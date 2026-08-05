//-*-c++-*-
/* CUDA implementation of the 3D heat conduction problem
         du/dt = div ( kappa(x,y,z) grad u ) + f(x,y,z)
   where kappa(x,y,z) = 0.5+0.45*sin(pi*x)*sin(pi*y)*sin(pi*z),
   using explicit time stepping.

   TWO-GPU VERSION, ASYNC + 2 HOST THREADS. Halo exchange staged through the
   host on a dedicated stream, overlapped with the interior kernel.
   ONE HOST THREAD PER DEVICE.

   Layout: u[k][j][i]  ->  k is slowest, i is fastest
     offset = k*(ny*nx) + j*nx + i
   Decomposition along the k axis, so an XY plane at fixed k is CONTIGUOUS
   and no pack/unpack kernels are needed.
   Neighbour offsets:
     i+-1  ->  +-1
     j+-1  ->  +-nx
     k+-1  ->  +-ny*nx

   This file is kept deliberately parallel to 2-openmp-stream-omp.c and
   differs from 2-cuda-stream.cu only in the host threading: same
   decomposition, same kernel split, same launch order, same prints, same
   timing.

   HOST THREADING
   --------------
   One OpenMP thread per device. Each thread calls cudaSetDevice() ONCE and
   never switches, so the CUDA context is thread-local and the two devices
   are driven genuinely concurrently rather than by one thread alternating
   between them.

   Two barriers per step:
     after PHASE 1   both halo planes must be on the host before either
                     device reads the other's buffer
     after PHASE 2   both devices must be finished before the pointer swap

   The pointer swap and time advance happen in a `single` region; its
   implicit barrier is what makes the updated `t` visible to both threads
   before the loop condition is re-evaluated.

   KERNEL SPLIT
   ------------
   The split is by INTERFACE, not by position:

     heat_update_boundary_bottom   k = 1        only where k=1 is SENT
     heat_update_boundary_top      k = nz-2     only where k=nz-2 is SENT
     heat_update_interior          k_start..k_end   everything else

   The boundary kernel runs on halo_stream so the exchanged plane finishes
   early and its D2H can start while the interior is still running on
   compute_stream. A plane that is never sent has no such urgency, so folding
   it into the interior keeps the boundary kernel -- which gates the exchange
   -- as small as possible.

     GPU0 has no lower neighbour -> plane k=1 folded into interior
          (k_start=1); runs TOP only.
     GPU1 has no upper neighbour -> plane k=nz1-2 folded into interior
          (k_end=nz1-2); runs BOTTOM only.

     GPU0   interior 1 .. nz0-3   + top       = 1 .. nz0-2
     GPU1   interior 2 .. nz1-2   + bottom    = 1 .. nz1-2

   HOST HALO BUFFERS ARE PAGEABLE, DELIBERATELY
   --------------------------------------------
   An earlier version of this file used cudaMallocHost (pinned), while
   2-cuda-stream.cu used plain malloc. That difference would have made the
   stream vs stream-omp comparison measure the copy path as well as the host
   threading. All four CUDA variants now use plain malloc, matching what
   2-openmp-stream-omp.c stages through. Pinned buffers would very likely be
   faster; if that number is wanted it belongs in a separate variant.

   HALO BUFFER NAMING
   ------------------
     h_haloN_up   plane device N sends UPWARD   (to device N+1)
     h_haloN_dn   plane device N sends DOWNWARD (to device N-1)

   BUILD NOTE:
     nvcc -O3 -arch=sm_90 --fmad=false -Xcompiler -fopenmp 2-cuda-stream-omp.cu -lm -o 2-cuda-stream-omp
   --fmad=false is the nvcc equivalent of clang's -ffp-contract=off. Here
   -fopenmp is needed for the host threading as well as omp_get_wtime().
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>   /* strcasecmp */
#include <math.h>
#include <omp.h>

#define IDX(k, j, i, ny, nx)  ((size_t)(k)*(size_t)(ny)*(size_t)(nx) \
                             + (size_t)(j)*(size_t)(nx) + (size_t)(i))

/*
DPCT1009:108: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced by a placeholder string. You need to rewrite this
code.
*/
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    dpct::err0 err = (call);                                                   \
                                                                               \
  } while (0)

static int should_write_txt(void)
{
  const char *e = getenv("WRITE_TXT");
  if (!e) return 1;
  if (!strcmp(e, "0")) return 0;
  if (!strcasecmp(e, "false")) return 0;
  if (!strcasecmp(e, "no")) return 0;
  return 1;
}

static void write_u_values(const char *fname, const double *u, int n)
{
  FILE *f = fopen(fname, "w");
  if (!f) { perror("fopen"); return; }
  for (int k=0; k<n; k++)
    for (int j=0; j<n; j++)
      for (int i=0; i<n; i++)
        fprintf(f, "%.17g\n", u[IDX(k, j, i, n, n)]);
  fclose(f);
  printf("Solution written to %s\n", fname);
}

// -- Partition initialisation  (layout: u[k][j][i]) --------------------------
// Identical to 2-openmp-stream-omp.c. Boundary values forced to exact zero.
static void init_partition(double *u, double *rhs, double *kappa,
                           int nz_local, int gk_start, int ng,
                           int n, int ny, int nx, double h, double dt)
{
  for (int k=0; k<nz_local; k++) {
    int gk = (k - ng) + gk_start;
    for (int j=0; j<ny; j++)
      for (int i=0; i<nx; i++) {
        size_t idx = IDX(k, j, i, ny, nx);
        u[idx]     = (gk<=0||gk>=n-1||j<=0||j>=n-1||i<=0||i>=n-1)
                     ? 0.0 : sin(M_PI*i*h)*sin(M_PI*j*h)*sin(M_PI*gk*h);
        rhs[idx]   = dt*(1.0+cos(M_PI*i*h)*cos(M_PI*j*h)*cos(M_PI*gk*h));
        kappa[idx] = 0.5+0.45*sin(M_PI*i*h)*sin(M_PI*j*h)*sin(M_PI*gk*h);
      }
  }
}

// -- BOTTOM boundary kernel  (layout: u[k][j][i]) ----------------------------
// Computes k=1, the plane sent DOWNWARD. Used by GPU 1.
// `nz` is unused here (k is fixed at 1); the parameter is kept so that all
// three kernels share one signature shape.
SYCL_EXTERNAL
void heat_update_boundary_bottom(double * __restrict__ u_new,
                                 const double * __restrict__ u_old,
                                 const double * __restrict__ rhs,
                                 const double * __restrict__ kappa,
                                 int nz, int ny, int nx, double factor,
                                 const sycl::nd_item<3> &item_ct1)
{
  int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2) + 1;
  int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1) + 1;

  if (i < nx-1 && j < ny-1) {
    int k = 1;

    size_t idx = IDX(k,   j,   i,   ny, nx);
    size_t ip1 = IDX(k,   j,   i+1, ny, nx);
    size_t im1 = IDX(k,   j,   i-1, ny, nx);
    size_t jp1 = IDX(k,   j+1, i,   ny, nx);
    size_t jm1 = IDX(k,   j-1, i,   ny, nx);
    size_t kp1 = IDX(k+1, j,   i,   ny, nx);
    size_t km1 = IDX(k-1, j,   i,   ny, nx);

    u_new[idx] = u_old[idx] + rhs[idx]
      + factor * (
          (kappa[ip1] + kappa[idx]) * (u_old[ip1] - u_old[idx])
        - (kappa[idx] + kappa[im1]) * (u_old[idx] - u_old[im1])
        + (kappa[jp1] + kappa[idx]) * (u_old[jp1] - u_old[idx])
        - (kappa[idx] + kappa[jm1]) * (u_old[idx] - u_old[jm1])
        + (kappa[kp1] + kappa[idx]) * (u_old[kp1] - u_old[idx])
        - (kappa[idx] + kappa[km1]) * (u_old[idx] - u_old[km1])
        );
  }
}

// -- TOP boundary kernel  (layout: u[k][j][i]) -------------------------------
// Computes k=nz-2, the plane sent UPWARD. Used by GPU 0.
SYCL_EXTERNAL
void heat_update_boundary_top(double * __restrict__ u_new,
                              const double * __restrict__ u_old,
                              const double * __restrict__ rhs,
                              const double * __restrict__ kappa,
                              int nz, int ny, int nx, double factor,
                              const sycl::nd_item<3> &item_ct1)
{
  int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2) + 1;
  int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1) + 1;

  if (i < nx-1 && j < ny-1) {
    int k = nz - 2;

    size_t idx = IDX(k,   j,   i,   ny, nx);
    size_t ip1 = IDX(k,   j,   i+1, ny, nx);
    size_t im1 = IDX(k,   j,   i-1, ny, nx);
    size_t jp1 = IDX(k,   j+1, i,   ny, nx);
    size_t jm1 = IDX(k,   j-1, i,   ny, nx);
    size_t kp1 = IDX(k+1, j,   i,   ny, nx);
    size_t km1 = IDX(k-1, j,   i,   ny, nx);

    u_new[idx] = u_old[idx] + rhs[idx]
      + factor * (
          (kappa[ip1] + kappa[idx]) * (u_old[ip1] - u_old[idx])
        - (kappa[idx] + kappa[im1]) * (u_old[idx] - u_old[im1])
        + (kappa[jp1] + kappa[idx]) * (u_old[jp1] - u_old[idx])
        - (kappa[idx] + kappa[jm1]) * (u_old[idx] - u_old[jm1])
        + (kappa[kp1] + kappa[idx]) * (u_old[kp1] - u_old[idx])
        - (kappa[idx] + kappa[km1]) * (u_old[idx] - u_old[km1])
        );
  }
}

// -- INTERIOR kernel  (layout: u[k][j][i]) -----------------------------------
// Computes k = k_start .. k_end. The explicit bounds let each device fold in
// the plane it never sends, which has no reason to sit in a boundary kernel.
// `nz` is unused here (the range is given explicitly); the parameter is kept
// so that all three kernels share one signature shape.
//
// Block shape (32,4,2): 32 threads along i is a full warp on contiguous
// addresses, and 256 threads/block keeps register pressure below what an
// (8,8,8)=512 block costs.
SYCL_EXTERNAL
void heat_update_interior(double * __restrict__ u_new,
                          const double * __restrict__ u_old,
                          const double * __restrict__ rhs,
                          const double * __restrict__ kappa,
                          int nz, int ny, int nx, double factor,
                          int k_start, int k_end,
                          const sycl::nd_item<3> &item_ct1)
{
  int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2) + 1;
  int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1) + 1;
  int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
          item_ct1.get_local_id(0) + k_start;

  if (i < nx-1 && j < ny-1 && k <= k_end) {
    size_t idx = IDX(k,   j,   i,   ny, nx);
    size_t ip1 = IDX(k,   j,   i+1, ny, nx);
    size_t im1 = IDX(k,   j,   i-1, ny, nx);
    size_t jp1 = IDX(k,   j+1, i,   ny, nx);
    size_t jm1 = IDX(k,   j-1, i,   ny, nx);
    size_t kp1 = IDX(k+1, j,   i,   ny, nx);
    size_t km1 = IDX(k-1, j,   i,   ny, nx);

    u_new[idx] = u_old[idx] + rhs[idx]
      + factor * (
          (kappa[ip1] + kappa[idx]) * (u_old[ip1] - u_old[idx])
        - (kappa[idx] + kappa[im1]) * (u_old[idx] - u_old[im1])
        + (kappa[jp1] + kappa[idx]) * (u_old[jp1] - u_old[idx])
        - (kappa[idx] + kappa[jm1]) * (u_old[idx] - u_old[jm1])
        + (kappa[kp1] + kappa[idx]) * (u_old[kp1] - u_old[idx])
        - (kappa[idx] + kappa[km1]) * (u_old[idx] - u_old[km1])
        );
  }
}

int main(int argc, char **argv) try {
  setvbuf(stdout, NULL, _IONBF, 0);  /* unbuffered: never lose output on a fault */

  printf("\n**** TWO GPUS CUDA, ASYNC + 2 HOST THREADS STARTS ****\n");

  int    n = (argc > 1) ? atoi(argv[1]) : 51;
  double T = (argc > 2) ? atof(argv[2]) : 1.0;

  int nx = n, ny = n;
  size_t total_points = (size_t)n * (size_t)n * (size_t)n;
  size_t global_bytes = total_points * sizeof(double);

  printf("\n=== MEMORY ESTIMATE (GLOBAL) ===\n");
  printf("Grid size: %d x %d x %d = %zu points\n", n, n, n, total_points);
  printf("Memory per global array: %.3f GB\n", global_bytes / 1e9);
  printf("================================\n\n");

  double h         = 1.0/(n-1);
  double kappa_max = 0.95;
  double dt        = h*h / (12.0 * kappa_max);
  double factor    = dt/h/h/2.0;

  printf("=== SIMULATION PARAMETERS ===\n");
  printf("Grid size (n):           %d\n", n);
  printf("Final time (T):          %.6f\n", T);
  printf("Grid spacing (h):        %.6e\n", h);
  printf("Time step (dt):          %.6e\n", dt);
  printf("Kappa_max:               %.6f\n", kappa_max);
  printf("Diffusion factor:        %.6e\n", factor);
  printf("Estimated timesteps:     ~%d\n", (int)(T/dt));
  printf("CFL condition (dt/h^2):  %.6f (should be < 0.5)\n", dt/(h*h));
  printf("==============================\n\n");

  // -- Device selection -----------------------------------------
  int num_devices = 0;
  CUDA_CHECK(
      DPCT_CHECK_ERROR(num_devices = dpct::dev_mgr::instance().device_count()));

  if (num_devices < 2) {
    fprintf(stderr, "ERROR: this version needs 2 CUDA devices, %d present\n",
            num_devices);
    exit(EXIT_FAILURE);
  }

  printf("Using CUDA devices 0 and 1 of %d\n", num_devices);

  // -- Domain decomposition along k -----------------------------
  // The n-2 interior planes are split two ways; the remainder, if any, goes
  // to the higher-numbered device.
  int ng     = 1;
  int nz_int = n - 2;
  int mid    = nz_int / 2;

  int nz0 = mid                + 2*ng;
  int nz1 = (nz_int - mid)     + 2*ng;

  int gk_start0 = 1;
  int gk_start1 = gk_start0 + (nz0 - 2*ng);

  printf("Block sizes: nz0=%d, nz1=%d, mid=%d\n", nz0, nz1, mid);

  // The boundary kernels write k=1 and k=nz-2; with nz<4 those coincide or
  // overlap the ghost planes, so the decomposition is only valid above that.
  if (nz0 < 4 || nz1 < 4) {
    fprintf(stderr,
            "ERROR: n=%d gives nz0=%d nz1=%d; each needs >= 4 planes\n",
            n, nz0, nz1);
    exit(EXIT_FAILURE);
  }

  // -- Partition sizes  (layout [k][j][i]) ----------------------
  size_t size0      = (size_t)nz0*(size_t)ny*(size_t)nx*sizeof(double);
  size_t size1      = (size_t)nz1*(size_t)ny*(size_t)nx*sizeof(double);
  size_t halo_bytes = (size_t)ny*(size_t)nx*sizeof(double);  // one XY plane

  printf("\n=== MEMORY ESTIMATE (LOCAL PER GPU) ===\n");
  printf("GPU0 local bytes per array: %.3f GB\n", size0 / 1e9);
  printf("GPU1 local bytes per array: %.3f GB\n", size1 / 1e9);
  printf("======================================\n\n");

  // -- Host arrays -----------------------------------------------
  double *h_u_global = (double*)malloc(global_bytes);
  double *h_u0       = (double*)malloc(size0);
  double *h_u1       = (double*)malloc(size1);
  double *h_rhs0     = (double*)malloc(size0);
  double *h_rhs1     = (double*)malloc(size1);
  double *h_kappa0   = (double*)malloc(size0);
  double *h_kappa1   = (double*)malloc(size1);

  if (!h_u_global || !h_u0 || !h_u1 ||
      !h_rhs0 || !h_rhs1 || !h_kappa0 || !h_kappa1) {
    fprintf(stderr, "ERROR: malloc failed (host arrays)\n");
    return 1;
  }

  // -- Initialize the two partitions  (layout [k][j][i]) --------
  init_partition(h_u0, h_rhs0, h_kappa0, nz0, gk_start0, ng, n, ny, nx, h, dt);
  init_partition(h_u1, h_rhs1, h_kappa1, nz1, gk_start1, ng, n, ny, nx, h, dt);

  // -- GPU 0 device allocation + H2D ----------------------------
  double *d0_u_old, *d0_u_new, *d0_rhs, *d0_kappa;

  /*
  DPCT1093:109: The "0" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(0)));
  CUDA_CHECK(DPCT_CHECK_ERROR(d0_u_old = (double *)sycl::malloc_device(
                                  size0, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(d0_u_new = (double *)sycl::malloc_device(
                                  size0, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(d0_rhs = (double *)sycl::malloc_device(
                                  size0, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(d0_kappa = (double *)sycl::malloc_device(
                                  size0, dpct::get_in_order_queue())));

  // u_new is seeded from the same array as u_old so its BOUNDARY values are
  // the exact zeros above -- the kernels only ever write interior points.
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d0_u_old, h_u0, size0).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d0_u_new, h_u0, size0).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d0_rhs, h_rhs0, size0).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d0_kappa, h_kappa0, size0).wait()));

  // -- GPU 1 device allocation + H2D ----------------------------
  double *d1_u_old, *d1_u_new, *d1_rhs, *d1_kappa;

  /*
  DPCT1093:110: The "1" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(1)));
  CUDA_CHECK(DPCT_CHECK_ERROR(d1_u_old = (double *)sycl::malloc_device(
                                  size1, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(d1_u_new = (double *)sycl::malloc_device(
                                  size1, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(d1_rhs = (double *)sycl::malloc_device(
                                  size1, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(d1_kappa = (double *)sycl::malloc_device(
                                  size1, dpct::get_in_order_queue())));

  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d1_u_old, h_u1, size1).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d1_u_new, h_u1, size1).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d1_rhs, h_rhs1, size1).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d1_kappa, h_kappa1, size1).wait()));

  // -- Host halo staging buffers ---------------------------------
  // Pageable on purpose -- see the header note. Two buffers: one interface,
  // two directions.
  double *h_halo0_up = (double*)malloc(halo_bytes);  // dev0 -> dev1
  double *h_halo1_dn = (double*)malloc(halo_bytes);  // dev1 -> dev0

  if (!h_halo0_up || !h_halo1_dn) {
    fprintf(stderr, "ERROR: malloc failed (halo host)\n");
    return 1;
  }

  // -- Kernel configs -------------------------------------------
  //   GPU0 interior 1 .. nz0-3
  //   GPU1 interior 2 .. nz1-2
  sycl::range<3> block(2, 4, 32);

  int k_start0 = 1,  k_end0 = nz0 - 3;
  int k_start1 = 2,  k_end1 = nz1 - 2;

  sycl::range<3> grid0_interior(
      (k_end0 - k_start0 + 1 + block[0] - 1) / block[0],
      (ny - 2 + block[1] - 1) / block[1], (nx - 2 + block[2] - 1) / block[2]);

  sycl::range<3> grid1_interior(
      (k_end1 - k_start1 + 1 + block[0] - 1) / block[0],
      (ny - 2 + block[1] - 1) / block[1], (nx - 2 + block[2] - 1) / block[2]);

  // Boundary: a single XY plane, so the grid is 2D.
  sycl::range<3> block_boundary(1, 8, 32);
  sycl::range<3> grid_boundary(
      1, (ny - 2 + block_boundary[1] - 1) / block_boundary[1],
      (nx - 2 + block_boundary[2] - 1) / block_boundary[2]);

  // -- Streams ---------------------------------------------------
  const int ngpus = 2;
  dpct::queue_ptr compute_stream[ngpus];
  dpct::queue_ptr halo_stream[ngpus];

  for (int g = 0; g < ngpus; ++g) {
    /*
    DPCT1093:111: The "g" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(g)));
    CUDA_CHECK(DPCT_CHECK_ERROR(compute_stream[g] =
                                    dpct::get_current_device().create_queue()));
    CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream[g] =
                                    dpct::get_current_device().create_queue()));
  }

  double t     = 0.0;
  int    steps = 0;

  printf("\n=== SIMULATION STARTS ===\n");

  double compute_timer = 0.0;
  compute_timer -= omp_get_wtime();

  // ======================================================================
  // Time loop, one host thread per device.
  //
  // PHASE 1: interior on compute_stream, boundary on halo_stream, then D2H
  //   from the contiguous boundary plane on halo_stream -- no pack kernel.
  //   Boundary and D2H are on the SAME stream, so the copy cannot start
  //   before the plane is written; no event is needed.
  //
  //   barrier: both halo planes must be on the host before PHASE 2.
  //
  // PHASE 2: H2D into the contiguous ghost plane on halo_stream -- no unpack.
  //
  //   barrier: both devices must be idle before the pointer swap.
  //
  // Transfer map (two transfers, one interface):
  //   GPU0 plane k=nz0-2  ->  h_halo0_up  ->  GPU1 ghost k=0
  //   GPU1 plane k=1      ->  h_halo1_dn  ->  GPU0 ghost k=nz0-1
  //
  // The ghost planes at GPU0 k=0 and GPU1 k=nz1-1 are physical domain
  // boundaries. They are initialised to zero and never written.
  // ======================================================================

#pragma omp parallel num_threads(ngpus)
  {
    int tid = omp_get_thread_num();

    // Each host thread owns one device: set once, never switch.
    /*
    DPCT1093:112: The "tid" device may be not the one intended for use. Adjust
    the selected device if needed.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(tid)));

    while (t < T) {

      // -- PHASE 1: compute + D2H ---------------------------------
      if (tid == 0) {
        // GPU0: no bottom boundary -- k=1 is never sent.
        /*
        DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(compute_stream[0]->get_device(),
                                       {sycl::aspect::fp64});

          compute_stream[0]->parallel_for(
              sycl::nd_range<3>(grid0_interior * block, block),
              [=](sycl::nd_item<3> item_ct1) {
                heat_update_interior(d0_u_new, d0_u_old, d0_rhs, d0_kappa, nz0,
                                     ny, nx, factor, k_start0, k_end0,
                                     item_ct1);
              });
        }
        /*
        DPCT1010:113: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);
        /*
        DPCT1049:10: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(halo_stream[0]->get_device(),
                                       {sycl::aspect::fp64});

          halo_stream[0]->parallel_for(
              sycl::nd_range<3>(grid_boundary * block_boundary, block_boundary),
              [=](sycl::nd_item<3> item_ct1) {
                heat_update_boundary_top(d0_u_new, d0_u_old, d0_rhs, d0_kappa,
                                         nz0, ny, nx, factor, item_ct1);
              });
        }
        /*
        DPCT1010:114: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);

        /*
        DPCT1124:115: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream[0]->memcpy(
            h_halo0_up, d0_u_new + (size_t)(nz0 - 2) * ny * nx, halo_bytes)));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream[0]->wait()));
      }
      else {
        // GPU1: no top boundary -- k=nz1-2 is never sent.
        /*
        DPCT1049:11: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(compute_stream[1]->get_device(),
                                       {sycl::aspect::fp64});

          compute_stream[1]->parallel_for(
              sycl::nd_range<3>(grid1_interior * block, block),
              [=](sycl::nd_item<3> item_ct1) {
                heat_update_interior(d1_u_new, d1_u_old, d1_rhs, d1_kappa, nz1,
                                     ny, nx, factor, k_start1, k_end1,
                                     item_ct1);
              });
        }
        /*
        DPCT1010:116: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);
        /*
        DPCT1049:12: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(halo_stream[1]->get_device(),
                                       {sycl::aspect::fp64});

          halo_stream[1]->parallel_for(
              sycl::nd_range<3>(grid_boundary * block_boundary, block_boundary),
              [=](sycl::nd_item<3> item_ct1) {
                heat_update_boundary_bottom(d1_u_new, d1_u_old, d1_rhs,
                                            d1_kappa, nz1, ny, nx, factor,
                                            item_ct1);
              });
        }
        /*
        DPCT1010:117: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);

        /*
        DPCT1124:118: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream[1]->memcpy(
            h_halo1_dn, d1_u_new + (size_t)1 * ny * nx, halo_bytes)));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream[1]->wait()));
      }

      // Both halo planes are on the host -- safe to cross-exchange.
#pragma omp barrier

      // -- PHASE 2: H2D into the ghost planes ---------------------
      if (tid == 0) {
        /*
        DPCT1124:119: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream[0]->memcpy(
            d0_u_new + (size_t)(nz0 - 1) * ny * nx, h_halo1_dn, halo_bytes)));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream[0]->wait()));
        CUDA_CHECK(DPCT_CHECK_ERROR(compute_stream[0]->wait()));
      }
      else {
        /*
        DPCT1124:120: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream[1]->memcpy(
            d1_u_new + (size_t)0 * ny * nx, h_halo0_up, halo_bytes)));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream[1]->wait()));
        CUDA_CHECK(DPCT_CHECK_ERROR(compute_stream[1]->wait()));
      }

      // Both devices are idle -- safe to swap.
#pragma omp barrier

      // The implicit barrier at the end of `single` is what publishes the
      // updated `t` to both threads before the loop condition is re-read.
#pragma omp single
      {
        double *tmp;
        tmp = d0_u_old; d0_u_old = d0_u_new; d0_u_new = tmp;
        tmp = d1_u_old; d1_u_old = d1_u_new; d1_u_new = tmp;
        steps++;
        t += dt;
      }

    } // end while
  }   // end omp parallel

  // Every stream is drained inside the loop, so both devices are already
  // idle. Make it explicit so the timed region cannot drift if the internal
  // synchronisation is ever relaxed.
  /*
  DPCT1093:121: The "0" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(0))); CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
  /*
  DPCT1093:122: The "1" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(1))); CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));

  compute_timer += omp_get_wtime();

  printf("\n%d timesteps complete\n", steps);
  printf("Time = %f seconds\n", compute_timer);
  printf("=== SIMULATION ENDS ===\n");

  // -- Copy back (layout [k][j][i]) ------------------------------
  /*
  DPCT1093:123: The "0" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(0)));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(h_u0, d0_u_old, size0).wait()));
  /*
  DPCT1093:124: The "1" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(1)));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(h_u1, d1_u_old, size1).wait()));

  // -- Reconstruct global solution (global output is [k][j][i]) --
  // Each device contributes its owned planes only; the interface ghosts are
  // excluded because the owner writes them. The two physical-boundary
  // ghosts (GPU0 k=0, GPU1 k=nz1-1) are included and hold exact zero.
  memset(h_u_global, 0, global_bytes);

  // GPU0 contribution: global k = 0 .. gk_start1-1
  for (int k=0; k<nz0; k++) {
    int gk = (k - ng) + gk_start0;
    if (gk >= 0 && gk < gk_start1) {
      for (int j=0; j<n; j++)
        for (int i=0; i<n; i++)
          h_u_global[IDX(gk, j, i, n, n)] = h_u0[IDX(k, j, i, ny, nx)];
    }
  }

  // GPU1 contribution: global k = gk_start1 .. n-1
  for (int k=0; k<nz1; k++) {
    int gk = (k - ng) + gk_start1;
    if (gk >= gk_start1 && gk <= n-1) {
      for (int j=0; j<n; j++)
        for (int i=0; i<n; i++)
          h_u_global[IDX(gk, j, i, n, n)] = h_u1[IDX(k, j, i, ny, nx)];
    }
  }

  if (should_write_txt())
    write_u_values("u_sycl_stream_omp_2gpu.txt", h_u_global, n);

  // -- Cleanup streams -------------------------------------------
  for (int g = 0; g < ngpus; ++g) {
    /*
    DPCT1093:125: The "g" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(g)));
    CUDA_CHECK(DPCT_CHECK_ERROR(
        dpct::get_current_device().destroy_queue(compute_stream[g])));
    CUDA_CHECK(DPCT_CHECK_ERROR(
        dpct::get_current_device().destroy_queue(halo_stream[g])));
  }

  // -- Cleanup ---------------------------------------------------
  /*
  DPCT1093:126: The "0" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(0)));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(d0_u_old, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(
          dpct::dpct_free(d0_u_new, dpct::get_in_order_queue())));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(d0_rhs, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(
          dpct::dpct_free(d0_kappa, dpct::get_in_order_queue())));

  /*
  DPCT1093:127: The "1" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(1)));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(d1_u_old, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(
          dpct::dpct_free(d1_u_new, dpct::get_in_order_queue())));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(d1_rhs, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(
          dpct::dpct_free(d1_kappa, dpct::get_in_order_queue())));

  free(h_u_global);
  free(h_u0);     free(h_u1);
  free(h_rhs0);   free(h_rhs1);
  free(h_kappa0); free(h_kappa1);
  free(h_halo0_up); free(h_halo1_dn);

  printf("\n**** TWO GPUS CUDA, ASYNC + 2 HOST THREADS FINISHED ****\n");
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
