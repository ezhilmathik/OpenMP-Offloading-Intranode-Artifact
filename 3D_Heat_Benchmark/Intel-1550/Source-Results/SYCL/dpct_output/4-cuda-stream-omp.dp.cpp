//-*-c++-*-
/* CUDA implementation of the 3D heat conduction problem
         du/dt = div ( kappa(x,y,z) grad u ) + f(x,y,z)
   where kappa(x,y,z) = 0.5+0.45*sin(pi*x)*sin(pi*y)*sin(pi*z),
   using explicit time stepping.

   FOUR-GPU VERSION, ASYNC + 4 HOST THREADS. Halo exchange staged through the
   host on dedicated streams, overlapped with the interior kernel.
   ONE HOST THREAD PER DEVICE.

   Layout: u[k][j][i]  ->  k is slowest, i is fastest
     offset = k*(ny*nx) + j*nx + i
   Decomposition along the k axis, so an XY plane at fixed k is CONTIGUOUS
   and no pack/unpack kernels are needed.
   Neighbour offsets:
     i+-1  ->  +-1
     j+-1  ->  +-nx
     k+-1  ->  +-ny*nx

   This file is kept deliberately parallel to 4-openmp-stream-omp.c and
   differs from 4-cuda-stream.cu only in the host threading.

   HOST THREADING
   --------------
   One OpenMP thread per device. Each thread calls cudaSetDevice() ONCE and
   never switches, so the CUDA context is thread-local and all four devices
   are driven genuinely concurrently.

   Two barriers per step:
     after PHASE A   every host halo buffer must be filled before any device
                     reads a neighbour's buffer
     after PHASE B   every device must be finished before the pointer swap

   The pointer swap and time advance happen in a `single` region; its
   implicit barrier is what makes the updated `t` visible to all four threads
   before the loop condition is re-evaluated.

   KERNEL SPLIT
   ------------
   The split is by INTERFACE, not by position:

     heat_update_boundary_bottom   k = 1        only where k=1 is SENT
     heat_update_boundary_top      k = nz-2     only where k=nz-2 is SENT
     heat_update_interior          k_start..k_end   everything else

   With four devices there are THREE interfaces. The two END devices send one
   plane each and run one boundary kernel; the two MIDDLE devices send both
   planes and run BOTH:

     GPU0 no lower neighbour -> k=1 folded into interior (k_start=1);
          runs TOP only.
     GPU1 both neighbours    -> runs BOTTOM and TOP.
     GPU2 both neighbours    -> runs BOTTOM and TOP.
     GPU3 no upper neighbour -> k=nz3-2 folded into interior (k_end=nz3-2);
          runs BOTTOM only.

     GPU0   interior 1 .. nz0-3   + top             = 1 .. nz0-2
     GPU1   interior 2 .. nz1-3   + bottom + top    = 1 .. nz1-2
     GPU2   interior 2 .. nz2-3   + bottom + top    = 1 .. nz2-2
     GPU3   interior 2 .. nz3-2   + bottom          = 1 .. nz3-2

   STREAM ASSIGNMENT
   -----------------
   THREE streams per device:

     compute_stream[d]        interior kernel
     halo_stream_bottom[d]    bottom boundary kernel, its D2H, and the H2D
                              into ghost k=0        (the LOWER interface)
     halo_stream_top[d]       top boundary kernel, its D2H, and the H2D
                              into ghost k=nz-1     (the UPPER interface)

   A middle device has two interfaces that have no reason to serialise
   against each other, so each gets its own stream. Kernel -> D2H -> H2D for
   one interface all sit on the same stream, so their order is guaranteed
   without any explicit event.

   HOST HALO BUFFERS ARE PAGEABLE, DELIBERATELY
   --------------------------------------------
   See 2-cuda-stream.cu's header. Every file in this set uses plain malloc so
   that the CUDA and OpenMP measurements stay comparable, and so that the
   stream / stream-omp / p2p differences are not contaminated by a change of
   copy path.

   HALO BUFFER NAMING
   ------------------
     h_haloN_up   plane device N sends UPWARD   (to device N+1)
     h_haloN_dn   plane device N sends DOWNWARD (to device N-1)

   Six transfers per step, three interfaces:
     GPU0 plane k=nz0-2  ->  h_halo0_up  ->  GPU1 ghost k=0
     GPU1 plane k=1      ->  h_halo1_dn  ->  GPU0 ghost k=nz0-1
     GPU1 plane k=nz1-2  ->  h_halo1_up  ->  GPU2 ghost k=0
     GPU2 plane k=1      ->  h_halo2_dn  ->  GPU1 ghost k=nz1-1
     GPU2 plane k=nz2-2  ->  h_halo2_up  ->  GPU3 ghost k=0
     GPU3 plane k=1      ->  h_halo3_dn  ->  GPU2 ghost k=nz2-1

   BUILD NOTE:
     nvcc -O3 -arch=sm_90 --fmad=false -Xcompiler -fopenmp 4-cuda-stream-omp.cu -lm -o 4-cuda-stream-omp
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
DPCT1009:241: SYCL uses exceptions to report errors and does not use the error
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
// Identical to 4-openmp-stream-omp.c. Boundaries forced to exact zero.
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

// -- Global reconstruction: copy one partition's owned planes into the global
// array. Each device contributes global k in [gk_lo, gk_hi); the interface
// ghosts are excluded because the owner writes them, and the two
// physical-boundary ghosts are included and hold exact zero.
static void gather_partition(double *u_global, const double *u_local,
                             int nz_local, int gk_start, int ng,
                             int gk_lo, int gk_hi, int n, int ny, int nx)
{
  for (int k=0; k<nz_local; k++) {
    int gk = (k - ng) + gk_start;
    if (gk >= gk_lo && gk < gk_hi) {
      for (int j=0; j<n; j++)
        for (int i=0; i<n; i++)
          u_global[IDX(gk, j, i, n, n)] = u_local[IDX(k, j, i, ny, nx)];
    }
  }
}

// -- BOTTOM boundary kernel  (layout: u[k][j][i]) ----------------------------
// Computes k=1, the plane sent DOWNWARD. Used by GPU 1, 2 and 3.
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
// Computes k=nz-2, the plane sent UPWARD. Used by GPU 0, 1 and 2.
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
// Computes k = k_start .. k_end. The explicit bounds let each end device fold
// in the plane it never sends:
//   GPU0  k_start=1, k_end=nz0-3   GPU1  k_start=2, k_end=nz1-3
//   GPU2  k_start=2, k_end=nz2-3   GPU3  k_start=2, k_end=nz3-2
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

  printf("\n**** FOUR GPUS CUDA, ASYNC + 4 HOST THREADS STARTS ****\n");

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

  if (num_devices < 4) {
    fprintf(stderr, "ERROR: this version needs 4 CUDA devices, %d present\n",
            num_devices);
    exit(EXIT_FAILURE);
  }

  printf("Using CUDA devices 0, 1, 2 and 3 of %d\n", num_devices);

  // -- Domain decomposition along k -----------------------------
  // The n-2 interior planes are split four ways; the remainder, if any, is
  // handed to the lowest-numbered devices one plane at a time.
  int ng        = 1;
  int nz_int    = n - 2;
  int quarter   = nz_int / 4;
  int remainder = nz_int % 4;

  int nz0 = quarter + (remainder > 0 ? 1 : 0) + 2*ng;
  int nz1 = quarter + (remainder > 1 ? 1 : 0) + 2*ng;
  int nz2 = quarter + (remainder > 2 ? 1 : 0) + 2*ng;
  int nz3 = quarter                           + 2*ng;

  int gk_start0 = 1;
  int gk_start1 = gk_start0 + (nz0 - 2*ng);
  int gk_start2 = gk_start1 + (nz1 - 2*ng);
  int gk_start3 = gk_start2 + (nz2 - 2*ng);

  printf("Block sizes: nz0=%d, nz1=%d, nz2=%d, nz3=%d\n", nz0, nz1, nz2, nz3);

  // The boundary kernels write k=1 and k=nz-2; with nz<4 those coincide or
  // overlap the ghost planes, so the decomposition is only valid above that.
  if (nz0 < 4 || nz1 < 4 || nz2 < 4 || nz3 < 4) {
    fprintf(stderr,
            "ERROR: n=%d gives nz0=%d nz1=%d nz2=%d nz3=%d; each needs >= 4 planes\n",
            n, nz0, nz1, nz2, nz3);
    exit(EXIT_FAILURE);
  }

  // -- Partition sizes  (layout [k][j][i]) ----------------------
  size_t size0      = (size_t)nz0*(size_t)ny*(size_t)nx*sizeof(double);
  size_t size1      = (size_t)nz1*(size_t)ny*(size_t)nx*sizeof(double);
  size_t size2      = (size_t)nz2*(size_t)ny*(size_t)nx*sizeof(double);
  size_t size3      = (size_t)nz3*(size_t)ny*(size_t)nx*sizeof(double);
  size_t halo_bytes = (size_t)ny*(size_t)nx*sizeof(double);  // one XY plane

  printf("\n=== MEMORY ESTIMATE (LOCAL PER GPU) ===\n");
  printf("GPU0 local bytes per array: %.3f GB\n", size0 / 1e9);
  printf("GPU1 local bytes per array: %.3f GB\n", size1 / 1e9);
  printf("GPU2 local bytes per array: %.3f GB\n", size2 / 1e9);
  printf("GPU3 local bytes per array: %.3f GB\n", size3 / 1e9);
  printf("======================================\n\n");

  // -- Host arrays -----------------------------------------------
  double *h_u_global = (double*)malloc(global_bytes);
  double *h_u0       = (double*)malloc(size0);
  double *h_u1       = (double*)malloc(size1);
  double *h_u2       = (double*)malloc(size2);
  double *h_u3       = (double*)malloc(size3);
  double *h_rhs0     = (double*)malloc(size0);
  double *h_rhs1     = (double*)malloc(size1);
  double *h_rhs2     = (double*)malloc(size2);
  double *h_rhs3     = (double*)malloc(size3);
  double *h_kappa0   = (double*)malloc(size0);
  double *h_kappa1   = (double*)malloc(size1);
  double *h_kappa2   = (double*)malloc(size2);
  double *h_kappa3   = (double*)malloc(size3);

  if (!h_u_global || !h_u0 || !h_u1 || !h_u2 || !h_u3 ||
      !h_rhs0 || !h_rhs1 || !h_rhs2 || !h_rhs3 ||
      !h_kappa0 || !h_kappa1 || !h_kappa2 || !h_kappa3) {
    fprintf(stderr, "ERROR: malloc failed (host arrays)\n");
    return 1;
  }

  // -- Initialize the four partitions  (layout [k][j][i]) -------
  init_partition(h_u0, h_rhs0, h_kappa0, nz0, gk_start0, ng, n, ny, nx, h, dt);
  init_partition(h_u1, h_rhs1, h_kappa1, nz1, gk_start1, ng, n, ny, nx, h, dt);
  init_partition(h_u2, h_rhs2, h_kappa2, nz2, gk_start2, ng, n, ny, nx, h, dt);
  init_partition(h_u3, h_rhs3, h_kappa3, nz3, gk_start3, ng, n, ny, nx, h, dt);

  // -- Device allocation + H2D ----------------------------------
  // No halo send/recv device buffers: planes are contiguous in [k][j][i].
  // u_new is seeded from the same array as u_old so its BOUNDARY values are
  // the exact zeros above -- the kernels only ever write interior points.
  double *d0_u_old, *d0_u_new, *d0_rhs, *d0_kappa;
  double *d1_u_old, *d1_u_new, *d1_rhs, *d1_kappa;
  double *d2_u_old, *d2_u_new, *d2_rhs, *d2_kappa;
  double *d3_u_old, *d3_u_new, *d3_rhs, *d3_kappa;

  /*
  DPCT1093:242: The "0" device may be not the one intended for use. Adjust the
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
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d0_u_old, h_u0, size0).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d0_u_new, h_u0, size0).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d0_rhs, h_rhs0, size0).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d0_kappa, h_kappa0, size0).wait()));

  /*
  DPCT1093:243: The "1" device may be not the one intended for use. Adjust the
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

  /*
  DPCT1093:244: The "2" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(2)));
  CUDA_CHECK(DPCT_CHECK_ERROR(d2_u_old = (double *)sycl::malloc_device(
                                  size2, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(d2_u_new = (double *)sycl::malloc_device(
                                      size2, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(d2_rhs = (double *)sycl::malloc_device(
                                  size2, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(d2_kappa = (double *)sycl::malloc_device(
                                      size2, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d2_u_old, h_u2, size2).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d2_u_new, h_u2, size2).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d2_rhs, h_rhs2, size2).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d2_kappa, h_kappa2, size2).wait()));

  /*
  DPCT1093:245: The "3" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(3)));
  CUDA_CHECK(DPCT_CHECK_ERROR(d3_u_old = (double *)sycl::malloc_device(
                                  size3, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(d3_u_new = (double *)sycl::malloc_device(
                                      size3, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(d3_rhs = (double *)sycl::malloc_device(
                                  size3, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(d3_kappa = (double *)sycl::malloc_device(
                                      size3, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d3_u_old, h_u3, size3).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d3_u_new, h_u3, size3).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d3_rhs, h_rhs3, size3).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d3_kappa, h_kappa3, size3).wait()));

  // -- Host halo staging buffers ---------------------------------
  // Six buffers: three interfaces, two directions each. Pageable on purpose.
  double *h_halo0_up = (double*)malloc(halo_bytes);  // dev0 -> dev1
  double *h_halo1_dn = (double*)malloc(halo_bytes);  // dev1 -> dev0
  double *h_halo1_up = (double*)malloc(halo_bytes);  // dev1 -> dev2
  double *h_halo2_dn = (double*)malloc(halo_bytes);  // dev2 -> dev1
  double *h_halo2_up = (double*)malloc(halo_bytes);  // dev2 -> dev3
  double *h_halo3_dn = (double*)malloc(halo_bytes);  // dev3 -> dev2

  if (!h_halo0_up || !h_halo1_dn || !h_halo1_up ||
      !h_halo2_dn || !h_halo2_up || !h_halo3_dn) {
    fprintf(stderr, "ERROR: malloc failed (halo host)\n");
    return 1;
  }

  // -- Kernel configs -------------------------------------------
  sycl::range<3> block(2, 4, 32);

  int k_start0 = 1,  k_end0 = nz0 - 3;
  int k_start1 = 2,  k_end1 = nz1 - 3;
  int k_start2 = 2,  k_end2 = nz2 - 3;
  int k_start3 = 2,  k_end3 = nz3 - 2;

  sycl::range<3> grid0_interior(
      (k_end0 - k_start0 + 1 + block[0] - 1) / block[0],
      (ny - 2 + block[1] - 1) / block[1], (nx - 2 + block[2] - 1) / block[2]);
  sycl::range<3> grid1_interior(
      (k_end1 - k_start1 + 1 + block[0] - 1) / block[0],
      (ny - 2 + block[1] - 1) / block[1], (nx - 2 + block[2] - 1) / block[2]);
  sycl::range<3> grid2_interior(
      (k_end2 - k_start2 + 1 + block[0] - 1) / block[0],
      (ny - 2 + block[1] - 1) / block[1], (nx - 2 + block[2] - 1) / block[2]);
  sycl::range<3> grid3_interior(
      (k_end3 - k_start3 + 1 + block[0] - 1) / block[0],
      (ny - 2 + block[1] - 1) / block[1], (nx - 2 + block[2] - 1) / block[2]);

  // Boundary: a single XY plane, so the grid is 2D.
  sycl::range<3> block_boundary(1, 8, 32);
  sycl::range<3> grid_boundary(
      1, (ny - 2 + block_boundary[1] - 1) / block_boundary[1],
      (nx - 2 + block_boundary[2] - 1) / block_boundary[2]);

  // -- Streams: three per device --------------------------------
  const int ngpus = 4;
  dpct::queue_ptr compute_stream[ngpus];
  dpct::queue_ptr halo_stream_bottom[ngpus];
  dpct::queue_ptr halo_stream_top[ngpus];

  for (int g = 0; g < ngpus; ++g) {
    /*
    DPCT1093:246: The "g" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(g)));
    CUDA_CHECK(DPCT_CHECK_ERROR(compute_stream[g] =
                                    dpct::get_current_device().create_queue()));
    CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[g] =
                                    dpct::get_current_device().create_queue()));
    CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[g] =
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
  // PHASE A: interior on compute_stream, boundary kernels on their interface
  //   streams, then D2H from the contiguous boundary planes on those same
  //   streams -- no pack kernel, and no event needed for ordering. Each
  //   thread synchronises only its OWN halo streams.
  //
  //   barrier: every host halo buffer must be filled before any device reads
  //            a neighbour's buffer.
  //
  // PHASE B: H2D into the contiguous ghost planes -- no unpack.
  //
  //   barrier: every device must be idle before the pointer swap.
  //
  // The ghost planes at GPU0 k=0 and GPU3 k=nz3-1 are physical domain
  // boundaries. They are initialised to zero and never written.
  // ======================================================================

#pragma omp parallel num_threads(ngpus)
  {
    int tid = omp_get_thread_num();

    // Each host thread owns one device: set once, never switch.
    /*
    DPCT1093:247: The "tid" device may be not the one intended for use. Adjust
    the selected device if needed.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(tid)));

    while (t < T) {

      // -- PHASE A: compute + D2H ---------------------------------
      if (tid == 0) {
        // GPU0: no bottom boundary -- k=1 is never sent.
        /*
        DPCT1049:37: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
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
        DPCT1010:248: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);
        /*
        DPCT1049:38: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(halo_stream_top[0]->get_device(),
                                       {sycl::aspect::fp64});

          halo_stream_top[0]->parallel_for(
              sycl::nd_range<3>(grid_boundary * block_boundary, block_boundary),
              [=](sycl::nd_item<3> item_ct1) {
                heat_update_boundary_top(d0_u_new, d0_u_old, d0_rhs, d0_kappa,
                                         nz0, ny, nx, factor, item_ct1);
              });
        }
        /*
        DPCT1010:249: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);

        /*
        DPCT1124:250: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[0]->memcpy(
            h_halo0_up, d0_u_new + (size_t)(nz0 - 2) * ny * nx, halo_bytes)));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[0]->wait()));
      }
      else if (tid == 1) {
        /*
        DPCT1049:39: The work-group size passed to the SYCL kernel may exceed
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
        DPCT1010:251: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);
        /*
        DPCT1049:40: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(halo_stream_bottom[1]->get_device(),
                                       {sycl::aspect::fp64});

          halo_stream_bottom[1]->parallel_for(
              sycl::nd_range<3>(grid_boundary * block_boundary, block_boundary),
              [=](sycl::nd_item<3> item_ct1) {
                heat_update_boundary_bottom(d1_u_new, d1_u_old, d1_rhs,
                                            d1_kappa, nz1, ny, nx, factor,
                                            item_ct1);
              });
        }
        /*
        DPCT1010:252: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);
        /*
        DPCT1049:41: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(halo_stream_top[1]->get_device(),
                                       {sycl::aspect::fp64});

          halo_stream_top[1]->parallel_for(
              sycl::nd_range<3>(grid_boundary * block_boundary, block_boundary),
              [=](sycl::nd_item<3> item_ct1) {
                heat_update_boundary_top(d1_u_new, d1_u_old, d1_rhs, d1_kappa,
                                         nz1, ny, nx, factor, item_ct1);
              });
        }
        /*
        DPCT1010:253: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);

        /*
        DPCT1124:254: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[1]->memcpy(
            h_halo1_dn, d1_u_new + (size_t)1 * ny * nx, halo_bytes)));
        /*
        DPCT1124:255: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[1]->memcpy(
            h_halo1_up, d1_u_new + (size_t)(nz1 - 2) * ny * nx, halo_bytes)));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[1]->wait()));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[1]->wait()));
      }
      else if (tid == 2) {
        /*
        DPCT1049:42: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(compute_stream[2]->get_device(),
                                       {sycl::aspect::fp64});

          compute_stream[2]->parallel_for(
              sycl::nd_range<3>(grid2_interior * block, block),
              [=](sycl::nd_item<3> item_ct1) {
                heat_update_interior(d2_u_new, d2_u_old, d2_rhs, d2_kappa, nz2,
                                     ny, nx, factor, k_start2, k_end2,
                                     item_ct1);
              });
        }
        /*
        DPCT1010:256: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);
        /*
        DPCT1049:43: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(halo_stream_bottom[2]->get_device(),
                                       {sycl::aspect::fp64});

          halo_stream_bottom[2]->parallel_for(
              sycl::nd_range<3>(grid_boundary * block_boundary, block_boundary),
              [=](sycl::nd_item<3> item_ct1) {
                heat_update_boundary_bottom(d2_u_new, d2_u_old, d2_rhs,
                                            d2_kappa, nz2, ny, nx, factor,
                                            item_ct1);
              });
        }
        /*
        DPCT1010:257: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);
        /*
        DPCT1049:44: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(halo_stream_top[2]->get_device(),
                                       {sycl::aspect::fp64});

          halo_stream_top[2]->parallel_for(
              sycl::nd_range<3>(grid_boundary * block_boundary, block_boundary),
              [=](sycl::nd_item<3> item_ct1) {
                heat_update_boundary_top(d2_u_new, d2_u_old, d2_rhs, d2_kappa,
                                         nz2, ny, nx, factor, item_ct1);
              });
        }
        /*
        DPCT1010:258: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);

        /*
        DPCT1124:259: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[2]->memcpy(
            h_halo2_dn, d2_u_new + (size_t)1 * ny * nx, halo_bytes)));
        /*
        DPCT1124:260: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[2]->memcpy(
            h_halo2_up, d2_u_new + (size_t)(nz2 - 2) * ny * nx, halo_bytes)));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[2]->wait()));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[2]->wait()));
      }
      else {
        // GPU3: no top boundary -- k=nz3-2 is never sent.
        /*
        DPCT1049:45: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(compute_stream[3]->get_device(),
                                       {sycl::aspect::fp64});

          compute_stream[3]->parallel_for(
              sycl::nd_range<3>(grid3_interior * block, block),
              [=](sycl::nd_item<3> item_ct1) {
                heat_update_interior(d3_u_new, d3_u_old, d3_rhs, d3_kappa, nz3,
                                     ny, nx, factor, k_start3, k_end3,
                                     item_ct1);
              });
        }
        /*
        DPCT1010:261: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);
        /*
        DPCT1049:46: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
          dpct::has_capability_or_fail(halo_stream_bottom[3]->get_device(),
                                       {sycl::aspect::fp64});

          halo_stream_bottom[3]->parallel_for(
              sycl::nd_range<3>(grid_boundary * block_boundary, block_boundary),
              [=](sycl::nd_item<3> item_ct1) {
                heat_update_boundary_bottom(d3_u_new, d3_u_old, d3_rhs,
                                            d3_kappa, nz3, ny, nx, factor,
                                            item_ct1);
              });
        }
        /*
        DPCT1010:262: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);

        /*
        DPCT1124:263: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[3]->memcpy(
            h_halo3_dn, d3_u_new + (size_t)1 * ny * nx, halo_bytes)));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[3]->wait()));
      }

      // Every host halo buffer is filled -- safe to cross-exchange.
#pragma omp barrier

      // -- PHASE B: H2D into the ghost planes ---------------------
      if (tid == 0) {
        /*
        DPCT1124:264: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[0]->memcpy(
            d0_u_new + (size_t)(nz0 - 1) * ny * nx, h_halo1_dn, halo_bytes)));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[0]->wait()));
        CUDA_CHECK(DPCT_CHECK_ERROR(compute_stream[0]->wait()));
      }
      else if (tid == 1) {
        /*
        DPCT1124:265: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[1]->memcpy(
            d1_u_new + (size_t)0 * ny * nx, h_halo0_up, halo_bytes)));
        /*
        DPCT1124:266: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[1]->memcpy(
            d1_u_new + (size_t)(nz1 - 1) * ny * nx, h_halo2_dn, halo_bytes)));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[1]->wait()));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[1]->wait()));
        CUDA_CHECK(DPCT_CHECK_ERROR(compute_stream[1]->wait()));
      }
      else if (tid == 2) {
        /*
        DPCT1124:267: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[2]->memcpy(
            d2_u_new + (size_t)0 * ny * nx, h_halo1_up, halo_bytes)));
        /*
        DPCT1124:268: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[2]->memcpy(
            d2_u_new + (size_t)(nz2 - 1) * ny * nx, h_halo3_dn, halo_bytes)));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[2]->wait()));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_top[2]->wait()));
        CUDA_CHECK(DPCT_CHECK_ERROR(compute_stream[2]->wait()));
      }
      else {
        /*
        DPCT1124:269: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[3]->memcpy(
            d3_u_new + (size_t)0 * ny * nx, h_halo2_up, halo_bytes)));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream_bottom[3]->wait()));
        CUDA_CHECK(DPCT_CHECK_ERROR(compute_stream[3]->wait()));
      }

      // Every device is idle -- safe to swap.
#pragma omp barrier

      // The implicit barrier at the end of `single` is what publishes the
      // updated `t` to all four threads before the loop condition is re-read.
#pragma omp single
      {
        double *tmp;
        tmp = d0_u_old; d0_u_old = d0_u_new; d0_u_new = tmp;
        tmp = d1_u_old; d1_u_old = d1_u_new; d1_u_new = tmp;
        tmp = d2_u_old; d2_u_old = d2_u_new; d2_u_new = tmp;
        tmp = d3_u_old; d3_u_old = d3_u_new; d3_u_new = tmp;
        steps++;
        t += dt;
      }

    } // end while
  }   // end omp parallel

  // Every stream is drained inside the loop, so all four devices are already
  // idle. Make it explicit so the timed region cannot drift if the internal
  // synchronisation is ever relaxed.
  for (int g = 0; g < ngpus; ++g) {
    /*
    DPCT1093:270: The "g" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(g)));
    CUDA_CHECK(
        DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
  }

  compute_timer += omp_get_wtime();

  printf("\n%d timesteps complete\n", steps);
  printf("Time = %f seconds\n", compute_timer);
  printf("=== SIMULATION ENDS ===\n");

  // -- Copy back (layout [k][j][i]) ------------------------------
  /*
  DPCT1093:271: The "0" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(0)));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(h_u0, d0_u_old, size0).wait()));
  /*
  DPCT1093:272: The "1" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(1)));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(h_u1, d1_u_old, size1).wait()));
  /*
  DPCT1093:273: The "2" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(2)));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(h_u2, d2_u_old, size2).wait()));
  /*
  DPCT1093:274: The "3" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(3)));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(h_u3, d3_u_old, size3).wait()));

  // -- Reconstruct global solution (global output is [k][j][i]) --
  memset(h_u_global, 0, global_bytes);

  gather_partition(h_u_global, h_u0, nz0, gk_start0, ng, 0,         gk_start1, n, ny, nx);
  gather_partition(h_u_global, h_u1, nz1, gk_start1, ng, gk_start1, gk_start2, n, ny, nx);
  gather_partition(h_u_global, h_u2, nz2, gk_start2, ng, gk_start2, gk_start3, n, ny, nx);
  gather_partition(h_u_global, h_u3, nz3, gk_start3, ng, gk_start3, n,         n, ny, nx);

  if (should_write_txt())
    write_u_values("u_cuda_stream_omp_4gpu.txt", h_u_global, n);

  // -- Cleanup streams -------------------------------------------
  for (int g = 0; g < ngpus; ++g) {
    /*
    DPCT1093:275: The "g" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(g)));
    CUDA_CHECK(DPCT_CHECK_ERROR(
        dpct::get_current_device().destroy_queue(compute_stream[g])));
    CUDA_CHECK(DPCT_CHECK_ERROR(
        dpct::get_current_device().destroy_queue(halo_stream_bottom[g])));
    CUDA_CHECK(DPCT_CHECK_ERROR(
        dpct::get_current_device().destroy_queue(halo_stream_top[g])));
  }

  // -- Cleanup ---------------------------------------------------
  /*
  DPCT1093:276: The "0" device may be not the one intended for use. Adjust the
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
  DPCT1093:277: The "1" device may be not the one intended for use. Adjust the
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

  /*
  DPCT1093:278: The "2" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(2)));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(d2_u_old, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(
          dpct::dpct_free(d2_u_new, dpct::get_in_order_queue())));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(d2_rhs, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(
          dpct::dpct_free(d2_kappa, dpct::get_in_order_queue())));

  /*
  DPCT1093:279: The "3" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(3)));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(d3_u_old, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(
          dpct::dpct_free(d3_u_new, dpct::get_in_order_queue())));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(d3_rhs, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(
          dpct::dpct_free(d3_kappa, dpct::get_in_order_queue())));

  free(h_u_global);
  free(h_u0);     free(h_u1);     free(h_u2);     free(h_u3);
  free(h_rhs0);   free(h_rhs1);   free(h_rhs2);   free(h_rhs3);
  free(h_kappa0); free(h_kappa1); free(h_kappa2); free(h_kappa3);
  free(h_halo0_up); free(h_halo1_dn);
  free(h_halo1_up); free(h_halo2_dn);
  free(h_halo2_up); free(h_halo3_dn);

  printf("\n**** FOUR GPUS CUDA, ASYNC + 4 HOST THREADS FINISHED ****\n");
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
