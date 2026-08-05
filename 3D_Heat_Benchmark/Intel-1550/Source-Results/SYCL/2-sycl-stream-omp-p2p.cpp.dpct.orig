//-*-c++-*-
/* CUDA implementation of the 3D heat conduction problem
         du/dt = div ( kappa(x,y,z) grad u ) + f(x,y,z)
   where kappa(x,y,z) = 0.5+0.45*sin(pi*x)*sin(pi*y)*sin(pi*z),
   using explicit time stepping.

   TWO-GPU VERSION, P2P + 2 HOST THREADS. Halo exchange goes DIRECTLY
   device-to-device via cudaMemcpyPeerAsync; the host is never touched.
   ONE HOST THREAD PER DEVICE.

   Layout: u[k][j][i]  ->  k is slowest, i is fastest
     offset = k*(ny*nx) + j*nx + i
   Decomposition along the k axis, so an XY plane at fixed k is CONTIGUOUS.
   cudaMemcpyPeerAsync addresses it directly by pointer offset:
     plane k  ->  base_ptr + k*(ny*nx)
   so there are no pack/unpack kernels AND no host staging buffers.
   Neighbour offsets:
     i+-1  ->  +-1
     j+-1  ->  +-nx
     k+-1  ->  +-ny*nx

   This file is kept deliberately parallel to 2-openmp-stream-omp-p2p.c.
   Against 2-cuda-stream-omp.cu the ONLY difference is the transfer path:
   two cudaMemcpyPeerAsync calls replace the D2H/H2D pair and the two host
   halo buffers disappear. Everything else -- decomposition, kernel split,
   launch order, host threading, prints, timing -- is identical, so the
   measured difference is the transfer path and nothing else.

   KERNEL SPLIT
   ------------
   The split is by INTERFACE, not by position:

     heat_update_boundary_bottom   k = 1        only where k=1 is SENT
     heat_update_boundary_top      k = nz-2     only where k=nz-2 is SENT
     heat_update_interior          k_start..k_end   everything else

   Here the correspondence is exact: each device's boundary kernel computes
   precisely the plane it sends, and the peer copy cannot be issued until
   that kernel has completed. Keeping the kernel to one plane keeps the
   critical path as short as possible.

     GPU0 has no lower neighbour -> plane k=1 folded into interior
          (k_start=1); runs TOP only, and sends k=nz0-2 upward.
     GPU1 has no upper neighbour -> plane k=nz1-2 folded into interior
          (k_end=nz1-2); runs BOTTOM only, and sends k=1 downward.

     GPU0   interior 1 .. nz0-3   + top       = 1 .. nz0-2
     GPU1   interior 2 .. nz1-2   + bottom    = 1 .. nz1-2

   WHY THE CONCURRENT PEER COPIES DO NOT RACE
   ------------------------------------------
   Both threads issue a peer copy that READS the other device's u_new while
   WRITING its own ghost plane:

     tid0   reads  GPU1 u_new k=1        writes GPU0 u_new k=nz0-1
     tid1   reads  GPU0 u_new k=nz0-2    writes GPU1 u_new k=0

   The plane read on each device is disjoint from the plane written there,
   so the two copies never touch the same memory. The interior kernel runs
   concurrently on compute_stream but reads u_old and writes only
   k_start..k_end, which excludes both ghost planes. The barrier before
   PHASE 2 guarantees both boundary kernels have completed, so the data
   being pulled is already final.

   NO HOST HALO BUFFERS
   --------------------
   2-cuda-stream.cu and 2-cuda-stream-omp.cu stage through plain malloc'd
   host memory (never pinned -- see their headers). This file has no host
   staging at all, which is the entire point of the variant.

   BUILD NOTE:
     nvcc -O3 -arch=sm_90 --fmad=false -Xcompiler -fopenmp 2-cuda-stream-omp-p2p.cu -lm -o 2-cuda-stream-omp-p2p
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
DPCT1009:128: SYCL uses exceptions to report errors and does not use the error
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
// Identical to 2-openmp-stream-omp-p2p.c. Boundaries forced to exact zero.
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

  printf("\n**** TWO GPUS CUDA, P2P + 2 HOST THREADS STARTS ****\n");

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

  // -- Enable P2P (required -- this variant has no host fallback) -
  int can01 = 0, can10 = 0;
  CUDA_CHECK(DPCT_CHECK_ERROR(
      can01 =
          dpct::dev_mgr::instance().get_device(0).ext_oneapi_can_access_peer(
              dpct::dev_mgr::instance().get_device(1))));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      can10 =
          dpct::dev_mgr::instance().get_device(1).ext_oneapi_can_access_peer(
              dpct::dev_mgr::instance().get_device(0))));
  if (!can01 || !can10) {
    fprintf(stderr, "ERROR: P2P not available between GPU 0 and GPU 1; "
                    "this version requires it\n");
    exit(EXIT_FAILURE);
  }

  /*
  DPCT1093:129: The "0" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(0)));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_current_device().ext_oneapi_enable_peer_access(
          dpct::dev_mgr::instance().get_device(1))));
  /*
  DPCT1093:130: The "1" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(1)));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_current_device().ext_oneapi_enable_peer_access(
          dpct::dev_mgr::instance().get_device(0))));
  printf("P2P enabled between GPU 0 and GPU 1\n");

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
  DPCT1093:131: The "0" device may be not the one intended for use. Adjust the
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
  DPCT1093:132: The "1" device may be not the one intended for use. Adjust the
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

  // -- NO host halo buffers ------------------------------------
  // In [k][j][i] layout each XY plane is already contiguous, and
  // cudaMemcpyPeerAsync addresses it directly by pointer offset. There is no
  // pack/unpack and no host staging -- that is the whole point of this file.

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
    DPCT1093:133: The "g" device may be not the one intended for use. Adjust the
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
  // PHASE 1: interior on compute_stream, boundary on halo_stream, then sync
  //   halo_stream so the SENT plane is final.
  //
  //   barrier: both boundary planes must be written before either peer copy
  //            reads the other device's memory.
  //
  // PHASE 2: cudaMemcpyPeerAsync straight into the ghost plane -- no host.
  //
  //   barrier: both devices must be idle before the pointer swap.
  //
  // Transfer map (two peer copies, one interface):
  //   GPU0 ghost k=nz0-1  <-  GPU1 plane k=1
  //   GPU1 ghost k=0      <-  GPU0 plane k=nz0-2
  //
  // The ghost planes at GPU0 k=0 and GPU1 k=nz1-1 are physical domain
  // boundaries. They are initialised to zero and never written.
  // ======================================================================

#pragma omp parallel num_threads(ngpus)
  {
    int tid = omp_get_thread_num();

    // Each host thread owns one device: set once, never switch.
    /*
    DPCT1093:134: The "tid" device may be not the one intended for use. Adjust
    the selected device if needed.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(tid)));

    while (t < T) {

      // -- PHASE 1: compute ---------------------------------------
      if (tid == 0) {
        // GPU0: no bottom boundary -- k=1 is never sent.
        /*
        DPCT1049:13: The work-group size passed to the SYCL kernel may exceed
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
        DPCT1010:135: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);
        /*
        DPCT1049:14: The work-group size passed to the SYCL kernel may exceed
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
        DPCT1010:136: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);

        // plane k=nz0-2 must be final before GPU1 pulls it
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream[0]->wait()));
      }
      else {
        // GPU1: no top boundary -- k=nz1-2 is never sent.
        /*
        DPCT1049:15: The work-group size passed to the SYCL kernel may exceed
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
        DPCT1010:137: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);
        /*
        DPCT1049:16: The work-group size passed to the SYCL kernel may exceed
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
        DPCT1010:138: SYCL uses exceptions to report errors and does not use the
        error codes. The call was replaced with 0. You need to rewrite this
        code.
        */
        CUDA_CHECK(0);

        // plane k=1 must be final before GPU0 pulls it
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream[1]->wait()));
      }

      // Both sent planes are final -- safe to peer-copy.
#pragma omp barrier

      // -- PHASE 2: peer copy straight into the ghost plane -------
      if (tid == 0) {
        /*
        DPCT1007:139: Migration of cudaMemcpyPeerAsync is not supported.
        */
        CUDA_CHECK(cudaMemcpyPeerAsync(
            d0_u_new + (size_t)(nz0 - 1) * ny * nx, 0, // dst: GPU0 top ghost
            d1_u_new + (size_t)1 * ny * nx, 1,         // src: GPU1 plane k=1
            halo_bytes, halo_stream[0]));
        CUDA_CHECK(DPCT_CHECK_ERROR(halo_stream[0]->wait()));
        CUDA_CHECK(DPCT_CHECK_ERROR(compute_stream[0]->wait()));
      }
      else {
        /*
        DPCT1007:140: Migration of cudaMemcpyPeerAsync is not supported.
        */
        CUDA_CHECK(cudaMemcpyPeerAsync(d1_u_new + (size_t)0 * ny * nx,
                                       1, // dst: GPU1 bottom ghost
                                       d0_u_new + (size_t)(nz0 - 2) * ny * nx,
                                       0, // src: GPU0 plane k=nz0-2
                                       halo_bytes, halo_stream[1]));
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
  DPCT1093:141: The "0" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(0))); CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));
  /*
  DPCT1093:142: The "1" device may be not the one intended for use. Adjust the
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
  DPCT1093:143: The "0" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(0)));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(h_u0, d0_u_old, size0).wait()));
  /*
  DPCT1093:144: The "1" device may be not the one intended for use. Adjust the
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
    write_u_values("u_cuda_stream_omp_p2p_2gpu.txt", h_u_global, n);

  // -- Cleanup streams -------------------------------------------
  for (int g = 0; g < ngpus; ++g) {
    /*
    DPCT1093:145: The "g" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(g)));
    CUDA_CHECK(DPCT_CHECK_ERROR(
        dpct::get_current_device().destroy_queue(compute_stream[g])));
    CUDA_CHECK(DPCT_CHECK_ERROR(
        dpct::get_current_device().destroy_queue(halo_stream[g])));
  }

  // -- Disable P2P -----------------------------------------------
  // Not CUDA_CHECK'd: teardown, and a failure here cannot affect the result
  // that was already computed and written.
  /*
  DPCT1093:146: The "0" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(0)));
      dpct::get_current_device().ext_oneapi_disable_peer_access(
          dpct::dev_mgr::instance().get_device(1));
  /*
  DPCT1093:147: The "1" device may be not the one intended for use. Adjust the
  selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(1)));
      dpct::get_current_device().ext_oneapi_disable_peer_access(
          dpct::dev_mgr::instance().get_device(0));

  // -- Cleanup ---------------------------------------------------
  /*
  DPCT1093:148: The "0" device may be not the one intended for use. Adjust the
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
  DPCT1093:149: The "1" device may be not the one intended for use. Adjust the
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

  printf("\n**** TWO GPUS CUDA, P2P + 2 HOST THREADS FINISHED ****\n");
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
