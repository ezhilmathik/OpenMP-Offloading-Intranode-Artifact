#include "hip/hip_runtime.h"
//-*-c++-*-
/* CUDA implementation of the 3D heat conduction problem
         du/dt = div ( kappa(x,y,z) grad u ) + f(x,y,z)
   where kappa(x,y,z) = 0.5+0.45*sin(pi*x)*sin(pi*y)*sin(pi*z),
   using explicit time stepping.

   TWO-GPU VERSION, halo exchange staged through the host.
   ONE HOST THREAD drives both devices.

   Layout: u[k][j][i]  ->  k is slowest, i is fastest
     offset = k*(ny*nx) + j*nx + i
   Decomposition along the k axis, so an XY plane at fixed k is CONTIGUOUS
   and no pack/unpack kernels are needed:
     D2H: hipMemcpy(h_halo, d_u_new + k*(ny*nx), halo_bytes, D2H)
     H2D: hipMemcpy(d_u_new + ghost_k*(ny*nx), h_halo, halo_bytes, H2D)
   Neighbour offsets:
     i+-1  ->  +-1
     j+-1  ->  +-nx
     k+-1  ->  +-ny*nx

   This file is kept deliberately parallel to 2-openmp.c: identical
   decomposition, identical kernel split, identical launch order, identical
   prints, identical timing. A diff against 2-openmp.c should show only what
   is intrinsic to using CUDA instead of OpenMP target offload.

   KERNEL SPLIT
   ------------
   The split is by INTERFACE, not by position:

     heat_update_boundary_bottom   k = 1        only where k=1 is SENT
     heat_update_boundary_top      k = nz-2     only where k=nz-2 is SENT
     heat_update_interior          k_start..k_end   everything else

   A boundary kernel exists so that an EXCHANGED plane finishes early and
   its halo copy can be issued without waiting for the bulk of the domain.
   A plane that is never sent has no such urgency, so folding it into the
   interior kernel is strictly better: it keeps the boundary kernel as small
   as possible, and the boundary kernel is what gates the exchange.

   With two devices there is ONE interface, so each device sends exactly one
   plane and runs exactly one boundary kernel:

     GPU0 has no lower neighbour -> its plane k=1 is folded into the
          interior (k_start=1); it runs TOP only.
     GPU1 has no upper neighbour -> its plane k=nz1-2 is folded into the
          interior (k_end=nz1-2); it runs BOTTOM only.

   Per device the union of the kernels is k = 1 .. nz-2, exactly the owned
   planes:

     GPU0   interior 1 .. nz0-3   + top       = 1 .. nz0-2
     GPU1   interior 2 .. nz1-2   + bottom    = 1 .. nz1-2

   LAUNCH ORDER
   ------------
   The interior kernel is launched BEFORE the boundary kernel on each
   device, matching 2-openmp.c. In THIS file the order is immaterial: the
   synchronous copies below drain every kernel before any transfer is
   issued, so nothing is gated on a boundary kernel finishing early.

   HALO BUFFER NAMING
   ------------------
   Buffers are indexed by sender and direction, as in 2-openmp.c:

     h_haloN_up   plane device N sends UPWARD   (to device N+1)
     h_haloN_dn   plane device N sends DOWNWARD (to device N-1)

   With two devices only h_halo0_up and h_halo1_dn exist -- one interface,
   two directions.

   BUILD NOTE: two flags are mandatory for a valid comparison.
     --fmad=false         nvcc equivalent of clang's -ffp-contract=off. Without
                          it nvcc contracts a*b+c into FMA and the result will
                          NOT match the OpenMP/serial reference bit-for-bit.
     -Xcompiler -fopenmp  needed only for omp_get_wtime(), so this file times
                          the loop with the SAME function 2-openmp.c uses.
                          No OpenMP parallelism is used here.

     nvcc -O3 -arch=sm_90 --fmad=false -Xcompiler -fopenmp 2-cuda.cu -lm -o 2-cuda
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>   /* strcasecmp */
#include <math.h>
#include <omp.h>
#include <hip/hip_runtime.h>

#define IDX(k, j, i, ny, nx)  ((size_t)(k)*(size_t)(ny)*(size_t)(nx) \
                             + (size_t)(j)*(size_t)(nx) + (size_t)(i))

#define CUDA_CHECK(call)                                                \
  do {                                                                  \
    hipError_t err = (call);                                           \
    if (err != hipSuccess) {                                           \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
              __FILE__, __LINE__, hipGetErrorString(err));             \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
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
// Identical to 2-openmp.c. Boundary values are forced to exact zero.
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
//
// threadIdx.x -> i (coalesced, full warp on contiguous addresses),
// threadIdx.y -> j.
__global__
void heat_update_boundary_bottom(double * __restrict__ u_new,
                                 const double * __restrict__ u_old,
                                 const double * __restrict__ rhs,
                                 const double * __restrict__ kappa,
                                 int nz, int ny, int nx, double factor)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

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
__global__
void heat_update_boundary_top(double * __restrict__ u_new,
                              const double * __restrict__ u_old,
                              const double * __restrict__ rhs,
                              const double * __restrict__ kappa,
                              int nz, int ny, int nx, double factor)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;

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
__global__
void heat_update_interior(double * __restrict__ u_new,
                          const double * __restrict__ u_old,
                          const double * __restrict__ rhs,
                          const double * __restrict__ kappa,
                          int nz, int ny, int nx, double factor,
                          int k_start, int k_end)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + k_start;

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


int main(int argc, char** argv)
{
  setvbuf(stdout, NULL, _IONBF, 0);  /* unbuffered: never lose output on a fault */

  printf("\n**** TWO GPUS HIP STARTS ****\n");

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
  CUDA_CHECK(hipGetDeviceCount(&num_devices));

  if (num_devices < 2) {
    fprintf(stderr, "ERROR: this version needs 2 HIP devices, %d present\n",
            num_devices);
    exit(EXIT_FAILURE);
  }

  printf("Using HIP devices 0 and 1 of %d\n", num_devices);

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

  CUDA_CHECK(hipSetDevice(0));
  CUDA_CHECK(hipMalloc(&d0_u_old, size0));
  CUDA_CHECK(hipMalloc(&d0_u_new, size0));
  CUDA_CHECK(hipMalloc(&d0_rhs,   size0));
  CUDA_CHECK(hipMalloc(&d0_kappa, size0));

  // u_new is seeded from the same array as u_old so its BOUNDARY values are
  // the exact zeros above -- the kernels only ever write interior points.
  CUDA_CHECK(hipMemcpy(d0_u_old, h_u0,     size0, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d0_u_new, h_u0,     size0, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d0_rhs,   h_rhs0,   size0, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d0_kappa, h_kappa0, size0, hipMemcpyHostToDevice));

  // -- GPU 1 device allocation + H2D ----------------------------
  double *d1_u_old, *d1_u_new, *d1_rhs, *d1_kappa;

  CUDA_CHECK(hipSetDevice(1));
  CUDA_CHECK(hipMalloc(&d1_u_old, size1));
  CUDA_CHECK(hipMalloc(&d1_u_new, size1));
  CUDA_CHECK(hipMalloc(&d1_rhs,   size1));
  CUDA_CHECK(hipMalloc(&d1_kappa, size1));

  CUDA_CHECK(hipMemcpy(d1_u_old, h_u1,     size1, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d1_u_new, h_u1,     size1, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d1_rhs,   h_rhs1,   size1, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d1_kappa, h_kappa1, size1, hipMemcpyHostToDevice));

  // -- Host halo staging buffers ---------------------------------
  // In [k][j][i] layout XY planes are contiguous, so D2H/H2D go directly
  // via pointer offset. No device halo buffers needed.
  // Two buffers: one interface, two directions.
  double *h_halo0_up = (double*)malloc(halo_bytes);  // dev0 -> dev1
  double *h_halo1_dn = (double*)malloc(halo_bytes);  // dev1 -> dev0

  if (!h_halo0_up || !h_halo1_dn) {
    fprintf(stderr, "ERROR: malloc failed (halo host)\n");
    return 1;
  }

  // -- Kernel configs -------------------------------------------
  // Interior: k range is [k_start, k_end], so grid.z covers k_end-k_start+1.
  //   GPU0 interior 1 .. nz0-3   ->  nz0-3 planes
  //   GPU1 interior 2 .. nz1-2   ->  nz1-3 planes
  dim3 block(32, 4, 2);

  int k_start0 = 1,  k_end0 = nz0 - 3;
  int k_start1 = 2,  k_end1 = nz1 - 2;

  dim3 grid0_interior((nx-2 + block.x - 1)/block.x,
                      (ny-2 + block.y - 1)/block.y,
                      (k_end0 - k_start0 + 1 + block.z - 1)/block.z);

  dim3 grid1_interior((nx-2 + block.x - 1)/block.x,
                      (ny-2 + block.y - 1)/block.y,
                      (k_end1 - k_start1 + 1 + block.z - 1)/block.z);

  // Boundary: a single XY plane, so the grid is 2D.
  dim3 block_boundary(32, 8, 1);
  dim3 grid_boundary((nx-2 + block_boundary.x - 1)/block_boundary.x,
                     (ny-2 + block_boundary.y - 1)/block_boundary.y,
                     1);

  double t     = 0.0;
  int    steps = 0;

  printf("\n=== SIMULATION STARTS ===\n");

  double compute_timer = 0.0;
  compute_timer -= omp_get_wtime();

  // ======================================================================
  // Time loop
  //
  // PHASE 1: each GPU computes, then D2H directly from the contiguous
  //   boundary plane -- no pack kernel needed.
  //
  // PHASE 2: H2D directly into the contiguous ghost plane -- no unpack.
  //
  // Transfer map (two transfers, one interface):
  //   GPU0 plane k=nz0-2  ->  h_halo0_up  ->  GPU1 ghost k=0
  //   GPU1 plane k=1      ->  h_halo1_dn  ->  GPU0 ghost k=nz0-1
  //
  // The ghost planes at GPU0 k=0 and GPU1 k=nz1-1 are physical domain
  // boundaries. They are initialised to zero and never written.
  // ======================================================================

  while (t < T) {

    // -- GPU 0: compute (no bottom boundary -- k=1 is never sent) -
    CUDA_CHECK(hipSetDevice(0));
    heat_update_interior<<<grid0_interior, block>>>(
      d0_u_new, d0_u_old, d0_rhs, d0_kappa, nz0, ny, nx, factor,
      k_start0, k_end0);
    CUDA_CHECK(hipGetLastError());
    heat_update_boundary_top<<<grid_boundary, block_boundary>>>(
      d0_u_new, d0_u_old, d0_rhs, d0_kappa, nz0, ny, nx, factor);
    CUDA_CHECK(hipGetLastError());

    // -- GPU 1: compute (no top boundary -- k=nz1-2 is never sent) -
    CUDA_CHECK(hipSetDevice(1));
    heat_update_interior<<<grid1_interior, block>>>(
      d1_u_new, d1_u_old, d1_rhs, d1_kappa, nz1, ny, nx, factor,
      k_start1, k_end1);
    CUDA_CHECK(hipGetLastError());
    heat_update_boundary_bottom<<<grid_boundary, block_boundary>>>(
      d1_u_new, d1_u_old, d1_rhs, d1_kappa, nz1, ny, nx, factor);
    CUDA_CHECK(hipGetLastError());

    // -- PHASE 1: D2H from contiguous boundary planes -------------
    CUDA_CHECK(hipSetDevice(0));
    CUDA_CHECK(hipMemcpy(h_halo0_up,
                          d0_u_new + (size_t)(nz0-2) * ny * nx,
                          halo_bytes, hipMemcpyDeviceToHost));

    CUDA_CHECK(hipSetDevice(1));
    CUDA_CHECK(hipMemcpy(h_halo1_dn,
                          d1_u_new + (size_t)1 * ny * nx,
                          halo_bytes, hipMemcpyDeviceToHost));

    // -- PHASE 2: H2D into contiguous ghost planes ----------------
    CUDA_CHECK(hipSetDevice(0));
    CUDA_CHECK(hipMemcpy(d0_u_new + (size_t)(nz0-1) * ny * nx,
                          h_halo1_dn,
                          halo_bytes, hipMemcpyHostToDevice));

    CUDA_CHECK(hipSetDevice(1));
    CUDA_CHECK(hipMemcpy(d1_u_new + (size_t)0 * ny * nx,
                          h_halo0_up,
                          halo_bytes, hipMemcpyHostToDevice));

    // -- Pointer swap --------------------------------------------
    double *tmp;
    tmp = d0_u_old; d0_u_old = d0_u_new; d0_u_new = tmp;
    tmp = d1_u_old; d1_u_old = d1_u_new; d1_u_new = tmp;

    steps++;
    t += dt;
  }

  // 2-openmp.c drains every kernel with `taskwait` and its copies are
  // blocking, so its timer closes on quiesced devices. The hipMemcpy calls
  // above are synchronous, so both devices are already idle here -- but make
  // it explicit rather than relying on that, so the timed region cannot drift
  // if the copies ever become async.
  CUDA_CHECK(hipSetDevice(0)); CUDA_CHECK(hipDeviceSynchronize());
  CUDA_CHECK(hipSetDevice(1)); CUDA_CHECK(hipDeviceSynchronize());

  compute_timer += omp_get_wtime();

  printf("\n%d timesteps complete\n", steps);
  printf("Time = %f seconds\n", compute_timer);
  printf("=== SIMULATION ENDS ===\n");

  // -- Copy back (layout [k][j][i]) ------------------------------
  CUDA_CHECK(hipSetDevice(0));
  CUDA_CHECK(hipMemcpy(h_u0, d0_u_old, size0, hipMemcpyDeviceToHost));
  CUDA_CHECK(hipSetDevice(1));
  CUDA_CHECK(hipMemcpy(h_u1, d1_u_old, size1, hipMemcpyDeviceToHost));

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
    write_u_values("u_hip_2gpu.txt", h_u_global, n);

  // -- Cleanup ---------------------------------------------------
  CUDA_CHECK(hipSetDevice(0));
  CUDA_CHECK(hipFree(d0_u_old)); CUDA_CHECK(hipFree(d0_u_new));
  CUDA_CHECK(hipFree(d0_rhs));   CUDA_CHECK(hipFree(d0_kappa));

  CUDA_CHECK(hipSetDevice(1));
  CUDA_CHECK(hipFree(d1_u_old)); CUDA_CHECK(hipFree(d1_u_new));
  CUDA_CHECK(hipFree(d1_rhs));   CUDA_CHECK(hipFree(d1_kappa));

  free(h_u_global);
  free(h_u0);     free(h_u1);
  free(h_rhs0);   free(h_rhs1);
  free(h_kappa0); free(h_kappa1);
  free(h_halo0_up); free(h_halo1_dn);

  printf("\n**** TWO GPUS HIP FINISHED ****\n");
  return 0;
}