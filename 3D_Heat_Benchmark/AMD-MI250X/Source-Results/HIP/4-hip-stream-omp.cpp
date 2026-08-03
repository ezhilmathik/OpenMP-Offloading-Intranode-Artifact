#include "hip/hip_runtime.h"
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
   One OpenMP thread per device. Each thread calls hipSetDevice() ONCE and
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
// Computes k=nz-2, the plane sent UPWARD. Used by GPU 0, 1 and 2.
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

  printf("\n**** FOUR GPUS HIP, ASYNC + 4 HOST THREADS STARTS ****\n");

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

  if (num_devices < 4) {
    fprintf(stderr, "ERROR: this version needs 4 HIP devices, %d present\n",
            num_devices);
    exit(EXIT_FAILURE);
  }

  printf("Using HIP devices 0, 1, 2 and 3 of %d\n", num_devices);

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

  CUDA_CHECK(hipSetDevice(0));
  CUDA_CHECK(hipMalloc(&d0_u_old, size0)); CUDA_CHECK(hipMalloc(&d0_u_new, size0));
  CUDA_CHECK(hipMalloc(&d0_rhs,   size0)); CUDA_CHECK(hipMalloc(&d0_kappa, size0));
  CUDA_CHECK(hipMemcpy(d0_u_old, h_u0,     size0, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d0_u_new, h_u0,     size0, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d0_rhs,   h_rhs0,   size0, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d0_kappa, h_kappa0, size0, hipMemcpyHostToDevice));

  CUDA_CHECK(hipSetDevice(1));
  CUDA_CHECK(hipMalloc(&d1_u_old, size1)); CUDA_CHECK(hipMalloc(&d1_u_new, size1));
  CUDA_CHECK(hipMalloc(&d1_rhs,   size1)); CUDA_CHECK(hipMalloc(&d1_kappa, size1));
  CUDA_CHECK(hipMemcpy(d1_u_old, h_u1,     size1, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d1_u_new, h_u1,     size1, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d1_rhs,   h_rhs1,   size1, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d1_kappa, h_kappa1, size1, hipMemcpyHostToDevice));

  CUDA_CHECK(hipSetDevice(2));
  CUDA_CHECK(hipMalloc(&d2_u_old, size2)); CUDA_CHECK(hipMalloc(&d2_u_new, size2));
  CUDA_CHECK(hipMalloc(&d2_rhs,   size2)); CUDA_CHECK(hipMalloc(&d2_kappa, size2));
  CUDA_CHECK(hipMemcpy(d2_u_old, h_u2,     size2, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d2_u_new, h_u2,     size2, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d2_rhs,   h_rhs2,   size2, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d2_kappa, h_kappa2, size2, hipMemcpyHostToDevice));

  CUDA_CHECK(hipSetDevice(3));
  CUDA_CHECK(hipMalloc(&d3_u_old, size3)); CUDA_CHECK(hipMalloc(&d3_u_new, size3));
  CUDA_CHECK(hipMalloc(&d3_rhs,   size3)); CUDA_CHECK(hipMalloc(&d3_kappa, size3));
  CUDA_CHECK(hipMemcpy(d3_u_old, h_u3,     size3, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d3_u_new, h_u3,     size3, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d3_rhs,   h_rhs3,   size3, hipMemcpyHostToDevice));
  CUDA_CHECK(hipMemcpy(d3_kappa, h_kappa3, size3, hipMemcpyHostToDevice));

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
  dim3 block(32, 4, 2);

  int k_start0 = 1,  k_end0 = nz0 - 3;
  int k_start1 = 2,  k_end1 = nz1 - 3;
  int k_start2 = 2,  k_end2 = nz2 - 3;
  int k_start3 = 2,  k_end3 = nz3 - 2;

  dim3 grid0_interior((nx-2 + block.x - 1)/block.x,
                      (ny-2 + block.y - 1)/block.y,
                      (k_end0 - k_start0 + 1 + block.z - 1)/block.z);
  dim3 grid1_interior((nx-2 + block.x - 1)/block.x,
                      (ny-2 + block.y - 1)/block.y,
                      (k_end1 - k_start1 + 1 + block.z - 1)/block.z);
  dim3 grid2_interior((nx-2 + block.x - 1)/block.x,
                      (ny-2 + block.y - 1)/block.y,
                      (k_end2 - k_start2 + 1 + block.z - 1)/block.z);
  dim3 grid3_interior((nx-2 + block.x - 1)/block.x,
                      (ny-2 + block.y - 1)/block.y,
                      (k_end3 - k_start3 + 1 + block.z - 1)/block.z);

  // Boundary: a single XY plane, so the grid is 2D.
  dim3 block_boundary(32, 8, 1);
  dim3 grid_boundary((nx-2 + block_boundary.x - 1)/block_boundary.x,
                     (ny-2 + block_boundary.y - 1)/block_boundary.y,
                     1);

  // -- Streams: three per device --------------------------------
  const int ngpus = 4;
  hipStream_t compute_stream[ngpus];
  hipStream_t halo_stream_bottom[ngpus];
  hipStream_t halo_stream_top[ngpus];

  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(hipSetDevice(g));
    CUDA_CHECK(hipStreamCreate(&compute_stream[g]));
    CUDA_CHECK(hipStreamCreate(&halo_stream_bottom[g]));
    CUDA_CHECK(hipStreamCreate(&halo_stream_top[g]));
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
    CUDA_CHECK(hipSetDevice(tid));

    while (t < T) {

      // -- PHASE A: compute + D2H ---------------------------------
      if (tid == 0) {
        // GPU0: no bottom boundary -- k=1 is never sent.
        heat_update_interior<<<grid0_interior, block, 0, compute_stream[0]>>>(
          d0_u_new, d0_u_old, d0_rhs, d0_kappa, nz0, ny, nx, factor,
          k_start0, k_end0);
        CUDA_CHECK(hipGetLastError());
        heat_update_boundary_top<<<grid_boundary, block_boundary, 0, halo_stream_top[0]>>>(
          d0_u_new, d0_u_old, d0_rhs, d0_kappa, nz0, ny, nx, factor);
        CUDA_CHECK(hipGetLastError());

        CUDA_CHECK(hipMemcpyAsync(h_halo0_up,
                                   d0_u_new + (size_t)(nz0-2) * ny * nx,
                                   halo_bytes, hipMemcpyDeviceToHost,
                                   halo_stream_top[0]));
        CUDA_CHECK(hipStreamSynchronize(halo_stream_top[0]));
      }
      else if (tid == 1) {
        heat_update_interior<<<grid1_interior, block, 0, compute_stream[1]>>>(
          d1_u_new, d1_u_old, d1_rhs, d1_kappa, nz1, ny, nx, factor,
          k_start1, k_end1);
        CUDA_CHECK(hipGetLastError());
        heat_update_boundary_bottom<<<grid_boundary, block_boundary, 0, halo_stream_bottom[1]>>>(
          d1_u_new, d1_u_old, d1_rhs, d1_kappa, nz1, ny, nx, factor);
        CUDA_CHECK(hipGetLastError());
        heat_update_boundary_top<<<grid_boundary, block_boundary, 0, halo_stream_top[1]>>>(
          d1_u_new, d1_u_old, d1_rhs, d1_kappa, nz1, ny, nx, factor);
        CUDA_CHECK(hipGetLastError());

        CUDA_CHECK(hipMemcpyAsync(h_halo1_dn,
                                   d1_u_new + (size_t)1 * ny * nx,
                                   halo_bytes, hipMemcpyDeviceToHost,
                                   halo_stream_bottom[1]));
        CUDA_CHECK(hipMemcpyAsync(h_halo1_up,
                                   d1_u_new + (size_t)(nz1-2) * ny * nx,
                                   halo_bytes, hipMemcpyDeviceToHost,
                                   halo_stream_top[1]));
        CUDA_CHECK(hipStreamSynchronize(halo_stream_bottom[1]));
        CUDA_CHECK(hipStreamSynchronize(halo_stream_top[1]));
      }
      else if (tid == 2) {
        heat_update_interior<<<grid2_interior, block, 0, compute_stream[2]>>>(
          d2_u_new, d2_u_old, d2_rhs, d2_kappa, nz2, ny, nx, factor,
          k_start2, k_end2);
        CUDA_CHECK(hipGetLastError());
        heat_update_boundary_bottom<<<grid_boundary, block_boundary, 0, halo_stream_bottom[2]>>>(
          d2_u_new, d2_u_old, d2_rhs, d2_kappa, nz2, ny, nx, factor);
        CUDA_CHECK(hipGetLastError());
        heat_update_boundary_top<<<grid_boundary, block_boundary, 0, halo_stream_top[2]>>>(
          d2_u_new, d2_u_old, d2_rhs, d2_kappa, nz2, ny, nx, factor);
        CUDA_CHECK(hipGetLastError());

        CUDA_CHECK(hipMemcpyAsync(h_halo2_dn,
                                   d2_u_new + (size_t)1 * ny * nx,
                                   halo_bytes, hipMemcpyDeviceToHost,
                                   halo_stream_bottom[2]));
        CUDA_CHECK(hipMemcpyAsync(h_halo2_up,
                                   d2_u_new + (size_t)(nz2-2) * ny * nx,
                                   halo_bytes, hipMemcpyDeviceToHost,
                                   halo_stream_top[2]));
        CUDA_CHECK(hipStreamSynchronize(halo_stream_bottom[2]));
        CUDA_CHECK(hipStreamSynchronize(halo_stream_top[2]));
      }
      else {
        // GPU3: no top boundary -- k=nz3-2 is never sent.
        heat_update_interior<<<grid3_interior, block, 0, compute_stream[3]>>>(
          d3_u_new, d3_u_old, d3_rhs, d3_kappa, nz3, ny, nx, factor,
          k_start3, k_end3);
        CUDA_CHECK(hipGetLastError());
        heat_update_boundary_bottom<<<grid_boundary, block_boundary, 0, halo_stream_bottom[3]>>>(
          d3_u_new, d3_u_old, d3_rhs, d3_kappa, nz3, ny, nx, factor);
        CUDA_CHECK(hipGetLastError());

        CUDA_CHECK(hipMemcpyAsync(h_halo3_dn,
                                   d3_u_new + (size_t)1 * ny * nx,
                                   halo_bytes, hipMemcpyDeviceToHost,
                                   halo_stream_bottom[3]));
        CUDA_CHECK(hipStreamSynchronize(halo_stream_bottom[3]));
      }

      // Every host halo buffer is filled -- safe to cross-exchange.
#pragma omp barrier

      // -- PHASE B: H2D into the ghost planes ---------------------
      if (tid == 0) {
        CUDA_CHECK(hipMemcpyAsync(d0_u_new + (size_t)(nz0-1) * ny * nx,
                                   h_halo1_dn, halo_bytes,
                                   hipMemcpyHostToDevice, halo_stream_top[0]));
        CUDA_CHECK(hipStreamSynchronize(halo_stream_top[0]));
        CUDA_CHECK(hipStreamSynchronize(compute_stream[0]));
      }
      else if (tid == 1) {
        CUDA_CHECK(hipMemcpyAsync(d1_u_new + (size_t)0 * ny * nx,
                                   h_halo0_up, halo_bytes,
                                   hipMemcpyHostToDevice, halo_stream_bottom[1]));
        CUDA_CHECK(hipMemcpyAsync(d1_u_new + (size_t)(nz1-1) * ny * nx,
                                   h_halo2_dn, halo_bytes,
                                   hipMemcpyHostToDevice, halo_stream_top[1]));
        CUDA_CHECK(hipStreamSynchronize(halo_stream_bottom[1]));
        CUDA_CHECK(hipStreamSynchronize(halo_stream_top[1]));
        CUDA_CHECK(hipStreamSynchronize(compute_stream[1]));
      }
      else if (tid == 2) {
        CUDA_CHECK(hipMemcpyAsync(d2_u_new + (size_t)0 * ny * nx,
                                   h_halo1_up, halo_bytes,
                                   hipMemcpyHostToDevice, halo_stream_bottom[2]));
        CUDA_CHECK(hipMemcpyAsync(d2_u_new + (size_t)(nz2-1) * ny * nx,
                                   h_halo3_dn, halo_bytes,
                                   hipMemcpyHostToDevice, halo_stream_top[2]));
        CUDA_CHECK(hipStreamSynchronize(halo_stream_bottom[2]));
        CUDA_CHECK(hipStreamSynchronize(halo_stream_top[2]));
        CUDA_CHECK(hipStreamSynchronize(compute_stream[2]));
      }
      else {
        CUDA_CHECK(hipMemcpyAsync(d3_u_new + (size_t)0 * ny * nx,
                                   h_halo2_up, halo_bytes,
                                   hipMemcpyHostToDevice, halo_stream_bottom[3]));
        CUDA_CHECK(hipStreamSynchronize(halo_stream_bottom[3]));
        CUDA_CHECK(hipStreamSynchronize(compute_stream[3]));
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
    CUDA_CHECK(hipSetDevice(g));
    CUDA_CHECK(hipDeviceSynchronize());
  }

  compute_timer += omp_get_wtime();

  printf("\n%d timesteps complete\n", steps);
  printf("Time = %f seconds\n", compute_timer);
  printf("=== SIMULATION ENDS ===\n");

  // -- Copy back (layout [k][j][i]) ------------------------------
  CUDA_CHECK(hipSetDevice(0));
  CUDA_CHECK(hipMemcpy(h_u0, d0_u_old, size0, hipMemcpyDeviceToHost));
  CUDA_CHECK(hipSetDevice(1));
  CUDA_CHECK(hipMemcpy(h_u1, d1_u_old, size1, hipMemcpyDeviceToHost));
  CUDA_CHECK(hipSetDevice(2));
  CUDA_CHECK(hipMemcpy(h_u2, d2_u_old, size2, hipMemcpyDeviceToHost));
  CUDA_CHECK(hipSetDevice(3));
  CUDA_CHECK(hipMemcpy(h_u3, d3_u_old, size3, hipMemcpyDeviceToHost));

  // -- Reconstruct global solution (global output is [k][j][i]) --
  memset(h_u_global, 0, global_bytes);

  gather_partition(h_u_global, h_u0, nz0, gk_start0, ng, 0,         gk_start1, n, ny, nx);
  gather_partition(h_u_global, h_u1, nz1, gk_start1, ng, gk_start1, gk_start2, n, ny, nx);
  gather_partition(h_u_global, h_u2, nz2, gk_start2, ng, gk_start2, gk_start3, n, ny, nx);
  gather_partition(h_u_global, h_u3, nz3, gk_start3, ng, gk_start3, n,         n, ny, nx);

  if (should_write_txt())
    write_u_values("u_hip_stream_omp_4gpu.txt", h_u_global, n);

  // -- Cleanup streams -------------------------------------------
  for (int g = 0; g < ngpus; ++g) {
    CUDA_CHECK(hipSetDevice(g));
    CUDA_CHECK(hipStreamDestroy(compute_stream[g]));
    CUDA_CHECK(hipStreamDestroy(halo_stream_bottom[g]));
    CUDA_CHECK(hipStreamDestroy(halo_stream_top[g]));
  }

  // -- Cleanup ---------------------------------------------------
  CUDA_CHECK(hipSetDevice(0));
  CUDA_CHECK(hipFree(d0_u_old)); CUDA_CHECK(hipFree(d0_u_new));
  CUDA_CHECK(hipFree(d0_rhs));   CUDA_CHECK(hipFree(d0_kappa));

  CUDA_CHECK(hipSetDevice(1));
  CUDA_CHECK(hipFree(d1_u_old)); CUDA_CHECK(hipFree(d1_u_new));
  CUDA_CHECK(hipFree(d1_rhs));   CUDA_CHECK(hipFree(d1_kappa));

  CUDA_CHECK(hipSetDevice(2));
  CUDA_CHECK(hipFree(d2_u_old)); CUDA_CHECK(hipFree(d2_u_new));
  CUDA_CHECK(hipFree(d2_rhs));   CUDA_CHECK(hipFree(d2_kappa));

  CUDA_CHECK(hipSetDevice(3));
  CUDA_CHECK(hipFree(d3_u_old)); CUDA_CHECK(hipFree(d3_u_new));
  CUDA_CHECK(hipFree(d3_rhs));   CUDA_CHECK(hipFree(d3_kappa));

  free(h_u_global);
  free(h_u0);     free(h_u1);     free(h_u2);     free(h_u3);
  free(h_rhs0);   free(h_rhs1);   free(h_rhs2);   free(h_rhs3);
  free(h_kappa0); free(h_kappa1); free(h_kappa2); free(h_kappa3);
  free(h_halo0_up); free(h_halo1_dn);
  free(h_halo1_up); free(h_halo2_dn);
  free(h_halo2_up); free(h_halo3_dn);

  printf("\n**** FOUR GPUS HIP, ASYNC + 4 HOST THREADS FINISHED ****\n");
  return 0;
}
