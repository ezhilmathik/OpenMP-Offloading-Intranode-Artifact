//-*-c++-*-
/* CUDA implementation of the 3D heat conduction problem
         du/dt = div ( kappa(x,y,z) grad u ) + f(x,y,z)
   where kappa(x,y,z) = 0.5+0.45*sin(pi*x)*sin(pi*y)*sin(pi*z),
   using explicit time stepping.

   SINGLE-GPU VERSION.

   Layout: u[k][j][i]  ->  k is slowest, i is fastest
     offset = k*(ny*nx) + j*nx + i
   Neighbour offsets:
     i+-1  ->  +-1
     j+-1  ->  +-nx
     k+-1  ->  +-ny*nx

   This file is kept deliberately parallel to 1-openmp.c: identical variable
   names, identical initialisation, identical prints, identical timing method.
   A diff against 1-openmp.c should show only what is intrinsic to using CUDA
   instead of OpenMP target offload.

   BUILD NOTE: two flags are mandatory for a valid comparison.
     --fmad=false        nvcc equivalent of clang's -ffp-contract=off. Without
                         it nvcc contracts a*b+c into FMA and the result will
                         NOT match the OpenMP/serial reference bit-for-bit.
     -Xcompiler -fopenmp  needed only for omp_get_wtime(), so that this file
                         times the loop with the SAME function 1-openmp.c uses.
                         No OpenMP parallelism is used here.

     nvcc -O3 -arch=sm_90 --fmad=false -Xcompiler -fopenmp 1-cuda.cu -lm -o 1-cuda
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define IDX(k, j, i, ny, nx)  ((size_t)(k)*(size_t)(ny)*(size_t)(nx) \
                             + (size_t)(j)*(size_t)(nx) + (size_t)(i))

/*
DPCT1009:58: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced by a placeholder string. You need to rewrite this
code.
*/
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    dpct::err0 err = call;                                                     \
                                                                               \
  } while (0)

/* Timing: omp_get_wtime(), the SAME function 1-openmp.c uses -- not merely an
   equivalent clock. libomp and libgomp do not implement omp_get_wtime() over
   the same underlying clock in every version, so using the identical call is
   the only way to guarantee the two sides measure the same quantity.

   Deliberately NOT cudaEvent: events time the GPU stream, whereas
   omp_get_wtime() around a blocking `target` region also includes host-side
   launch cost. Same function, same bracket, same semantics. */

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

// -- UPDATE kernel  (layout: u[k][j][i]) -------------------------------------
// Computes every owned point: k = 1 .. nz-2
void heat_update_kernel(double *u_new, const double *u_old,
                                   const double *rhs, const double *kappa,
                                   int nz, int ny, int nx, double factor,
                                   const sycl::nd_item<3> &item_ct1)
{
  int i = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
          item_ct1.get_local_id(2);
  int j = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
          item_ct1.get_local_id(1);
  int k = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
          item_ct1.get_local_id(0);

  if (k >= 1 && k < nz-1 && j >= 1 && j < ny-1 && i >= 1 && i < nx-1) {

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

// Host-side wrapper, mirroring heat_update() in 1-openmp.c.
static void heat_update(double *u_new, const double *u_old,
                        const double *rhs, const double *kappa,
                        int nz, int ny, int nx,
                        double factor, int device_id)
{
  (void)device_id;   /* single device: already current */

  sycl::range<3> block(8, 8, 8);
  sycl::range<3> grid((nz + block[0] - 1) / block[0],
                      (ny + block[1] - 1) / block[1],
                      (nx + block[2] - 1) / block[2]);

  /*
  DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  {
    dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                 {sycl::aspect::fp64});

    dpct::get_in_order_queue().parallel_for(
        sycl::nd_range<3>(grid * block, block), [=](sycl::nd_item<3> item_ct1) {
          heat_update_kernel(u_new, u_old, rhs, kappa, nz, ny, nx, factor,
                             item_ct1);
        });
  }
  /*
  DPCT1010:57: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  CUDA_CHECK(0);
}

int main(int argc, char **argv) try {
  printf("\n**** ONE GPU CUDA STARTS ****\n");

  int    n = (argc > 1) ? atoi(argv[1]) : 51;
  double T = (argc > 2) ? atof(argv[2]) : 1.0;

  int nx = n, ny = n, nz = n;
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
  int device_id   = 0;
  int num_devices = 0;
  CUDA_CHECK(
      DPCT_CHECK_ERROR(num_devices = dpct::dev_mgr::instance().device_count()));

  if (num_devices < 1) {
    fprintf(stderr, "ERROR: this version needs 1 CUDA device, %d present\n",
            num_devices);
    exit(EXIT_FAILURE);
  }

  /*
  DPCT1093:59: The "device_id" device may be not the one intended for use.
  Adjust the selected device if needed.
  */
  CUDA_CHECK(DPCT_CHECK_ERROR(dpct::select_device(device_id)));
  printf("Using CUDA device %d of %d\n", device_id, num_devices);

  // -- Partition sizes  (layout [k][j][i]) ----------------------
  size_t size0 = (size_t)nz*(size_t)ny*(size_t)nx*sizeof(double);

  printf("\n=== MEMORY ESTIMATE (LOCAL PER GPU) ===\n");
  printf("GPU0 local bytes per array: %.3f GB\n", size0 / 1e9);
  printf("======================================\n\n");

  // -- Host arrays -----------------------------------------------
  double *h_u_global = (double*)malloc(global_bytes);
  double *h_u0       = (double*)malloc(size0);
  double *h_rhs0     = (double*)malloc(size0);
  double *h_kappa0   = (double*)malloc(size0);

  if (!h_u_global || !h_u0 || !h_rhs0 || !h_kappa0) {
    fprintf(stderr, "ERROR: malloc failed (host arrays)\n");
    return 1;
  }

  // -- Initialize GPU 0 partition  (layout [k][j][i]) -----------
  // Boundary values are forced to exact zero.
  for (int k=0; k<nz; k++) {
    int gk = k;
    for (int j=0; j<ny; j++)
      for (int i=0; i<nx; i++) {
        size_t idx = IDX(k, j, i, ny, nx);
        h_u0[idx]     = (gk<=0||gk>=n-1||j<=0||j>=n-1||i<=0||i>=n-1)
                        ? 0.0 : sin(M_PI*i*h)*sin(M_PI*j*h)*sin(M_PI*gk*h);
        h_rhs0[idx]   = dt*(1.0+cos(M_PI*i*h)*cos(M_PI*j*h)*cos(M_PI*gk*h));
        h_kappa0[idx] = 0.5+0.45*sin(M_PI*i*h)*sin(M_PI*j*h)*sin(M_PI*gk*h);
      }
  }

  // -- GPU 0 device allocation + H2D ----------------------------
  double *d0_u_old, *d0_u_new, *d0_rhs, *d0_kappa;
  CUDA_CHECK(DPCT_CHECK_ERROR(d0_u_old = (double *)sycl::malloc_device(
                                  size0, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(d0_u_new = (double *)sycl::malloc_device(
                                  size0, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(d0_rhs = (double *)sycl::malloc_device(
                                  size0, dpct::get_in_order_queue())));
  CUDA_CHECK(DPCT_CHECK_ERROR(d0_kappa = (double *)sycl::malloc_device(
                                  size0, dpct::get_in_order_queue())));

  // u_new is seeded from the same array as u_old so its BOUNDARY values are
  // the exact zeros above -- the kernel only ever writes interior points.
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d0_u_old, h_u0, size0).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d0_u_new, h_u0, size0).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d0_rhs, h_rhs0, size0).wait()));
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(d0_kappa, h_kappa0, size0).wait()));

  double t     = 0.0;
  int    steps = 0;

  printf("\n=== SIMULATION STARTS ===\n");

  double compute_timer = 0.0;
  compute_timer -= omp_get_wtime();

  // ======================================================================
  // Time loop
  // ======================================================================

  while (t < T) {

    // -- GPU 0: compute ------------------------------------------
    heat_update(d0_u_new, d0_u_old, d0_rhs, d0_kappa,
                nz, ny, nx, factor, 0);

    // -- Pointer swap --------------------------------------------
    double *tmp;
    tmp = d0_u_old; d0_u_old = d0_u_new; d0_u_new = tmp;

    steps++;
    t += dt;
  }

  // The OpenMP version's `target` regions are blocking, so its timer closes on
  // a quiesced device. CUDA launches are asynchronous, so we must synchronise
  // before stopping the clock or we would be timing launch cost only.
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::get_current_device().queues_wait_and_throw()));

  compute_timer += omp_get_wtime();

  printf("\n%d timesteps complete\n", steps);
  printf("Time = %f seconds\n", compute_timer);
  printf("=== SIMULATION ENDS ===\n");

  // -- Copy back (layout [k][j][i]) ------------------------------
  CUDA_CHECK(DPCT_CHECK_ERROR(
      dpct::get_in_order_queue().memcpy(h_u0, d0_u_old, size0).wait()));

  // -- Reconstruct global solution (global output is [k][j][i]) --
  memset(h_u_global, 0, global_bytes);

  // GPU0 contribution
  for (int k=0; k<nz; k++) {
    int gk = k;
    for (int j=0; j<n; j++)
      for (int i=0; i<n; i++)
        h_u_global[IDX(gk, j, i, n, n)] = h_u0[IDX(k, j, i, ny, nx)];
  }

  if (should_write_txt())
    write_u_values("u_cuda_1gpu.txt", h_u_global, n);

  // -- Cleanup ---------------------------------------------------
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(d0_u_old, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(
          dpct::dpct_free(d0_u_new, dpct::get_in_order_queue())));
  CUDA_CHECK(
      DPCT_CHECK_ERROR(dpct::dpct_free(d0_rhs, dpct::get_in_order_queue())));
      CUDA_CHECK(DPCT_CHECK_ERROR(
          dpct::dpct_free(d0_kappa, dpct::get_in_order_queue())));

  free(h_u_global);
  free(h_u0);
  free(h_rhs0);
  free(h_kappa0);

  printf("\n**** ONE GPU CUDA FINISHED ****\n");
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}