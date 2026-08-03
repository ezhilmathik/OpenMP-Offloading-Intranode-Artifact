//-*-c++-*-
/* OpenMP target offload implementation of the 3D heat conduction problem
         du/dt = div ( kappa(x,y,z) grad u ) + f(x,y,z)
   where kappa(x,y,z) = 0.5+0.45*sin(pi*x)*sin(pi*y)*sin(pi*z),
   using explicit time stepping.

   FOUR-GPU VERSION, DIRECT PEER-TO-PEER halo exchange.
   ONE HOST THREAD PER DEVICE.

   Layout: u[k][j][i]  ->  k is slowest, i is fastest
     offset = k*(ny*nx) + j*nx + i
   Decomposition along the k axis, so an XY plane at fixed k is CONTIGUOUS
   and no pack/unpack kernels AND no staging buffers are needed. One call
   moves a plane straight from one device to the other:
     omp_target_memcpy_async(d_dst + ghost_k*(ny*nx),
                             d_src + k*(ny*nx),
                             halo_bytes, 0, 0, dst_dev, src_dev, ...)
   Neighbour offsets:
     i+-1  ->  +-1
     j+-1  ->  +-nx
     k+-1  ->  +-ny*nx

   This file is 4-openmp-stream-omp.c with the host taken out of the halo
   path. Everything else is unchanged -- same setup, same allocation, same
   kernels, same threading, same tokens, same names. It is the two-GPU peer
   version generalised from one interface to three.

   WHAT IS NEW HERE RELATIVE TO 4-openmp-stream-omp.c
   --------------------------------------------------
   That file stages every halo through a host buffer: D2H into h_halo*, then
   H2D out of it. Here a single omp_target_memcpy_async names two DEVICE
   numbers and the plane never touches host memory. Twelve transfers per step
   become six, the host buffers are gone entirely, and each chain is two
   tasks instead of three.

   Nothing else changes. The boundary kernels are byte-identical to the
   staged version's -- they read the off-device neighbour from the ghost
   plane of u_old exactly as before -- and there are no halo allocations of
   any kind on either side of the copy.

   WHETHER THE TRANSFER IS ACTUALLY PEER-TO-PEER IS A RUNTIME PROPERTY.
   omp_target_memcpy with two device endpoints is permitted to stage through
   the host internally. If it does, this version measures nearly the same
   data path as 4-openmp-stream-omp.c and the timings will be close. Check it
   rather than assume it.

   WHY NO BARRIER IS NEEDED
   ------------------------
   Each copy writes a GHOST PLANE OF u_new. Nothing reads u_new's ghost
   during the step: the interior kernel runs k_start..k_end and the boundary
   kernels run k=1 and k=nz-2, all strictly inside, and all read u_old. The
   write and the neighbour's read are therefore a full step apart, and it is
   the u_old/u_new swap that keeps them apart.

   That is what the block-based revision lost -- with a single halo_recv the
   neighbour's copy wrote the very block a boundary kernel was reading, in
   the same step -- and why it needed either a mid-step barrier or a second
   receive buffer. Landing in the ghost plane gets the separation back for
   free, so the loop keeps the staged reference's shape and full overlap.

   PUSH, NOT PULL
   --------------
   Each thread copies OUT OF its own device: thread tid issues the copies
   whose SOURCE device is tid, writing into its neighbours' ghost planes.

   Two reasons.

   First, the dependence graph. Push puts both ends of every halo chain in
   ONE thread -- this thread's boundary kernel writes the plane, this
   thread's copy sends it -- so a single `inout` depend-object expresses the
   whole chain and no cross-thread dependence is required.

   Second, the pull formulation -- each thread reading its neighbours' planes
   -- SEGFAULTS reproducibly on clang 18.1.8 + CUDA 12.8 with four devices.
   With pull, devices 1 and 2 are each read as a SOURCE by two host threads
   at the same time (thread 0 and thread 2 both read device 1; threads 1 and
   3 both read device 2). With push, every device is the source of exactly
   one thread's copies, and the program runs.

   Supporting evidence that this is a runtime defect rather than a defect in
   the halo logic:

     - 2-openmp-stream-omp-p2p.c works with either assignment, because with
       only two devices each device is already the source for exactly one
       thread.
     - 4-openmp-stream-omp.c -- same four host threads, same kernels, same
       barriers, host-staged instead of peer -- runs correctly. So the
       four-thread structure is not the problem.
     - nvidia-smi topo -m reports NV6 between every pair, so no peer link is
       missing and no silent host-staged fallback is involved.

   Do not "simplify" this back to the symmetric-looking pull form.

   ONE TOKEN PER HALO -- SAME SCHEME AND SAME NAMES
   ------------------------------------------------
   Each token describes a TWO-TASK CHAIN, and one `inout` depend-object per
   token expresses it:

     tk_halo0_up :  GPU0 top boundary -> peer copy into GPU1's ghost k=0
     tk_halo1_dn :  GPU1 bot boundary -> peer copy into GPU0's ghost nz0-1
     tk_halo1_up :  GPU1 top boundary -> peer copy into GPU2's ghost k=0
     tk_halo2_dn :  GPU2 bot boundary -> peer copy into GPU1's ghost nz1-1
     tk_halo2_up :  GPU2 top boundary -> peer copy into GPU3's ghost k=0
     tk_halo3_dn :  GPU3 bot boundary -> peer copy into GPU2's ghost nz2-1

   Six tokens, six objects, the same names and the same scheme as
   4-openmp-stream.c and 4-openmp-stream-omp.c. Removing the host hop removes
   the middle link of each chain and nothing else.

   Declared OUTSIDE the parallel region: thread t owns every chain named
   after device t and never touches any other. Nothing crosses, so nothing
   needs to be private. That is what the PUSH arrangement buys.

   The interior kernels carry NO token. Nothing consumes their completion;
   they are drained by the plain taskwait at the end of the step.

   KERNEL SPLIT -- SAME AS THE REST OF THE FOUR-GPU SET
   ----------------------------------------------------
     heat_update_boundary_bottom   k = 1        only where k=1 is SENT
     heat_update_boundary_top      k = nz-2     only where k=nz-2 is SENT
     heat_update_interior          k_start..k_end   everything else

   Device 0 folds k=1 into the interior (k_start=1); device 3 folds k=nz3-2
   in at the other end (k_end=nz3-2). Devices 1 and 2 run both boundary
   kernels, because they have an interface at each end.

   Per timestep, per thread:
     interior   nowait                              long pole, no token
     bottom     nowait  depend(out:   tk_halo<tid>_dn)   if it sends down
     top        nowait  depend(out:   tk_halo<tid>_up)   if it sends up
     peer copy  async   depend(inout: that halo's token) -- PUSH straight
                                    into the neighbour's ghost, sourced from
                                    OUR OWN device
     taskwait                       -- drains interior and this thread's copies
     barrier                        -- all six pushes have landed
     swap by one thread             -- single, implicit barrier at its end

   There is NO mid-step barrier. The only cross-thread synchronisation is the
   end-of-step barrier, by which point the taskwait has already drained
   everything, so it costs nothing.

   OVERLAP -- FULL
   ---------------
   An `omp barrier` is a task barrier: every explicit task bound to the
   parallel region must complete before any thread passes it, and a
   `target nowait` region is an explicit task. A barrier placed mid-step
   would therefore drain the interior kernels.

   In the push form there is no mid-step barrier, so the interior kernel is
   in flight across the WHOLE exchange:

     boundary kernels  overlap the interior
     peer copies       overlap the interior

   The interior is drained only by the taskwait immediately before the swap.

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>   /* strcasecmp */
#include <math.h>
#include <omp.h>

#define IDX(k, j, i, ny, nx)  ((size_t)(k)*(size_t)(ny)*(size_t)(nx) \
                             + (size_t)(j)*(size_t)(nx) + (size_t)(i))

static int should_write_txt(void)
{
  const char *e = getenv("WRITE_TXT");
  if (!e) return 1;
  if (!strcmp(e, "0")) return 0;
  if (!strcasecmp(e, "false")) return 0;
  if (!strcasecmp(e, "no")) return 0;
  return 1;
}

/* Uniform check for every transfer in the set. A silent failure would
   produce a plausible-looking but wrong answer, which is the worst outcome
   for a reproducibility artifact. `critical` so the message is not
   interleaved when several host threads fail at once; harmless outside a
   parallel region.
   NOTE: for the *_async form this checks only that the copy was accepted for
   execution, not that it completed. */
static void check_copy(int rc, const char *what)
{
  if (rc == 0) return;
#pragma omp critical
  fprintf(stderr, "ERROR: transfer '%s' failed, rc=%d\n", what, rc);
  abort();
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
// Identical to 4-openmp.c. Boundary values are forced to exact zero.
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
// Computes k=1, the plane sent DOWNWARD. Used by GPU 1, 2 and 3.
// ASYNC: returns immediately; completion is published on dep_token.
// `nz` is unused here (k is fixed at 1); the parameter is kept so that all
// three kernels share one signature shape.
static void heat_update_boundary_bottom(double *u_new, const double *u_old,
                                        const double *rhs, const double *kappa,
                                        int nz, int ny, int nx,
                                        double factor, int device_id,
                                        int *dep_token)
{
  (void)nz;
#pragma omp target is_device_ptr(u_new, u_old, rhs, kappa) \
                   device(device_id) nowait depend(out: dep_token[0])
#pragma omp teams distribute parallel for collapse(2)
  for (int j=1; j<ny-1; j++)
    for (int i=1; i<nx-1; i++) {
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
// ASYNC: returns immediately; completion is published on dep_token.
static void heat_update_boundary_top(double *u_new, const double *u_old,
                                     const double *rhs, const double *kappa,
                                     int nz, int ny, int nx,
                                     double factor, int device_id,
                                     int *dep_token)
{
#pragma omp target is_device_ptr(u_new, u_old, rhs, kappa) \
                   device(device_id) nowait depend(out: dep_token[0])
#pragma omp teams distribute parallel for collapse(2)
  for (int j=1; j<ny-1; j++)
    for (int i=1; i<nx-1; i++) {
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
// Computes k = k_start .. k_end. The explicit bounds let devices 0 and 3
// fold in the plane they never send, which has no reason to sit in a
// boundary kernel.
// ASYNC: returns immediately. No dependency token -- nothing consumes this
// kernel's completion; it is drained by the plain taskwait at the end of
// the step.
// `nz` is unused here (the range is given explicitly); the parameter is kept
// so that all three kernels share one signature shape.
static void heat_update_interior(double *u_new, const double *u_old,
                                 const double *rhs, const double *kappa,
                                 int nz, int ny, int nx,
                                 double factor,
                                 int k_start, int k_end,
                                 int device_id)
{
  (void)nz;
#pragma omp target is_device_ptr(u_new, u_old, rhs, kappa) \
                   device(device_id) nowait
#pragma omp teams distribute parallel for collapse(3)
  for (int k=k_start; k<=k_end; k++)
    for (int j=1; j<ny-1; j++)
      for (int i=1; i<nx-1; i++) {

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

  printf("\n**** FOUR GPUS OpenMP Offloading, P2P + 4 HOST THREADS STARTS ****\n");

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
  int host_id     = omp_get_initial_device();
  int num_devices = omp_get_num_devices();

  if (num_devices < 4) {
    fprintf(stderr, "ERROR: this version needs 4 OpenMP devices, %d present\n",
            num_devices);
    exit(EXIT_FAILURE);
  }

  printf("Using OpenMP target devices 0, 1, 2 and 3 of %d\n", num_devices);

  // -- Domain decomposition along k -----------------------------
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

  if (nz0 < 4 || nz1 < 4 || nz2 < 4 || nz3 < 4) {
    fprintf(stderr,
            "ERROR: n=%d gives nz0=%d nz1=%d nz2=%d nz3=%d; "
            "each needs >= 4 planes\n",
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

  // -- GPU 0 device allocation + H2D ----------------------------
  double *d0_u_old = (double*)omp_target_alloc(size0, 0);
  double *d0_u_new = (double*)omp_target_alloc(size0, 0);
  double *d0_rhs   = (double*)omp_target_alloc(size0, 0);
  double *d0_kappa = (double*)omp_target_alloc(size0, 0);

  if (!d0_u_old || !d0_u_new || !d0_rhs || !d0_kappa) {
    fprintf(stderr, "ERROR: omp_target_alloc failed for GPU 0\n");
    return 1;
  }

  check_copy(omp_target_memcpy(d0_u_old, h_u0,     size0, 0, 0, 0, host_id),
             "init H2D u_old GPU0");
  check_copy(omp_target_memcpy(d0_u_new, h_u0,     size0, 0, 0, 0, host_id),
             "init H2D u_new GPU0");
  check_copy(omp_target_memcpy(d0_rhs,   h_rhs0,   size0, 0, 0, 0, host_id),
             "init H2D rhs GPU0");
  check_copy(omp_target_memcpy(d0_kappa, h_kappa0, size0, 0, 0, 0, host_id),
             "init H2D kappa GPU0");

  // -- GPU 1 device allocation + H2D ----------------------------
  double *d1_u_old = (double*)omp_target_alloc(size1, 1);
  double *d1_u_new = (double*)omp_target_alloc(size1, 1);
  double *d1_rhs   = (double*)omp_target_alloc(size1, 1);
  double *d1_kappa = (double*)omp_target_alloc(size1, 1);

  if (!d1_u_old || !d1_u_new || !d1_rhs || !d1_kappa) {
    fprintf(stderr, "ERROR: omp_target_alloc failed for GPU 1\n");
    return 1;
  }

  check_copy(omp_target_memcpy(d1_u_old, h_u1,     size1, 0, 0, 1, host_id),
             "init H2D u_old GPU1");
  check_copy(omp_target_memcpy(d1_u_new, h_u1,     size1, 0, 0, 1, host_id),
             "init H2D u_new GPU1");
  check_copy(omp_target_memcpy(d1_rhs,   h_rhs1,   size1, 0, 0, 1, host_id),
             "init H2D rhs GPU1");
  check_copy(omp_target_memcpy(d1_kappa, h_kappa1, size1, 0, 0, 1, host_id),
             "init H2D kappa GPU1");

  // -- GPU 2 device allocation + H2D ----------------------------
  double *d2_u_old = (double*)omp_target_alloc(size2, 2);
  double *d2_u_new = (double*)omp_target_alloc(size2, 2);
  double *d2_rhs   = (double*)omp_target_alloc(size2, 2);
  double *d2_kappa = (double*)omp_target_alloc(size2, 2);

  if (!d2_u_old || !d2_u_new || !d2_rhs || !d2_kappa) {
    fprintf(stderr, "ERROR: omp_target_alloc failed for GPU 2\n");
    return 1;
  }

  check_copy(omp_target_memcpy(d2_u_old, h_u2,     size2, 0, 0, 2, host_id),
             "init H2D u_old GPU2");
  check_copy(omp_target_memcpy(d2_u_new, h_u2,     size2, 0, 0, 2, host_id),
             "init H2D u_new GPU2");
  check_copy(omp_target_memcpy(d2_rhs,   h_rhs2,   size2, 0, 0, 2, host_id),
             "init H2D rhs GPU2");
  check_copy(omp_target_memcpy(d2_kappa, h_kappa2, size2, 0, 0, 2, host_id),
             "init H2D kappa GPU2");

  // -- GPU 3 device allocation + H2D ----------------------------
  double *d3_u_old = (double*)omp_target_alloc(size3, 3);
  double *d3_u_new = (double*)omp_target_alloc(size3, 3);
  double *d3_rhs   = (double*)omp_target_alloc(size3, 3);
  double *d3_kappa = (double*)omp_target_alloc(size3, 3);

  if (!d3_u_old || !d3_u_new || !d3_rhs || !d3_kappa) {
    fprintf(stderr, "ERROR: omp_target_alloc failed for GPU 3\n");
    return 1;
  }

  check_copy(omp_target_memcpy(d3_u_old, h_u3,     size3, 0, 0, 3, host_id),
             "init H2D u_old GPU3");
  check_copy(omp_target_memcpy(d3_u_new, h_u3,     size3, 0, 0, 3, host_id),
             "init H2D u_new GPU3");
  check_copy(omp_target_memcpy(d3_rhs,   h_rhs3,   size3, 0, 0, 3, host_id),
             "init H2D rhs GPU3");
  check_copy(omp_target_memcpy(d3_kappa, h_kappa3, size3, 0, 0, 3, host_id),
             "init H2D kappa GPU3");

  // -- NO halo buffers of any kind -------------------------------
  // In [k][j][i] layout XY planes are contiguous, so a peer copy addresses
  // one directly by pointer offset on each side. Nothing is staged, packed
  // or unpacked, on the host or on the device. halo_bytes is still needed,
  // as the transfer size.
  //
  // 4-openmp-stream.c and 4-openmp-stream-omp.c allocate six h_halo* buffers
  // here. Their absence is the whole difference. The token names keep the
  // _up / _dn suffixes so the three files stay diffable.

  // Fix the step count before the parallel region: all four threads must run
  // exactly the same number of iterations, and none should accumulate its
  // own floating-point clock.
  int nsteps = 0;
  for (double tt=0.0; tt < T; tt += dt) nsteps++;

  printf("\n=== SIMULATION STARTS ===\n");

  double compute_timer = 0.0;

  // -- Dependency tokens and depend-objects ----------------------
  // ASYNC ONLY. The tokens are never read or written; the runtime orders
  // tasks by their addresses.
  //
  // ONE token per halo, exactly as in 4-openmp-stream.c and
  // 4-openmp-stream-omp.c. Each describes a two-task chain here, since there
  // is no host hop in the middle:
  //
  //   tk_halo0_up :  GPU0 top boundary -> peer copy into GPU1's ghost k=0
  //   tk_halo1_dn :  GPU1 bot boundary -> peer copy into GPU0's ghost nz0-1
  //   tk_halo1_up :  GPU1 top boundary -> peer copy into GPU2's ghost k=0
  //   tk_halo2_dn :  GPU2 bot boundary -> peer copy into GPU1's ghost nz1-1
  //   tk_halo2_up :  GPU2 top boundary -> peer copy into GPU3's ghost k=0
  //   tk_halo3_dn :  GPU3 bot boundary -> peer copy into GPU2's ghost nz2-1
  //
  // Declared OUTSIDE the parallel region even though this file has one host
  // thread per device: thread t owns every chain named after device t, and
  // never touches any other. Nothing crosses, so nothing needs to be
  // private. That is what the PUSH arrangement buys.
  //
  // Only the boundary kernels publish completion. The interior kernels are
  // synchronised by the taskwait at the end of the step and need no token.
  int tk_halo0_up = 0;
  int tk_halo1_dn = 0;
  int tk_halo1_up = 0;
  int tk_halo2_dn = 0;
  int tk_halo2_up = 0;
  int tk_halo3_dn = 0;

  omp_depend_t halo0_up_dep;
  omp_depend_t halo1_dn_dep;
  omp_depend_t halo1_up_dep;
  omp_depend_t halo2_dn_dep;
  omp_depend_t halo2_up_dep;
  omp_depend_t halo3_dn_dep;

#pragma omp depobj(halo0_up_dep) depend(inout: tk_halo0_up)
#pragma omp depobj(halo1_dn_dep) depend(inout: tk_halo1_dn)
#pragma omp depobj(halo1_up_dep) depend(inout: tk_halo1_up)
#pragma omp depobj(halo2_dn_dep) depend(inout: tk_halo2_dn)
#pragma omp depobj(halo2_up_dep) depend(inout: tk_halo2_up)
#pragma omp depobj(halo3_dn_dep) depend(inout: tk_halo3_dn)

  // ======================================================================
  // Time loop -- thread tid drives device tid
  //
  // Each GPU computes, then pushes its boundary planes straight into the
  //   neighbours' ghost planes. One transfer per halo, no host hop, no pack
  //   or unpack.
  //
  // Transfer map (six transfers, three interfaces):
  //   GPU0 plane k=nz0-2  ->  GPU1 ghost k=0         (thread 0)
  //   GPU1 plane k=1      ->  GPU0 ghost k=nz0-1     (thread 1)
  //   GPU1 plane k=nz1-2  ->  GPU2 ghost k=0         (thread 1)
  //   GPU2 plane k=1      ->  GPU1 ghost k=nz1-1     (thread 2)
  //   GPU2 plane k=nz2-2  ->  GPU3 ghost k=0         (thread 2)
  //   GPU3 plane k=1      ->  GPU2 ghost k=nz2-1     (thread 3)
  //
  // Each thread PUSHES out of its own device, so thread tid issues the
  // copies whose src_device is tid. See PUSH, NOT PULL in the header: the
  // reverse assignment segfaults on LLVM/CUDA with four devices.
  //
  // The ghost planes at GPU0 k=0 and GPU3 k=nz3-1 are physical domain
  // boundaries. They are initialised to zero and never written.
  //
  // ASYNC: interior is launched first so it has the longest possible time to
  //   run behind the boundary kernels and the exchange.
  // ======================================================================

#pragma omp parallel num_threads(4) default(shared)
  {
    int tid       = omp_get_thread_num();
    int device_id = tid;

    if (omp_get_num_threads() != 4) {
#pragma omp critical
      fprintf(stderr, "ERROR: this version needs 4 host threads, got %d\n",
              omp_get_num_threads());
      abort();
    }

    // Time the loop only. Both reads must come from the SAME thread:
    // omp_get_wtime is only required to be consistent per thread.
#pragma omp barrier
    if (tid == 0) compute_timer -= omp_get_wtime();
#pragma omp barrier

    for (int s=0; s<nsteps; s++) {

      // -- This thread's compute, then its peer PUSHes -------------
      // Every operation below SOURCES only data THIS thread produced, so
      // each chain is ordered by this thread's own token. See PUSH, NOT
      // PULL in the header.
      if (tid == 0) {
        // GPU0: no bottom boundary -- k=1 is never sent, so it is folded
        // into the interior.
        heat_update_interior(d0_u_new, d0_u_old, d0_rhs, d0_kappa,
                             nz0, ny, nx, factor, 1, nz0-3, device_id);
        heat_update_boundary_top(d0_u_new, d0_u_old, d0_rhs, d0_kappa,
                                 nz0, ny, nx, factor, device_id, &tk_halo0_up);

        // PEER COPY: PUSH our plane k=nz0-2 straight into GPU1's ghost k=0.
        // The destination is the neighbour's device, but the SOURCE is ours,
        // so this is simply the last link on our own chain. Both the plane
        // and the ghost are contiguous, so nothing is packed or unpacked.
        check_copy(omp_target_memcpy_async(d1_u_new + (size_t)0 * ny * nx,
                                           d0_u_new + (size_t)(nz0-2) * ny * nx,
                                           halo_bytes, 0, 0, 1, device_id,
                                           1, &halo0_up_dep),
                   "P2P dev0 -> dev1 ghost bottom");
      }
      else if (tid == 1) {
        // GPU1: an interface at each end, so both boundary kernels run and
        // this thread owns two chains.
        heat_update_interior(d1_u_new, d1_u_old, d1_rhs, d1_kappa,
                             nz1, ny, nx, factor, 2, nz1-3, device_id);
        heat_update_boundary_bottom(d1_u_new, d1_u_old, d1_rhs, d1_kappa,
                                    nz1, ny, nx, factor, device_id,
                                    &tk_halo1_dn);
        heat_update_boundary_top(d1_u_new, d1_u_old, d1_rhs, d1_kappa,
                                 nz1, ny, nx, factor, device_id, &tk_halo1_up);

        check_copy(omp_target_memcpy_async(d0_u_new + (size_t)(nz0-1) * ny * nx,
                                           d1_u_new + (size_t)1 * ny * nx,
                                           halo_bytes, 0, 0, 0, device_id,
                                           1, &halo1_dn_dep),
                   "P2P dev1 -> dev0 ghost top");
        check_copy(omp_target_memcpy_async(d2_u_new + (size_t)0 * ny * nx,
                                           d1_u_new + (size_t)(nz1-2) * ny * nx,
                                           halo_bytes, 0, 0, 2, device_id,
                                           1, &halo1_up_dep),
                   "P2P dev1 -> dev2 ghost bottom");
      }
      else if (tid == 2) {
        // GPU2: an interface at each end, same shape as GPU1.
        heat_update_interior(d2_u_new, d2_u_old, d2_rhs, d2_kappa,
                             nz2, ny, nx, factor, 2, nz2-3, device_id);
        heat_update_boundary_bottom(d2_u_new, d2_u_old, d2_rhs, d2_kappa,
                                    nz2, ny, nx, factor, device_id,
                                    &tk_halo2_dn);
        heat_update_boundary_top(d2_u_new, d2_u_old, d2_rhs, d2_kappa,
                                 nz2, ny, nx, factor, device_id, &tk_halo2_up);

        check_copy(omp_target_memcpy_async(d1_u_new + (size_t)(nz1-1) * ny * nx,
                                           d2_u_new + (size_t)1 * ny * nx,
                                           halo_bytes, 0, 0, 1, device_id,
                                           1, &halo2_dn_dep),
                   "P2P dev2 -> dev1 ghost top");
        check_copy(omp_target_memcpy_async(d3_u_new + (size_t)0 * ny * nx,
                                           d2_u_new + (size_t)(nz2-2) * ny * nx,
                                           halo_bytes, 0, 0, 3, device_id,
                                           1, &halo2_up_dep),
                   "P2P dev2 -> dev3 ghost bottom");
      }
      else {  // tid == 3
        // GPU3: no top boundary -- k=nz3-2 is never sent, so it is folded
        // into the interior.
        heat_update_interior(d3_u_new, d3_u_old, d3_rhs, d3_kappa,
                             nz3, ny, nx, factor, 2, nz3-2, device_id);
        heat_update_boundary_bottom(d3_u_new, d3_u_old, d3_rhs, d3_kappa,
                                    nz3, ny, nx, factor, device_id,
                                    &tk_halo3_dn);

        check_copy(omp_target_memcpy_async(d2_u_new + (size_t)(nz2-1) * ny * nx,
                                           d3_u_new + (size_t)1 * ny * nx,
                                           halo_bytes, 0, 0, 2, device_id,
                                           1, &halo3_dn_dep),
                   "P2P dev3 -> dev2 ghost top");
      }

      // -- Drain: this thread's interior kernel and its peer copies -
      // The interior has been in flight across the entire exchange.
      //#pragma omp taskwait

      // -- Barrier: all four threads have drained, so every ghost plane is
      //    written before the swap turns it into u_old ----------------
#pragma omp barrier

      // -- Pointer swap (one thread swaps all four sets) ------------
      // `single` carries an implicit barrier at its end, which is what
      // makes the swapped pointers visible to every thread.
#pragma omp single
      {
        double *tmp;
        tmp = d0_u_old; d0_u_old = d0_u_new; d0_u_new = tmp;
        tmp = d1_u_old; d1_u_old = d1_u_new; d1_u_new = tmp;
        tmp = d2_u_old; d2_u_old = d2_u_new; d2_u_new = tmp;
        tmp = d3_u_old; d3_u_old = d3_u_new; d3_u_new = tmp;
      }
    }

#pragma omp barrier
    if (tid == 0) compute_timer += omp_get_wtime();
  }

  // -- Tear down depend-objects ----------------------------------
#pragma omp depobj(halo0_up_dep) destroy
#pragma omp depobj(halo1_dn_dep) destroy
#pragma omp depobj(halo1_up_dep) destroy
#pragma omp depobj(halo2_dn_dep) destroy
#pragma omp depobj(halo2_up_dep) destroy
#pragma omp depobj(halo3_dn_dep) destroy

  printf("\n%d timesteps complete\n", nsteps);
  printf("Time = %f seconds\n", compute_timer);
  printf("=== SIMULATION ENDS ===\n");

  // -- Copy back (layout [k][j][i]) ------------------------------
  check_copy(omp_target_memcpy(h_u0, d0_u_old, size0, 0, 0, host_id, 0),
             "copy-back GPU0");
  check_copy(omp_target_memcpy(h_u1, d1_u_old, size1, 0, 0, host_id, 1),
             "copy-back GPU1");
  check_copy(omp_target_memcpy(h_u2, d2_u_old, size2, 0, 0, host_id, 2),
             "copy-back GPU2");
  check_copy(omp_target_memcpy(h_u3, d3_u_old, size3, 0, 0, host_id, 3),
             "copy-back GPU3");

  // -- Reconstruct global solution (global output is [k][j][i]) --
  // Each device contributes its owned planes only; interface ghosts are
  // excluded because the owner writes them. The two physical-boundary
  // ghosts (GPU0 k=0, GPU3 k=nz3-1) are included and hold exact zero.
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

  // GPU1 contribution: global k = gk_start1 .. gk_start2-1
  for (int k=0; k<nz1; k++) {
    int gk = (k - ng) + gk_start1;
    if (gk >= gk_start1 && gk < gk_start2) {
      for (int j=0; j<n; j++)
        for (int i=0; i<n; i++)
          h_u_global[IDX(gk, j, i, n, n)] = h_u1[IDX(k, j, i, ny, nx)];
    }
  }

  // GPU2 contribution: global k = gk_start2 .. gk_start3-1
  for (int k=0; k<nz2; k++) {
    int gk = (k - ng) + gk_start2;
    if (gk >= gk_start2 && gk < gk_start3) {
      for (int j=0; j<n; j++)
        for (int i=0; i<n; i++)
          h_u_global[IDX(gk, j, i, n, n)] = h_u2[IDX(k, j, i, ny, nx)];
    }
  }

  // GPU3 contribution: global k = gk_start3 .. n-1
  for (int k=0; k<nz3; k++) {
    int gk = (k - ng) + gk_start3;
    if (gk >= gk_start3 && gk <= n-1) {
      for (int j=0; j<n; j++)
        for (int i=0; i<n; i++)
          h_u_global[IDX(gk, j, i, n, n)] = h_u3[IDX(k, j, i, ny, nx)];
    }
  }

  if (should_write_txt())
    write_u_values("u_openmp_stream_omp_p2p_4gpu.txt", h_u_global, n);

  // -- Cleanup ---------------------------------------------------
  omp_target_free(d0_u_old, 0); omp_target_free(d0_u_new, 0);
  omp_target_free(d0_rhs, 0);   omp_target_free(d0_kappa, 0);

  omp_target_free(d1_u_old, 1); omp_target_free(d1_u_new, 1);
  omp_target_free(d1_rhs, 1);   omp_target_free(d1_kappa, 1);

  omp_target_free(d2_u_old, 2); omp_target_free(d2_u_new, 2);
  omp_target_free(d2_rhs, 2);   omp_target_free(d2_kappa, 2);

  omp_target_free(d3_u_old, 3); omp_target_free(d3_u_new, 3);
  omp_target_free(d3_rhs, 3);   omp_target_free(d3_kappa, 3);

  free(h_u_global);
  free(h_u0);     free(h_u1);     free(h_u2);     free(h_u3);
  free(h_rhs0);   free(h_rhs1);   free(h_rhs2);   free(h_rhs3);
  free(h_kappa0); free(h_kappa1); free(h_kappa2); free(h_kappa3);

  printf("\n**** FOUR GPUS OpenMP Offloading, P2P + 4 HOST THREADS FINISHED ****\n");
  return 0;
}
