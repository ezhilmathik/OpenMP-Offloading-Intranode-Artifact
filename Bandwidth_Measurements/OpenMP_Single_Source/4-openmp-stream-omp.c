//-*-c++-*-
/* OpenMP target offload implementation of the 3D heat conduction problem
         du/dt = div ( kappa(x,y,z) grad u ) + f(x,y,z)
   where kappa(x,y,z) = 0.5+0.45*sin(pi*x)*sin(pi*y)*sin(pi*z),
   using explicit time stepping.

   FOUR-GPU VERSION, ASYNCHRONOUS, halo exchange staged through the host.
   ONE HOST THREAD PER DEVICE.

   Layout: u[k][j][i]  ->  k is slowest, i is fastest
     offset = k*(ny*nx) + j*nx + i
   Decomposition along the k axis, so an XY plane at fixed k is CONTIGUOUS
   and no pack/unpack kernels are needed:
     D2H: omp_target_memcpy_async(h_halo, d_u_new + k*(ny*nx), halo_bytes, ...)
     H2D: omp_target_memcpy_async(d_u_new + ghost_k*(ny*nx), h_halo, ...)
   Neighbour offsets:
     i+-1  ->  +-1
     j+-1  ->  +-nx
     k+-1  ->  +-ny*nx

   This file is 4-openmp-stream.c with one host thread per device. Everything
   outside the time loop is unchanged -- same setup, same allocation, same
   copy-back -- so that a diff shows only what the extra host threads add.

   WHAT IS NEW HERE RELATIVE TO 4-openmp-stream.c
   ----------------------------------------------
   4-openmp-stream.c drives all four devices from one host thread: the
   launches for GPU3 cannot be issued until those for GPU0, 1 and 2 have
   been. Here thread tid drives device tid, so the four launch sequences
   proceed independently.

   The dependency machinery gets simpler, not harder. A thread cannot wait on
   another thread's tasks -- depend clauses order sibling tasks, and the four
   threads' target tasks live in different implicit task regions -- so the
   exchange is arranged as a PUSH to keep every dependence inside one thread.
   See PUSH, NOT PULL below.

   ONE TOKEN PER HALO -- SAME SCHEME AS 4-openmp-stream.c
   ------------------------------------------------------
   Each token describes a THREE-TASK CHAIN, and one `inout` depend-object per
   token expresses the whole chain, so the SAME object serves that chain's
   D2H and its H2D:

     tk_halo0_up :  GPU0 top boundary -> D2H0 -> H2D into GPU1's ghost k=0
     tk_halo1_dn :  GPU1 bot boundary -> D2H1 -> H2D into GPU0's ghost nz0-1
     tk_halo1_up :  GPU1 top boundary -> D2H1 -> H2D into GPU2's ghost k=0
     tk_halo2_dn :  GPU2 bot boundary -> D2H2 -> H2D into GPU1's ghost nz1-1
     tk_halo2_up :  GPU2 top boundary -> D2H2 -> H2D into GPU3's ghost k=0
     tk_halo3_dn :  GPU3 bot boundary -> D2H3 -> H2D into GPU2's ghost nz2-1

   Six tokens, six objects -- one per halo, identical in form to
   4-openmp-stream.c, so a diff between the two files shows only the
   threading. This mirrors 2-openmp-stream-omp.c exactly, with three
   interfaces instead of one.

   Here the chains are per THREAD as well as per halo. Thread 0 owns
   tk_halo0_up end to end; thread 1 owns tk_halo1_dn and tk_halo1_up end to
   end; thread 2 owns tk_halo2_dn and tk_halo2_up; thread 3 owns
   tk_halo3_dn. The interior threads own two chains each because they have
   two interfaces, but no chain is ever shared. That is what PUSH buys.

   The tokens are declared OUTSIDE the parallel region, as in
   4-openmp-stream.c and 2-openmp-stream-omp.c. Safe precisely because each
   is touched by one thread only; nothing crosses.

   PUSH, NOT PULL
   --------------
   A thread does not fill its OWN ghost from the neighbour's buffer. It
   sends its OWN buffer into the NEIGHBOUR's ghost:

     thread 0:  bnd_top -> D2H -> h_halo0_up -> H2D -> GPU1 ghost k=0
     thread 1:  bnd_bot -> D2H -> h_halo1_dn -> H2D -> GPU0 ghost k=nz0-1
                bnd_top -> D2H -> h_halo1_up -> H2D -> GPU2 ghost k=0
     thread 2:  bnd_bot -> D2H -> h_halo2_dn -> H2D -> GPU1 ghost k=nz1-1
                bnd_top -> D2H -> h_halo2_up -> H2D -> GPU3 ghost k=0
     thread 3:  bnd_bot -> D2H -> h_halo3_dn -> H2D -> GPU2 ghost k=nz2-1

   The transfer map is unchanged -- GPU0's ghost still receives GPU1's plane
   k=1, GPU1's ghost still receives GPU0's plane k=nz0-2, and so on. What
   changes is WHICH THREAD performs each copy.

   That is the whole point. Both endpoints of each chain now belong to one
   thread, so the H2D is simply the last link on that thread's own halo
   chain -- a genuine sibling dependence. In the pull form, thread 0's H2D
   read h_halo1_dn, which thread 1 wrote, and no depend clause can express
   that; it needed a mid-step barrier, and an `omp barrier` completes every
   explicit task bound to the region, including the interior kernel. Push
   removes the barrier and with it the stall.

   The interior kernels carry NO token. Nothing consumes their completion;
   they are drained by the plain taskwait at the end of the step.

   KERNEL SPLIT -- SAME AS 4-openmp.c AND 4-openmp-stream.c
   --------------------------------------------------------
   The split is by INTERFACE, not by position:

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
     D2H        async   depend(inout: that halo's token)
     H2D        async   depend(inout: that halo's token)  -- PUSH into the
                                    neighbour's ghost, from OUR OWN buffer
     taskwait                       -- drains interior, D2Hs and H2Ds
     barrier                        -- all four threads' pushes have landed
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
     D2H               overlaps the interior
     H2D               overlaps the interior

   The interior is drained only by the taskwait immediately before the swap.
   This matches 4-openmp-stream.c, which achieves the same coverage from one
   driver thread with a single selective taskwait.

   NOTE. An earlier revision of this file used a pull exchange with a
   mid-step barrier and reported PARTIAL overlap -- D2H covered, H2D not --
   as an inherent cost of one-thread-per-device. It is not inherent; it was
   the cost of pulling. Push removes it.

   WHY THE H2D MAY WRITE THE NEIGHBOUR'S u_new SAFELY
   --------------------------------------------------
   Thread t pushes into a plane of the neighbour's u_new while the
   neighbour's own kernels are running on that same array. There is no race:
   the pushed plane is a GHOST (k=0 or k=nz-1), and no kernel writes a ghost
   plane -- the interior runs k_start..k_end and the boundary kernels run
   k=1 and k=nz-2, all strictly inside. Nor does anything READ u_new's ghost
   this step; the ghosts are read as u_old only after the swap, which is
   separated from the pushes by the taskwait and the barrier.

   The pointer read is likewise safe. Thread 0 reads d1_u_new to name the
   destination; that variable is only ever written inside the `single`, whose
   implicit barrier publishes the swapped value to every thread.

   LAUNCH ORDER
   ------------
   The interior kernel is launched FIRST on each device, matching
   4-openmp-stream.c and 2-openmp-stream-omp.c, so it has the longest
   possible time to run behind the boundary kernels and the exchange.

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

/* Uniform check for every omp_target_memcpy / omp_target_memcpy_async in
   the set. A silent transfer failure would produce a plausible-looking but
   wrong answer, which is the worst outcome for a reproducibility artifact.
   `critical` so the message is not interleaved when several host threads
   fail at once; harmless outside a parallel region.
   NOTE: for the *_async form this checks only that the copy was accepted
   for execution, not that it completed. */
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

  printf("\n**** FOUR GPUS OpenMP Offloading, ASYNC + 4 HOST THREADS STARTS ****\n");

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
  // The n-2 interior planes are split four ways; the remainder is handed
  // to the lowest-numbered devices, one plane each.
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

  // -- Host halo staging buffers ---------------------------------
  // In [k][j][i] layout XY planes are contiguous, so D2H/H2D go directly
  // via pointer offset. No device halo buffers needed.
  // Six buffers: three interfaces, two directions each. Each buffer is
  // written AND read by exactly ONE thread -- that is the PUSH arrangement.
  // The end-of-step barrier is what makes reuse next step safe.
  double *h_halo0_up = (double*)malloc(halo_bytes);  // dev0 -> dev1, thread 0
  double *h_halo1_dn = (double*)malloc(halo_bytes);  // dev1 -> dev0, thread 1
  double *h_halo1_up = (double*)malloc(halo_bytes);  // dev1 -> dev2, thread 1
  double *h_halo2_dn = (double*)malloc(halo_bytes);  // dev2 -> dev1, thread 2
  double *h_halo2_up = (double*)malloc(halo_bytes);  // dev2 -> dev3, thread 2
  double *h_halo3_dn = (double*)malloc(halo_bytes);  // dev3 -> dev2, thread 3

  if (!h_halo0_up || !h_halo1_dn || !h_halo1_up ||
      !h_halo2_dn || !h_halo2_up || !h_halo3_dn) {
    fprintf(stderr, "ERROR: malloc failed (halo host)\n");
    return 1;
  }

  // Fix the step count before the parallel region: all four threads must run
  // exactly the same number of iterations, and none should accumulate its
  // own floating-point clock.
  int nsteps = 0;
  for (double tt=0.0; tt < T; tt += dt) nsteps++;

  printf("\n=== SIMULATION STARTS ===\n");

  // Accumulated inside the parallel region, around the time loop only.
  // See the note at the timer itself.
  double compute_timer = 0.0;

  // -- Dependency tokens and depend-objects ----------------------
  // ASYNC ONLY. The tokens are never read or written; the runtime orders
  // tasks by their addresses.
  //
  // ONE token per halo, exactly as in 4-openmp-stream.c. Each describes a
  // three-task chain, and one `inout` depend-object expresses the whole
  // chain, so the SAME object serves that chain's D2H and its H2D:
  //
  //   tk_halo0_up :  GPU0 top boundary -> D2H0 -> H2D into GPU1's ghost k=0
  //   tk_halo1_dn :  GPU1 bot boundary -> D2H1 -> H2D into GPU0's ghost nz0-1
  //   tk_halo1_up :  GPU1 top boundary -> D2H1 -> H2D into GPU2's ghost k=0
  //   tk_halo2_dn :  GPU2 bot boundary -> D2H2 -> H2D into GPU1's ghost nz1-1
  //   tk_halo2_up :  GPU2 top boundary -> D2H2 -> H2D into GPU3's ghost k=0
  //   tk_halo3_dn :  GPU3 bot boundary -> D2H3 -> H2D into GPU2's ghost nz2-1
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
  // PHASE 1: each GPU computes, then async D2H directly from the contiguous
  //   boundary plane -- no pack kernel needed.
  //
  // PHASE 2: async H2D directly into the NEIGHBOUR's contiguous ghost plane
  //   -- no unpack. PUSH, so the source buffer is always this thread's own.
  //
  // Transfer map (six transfers, three interfaces):
  //   GPU0 plane k=nz0-2  ->  h_halo0_up  ->  GPU1 ghost k=0       (thread 0)
  //   GPU1 plane k=1      ->  h_halo1_dn  ->  GPU0 ghost k=nz0-1   (thread 1)
  //   GPU1 plane k=nz1-2  ->  h_halo1_up  ->  GPU2 ghost k=0       (thread 1)
  //   GPU2 plane k=1      ->  h_halo2_dn  ->  GPU1 ghost k=nz1-1   (thread 2)
  //   GPU2 plane k=nz2-2  ->  h_halo2_up  ->  GPU3 ghost k=0       (thread 2)
  //   GPU3 plane k=1      ->  h_halo3_dn  ->  GPU2 ghost k=nz2-1   (thread 3)
  //
  // The ghost planes at GPU0 k=0 and GPU3 k=nz3-1 are physical domain
  // boundaries. They are initialised to zero and never written.
  //
  // ASYNC: interior is launched first so it has the longest possible time to
  //   run behind the boundary kernels, the D2H, the exchange and the H2D.
  // ======================================================================

#pragma omp parallel num_threads(4) default(shared)
  {
    int tid       = omp_get_thread_num();
    int device_id = tid;

    // num_threads() is a request, not a guarantee. With fewer threads the
    // barriers still work but some devices are never driven, which would
    // silently produce a wrong answer rather than fail.
    if (omp_get_num_threads() != 4) {
#pragma omp critical
      fprintf(stderr, "ERROR: this version needs 4 host threads, got %d\n",
              omp_get_num_threads());
      abort();
    }

    // Time the loop only, matching the other multi-GPU versions.
    // Both reads must come from the SAME thread: omp_get_wtime is only
    // required to be consistent per thread, not across threads -- which is
    // why this is `if (tid == 0)` and not `single`, since two `single`
    // regions may be executed by two different threads. The barriers make
    // the interval cover all four threads' work rather than just thread 0's.
#pragma omp barrier
    if (tid == 0) compute_timer -= omp_get_wtime();
#pragma omp barrier

    for (int s=0; s<nsteps; s++) {

      // -- This thread's compute, its D2Hs, and its PUSHes ---------
      // Every operation below reads only data THIS thread produced, so the
      // whole chain is ordered by this thread's own tokens. The only writes
      // that leave the thread are the pushes, and they land in ghost planes
      // that no kernel touches. See PUSH, NOT PULL in the header.
      if (tid == 0) {
        // GPU0: no bottom boundary -- k=1 is never sent, so it is folded
        // into the interior.
        heat_update_interior(d0_u_new, d0_u_old, d0_rhs, d0_kappa,
                             nz0, ny, nx, factor, 1, nz0-3, device_id);
        heat_update_boundary_top(d0_u_new, d0_u_old, d0_rhs, d0_kappa,
                                 nz0, ny, nx, factor, device_id, &tk_halo0_up);

        // D2H: our plane k=nz0-2 -> our own buffer. Next link on the _up chain.
        check_copy(omp_target_memcpy_async(h_halo0_up,
                                           d0_u_new + (size_t)(nz0-2) * ny * nx,
                                           halo_bytes, 0, 0, host_id, device_id,
                                           1, &halo0_up_dep),
                   "D2H dev0 up");

        // H2D: PUSH our buffer into GPU1's ghost k=0. The destination is the
        // neighbour's device, but the SOURCE is ours, so this is simply the
        // last link on our own chain -- same object as the D2H above.
        check_copy(omp_target_memcpy_async(d1_u_new + (size_t)0 * ny * nx,
                                           h_halo0_up,
                                           halo_bytes, 0, 0, 1, host_id,
                                           1, &halo0_up_dep),
                   "H2D dev1 ghost bottom");
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

        // _dn chain: our plane k=1 -> our buffer -> GPU0's ghost k=nz0-1.
        check_copy(omp_target_memcpy_async(h_halo1_dn,
                                           d1_u_new + (size_t)1 * ny * nx,
                                           halo_bytes, 0, 0, host_id, device_id,
                                           1, &halo1_dn_dep),
                   "D2H dev1 dn");
        check_copy(omp_target_memcpy_async(d0_u_new + (size_t)(nz0-1) * ny * nx,
                                           h_halo1_dn,
                                           halo_bytes, 0, 0, 0, host_id,
                                           1, &halo1_dn_dep),
                   "H2D dev0 ghost top");

        // _up chain: our plane k=nz1-2 -> our buffer -> GPU2's ghost k=0.
        check_copy(omp_target_memcpy_async(h_halo1_up,
                                           d1_u_new + (size_t)(nz1-2) * ny * nx,
                                           halo_bytes, 0, 0, host_id, device_id,
                                           1, &halo1_up_dep),
                   "D2H dev1 up");
        check_copy(omp_target_memcpy_async(d2_u_new + (size_t)0 * ny * nx,
                                           h_halo1_up,
                                           halo_bytes, 0, 0, 2, host_id,
                                           1, &halo1_up_dep),
                   "H2D dev2 ghost bottom");
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

        // _dn chain: our plane k=1 -> our buffer -> GPU1's ghost k=nz1-1.
        check_copy(omp_target_memcpy_async(h_halo2_dn,
                                           d2_u_new + (size_t)1 * ny * nx,
                                           halo_bytes, 0, 0, host_id, device_id,
                                           1, &halo2_dn_dep),
                   "D2H dev2 dn");
        check_copy(omp_target_memcpy_async(d1_u_new + (size_t)(nz1-1) * ny * nx,
                                           h_halo2_dn,
                                           halo_bytes, 0, 0, 1, host_id,
                                           1, &halo2_dn_dep),
                   "H2D dev1 ghost top");

        // _up chain: our plane k=nz2-2 -> our buffer -> GPU3's ghost k=0.
        check_copy(omp_target_memcpy_async(h_halo2_up,
                                           d2_u_new + (size_t)(nz2-2) * ny * nx,
                                           halo_bytes, 0, 0, host_id, device_id,
                                           1, &halo2_up_dep),
                   "D2H dev2 up");
        check_copy(omp_target_memcpy_async(d3_u_new + (size_t)0 * ny * nx,
                                           h_halo2_up,
                                           halo_bytes, 0, 0, 3, host_id,
                                           1, &halo2_up_dep),
                   "H2D dev3 ghost bottom");
      }
      else {  // tid == 3
        // GPU3: no top boundary -- k=nz3-2 is never sent, so it is folded
        // into the interior.
        heat_update_interior(d3_u_new, d3_u_old, d3_rhs, d3_kappa,
                             nz3, ny, nx, factor, 2, nz3-2, device_id);
        heat_update_boundary_bottom(d3_u_new, d3_u_old, d3_rhs, d3_kappa,
                                    nz3, ny, nx, factor, device_id,
                                    &tk_halo3_dn);

        // _dn chain: our plane k=1 -> our buffer -> GPU2's ghost k=nz2-1.
        check_copy(omp_target_memcpy_async(h_halo3_dn,
                                           d3_u_new + (size_t)1 * ny * nx,
                                           halo_bytes, 0, 0, host_id, device_id,
                                           1, &halo3_dn_dep),
                   "D2H dev3 dn");
        check_copy(omp_target_memcpy_async(d2_u_new + (size_t)(nz2-1) * ny * nx,
                                           h_halo3_dn,
                                           halo_bytes, 0, 0, 2, host_id,
                                           1, &halo3_dn_dep),
                   "H2D dev2 ghost top");
      }

      // -- Drain: this thread's interior kernel, its D2Hs and its H2Ds
      // The interior has been in flight across the entire exchange.
      //#pragma omp taskwait

      // -- Barrier: all four threads have drained, so every ghost plane is
      //    written and the halo buffers may be reused next step -----------
#pragma omp barrier

      // -- Pointer swap (one thread swaps all four sets) ------------
      // `single` carries an implicit barrier at its end, which is what
      // makes the swapped pointers visible to every thread. No explicit
      // barrier is needed here.
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
    write_u_values("u_openmp_stream_omp_4gpu.txt", h_u_global, n);

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
  free(h_halo0_up); free(h_halo1_dn);
  free(h_halo1_up); free(h_halo2_dn);
  free(h_halo2_up); free(h_halo3_dn);

  printf("\n**** FOUR GPUS OpenMP Offloading, ASYNC + 4 HOST THREADS FINISHED ****\n");
  return 0;
}
