//-*-c++-*-
/* OpenMP target offload implementation of the 3D heat conduction problem
         du/dt = div ( kappa(x,y,z) grad u ) + f(x,y,z)
   where kappa(x,y,z) = 0.5+0.45*sin(pi*x)*sin(pi*y)*sin(pi*z),
   using explicit time stepping.

   TWO-GPU VERSION, DIRECT PEER-TO-PEER halo exchange.
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

   This file is 2-openmp-stream-omp.c with the host taken out of the halo
   path. Everything else is unchanged -- same setup, same allocation, same
   kernels, same threading, same tokens, same names -- so that a diff against
   it shows only the transfer path.

   WHAT IS NEW HERE RELATIVE TO 2-openmp-stream-omp.c
   --------------------------------------------------
   That file stages every halo through a host buffer: D2H into h_halo*, then
   H2D out of it. Here a single omp_target_memcpy_async names two DEVICE
   numbers and the plane never touches host memory. The host buffers are gone
   entirely, and each chain is two tasks instead of three.

   Nothing else changes. The boundary kernels are byte-identical to the
   staged version's -- they read the off-device neighbour from the ghost
   plane of u_old exactly as before -- and there are no halo allocations of
   any kind on either side of the copy.

   WHETHER THE TRANSFER IS ACTUALLY PEER-TO-PEER IS A RUNTIME PROPERTY.
   omp_target_memcpy with two device endpoints is permitted to stage through
   the host internally. If it does, this version measures nearly the same
   data path as 2-openmp-stream-omp.c and the timings will be close. A
   genuine peer copy shows up as a clear win at large N, where the halo is
   big enough for the host round trip to matter. Check it rather than
   assume it.

   WHY NO BARRIER IS NEEDED
   ------------------------
   The copy writes a GHOST PLANE OF u_new. Nothing reads u_new's ghost during
   the step: the interior kernel runs k_start..k_end and the boundary kernel
   runs k=nz-2, both strictly inside, and both read u_old. The write and the
   neighbour's read are therefore a full step apart, and it is the
   u_old/u_new swap that keeps them apart.

   That is what the block-based revision lost, and why it needed either a
   mid-step barrier or a second receive buffer. Landing in the ghost plane
   gets the separation back for free, so the loop keeps the staged
   reference's shape and full overlap.

   PUSH, NOT PULL
   --------------
   A thread does not fill its OWN ghost from the neighbour. It sends its OWN
   plane into the NEIGHBOUR's ghost:

     thread 0:  bnd_top -> peer copy -> GPU1 ghost k=0
     thread 1:  bnd_bot -> peer copy -> GPU0 ghost k=nz0-1

   The transfer map is unchanged -- GPU0's ghost still receives GPU1's plane
   k=1, and GPU1's ghost still receives GPU0's plane k=nz0-2. What changes is
   WHICH THREAD performs each copy, and that no host buffer sits in between.

   That is the whole point. The SOURCE of each copy is the thread's own
   device, so the copy is simply the last link on that thread's own halo
   chain -- a genuine sibling dependence. Only the DESTINATION is the
   neighbour's device, and a write into a ghost plane nobody else touches
   needs no ordering against the neighbour at all.

   ONE TOKEN PER HALO -- SAME SCHEME AS 2-openmp-stream.c
   ------------------------------------------------------
   Each token describes a TWO-TASK CHAIN, and one `inout` depend-object per
   token expresses it:

     tk_halo0_up :  GPU0 top boundary -> peer copy into GPU1's ghost k=0
     tk_halo1_dn :  GPU1 bot boundary -> peer copy into GPU0's ghost nz0-1

   Two tokens, two objects, the same names and the same scheme as
   2-openmp-stream.c and 2-openmp-stream-omp.c. Removing the host hop removes
   the middle link of each chain and nothing else.

   Here the chains are per THREAD as well as per halo: thread 0 owns
   tk_halo0_up end to end, thread 1 owns tk_halo1_dn end to end. That is what
   PUSH buys. In the pull form thread 0's copy would READ GPU1's plane k=1,
   written by thread 1's boundary kernel, and no token could express it --
   depend clauses order sibling tasks, and the two threads' target tasks live
   in different implicit task regions.

   The tokens are declared OUTSIDE the parallel region, as in
   2-openmp-stream.c. Safe precisely because each is touched by one thread
   only; nothing crosses.

   The interior kernels carry NO token. Nothing consumes their completion;
   they are drained by the plain taskwait at the end of the step.

   KERNEL SPLIT -- SAME AS 2-openmp.c AND 2-openmp-stream.c
   --------------------------------------------------------
   The split is by INTERFACE, not by position:

     heat_update_boundary_bottom   k = 1        only where k=1 is SENT
     heat_update_boundary_top      k = nz-2     only where k=nz-2 is SENT
     heat_update_interior          k_start..k_end   everything else

   With two devices there is ONE interface, so each device sends exactly one
   plane and runs exactly one boundary kernel:

     GPU0 folds k=1      into the interior (k_start=1); runs TOP only.
     GPU1 folds k=nz1-2  into the interior (k_end=nz1-2); runs BOTTOM only.

   Per timestep, per thread:
     interior   nowait                              long pole, no token
     boundary   nowait  depend(out:   its halo token)
     peer copy  async   depend(inout: its halo token)  -- PUSH straight into
                                    the neighbour's ghost, sourced from OUR
                                    OWN device
     taskwait                       -- drains interior and the peer copy
     barrier                        -- both threads' pushes have landed
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

     boundary kernel   overlaps the interior
     peer copy         overlaps the interior

   The interior is drained only by the taskwait immediately before the swap.
   This matches 2-openmp-stream.c, which achieves the same coverage from one
   driver thread with a single selective taskwait.

   LAUNCH ORDER
   ------------
   The interior kernel is launched FIRST on each device, matching every other
   -stream variant in the set, so it has the longest possible time to run
   behind the boundary kernel and the exchange.
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
// ASYNC: returns immediately; completion is published on dep_token.
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
// Computes k=nz-2, the plane sent UPWARD. Used by GPU 0.
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
// Computes k = k_start .. k_end. The explicit bounds let each device fold in
// the plane it never sends, which has no reason to sit in a boundary kernel.
// ASYNC: returns immediately. No dependency token -- nothing consumes this
// kernel's completion; it is drained by the plain taskwait at the end of
// the step.
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

  printf("\n**** TWO GPUS OpenMP Offloading, P2P + 2 HOST THREADS STARTS ****\n");

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

  if (num_devices < 2) {
    fprintf(stderr, "ERROR: this version needs 2 OpenMP devices, %d present\n",
            num_devices);
    exit(EXIT_FAILURE);
  }

  printf("Using OpenMP target devices 0 and 1 of %d\n", num_devices);

  // -- Domain decomposition along k -----------------------------
  // The n-2 interior planes are split two ways; the remainder, if any, goes
  // to the higher-numbered device.
  int ng     = 1;
  int nz_int = n - 2;
  int mid    = nz_int / 2;

  int nz0 = mid            + 2*ng;
  int nz1 = (nz_int - mid) + 2*ng;

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

  // -- NO halo buffers of any kind -------------------------------
  // In [k][j][i] layout XY planes are contiguous, so a peer copy addresses
  // one directly by pointer offset on each side. Nothing is staged, packed
  // or unpacked, on the host or on the device. halo_bytes is still needed,
  // as the transfer size.
  //
  // 2-openmp-stream.c and 2-openmp-stream-omp.c allocate h_halo0_up and
  // h_halo1_dn here. Their absence is the whole difference. The token names
  // keep the _up / _dn suffixes so the three files stay diffable.

  // Fix the step count before the parallel region: both threads must run
  // exactly the same number of iterations, and neither should accumulate its
  // own floating-point clock.
  int nsteps = 0;
  for (double tt=0.0; tt < T; tt += dt) nsteps++;

  printf("\n=== SIMULATION STARTS ===\n");

  // Accumulated inside the parallel region, around the time loop only.
  double compute_timer = 0.0;

  // -- Dependency tokens and depend-objects ----------------------
  // ASYNC ONLY. The tokens are never read or written; the runtime orders
  // tasks by their addresses.
  //
  // ONE token per halo, exactly as in 2-openmp-stream.c and
  // 2-openmp-stream-omp.c. Each describes a two-task chain here, since there
  // is no host hop in the middle:
  //
  //   tk_halo0_up :  GPU0 top boundary -> peer copy into GPU1's ghost k=0
  //   tk_halo1_dn :  GPU1 bot boundary -> peer copy into GPU0's ghost nz0-1
  //
  // Declared OUTSIDE the parallel region even though this file has one host
  // thread per device: thread 0 owns the _up chain, thread 1 owns the _dn
  // chain, and neither ever touches the other's. Nothing crosses, so nothing
  // needs to be private. That is what the PUSH arrangement buys.
  //
  // Only the boundary kernels publish completion. The interior kernels are
  // synchronised by the taskwait at the end of the step and need no token.
  int tk_halo0_up = 0;
  int tk_halo1_dn = 0;

  omp_depend_t halo0_up_dep;
  omp_depend_t halo1_dn_dep;

#pragma omp depobj(halo0_up_dep) depend(inout: tk_halo0_up)
#pragma omp depobj(halo1_dn_dep) depend(inout: tk_halo1_dn)

  // ======================================================================
  // Time loop -- thread tid drives device tid
  //
  // Each GPU computes, then pushes its boundary plane straight into the
  //   neighbour's ghost plane. One transfer per interface, no host hop, no
  //   pack or unpack.
  //
  // Transfer map (two transfers, one interface):
  //   GPU0 plane k=nz0-2  ->  GPU1 ghost k=0
  //   GPU1 plane k=1      ->  GPU0 ghost k=nz0-1
  //
  // The ghost planes at GPU0 k=0 and GPU1 k=nz1-1 are physical domain
  // boundaries. They are initialised to zero and never written.
  //
  // ASYNC: interior is launched first so it has the longest possible time to
  //   run behind the boundary kernel and the exchange.
  // ======================================================================

#pragma omp parallel num_threads(2) default(shared)
  {
    int tid       = omp_get_thread_num();
    int device_id = tid;

    // num_threads() is a request, not a guarantee. With one thread the
    // barriers become no-ops and GPU1 is never driven, which would silently
    // produce a wrong answer rather than fail.
    if (omp_get_num_threads() != 2) {
#pragma omp critical
      fprintf(stderr, "ERROR: this version needs 2 host threads, got %d\n",
              omp_get_num_threads());
      abort();
    }

    // Time the loop only, matching the other multi-GPU versions.
    // Both reads must come from the SAME thread: omp_get_wtime is only
    // required to be consistent per thread, not across threads -- which is
    // why this is `if (tid == 0)` and not `single`, since two `single`
    // regions may be executed by two different threads.
#pragma omp barrier
    if (tid == 0) compute_timer -= omp_get_wtime();
#pragma omp barrier

    for (int s=0; s<nsteps; s++) {

      // -- This thread's compute, then its peer PUSH ---------------
      // Every operation below SOURCES only data THIS thread produced, so
      // the whole chain is ordered by this thread's own token. See PUSH,
      // NOT PULL in the header.
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
      else {  // tid == 1
        // GPU1: no top boundary -- k=nz1-2 is never sent, so it is folded
        // into the interior.
        heat_update_interior(d1_u_new, d1_u_old, d1_rhs, d1_kappa,
                             nz1, ny, nx, factor, 2, nz1-2, device_id);
        heat_update_boundary_bottom(d1_u_new, d1_u_old, d1_rhs, d1_kappa,
                                    nz1, ny, nx, factor, device_id,
                                    &tk_halo1_dn);

        // PEER COPY: PUSH our plane k=1 straight into GPU0's ghost k=nz0-1
        // -- last link on our own chain.
        check_copy(omp_target_memcpy_async(d0_u_new + (size_t)(nz0-1) * ny * nx,
                                           d1_u_new + (size_t)1 * ny * nx,
                                           halo_bytes, 0, 0, 0, device_id,
                                           1, &halo1_dn_dep),
                   "P2P dev1 -> dev0 ghost top");
      }

      // -- Drain: this thread's interior kernel and its peer copy ---
      // The interior has been in flight across the entire exchange.
      //#pragma omp taskwait

      // -- Barrier: both threads have drained, so both ghost planes are
      //    written before the swap turns them into u_old ---------------
#pragma omp barrier

      // -- Pointer swap (one thread swaps both sets) ----------------
      // `single` carries an implicit barrier at its end, which is what
      // makes the swapped pointers visible to both threads. No explicit
      // barrier is needed here.
#pragma omp single
      {
        double *tmp;
        tmp = d0_u_old; d0_u_old = d0_u_new; d0_u_new = tmp;
        tmp = d1_u_old; d1_u_old = d1_u_new; d1_u_new = tmp;
      }
    }

#pragma omp barrier
    if (tid == 0) compute_timer += omp_get_wtime();

  }

  // -- Tear down depend-objects ----------------------------------
#pragma omp depobj(halo0_up_dep) destroy
#pragma omp depobj(halo1_dn_dep) destroy

  printf("\n%d timesteps complete\n", nsteps);
  printf("Time = %f seconds\n", compute_timer);
  printf("=== SIMULATION ENDS ===\n");

  // -- Copy back (layout [k][j][i]) ------------------------------
  check_copy(omp_target_memcpy(h_u0, d0_u_old, size0, 0, 0, host_id, 0),
             "copy-back GPU0");
  check_copy(omp_target_memcpy(h_u1, d1_u_old, size1, 0, 0, host_id, 1),
             "copy-back GPU1");

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
    write_u_values("u_openmp_stream_omp_p2p_2gpu.txt", h_u_global, n);

  // -- Cleanup ---------------------------------------------------
  omp_target_free(d0_u_old, 0); omp_target_free(d0_u_new, 0);
  omp_target_free(d0_rhs, 0);   omp_target_free(d0_kappa, 0);

  omp_target_free(d1_u_old, 1); omp_target_free(d1_u_new, 1);
  omp_target_free(d1_rhs, 1);   omp_target_free(d1_kappa, 1);

  free(h_u_global);
  free(h_u0);     free(h_u1);
  free(h_rhs0);   free(h_rhs1);
  free(h_kappa0); free(h_kappa1);

  printf("\n**** TWO GPUS OpenMP Offloading, P2P + 2 HOST THREADS FINISHED ****\n");
  return 0;
}
