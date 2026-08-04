//-*-c++-*-
/* OpenMP target offload implementation of the 3D heat conduction problem
         du/dt = div ( kappa(x,y,z) grad u ) + f(x,y,z)
   where kappa(x,y,z) = 0.5+0.45*sin(pi*x)*sin(pi*y)*sin(pi*z),
   using explicit time stepping.

   FOUR-GPU VERSION, ASYNCHRONOUS, halo exchange staged through the host.
   ONE HOST THREAD drives all four devices.

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

   This file is kept deliberately parallel to 4-openmp.c: everything outside
   the asynchrony should be identical between them, so that a diff shows only
   what overlap adds. It is also kept parallel to 2-openmp-stream.c: the
   dependence machinery is the same, generalised from two halos to six.

   WHAT IS NEW HERE RELATIVE TO 4-openmp.c
   ---------------------------------------
   4-openmp.c splits the update into boundary and interior kernels but then
   waits for all of them before transferring, so the split buys nothing. Here
   the split earns its keep: the boundary planes are computed and shipped
   while the interior kernels are still running.

   Every kernel carries `nowait`, making it a deferred target task, and
   ordering is expressed through `depend` clauses on dependency tokens rather
   than through explicit queues. The tokens are plain ints; the runtime
   tracks dependences by their address, not their value.

   The whole step is ONE dependence graph. Nothing waits on the host until
   the final taskwait before the pointer swap.

   KERNEL SPLIT -- SAME AS 4-openmp.c
   ----------------------------------
   The split is by INTERFACE, not by position:

     heat_update_boundary_bottom   k = 1        only where k=1 is SENT
     heat_update_boundary_top      k = nz-2     only where k=nz-2 is SENT
     heat_update_interior          k_start..k_end   everything else

   A boundary kernel exists so that an EXCHANGED plane finishes early and its
   halo copy can be issued without waiting for the bulk of the domain -- which
   is precisely what this version exploits. A plane that is never sent has no
   such urgency, so folding it into the interior keeps the boundary kernel as
   small as possible. Device 0 folds in k=1 (k_start=1), device 3 folds in
   k=nz3-2 (k_end=nz3-2).

   ONE TOKEN PER HALO
   ------------------
   2-openmp-stream.c has exactly two dependence edges to express -- one halo
   per GPU -- so it has two tokens and two depend-objects. Here there are six
   halos, hence six tokens and six depend-objects, named to match the staging
   buffers:

     tk_halo0_up   published by GPU0's TOP    boundary kernel
     tk_halo1_dn   published by GPU1's BOTTOM boundary kernel
     tk_halo1_up   published by GPU1's TOP    boundary kernel
     tk_halo2_dn   published by GPU2's BOTTOM boundary kernel
     tk_halo2_up   published by GPU2's TOP    boundary kernel
     tk_halo3_dn   published by GPU3's BOTTOM boundary kernel

   Each token describes a THREE-TASK CHAIN, and one `inout` depend-object per
   token expresses the whole chain:

     tk_halo0_up :  GPU0 top boundary -> D2H -> H2D into GPU1's ghost k=0
     tk_halo1_dn :  GPU1 bot boundary -> D2H -> H2D into GPU0's ghost nz0-1
     tk_halo1_up :  GPU1 top boundary -> D2H -> H2D into GPU2's ghost k=0
     tk_halo2_dn :  GPU2 bot boundary -> D2H -> H2D into GPU1's ghost nz1-1
     tk_halo2_up :  GPU2 top boundary -> D2H -> H2D into GPU3's ghost k=0
     tk_halo3_dn :  GPU3 bot boundary -> D2H -> H2D into GPU2's ghost nz2-1

   `inout` on each link waits for the previous task on that chain and
   republishes, so no further edges are needed and the SAME depend-object
   serves both copies. SIX objects in total, not twelve -- a depend-object is
   a handle describing a dependence, not something consumed by use.

   The H2D on a chain reads the buffer the D2H on that same chain filled, so
   the H2D into GPU0's ghost carries halo1_dn_dep, not halo0_up_dep. The
   exchange crosses; the dependence chains do not.

   `inout` on the H2D also gives the anti-dependence for free: next step's
   D2H is a writer of the same token, so it cannot overwrite a staging buffer
   while this step's H2D is still reading it.

   The tokens are PER HALO, never shared. A shared token would create
   write-after-write edges between unrelated devices' kernels and serialise
   them, which is the opposite of what this version is for.

   The interior kernels carry NO token. Nothing consumes their completion;
   they are drained by the plain taskwait at the end of the step.

   Per timestep:
     GPU0  interior   nowait                             long pole, no token
           top        nowait  depend(out:   tk_halo0_up)
     GPU1  interior   nowait                             long pole, no token
           bottom     nowait  depend(out:   tk_halo1_dn)
           top        nowait  depend(out:   tk_halo1_up)
     GPU2  interior   nowait                             long pole, no token
           bottom     nowait  depend(out:   tk_halo2_dn)
           top        nowait  depend(out:   tk_halo2_up)
     GPU3  interior   nowait                             long pole, no token
           bottom     nowait  depend(out:   tk_halo3_dn)

     D2H x6           async   depend(inout: its own token)

     H2D x6           async   depend(inout: the token of the chain whose
                              buffer this copy READS -- same object as that
                              chain's D2H)

       GPU0's ghost is written from h_halo1_dn, so that H2D continues the
       tk_halo1_dn chain. GPU1's lower ghost is written from h_halo0_up, so
       that H2D continues the tk_halo0_up chain. And so on for all six. The
       dependence is expressed on the copy itself, so THE HOST NEVER BLOCKS
       MID-STEP: it issues sixteen tasks and the runtime orders them.

       This is legal because ONE host thread generates every task, so they
       are all siblings. A version with one host thread per device cannot do
       this: depend clauses order sibling tasks, and per-device threads put
       their target tasks in different implicit task regions, so cross-device
       ordering there needs either a barrier or a PUSH arrangement in which
       each thread's H2D reads only its own buffer. EXPRESSING THE ENTIRE
       STEP AS ONE DEPENDENCE GRAPH IS THIS VERSION'S DISTINGUISHING
       PROPERTY.

     plain taskwait   drains the interior kernels and all six H2D before the
                      swap. This is the ONLY host-side wait in the step.

   NOTE ON EARLIER REVISIONS. This file once used a selective
   `taskwait depend(in: ...)` on all six tokens between the D2H and the H2D
   block, with the H2D copies carrying no dependences (0, NULL). That works,
   but it blocks the host mid-step for no reason: stating the dependence on
   the H2D itself is strictly better and is what 2-openmp-stream.c does. The
   selective wait is gone.

   The whole -stream set now uses ONE token per halo with `inout`. An earlier
   note here claimed the -omp and -omp-p2p variants used a PAIR of tokens
   (tk_bnd_* / tk_d2h_*) with separate `in` and `out` edges; they no longer
   do, and the same stale note still stands in 2-openmp-stream.c.

   LAUNCH ORDER
   ------------
   The interior kernel is launched FIRST on each device, so it has the
   longest possible time to run behind the boundary kernels, the D2H, the
   exchange and the H2D. This matches 2-openmp-stream.c. (4-openmp.c is
   boundary-first, matching 2-openmp.c, where the order is immaterial because
   a single taskwait drains everything.)
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

  printf("\n**** FOUR GPUS OpenMP Offloading, ASYNC STARTS ****\n");

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
  // Six buffers: three interfaces, two directions each.
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

  // -- Dependency tokens and depend-objects ----------------------
  // ASYNC ONLY. The tokens are never read or written; the runtime orders
  // tasks by their addresses. One host thread owns all of them.
  //
  // There are exactly six dependence edges to express -- one per halo -- so
  // six tokens and six depend-objects, named to match the staging buffers.
  //
  // Only the boundary kernels publish completion. The interior kernels are
  // synchronised by the plain taskwait at the end of the step and need no
  // token. `omp_target_memcpy_async` takes a depend-object list rather than
  // a clause, which is why these objects exist at all.
  int tk_halo0_up = 0;
  int tk_halo1_dn = 0, tk_halo1_up = 0;
  int tk_halo2_dn = 0, tk_halo2_up = 0;
  int tk_halo3_dn = 0;

  // ONE depend-object per halo, `inout`, reused by BOTH copies on that halo.
  // Each token describes a three-task chain:
  //
  //   tk_halo0_up :  GPU0 top boundary -> D2H -> H2D into GPU1's ghost k=0
  //   tk_halo1_dn :  GPU1 bot boundary -> D2H -> H2D into GPU0's ghost nz0-1
  //   tk_halo1_up :  GPU1 top boundary -> D2H -> H2D into GPU2's ghost k=0
  //   tk_halo2_dn :  GPU2 bot boundary -> D2H -> H2D into GPU1's ghost nz1-1
  //   tk_halo2_up :  GPU2 top boundary -> D2H -> H2D into GPU3's ghost k=0
  //   tk_halo3_dn :  GPU3 bot boundary -> D2H -> H2D into GPU2's ghost nz2-1
  //
  // `inout` on every link both waits for the previous task on that chain and
  // republishes, so the chain is linear without any further edges. A
  // depend-object is a handle describing a dependence, not something consumed
  // by use, so the same object serves the D2H and the H2D.
  //
  // The H2D on a chain reads the buffer the D2H on that SAME chain filled --
  // which is why the H2D into GPU0's ghost carries halo1_dn_dep and not
  // halo0_up_dep. The exchange crosses; the dependence chains do not.
  omp_depend_t halo0_up_dep;
  omp_depend_t halo1_dn_dep, halo1_up_dep;
  omp_depend_t halo2_dn_dep, halo2_up_dep;
  omp_depend_t halo3_dn_dep;

#pragma omp depobj(halo0_up_dep) depend(inout: tk_halo0_up)
#pragma omp depobj(halo1_dn_dep) depend(inout: tk_halo1_dn)
#pragma omp depobj(halo1_up_dep) depend(inout: tk_halo1_up)
#pragma omp depobj(halo2_dn_dep) depend(inout: tk_halo2_dn)
#pragma omp depobj(halo2_up_dep) depend(inout: tk_halo2_up)
#pragma omp depobj(halo3_dn_dep) depend(inout: tk_halo3_dn)

  double t     = 0.0;
  int    steps = 0;

  printf("\n=== SIMULATION STARTS ===\n");

  double compute_timer = 0.0;
  compute_timer -= omp_get_wtime();

  // ======================================================================
  // Time loop
  //
  // PHASE 1: each GPU computes, then async D2H directly from the contiguous
  //   boundary plane -- no pack kernel needed.
  //
  // PHASE 2: async H2D directly into the contiguous ghost plane -- no
  //   unpack.
  //
  // Transfer map (six transfers, three interfaces):
  //   GPU0 plane k=nz0-2  ->  h_halo0_up  ->  GPU1 ghost k=0
  //   GPU1 plane k=1      ->  h_halo1_dn  ->  GPU0 ghost k=nz0-1
  //   GPU1 plane k=nz1-2  ->  h_halo1_up  ->  GPU2 ghost k=0
  //   GPU2 plane k=1      ->  h_halo2_dn  ->  GPU1 ghost k=nz1-1
  //   GPU2 plane k=nz2-2  ->  h_halo2_up  ->  GPU3 ghost k=0
  //   GPU3 plane k=1      ->  h_halo3_dn  ->  GPU2 ghost k=nz2-1
  //
  // The ghost planes at GPU0 k=0 and GPU3 k=nz3-1 are physical domain
  // boundaries. They are initialised to zero and never written.
  //
  // ASYNC: interior is launched first so it has the longest possible time to
  //   run behind the boundary kernels, the D2H, the exchange and the H2D.
  // ======================================================================

  while (t < T) {

    // -- GPU 0: compute (no bottom boundary -- k=1 is never sent) -
    heat_update_interior(d0_u_new, d0_u_old, d0_rhs, d0_kappa,
                         nz0, ny, nx, factor, 1, nz0-3, 0);
    heat_update_boundary_top(d0_u_new, d0_u_old, d0_rhs, d0_kappa,
                             nz0, ny, nx, factor, 0, &tk_halo0_up);

    // -- GPU 1: compute (both boundaries) -------------------------
    heat_update_interior(d1_u_new, d1_u_old, d1_rhs, d1_kappa,
                         nz1, ny, nx, factor, 2, nz1-3, 1);
    heat_update_boundary_bottom(d1_u_new, d1_u_old, d1_rhs, d1_kappa,
                                nz1, ny, nx, factor, 1, &tk_halo1_dn);
    heat_update_boundary_top(d1_u_new, d1_u_old, d1_rhs, d1_kappa,
                             nz1, ny, nx, factor, 1, &tk_halo1_up);

    // -- GPU 2: compute (both boundaries) -------------------------
    heat_update_interior(d2_u_new, d2_u_old, d2_rhs, d2_kappa,
                         nz2, ny, nx, factor, 2, nz2-3, 2);
    heat_update_boundary_bottom(d2_u_new, d2_u_old, d2_rhs, d2_kappa,
                                nz2, ny, nx, factor, 2, &tk_halo2_dn);
    heat_update_boundary_top(d2_u_new, d2_u_old, d2_rhs, d2_kappa,
                             nz2, ny, nx, factor, 2, &tk_halo2_up);

    // -- GPU 3: compute (no top boundary -- k=nz3-2 is never sent) -
    heat_update_interior(d3_u_new, d3_u_old, d3_rhs, d3_kappa,
                         nz3, ny, nx, factor, 2, nz3-2, 3);
    heat_update_boundary_bottom(d3_u_new, d3_u_old, d3_rhs, d3_kappa,
                                nz3, ny, nx, factor, 3, &tk_halo3_dn);

    // -- PHASE 1: D2H from contiguous boundary planes (async) -----
    // `inout` on the halo token: each copy waits for the boundary kernel
    // that wrote its plane, then becomes that token's last writer.
    check_copy(omp_target_memcpy_async(h_halo0_up,
                                       d0_u_new + (size_t)(nz0-2) * ny * nx,
                                       halo_bytes, 0, 0, host_id, 0,
                                       1, &halo0_up_dep),
               "D2H dev0 up");

    check_copy(omp_target_memcpy_async(h_halo1_dn,
                                       d1_u_new + (size_t)1 * ny * nx,
                                       halo_bytes, 0, 0, host_id, 1,
                                       1, &halo1_dn_dep),
               "D2H dev1 dn");
    check_copy(omp_target_memcpy_async(h_halo1_up,
                                       d1_u_new + (size_t)(nz1-2) * ny * nx,
                                       halo_bytes, 0, 0, host_id, 1,
                                       1, &halo1_up_dep),
               "D2H dev1 up");

    check_copy(omp_target_memcpy_async(h_halo2_dn,
                                       d2_u_new + (size_t)1 * ny * nx,
                                       halo_bytes, 0, 0, host_id, 2,
                                       1, &halo2_dn_dep),
               "D2H dev2 dn");
    check_copy(omp_target_memcpy_async(h_halo2_up,
                                       d2_u_new + (size_t)(nz2-2) * ny * nx,
                                       halo_bytes, 0, 0, host_id, 2,
                                       1, &halo2_up_dep),
               "D2H dev2 up");

    check_copy(omp_target_memcpy_async(h_halo3_dn,
                                       d3_u_new + (size_t)1 * ny * nx,
                                       halo_bytes, 0, 0, host_id, 3,
                                       1, &halo3_dn_dep),
               "D2H dev3 dn");

    // -- PHASE 2: H2D into contiguous ghost planes (async) ---------
    // Each copy waits, via its depend-object, for the D2H that filled the
    // buffer it reads -- so the object is the one belonging to the SENDING
    // device's chain, not the receiving device's. No host-side wait is
    // needed here: the runtime orders D2H before H2D, and all four interior
    // kernels stay in flight throughout.
    check_copy(omp_target_memcpy_async(d0_u_new + (size_t)(nz0-1) * ny * nx,
                                       h_halo1_dn,
                                       halo_bytes, 0, 0, 0, host_id,
                                       1, &halo1_dn_dep),
               "H2D dev0 ghost top");

    check_copy(omp_target_memcpy_async(d1_u_new + (size_t)0 * ny * nx,
                                       h_halo0_up,
                                       halo_bytes, 0, 0, 1, host_id,
                                       1, &halo0_up_dep),
               "H2D dev1 ghost bottom");
    check_copy(omp_target_memcpy_async(d1_u_new + (size_t)(nz1-1) * ny * nx,
                                       h_halo2_dn,
                                       halo_bytes, 0, 0, 1, host_id,
                                       1, &halo2_dn_dep),
               "H2D dev1 ghost top");

    check_copy(omp_target_memcpy_async(d2_u_new + (size_t)0 * ny * nx,
                                       h_halo1_up,
                                       halo_bytes, 0, 0, 2, host_id,
                                       1, &halo1_up_dep),
               "H2D dev2 ghost bottom");
    check_copy(omp_target_memcpy_async(d2_u_new + (size_t)(nz2-1) * ny * nx,
                                       h_halo3_dn,
                                       halo_bytes, 0, 0, 2, host_id,
                                       1, &halo3_dn_dep),
               "H2D dev2 ghost top");

    check_copy(omp_target_memcpy_async(d3_u_new + (size_t)0 * ny * nx,
                                       h_halo2_up,
                                       halo_bytes, 0, 0, 3, host_id,
                                       1, &halo2_up_dep),
               "H2D dev3 ghost bottom");

    // The end-of-step taskwait drains the interior kernels and both 
    // complete halo chains before pointer swapping.
#pragma omp taskwait

    // -- Pointer swap --------------------------------------------
    double *tmp;
    tmp = d0_u_old; d0_u_old = d0_u_new; d0_u_new = tmp;
    tmp = d1_u_old; d1_u_old = d1_u_new; d1_u_new = tmp;
    tmp = d2_u_old; d2_u_old = d2_u_new; d2_u_new = tmp;
    tmp = d3_u_old; d3_u_old = d3_u_new; d3_u_new = tmp;

    steps++;
    t += dt;
  }

  compute_timer += omp_get_wtime();

  // -- Tear down depend-objects ----------------------------------
#pragma omp depobj(halo0_up_dep) destroy
#pragma omp depobj(halo1_dn_dep) destroy
#pragma omp depobj(halo1_up_dep) destroy
#pragma omp depobj(halo2_dn_dep) destroy
#pragma omp depobj(halo2_up_dep) destroy
#pragma omp depobj(halo3_dn_dep) destroy

  printf("\n%d timesteps complete\n", steps);
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
    write_u_values("u_openmp_stream_4gpu.txt", h_u_global, n);

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

  printf("\n**** FOUR GPUS OpenMP Offloading, ASYNC FINISHED ****\n");
  return 0;
}
