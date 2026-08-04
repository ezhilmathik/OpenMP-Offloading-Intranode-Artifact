//-*-c++-*-
/* OpenMP target offload implementation of the 3D heat conduction problem
         du/dt = div ( kappa(x,y,z) grad u ) + f(x,y,z)
   where kappa(x,y,z) = 0.5+0.45*sin(pi*x)*sin(pi*y)*sin(pi*z),
   using explicit time stepping.

   FOUR-GPU VERSION, halo exchange staged through the host.
   ONE HOST THREAD drives all four devices.

   Layout: u[k][j][i]  ->  k is slowest, i is fastest
     offset = k*(ny*nx) + j*nx + i
   Decomposition along the k axis, so an XY plane at fixed k is CONTIGUOUS
   and no pack/unpack kernels are needed:
     D2H: omp_target_memcpy(h_halo, d_u_new + k*(ny*nx), halo_bytes)
     H2D: omp_target_memcpy(d_u_new + ghost_k*(ny*nx), h_halo, halo_bytes)
   Neighbour offsets:
     i+-1  ->  +-1
     j+-1  ->  +-nx
     k+-1  ->  +-ny*nx

   This file is kept deliberately parallel to 2-openmp.c: everything outside
   the device decomposition should be identical between them, so that a diff
   shows only what is intrinsic to using four devices rather than two.

   WHAT IS INTRINSIC TO FOUR DEVICES
   ---------------------------------
   Three interfaces instead of one, so six halo transfers per step instead
   of two, and six host staging buffers instead of two. Devices 1 and 2 have
   a neighbour on both sides; devices 0 and 3 have one neighbour and one
   physical domain boundary.

   KERNEL SPLIT -- WHY IT DIFFERS FROM THE TWO-GPU VERSION
   -------------------------------------------------------
   2-openmp.c uses one boundary kernel covering both planes k=1 and k=nz-2.
   With four devices the split is by INTERFACE rather than by position:

     heat_update_boundary_bottom   k = 1        only where k=1 is SENT
     heat_update_boundary_top      k = nz-2     only where k=nz-2 is SENT
     heat_update_interior          k_start..k_end   everything else

   A boundary kernel exists so that an EXCHANGED plane finishes early and
   its halo copy can be issued without waiting for the bulk of the domain.
   A plane that is never sent has no such urgency, so folding it into the
   interior kernel is strictly better: it keeps the boundary kernel as small
   as possible, and the boundary kernel is what gates the exchange.

   Device 0 has no lower neighbour, so its plane k=1 is never sent and is
   folded into the interior (k_start=1). Device 3 has no upper neighbour, so
   its plane k=nz3-2 is folded in at the other end (k_end=nz3-2). Devices 1
   and 2 send both planes and so run both boundary kernels.

   Per device the union of the kernels is k = 1 .. nz-2, exactly the owned
   planes:

     GPU0   interior 1 .. nz0-3   + top             = 1 .. nz0-2
     GPU1   interior 2 .. nz1-3   + bottom + top    = 1 .. nz1-2
     GPU2   interior 2 .. nz2-3   + bottom + top    = 1 .. nz2-2
     GPU3   interior 2 .. nz3-2   + bottom          = 1 .. nz3-2

   LAUNCH ORDER
   ------------
   The boundary kernels are launched BEFORE the interior kernel on each
   device, matching 2-openmp.c. In THIS file the order is immaterial: every
   kernel is drained by the single taskwait below before any transfer is
   issued, so nothing is gated on a boundary kernel finishing early. The
   -stream variants launch the interior FIRST, for the opposite reason --
   there the interior is meant to run behind the whole exchange. That
   difference mirrors the two-GPU set exactly (2-openmp.c is boundary-first,
   2-openmp-stream.c is interior-first) and is deliberate.

   HALO BUFFER NAMING
   ------------------
   2-openmp.c indexes its two staging buffers by sender: h_halo0 is the
   plane device 0 sends, h_halo1 the plane device 1 sends. With four
   devices the middle two each send two planes, so the sender index alone
   no longer identifies a buffer and a direction is appended:

     h_haloN_up   plane device N sends UPWARD   (to device N+1)
     h_haloN_dn   plane device N sends DOWNWARD (to device N-1)

   In those terms 2-openmp.c's h_halo0 is h_halo0_up and its h_halo1 is
   h_halo1_dn.
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
   fail at once; harmless outside a parallel region. */
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
// The loop body is identical to the two inline initialisation loops in
// 2-openmp.c; with four partitions it is factored into a function rather
// than written out four times. Boundary values are forced to exact zero.
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
// `nz` is unused here (k is fixed at 1); the parameter is kept so that all
// three kernels share one signature shape.
static void heat_update_boundary_bottom(double *u_new, const double *u_old,
                                        const double *rhs, const double *kappa,
                                        int nz, int ny, int nx,
                                        double factor, int device_id)
{
  (void)nz;
#pragma omp target is_device_ptr(u_new, u_old, rhs, kappa) device(device_id) nowait
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
static void heat_update_boundary_top(double *u_new, const double *u_old,
                                     const double *rhs, const double *kappa,
                                     int nz, int ny, int nx,
                                     double factor, int device_id)
{
#pragma omp target is_device_ptr(u_new, u_old, rhs, kappa) device(device_id) nowait
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
#pragma omp target is_device_ptr(u_new, u_old, rhs, kappa) device(device_id) nowait
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

  printf("\n**** FOUR GPUS OpenMP Offloading STARTS ****\n");

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
  // ======================================================================

  while (t < T) {

    // -- GPU 0: compute (no bottom boundary -- k=1 is never sent) -
    heat_update_interior(d0_u_new, d0_u_old, d0_rhs, d0_kappa,
                         nz0, ny, nx, factor, 1, nz0-3, 0);    
    heat_update_boundary_top(d0_u_new, d0_u_old, d0_rhs, d0_kappa,
                             nz0, ny, nx, factor, 0);

    // -- GPU 1: compute (both boundaries) -------------------------
    heat_update_interior(d1_u_new, d1_u_old, d1_rhs, d1_kappa,
                         nz1, ny, nx, factor, 2, nz1-3, 1);    
    heat_update_boundary_bottom(d1_u_new, d1_u_old, d1_rhs, d1_kappa,
                                nz1, ny, nx, factor, 1);
    heat_update_boundary_top(d1_u_new, d1_u_old, d1_rhs, d1_kappa,
                             nz1, ny, nx, factor, 1);

    // -- GPU 2: compute (both boundaries) -------------------------
    heat_update_interior(d2_u_new, d2_u_old, d2_rhs, d2_kappa,
                         nz2, ny, nx, factor, 2, nz2-3, 2);
    heat_update_boundary_bottom(d2_u_new, d2_u_old, d2_rhs, d2_kappa,
                                nz2, ny, nx, factor, 2);
    heat_update_boundary_top(d2_u_new, d2_u_old, d2_rhs, d2_kappa,
                             nz2, ny, nx, factor, 2);

    // -- GPU 3: compute (no top boundary -- k=nz3-2 is never sent) -
    heat_update_interior(d3_u_new, d3_u_old, d3_rhs, d3_kappa,
                         nz3, ny, nx, factor, 2, nz3-2, 3);    
    heat_update_boundary_bottom(d3_u_new, d3_u_old, d3_rhs, d3_kappa,
                                nz3, ny, nx, factor, 3);

#pragma omp taskwait

    // -- PHASE 1: D2H from contiguous boundary planes -------------
    check_copy(omp_target_memcpy(h_halo0_up,
                                 d0_u_new + (size_t)(nz0-2) * ny * nx,
                                 halo_bytes, 0, 0, host_id, 0),
               "D2H dev0 up");

    check_copy(omp_target_memcpy(h_halo1_dn,
                                 d1_u_new + (size_t)1 * ny * nx,
                                 halo_bytes, 0, 0, host_id, 1),
               "D2H dev1 dn");

    check_copy(omp_target_memcpy(h_halo1_up,
                                 d1_u_new + (size_t)(nz1-2) * ny * nx,
                                 halo_bytes, 0, 0, host_id, 1),
               "D2H dev1 up");

    check_copy(omp_target_memcpy(h_halo2_dn,
                                 d2_u_new + (size_t)1 * ny * nx,
                                 halo_bytes, 0, 0, host_id, 2),
               "D2H dev2 dn");

    check_copy(omp_target_memcpy(h_halo2_up,
                                 d2_u_new + (size_t)(nz2-2) * ny * nx,
                                 halo_bytes, 0, 0, host_id, 2),
               "D2H dev2 up");

    check_copy(omp_target_memcpy(h_halo3_dn,
                                 d3_u_new + (size_t)1 * ny * nx,
                                 halo_bytes, 0, 0, host_id, 3),
               "D2H dev3 dn");

    // -- PHASE 2: H2D into contiguous ghost planes ----------------
    check_copy(omp_target_memcpy(d0_u_new + (size_t)(nz0-1) * ny * nx,
                                 h_halo1_dn,
                                 halo_bytes, 0, 0, 0, host_id),
               "H2D dev0 ghost top");

    check_copy(omp_target_memcpy(d1_u_new + (size_t)0 * ny * nx,
                                 h_halo0_up,
                                 halo_bytes, 0, 0, 1, host_id),
               "H2D dev1 ghost bottom");

    check_copy(omp_target_memcpy(d1_u_new + (size_t)(nz1-1) * ny * nx,
                                 h_halo2_dn,
                                 halo_bytes, 0, 0, 1, host_id),
               "H2D dev1 ghost top");

    check_copy(omp_target_memcpy(d2_u_new + (size_t)0 * ny * nx,
                                 h_halo1_up,
                                 halo_bytes, 0, 0, 2, host_id),
               "H2D dev2 ghost bottom");

    check_copy(omp_target_memcpy(d2_u_new + (size_t)(nz2-1) * ny * nx,
                                 h_halo3_dn,
                                 halo_bytes, 0, 0, 2, host_id),
               "H2D dev2 ghost top");

    check_copy(omp_target_memcpy(d3_u_new + (size_t)0 * ny * nx,
                                 h_halo2_up,
                                 halo_bytes, 0, 0, 3, host_id),
               "H2D dev3 ghost bottom");

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
    write_u_values("u_openmp_4gpu.txt", h_u_global, n);

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

  printf("\n**** FOUR GPUS OpenMP Offloading FINISHED ****\n");
  return 0;
}
