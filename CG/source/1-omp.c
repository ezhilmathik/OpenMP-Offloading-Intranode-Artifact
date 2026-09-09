/* ===========================================================================
 * 1-omp.c -- Conjugate Gradient, unpreconditioned, OpenMP target offload,
 *            1 GPU.
/* ===========================================================================

#include "matgen.h"
#include <time.h>
#include <omp.h>

static int DEV  = 0;    /* target device, from omp_get_default_device() */
static int HOST = 0;    /* omp_get_initial_device() */

/* y = A*x */
static void spmv(int n, const ptr_t *rowptr, const idx_t *col, const double *val,
                 const double *x, double *y)
{
  ptr_t  *rp = (ptr_t *)rowptr;
  idx_t  *cl = (idx_t *)col;
  double *vl = (double *)val;
  double *xx = (double *)x;
  int i;

#pragma omp target teams distribute parallel for schedule(static, 1) \
        is_device_ptr(rp, cl, vl, xx, y)
  for (i = 0; i < n; i++) {
    double sum = 0.0;
    ptr_t k;
    for (k = rp[i]; k < rp[i + 1]; k++)
      sum += vl[k] * xx[cl[k]];
    y[i] = sum;
  }
}


static double dot(int n, const double *a, const double *b)
{
  double *aa = (double *)a;
  double *bb = (double *)b;
  double s = 0.0;
  int i;

#pragma omp target teams distribute parallel for schedule(static, 1) \
        is_device_ptr(aa, bb) reduction(+ : s)
  for (i = 0; i < n; i++) s += aa[i] * bb[i];
  return s;
}

/* y = y + a*x */
static void axpy(int n, double a, const double *x, double *y)
{
  double *xx = (double *)x;
  int i;

#pragma omp target teams distribute parallel for schedule(static, 1) \
        is_device_ptr(xx, y)
  for (i = 0; i < n; i++) y[i] = y[i] + a * xx[i];
}

/* y = x + b*y */
static void xpby(int n, const double *x, double b, double *y)
{
  double *xx = (double *)x;
  int i;

#pragma omp target teams distribute parallel for schedule(static, 1) \
        is_device_ptr(xx, y)
  for (i = 0; i < n; i++) y[i] = xx[i] + b * y[i];
}

/* y = x */
static void vcopy(int n, const double *x, double *y)
{
  double *xx = (double *)x;
  int i;

#pragma omp target teams distribute parallel for schedule(static, 1) \
        is_device_ptr(xx, y)
  for (i = 0; i < n; i++) y[i] = xx[i];
}

/* y = v -- not one of the five; used only to reset x between -runs, where
 * cg_omp.c writes the loop inline. */
static void vfill(int n, double v, double *y)
{
  int i;

#pragma omp target teams distribute parallel for schedule(static, 1) \
        is_device_ptr(y)
  for (i = 0; i < n; i++) y[i] = v;
}

/* ===========================================================================
 * HOST-TO-DEVICE TRANSFER
 *
 * Chunked. At the 40 GB sweep point d_val alone is 21.5 GB, and a single
 * omp_target_memcpy of that size leans on the runtime handling a length that
 * exceeds anything it is usually asked for. Splitting into 1 GB pieces costs
 * nothing measurable at setup and removes the question.
 * ========================================================================= */
#define XFER_CHUNK ((size_t)1 << 30)

static void h2d(void *dst, const void *src, size_t bytes, const char *what)
{
  size_t off = 0;
  while (off < bytes) {
    size_t nb = bytes - off;
    if (nb > XFER_CHUNK) nb = XFER_CHUNK;
    if (omp_target_memcpy(dst, (void *)src, nb, off, off, DEV, HOST) != 0) {
      fprintf(stderr, "omp_target_memcpy failed: %s (offset %zu, %zu bytes)\n",
              what, off, nb);
      exit(1);
    }
    off += nb;
  }
}

static void *dev_alloc(size_t bytes, const char *what)
{
  void *p = omp_target_alloc(bytes, DEV);
  if (!p) {
    fprintf(stderr, "omp_target_alloc failed: %s (%zu bytes)\n", what, bytes);
    exit(1);
  }
  return p;
}

/* ===========================================================================
 * CG -- unpreconditioned, standard formulation.
 * Identical to cg_omp.c and 1-cuda.cu.
 *
 *   r = b - A x ;  p = r ;  rr = r.r
 *   repeat
 *     Ap    = A p
 *     alpha = rr / (p.Ap)
 *     x     = x + alpha p
 *     r     = r - alpha Ap
 *     rr_new = r.r ;  beta = rr_new / rr ;  rr = rr_new
 *     p     = r + beta p
 *
 * The TRUE residual ||b - Ax|| is recomputed at the end. The recursively
 * updated r drifts from it over hundreds of iterations, and trusting the
 * recursive value is the classic way to report a convergence that never
 * happened.
 * ========================================================================= */
static double wtime(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int cg_solve(int n, const ptr_t *rowptr, const idx_t *col, const double *val,
                    const double *b, double *x,
                    double *r, double *p, double *Ap,
                    double rtol, int maxit,
                    double *relres, double *tsolve, double *tspmv_total)
{
  double rr, rr_new, pAp, alpha, beta, bnorm, t0, t1, ts;
  int k;

  bnorm = sqrt(dot(n, b, b));
  if (bnorm == 0.0) bnorm = 1.0;

  spmv(n, rowptr, col, val, x, Ap);          /* r = b - A x */
  vcopy(n, b, r);
  axpy(n, -1.0, Ap, r);

  vcopy(n, r, p);
  rr = dot(n, r, r);

  *tspmv_total = 0.0;
  t0 = wtime();

  for (k = 0; k < maxit; k++) {
    if (sqrt(rr) / bnorm <= rtol) break;

    /* a target region without nowait is synchronous, so the host clock
     * brackets the kernel exactly -- same timing code as cg_omp.c */
    ts = wtime();
    spmv(n, rowptr, col, val, p, Ap);
    *tspmv_total += wtime() - ts;

    pAp   = dot(n, p, Ap);
    alpha = rr / pAp;

    axpy(n,  alpha, p,  x);
    axpy(n, -alpha, Ap, r);

    rr_new = dot(n, r, r);
    beta   = rr_new / rr;
    rr     = rr_new;

    xpby(n, r, beta, p);

    if ((k + 1) % 100 == 0)
      printf("    iter %6d   relres %.3e\n", k + 1, sqrt(rr) / bnorm);
  }

  t1 = wtime();

  spmv(n, rowptr, col, val, x, Ap);          /* TRUE residual */
  vcopy(n, b, r);
  axpy(n, -1.0, Ap, r);
  *relres = sqrt(dot(n, r, r)) / bnorm;
  *tsolve = t1 - t0;
  return k;
}

/* ===========================================================================
 * MAIN
 * ========================================================================= */
static int arg_i(int c, char **v, const char *k, int d)
{
  int i;
  for (i = 1; i < c - 1; i++) if (!strcmp(v[i], k)) return atoi(v[i + 1]);
  return d;
}
static double arg_d(int c, char **v, const char *k, double d)
{
  int i;
  for (i = 1; i < c - 1; i++) if (!strcmp(v[i], k)) return atof(v[i + 1]);
  return d;
}
static const char *arg_s(int c, char **v, const char *k, const char *d)
{
  int i;
  for (i = 1; i < c - 1; i++) if (!strcmp(v[i], k)) return v[i + 1];
  return d;
}

int main(int argc, char **argv)
{
  CSR A;
  const char *mmfile, *order;
  int npts, maxit, iters = 0, i, max_len, denom = 1, nruns, ndev;
  double avg_deg, shift, rtol, relres = 0.0, tsolve = 0.0, tspmv_total = 0.0;
  double t0, t1, tspmv;
  double mean_len, std_len, mean_bw, bytes, bw;
  double best_iter, best_spmv, sum_iter, sum_spmv, tsolve_iter;
  unsigned long long seed;
  double *hb;
  ptr_t  *d_rowptr;
  idx_t  *d_col;
  double *d_val, *d_b, *d_x, *d_r, *d_p, *d_Ap;
  double gpu_mem;

  npts    = arg_i(argc, argv, "-n", 1000000);
  avg_deg = arg_d(argc, argv, "-deg", 15.0);
  shift   = arg_d(argc, argv, "-shift", 0.1);
  rtol    = arg_d(argc, argv, "-tol", 1e-8);
  maxit   = arg_i(argc, argv, "-maxit", 1000);
  nruns   = arg_i(argc, argv, "-runs", 1);
  if (nruns < 1) nruns = 1;
  seed    = (unsigned long long)arg_i(argc, argv, "-seed", 12345);
  mmfile  = arg_s(argc, argv, "-mm", NULL);
  order   = arg_s(argc, argv, "-order", "natural");

  ndev = omp_get_num_devices();
  if (ndev < 1) {
    fprintf(stderr, "no OpenMP target device found -- was this built with "
                    "--offload-arch, and is a GPU visible?\n");
    return 1;
  }
  DEV  = omp_get_default_device();
  HOST = omp_get_initial_device();

  /* ----- matrix, built on the host exactly as in cg_omp.c ----- */
  t0 = wtime();
  if (mmfile) {
    if (matgen_mm(&A, mmfile, 0.0)) return 1;
  } else {
    if (matgen_radius(&A, npts, avg_deg, shift, seed)) {
      fprintf(stderr, "generator failed\n"); return 1;
    }
  }
  t1 = wtime();

  if (!strcmp(order, "rcm"))         reorder_rcm(&A);
  else if (!strcmp(order, "random")) reorder_random(&A, seed);
  else if (strcmp(order, "natural")) {
    fprintf(stderr, "unknown -order %s (natural|rcm|random)\n", order); return 1;
  }

  csr_stats(&A, &mean_len, &std_len, &max_len, &mean_bw);

  /* ----- upload ----- */
  d_rowptr = (ptr_t *) dev_alloc((size_t)(A.n + 1) * sizeof(ptr_t), "rowptr");
  d_col    = (idx_t *) dev_alloc((size_t)A.nnz * sizeof(idx_t), "col");
  d_val    = (double *)dev_alloc((size_t)A.nnz * sizeof(double), "val");
  d_b      = (double *)dev_alloc((size_t)A.n * sizeof(double), "b");
  d_x      = (double *)dev_alloc((size_t)A.n * sizeof(double), "x");
  d_r      = (double *)dev_alloc((size_t)A.n * sizeof(double), "r");
  d_p      = (double *)dev_alloc((size_t)A.n * sizeof(double), "p");
  d_Ap     = (double *)dev_alloc((size_t)A.n * sizeof(double), "Ap");

  h2d(d_rowptr, A.rowptr, (size_t)(A.n + 1) * sizeof(ptr_t), "rowptr");
  h2d(d_col,    A.col,    (size_t)A.nnz * sizeof(idx_t), "col");
  h2d(d_val,    A.val,    (size_t)A.nnz * sizeof(double), "val");

  /* RHS built on the host with the same hash cg_omp.c uses, so every version
   * solves a bit-identical problem. NOT the constant vector: it is an
   * eigenvector of this matrix and CG would converge in one iteration. */
  hb = (double *)mg_alloc((size_t)A.n * sizeof(double));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (i = 0; i < A.n; i++) hb[i] = 2.0 * mg_hash01(i) - 1.0;
  h2d(d_b, hb, (size_t)A.n * sizeof(double), "b");
  free(hb);

  vfill(A.n, 0.0, d_x);
  vfill(A.n, 0.0, d_r);
  vfill(A.n, 0.0, d_p);
  vfill(A.n, 0.0, d_Ap);

  gpu_mem = ((double)A.nnz * 12.0 + (double)(A.n + 1) * 8.0
             + (double)A.n * 5.0 * 8.0) / 1e9;

  printf("=========================================================\n");
  printf("  GPUs            : 1  (OpenMP target device %d of %d)\n", DEV, ndev);
  printf("  backend         : OpenMP target offload\n");
  printf("  matrix          : %s\n", mmfile ? mmfile : "synthetic radius graph");
  printf("  ordering        : %s\n", order);
  printf("  rows            : %d\n", A.n);
  printf("  nonzeros        : %lld\n", (long long)A.nnz);
  printf("  nnz/row         : mean %.2f, std %.2f, max %d\n", mean_len, std_len, max_len);
  printf("  mean |i-col|    : %.1f  (locality: lower is better)\n", mean_bw);
  printf("  device memory   : %.2f GB\n", gpu_mem);
  printf("  build time      : %.2f s  (host)\n", t1 - t0);
  printf("---------------------------------------------------------\n");

  /* ----- solve -----
   * No standalone SpMV benchmark; see the note at the top of this file.
   * Two levels of averaging: within a run over every CG iteration, and across
   * runs via -runs for machine noise.
   *
   * The first target region in the process pays a one-off kernel load cost.
   * It lands in cg_solve's setup (the r = b - A x SpMV, before t0), so it
   * never enters the timed loop. */
  {
    int run;
    double s_iter, s_spmv;
    best_iter = 1e30; best_spmv = 1e30;
    sum_iter = 0.0; sum_spmv = 0.0;

    for (run = 0; run < nruns; run++) {
      vfill(A.n, 0.0, d_x);                        /* same start each run */

      iters = cg_solve(A.n, d_rowptr, d_col, d_val, d_b, d_x,
                       d_r, d_p, d_Ap, rtol, maxit, &relres, &tsolve, &tspmv_total);

      denom  = iters ? iters : 1;
      s_iter = tsolve / denom;
      s_spmv = tspmv_total / denom;

      sum_iter += s_iter;  if (s_iter < best_iter) best_iter = s_iter;
      sum_spmv += s_spmv;  if (s_spmv < best_spmv) best_spmv = s_spmv;

      if (nruns > 1)
        printf("  run %d/%d        : %.4f ms/iter  (%d iters, relres %.2e)\n",
               run + 1, nruns, 1e3 * s_iter, iters, relres);
    }
    tsolve_iter = sum_iter / nruns;
    tspmv       = sum_spmv / nruns;
  }

  /* values + indices + row pointers + one read of x + one write of y.
   * Assumes perfect reuse of x, so a LOWER bound on traffic and therefore an
   * UPPER bound on the bandwidth figure. */
  bytes = (double)A.nnz * 12.0 + (double)A.n * 24.0;
  bw    = bytes / tspmv / 1e9;

  printf("---------------------------------------------------------\n");
  printf("  iterations      : %d%s\n", iters, iters >= maxit ? "  (hit -maxit cap)" : "");
  printf("  true rel. resid : %.6e\n", relres);
  printf("  runs            : %d\n", nruns);
  printf("---------------------------------------------------------\n");
  printf("  FULL CG CYCLE   : %.4f ms/iter   <- headline number\n", 1e3 * tsolve_iter);
  if (nruns > 1)
    printf("                    (best %.4f ms, spread %.1f%%)\n",
           1e3 * best_iter, 100.0 * (tsolve_iter - best_iter) / best_iter);
  printf("    of which SpMV : %.4f ms/iter   (%.1f GB/s, lower-bound traffic)\n",
         tspmv * 1e3, bw);
  printf("    other (dots,  : %.4f ms/iter\n", 1e3 * (tsolve_iter - tspmv));
  printf("     axpy, copy)  \n");
  printf("  SpMV share      : %.1f%%\n", 100.0 * tspmv / tsolve_iter);
  printf("=========================================================\n");

  /* Machine-readable line for the harness. Same 20 fields in the same order as
   * cg_omp.c and the CUDA versions. impl/backend is "gpu,omp", which is what
   * distinguishes this from cg_omp.c's "cpu,omp" and 1-cuda.cu's "gpu,cuda".
   * The last field is the device count. */
  printf("RESULT,gpu,omp,%s,%d,%lld,%.2f,%.1f,%.3f,%d,%.6e,%.6f,%.6f,%.6f,%.1f,%.2f,%.6f,%.2f,%d,%d\n",
         order, A.n, (long long)A.nnz, mean_len, mean_bw, gpu_mem,
         iters, relres,
         1e3 * tsolve_iter, 1e3 * tspmv, 1e3 * (tsolve_iter - tspmv),
         100.0 * tspmv / tsolve_iter, bw,
         1e3 * best_iter, 100.0 * (tsolve_iter - best_iter) / best_iter,
         nruns, 1);

  omp_target_free(d_rowptr, DEV); omp_target_free(d_col, DEV);
  omp_target_free(d_val, DEV);    omp_target_free(d_b, DEV);
  omp_target_free(d_x, DEV);      omp_target_free(d_r, DEV);
  omp_target_free(d_p, DEV);      omp_target_free(d_Ap, DEV);
  csr_free(&A);
  return 0;
}

