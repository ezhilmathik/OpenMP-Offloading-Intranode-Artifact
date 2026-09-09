/* ===========================================================================
 * cg_omp.c -- Conjugate Gradient, unpreconditioned, OpenMP CPU.
 *
 * BUILD:  gcc -O3 -march=native -fopenmp -o cg_omp cg_omp.c -lm
 *         icx -O3 -xHost      -qopenmp -o cg_omp cg_omp.c -lm
 *
 * RUN:    ./cg_omp -n 4000000 -deg 15
 *         ./cg_omp -mm Queen_4147.mtx -order rcm
 *
 *   -n      rows for the synthetic matrix          (default 1000000)
 *   -deg    mean off-diagonal entries per row      (default 15)
 *   -shift  added to the diagonal, sets conditioning (default 0.1)
 *   -mm     read a Matrix Market file instead of generating
 *   -order  natural | rcm | random                 (default natural)
 *   -tol    relative residual tolerance            (default 1e-8)
 *   -maxit  iteration cap                          (default 1000)
 *   -runs   repeat the whole solve, for run-to-run variance (default 1)
 *   -seed   RNG seed                               (default 12345)
 *
 * ---------------------------------------------------------------------------
 * THIS FILE IS THE PORTABILITY BASE
 *
 * It is written so that the OpenACC and OpenMP-offload versions differ from it
 * only in the pragma above each compute loop. To keep that true:
 *
 *   - All data is FLAT TOP-LEVEL ARRAYS passed as arguments. No structs of
 *     pointers in the compute path. OpenACC and OpenMP target both want
 *     contiguous buffers with explicit extents; a struct forces you into
 *     nested deep-copy map clauses for no benefit.
 *
 *   - Each compute loop is a SEPARATE FUNCTION containing nothing but that
 *     loop. No fusing of unrelated work, no branches inside. This is the shape
 *     all three models compile well.
 *
 *   - The CG driver is plain serial host code. Only the five kernels below
 *     carry pragmas.
 *
 * The ports then look like:
 *
 *     #pragma omp parallel for schedule(static)              <- this file
 *     #pragma omp target teams distribute parallel for       <- OMP offload
 *     #pragma acc parallel loop present(rowptr,col,val,x,y)  <- OpenACC
 *
 * with an identical loop body underneath.
 *
 * ---------------------------------------------------------------------------
 * NUMA
 *
 * The vectors are initialised inside a parallel loop with the same schedule
 * the compute loops use, so first-touch places each page on the socket that
 * will read it. Without this a dual-socket run puts every page on socket 0 and
 * loses roughly half the memory bandwidth. Run with OMP_PROC_BIND=close and
 * OMP_PLACES=cores.
 *
 * The matrix arrays come from malloc inside matgen.h and are NOT first-touched
 * here, so on a dual-socket machine the SpMV still reads them from wherever
 * the generator landed. Fixing that properly means moving allocation into the
 * solver; it is called out in the limitations note at the bottom rather than
 * silently ignored.
 * ========================================================================= */

#include "matgen.h"
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* ===========================================================================
 * THE FIVE COMPUTE KERNELS
 *
 * Everything below this line up to the CG driver is what gets a different
 * pragma in the OpenACC and OpenMP-offload ports. Nothing else changes.
 * ========================================================================= */

/* y = A*x */
static void spmv(int n, const ptr_t *rowptr, const idx_t *col, const double *val,
                 const double *x, double *y)
{
  int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (i = 0; i < n; i++) {
    double sum = 0.0;
    ptr_t k;
    for (k = rowptr[i]; k < rowptr[i + 1]; k++)
      sum += val[k] * x[col[k]];
    y[i] = sum;
  }
}

/* returns a . b */
static double dot(int n, const double *a, const double *b)
{
  double s = 0.0;
  int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : s)
#endif
  for (i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

/* y = y + a*x */
static void axpy(int n, double a, const double *x, double *y)
{
  int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (i = 0; i < n; i++) y[i] = y[i] + a * x[i];
}

/* y = x + b*y */
static void xpby(int n, const double *x, double b, double *y)
{
  int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (i = 0; i < n; i++) y[i] = x[i] + b * y[i];
}

/* y = x */
static void vcopy(int n, const double *x, double *y)
{
  int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (i = 0; i < n; i++) y[i] = x[i];
}

/* ===========================================================================
 * CG -- unpreconditioned, standard formulation
 *
 * Per iteration: one SpMV, two dot products, three vector updates.
 *
 *   r = b - A x
 *   p = r
 *   rr = r.r
 *   repeat
 *     Ap    = A p
 *     alpha = rr / (p.Ap)
 *     x     = x + alpha p
 *     r     = r - alpha Ap
 *     rr_new = r.r
 *     beta  = rr_new / rr
 *     p     = r + beta p
 *
 * Note this reports the TRUE residual at the end, recomputed as ||b - Ax||.
 * The recursively updated r drifts from the true residual over hundreds of
 * iterations, and on an ill-conditioned matrix the gap can be large. Trusting
 * the recursive value is the classic way to report a convergence that did not
 * happen.
 * ========================================================================= */
static int cg_solve(int n, const ptr_t *rowptr, const idx_t *col, const double *val,
                    const double *b, double *x,
                    double *r, double *p, double *Ap,
                    double rtol, int maxit,
                    double *relres, double *tsolve, double *tspmv_total);

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
  int npts, maxit, iters = 0, i, nthreads = 1, max_len, denom = 1, nruns;
  double avg_deg, shift, rtol, relres = 0.0, tsolve = 0.0, tspmv_total = 0.0;
  double t0, t1, tspmv;
  double mean_len, std_len, mean_bw, bytes, bw;
  double best_iter, best_spmv, sum_iter, sum_spmv, tsolve_iter;
  unsigned long long seed;
  double *b, *x, *r, *p, *Ap;

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

#ifdef _OPENMP
  nthreads = omp_get_max_threads();
#endif

  /* ----- matrix ----- */
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

  /* ----- vectors, first-touched with the compute schedule ----- */
  b  = (double *)mg_alloc((size_t)A.n * sizeof(double));
  x  = (double *)mg_alloc((size_t)A.n * sizeof(double));
  r  = (double *)mg_alloc((size_t)A.n * sizeof(double));
  p  = (double *)mg_alloc((size_t)A.n * sizeof(double));
  Ap = (double *)mg_alloc((size_t)A.n * sizeof(double));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (i = 0; i < A.n; i++) {
    /* NOT the constant vector: it is an eigenvector of this matrix and CG
     * would converge in one iteration. See mg_hash01 in matgen.h. */
    b[i] = 2.0 * mg_hash01(i) - 1.0;
    x[i] = 0.0; r[i] = 0.0; p[i] = 0.0; Ap[i] = 0.0;
  }

  printf("=========================================================\n");
  printf("  threads         : %d\n", nthreads);
  printf("  matrix          : %s\n", mmfile ? mmfile : "synthetic radius graph");
  printf("  ordering        : %s\n", order);
  printf("  rows            : %d\n", A.n);
  printf("  nonzeros        : %lld\n", (long long)A.nnz);
  printf("  nnz/row         : mean %.2f, std %.2f, max %d\n", mean_len, std_len, max_len);
  printf("  mean |i-col|    : %.1f  (locality: lower is better)\n", mean_bw);
  printf("  build time      : %.2f s\n", t1 - t0);
  printf("---------------------------------------------------------\n");

  /* ----- solve -----
   * There is deliberately NO standalone SpMV benchmark. Calling spmv in a
   * tight loop on the same input vector leaves that vector resident in cache,
   * so the gather x[col[k]] hits far more often than it ever does inside CG,
   * where p was just overwritten and four other vectors compete for the same
   * cache. It flatters the kernel, and it flatters a well-ordered matrix more
   * than a badly-ordered one, which understates exactly the ordering effect
   * this code exists to measure.
   *
   * SpMV is timed inside the solve instead, under real cache pressure.
   *
   * TWO LEVELS OF AVERAGING:
   *   within a run  -- tsolve/iters and tspmv_total/iters already average over
   *                    every CG iteration, typically hundreds of samples
   *   across runs   -- -runs repeats the whole solve to expose run-to-run
   *                    variance (turbo, other load, page placement)
   *
   * Report the MEAN for a typical figure, the MIN for a best-case one. If
   * min and mean differ by more than a few percent, the machine is noisy and
   * you should say so rather than quoting a single number. */
  {
    int run;
    double s_iter, s_spmv;
    best_iter = 1e30; best_spmv = 1e30;
    sum_iter = 0.0; sum_spmv = 0.0;

    for (run = 0; run < nruns; run++) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (i = 0; i < A.n; i++) x[i] = 0.0;      /* same starting point each run */

      iters = cg_solve(A.n, A.rowptr, A.col, A.val, b, x, r, p, Ap,
                       rtol, maxit, &relres, &tsolve, &tspmv_total);

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
  bytes  = (double)A.nnz * 12.0 + (double)A.n * 24.0;
  bw     = bytes / tspmv / 1e9;

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

  /* Machine-readable line for the harness. Field order is identical in the
   * CUDA versions, so one parser handles every binary. */
  printf("RESULT,cpu,omp,%s,%d,%lld,%.2f,%.1f,%.3f,%d,%.6e,%.6f,%.6f,%.6f,%.1f,%.2f,%.6f,%.2f,%d,%d\n",
         order, A.n, (long long)A.nnz, mean_len, mean_bw,
         ((double)A.nnz * 12.0 + (double)(A.n + 1) * 8.0 + (double)A.n * 5.0 * 8.0) / 1e9,
         iters, relres,
         1e3 * tsolve_iter, 1e3 * tspmv, 1e3 * (tsolve_iter - tspmv),
         100.0 * tspmv / tsolve_iter, bw,
         1e3 * best_iter, 100.0 * (tsolve_iter - best_iter) / best_iter,
         nruns, nthreads);

  free(b); free(x); free(r); free(p); free(Ap);
  csr_free(&A);
  return 0;
}

/* ===========================================================================
 * NOTES
 *
 * WHAT TO MEASURE
 *   Unpreconditioned CG on an ill-conditioned matrix will run thousands of
 *   iterations and may never reach 1e-8. That is expected and not a problem
 *   here: the headline number is TIME PER ITERATION, and the residual is a
 *   correctness check, not a convergence claim. Raise -shift to make the
 *   matrix better conditioned if you want the solve to actually finish.
 *
 * THE ORDERING EXPERIMENT
 *   Run the same matrix with -order natural, rcm, random. A symmetric
 *   permutation leaves the spectrum unchanged, so the iteration count should
 *   be the same to within a step or two -- it can drift slightly because the
 *   dot products sum in a different order, not because the problem changed. A
 *   difference of more than a few iterations means something is wrong.
 *
 *   What does change is SpMV time, via cache hit rate on x.
 *
 *   Note that "natural" here is already spatially sorted (matgen_radius
 *   relabels points in cell order), so it has decent locality and RCM may not
 *   improve on it much. That is itself worth reporting: RCM pays off on
 *   badly-numbered input, not on input that is already coherent. The random
 *   ordering is the one that shows the full cost of losing locality.
 *
 * THE IRREGULARITY QUESTION
 *   Watch the printed nnz/row std. The radius graph gives Poisson-distributed
 *   degree, so std ~ sqrt(mean): about 3.9 at mean 15. That is close to real
 *   tetrahedral mesh valence spread. If the std is small relative to the mean,
 *   a simple one-thread-per-row (or one-warp-per-row on GPU) kernel is fine.
 *   If it is large, load imbalance within the kernel starts to matter, which
 *   is why cuSPARSE uses merge-path rather than row-per-thread.
 *
 * LIMITATIONS
 *   - The matrix arrays are allocated inside matgen.h with plain malloc, so
 *     they are not NUMA first-touched by the compute schedule. On a
 *     dual-socket machine the SpMV therefore reads the matrix from wherever
 *     the generator's allocation landed. The vectors here ARE first-touched
 *     correctly. Fixing the matrix properly means allocating and filling it
 *     under the same schedule the SpMV uses.
 *
 *   - Run with OMP_PROC_BIND=close OMP_PLACES=cores. Without pinning, thread
 *     migration alone can cost 30% on a memory-bound kernel.
 *
 *   - schedule(static) is right when rows are uniform. With a heavy-tailed
 *     degree distribution, schedule(dynamic, 512) can be better. Worth trying
 *     both; the answer depends on the matrix, which is the point.
 * ========================================================================= */
