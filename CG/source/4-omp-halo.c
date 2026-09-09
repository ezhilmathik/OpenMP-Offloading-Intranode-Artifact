/* ===========================================================================
 * 4-omp-halo.c -- Conjugate Gradient, unpreconditioned, OpenMP target offload,
 *                 4 GPUs (single node), REAL PACKED HALO.
 * ========================================================================= */

#include "matgen.h"
#include <time.h>
#include <omp.h>

#ifndef NGPU
#define NGPU 4
#endif


static int HOST = 0;                  /* omp_get_initial_device() */
static int DEV[NGPU];                 /* device number for each slot */
static int R0[NGPU], R1[NGPU], NLOC[NGPU];
static long long NNZLOC[NGPU];

static ptr_t  *d_rowptr[NGPU];        /* NLOC+1, rebased to 0 */
static idx_t  *d_col[NGPU];           /* NNZLOC, GLOBAL column indices */
static double *d_val[NGPU];           /* NNZLOC */
static double *d_b[NGPU], *d_x[NGPU], *d_r[NGPU], *d_p[NGPU], *d_Ap[NGPU];


static int *d_int[NGPU], *d_bnd[NGPU];
static int  N_INT[NGPU], N_BND[NGPU];

/* ----- halo -----
 * CNT[d][e] : entries device d must receive from device e
 * ROFF[d][e]: offset of that block inside d's receive buffer / scatter list
 * SOFF[e][d]: offset of that block inside e's send buffer / gather list
 * RLEN[d]   : total entries d receives per iteration  (= sum_e CNT[d][e])
 * SLEN[d]   : total entries d sends    per iteration  (= sum_e CNT[e][d])
 */
static int CNT[NGPU][NGPU];
static int ROFF[NGPU][NGPU], SOFF[NGPU][NGPU];
static int RLEN[NGPU], SLEN[NGPU];

static int    *d_gidx[NGPU];          /* SLEN[d] global indices, gather  */
static int    *d_sidx[NGPU];          /* RLEN[d] global indices, scatter */
static double *d_sbuf[NGPU];          /* SLEN[d] doubles */
static double *d_rbuf[NGPU];          /* RLEN[d] doubles */


static int OVERLAP = 0;

static int n_glob = 0;




static void spmv_rows(int dev, ptr_t *rp, idx_t *cl, double *vl,
                      double *pp, double *yy, int *list, int nlist)
{
  int i;
#pragma omp target teams distribute parallel for schedule(static, 1) \
        device(dev) is_device_ptr(rp, cl, vl, pp, yy, list)
  for (i = 0; i < nlist; i++) {
    int rr = list[i];
    double sum = 0.0;
    ptr_t k;
    for (k = rp[rr]; k < rp[rr + 1]; k++)
      sum += vl[k] * pp[cl[k]];
    yy[rr] = sum;
  }
}


static void spmv(void)
{
#pragma omp parallel num_threads(NGPU)
  {
    int d   = omp_get_thread_num();
    int dev = DEV[d];

    spmv_rows(dev, d_rowptr[d], d_col[d], d_val[d], d_p[d],
              d_Ap[d] + R0[d], d_int[d], N_INT[d]);
    spmv_rows(dev, d_rowptr[d], d_col[d], d_val[d], d_p[d],
              d_Ap[d] + R0[d], d_bnd[d], N_BND[d]);
  }
}


static double dot(double *const *a, double *const *b)
{
  double part[NGPU];
  double s = 0.0;
  int d;

  for (d = 0; d < NGPU; d++) part[d] = 0.0;

#pragma omp parallel num_threads(NGPU)
  {
    int t   = omp_get_thread_num();
    int dev = DEV[t];
    int nl  = NLOC[t];
    int i;
    double *aa = a[t] + R0[t];
    double *bb = b[t] + R0[t];
    double sd = 0.0;

#pragma omp target teams distribute parallel for schedule(static, 1) \
        device(dev) is_device_ptr(aa, bb) reduction(+ : sd)
    for (i = 0; i < nl; i++) sd += aa[i] * bb[i];

    part[t] = sd;
  }

  for (d = 0; d < NGPU; d++) s += part[d];
  return s;
}

/* y = y + a*x */
static void axpy(double a, double *const *x, double *const *y)
{
#pragma omp parallel num_threads(NGPU)
  {
    int d   = omp_get_thread_num();
    int dev = DEV[d];
    int nl  = NLOC[d];
    int i;
    double *xx = x[d] + R0[d];
    double *yy = y[d] + R0[d];

#pragma omp target teams distribute parallel for schedule(static, 1) \
        device(dev) is_device_ptr(xx, yy)
    for (i = 0; i < nl; i++) yy[i] = yy[i] + a * xx[i];
  }
}

/* y = x + b*y */
static void xpby(double *const *x, double b, double *const *y)
{
#pragma omp parallel num_threads(NGPU)
  {
    int d   = omp_get_thread_num();
    int dev = DEV[d];
    int nl  = NLOC[d];
    int i;
    double *xx = x[d] + R0[d];
    double *yy = y[d] + R0[d];

#pragma omp target teams distribute parallel for schedule(static, 1) \
        device(dev) is_device_ptr(xx, yy)
    for (i = 0; i < nl; i++) yy[i] = xx[i] + b * yy[i];
  }
}

/* y = x */
static void vcopy(double *const *x, double *const *y)
{
#pragma omp parallel num_threads(NGPU)
  {
    int d   = omp_get_thread_num();
    int dev = DEV[d];
    int nl  = NLOC[d];
    int i;
    double *xx = x[d] + R0[d];
    double *yy = y[d] + R0[d];

#pragma omp target teams distribute parallel for schedule(static, 1) \
        device(dev) is_device_ptr(xx, yy)
    for (i = 0; i < nl; i++) yy[i] = xx[i];
  }
}

/* y = v over the owned window */
static void vfill(double v, double *const *y)
{
#pragma omp parallel num_threads(NGPU)
  {
    int d   = omp_get_thread_num();
    int dev = DEV[d];
    int nl  = NLOC[d];
    int i;
    double *yy = y[d] + R0[d];

#pragma omp target teams distribute parallel for schedule(static, 1) \
        device(dev) is_device_ptr(yy)
    for (i = 0; i < nl; i++) yy[i] = v;
  }
}


static void vfill_full(double v, double *const *y)
{
#pragma omp parallel num_threads(NGPU)
  {
    int d   = omp_get_thread_num();
    int dev = DEV[d];
    int n   = n_glob;
    int i;
    double *yy = y[d];

#pragma omp target teams distribute parallel for schedule(static, 1) \
        device(dev) is_device_ptr(yy)
    for (i = 0; i < n; i++) yy[i] = v;
  }
}


static void pack_p(void)
{
#pragma omp parallel num_threads(NGPU)
  {
    int d   = omp_get_thread_num();
    int dev = DEV[d];
    int ns  = SLEN[d];
    int i;
    double *pp = d_p[d];
    double *sb = d_sbuf[d];
    int    *gi = d_gidx[d];

#pragma omp target teams distribute parallel for schedule(static, 1) \
        device(dev) is_device_ptr(pp, sb, gi)
    for (i = 0; i < ns; i++) sb[i] = pp[gi[i]];
  }
}


static void unpack_p(void)
{
#pragma omp parallel num_threads(NGPU)
  {
    int d   = omp_get_thread_num();
    int dev = DEV[d];
    int nr  = RLEN[d];
    int i;
    double *pp = d_p[d];
    double *rb = d_rbuf[d];
    int    *si = d_sidx[d];

#pragma omp target teams distribute parallel for schedule(static, 1) \
        device(dev) is_device_ptr(pp, rb, si)
    for (i = 0; i < nr; i++) pp[si[i]] = rb[i];
  }
}


static void copy_pair(int d, int e)
{
  if (CNT[d][e] <= 0) return;
  if (omp_target_memcpy(d_rbuf[d], d_sbuf[e],
                        (size_t)CNT[d][e] * sizeof(double),
                        (size_t)ROFF[d][e] * sizeof(double),
                        (size_t)SOFF[e][d] * sizeof(double),
                        DEV[d], DEV[e]) != 0) {
    fprintf(stderr, "omp_target_memcpy failed: halo %d <- %d (%d entries)\n",
            DEV[d], DEV[e], CNT[d][e]);
    exit(1);
  }
}

static void copy_halo(void)
{
  const int total = NGPU * (NGPU - 1);

#pragma omp parallel num_threads(total)
  {
    int t = omp_get_thread_num();
    int d = t / (NGPU - 1);
    int j = t % (NGPU - 1);
    int e = (j < d) ? j : j + 1;


    if (omp_get_num_threads() != total) {
      fprintf(stderr, "halo: asked for %d threads, got %d\n",
              total, omp_get_num_threads());
      exit(1);
    }

    copy_pair(d, e);
  }
}


static void exchange_p(void)
{
  pack_p();
  copy_halo();
  unpack_p();
}

/* ---------------------------------------------------------------------------
 * THE OVERLAPPED PHASE
 *
 *     pack
 *     ---- barrier
 *     interior SpMV (NGPU threads)  ||  peer copies (NGPU*(NGPU-1) threads)
 *     ---- barrier
 *     unpack
 *     boundary SpMV
 *
 * ------------------------------------------------------------------------- */
static void spmv_exchange(void)
{
  const int ncopy = NGPU * (NGPU - 1);
  const int total = NGPU + ncopy;

  pack_p();

#pragma omp parallel num_threads(total)
  {
    int t = omp_get_thread_num();

    if (omp_get_num_threads() != total) {
      fprintf(stderr, "overlap: asked for %d threads, got %d\n",
              total, omp_get_num_threads());
      exit(1);
    }

    if (t < NGPU) {
      int d = t;
      spmv_rows(DEV[d], d_rowptr[d], d_col[d], d_val[d], d_p[d],
                d_Ap[d] + R0[d], d_int[d], N_INT[d]);
    } else {
      int u = t - NGPU;
      int d = u / (NGPU - 1);
      int j = u % (NGPU - 1);
      int e = (j < d) ? j : j + 1;
      copy_pair(d, e);
    }
  }

  unpack_p();

#pragma omp parallel num_threads(NGPU)
  {
    int d = omp_get_thread_num();
    spmv_rows(DEV[d], d_rowptr[d], d_col[d], d_val[d], d_p[d],
              d_Ap[d] + R0[d], d_bnd[d], N_BND[d]);
  }
}


static void stage_into_p(double *const *v)
{
  vcopy(v, d_p);
  exchange_p();
}


#define XFER_CHUNK ((size_t)1 << 30)

static void h2d(void *dst, const void *src, size_t bytes, int dev,
                const char *what)
{
  size_t off = 0;
  while (off < bytes) {
    size_t nb = bytes - off;
    if (nb > XFER_CHUNK) nb = XFER_CHUNK;
    if (omp_target_memcpy(dst, (void *)src, nb, off, off, dev, HOST) != 0) {
      fprintf(stderr, "omp_target_memcpy failed: %s (offset %zu, %zu bytes)\n",
              what, off, nb);
      exit(1);
    }
    off += nb;
  }
}

static void *dev_alloc(size_t bytes, int dev, const char *what)
{
  void *p = omp_target_alloc(bytes, dev);
  if (!p) {
    fprintf(stderr, "omp_target_alloc failed: %s (%zu bytes, device %d)\n",
            what, bytes, dev);
    exit(1);
  }
  return p;
}

/* ===========================================================================
 * CG -- unpreconditioned, standard formulation. Identical to 4-omp.c.
 *
 *   r = b - A x ;  p = r ;  rr = r.r
 *   repeat
 *     EXCHANGE p                <- the only communication
 *     Ap    = A p
 *     alpha = rr / (p.Ap)
 *     x     = x + alpha p
 *     r     = r - alpha Ap
 *     rr_new = r.r ;  beta = rr_new / rr ;  rr = rr_new
 *     p     = r + beta p
 *
 * The exchange sits at the TOP of the loop, not after the p update at the
 * bottom. Same thing algorithmically, but it also covers the initial p = r
 * before the first iteration, so there is one code path instead of a special
 * case outside the loop.
 *
 * The TRUE residual ||b - Ax|| is recomputed at the end.
 * ========================================================================= */
static double wtime(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int cg_solve(double rtol, int maxit, double *relres, double *tsolve,
                    double *tspmv_total, double *thalo_total)
{
  double rr, rr_new, pAp, alpha, beta, bnorm, t0, t1, ts, h0;
  int k;

  bnorm = sqrt(dot(d_b, d_b));
  if (bnorm == 0.0) bnorm = 1.0;

  stage_into_p(d_x);                         /* r = b - A x */
  spmv();
  vcopy(d_b, d_r);
  axpy(-1.0, d_Ap, d_r);

  vcopy(d_r, d_p);
  rr = dot(d_r, d_r);

  *tspmv_total = 0.0;
  *thalo_total = 0.0;
  t0 = wtime();

  for (k = 0; k < maxit; k++) {
    if (sqrt(rr) / bnorm <= rtol) break;


    if (OVERLAP) {
      ts = wtime();
      spmv_exchange();
      *tspmv_total += wtime() - ts;
    } else {
      h0 = wtime();
      exchange_p();
      *thalo_total += wtime() - h0;

      ts = wtime();
      spmv();
      *tspmv_total += wtime() - ts;
    }

    pAp   = dot(d_p, d_Ap);
    alpha = rr / pAp;

    axpy( alpha, d_p,  d_x);
    axpy(-alpha, d_Ap, d_r);

    rr_new = dot(d_r, d_r);
    beta   = rr_new / rr;
    rr     = rr_new;

    xpby(d_r, beta, d_p);

    if ((k + 1) % 100 == 0)
      printf("    iter %6d   relres %.3e\n", k + 1, sqrt(rr) / bnorm);
  }

  t1 = wtime();

  stage_into_p(d_x);                         /* TRUE residual */
  spmv();
  vcopy(d_b, d_r);
  axpy(-1.0, d_Ap, d_r);
  *relres = sqrt(dot(d_r, d_r)) / bnorm;
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

/* nnz-balanced row boundaries: the smallest r with rowptr[r] >= target. */
static int split_at(const ptr_t *rowptr, int n, ptr_t target)
{
  int lo = 0, hi = n, mid;
  while (lo < hi) {
    mid = lo + (hi - lo) / 2;
    if (rowptr[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

int main(int argc, char **argv)
{
  CSR A;
  const char *mmfile, *order;
  int npts, maxit, iters = 0, i, d, e, max_len, denom = 1, nruns, ndev;
  double avg_deg, shift, rtol, relres = 0.0, tsolve = 0.0;
  double tspmv_total = 0.0, thalo_total = 0.0;
  double t0, t1, tspmv, thalo;
  double mean_len, std_len, mean_bw, bytes, bw;
  double best_iter, best_spmv, sum_iter, sum_spmv, sum_halo, tsolve_iter;
  unsigned long long seed;
  double *hb;
  ptr_t  *h_rp;
  double gpu_mem, gpu_mem_dev, halo_gb, repl_gb;
  int *L[NGPU][NGPU];                  /* host exchange lists, L[d][e] */
  int *h_gidx[NGPU], *h_sidx[NGPU];

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
  if (ndev < NGPU) {
    fprintf(stderr, "need %d OpenMP target devices, found %d -- was this built "
                    "with --offload-arch, and are %d GPUs visible?\n",
            NGPU, ndev, NGPU);
    return 1;
  }
  HOST = omp_get_initial_device();
  for (d = 0; d < NGPU; d++) DEV[d] = d;
  {
    const char *ov = getenv("CG_OVERLAP");
    if (ov && ov[0] == '1') OVERLAP = 1;
  }

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
  n_glob = A.n;

  /* ----- split rows so each device gets ~nnz/NGPU nonzeros ----- */
  for (d = 0; d < NGPU; d++) {
    ptr_t lo_t = (ptr_t)((double)A.nnz * (double)d       / (double)NGPU);
    ptr_t hi_t = (ptr_t)((double)A.nnz * (double)(d + 1) / (double)NGPU);
    R0[d] = (d == 0)        ? 0    : split_at(A.rowptr, A.n, lo_t);
    R1[d] = (d == NGPU - 1) ? A.n  : split_at(A.rowptr, A.n, hi_t);
  }
  for (d = 1; d < NGPU; d++) R0[d] = R1[d - 1];      /* no gaps, no overlap */
  for (d = 0; d < NGPU; d++) {
    NLOC[d]   = R1[d] - R0[d];
    NNZLOC[d] = (long long)(A.rowptr[R1[d]] - A.rowptr[R0[d]]);
    if (NLOC[d] <= 0) {
      fprintf(stderr, "degenerate split: device %d got %d rows\n", d, NLOC[d]);
      return 1;
    }
  }


  {
    char *mark = (char *)mg_alloc((size_t)A.n);

    for (d = 0; d < NGPU; d++) {
      long long kb = (long long)A.rowptr[R0[d]];
      long long ke = (long long)A.rowptr[R1[d]];
      long long k;

      memset(mark, 0, (size_t)A.n);

#pragma omp parallel for schedule(static)
      for (k = kb; k < ke; k++) {
        idx_t c = A.col[k];
        if ((int)c < R0[d] || (int)c >= R1[d]) mark[(int)c] = 1;
      }

      for (e = 0; e < NGPU; e++) {
        int cnt = 0, g;
        if (e == d) { CNT[d][e] = 0; L[d][e] = NULL; continue; }
        for (g = R0[e]; g < R1[e]; g++) if (mark[g]) cnt++;
        CNT[d][e] = cnt;
        L[d][e] = (int *)mg_alloc((size_t)(cnt ? cnt : 1) * sizeof(int));
        cnt = 0;
        for (g = R0[e]; g < R1[e]; g++) if (mark[g]) L[d][e][cnt++] = g;
      }
    }
    free(mark);
  }

  /* receive-side offsets and concatenated scatter lists */
  for (d = 0; d < NGPU; d++) {
    int off = 0;
    for (e = 0; e < NGPU; e++) { ROFF[d][e] = off; if (e != d) off += CNT[d][e]; }
    RLEN[d] = off;
    h_sidx[d] = (int *)mg_alloc((size_t)(off ? off : 1) * sizeof(int));
    for (e = 0; e < NGPU; e++) {
      if (e == d || CNT[d][e] == 0) continue;
      memcpy(h_sidx[d] + ROFF[d][e], L[d][e], (size_t)CNT[d][e] * sizeof(int));
    }
  }


  for (e = 0; e < NGPU; e++) {
    int off = 0;
    for (d = 0; d < NGPU; d++) { SOFF[e][d] = off; if (d != e) off += CNT[d][e]; }
    SLEN[e] = off;
    h_gidx[e] = (int *)mg_alloc((size_t)(off ? off : 1) * sizeof(int));
    for (d = 0; d < NGPU; d++) {
      if (d == e || CNT[d][e] == 0) continue;
      memcpy(h_gidx[e] + SOFF[e][d], L[d][e], (size_t)CNT[d][e] * sizeof(int));
    }
  }

  for (d = 0; d < NGPU; d++)
    for (e = 0; e < NGPU; e++)
      if (L[d][e]) free(L[d][e]);


  hb = (double *)mg_alloc((size_t)A.n * sizeof(double));
#pragma omp parallel for schedule(static)
  for (i = 0; i < A.n; i++) hb[i] = 2.0 * mg_hash01(i) - 1.0;

  h_rp = (ptr_t *)mg_alloc((size_t)(A.n + 1) * sizeof(ptr_t));


  for (d = 0; d < NGPU; d++) {
    ptr_t base = A.rowptr[R0[d]];
    int   nl   = NLOC[d];
    int   dev  = DEV[d];
    char *isint;
    int  *h_int, *h_bnd, ni = 0, nb = 0;

    for (i = 0; i <= nl; i++) h_rp[i] = A.rowptr[R0[d] + i] - base;


    isint = (char *)mg_alloc((size_t)nl);
#pragma omp parallel for schedule(static)
    for (i = 0; i < nl; i++) {
      ptr_t k;
      int   ok = 1;
      for (k = A.rowptr[R0[d] + i]; k < A.rowptr[R0[d] + i + 1]; k++) {
        idx_t c = A.col[k];
        if ((int)c < R0[d] || (int)c >= R1[d]) { ok = 0; break; }
      }
      isint[i] = (char)ok;
    }
    h_int = (int *)mg_alloc((size_t)nl * sizeof(int));
    h_bnd = (int *)mg_alloc((size_t)nl * sizeof(int));
    for (i = 0; i < nl; i++) {
      if (isint[i]) h_int[ni++] = i;
      else          h_bnd[nb++] = i;
    }
    N_INT[d] = ni;
    N_BND[d] = nb;
    free(isint);

    d_rowptr[d] = (ptr_t *) dev_alloc((size_t)(nl + 1) * sizeof(ptr_t), dev, "rowptr");
    d_col[d]    = (idx_t *) dev_alloc((size_t)NNZLOC[d] * sizeof(idx_t), dev, "col");
    d_val[d]    = (double *)dev_alloc((size_t)NNZLOC[d] * sizeof(double), dev, "val");
    d_int[d]    = (int *)   dev_alloc((size_t)(ni ? ni : 1) * sizeof(int), dev, "int rows");
    d_bnd[d]    = (int *)   dev_alloc((size_t)(nb ? nb : 1) * sizeof(int), dev, "bnd rows");
    d_b[d]      = (double *)dev_alloc((size_t)A.n * sizeof(double), dev, "b");
    d_x[d]      = (double *)dev_alloc((size_t)A.n * sizeof(double), dev, "x");
    d_r[d]      = (double *)dev_alloc((size_t)A.n * sizeof(double), dev, "r");
    d_p[d]      = (double *)dev_alloc((size_t)A.n * sizeof(double), dev, "p");
    d_Ap[d]     = (double *)dev_alloc((size_t)A.n * sizeof(double), dev, "Ap");

    /* halo: gather list + send buffer (this device as source),
     *       scatter list + receive buffer (this device as sink) */
    d_gidx[d] = (int *)   dev_alloc((size_t)(SLEN[d] ? SLEN[d] : 1) * sizeof(int),
                                    dev, "halo gather list");
    d_sidx[d] = (int *)   dev_alloc((size_t)(RLEN[d] ? RLEN[d] : 1) * sizeof(int),
                                    dev, "halo scatter list");
    d_sbuf[d] = (double *)dev_alloc((size_t)(SLEN[d] ? SLEN[d] : 1) * sizeof(double),
                                    dev, "halo send buffer");
    d_rbuf[d] = (double *)dev_alloc((size_t)(RLEN[d] ? RLEN[d] : 1) * sizeof(double),
                                    dev, "halo recv buffer");

    h2d(d_rowptr[d], h_rp, (size_t)(nl + 1) * sizeof(ptr_t), dev, "rowptr");
    h2d(d_col[d], A.col + base, (size_t)NNZLOC[d] * sizeof(idx_t), dev, "col");
    h2d(d_val[d], A.val + base, (size_t)NNZLOC[d] * sizeof(double), dev, "val");
    if (ni) h2d(d_int[d], h_int, (size_t)ni * sizeof(int), dev, "int rows");
    if (nb) h2d(d_bnd[d], h_bnd, (size_t)nb * sizeof(int), dev, "bnd rows");
    h2d(d_b[d], hb, (size_t)A.n * sizeof(double), dev, "b");
    if (SLEN[d]) h2d(d_gidx[d], h_gidx[d], (size_t)SLEN[d] * sizeof(int),
                     dev, "halo gather list");
    if (RLEN[d]) h2d(d_sidx[d], h_sidx[d], (size_t)RLEN[d] * sizeof(int),
                     dev, "halo scatter list");

    free(h_int);
    free(h_bnd);
  }
  free(hb);
  free(h_rp);
  for (d = 0; d < NGPU; d++) { free(h_gidx[d]); free(h_sidx[d]); }

  vfill_full(0.0, d_x);
  vfill_full(0.0, d_r);
  vfill_full(0.0, d_p);
  vfill_full(0.0, d_Ap);


  gpu_mem = 0.0;
  gpu_mem_dev = 0.0;
  for (d = 0; d < NGPU; d++) {
    double m = ((double)NNZLOC[d] * 12.0 + (double)(NLOC[d] + 1) * 8.0
                + (double)A.n * 5.0 * 8.0
                + (double)SLEN[d] * 12.0 + (double)RLEN[d] * 12.0) / 1e9;
    gpu_mem += m;
    if (m > gpu_mem_dev) gpu_mem_dev = m;
  }


  halo_gb = 0.0;
  for (d = 0; d < NGPU; d++) halo_gb += (double)RLEN[d] * 8.0;
  halo_gb /= (double)NGPU * 1e9;
  repl_gb = (double)A.n * 8.0 * (double)(NGPU - 1) / (double)NGPU / 1e9;

  printf("=========================================================\n");
  printf("  GPUs            : %d  (OpenMP target devices 0-%d of %d)\n",
         NGPU, NGPU - 1, ndev);
  printf("  backend         : OpenMP target offload\n");
  printf("  halo path       : packed lists (gather, peer copy, scatter)\n");
  printf("  matrix          : %s\n", mmfile ? mmfile : "synthetic radius graph");
  printf("  ordering        : %s\n", order);
  printf("  rows            : %d\n", A.n);
  printf("  nonzeros        : %lld\n", (long long)A.nnz);
  printf("  nnz/row         : mean %.2f, std %.2f, max %d\n", mean_len, std_len, max_len);
  printf("  mean |i-col|    : %.1f  (locality: lower is better)\n", mean_bw);
  printf("  rows / GPU      :");
  for (d = 0; d < NGPU; d++) printf(" %d", NLOC[d]);
  printf("\n");
  printf("  nnz / GPU       :");
  for (d = 0; d < NGPU; d++) printf(" %lld", NNZLOC[d]);
  printf("\n");
  {
    double mean_nnz = (double)A.nnz / (double)NGPU, worst = 0.0;
    for (d = 0; d < NGPU; d++)
      if ((double)NNZLOC[d] > worst) worst = (double)NNZLOC[d];
    printf("  nnz imbalance   : %.2f%%  (max vs mean)\n",
           100.0 * (worst - mean_nnz) / mean_nnz);
  }
  printf("  device memory   : %.2f GB total, %.2f GB max/GPU\n", gpu_mem, gpu_mem_dev);
  printf("  halo entries    :");
  for (d = 0; d < NGPU; d++) printf(" %d", RLEN[d]);
  printf("   (recv/GPU/iter)\n");
  printf("  halo fraction   : %.3f%% of n\n",
         100.0 * halo_gb * 1e9 / 8.0 / (double)A.n);
  printf("  p exchanged     : %.6f GB/GPU/iter  (replicated would be %.3f, %.0fx more)\n",
         halo_gb, repl_gb, halo_gb > 0.0 ? repl_gb / halo_gb : 0.0);
  printf("  exchange mode   : %s\n",
         OVERLAP ? "overlapped with interior SpMV (wide host team, no nowait)"
                 : "sequential (CG_OVERLAP=0)");
  {
    double tot = 0.0, ints = 0.0;
    for (d = 0; d < NGPU; d++) { tot += (double)NLOC[d]; ints += (double)N_INT[d]; }
    printf("  interior rows   : %.2f%%  (rows computable before the halo lands)\n",
           100.0 * ints / tot);
    printf("  boundary rows   :");
    for (d = 0; d < NGPU; d++) printf(" %d", N_BND[d]);
    printf("\n");
  }
  printf("  build time      : %.2f s  (host)\n", t1 - t0);
  printf("---------------------------------------------------------\n");


  {
    int run;
    double s_iter, s_spmv, s_halo;
    best_iter = 1e30; best_spmv = 1e30;
    sum_iter = 0.0; sum_spmv = 0.0; sum_halo = 0.0;

    for (run = 0; run < nruns; run++) {
      vfill(0.0, d_x);                           /* same start each run */

      iters = cg_solve(rtol, maxit, &relres, &tsolve, &tspmv_total, &thalo_total);

      denom  = iters ? iters : 1;
      s_iter = tsolve / denom;
      s_spmv = tspmv_total / denom;
      s_halo = thalo_total / denom;

      sum_iter += s_iter;  if (s_iter < best_iter) best_iter = s_iter;
      sum_spmv += s_spmv;  if (s_spmv < best_spmv) best_spmv = s_spmv;
      sum_halo += s_halo;

      if (nruns > 1)
        printf("  run %d/%d        : %.4f ms/iter  (%d iters, relres %.2e)\n",
               run + 1, nruns, 1e3 * s_iter, iters, relres);
    }
    tsolve_iter = sum_iter / nruns;
    tspmv       = sum_spmv / nruns;
    thalo       = sum_halo / nruns;
  }


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
  if (OVERLAP) {
    printf("    SpMV+halo     : %.4f ms/iter   (%.1f GB/s on SpMV traffic alone)\n",
           tspmv * 1e3, bw);
    printf("                    pack and unpack are inside this; the peer\n");
    printf("                    copies run underneath the interior rows, so\n");
    printf("                    they have no separate duration. CG_OVERLAP=0\n");
    printf("                    times the two apart.\n");
  } else {
    printf("    of which SpMV : %.4f ms/iter   (%.1f GB/s, lower-bound traffic)\n",
           tspmv * 1e3, bw);
    printf("    halo exchange : %.4f ms/iter   (%.6f GB/GPU, pack+copy+scatter)\n",
           thalo * 1e3, halo_gb);
  }
  printf("    other (dots,  : %.4f ms/iter\n",
         1e3 * (tsolve_iter - tspmv - thalo));
  printf("     axpy, copy)  \n");
  printf("  SpMV share      : %.1f%%\n", 100.0 * tspmv / tsolve_iter);
  if (!OVERLAP)
    printf("  halo share      : %.1f%%\n", 100.0 * thalo / tsolve_iter);
  printf("=========================================================\n");


  printf("RESULT,gpu,omp,%s,%d,%lld,%.2f,%.1f,%.3f,%d,%.6e,%.6f,%.6f,%.6f,%.1f,%.2f,%.6f,%.2f,%d,%d\n",
         order, A.n, (long long)A.nnz, mean_len, mean_bw, gpu_mem,
         iters, relres,
         1e3 * tsolve_iter, 1e3 * tspmv, 1e3 * (tsolve_iter - tspmv),
         100.0 * tspmv / tsolve_iter, bw,
         1e3 * best_iter, 100.0 * (tsolve_iter - best_iter) / best_iter,
         nruns, NGPU);

  printf("HALO,%s,%d,%d,%.6f,%.6f,%s\n",
         order, A.n, NGPU, 1e3 * thalo, halo_gb,
         OVERLAP ? "packedhalo-overlapped" : "packedhalo");

  for (d = 0; d < NGPU; d++) {
    int dev = DEV[d];
    omp_target_free(d_rowptr[d], dev); omp_target_free(d_col[d], dev);
    omp_target_free(d_val[d], dev);    omp_target_free(d_b[d], dev);
    omp_target_free(d_x[d], dev);      omp_target_free(d_r[d], dev);
    omp_target_free(d_p[d], dev);      omp_target_free(d_Ap[d], dev);
    omp_target_free(d_int[d], dev);    omp_target_free(d_bnd[d], dev);
    omp_target_free(d_gidx[d], dev);   omp_target_free(d_sidx[d], dev);
    omp_target_free(d_sbuf[d], dev);   omp_target_free(d_rbuf[d], dev);
  }
  csr_free(&A);
  return 0;
}


