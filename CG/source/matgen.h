/* ===========================================================================
 * matgen.h -- unstructured sparse matrix production. SHARED by all versions.
 *
 * Included by cg_omp.c, 1-cuda.cu, 2-cuda.cu, 4-cuda.cu.
 * Header-only with static functions, so it compiles under gcc and nvcc alike
 * and needs no separate compilation step.
 *
 * This file knows nothing about CG, GPUs, or partitioning. It only produces a
 * CSR matrix, by one of two routes:
 *
 *   matgen_radius()  synthetic unstructured matrix at any scale
 *   matgen_mm()      Matrix Market file (SuiteSparse)
 *
 * plus optional symmetric reorderings, which are the experimental knob:
 *
 *   reorder_rcm()     bandwidth-reducing, columns cluster near the diagonal
 *   reorder_random()  destroys all locality, the pessimistic reference
 *   (natural)         whatever order the generator or file gave
 *
 * ---------------------------------------------------------------------------
 * THE SYNTHETIC MATRIX
 *
 * N random points in the unit cube; connect every pair closer than r. This is
 * a radius graph, and it has the three properties that matter:
 *
 *   symmetric by construction  -- distance is symmetric, so no mirroring pass
 *   irregular valence          -- degree is Poisson(avg_deg), so std ~ sqrt(k).
 *                                 At avg_deg=15 that is a spread of about 4,
 *                                 close to real tetrahedral mesh valence.
 *   geometric locality         -- nearby nodes connect, so it partitions and
 *                                 reorders like a real mesh
 *
 * That last point is why this is not just a random sparse matrix. A matrix
 * with random column indices has no locality at all, would partition
 * catastrophically, and would give pessimistic numbers no real mesh produces.
 *
 * Values: off-diagonal -1, diagonal = degree + shift. That is a graph
 * Laplacian plus a shift. Without the shift the matrix is singular (the
 * constant vector is in the null space); with it the matrix is strictly
 * diagonally dominant, hence SPD. The shift controls conditioning, and for
 * unpreconditioned CG it controls the iteration count directly -- small shift
 * means ill-conditioned and many iterations.
 *
 * Construction is O(N): points are binned into a uniform grid with cell size
 * r, so each point only tests the 27 surrounding cells.
 *
 * Memory at N = 64M, avg_deg = 15: coordinates ~1.5 GB, grid ~0.7 GB,
 * CSR ~12 GB. All host-side and all transient except the CSR.
 * ========================================================================= */

#ifndef MATGEN_H
#define MATGEN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* int32 columns: row counts stay far below 2^31 and index traffic is a third
 * of SpMV bandwidth, so 4 bytes matters.
 * int64 row pointers: 64M rows x 15 nnz = 1e9 nonzeros, close enough to the
 * int32 ceiling of 2.147e9 that economising here is a mistake. */
typedef long long ptr_t;
typedef int       idx_t;

typedef struct {
  int     n;        /* rows */
  ptr_t   nnz;
  ptr_t  *rowptr;   /* n+1 */
  idx_t  *col;      /* nnz, sorted within each row */
  double *val;      /* nnz */
} CSR;

/* ----- small utilities --------------------------------------------------- */

static void *mg_alloc(size_t bytes)
{
  void *p = malloc(bytes ? bytes : 1);
  if (!p) { fprintf(stderr, "matgen: out of host memory (%zu bytes)\n", bytes); exit(1); }
  return p;
}

static void csr_free(CSR *A)
{
  free(A->rowptr); free(A->col); free(A->val);
  A->rowptr = NULL; A->col = NULL; A->val = NULL;
  A->n = 0; A->nnz = 0;
}

/* deterministic 64-bit LCG, so runs are reproducible across machines */
static unsigned long long mg_rng_state = 88172645463325252ULL;

static void mg_seed(unsigned long long s) { mg_rng_state = s ? s : 1; }

static double mg_rand(void)   /* uniform [0,1) */
{
  mg_rng_state = mg_rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
  return (double)(mg_rng_state >> 11) * (1.0 / 9007199254740992.0);
}

/* Deterministic hash of an index to [0,1). Used to build the right-hand side.
 *
 * This exists because the RHS must not be the constant vector: for a graph
 * Laplacian plus a shift, the constant vector is an eigenvector, so CG
 * converges in ONE iteration and measures nothing. It also must be
 * reproducible across CPU and GPU and fillable in parallel, which rules out a
 * sequential RNG. A hash of the index satisfies all three.
 *
 * The CUDA files carry a __device__ copy of this function so every version
 * solves bit-identical problems. */
static double mg_hash01(int i)
{
  unsigned long long z = (unsigned long long)(unsigned)i * 0x9E3779B97F4A7C15ULL
                       + 0x853C49E6748FEA9BULL;
  z ^= z >> 30; z *= 0xBF58476D1CE4E5B9ULL;
  z ^= z >> 27; z *= 0x94D049BB133111EBULL;
  z ^= z >> 31;
  return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

/* insertion sort of one row's (col,val) pairs -- rows are ~15 long, so this
 * is the right algorithm. Sorted columns help cache locality and are what
 * cuSPARSE expects if you ever want to cross-check against it. */
static void mg_sort_row(idx_t *col, double *val, int len)
{
  int i, j;
  for (i = 1; i < len; ++i) {
    idx_t c = col[i];
    double v = val[i];
    for (j = i - 1; j >= 0 && col[j] > c; --j) {
      col[j + 1] = col[j];
      val[j + 1] = val[j];
    }
    col[j + 1] = c;
    val[j + 1] = v;
  }
}

/* ===========================================================================
 * SYNTHETIC: radius graph on a random point cloud
 *
 * npts     number of rows
 * avg_deg  target mean number of off-diagonal entries per row
 * shift    added to the diagonal; controls conditioning
 * seed     RNG seed
 *
 * Returns 0 on success.
 * ========================================================================= */
static int matgen_radius(CSR *A, int npts, double avg_deg, double shift,
                         unsigned long long seed)
{
  double *px, *py, *pz;
  double r, r2, h;
  int m, i, mm3;
  int *cell_start, *cell_pts, *cell_of;
  ptr_t *rowptr;
  idx_t *col;
  double *val;
  ptr_t nnz;

  if (npts < 2 || avg_deg < 1.0) return 1;

  mg_seed(seed);
  px = (double *)mg_alloc((size_t)npts * sizeof(double));
  py = (double *)mg_alloc((size_t)npts * sizeof(double));
  pz = (double *)mg_alloc((size_t)npts * sizeof(double));
  for (i = 0; i < npts; ++i) { px[i] = mg_rand(); py[i] = mg_rand(); pz[i] = mg_rand(); }

  /* mean degree of a radius graph = density * volume of the ball
   *   avg_deg = npts * (4/3) pi r^3   ->   r = cbrt(3 avg_deg / (4 pi npts)) */
  r  = pow(3.0 * avg_deg / (4.0 * M_PI * (double)npts), 1.0 / 3.0);
  r2 = r * r;

  /* uniform grid with cell size >= r, so only the 27 surrounding cells can
   * hold a neighbour */
  m = (int)floor(1.0 / r);
  if (m < 1) m = 1;
  if (m > 1024) m = 1024;            /* cap memory: 1024^3 cells is 4 GB of ints */
  h = 1.0 / (double)m;
  mm3 = m * m * m;

  /* counting sort of points into cells */
  cell_of    = (int *)mg_alloc((size_t)npts * sizeof(int));
  cell_start = (int *)mg_alloc(((size_t)mm3 + 1) * sizeof(int));
  cell_pts   = (int *)mg_alloc((size_t)npts * sizeof(int));
  memset(cell_start, 0, ((size_t)mm3 + 1) * sizeof(int));

  for (i = 0; i < npts; ++i) {
    int cx = (int)(px[i] / h), cy = (int)(py[i] / h), cz = (int)(pz[i] / h);
    if (cx >= m) cx = m - 1;
    if (cy >= m) cy = m - 1;
    if (cz >= m) cz = m - 1;
    cell_of[i] = cx + m * (cy + m * cz);
    cell_start[cell_of[i] + 1]++;
  }
  for (i = 0; i < mm3; ++i) cell_start[i + 1] += cell_start[i];
  {
    int *fill = (int *)mg_alloc((size_t)mm3 * sizeof(int));
    memcpy(fill, cell_start, (size_t)mm3 * sizeof(int));
    for (i = 0; i < npts; ++i) cell_pts[fill[cell_of[i]]++] = i;
    free(fill);
  }

  /* Relabel points into cell order, so node i and node i+1 are usually
   * spatially close. Without this the "natural" ordering is the order the RNG
   * produced points in -- effectively random -- and the natural/rcm/random
   * comparison would have nothing to show. A real mesher emits nodes with
   * some spatial coherence, so this makes the synthetic matrix behave like a
   * mesh file rather than like a random graph. */
  {
    double *qx = (double *)mg_alloc((size_t)npts * sizeof(double));
    double *qy = (double *)mg_alloc((size_t)npts * sizeof(double));
    double *qz = (double *)mg_alloc((size_t)npts * sizeof(double));
    int    *qc = (int *)mg_alloc((size_t)npts * sizeof(int));
    for (i = 0; i < npts; ++i) {
      int o = cell_pts[i];
      qx[i] = px[o]; qy[i] = py[o]; qz[i] = pz[o];
      qc[i] = cell_of[o];
    }
    free(px); free(py); free(pz); free(cell_of);
    px = qx; py = qy; pz = qz; cell_of = qc;
    for (i = 0; i < npts; ++i) cell_pts[i] = i;   /* now the identity */
  }

  /* pass 1: count neighbours per row (+1 for the diagonal) */
  rowptr = (ptr_t *)mg_alloc(((size_t)npts + 1) * sizeof(ptr_t));
  memset(rowptr, 0, ((size_t)npts + 1) * sizeof(ptr_t));

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4096)
#endif
  for (i = 0; i < npts; ++i) {
    int cx = cell_of[i] % m;
    int cy = (cell_of[i] / m) % m;
    int cz = cell_of[i] / (m * m);
    int dx, dy, dz, cnt = 1;                       /* diagonal */
    for (dz = -1; dz <= 1; ++dz)
      for (dy = -1; dy <= 1; ++dy)
        for (dx = -1; dx <= 1; ++dx) {
          int nxx = cx + dx, nyy = cy + dy, nzz = cz + dz, c, s, e, t;
          if (nxx < 0 || nxx >= m || nyy < 0 || nyy >= m || nzz < 0 || nzz >= m) continue;
          c = nxx + m * (nyy + m * nzz);
          s = cell_start[c]; e = cell_start[c + 1];
          for (t = s; t < e; ++t) {
            int j = cell_pts[t];
            double ddx, ddy, ddz;
            if (j == i) continue;
            ddx = px[i] - px[j]; ddy = py[i] - py[j]; ddz = pz[i] - pz[j];
            if (ddx * ddx + ddy * ddy + ddz * ddz < r2) cnt++;
          }
        }
    rowptr[i + 1] = cnt;
  }
  for (i = 0; i < npts; ++i) rowptr[i + 1] += rowptr[i];
  nnz = rowptr[npts];

  col = (idx_t *)mg_alloc((size_t)nnz * sizeof(idx_t));
  val = (double *)mg_alloc((size_t)nnz * sizeof(double));

  /* pass 2: fill */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4096)
#endif
  for (i = 0; i < npts; ++i) {
    int cx = cell_of[i] % m;
    int cy = (cell_of[i] / m) % m;
    int cz = cell_of[i] / (m * m);
    int dx, dy, dz;
    ptr_t k = rowptr[i];
    ptr_t diag_slot = k;
    int deg = 0;

    col[k] = i; val[k] = 0.0; k++;                 /* reserve the diagonal */

    for (dz = -1; dz <= 1; ++dz)
      for (dy = -1; dy <= 1; ++dy)
        for (dx = -1; dx <= 1; ++dx) {
          int nxx = cx + dx, nyy = cy + dy, nzz = cz + dz, c, s, e, t;
          if (nxx < 0 || nxx >= m || nyy < 0 || nyy >= m || nzz < 0 || nzz >= m) continue;
          c = nxx + m * (nyy + m * nzz);
          s = cell_start[c]; e = cell_start[c + 1];
          for (t = s; t < e; ++t) {
            int j = cell_pts[t];
            double ddx, ddy, ddz;
            if (j == i) continue;
            ddx = px[i] - px[j]; ddy = py[i] - py[j]; ddz = pz[i] - pz[j];
            if (ddx * ddx + ddy * ddy + ddz * ddz < r2) {
              col[k] = j; val[k] = -1.0; k++; deg++;
            }
          }
        }
    val[diag_slot] = (double)deg + shift;
    mg_sort_row(&col[rowptr[i]], &val[rowptr[i]], (int)(rowptr[i + 1] - rowptr[i]));
  }

  free(px); free(py); free(pz);
  free(cell_of); free(cell_start); free(cell_pts);

  A->n = npts; A->nnz = nnz; A->rowptr = rowptr; A->col = col; A->val = val;
  return 0;
}

/* ===========================================================================
 * MATRIX MARKET reader (SuiteSparse)
 *
 * Handles: matrix coordinate real|integer|pattern  symmetric|general
 *
 * Symmetric files store only the lower triangle, so off-diagonal entries are
 * mirrored on read. Forgetting that mirroring is the classic bug here -- the
 * matrix silently becomes triangular and CG produces nonsense.
 *
 * If diag_shift > 0 it is added to every diagonal entry. Leave it at 0 for
 * genuine SuiteSparse SPD matrices; it exists so pattern-only files can be
 * turned into something SPD.
 * ========================================================================= */
static int matgen_mm(CSR *A, const char *path, double diag_shift)
{
  FILE *f = fopen(path, "r");
  char line[1024], banner[64], mtx[64], crd[64], dtype[64], sym[64];
  int nrow, ncol, i, is_sym = 0, is_pattern = 0;
  ptr_t declared = 0, k, nnz = 0;
  int    *I, *J;
  double *V;
  ptr_t  *rowptr;
  idx_t  *col;
  double *val;
  ptr_t  *fill;
  int has_diag_missing = 0;

  if (!f) { fprintf(stderr, "matgen: cannot open %s\n", path); return 1; }

  if (!fgets(line, sizeof(line), f)) { fclose(f); return 1; }
  if (sscanf(line, "%63s %63s %63s %63s %63s", banner, mtx, crd, dtype, sym) != 5) {
    fprintf(stderr, "matgen: bad Matrix Market banner\n"); fclose(f); return 1;
  }
  if (strcmp(crd, "coordinate") != 0) {
    fprintf(stderr, "matgen: only coordinate format supported\n"); fclose(f); return 1;
  }
  is_sym     = (strcmp(sym, "symmetric") == 0);
  is_pattern = (strcmp(dtype, "pattern") == 0);

  do {
    if (!fgets(line, sizeof(line), f)) { fclose(f); return 1; }
  } while (line[0] == '%');
  if (sscanf(line, "%d %d %lld", &nrow, &ncol, &declared) != 3) { fclose(f); return 1; }
  if (nrow != ncol) {
    fprintf(stderr, "matgen: matrix is not square (%d x %d)\n", nrow, ncol);
    fclose(f); return 1;
  }

  I = (int *)mg_alloc((size_t)declared * sizeof(int));
  J = (int *)mg_alloc((size_t)declared * sizeof(int));
  V = (double *)mg_alloc((size_t)declared * sizeof(double));

  for (k = 0; k < declared; ++k) {
    int a, b;
    double v = 1.0;
    if (!fgets(line, sizeof(line), f)) { fclose(f); return 1; }
    if (is_pattern) { if (sscanf(line, "%d %d", &a, &b) != 2) { fclose(f); return 1; } }
    else            { if (sscanf(line, "%d %d %lf", &a, &b, &v) != 3) { fclose(f); return 1; } }
    I[k] = a - 1; J[k] = b - 1; V[k] = v;          /* Matrix Market is 1-based */
    nnz += (is_sym && a != b) ? 2 : 1;
  }
  fclose(f);

  rowptr = (ptr_t *)mg_alloc(((size_t)nrow + 1) * sizeof(ptr_t));
  memset(rowptr, 0, ((size_t)nrow + 1) * sizeof(ptr_t));
  for (k = 0; k < declared; ++k) {
    rowptr[I[k] + 1]++;
    if (is_sym && I[k] != J[k]) rowptr[J[k] + 1]++;
  }
  for (i = 0; i < nrow; ++i) rowptr[i + 1] += rowptr[i];

  col = (idx_t *)mg_alloc((size_t)nnz * sizeof(idx_t));
  val = (double *)mg_alloc((size_t)nnz * sizeof(double));
  fill = (ptr_t *)mg_alloc((size_t)nrow * sizeof(ptr_t));
  memcpy(fill, rowptr, (size_t)nrow * sizeof(ptr_t));

  for (k = 0; k < declared; ++k) {
    col[fill[I[k]]] = J[k]; val[fill[I[k]]] = V[k]; fill[I[k]]++;
    if (is_sym && I[k] != J[k]) {
      col[fill[J[k]]] = I[k]; val[fill[J[k]]] = V[k]; fill[J[k]]++;
    }
  }
  free(I); free(J); free(V); free(fill);

  for (i = 0; i < nrow; ++i)
    mg_sort_row(&col[rowptr[i]], &val[rowptr[i]], (int)(rowptr[i + 1] - rowptr[i]));

  if (diag_shift != 0.0) {
    for (i = 0; i < nrow; ++i) {
      int found = 0;
      for (k = rowptr[i]; k < rowptr[i + 1]; ++k)
        if (col[k] == i) { val[k] += diag_shift; found = 1; break; }
      if (!found) has_diag_missing = 1;
    }
    if (has_diag_missing)
      fprintf(stderr, "matgen: warning, some rows have no stored diagonal; "
                      "shift not applied there\n");
  }

  A->n = nrow; A->nnz = nnz; A->rowptr = rowptr; A->col = col; A->val = val;
  return 0;
}

/* ===========================================================================
 * REORDERING
 *
 * Symmetric permutation: A' = P A P^T, with perm[old] = new.
 * The matrix is mathematically identical -- same eigenvalues, same iteration
 * count -- only the memory access pattern changes. That is what makes this a
 * clean experiment.
 * ========================================================================= */
static void mg_apply_perm(CSR *A, const int *perm)
{
  const int n = A->n;
  ptr_t *nrp = (ptr_t *)mg_alloc(((size_t)n + 1) * sizeof(ptr_t));
  idx_t *nc  = (idx_t *)mg_alloc((size_t)A->nnz * sizeof(idx_t));
  double *nv = (double *)mg_alloc((size_t)A->nnz * sizeof(double));
  int i;
  ptr_t k;

  memset(nrp, 0, ((size_t)n + 1) * sizeof(ptr_t));
  for (i = 0; i < n; ++i) nrp[perm[i] + 1] = A->rowptr[i + 1] - A->rowptr[i];
  for (i = 0; i < n; ++i) nrp[i + 1] += nrp[i];

  for (i = 0; i < n; ++i) {
    ptr_t dst = nrp[perm[i]];
    for (k = A->rowptr[i]; k < A->rowptr[i + 1]; ++k) {
      nc[dst] = perm[A->col[k]];
      nv[dst] = A->val[k];
      dst++;
    }
    mg_sort_row(&nc[nrp[perm[i]]], &nv[nrp[perm[i]]],
                (int)(nrp[perm[i] + 1] - nrp[perm[i]]));
  }

  free(A->rowptr); free(A->col); free(A->val);
  A->rowptr = nrp; A->col = nc; A->val = nv;
}

/* Reverse Cuthill-McKee. BFS from the lowest-degree node, visiting each
 * level's neighbours in increasing degree order, then reverse the result.
 * Clusters column indices near the diagonal, which both improves cache hits
 * on x and -- in the multi-GPU versions -- shrinks the range of x each GPU
 * needs, so it cuts communication volume too. */
static void reorder_rcm(CSR *A)
{
  const int n = A->n;
  int *perm  = (int *)mg_alloc((size_t)n * sizeof(int));
  int *order = (int *)mg_alloc((size_t)n * sizeof(int));
  char *seen = (char *)mg_alloc((size_t)n);
  int *deg   = (int *)mg_alloc((size_t)n * sizeof(int));
  int head = 0, tail = 0, i, start, best;
  ptr_t k;

  memset(seen, 0, (size_t)n);
  for (i = 0; i < n; ++i) deg[i] = (int)(A->rowptr[i + 1] - A->rowptr[i]);

  for (;;) {
    /* next unvisited component, seeded at its lowest-degree node */
    start = -1; best = 0;
    for (i = 0; i < n; ++i)
      if (!seen[i] && (start < 0 || deg[i] < best)) { start = i; best = deg[i]; }
    if (start < 0) break;

    seen[start] = 1;
    order[tail++] = start;
    while (head < tail) {
      int u = order[head++], cnt = 0, a, b;
      int first = tail;
      for (k = A->rowptr[u]; k < A->rowptr[u + 1]; ++k) {
        int v = A->col[k];
        if (v != u && !seen[v]) { seen[v] = 1; order[tail++] = v; cnt++; }
      }
      /* sort this level's new nodes by degree (insertion; levels are short) */
      for (a = first + 1; a < first + cnt; ++a) {
        int t = order[a];
        for (b = a - 1; b >= first && deg[order[b]] > deg[t]; --b) order[b + 1] = order[b];
        order[b + 1] = t;
      }
    }
  }

  for (i = 0; i < n; ++i) perm[order[i]] = n - 1 - i;   /* reverse */
  mg_apply_perm(A, perm);
  free(perm); free(order); free(seen); free(deg);
}

/* Fisher-Yates. Destroys all locality; the pessimistic reference point. */
static void reorder_random(CSR *A, unsigned long long seed)
{
  const int n = A->n;
  int *perm = (int *)mg_alloc((size_t)n * sizeof(int));
  int i, j, t;

  mg_seed(seed);
  for (i = 0; i < n; ++i) perm[i] = i;
  for (i = n - 1; i > 0; --i) {
    j = (int)(mg_rand() * (double)(i + 1));
    if (j > i) j = i;
    t = perm[i]; perm[i] = perm[j]; perm[j] = t;
  }
  mg_apply_perm(A, perm);
  free(perm);
}

/* ===========================================================================
 * STATS
 *
 * Row-length mean and standard deviation say whether a vector-per-row SpMV
 * kernel is appropriate: uniform rows suit it, wide spread wastes lanes on
 * short rows and serialises long ones.
 *
 * Mean bandwidth |i - col| measures locality, which is what the reorderings
 * change. Lower is better for cache hits and, in the multi-GPU versions, for
 * communication volume.
 * ========================================================================= */
static void csr_stats(const CSR *A, double *mean_len, double *std_len,
                      int *max_len, double *mean_bw)
{
  const int n = A->n;
  double s = 0.0, s2 = 0.0, bw = 0.0;
  int mx = 0, i;
  ptr_t k;

  for (i = 0; i < n; ++i) {
    int len = (int)(A->rowptr[i + 1] - A->rowptr[i]);
    s  += (double)len;
    s2 += (double)len * (double)len;
    if (len > mx) mx = len;
    for (k = A->rowptr[i]; k < A->rowptr[i + 1]; ++k) {
      int d = A->col[k] - i;
      bw += (d < 0) ? -(double)d : (double)d;
    }
  }
  *mean_len = s / (double)n;
  *std_len  = sqrt(s2 / (double)n - (s / (double)n) * (s / (double)n));
  *max_len  = mx;
  *mean_bw  = bw / (double)A->nnz;
}

#endif /* MATGEN_H */
