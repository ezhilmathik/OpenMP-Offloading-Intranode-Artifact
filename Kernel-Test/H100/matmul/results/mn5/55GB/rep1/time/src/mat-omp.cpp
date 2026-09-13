// mat-omp.cpp -- dense C = A * B  (N x N, row-major, double), OpenMP target offload
//
// Build (Clang):
//   clang++ -O3 -fopenmp --offload-arch=sm_80 -ffp-contract=off -gline-tables-only \
//           -o matmul_omp mat-omp.cpp
//
// Run:     OMP_TARGET_OFFLOAD=MANDATORY ./matmul_omp <N> [reps=5]
//          (MANDATORY makes the run fail instead of silently falling back to the CPU)
//
// Counterpart of mat-cuda.cu: same N, same initial values, same timing method
// (one warm-up call, then the mean of `reps` calls, transfers excluded), and the
// same bitwise check against a sequential CPU loop. The kernel is the plain
// "teams distribute parallel for collapse(2)" loop nest; its CUDA equivalents
// are dgemm_naive and dgemm_tiled32 in mat-cuda.cu.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <omp.h>

// Your kernel. Only change: the loop variables are declared inside the loops.
// Declared at function scope, `t` (the sequential inner loop) is SHARED by all
// threads of a team under the C/C++ OpenMP rules, which is a data race. Only
// the loop variables of the collapsed loops (row, col) are privatized
// automatically.
void matrixMultiplyOpenMP(double *A, double *B, double *C, size_t N, int device_id)
{
#pragma omp target is_device_ptr(A, B, C) device(device_id)
#pragma omp teams distribute parallel for collapse(2)
  for (size_t row = 0; row < N; ++row)
    {
      for (size_t col = 0; col < N; ++col)
	{
	  double value = 0.0;
	  for (size_t t = 0; t < N; ++t)
	    {
	      value += A[row * N + t] * B[t * N + col];
	    }
	  C[row * N + col] = value;
	}
    }
}

// Launch-overhead probe: an empty target region (launch + wait for completion).
void emptyTargetRegion(int device_id)
{
#pragma omp target device(device_id)
  { }
}

// Compare ~512 entries of C (including all four corners) against a sequential
// CPU dot product in the same k order. Identical to the check in mat-cuda.cu.
static void check_sample(const double* A, const double* B, const double* C, int N,
                         double* max_rel, int* n_exact, int* n_checked)
{
    const int S = N < 23 ? N * N : 512;
    *max_rel = 0.0; *n_exact = 0; *n_checked = S;
    for (int s = 0; s < S; ++s) {
        int i, j;
        if (N < 23)      { i = s / N; j = s % N; }
        else if (s < 4)  { i = (s & 1) ? N - 1 : 0; j = (s & 2) ? N - 1 : 0; }
        else             { i = (int)((1103515245u * (unsigned)s + 12345u) % (unsigned)N);
                           j = (int)((2654435761u * (unsigned)s + 7u)     % (unsigned)N); }
        double ref = 0.0;
        for (int k = 0; k < N; ++k) ref += A[(size_t)i * N + k] * B[(size_t)k * N + j];
        const double got = C[(size_t)i * N + j];
        if (got == ref) ++*n_exact;
        const double d = std::fabs(got - ref) / std::fmax(std::fabs(ref), 1e-300);
        if (!(d <= *max_rel)) *max_rel = d;          // also catches NaN
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <N> [reps=5]\n", argv[0]); return 1; }
    const int N    = atoi(argv[1]);
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    if (N <= 0 || reps <= 0) { fprintf(stderr, "N and reps must be positive\n"); return 1; }

    const size_t nn    = (size_t)N * N;              // size_t: N*N overflows int above 46340
    const size_t bytes = nn * sizeof(double);
    const double flops = 2.0 * (double)N * N * N;

    const int host = omp_get_initial_device();
    const int dev  = omp_get_default_device();
    printf("OpenMP offload: %d device(s) visible, using device %d%s\n",
           omp_get_num_devices(), dev,
           dev == host ? "  WARNING: this is the host, no GPU offload" : "");
    printf("N = %d, each matrix %.3f GB, reps = %d, %.3f GFLOP per multiply\n",
           N, bytes / 1e9, reps, flops / 1e9);

    // Host matrices: same values as mat-cuda.cu.
    double* hA = (double*)malloc(bytes);
    double* hB = (double*)malloc(bytes);
    double* hC = (double*)malloc(bytes);
    if (!hA || !hB || !hC) { fprintf(stderr, "host malloc failed\n"); return 1; }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            hA[(size_t)i * N + j] = ((i * 131 + j * 17) % 100) / 100.0;
            hB[(size_t)i * N + j] = ((i * 7 + j * 29) % 10 + 1) / 10.0;
        }

    double* dA = (double*)omp_target_alloc(bytes, dev);
    double* dB = (double*)omp_target_alloc(bytes, dev);
    double* dC = (double*)omp_target_alloc(bytes, dev);
    if (!dA || !dB || !dC) { fprintf(stderr, "omp_target_alloc failed\n"); return 1; }

    double t0 = omp_get_wtime();
    if (omp_target_memcpy(dA, hA, bytes, 0, 0, dev, host) ||
        omp_target_memcpy(dB, hB, bytes, 0, 0, dev, host)) {
        fprintf(stderr, "omp_target_memcpy H2D failed\n"); return 1;
    }
    const double h2d_ms = (omp_get_wtime() - t0) * 1e3;
    printf("%-26s %11.3f ms  (A and B, not included in kernel times)\n", "H2D copy", h2d_ms);

    // Fill C with NaN so that entries the kernel never writes fail the check.
    memset(hC, 0xFF, bytes);
    if (omp_target_memcpy(dC, hC, bytes, 0, 0, dev, host)) {
        fprintf(stderr, "omp_target_memcpy NaN fill failed\n"); return 1;
    }

    // Warm-up. The first target region also pays one-time runtime setup,
    // which is why it must not be inside the timed loop.
    t0 = omp_get_wtime();
    matrixMultiplyOpenMP(dA, dB, dC, (size_t)N, dev);
    const double first_ms = (omp_get_wtime() - t0) * 1e3;

    // A target region without nowait is blocking, so wall time = launch + kernel.
    t0 = omp_get_wtime();
    for (int r = 0; r < reps; ++r) matrixMultiplyOpenMP(dA, dB, dC, (size_t)N, dev);
    const double ms = (omp_get_wtime() - t0) * 1e3 / reps;

    if (omp_target_memcpy(hC, dC, bytes, 0, 0, host, dev)) {
        fprintf(stderr, "omp_target_memcpy D2H failed\n"); return 1;
    }
    double max_rel; int n_exact, n_checked;
    check_sample(hA, hB, hC, N, &max_rel, &n_exact, &n_checked);
    const char* verdict = (n_exact == n_checked) ? "bitwise exact"
                        : (max_rel < 1e-12)      ? "OK (rounding differs)" : "FAIL";

    printf("%-26s %11.3f ms  (includes one-time runtime setup)\n", "first call (warm-up)", first_ms);
    printf("%-26s %11s     %10s   %s\n", "kernel", "time/call", "GFLOP/s", "check vs CPU");
    printf("%-26s %11.3f ms  %10.1f   %s (%d/%d exact, max rel err %.1e)\n",
           "omp collapse(2)", ms, flops / (ms / 1e3) / 1e9, verdict, n_exact, n_checked, max_rel);

    // Launch overhead: empty target region, launch + wait, averaged over L calls.
    const int L = 1000;
    emptyTargetRegion(dev);
    t0 = omp_get_wtime();
    for (int i = 0; i < L; ++i) emptyTargetRegion(dev);
    printf("%-26s %11.2f us  (empty target region, launch + wait, mean of %d)\n",
           "launch overhead", (omp_get_wtime() - t0) * 1e6 / L, L);

    omp_target_free(dA, dev);
    omp_target_free(dB, dev);
    omp_target_free(dC, dev);
    free(hA); free(hB); free(hC);
    return 0;
}
