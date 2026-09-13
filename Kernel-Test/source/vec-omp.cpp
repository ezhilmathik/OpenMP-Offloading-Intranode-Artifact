// vec-omp.cpp -- vector addition C = A + B (double precision), OpenMP target offload
//
// Build (Clang):
//   clang++ -O3 -fopenmp --offload-arch=sm_80 -ffp-contract=off -gline-tables-only \
//           -o vecadd_omp vec-omp.cpp
//
// Run:     OMP_TARGET_OFFLOAD=MANDATORY ./vecadd_omp <N elements> [reps=100]
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

// Your kernel, unchanged. `i` is declared outside the loop, which is fine here:
// it is the loop variable of the distribute/for construct itself, so OpenMP
// makes it private automatically.
void vectorAddOpenMP(double *A, double *B, double *C, size_t N, int device_id)
{
  size_t i;
#pragma omp target is_device_ptr(A, B, C) device(device_id)
#pragma omp teams distribute parallel for
  for (i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
}

// Launch-overhead probe: an empty target region (launch + wait for completion).
void emptyTargetRegion(int device_id)
{
#pragma omp target device(device_id)
  { }
}

// Every element against the CPU; returns the number of wrong (or NaN) entries.
static size_t check_all(const double* A, const double* B, const double* C, size_t N)
{
    size_t bad = 0;
    #pragma omp parallel for reduction(+:bad) schedule(static)
    for (long long i = 0; i < (long long)N; ++i)
        if (!(C[i] == A[i] + B[i])) ++bad;
    return bad;
}

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <N elements> [reps=100]\n", argv[0]); return 1; }
    const size_t N    = strtoull(argv[1], nullptr, 10);   // size_t: 55 GB is 2.3e9 elements
    const int    reps = argc > 2 ? atoi(argv[2]) : 100;
    if (N == 0 || reps <= 0) { fprintf(stderr, "N and reps must be positive\n"); return 1; }

    const size_t bytes = N * sizeof(double);
    const double moved = 3.0 * (double)bytes;

    const int host = omp_get_initial_device();
    const int dev  = omp_get_default_device();
    printf("OpenMP offload: %d device(s) visible, using device %d%s\n",
           omp_get_num_devices(), dev,
           dev == host ? "  WARNING: this is the host, no GPU offload" : "");
    printf("N = %zu elements, each vector %.3f GB, reps = %d, %.3f GB moved per call\n",
           N, bytes / 1e9, reps, moved / 1e9);

    double* hA = (double*)malloc(bytes);
    double* hB = (double*)malloc(bytes);
    double* hC = (double*)malloc(bytes);
    if (!hA || !hB || !hC) { fprintf(stderr, "host malloc failed\n"); return 1; }
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; ++i) { hA[i] = 1.0; hB[i] = (double)i; }

    double* dA = (double*)omp_target_alloc(bytes, dev);
    double* dB = (double*)omp_target_alloc(bytes, dev);
    double* dC = (double*)omp_target_alloc(bytes, dev);
    if (!dA || !dB || !dC) { fprintf(stderr, "omp_target_alloc failed\n"); return 1; }

    double t0 = omp_get_wtime();
    if (omp_target_memcpy(dA, hA, bytes, 0, 0, dev, host) ||
        omp_target_memcpy(dB, hB, bytes, 0, 0, dev, host)) {
        fprintf(stderr, "omp_target_memcpy H2D failed\n"); return 1;
    }
    printf("%-26s %11.3f ms  (A and B, not included in kernel times)\n", "H2D copy",
           (omp_get_wtime() - t0) * 1e3);

    memset(hC, 0xFF, bytes);                               // NaN: unwritten entries fail the check
    if (omp_target_memcpy(dC, hC, bytes, 0, 0, dev, host)) {
        fprintf(stderr, "omp_target_memcpy NaN fill failed\n"); return 1;
    }

    t0 = omp_get_wtime();                                   // warm-up (one-time runtime setup)
    vectorAddOpenMP(dA, dB, dC, N, dev);
    const double first_ms = (omp_get_wtime() - t0) * 1e3;

    t0 = omp_get_wtime();                                   // blocking calls: launch + kernel + wait
    for (int r = 0; r < reps; ++r) vectorAddOpenMP(dA, dB, dC, N, dev);
    const double ms = (omp_get_wtime() - t0) * 1e3 / reps;

    if (omp_target_memcpy(hC, dC, bytes, 0, 0, host, dev)) {
        fprintf(stderr, "omp_target_memcpy D2H failed\n"); return 1;
    }
    const size_t bad = check_all(hA, hB, hC, N);

    printf("%-26s %11.3f ms  (includes one-time runtime setup)\n", "first call (warm-up)", first_ms);
    printf("%-26s %11s     %10s   %s\n", "kernel", "time/call", "GB/s", "check vs CPU (all elements)");
    printf("%-26s %11.4f ms  %10.1f   %s (%zu of %zu wrong)\n", "omp teams distribute", ms,
           moved / (ms / 1e3) / 1e9, bad ? "FAIL" : "bitwise exact", bad, N);

    const int L = 1000;                                     // launch overhead, as in mat-omp.cpp
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
