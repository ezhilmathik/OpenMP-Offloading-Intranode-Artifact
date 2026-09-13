// mv-omp.cpp -- dense matrix-vector product y = A x  (N x N, row-major, double),
//               OpenMP target offload
//
// Build (Clang):
//   clang++ -O3 -fopenmp --offload-arch=sm_80 -ffp-contract=off -gline-tables-only \
//           -o matvec_omp mv-omp.cpp
//
// Run:     OMP_TARGET_OFFLOAD=MANDATORY ./matvec_omp <N> [reps=20]
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

// Your kernel. One change: temp is private (declared inside the loop). With
// reduction(+:temp), every thread's last temp was also summed across all teams
// at the end of the kernel -- extra runtime work that nothing used; y was not
// affected. i and j are private as loop variables.
void matrix_vector_multiply(double *d_local_A, double *d_x, double *d_local_y,
                            size_t rows, size_t cols, int device_id)
{
#pragma omp target is_device_ptr(d_local_A, d_x, d_local_y) device(device_id)
#pragma omp teams distribute parallel for
  for (size_t i = 0; i < rows; i++)
    {
      double temp = 0;
      for (size_t j = 0; j < cols; j++)
        {
          temp += d_local_A[i * cols + j] * d_x[j];
        }
      d_local_y[i] = temp;
    }
}

// One team per row; the team's threads share the columns and reduce.
void matrix_vector_multiply_team(double *d_local_A, double *d_x, double *d_local_y,
                                 size_t rows, size_t cols, int device_id)
{
#pragma omp target is_device_ptr(d_local_A, d_x, d_local_y) device(device_id)
#pragma omp teams distribute
  for (size_t i = 0; i < rows; i++)
    {
      double temp = 0;
#pragma omp parallel for reduction(+:temp)
      for (size_t j = 0; j < cols; j++)
        {
          temp += d_local_A[i * cols + j] * d_x[j];
        }
      d_local_y[i] = temp;
    }
}

// Launch-overhead probe: an empty target region (launch + wait for completion).
void emptyTargetRegion(int device_id)
{
#pragma omp target device(device_id)
  { }
}

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <N> [reps=20]\n", argv[0]); return 1; }
    const size_t N    = strtoull(argv[1], nullptr, 10);
    const int    reps = argc > 2 ? atoi(argv[2]) : 20;
    if (N == 0 || reps <= 0) { fprintf(stderr, "N and reps must be positive\n"); return 1; }

    const size_t bytesA = N * N * sizeof(double);
    const size_t bytesV = N * sizeof(double);
    const double moved  = (double)bytesA + 2.0 * (double)bytesV;

    const int host = omp_get_initial_device();
    const int dev  = omp_get_default_device();
    printf("OpenMP offload: %d device(s) visible, using device %d%s\n",
           omp_get_num_devices(), dev,
           dev == host ? "  WARNING: this is the host, no GPU offload" : "");
    printf("N = %zu, A = %.3f GB, reps = %d, %.3f GB moved per call\n",
           N, bytesA / 1e9, reps, moved / 1e9);

    double* hA   = (double*)malloc(bytesA);
    double* hx   = (double*)malloc(bytesV);
    double* hy   = (double*)malloc(bytesV);
    double* href = (double*)malloc(bytesV);
    if (!hA || !hx || !hy || !href) { fprintf(stderr, "host malloc failed\n"); return 1; }
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; ++i)
        for (size_t j = 0; j < N; ++j)
            hA[(size_t)i * N + j] = (double)((size_t)i * N + j + 1);   // your "count"
    for (size_t j = 0; j < N; ++j) hx[j] = (double)(1 + j % 3);

    #pragma omp parallel for schedule(static)                          // CPU reference (exact)
    for (long long i = 0; i < (long long)N; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < N; ++j) s += hA[(size_t)i * N + j] * hx[j];
        href[i] = s;
    }

    double* dA = (double*)omp_target_alloc(bytesA, dev);
    double* dx = (double*)omp_target_alloc(bytesV, dev);
    double* dy = (double*)omp_target_alloc(bytesV, dev);
    if (!dA || !dx || !dy) { fprintf(stderr, "omp_target_alloc failed\n"); return 1; }

    double t0 = omp_get_wtime();
    if (omp_target_memcpy(dA, hA, bytesA, 0, 0, dev, host) ||
        omp_target_memcpy(dx, hx, bytesV, 0, 0, dev, host)) {
        fprintf(stderr, "omp_target_memcpy H2D failed\n"); return 1;
    }
    printf("%-26s %11.3f ms  (A and x, not included in kernel times)\n", "H2D copy",
           (omp_get_wtime() - t0) * 1e3);

    double first_ms = 0.0;
    auto run = [&](const char* name, void (*kernel)(double*, double*, double*, size_t, size_t, int),
                   bool header) {
        memset(hy, 0xFF, bytesV);                          // NaN: unwritten entries fail the check
        if (omp_target_memcpy(dy, hy, bytesV, 0, 0, dev, host)) { fprintf(stderr, "NaN fill failed\n"); exit(1); }
        double t = omp_get_wtime();                        // warm-up
        kernel(dA, dx, dy, N, N, dev);
        if (header) first_ms = (omp_get_wtime() - t) * 1e3;
        t = omp_get_wtime();                               // blocking calls: launch + kernel + wait
        for (int r = 0; r < reps; ++r) kernel(dA, dx, dy, N, N, dev);
        const double ms = (omp_get_wtime() - t) * 1e3 / reps;
        if (omp_target_memcpy(hy, dy, bytesV, 0, 0, host, dev)) { fprintf(stderr, "D2H failed\n"); exit(1); }
        size_t bad = 0;
        for (size_t i = 0; i < N; ++i) if (!(hy[i] == href[i])) ++bad;
        if (header) {
            printf("%-26s %11.3f ms  (includes one-time runtime setup)\n", "first call (warm-up)", first_ms);
            printf("%-26s %11s     %10s   %s\n", "kernel", "time/call", "GB/s", "check vs CPU (all of y)");
        }
        printf("%-26s %11.4f ms  %10.1f   %s (%zu of %zu wrong)\n", name, ms,
               moved / (ms / 1e3) / 1e9, bad ? "FAIL" : "bitwise exact", bad, N);
    };
    run("omp thread/row (yours)", matrix_vector_multiply, true);
    run("omp team/row (reduction)", matrix_vector_multiply_team, false);

    const int L = 1000;                                    // launch overhead, as in the other programs
    emptyTargetRegion(dev);
    t0 = omp_get_wtime();
    for (int i = 0; i < L; ++i) emptyTargetRegion(dev);
    printf("%-26s %11.2f us  (empty target region, launch + wait, mean of %d)\n",
           "launch overhead", (omp_get_wtime() - t0) * 1e6 / L, L);

    omp_target_free(dA, dev);
    omp_target_free(dx, dev);
    omp_target_free(dy, dev);
    free(hA); free(hx); free(hy); free(href);
    return 0;
}
