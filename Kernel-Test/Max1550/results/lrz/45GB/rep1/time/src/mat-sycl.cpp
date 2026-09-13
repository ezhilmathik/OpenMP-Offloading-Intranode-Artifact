// mat-sycl.cpp -- dense C = A * B  (N x N, row-major, double precision), SYCL
//
// Build (oneAPI):
//   icpx -fsycl -O3 -ffp-contract=off -fno-fast-math -fiopenmp -o matmul_sycl mat-sycl.cpp
//
// Run:   ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0 ./matmul_sycl <N> [reps=5]
//
// The four kernels of mat-cuda.cu, ported without changing the algorithm: same
// work per work item, same tile sizes, same loop order, same summation order.
// Only the constructs that have no SYCL equivalent are replaced:
//   blockIdx/threadIdx -> nd_item,  __shared__ -> local_accessor,
//   __syncthreads()    -> it.barrier(access::fence_space::local_space).
//
//   naive      one work item per C element, 32x8 work-group (the direct
//              counterpart of an "omp target teams distribute parallel for
//              collapse(2)" loop nest)
//   naive 1-D  the same kernel with a 128x1 work-group: the OpenMP layout
//   tiled32    32x32 tiles in local memory, one C element per work item
//   regtile_*  local-memory tiles plus a TM x TN block of C per work item,
//              kept in private variables (registers)
//
// All kernels add the k terms in order 0..N-1, one rounded multiply and one
// rounded add per term. With -ffp-contract=off (and -fno-fast-math) the results
// are bitwise identical to a sequential CPU triple loop, which is what the
// check tests -- the same guarantee as --fmad=false gives with nvcc.
//
// Kernel time comes from SYCL event profiling; the wall time per call is shown
// next to it.

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

using namespace sycl;

// Named kernels: profilers (VTune) report these names instead of lambda numbers.
class k_mat_naive; class k_mat_tiled32; class k_empty;
template <int BM, int BN, int BK, int TM, int TN> class k_mat_regtile;

static constexpr int TILE = 32;

static double event_ms(const event &e)
{
    const auto t0 = e.get_profiling_info<info::event_profiling::command_start>();
    const auto t1 = e.get_profiling_info<info::event_profiling::command_end>();
    return (t1 - t0) / 1.0e6;
}

static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n ? (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : 0.0;
}

// ~512 entries of C (including all four corners) against a sequential CPU dot
// product in the same k order. Identical to the check in mat-cuda.cu.
static void check_sample(const double *A, const double *B, const double *C, int N,
                         double *max_rel, int *n_exact, int *n_checked)
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

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <N> [reps=5]\n", argv[0]); return 1; }
    const int N    = atoi(argv[1]);
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    if (N <= 0 || reps <= 0) { fprintf(stderr, "N and reps must be positive\n"); return 1; }

    const size_t nn    = (size_t)N * N;            // size_t: N*N overflows int above 46340
    const size_t bytes = nn * sizeof(double);
    const double flops = 2.0 * (double)N * N * N;

    queue q{gpu_selector_v, property::queue::enable_profiling{}};
    const device dev = q.get_device();
    printf("GPU : %s (%u EUs, %zu GB)\n",
           dev.get_info<info::device::name>().c_str(),
           dev.get_info<info::device::max_compute_units>(),
           (size_t)(dev.get_info<info::device::global_mem_size>() / (1ull << 30)));
    printf("N = %d, each matrix %.3f GB, reps = %d, %.3f GFLOP per multiply\n",
           N, bytes / 1e9, reps, flops / 1e9);

    double *hA = (double *)malloc(bytes);
    double *hB = (double *)malloc(bytes);
    double *hC = (double *)malloc(bytes);
    if (!hA || !hB || !hC) { fprintf(stderr, "host malloc failed\n"); return 1; }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            hA[(size_t)i * N + j] = ((i * 131 + j * 17) % 100) / 100.0;
            hB[(size_t)i * N + j] = ((i * 7 + j * 29) % 10 + 1) / 10.0;
        }

    double *dA = malloc_device<double>(nn, q);
    double *dB = malloc_device<double>(nn, q);
    double *dC = malloc_device<double>(nn, q);
    if (!dA || !dB || !dC) { fprintf(stderr, "malloc_device failed\n"); return 1; }

    auto t0 = std::chrono::steady_clock::now();
    q.memcpy(dA, hA, bytes).wait();
    q.memcpy(dB, hB, bytes).wait();
    printf("%-26s %11.3f ms  (A and B, not included in kernel times)\n", "H2D copy",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    printf("%-26s %11s     %10s   %s\n", "kernel", "time/call", "GFLOP/s", "check vs CPU");

    auto run = [&](const char *name, const char *cfg, auto submit) {
        q.memset(dC, 0xFF, bytes).wait();            // NaN: unwritten entries fail the check
        submit().wait();                              // warm-up
        std::vector<double> kms;
        const auto w0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) { event e = submit(); e.wait(); kms.push_back(event_ms(e)); }
        const double wall = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - w0).count() / reps;
        q.memcpy(hC, dC, bytes).wait();
        double max_rel; int n_exact, n_checked;
        check_sample(hA, hB, hC, N, &max_rel, &n_exact, &n_checked);
        const char *verdict = (n_exact == n_checked) ? "bitwise exact"
                            : (max_rel < 1e-12)      ? "OK (rounding differs)" : "FAIL";
        const double kmed = median(kms);
        printf("%-26s %11.3f ms  %10.1f   %s (%d/%d exact, max rel err %.1e)  [min %.3f max %.3f, wall %.3f ms] %s\n",
               name, kmed, flops / (kmed / 1e3) / 1e9, verdict, n_exact, n_checked, max_rel,
               *std::min_element(kms.begin(), kms.end()),
               *std::max_element(kms.begin(), kms.end()), wall, cfg);
    };

    // ---- 1. naive: one work item per C element -----------------------------
    auto naive = [&](size_t wgx, size_t wgy) {
        const size_t gx = ((N + wgx - 1) / wgx) * wgx;
        const size_t gy = ((N + wgy - 1) / wgy) * wgy;
        return q.parallel_for<k_mat_naive>(nd_range<2>(range<2>(gy, gx), range<2>(wgy, wgx)), [=](nd_item<2> it) {
            const int col = (int)it.get_global_id(1);
            const int row = (int)it.get_global_id(0);
            if (row >= N || col >= N) return;
            double sum = 0.0;
            for (int k = 0; k < N; ++k)
                sum += dA[(size_t)row * N + k] * dB[(size_t)k * N + col];
            dC[(size_t)row * N + col] = sum;
        });
    };
    static char cfg[5][96];
    auto cfg_naive = [&](int i, size_t wx, size_t wy) {
        snprintf(cfg[i], sizeof cfg[i], "{groups %zu, wg %zu, per_item 1}",
                 (size_t)((N + wx - 1) / wx) * (size_t)((N + wy - 1) / wy), wx * wy);
    };
    cfg_naive(0, 32, 8); cfg_naive(1, 128, 1);
    snprintf(cfg[2], sizeof cfg[2], "{groups %zu, wg %d, per_item 1}",
             (size_t)((N + TILE - 1) / TILE) * (size_t)((N + TILE - 1) / TILE), TILE * TILE);
    snprintf(cfg[3], sizeof cfg[3], "{groups %zu, wg 256, per_item 16}",
             (size_t)((N + 63) / 64) * (size_t)((N + 63) / 64));
    snprintf(cfg[4], sizeof cfg[4], "{groups %zu, wg 256, per_item 64}",
             (size_t)((N + 127) / 128) * (size_t)((N + 127) / 128));

    run("naive (32x8 threads)",    cfg[0], [&] { return naive(32, 8); });
    run("naive (128x1, OMP-like)", cfg[1], [&] { return naive(128, 1); });

    // ---- 2. tiled32: 32x32 tiles in local memory ---------------------------
    run("tiled32 (your original)", cfg[2], [&] {
        const size_t g = ((N + TILE - 1) / TILE) * (size_t)TILE;
        return q.submit([&](handler &h) {
            local_accessor<double, 2> sA(range<2>(TILE, TILE), h);
            local_accessor<double, 2> sB(range<2>(TILE, TILE), h);
            h.parallel_for<k_mat_tiled32>(nd_range<2>(range<2>(g, g), range<2>(TILE, TILE)), [=](nd_item<2> it) {
                const int ty = (int)it.get_local_id(0), tx = (int)it.get_local_id(1);
                const int row = (int)it.get_group(0) * TILE + ty;
                const int col = (int)it.get_group(1) * TILE + tx;
                double value = 0.0;
                for (int t = 0; t < N; t += TILE) {
                    sA[ty][tx] = (row < N && t + tx < N) ? dA[(size_t)row * N + t + tx] : 0.0;
                    sB[ty][tx] = (col < N && t + ty < N) ? dB[(size_t)(t + ty) * N + col] : 0.0;
                    it.barrier(access::fence_space::local_space);
                    for (int i = 0; i < TILE; ++i) value += sA[ty][i] * sB[i][tx];
                    it.barrier(access::fence_space::local_space);
                }
                if (row < N && col < N) dC[(size_t)row * N + col] = value;
            });
        });
    });

    // ---- 3. register-tiled: a BM x BN tile of C per work-group -------------
    // Identical to dgemm_regtile in mat-cuda.cu: the same tile constants, the
    // same strided ownership, the same loop order; acc lives in private memory.
    auto regtile = [&](auto BMc, auto BNc, auto BKc, auto TMc, auto TNc) {
        constexpr int BM = decltype(BMc)::value, BN = decltype(BNc)::value;
        constexpr int BK = decltype(BKc)::value, TM = decltype(TMc)::value, TN = decltype(TNc)::value;
        constexpr int RX = BN / TN, RY = BM / TM, NT = RX * RY;
        const size_t gx = (size_t)((N + BN - 1) / BN) * NT;
        return q.submit([&](handler &h) {
            local_accessor<double, 2> sA(range<2>(BM, BK + 1), h);
            local_accessor<double, 2> sB(range<2>(BK, BN), h);
            h.parallel_for<k_mat_regtile<BM, BN, BK, TM, TN>>(nd_range<2>(range<2>((size_t)((N + BM - 1) / BM), gx), range<2>(1, NT)),
                           [=](nd_item<2> it) {
                const int tid  = (int)it.get_local_id(1);
                const int tx   = tid % RX, ty = tid / RX;
                const int row0 = (int)it.get_group(0) * BM;
                const int col0 = (int)it.get_group(1) * BN;

                double acc[TM][TN];
                for (int i = 0; i < TM; ++i)
                    for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

                for (int t = 0; t < N; t += BK) {
                    for (int p = 0; p < (BM * BK) / NT; ++p) {
                        const int e = tid + p * NT, m = e / BK, k = e % BK;
                        const int r = row0 + m, c = t + k;
                        sA[m][k] = (r < N && c < N) ? dA[(size_t)r * N + c] : 0.0;
                    }
                    for (int p = 0; p < (BK * BN) / NT; ++p) {
                        const int e = tid + p * NT, k = e / BN, n = e % BN;
                        const int r = t + k, c = col0 + n;
                        sB[k][n] = (r < N && c < N) ? dB[(size_t)r * N + c] : 0.0;
                    }
                    it.barrier(access::fence_space::local_space);

                    for (int k = 0; k < BK; ++k) {
                        double a[TM], b[TN];
                        for (int i = 0; i < TM; ++i) a[i] = sA[ty + i * RY][k];
                        for (int j = 0; j < TN; ++j) b[j] = sB[k][tx + j * RX];
                        for (int i = 0; i < TM; ++i)
                            for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
                    }
                    it.barrier(access::fence_space::local_space);
                }

                for (int i = 0; i < TM; ++i) {
                    const int r = row0 + ty + i * RY;
                    for (int j = 0; j < TN; ++j) {
                        const int c = col0 + tx + j * RX;
                        if (r < N && c < N) dC[(size_t)r * N + c] = acc[i][j];
                    }
                }
            });
        });
    };
    run("regtile 64x64x16, 4x4",  cfg[3], [&] {
        return regtile(std::integral_constant<int, 64>{},  std::integral_constant<int, 64>{},
                       std::integral_constant<int, 16>{},  std::integral_constant<int, 4>{},
                       std::integral_constant<int, 4>{});
    });
    run("regtile 128x128x8, 8x8", cfg[4], [&] {
        return regtile(std::integral_constant<int, 128>{}, std::integral_constant<int, 128>{},
                       std::integral_constant<int, 8>{},   std::integral_constant<int, 8>{},
                       std::integral_constant<int, 8>{});
    });

    {   // launch overhead: empty kernel, submit + wait, mean of L calls
        const int L = 1000;
        auto empty = [] {};                          // one lambda -> one kernel name
        q.single_task<k_empty>(empty).wait();
        const auto e0 = std::chrono::steady_clock::now();
        for (int i = 0; i < L; ++i) q.single_task<k_empty>(empty).wait();
        printf("%-26s %11.2f us  (empty kernel, submit + wait, mean of %d)\n", "launch overhead",
               std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now() - e0).count() / L, L);
    }

    free(dA, q); free(dB, q); free(dC, q);
    free(hA); free(hB); free(hC);
    return 0;
}
