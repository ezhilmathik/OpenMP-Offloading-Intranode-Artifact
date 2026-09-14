// mv-sycl.cpp -- dense matrix-vector product y = A x  (N x N, row-major, double), SYCL
//
// Build (oneAPI):
//   icpx -fsycl -O3 -ffp-contract=off -fno-fast-math -fiopenmp -o matvec_sycl mv-sycl.cpp
//
// Run:   ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0 \
//        PEAK_BW_GBS=1638 ./matvec_sycl <N> [reps=5]
//
// Counterpart of mv-omp.cpp, mv-cuda.cu and mv-hip.cpp:
//   mv_thread_row   one work item per row, x staged in local memory in chunks of
//                   256. Same mapping as the OpenMP thread-per-row loop: the fair
//                   counterpart of the directive-based version.
//   mv_sg_row       one sub-group per row: its lanes read consecutive columns
//                   (coalesced) and the partial sums are combined with
//                   reduce_over_group. The counterpart of the CUDA warp-per-row
//                   and HIP wavefront-per-row kernels. The sub-group size is
//                   requested explicitly (SG, 32 by default on Xe) so the mapping
//                   is the same on every vendor.
//
// Data: A holds the consecutive integers 1, 2, 3, ... and x = 1, 2, 3, 1, 2, 3, ...
// Every product and partial sum is an integer below 2^53, so y is bitwise exact
// whatever the order of summation, including the sub-group and OpenMP reductions.
//
// Kernel time comes from SYCL event profiling; the wall time per call is shown
// next to it. y = A x reads all of A once: the figure of merit is GB/s.

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <vector>

using namespace sycl;

// Named kernels: profilers (VTune) report these names instead of lambda numbers.
class k_mv_thread_row; class k_mv_sg_row; class k_empty;

static constexpr size_t WG = 256;      // work-group size
static constexpr size_t SG = 32;       // sub-group size (lanes per row in mv_sg_row)

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

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <N> [reps=5]\n", argv[0]); return 1; }
    const size_t N    = strtoull(argv[1], nullptr, 10);
    const int    reps = argc > 2 ? atoi(argv[2]) : 5;
    if (N == 0 || reps <= 0) { fprintf(stderr, "N and reps must be positive\n"); return 1; }

    const size_t bytesA = N * N * sizeof(double);
    const size_t bytesV = N * sizeof(double);
    const double moved  = (double)bytesA + 2.0 * (double)bytesV;

    queue q{gpu_selector_v, property::queue::enable_profiling{}};
    const device dev = q.get_device();
    const char *pk = getenv("PEAK_BW_GBS");
    const double peak = pk ? atof(pk) : 0.0;
    printf("GPU : %s (%u EUs, %zu GB), theoretical DRAM bandwidth %.0f GB/s\n",
           dev.get_info<info::device::name>().c_str(),
           dev.get_info<info::device::max_compute_units>(),
           (size_t)(dev.get_info<info::device::global_mem_size>() / (1ull << 30)), peak);
    printf("N = %zu, A = %.3f GB, reps = %d, %.3f GB moved per call\n",
           N, bytesA / 1e9, reps, moved / 1e9);

    double *hA   = (double *)malloc(bytesA);
    double *hx   = (double *)malloc(bytesV);
    double *hy   = (double *)malloc(bytesV);
    double *href = (double *)malloc(bytesV);
    if (!hA || !hx || !hy || !href) { fprintf(stderr, "host malloc failed\n"); return 1; }
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; ++i)
        for (size_t j = 0; j < N; ++j)
            hA[(size_t)i * N + j] = (double)((size_t)i * N + j + 1);
    for (size_t j = 0; j < N; ++j) hx[j] = (double)(1 + j % 3);

    #pragma omp parallel for schedule(static)                       // CPU reference (exact)
    for (long long i = 0; i < (long long)N; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < N; ++j) s += hA[(size_t)i * N + j] * hx[j];
        href[i] = s;
    }

    double *dA = malloc_device<double>(N * N, q);
    double *dx = malloc_device<double>(N, q);
    double *dy = malloc_device<double>(N, q);
    if (!dA || !dx || !dy) { fprintf(stderr, "malloc_device failed\n"); return 1; }

    auto t0 = std::chrono::steady_clock::now();
    q.memcpy(dA, hA, bytesA).wait();
    q.memcpy(dx, hx, bytesV).wait();
    printf("%-26s %11.3f ms  (A and x, not included in kernel times)\n", "H2D copy",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    printf("%-26s %11s     %10s   %s\n", "kernel", "time/call", "GB/s", "check vs CPU (all of y)");

    auto run = [&](const char *name, const char *cfg, auto submit) {
        q.memset(dy, 0xFF, bytesV).wait();               // NaN: unwritten entries fail the check
        submit().wait();                                  // warm-up
        std::vector<double> kms;
        const auto w0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) { event e = submit(); e.wait(); kms.push_back(event_ms(e)); }
        const double wall = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - w0).count() / reps;
        q.memcpy(hy, dy, bytesV).wait();
        size_t bad = 0;
        for (size_t i = 0; i < N; ++i) if (!(hy[i] == href[i])) ++bad;
        const double kmed = median(kms);
        printf("%-26s %11.4f ms  %10.1f   %s (%zu of %zu wrong)  [min %.4f max %.4f, wall %.4f ms] %s\n",
               name, kmed, moved / (kmed / 1e3) / 1e9, bad ? "FAIL" : "bitwise exact", bad, N,
               *std::min_element(kms.begin(), kms.end()),
               *std::max_element(kms.begin(), kms.end()), wall, cfg);
    };

    static char cfg[2][96];
    snprintf(cfg[0], sizeof cfg[0], "{groups %zu, wg %zu, per_item 1}", (N + WG - 1) / WG, WG);
    snprintf(cfg[1], sizeof cfg[1], "{groups %zu, wg %zu, sg %zu, rows/wg %zu}",
             (N + WG / SG - 1) / (WG / SG), WG, SG, WG / SG);

    // One work item per row; the work-group stages a chunk of x in local memory.
    run("mv thread/row", cfg[0], [&] {
        const size_t global = ((N + WG - 1) / WG) * WG;
        return q.submit([&](handler &h) {
            local_accessor<double, 1> xs(range<1>(WG), h);
            h.parallel_for<k_mv_thread_row>(nd_range<1>(range<1>(global), range<1>(WG)), [=](nd_item<1> it) {
                const size_t row = it.get_global_id(0);
                const size_t lid = it.get_local_id(0);
                double temp = 0.0;
                for (size_t blk = 0; blk < (N + WG - 1) / WG; ++blk) {
                    const size_t vi = blk * WG + lid;
                    xs[lid] = (vi < N) ? dx[vi] : 0.0;
                    it.barrier(access::fence_space::local_space);
                    for (size_t i = 0; i < WG; ++i) {
                        const size_t col = blk * WG + i;
                        if (row < N && col < N) temp += dA[row * N + col] * xs[i];
                    }
                    it.barrier(access::fence_space::local_space);
                }
                if (row < N) dy[row] = temp;
            });
        });
    });

    // One sub-group per row: coalesced reads, two accumulators, sub-group reduction.
    run("mv sub-group/row", cfg[1], [&] {
        const size_t rows_per_wg = WG / SG;
        const size_t groups = (N + rows_per_wg - 1) / rows_per_wg;
        return q.parallel_for<k_mv_sg_row>(nd_range<1>(range<1>(groups * WG), range<1>(WG)),
                              [=](nd_item<1> it) [[intel::reqd_sub_group_size(SG)]] {
            const sub_group sg = it.get_sub_group();
            const size_t lane  = sg.get_local_linear_id();
            const size_t row   = it.get_global_id(0) / SG;
            if (row >= N) return;
            const double *a = dA + row * N;
            double s0 = 0.0, s1 = 0.0;
            size_t j = lane;
            for (; j + SG < N; j += 2 * SG) {
                s0 += a[j]      * dx[j];
                s1 += a[j + SG] * dx[j + SG];
            }
            if (j < N) s0 += a[j] * dx[j];
            const double sum = reduce_over_group(sg, s0 + s1, plus<double>());
            if (lane == 0) dy[row] = sum;
        });
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

    free(dA, q); free(dx, q); free(dy, q);
    free(hA); free(hx); free(hy); free(href);
    return 0;
}
