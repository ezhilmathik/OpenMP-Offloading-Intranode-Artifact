// vec-sycl.cpp -- vector addition C = A + B (double precision), SYCL
//
// Build (oneAPI):
//   icpx -fsycl -O3 -ffp-contract=off -fno-fast-math -fiopenmp \
//        -fsycl-targets=spir64_gen -Xs "-device pvc" \
//        -o vecadd_sycl vec-sycl.cpp
//   (plain JIT also works: drop -fsycl-targets and -Xs)
//
// Run:   ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0 ./vecadd_sycl <N> [reps=5]
//
// Counterpart of vec-omp.cpp on Intel GPUs, and of vec-cuda.cu / vec-hip.cpp on
// the other two vendors: same N, same values (A = 1, B = i), same timing (one
// untimed warm-up call, then the median of `reps` calls), and every element
// checked against the CPU.
//
//   vadd_range      basic parallel_for over a 1-D range: the work-group shape is
//                   left to the runtime. The direct counterpart of the OpenMP
//                   "teams distribute parallel for" loop.
//   vadd_nd         nd_range with an explicit work-group of 256, the counterpart
//                   of the CUDA/HIP "one thread per element, 256 per block".
//   vadd_vec2       128-bit accesses through sycl::vec<double,2>, two per work
//                   item: the counterpart of the double2 kernel.
//   vadd_stride     the OpenMP runtime's layout: a fixed number of work-groups,
//                   each work item striding through the vector (reference).
//
// Kernel time comes from SYCL event profiling (device timestamps, in the same
// sense as the Nsight Systems / rocprofv3 traces used on the other machines);
// the wall time per call (submit + wait) is reported next to it.
//
// Vector addition moves 24 bytes per element and does one addition, so it is
// limited by memory bandwidth: the figure of merit is GB/s. A double addition
// is correctly rounded everywhere, so the check demands bitwise equality.

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cstring>

using namespace sycl;

// Named kernels: profilers (VTune) report these names instead of lambda numbers.
class k_vadd_range; class k_vadd_nd; class k_vadd_vec2; class k_vadd_stride; class k_empty;

static constexpr size_t WG   = 256;    // work-group size (like a CUDA block)
static constexpr size_t GRPS = 3200;   // work-groups for the OpenMP-layout kernel

// device time of one call, in ms, from the event's profiling timestamps
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
    if (argc < 2) { fprintf(stderr, "Usage: %s <N elements> [reps=5]\n", argv[0]); return 1; }
    const size_t N    = strtoull(argv[1], nullptr, 10);
    const int    reps = argc > 2 ? atoi(argv[2]) : 5;
    if (N == 0 || reps <= 0) { fprintf(stderr, "N and reps must be positive\n"); return 1; }

    const size_t bytes = N * sizeof(double);
    const double moved = 3.0 * (double)bytes;          // A and B read, C written

    queue q{gpu_selector_v, property::queue::enable_profiling{}};
    const device dev = q.get_device();

    // The device query for the memory clock is not reliable here, so the peak is
    // taken from PEAK_BW_GBS (about 1638 GB/s for one Max 1550 tile) when set.
    const char *pk = getenv("PEAK_BW_GBS");
    const double peak = pk ? atof(pk) : 0.0;
    printf("GPU : %s (%u EUs, %zu GB), theoretical DRAM bandwidth %.0f GB/s\n",
           dev.get_info<info::device::name>().c_str(),
           dev.get_info<info::device::max_compute_units>(),
           (size_t)(dev.get_info<info::device::global_mem_size>() / (1ull << 30)), peak);
    printf("N = %zu elements, each vector %.3f GB, reps = %d, %.3f GB moved per call\n",
           N, bytes / 1e9, reps, moved / 1e9);

    double *hA = (double *)malloc(bytes);
    double *hB = (double *)malloc(bytes);
    double *hC = (double *)malloc(bytes);
    if (!hA || !hB || !hC) { fprintf(stderr, "host malloc failed\n"); return 1; }
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; ++i) { hA[i] = 1.0; hB[i] = (double)i; }

    double *dA = malloc_device<double>(N, q);
    double *dB = malloc_device<double>(N, q);
    double *dC = malloc_device<double>(N, q);
    if (!dA || !dB || !dC) { fprintf(stderr, "malloc_device failed\n"); return 1; }

    auto t0 = std::chrono::steady_clock::now();
    q.memcpy(dA, hA, bytes).wait();
    q.memcpy(dB, hB, bytes).wait();
    const double h2d_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0).count();
    printf("%-26s %11.3f ms  (A and B, not included in kernel times)\n", "H2D copy", h2d_ms);
    printf("%-26s %11s     %10s   %s\n", "kernel", "time/call", "GB/s", "check vs CPU (all elements)");

    // One untimed warm-up call, then `reps` calls; kernel time from the event,
    // wall time from submit to completion.
    auto run = [&](const char *name, const char *cfg, auto submit) {
        q.memset(dC, 0xFF, bytes).wait();               // NaN: unwritten entries fail the check
        submit().wait();                                 // warm-up (JIT, first touch)
        std::vector<double> kms;
        const auto w0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            event e = submit();
            e.wait();
            kms.push_back(event_ms(e));
        }
        const double wall = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - w0).count() / reps;
        q.memcpy(hC, dC, bytes).wait();
        size_t bad = 0;
        #pragma omp parallel for reduction(+:bad) schedule(static)
        for (long long i = 0; i < (long long)N; ++i)
            if (!(hC[i] == hA[i] + hB[i])) ++bad;
        const double kmed = median(kms);
        printf("%-26s %11.4f ms  %10.1f   %s (%zu of %zu wrong)  [min %.4f max %.4f, wall %.4f ms] %s\n",
               name, kmed, moved / (kmed / 1e3) / 1e9, bad ? "FAIL" : "bitwise exact", bad, N,
               *std::min_element(kms.begin(), kms.end()),
               *std::max_element(kms.begin(), kms.end()), wall, cfg);
    };

    static char cfg[5][96];
    snprintf(cfg[0], sizeof cfg[0], "{groups %zu, wg %zu, per_item 1}", (N + WG - 1) / WG, WG);
    snprintf(cfg[1], sizeof cfg[1], "{groups %zu, wg %zu, per_item 1}", (N + WG - 1) / WG, WG);
    snprintf(cfg[2], sizeof cfg[2], "{groups %zu, wg %zu, per_item 4}",
             std::max<size_t>(1, (N / 2 + 2 * WG - 1) / (2 * WG)), WG);
    snprintf(cfg[3], sizeof cfg[3], "{groups %zu, wg %zu, per_item %.0f}",
             std::min<size_t>(GRPS, std::max<size_t>(1, (N + WG - 1) / WG)), WG,
             (double)N / (std::min<size_t>(GRPS, std::max<size_t>(1, (N + WG - 1) / WG)) * WG));
    run("vadd range (runtime wg)", cfg[0], [&] {
        return q.parallel_for<k_vadd_range>(range<1>(N), [=](id<1> i) { dC[i] = dA[i] + dB[i]; });
    });

    run("vadd nd_range (256/wg)", cfg[1], [&] {
        const size_t global = ((N + WG - 1) / WG) * WG;
        return q.parallel_for<k_vadd_nd>(nd_range<1>(range<1>(global), range<1>(WG)), [=](nd_item<1> it) {
            const size_t i = it.get_global_id(0);
            if (i < N) dC[i] = dA[i] + dB[i];
        });
    });

    run("vadd vectorized (vec2)", cfg[2], [&] {
        const size_t n2 = N / 2;
        const size_t groups = std::max<size_t>(1, (n2 + 2 * WG - 1) / (2 * WG));
        return q.parallel_for<k_vadd_vec2>(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
            using d2 = vec<double, 2>;
            const d2 *A2 = reinterpret_cast<const d2 *>(dA);
            const d2 *B2 = reinterpret_cast<const d2 *>(dB);
            d2 *C2 = reinterpret_cast<d2 *>(dC);
            const size_t base = it.get_group(0) * WG * 2 + it.get_local_id(0);
            for (int u = 0; u < 2; ++u) {
                const size_t j = base + (size_t)u * WG;
                if (j < n2) C2[j] = A2[j] + B2[j];
            }
            if ((N & 1) && it.get_global_id(0) == 0) dC[N - 1] = dA[N - 1] + dB[N - 1];
        });
    });

    run("vadd stride (OMP layout)", cfg[3], [&] {
        const size_t groups = std::min<size_t>(GRPS, std::max<size_t>(1, (N + WG - 1) / WG));
        return q.parallel_for<k_vadd_stride>(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
            const size_t stride = it.get_global_range(0);
            for (size_t i = it.get_global_id(0); i < N; i += stride) dC[i] = dA[i] + dB[i];
        });
    });

    {   // launch overhead: empty kernel, submit + wait, mean of L calls
        const int L = 1000;
        auto empty = [] {};                          // one lambda -> one kernel name
        q.single_task<k_empty>(empty).wait();
        const auto e0 = std::chrono::steady_clock::now();
        for (int i = 0; i < L; ++i) q.single_task<k_empty>(empty).wait();
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - e0).count() / L;
        printf("%-26s %11.2f us  (empty kernel, submit + wait, mean of %d)\n",
               "launch overhead", us, L);
    }

    free(dA, q); free(dB, q); free(dC, q);
    free(hA); free(hB); free(hC);
    return 0;
}
