//-*-c++-*-
// version-2.cpp — P2P bandwidth, 2 simultaneous transfers on 2 devices:
//   d_C (dev1) <- d_A (dev0)   expect 1.0
//   d_D (dev0) <- d_B (dev1)   expect 2.0
//
// Two host threads, one queue each. Bytes moved = 2 x N x sizeof(double).
#include "p2p_common.hpp"

using std::cout;
using std::cerr;
using std::fixed;
using std::setprecision;
#include <omp.h>

int main(int argc, char **argv) try {
    size_t N     = p2p::parse_n(argc, argv);
    size_t bytes = N * sizeof(double);

    p2p::Ctx c = p2p::setup(2);
    p2p::enable_peer_access(c);

    double *d_A = sycl::malloc_device<double>(N, c.dev[0], c.ctx);  // src 1.0
    double *d_D = sycl::malloc_device<double>(N, c.dev[0], c.ctx);  // dst <- dev1
    double *d_B = sycl::malloc_device<double>(N, c.dev[1], c.ctx);  // src 2.0
    double *d_C = sycl::malloc_device<double>(N, c.dev[1], c.ctx);  // dst <- dev0
    if (!d_A || !d_B || !d_C || !d_D) p2p::die("device allocation failed");

    {
        std::vector<double> h(N);
        std::fill(h.begin(), h.end(), 1.0); c.q[0].memcpy(d_A, h.data(), bytes).wait();
        std::fill(h.begin(), h.end(), 4.0); c.q[0].memcpy(d_D, h.data(), bytes).wait();
        std::fill(h.begin(), h.end(), 2.0); c.q[1].memcpy(d_B, h.data(), bytes).wait();
        std::fill(h.begin(), h.end(), 3.0); c.q[1].memcpy(d_C, h.data(), bytes).wait();
    }

    // Warm-up (not timed)
    for (int w = 0; w < 2; ++w) {
        c.q[0].memcpy(d_C, d_A, bytes).wait();
        c.q[1].memcpy(d_D, d_B, bytes).wait();
    }

    // Timed: both directions at once, one host thread driving each queue.
    // No device selection needed — a queue already knows its device, so the
    // threads cannot interfere with each other.
    omp_set_num_threads(2);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        { c.q[0].memcpy(d_C, d_A, bytes).wait(); }   // dev0 -> dev1
        #pragma omp section
        { c.q[1].memcpy(d_D, d_B, bytes).wait(); }   // dev1 -> dev0
    }
    double end = omp_get_wtime();

    // ── Timer sanity check ───────────────────────────────────────────────────
    if (end <= start) {
        cerr << "Warning: timer returned non-positive elapsed time, "
             << "results may be unreliable\n";
    }

    // ── Bandwidth calculation ────────────────────────────────────────────────
    double seconds       = end - start;
    double total_bytes   = 2.0 * static_cast<double>(bytes); // 2 P2P transfers
    double bandwidthGBps = total_bytes / (seconds * 1e9);

    cout << fixed << setprecision(6);
    cout << "Elapsed time:      " << seconds              << " s\n";
    cout << "Total transferred: " << total_bytes / 1e9     << " GB\n";
    cout << "P2P Bandwidth:     " << bandwidthGBps         << " GB/s\n";

    p2p::verify(c.q[1], d_C, N, 1.0, "d_C (dev0->dev1)");
    p2p::verify(c.q[0], d_D, N, 2.0, "d_D (dev1->dev0)");

    sycl::free(d_A, c.ctx); sycl::free(d_D, c.ctx);
    sycl::free(d_B, c.ctx); sycl::free(d_C, c.ctx);
    return EXIT_SUCCESS;
}
catch (const sycl::exception &e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
    return EXIT_FAILURE;
}
