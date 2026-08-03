//-*-c++-*-
// version-4.cpp — P2P bandwidth, 3 simultaneous transfers into dev0:
//   d_B1 (dev0) <- d_B2 (dev1)   expect 2.0
//   d_C1 (dev0) <- d_C3 (dev2)   expect 3.0
//   d_D1 (dev0) <- d_D4 (dev3)   expect 4.0
//
// Three host threads, one queue each. 4 devices, but only 3 transfers, so
// bytes moved = 3 x N x sizeof(double) — NOT 4x. The reported aggregate is
// bounded by 3 x (slowest link), not by the sum of all link capacities.
#include "p2p_common.hpp"

using std::cout;
using std::cerr;
using std::fixed;
using std::setprecision;
#include <omp.h>

int main(int argc, char **argv) try {
    size_t N     = p2p::parse_n(argc, argv);
    size_t bytes = N * sizeof(double);

    p2p::Ctx c = p2p::setup(4);
    p2p::enable_peer_access(c);

    // Only the 6 buffers actually involved in a transfer are allocated.
    // (The dpct-migrated original allocated 16 device + 16 host buffers, of
    // which 10 were never touched — that capped N at ~1/3 of what fits.)
    double *d_B1 = sycl::malloc_device<double>(N, c.dev[0], c.ctx);  // dst
    double *d_C1 = sycl::malloc_device<double>(N, c.dev[0], c.ctx);  // dst
    double *d_D1 = sycl::malloc_device<double>(N, c.dev[0], c.ctx);  // dst
    double *d_B2 = sycl::malloc_device<double>(N, c.dev[1], c.ctx);  // src 2.0
    double *d_C3 = sycl::malloc_device<double>(N, c.dev[2], c.ctx);  // src 3.0
    double *d_D4 = sycl::malloc_device<double>(N, c.dev[3], c.ctx);  // src 4.0
    if (!d_B1 || !d_C1 || !d_D1 || !d_B2 || !d_C3 || !d_D4)
        p2p::die("device allocation failed");

    {
        std::vector<double> h(N);
        std::fill(h.begin(), h.end(), 1.0);
        c.q[0].memcpy(d_B1, h.data(), bytes).wait();
        c.q[0].memcpy(d_C1, h.data(), bytes).wait();
        c.q[0].memcpy(d_D1, h.data(), bytes).wait();
        std::fill(h.begin(), h.end(), 2.0); c.q[1].memcpy(d_B2, h.data(), bytes).wait();
        std::fill(h.begin(), h.end(), 3.0); c.q[2].memcpy(d_C3, h.data(), bytes).wait();
        std::fill(h.begin(), h.end(), 4.0); c.q[3].memcpy(d_D4, h.data(), bytes).wait();
    }

    // Warm-up (not timed)
    for (int w = 0; w < 2; ++w) {
        c.q[1].memcpy(d_B1, d_B2, bytes).wait();
        c.q[2].memcpy(d_C1, d_C3, bytes).wait();
        c.q[3].memcpy(d_D1, d_D4, bytes).wait();
    }

    // Timed: dev1, dev2, dev3 -> dev0, concurrently. Each section drives the
    // sender's queue, so all three transfers are in flight together.
    omp_set_num_threads(3);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        { c.q[1].memcpy(d_B1, d_B2, bytes).wait(); }
        #pragma omp section
        { c.q[2].memcpy(d_C1, d_C3, bytes).wait(); }
        #pragma omp section
        { c.q[3].memcpy(d_D1, d_D4, bytes).wait(); }
    }
    double end = omp_get_wtime();

    // ── Timer sanity check ───────────────────────────────────────────────────
    if (end <= start) {
        cerr << "Warning: timer returned non-positive elapsed time, "
             << "results may be unreliable\n";
    }

    // ── Bandwidth calculation ────────────────────────────────────────────────
    double seconds       = end - start;
    double total_bytes   = 3.0 * static_cast<double>(bytes); // 3 P2P transfers
    double bandwidthGBps = total_bytes / (seconds * 1e9);

    cout << fixed << setprecision(6);
    cout << "Elapsed time:      " << seconds              << " s\n";
    cout << "Total transferred: " << total_bytes / 1e9     << " GB\n";
    cout << "P2P Bandwidth:     " << bandwidthGBps         << " GB/s\n";

    p2p::verify(c.q[0], d_B1, N, 2.0, "d_B1 (dev1->dev0)");
    p2p::verify(c.q[0], d_C1, N, 3.0, "d_C1 (dev2->dev0)");
    p2p::verify(c.q[0], d_D1, N, 4.0, "d_D1 (dev3->dev0)");

    sycl::free(d_B1, c.ctx); sycl::free(d_C1, c.ctx); sycl::free(d_D1, c.ctx);
    sycl::free(d_B2, c.ctx); sycl::free(d_C3, c.ctx); sycl::free(d_D4, c.ctx);
    return EXIT_SUCCESS;
}
catch (const sycl::exception &e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
    return EXIT_FAILURE;
}
