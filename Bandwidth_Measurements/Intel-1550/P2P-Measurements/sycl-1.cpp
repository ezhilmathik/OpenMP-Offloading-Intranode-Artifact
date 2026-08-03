//-*-c++-*-
// version-1.cpp — P2P bandwidth, 1 transfer: dev0 -> dev1.
//
// One host thread, one queue. Bytes moved = 1 x N x sizeof(double).
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

    // Host staging: fill the source, then release. Only `bytes` of host
    // memory is live at once, so N is bounded by device HBM, not host RAM.
    {
        std::vector<double> h(N, 1.0);
        double *d_A = sycl::malloc_device<double>(N, c.dev[0], c.ctx);
        double *d_B = sycl::malloc_device<double>(N, c.dev[1], c.ctx);
        if (!d_A || !d_B) p2p::die("device allocation failed");

        c.q[0].memcpy(d_A, h.data(), bytes).wait();   // src = 1.0
        std::fill(h.begin(), h.end(), 2.0);
        c.q[1].memcpy(d_B, h.data(), bytes).wait();   // dst = 2.0, overwritten

        // Warm-up (not timed). Peer copy: same context, so a plain memcpy
        // between two device pointers IS the peer transfer.
        for (int w = 0; w < 2; ++w)
            c.q[0].memcpy(d_B, d_A, bytes).wait();

        // Timed: dev0 -> dev1
        double start = omp_get_wtime();
        c.q[0].memcpy(d_B, d_A, bytes).wait();
        double end = omp_get_wtime();

        // ── Timer sanity check ───────────────────────────────────────────────────
        if (end <= start) {
            cerr << "Warning: timer returned non-positive elapsed time, "
                 << "results may be unreliable\n";
        }

        // ── Bandwidth calculation ────────────────────────────────────────────────
        double seconds       = end - start;
        double total_bytes   = static_cast<double>(bytes); // 1 P2P transfer
        double bandwidthGBps = total_bytes / (seconds * 1e9);

        cout << fixed << setprecision(6);
        cout << "Elapsed time:      " << seconds              << " s\n";
        cout << "Total transferred: " << total_bytes / 1e9     << " GB\n";
        cout << "P2P Bandwidth:     " << bandwidthGBps         << " GB/s\n";
        p2p::verify(c.q[1], d_B, N, 1.0, "d_B (dev0->dev1)");

        sycl::free(d_A, c.ctx);
        sycl::free(d_B, c.ctx);
    }
    return EXIT_SUCCESS;
}
catch (const sycl::exception &e) {
    std::cerr << "SYCL exception: " << e.what() << "\n";
    return EXIT_FAILURE;
}
