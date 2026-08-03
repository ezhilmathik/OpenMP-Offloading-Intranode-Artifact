//-*-c++-*-
// p2p_common.hpp — shared setup for the SYCL P2P bandwidth benchmarks.
//
// The one rule that makes SYCL P2P work: every device pointer involved in a
// peer copy must belong to the SAME sycl::context.  Build one context over all
// the devices up front, allocate against it, and queue.memcpy() does the peer
// transfer directly.  Separate contexts -> the runtime cannot do a direct copy
// and will either throw or silently stage through host memory, which would
// quietly turn a "P2P bandwidth" number into a host round-trip.
#pragma once
#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <vector>

namespace p2p {

inline void die(const std::string &msg) {
    std::cerr << "Error: " << msg << "\n";
    std::exit(EXIT_FAILURE);
}

// Parse and validate the vector length argument.
inline size_t parse_n(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <vector_size>\n";
        std::exit(EXIT_FAILURE);
    }
    long long val = 0;
    try {
        val = std::stoll(argv[1]);
        if (val <= 0) throw std::out_of_range("must be positive");
    } catch (const std::exception &e) {
        die(std::string("invalid vector_size '") + argv[1] + "' (" + e.what() + ")");
    }
    size_t N = static_cast<size_t>(val);
    if (N > (static_cast<size_t>(-1) / sizeof(double)))
        die("vector_size too large, would overflow size_t");
    return N;
}

struct Ctx {
    std::vector<sycl::device> dev;
    sycl::context             ctx;   // one context spanning every device
    std::vector<sycl::queue>  q;     // one in-order queue per device
};

// Grab `needed` GPUs, put them in a single shared context, one queue each.
inline Ctx setup(int needed) {
    auto all = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (static_cast<int>(all.size()) < needed) {
        die("need at least " + std::to_string(needed) + " GPU devices, found " +
            std::to_string(all.size()) +
            ". On Intel GPUs try ZE_FLAT_DEVICE_HIERARCHY=FLAT to expose each "
            "stack as its own device.");
    }
    Ctx c;
    c.dev.assign(all.begin(), all.begin() + needed);
    c.ctx = sycl::context(c.dev);
    for (auto &d : c.dev)
        c.q.emplace_back(c.ctx, d, sycl::property::queue::in_order{});

    std::cerr << "Devices:\n";
    for (int i = 0; i < needed; ++i)
        std::cerr << "  [" << i << "] "
                  << c.dev[i].get_info<sycl::info::device::name>() << "\n";
    return c;
}

// Enable peer access both ways for every pair. Reports what is actually
// supported — if this prints warnings, any "P2P" number is really a staged
// copy through the host, so check it before trusting the bandwidth.
inline void enable_peer_access(Ctx &c) {
    const int n = static_cast<int>(c.dev.size());
    bool all_ok = true;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            bool ok = c.dev[i].ext_oneapi_can_access_peer(
                c.dev[j], sycl::ext::oneapi::peer_access::access_supported);
            if (!ok) {
                std::cerr << "Warning: peer access NOT supported, device " << i
                          << " -> device " << j << "\n";
                all_ok = false;
                continue;
            }
            try {
                c.dev[i].ext_oneapi_enable_peer_access(c.dev[j]);
            } catch (const sycl::exception &) {
                // Already enabled — harmless, the CUDA original swallowed
                // cudaErrorPeerAccessAlreadyEnabled (704) the same way.
            }
        }
    }
    std::cerr << "Peer access: " << (all_ok ? "all pairs enabled"
                                            : "INCOMPLETE — see warnings above")
              << "\n";
}

// Spot-check a destination buffer against its expected fill value.
inline bool verify(sycl::queue &q, const double *d_ptr, size_t N,
                   double expected, const char *label) {
    size_t sample = std::min(N, static_cast<size_t>(1000));
    std::vector<double> h(sample);
    q.memcpy(h.data(), d_ptr, sample * sizeof(double)).wait();

    bool pass = true;
    for (size_t i = 0; i < sample; ++i) {
        if (h[i] != expected) {
            std::cerr << label << " mismatch at index " << i << ": expected "
                      << expected << ", got " << h[i] << "\n";
            pass = false;
            break;
        }
    }
    std::cout << "Verification " << label << ": " << (pass ? "PASSED" : "FAILED")
              << " (" << sample << "/" << N << " samples)\n";
    return pass;
}

}  // namespace p2p
