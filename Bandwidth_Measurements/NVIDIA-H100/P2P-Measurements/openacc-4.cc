#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdint>
#include <openacc.h>
#include <omp.h>
#include <cuda.h>

using namespace std;

int main(int argc, char *argv[]) {

    // ── Argument validation ──────────────────────────────────────────────────
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <vector_size>\n";
        return EXIT_FAILURE;
    }

    size_t N = 0;
    try {
        long long val = stoll(argv[1]);
        if (val <= 0) throw out_of_range("must be positive");
        N = static_cast<size_t>(val);
    } catch (const exception &e) {
        cerr << "Error: invalid vector_size '" << argv[1]
             << "' (" << e.what() << ")\n";
        return EXIT_FAILURE;
    }

    // ── Overflow guard ───────────────────────────────────────────────────────
    if (N > (SIZE_MAX / sizeof(double))) {
        cerr << "Error: vector_size too large, would overflow size_t\n";
        return EXIT_FAILURE;
    }
    size_t bytes = N * sizeof(double);

    // ── Device check ─────────────────────────────────────────────────────────
    int num_devices = acc_get_num_devices(acc_device_nvidia);
    if (num_devices < 4) {
        cerr << "Error: need at least 4 NVIDIA OpenACC devices, found "
             << num_devices << "\n";
        return EXIT_FAILURE;
    }

    // ── Enable P2P access between all device pairs ────────────────────────────
    for (int i = 0; i < 4; ++i) {
        cudaSetDevice(i);
        for (int j = 0; j < 4; ++j) {
            if (i == j) continue;
            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            if (canAccess)
                cudaDeviceEnablePeerAccess(j, 0);
            else
                cerr << "Warning: P2P not supported device " << i
                     << " -> device " << j << "\n";
        }
    }

    // ── Host allocation ──────────────────────────────────────────────────────
    double *h_A1 = (double*) malloc(bytes); double *h_A2 = (double*) malloc(bytes);
    double *h_A3 = (double*) malloc(bytes); double *h_A4 = (double*) malloc(bytes);
    double *h_B1 = (double*) malloc(bytes); double *h_B2 = (double*) malloc(bytes);
    double *h_B3 = (double*) malloc(bytes); double *h_B4 = (double*) malloc(bytes);
    double *h_C1 = (double*) malloc(bytes); double *h_C2 = (double*) malloc(bytes);
    double *h_C3 = (double*) malloc(bytes); double *h_C4 = (double*) malloc(bytes);
    double *h_D1 = (double*) malloc(bytes); double *h_D2 = (double*) malloc(bytes);
    double *h_D3 = (double*) malloc(bytes); double *h_D4 = (double*) malloc(bytes);

    if (!h_A1 || !h_A2 || !h_A3 || !h_A4 ||
        !h_B1 || !h_B2 || !h_B3 || !h_B4 ||
        !h_C1 || !h_C2 || !h_C3 || !h_C4 ||
        !h_D1 || !h_D2 || !h_D3 || !h_D4) {
        cerr << "Error: malloc failed for host buffers\n";
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i) {
        h_A1[i] = h_B1[i] = h_C1[i] = h_D1[i] = 1.0;
        h_A2[i] = h_B2[i] = h_C2[i] = h_D2[i] = 2.0;
        h_A3[i] = h_B3[i] = h_C3[i] = h_D3[i] = 3.0;
        h_A4[i] = h_B4[i] = h_C4[i] = h_D4[i] = 4.0;
    }

    // ── Device allocation + H→D population ───────────────────────────────────
    // Transfer map (all destinations are on device 0):
    //   d_B1 (dev0) ← d_B2 (dev1): expect 2.0
    //   d_C1 (dev0) ← d_C3 (dev2): expect 3.0
    //   d_D1 (dev0) ← d_D4 (dev3): expect 4.0

    acc_set_device_num(0, acc_device_nvidia);
    double *d_A1 = (double*) acc_malloc(bytes);
    double *d_B1 = (double*) acc_malloc(bytes); // dst: receives from dev1 (2.0)
    double *d_C1 = (double*) acc_malloc(bytes); // dst: receives from dev2 (3.0)
    double *d_D1 = (double*) acc_malloc(bytes); // dst: receives from dev3 (4.0)
    if (!d_A1 || !d_B1 || !d_C1 || !d_D1) {
        cerr << "Error: acc_malloc failed on device 0\n";
        free(h_A1); free(h_B1); free(h_C1); free(h_D1);
        free(h_A2); free(h_B2); free(h_C2); free(h_D2);
        free(h_A3); free(h_B3); free(h_C3); free(h_D3);
        free(h_A4); free(h_B4); free(h_C4); free(h_D4);
        return EXIT_FAILURE;
    }
    acc_memcpy_to_device(d_A1, h_A1, bytes);
    acc_memcpy_to_device(d_B1, h_B1, bytes);
    acc_memcpy_to_device(d_C1, h_C1, bytes);
    acc_memcpy_to_device(d_D1, h_D1, bytes);

    acc_set_device_num(1, acc_device_nvidia);
    double *d_A2 = (double*) acc_malloc(bytes);
    double *d_B2 = (double*) acc_malloc(bytes); // src: value 2.0
    double *d_C2 = (double*) acc_malloc(bytes);
    double *d_D2 = (double*) acc_malloc(bytes);
    if (!d_A2 || !d_B2 || !d_C2 || !d_D2) {
        cerr << "Error: acc_malloc failed on device 1\n";
        return EXIT_FAILURE;
    }
    acc_memcpy_to_device(d_A2, h_A2, bytes);
    acc_memcpy_to_device(d_B2, h_B2, bytes);
    acc_memcpy_to_device(d_C2, h_C2, bytes);
    acc_memcpy_to_device(d_D2, h_D2, bytes);

    acc_set_device_num(2, acc_device_nvidia);
    double *d_A3 = (double*) acc_malloc(bytes);
    double *d_B3 = (double*) acc_malloc(bytes);
    double *d_C3 = (double*) acc_malloc(bytes); // src: value 3.0
    double *d_D3 = (double*) acc_malloc(bytes);
    if (!d_A3 || !d_B3 || !d_C3 || !d_D3) {
        cerr << "Error: acc_malloc failed on device 2\n";
        return EXIT_FAILURE;
    }
    acc_memcpy_to_device(d_A3, h_A3, bytes);
    acc_memcpy_to_device(d_B3, h_B3, bytes);
    acc_memcpy_to_device(d_C3, h_C3, bytes);
    acc_memcpy_to_device(d_D3, h_D3, bytes);

    acc_set_device_num(3, acc_device_nvidia);
    double *d_A4 = (double*) acc_malloc(bytes);
    double *d_B4 = (double*) acc_malloc(bytes);
    double *d_C4 = (double*) acc_malloc(bytes);
    double *d_D4 = (double*) acc_malloc(bytes); // src: value 4.0
    if (!d_A4 || !d_B4 || !d_C4 || !d_D4) {
        cerr << "Error: acc_malloc failed on device 3\n";
        return EXIT_FAILURE;
    }
    acc_memcpy_to_device(d_A4, h_A4, bytes);
    acc_memcpy_to_device(d_B4, h_B4, bytes);
    acc_memcpy_to_device(d_C4, h_C4, bytes);
    acc_memcpy_to_device(d_D4, h_D4, bytes);

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for (int w = 0; w < 2; ++w) {
        acc_set_device_num(1, acc_device_nvidia);
        acc_memcpy_device_async(d_B1, d_B2, bytes, 0);
        acc_wait(0);
        acc_set_device_num(2, acc_device_nvidia);
        acc_memcpy_device_async(d_C1, d_C3, bytes, 0);
        acc_wait(0);
        acc_set_device_num(3, acc_device_nvidia);
        acc_memcpy_device_async(d_D1, d_D4, bytes, 0);
        acc_wait(0);
    }

    // ── Timed P2P transfer (3 devices → device 0, parallel) ──────────────────
    omp_set_num_threads(3);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            acc_set_device_num(1, acc_device_nvidia);
            acc_memcpy_device_async(d_B1, d_B2, bytes, 0);
            acc_wait(0);   // wait only for queue 0 on device 1
        }
        #pragma omp section
        {
            acc_set_device_num(2, acc_device_nvidia);
            acc_memcpy_device_async(d_C1, d_C3, bytes, 0);
            acc_wait(0);   // wait only for queue 0 on device 2
        }
        #pragma omp section
        {
            acc_set_device_num(3, acc_device_nvidia);
            acc_memcpy_device_async(d_D1, d_D4, bytes, 0);
            acc_wait(0);   // wait only for queue 0 on device 3
        }
    }
    double end = omp_get_wtime();

    // ── Timer sanity check ───────────────────────────────────────────────────
    if (end <= start) {
        cerr << "Warning: timer returned non-positive elapsed time, "
             << "results may be unreliable\n";
    }

    // ── Bandwidth calculation ─────────────────────────────────────────────────
    double seconds       = end - start;
    double total_bytes   = 3.0 * static_cast<double>(bytes); // 3 P2P transfers
    double bandwidthGBps = total_bytes / (seconds * 1e9);

    cout << fixed << setprecision(6);
    cout << "Elapsed time:      " << seconds                                   << " s\n";
    cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    cout << "P2P Bandwidth:     " << bandwidthGBps                             << " GB/s\n";

    // ── Spot-check verification ───────────────────────────────────────────────
    // All 3 destination buffers live on device 0
    //   d_B1 ← d_B2 (dev1): expect 2.0
    //   d_C1 ← d_C3 (dev2): expect 3.0
    //   d_D1 ← d_D4 (dev3): expect 4.0
    size_t sample_size  = min(N, (size_t) 1000);
    size_t sample_bytes = sample_size * sizeof(double);

    double *h_verify = (double*) malloc(sample_bytes);
    if (!h_verify) {
        cerr << "Error: malloc failed for verification buffer\n";
    } else {
        struct Check { double *ptr; double expected; const char *label; };
        Check checks[] = {
            { d_B1, 2.0, "d_B1 (dev1→dev0)" },
            { d_C1, 3.0, "d_C1 (dev2→dev0)" },
            { d_D1, 4.0, "d_D1 (dev3→dev0)" },
        };

        acc_set_device_num(0, acc_device_nvidia);
        srand(42);
        for (auto &ck : checks) {
            acc_memcpy_from_device(h_verify, ck.ptr, sample_bytes);
            bool pass = true;
            for (size_t s = 0; s < sample_size; ++s) {
                size_t i = (size_t) rand() % sample_size;
                if (h_verify[i] != ck.expected) {
                    cerr << ck.label << " mismatch at index " << i
                         << ": expected " << ck.expected
                         << ", got "      << h_verify[i] << "\n";
                    pass = false;
                    break;
                }
            }
            cout << "Verification " << ck.label << ": "
                 << (pass ? "PASSED" : "FAILED")
                 << " (" << sample_size << "/" << N << " samples)\n";
        }
        free(h_verify);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    acc_set_device_num(0, acc_device_nvidia);
    acc_free(d_A1); acc_free(d_B1); acc_free(d_C1); acc_free(d_D1);

    acc_set_device_num(1, acc_device_nvidia);
    acc_free(d_A2); acc_free(d_B2); acc_free(d_C2); acc_free(d_D2);

    acc_set_device_num(2, acc_device_nvidia);
    acc_free(d_A3); acc_free(d_B3); acc_free(d_C3); acc_free(d_D3);

    acc_set_device_num(3, acc_device_nvidia);
    acc_free(d_A4); acc_free(d_B4); acc_free(d_C4); acc_free(d_D4);

    free(h_A1); free(h_B1); free(h_C1); free(h_D1);
    free(h_A2); free(h_B2); free(h_C2); free(h_D2);
    free(h_A3); free(h_B3); free(h_C3); free(h_D3);
    free(h_A4); free(h_B4); free(h_C4); free(h_D4);
    return EXIT_SUCCESS;
}
