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
    if (num_devices < 2) {
        cerr << "Error: need at least 2 NVIDIA OpenACC devices, found "
             << num_devices << "\n";
        return EXIT_FAILURE;
    }

    // ── Enable P2P access between device 0 and device 1 ──────────────────────
    for (int i = 0; i < 2; ++i) {
        cudaSetDevice(i);
        for (int j = 0; j < 2; ++j) {
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
    double *h_A = (double*) malloc(bytes); // dev0 src: 1.0
    double *h_B = (double*) malloc(bytes); // dev1 src: 2.0
    double *h_C = (double*) malloc(bytes); // dev1 dst: receives from dev0 (1.0)
    double *h_D = (double*) malloc(bytes); // dev0 dst: receives from dev1 (2.0)
    if (!h_A || !h_B || !h_C || !h_D) {
        cerr << "Error: malloc failed for host buffers\n";
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
        h_C[i] = 3.0;  // will be overwritten by d_A (1.0)
        h_D[i] = 4.0;  // will be overwritten by d_B (2.0)
    }

    // ── Device allocation + H→D population ───────────────────────────────────
    // Transfer map:
    //   d_C (dev1) ← d_A (dev0): expect 1.0
    //   d_D (dev0) ← d_B (dev1): expect 2.0

    acc_set_device_num(0, acc_device_nvidia);
    double *d_A = (double*) acc_malloc(bytes); // src: 1.0
    double *d_D = (double*) acc_malloc(bytes); // dst: receives from dev1 (2.0)
    if (!d_A || !d_D) {
        cerr << "Error: acc_malloc failed on device 0\n";
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }
    acc_memcpy_to_device(d_A, h_A, bytes);
    acc_memcpy_to_device(d_D, h_D, bytes);

    acc_set_device_num(1, acc_device_nvidia);
    double *d_B = (double*) acc_malloc(bytes); // src: 2.0
    double *d_C = (double*) acc_malloc(bytes); // dst: receives from dev0 (1.0)
    if (!d_B || !d_C) {
        cerr << "Error: acc_malloc failed on device 1\n";
        acc_set_device_num(0, acc_device_nvidia);
        acc_free(d_A); acc_free(d_D);
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }
    acc_memcpy_to_device(d_B, h_B, bytes);
    acc_memcpy_to_device(d_C, h_C, bytes);

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for (int w = 0; w < 2; ++w) {
        acc_set_device_num(0, acc_device_nvidia);
        acc_memcpy_device_async(d_C, d_A, bytes, 0);
        acc_wait(0);
        acc_set_device_num(1, acc_device_nvidia);
        acc_memcpy_device_async(d_D, d_B, bytes, 0);
        acc_wait(0);
    }

    // ── Timed P2P transfer (both devices simultaneously) ─────────────────────
    omp_set_num_threads(2);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            acc_set_device_num(0, acc_device_nvidia);
            acc_memcpy_device_async(d_C, d_A, bytes, 0);
            acc_wait(0);   // wait only for queue 0 on device 0
        }
        #pragma omp section
        {
            acc_set_device_num(1, acc_device_nvidia);
            acc_memcpy_device_async(d_D, d_B, bytes, 0);
            acc_wait(0);   // wait only for queue 0 on device 1
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
    double total_bytes   = 2.0 * static_cast<double>(bytes); // 2 P2P transfers
    double bandwidthGBps = total_bytes / (seconds * 1e9);

    cout << fixed << setprecision(6);
    cout << "Elapsed time:      " << seconds                                   << " s\n";
    cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    cout << "P2P Bandwidth:     " << bandwidthGBps                             << " GB/s\n";

    // ── Spot-check verification ───────────────────────────────────────────────
    // d_C lives on dev1, d_D lives on dev0
    size_t sample_size  = min(N, (size_t) 1000);
    size_t sample_bytes = sample_size * sizeof(double);

    double *h_verify = (double*) malloc(sample_bytes);
    if (!h_verify) {
        cerr << "Error: malloc failed for verification buffer\n";
    } else {
        struct Check { int dev; double *ptr; double expected; const char *label; };
        Check checks[] = {
            { 1, d_C, 1.0, "d_C (dev0→dev1)" },
            { 0, d_D, 2.0, "d_D (dev1→dev0)" },
        };

        srand(42);
        for (auto &ck : checks) {
            acc_set_device_num(ck.dev, acc_device_nvidia);
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
    acc_free(d_A); acc_free(d_D);
    acc_set_device_num(1, acc_device_nvidia);
    acc_free(d_B); acc_free(d_C);
    free(h_A); free(h_B); free(h_C); free(h_D);
    return EXIT_SUCCESS;
}
