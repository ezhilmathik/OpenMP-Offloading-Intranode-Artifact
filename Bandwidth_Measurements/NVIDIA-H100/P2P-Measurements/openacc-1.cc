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

    // ── Enable P2P access (both directions) ───────────────────────────────────
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
    double *h_B = (double*) malloc(bytes); // dev1 dst: receives from dev0 (1.0)
    if (!h_A || !h_B) {
        cerr << "Error: malloc failed for " << bytes << " bytes\n";
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;  // will be overwritten by d_A (1.0)
    }

    // ── Device allocation + H→D population ───────────────────────────────────
    // Transfer map:
    //   d_B (dev1) ← d_A (dev0): expect 1.0

    acc_set_device_num(0, acc_device_nvidia);
    double *d_A = (double*) acc_malloc(bytes); // src: 1.0
    if (!d_A) {
        cerr << "Error: acc_malloc failed for d_A on device 0\n";
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }
    acc_memcpy_to_device(d_A, h_A, bytes);

    acc_set_device_num(1, acc_device_nvidia);
    double *d_B = (double*) acc_malloc(bytes); // dst: receives from dev0 (1.0)
    if (!d_B) {
        cerr << "Error: acc_malloc failed for d_B on device 1\n";
        acc_set_device_num(0, acc_device_nvidia); acc_free(d_A);
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }
    acc_memcpy_to_device(d_B, h_B, bytes);

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    acc_set_device_num(0, acc_device_nvidia);
    acc_memcpy_device_async(d_B, d_A, bytes, 0);
    acc_wait(0);
    acc_memcpy_device_async(d_B, d_A, bytes, 0);
    acc_wait(0);

    // ── Timed P2P transfer (dev0 → dev1) ─────────────────────────────────────
    double start = omp_get_wtime();
    acc_set_device_num(0, acc_device_nvidia);
    acc_memcpy_device_async(d_B, d_A, bytes, 0);
    acc_wait(0);
    double end = omp_get_wtime();

    // ── Timer sanity check ───────────────────────────────────────────────────
    if (end <= start) {
        cerr << "Warning: timer returned non-positive elapsed time, "
             << "results may be unreliable\n";
    }

    // ── Bandwidth calculation ─────────────────────────────────────────────────
    double seconds       = end - start;
    double total_bytes   = static_cast<double>(bytes);
    double bandwidthGBps = total_bytes / (seconds * 1e9);

    cout << fixed << setprecision(6);
    cout << "Elapsed time:      " << seconds                                   << " s\n";
    cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    cout << "P2P Bandwidth:     " << bandwidthGBps                             << " GB/s\n";

    // ── Spot-check verification ───────────────────────────────────────────────
    // d_B lives on dev1 — should now contain 1.0 (copied from d_A on dev0)
    size_t sample_size  = min(N, (size_t) 1000);
    size_t sample_bytes = sample_size * sizeof(double);

    double *h_verify = (double*) malloc(sample_bytes);
    if (!h_verify) {
        cerr << "Error: malloc failed for verification buffer\n";
    } else {
        acc_set_device_num(1, acc_device_nvidia);
        acc_memcpy_from_device(h_verify, d_B, sample_bytes);

        srand(42);
        bool pass = true;
        for (size_t s = 0; s < sample_size; ++s) {
            size_t i = (size_t) rand() % sample_size;
            if (h_verify[i] != 1.0) {
                cerr << "d_B (dev0→dev1) mismatch at index " << i
                     << ": expected 1.0, got " << h_verify[i] << "\n";
                pass = false;
                break;
            }
        }
        cout << "Verification d_B (dev0→dev1): "
             << (pass ? "PASSED" : "FAILED")
             << " (" << sample_size << "/" << N << " samples)\n";
        free(h_verify);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    acc_set_device_num(0, acc_device_nvidia); acc_free(d_A);
    acc_set_device_num(1, acc_device_nvidia); acc_free(d_B);
    free(h_A); free(h_B);
    return EXIT_SUCCESS;
}
