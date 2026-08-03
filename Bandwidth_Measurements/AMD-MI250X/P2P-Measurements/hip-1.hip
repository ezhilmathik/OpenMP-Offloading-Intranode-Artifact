#include "hip/hip_runtime.h"
//-*-c++-*-
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <climits>    // ← fixes SIZE_MAX with nvcc
#include <hip/hip_runtime.h>
#include <omp.h>

using namespace std;

// ── CUDA error helper ────────────────────────────────────────────────────────
void checkCuda(hipError_t result, const char *func) {
    if (result != hipSuccess) {
        cerr << "CUDA error in " << func << ": "
             << hipGetErrorString(result) << "\n";
        exit(EXIT_FAILURE);
    }
}

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
    if (N > ((size_t)-1 / sizeof(double))) {
        cerr << "Error: vector_size too large, would overflow size_t\n";
        return EXIT_FAILURE;
    }
    size_t bytes = N * sizeof(double);

    // ── Device check ─────────────────────────────────────────────────────────
    int num_devices = 0;
    checkCuda(hipGetDeviceCount(&num_devices), "hipGetDeviceCount");
    if (num_devices < 2) {
        cerr << "Error: need at least 2 CUDA devices, found "
             << num_devices << "\n";
        return EXIT_FAILURE;
    }

    // ── Enable P2P access (both directions) ───────────────────────────────────
    for (int i = 0; i < 2; ++i) {
        checkCuda(hipSetDevice(i), "hipSetDevice P2P");
        for (int j = 0; j < 2; ++j) {
            if (i == j) continue;
            int canAccess = 0;
            checkCuda(hipDeviceCanAccessPeer(&canAccess, i, j), "hipDeviceCanAccessPeer");
            if (canAccess)
                checkCuda(hipDeviceEnablePeerAccess(j, 0), "hipDeviceEnablePeerAccess");
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
        h_B[i] = 2.0;  // will be overwritten by d_A (1.0) via P2P
    }

    // ── Device allocation + H→D population ───────────────────────────────────
    // Transfer map:
    //   d_B (dev1) ← d_A (dev0): expect 1.0

    double *d_A = nullptr, *d_B = nullptr;

    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipMalloc(&d_A, bytes), "hipMalloc d_A");
    checkCuda(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice), "initial hipMemcpy H→D d_A");

    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipMalloc(&d_B, bytes), "hipMalloc d_B");
    if (!d_B) {
        checkCuda(hipSetDevice(0), "hipSetDevice 0");
        checkCuda(hipFree(d_A), "hipFree d_A");
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }
    checkCuda(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice), "initial hipMemcpy H→D d_B");

    // ── Stream creation (one per device) ─────────────────────────────────────
    hipStream_t stream[2];
    for (int i = 0; i < 2; ++i) {
        checkCuda(hipSetDevice(i),             "hipSetDevice stream init");
        checkCuda(hipStreamCreate(&stream[i]), "hipStreamCreate");
    }
   
    // ── Warm-up P2P transfers (not timed) ────────────────────────────────────
    for (int w = 0; w < 2; ++w)
      {
	// ── Warm-up transfers (not timed) ────────────────────────────────────────
	checkCuda(hipSetDevice(0), "hipSetDevice 0");
	checkCuda(hipMemcpyPeerAsync(d_B, 1, d_A, 0, bytes, stream[0]), "warmup P2P d_A→d_B");
	checkCuda(hipStreamSynchronize(stream[0]), "hipStreamSynchronize warmup 2");
      }
    
    // ── Timed P2P transfer (dev0 → dev1) ─────────────────────────────────────
    double start = omp_get_wtime();
    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipMemcpyPeerAsync(d_B, 1, d_A, 0, bytes, stream[0]), "timed hipMemcpyPeerAsync d_A→d_B");
    checkCuda(hipStreamSynchronize(stream[0]), "hipStreamSynchronize timed");
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
        checkCuda(hipSetDevice(1), "hipSetDevice 1");
        checkCuda(hipMemcpy(h_verify, d_B, sample_bytes, hipMemcpyDeviceToHost),
                  "verify hipMemcpy D→H d_B");

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
    checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
    checkCuda(hipSetDevice(1), "hipSetDevice 1"); checkCuda(hipFree(d_B), "hipFree d_B");
    checkCuda(hipStreamDestroy(stream[0]), "hipStreamDestroy stream[0]");
    checkCuda(hipStreamDestroy(stream[1]), "hipStreamDestroy stream[1]");
    free(h_A); free(h_B);
    return EXIT_SUCCESS;
}
