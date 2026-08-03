#include "hip/hip_runtime.h"
//-*-c++-*-
#include <iostream>
#include <iomanip>
#include <cstdlib>
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
    if (N > (SIZE_MAX / sizeof(double))) {
        cerr << "Error: vector_size too large, would overflow size_t\n";
        return EXIT_FAILURE;
    }
    size_t bytes = N * sizeof(double);

    // ── Device check (need at least 2 GPUs) ──────────────────────────────────
    int num_devices = 0;
    checkCuda(hipGetDeviceCount(&num_devices), "hipGetDeviceCount");
    if (num_devices < 2) {
        cerr << "Error: need at least 2 CUDA devices, found "
             << num_devices << "\n";
        return EXIT_FAILURE;
    }

    // ── Host allocation ──────────────────────────────────────────────────────
    double *h_A = (double*) malloc(bytes);
    double *h_B = (double*) malloc(bytes);
    if (!h_A || !h_B) {
        cerr << "Error: malloc failed for " << bytes << " bytes\n";
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;   // distinct value so cross-buffer mix-ups are caught
    }

    // ── Device allocation + initial H→D to populate device buffers ───────────
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
    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipStreamCreate(&stream[0]), "hipStreamCreate stream[0]");
    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipStreamCreate(&stream[1]), "hipStreamCreate stream[1]");

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for(int w=0; w<2; w++)
      {
	checkCuda(hipSetDevice(0), "hipSetDevice 0");
	checkCuda(hipMemcpyAsync(h_A, d_A, bytes, hipMemcpyDeviceToHost, stream[0]), "warmup D→H d_A");
	checkCuda(hipStreamSynchronize(stream[0]), "hipStreamSynchronize warmup stream[0]");
	checkCuda(hipSetDevice(1), "hipSetDevice 1");
	checkCuda(hipMemcpyAsync(h_B, d_B, bytes, hipMemcpyDeviceToHost, stream[1]), "warmup D→H d_B");
	checkCuda(hipStreamSynchronize(stream[1]), "hipStreamSynchronize warmup stream[1]");
      }
    
    // ── Timed transfer (async D→H, parallel across both devices) ─────────────
    omp_set_num_threads(2);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            checkCuda(hipSetDevice(0), "hipSetDevice 0");
            checkCuda(hipMemcpyAsync(h_A, d_A, bytes, hipMemcpyDeviceToHost, stream[0]), "timed hipMemcpyAsync D→H d_A");
            checkCuda(hipStreamSynchronize(stream[0]), "hipStreamSynchronize timed stream[0]");
        }
        #pragma omp section
        {
            checkCuda(hipSetDevice(1), "hipSetDevice 1");
            checkCuda(hipMemcpyAsync(h_B, d_B, bytes, hipMemcpyDeviceToHost, stream[1]), "timed hipMemcpyAsync D→H d_B");
            checkCuda(hipStreamSynchronize(stream[1]), "hipStreamSynchronize timed stream[1]");
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
    double total_bytes   = 2.0 * static_cast<double>(bytes);
    double bandwidthGBps = total_bytes / (seconds * 1e9);

    cout << fixed << setprecision(6);
    cout << "Elapsed time:      " << seconds                                   << " s\n";
    cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    cout << "PCIe Bandwidth:    " << bandwidthGBps                             << " GB/s\n";

    // ── Spot-check verification ───────────────────────────────────────────────
    // D→H lands back into h_A/h_B — check sample against known init values
    size_t sample_size = min(N, (size_t) 1000);

    srand(42);
    bool passA = true, passB = true;
    for (size_t s = 0; s < sample_size; ++s) {
        size_t i = (size_t) rand() % sample_size;
        if (h_A[i] != 1.0) {
            cerr << "h_A mismatch at index " << i
                 << ": expected 1.0, got " << h_A[i] << "\n";
            passA = false;
            break;
        }
        if (h_B[i] != 2.0) {
            cerr << "h_B mismatch at index " << i
                 << ": expected 2.0, got " << h_B[i] << "\n";
            passB = false;
            break;
        }
    }

    cout << "Verification h_A (device 0 → host): "
         << (passA ? "PASSED" : "FAILED")
         << " (" << sample_size << "/" << N << " samples)\n";
    cout << "Verification h_B (device 1 → host): "
         << (passB ? "PASSED" : "FAILED")
         << " (" << sample_size << "/" << N << " samples)\n";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
    checkCuda(hipSetDevice(1), "hipSetDevice 1"); checkCuda(hipFree(d_B), "hipFree d_B");
    checkCuda(hipStreamDestroy(stream[0]), "hipStreamDestroy stream[0]");
    checkCuda(hipStreamDestroy(stream[1]), "hipStreamDestroy stream[1]");
    free(h_A); free(h_B);
    return EXIT_SUCCESS;
}
