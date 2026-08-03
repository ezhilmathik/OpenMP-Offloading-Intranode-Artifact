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
    if (N > ((size_t)-1 / sizeof(double))) {
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

    // ── Device allocation ────────────────────────────────────────────────────
    double *d_A = nullptr, *d_B = nullptr;

    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipMalloc(&d_A, bytes), "hipMalloc d_A");

    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipMalloc(&d_B, bytes), "hipMalloc d_B");
    if (!d_B) {
        checkCuda(hipSetDevice(0), "hipSetDevice 0");
        checkCuda(hipFree(d_A), "hipFree d_A");
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }

    // ── Stream creation (one per device) ─────────────────────────────────────
    hipStream_t stream[2];
    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipStreamCreate(&stream[0]), "hipStreamCreate stream[0]");
    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipStreamCreate(&stream[1]), "hipStreamCreate stream[1]");

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for (int w=0; w<2; w++)
      {
	checkCuda(hipSetDevice(0), "hipSetDevice 0");
	checkCuda(hipMemcpyAsync(d_A, h_A, bytes, hipMemcpyHostToDevice, stream[0]), "warmup H→D d_A");
	checkCuda(hipStreamSynchronize(stream[0]), "hipStreamSynchronize warmup stream[0]");
	checkCuda(hipSetDevice(1), "hipSetDevice 1");
	checkCuda(hipMemcpyAsync(d_B, h_B, bytes, hipMemcpyHostToDevice, stream[1]), "warmup H→D d_B");
	checkCuda(hipStreamSynchronize(stream[1]), "hipStreamSynchronize warmup stream[1]");
      }
    
    // ── Timed transfer (async H→D, parallel across both devices) ─────────────
    omp_set_num_threads(2);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            checkCuda(hipSetDevice(0), "hipSetDevice 0");
            checkCuda(hipMemcpyAsync(d_A, h_A, bytes, hipMemcpyHostToDevice, stream[0]), "timed hipMemcpyAsync H→D d_A");
            checkCuda(hipStreamSynchronize(stream[0]), "hipStreamSynchronize timed stream[0]");
        }
        #pragma omp section
        {
            checkCuda(hipSetDevice(1), "hipSetDevice 1");
            checkCuda(hipMemcpyAsync(d_B, h_B, bytes, hipMemcpyHostToDevice, stream[1]), "timed hipMemcpyAsync H→D d_B");
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

    // ── Spot-check verification (both devices) ────────────────────────────────
    size_t sample_size  = min(N, (size_t) 1000);
    size_t sample_bytes = sample_size * sizeof(double);

    double *h_verifyA = (double*) malloc(sample_bytes);
    double *h_verifyB = (double*) malloc(sample_bytes);
    if (!h_verifyA || !h_verifyB) {
        cerr << "Error: malloc failed for verification buffers\n";
        free(h_verifyA); free(h_verifyB);
        checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
        checkCuda(hipSetDevice(1), "hipSetDevice 1"); checkCuda(hipFree(d_B), "hipFree d_B");
        hipStreamDestroy(stream[0]); hipStreamDestroy(stream[1]);
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }

    // Pull back only sample_size elements from each device to confirm transfer
    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipMemcpy(h_verifyA, d_A, sample_bytes, hipMemcpyDeviceToHost), "verify D→H d_A");
    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipMemcpy(h_verifyB, d_B, sample_bytes, hipMemcpyDeviceToHost), "verify D→H d_B");

    srand(42);
    bool passA = true, passB = true;
    for (size_t s = 0; s < sample_size; ++s) {
        size_t i = (size_t) rand() % sample_size;
        if (h_verifyA[i] != h_A[i]) {
            cerr << "d_A mismatch at index " << i
                 << ": expected " << h_A[i]
                 << ", got "      << h_verifyA[i] << "\n";
            passA = false;
            break;
        }
        if (h_verifyB[i] != h_B[i]) {
            cerr << "d_B mismatch at index " << i
                 << ": expected " << h_B[i]
                 << ", got "      << h_verifyB[i] << "\n";
            passB = false;
            break;
        }
    }

    cout << "Verification d_A (device 0): "
         << (passA ? "PASSED" : "FAILED")
         << " (" << sample_size << "/" << N << " samples)\n";
    cout << "Verification d_B (device 1): "
         << (passB ? "PASSED" : "FAILED")
         << " (" << sample_size << "/" << N << " samples)\n";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    free(h_verifyA); free(h_verifyB);
    checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
    checkCuda(hipSetDevice(1), "hipSetDevice 1"); checkCuda(hipFree(d_B), "hipFree d_B");
    checkCuda(hipStreamDestroy(stream[0]), "hipStreamDestroy stream[0]");
    checkCuda(hipStreamDestroy(stream[1]), "hipStreamDestroy stream[1]");
    free(h_A); free(h_B);
    return EXIT_SUCCESS;
}
