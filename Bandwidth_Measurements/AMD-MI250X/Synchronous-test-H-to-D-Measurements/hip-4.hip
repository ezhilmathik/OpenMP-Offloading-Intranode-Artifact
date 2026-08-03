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

    // ── Device check (need at least 4 GPUs) ──────────────────────────────────
    int num_devices = 0;
    checkCuda(hipGetDeviceCount(&num_devices), "hipGetDeviceCount");
    if (num_devices < 4) {
        cerr << "Error: need at least 4 CUDA devices, found "
             << num_devices << "\n";
        return EXIT_FAILURE;
    }

    // ── Host allocation ──────────────────────────────────────────────────────
    double *h_A = (double*) malloc(bytes);
    double *h_B = (double*) malloc(bytes);
    double *h_C = (double*) malloc(bytes);
    double *h_D = (double*) malloc(bytes);
    if (!h_A || !h_B || !h_C || !h_D) {
        cerr << "Error: malloc failed for " << bytes << " bytes\n";
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;   // distinct values so cross-buffer mix-ups are caught
        h_C[i] = 3.0;
        h_D[i] = 4.0;
    }

    // ── Device allocation ────────────────────────────────────────────────────
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;

    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipMalloc(&d_A, bytes), "hipMalloc d_A");

    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipMalloc(&d_B, bytes), "hipMalloc d_B");
    if (!d_B) {
        checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    checkCuda(hipSetDevice(2), "hipSetDevice 2");
    checkCuda(hipMalloc(&d_C, bytes), "hipMalloc d_C");
    if (!d_C) {
        checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
        checkCuda(hipSetDevice(1), "hipSetDevice 1"); checkCuda(hipFree(d_B), "hipFree d_B");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    checkCuda(hipSetDevice(3), "hipSetDevice 3");
    checkCuda(hipMalloc(&d_D, bytes), "hipMalloc d_D");
    if (!d_D) {
        checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
        checkCuda(hipSetDevice(1), "hipSetDevice 1"); checkCuda(hipFree(d_B), "hipFree d_B");
        checkCuda(hipSetDevice(2), "hipSetDevice 2"); checkCuda(hipFree(d_C), "hipFree d_C");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice), "warmup d_A");
    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice), "warmup d_B");
    checkCuda(hipSetDevice(2), "hipSetDevice 2");
    checkCuda(hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice), "warmup d_C");
    checkCuda(hipSetDevice(3), "hipSetDevice 3");
    checkCuda(hipMemcpy(d_D, h_D, bytes, hipMemcpyHostToDevice), "warmup d_D");

    // ── Timed transfer ───────────────────────────────────────────────────────
    omp_set_num_threads(4);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            checkCuda(hipSetDevice(0), "hipSetDevice 0");
            checkCuda(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice), "timed hipMemcpy d_A");
        }
        #pragma omp section
        {
            checkCuda(hipSetDevice(1), "hipSetDevice 1");
            checkCuda(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice), "timed hipMemcpy d_B");
        }
        #pragma omp section
        {
            checkCuda(hipSetDevice(2), "hipSetDevice 2");
            checkCuda(hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice), "timed hipMemcpy d_C");
        }
        #pragma omp section
        {
            checkCuda(hipSetDevice(3), "hipSetDevice 3");
            checkCuda(hipMemcpy(d_D, h_D, bytes, hipMemcpyHostToDevice), "timed hipMemcpy d_D");
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
    double total_bytes   = 4.0 * static_cast<double>(bytes);
    double bandwidthGBps = total_bytes / (seconds * 1e9);

    cout << fixed << setprecision(6);
    cout << "Elapsed time:      " << seconds                                   << " s\n";
    cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    cout << "PCIe Bandwidth:    " << bandwidthGBps                             << " GB/s\n";

    // ── Spot-check verification (all 4 devices) ───────────────────────────────
    size_t sample_size  = min(N, (size_t) 1000);
    size_t sample_bytes = sample_size * sizeof(double);

    double *h_verifyA = (double*) malloc(sample_bytes);
    double *h_verifyB = (double*) malloc(sample_bytes);
    double *h_verifyC = (double*) malloc(sample_bytes);
    double *h_verifyD = (double*) malloc(sample_bytes);
    if (!h_verifyA || !h_verifyB || !h_verifyC || !h_verifyD) {
        cerr << "Error: malloc failed for verification buffers\n";
        free(h_verifyA); free(h_verifyB); free(h_verifyC); free(h_verifyD);
        checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
        checkCuda(hipSetDevice(1), "hipSetDevice 1"); checkCuda(hipFree(d_B), "hipFree d_B");
        checkCuda(hipSetDevice(2), "hipSetDevice 2"); checkCuda(hipFree(d_C), "hipFree d_C");
        checkCuda(hipSetDevice(3), "hipSetDevice 3"); checkCuda(hipFree(d_D), "hipFree d_D");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    // Pull back only sample_size elements from each device
    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipMemcpy(h_verifyA, d_A, sample_bytes, hipMemcpyDeviceToHost), "verify d_A");
    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipMemcpy(h_verifyB, d_B, sample_bytes, hipMemcpyDeviceToHost), "verify d_B");
    checkCuda(hipSetDevice(2), "hipSetDevice 2");
    checkCuda(hipMemcpy(h_verifyC, d_C, sample_bytes, hipMemcpyDeviceToHost), "verify d_C");
    checkCuda(hipSetDevice(3), "hipSetDevice 3");
    checkCuda(hipMemcpy(h_verifyD, d_D, sample_bytes, hipMemcpyDeviceToHost), "verify d_D");

    // Pointer arrays for compact verification loop
    double *h_src[4]    = { h_A,       h_B,       h_C,       h_D       };
    double *h_verify[4] = { h_verifyA, h_verifyB, h_verifyC, h_verifyD };
    bool    pass[4]     = { true, true, true, true };

    srand(42);
    for (int d = 0; d < 4; ++d) {
        for (size_t s = 0; s < sample_size; ++s) {
            size_t i = (size_t) rand() % sample_size;
            if (h_verify[d][i] != h_src[d][i]) {
                cerr << "d_" << (char)('A'+d) << " mismatch at index " << i
                     << ": expected " << h_src[d][i]
                     << ", got "      << h_verify[d][i] << "\n";
                pass[d] = false;
                break;
            }
        }
        cout << "Verification d_" << (char)('A'+d) << " (device " << d << "): "
             << (pass[d] ? "PASSED" : "FAILED")
             << " (" << sample_size << "/" << N << " samples)\n";
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    free(h_verifyA); free(h_verifyB); free(h_verifyC); free(h_verifyD);
    checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
    checkCuda(hipSetDevice(1), "hipSetDevice 1"); checkCuda(hipFree(d_B), "hipFree d_B");
    checkCuda(hipSetDevice(2), "hipSetDevice 2"); checkCuda(hipFree(d_C), "hipFree d_C");
    checkCuda(hipSetDevice(3), "hipSetDevice 3"); checkCuda(hipFree(d_D), "hipFree d_D");
    free(h_A); free(h_B); free(h_C); free(h_D);
    return EXIT_SUCCESS;
}