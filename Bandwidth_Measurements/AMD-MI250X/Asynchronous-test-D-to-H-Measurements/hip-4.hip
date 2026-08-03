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

    // ── Device allocation + initial H→D to populate device buffers ───────────
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;

    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipMalloc(&d_A, bytes), "hipMalloc d_A");
    checkCuda(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice), "initial hipMemcpy H→D d_A");

    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipMalloc(&d_B, bytes), "hipMalloc d_B");
    if (!d_B) {
        checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }
    checkCuda(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice), "initial hipMemcpy H→D d_B");

    checkCuda(hipSetDevice(2), "hipSetDevice 2");
    checkCuda(hipMalloc(&d_C, bytes), "hipMalloc d_C");
    if (!d_C) {
        checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
        checkCuda(hipSetDevice(1), "hipSetDevice 1"); checkCuda(hipFree(d_B), "hipFree d_B");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }
    checkCuda(hipMemcpy(d_C, h_C, bytes, hipMemcpyHostToDevice), "initial hipMemcpy H→D d_C");

    checkCuda(hipSetDevice(3), "hipSetDevice 3");
    checkCuda(hipMalloc(&d_D, bytes), "hipMalloc d_D");
    if (!d_D) {
        checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
        checkCuda(hipSetDevice(1), "hipSetDevice 1"); checkCuda(hipFree(d_B), "hipFree d_B");
        checkCuda(hipSetDevice(2), "hipSetDevice 2"); checkCuda(hipFree(d_C), "hipFree d_C");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }
    checkCuda(hipMemcpy(d_D, h_D, bytes, hipMemcpyHostToDevice), "initial hipMemcpy H→D d_D");

    // ── Stream creation (one per device) ─────────────────────────────────────
    hipStream_t stream[4];
    for (int i = 0; i < 4; ++i) {
        checkCuda(hipSetDevice(i),             "hipSetDevice stream init");
        checkCuda(hipStreamCreate(&stream[i]), "hipStreamCreate");
    }

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for(int w=0; w<2; w++)
      {
	checkCuda(hipSetDevice(0), "hipSetDevice 0");
	checkCuda(hipMemcpyAsync(h_A, d_A, bytes, hipMemcpyDeviceToHost, stream[0]), "warmup D→H d_A");
	checkCuda(hipStreamSynchronize(stream[0]), "hipStreamSynchronize warmup stream[0]");
	checkCuda(hipSetDevice(1), "hipSetDevice 1");
	checkCuda(hipMemcpyAsync(h_B, d_B, bytes, hipMemcpyDeviceToHost, stream[1]), "warmup D→H d_B");
	checkCuda(hipStreamSynchronize(stream[1]), "hipStreamSynchronize warmup stream[1]");
	checkCuda(hipSetDevice(2), "hipSetDevice 2");
	checkCuda(hipMemcpyAsync(h_C, d_C, bytes, hipMemcpyDeviceToHost, stream[2]), "warmup D→H d_C");
	checkCuda(hipStreamSynchronize(stream[2]), "hipStreamSynchronize warmup stream[2]");
	checkCuda(hipSetDevice(3), "hipSetDevice 3");
	checkCuda(hipMemcpyAsync(h_D, d_D, bytes, hipMemcpyDeviceToHost, stream[3]), "warmup D→H d_D");
	checkCuda(hipStreamSynchronize(stream[3]), "hipStreamSynchronize warmup stream[3]");
      }
    
    // ── Timed transfer (async D→H, parallel across all 4 devices) ────────────
    omp_set_num_threads(4);
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
        #pragma omp section
        {
            checkCuda(hipSetDevice(2), "hipSetDevice 2");
            checkCuda(hipMemcpyAsync(h_C, d_C, bytes, hipMemcpyDeviceToHost, stream[2]), "timed hipMemcpyAsync D→H d_C");
            checkCuda(hipStreamSynchronize(stream[2]), "hipStreamSynchronize timed stream[2]");
        }
        #pragma omp section
        {
            checkCuda(hipSetDevice(3), "hipSetDevice 3");
            checkCuda(hipMemcpyAsync(h_D, d_D, bytes, hipMemcpyDeviceToHost, stream[3]), "timed hipMemcpyAsync D→H d_D");
            checkCuda(hipStreamSynchronize(stream[3]), "hipStreamSynchronize timed stream[3]");
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

    // ── Spot-check verification ───────────────────────────────────────────────
    // D→H lands back into h_A/h_B/h_C/h_D — check against known init values
    size_t sample_size = min(N, (size_t) 1000);

    double  expected[4] = { 1.0, 2.0, 3.0, 4.0 };
    double *h_buf[4]    = { h_A, h_B, h_C, h_D };
    bool    pass[4]     = { true, true, true, true };

    srand(42);
    for (int d = 0; d < 4; ++d) {
        for (size_t s = 0; s < sample_size; ++s) {
            size_t i = (size_t) rand() % sample_size;
            if (h_buf[d][i] != expected[d]) {
                cerr << "h_" << (char)('A'+d) << " mismatch at index " << i
                     << ": expected " << expected[d]
                     << ", got "      << h_buf[d][i] << "\n";
                pass[d] = false;
                break;
            }
        }
        cout << "Verification h_" << (char)('A'+d)
             << " (device " << d << " → host): "
             << (pass[d] ? "PASSED" : "FAILED")
             << " (" << sample_size << "/" << N << " samples)\n";
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    checkCuda(hipSetDevice(0), "hipSetDevice 0"); checkCuda(hipFree(d_A), "hipFree d_A");
    checkCuda(hipSetDevice(1), "hipSetDevice 1"); checkCuda(hipFree(d_B), "hipFree d_B");
    checkCuda(hipSetDevice(2), "hipSetDevice 2"); checkCuda(hipFree(d_C), "hipFree d_C");
    checkCuda(hipSetDevice(3), "hipSetDevice 3"); checkCuda(hipFree(d_D), "hipFree d_D");
    for (int i = 0; i < 4; ++i)
        checkCuda(hipStreamDestroy(stream[i]), "hipStreamDestroy");
    free(h_A); free(h_B); free(h_C); free(h_D);
    return EXIT_SUCCESS;
}
