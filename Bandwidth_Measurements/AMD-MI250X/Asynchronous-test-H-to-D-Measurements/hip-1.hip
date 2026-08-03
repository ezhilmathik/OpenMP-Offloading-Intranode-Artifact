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

    // ── Device check ─────────────────────────────────────────────────────────
    int num_devices = 0;
    checkCuda(hipGetDeviceCount(&num_devices), "hipGetDeviceCount");
    if (num_devices < 1) {
        cerr << "Error: no CUDA devices found\n";
        return EXIT_FAILURE;
    }
    checkCuda(hipSetDevice(0), "hipSetDevice 0");

    // ── Host allocation ──────────────────────────────────────────────────────
    double *h_A = (double*) malloc(bytes);
    if (!h_A) {
        cerr << "Error: malloc failed for " << bytes << " bytes\n";
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i)
        h_A[i] = 1.0;

    // ── Device allocation ────────────────────────────────────────────────────
    double *d_A = nullptr;
    checkCuda(hipMalloc(&d_A, bytes), "hipMalloc d_A");

    // ── Stream creation ──────────────────────────────────────────────────────
    hipStream_t stream;
    checkCuda(hipStreamCreate(&stream), "hipStreamCreate");

    // ── CUDA event creation ──────────────────────────────────────────────────
    hipEvent_t start, stop;
    checkCuda(hipEventCreate(&start), "hipEventCreate start");
    checkCuda(hipEventCreate(&stop),  "hipEventCreate stop");

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    checkCuda(hipMemcpyAsync(d_A, h_A, bytes, hipMemcpyHostToDevice, stream), "warmup 1 H→D d_A");
    checkCuda(hipStreamSynchronize(stream), "hipStreamSynchronize warmup 1");
    checkCuda(hipMemcpyAsync(d_A, h_A, bytes, hipMemcpyHostToDevice, stream), "warmup 2 H→D d_A");
    checkCuda(hipStreamSynchronize(stream), "hipStreamSynchronize warmup 2");

    // ── Timed transfer (async H→D via CUDA events) ───────────────────────────
    checkCuda(hipEventRecord(start, stream), "hipEventRecord start");
    checkCuda(hipMemcpyAsync(d_A, h_A, bytes, hipMemcpyHostToDevice, stream), "timed hipMemcpyAsync H→D d_A");
    checkCuda(hipEventRecord(stop, stream),  "hipEventRecord stop");
    checkCuda(hipStreamSynchronize(stream),  "hipStreamSynchronize timed");
    checkCuda(hipEventSynchronize(stop),     "hipEventSynchronize stop");

    // ── Elapsed time from CUDA events ────────────────────────────────────────
    float milliseconds = 0.0f;
    checkCuda(hipEventElapsedTime(&milliseconds, start, stop), "hipEventElapsedTime");

    if (milliseconds <= 0.0f) {
        cerr << "Warning: CUDA event elapsed time is non-positive, "
             << "results may be unreliable\n";
    }

    // ── Bandwidth calculation ─────────────────────────────────────────────────
    double seconds       = milliseconds / 1000.0;
    double total_bytes   = static_cast<double>(bytes);
    double bandwidthGBps = total_bytes / (seconds * 1e9);

    cout << fixed << setprecision(6);
    cout << "Elapsed time:      " << seconds                                   << " s\n";
    cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    cout << "PCIe Bandwidth:    " << bandwidthGBps                             << " GB/s\n";

    // ── Spot-check verification ───────────────────────────────────────────────
    size_t sample_size  = min(N, (size_t) 1000);
    size_t sample_bytes = sample_size * sizeof(double);

    double *h_verify = (double*) malloc(sample_bytes);
    if (!h_verify) {
        cerr << "Error: malloc failed for verification buffer\n";
        hipFree(d_A);
        hipStreamDestroy(stream);
        hipEventDestroy(start);
        hipEventDestroy(stop);
        free(h_A);
        return EXIT_FAILURE;
    }

    // Pull back only sample_size elements to confirm async transfer landed
    checkCuda(hipMemcpy(h_verify, d_A, sample_bytes, hipMemcpyDeviceToHost), "verify hipMemcpy D→H d_A");

    srand(42);
    bool pass = true;
    for (size_t s = 0; s < sample_size; ++s) {
        size_t i = (size_t) rand() % sample_size;
        if (h_verify[i] != h_A[i]) {
            cerr << "d_A mismatch at index " << i
                 << ": expected " << h_A[i]
                 << ", got "      << h_verify[i] << "\n";
            pass = false;
            break;
        }
    }

    cout << "Verification d_A (device 0): "
         << (pass ? "PASSED" : "FAILED")
         << " (" << sample_size << "/" << N << " samples)\n";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    free(h_verify);
    checkCuda(hipFree(d_A),           "hipFree d_A");
    checkCuda(hipStreamDestroy(stream), "hipStreamDestroy");
    checkCuda(hipEventDestroy(start), "hipEventDestroy start");
    checkCuda(hipEventDestroy(stop),  "hipEventDestroy stop");
    free(h_A);
    return EXIT_SUCCESS;
}