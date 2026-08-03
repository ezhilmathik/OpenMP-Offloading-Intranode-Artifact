//-*-c++-*-
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>
#include <omp.h>

using namespace std;

// ── CUDA error helper ────────────────────────────────────────────────────────
void checkCuda(cudaError_t result, const char *func) {
    if (result != cudaSuccess) {
        cerr << "CUDA error in " << func << ": "
             << cudaGetErrorString(result) << "\n";
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
    checkCuda(cudaGetDeviceCount(&num_devices), "cudaGetDeviceCount");
    if (num_devices < 1) {
        cerr << "Error: no CUDA devices found\n";
        return EXIT_FAILURE;
    }
    checkCuda(cudaSetDevice(0), "cudaSetDevice 0");

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
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");

    // ── Stream creation ──────────────────────────────────────────────────────
    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    // ── CUDA event creation ──────────────────────────────────────────────────
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop),  "cudaEventCreate stop");

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    checkCuda(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream), "warmup 1 H→D d_A");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup 1");
    checkCuda(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream), "warmup 2 H→D d_A");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup 2");

    // ── Timed transfer (async H→D via CUDA events) ───────────────────────────
    checkCuda(cudaEventRecord(start, stream), "cudaEventRecord start");
    checkCuda(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream), "timed cudaMemcpyAsync H→D d_A");
    checkCuda(cudaEventRecord(stop, stream),  "cudaEventRecord stop");
    checkCuda(cudaStreamSynchronize(stream),  "cudaStreamSynchronize timed");
    checkCuda(cudaEventSynchronize(stop),     "cudaEventSynchronize stop");

    // ── Elapsed time from CUDA events ────────────────────────────────────────
    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");

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
        cudaFree(d_A);
        cudaStreamDestroy(stream);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(h_A);
        return EXIT_FAILURE;
    }

    // Pull back only sample_size elements to confirm async transfer landed
    checkCuda(cudaMemcpy(h_verify, d_A, sample_bytes, cudaMemcpyDeviceToHost), "verify cudaMemcpy D→H d_A");

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
    checkCuda(cudaFree(d_A),           "cudaFree d_A");
    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCuda(cudaEventDestroy(stop),  "cudaEventDestroy stop");
    free(h_A);
    return EXIT_SUCCESS;
}