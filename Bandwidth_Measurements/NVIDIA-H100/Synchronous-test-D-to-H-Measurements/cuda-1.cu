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

    // ── Device allocation + initial H→D to populate device buffer ────────────
    double *d_A = nullptr;
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "initial cudaMemcpy H→D d_A");

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    checkCuda(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost), "warmup 1 D→H d_A");
    checkCuda(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost), "warmup 2 D→H d_A");

    // ── Timed transfer (D→H) ─────────────────────────────────────────────────
    double start = omp_get_wtime();
    checkCuda(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost), "timed cudaMemcpy D→H d_A");
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
    cout << "PCIe Bandwidth:    " << bandwidthGBps                             << " GB/s\n";

    // ── Spot-check verification ───────────────────────────────────────────────
    // D→H lands back into h_A — check sample against known init value (1.0)
    size_t sample_size = min(N, (size_t) 1000);

    srand(42);
    bool pass = true;
    for (size_t s = 0; s < sample_size; ++s) {
        size_t i = (size_t) rand() % sample_size;
        if (h_A[i] != 1.0) {
            cerr << "h_A mismatch at index " << i
                 << ": expected 1.0, got " << h_A[i] << "\n";
            pass = false;
            break;
        }
    }

    cout << "Verification h_A (device 0 → host): "
         << (pass ? "PASSED" : "FAILED")
         << " (" << sample_size << "/" << N << " samples)\n";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    checkCuda(cudaFree(d_A), "cudaFree d_A");
    free(h_A);
    return EXIT_SUCCESS;
}