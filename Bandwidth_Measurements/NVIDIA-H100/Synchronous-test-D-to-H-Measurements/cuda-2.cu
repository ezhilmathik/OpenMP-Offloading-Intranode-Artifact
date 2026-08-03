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

    // ── Device check (need at least 2 GPUs) ──────────────────────────────────
    int num_devices = 0;
    checkCuda(cudaGetDeviceCount(&num_devices), "cudaGetDeviceCount");
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

    checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "initial cudaMemcpy H→D d_A");

    checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    if (!d_B) {
        checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
        checkCuda(cudaFree(d_A), "cudaFree d_A");
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "initial cudaMemcpy H→D d_B");

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
    checkCuda(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost), "warmup D→H d_A");
    checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
    checkCuda(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost), "warmup D→H d_B");

    // ── Timed transfer (D→H) ─────────────────────────────────────────────────
    omp_set_num_threads(2);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
            checkCuda(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost), "timed cudaMemcpy D→H d_A");
        }
        #pragma omp section
        {
            checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
            checkCuda(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost), "timed cudaMemcpy D→H d_B");
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
    checkCuda(cudaSetDevice(0), "cudaSetDevice 0"); checkCuda(cudaFree(d_A), "cudaFree d_A");
    checkCuda(cudaSetDevice(1), "cudaSetDevice 1"); checkCuda(cudaFree(d_B), "cudaFree d_B");
    free(h_A); free(h_B);
    return EXIT_SUCCESS;
}