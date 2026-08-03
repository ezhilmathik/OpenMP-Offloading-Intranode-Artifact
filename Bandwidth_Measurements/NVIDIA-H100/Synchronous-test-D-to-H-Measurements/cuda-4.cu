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

    // ── Device check (need at least 4 GPUs) ──────────────────────────────────
    int num_devices = 0;
    checkCuda(cudaGetDeviceCount(&num_devices), "cudaGetDeviceCount");
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

    checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "initial cudaMemcpy H→D d_A");

    checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    if (!d_B) {
        checkCuda(cudaSetDevice(0), "cudaSetDevice 0"); checkCuda(cudaFree(d_A), "cudaFree d_A");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "initial cudaMemcpy H→D d_B");

    checkCuda(cudaSetDevice(2), "cudaSetDevice 2");
    checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");
    if (!d_C) {
        checkCuda(cudaSetDevice(0), "cudaSetDevice 0"); checkCuda(cudaFree(d_A), "cudaFree d_A");
        checkCuda(cudaSetDevice(1), "cudaSetDevice 1"); checkCuda(cudaFree(d_B), "cudaFree d_B");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }
    checkCuda(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice), "initial cudaMemcpy H→D d_C");

    checkCuda(cudaSetDevice(3), "cudaSetDevice 3");
    checkCuda(cudaMalloc(&d_D, bytes), "cudaMalloc d_D");
    if (!d_D) {
        checkCuda(cudaSetDevice(0), "cudaSetDevice 0"); checkCuda(cudaFree(d_A), "cudaFree d_A");
        checkCuda(cudaSetDevice(1), "cudaSetDevice 1"); checkCuda(cudaFree(d_B), "cudaFree d_B");
        checkCuda(cudaSetDevice(2), "cudaSetDevice 2"); checkCuda(cudaFree(d_C), "cudaFree d_C");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }
    checkCuda(cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice), "initial cudaMemcpy H→D d_D");

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
    checkCuda(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost), "warmup D→H d_A");
    checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
    checkCuda(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost), "warmup D→H d_B");
    checkCuda(cudaSetDevice(2), "cudaSetDevice 2");
    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "warmup D→H d_C");
    checkCuda(cudaSetDevice(3), "cudaSetDevice 3");
    checkCuda(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost), "warmup D→H d_D");

    // ── Timed transfer (D→H) ─────────────────────────────────────────────────
    omp_set_num_threads(4);
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
        #pragma omp section
        {
            checkCuda(cudaSetDevice(2), "cudaSetDevice 2");
            checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "timed cudaMemcpy D→H d_C");
        }
        #pragma omp section
        {
            checkCuda(cudaSetDevice(3), "cudaSetDevice 3");
            checkCuda(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost), "timed cudaMemcpy D→H d_D");
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
    checkCuda(cudaSetDevice(0), "cudaSetDevice 0"); checkCuda(cudaFree(d_A), "cudaFree d_A");
    checkCuda(cudaSetDevice(1), "cudaSetDevice 1"); checkCuda(cudaFree(d_B), "cudaFree d_B");
    checkCuda(cudaSetDevice(2), "cudaSetDevice 2"); checkCuda(cudaFree(d_C), "cudaFree d_C");
    checkCuda(cudaSetDevice(3), "cudaSetDevice 3"); checkCuda(cudaFree(d_D), "cudaFree d_D");
    free(h_A); free(h_B); free(h_C); free(h_D);
    return EXIT_SUCCESS;
}