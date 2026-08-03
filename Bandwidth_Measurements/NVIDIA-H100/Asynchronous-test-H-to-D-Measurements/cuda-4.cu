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

    // ── Device allocation ────────────────────────────────────────────────────
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;

    checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");

    checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    if (!d_B) {
        checkCuda(cudaSetDevice(0), "cudaSetDevice 0"); checkCuda(cudaFree(d_A), "cudaFree d_A");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    checkCuda(cudaSetDevice(2), "cudaSetDevice 2");
    checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");
    if (!d_C) {
        checkCuda(cudaSetDevice(0), "cudaSetDevice 0"); checkCuda(cudaFree(d_A), "cudaFree d_A");
        checkCuda(cudaSetDevice(1), "cudaSetDevice 1"); checkCuda(cudaFree(d_B), "cudaFree d_B");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    checkCuda(cudaSetDevice(3), "cudaSetDevice 3");
    checkCuda(cudaMalloc(&d_D, bytes), "cudaMalloc d_D");
    if (!d_D) {
        checkCuda(cudaSetDevice(0), "cudaSetDevice 0"); checkCuda(cudaFree(d_A), "cudaFree d_A");
        checkCuda(cudaSetDevice(1), "cudaSetDevice 1"); checkCuda(cudaFree(d_B), "cudaFree d_B");
        checkCuda(cudaSetDevice(2), "cudaSetDevice 2"); checkCuda(cudaFree(d_C), "cudaFree d_C");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    // ── Stream creation (one per device) ─────────────────────────────────────
    cudaStream_t stream[4];
    for (int i = 0; i < 4; ++i) {
        checkCuda(cudaSetDevice(i),            "cudaSetDevice stream init");
        checkCuda(cudaStreamCreate(&stream[i]), "cudaStreamCreate");
    }

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for (int w=0; w<2; w++)
      {
	checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
	checkCuda(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream[0]), "warmup H→D d_A");
	checkCuda(cudaStreamSynchronize(stream[0]), "cudaStreamSynchronize warmup stream[0]");
	checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
	checkCuda(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream[1]), "warmup H→D d_B");
	checkCuda(cudaStreamSynchronize(stream[1]), "cudaStreamSynchronize warmup stream[1]");
	checkCuda(cudaSetDevice(2), "cudaSetDevice 2");
	checkCuda(cudaMemcpyAsync(d_C, h_C, bytes, cudaMemcpyHostToDevice, stream[2]), "warmup H→D d_C");
	checkCuda(cudaStreamSynchronize(stream[2]), "cudaStreamSynchronize warmup stream[2]");
	checkCuda(cudaSetDevice(3), "cudaSetDevice 3");
	checkCuda(cudaMemcpyAsync(d_D, h_D, bytes, cudaMemcpyHostToDevice, stream[3]), "warmup H→D d_D");
	checkCuda(cudaStreamSynchronize(stream[3]), "cudaStreamSynchronize warmup stream[3]");
      }
    
    // ── Timed transfer (async H→D, parallel across all 4 devices) ────────────
    omp_set_num_threads(4);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
            checkCuda(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, stream[0]), "timed cudaMemcpyAsync H→D d_A");
            checkCuda(cudaStreamSynchronize(stream[0]), "cudaStreamSynchronize timed stream[0]");
        }
        #pragma omp section
        {
            checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
            checkCuda(cudaMemcpyAsync(d_B, h_B, bytes, cudaMemcpyHostToDevice, stream[1]), "timed cudaMemcpyAsync H→D d_B");
            checkCuda(cudaStreamSynchronize(stream[1]), "cudaStreamSynchronize timed stream[1]");
        }
        #pragma omp section
        {
            checkCuda(cudaSetDevice(2), "cudaSetDevice 2");
            checkCuda(cudaMemcpyAsync(d_C, h_C, bytes, cudaMemcpyHostToDevice, stream[2]), "timed cudaMemcpyAsync H→D d_C");
            checkCuda(cudaStreamSynchronize(stream[2]), "cudaStreamSynchronize timed stream[2]");
        }
        #pragma omp section
        {
            checkCuda(cudaSetDevice(3), "cudaSetDevice 3");
            checkCuda(cudaMemcpyAsync(d_D, h_D, bytes, cudaMemcpyHostToDevice, stream[3]), "timed cudaMemcpyAsync H→D d_D");
            checkCuda(cudaStreamSynchronize(stream[3]), "cudaStreamSynchronize timed stream[3]");
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
        checkCuda(cudaSetDevice(0), "cudaSetDevice 0"); checkCuda(cudaFree(d_A), "cudaFree d_A");
        checkCuda(cudaSetDevice(1), "cudaSetDevice 1"); checkCuda(cudaFree(d_B), "cudaFree d_B");
        checkCuda(cudaSetDevice(2), "cudaSetDevice 2"); checkCuda(cudaFree(d_C), "cudaFree d_C");
        checkCuda(cudaSetDevice(3), "cudaSetDevice 3"); checkCuda(cudaFree(d_D), "cudaFree d_D");
        for (int i = 0; i < 4; ++i) cudaStreamDestroy(stream[i]);
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    // Pull back only sample_size elements from each device to confirm transfer
    checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
    checkCuda(cudaMemcpy(h_verifyA, d_A, sample_bytes, cudaMemcpyDeviceToHost), "verify D→H d_A");
    checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
    checkCuda(cudaMemcpy(h_verifyB, d_B, sample_bytes, cudaMemcpyDeviceToHost), "verify D→H d_B");
    checkCuda(cudaSetDevice(2), "cudaSetDevice 2");
    checkCuda(cudaMemcpy(h_verifyC, d_C, sample_bytes, cudaMemcpyDeviceToHost), "verify D→H d_C");
    checkCuda(cudaSetDevice(3), "cudaSetDevice 3");
    checkCuda(cudaMemcpy(h_verifyD, d_D, sample_bytes, cudaMemcpyDeviceToHost), "verify D→H d_D");

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
    checkCuda(cudaSetDevice(0), "cudaSetDevice 0"); checkCuda(cudaFree(d_A), "cudaFree d_A");
    checkCuda(cudaSetDevice(1), "cudaSetDevice 1"); checkCuda(cudaFree(d_B), "cudaFree d_B");
    checkCuda(cudaSetDevice(2), "cudaSetDevice 2"); checkCuda(cudaFree(d_C), "cudaFree d_C");
    checkCuda(cudaSetDevice(3), "cudaSetDevice 3"); checkCuda(cudaFree(d_D), "cudaFree d_D");
    for (int i = 0; i < 4; ++i)
        checkCuda(cudaStreamDestroy(stream[i]), "cudaStreamDestroy");
    free(h_A); free(h_B); free(h_C); free(h_D);
    return EXIT_SUCCESS;
}
