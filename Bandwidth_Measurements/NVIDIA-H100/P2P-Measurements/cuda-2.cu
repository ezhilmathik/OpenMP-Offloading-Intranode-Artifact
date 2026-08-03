//-*-c++-*-
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdint>
#include <climits>    // ← fixes SIZE_MAX with nvcc
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
    if (num_devices < 2) {
        cerr << "Error: need at least 2 CUDA devices, found "
             << num_devices << "\n";
        return EXIT_FAILURE;
    }

    // ── Enable P2P access (both directions) ───────────────────────────────────
    for (int i = 0; i < 2; ++i) {
        checkCuda(cudaSetDevice(i), "cudaSetDevice P2P");
        for (int j = 0; j < 2; ++j) {
            if (i == j) continue;
            int canAccess = 0;
            checkCuda(cudaDeviceCanAccessPeer(&canAccess, i, j), "cudaDeviceCanAccessPeer");
            if (canAccess)
                checkCuda(cudaDeviceEnablePeerAccess(j, 0), "cudaDeviceEnablePeerAccess");
            else
                cerr << "Warning: P2P not supported device " << i
                     << " -> device " << j << "\n";
        }
    }

    // ── Host allocation ──────────────────────────────────────────────────────
    double *h_A = (double*) malloc(bytes); // dev0 src: 1.0
    double *h_B = (double*) malloc(bytes); // dev1 src: 2.0
    double *h_C = (double*) malloc(bytes); // dev1 dst: receives from dev0 (1.0)
    double *h_D = (double*) malloc(bytes); // dev0 dst: receives from dev1 (2.0)
    if (!h_A || !h_B || !h_C || !h_D) {
        cerr << "Error: malloc failed for " << bytes << " bytes\n";
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
        h_C[i] = 3.0;  // will be overwritten by d_A (1.0)
        h_D[i] = 4.0;  // will be overwritten by d_B (2.0)
    }

    // ── Device allocation + H→D population ───────────────────────────────────
    // Transfer map:
    //   d_C (dev1) ← d_A (dev0): expect 1.0
    //   d_D (dev0) ← d_B (dev1): expect 2.0

    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;

    checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
    checkCuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc(&d_D, bytes), "cudaMalloc d_D");
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "initial H→D d_A");
    checkCuda(cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice), "initial H→D d_D");

    checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
    checkCuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    checkCuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");
    if (!d_B || !d_C) {
        cerr << "Error: cudaMalloc failed on device 1\n";
        checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
        checkCuda(cudaFree(d_A), "cudaFree d_A");
        checkCuda(cudaFree(d_D), "cudaFree d_D");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "initial H→D d_B");
    checkCuda(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice), "initial H→D d_C");

    // ── Stream creation (one per device) ─────────────────────────────────────
    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i) {
        checkCuda(cudaSetDevice(i),             "cudaSetDevice stream init");
        checkCuda(cudaStreamCreate(&stream[i]), "cudaStreamCreate");
    }

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for (int w = 0; w < 2; ++w) {
        checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
        checkCuda(cudaMemcpyPeerAsync(d_C, 1, d_A, 0, bytes, stream[0]), "warmup P2P d_A→d_C");
        checkCuda(cudaStreamSynchronize(stream[0]), "cudaStreamSynchronize warmup stream[0]");
        checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
        checkCuda(cudaMemcpyPeerAsync(d_D, 0, d_B, 1, bytes, stream[1]), "warmup P2P d_B→d_D");
        checkCuda(cudaStreamSynchronize(stream[1]), "cudaStreamSynchronize warmup stream[1]");
    }

    // ── Timed P2P transfer (both devices simultaneously) ─────────────────────
    omp_set_num_threads(2);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
            checkCuda(cudaMemcpyPeerAsync(d_C, 1, d_A, 0, bytes, stream[0]), "timed P2P d_A→d_C");
            checkCuda(cudaStreamSynchronize(stream[0]), "cudaStreamSynchronize timed stream[0]");
        }
        #pragma omp section
        {
            checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
            checkCuda(cudaMemcpyPeerAsync(d_D, 0, d_B, 1, bytes, stream[1]), "timed P2P d_B→d_D");
            checkCuda(cudaStreamSynchronize(stream[1]), "cudaStreamSynchronize timed stream[1]");
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
    double total_bytes   = 2.0 * static_cast<double>(bytes); // 2 P2P transfers
    double bandwidthGBps = total_bytes / (seconds * 1e9);

    cout << fixed << setprecision(6);
    cout << "Elapsed time:      " << seconds                                   << " s\n";
    cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    cout << "P2P Bandwidth:     " << bandwidthGBps                             << " GB/s\n";

    // ── Spot-check verification ───────────────────────────────────────────────
    // d_C (dev1) ← d_A (dev0): expect 1.0
    // d_D (dev0) ← d_B (dev1): expect 2.0
    size_t sample_size  = min(N, (size_t) 1000);
    size_t sample_bytes = sample_size * sizeof(double);

    double *h_verify = (double*) malloc(sample_bytes);
    if (!h_verify) {
        cerr << "Error: malloc failed for verification buffer\n";
    } else {
        struct Check { int dev; double *ptr; double expected; const char *label; };
        Check checks[] = {
            { 1, d_C, 1.0, "d_C (dev0→dev1)" },
            { 0, d_D, 2.0, "d_D (dev1→dev0)" },
        };

        srand(42);
        for (auto &ck : checks) {
            checkCuda(cudaSetDevice(ck.dev), "cudaSetDevice verify");
            checkCuda(cudaMemcpy(h_verify, ck.ptr, sample_bytes, cudaMemcpyDeviceToHost),
                      "verify cudaMemcpy D→H");
            bool pass = true;
            for (size_t s = 0; s < sample_size; ++s) {
                size_t i = (size_t) rand() % sample_size;
                if (h_verify[i] != ck.expected) {
                    cerr << ck.label << " mismatch at index " << i
                         << ": expected " << ck.expected
                         << ", got "      << h_verify[i] << "\n";
                    pass = false;
                    break;
                }
            }
            cout << "Verification " << ck.label << ": "
                 << (pass ? "PASSED" : "FAILED")
                 << " (" << sample_size << "/" << N << " samples)\n";
        }
        free(h_verify);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
    checkCuda(cudaFree(d_A), "cudaFree d_A");
    checkCuda(cudaFree(d_D), "cudaFree d_D");
    checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
    checkCuda(cudaFree(d_B), "cudaFree d_B");
    checkCuda(cudaFree(d_C), "cudaFree d_C");
    checkCuda(cudaStreamDestroy(stream[0]), "cudaStreamDestroy stream[0]");
    checkCuda(cudaStreamDestroy(stream[1]), "cudaStreamDestroy stream[1]");
    free(h_A); free(h_B); free(h_C); free(h_D);
    return EXIT_SUCCESS;
}
