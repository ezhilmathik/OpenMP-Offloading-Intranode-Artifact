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
    double *h_B = (double*) malloc(bytes); // dev1 dst: receives from dev0 (1.0)
    if (!h_A || !h_B) {
        cerr << "Error: malloc failed for " << bytes << " bytes\n";
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;  // will be overwritten by d_A (1.0) via P2P
    }

    // ── Device allocation + H→D population ───────────────────────────────────
    // Transfer map:
    //   d_B (dev1) ← d_A (dev0): expect 1.0

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

    // ── Stream creation (one per device) ─────────────────────────────────────
    cudaStream_t stream[2];
    for (int i = 0; i < 2; ++i) {
        checkCuda(cudaSetDevice(i),             "cudaSetDevice stream init");
        checkCuda(cudaStreamCreate(&stream[i]), "cudaStreamCreate");
    }
   
    // ── Warm-up P2P transfers (not timed) ────────────────────────────────────
    for (int w = 0; w < 2; ++w)
      {
	// ── Warm-up transfers (not timed) ────────────────────────────────────────
	checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
	checkCuda(cudaMemcpyPeerAsync(d_B, 1, d_A, 0, bytes, stream[0]), "warmup P2P d_A→d_B");
	checkCuda(cudaStreamSynchronize(stream[0]), "cudaStreamSynchronize warmup 2");
      }
    
    // ── Timed P2P transfer (dev0 → dev1) ─────────────────────────────────────
    double start = omp_get_wtime();
    checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
    checkCuda(cudaMemcpyPeerAsync(d_B, 1, d_A, 0, bytes, stream[0]), "timed cudaMemcpyPeerAsync d_A→d_B");
    checkCuda(cudaStreamSynchronize(stream[0]), "cudaStreamSynchronize timed");
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
    cout << "P2P Bandwidth:     " << bandwidthGBps                             << " GB/s\n";

    // ── Spot-check verification ───────────────────────────────────────────────
    // d_B lives on dev1 — should now contain 1.0 (copied from d_A on dev0)
    size_t sample_size  = min(N, (size_t) 1000);
    size_t sample_bytes = sample_size * sizeof(double);

    double *h_verify = (double*) malloc(sample_bytes);
    if (!h_verify) {
        cerr << "Error: malloc failed for verification buffer\n";
    } else {
        checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
        checkCuda(cudaMemcpy(h_verify, d_B, sample_bytes, cudaMemcpyDeviceToHost),
                  "verify cudaMemcpy D→H d_B");

        srand(42);
        bool pass = true;
        for (size_t s = 0; s < sample_size; ++s) {
            size_t i = (size_t) rand() % sample_size;
            if (h_verify[i] != 1.0) {
                cerr << "d_B (dev0→dev1) mismatch at index " << i
                     << ": expected 1.0, got " << h_verify[i] << "\n";
                pass = false;
                break;
            }
        }
        cout << "Verification d_B (dev0→dev1): "
             << (pass ? "PASSED" : "FAILED")
             << " (" << sample_size << "/" << N << " samples)\n";
        free(h_verify);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    checkCuda(cudaSetDevice(0), "cudaSetDevice 0"); checkCuda(cudaFree(d_A), "cudaFree d_A");
    checkCuda(cudaSetDevice(1), "cudaSetDevice 1"); checkCuda(cudaFree(d_B), "cudaFree d_B");
    checkCuda(cudaStreamDestroy(stream[0]), "cudaStreamDestroy stream[0]");
    checkCuda(cudaStreamDestroy(stream[1]), "cudaStreamDestroy stream[1]");
    free(h_A); free(h_B);
    return EXIT_SUCCESS;
}
