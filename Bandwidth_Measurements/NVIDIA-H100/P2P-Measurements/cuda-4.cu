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
    if (num_devices < 4) {
        cerr << "Error: need at least 4 CUDA devices, found "
             << num_devices << "\n";
        return EXIT_FAILURE;
    }

    // ── Enable P2P access between all device pairs ────────────────────────────
    for (int i = 0; i < 4; ++i) {
        checkCuda(cudaSetDevice(i), "cudaSetDevice P2P");
        for (int j = 0; j < 4; ++j) {
            if (i == j) continue;
            int canAccess = 0;
            checkCuda(cudaDeviceCanAccessPeer(&canAccess, i, j), "cudaDeviceCanAccessPeer");
            if (canAccess) {
                cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
                    cerr << "Warning: P2P enable failed device " << i
                         << " -> device " << j << ": "
                         << cudaGetErrorString(err) << "\n";
            } else {
                cerr << "Warning: P2P not supported device " << i
                     << " -> device " << j << "\n";
            }
        }
    }

    // ── Host allocation ──────────────────────────────────────────────────────
    double *h_A1 = (double*) malloc(bytes); double *h_A2 = (double*) malloc(bytes);
    double *h_A3 = (double*) malloc(bytes); double *h_A4 = (double*) malloc(bytes);
    double *h_B1 = (double*) malloc(bytes); double *h_B2 = (double*) malloc(bytes);
    double *h_B3 = (double*) malloc(bytes); double *h_B4 = (double*) malloc(bytes);
    double *h_C1 = (double*) malloc(bytes); double *h_C2 = (double*) malloc(bytes);
    double *h_C3 = (double*) malloc(bytes); double *h_C4 = (double*) malloc(bytes);
    double *h_D1 = (double*) malloc(bytes); double *h_D2 = (double*) malloc(bytes);
    double *h_D3 = (double*) malloc(bytes); double *h_D4 = (double*) malloc(bytes);

    if (!h_A1 || !h_A2 || !h_A3 || !h_A4 ||
        !h_B1 || !h_B2 || !h_B3 || !h_B4 ||
        !h_C1 || !h_C2 || !h_C3 || !h_C4 ||
        !h_D1 || !h_D2 || !h_D3 || !h_D4) {
        cerr << "Error: malloc failed for host buffers\n";
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i) {
        h_A1[i] = h_B1[i] = h_C1[i] = h_D1[i] = 1.0;
        h_A2[i] = h_B2[i] = h_C2[i] = h_D2[i] = 2.0;
        h_A3[i] = h_B3[i] = h_C3[i] = h_D3[i] = 3.0;
        h_A4[i] = h_B4[i] = h_C4[i] = h_D4[i] = 4.0;
    }

    // ── Device allocation + H→D population ───────────────────────────────────
    // Transfer map (all destinations on device 0):
    //   d_B1 (dev0) ← d_B2 (dev1): expect 2.0
    //   d_C1 (dev0) ← d_C3 (dev2): expect 3.0
    //   d_D1 (dev0) ← d_D4 (dev3): expect 4.0

    double *d_A1 = nullptr, *d_B1 = nullptr, *d_C1 = nullptr, *d_D1 = nullptr;
    double *d_A2 = nullptr, *d_B2 = nullptr, *d_C2 = nullptr, *d_D2 = nullptr;
    double *d_A3 = nullptr, *d_B3 = nullptr, *d_C3 = nullptr, *d_D3 = nullptr;
    double *d_A4 = nullptr, *d_B4 = nullptr, *d_C4 = nullptr, *d_D4 = nullptr;

    checkCuda(cudaSetDevice(0), "cudaSetDevice 0");
    checkCuda(cudaMalloc(&d_A1, bytes), "cudaMalloc d_A1");
    checkCuda(cudaMalloc(&d_B1, bytes), "cudaMalloc d_B1"); // dst: receives from dev1 (2.0)
    checkCuda(cudaMalloc(&d_C1, bytes), "cudaMalloc d_C1"); // dst: receives from dev2 (3.0)
    checkCuda(cudaMalloc(&d_D1, bytes), "cudaMalloc d_D1"); // dst: receives from dev3 (4.0)
    checkCuda(cudaMemcpy(d_A1, h_A1, bytes, cudaMemcpyHostToDevice), "H→D d_A1");
    checkCuda(cudaMemcpy(d_B1, h_B1, bytes, cudaMemcpyHostToDevice), "H→D d_B1");
    checkCuda(cudaMemcpy(d_C1, h_C1, bytes, cudaMemcpyHostToDevice), "H→D d_C1");
    checkCuda(cudaMemcpy(d_D1, h_D1, bytes, cudaMemcpyHostToDevice), "H→D d_D1");

    checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
    checkCuda(cudaMalloc(&d_A2, bytes), "cudaMalloc d_A2");
    checkCuda(cudaMalloc(&d_B2, bytes), "cudaMalloc d_B2"); // src: 2.0
    checkCuda(cudaMalloc(&d_C2, bytes), "cudaMalloc d_C2");
    checkCuda(cudaMalloc(&d_D2, bytes), "cudaMalloc d_D2");
    checkCuda(cudaMemcpy(d_A2, h_A2, bytes, cudaMemcpyHostToDevice), "H→D d_A2");
    checkCuda(cudaMemcpy(d_B2, h_B2, bytes, cudaMemcpyHostToDevice), "H→D d_B2");
    checkCuda(cudaMemcpy(d_C2, h_C2, bytes, cudaMemcpyHostToDevice), "H→D d_C2");
    checkCuda(cudaMemcpy(d_D2, h_D2, bytes, cudaMemcpyHostToDevice), "H→D d_D2");

    checkCuda(cudaSetDevice(2), "cudaSetDevice 2");
    checkCuda(cudaMalloc(&d_A3, bytes), "cudaMalloc d_A3");
    checkCuda(cudaMalloc(&d_B3, bytes), "cudaMalloc d_B3");
    checkCuda(cudaMalloc(&d_C3, bytes), "cudaMalloc d_C3"); // src: 3.0
    checkCuda(cudaMalloc(&d_D3, bytes), "cudaMalloc d_D3");
    checkCuda(cudaMemcpy(d_A3, h_A3, bytes, cudaMemcpyHostToDevice), "H→D d_A3");
    checkCuda(cudaMemcpy(d_B3, h_B3, bytes, cudaMemcpyHostToDevice), "H→D d_B3");
    checkCuda(cudaMemcpy(d_C3, h_C3, bytes, cudaMemcpyHostToDevice), "H→D d_C3");
    checkCuda(cudaMemcpy(d_D3, h_D3, bytes, cudaMemcpyHostToDevice), "H→D d_D3");

    checkCuda(cudaSetDevice(3), "cudaSetDevice 3");
    checkCuda(cudaMalloc(&d_A4, bytes), "cudaMalloc d_A4");
    checkCuda(cudaMalloc(&d_B4, bytes), "cudaMalloc d_B4");
    checkCuda(cudaMalloc(&d_C4, bytes), "cudaMalloc d_C4");
    checkCuda(cudaMalloc(&d_D4, bytes), "cudaMalloc d_D4"); // src: 4.0
    checkCuda(cudaMemcpy(d_A4, h_A4, bytes, cudaMemcpyHostToDevice), "H→D d_A4");
    checkCuda(cudaMemcpy(d_B4, h_B4, bytes, cudaMemcpyHostToDevice), "H→D d_B4");
    checkCuda(cudaMemcpy(d_C4, h_C4, bytes, cudaMemcpyHostToDevice), "H→D d_C4");
    checkCuda(cudaMemcpy(d_D4, h_D4, bytes, cudaMemcpyHostToDevice), "H→D d_D4");

    // ── Stream creation (one per device) ─────────────────────────────────────
    cudaStream_t stream[4];
    for (int i = 0; i < 4; ++i) {
        checkCuda(cudaSetDevice(i),             "cudaSetDevice stream init");
        checkCuda(cudaStreamCreate(&stream[i]), "cudaStreamCreate");
    }

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for (int w = 0; w < 2; ++w) {
        checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
        checkCuda(cudaMemcpyPeerAsync(d_B1, 0, d_B2, 1, bytes, stream[1]), "warmup P2P d_B2→d_B1");
        checkCuda(cudaStreamSynchronize(stream[1]), "cudaStreamSynchronize warmup stream[1]");
        checkCuda(cudaSetDevice(2), "cudaSetDevice 2");
        checkCuda(cudaMemcpyPeerAsync(d_C1, 0, d_C3, 2, bytes, stream[2]), "warmup P2P d_C3→d_C1");
        checkCuda(cudaStreamSynchronize(stream[2]), "cudaStreamSynchronize warmup stream[2]");
        checkCuda(cudaSetDevice(3), "cudaSetDevice 3");
        checkCuda(cudaMemcpyPeerAsync(d_D1, 0, d_D4, 3, bytes, stream[3]), "warmup P2P d_D4→d_D1");
        checkCuda(cudaStreamSynchronize(stream[3]), "cudaStreamSynchronize warmup stream[3]");
    }

    // ── Timed P2P transfer (dev1,2,3 → dev0, parallel) ───────────────────────
    omp_set_num_threads(3);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
            checkCuda(cudaMemcpyPeerAsync(d_B1, 0, d_B2, 1, bytes, stream[1]), "timed P2P d_B2→d_B1");
            checkCuda(cudaStreamSynchronize(stream[1]), "cudaStreamSynchronize timed stream[1]");
        }
        #pragma omp section
        {
            checkCuda(cudaSetDevice(2), "cudaSetDevice 2");
            checkCuda(cudaMemcpyPeerAsync(d_C1, 0, d_C3, 2, bytes, stream[2]), "timed P2P d_C3→d_C1");
            checkCuda(cudaStreamSynchronize(stream[2]), "cudaStreamSynchronize timed stream[2]");
        }
        #pragma omp section
        {
            checkCuda(cudaSetDevice(3), "cudaSetDevice 3");
            checkCuda(cudaMemcpyPeerAsync(d_D1, 0, d_D4, 3, bytes, stream[3]), "timed P2P d_D4→d_D1");
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
    double total_bytes   = 3.0 * static_cast<double>(bytes); // 3 P2P transfers
    double bandwidthGBps = total_bytes / (seconds * 1e9);

    cout << fixed << setprecision(6);
    cout << "Elapsed time:      " << seconds                                   << " s\n";
    cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    cout << "P2P Bandwidth:     " << bandwidthGBps                             << " GB/s\n";

    // ── Spot-check verification ───────────────────────────────────────────────
    // All 3 destination buffers live on device 0
    //   d_B1 ← d_B2 (dev1): expect 2.0
    //   d_C1 ← d_C3 (dev2): expect 3.0
    //   d_D1 ← d_D4 (dev3): expect 4.0
    size_t sample_size  = min(N, (size_t) 1000);
    size_t sample_bytes = sample_size * sizeof(double);

    double *h_verify = (double*) malloc(sample_bytes);
    if (!h_verify) {
        cerr << "Error: malloc failed for verification buffer\n";
    } else {
        struct Check { double *ptr; double expected; const char *label; };
        Check checks[] = {
            { d_B1, 2.0, "d_B1 (dev1→dev0)" },
            { d_C1, 3.0, "d_C1 (dev2→dev0)" },
            { d_D1, 4.0, "d_D1 (dev3→dev0)" },
        };

        checkCuda(cudaSetDevice(0), "cudaSetDevice 0 verify");
        srand(42);
        for (auto &ck : checks) {
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
    checkCuda(cudaFree(d_A1), "cudaFree d_A1"); checkCuda(cudaFree(d_B1), "cudaFree d_B1");
    checkCuda(cudaFree(d_C1), "cudaFree d_C1"); checkCuda(cudaFree(d_D1), "cudaFree d_D1");

    checkCuda(cudaSetDevice(1), "cudaSetDevice 1");
    checkCuda(cudaFree(d_A2), "cudaFree d_A2"); checkCuda(cudaFree(d_B2), "cudaFree d_B2");
    checkCuda(cudaFree(d_C2), "cudaFree d_C2"); checkCuda(cudaFree(d_D2), "cudaFree d_D2");

    checkCuda(cudaSetDevice(2), "cudaSetDevice 2");
    checkCuda(cudaFree(d_A3), "cudaFree d_A3"); checkCuda(cudaFree(d_B3), "cudaFree d_B3");
    checkCuda(cudaFree(d_C3), "cudaFree d_C3"); checkCuda(cudaFree(d_D3), "cudaFree d_D3");

    checkCuda(cudaSetDevice(3), "cudaSetDevice 3");
    checkCuda(cudaFree(d_A4), "cudaFree d_A4"); checkCuda(cudaFree(d_B4), "cudaFree d_B4");
    checkCuda(cudaFree(d_C4), "cudaFree d_C4"); checkCuda(cudaFree(d_D4), "cudaFree d_D4");

    for (int i = 0; i < 4; ++i)
        checkCuda(cudaStreamDestroy(stream[i]), "cudaStreamDestroy");

    free(h_A1); free(h_B1); free(h_C1); free(h_D1);
    free(h_A2); free(h_B2); free(h_C2); free(h_D2);
    free(h_A3); free(h_B3); free(h_C3); free(h_D3);
    free(h_A4); free(h_B4); free(h_C4); free(h_D4);
    return EXIT_SUCCESS;
}
