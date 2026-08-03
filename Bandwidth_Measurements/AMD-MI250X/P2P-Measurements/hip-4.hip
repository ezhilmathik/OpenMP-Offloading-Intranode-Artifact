#include "hip/hip_runtime.h"
//-*-c++-*-
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <climits>    // ← fixes SIZE_MAX with nvcc
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
    if (num_devices < 4) {
        cerr << "Error: need at least 4 CUDA devices, found "
             << num_devices << "\n";
        return EXIT_FAILURE;
    }

    // ── Enable P2P access between all device pairs ────────────────────────────
    for (int i = 0; i < 4; ++i) {
        checkCuda(hipSetDevice(i), "hipSetDevice P2P");
        for (int j = 0; j < 4; ++j) {
            if (i == j) continue;
            int canAccess = 0;
            checkCuda(hipDeviceCanAccessPeer(&canAccess, i, j), "hipDeviceCanAccessPeer");
            if (canAccess) {
                hipError_t err = hipDeviceEnablePeerAccess(j, 0);
                if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled)
                    cerr << "Warning: P2P enable failed device " << i
                         << " -> device " << j << ": "
                         << hipGetErrorString(err) << "\n";
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

    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipMalloc(&d_A1, bytes), "hipMalloc d_A1");
    checkCuda(hipMalloc(&d_B1, bytes), "hipMalloc d_B1"); // dst: receives from dev1 (2.0)
    checkCuda(hipMalloc(&d_C1, bytes), "hipMalloc d_C1"); // dst: receives from dev2 (3.0)
    checkCuda(hipMalloc(&d_D1, bytes), "hipMalloc d_D1"); // dst: receives from dev3 (4.0)
    checkCuda(hipMemcpy(d_A1, h_A1, bytes, hipMemcpyHostToDevice), "H→D d_A1");
    checkCuda(hipMemcpy(d_B1, h_B1, bytes, hipMemcpyHostToDevice), "H→D d_B1");
    checkCuda(hipMemcpy(d_C1, h_C1, bytes, hipMemcpyHostToDevice), "H→D d_C1");
    checkCuda(hipMemcpy(d_D1, h_D1, bytes, hipMemcpyHostToDevice), "H→D d_D1");

    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipMalloc(&d_A2, bytes), "hipMalloc d_A2");
    checkCuda(hipMalloc(&d_B2, bytes), "hipMalloc d_B2"); // src: 2.0
    checkCuda(hipMalloc(&d_C2, bytes), "hipMalloc d_C2");
    checkCuda(hipMalloc(&d_D2, bytes), "hipMalloc d_D2");
    checkCuda(hipMemcpy(d_A2, h_A2, bytes, hipMemcpyHostToDevice), "H→D d_A2");
    checkCuda(hipMemcpy(d_B2, h_B2, bytes, hipMemcpyHostToDevice), "H→D d_B2");
    checkCuda(hipMemcpy(d_C2, h_C2, bytes, hipMemcpyHostToDevice), "H→D d_C2");
    checkCuda(hipMemcpy(d_D2, h_D2, bytes, hipMemcpyHostToDevice), "H→D d_D2");

    checkCuda(hipSetDevice(2), "hipSetDevice 2");
    checkCuda(hipMalloc(&d_A3, bytes), "hipMalloc d_A3");
    checkCuda(hipMalloc(&d_B3, bytes), "hipMalloc d_B3");
    checkCuda(hipMalloc(&d_C3, bytes), "hipMalloc d_C3"); // src: 3.0
    checkCuda(hipMalloc(&d_D3, bytes), "hipMalloc d_D3");
    checkCuda(hipMemcpy(d_A3, h_A3, bytes, hipMemcpyHostToDevice), "H→D d_A3");
    checkCuda(hipMemcpy(d_B3, h_B3, bytes, hipMemcpyHostToDevice), "H→D d_B3");
    checkCuda(hipMemcpy(d_C3, h_C3, bytes, hipMemcpyHostToDevice), "H→D d_C3");
    checkCuda(hipMemcpy(d_D3, h_D3, bytes, hipMemcpyHostToDevice), "H→D d_D3");

    checkCuda(hipSetDevice(3), "hipSetDevice 3");
    checkCuda(hipMalloc(&d_A4, bytes), "hipMalloc d_A4");
    checkCuda(hipMalloc(&d_B4, bytes), "hipMalloc d_B4");
    checkCuda(hipMalloc(&d_C4, bytes), "hipMalloc d_C4");
    checkCuda(hipMalloc(&d_D4, bytes), "hipMalloc d_D4"); // src: 4.0
    checkCuda(hipMemcpy(d_A4, h_A4, bytes, hipMemcpyHostToDevice), "H→D d_A4");
    checkCuda(hipMemcpy(d_B4, h_B4, bytes, hipMemcpyHostToDevice), "H→D d_B4");
    checkCuda(hipMemcpy(d_C4, h_C4, bytes, hipMemcpyHostToDevice), "H→D d_C4");
    checkCuda(hipMemcpy(d_D4, h_D4, bytes, hipMemcpyHostToDevice), "H→D d_D4");

    // ── Stream creation (one per device) ─────────────────────────────────────
    hipStream_t stream[4];
    for (int i = 0; i < 4; ++i) {
        checkCuda(hipSetDevice(i),             "hipSetDevice stream init");
        checkCuda(hipStreamCreate(&stream[i]), "hipStreamCreate");
    }

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for (int w = 0; w < 2; ++w) {
        checkCuda(hipSetDevice(1), "hipSetDevice 1");
        checkCuda(hipMemcpyPeerAsync(d_B1, 0, d_B2, 1, bytes, stream[1]), "warmup P2P d_B2→d_B1");
        checkCuda(hipStreamSynchronize(stream[1]), "hipStreamSynchronize warmup stream[1]");
        checkCuda(hipSetDevice(2), "hipSetDevice 2");
        checkCuda(hipMemcpyPeerAsync(d_C1, 0, d_C3, 2, bytes, stream[2]), "warmup P2P d_C3→d_C1");
        checkCuda(hipStreamSynchronize(stream[2]), "hipStreamSynchronize warmup stream[2]");
        checkCuda(hipSetDevice(3), "hipSetDevice 3");
        checkCuda(hipMemcpyPeerAsync(d_D1, 0, d_D4, 3, bytes, stream[3]), "warmup P2P d_D4→d_D1");
        checkCuda(hipStreamSynchronize(stream[3]), "hipStreamSynchronize warmup stream[3]");
    }

    // ── Timed P2P transfer (dev1,2,3 → dev0, parallel) ───────────────────────
    omp_set_num_threads(3);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            checkCuda(hipSetDevice(1), "hipSetDevice 1");
            checkCuda(hipMemcpyPeerAsync(d_B1, 0, d_B2, 1, bytes, stream[1]), "timed P2P d_B2→d_B1");
            checkCuda(hipStreamSynchronize(stream[1]), "hipStreamSynchronize timed stream[1]");
        }
        #pragma omp section
        {
            checkCuda(hipSetDevice(2), "hipSetDevice 2");
            checkCuda(hipMemcpyPeerAsync(d_C1, 0, d_C3, 2, bytes, stream[2]), "timed P2P d_C3→d_C1");
            checkCuda(hipStreamSynchronize(stream[2]), "hipStreamSynchronize timed stream[2]");
        }
        #pragma omp section
        {
            checkCuda(hipSetDevice(3), "hipSetDevice 3");
            checkCuda(hipMemcpyPeerAsync(d_D1, 0, d_D4, 3, bytes, stream[3]), "timed P2P d_D4→d_D1");
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

        checkCuda(hipSetDevice(0), "hipSetDevice 0 verify");
        srand(42);
        for (auto &ck : checks) {
            checkCuda(hipMemcpy(h_verify, ck.ptr, sample_bytes, hipMemcpyDeviceToHost),
                      "verify hipMemcpy D→H");
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
    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipFree(d_A1), "hipFree d_A1"); checkCuda(hipFree(d_B1), "hipFree d_B1");
    checkCuda(hipFree(d_C1), "hipFree d_C1"); checkCuda(hipFree(d_D1), "hipFree d_D1");

    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipFree(d_A2), "hipFree d_A2"); checkCuda(hipFree(d_B2), "hipFree d_B2");
    checkCuda(hipFree(d_C2), "hipFree d_C2"); checkCuda(hipFree(d_D2), "hipFree d_D2");

    checkCuda(hipSetDevice(2), "hipSetDevice 2");
    checkCuda(hipFree(d_A3), "hipFree d_A3"); checkCuda(hipFree(d_B3), "hipFree d_B3");
    checkCuda(hipFree(d_C3), "hipFree d_C3"); checkCuda(hipFree(d_D3), "hipFree d_D3");

    checkCuda(hipSetDevice(3), "hipSetDevice 3");
    checkCuda(hipFree(d_A4), "hipFree d_A4"); checkCuda(hipFree(d_B4), "hipFree d_B4");
    checkCuda(hipFree(d_C4), "hipFree d_C4"); checkCuda(hipFree(d_D4), "hipFree d_D4");

    for (int i = 0; i < 4; ++i)
        checkCuda(hipStreamDestroy(stream[i]), "hipStreamDestroy");

    free(h_A1); free(h_B1); free(h_C1); free(h_D1);
    free(h_A2); free(h_B2); free(h_C2); free(h_D2);
    free(h_A3); free(h_B3); free(h_C3); free(h_D3);
    free(h_A4); free(h_B4); free(h_C4); free(h_D4);
    return EXIT_SUCCESS;
}
