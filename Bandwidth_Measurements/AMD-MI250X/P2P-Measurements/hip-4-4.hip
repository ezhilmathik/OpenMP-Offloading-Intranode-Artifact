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
    // Transfer map (12 all-to-all P2P transfers):
    //   dev0→dev1 (d_A2)  src=d_B11(1.0)  dev0→dev2 (d_A3)  src=d_C11(1.0)
    //   dev0→dev3 (d_A4)  src=d_D11(1.0)  dev1→dev0 (d_B1)  src=d_A22(2.0)
    //   dev1→dev2 (d_B3)  src=d_C22(2.0)  dev1→dev3 (d_B4)  src=d_D22(2.0)
    //   dev2→dev0 (d_C1)  src=d_A33(3.0)  dev2→dev1 (d_C2)  src=d_B33(3.0)
    //   dev2→dev3 (d_C4)  src=d_D33(3.0)  dev3→dev0 (d_D1)  src=d_A44(4.0)
    //   dev3→dev1 (d_D2)  src=d_B44(4.0)  dev3→dev2 (d_D3)  src=d_C44(4.0)

    double *d_A1  = nullptr, *d_B1  = nullptr, *d_C1  = nullptr, *d_D1  = nullptr;
    double *d_A2  = nullptr, *d_B2  = nullptr, *d_C2  = nullptr, *d_D2  = nullptr;
    double *d_A3  = nullptr, *d_B3  = nullptr, *d_C3  = nullptr, *d_D3  = nullptr;
    double *d_A4  = nullptr, *d_B4  = nullptr, *d_C4  = nullptr, *d_D4  = nullptr;
    double *d_A11 = nullptr, *d_B11 = nullptr, *d_C11 = nullptr, *d_D11 = nullptr;
    double *d_A22 = nullptr, *d_B22 = nullptr, *d_C22 = nullptr, *d_D22 = nullptr;
    double *d_A33 = nullptr, *d_B33 = nullptr, *d_C33 = nullptr, *d_D33 = nullptr;
    double *d_A44 = nullptr, *d_B44 = nullptr, *d_C44 = nullptr, *d_D44 = nullptr;

    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipMalloc(&d_A1,  bytes), "hipMalloc d_A1");
    checkCuda(hipMalloc(&d_B1,  bytes), "hipMalloc d_B1");  // dst: receives from dev1 (2.0)
    checkCuda(hipMalloc(&d_C1,  bytes), "hipMalloc d_C1");  // dst: receives from dev2 (3.0)
    checkCuda(hipMalloc(&d_D1,  bytes), "hipMalloc d_D1");  // dst: receives from dev3 (4.0)
    checkCuda(hipMalloc(&d_A11, bytes), "hipMalloc d_A11");
    checkCuda(hipMalloc(&d_B11, bytes), "hipMalloc d_B11"); // src → dev1
    checkCuda(hipMalloc(&d_C11, bytes), "hipMalloc d_C11"); // src → dev2
    checkCuda(hipMalloc(&d_D11, bytes), "hipMalloc d_D11"); // src → dev3
    checkCuda(hipMemcpy(d_A1,  h_A1, bytes, hipMemcpyHostToDevice), "H→D d_A1");
    checkCuda(hipMemcpy(d_B1,  h_B1, bytes, hipMemcpyHostToDevice), "H→D d_B1");
    checkCuda(hipMemcpy(d_C1,  h_C1, bytes, hipMemcpyHostToDevice), "H→D d_C1");
    checkCuda(hipMemcpy(d_D1,  h_D1, bytes, hipMemcpyHostToDevice), "H→D d_D1");
    checkCuda(hipMemcpy(d_A11, h_A1, bytes, hipMemcpyHostToDevice), "H→D d_A11");
    checkCuda(hipMemcpy(d_B11, h_B1, bytes, hipMemcpyHostToDevice), "H→D d_B11");
    checkCuda(hipMemcpy(d_C11, h_C1, bytes, hipMemcpyHostToDevice), "H→D d_C11");
    checkCuda(hipMemcpy(d_D11, h_D1, bytes, hipMemcpyHostToDevice), "H→D d_D11");

    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipMalloc(&d_A2,  bytes), "hipMalloc d_A2");  // dst: receives from dev0 (1.0)
    checkCuda(hipMalloc(&d_B2,  bytes), "hipMalloc d_B2");
    checkCuda(hipMalloc(&d_C2,  bytes), "hipMalloc d_C2");  // dst: receives from dev2 (3.0)
    checkCuda(hipMalloc(&d_D2,  bytes), "hipMalloc d_D2");  // dst: receives from dev3 (4.0)
    checkCuda(hipMalloc(&d_A22, bytes), "hipMalloc d_A22"); // src → dev0
    checkCuda(hipMalloc(&d_B22, bytes), "hipMalloc d_B22");
    checkCuda(hipMalloc(&d_C22, bytes), "hipMalloc d_C22"); // src → dev2
    checkCuda(hipMalloc(&d_D22, bytes), "hipMalloc d_D22"); // src → dev3
    checkCuda(hipMemcpy(d_A2,  h_A2, bytes, hipMemcpyHostToDevice), "H→D d_A2");
    checkCuda(hipMemcpy(d_B2,  h_B2, bytes, hipMemcpyHostToDevice), "H→D d_B2");
    checkCuda(hipMemcpy(d_C2,  h_C2, bytes, hipMemcpyHostToDevice), "H→D d_C2");
    checkCuda(hipMemcpy(d_D2,  h_D2, bytes, hipMemcpyHostToDevice), "H→D d_D2");
    checkCuda(hipMemcpy(d_A22, h_A2, bytes, hipMemcpyHostToDevice), "H→D d_A22");
    checkCuda(hipMemcpy(d_B22, h_B2, bytes, hipMemcpyHostToDevice), "H→D d_B22");
    checkCuda(hipMemcpy(d_C22, h_C2, bytes, hipMemcpyHostToDevice), "H→D d_C22");
    checkCuda(hipMemcpy(d_D22, h_D2, bytes, hipMemcpyHostToDevice), "H→D d_D22");

    checkCuda(hipSetDevice(2), "hipSetDevice 2");
    checkCuda(hipMalloc(&d_A3,  bytes), "hipMalloc d_A3");  // dst: receives from dev0 (1.0)
    checkCuda(hipMalloc(&d_B3,  bytes), "hipMalloc d_B3");  // dst: receives from dev1 (2.0)
    checkCuda(hipMalloc(&d_C3,  bytes), "hipMalloc d_C3");
    checkCuda(hipMalloc(&d_D3,  bytes), "hipMalloc d_D3");  // dst: receives from dev3 (4.0)
    checkCuda(hipMalloc(&d_A33, bytes), "hipMalloc d_A33"); // src → dev0
    checkCuda(hipMalloc(&d_B33, bytes), "hipMalloc d_B33"); // src → dev1
    checkCuda(hipMalloc(&d_C33, bytes), "hipMalloc d_C33");
    checkCuda(hipMalloc(&d_D33, bytes), "hipMalloc d_D33"); // src → dev3
    checkCuda(hipMemcpy(d_A3,  h_A3, bytes, hipMemcpyHostToDevice), "H→D d_A3");
    checkCuda(hipMemcpy(d_B3,  h_B3, bytes, hipMemcpyHostToDevice), "H→D d_B3");
    checkCuda(hipMemcpy(d_C3,  h_C3, bytes, hipMemcpyHostToDevice), "H→D d_C3");
    checkCuda(hipMemcpy(d_D3,  h_D3, bytes, hipMemcpyHostToDevice), "H→D d_D3");
    checkCuda(hipMemcpy(d_A33, h_A3, bytes, hipMemcpyHostToDevice), "H→D d_A33");
    checkCuda(hipMemcpy(d_B33, h_B3, bytes, hipMemcpyHostToDevice), "H→D d_B33");
    checkCuda(hipMemcpy(d_C33, h_C3, bytes, hipMemcpyHostToDevice), "H→D d_C33");
    checkCuda(hipMemcpy(d_D33, h_D3, bytes, hipMemcpyHostToDevice), "H→D d_D33");

    checkCuda(hipSetDevice(3), "hipSetDevice 3");
    checkCuda(hipMalloc(&d_A4,  bytes), "hipMalloc d_A4");  // dst: receives from dev0 (1.0)
    checkCuda(hipMalloc(&d_B4,  bytes), "hipMalloc d_B4");  // dst: receives from dev1 (2.0)
    checkCuda(hipMalloc(&d_C4,  bytes), "hipMalloc d_C4");  // dst: receives from dev2 (3.0)
    checkCuda(hipMalloc(&d_D4,  bytes), "hipMalloc d_D4");
    checkCuda(hipMalloc(&d_A44, bytes), "hipMalloc d_A44"); // src → dev0
    checkCuda(hipMalloc(&d_B44, bytes), "hipMalloc d_B44"); // src → dev1
    checkCuda(hipMalloc(&d_C44, bytes), "hipMalloc d_C44"); // src → dev2
    checkCuda(hipMalloc(&d_D44, bytes), "hipMalloc d_D44");
    checkCuda(hipMemcpy(d_A4,  h_A4, bytes, hipMemcpyHostToDevice), "H→D d_A4");
    checkCuda(hipMemcpy(d_B4,  h_B4, bytes, hipMemcpyHostToDevice), "H→D d_B4");
    checkCuda(hipMemcpy(d_C4,  h_C4, bytes, hipMemcpyHostToDevice), "H→D d_C4");
    checkCuda(hipMemcpy(d_D4,  h_D4, bytes, hipMemcpyHostToDevice), "H→D d_D4");
    checkCuda(hipMemcpy(d_A44, h_A4, bytes, hipMemcpyHostToDevice), "H→D d_A44");
    checkCuda(hipMemcpy(d_B44, h_B4, bytes, hipMemcpyHostToDevice), "H→D d_B44");
    checkCuda(hipMemcpy(d_C44, h_C4, bytes, hipMemcpyHostToDevice), "H→D d_C44");
    checkCuda(hipMemcpy(d_D44, h_D4, bytes, hipMemcpyHostToDevice), "H→D d_D44");

    // ── Stream creation: one per ordered (src→dst) pair ──────────────────────
    hipStream_t stream[4][4];
    memset(stream, 0, sizeof(stream));
    for (int src = 0; src < 4; ++src) {
      checkCuda(hipSetDevice(src), "hipSetDevice stream init");
      for (int dst = 0; dst < 4; ++dst) {
        if (src == dst) continue;
        checkCuda(hipStreamCreate(&stream[src][dst]), "hipStreamCreate");
      }
    }

    // ── Warm-up P2P transfers (not timed) ────────────────────────────────────
    for (int w = 0; w < 2; ++w) {
      // Warm-up and timed section — device 0 block
      checkCuda(hipSetDevice(0), "hipSetDevice 0");
      checkCuda(hipMemcpyPeerAsync(d_A2, 1, d_B11, 0, bytes, stream[0][1]), "d_B11→d_A2");
      checkCuda(hipMemcpyPeerAsync(d_A3, 2, d_C11, 0, bytes, stream[0][2]), "d_C11→d_A3");
      checkCuda(hipMemcpyPeerAsync(d_A4, 3, d_D11, 0, bytes, stream[0][3]), "d_D11→d_A4");
      checkCuda(hipStreamSynchronize(stream[0][1]), "sync stream[0][1]");
      checkCuda(hipStreamSynchronize(stream[0][2]), "sync stream[0][2]");
      checkCuda(hipStreamSynchronize(stream[0][3]), "sync stream[0][3]");

      // device 1 block
      checkCuda(hipSetDevice(1), "hipSetDevice 1");
      checkCuda(hipMemcpyPeerAsync(d_B1, 0, d_A22, 1, bytes, stream[1][0]), "d_A22→d_B1");
      checkCuda(hipMemcpyPeerAsync(d_B3, 2, d_C22, 1, bytes, stream[1][2]), "d_C22→d_B3");
      checkCuda(hipMemcpyPeerAsync(d_B4, 3, d_D22, 1, bytes, stream[1][3]), "d_D22→d_B4");
      checkCuda(hipStreamSynchronize(stream[1][0]), "sync stream[1][0]");
      checkCuda(hipStreamSynchronize(stream[1][2]), "sync stream[1][2]");
      checkCuda(hipStreamSynchronize(stream[1][3]), "sync stream[1][3]");

      // device 2 block
      checkCuda(hipSetDevice(2), "hipSetDevice 2");
      checkCuda(hipMemcpyPeerAsync(d_C1, 0, d_A33, 2, bytes, stream[2][0]), "d_A33→d_C1");
      checkCuda(hipMemcpyPeerAsync(d_C2, 1, d_B33, 2, bytes, stream[2][1]), "d_B33→d_C2");
      checkCuda(hipMemcpyPeerAsync(d_C4, 3, d_D33, 2, bytes, stream[2][3]), "d_D33→d_C4");
      checkCuda(hipStreamSynchronize(stream[2][0]), "sync stream[2][0]");
      checkCuda(hipStreamSynchronize(stream[2][1]), "sync stream[2][1]");
      checkCuda(hipStreamSynchronize(stream[2][3]), "sync stream[2][3]");

      // device 3 block
      checkCuda(hipSetDevice(3), "hipSetDevice 3");
      checkCuda(hipMemcpyPeerAsync(d_D1, 0, d_A44, 3, bytes, stream[3][0]), "d_A44→d_D1");
      checkCuda(hipMemcpyPeerAsync(d_D2, 1, d_B44, 3, bytes, stream[3][1]), "d_B44→d_D2");
      checkCuda(hipMemcpyPeerAsync(d_D3, 2, d_C44, 3, bytes, stream[3][2]), "d_C44→d_D3");
      checkCuda(hipStreamSynchronize(stream[3][0]), "sync stream[3][0]");
      checkCuda(hipStreamSynchronize(stream[3][1]), "sync stream[3][1]");
      checkCuda(hipStreamSynchronize(stream[3][2]), "sync stream[3][2]");
    }
 
    // ── Timed P2P transfer (all 4 devices simultaneously) ────────────────────
    omp_set_num_threads(4);
    double start = omp_get_wtime();
#pragma omp parallel sections
    {
#pragma omp section
      {
	checkCuda(hipSetDevice(0), "hipSetDevice 0");
	checkCuda(hipMemcpyPeerAsync(d_A2, 1, d_B11, 0, bytes, stream[0][1]), "timed d_B11→d_A2");
	checkCuda(hipMemcpyPeerAsync(d_A3, 2, d_C11, 0, bytes, stream[0][2]), "timed d_C11→d_A3");
	checkCuda(hipMemcpyPeerAsync(d_A4, 3, d_D11, 0, bytes, stream[0][3]), "timed d_D11→d_A4");
	checkCuda(hipStreamSynchronize(stream[0][1]), "sync timed stream[0][1]");
	checkCuda(hipStreamSynchronize(stream[0][2]), "sync timed stream[0][2]");
	checkCuda(hipStreamSynchronize(stream[0][3]), "sync timed stream[0][3]");
      }
#pragma omp section
      {
	checkCuda(hipSetDevice(1), "hipSetDevice 1");
	checkCuda(hipMemcpyPeerAsync(d_B1, 0, d_A22, 1, bytes, stream[1][0]), "timed d_A22→d_B1");
	checkCuda(hipMemcpyPeerAsync(d_B3, 2, d_C22, 1, bytes, stream[1][2]), "timed d_C22→d_B3");
	checkCuda(hipMemcpyPeerAsync(d_B4, 3, d_D22, 1, bytes, stream[1][3]), "timed d_D22→d_B4");
	checkCuda(hipStreamSynchronize(stream[1][0]), "sync timed stream[1][0]");
	checkCuda(hipStreamSynchronize(stream[1][2]), "sync timed stream[1][2]");
	checkCuda(hipStreamSynchronize(stream[1][3]), "sync timed stream[1][3]");
      }
#pragma omp section
      {
	checkCuda(hipSetDevice(2), "hipSetDevice 2");
	checkCuda(hipMemcpyPeerAsync(d_C1, 0, d_A33, 2, bytes, stream[2][0]), "timed d_A33→d_C1");
	checkCuda(hipMemcpyPeerAsync(d_C2, 1, d_B33, 2, bytes, stream[2][1]), "timed d_B33→d_C2");
	checkCuda(hipMemcpyPeerAsync(d_C4, 3, d_D33, 2, bytes, stream[2][3]), "timed d_D33→d_C4");
	checkCuda(hipStreamSynchronize(stream[2][0]), "sync timed stream[2][0]");
	checkCuda(hipStreamSynchronize(stream[2][1]), "sync timed stream[2][1]");
	checkCuda(hipStreamSynchronize(stream[2][3]), "sync timed stream[2][3]");
      }
#pragma omp section
      {
	checkCuda(hipSetDevice(3), "hipSetDevice 3");
	checkCuda(hipMemcpyPeerAsync(d_D1, 0, d_A44, 3, bytes, stream[3][0]), "timed d_A44→d_D1");
	checkCuda(hipMemcpyPeerAsync(d_D2, 1, d_B44, 3, bytes, stream[3][1]), "timed d_B44→d_D2");
	checkCuda(hipMemcpyPeerAsync(d_D3, 2, d_C44, 3, bytes, stream[3][2]), "timed d_C44→d_D3");
	checkCuda(hipStreamSynchronize(stream[3][0]), "sync timed stream[3][0]");
	checkCuda(hipStreamSynchronize(stream[3][1]), "sync timed stream[3][1]");
	checkCuda(hipStreamSynchronize(stream[3][2]), "sync timed stream[3][2]");
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
    double total_bytes   = 12.0 * static_cast<double>(bytes); // 12 P2P transfers
    double bandwidthGBps = total_bytes / (seconds * 1e9);

    cout << fixed << setprecision(6);
    cout << "Elapsed time:      " << seconds                                   << " s\n";
    cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    cout << "P2P Bandwidth:     " << bandwidthGBps                             << " GB/s\n";

    // ── Spot-check verification ───────────────────────────────────────────────
    size_t sample_size  = min(N, (size_t) 1000);
    size_t sample_bytes = sample_size * sizeof(double);

    double *h_verify = (double*) malloc(sample_bytes);
    if (!h_verify) {
        cerr << "Error: malloc failed for verification buffer\n";
    } else {
        struct Check { int dev; double *ptr; double expected; const char *label; };
        Check checks[] = {
            { 1, d_A2, 1.0, "d_A2  (dev0→dev1)" },
            { 2, d_A3, 1.0, "d_A3  (dev0→dev2)" },
            { 3, d_A4, 1.0, "d_A4  (dev0→dev3)" },
            { 0, d_B1, 2.0, "d_B1  (dev1→dev0)" },
            { 2, d_B3, 2.0, "d_B3  (dev1→dev2)" },
            { 3, d_B4, 2.0, "d_B4  (dev1→dev3)" },
            { 0, d_C1, 3.0, "d_C1  (dev2→dev0)" },
            { 1, d_C2, 3.0, "d_C2  (dev2→dev1)" },
            { 3, d_C4, 3.0, "d_C4  (dev2→dev3)" },
            { 0, d_D1, 4.0, "d_D1  (dev3→dev0)" },
            { 1, d_D2, 4.0, "d_D2  (dev3→dev1)" },
            { 2, d_D3, 4.0, "d_D3  (dev3→dev2)" },
        };
        int num_checks = sizeof(checks) / sizeof(checks[0]);

        srand(42);
        for (int c = 0; c < num_checks; ++c) {
            checkCuda(hipSetDevice(checks[c].dev), "hipSetDevice verify");
            checkCuda(hipMemcpy(h_verify, checks[c].ptr, sample_bytes,
                                 hipMemcpyDeviceToHost), "verify D→H");
            bool pass = true;
            for (size_t s = 0; s < sample_size; ++s) {
                size_t i = (size_t) rand() % sample_size;
                if (h_verify[i] != checks[c].expected) {
                    cerr << checks[c].label << " mismatch at index " << i
                         << ": expected " << checks[c].expected
                         << ", got "      << h_verify[i] << "\n";
                    pass = false;
                    break;
                }
            }
            cout << "Verification " << checks[c].label << ": "
                 << (pass ? "PASSED" : "FAILED")
                 << " (" << sample_size << "/" << N << " samples)\n";
        }
        free(h_verify);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    checkCuda(hipSetDevice(0), "hipSetDevice 0");
    checkCuda(hipFree(d_A1),  "hipFree d_A1");  checkCuda(hipFree(d_B1),  "hipFree d_B1");
    checkCuda(hipFree(d_C1),  "hipFree d_C1");  checkCuda(hipFree(d_D1),  "hipFree d_D1");
    checkCuda(hipFree(d_A11), "hipFree d_A11"); checkCuda(hipFree(d_B11), "hipFree d_B11");
    checkCuda(hipFree(d_C11), "hipFree d_C11"); checkCuda(hipFree(d_D11), "hipFree d_D11");

    checkCuda(hipSetDevice(1), "hipSetDevice 1");
    checkCuda(hipFree(d_A2),  "hipFree d_A2");  checkCuda(hipFree(d_B2),  "hipFree d_B2");
    checkCuda(hipFree(d_C2),  "hipFree d_C2");  checkCuda(hipFree(d_D2),  "hipFree d_D2");
    checkCuda(hipFree(d_A22), "hipFree d_A22"); checkCuda(hipFree(d_B22), "hipFree d_B22");
    checkCuda(hipFree(d_C22), "hipFree d_C22"); checkCuda(hipFree(d_D22), "hipFree d_D22");

    checkCuda(hipSetDevice(2), "hipSetDevice 2");
    checkCuda(hipFree(d_A3),  "hipFree d_A3");  checkCuda(hipFree(d_B3),  "hipFree d_B3");
    checkCuda(hipFree(d_C3),  "hipFree d_C3");  checkCuda(hipFree(d_D3),  "hipFree d_D3");
    checkCuda(hipFree(d_A33), "hipFree d_A33"); checkCuda(hipFree(d_B33), "hipFree d_B33");
    checkCuda(hipFree(d_C33), "hipFree d_C33"); checkCuda(hipFree(d_D33), "hipFree d_D33");

    checkCuda(hipSetDevice(3), "hipSetDevice 3");
    checkCuda(hipFree(d_A4),  "hipFree d_A4");  checkCuda(hipFree(d_B4),  "hipFree d_B4");
    checkCuda(hipFree(d_C4),  "hipFree d_C4");  checkCuda(hipFree(d_D4),  "hipFree d_D4");
    checkCuda(hipFree(d_A44), "hipFree d_A44"); checkCuda(hipFree(d_B44), "hipFree d_B44");
    checkCuda(hipFree(d_C44), "hipFree d_C44"); checkCuda(hipFree(d_D44), "hipFree d_D44");

    for (int src = 0; src < 4; ++src)
      for (int dst = 0; dst < 4; ++dst)
        if (src != dst)
	  checkCuda(hipStreamDestroy(stream[src][dst]), "hipStreamDestroy");
    
    free(h_A1); free(h_B1); free(h_C1); free(h_D1);
    free(h_A2); free(h_B2); free(h_C2); free(h_D2);
    free(h_A3); free(h_B3); free(h_C3); free(h_D3);
    free(h_A4); free(h_B4); free(h_C4); free(h_D4);
    return EXIT_SUCCESS;
}
