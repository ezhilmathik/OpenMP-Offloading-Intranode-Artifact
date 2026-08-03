//-*-c++-*-
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <omp.h>
#include <cmath>

using namespace std;

// ── CUDA error helper ────────────────────────────────────────────────────────
void checkCuda(dpct::err0 result, const char *func) {
    /*
    DPCT1000:1: Error handling if-stmt was detected but could not be rewritten.
    */
    if (result != 0) {
        /*
        DPCT1001:0: The statement could not be removed.
        */
        cerr << "CUDA error in " << func
             << ": "
             /*
             DPCT1009:2: SYCL uses exceptions to report errors and does not use
             the error codes. The call was replaced by a placeholder string. You
             need to rewrite this code.
             */
             << "<Placeholder string>" << "\n";
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
    checkCuda(DPCT_CHECK_ERROR(num_devices =
                                   dpct::dev_mgr::instance().device_count()),
              "cudaGetDeviceCount");
    if (num_devices < 1) {
        cerr << "Error: no CUDA devices found\n";
        return EXIT_FAILURE;
    }
    /*
    DPCT1093:3: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");

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
    checkCuda(DPCT_CHECK_ERROR(d_A = (double *)sycl::malloc_device(
                                   bytes, dpct::get_in_order_queue())),
              "cudaMalloc d_A");

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    checkCuda(DPCT_CHECK_ERROR(
                  dpct::get_in_order_queue().memcpy(d_A, h_A, bytes).wait()),
              "warmup 1 d_A");
    checkCuda(DPCT_CHECK_ERROR(
                  dpct::get_in_order_queue().memcpy(d_A, h_A, bytes).wait()),
              "warmup 2 d_A");

    // ── Timed transfer ───────────────────────────────────────────────────────
    double start = omp_get_wtime();
    checkCuda(DPCT_CHECK_ERROR(
                  dpct::get_in_order_queue().memcpy(d_A, h_A, bytes).wait()),
              "timed cudaMemcpy d_A");
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
    size_t sample_size  = min(N, (size_t) 1000);
    size_t sample_bytes = sample_size * sizeof(double);

    double *h_verify = (double*) malloc(sample_bytes);
    if (!h_verify) {
        cerr << "Error: malloc failed for verification buffer\n";
        checkCuda(
            DPCT_CHECK_ERROR(dpct::dpct_free(d_A, dpct::get_in_order_queue())),
            "cudaFree d_A");
        free(h_A);
        return EXIT_FAILURE;
    }

    // Pull back only sample_size elements across PCIe
    checkCuda(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memcpy(h_verify, d_A, sample_bytes)
                                   .wait()),
              "verify cudaMemcpy d_A");

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
    checkCuda(
        DPCT_CHECK_ERROR(dpct::dpct_free(d_A, dpct::get_in_order_queue())),
        "cudaFree d_A");
    free(h_A);
    return EXIT_SUCCESS;
}