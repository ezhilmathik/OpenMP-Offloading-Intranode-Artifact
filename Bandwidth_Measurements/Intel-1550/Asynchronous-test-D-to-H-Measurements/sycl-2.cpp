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

    // ── Device check (need at least 2 GPUs) ──────────────────────────────────
    int num_devices = 0;
    checkCuda(DPCT_CHECK_ERROR(num_devices =
                                   dpct::dev_mgr::instance().device_count()),
              "cudaGetDeviceCount");
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

    /*
    DPCT1093:3: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
    checkCuda(DPCT_CHECK_ERROR(d_A = (double *)sycl::malloc_device(
                                   bytes, dpct::get_in_order_queue())),
              "cudaMalloc d_A");
    checkCuda(DPCT_CHECK_ERROR(
                  dpct::get_in_order_queue().memcpy(d_A, h_A, bytes).wait()),
              "initial cudaMemcpy H→D d_A");

    /*
    DPCT1093:4: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)), "cudaSetDevice 1");
    checkCuda(DPCT_CHECK_ERROR(d_B = (double *)sycl::malloc_device(
                                   bytes, dpct::get_in_order_queue())),
              "cudaMalloc d_B");
    if (!d_B) {
        /*
        DPCT1093:5: The "0" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
        checkCuda(
            DPCT_CHECK_ERROR(dpct::dpct_free(d_A, dpct::get_in_order_queue())),
            "cudaFree d_A");
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }
    checkCuda(DPCT_CHECK_ERROR(
                  dpct::get_in_order_queue().memcpy(d_B, h_B, bytes).wait()),
              "initial cudaMemcpy H→D d_B");

    // ── Stream creation (one per device) ─────────────────────────────────────
    dpct::queue_ptr stream[2];
    /*
    DPCT1093:6: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
    checkCuda(
        DPCT_CHECK_ERROR(stream[0] = dpct::get_current_device().create_queue()),
        "cudaStreamCreate stream[0]");
    /*
    DPCT1093:7: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)), "cudaSetDevice 1");
    checkCuda(
        DPCT_CHECK_ERROR(stream[1] = dpct::get_current_device().create_queue()),
        "cudaStreamCreate stream[1]");

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for(int w=0; w<2; w++)
      {
        /*
        DPCT1093:8: The "0" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
        /*
        DPCT1124:9: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        checkCuda(DPCT_CHECK_ERROR(stream[0]->memcpy(h_A, d_A, bytes)),
                  "warmup D→H d_A");
        checkCuda(DPCT_CHECK_ERROR(stream[0]->wait()),
                  "cudaStreamSynchronize warmup stream[0]");
        /*
        DPCT1093:10: The "1" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)), "cudaSetDevice 1");
        /*
        DPCT1124:11: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        checkCuda(DPCT_CHECK_ERROR(stream[1]->memcpy(h_B, d_B, bytes)),
                  "warmup D→H d_B");
        checkCuda(DPCT_CHECK_ERROR(stream[1]->wait()),
                  "cudaStreamSynchronize warmup stream[1]");
      }
    
    // ── Timed transfer (async D→H, parallel across both devices) ─────────────
    omp_set_num_threads(2);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            /*
            DPCT1093:12: The "0" device may be not the one intended for use.
            Adjust the selected device if needed.
            */
            checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)),
                      "cudaSetDevice 0");
            /*
            DPCT1124:13: cudaMemcpyAsync is migrated to asynchronous memcpy API.
            While the origin API might be synchronous, it depends on the type of
            operand memory, so you may need to call wait() on event return by
            memcpy API to ensure synchronization behavior.
            */
            checkCuda(DPCT_CHECK_ERROR(stream[0]->memcpy(h_A, d_A, bytes)),
                      "timed cudaMemcpyAsync D→H d_A");
            checkCuda(DPCT_CHECK_ERROR(stream[0]->wait()),
                      "cudaStreamSynchronize timed stream[0]");
        }
        #pragma omp section
        {
            /*
            DPCT1093:14: The "1" device may be not the one intended for use.
            Adjust the selected device if needed.
            */
            checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)),
                      "cudaSetDevice 1");
            /*
            DPCT1124:15: cudaMemcpyAsync is migrated to asynchronous memcpy API.
            While the origin API might be synchronous, it depends on the type of
            operand memory, so you may need to call wait() on event return by
            memcpy API to ensure synchronization behavior.
            */
            checkCuda(DPCT_CHECK_ERROR(stream[1]->memcpy(h_B, d_B, bytes)),
                      "timed cudaMemcpyAsync D→H d_B");
            checkCuda(DPCT_CHECK_ERROR(stream[1]->wait()),
                      "cudaStreamSynchronize timed stream[1]");
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
    /*
    DPCT1093:16: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
        checkCuda(
            DPCT_CHECK_ERROR(dpct::dpct_free(d_A, dpct::get_in_order_queue())),
            "cudaFree d_A");
    /*
    DPCT1093:17: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)), "cudaSetDevice 1");
        checkCuda(
            DPCT_CHECK_ERROR(dpct::dpct_free(d_B, dpct::get_in_order_queue())),
            "cudaFree d_B");
    checkCuda(
        DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(stream[0])),
        "cudaStreamDestroy stream[0]");
    checkCuda(
        DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(stream[1])),
        "cudaStreamDestroy stream[1]");
    free(h_A); free(h_B);
    return EXIT_SUCCESS;
}
