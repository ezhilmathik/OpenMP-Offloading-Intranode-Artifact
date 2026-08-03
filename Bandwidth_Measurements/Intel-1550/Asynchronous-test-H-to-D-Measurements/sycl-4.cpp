//-*-c++-*-
#define DPCT_PROFILING_ENABLED
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
    DPCT1000:10: Error handling if-stmt was detected but could not be rewritten.
    */
    if (result != 0) {
        /*
        DPCT1001:9: The statement could not be removed.
        */
        cerr << "CUDA error in " << func
             << ": "
             /*
             DPCT1009:11: SYCL uses exceptions to report errors and does not use
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

    // ── Device check (need at least 4 GPUs) ──────────────────────────────────
    int num_devices = 0;
    checkCuda(DPCT_CHECK_ERROR(num_devices =
                                   dpct::dev_mgr::instance().device_count()),
              "cudaGetDeviceCount");
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

    /*
    DPCT1093:12: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
    checkCuda(DPCT_CHECK_ERROR(d_A = (double *)sycl::malloc_device(
                                   bytes, dpct::get_in_order_queue())),
              "cudaMalloc d_A");

    /*
    DPCT1093:13: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)), "cudaSetDevice 1");
    checkCuda(DPCT_CHECK_ERROR(d_B = (double *)sycl::malloc_device(
                                   bytes, dpct::get_in_order_queue())),
              "cudaMalloc d_B");
    if (!d_B) {
        /*
        DPCT1093:14: The "0" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
            checkCuda(DPCT_CHECK_ERROR(
                          dpct::dpct_free(d_A, dpct::get_in_order_queue())),
                      "cudaFree d_A");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    /*
    DPCT1093:15: The "2" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(2)), "cudaSetDevice 2");
    checkCuda(DPCT_CHECK_ERROR(d_C = (double *)sycl::malloc_device(
                                   bytes, dpct::get_in_order_queue())),
              "cudaMalloc d_C");
    if (!d_C) {
        /*
        DPCT1093:16: The "0" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
            checkCuda(DPCT_CHECK_ERROR(
                          dpct::dpct_free(d_A, dpct::get_in_order_queue())),
                      "cudaFree d_A");
        /*
        DPCT1093:17: The "1" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)), "cudaSetDevice 1");
            checkCuda(DPCT_CHECK_ERROR(
                          dpct::dpct_free(d_B, dpct::get_in_order_queue())),
                      "cudaFree d_B");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    /*
    DPCT1093:18: The "3" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(3)), "cudaSetDevice 3");
    checkCuda(DPCT_CHECK_ERROR(d_D = (double *)sycl::malloc_device(
                                   bytes, dpct::get_in_order_queue())),
              "cudaMalloc d_D");
    if (!d_D) {
        /*
        DPCT1093:19: The "0" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
            checkCuda(DPCT_CHECK_ERROR(
                          dpct::dpct_free(d_A, dpct::get_in_order_queue())),
                      "cudaFree d_A");
        /*
        DPCT1093:20: The "1" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)), "cudaSetDevice 1");
            checkCuda(DPCT_CHECK_ERROR(
                          dpct::dpct_free(d_B, dpct::get_in_order_queue())),
                      "cudaFree d_B");
        /*
        DPCT1093:21: The "2" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(2)), "cudaSetDevice 2");
            checkCuda(DPCT_CHECK_ERROR(
                          dpct::dpct_free(d_C, dpct::get_in_order_queue())),
                      "cudaFree d_C");
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    // ── Stream creation (one per device) ─────────────────────────────────────
    dpct::queue_ptr stream[4];
    for (int i = 0; i < 4; ++i) {
        /*
        DPCT1093:22: The "i" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(i)),
                  "cudaSetDevice stream init");
        checkCuda(DPCT_CHECK_ERROR(
                      stream[i] = dpct::get_current_device().create_queue()),
                  "cudaStreamCreate");
    }

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for (int w=0; w<2; w++)
      {
        /*
        DPCT1093:23: The "0" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
        /*
        DPCT1124:24: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        checkCuda(DPCT_CHECK_ERROR(stream[0]->memcpy(d_A, h_A, bytes)),
                  "warmup H→D d_A");
        checkCuda(DPCT_CHECK_ERROR(stream[0]->wait()),
                  "cudaStreamSynchronize warmup stream[0]");
        /*
        DPCT1093:25: The "1" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)), "cudaSetDevice 1");
        /*
        DPCT1124:26: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        checkCuda(DPCT_CHECK_ERROR(stream[1]->memcpy(d_B, h_B, bytes)),
                  "warmup H→D d_B");
        checkCuda(DPCT_CHECK_ERROR(stream[1]->wait()),
                  "cudaStreamSynchronize warmup stream[1]");
        /*
        DPCT1093:27: The "2" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(2)), "cudaSetDevice 2");
        /*
        DPCT1124:28: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        checkCuda(DPCT_CHECK_ERROR(stream[2]->memcpy(d_C, h_C, bytes)),
                  "warmup H→D d_C");
        checkCuda(DPCT_CHECK_ERROR(stream[2]->wait()),
                  "cudaStreamSynchronize warmup stream[2]");
        /*
        DPCT1093:29: The "3" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(3)), "cudaSetDevice 3");
        /*
        DPCT1124:30: cudaMemcpyAsync is migrated to asynchronous memcpy API.
        While the origin API might be synchronous, it depends on the type of
        operand memory, so you may need to call wait() on event return by memcpy
        API to ensure synchronization behavior.
        */
        checkCuda(DPCT_CHECK_ERROR(stream[3]->memcpy(d_D, h_D, bytes)),
                  "warmup H→D d_D");
        checkCuda(DPCT_CHECK_ERROR(stream[3]->wait()),
                  "cudaStreamSynchronize warmup stream[3]");
      }
    
    // ── Timed transfer (async H→D, parallel across all 4 devices) ────────────
    omp_set_num_threads(4);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            /*
            DPCT1093:31: The "0" device may be not the one intended for use.
            Adjust the selected device if needed.
            */
            checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)),
                      "cudaSetDevice 0");
            /*
            DPCT1124:32: cudaMemcpyAsync is migrated to asynchronous memcpy API.
            While the origin API might be synchronous, it depends on the type of
            operand memory, so you may need to call wait() on event return by
            memcpy API to ensure synchronization behavior.
            */
            checkCuda(DPCT_CHECK_ERROR(stream[0]->memcpy(d_A, h_A, bytes)),
                      "timed cudaMemcpyAsync H→D d_A");
            checkCuda(DPCT_CHECK_ERROR(stream[0]->wait()),
                      "cudaStreamSynchronize timed stream[0]");
        }
        #pragma omp section
        {
            /*
            DPCT1093:33: The "1" device may be not the one intended for use.
            Adjust the selected device if needed.
            */
            checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)),
                      "cudaSetDevice 1");
            /*
            DPCT1124:34: cudaMemcpyAsync is migrated to asynchronous memcpy API.
            While the origin API might be synchronous, it depends on the type of
            operand memory, so you may need to call wait() on event return by
            memcpy API to ensure synchronization behavior.
            */
            checkCuda(DPCT_CHECK_ERROR(stream[1]->memcpy(d_B, h_B, bytes)),
                      "timed cudaMemcpyAsync H→D d_B");
            checkCuda(DPCT_CHECK_ERROR(stream[1]->wait()),
                      "cudaStreamSynchronize timed stream[1]");
        }
        #pragma omp section
        {
            /*
            DPCT1093:35: The "2" device may be not the one intended for use.
            Adjust the selected device if needed.
            */
            checkCuda(DPCT_CHECK_ERROR(dpct::select_device(2)),
                      "cudaSetDevice 2");
            /*
            DPCT1124:36: cudaMemcpyAsync is migrated to asynchronous memcpy API.
            While the origin API might be synchronous, it depends on the type of
            operand memory, so you may need to call wait() on event return by
            memcpy API to ensure synchronization behavior.
            */
            checkCuda(DPCT_CHECK_ERROR(stream[2]->memcpy(d_C, h_C, bytes)),
                      "timed cudaMemcpyAsync H→D d_C");
            checkCuda(DPCT_CHECK_ERROR(stream[2]->wait()),
                      "cudaStreamSynchronize timed stream[2]");
        }
        #pragma omp section
        {
            /*
            DPCT1093:37: The "3" device may be not the one intended for use.
            Adjust the selected device if needed.
            */
            checkCuda(DPCT_CHECK_ERROR(dpct::select_device(3)),
                      "cudaSetDevice 3");
            /*
            DPCT1124:38: cudaMemcpyAsync is migrated to asynchronous memcpy API.
            While the origin API might be synchronous, it depends on the type of
            operand memory, so you may need to call wait() on event return by
            memcpy API to ensure synchronization behavior.
            */
            checkCuda(DPCT_CHECK_ERROR(stream[3]->memcpy(d_D, h_D, bytes)),
                      "timed cudaMemcpyAsync H→D d_D");
            checkCuda(DPCT_CHECK_ERROR(stream[3]->wait()),
                      "cudaStreamSynchronize timed stream[3]");
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
        /*
        DPCT1093:39: The "0" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
            checkCuda(DPCT_CHECK_ERROR(
                          dpct::dpct_free(d_A, dpct::get_in_order_queue())),
                      "cudaFree d_A");
        /*
        DPCT1093:40: The "1" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)), "cudaSetDevice 1");
            checkCuda(DPCT_CHECK_ERROR(
                          dpct::dpct_free(d_B, dpct::get_in_order_queue())),
                      "cudaFree d_B");
        /*
        DPCT1093:41: The "2" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(2)), "cudaSetDevice 2");
            checkCuda(DPCT_CHECK_ERROR(
                          dpct::dpct_free(d_C, dpct::get_in_order_queue())),
                      "cudaFree d_C");
        /*
        DPCT1093:42: The "3" device may be not the one intended for use. Adjust
        the selected device if needed.
        */
        checkCuda(DPCT_CHECK_ERROR(dpct::select_device(3)), "cudaSetDevice 3");
            checkCuda(DPCT_CHECK_ERROR(
                          dpct::dpct_free(d_D, dpct::get_in_order_queue())),
                      "cudaFree d_D");
        for (int i = 0; i < 4; ++i) dpct::get_current_device().destroy_queue(
            stream[i]);
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    // Pull back only sample_size elements from each device to confirm transfer
    /*
    DPCT1093:43: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
    checkCuda(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memcpy(h_verifyA, d_A, sample_bytes)
                                   .wait()),
              "verify D→H d_A");
    /*
    DPCT1093:44: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)), "cudaSetDevice 1");
    checkCuda(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memcpy(h_verifyB, d_B, sample_bytes)
                                   .wait()),
              "verify D→H d_B");
    /*
    DPCT1093:45: The "2" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(2)), "cudaSetDevice 2");
    checkCuda(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memcpy(h_verifyC, d_C, sample_bytes)
                                   .wait()),
              "verify D→H d_C");
    /*
    DPCT1093:46: The "3" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(3)), "cudaSetDevice 3");
    checkCuda(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memcpy(h_verifyD, d_D, sample_bytes)
                                   .wait()),
              "verify D→H d_D");

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
    /*
    DPCT1093:47: The "0" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(0)), "cudaSetDevice 0");
        checkCuda(
            DPCT_CHECK_ERROR(dpct::dpct_free(d_A, dpct::get_in_order_queue())),
            "cudaFree d_A");
    /*
    DPCT1093:48: The "1" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(1)), "cudaSetDevice 1");
        checkCuda(
            DPCT_CHECK_ERROR(dpct::dpct_free(d_B, dpct::get_in_order_queue())),
            "cudaFree d_B");
    /*
    DPCT1093:49: The "2" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(2)), "cudaSetDevice 2");
        checkCuda(
            DPCT_CHECK_ERROR(dpct::dpct_free(d_C, dpct::get_in_order_queue())),
            "cudaFree d_C");
    /*
    DPCT1093:50: The "3" device may be not the one intended for use. Adjust the
    selected device if needed.
    */
    checkCuda(DPCT_CHECK_ERROR(dpct::select_device(3)), "cudaSetDevice 3");
        checkCuda(
            DPCT_CHECK_ERROR(dpct::dpct_free(d_D, dpct::get_in_order_queue())),
            "cudaFree d_D");
    for (int i = 0; i < 4; ++i)
        checkCuda(DPCT_CHECK_ERROR(
                      dpct::get_current_device().destroy_queue(stream[i])),
                  "cudaStreamDestroy");
    free(h_A); free(h_B); free(h_C); free(h_D);
    return EXIT_SUCCESS;
}
