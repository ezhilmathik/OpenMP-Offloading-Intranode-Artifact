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

    // ── Stream creation ──────────────────────────────────────────────────────
    dpct::queue_ptr stream;
    checkCuda(
        DPCT_CHECK_ERROR(stream = dpct::get_current_device().create_queue()),
        "cudaStreamCreate");

    // ── CUDA event creation ──────────────────────────────────────────────────
    dpct::event_ptr start, stop;
    checkCuda(DPCT_CHECK_ERROR(start = new sycl::event()),
              "cudaEventCreate start");
    checkCuda(DPCT_CHECK_ERROR(stop = new sycl::event()),
              "cudaEventCreate stop");

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    /*
    DPCT1124:4: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    checkCuda(DPCT_CHECK_ERROR(stream->memcpy(d_A, h_A, bytes)),
              "warmup 1 H→D d_A");
    checkCuda(DPCT_CHECK_ERROR(stream->wait()),
              "cudaStreamSynchronize warmup 1");
    /*
    DPCT1124:5: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    checkCuda(DPCT_CHECK_ERROR(stream->memcpy(d_A, h_A, bytes)),
              "warmup 2 H→D d_A");
    checkCuda(DPCT_CHECK_ERROR(stream->wait()),
              "cudaStreamSynchronize warmup 2");

    // ── Timed transfer (async H→D via CUDA events) ───────────────────────────
    /*
    DPCT1024:6: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    checkCuda(DPCT_CHECK_ERROR(*start = stream->ext_oneapi_submit_barrier()),
              "cudaEventRecord start");
    /*
    DPCT1124:7: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    checkCuda(DPCT_CHECK_ERROR(stream->memcpy(d_A, h_A, bytes)),
              "timed cudaMemcpyAsync H→D d_A");
    /*
    DPCT1024:8: The original code returned the error code that was further
    consumed by the program logic. This original code was replaced with 0. You
    may need to rewrite the program logic consuming the error code.
    */
    checkCuda(DPCT_CHECK_ERROR(*stop = stream->ext_oneapi_submit_barrier()),
              "cudaEventRecord stop");
    checkCuda(DPCT_CHECK_ERROR(stream->wait()), "cudaStreamSynchronize timed");
    checkCuda(DPCT_CHECK_ERROR(stop->wait_and_throw()),
              "cudaEventSynchronize stop");

    // ── Elapsed time from CUDA events ────────────────────────────────────────
    float milliseconds = 0.0f;
    checkCuda(
        DPCT_CHECK_ERROR(
            milliseconds = (stop->get_profiling_info<
                                sycl::info::event_profiling::command_end>() -
                            start->get_profiling_info<
                                sycl::info::event_profiling::command_start>()) /
                           1000000.0f),
        "cudaEventElapsedTime");

    if (milliseconds <= 0.0f) {
        cerr << "Warning: CUDA event elapsed time is non-positive, "
             << "results may be unreliable\n";
    }

    // ── Bandwidth calculation ─────────────────────────────────────────────────
    double seconds       = milliseconds / 1000.0;
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
        dpct::dpct_free(d_A, dpct::get_in_order_queue());
        dpct::get_current_device().destroy_queue(stream);
        dpct::destroy_event(start);
        dpct::destroy_event(stop);
        free(h_A);
        return EXIT_FAILURE;
    }

    // Pull back only sample_size elements to confirm async transfer landed
    checkCuda(DPCT_CHECK_ERROR(dpct::get_in_order_queue()
                                   .memcpy(h_verify, d_A, sample_bytes)
                                   .wait()),
              "verify cudaMemcpy D→H d_A");

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
    checkCuda(
        DPCT_CHECK_ERROR(dpct::get_current_device().destroy_queue(stream)),
        "cudaStreamDestroy");
    checkCuda(DPCT_CHECK_ERROR(dpct::destroy_event(start)),
              "cudaEventDestroy start");
    checkCuda(DPCT_CHECK_ERROR(dpct::destroy_event(stop)),
              "cudaEventDestroy stop");
    free(h_A);
    return EXIT_SUCCESS;
}