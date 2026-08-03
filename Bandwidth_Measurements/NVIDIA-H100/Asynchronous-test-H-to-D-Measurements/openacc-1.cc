#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <openacc.h>
#include <omp.h>

using namespace std;

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
    int num_devices = acc_get_num_devices(acc_device_nvidia);
    if (num_devices < 1) {
        cerr << "Error: no NVIDIA OpenACC devices found\n";
        return EXIT_FAILURE;
    }
    acc_set_device_num(0, acc_device_nvidia);

    // ── Host allocation ──────────────────────────────────────────────────────
    double *h_A = (double*) malloc(bytes);
    if (!h_A) {
        cerr << "Error: malloc failed for " << bytes << " bytes\n";
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; ++i)
        h_A[i] = 1.0;

    // ── Device allocation ────────────────────────────────────────────────────
    double *d_A = (double*) acc_malloc(bytes);
    if (!d_A) {
        cerr << "Error: acc_malloc failed for " << bytes << " bytes\n";
        free(h_A);
        return EXIT_FAILURE;
    }

    // ── Warm-up transfers (not timed, async on queue 0) ───────────────────────
    acc_memcpy_to_device_async(d_A, h_A, bytes, 0);
    acc_wait(0);   // ensure warm-up 1 completes before next launch
    acc_memcpy_to_device_async(d_A, h_A, bytes, 0);
    acc_wait(0);   // ensure warm-up 2 completes before timed launch

    // ── Timed transfer (async H→D, queue 0) ──────────────────────────────────
    double start = omp_get_wtime();
    acc_memcpy_to_device_async(d_A, h_A, bytes, 0);
    acc_wait(0);   // block until transfer is fully complete before stopping timer
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
        acc_free(d_A);
        free(h_A);
        return EXIT_FAILURE;
    }

    // Pull back only sample_size elements to verify the async transfer landed
    acc_memcpy_from_device(h_verify, d_A, sample_bytes);

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
    acc_free(d_A);
    free(h_A);
    return EXIT_SUCCESS;
}
