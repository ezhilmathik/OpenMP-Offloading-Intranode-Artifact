#include <iostream>
#include <iomanip>
#include <cstdlib>
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

    // ── Device check (need at least 2 GPUs) ──────────────────────────────────
    int num_devices = omp_get_num_devices();
    if (num_devices < 2) {
        cerr << "Error: need at least 2 OpenMP target devices, found "
             << num_devices << "\n";
        return EXIT_FAILURE;
    }

    int host_id = omp_get_initial_device();
    int dev0    = 0;
    int dev1    = 1;

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

    // ── Device allocation ────────────────────────────────────────────────────
    double *d_A = (double*) omp_target_alloc(bytes, dev0);
    double *d_B = (double*) omp_target_alloc(bytes, dev1);
    if (!d_A || !d_B) {
        cerr << "Error: omp_target_alloc failed\n";
        if (d_A) omp_target_free(d_A, dev0);
        if (d_B) omp_target_free(d_B, dev1);
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    omp_target_memcpy(d_A, h_A, bytes, 0, 0, dev0, host_id);
    omp_target_memcpy(d_B, h_B, bytes, 0, 0, dev1, host_id);

    // ── Timed transfer ───────────────────────────────────────────────────────
    omp_set_num_threads(2);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            omp_target_memcpy(d_A, h_A, bytes, 0, 0, dev0, host_id);
        }
        #pragma omp section
        {
            omp_target_memcpy(d_B, h_B, bytes, 0, 0, dev1, host_id);
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

    // ── Spot-check verification (both devices) ────────────────────────────────
    size_t sample_size  = min(N, (size_t) 1000);
    size_t sample_bytes = sample_size * sizeof(double);

    double *h_verifyA = (double*) malloc(sample_bytes);
    double *h_verifyB = (double*) malloc(sample_bytes);
    if (!h_verifyA || !h_verifyB) {
        cerr << "Error: malloc failed for verification buffers\n";
        free(h_verifyA); free(h_verifyB);
        omp_target_free(d_A, dev0);
        omp_target_free(d_B, dev1);
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }

    // Pull back only sample_size elements from each device
    omp_target_memcpy(h_verifyA, d_A, sample_bytes, 0, 0, host_id, dev0);
    omp_target_memcpy(h_verifyB, d_B, sample_bytes, 0, 0, host_id, dev1);

    srand(42);
    bool passA = true, passB = true;
    for (size_t s = 0; s < sample_size; ++s) {
        size_t i = (size_t) rand() % sample_size;
        if (h_verifyA[i] != h_A[i]) {
            cerr << "d_A mismatch at index " << i
                 << ": expected " << h_A[i] << ", got " << h_verifyA[i] << "\n";
            passA = false;
            break;
        }
        if (h_verifyB[i] != h_B[i]) {
            cerr << "d_B mismatch at index " << i
                 << ": expected " << h_B[i] << ", got " << h_verifyB[i] << "\n";
            passB = false;
            break;
        }
    }

    cout << "Verification d_A (device " << dev0 << "): "
         << (passA ? "PASSED" : "FAILED")
         << " (" << sample_size << "/" << N << " samples)\n";
    cout << "Verification d_B (device " << dev1 << "): "
         << (passB ? "PASSED" : "FAILED")
         << " (" << sample_size << "/" << N << " samples)\n";

    // ── Cleanup ───────────────────────────────────────────────────────────────
    free(h_verifyA); free(h_verifyB);
    omp_target_free(d_A, dev0);
    omp_target_free(d_B, dev1);
    free(h_A); free(h_B);
    return EXIT_SUCCESS;
}
