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

    // ── Device check (need at least 2 GPUs) ──────────────────────────────────
    int num_devices = acc_get_num_devices(acc_device_nvidia);
    if (num_devices < 2) {
        cerr << "Error: need at least 2 NVIDIA OpenACC devices, found "
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
    acc_set_device_num(0, acc_device_nvidia);
    double *d_A = (double*) acc_malloc(bytes);
    if (!d_A) {
        cerr << "Error: acc_malloc failed for d_A on device 0\n";
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }
    acc_memcpy_to_device(d_A, h_A, bytes);

    acc_set_device_num(1, acc_device_nvidia);
    double *d_B = (double*) acc_malloc(bytes);
    if (!d_B) {
        cerr << "Error: acc_malloc failed for d_B on device 1\n";
        acc_set_device_num(0, acc_device_nvidia); acc_free(d_A);
        free(h_A); free(h_B);
        return EXIT_FAILURE;
    }
    acc_memcpy_to_device(d_B, h_B, bytes);

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for(int w=0; w<2; w++)
      {
	acc_set_device_num(0, acc_device_nvidia);
	acc_memcpy_from_device(h_A, d_A, bytes);
	acc_set_device_num(1, acc_device_nvidia);
	acc_memcpy_from_device(h_B, d_B, bytes);
      }
    
    // ── Timed transfer (D→H) ─────────────────────────────────────────────────
    omp_set_num_threads(2);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            acc_set_device_num(0, acc_device_nvidia);
            acc_memcpy_from_device(h_A, d_A, bytes);
        }
        #pragma omp section
        {
            acc_set_device_num(1, acc_device_nvidia);
            acc_memcpy_from_device(h_B, d_B, bytes);
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
    // D→H lands back into h_A/h_B — spot-check against known init values
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
    acc_set_device_num(0, acc_device_nvidia); acc_free(d_A);
    acc_set_device_num(1, acc_device_nvidia); acc_free(d_B);
    free(h_A); free(h_B);
    return EXIT_SUCCESS;
}
