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

    // ── Device check (need at least 4 GPUs) ──────────────────────────────────
    int num_devices = acc_get_num_devices(acc_device_nvidia);
    if (num_devices < 4) {
        cerr << "Error: need at least 4 NVIDIA OpenACC devices, found "
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
    acc_set_device_num(0, acc_device_nvidia);
    double *d_A = (double*) acc_malloc(bytes);
    if (!d_A) {
        cerr << "Error: acc_malloc failed for d_A on device 0\n";
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    acc_set_device_num(1, acc_device_nvidia);
    double *d_B = (double*) acc_malloc(bytes);
    if (!d_B) {
        cerr << "Error: acc_malloc failed for d_B on device 1\n";
        acc_set_device_num(0, acc_device_nvidia); acc_free(d_A);
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    acc_set_device_num(2, acc_device_nvidia);
    double *d_C = (double*) acc_malloc(bytes);
    if (!d_C) {
        cerr << "Error: acc_malloc failed for d_C on device 2\n";
        acc_set_device_num(0, acc_device_nvidia); acc_free(d_A);
        acc_set_device_num(1, acc_device_nvidia); acc_free(d_B);
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    acc_set_device_num(3, acc_device_nvidia);
    double *d_D = (double*) acc_malloc(bytes);
    if (!d_D) {
        cerr << "Error: acc_malloc failed for d_D on device 3\n";
        acc_set_device_num(0, acc_device_nvidia); acc_free(d_A);
        acc_set_device_num(1, acc_device_nvidia); acc_free(d_B);
        acc_set_device_num(2, acc_device_nvidia); acc_free(d_C);
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    // ── Warm-up transfers (not timed, async on queue 0 per device) ───────────
    for (int w=0; w<2; w++)
      {
	acc_set_device_num(0, acc_device_nvidia);
	acc_memcpy_to_device_async(d_A, h_A, bytes, 0);
	acc_wait(0);
	acc_set_device_num(1, acc_device_nvidia);
	acc_memcpy_to_device_async(d_B, h_B, bytes, 0);
	acc_wait(0);
	acc_set_device_num(2, acc_device_nvidia);
	acc_memcpy_to_device_async(d_C, h_C, bytes, 0);
	acc_wait(0);
	acc_set_device_num(3, acc_device_nvidia);
	acc_memcpy_to_device_async(d_D, h_D, bytes, 0);
	acc_wait(0);
      }
    
    // ── Timed transfer (async H→D, parallel across all 4 devices) ────────────
    omp_set_num_threads(4);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            acc_set_device_num(0, acc_device_nvidia);
            acc_memcpy_to_device_async(d_A, h_A, bytes, 0);
            acc_wait(0);   // block until device 0 transfer completes
        }
        #pragma omp section
        {
            acc_set_device_num(1, acc_device_nvidia);
            acc_memcpy_to_device_async(d_B, h_B, bytes, 0);
            acc_wait(0);   // block until device 1 transfer completes
        }
        #pragma omp section
        {
            acc_set_device_num(2, acc_device_nvidia);
            acc_memcpy_to_device_async(d_C, h_C, bytes, 0);
            acc_wait(0);   // block until device 2 transfer completes
        }
        #pragma omp section
        {
            acc_set_device_num(3, acc_device_nvidia);
            acc_memcpy_to_device_async(d_D, h_D, bytes, 0);
            acc_wait(0);   // block until device 3 transfer completes
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
        acc_set_device_num(0, acc_device_nvidia); acc_free(d_A);
        acc_set_device_num(1, acc_device_nvidia); acc_free(d_B);
        acc_set_device_num(2, acc_device_nvidia); acc_free(d_C);
        acc_set_device_num(3, acc_device_nvidia); acc_free(d_D);
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    // Pull back only sample_size elements from each device to confirm transfer
    acc_set_device_num(0, acc_device_nvidia);
    acc_memcpy_from_device(h_verifyA, d_A, sample_bytes);
    acc_set_device_num(1, acc_device_nvidia);
    acc_memcpy_from_device(h_verifyB, d_B, sample_bytes);
    acc_set_device_num(2, acc_device_nvidia);
    acc_memcpy_from_device(h_verifyC, d_C, sample_bytes);
    acc_set_device_num(3, acc_device_nvidia);
    acc_memcpy_from_device(h_verifyD, d_D, sample_bytes);

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
    acc_set_device_num(0, acc_device_nvidia); acc_free(d_A);
    acc_set_device_num(1, acc_device_nvidia); acc_free(d_B);
    acc_set_device_num(2, acc_device_nvidia); acc_free(d_C);
    acc_set_device_num(3, acc_device_nvidia); acc_free(d_D);
    free(h_A); free(h_B); free(h_C); free(h_D);
    return EXIT_SUCCESS;
}
