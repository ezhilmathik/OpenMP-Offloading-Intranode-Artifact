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
    if (N > (SIZE_MAX / sizeof(double))) {
        cerr << "Error: vector_size too large, would overflow size_t\n";
        return EXIT_FAILURE;
    }
    size_t bytes = N * sizeof(double);

    // ── Device check (need at least 4 GPUs) ──────────────────────────────────
    int num_devices = omp_get_num_devices();
    if (num_devices < 4) {
        cerr << "Error: need at least 4 OpenMP target devices, found "
             << num_devices << "\n";
        return EXIT_FAILURE;
    }

    int host_id = omp_get_initial_device();
    int dev0    = 0;
    int dev1    = 1;
    int dev2    = 2;
    int dev3    = 3;

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
    double *d_A = (double*) omp_target_alloc(bytes, dev0);
    double *d_B = (double*) omp_target_alloc(bytes, dev1);
    double *d_C = (double*) omp_target_alloc(bytes, dev2);
    double *d_D = (double*) omp_target_alloc(bytes, dev3);
    if (!d_A || !d_B || !d_C || !d_D) {
        cerr << "Error: omp_target_alloc failed\n";
        if (d_A) omp_target_free(d_A, dev0);
        if (d_B) omp_target_free(d_B, dev1);
        if (d_C) omp_target_free(d_C, dev2);
        if (d_D) omp_target_free(d_D, dev3);
        free(h_A); free(h_B); free(h_C); free(h_D);
        return EXIT_FAILURE;
    }

    // ── Initial H→D to populate device buffers ────────────────────────────────
    omp_target_memcpy(d_A, h_A, bytes, 0, 0, dev0, host_id);
    omp_target_memcpy(d_B, h_B, bytes, 0, 0, dev1, host_id);
    omp_target_memcpy(d_C, h_C, bytes, 0, 0, dev2, host_id);
    omp_target_memcpy(d_D, h_D, bytes, 0, 0, dev3, host_id);

    // ── Warm-up transfers (not timed) ────────────────────────────────────────
    for(int w=0; w<2; w++)
      {
	omp_target_memcpy(h_A, d_A, bytes, 0, 0, host_id, dev0);
	omp_target_memcpy(h_B, d_B, bytes, 0, 0, host_id, dev1);
	omp_target_memcpy(h_C, d_C, bytes, 0, 0, host_id, dev2);
	omp_target_memcpy(h_D, d_D, bytes, 0, 0, host_id, dev3);
      }
    
    // ── Timed transfer (D→H) ─────────────────────────────────────────────────
    omp_set_num_threads(4);
    double start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            omp_target_memcpy(h_A, d_A, bytes, 0, 0, host_id, dev0);
        }
        #pragma omp section
        {
            omp_target_memcpy(h_B, d_B, bytes, 0, 0, host_id, dev1);
        }
        #pragma omp section
        {
            omp_target_memcpy(h_C, d_C, bytes, 0, 0, host_id, dev2);
        }
        #pragma omp section
        {
            omp_target_memcpy(h_D, d_D, bytes, 0, 0, host_id, dev3);
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

    // ── Spot-check verification ───────────────────────────────────────────────
    // D→H lands back into h_A/h_B/h_C/h_D — check against known init values
    size_t sample_size = min(N, (size_t) 1000);

    double  expected[4] = { 1.0, 2.0, 3.0, 4.0 };
    double *h_buf[4]    = { h_A, h_B, h_C, h_D };
    int     devs[4]     = { dev0, dev1, dev2, dev3 };
    bool    pass[4]     = { true, true, true, true };

    srand(42);
    for (int d = 0; d < 4; ++d) {
        for (size_t s = 0; s < sample_size; ++s) {
            size_t i = (size_t) rand() % sample_size;
            if (h_buf[d][i] != expected[d]) {
                cerr << "h_" << (char)('A'+d) << " mismatch at index " << i
                     << ": expected " << expected[d]
                     << ", got "      << h_buf[d][i] << "\n";
                pass[d] = false;
                break;
            }
        }
        cout << "Verification h_" << (char)('A'+d)
             << " (device " << devs[d] << " → host): "
             << (pass[d] ? "PASSED" : "FAILED")
             << " (" << sample_size << "/" << N << " samples)\n";
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    omp_target_free(d_A, dev0); omp_target_free(d_B, dev1);
    omp_target_free(d_C, dev2); omp_target_free(d_D, dev3);
    free(h_A); free(h_B); free(h_C); free(h_D);
    return EXIT_SUCCESS;
}
