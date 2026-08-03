#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <omp.h>

using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " <vector_size>\n";
    return EXIT_FAILURE;
  }

  const size_t N     = stoull(argv[1]);
  const size_t bytes = N * sizeof(double);

  const int host_id = omp_get_initial_device();
  const int dev0    = 0;
  const int dev1    = 1;

  cout << "omp_get_num_devices()    = " << omp_get_num_devices() << "\n";
  cout << "omp_get_initial_device() = " << host_id << "\n";

  if (omp_get_num_devices() < 2) {
    cerr << "Need at least 2 target devices\n";
    return EXIT_FAILURE;
  }

  // ── Host allocation ──────────────────────────────────────────────────────
  double *h_A = (double*) malloc(bytes);
  double *h_B = (double*) malloc(bytes);
  double *h_C = (double*) malloc(bytes);
  double *h_D = (double*) malloc(bytes);

  if (!h_A || !h_B || !h_C || !h_D) {
    cerr << "Host allocation failed\n";
    free(h_A); free(h_B); free(h_C); free(h_D);
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < N; ++i) {
    h_A[i] = 1.0;
    h_B[i] = 2.0;
    h_C[i] = 1.0;
    h_D[i] = 2.0;
  }

  // ── Device allocation ────────────────────────────────────────────────────
  double *d_A = (double*) omp_target_alloc(bytes, dev0);  // src: dev0 -> dev1
  double *d_B = (double*) omp_target_alloc(bytes, dev1);  // recv on dev1 from dev0
  double *d_C = (double*) omp_target_alloc(bytes, dev1);  // src: dev1 -> dev0
  double *d_D = (double*) omp_target_alloc(bytes, dev0);  // recv on dev0 from dev1

  if (!d_A || !d_B || !d_C || !d_D) {
    cerr << "Device allocation failed\n";
    if (d_A) omp_target_free(d_A, dev0);
    if (d_B) omp_target_free(d_B, dev1);
    if (d_C) omp_target_free(d_C, dev1);
    if (d_D) omp_target_free(d_D, dev0);
    free(h_A); free(h_B); free(h_C); free(h_D);
    return EXIT_FAILURE;
  }

  // ── Host → Device copies ─────────────────────────────────────────────────
  if (omp_target_memcpy(d_A, h_A, bytes, 0, 0, dev0, host_id) != 0) {
    cerr << "h_A -> d_A failed\n"; return EXIT_FAILURE;
  }
  if (omp_target_memcpy(d_D, h_D, bytes, 0, 0, dev0, host_id) != 0) {
    cerr << "h_D -> d_D failed\n"; return EXIT_FAILURE;
  }
  if (omp_target_memcpy(d_B, h_B, bytes, 0, 0, dev1, host_id) != 0) {
    cerr << "h_B -> d_B failed\n"; return EXIT_FAILURE;
  }
  if (omp_target_memcpy(d_C, h_C, bytes, 0, 0, dev1, host_id) != 0) {
    cerr << "h_C -> d_C failed\n"; return EXIT_FAILURE;
  }
  
  // ── Async bidirectional P2P transfers ────────────────────────────────────
  // Two concurrent sections: dev0 -> dev1 and dev1 -> dev0 simultaneously.
  int rc1 = 0, rc2 = 0;
  omp_set_num_threads(2);  

  // ── Warm-up P2P transfers (not timed) ────────────────────────────────────
  for (int w = 0; w < 2; ++w)
    {
      omp_target_memcpy_async(d_C, d_A, bytes, 0, 0, dev1, dev0, 0, NULL);
      omp_target_memcpy_async(d_D, d_B, bytes, 0, 0, dev0, dev1, 0, NULL);
#pragma omp taskwait
    }
#pragma omp taskwait

  const double start = omp_get_wtime();  
#pragma omp parallel
  {
#pragma omp sections
    {
      // ── dev0 -> dev1 ─────────────────────────────────────────────────────
#pragma omp section
      {
        rc1 = omp_target_memcpy_async(d_C, d_A, bytes, 0, 0, dev1, dev0, 0, NULL);
      }
      // ── dev1 -> dev0 ─────────────────────────────────────────────────────
#pragma omp section
      {
        rc2 = omp_target_memcpy_async(d_D, d_B, bytes, 0, 0, dev0, dev1, 0, NULL);
      }
    }
  }
#pragma omp taskwait
  const double end = omp_get_wtime();
  
  if (rc1 != 0 || rc2 != 0) {
    cerr << "Direct D2D copy failed\n";
    omp_target_free(d_A, dev0); omp_target_free(d_B, dev1);
    omp_target_free(d_C, dev1); omp_target_free(d_D, dev0);
    free(h_A); free(h_B); free(h_C); free(h_D);
    return EXIT_FAILURE;
  }

  // ── Verification ─────────────────────────────────────────────────────────
  // Check dev0->dev1 transfer: d_C should now contain 1.0 (from d_A)
  if (omp_target_memcpy(h_C, d_C, bytes, 0, 0, host_id, dev1) != 0) {
    cerr << "d_C -> h_C failed\n"; return EXIT_FAILURE;
  }
  // Check dev1->dev0 transfer: d_D should now contain 2.0 (from d_B)
  if (omp_target_memcpy(h_D, d_D, bytes, 0, 0, host_id, dev0) != 0) {
    cerr << "d_D -> h_D failed\n"; return EXIT_FAILURE;
  }

  bool ok = true;
  for (size_t i = 0; i < min<size_t>(N, 10); ++i) {
    if (h_C[i] != 1.0 || h_D[i] != 2.0) {
      ok = false;
      break;
    }
  }
  
  // ── Bandwidth report ─────────────────────────────────────────────────────
  const double seconds       = end - start;
  const double total_bytes   = 2.0 * bytes;   // two transfers
  const double bandwidthGBps = total_bytes / (seconds * 1e9);

  cout << fixed << setprecision(6);
  cout << "Elapsed time:      " << seconds << " seconds\n";
  cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
  cout << "P2P Bandwidth:     " << bandwidthGBps << " GB/s\n";
  cout << "Verification:      " << (ok ? "PASS" : "FAIL") << "\n";

  // ── Free device memory ───────────────────────────────────────────────────
  omp_target_free(d_A, dev0); omp_target_free(d_B, dev1);
  omp_target_free(d_C, dev1); omp_target_free(d_D, dev0);

  // ── Free host memory ─────────────────────────────────────────────────────
  free(h_A); free(h_B); free(h_C); free(h_D);

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
