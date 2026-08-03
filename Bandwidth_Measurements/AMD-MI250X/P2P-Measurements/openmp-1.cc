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

  if (!h_A || !h_B) {
    cerr << "Host allocation failed\n";
    free(h_A);
    free(h_B);
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < N; ++i) {
    h_A[i] = 1.0;
    h_B[i] = 2.0;
  }

  // ── Device allocation ────────────────────────────────────────────────────
  double *d_A = (double*) omp_target_alloc(bytes, dev0);
  double *d_B = (double*) omp_target_alloc(bytes, dev1);

  if (!d_A || !d_B) {
    cerr << "Device allocation failed\n";
    if (d_A) omp_target_free(d_A, dev0);
    if (d_B) omp_target_free(d_B, dev1);
    free(h_A);
    free(h_B);
    return EXIT_FAILURE;
  }

  // ── Host → Device copies ─────────────────────────────────────────────────
  if (omp_target_memcpy(d_A, h_A, bytes, 0, 0, dev0, host_id) != 0) {
    cerr << "h_A -> d_A failed\n";
    return EXIT_FAILURE;
  }
  if (omp_target_memcpy(d_B, h_B, bytes, 0, 0, dev1, host_id) != 0) {
    cerr << "h_B -> d_B failed\n";
    return EXIT_FAILURE;
  }

  // ── Async P2P transfer (dev0 -> dev1) ────────────────────────────────────
  // Uses depobj/depend(out) + taskwait depend(depobj) to sync a single async
  // transfer — an alternative to the omp_depend_t dummy pattern used in the
  // multi-transfer variants (openmp-2, openmp-4, openmp-4-4).
  int rc = 0;

  // ── Warm-up P2P transfers (not timed) ────────────────────────────────────
  for (int w = 0; w < 2; ++w)
    {
      omp_target_memcpy_async(d_B, d_A, bytes, 0, 0, dev1, dev0, 0, NULL);
#pragma omp taskwait
    }
#pragma omp taskwait
  
  const double start = omp_get_wtime();
  rc = omp_target_memcpy_async(d_B, d_A, bytes, 0, 0, dev1, dev0, 0, NULL);
#pragma omp taskwait
  const double end = omp_get_wtime();
  
  if (rc != 0) {
    cerr << "Direct D2D copy failed: dev0 -> dev1\n";
    omp_target_free(d_A, dev0);
    omp_target_free(d_B, dev1);
    free(h_A);
    free(h_B);
    return EXIT_FAILURE;
  }

  // ── Verification ─────────────────────────────────────────────────────────
  if (omp_target_memcpy(h_B, d_B, bytes, 0, 0, host_id, dev1) != 0) {
    cerr << "d_B -> h_B failed\n";
    return EXIT_FAILURE;
  }

  bool ok = true;
  for (size_t i = 0; i < min<size_t>(N, 10); ++i) {
    if (h_B[i] != 1.0) {
      ok = false;
      break;
    }
  }

  // ── Bandwidth report ─────────────────────────────────────────────────────
  const double seconds       = end - start;
  const double total_bytes   = 1.0 * bytes;   // one transfer
  const double bandwidthGBps = total_bytes / (seconds * 1e9);

  cout << fixed << setprecision(6);
  cout << "Elapsed time:      " << seconds << " seconds\n";
  cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
  cout << "P2P Bandwidth:     " << bandwidthGBps << " GB/s\n";
  cout << "Verification:      " << (ok ? "PASS" : "FAIL") << "\n";

  // ── Free device memory ───────────────────────────────────────────────────
  omp_target_free(d_A, dev0);
  omp_target_free(d_B, dev1);

  // ── Free host memory ─────────────────────────────────────────────────────
  free(h_A);
  free(h_B);

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
