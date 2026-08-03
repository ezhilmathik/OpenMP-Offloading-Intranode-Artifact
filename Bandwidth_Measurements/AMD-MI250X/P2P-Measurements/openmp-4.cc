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
  const int dev2    = 2;
  const int dev3    = 3;

  cout << "omp_get_num_devices()    = " << omp_get_num_devices() << "\n";
  cout << "omp_get_initial_device() = " << host_id << "\n";

  if (omp_get_num_devices() < 4) {
    cerr << "Need at least 4 target devices\n";
    return EXIT_FAILURE;
  }

  // ── Host allocation ──────────────────────────────────────────────────────
  double *h_A1 = (double*) malloc(bytes);
  double *h_B1 = (double*) malloc(bytes);
  double *h_C1 = (double*) malloc(bytes);
  double *h_D1 = (double*) malloc(bytes);

  double *h_A2 = (double*) malloc(bytes);
  double *h_B2 = (double*) malloc(bytes);
  double *h_C2 = (double*) malloc(bytes);
  double *h_D2 = (double*) malloc(bytes);

  double *h_A3 = (double*) malloc(bytes);
  double *h_B3 = (double*) malloc(bytes);
  double *h_C3 = (double*) malloc(bytes);
  double *h_D3 = (double*) malloc(bytes);

  double *h_A4 = (double*) malloc(bytes);
  double *h_B4 = (double*) malloc(bytes);
  double *h_C4 = (double*) malloc(bytes);
  double *h_D4 = (double*) malloc(bytes);

  if (!h_A1 || !h_B1 || !h_C1 || !h_D1 ||
      !h_A2 || !h_B2 || !h_C2 || !h_D2 ||
      !h_A3 || !h_B3 || !h_C3 || !h_D3 ||
      !h_A4 || !h_B4 || !h_C4 || !h_D4) {
    cerr << "Host allocation failed\n";
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < N; ++i) {
    h_A1[i] = h_B1[i] = h_C1[i] = h_D1[i] = 1.0;
    h_A2[i] = h_B2[i] = h_C2[i] = h_D2[i] = 2.0;
    h_A3[i] = h_B3[i] = h_C3[i] = h_D3[i] = 3.0;
    h_A4[i] = h_B4[i] = h_C4[i] = h_D4[i] = 4.0;
  }

  // ── Device allocation ────────────────────────────────────────────────────
  double *d_A1 = (double*) omp_target_alloc(bytes, dev0);
  double *d_B1 = (double*) omp_target_alloc(bytes, dev0);  // recv on dev0 from dev1
  double *d_C1 = (double*) omp_target_alloc(bytes, dev0);  // recv on dev0 from dev2
  double *d_D1 = (double*) omp_target_alloc(bytes, dev0);  // recv on dev0 from dev3

  double *d_A2 = (double*) omp_target_alloc(bytes, dev1);
  double *d_B2 = (double*) omp_target_alloc(bytes, dev1);
  double *d_C2 = (double*) omp_target_alloc(bytes, dev1);
  double *d_D2 = (double*) omp_target_alloc(bytes, dev1);

  double *d_A3 = (double*) omp_target_alloc(bytes, dev2);
  double *d_B3 = (double*) omp_target_alloc(bytes, dev2);
  double *d_C3 = (double*) omp_target_alloc(bytes, dev2);
  double *d_D3 = (double*) omp_target_alloc(bytes, dev2);

  double *d_A4 = (double*) omp_target_alloc(bytes, dev3);
  double *d_B4 = (double*) omp_target_alloc(bytes, dev3);
  double *d_C4 = (double*) omp_target_alloc(bytes, dev3);
  double *d_D4 = (double*) omp_target_alloc(bytes, dev3);

  if (!d_A1 || !d_B1 || !d_C1 || !d_D1 ||
      !d_A2 || !d_B2 || !d_C2 || !d_D2 ||
      !d_A3 || !d_B3 || !d_C3 || !d_D3 ||
      !d_A4 || !d_B4 || !d_C4 || !d_D4) {
    cerr << "Device allocation failed\n";
    return EXIT_FAILURE;
  }

  // ── Host → Device copies ─────────────────────────────────────────────────
  if (omp_target_memcpy(d_A1, h_A1, bytes, 0, 0, dev0, host_id) != 0 ||
      omp_target_memcpy(d_B1, h_B1, bytes, 0, 0, dev0, host_id) != 0 ||
      omp_target_memcpy(d_C1, h_C1, bytes, 0, 0, dev0, host_id) != 0 ||
      omp_target_memcpy(d_D1, h_D1, bytes, 0, 0, dev0, host_id) != 0) {
    cerr << "H->D copy for dev0 failed\n"; return EXIT_FAILURE;
  }
  if (omp_target_memcpy(d_A2, h_A2, bytes, 0, 0, dev1, host_id) != 0 ||
      omp_target_memcpy(d_B2, h_B2, bytes, 0, 0, dev1, host_id) != 0 ||
      omp_target_memcpy(d_C2, h_C2, bytes, 0, 0, dev1, host_id) != 0 ||
      omp_target_memcpy(d_D2, h_D2, bytes, 0, 0, dev1, host_id) != 0) {
    cerr << "H->D copy for dev1 failed\n"; return EXIT_FAILURE;
  }
  if (omp_target_memcpy(d_A3, h_A3, bytes, 0, 0, dev2, host_id) != 0 ||
      omp_target_memcpy(d_B3, h_B3, bytes, 0, 0, dev2, host_id) != 0 ||
      omp_target_memcpy(d_C3, h_C3, bytes, 0, 0, dev2, host_id) != 0 ||
      omp_target_memcpy(d_D3, h_D3, bytes, 0, 0, dev2, host_id) != 0) {
    cerr << "H->D copy for dev2 failed\n"; return EXIT_FAILURE;
  }
  if (omp_target_memcpy(d_A4, h_A4, bytes, 0, 0, dev3, host_id) != 0 ||
      omp_target_memcpy(d_B4, h_B4, bytes, 0, 0, dev3, host_id) != 0 ||
      omp_target_memcpy(d_C4, h_C4, bytes, 0, 0, dev3, host_id) != 0 ||
      omp_target_memcpy(d_D4, h_D4, bytes, 0, 0, dev3, host_id) != 0) {
    cerr << "H->D copy for dev3 failed\n"; return EXIT_FAILURE;
  }

   
  // ── Async P2P transfers (dev1 -> dev0, dev2 -> dev0, dev3 -> dev0) ───────
  // Three concurrent sections each sending one buffer into dev0.
  int rc1 = 0, rc2 = 0, rc3 = 0;
  omp_depend_t dummy;
  omp_set_num_threads(3);

  // ── Warm-up P2P transfers (not timed) ────────────────────────────────────
  for (int w = 0; w < 2; ++w)
    {
      omp_target_memcpy_async(d_B1, d_B2, bytes, 0, 0, dev0, dev1, 0, NULL);
      omp_target_memcpy_async(d_C1, d_C3, bytes, 0, 0, dev0, dev2, 0, NULL);
      omp_target_memcpy_async(d_D1, d_D4, bytes, 0, 0, dev0, dev3, 0, NULL);
#pragma omp taskwait  
    }  
#pragma omp taskwait
  
  const double start = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp sections
    {
      // ── dev1 -> dev0 ─────────────────────────────────────────────────────
#pragma omp section
      {
        rc1 = omp_target_memcpy_async(d_B1, d_B2, bytes, 0, 0, dev0, dev1, 0, NULL);
      }
      // ── dev2 -> dev0 ─────────────────────────────────────────────────────
#pragma omp section
      {
        rc2 = omp_target_memcpy_async(d_C1, d_C3, bytes, 0, 0, dev0, dev2, 0, NULL);
      }
      // ── dev3 -> dev0 ─────────────────────────────────────────────────────
#pragma omp section
      {
        rc3 = omp_target_memcpy_async(d_D1, d_D4, bytes, 0, 0, dev0, dev3, 0, NULL);
      }
    }
  }
#pragma omp taskwait

  const double end = omp_get_wtime();

  if (rc1 != 0 || rc2 != 0 || rc3 != 0) {
    cerr << "Direct D2D copy failed\n";
    return EXIT_FAILURE;
  }

  // ── Verification ─────────────────────────────────────────────────────────
  if (omp_target_memcpy(h_B1, d_B1, bytes, 0, 0, host_id, dev0) != 0) {
    cerr << "d_B1 -> h_B1 failed\n"; return EXIT_FAILURE;
  }
  if (omp_target_memcpy(h_C1, d_C1, bytes, 0, 0, host_id, dev0) != 0) {
    cerr << "d_C1 -> h_C1 failed\n"; return EXIT_FAILURE;
  }
  if (omp_target_memcpy(h_D1, d_D1, bytes, 0, 0, host_id, dev0) != 0) {
    cerr << "d_D1 -> h_D1 failed\n"; return EXIT_FAILURE;
  }

  bool ok = true;
  for (size_t i = 0; i < min<size_t>(N, 10); ++i) {
    if (h_B1[i] != 2.0 || h_C1[i] != 3.0 || h_D1[i] != 4.0) {
      ok = false;
      break;
    }
  }

  // ── Bandwidth report ─────────────────────────────────────────────────────
  const double seconds       = end - start;
  const double total_bytes   = 3.0 * bytes;   // three transfers
  const double bandwidthGBps = total_bytes / (seconds * 1e9);

  cout << fixed << setprecision(6);
  cout << "Elapsed time:      " << seconds << " seconds\n";
  cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
  cout << "P2P Bandwidth:     " << bandwidthGBps << " GB/s\n";
  cout << "Verification:      " << (ok ? "PASS" : "FAIL") << "\n";

  // ── Free device memory ───────────────────────────────────────────────────
  omp_target_free(d_A1, dev0); omp_target_free(d_B1, dev0);
  omp_target_free(d_C1, dev0); omp_target_free(d_D1, dev0);

  omp_target_free(d_A2, dev1); omp_target_free(d_B2, dev1);
  omp_target_free(d_C2, dev1); omp_target_free(d_D2, dev1);

  omp_target_free(d_A3, dev2); omp_target_free(d_B3, dev2);
  omp_target_free(d_C3, dev2); omp_target_free(d_D3, dev2);

  omp_target_free(d_A4, dev3); omp_target_free(d_B4, dev3);
  omp_target_free(d_C4, dev3); omp_target_free(d_D4, dev3);

  // ── Free host memory ─────────────────────────────────────────────────────
  free(h_A1); free(h_B1); free(h_C1); free(h_D1);
  free(h_A2); free(h_B2); free(h_C2); free(h_D2);
  free(h_A3); free(h_B3); free(h_C3); free(h_D3);
  free(h_A4); free(h_B4); free(h_C4); free(h_D4);

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
