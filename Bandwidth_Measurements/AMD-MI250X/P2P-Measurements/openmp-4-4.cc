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
  // Each GPU has 3 source buffers (one per peer it will send to)
  // Named: h_<gpu><slot>  where slot A/B/C = dest GPU0/1/2/3 (skipping self)
  double *h_A1 = (double*) malloc(bytes);  // GPU0 -> GPU1
  double *h_B1 = (double*) malloc(bytes);  // GPU0 -> GPU2
  double *h_C1 = (double*) malloc(bytes);  // GPU0 -> GPU3

  double *h_A2 = (double*) malloc(bytes);  // GPU1 -> GPU0
  double *h_B2 = (double*) malloc(bytes);  // GPU1 -> GPU2
  double *h_C2 = (double*) malloc(bytes);  // GPU1 -> GPU3

  double *h_A3 = (double*) malloc(bytes);  // GPU2 -> GPU0
  double *h_B3 = (double*) malloc(bytes);  // GPU2 -> GPU1
  double *h_C3 = (double*) malloc(bytes);  // GPU2 -> GPU3

  double *h_A4 = (double*) malloc(bytes);  // GPU3 -> GPU0
  double *h_B4 = (double*) malloc(bytes);  // GPU3 -> GPU1
  double *h_C4 = (double*) malloc(bytes);  // GPU3 -> GPU2

  // Receive buffers (one per incoming transfer, used for verification)
  double *h_recv_0_from_1 = (double*) malloc(bytes);
  double *h_recv_0_from_2 = (double*) malloc(bytes);
  double *h_recv_0_from_3 = (double*) malloc(bytes);

  if (!h_A1 || !h_B1 || !h_C1 ||
      !h_A2 || !h_B2 || !h_C2 ||
      !h_A3 || !h_B3 || !h_C3 ||
      !h_A4 || !h_B4 || !h_C4 ||
      !h_recv_0_from_1 || !h_recv_0_from_2 || !h_recv_0_from_3) {
    cerr << "Host allocation failed\n";
    return EXIT_FAILURE;
  }

  for (size_t i = 0; i < N; ++i) {
    h_A1[i] = h_B1[i] = h_C1[i] = 1.0;   // GPU0 sends value 1
    h_A2[i] = h_B2[i] = h_C2[i] = 2.0;   // GPU1 sends value 2
    h_A3[i] = h_B3[i] = h_C3[i] = 3.0;   // GPU2 sends value 3
    h_A4[i] = h_B4[i] = h_C4[i] = 4.0;   // GPU3 sends value 4
  }

  // ── Device allocation ────────────────────────────────────────────────────
  // Source buffers (d_X<gpu>) and receive buffers (d_recv_<dst>_from_<src>)

  // GPU 0: source buffers
  double *d_A1 = (double*) omp_target_alloc(bytes, dev0);  // -> GPU1
  double *d_B1 = (double*) omp_target_alloc(bytes, dev0);  // -> GPU2
  double *d_C1 = (double*) omp_target_alloc(bytes, dev0);  // -> GPU3
  // GPU 0: receive buffers
  double *d_recv_0_from_1 = (double*) omp_target_alloc(bytes, dev0);
  double *d_recv_0_from_2 = (double*) omp_target_alloc(bytes, dev0);
  double *d_recv_0_from_3 = (double*) omp_target_alloc(bytes, dev0);

  // GPU 1: source buffers
  double *d_A2 = (double*) omp_target_alloc(bytes, dev1);  // -> GPU0
  double *d_B2 = (double*) omp_target_alloc(bytes, dev1);  // -> GPU2
  double *d_C2 = (double*) omp_target_alloc(bytes, dev1);  // -> GPU3
  // GPU 1: receive buffers
  double *d_recv_1_from_0 = (double*) omp_target_alloc(bytes, dev1);
  double *d_recv_1_from_2 = (double*) omp_target_alloc(bytes, dev1);
  double *d_recv_1_from_3 = (double*) omp_target_alloc(bytes, dev1);

  // GPU 2: source buffers
  double *d_A3 = (double*) omp_target_alloc(bytes, dev2);  // -> GPU0
  double *d_B3 = (double*) omp_target_alloc(bytes, dev2);  // -> GPU1
  double *d_C3 = (double*) omp_target_alloc(bytes, dev2);  // -> GPU3
  // GPU 2: receive buffers
  double *d_recv_2_from_0 = (double*) omp_target_alloc(bytes, dev2);
  double *d_recv_2_from_1 = (double*) omp_target_alloc(bytes, dev2);
  double *d_recv_2_from_3 = (double*) omp_target_alloc(bytes, dev2);

  // GPU 3: source buffers
  double *d_A4 = (double*) omp_target_alloc(bytes, dev3);  // -> GPU0
  double *d_B4 = (double*) omp_target_alloc(bytes, dev3);  // -> GPU1
  double *d_C4 = (double*) omp_target_alloc(bytes, dev3);  // -> GPU2
  // GPU 3: receive buffers
  double *d_recv_3_from_0 = (double*) omp_target_alloc(bytes, dev3);
  double *d_recv_3_from_1 = (double*) omp_target_alloc(bytes, dev3);
  double *d_recv_3_from_2 = (double*) omp_target_alloc(bytes, dev3);

  if (!d_A1 || !d_B1 || !d_C1 || !d_recv_0_from_1 || !d_recv_0_from_2 || !d_recv_0_from_3 ||
      !d_A2 || !d_B2 || !d_C2 || !d_recv_1_from_0 || !d_recv_1_from_2 || !d_recv_1_from_3 ||
      !d_A3 || !d_B3 || !d_C3 || !d_recv_2_from_0 || !d_recv_2_from_1 || !d_recv_2_from_3 ||
      !d_A4 || !d_B4 || !d_C4 || !d_recv_3_from_0 || !d_recv_3_from_1 || !d_recv_3_from_2) {
    cerr << "Device allocation failed\n";
    return EXIT_FAILURE;
  }

  // ── Host → Device copies ─────────────────────────────────────────────────
  if (omp_target_memcpy(d_A1, h_A1, bytes, 0, 0, dev0, host_id) != 0 ||
      omp_target_memcpy(d_B1, h_B1, bytes, 0, 0, dev0, host_id) != 0 ||
      omp_target_memcpy(d_C1, h_C1, bytes, 0, 0, dev0, host_id) != 0) {
    cerr << "H->D copy for dev0 failed\n"; return EXIT_FAILURE;
  }
  if (omp_target_memcpy(d_A2, h_A2, bytes, 0, 0, dev1, host_id) != 0 ||
      omp_target_memcpy(d_B2, h_B2, bytes, 0, 0, dev1, host_id) != 0 ||
      omp_target_memcpy(d_C2, h_C2, bytes, 0, 0, dev1, host_id) != 0) {
    cerr << "H->D copy for dev1 failed\n"; return EXIT_FAILURE;
  }
  if (omp_target_memcpy(d_A3, h_A3, bytes, 0, 0, dev2, host_id) != 0 ||
      omp_target_memcpy(d_B3, h_B3, bytes, 0, 0, dev2, host_id) != 0 ||
      omp_target_memcpy(d_C3, h_C3, bytes, 0, 0, dev2, host_id) != 0) {
    cerr << "H->D copy for dev2 failed\n"; return EXIT_FAILURE;
  }
  if (omp_target_memcpy(d_A4, h_A4, bytes, 0, 0, dev3, host_id) != 0 ||
      omp_target_memcpy(d_B4, h_B4, bytes, 0, 0, dev3, host_id) != 0 ||
      omp_target_memcpy(d_C4, h_C4, bytes, 0, 0, dev3, host_id) != 0) {
    cerr << "H->D copy for dev3 failed\n"; return EXIT_FAILURE;
  }

  // ── All-to-all async P2P transfers ───────────────────────────────────────
  // 12 transfers total, grouped by source GPU into 4 concurrent sections.
  // Each section launches 3 async transfers then returns; taskwait below
  // ensures all 12 complete before timing stops.
  //
  //  Section 0  GPU0 -> GPU1, GPU0 -> GPU2, GPU0 -> GPU3
  //  Section 1  GPU1 -> GPU0, GPU1 -> GPU2, GPU1 -> GPU3
  //  Section 2  GPU2 -> GPU0, GPU2 -> GPU1, GPU2 -> GPU3
  //  Section 3  GPU3 -> GPU0, GPU3 -> GPU1, GPU3 -> GPU2

  int rc00 = 0, rc01 = 0, rc02 = 0;  // GPU0 -> 1,2,3
  int rc10 = 0, rc12 = 0, rc13 = 0;  // GPU1 -> 0,2,3
  int rc20 = 0, rc21 = 0, rc23 = 0;  // GPU2 -> 0,1,3
  int rc30 = 0, rc31 = 0, rc32 = 0;  // GPU3 -> 0,1,2

  omp_depend_t dummy;
  omp_set_num_threads(4);

  // ── Warm-up P2P transfers (not timed) ────────────────────────────────────
  for (int w = 0; w < 2; ++w)
    {
      omp_target_memcpy_async(d_recv_1_from_0, d_A1, bytes, 0, 0, dev1, dev0, 0, NULL);
      omp_target_memcpy_async(d_recv_2_from_0, d_B1, bytes, 0, 0, dev2, dev0, 0, NULL);
      omp_target_memcpy_async(d_recv_3_from_0, d_C1, bytes, 0, 0, dev3, dev0, 0, NULL);
      omp_target_memcpy_async(d_recv_0_from_1, d_A2, bytes, 0, 0, dev0, dev1, 0, NULL);
      omp_target_memcpy_async(d_recv_2_from_1, d_B2, bytes, 0, 0, dev2, dev1, 0, NULL);
      omp_target_memcpy_async(d_recv_3_from_1, d_C2, bytes, 0, 0, dev3, dev1, 0, NULL);
      omp_target_memcpy_async(d_recv_0_from_2, d_A3, bytes, 0, 0, dev0, dev2, 0, NULL);
      omp_target_memcpy_async(d_recv_1_from_2, d_B3, bytes, 0, 0, dev1, dev2, 0, NULL);
      omp_target_memcpy_async(d_recv_3_from_2, d_C3, bytes, 0, 0, dev3, dev2, 0, NULL);
      omp_target_memcpy_async(d_recv_0_from_3, d_A4, bytes, 0, 0, dev0, dev3, 0, NULL);
      omp_target_memcpy_async(d_recv_1_from_3, d_B4, bytes, 0, 0, dev1, dev3, 0, NULL);
      omp_target_memcpy_async(d_recv_2_from_3, d_C4, bytes, 0, 0, dev2, dev3, 0, NULL);
#pragma omp taskwait
    }
#pragma omp taskwait
	
  const double start = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp sections
    {
      // ── GPU 0 sends to GPUs 1, 2, 3 ─────────────────────────────────────
#pragma omp section
      {
        rc00 = omp_target_memcpy_async(d_recv_1_from_0, d_A1, bytes, 0, 0, dev1, dev0, 0, NULL);
        rc01 = omp_target_memcpy_async(d_recv_2_from_0, d_B1, bytes, 0, 0, dev2, dev0, 0, NULL);
        rc02 = omp_target_memcpy_async(d_recv_3_from_0, d_C1, bytes, 0, 0, dev3, dev0, 0, NULL);
      }
      // ── GPU 1 sends to GPUs 0, 2, 3 ─────────────────────────────────────
#pragma omp section
      {
        rc10 = omp_target_memcpy_async(d_recv_0_from_1, d_A2, bytes, 0, 0, dev0, dev1, 0, NULL);
        rc12 = omp_target_memcpy_async(d_recv_2_from_1, d_B2, bytes, 0, 0, dev2, dev1, 0, NULL);
        rc13 = omp_target_memcpy_async(d_recv_3_from_1, d_C2, bytes, 0, 0, dev3, dev1, 0, NULL);
      }
      // ── GPU 2 sends to GPUs 0, 1, 3 ─────────────────────────────────────
#pragma omp section
      {
        rc20 = omp_target_memcpy_async(d_recv_0_from_2, d_A3, bytes, 0, 0, dev0, dev2, 0, NULL);
        rc21 = omp_target_memcpy_async(d_recv_1_from_2, d_B3, bytes, 0, 0, dev1, dev2, 0, NULL);
        rc23 = omp_target_memcpy_async(d_recv_3_from_2, d_C3, bytes, 0, 0, dev3, dev2, 0, NULL);
      }
      // ── GPU 3 sends to GPUs 0, 1, 2 ─────────────────────────────────────
#pragma omp section
      {
        rc30 = omp_target_memcpy_async(d_recv_0_from_3, d_A4, bytes, 0, 0, dev0, dev3, 0, NULL);
        rc31 = omp_target_memcpy_async(d_recv_1_from_3, d_B4, bytes, 0, 0, dev1, dev3, 0, NULL);
        rc32 = omp_target_memcpy_async(d_recv_2_from_3, d_C4, bytes, 0, 0, dev2, dev3, 0, NULL);
      }
    }
  }
#pragma omp taskwait

  const double end = omp_get_wtime();

  if (rc00 != 0 || rc01 != 0 || rc02 != 0 ||
      rc10 != 0 || rc12 != 0 || rc13 != 0 ||
      rc20 != 0 || rc21 != 0 || rc23 != 0 ||
      rc30 != 0 || rc31 != 0 || rc32 != 0) {
    cerr << "One or more async P2P transfers failed\n";
    return EXIT_FAILURE;
  }

  // ── Verification (spot-check GPU0's three receive buffers) ───────────────
  if (omp_target_memcpy(h_recv_0_from_1, d_recv_0_from_1, bytes, 0, 0, host_id, dev0) != 0 ||
      omp_target_memcpy(h_recv_0_from_2, d_recv_0_from_2, bytes, 0, 0, host_id, dev0) != 0 ||
      omp_target_memcpy(h_recv_0_from_3, d_recv_0_from_3, bytes, 0, 0, host_id, dev0) != 0) {
    cerr << "D->H verification copy failed\n"; return EXIT_FAILURE;
  }

  bool ok = true;
  for (size_t i = 0; i < min<size_t>(N, 10); ++i) {
    if (h_recv_0_from_1[i] != 2.0 ||
        h_recv_0_from_2[i] != 3.0 ||
        h_recv_0_from_3[i] != 4.0) {
      ok = false;
      break;
    }
  }

  // ── Bandwidth report ─────────────────────────────────────────────────────
  const double seconds       = end - start;
  const double total_bytes   = 12.0 * bytes;   // 12 P2P transfers
  const double bandwidthGBps = total_bytes / (seconds * 1e9);

  cout << fixed << setprecision(6);
  cout << "Elapsed time:      " << seconds << " seconds\n";
  cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
  cout << "P2P Bandwidth:     " << bandwidthGBps << " GB/s\n";
  cout << "Verification:      " << (ok ? "PASS" : "FAIL") << "\n";

  // ── Free device memory ───────────────────────────────────────────────────
  omp_target_free(d_A1, dev0); omp_target_free(d_B1, dev0); omp_target_free(d_C1, dev0);
  omp_target_free(d_recv_0_from_1, dev0);
  omp_target_free(d_recv_0_from_2, dev0);
  omp_target_free(d_recv_0_from_3, dev0);

  omp_target_free(d_A2, dev1); omp_target_free(d_B2, dev1); omp_target_free(d_C2, dev1);
  omp_target_free(d_recv_1_from_0, dev1);
  omp_target_free(d_recv_1_from_2, dev1);
  omp_target_free(d_recv_1_from_3, dev1);

  omp_target_free(d_A3, dev2); omp_target_free(d_B3, dev2); omp_target_free(d_C3, dev2);
  omp_target_free(d_recv_2_from_0, dev2);
  omp_target_free(d_recv_2_from_1, dev2);
  omp_target_free(d_recv_2_from_3, dev2);

  omp_target_free(d_A4, dev3); omp_target_free(d_B4, dev3); omp_target_free(d_C4, dev3);
  omp_target_free(d_recv_3_from_0, dev3);
  omp_target_free(d_recv_3_from_1, dev3);
  omp_target_free(d_recv_3_from_2, dev3);

  // ── Free host memory ─────────────────────────────────────────────────────
  free(h_A1); free(h_B1); free(h_C1);
  free(h_A2); free(h_B2); free(h_C2);
  free(h_A3); free(h_B3); free(h_C3);
  free(h_A4); free(h_B4); free(h_C4);
  free(h_recv_0_from_1); free(h_recv_0_from_2); free(h_recv_0_from_3);

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
