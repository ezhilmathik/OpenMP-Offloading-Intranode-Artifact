#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <set>

#define CHECK(cmd)                                                        \
  do {                                                                    \
    hipError_t e = (cmd);                                                 \
    if (e != hipSuccess) {                                                \
      fprintf(stderr, "HIP error %s:%d — %s\n",                         \
              __FILE__, __LINE__, hipGetErrorString(e));                  \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)

static double wall() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main() {
  // ── Device count ────────────────────────────────────────────────────────
  int n = 0;
  CHECK(hipGetDeviceCount(&n));
  printf("HIP sees %d device(s)\n\n", n);

  if (n == 0) {
    fprintf(stderr, "No HIP devices found. Is ROCR_VISIBLE_DEVICES set?\n");
    return EXIT_FAILURE;
  }

  // ── Per-device properties ────────────────────────────────────────────────
  printf("%-8s %-30s %-14s %-6s\n", "Device", "Name", "pciBusID", "CUs");
  printf("%-8s %-30s %-14s %-6s\n", "------", "----", "--------", "---");

  std::set<int> busIDs;
  for (int i = 0; i < n; i++) {
    hipDeviceProp_t p;
    CHECK(hipGetDeviceProperties(&p, i));
    bool dup = busIDs.count(p.pciBusID) > 0;
    busIDs.insert(p.pciBusID);
    printf("%-8d %-30s 0x%08x     %-6d %s\n",
           i, p.name, p.pciBusID, p.multiProcessorCount,
           dup ? "  <-- SAME PHYSICAL CARD as another device!" : "");
  }

  // ── Card uniqueness verdict ──────────────────────────────────────────────
  printf("\n");
  if ((int)busIDs.size() == n)
    printf("OK   — all %d devices have distinct pciBusIDs"
           " (one GCD per physical card)\n", n);
  else
    printf("WARN — only %d unique pciBusIDs for %d devices:"
           " some share a physical card\n", (int)busIDs.size(), n);

  // ── Peer access matrix ───────────────────────────────────────────────────
  printf("\nPeer access matrix (1 = direct, 0 = none):\n     ");
  for (int j = 0; j < n; j++) printf("GPU%-4d", j);
  printf("\n");
  for (int i = 0; i < n; i++) {
    printf("GPU%d ", i);
    for (int j = 0; j < n; j++) {
      if (i == j) { printf("  -   "); continue; }
      int can = 0;
      CHECK(hipDeviceCanAccessPeer(&can, i, j));
      printf("  %d   ", can);
    }
    printf("\n");
  }

  // ── Enable peer access ───────────────────────────────────────────────────
  printf("\nEnabling peer access:\n");
  for (int i = 0; i < n; i++) {
    CHECK(hipSetDevice(i));
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      int can = 0;
      CHECK(hipDeviceCanAccessPeer(&can, i, j));
      if (can) {
        hipError_t err = hipDeviceEnablePeerAccess(j, 0);
        if (err == hipSuccess)
          printf("  GPU%d -> GPU%d : enabled\n", i, j);
        else if (err == hipErrorPeerAccessAlreadyEnabled)
          printf("  GPU%d -> GPU%d : already enabled\n", i, j);
        else
          printf("  GPU%d -> GPU%d : FAILED (%s)\n",
                 i, j, hipGetErrorString(err));
      } else {
        printf("  GPU%d -> GPU%d : not supported"
               " (host-staged fallback will be used)\n", i, j);
      }
    }
  }

  // ── P2P smoke test — one stream per directed pair ────────────────────────
  printf("\nP2P smoke test (64 MB per directed pair, %d reps):\n", 5);
  const size_t bytes = 64UL * 1024 * 1024;
  const int    reps  = 5;

  void **src = new void*[n];
  void **dst = new void*[n];
  for (int i = 0; i < n; i++) {
    CHECK(hipSetDevice(i));
    CHECK(hipMalloc(&src[i], bytes));
    CHECK(hipMalloc(&dst[i], bytes));
    CHECK(hipMemset(src[i], i + 1, bytes));
  }

  hipStream_t *st = new hipStream_t[n * n];
  for (int i = 0; i < n; i++) {
    CHECK(hipSetDevice(i));
    for (int j = 0; j < n; j++)
      if (i != j) CHECK(hipStreamCreate(&st[i * n + j]));
  }

  auto sync_all = [&]() {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        if (i != j) CHECK(hipStreamSynchronize(st[i * n + j]));
  };

  auto fire_all = [&]() {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        if (i == j) continue;
        CHECK(hipMemcpyPeerAsync(dst[j], j, src[i], i,
                                 bytes, st[i * n + j]));
      }
  };

  // warm-up
  fire_all(); sync_all();

  // timed
  double t0 = wall();
  for (int r = 0; r < reps; r++) { fire_all(); sync_all(); }
  double elapsed  = (wall() - t0) / reps;
  int    npairs   = n * (n - 1);
  double bw       = (double)npairs * bytes / 1e9 / elapsed;

  printf("  directed pairs : %d\n", npairs);
  printf("  avg elapsed    : %.4f s\n", elapsed);
  printf("  aggregate BW   : %.2f GB/s\n", bw);
  printf("  per-pair BW    : %.2f GB/s\n", bw / npairs);

  // ── Cleanup ──────────────────────────────────────────────────────────────
  for (int i = 0; i < n; i++) {
    CHECK(hipSetDevice(i));
    for (int j = 0; j < n; j++)
      if (i != j) CHECK(hipStreamDestroy(st[i * n + j]));
    CHECK(hipFree(src[i]));
    CHECK(hipFree(dst[i]));
  }
  delete[] st;
  delete[] src;
  delete[] dst;

  return EXIT_SUCCESS;
}
