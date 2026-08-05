#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>
#include <vector>

int main() {
  auto d = sycl::device::get_devices(sycl::info::device_type::gpu);
  printf("visible GPUs: %zu\n", d.size());
  for (size_t i = 0; i < d.size(); ++i)
    printf("  [%zu] %s\n", i, d[i].get_info<sycl::info::device::name>().c_str());
  if (d.size() < 2) { printf("need >=2 devices\n"); return 1; }

  using pa = sycl::ext::oneapi::peer_access;
  printf("can_access_peer 0<->1 : %s / %s\n",
         d[0].ext_oneapi_can_access_peer(d[1], pa::access_supported) ? "yes" : "NO",
         d[1].ext_oneapi_can_access_peer(d[0], pa::access_supported) ? "yes" : "NO");

  try {
    sycl::context ctx(d);
    printf("shared context: built\n");
    try { d[0].ext_oneapi_enable_peer_access(d[1]);
          d[1].ext_oneapi_enable_peer_access(d[0]);
          printf("enable_peer_access: ok\n"); }
    catch (sycl::exception const &e) { printf("enable threw: %s\n", e.what()); }

    sycl::property_list io{sycl::property::queue::in_order()};
    sycl::queue q0(ctx, d[0], io), q1(ctx, d[1], io);

    const size_t N = 8ull*1024*1024;            // 64 MB of doubles
    double *b0 = sycl::malloc_device<double>(N, d[0], ctx);
    double *b1 = sycl::malloc_device<double>(N, d[1], ctx);
    double *bh = sycl::malloc_host<double>(N, ctx);

    std::vector<double> src(N), back(N, 0.0);
    for (size_t i = 0; i < N; ++i) src[i] = (double)i * 1.5;
    q1.memcpy(b1, src.data(), N*sizeof(double)).wait();

    q0.memcpy(b0, b1, N*sizeof(double)).wait();     // the peer copy
    q0.memcpy(back.data(), b0, N*sizeof(double)).wait();
    size_t bad = 0;
    for (size_t i = 0; i < N; ++i) if (back[i] != src[i]) ++bad;
    printf("peer copy contents: %s (%zu bad)\n", bad ? "WRONG" : "correct", bad);

    const int R = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < R; ++r) q0.memcpy(b0, b1, N*sizeof(double));
    q0.wait();
    double direct = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();

    t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < R; ++r) {
      q1.memcpy(bh, b1, N*sizeof(double)).wait();
      q0.memcpy(b0, bh, N*sizeof(double)).wait();
    }
    double staged = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();

    double gb = (double)N*sizeof(double)*R/1e9;
    printf("direct  d1->d0      : %6.1f GB/s\n", gb/direct);
    printf("staged  d1->host->d0: %6.1f GB/s\n", gb/staged);
    printf("ratio               : %6.2fx\n", staged/direct);

    sycl::free(b0,ctx); sycl::free(b1,ctx); sycl::free(bh,ctx);
    return bad ? 1 : 0;
  } catch (sycl::exception const &e) {
    printf("THREW: %s\n", e.what());
    return 1;
  }
}
