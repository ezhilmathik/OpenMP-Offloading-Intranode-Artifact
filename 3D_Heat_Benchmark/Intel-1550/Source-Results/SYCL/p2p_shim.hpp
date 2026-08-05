#ifndef P2P_SHIM_HPP
#define P2P_SHIM_HPP
// ===========================================================================
// P2P support shim for the DPCT-migrated *-stream-omp-p2p sources.
// Hand-written: DPCT cannot generate this (it is the DPCT1007 it gave up on).
//
// WHY THE MIGRATION STOPPED
// ------------------------
// SYCL has no cudaMemcpyPeerAsync, and does not need one: with USM device
// pointers the peer copy IS just
//
//     q.memcpy(dst, src, bytes);
//
// but only when BOTH pointers belong to the same sycl::context as q. DPCT
// gives every device_ext its own single-device context, so a pointer returned
// by sycl::malloc_device(..., dpct::get_in_order_queue()) on device 1 is not a
// valid address for any queue on device 0 -- the copy would throw no matter
// how it is written. That context split, not a missing API, is what blocked
// the migration.
//
// WHAT THIS FILE DOES
// -------------------
//   1. Builds ONE sycl::context containing every device the run uses.
//   2. Hands out in-order queues on that context, one per device, so the
//      existing dpct::get_in_order_queue() call sites keep working unchanged
//      after a one-token rename to q_cur().
//   3. Provides a cudaMemcpyPeerAsync() with the CUDA signature, so the peer
//      copy call sites do not change AT ALL -- including their surrounding
//      CUDA_CHECK(...) wrappers.
//
// The queues are in-order, matching dpct::device_ext::create_queue()'s default
// (in_order = true). The sources depend on it: on halo_stream the boundary
// kernel must complete before the peer copy that reads the plane it wrote.
// ===========================================================================

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>

// One context spanning every device. This is the whole fix.
static sycl::context              *g_p2p_ctx = nullptr;
static std::vector<sycl::queue *>  g_p2p_default_q;   // one per device
static std::vector<sycl::queue *>  g_p2p_all_q;       // default + extra streams

// Call once, after the device-count check and before any allocation.
static inline void p2p_init(int ngpu)
{
  std::vector<sycl::device> devs;
  for (int d = 0; d < ngpu; ++d)
    devs.push_back(dpct::dev_mgr::instance().get_device(d));

  // All devices must share a platform for this to be constructible; with
  // ONEAPI_DEVICE_SELECTOR=level_zero:gpu they do.
  g_p2p_ctx = new sycl::context(devs);

  for (int d = 0; d < ngpu; ++d) {
    auto *q = new sycl::queue(*g_p2p_ctx, devs[d],
                              sycl::property_list{sycl::property::queue::in_order()});
    g_p2p_default_q.push_back(q);
    g_p2p_all_q.push_back(q);
  }

  printf("P2P shim: one shared context over %d device(s)\n", ngpu);
}

// Index of the device dpct currently has selected. dpct tracks this PER HOST
// THREAD, which is exactly what the `#pragma omp parallel` regions rely on:
// each thread calls dpct::select_device(tid) once and then owns that device.
static inline int p2p_cur_id()
{
  // If your dpct headers do not have get_current_device_id(), use:
  //   return (int)dpct::dev_mgr::instance().current_device_id();
  return (int)dpct::get_current_device_id();
}

// In-order queue on the CURRENT dpct device, in the shared context.
// Drop-in replacement for dpct::get_in_order_queue().
static inline sycl::queue &q_cur()
{
  int d = p2p_cur_id();
  if (d < 0 || d >= (int)g_p2p_default_q.size()) {
    fprintf(stderr, "ERROR: p2p shim used for device %d, only %zu initialised "
                    "(is p2p_init() called before the first allocation?)\n",
            d, g_p2p_default_q.size());
    exit(EXIT_FAILURE);
  }
  return *g_p2p_default_q[d];
}

// Extra streams (compute_stream / halo_stream), also in the shared context.
// Replaces dpct::get_current_device().create_queue().
static inline dpct::queue_ptr p2p_create_queue()
{
  auto *q = new sycl::queue(*g_p2p_ctx,
                            dpct::dev_mgr::instance().get_device(p2p_cur_id()),
                            sycl::property_list{sycl::property::queue::in_order()});
  g_p2p_all_q.push_back(q);
  return q;
}

// Replaces dpct::get_current_device().destroy_queue(q).
static inline void p2p_destroy_queue(dpct::queue_ptr q)
{
  for (size_t i = 0; i < g_p2p_all_q.size(); ++i)
    if (g_p2p_all_q[i] == q) { g_p2p_all_q.erase(g_p2p_all_q.begin() + i); break; }
  delete q;
}

// Replaces dpct::get_current_device().queues_wait_and_throw(). That call only
// drains queues owned by dpct's device_ext -- ours are not, so it would return
// immediately and silently, which is worse than not calling it.
static inline void p2p_wait_current()
{
  sycl::device cur = dpct::dev_mgr::instance().get_device(p2p_cur_id());
  for (auto *q : g_p2p_all_q)
    if (q->get_device() == cur) q->wait_and_throw();
}

// ---------------------------------------------------------------------------
// The peer copy, with the CUDA signature so no call site has to change.
//
// The device-id arguments are ignored on purpose: a USM pointer already
// carries its device, and with the shared context above the runtime resolves
// both ends itself. On Level Zero this issues a direct card-to-card transfer
// over Xe Link once ext_oneapi_enable_peer_access() has been called for the
// pair -- which the sources already do, and already abort without.
//
// Returning dpct::err0(0) keeps the existing CUDA_CHECK(...) wrappers valid.
// If your dpct's err0 is not implicitly constructible from int, use
//   return dpct::err0();
// ---------------------------------------------------------------------------
static inline dpct::err0 cudaMemcpyPeerAsync(void *dst, int /*dst_device*/,
                                             const void *src, int /*src_device*/,
                                             size_t bytes,
                                             dpct::queue_ptr q)
{
  q->memcpy(dst, src, bytes);
  return 0;
}

#endif // P2P_SHIM_HPP
