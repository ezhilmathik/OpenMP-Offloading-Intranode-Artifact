#!/bin/bash
# ---------------------------------------------------------------------------
# Relax the P2P availability guard in the two peer-to-peer sources.
#
# WHY, measured on this node (probe_p2p, 64 MB buffers, oneAPI 2024.1.0):
#
#   ZE_FLAT_DEVICE_HIERARCHY   can_access_peer   direct      host-staged
#   COMPOSITE (card)           NO / NO           62.9 GB/s   25.3 GB/s
#   FLAT (stack)               NO / NO           31.6 GB/s   25.6 GB/s
#
#   ext_oneapi_enable_peer_access: PI_ERROR_INVALID_OPERATION (-59), both modes
#   xpu-smi topology -m: XL8 / XL* between every card pair (Xe Link present)
#   peer copy contents: correct, 0 bad values
#
# A copy that is 2.5x faster than the host-staged path is not going through
# the host. The COMPOSITE rate being 2x the FLAT rate is the signature of both
# stacks of each card driving the link, which host staging would not show --
# and indeed the staged rate is flat across the two modes (25.3 vs 25.6).
#
# So ext_oneapi_can_access_peer() is a false negative on this toolkit, and the
# abort it triggers is the only thing preventing a working variant from
# running. This script turns that abort into a warning. It does NOT touch the
# enable_peer_access call: DPCT_CHECK_ERROR already swallows the exception.
#
# THIS IS A CALIBRATED DECISION, NOT A BLANKET ONE. It is valid only while the
# direct rate stays well above the staged rate. Re-run probe_p2p after any
# toolkit or driver change: if the two rates converge, the variant has become
# a staged copy wearing a p2p label, and the guard must go back to fatal.
#
#   ./relax-p2p-guard.sh
# ---------------------------------------------------------------------------
set -euo pipefail

F2=2-sycl-stream-omp-p2p.cpp
F4=4-sycl-stream-omp-p2p.cpp

for f in "$F2" "$F4"; do
    [[ -f "$f" ]] || { echo "ERROR: $f not found" >&2; exit 1; }
    if grep -q 'known false negative' "$f"; then
        echo "SKIP (already relaxed): $f"; continue
    fi
    cp -n "$f" "$f.hardguard" 2>/dev/null || true
done

# --- 2-GPU ------------------------------------------------------------------
if ! grep -q 'known false negative' "$F2"; then
perl -0777 -i -pe 's{
  \x20\x20if\ \(!can01\ \|\|\ !can10\)\ \{\n
  .*?exit\(EXIT_FAILURE\);\n
  \x20\x20\}\n
}{
  // ext_oneapi_can_access_peer() is a FALSE NEGATIVE on oneAPI 2024.1.0 /
  // Max 1550: it reports no peer access and ext_oneapi_enable_peer_access()
  // throws PI_ERROR_INVALID_OPERATION, yet the cross-device copy in a shared
  // context measures 62.9 GB/s against 25.3 GB/s for the same copy staged
  // through host memory (probe_p2p, 64 MB, COMPOSITE). xpu-smi topology -m
  // shows XL8/XL* Xe Link between every card pair. The capability is there;
  // only the query is broken, so this is a warning rather than an abort.
  //
  // Revisit after any toolkit upgrade: if probe_p2p ever shows the direct and
  // staged rates converging, this variant is silently host-staged and this
  // guard must be fatal again.
  if (!can01 || !can10) {
    fprintf(stderr, "WARNING: ext_oneapi_can_access_peer reports no P2P between "
                    "GPU 0 and GPU 1 -- known false negative on this toolkit; "
                    "proceeding (measured 62.9 vs 25.3 GB/s staged).\\n");
  }
}x' "$F2"
echo "patched: $F2"
fi

# --- 4-GPU ------------------------------------------------------------------
if ! grep -q 'known false negative' "$F4"; then
perl -0777 -i -pe 's{
  \x20\x20\x20\x20if\ \(!can_ab\ \|\|\ !can_ba\)\ \{\n
  .*?exit\(EXIT_FAILURE\);\n
  \x20\x20\x20\x20\}\n
}{
    // False negative on oneAPI 2024.1.0 / Max 1550 -- see the note in
    // 2-sycl-stream-omp-p2p.cpp and the numbers in relax-p2p-guard.sh.
    // Measured 62.9 GB/s direct vs 25.3 GB/s host-staged, and xpu-smi shows
    // Xe Link between every card pair, so the query is wrong, not the link.
    if (!can_ab || !can_ba) {
      fprintf(stderr, "WARNING: ext_oneapi_can_access_peer reports no P2P between "
                      "GPU %d and GPU %d -- known false negative on this "
                      "toolkit; proceeding.\\n", a, b);
    }
}x' "$F4"
echo "patched: $F4"
fi

echo ""
echo "Checking both files still abort on nothing else:"
grep -n 'exit(EXIT_FAILURE)' "$F2" "$F4" || echo "  (no EXIT_FAILURE paths left in the P2P setup)"
echo ""
echo "Next: make clean && make all gpu-baseline, then sbatch verify.sh"
