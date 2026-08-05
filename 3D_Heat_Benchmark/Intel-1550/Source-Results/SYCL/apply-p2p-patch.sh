#!/bin/bash
# ---------------------------------------------------------------------------
# Apply the P2P shim to the two DPCT-migrated peer-to-peer sources.
#
#   ./apply-p2p-patch.sh                 # patches 2- and 4-sycl-stream-omp-p2p.cpp
#   ./apply-p2p-patch.sh 4-sycl-stream-omp-p2p.cpp
#
# Four mechanical renames and two insertions per file. The kernels, the
# decomposition, the stream assignment, the barriers and every peer-copy call
# site are untouched.
#
# Originals are kept as <file>.dpct.orig -- revert with:
#   for f in *.dpct.orig; do mv "$f" "${f%.dpct.orig}"; done
# ---------------------------------------------------------------------------
set -euo pipefail

FILES=("$@")
[[ ${#FILES[@]} -gt 0 ]] || FILES=(2-sycl-stream-omp-p2p.cpp 4-sycl-stream-omp-p2p.cpp)

[[ -f p2p_shim.hpp ]] || { echo "ERROR: p2p_shim.hpp not found in $PWD" >&2; exit 1; }

for f in "${FILES[@]}"; do
    [[ -f "$f" ]] || { echo "SKIP (missing): $f"; continue; }

    case "$f" in
        2-*) NGPU=2 ;;
        4-*) NGPU=4 ;;
        *)   echo "ERROR: cannot tell the device count from '$f'" >&2; exit 1 ;;
    esac

    if grep -q 'p2p_shim.hpp' "$f"; then
        echo "SKIP (already patched): $f"
        continue
    fi

    cp -n "$f" "$f.dpct.orig"
    echo "=== $f  (${NGPU} devices) ==="

    # 1. Pull in the shim. Placed after <omp.h> so it sees dpct's types.
    sed -i 's|^#include <omp.h>$|#include <omp.h>\n#include "p2p_shim.hpp"|' "$f"

    # 2. Build the shared context, immediately after the device-count check
    #    and before the first allocation.
    sed -i "/printf(\"Using CUDA devices/a\\
\\
  // Build the one context that spans all ${NGPU} devices. Must precede every\\
  // sycl::malloc_device below: that is what makes the peer copies legal.\\
  p2p_init(${NGPU});" "$f"

    # 3. Every allocation / H2D / D2H moves into the shared context.
    n=$(grep -c 'dpct::get_in_order_queue()' "$f" || true)
    sed -i 's|dpct::get_in_order_queue()|q_cur()|g' "$f"
    echo "  allocations + transfers rehosted : $n"

    # 4. compute_stream / halo_stream created on the shared context too.
    n=$(grep -c 'dpct::get_current_device().create_queue()' "$f" || true)
    sed -i 's|dpct::get_current_device().create_queue()|p2p_create_queue()|g' "$f"
    echo "  streams created on shared ctx    : $n"

    n=$(grep -c 'dpct::get_current_device().destroy_queue(' "$f" || true)
    sed -i 's|dpct::get_current_device().destroy_queue(|p2p_destroy_queue(|g' "$f"
    echo "  stream teardown                  : $n"

    # 5. queues_wait_and_throw() does not know about our queues.
    n=$(grep -c 'dpct::get_current_device().queues_wait_and_throw()' "$f" || true)
    sed -i 's|dpct::get_current_device().queues_wait_and_throw()|p2p_wait_current()|g' "$f"
    echo "  final drain                      : $n"

    # 6. The peer copies themselves need NO edit: the shim declares
    #    cudaMemcpyPeerAsync() with the CUDA signature. Just report them.
    n=$(grep -c 'cudaMemcpyPeerAsync' "$f" || true)
    echo "  peer copies now resolving to shim: $n  (call sites unchanged)"

    left=$(grep -c 'dpct::get_in_order_queue()\|dpct::get_current_device().create_queue()' "$f" || true)
    [[ "$left" -eq 0 ]] || { echo "  WARNING: $left dpct queue call(s) left unpatched" >&2; }
done

echo ""
echo "Done. Build with the usual flags, e.g.:"
echo "  icpx -O3 -fno-fast-math -fsycl -fiopenmp -ffp-contract=off \\"
echo "       -Wl,--no-warn-execstack 2-sycl-stream-omp-p2p.cpp -lm -o 2-sycl-stream-omp-p2p"
