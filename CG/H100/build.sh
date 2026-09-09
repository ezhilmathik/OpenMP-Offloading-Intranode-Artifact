#!/bin/bash
# ---------------------------------------------------------------------------
# build.sh -- the build step. NOT an sbatch job.
#
# verify.sh and production.sh both inline their own build, so this script is
# only for building by hand inside an existing allocation:
#
#     salloc -A ehpc681 -p acc --qos acc_ehpc -N 1 --gres=gpu:4 -t 00:30:00
#     ./build.sh
#
# It must run on a COMPUTE node: cg-cpu is compiled with -march=native and the
# GPU arch is detected from nvidia-smi, so a login-node build would target the
# wrong CPU and have no GPU to detect.
#
# Environment:
#   CG_MODULES   modules to load, in order
#   ARCH         GPU arch; auto-detected from nvidia-smi when unset
# ---------------------------------------------------------------------------

set -uo pipefail

CG_MODULES="${CG_MODULES:-EB/apps GCC/13.2.0 cuda/12.8 clang/18.1.8-cuda12.8}"

module purge 2>/dev/null
for m in $CG_MODULES; do
    if ! module load "$m" 2>&1; then
        echo "ERROR: module load $m failed -- check 'module spider' on this site" >&2
        exit 1
    fi
done

command -v clang >/dev/null || { echo "ERROR: clang not on PATH after module load" >&2; exit 1; }

if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not found -- are you on a login node?" >&2
    echo "       Build inside an allocation, or set ARCH=sm_XX by hand." >&2
    exit 1
fi

# compute_cap comes back as e.g. 9.0 (H100); strip the dot.
if [[ -z "${ARCH:-}" ]]; then
    CC_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
    [[ -n "$CC_RAW" ]] || { echo "ERROR: could not detect compute capability" >&2; exit 1; }
    ARCH="sm_${CC_RAW}"
fi
export ARCH

echo "============================================================"
echo "BUILD"
echo "  host    : $(hostname)"
echo "  date    : $(date)"
echo "  modules : ${CG_MODULES}"
echo "  arch    : ${ARCH}"
echo "  cpu cc  : $(clang --version 2>/dev/null | head -1)"
echo "  cpu     : $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')"
nvidia-smi -L 2>/dev/null || echo "  (no GPU visible)"
echo "============================================================"

# Always from scratch. A stale object built for a different node type, or
# against a different module set, is a silent source of wrong numbers.
make clean
make list
echo ""

if ! make -j4 all; then
    echo ""
    echo "BUILD FAILED"
    exit 1
fi

echo ""
echo "Binaries:"

EXPECTED=(cg-cpu)
for n in 1 2 4; do
    [[ -f "${n}-omp.c"      ]] && EXPECTED+=("cg-ompgpu${n}")
    [[ -f "${n}-omp-halo.c" ]] && EXPECTED+=("cg-ompgpu${n}halo")
done

MISSING=()
for bin in "${EXPECTED[@]}"; do
    if [[ -x "./$bin" ]]; then
        echo "  OK      : $bin"
    else
        echo "  MISSING : $bin"
        MISSING+=("$bin")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo ""
    echo "BUILD FAILED: missing ${MISSING[*]}"
    exit 1
fi

echo ""
echo "BUILD OK"
echo ""
