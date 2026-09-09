#!/bin/bash -l
#SBATCH --job-name=cg-build
#SBATCH --account=pn67qe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=build-%j.out
#
# Intel Data Center GPU Max 1550 (PVC). Whole node = 4 cards, no --gres needed.
# If sinfo shows a different GPU partition on this system, change --partition
# here, in verify.sh and in production.sh together.
#
# verify.sh and production.sh both inline their own build, so this script is
# only for building by hand.

set -uo pipefail

module load slurm_setup
module load intel-toolkit

echo "============================================================"
echo "BUILD"
echo "  host     : $(hostname)"
echo "  date     : $(date)"
echo "  compiler : $(icx --version 2>/dev/null | head -1)"
echo "  cpu      : $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')"
echo "============================================================"
sycl-ls 2>/dev/null || xpu-smi discovery 2>/dev/null || true
echo ""

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
echo "BUILD OK -- all built with identical flags (see OPT in the Makefile)."
echo ""
