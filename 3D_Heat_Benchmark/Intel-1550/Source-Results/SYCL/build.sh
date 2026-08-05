#!/bin/bash -l
#SBATCH --job-name=heat-sycl-build
#SBATCH --account=pn67qe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --output=build-sycl-%j.out
#
# Intel Data Center GPU Max 1550 (PVC). Whole node = 4 cards, no --gres needed.
# Longer walltime than the OpenMP build: AOT (-fsycl-targets=spir64_gen) runs
# ocloc per binary, which is slower than SPIR-V-only codegen.
#
#   sbatch build.sh          # AOT (default)
#   AOT=0 sbatch build.sh    # JIT fallback if ocloc is unavailable

module load slurm_setup
module load intel-toolkit
# DPCT-migrated sources need <dpct/dpct.hpp>; harmless if already pulled in.
module load intel-dpct 2>/dev/null || true

AOT="${AOT:-1}"

echo "=== Host     : $(hostname) ==="
echo "=== Date     : $(date) ==="
echo "=== Compiler : $(icpx --version | head -1) ==="
echo "=== ocloc    : $(command -v ocloc || echo 'NOT FOUND') ==="
echo ""
make AOT="$AOT" flags
echo ""

make clean
if ! make -j4 AOT="$AOT" all gpu-baseline; then
    if [[ "$AOT" == "1" ]]; then
        echo ""
        echo "ERROR: AOT build failed. If the failure is from ocloc / spir64_gen,"
        echo "       retry with:  AOT=0 sbatch build.sh"
        echo "       (JIT then happens inside the timed region -- discard rep 1.)"
    fi
    exit 1
fi

# NOTE: the CPU reference (`serial`) is deliberately NOT built here. Production
# timing runs never execute it and never write .txt solutions. Build it with
# `make serial-ref` when a correctness check needs it.

echo ""
echo "Verifying binaries..."

EXPECTED=(
    2-sycl 2-sycl-stream 2-sycl-stream-omp 2-sycl-stream-omp-p2p
    4-sycl 4-sycl-stream 4-sycl-stream-omp 4-sycl-stream-omp-p2p
    1-sycl
)

MISSING=()
for bin in "${EXPECTED[@]}"; do
    if [[ -x "./$bin" ]]; then
        echo "  OK : $bin"
    else
        echo "  MISSING : $bin"
        MISSING+=("$bin")
    fi
done

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo ""
    echo "ERROR: ${#MISSING[@]} binary/binaries missing: ${MISSING[*]}"
    exit 1
fi

echo ""
echo "All binaries present -- run jobs are cleared to start."
echo "All built with identical flags (see OPT/SYCL_FLAGS in the Makefile), AOT=${AOT}."
