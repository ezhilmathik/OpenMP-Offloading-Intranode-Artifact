#!/bin/bash -l
#SBATCH --job-name=heat-build
#SBATCH --account=pn67qe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=build-%j.out
#
# Intel Data Center GPU Max 1550 (PVC). Whole node = 4 cards, no --gres needed.
# If sinfo shows a different GPU partition on this system, change --partition
# here, in run.sh and in verify.sh together.

module load slurm_setup
module load intel-toolkit

echo "=== Host     : $(hostname) ==="
echo "=== Date     : $(date) ==="
echo "=== Compiler : $(icx --version | head -1) ==="

make clean
make -j4 all gpu-baseline || exit 1

# NOTE: the CPU reference (`serial`) is deliberately NOT built here. Production
# timing runs never execute it and never write .txt solutions. Build it with
# `make serial-ref` when verify.sh needs it.

echo ""
echo "Verifying binaries..."

EXPECTED=(
    2-omp 2-omp-stream 2-omp-stream-omp 2-omp-stream-omp-p2p
    4-omp 4-omp-stream 4-omp-stream-omp 4-omp-stream-omp-p2p
    1-omp
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
echo "All built with identical flags (see OPT in the Makefile)."
