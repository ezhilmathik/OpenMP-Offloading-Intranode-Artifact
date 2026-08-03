#!/bin/bash -l
#SBATCH --job-name=heat-build
#SBATCH --account=ehpc681
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=build-%j.out
#
# NVIDIA H100 (sm_90), MareNostrum 5 acc partition.
# --exclusive on one acc node gives all 4 H100s; no --gres needed.
# If your site requires it instead, add:  #SBATCH --gres=gpu:4

module purge
module load EB/apps
module load GCC/13.2.0
module load cuda/12.8
module load clang/18.1.8-cuda12.8

echo "=== Host     : $(hostname) ==="
echo "=== Date     : $(date) ==="
echo "=== Compiler : $(clang --version | head -1) ==="
nvidia-smi -L || true

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
echo "All built with identical flags (see OPT/OMP_FLAGS in the Makefile)."
