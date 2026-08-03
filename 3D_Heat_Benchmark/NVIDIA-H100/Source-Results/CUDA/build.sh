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
# CUDA / NVIDIA H100 (sm_90), MareNostrum 5 acc partition.
# --exclusive on one acc node gives all 4 H100s; no --gres needed.

module purge
module load EB/apps
module load NVHPC/25.3-CUDA-12.8.0

echo "=== Host   : $(hostname) ==="
echo "=== Date   : $(date) ==="
echo "=== nvcc   : $(nvcc --version | tail -2 | head -1) ==="
echo "=== nvc    : $(nvc --version 2>/dev/null | head -2 | tail -1) ==="
nvidia-smi -L || true

echo ""
echo "=== GPU topology (P2P reachability for the *-p2p variants) ==="
nvidia-smi topo -m || true

make clean
make -j4 all gpu-baseline || exit 1

# NOTE: the CPU reference (`serial`) is deliberately NOT built here. Production
# timing runs never execute it and never write .txt solutions. Build it with
# `make serial-ref` when verify.sh needs it.

echo ""
echo "Verifying binaries..."

EXPECTED=(
    2-cuda 2-cuda-stream 2-cuda-stream-omp 2-cuda-stream-omp-p2p
    4-cuda 4-cuda-stream 4-cuda-stream-omp 4-cuda-stream-omp-p2p
    1-cuda
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
echo "All built with identical flags (see NVCC_FLAGS in the Makefile)."
