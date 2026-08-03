#!/bin/bash
#SBATCH --job-name=heat-build
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465003145
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=build-%j.out

set -euo pipefail

module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4
module load lumi-CrayPath

# Slurm runs a copied script from its spool directory. Build in the directory
# from which ./submit was executed.
WORK_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORK_DIR"

make clean
make -j4 all gpu-baseline

# NOTE: the CPU reference (`serial`) is deliberately NOT built here.
# Production timing jobs never execute it and never write solution files.
# Build it manually with `make serial-ref` only when needed.

echo ""
echo "Verifying binaries..."

EXPECTED=(
    2-hip 2-hip-stream 2-hip-stream-omp 2-hip-stream-omp-p2p
    4-hip 4-hip-stream 4-hip-stream-omp 4-hip-stream-omp-p2p
    1-hip
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
