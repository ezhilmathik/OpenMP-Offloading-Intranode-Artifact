#!/bin/bash -l
#SBATCH --job-name=build
#SBATCH --time=00:15:00
#SBATCH --account=ehpc681
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=build-%j.out

# ---------------------------------------------------------------------------
# What to build. Defaults match what this study measures: v1 and v2 with
# OpenMP target offload and CUDA. Keep these in step with the matching flags
# in run.sh.
# ---------------------------------------------------------------------------
BUILD_ACC=0      # 1 also builds the OpenACC binaries (requires NVHPC)
BUILD_FUTURE=0   # 1 also builds the four-GPU configurations

set -e

# ---------------------------------------------------------------------------
# OpenMP offload (clang++) + CUDA (nvcc)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "OpenMP offload + CUDA"
echo "============================================================"

module purge
module load EB/apps
module load GCC/13.2.0
module load cuda/12.8
module load clang/18.1.8-cuda12.8

make omp
make cuda

EXPECTED=(
    version-1-off   version-2-off
    version-1-cuda  version-2-cuda
)

if (( BUILD_FUTURE )); then
    echo ""
    echo "============================================================"
    echo "Four-GPU configurations (future work)"
    echo "============================================================"
    make version-4-off version-4-cuda
    EXPECTED+=(version-4-off version-4-cuda)
fi

# ---------------------------------------------------------------------------
# OpenACC (nvc++) — reference only, requires NVHPC
# ---------------------------------------------------------------------------
if (( BUILD_ACC )); then
    echo ""
    echo "============================================================"
    echo "OpenACC (NVHPC)"
    echo "============================================================"

    module purge
    module load EB/apps
    module load NVHPC/23.7-CUDA-12.2.0

    make acc
    EXPECTED+=(version-1-acc version-2-acc)

    if (( BUILD_FUTURE )); then
        make version-4-acc
        EXPECTED+=(version-4-acc)
    fi
fi

# ---------------------------------------------------------------------------
# Verify before declaring success — run jobs depend on this via afterok
# ---------------------------------------------------------------------------
set +e

echo ""
echo "============================================================"
echo "Verifying binaries"
echo "============================================================"

MISSING=()
for bin in "${EXPECTED[@]}"; do
    if [[ -x "./$bin" ]]; then
        echo "  OK      : $bin"
    else
        echo "  MISSING : $bin"
        MISSING+=("$bin")
    fi
done

if (( ${#MISSING[@]} > 0 )); then
    echo ""
    echo "ERROR: ${#MISSING[@]} binary/binaries missing: ${MISSING[*]}"
    echo "Run jobs will NOT start (the afterok dependency will block them)."
    exit 1
fi

echo ""
echo "All ${#EXPECTED[@]} binaries present — run jobs are cleared to start."
echo "============================================================"
