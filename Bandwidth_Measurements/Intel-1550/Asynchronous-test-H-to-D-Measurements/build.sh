#!/bin/bash -l
#SBATCH --job-name=asyn-hd-build
#SBATCH --account=pn67qe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=build-%j.out

# =============================================================================
# Intel PVC counterpart of the LUMI build.sh.
#   module --force purge / LUMI / PrgEnv-amd / rocm  ->  slurm_setup / intel-toolkit
# =============================================================================

set -uo pipefail

module load slurm_setup
module load intel-toolkit

echo "=== Host: $(hostname) ==="
echo "=== Date: $(date) ==="
echo "=== Compiler: $(icpx --version | head -1) ==="

make clean
make all || { echo "ERROR: build failed"; exit 1; }

echo ""
echo "Verifying binaries..."

EXPECTED=(
    version-1-sycl version-2-sycl
    version-1-off  version-2-off
)

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
    echo "ERROR: ${#MISSING[@]} binary/binaries missing: ${MISSING[*]}"
    exit 1
fi

echo ""
echo "All binaries present -- run jobs are cleared to start."
