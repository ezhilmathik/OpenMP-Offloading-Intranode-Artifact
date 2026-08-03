#!/bin/bash
#SBATCH --job-name=asyn-hd-build
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --account=project_465003145
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=build-%j.out

module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4

make clean
make all

echo ""
echo "Verifying binaries..."

EXPECTED=(
    version-1-hip version-2-hip
    version-1-off version-2-off
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
