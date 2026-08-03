#!/bin/bash
#SBATCH --job-name=heat-build
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465003145
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=build-%j.out

module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4
module load lumi-CrayPath

make clean
make -j4 all gpu-baseline
make check-p2p-lib || exit 1

# NOTE: the CPU reference (`serial`) is deliberately NOT built here. Production
# timing runs never execute it and never write .txt solutions. Build it by hand
# with `make serial-ref` when you want to regenerate u_serial.txt for a diff.

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
