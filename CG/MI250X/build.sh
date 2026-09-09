#!/bin/bash
#SBATCH --job-name=cg-build
#SBATCH --account=project_465003145
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=build-%j.out
# ---------------------------------------------------------------------------
# build.sh -- the build step for LUMI-G.
#
# verify.sh and production.sh both inline their own build, so this script is
# only for building by hand:
#
#     sbatch build.sh
#     # or inside an allocation:
#     salloc -A project_465003145 -p standard-g -N 1 --gpus-per-node=8 -t 00:30:00
#     ./build.sh
#
# Unlike the NVIDIA version this can run on a login node too -- the offload
# arch is fixed at gfx90a rather than probed from the hardware, and nothing
# here uses -march=native.
#
# Environment:
#   CG_MODULES          modules to load, in order
#   ARCH                offload arch (default gfx90a)
#   OMPTARGET_P2P_LIB   patched offload runtime
# ---------------------------------------------------------------------------

set -uo pipefail

CG_MODULES="${CG_MODULES:-LUMI/25.09 partition/G craype-accel-amd-gfx90a PrgEnv-amd rocm/6.4.4 lumi-CrayPath}"
ARCH="${ARCH:-gfx90a}"
export ARCH
export OMPTARGET_P2P_LIB="${OMPTARGET_P2P_LIB:-/scratch/project_465003145/omptarget-p2p/lib}"

module --force purge
for m in $CG_MODULES; do
    if ! module load "$m" 2>&1; then
	echo "ERROR: module load $m failed -- check 'module spider' on this site" >&2
	exit 1
    fi
done

command -v cc >/dev/null || { echo "ERROR: cc not on PATH after module load" >&2; exit 1; }

echo "============================================================"
echo "BUILD"
echo "  host    : $(hostname)"
echo "  date    : $(date)"
echo "  modules : ${CG_MODULES}"
echo "  arch    : ${ARCH}"
echo "  compiler: $(cc --version 2>/dev/null | head -1)"
echo "  cpu     : $(lscpu | awk -F: '/Mode
