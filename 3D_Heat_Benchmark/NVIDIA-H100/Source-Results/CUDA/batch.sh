#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time=00:30:00
#SBATCH --account=p201103
#SBATCH --exclusive
# Default GPU count (can be overridden by: sbatch --gpus=2 or --gpus=4)
#SBATCH --gpus=4

set -euo pipefail

# -----------------------------
# Args
#   batch.sh <N> <DT> <test|no-test> <write|no-write> <GPU_MODE>
#   GPU_MODE: 2 | 4 | all
# -----------------------------
#N="${1:-256}"
#DT="${2:-0.00675}"
N="${1:-128}"
DT="${2:-0.007}"
MODE="${3:-test}"          # test | no-test
WRITE_MODE="${4:-write}"   # write | no-write
GPU_MODE="${5:-all}"       # all | 2 | 4

if [[ "$MODE" != "test" && "$MODE" != "no-test" ]]; then
  echo "ERROR: MODE must be 'test' or 'no-test' (got: '$MODE')"
  echo "Usage: sbatch batch.sh <N> <DT> <test|no-test> <write|no-write> <all|2|4>"
  exit 1
fi

if [[ "$WRITE_MODE" != "write" && "$WRITE_MODE" != "no-write" ]]; then
  echo "ERROR: WRITE_MODE must be 'write' or 'no-write' (got: '$WRITE_MODE')"
  echo "Usage: sbatch batch.sh <N> <DT> <test|no-test> <write|no-write> <all|2|4>"
  exit 1
fi

if [[ "$GPU_MODE" != "all" && "$GPU_MODE" != "2" && "$GPU_MODE" != "4" ]]; then
  echo "ERROR: GPU_MODE must be all|2|4 (got: '$GPU_MODE')"
  echo "Usage: sbatch batch.sh <N> <DT> <test|no-test> <write|no-write> <all|2|4>"
  exit 1
fi

DO_COMPARE=false
[[ "$MODE" == "test" ]] && DO_COMPARE=true

# Control writing inside codes via environment variable:
WRITE_TXT=1
[[ "$WRITE_MODE" == "no-write" ]] && WRITE_TXT=0
export WRITE_TXT

if [[ "$WRITE_TXT" -eq 0 ]]; then
  DO_COMPARE=false
fi

# -----------------------------
# Run 1-GPU baseline ONLY in test+write mode
# -----------------------------
RUN_1GPU=true
if [[ "$MODE" != "test" || "$WRITE_MODE" != "write" ]]; then
  RUN_1GPU=false
fi

echo "Using N=${N}, DT=${DT}, MODE=${MODE}, WRITE_MODE=${WRITE_MODE}, GPU_MODE=${GPU_MODE} (WRITE_TXT=${WRITE_TXT}, RUN_1GPU=${RUN_1GPU})"

# -----------------------------
# GPU visibility sanity
# -----------------------------
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

# Fail fast if GPUs allocated < requested mode
ALLOC_GPUS=0
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  ALLOC_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  ALLOC_GPUS=$(awk -F',' '{print NF}' <<< "${SLURM_JOB_GPUS}")
fi

if [[ "$GPU_MODE" == "4" && "$ALLOC_GPUS" -gt 0 && "$ALLOC_GPUS" -lt 4 ]]; then
  echo "ERROR: GPU_MODE=4 but only ${ALLOC_GPUS} GPU(s) visible/allocated."
  exit 1
fi
if [[ "$GPU_MODE" == "2" && "$ALLOC_GPUS" -gt 0 && "$ALLOC_GPUS" -lt 2 ]]; then
  echo "ERROR: GPU_MODE=2 but only ${ALLOC_GPUS} GPU(s) visible/allocated."
  exit 1
fi

# -----------------------------
# Modules / environment
# -----------------------------
module load env/release/2023.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.2.0

# -----------------------------
# Config
# -----------------------------
ARCH="sm_80"
CUDA_COMMON_FLAGS=(--fmad=false --prec-div=true --prec-sqrt=true -arch=${ARCH} -Xcompiler -fomp)
CUDA_O2=(-O2)

# -----------------------------
# Output files
# -----------------------------
U_1GPU="u_cuda_1gpu.txt"

U_2GPU="u_cuda_2gpu.txt"
U_2GPU_STREAM="u_cuda_2gpu_stream.txt"
U_2GPU_STREAM_OMP="u_cuda_2gpu_stream_omp.txt"
U_2GPU_STREAM_OMP_P2P="u_cuda_2gpu_stream_omp_p2p.txt"

U_4GPU="u_cuda_4gpu.txt"
U_4GPU_STREAM="u_cuda_4gpu_stream.txt"
U_4GPU_STREAM_OMP="u_cuda_4gpu_stream_omp.txt"
U_4GPU_STREAM_OMP_P2P="u_cuda_4gpu_stream_omp_p2p.txt"

# -----------------------------
# Sources / executables
# -----------------------------
SRC_SERIAL="heat3D_variable_coeff.c"
EXE_SERIAL="heat3D_serial"

SRC_1GPU="1-gpu-correct.cu"
EXE_1GPU="heat3D_cuda_1gpu"

SRC_2GPU="2-gpu-correct.cu"
EXE_2GPU="heat3D_cuda_2gpu"

SRC_2GPU_STREAM="2-gpu-stream-correct.cu"
EXE_2GPU_STREAM="2-gpu-stream"

SRC_2GPU_STREAM_OMP="2-gpu-stream-omp-correct.cu"
EXE_2GPU_STREAM_OMP="2-gpu-stream-omp"

SRC_2GPU_STREAM_OMP_P2P="2-gpu-stream-omp-p2p-correct.cu"
EXE_2GPU_STREAM_OMP_P2P="2-gpu-stream-omp-p2p"

SRC_4GPU="4-gpu-correct.cu"
EXE_4GPU="heat3D_cuda_4gpu"

SRC_4GPU_STREAM="4-gpu-stream-correct.cu"
EXE_4GPU_STREAM="4-gpu-stream"

SRC_4GPU_STREAM_OMP="4-gpu-stream-omp-correct.cu"
EXE_4GPU_STREAM_OMP="4-gpu-stream-omp"

SRC_4GPU_STREAM_OMP_P2P="4-gpu-stream-omp-p2p-correct.cu"
EXE_4GPU_STREAM_OMP_P2P="4-gpu-stream-omp-p2p"

CMP="compare_solutions.py"

# -----------------------------
# Helpers
# -----------------------------
banner() { echo -e "\n========== $* ==========\n"; }

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing '$1' in PATH"; exit 1; }
}

run_only() {
  local exe="$1"
  echo "Running: ./${exe} ${N} ${DT}  (WRITE_TXT=${WRITE_TXT})"
  ./"$exe" "$N" "$DT"
}

run_and_check() {
  local exe="$1"
  local out="$2"
  run_only "$exe"
  if [[ ! -f "$out" ]]; then
    echo "ERROR: Expected output file '$out' was not created by '$exe'."
    exit 1
  fi
}

RUNNER="run_only"
$DO_COMPARE && RUNNER="run_and_check"

# -----------------------------
# Sanity checks
# -----------------------------
need gcc
need nvcc
need python3
test -f "$CMP" || { echo "ERROR: missing $CMP"; exit 1; }

for f in \
  "$SRC_SERIAL" \
  "$SRC_1GPU" \
  "$SRC_2GPU" \
  "$SRC_2GPU_STREAM" \
  "$SRC_2GPU_STREAM_OMP" \
  "$SRC_2GPU_STREAM_OMP_P2P" \
  "$SRC_4GPU" \
  "$SRC_4GPU_STREAM" \
  "$SRC_4GPU_STREAM_OMP" \
  "$SRC_4GPU_STREAM_OMP_P2P"
do
  test -f "$f" || { echo "ERROR: missing source file: $f"; exit 1; }
done

# -----------------------------
# Build
# -----------------------------
banner "Building"

echo "gcc  -> $EXE_SERIAL"
gcc "$SRC_SERIAL" -lm -fomp -O3 -o "$EXE_SERIAL"

echo "nvcc -> $EXE_1GPU"
nvcc "${CUDA_O2[@]}" "${CUDA_COMMON_FLAGS[@]}" -o "$EXE_1GPU" "$SRC_1GPU"

echo "nvcc -> $EXE_2GPU"
nvcc "${CUDA_O2[@]}" "${CUDA_COMMON_FLAGS[@]}" -o "$EXE_2GPU" "$SRC_2GPU"

echo "nvcc -> $EXE_2GPU_STREAM"
nvcc "${CUDA_O2[@]}" "${CUDA_COMMON_FLAGS[@]}" -o "$EXE_2GPU_STREAM" "$SRC_2GPU_STREAM"

echo "nvcc -> $EXE_2GPU_STREAM_OMP"
nvcc "${CUDA_O2[@]}" "${CUDA_COMMON_FLAGS[@]}" -o "$EXE_2GPU_STREAM_OMP" "$SRC_2GPU_STREAM_OMP"

echo "nvcc -> $EXE_2GPU_STREAM_OMP_P2P"
nvcc "${CUDA_O2[@]}" "${CUDA_COMMON_FLAGS[@]}" -o "$EXE_2GPU_STREAM_OMP_P2P" "$SRC_2GPU_STREAM_OMP_P2P"

echo "nvcc -> $EXE_4GPU"
nvcc "${CUDA_O2[@]}" "${CUDA_COMMON_FLAGS[@]}" -o "$EXE_4GPU" "$SRC_4GPU"

echo "nvcc -> $EXE_4GPU_STREAM"
nvcc "${CUDA_O2[@]}" "${CUDA_COMMON_FLAGS[@]}" -o "$EXE_4GPU_STREAM" "$SRC_4GPU_STREAM"

echo "nvcc -> $EXE_4GPU_STREAM_OMP"
nvcc "${CUDA_O2[@]}" "${CUDA_COMMON_FLAGS[@]}" -o "$EXE_4GPU_STREAM_OMP" "$SRC_4GPU_STREAM_OMP"

echo "nvcc -> $EXE_4GPU_STREAM_OMP_P2P"
nvcc "${CUDA_O2[@]}" "${CUDA_COMMON_FLAGS[@]}" -o "$EXE_4GPU_STREAM_OMP_P2P" "$SRC_4GPU_STREAM_OMP_P2P"

# -----------------------------
# Run
# -----------------------------
banner "Running (N=${N}, dt=${DT}, GPU_MODE=${GPU_MODE})"

# Baseline (only in test+write)
if $RUN_1GPU; then
  $RUNNER "$EXE_1GPU" "$U_1GPU"
else
  echo "Skipping 1-GPU baseline (MODE=${MODE}, WRITE_MODE=${WRITE_MODE})."
fi

if [[ "$GPU_MODE" == "2" || "$GPU_MODE" == "all" ]]; then
  $RUNNER "$EXE_2GPU" "$U_2GPU"
  $RUNNER "$EXE_2GPU_STREAM" "$U_2GPU_STREAM"
  $RUNNER "$EXE_2GPU_STREAM_OMP" "$U_2GPU_STREAM_OMP"
  $RUNNER "$EXE_2GPU_STREAM_OMP_P2P" "$U_2GPU_STREAM_OMP_P2P"
fi

if [[ "$GPU_MODE" == "4" || "$GPU_MODE" == "all" ]]; then
  $RUNNER "$EXE_4GPU" "$U_4GPU"
  $RUNNER "$EXE_4GPU_STREAM" "$U_4GPU_STREAM"
  $RUNNER "$EXE_4GPU_STREAM_OMP" "$U_4GPU_STREAM_OMP"
  $RUNNER "$EXE_4GPU_STREAM_OMP_P2P" "$U_4GPU_STREAM_OMP_P2P"
fi

# -----------------------------
# Compare (only in test+write)
# -----------------------------
if $DO_COMPARE; then
  banner "Comparisons"

  if ! $RUN_1GPU; then
    echo "Skipping comparisons because 1-GPU reference was not generated (not in test+write)."
    exit 0
  fi

  if [[ "$GPU_MODE" == "2" || "$GPU_MODE" == "all" ]]; then
    python3 "$CMP" "$U_1GPU" "$U_2GPU"
    python3 "$CMP" "$U_1GPU" "$U_2GPU_STREAM"
    python3 "$CMP" "$U_1GPU" "$U_2GPU_STREAM_OMP"
    python3 "$CMP" "$U_1GPU" "$U_2GPU_STREAM_OMP_P2P"
  fi

  if [[ "$GPU_MODE" == "4" || "$GPU_MODE" == "all" ]]; then
    python3 "$CMP" "$U_1GPU" "$U_4GPU"
    python3 "$CMP" "$U_1GPU" "$U_4GPU_STREAM"
    python3 "$CMP" "$U_1GPU" "$U_4GPU_STREAM_OMP"
    python3 "$CMP" "$U_1GPU" "$U_4GPU_STREAM_OMP_P2P"
  fi
else
  banner "Comparisons skipped (MODE=${MODE}, WRITE_MODE=${WRITE_MODE})"
fi

banner "Done"
