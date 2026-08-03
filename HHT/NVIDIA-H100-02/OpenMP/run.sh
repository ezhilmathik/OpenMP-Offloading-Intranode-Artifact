#!/bin/bash -l
#SBATCH --job-name=heat-run
#SBATCH --account=ehpc681
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=run-%j.out
#
# One job = one grid size N, one repetition, all requested GPU families.
# The job owns one complete ACC node exclusively. Each benchmark selects
# 1, 2, or 4 visible H100 GPUs through CUDA_VISIBLE_DEVICES.
#
# Inputs (via sbatch --export):
#   N        grid size                      (required)
#   DT       total simulated time           (required)
#   STEPS    step count DT was derived from (recorded in the CSV only)
#   REP      repetition index               (default 1)
#   FAMILIES GPU families to run: "2 4"     (default "2 4")
#   BASELINE 1 -> also run the 1-GPU baseline (default 1)
#
# Optional:
# Fixed runtime policy:
#   1/2/4-omp:
#       one application host thread, no LLVM hidden helpers
#   2-omp-stream:
#       one application host thread, two LLVM hidden helpers
#   4-omp-stream:
#       one application host thread, four LLVM hidden helpers
#   *-stream-omp and P2P:
#       one explicit application host thread per GPU, no hidden helpers
#
# Optional:
#   SLEEP_SEC
#            delay between variants in seconds (default 15)
#   TIMEOUT_SEC
#            timeout per binary in seconds (default 1800)
#   CHECK_GPU_MEM
#            1 -> print GPU memory after each run (default 0)
#
# Examples:
#   Normal mode:
#     sbatch --export=ALL,N=640,DT=0.0001074146,STEPS=500 run.sh
#
#   Strict single-host diagnostic:
#     sbatch --export=ALL,N=640,DT=0.0001074146,STEPS=500,\
#STRICT_SINGLE_HOST=1 run.sh

set -uo pipefail
# Deliberately not using -e: one failing variant must not stop the full sweep.

module purge
module load EB/apps
module load GCC/13.2.0
module load cuda/12.8
module load clang/18.1.8-cuda12.8

# Always run from the directory where sbatch was submitted.
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# -----------------------------------------------------------------------------
# Required inputs and configurable controls
# -----------------------------------------------------------------------------
N="${N:?ERROR: N not set (use: sbatch --export=ALL,N=...,DT=... run.sh)}"
DT="${DT:?ERROR: DT not set}"

STEPS="${STEPS:-unknown}"
REP="${REP:-1}"
FAMILIES="${FAMILIES:-2 4}"
BASELINE="${BASELINE:-1}"
STREAM_HELPERS_2=2
STREAM_HELPERS_4=4
HELPERS=8
SLEEP_SEC="${SLEEP_SEC:-15}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
CHECK_GPU_MEM="${CHECK_GPU_MEM:-0}"


case "$BASELINE" in
    0|1) ;;
    *)
        echo "ERROR: BASELINE must be 0 or 1" >&2
        exit 1
        ;;
esac

# -----------------------------------------------------------------------------
# Common environment
# -----------------------------------------------------------------------------
export OMP_TARGET_OFFLOAD=MANDATORY
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active
export WRITE_TXT=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Required by application-level checks in the *-stream-omp executables.
export OMP_NUM_HIDDEN_HELPER_THREADS=0

# These are set per binary below.
unset CUDA_VISIBLE_DEVICES
unset OMP_THREAD_LIMIT
unset LIBOMP_USE_HIDDEN_HELPER_TASK
unset LIBOMP_NUM_HIDDEN_HELPER_THREADS

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_DIR="results_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

TIMING_CSV="${RESULTS_DIR}/timings.csv"
RUN_LOG="${RESULTS_DIR}/run.log"
SKIPPED_LOG="${RESULTS_DIR}/skipped.log"

echo "rep,job_id,variant,family,gpus,cuda_mask,N,dt,steps,wall_sec,reported_sec,status" \
    > "$TIMING_CSV"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "============================================================"
echo "Host               : $(hostname)"
echo "Job ID             : ${JOB_ID}"
echo "Repetition         : ${REP}"
echo "Timestamp          : ${TIMESTAMP}"
echo "Results dir        : ${RESULTS_DIR}"
echo "N                  : ${N}"
echo "DT                 : ${DT}"
echo "STEPS              : ${STEPS}"
echo "FAMILIES           : '${FAMILIES}'"
echo "BASELINE           : ${BASELINE}"
echo "STREAM_HELPERS_2   : ${STREAM_HELPERS_2}"
echo "STREAM_HELPERS_4   : ${STREAM_HELPERS_4}"
echo "Compiler           : $(clang --version | head -1)"
echo "============================================================"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES from Slurm=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)

if (( GPU_COUNT < 4 )); then
    echo "ERROR: four visible GPUs are required; found ${GPU_COUNT}" >&2
    exit 1
fi

if ! nvidia-smi --query-gpu=name --format=csv,noheader |
        grep -q 'H100'; then
    echo "ERROR: no NVIDIA H100 detected in the allocation" >&2
    exit 1
fi

echo "GPU inventory:"
nvidia-smi \
    --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv,noheader

echo "GPU topology:"
nvidia-smi topo -m || true


trap 'echo "ERROR at line $LINENO" | tee -a "$SKIPPED_LOG"' ERR

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
mask_for() {
    local n="$1"
    local values=()
    local i

    for ((i = 0; i < n; i++)); do
        values+=("$i")
    done

    (IFS=,; echo "${values[*]}")
}

gpu_mem() {
    [[ "$CHECK_GPU_MEM" == "1" ]] || return 0

    echo "  [GPU memory after run]"
    nvidia-smi \
        --query-gpu=index,memory.used \
        --format=csv,noheader 2>/dev/null || true
}

print_runtime_config() {
    local item

    echo "    Runtime configuration:"

    for item in "$@"; do
        case "$item" in
            CUDA_VISIBLE_DEVICES=*|\
            OMP_NUM_THREADS=*|\
            OMP_THREAD_LIMIT=*|\
            OMP_NUM_HIDDEN_HELPER_THREADS=*|\
            LIBOMP_USE_HIDDEN_HELPER_TASK=*|\
            LIBOMP_NUM_HIDDEN_HELPER_THREADS=*)
                echo "      $item"
                ;;
        esac
    done
}

run_bench() {
    local binary="$1"
    local gpus="$2"
    local family="${gpus}gpu"
    local mask
    local mask_csv

    mask=$(mask_for "$gpus")
    mask_csv="${mask//,/-}"

    if [[ ! -x "./$binary" ]]; then
        echo "  SKIP (missing): $binary  N=$N" | tee -a "$SKIPPED_LOG"
        echo "${REP},${JOB_ID},${binary},${family},${gpus},${mask_csv},${N},${DT},${STEPS},N/A,N/A,missing_binary" \
            >> "$TIMING_CSV"
        return 0
    fi

    echo
    echo ">>> ${binary}  gpus=${gpus}  mask=${mask}  N=${N}"

    # No srun step is used. Every executable is already a separate process.
    local run_cmd=(
	env
	-u LD_PRELOAD
	-u CUDA_VISIBLE_DEVICES
	-u CUDA_LAUNCH_BLOCKING
	-u OMP_NUM_THREADS
	-u OMP_THREAD_LIMIT
	-u LIBOMP_USE_HIDDEN_HELPER_TASK
	-u OMP_NUM_HIDDEN_HELPER_THREADS      # add	
	-u LIBOMP_NUM_HIDDEN_HELPER_THREADS
	"CUDA_DEVICE_ORDER=PCI_BUS_ID"
	"CUDA_VISIBLE_DEVICES=${mask}"
	"OMP_PROC_BIND=spread"
	"OMP_PLACES=cores"
	"OMP_WAIT_POLICY=active"
    )

    case "$binary" in
	2-omp-stream|2-omp)
            run_cmd+=(
		"OMP_NUM_THREADS=1"
		"LIBOMP_USE_HIDDEN_HELPER_TASK=1"
		"LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"
            )
            ;;

	4-omp-stream|4-omp)
            run_cmd+=(
		"OMP_NUM_THREADS=1"
		"LIBOMP_USE_HIDDEN_HELPER_TASK=1"
		"LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"		
            )
            ;;

	*-stream-omp-p2p|*-stream-omp)
            run_cmd+=(
		"OMP_NUM_THREADS=${gpus}"
		"LIBOMP_USE_HIDDEN_HELPER_TASK=1"
		"LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"
            )
            ;;
		
	1-omp)
            run_cmd+=(
		"OMP_NUM_THREADS=1"
		"LIBOMP_USE_HIDDEN_HELPER_TASK=1"
		"LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"		
            )
            ;;

	*)
            echo "ERROR: no runtime configuration for binary '$binary'" >&2
            echo "${REP},${JOB_ID},${binary},${family},${gpus},${mask_csv},${N},${DT},${STEPS},N/A,N/A,configuration_error" \
		 >> "$TIMING_CSV"
            return 0
            ;;
    esac

    print_runtime_config "${run_cmd[@]}"

    run_cmd+=("./$binary" "$N" "$DT")

    local stdout_file
    local status
    local wall
    local reported
    local t_start
    local t_end
    local exit_code

    stdout_file=$(mktemp "${TMPDIR:-/tmp}/heat_stdout.XXXXXX")
    t_start=$(date +%s%3N)

    if timeout "$TIMEOUT_SEC" "${run_cmd[@]}" > "$stdout_file" 2>&1; then
        status="ok"
    else
        exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            status="timeout"
        else
            status="runtime_error"
        fi
    fi

    t_end=$(date +%s%3N)
    wall=$(awk -v start="$t_start" -v end="$t_end" \
        'BEGIN { printf "%.4f", (end - start) / 1000.0 }')

    cat "$stdout_file"

    # Solver line:
    #   Time = 1.176864 seconds
    reported=$(awk \
        '/^Time = / { value=$3 } END { if (value != "") print value }' \
        "$stdout_file")
    [[ -n "$reported" ]] || reported="N/A"

    rm -f "$stdout_file"

    echo "    -> wall=${wall}s  reported=${reported}s  status=${status}"
    gpu_mem

    echo "${REP},${JOB_ID},${binary},${family},${gpus},${mask_csv},${N},${DT},${STEPS},${wall},${reported},${status}" \
        >> "$TIMING_CSV"

    sleep "$SLEEP_SEC"
}

# -----------------------------------------------------------------------------
# Run benchmarks
# -----------------------------------------------------------------------------
echo
echo "Starting benchmarks"
echo "==========================="
echo "N = $N   dt = $DT"
echo "==========================="

if [[ "$BASELINE" == "1" ]]; then
    run_bench 1-omp 1
fi

for fam in $FAMILIES; do
    case "$fam" in
        2|4) ;;
        *)
            echo "SKIP: unsupported family '$fam'; expected 2 or 4" |
                tee -a "$SKIPPED_LOG"
            continue
            ;;
    esac

    run_bench "${fam}-omp"                "$fam"
    run_bench "${fam}-omp-stream"         "$fam"
    run_bench "${fam}-omp-stream-omp"     "$fam"
    run_bench "${fam}-omp-stream-omp-p2p" "$fam"
done

echo
echo "All done."
echo "  Timing CSV : $TIMING_CSV"
echo "  Run log    : $RUN_LOG"

if [[ -s "$SKIPPED_LOG" ]]; then
    echo "  Skipped    : $SKIPPED_LOG ($(wc -l < "$SKIPPED_LOG") entries)"
fi

exit 0
