#!/bin/bash -l
#SBATCH --job-name=heat-run
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465003145
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=run-%j.out
#
# One job = one grid size N, one repetition, selected HIP families.
# submit launches independent jobs for each N and repetition after build.sh succeeds.
#
# Inputs supplied through sbatch --export:
#   N          grid size                                      required
#   DT         total simulated time                           required
#   STEPS      step count used to derive DT                   default unknown
#   REP        repetition number                              default 1
#   FAMILIES   comma/space list: 2,4 | 2 | 4                  default 2,4
#   BASELINE   1 -> also run 1-hip                            default 1
#
# HIP P2P variants use the same runtime as all other HIP variants. There is
# no patched OpenMP-target runtime, P2P shim, or LD_PRELOAD in this script.

# Deliberately no `set -e`: one failed variant must not kill the whole job.
set -Euo pipefail

module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4
module load lumi-CrayPath

WORK_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORK_DIR"

# Clean inherited runtime settings. Each process receives an explicit, minimal
# environment in run_bench().
unset LD_PRELOAD
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset OMP_TARGET_OFFLOAD
unset HSA_ENABLE_SDMA
unset HSA_ENABLE_PEER_SDMA
unset OMP_NUM_THREADS
unset OMP_THREAD_LIMIT
unset OMP_NUM_HIDDEN_HELPER_THREADS

export WRITE_TXT=0

N="${N:?ERROR: N is not set}"
DT="${DT:?ERROR: DT is not set}"
STEPS="${STEPS:-unknown}"
REP="${REP:-1}"
FAMILIES="${FAMILIES:-2,4}"
FAMILIES="${FAMILIES//,/ }"
FAMILIES="${FAMILIES//+/ }"
BASELINE="${BASELINE:-1}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_DIR="results_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

TIMING_CSV="${RESULTS_DIR}/timings.csv"
RUN_LOG="${RESULTS_DIR}/run.log"
SKIPPED_LOG="${RESULTS_DIR}/skipped.log"

echo "rep,job_id,variant,family,gpus,rocr_mask,N,dt,steps,wall_sec,reported_sec,status" > "$TIMING_CSV"
exec > >(tee -a "$RUN_LOG") 2>&1

SLEEP_SEC="${SLEEP_SEC:-15}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"

echo "Job ID       : $JOB_ID"
echo "Repetition   : $REP"
echo "Timestamp    : $TIMESTAMP"
echo "Results dir  : $RESULTS_DIR"
echo "N=$N  DT=$DT  STEPS=$STEPS  FAMILIES='$FAMILIES'  BASELINE=$BASELINE"
echo "WRITE_TXT    : $WRITE_TXT"

rocm-smi --showproductname || rocm-smi || true

trap 'echo "ERROR at line $LINENO" | tee -a "$SKIPPED_LOG"' ERR

# One GCD from each physical MI250X card:
#   1 GPU  -> 0
#   2 GPUs -> 0,2
#   4 GPUs -> 0,2,4,6
mask_for() {
    local n="$1" values=() i
    for ((i = 0; i < n; i++)); do
        values+=("$((2 * i))")
    done
    (IFS=,; echo "${values[*]}")
}

valid_family() {
    [[ "$1" == "2" || "$1" == "4" ]]
}

for fam in $FAMILIES; do
    if ! valid_family "$fam"; then
        echo "ERROR: FAMILIES accepts only 2 and/or 4; got '$fam'."
        exit 1
    fi
done

run_bench() {
    local binary="$1" gpus="$2"
    local family="${gpus}gpu" mask
    mask=$(mask_for "$gpus")

    if [[ ! -x "./$binary" ]]; then
        echo "  SKIP (missing binary): $binary  N=$N" | tee -a "$SKIPPED_LOG"
        echo "${REP},${JOB_ID},${binary},${family},${gpus},${mask},${N},${DT},${STEPS},N/A,N/A,missing_binary" >> "$TIMING_CSV"
        return 0
    fi

    echo
echo ">>> Running: ${binary}  gpus=${gpus}  mask=${mask}  N=${N}"

    local run_cmd=(
        env
        -u LD_PRELOAD
        -u HIP_VISIBLE_DEVICES
        -u CUDA_VISIBLE_DEVICES
        -u OMP_TARGET_OFFLOAD
        -u HSA_ENABLE_SDMA
        -u HSA_ENABLE_PEER_SDMA
        -u OMP_NUM_THREADS
        -u OMP_THREAD_LIMIT
        -u OMP_NUM_HIDDEN_HELPER_THREADS
        "ROCR_VISIBLE_DEVICES=${mask}"
        "WRITE_TXT=0"
        "HSA_ENABLE_SDMA=1"
    )

    case "$binary" in
	*-stream-omp-p2p)
            run_cmd+=(
		"OMP_NUM_THREADS=${gpus}"
		"OMP_THREAD_LIMIT=${gpus}"
		"OMP_NUM_HIDDEN_HELPER_THREADS=0"
		"LIBOMP_USE_HIDDEN_HELPER_TASK=0"
		"LIBOMP_NUM_HIDDEN_HELPER_THREADS=0"
		"HSA_ENABLE_SDMA=1"
		"HSA_ENABLE_PEER_SDMA=1"
            )
            ;;

	*-stream-omp)
            run_cmd+=(
		"OMP_NUM_THREADS=${gpus}"
		"OMP_THREAD_LIMIT=${gpus}"
		"OMP_NUM_HIDDEN_HELPER_THREADS=0"
		"LIBOMP_USE_HIDDEN_HELPER_TASK=0"
		"LIBOMP_NUM_HIDDEN_HELPER_THREADS=0"
		"HSA_ENABLE_SDMA=1"
            )
            ;;

	*-omp-stream)
            run_cmd+=(
		"OMP_NUM_THREADS=1"
		"OMP_NUM_HIDDEN_HELPER_THREADS=0"
		"LIBOMP_USE_HIDDEN_HELPER_TASK=1"
		"LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"
		"HSA_ENABLE_SDMA=1"
            )
            ;;

	*)
            run_cmd+=(
		"OMP_NUM_THREADS=1"
		"OMP_NUM_HIDDEN_HELPER_THREADS=0"
		"LIBOMP_USE_HIDDEN_HELPER_TASK=1"
		"LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"
		"HSA_ENABLE_SDMA=1"		
            )
            ;;
    esac
	
    echo "    Runtime configuration:"
    for item in "${run_cmd[@]}"; do
        case "$item" in
            ROCR_VISIBLE_DEVICES=*|WRITE_TXT=*|OMP_NUM_THREADS=*|OMP_THREAD_LIMIT=*|\
            OMP_NUM_HIDDEN_HELPER_THREADS=*|HSA_ENABLE_SDMA=*|HSA_ENABLE_PEER_SDMA=*)
                echo "      $item"
                ;;
        esac
    done

    run_cmd+=("./$binary" "$N" "$DT")

    printf '    command:'
    printf ' %q' "${run_cmd[@]}"
    printf '\n'

    local stdout_file status wall reported t_start t_end exit_code
    stdout_file=$(mktemp /tmp/heat_hip_stdout.XXXXXX)
    t_start=$(date +%s%3N)

    if timeout "$TIMEOUT_SEC" "${run_cmd[@]}" > "$stdout_file" 2>&1; then
        status="ok"
        exit_code=0
    else
        exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            status="timeout"
        else
            status="runtime_error"
        fi
    fi

    t_end=$(date +%s%3N)
    wall=$(awk -v start="$t_start" -v end="$t_end" 'BEGIN { printf "%.4f", (end-start)/1000.0 }')
    cat "$stdout_file"

    # Expected solver output: "Time = <number> seconds"
    reported=$(
        sed -nE \
            's/^[[:space:]]*Time[[:space:]]*=[[:space:]]*([0-9.eE+-]+)[[:space:]]*seconds.*/\1/p' \
            "$stdout_file" |
        tail -n 1
    )
    [[ -n "$reported" ]] || reported="N/A"

    rm -f "$stdout_file"
    echo "    -> wall=${wall}s  reported=${reported}s  status=${status}  exit=${exit_code}"
    echo "${REP},${JOB_ID},${binary},${family},${gpus},${mask},${N},${DT},${STEPS},${wall},${reported},${status}" >> "$TIMING_CSV"

    sleep "$SLEEP_SEC"
}

echo
echo "Starting HIP benchmarks"
echo "==========================="
echo "N = $N   dt = $DT"
echo "==========================="

if [[ "$BASELINE" == "1" ]]; then
    run_bench 1-hip 1
fi

for fam in $FAMILIES; do
    run_bench "${fam}-hip"                "$fam"
    run_bench "${fam}-hip-stream"         "$fam"
    run_bench "${fam}-hip-stream-omp"     "$fam"
    run_bench "${fam}-hip-stream-omp-p2p" "$fam"
done

echo
echo "All done."
echo "  Timing CSV : $TIMING_CSV"
echo "  Run log    : $RUN_LOG"
if [[ -s "$SKIPPED_LOG" ]]; then
    echo "  Skipped    : $SKIPPED_LOG ($(wc -l < "$SKIPPED_LOG") entries)"
fi
exit 0
