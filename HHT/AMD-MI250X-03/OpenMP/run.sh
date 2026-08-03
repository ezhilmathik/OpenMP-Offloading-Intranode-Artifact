#!/bin/bash
#SBATCH --job-name=heat-run
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465003145
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=run-%j.out
#
# One job = one grid size N, one repetition, all 2- and 4-GPU variants.
# submit.sh launches N_RUNS of these per N so you get independent samples.
#
# Inputs (via sbatch --export):
#   N        grid size                      (required)
#   DT       total simulated time           (required)
#   STEPS    step count DT was derived from (recorded in the CSV only)
#   REP      repetition index               (default 1)
#   FAMILIES which families to run: "2 4"   (default "2 4")
#   BASELINE 1 -> also run the 1-GCD baseline (default 0)
#   HELPERS  hidden-helper thread count     (default 8 = libomp's own default)
#
# Optional P2P paths:
#   OMPTARGET_P2P_LIB  patched OpenMP runtime directory
#   P2P_SHIM           engine-selection/peer-access shim
#
# ===========================================================================
#  PREREQUISITE FOR THIS VERSION
# ===========================================================================
#  The *-stream-omp and *-stream-omp-p2p sources call
#      getenv("OMP_NUM_HIDDEN_HELPER_THREADS")
#  and exit(EXIT_FAILURE) unless it is "0". This script no longer sets it, so
#  those guards MUST be relaxed first:
#      python3 relax_hidden_helper_guard.py *.c
#  Otherwise those four variants abort immediately and land in the CSV as
#  runtime_error.
#
#  The guard documents a real fault: concurrent omp_target_memcpy_async from
#  several host threads corrupting task-team state and faulting inside
#  __kmpc_barrier on libomp 18.1.8. That is close to the runtime in use here,
#  and closer still for the p2p variant, which LD_PRELOADs a patched
#  libomptarget. Prove one job survives before committing to a sweep.
# ===========================================================================
#
#  WHAT CHANGED, AND WHY (measured on Intel Max 1550, to be re-checked here)
#  1. V1/V2 hidden helpers were LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus},
#     i.e. 2 or 4 -- BELOW libomp's default of 8. A helper blocked on a
#     transfer cannot submit the next task, so one-helper-per-device starves
#     the runtime. Raising to 8 cut V2 4-GPU N=1280 from 23.9 s to 18.0 s.
#  2. Every variant also had a global OMP_NUM_HIDDEN_HELPER_THREADS=0 in the
#     environment while asking for N helper threads. Contradictory; removed
#     for every variant that wants helpers.
#  3. V3/V4 had OMP_THREAD_LIMIT=${gpus}. thread-limit-var is inherited by the
#     initial task on the DEVICE, so it capped each team inside the target
#     regions at 2-4 threads and cost roughly 3x on Intel. Removed.
#  4. V3/V4 now also get hidden helpers: disabling them cost V3 20-30% at
#     4 GPUs on Intel.

# Deliberately NOT -e: one failing variant must not kill the repetition.
set -Euo pipefail

module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4
module load lumi-CrayPath

# Slurm executes a copied script from its spool directory. Use the original
# submission directory, where the binaries and shim are located.
WORK_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORK_DIR"

# ---------------------------------------------------------------------------
# Environment common to every binary.
# Transfer/runtime settings are applied per binary in run_bench().
# ---------------------------------------------------------------------------
unset LD_PRELOAD
unset HSA_ENABLE_SDMA
unset HSA_ENABLE_PEER_SDMA
unset OMP_NUM_HIDDEN_HELPER_THREADS
unset OMP_THREAD_LIMIT

export OMP_TARGET_OFFLOAD=mandatory
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export WRITE_TXT=0

OMPTARGET_P2P_LIB="${OMPTARGET_P2P_LIB:-/scratch/project_465003145/omptarget-p2p/lib}"
P2P_SHIM="${P2P_SHIM:-${WORK_DIR}/libp2p_copy_shim.so}"
P2P_PRELOAD="${P2P_SHIM}:${OMPTARGET_P2P_LIB}/libomp.so:${OMPTARGET_P2P_LIB}/libomptarget.so.19.0git"

# One GCD per physical MI250X module:
#   2 GPUs -> 0,2
#   4 GPUs -> 0,2,4,6
N="${N:?ERROR: N not set (use: sbatch --export=ALL,N=...,DT=... run.sh)}"
DT="${DT:?ERROR: DT not set}"
STEPS="${STEPS:-unknown}"
REP="${REP:-1}"
FAMILIES="${FAMILIES:-2 4}"
BASELINE="${BASELINE:-0}"
HELPERS="${HELPERS:-8}"

[[ "$HELPERS" =~ ^[0-9]+$ ]] || {
    echo "ERROR: HELPERS must be a non-negative integer, got '$HELPERS'." >&2; exit 1; }

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_DIR="results_h${HELPERS}_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

TIMING_CSV="${RESULTS_DIR}/timings.csv"
RUN_LOG="${RESULTS_DIR}/run.log"
SKIPPED_LOG="${RESULTS_DIR}/skipped.log"

echo "rep,job_id,variant,family,gpus,rocr_mask,N,dt,steps,wall_sec,reported_sec,status" > "$TIMING_CSV"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "Job ID       : ${JOB_ID}"
echo "Repetition   : ${REP}"
echo "Timestamp    : ${TIMESTAMP}"
echo "Results dir  : ${RESULTS_DIR}"
echo "N=${N}  DT=${DT}  STEPS=${STEPS}  FAMILIES='${FAMILIES}'"
echo "Hidden helper threads : ${HELPERS}"
echo "P2P runtime  : ${OMPTARGET_P2P_LIB}"
echo "P2P shim     : ${P2P_SHIM}"

trap 'echo "ERROR at line $LINENO" | tee -a "$SKIPPED_LOG"' ERR

SLEEP_SEC="${SLEEP_SEC:-15}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"

mask_for() {
    local n="$1" v=() i
    for ((i = 0; i < n; i++)); do
        v+=("$((2 * i))")
    done
    (IFS=,; echo "${v[*]}")
}

check_p2p_files() {
    local missing=0 file

    for file in \
        "$P2P_SHIM" \
        "$OMPTARGET_P2P_LIB/libomp.so" \
        "$OMPTARGET_P2P_LIB/libomptarget.so.19.0git"
    do
        if [[ ! -r "$file" ]]; then
            echo "  P2P requirement missing: $file" | tee -a "$SKIPPED_LOG"
            missing=1
        fi
    done

    [[ $missing -eq 0 ]]
}

run_bench() {
    local binary="$1" gpus="$2"
    local family="${gpus}gpu"
    local mask
    mask=$(mask_for "$gpus")

    if [[ ! -x "./$binary" ]]; then
        echo "  SKIP (missing binary): $binary  N=$N" | tee -a "$SKIPPED_LOG"
        echo "${REP},${JOB_ID},${binary},${family},${gpus},${mask},${N},${DT},${STEPS},N/A,N/A,missing_binary" >> "$TIMING_CSV"
        return 0
    fi

    if [[ "$binary" == *-stream-omp-p2p ]] && ! check_p2p_files; then
        echo "  SKIP (missing P2P runtime): $binary  N=$N" | tee -a "$SKIPPED_LOG"
        echo "${REP},${JOB_ID},${binary},${family},${gpus},${mask},${N},${DT},${STEPS},N/A,N/A,missing_p2p_runtime" >> "$TIMING_CSV"
        return 0
    fi

    echo ""
    echo ">>> Running: ${binary}  gpus=${gpus}  mask=${mask}  N=${N}"

    # Start from a clean per-process runtime environment. Assignments appended
    # below intentionally override the variables removed with env -u.
    #
    # OMP_THREAD_LIMIT and OMP_NUM_HIDDEN_HELPER_THREADS are stripped and never
    # re-added: the first throttles the device teams, the second contradicts
    # the LIBOMP_NUM_HIDDEN_HELPER_THREADS request.
    local run_cmd=(
        env
        -u LD_PRELOAD
        -u HSA_ENABLE_SDMA
        -u HSA_ENABLE_PEER_SDMA
        -u OMP_NUM_THREADS
        -u OMP_THREAD_LIMIT
        -u OMP_NUM_HIDDEN_HELPER_THREADS
        -u LIBOMP_USE_HIDDEN_HELPER_TASK
        -u LIBOMP_NUM_HIDDEN_HELPER_THREADS
        "ROCR_VISIBLE_DEVICES=${mask}"
    )

    case "$binary" in
        *-stream-omp-p2p)
            # Version 4: one explicit host thread per GCD, plus runtime helpers.
            run_cmd+=(
                "OMP_NUM_THREADS=${gpus}"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=1"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=${HELPERS}"
                "HSA_ENABLE_SDMA=1"
                "HSA_ENABLE_PEER_SDMA=1"
                "LD_PRELOAD=${P2P_PRELOAD}"
            )
            ;;

        *-stream-omp)
            # Version 3: one explicit host thread per GCD, plus runtime helpers.
            run_cmd+=(
                "OMP_NUM_THREADS=${gpus}"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=1"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=${HELPERS}"
                "HSA_ENABLE_SDMA=1"
            )
            ;;

        *-omp-stream)
            # Version 2: one application host thread; runtime helpers progress
            # the asynchronous transfers and kernels.
            run_cmd+=(
                "OMP_NUM_THREADS=1"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=1"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=${HELPERS}"
                "HSA_ENABLE_SDMA=1"
            )
            ;;

        *)
            # Version 1 and the 1-GCD baseline.
            run_cmd+=(
                "OMP_NUM_THREADS=1"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=1"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=${HELPERS}"
                "HSA_ENABLE_SDMA=1"
            )
            ;;
    esac

    echo "    Runtime configuration:"
    for item in "${run_cmd[@]}"; do
        case "$item" in
            ROCR_VISIBLE_DEVICES=*|\
                OMP_NUM_THREADS=*|\
                OMP_THREAD_LIMIT=*|\
                OMP_NUM_HIDDEN_HELPER_THREADS=*|\
                LIBOMP_USE_HIDDEN_HELPER_TASK=*|\
                LIBOMP_NUM_HIDDEN_HELPER_THREADS=*|\
                HSA_ENABLE_SDMA=*|\
                HSA_ENABLE_PEER_SDMA=*|\
                LIBOMPTARGET_AMDGPU_MAX_ASYNC_COPY_BYTES=*|\
                LD_PRELOAD=*)
                echo "      $item"
                ;;
        esac
    done

    run_cmd+=("./$binary" "$N" "$DT")

    printf '    command:'
    printf ' %q' "${run_cmd[@]}"
    printf '\n'

    local stdout_file status wall reported t_start t_end exit_code
    stdout_file=$(mktemp /tmp/heat_stdout.XXXXXX)
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
    wall=$(awk -v s="$t_start" -v e="$t_end" 'BEGIN { printf "%.4f", (e-s)/1000.0 }')
    cat "$stdout_file"

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

echo ""
echo "Starting benchmarks"
echo "==========================="
echo "N = $N   dt = $DT   helpers = $HELPERS"
echo "==========================="

if [[ "$BASELINE" == "1" ]]; then
    run_bench 1-omp 1
fi

for fam in $FAMILIES; do
    run_bench "${fam}-omp"                "$fam"
    run_bench "${fam}-omp-stream"         "$fam"
    run_bench "${fam}-omp-stream-omp"     "$fam"
    run_bench "${fam}-omp-stream-omp-p2p" "$fam"
done

echo ""
echo "All done."
echo "  Timing CSV : $TIMING_CSV"
echo "  Run log    : $RUN_LOG"
[[ -s "$SKIPPED_LOG" ]] && \
    echo "  Skipped    : $SKIPPED_LOG  ($(wc -l < "$SKIPPED_LOG") entries)"
exit 0
