#!/bin/bash -l
#SBATCH --job-name=heat-run
#SBATCH --account=ehpc681
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=run-%j.out
#
# One job = one grid size N, one repetition, all 2- and 4-GPU CUDA variants.
# submit.sh launches N_RUNS of these per N so you get 5 independent samples.
#
# Inputs (via sbatch --export):
#   N        grid size                      (required)
#   DT       total simulated time           (required)
#   STEPS    step count DT was derived from (recorded in the CSV only)
#   REP      repetition index 1..N_RUNS     (default 1)
#   FAMILIES which families to run: "2 4"   (default "2 4")
#   BASELINE 1 -> also run the 1-GPU baseline (default 1)
#   USE_SRUN 1 -> launch each binary in its own srun step (see below)

set -uo pipefail   # deliberately NOT -e: one failing variant must not kill the rep

module purge
module load EB/apps
module load NVHPC/25.3-CUDA-12.8.0

# ---------------------------------------------------------------------------
# Environment, identical for every binary in the comparison set.
#
# There is no OMP_TARGET_OFFLOAD here and no OMP_NUM_HIDDEN_HELPER_THREADS:
# these are CUDA binaries, and OpenMP is used only for host threading in the
# *-stream-omp* variants and for omp_get_wtime() everywhere.
# ---------------------------------------------------------------------------
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active
export WRITE_TXT=0                          # timing runs never write solutions

# NOTE: CUDA_VISIBLE_DEVICES is deliberately NOT set globally. The 2-GPU codes
# must see exactly 2 devices and the 4-GPU codes exactly 4, so the mask is set
# per binary in run_bench():
#   1 GPU -> 0     2 GPUs -> 0,1     4 GPUs -> 0,1,2,3
#
# One H100 = one device. There is no tile/GCD subdivision here, so this matches
# the Intel COMPOSITE convention (whole card) and NOT the AMD one (one GCD =
# half an MI250X module). State that in any cross-platform plot.

N="${N:?ERROR: N not set (use: sbatch --export=ALL,N=...,DT=... run.sh)}"
DT="${DT:?ERROR: DT not set}"
STEPS="${STEPS:-unknown}"
REP="${REP:-1}"
FAMILIES="${FAMILIES:-2 4}"
BASELINE="${BASELINE:-1}"

# ---------------------------------------------------------------------------
# Isolation control. Binaries that work standalone but crash when run after the
# others inside one job are the signature of CUDA runtime state leaking between
# programs. USE_SRUN=1 puts each binary in its own srun step, giving it a fresh
# process. Off by default; turn it on if you see failures that only happen
# partway through a job.
# ---------------------------------------------------------------------------
USE_SRUN="${USE_SRUN:-0}"
SRUN_OPTS=(--ntasks=1 --cpu-bind=cores)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_DIR="results_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

TIMING_CSV="${RESULTS_DIR}/timings.csv"
RUN_LOG="${RESULTS_DIR}/run.log"
SKIPPED_LOG="${RESULTS_DIR}/skipped.log"

echo "rep,job_id,variant,family,gpus,cuda_mask,N,dt,steps,wall_sec,reported_sec,status" > "$TIMING_CSV"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "============================================================"
echo "Host       : $(hostname)"
echo "Job ID     : ${JOB_ID}   Repetition: ${REP}"
echo "Timestamp  : ${TIMESTAMP}"
echo "Results dir: ${RESULTS_DIR}"
echo "N=${N}  DT=${DT}  STEPS=${STEPS}  FAMILIES='${FAMILIES}'  BASELINE=${BASELINE}"
echo "USE_SRUN=${USE_SRUN}"
echo "nvcc       : $(nvcc --version | tail -2 | head -1)"
echo "============================================================"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES (from SLURM)=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true

trap 'echo "ERROR at line $LINENO" | tee -a "$SKIPPED_LOG"' ERR

SLEEP_SEC="${SLEEP_SEC:-15}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"

# ---------------------------------------------------------------------------
# mask_for <ngpus>  ->  0 | 0,1 | 0,1,2,3
# ---------------------------------------------------------------------------
mask_for() {
    local n="$1" v=() i
    for ((i = 0; i < n; i++)); do v+=("$i"); done
    (IFS=,; echo "${v[*]}")
}

# Residual device memory after a run means the previous program did not release
# its allocations -- the usual precursor to a crash in the NEXT binary.
gpu_mem() {
    echo "  [GPU mem after run]"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null || true
}

run_bench() {
    local binary="$1" gpus="$2"
    local family="${gpus}gpu"
    local mask; mask=$(mask_for "$gpus")
    # Dash-joined for the CSV: a comma here would add fields and shift every
    # column after it.
    local mask_csv="${mask//,/-}"

    if [[ ! -x "./$binary" ]]; then
        echo "  SKIP (missing): $binary  N=$N" | tee -a "$SKIPPED_LOG"
        echo "${REP},${JOB_ID},${binary},${family},${gpus},${mask_csv},${N},${DT},${STEPS},N/A,N/A,missing_binary" >> "$TIMING_CSV"
        return 0
    fi

    echo ""
    echo ">>> ${binary}  gpus=${gpus}  mask=${mask}  N=${N}"

    local run_cmd=(env "CUDA_VISIBLE_DEVICES=${mask}" "OMP_NUM_THREADS=${gpus}")
    if [[ "$USE_SRUN" == "1" ]] && command -v srun >/dev/null 2>&1; then
        run_cmd+=(srun "${SRUN_OPTS[@]}")
    fi
    run_cmd+=("./$binary" "$N" "$DT")

    local stdout_file status wall reported t_start t_end
    stdout_file=$(mktemp /tmp/heat_stdout.XXXXXX)
    t_start=$(date +%s%3N)

    if timeout "$TIMEOUT_SEC" "${run_cmd[@]}" > "$stdout_file" 2>&1; then
        status="ok"
    else
        local exit_code=$?
        [[ $exit_code -eq 124 ]] && status="timeout" || status="runtime_error"
    fi

    t_end=$(date +%s%3N)
    wall=$(echo "scale=4; ($t_end - $t_start) / 1000" | bc)
    cat "$stdout_file"

    # The solver's own timer, anchored on the exact line it prints:
    #     Time = 1.176864 seconds
    # A loose /time/ match is WRONG: the parameter banner prints
    # "Final time (T):  0.000135" earlier, so a fuzzy regex captures DT.
    reported=$(awk '/^Time = / { print $3; exit }' "$stdout_file")
    [[ -z "$reported" ]] && reported="N/A"

    rm -f "$stdout_file"
    echo "    -> wall=${wall}s  reported=${reported}s  status=${status}"
    gpu_mem
    echo "${REP},${JOB_ID},${binary},${family},${gpus},${mask_csv},${N},${DT},${STEPS},${wall},${reported},${status}" >> "$TIMING_CSV"
    sleep "$SLEEP_SEC"
}

echo ""
echo "Starting benchmarks"
echo "==========================="
echo "N = $N   dt = $DT"
echo "==========================="

if [[ "$BASELINE" == "1" ]]; then
    run_bench 1-cuda 1
fi

for fam in $FAMILIES; do
    run_bench "${fam}-cuda"                "$fam"
    run_bench "${fam}-cuda-stream"         "$fam"
    run_bench "${fam}-cuda-stream-omp"     "$fam"
    run_bench "${fam}-cuda-stream-omp-p2p" "$fam"
done

echo ""
echo "All done."
echo "  Timing CSV : $TIMING_CSV"
echo "  Run log    : $RUN_LOG"
[[ -s "$SKIPPED_LOG" ]] && \
    echo "  Skipped    : $SKIPPED_LOG  ($(wc -l < "$SKIPPED_LOG") entries)"
exit 0
