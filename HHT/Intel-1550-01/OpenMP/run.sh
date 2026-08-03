#!/bin/bash -l
#SBATCH --job-name=heat-run
#SBATCH --account=pn67qe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --output=run-%j.out
#
# One job = one grid size N, one repetition, all 2- and 4-GPU variants.
# submit.sh launches N_RUNS of these per N so you get 5 independent samples.
#
# Inputs (via sbatch --export):
#   N        grid size                      (required)
#   DT       total simulated time           (required)
#   STEPS    step count DT was derived from (recorded in the CSV only)
#   REP      repetition index 1..N_RUNS     (default 1)
#   HIER     COMPOSITE | FLAT               (default COMPOSITE)
#   FAMILIES which families to run: "2 4"   (default "2 4")
#   BASELINE 1 -> also run the 1-card baseline (default 1)
#   HELPERS  hidden-helper threads for V1/V2 (default 8 = libomp's own default)
#
# HIDDEN-HELPER COUNT
# -------------------
# Versions 1 and 2 previously used LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus},
# i.e. 2 or 4. That is BELOW libomp's own default of 8, so it throttled the
# runtime rather than tuning it: a helper blocked on a transfer cannot submit
# the next task, and with one helper per device there is no slack. Measured on
# Max 1550, dropping the setting entirely (8 helpers) cut V2 4-GPU N=1280 from
# 23.9 s to 18.0 s and V1 from 29.5 s to 19.7 s, while V3/V4 -- which disable
# helpers -- barely moved.
#
# HELPERS is now a knob so the count can be swept independently:
#   sbatch --export=ALL,N=1280,DT=...,HELPERS=8   run.sh   # = runtime default
#   sbatch --export=ALL,N=1280,DT=...,HELPERS=16  run.sh
#   sbatch --export=ALL,N=1280,DT=...,HELPERS=32  run.sh
# HELPERS=8 should reproduce the stock-default timings. If it does not, the
# cause is elsewhere and the sweep will say so.

set -Euo pipefail   # deliberately NOT -e: one failing variant must not kill the rep

module load slurm_setup
module load intel-toolkit

WORK_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORK_DIR"

# ---------------------------------------------------------------------------
# Environment, identical for every binary in the comparison set
# ---------------------------------------------------------------------------
export OMP_TARGET_OFFLOAD=MANDATORY

# Required by the source-code checks in the explicit host-thread variants
# (*-stream-omp, *-stream-omp-p2p). V1/V2 do not read it and have it REMOVED
# from their environment below -- leaving a global 0 in place while asking for
# N helper threads is a conflicting request.
export OMP_NUM_HIDDEN_HELPER_THREADS=0
unset LIBOMP_USE_HIDDEN_HELPER_TASK
unset LIBOMP_NUM_HIDDEN_HELPER_THREADS
unset OMP_THREAD_LIMIT

# One OpenMP device must be one whole card: COMPOSITE makes the Level Zero
# root device the card, and LIBOMPTARGET_DEVICES must not downgrade it to a
# stack. Both are required; either one alone is not enough.
unset LIBOMPTARGET_DEVICES

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1     # stable, PCI-ordered device numbering
unset ONEAPI_DEVICE_SELECTOR               # ZE_AFFINITY_MASK controls OpenMP devices
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export WRITE_TXT=0                          # timing runs never write solutions

N="${N:?ERROR: N not set (use: sbatch --export=ALL,N=...,DT=... run.sh)}"
DT="${DT:?ERROR: DT not set}"
STEPS="${STEPS:-unknown}"
REP="${REP:-1}"
HIER="COMPOSITE"   # force one OpenMP device = one whole Max 1550 card
FAMILIES="${FAMILIES:-2 4}"
BASELINE="${BASELINE:-1}"
HELPERS="${HELPERS:-8}"

[[ "$HELPERS" =~ ^[0-9]+$ ]] || {
    echo "ERROR: HELPERS must be a non-negative integer, got '$HELPERS'." >&2; exit 1; }

[[ "$HIER" == "COMPOSITE" ]] || {
    echo "ERROR: this run.sh is locked to whole-card COMPOSITE mode." >&2; exit 1; }

export ZE_FLAT_DEVICE_HIERARCHY="$HIER"

# ---------------------------------------------------------------------------
# Device mask in whole-card COMPOSITE mode:
#
#   index = CARD. One OpenMP device = one whole Max 1550 card,
#   with both stacks/tiles available to the Intel runtime.
#
#   1-card code -> "0"
#   2-card code -> "0,1"
#   4-card code -> "0,1,2,3"
#
# The mask is set PER BINARY, not globally: the 2-GPU codes must see exactly
# two complete cards and the 4-GPU codes exactly four complete cards.
# ---------------------------------------------------------------------------
mask_for() {
    local n="$1" v=() i
    for ((i = 0; i < n; i++)); do v+=("$i"); done              # 0,1,2,3
    (IFS=,; echo "${v[*]}")
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_DIR="results_${HIER}_h${HELPERS}_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

TIMING_CSV="${RESULTS_DIR}/timings.csv"
RUN_LOG="${RESULTS_DIR}/run.log"
SKIPPED_LOG="${RESULTS_DIR}/skipped.log"

echo "rep,job_id,hier,variant,family,gpus,ze_mask,N,dt,steps,wall_sec,reported_sec,status" > "$TIMING_CSV"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "============================================================"
echo "Host       : $(hostname)"
echo "Job ID     : ${JOB_ID}   Repetition: ${REP}"
echo "Timestamp  : ${TIMESTAMP}"
echo "Results dir: ${RESULTS_DIR}"
echo "N=${N}  DT=${DT}  STEPS=${STEPS}  FAMILIES='${FAMILIES}'  BASELINE=${BASELINE}"
echo "ZE_FLAT_DEVICE_HIERARCHY=${HIER}  (1 device = $( [[ $HIER == COMPOSITE ]] && echo 'whole card, 2 stacks' || echo 'one stack' ))"
echo "Hidden helpers for V1/V2 : ${HELPERS}"
echo "Compiler   : $(icx --version | head -1)"
echo "============================================================"
sycl-ls 2>/dev/null || xpu-smi discovery 2>/dev/null || true

trap 'echo "ERROR at line $LINENO" | tee -a "$SKIPPED_LOG"' ERR

SLEEP_SEC="${SLEEP_SEC:-15}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"

run_bench() {
    local binary="$1" gpus="$2"
    local family="${gpus}gpu"
    local mask; mask=$(mask_for "$gpus")

    if [[ ! -x "./$binary" ]]; then
        echo "  SKIP (missing): $binary  N=$N" | tee -a "$SKIPPED_LOG"
        echo "${REP},${JOB_ID},${HIER},${binary},${family},${gpus},${mask},${N},${DT},${STEPS},N/A,N/A,missing_binary" >> "$TIMING_CSV"
        return 0
    fi

    echo ""
    echo ">>> ${binary}  gpus=${gpus}  ZE_AFFINITY_MASK=${mask}  N=${N}"

    # No LD_PRELOAD here: unlike the AMD build, the Intel *-p2p binaries run on
    # the stock runtime. If cross-device P2P is not wired up the runtime stages
    # through the host, which shows up as roughly halved effective bandwidth
    # rather than an error.
    local run_cmd

    case "$binary" in
        1-omp)
            # One-card baseline: one application thread, no runtime helpers.
            run_cmd=(
                env
                -u LIBOMPTARGET_DEVICES
                "ZE_FLAT_DEVICE_HIERARCHY=${HIER}"
                "ZE_AFFINITY_MASK=${mask}"
                "OMP_NUM_THREADS=1"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=1"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"
                "./$binary" "$N" "$DT"
            )
            ;;

        *-omp-stream-omp-p2p|*-omp-stream-omp)
            # Versions 3 and 4:
            # one explicit application host thread per complete GPU card;
            # LLVM hidden-helper tasks are disabled. These sources check
            # OMP_NUM_HIDDEN_HELPER_THREADS=0, so it stays in the environment.
            #
            # OMP_THREAD_LIMIT is deliberately NOT set. thread-limit-var also
            # caps the threads in each team INSIDE the target regions, so
            # OMP_THREAD_LIMIT=${gpus} makes the GPU kernels run ~2-4 threads
            # per team and costs roughly 3x. Table 1 lists it; the table is
            # wrong, not the script.
            run_cmd=(
                env
                -u OMP_THREAD_LIMIT
                -u LIBOMPTARGET_DEVICES
                "ZE_FLAT_DEVICE_HIERARCHY=${HIER}"
                "ZE_AFFINITY_MASK=${mask}"
                "OMP_NUM_THREADS=${gpus}"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=0"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=0"		
                "./$binary" "$N" "$DT"
            )
            ;;

        *-omp-stream|*-omp)
            # Versions 1 and 2:
            # one application host thread; HELPERS LLVM runtime helper threads
            # progress the asynchronous target tasks. The helper count is NOT
            # tied to the device count -- see the note in the header.
            #
            # OMP_NUM_HIDDEN_HELPER_THREADS is removed here: these sources do
            # not read it, and leaving a global 0 in place would contradict the
            # LIBOMP_NUM_HIDDEN_HELPER_THREADS request below.
            run_cmd=(
                env
                -u OMP_NUM_HIDDEN_HELPER_THREADS
                -u OMP_THREAD_LIMIT
                -u LIBOMPTARGET_DEVICES
                "ZE_FLAT_DEVICE_HIERARCHY=${HIER}"
                "ZE_AFFINITY_MASK=${mask}"
                "OMP_NUM_THREADS=1"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=0"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=0"
                "./$binary" "$N" "$DT"
            )
            ;;

        *)
            echo "ERROR: unknown benchmark variant '$binary'" >&2
            echo "${REP},${JOB_ID},${HIER},${binary},${family},${gpus},${mask},${N},${DT},${STEPS},N/A,N/A,unknown_variant" >> "$TIMING_CSV"
            return 0
            ;;
    esac

    case "$binary" in
        1-omp)
            echo "    runtime: baseline; OMP_NUM_THREADS=1  helpers=disabled"
            ;;
        *-omp-stream-omp-p2p|*-omp-stream-omp)
            echo "    runtime: explicit threads=${gpus}  helpers=disabled"
            ;;
        *)
            echo "    runtime: application threads=1  helpers=${HELPERS}"
            ;;
    esac

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

    # Anchored on the solver's own "Time = <number> seconds" line. The previous
    # loose grep also matched "Final time (T): 0.000168" from the parameter
    # banner and recorded dt as the runtime.
    reported=$(
        sed -nE \
            's/^[[:space:]]*Time[[:space:]]*=[[:space:]]*([0-9.eE+-]+)[[:space:]]*seconds.*/\1/p' \
            "$stdout_file" |
        tail -n 1
    )
    [[ -n "$reported" ]] || reported="N/A"

    rm -f "$stdout_file"
    echo "    -> wall=${wall}s  reported=${reported}s  status=${status}  exit=${exit_code}"
    echo "${REP},${JOB_ID},${HIER},${binary},${family},${gpus},${mask},${N},${DT},${STEPS},${wall},${reported},${status}" >> "$TIMING_CSV"
    sleep "$SLEEP_SEC"
}

echo ""
echo "Starting benchmarks"
echo "==========================="
echo "N = $N   dt = $DT   helpers(V1/V2) = $HELPERS"
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
