#!/bin/bash -l
#SBATCH --job-name=heat-sycl-run
#SBATCH --account=pn67qe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --output=run-sycl-%j.out
#
# One job = one grid size N, one repetition, all 2- and 4-GPU SYCL variants.
# submit.sh launches N_RUNS of these per N so you get N_RUNS independent samples.
#
# Inputs (via sbatch --export):
#   N        grid size                      (required)
#   DT       total simulated time           (required)
#   STEPS    step count DT was derived from (recorded in the CSV only)
#   REP      repetition index 1..N_RUNS     (default 1)
#   HIER     COMPOSITE                      (locked; see below)
#   FAMILIES which families to run: "2 4"   (default "2 4")
#   BASELINE 1 -> also run the 1-card baseline (default 1)
#
# WHAT IS *NOT* HERE, AND WHY
# ---------------------------
# The OpenMP run.sh spends most of its environment block on hidden-helper
# threads (LIBOMP_NUM_HIDDEN_HELPER_THREADS, OMP_NUM_HIDDEN_HELPER_THREADS,
# LIBOMP_USE_HIDDEN_HELPER_TASK) and on LIBOMPTARGET_DEVICES. Every one of
# those is an LLVM OpenMP *offload* knob. SYCL does not use libomptarget at
# all: asynchrony comes from the SYCL queues themselves (out-of-order queues,
# events, and in the *-stream variants explicit sub-queues), so there is no
# helper-thread pool to size and no HELPERS sweep to run. Setting them here
# would change nothing and would suggest the two pipelines are more comparable
# in their tuning than they are.
#
# Host OpenMP IS still live, because every source calls omp_get_wtime() and the
# *-stream-omp / *-stream-omp-p2p variants drive one device per host thread.
# So OMP_NUM_THREADS is set per binary, exactly as in the OpenMP script.

set -Euo pipefail   # deliberately NOT -e: one failing variant must not kill the rep

module load slurm_setup
module load intel-toolkit
module load intel-dpct 2>/dev/null || true

WORK_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORK_DIR"

# ---------------------------------------------------------------------------
# Environment, identical for every binary in the comparison set
# ---------------------------------------------------------------------------

# Keep only the Level Zero GPUs. Without this the OpenCL backend exposes the
# SAME cards a second time, dpct::select_device(0..K-1) / device enumeration
# picks up duplicates, and a "4 GPU" run can silently land two ranks on one
# card through two different backends.
#
# (This is the one place the SYCL script does the OPPOSITE of the OpenMP one:
#  there ONEAPI_DEVICE_SELECTOR is unset because libomptarget does its own
#  device discovery and the variable only confuses it.)
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1     # stable, PCI-ordered device numbering

# JIT fallback insurance: if the binaries were built with AOT=0, the SPIR-V ->
# PVC translation lands inside the solver's own timed region on a cold cache.
# A persistent cache means only the first job on a node pays it. With AOT=1
# (the default) this is a no-op.
export SYCL_CACHE_PERSISTENT=1

export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export WRITE_TXT=0                          # timing runs never write solutions

N="${N:?ERROR: N not set (use: sbatch --export=ALL,N=...,DT=... run.sh)}"
DT="${DT:?ERROR: DT not set}"
STEPS="${STEPS:-unknown}"
REP="${REP:-1}"
HIER="COMPOSITE"   # force one SYCL device = one whole Max 1550 card
FAMILIES="${FAMILIES:-2 4}"
BASELINE="${BASELINE:-1}"

[[ "$HIER" == "COMPOSITE" ]] || {
    echo "ERROR: this run.sh is locked to whole-card COMPOSITE mode." >&2; exit 1; }

# COMPOSITE: one root SYCL device = one whole PVC card, both stacks driven by
# implicit scaling. FLAT would make a device a single stack, so "2-sycl" would
# then mean two halves of one card -- a different experiment, and not the one
# the OpenMP numbers were taken in. Keep them comparable: COMPOSITE both sides.
export ZE_FLAT_DEVICE_HIERARCHY="$HIER"

# ---------------------------------------------------------------------------
# Device mask in whole-card COMPOSITE mode:
#
#   index = CARD. One SYCL device = one whole Max 1550 card.
#
#   1-card code -> "0"
#   2-card code -> "0,1"
#   4-card code -> "0,1,2,3"
#
# The mask is set PER BINARY, not globally: the 2-GPU codes must see exactly
# two complete cards and the 4-GPU codes exactly four complete cards. A stray
# inherited ZE_AFFINITY_MASK riding in through --export=ALL would otherwise cap
# a 4-GPU run at one card ("Need 4 GPUs, found 1"), so it is always overwritten.
# ---------------------------------------------------------------------------
mask_for() {
    local n="$1" v=() i
    for ((i = 0; i < n; i++)); do v+=("$i"); done              # 0,1,2,3
    (IFS=,; echo "${v[*]}")
}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_DIR="results_sycl_${HIER}_${JOB_ID}_${TIMESTAMP}"
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
echo "ZE_FLAT_DEVICE_HIERARCHY=${HIER}  (1 device = whole card, 2 stacks)"
echo "ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR}"
echo "Compiler   : $(icpx --version | head -1)"
echo "============================================================"
sycl-ls 2>/dev/null || xpu-smi discovery 2>/dev/null || true

trap 'echo "ERROR at line $LINENO" | tee -a "$SKIPPED_LOG"' ERR

SLEEP_SEC="${SLEEP_SEC:-15}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"

# Count the Level Zero devices actually visible under a given mask. Catching a
# mismatch here turns a confusing mid-run abort into one clear line.
visible_devices() {
    local mask="$1"
    env "ZE_FLAT_DEVICE_HIERARCHY=${HIER}" \
        "ZE_AFFINITY_MASK=${mask}" \
        "ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR}" \
        "ZE_ENABLE_PCI_ID_DEVICE_ORDER=1" \
        sycl-ls 2>/dev/null | grep -c ext_oneapi_level_zero || true
}

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

    local seen; seen=$(visible_devices "$mask")
    if [[ "$seen" -ne "$gpus" ]]; then
        echo "  SKIP (device mismatch): mask '${mask}' exposes ${seen} L0 device(s), need ${gpus}." | tee -a "$SKIPPED_LOG"
        echo "${REP},${JOB_ID},${HIER},${binary},${family},${gpus},${mask},${N},${DT},${STEPS},N/A,N/A,device_mismatch" >> "$TIMING_CSV"
        return 0
    fi

    # Host thread count. The device-side parallelism is entirely the SYCL
    # runtime's business; this only decides how many host threads submit work.
    local threads runtime_note
    case "$binary" in
        *-sycl-stream-omp-p2p|*-sycl-stream-omp)
            # Versions 3 and 4: one explicit host thread per complete card,
            # each owning its own SYCL queue(s).
            threads="$gpus"
            runtime_note="explicit host threads=${gpus} (one per card)"
            ;;
        1-sycl|*-sycl-stream|*-sycl)
            # Baseline and versions 1/2: a single host thread submits to all
            # devices; overlap comes from the queues, not from host threads.
            threads=1
            runtime_note="host threads=1; overlap from SYCL queues"
            ;;
        *)
            echo "ERROR: unknown benchmark variant '$binary'" >&2
            echo "${REP},${JOB_ID},${HIER},${binary},${family},${gpus},${mask},${N},${DT},${STEPS},N/A,N/A,unknown_variant" >> "$TIMING_CSV"
            return 0
            ;;
    esac

    # No LD_PRELOAD and no P2P shim: the Intel *-p2p binaries run on the stock
    # runtime. If cross-card P2P copies are not wired up the runtime stages
    # through the host, which shows up as roughly halved effective bandwidth
    # rather than as an error -- so compare p2p against stream-omp, do not
    # assume a missing feature would have failed loudly.
    local run_cmd=(
        env
        "ZE_FLAT_DEVICE_HIERARCHY=${HIER}"
        "ZE_AFFINITY_MASK=${mask}"
        "ZE_ENABLE_PCI_ID_DEVICE_ORDER=1"
        "ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR}"
        "OMP_NUM_THREADS=${threads}"
        "./$binary" "$N" "$DT"
    )

    echo "    runtime: ${runtime_note}"
    printf '    command:'
    printf ' %q' "${run_cmd[@]}"
    printf '\n'

    local stdout_file status wall reported t_start t_end exit_code
    stdout_file=$(mktemp /tmp/heat_sycl_stdout.XXXXXX)
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

    # Anchored on the solver's own "Time = <number> seconds" line. A loose grep
    # also matches "Final time (T): 0.000168" from the parameter banner and
    # records dt as the runtime.
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
echo "N = $N   dt = $DT"
echo "==========================="

if [[ "$BASELINE" == "1" ]]; then
    run_bench 1-sycl 1
fi

for fam in $FAMILIES; do
    run_bench "${fam}-sycl"                "$fam"
    run_bench "${fam}-sycl-stream"         "$fam"
    run_bench "${fam}-sycl-stream-omp"     "$fam"
    run_bench "${fam}-sycl-stream-omp-p2p" "$fam"
done

echo ""
echo "All done."
echo "  Timing CSV : $TIMING_CSV"
echo "  Run log    : $RUN_LOG"
[[ -s "$SKIPPED_LOG" ]] && \
    echo "  Skipped    : $SKIPPED_LOG  ($(wc -l < "$SKIPPED_LOG") entries)"
exit 0
