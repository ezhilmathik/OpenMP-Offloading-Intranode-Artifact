#!/bin/bash -l
# ---------------------------------------------------------------------------
# 3D heat equation, SYCL: build once, then N_RUNS independent timing
# repetitions for every grid size. Intel Data Center GPU Max 1550 (PVC).
# ---------------------------------------------------------------------------
#
#  Usage:  ./submit.sh [N_RUNS]        (default 5)
#
#  Knobs:
#    NS         grid sizes to sweep
#    FAMILIES   "2 4" | "2" | "4"
#    BASELINE   1 -> also time the 1-card baseline in every job
#    STEPS      time-step count used to derive dt
#    AOT        1 (default) ahead-of-time PVC codegen | 0 JIT
#    SKIP_BUILD 1 -> reuse existing binaries
#
#  Keep NS, STEPS and KAPPA_MAX identical to the OpenMP sweep, or the two sets
#  of numbers are not comparable: dt is derived from them, so a different STEPS
#  means a different amount of work per run, not a different runtime.
# ---------------------------------------------------------------------------

set -euo pipefail

N_RUNS="${1:-5}"
NS="${NS:-512 640 768 896 1024 1152 1280}"
HIER="COMPOSITE"                 # run.sh is locked to whole-card mode
FAMILIES="${FAMILIES:-2 4}"
BASELINE="${BASELINE:-1}"
STEPS="${STEPS:-500}"
KAPPA_MAX="${KAPPA_MAX:-0.95}"
AOT="${AOT:-1}"

[[ "$N_RUNS" =~ ^[0-9]+$ && "$N_RUNS" -ge 1 ]] || {
    echo "ERROR: N_RUNS must be a positive integer (got '$N_RUNS')" >&2; exit 1; }

# The first argument is the REPETITION COUNT, not a grid size.
if [[ "$N_RUNS" -gt 20 ]]; then
    echo "ERROR: N_RUNS=${N_RUNS} would submit $(( N_RUNS * $(wc -w <<< "$NS") )) run jobs." >&2
    echo "" >&2
    echo "  The first argument is the number of REPETITIONS, not the grid size." >&2
    echo "  To run one repetition at one grid size:" >&2
    echo "      NS=\"${N_RUNS}\" ./submit.sh 1" >&2
    echo "" >&2
    echo "  If you really do want ${N_RUNS} repetitions, set FORCE_RUNS=1." >&2
    [[ "${FORCE_RUNS:-0}" == "1" ]] || exit 1
    echo "  FORCE_RUNS=1 set -- proceeding anyway." >&2
fi

echo "============================================================"
echo "Submitting 3D-heat SYCL timing pipeline  (Intel PVC Max 1550)"
echo "============================================================"
echo "  Repetitions : $N_RUNS"
echo "  Grid sizes  : $NS"
echo "  Hierarchy   : $HIER  -- 1 device = 1 whole Max 1550 card (2 stacks)"
echo "  Families    : $FAMILIES  (baseline=$BASELINE)"
echo "  Steps       : $STEPS"
echo "  AOT         : $AOT"
echo ""

MAX_QUEUE="${MAX_QUEUE:-25}"
POLL="${POLL:-60}"

# Sites cap how many jobs a user may have queued at once. Without this, sbatch
# starts refusing partway through the sweep and `set -e` aborts the script --
# leaving a silently incomplete matrix.
wait_for_slot() {
    local n
    while :; do
        n=$(squeue -h -u "${USER:-$(id -un)}" 2>/dev/null | wc -l)
        [[ "$n" -lt "$MAX_QUEUE" ]] && return 0
        echo "    queue full (${n}/${MAX_QUEUE}) -- waiting ${POLL}s..."
        sleep "$POLL"
    done
}

# No --dependency=afterok: once the throttle pauses, the build finishes and
# ages out of Slurm, and later submissions are rejected ("Job dependency
# problem") or cancelled as DependencyNeverSatisfied -- jobs that never run.
# Wait for the build instead, and verify it actually produced binaries.
if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo "  Build        : SKIPPED (reusing existing binaries)"
else
    BUILD_JOB=$(sbatch --parsable --export="ALL,AOT=${AOT}" build.sh)
    echo "  Build job    : $BUILD_JOB  -- waiting..."
    while squeue -h -j "$BUILD_JOB" 2>/dev/null | grep -q .; do sleep 20; done
    echo "  Build        : finished"
fi

MISSING=()
for b in 1-sycl 2-sycl 2-sycl-stream 2-sycl-stream-omp 2-sycl-stream-omp-p2p \
         4-sycl 4-sycl-stream 4-sycl-stream-omp 4-sycl-stream-omp-p2p; do
    [[ -x "./$b" ]] || MISSING+=("$b")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: missing binaries: ${MISSING[*]}" >&2
    echo "       Check build-sycl-*.out -- not submitting the sweep." >&2
    exit 1
fi
echo "  Binaries     : all 9 present"
echo ""

TOTAL=0
FAILED_SUBMITS=0

for N in $NS; do
    # dt = h^2 / (12*kappa_max),  h = 1/(N-1),  T = STEPS*dt
    DT=$(awk -v N="$N" -v k="$KAPPA_MAX" -v s="$STEPS" \
          'BEGIN { h = 1.0/(N-1); dt = h*h/(12.0*k); printf "%.7g", s*dt }')

    for rep in $(seq 1 "$N_RUNS"); do
        wait_for_slot
        JID=$(sbatch --parsable \
                     --job-name="heat-sycl-N${N}-r${rep}" \
                     --export="ALL,N=${N},DT=${DT},STEPS=${STEPS},REP=${rep},HIER=${HIER},FAMILIES=${FAMILIES},BASELINE=${BASELINE}" \
                     run.sh) || {
            echo "  N=${N} rep=${rep}  : SUBMIT FAILED -- rerun for this size" >&2
            FAILED_SUBMITS=$((FAILED_SUBMITS + 1))
            continue
        }
        echo "  N=${N} rep=${rep}  : $JID"
        TOTAL=$((TOTAL + 1))
    done
done

echo ""
EXPECTED=$(( $(wc -w <<< "$NS") * N_RUNS ))
echo "Pipeline submitted: 1 build + ${TOTAL} run jobs (expected ${EXPECTED})."
if [[ "$TOTAL" -ne "$EXPECTED" || "$FAILED_SUBMITS" -gt 0 ]]; then
    echo "WARNING: ${FAILED_SUBMITS} submission(s) failed -- the result matrix will have holes." >&2
fi
echo "Track with:"
echo "  squeue -u \$USER"
echo "  Once done:  ./aggregate.sh   (merges results_*/timings.csv, takes medians)"
echo "============================================================"
