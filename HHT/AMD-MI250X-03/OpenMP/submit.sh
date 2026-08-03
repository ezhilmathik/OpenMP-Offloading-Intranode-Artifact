#!/bin/bash
# ---------------------------------------------------------------------------
# 3D heat equation: build once, then N_RUNS independent timing repetitions
# for every grid size. MI250X / LUMI, one GCD per physical module.
# ---------------------------------------------------------------------------
#
#  Usage:  ./submit.sh [N_RUNS]        (default 5)
#
#  Knobs (edit below or export before calling):
#    NS         grid sizes to sweep
#    FAMILIES   "2 4" | "2" | "4"
#    BASELINE   1 -> also time the 1-GCD baseline in every job
#    STEPS      time-step count used to derive dt
# ---------------------------------------------------------------------------

set -euo pipefail

N_RUNS="${1:-5}"
NS="${NS:-512 640 768 896 1024 1152 1280}"
FAMILIES="${FAMILIES:-2 4}"
BASELINE="${BASELINE:-1}"
STEPS="${STEPS:-500}"
KAPPA_MAX="${KAPPA_MAX:-0.95}"

[[ "$N_RUNS" =~ ^[0-9]+$ && "$N_RUNS" -ge 1 ]] || {
    echo "ERROR: N_RUNS must be a positive integer (got '$N_RUNS')" >&2; exit 1; }

# The first argument is the REPETITION COUNT, not a grid size. A value that
# looks like an N (128, 256, 512, ...) is almost certainly the confusion
# `./submit.sh 512` when `NS="512" ./submit.sh 1` was meant.
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
echo "Submitting 3D-heat timing pipeline  (MI250X / LUMI)"
echo "============================================================"
echo "  Repetitions : $N_RUNS"
echo "  Grid sizes  : $NS"
echo "  Families    : $FAMILIES  (baseline=$BASELINE)"
echo "  Steps       : $STEPS"
echo ""

BUILD_JOB=$(sbatch --parsable build.sh)
echo "  Build job    : $BUILD_JOB"
echo ""

RUN_DEP="afterok:$BUILD_JOB"
TOTAL=0

for N in $NS; do
    # dt = h^2 / (12*kappa_max),  h = 1/(N-1),  T = STEPS*dt
    DT=$(awk -v N="$N" -v k="$KAPPA_MAX" -v s="$STEPS" \
          'BEGIN { h = 1.0/(N-1); dt = h*h/(12.0*k); printf "%.7g", s*dt }')

    for rep in $(seq 1 "$N_RUNS"); do
        JID=$(sbatch --parsable \
                     --dependency="$RUN_DEP" \
                     --job-name="heat-N${N}-r${rep}" \
                     --export="ALL,N=${N},DT=${DT},STEPS=${STEPS},REP=${rep},FAMILIES=${FAMILIES},BASELINE=${BASELINE}" \
                     run.sh)
        echo "  N=${N} rep=${rep}  : $JID"
        TOTAL=$((TOTAL + 1))
    done
done

echo ""
echo "Pipeline submitted: 1 build + ${TOTAL} run jobs."
echo "Track with:"
echo "  squeue -u \$USER"
echo "  Once done:  ./aggregate.sh   (merges results_*/timings.csv, takes medians)"
echo "============================================================"
