#!/bin/bash
# Build once, then submit a production HIP timing sweep.
#
# Usage:
#   ./submit 5
#
# The positional argument is the number of independent repetitions per N.
# Output naming follows the OpenMP workflow:
#   build-<jobid>.out
#   run-<jobid>.out
#   results_<jobid>_<timestamp>/
#
# Optional environment overrides:
#   FAMILIES=2,4
#   BASELINE=1
#   STEPS=500
#   TIME_LIMIT=01:30:00
#
# Examples:
#   ./submit 5
#   FAMILIES=2 ./submit 5
#   FAMILIES=4 BASELINE=1 ./submit 3

set -euo pipefail

N_RUNS="${1:-3}"
FAMILIES="${FAMILIES:-2,4}"
BASELINE="${BASELINE:-1}"
KAPPA_MAX="${KAPPA_MAX:-0.95}"
STEPS="${STEPS:-500}"
TIME_LIMIT="${TIME_LIMIT:-01:30:00}"

N_LIST=(512 640 768 896 1024 1152 1280)

if [[ ! "$N_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: ./submit <positive-number-of-repetitions>" >&2
    echo "Example: ./submit 5" >&2
    exit 1
fi

case "$FAMILIES" in
    2|4|2,4|4,2|2+4|4+2) ;;
    *)
        echo "ERROR: FAMILIES must be 2, 4, or 2,4; got '$FAMILIES'." >&2
        exit 1
        ;;
esac

if [[ "$BASELINE" != "0" && "$BASELINE" != "1" ]]; then
    echo "ERROR: BASELINE must be 0 or 1." >&2
    exit 1
fi

for required in Makefile build.sh run.sh; do
    if [[ ! -f "$required" ]]; then
        echo "ERROR: required file '$required' is missing from $PWD" >&2
        exit 1
    fi
done

# Submit the Makefile-driven build first.
BUILD_JOB_ID=$(sbatch --parsable build.sh)
BUILD_JOB_ID="${BUILD_JOB_ID%%;*}"
echo "Submitted build job ${BUILD_JOB_ID}: build-${BUILD_JOB_ID}.out"
echo "Run jobs will start only if build job ${BUILD_JOB_ID} succeeds."

FAMILIES_EXPORT="${FAMILIES//,/+}"
submitted=0

for N in "${N_LIST[@]}"; do
    # DT is total simulated time:
    #   STEPS * h^2/(12*kappa_max), h = 1/(N-1)
    DT=$(awk -v N="$N" -v k="$KAPPA_MAX" -v s="$STEPS" \
        'BEGIN { h=1.0/(N-1); dt=h*h/(12.0*k); printf "%.7g", s*dt }')

    for ((REP=1; REP<=N_RUNS; REP++)); do
        RUN_JOB_ID=$(
            sbatch --parsable \
                --dependency="afterok:${BUILD_JOB_ID}" \
                --time="$TIME_LIMIT" \
                --export="ALL,N=${N},DT=${DT},STEPS=${STEPS},REP=${REP},FAMILIES=${FAMILIES_EXPORT},BASELINE=${BASELINE}" \
                run.sh
        )
        RUN_JOB_ID="${RUN_JOB_ID%%;*}"

        echo "Submitted run job ${RUN_JOB_ID}: N=${N} DT=${DT} REP=${REP} FAMILIES=${FAMILIES} BASELINE=${BASELINE}"
        ((submitted+=1))
    done
done

echo ""
echo "Submitted 1 build job and ${submitted} run job(s)."
echo "Build output : build-${BUILD_JOB_ID}.out"
echo "Run outputs  : run-<jobid>.out"
echo "Results      : results_<jobid>_<timestamp>/"
