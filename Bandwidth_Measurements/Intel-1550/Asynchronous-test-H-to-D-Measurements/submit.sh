#!/bin/bash
# Intel PVC counterpart of the LUMI submit.sh: one build job, then N run jobs
# that only start if the build succeeded (afterok).
#
# Usage:  ./submit.sh [N_RUNS] [HIER]
#   N_RUNS : repetitions for run-to-run variance (default 5)
#   HIER   : COMPOSITE | FLAT   (default COMPOSITE)

N_RUNS="${1:-5}"
HIER="${2:-COMPOSITE}"

if [[ "$HIER" != "COMPOSITE" && "$HIER" != "FLAT" ]]; then
  echo "ERROR: HIER must be COMPOSITE or FLAT (got: '$HIER')"; exit 1
fi

echo "============================================================"
echo "Submitting bandwidth pipeline  (Intel PVC / oneAPI)"
echo "  N_RUNS=${N_RUNS}  HIER=${HIER}"
echo "============================================================"

BUILD_JOB=$(sbatch --parsable build.sh)
echo "  Build job    : $BUILD_JOB"

RUN_DEP="afterok:$BUILD_JOB"
for i in $(seq 1 "$N_RUNS"); do
    JID=$(sbatch --parsable --dependency="$RUN_DEP" --export=ALL run.sh "$HIER")
    echo "  Run job $i    : $JID"
done

echo ""
echo "Pipeline submitted. Track with:"
echo "  squeue -u \$USER"
echo "  Once done, run ../all-plots.sh to regenerate the figures"
echo "============================================================"
