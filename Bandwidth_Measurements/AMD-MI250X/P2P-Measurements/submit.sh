#!/bin/bash

N_RUNS=5

echo "============================================================"
echo "Submitting P2P benchmark pipeline  (MI250X / LUMI)"
echo "============================================================"

BUILD_JOB=$(sbatch --parsable build.sh)
echo "  Build job    : $BUILD_JOB"

RUN_DEP="afterok:$BUILD_JOB"
RUN_JOBS=""
for i in $(seq 1 $N_RUNS); do
    JID=$(sbatch --parsable --dependency="$RUN_DEP" run.sh)
    echo "  Run job $i    : $JID"
    RUN_JOBS="${RUN_JOBS}:${JID}"
done

echo ""
echo "Pipeline submitted. Track with:"
echo "  squeue -u \$USER"
echo "  Once done, run ../all-plots.sh to regenerate the figures"
echo "============================================================"
