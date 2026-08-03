#!/bin/bash
#SBATCH --job-name=asyn-dh-run
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --account=project_465003145
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=run-%j.out

module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4

export HSA_ENABLE_SDMA=0
export OMP_TARGET_OFFLOAD=mandatory
export ROCR_VISIBLE_DEVICES=0,2,4,6        # one GCD per physical MI250X card
OMPTARGET_P2P_LIB="${OMPTARGET_P2P_LIB:-/scratch/project_465003145/omptarget-p2p/lib}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_DIR="results_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

TIMING_CSV="${RESULTS_DIR}/timings.csv"
RUN_LOG="${RESULTS_DIR}/run.log"
SKIPPED_LOG="${RESULTS_DIR}/skipped.log"

echo "framework,version,N,elapsed_sec,bandwidth_gbs,status" > "$TIMING_CSV"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "Job ID     : ${JOB_ID}"
echo "Timestamp  : ${TIMESTAMP}"
echo "Results dir: ${RESULTS_DIR}"

trap 'echo "ERROR at line $LINENO" | tee -a "$SKIPPED_LOG"' ERR

SLEEP_SEC=15
TIMEOUT_SEC=300

run_bench() {
    local framework="$1" version="$2" N="$3" use_p2p="$4" binary="$5"

    if [[ ! -x "./$binary" ]]; then
        echo "  SKIP (missing): $binary  N=$N" | tee -a "$SKIPPED_LOG"
        echo "${framework},${version},${N},N/A,N/A,missing_binary" >> "$TIMING_CSV"
        return 0
    fi

    echo ""
    echo ">>> Running: framework=${framework}  version=${version}  N=${N}"

    if [[ "$use_p2p" == "yes" ]]; then
        run_cmd=(
            env
            LD_PRELOAD="$OMPTARGET_P2P_LIB/libomp.so:$OMPTARGET_P2P_LIB/libomptarget.so.19.0git"
            "./$binary" "$N"
        )
    else
        run_cmd=("./$binary" "$N")
    fi

    local stdout_file status elapsed bw t_start t_end
    stdout_file=$(mktemp /tmp/bench_stdout.XXXXXX)
    t_start=$(date +%s%3N)

    if timeout "$TIMEOUT_SEC" "${run_cmd[@]}" > "$stdout_file" 2>&1; then
        status="ok"
    else
        local exit_code=$?
        [[ $exit_code -eq 124 ]] && status="timeout" || status="runtime_error"
    fi

    t_end=$(date +%s%3N)
    elapsed=$(echo "scale=4; ($t_end - $t_start) / 1000" | bc)
    cat "$stdout_file"

    local reported_time
    reported_time=$(grep -iEo '(time|elapsed)[^0-9]*([0-9]+\.?[0-9]*)' "$stdout_file" \
                    | grep -oE '[0-9]+\.?[0-9]*' | head -1 || true)
    [[ -n "$reported_time" ]] && elapsed="$reported_time"

    bw=$(grep -iEo '([0-9]+\.?[0-9]+)[[:space:]]*(GB/s|gb/s|GBs|gbps)' "$stdout_file" \
         | grep -oE '[0-9]+\.?[0-9]+' | head -1 || true)
    [[ -z "$bw" ]] && bw="N/A"

    rm -f "$stdout_file"
    echo "    -> elapsed=${elapsed}s  bandwidth=${bw} GB/s  status=${status}"
    echo "${framework},${version},${N},${elapsed},${bw},${status}" >> "$TIMING_CSV"
    sleep "$SLEEP_SEC"
}

Ns=(940000000 1740000000 2550000000 3350000000 4160000000 4960000000)

echo ""
echo "Starting benchmarks"

for N in "${Ns[@]}"; do
    echo ""
    echo "==========================="
    echo "N = $N"
    echo "==========================="

    run_bench "omp_off" "v1" "$N" yes version-1-off
    run_bench "omp_off" "v2" "$N" yes version-2-off

    run_bench "hip"     "v1" "$N" no  version-1-hip
    run_bench "hip"     "v2" "$N" no  version-2-hip
done

echo ""
echo "All done."
echo "  Timing CSV : $TIMING_CSV"
echo "  Run log    : $RUN_LOG"
[[ -s "$SKIPPED_LOG" ]] && \
    echo "  Skipped    : $SKIPPED_LOG  ($(wc -l < "$SKIPPED_LOG") entries)"
