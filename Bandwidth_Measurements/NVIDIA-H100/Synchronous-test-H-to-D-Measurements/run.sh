#!/bin/bash -l                                                                                   
#SBATCH --job-name=run                                                                       
#SBATCH --time=00:30:00                                                                          
#SBATCH --account=ehpc681                                                                        
#SBATCH --partition=acc                                                                          
#SBATCH --qos=acc_ehpc                                                                           
#SBATCH --nodes=1                                                                                
#SBATCH --exclusive                                                                               
#SBATCH --output=run-%j.out

# Load clang + CUDA first, capture clean LD path before NVHPC pollutes it
module purge
module load EB/apps
module load GCC/13.2.0
module load cuda/12.8
module load clang/18.1.8-cuda12.8

CLEAN_LD_PATH="${LD_LIBRARY_PATH:-}"

# Now load NVHPC for OpenACC runtime
module purge
module load EB/apps
module load NVHPC/23.7-CUDA-12.2.0

# Output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_DIR="results_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

TIMING_CSV="${RESULTS_DIR}/timings.csv"
RUN_LOG="${RESULTS_DIR}/run.log"
SKIPPED_LOG="${RESULTS_DIR}/skipped.log"

echo "framework,version,N,elapsed_sec,bandwidth_gbs,status" > "$TIMING_CSV"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "============================================================"
echo "Job ID     : ${JOB_ID}"
echo "Timestamp  : ${TIMESTAMP}"
echo "Results dir: ${RESULTS_DIR}"
echo "CLEAN_LD_PATH: ${CLEAN_LD_PATH}"
echo "============================================================"

trap 'echo "ERROR at line $LINENO" | tee -a "$SKIPPED_LOG"' ERR

SLEEP_SEC=15
TIMEOUT_SEC=300

run_bench() {
    local framework="$1" version="$2" N="$3" env_prefix="$4" binary="$5"

    if [[ ! -x "./$binary" ]]; then
        echo "  SKIP (missing): $binary  N=$N" | tee -a "$SKIPPED_LOG"
        echo "${framework},${version},${N},N/A,N/A,missing_binary" >> "$TIMING_CSV"
        return 0
    fi

    echo ""
    echo ">>> Running: framework=${framework}  version=${version}  N=${N}"

    if [[ "$env_prefix" == "clean_ld" ]]; then
        run_cmd=(env LD_LIBRARY_PATH="$CLEAN_LD_PATH" "./$binary" "$N")
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
echo "============================================================"
echo "Starting benchmarks"
echo "============================================================"

for N in "${Ns[@]}"; do
    echo ""
    echo "==========================="
    echo "N = $N"
    echo "==========================="

    run_bench "omp_off" "v1"  "$N" clean_ld version-1-off
    run_bench "omp_off" "v2"  "$N" clean_ld version-2-off

    run_bench "cuda"    "v1"  "$N" default  version-1-cuda
    run_bench "cuda"    "v2"  "$N" default  version-2-cuda

done

