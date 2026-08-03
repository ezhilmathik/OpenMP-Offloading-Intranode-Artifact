#!/bin/bash -l
#SBATCH --job-name=run
#SBATCH --time=01:00:00
#SBATCH --account=ehpc681
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --output=run-%j.out

# ---------------------------------------------------------------------------
# What to measure.
#
# This study uses v1 (unidirectional, GPU 0 -> GPU 1) and v2 (bidirectional,
# GPU 0 <-> GPU 1) with OpenMP target offload and CUDA. Both flags below
# default to 0.
# ---------------------------------------------------------------------------
RUN_FUTURE=0     # 1 also measures v4 (gather) and v4_4 (all-to-all)
RUN_ACC=0        # 1 also measures OpenACC; requires NVHPC and `make acc`

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load EB/apps
module load GCC/13.2.0
module load cuda/12.8
module load clang/18.1.8-cuda12.8

# Captured before any NVHPC load, so the clang-built binaries can be run with
# a clean library path even when NVHPC is present.
CLEAN_LD_PATH="${LD_LIBRARY_PATH:-}"

if (( RUN_ACC )); then
    # NVHPC supplies the OpenACC runtime but overwrites LD_LIBRARY_PATH,
    # which is why CLEAN_LD_PATH is captured above and restored per run.
    module purge
    module load EB/apps
    module load NVHPC/23.7-CUDA-12.2.0
fi

# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_DIR="results_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

TIMING_CSV="${RESULTS_DIR}/timings.csv"
RUN_LOG="${RESULTS_DIR}/run.log"
SKIPPED_LOG="${RESULTS_DIR}/skipped.log"

# Keep a copy of the script that produced these results, so the parameters
# used are always recoverable from the results directory itself.
if [[ -f "${BASH_SOURCE[0]}" ]]; then
    cp "${BASH_SOURCE[0]}" "${RESULTS_DIR}/run.sh.used"
fi

echo "framework,version,N,elapsed_sec,bandwidth_gbs,status" > "$TIMING_CSV"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "============================================================"
echo "Job ID     : ${JOB_ID}"
echo "Timestamp  : ${TIMESTAMP}"
echo "Results dir: ${RESULTS_DIR}"
echo "RUN_FUTURE : ${RUN_FUTURE}"
echo "RUN_ACC    : ${RUN_ACC}"
echo "CLEAN_LD_PATH: ${CLEAN_LD_PATH}"
echo "============================================================"

trap 'echo "ERROR at line $LINENO" | tee -a "$SKIPPED_LOG"' ERR

SLEEP_SEC=15
TIMEOUT_SEC=300

# ---------------------------------------------------------------------------
# run_bench <framework> <version> <N> <clean_ld|default> <binary>
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Buffer sizes, in elements (double, 8 bytes each)
# ---------------------------------------------------------------------------
Ns=(117500000 217500000 318750000 418750000 520000000 1240000000)

# Halved sizes for the 4-to-4 all-to-all case: each GPU drives three peers,
# so per-peer buffers are smaller to keep total device memory comparable.
Ns_4to4=(58750000 108750000 159375000 209375000 260000000 310000000)

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

    if (( RUN_ACC )); then
        run_bench "acc" "v1"  "$N" default  version-1-acc
        run_bench "acc" "v2"  "$N" default  version-2-acc
    fi

    if (( RUN_FUTURE )); then
        run_bench "omp_off" "v4" "$N" clean_ld version-4-off
        run_bench "cuda"    "v4" "$N" default  version-4-cuda
        if (( RUN_ACC )); then
            run_bench "acc" "v4" "$N" default version-4-acc
        fi
    fi
done

if (( RUN_FUTURE )); then
    for N in "${Ns_4to4[@]}"; do
        echo ""
        echo "==========================="
        echo "N = $N  (4-to-4)"
        echo "==========================="

        run_bench "omp_off" "v4_4" "$N" clean_ld version-4-4-off
        run_bench "cuda"    "v4_4" "$N" default  version-4-4-cuda
        if (( RUN_ACC )); then
            run_bench "acc" "v4_4" "$N" default version-4-4-acc
        fi
    done
fi

echo ""
echo "============================================================"
echo "Done. Results in ${RESULTS_DIR}"
echo "============================================================"
