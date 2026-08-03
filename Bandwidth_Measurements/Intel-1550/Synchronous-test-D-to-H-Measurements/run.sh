#!/bin/bash -l
#SBATCH --job-name=syn-dh-run
#SBATCH --account=pn67qe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=run-%j.out

# =============================================================================
# Intel PVC port of the LUMI/MI250X run.sh.
#
# Structure is unchanged: run_bench() -> timings.csv + run.log + skipped.log,
# same CSV schema (one column appended, see below), same timeout/sleep logic.
#
# What changed, and why:
#
#  1. ROCR_VISIBLE_DEVICES=0,2,4,6  ->  ZE_AFFINITY_MASK (hierarchy-dependent).
#     Your LUMI mask deliberately took ONE GCD PER PHYSICAL CARD, so no two
#     devices shared a card's host link. The Intel equivalent of that INTENT is
#     COMPOSITE + "0,1,2,3": 4 whole cards, one full link each.
#     The literal structural analogue (half a card each, 4 different cards) is
#     FLAT + "0,2,4,6". Both are provided; pick with the HIER argument.
#
#  2. HSA_ENABLE_SDMA=0  ->  DROPPED (see notes at the bottom).
#
#  3. LD_PRELOAD of omptarget-p2p  ->  DROPPED. That was a patched LLVM
#     libomptarget built for LUMI. Intel's Level Zero exposes P2P natively and
#     icx's libomptarget uses it; there is no library to preload. The `use_p2p`
#     argument to run_bench is therefore gone.
#
#  4. OMP_TARGET_OFFLOAD=mandatory  ->  KEPT. It is a standard OpenMP 5.0
#     variable, not a ROCm one, and icx honours it. Worth keeping: it turns a
#     silent host fallback into a hard failure, which is exactly what you want
#     when the number you are reporting is a GPU bandwidth.
#
#  5. Ns array: the commas are removed (they were a bug, see below).
#
# Usage:  sbatch run.sh [HIER]      HIER = COMPOSITE (default) | FLAT
# =============================================================================

HIER="${1:-COMPOSITE}"
if [[ "$HIER" != "COMPOSITE" && "$HIER" != "FLAT" ]]; then
  echo "ERROR: HIER must be COMPOSITE or FLAT (got: '$HIER')"; exit 1
fi

module load slurm_setup
module load intel-toolkit

# --- Offload / Level Zero environment ---------------------------------------
export OMP_TARGET_OFFLOAD=mandatory       # fail loudly instead of running on host
export ZE_FLAT_DEVICE_HIERARCHY="$HIER"
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1    # stable PCI-ordered device numbering
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu   # hide the duplicate OpenCL GPUs

# One device per physical card (the ROCR_VISIBLE_DEVICES=0,2,4,6 analogue).
#   COMPOSITE: index = card  -> "0,1,2,3" = 4 whole cards (1 full link each)
#   FLAT     : index = stack -> "0,2,4,6" = 1 stack from each of 4 cards
if [[ "$HIER" == "COMPOSITE" ]]; then
  export ZE_AFFINITY_MASK="0,1,2,3"
else
  export ZE_AFFINITY_MASK="0,2,4,6"
fi

# Persistent JIT cache: without AOT, the FIRST SYCL run of each binary pays for
# SPIR-V -> PVC codegen inside the timed region. Harmless if your binaries print
# their own timing (we prefer that below), but it poisons the wall-clock
# fallback. Better still: build AOT (see Makefile).
export SYCL_CACHE_PERSISTENT=1

# NOT SET, deliberately -- the SDMA analogue. See notes at the bottom.
# export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_DIR="results_${HIER}_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

TIMING_CSV="${RESULTS_DIR}/timings.csv"
RUN_LOG="${RESULTS_DIR}/run.log"
SKIPPED_LOG="${RESULTS_DIR}/skipped.log"
ENV_TXT="${RESULTS_DIR}/env.txt"

# NOTE: 'hierarchy' appended as a 7th column. On LUMI a GCD is a GCD; on PVC the
# same "4 devices" is different silicon under FLAT vs COMPOSITE, so the number is
# meaningless without it. Your plot.py needs to tolerate the extra column.
echo "framework,version,N,elapsed_sec,bandwidth_gbs,status,hierarchy" > "$TIMING_CSV"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "Job ID     : ${JOB_ID}"
echo "Timestamp  : ${TIMESTAMP}"
echo "Hierarchy  : ${HIER}  (ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK})"
echo "Results dir: ${RESULTS_DIR}"

{
  echo "host=$(hostname)"
  echo "date=$(date -Is)"
  echo "jobid=${JOB_ID}"
  echo "hierarchy=${ZE_FLAT_DEVICE_HIERARCHY}"
  echo "ze_affinity_mask=${ZE_AFFINITY_MASK}"
  echo "compiler=$(icpx --version 2>/dev/null | head -1)"
  echo "--- sycl-ls ---"
  sycl-ls 2>/dev/null
} > "$ENV_TXT"

echo ""
echo "Visible L0 GPUs: $(sycl-ls 2>/dev/null | grep -c ext_oneapi_level_zero)"
sycl-ls 2>/dev/null || xpu-smi discovery 2>/dev/null || true

trap 'echo "ERROR at line $LINENO" | tee -a "$SKIPPED_LOG"' ERR

SLEEP_SEC=15
TIMEOUT_SEC=300

run_bench() {
    local framework="$1" version="$2" N="$3" binary="$4"

    if [[ ! -x "./$binary" ]]; then
        echo "  SKIP (missing): $binary  N=$N" | tee -a "$SKIPPED_LOG"
        echo "${framework},${version},${N},N/A,N/A,missing_binary,${HIER}" >> "$TIMING_CSV"
        return 0
    fi

    echo ""
    echo ">>> Running: framework=${framework}  version=${version}  N=${N}"

    local run_cmd=("./$binary" "$N")

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
    echo "${framework},${version},${N},${elapsed},${bw},${status},${HIER}" >> "$TIMING_CSV"
    sleep "$SLEEP_SEC"
}

# NOTE: your LUMI version had commas INSIDE the bash array:
#     Ns=(940000000, 1740000000, ...)
# Bash does not use commas as separators, so five of the six elements were the
# literal strings "940000000," etc. atol() stops at the comma so the binaries
# still got the right N -- but the trailing comma went straight into the CSV,
# producing a phantom 7th field on 5 of every 6 rows. Commas removed here.
Ns=(940000000 1740000000 2550000000 3350000000 4160000000 4960000000)

echo ""
echo "Starting benchmarks"

for N in "${Ns[@]}"; do
    echo ""
    echo "==========================="
    echo "N = $N"
    echo "==========================="
    run_bench "omp_off" "v1" "$N" version-1-off
    run_bench "omp_off" "v2" "$N" version-2-off
    run_bench "sycl"    "v1" "$N" version-1-sycl
    run_bench "sycl"    "v2" "$N" version-2-sycl
done

echo ""
echo "All done."
echo "  Timing CSV : $TIMING_CSV"
echo "  Run log    : $RUN_LOG"
echo "  Environment: $ENV_TXT"
[[ -s "$SKIPPED_LOG" ]] && \
    echo "  Skipped    : $SKIPPED_LOG  ($(wc -l < "$SKIPPED_LOG") entries)"

# =============================================================================
# NOTES
#
# HSA_ENABLE_SDMA=0 has no clean Intel equivalent, and you probably do not want
# one. On LUMI it forces D2H copies off the SDMA engines onto blit kernels -- a
# well-known workaround for MI250X SDMA throughput. The nearest Intel knob is
#     SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0
# which controls whether Level Zero copy engines are used for host<->device
# buffer transfers. Two reasons it is commented out above:
#
#   * It is SYCL-only. The OpenMP offload plugin has its own LIBOMPTARGET_*
#     controls. Setting it would change the SYCL numbers and not the OpenMP
#     ones -- silently biasing the exact comparison this benchmark exists to
#     make.
#   * Intel documents these as development/debugging variables whose semantics
#     are subject to change between releases, and explicitly says not to rely on
#     them. Verify against the docs for YOUR oneAPI version before trusting a
#     result taken with it set. (Newer stacks moved to UR_L0_* names.)
#
# If you want the copy-engine sensitivity as a data point, run it as a separate
# labelled sweep rather than folding it into the baseline.
#
# MEMORY: 4.96e9 elements x 8 B = ~39.7 GB per buffer. A PVC 1550 card has 128 GB
# (64 GB per stack). Under COMPOSITE that fits comfortably; under FLAT a device
# is a 64 GB stack, so if the code allocates more than one buffer of size N the
# top one or two sizes may OOM. That will show up as status=runtime_error rather
# than a wrong number, which is the good failure mode.
# =============================================================================
