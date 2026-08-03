#!/bin/bash -l
#SBATCH --job-name=p2p-run
#SBATCH --account=pn67qe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=run-%j.out

# =============================================================================
# Intel PVC P2P bandwidth sweep.
#
# CHANGES FROM THE PREVIOUS run.sh -- read these, they affect the numbers:
#
#  1. LIBOMPTARGET_DEVICES is now set explicitly.  THIS IS THE IMPORTANT ONE.
#     ZE_FLAT_DEVICE_HIERARCHY / ZE_AFFINITY_MASK configure the Level Zero
#     driver, which is what SYCL sees.  OpenMP has its OWN mapping control --
#     LIBOMPTARGET_DEVICES=[device|subdevice|subsubdevice] -- which maps an
#     OpenMP "device" to a card / stack / CCS and does not apply to SYCL.
#     Leaving it unset meant SYCL's device model was pinned and OpenMP's was
#     left to a driver default that Intel documents as having changed (older
#     drivers defaulted to COMPOSITE + implicit scaling; newer default to
#     FLAT).  If OpenMP resolved to stacks while SYCL used cards, then
#     "omp_off v1" and "sycl v1" were copying over different links and the
#     comparison was invalid.  Suspected cause of omp_off=31.7 vs sycl=76.1.
#
#  2. Toolkit version PINNED.  Was `module load intel-toolkit` (floating).
#     Since the defaults above shift between versions, an unpinned module made
#     runs non-reproducible in exactly the dimension under investigation.
#
#  3. PREFLIGHT CHECK added.  Builds two tiny probes and refuses to run the
#     sweep unless OpenMP and SYCL agree on the device count.  This is the
#     4-vs-8 question; the sweep is meaningless if they disagree.
#
#  4. OMP_NUM_THREADS set explicitly.  Was unset with --exclusive and no
#     --cpus-per-task, i.e. whatever the runtime felt like.
#
#  5. Diagnostics (LIBOMPTARGET_PLUGIN_PROFILE / _INFO) available via DIAG=1.
#
# Usage:  sbatch run.sh [HIER] [DIAG]
#           HIER = COMPOSITE (default) | FLAT
#           DIAG = 0 (default) | 1     -> enable libomptarget profiling/info
# =============================================================================

HIER="${1:-COMPOSITE}"
DIAG="${2:-0}"
if [[ "$HIER" != "COMPOSITE" && "$HIER" != "FLAT" ]]; then
  echo "ERROR: HIER must be COMPOSITE or FLAT (got: '$HIER')"; exit 1
fi

module load slurm_setup
module load intel-toolkit/2024.1.0        # PINNED -- see note 2 above

# --- Offload / Level Zero environment ---------------------------------------
export OMP_TARGET_OFFLOAD=mandatory       # fail loudly instead of running on host
export ZE_FLAT_DEVICE_HIERARCHY="$HIER"
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1    # stable PCI-ordered device numbering
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu   # hide the duplicate OpenCL GPUs
export OMP_NUM_THREADS=8                  # enough host threads for v4's sections

# One device per physical card (the ROCR_VISIBLE_DEVICES=0,2,4,6 analogue).
#   COMPOSITE: index = card  -> "0,1,2,3" = 4 whole cards (1 full link each)
#   FLAT     : index = stack -> "0,2,4,6" = 1 stack from each of 4 cards
#
# LIBOMPTARGET_DEVICES must AGREE with the hierarchy, or OpenMP and SYCL will
# be benchmarking different hardware under the same label:
#   COMPOSITE -> device    (OpenMP device = whole card, matches SYCL)
#   FLAT      -> subdevice (OpenMP device = stack,      matches SYCL)
if [[ "$HIER" == "COMPOSITE" ]]; then
  export ZE_AFFINITY_MASK="0,1,2,3"
  export LIBOMPTARGET_DEVICES=device
else
  export ZE_AFFINITY_MASK="0,2,4,6"
  export LIBOMPTARGET_DEVICES=subdevice
fi

# Persistent JIT cache: without AOT, the FIRST SYCL run of each binary pays for
# SPIR-V -> PVC codegen. The binaries print their own timing so the timed region
# is unaffected, but this keeps the wall-clock fallback honest.
export SYCL_CACHE_PERSISTENT=1

# Diagnostics. Off by default -- LIBOMPTARGET_DEBUG in particular is very
# verbose and its output would swamp the CSV parsing in run_bench.
if [[ "$DIAG" == "1" ]]; then
  export LIBOMPTARGET_PLUGIN_PROFILE=T    # per-engine copy/kernel profile
  export LIBOMPTARGET_INFO=1              # data-transfer decisions
  echo "DIAG enabled: libomptarget profiling on. Do NOT use this run's CSV for"
  echo "              reported numbers -- profiling perturbs timings."
fi

# NOT SET, deliberately -- the HSA_ENABLE_SDMA analogue. See notes at bottom.
# export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS_DIR="results_${HIER}_${JOB_ID}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

TIMING_CSV="${RESULTS_DIR}/timings.csv"
RUN_LOG="${RESULTS_DIR}/run.log"
SKIPPED_LOG="${RESULTS_DIR}/skipped.log"
ENV_TXT="${RESULTS_DIR}/env.txt"
PROBE_TXT="${RESULTS_DIR}/probe.txt"

# 'hierarchy' is a 7th column: on PVC the same "4 devices" is different silicon
# under FLAT vs COMPOSITE, so the number is meaningless without it.
echo "framework,version,N,elapsed_sec,bandwidth_gbs,status,hierarchy" > "$TIMING_CSV"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "Job ID     : ${JOB_ID}"
echo "Timestamp  : ${TIMESTAMP}"
echo "Hierarchy  : ${HIER}"
echo "  ZE_AFFINITY_MASK     = ${ZE_AFFINITY_MASK}"
echo "  LIBOMPTARGET_DEVICES = ${LIBOMPTARGET_DEVICES}"
echo "Results dir: ${RESULTS_DIR}"

{
  echo "host=$(hostname)"
  echo "date=$(date -Is)"
  echo "jobid=${JOB_ID}"
  echo "hierarchy=${ZE_FLAT_DEVICE_HIERARCHY}"
  echo "ze_affinity_mask=${ZE_AFFINITY_MASK}"
  echo "libomptarget_devices=${LIBOMPTARGET_DEVICES}"
  echo "omp_target_offload=${OMP_TARGET_OFFLOAD}"
  echo "omp_num_threads=${OMP_NUM_THREADS}"
  echo "compiler=$(icpx --version 2>/dev/null | head -1)"
  echo "--- sycl-ls ---"
  sycl-ls 2>/dev/null
} > "$ENV_TXT"

# =============================================================================
# PREFLIGHT: do OpenMP and SYCL agree on what a device is?
#
# This is the 4-vs-8 check. If OpenMP reports 8 devices while SYCL reports 4,
# OpenMP is treating stacks as devices and every "N GPUs" label in the CSV
# means something different for the two frameworks. Abort rather than produce
# a plot nobody can defend.
# =============================================================================
echo ""
echo "=== Preflight: device model agreement ==="

PROBE_DIR=$(mktemp -d)
trap 'rm -rf "$PROBE_DIR"' EXIT

cat > "$PROBE_DIR/probe_omp.c" << 'PROBE_OMP'
#include <omp.h>
#include <stdio.h>
int main(void) {
    printf("%d\n", omp_get_num_devices());
    return 0;
}
PROBE_OMP

cat > "$PROBE_DIR/probe_sycl.cpp" << 'PROBE_SYCL'
#include <sycl/sycl.hpp>
#include <cstdio>
int main() {
    auto d = sycl::device::get_devices(sycl::info::device_type::gpu);
    printf("%zu\n", d.size());
    return 0;
}
PROBE_SYCL

icx -fiopenmp -fopenmp-targets=spir64 "$PROBE_DIR/probe_omp.c" \
    -o "$PROBE_DIR/probe_omp" 2>/dev/null \
    || { echo "ERROR: could not build the OpenMP probe"; exit 1; }
icpx -fsycl "$PROBE_DIR/probe_sycl.cpp" \
    -o "$PROBE_DIR/probe_sycl" 2>/dev/null \
    || { echo "ERROR: could not build the SYCL probe"; exit 1; }

OMP_DEVS=$("$PROBE_DIR/probe_omp")
SYCL_DEVS=$("$PROBE_DIR/probe_sycl")

{
  echo "omp_get_num_devices()  = ${OMP_DEVS}"
  echo "sycl gpu device count  = ${SYCL_DEVS}"
} | tee "$PROBE_TXT"

if [[ "$OMP_DEVS" != "$SYCL_DEVS" ]]; then
    echo ""
    echo "ABORT: OpenMP sees ${OMP_DEVS} devices, SYCL sees ${SYCL_DEVS}."
    echo "       The two frameworks disagree on what a device is, so any"
    echo "       omp_off-vs-sycl comparison from this run would be invalid."
    echo "       Check LIBOMPTARGET_DEVICES (currently '${LIBOMPTARGET_DEVICES}')"
    echo "       against ZE_FLAT_DEVICE_HIERARCHY ('${HIER}')."
    exit 1
fi

EXPECTED=$(awk -F',' '{print NF}' <<< "${ZE_AFFINITY_MASK}")
if [[ "$OMP_DEVS" != "$EXPECTED" ]]; then
    echo ""
    echo "ABORT: expected ${EXPECTED} devices from ZE_AFFINITY_MASK="
    echo "       '${ZE_AFFINITY_MASK}', but both runtimes report ${OMP_DEVS}."
    echo "       The affinity mask is not taking effect as intended."
    exit 1
fi

echo "OK: both runtimes see ${OMP_DEVS} devices, matching ZE_AFFINITY_MASK."

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

    # A verification FAILED line means the transfer did not do what it claims,
    # so the bandwidth is measuring the wrong thing. Do not let it into the CSV
    # as "ok".
    if grep -qi 'FAILED' "$stdout_file"; then
        status="verify_failed"
    fi

    rm -f "$stdout_file"

    echo "    -> elapsed=${elapsed}s  bandwidth=${bw} GB/s  status=${status}"
    echo "${framework},${version},${N},${elapsed},${bw},${status},${HIER}" >> "$TIMING_CSV"
    sleep "$SLEEP_SEC"
}

# Bash does not treat commas as separators -- keep this array comma-free.
#Ns=(940000000 1740000000 2550000000 3350000000 4160000000 4960000000)
Ns=(117500000 217500000 318750000 418750000 520000000 1240000000)

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
echo "  Probe      : $PROBE_TXT"
[[ -s "$SKIPPED_LOG" ]] && \
    echo "  Skipped    : $SKIPPED_LOG  ($(wc -l < "$SKIPPED_LOG") entries)"

# =============================================================================
# NOTES
#
# STILL UNRESOLVED -- do not treat this script as a clean bill of health:
#
#   * omp_off v4 is SERIALIZED. Elapsed is exactly 3x v1's (0.711128/0.237350
#     = 2.996), so the three transfers run back-to-back and v4 reports the same
#     31.7 GB/s as v1. v2's two sections DO overlap under the same environment,
#     so this is a bug in openmp-4.cc, not an environment variable. No setting
#     in this script will fix it.
#
#   * The LD_PRELOAD of omptarget-p2p was dropped when porting from LUMI on the
#     assumption that Intel's Level Zero exposes P2P natively to libomptarget.
#     That assumption was never verified. If the preflight passes (OpenMP and
#     SYCL agree on device count) and omp_off is STILL ~31.7 against sycl's
#     ~76.1, then the device mapping was not the cause and this assumption is
#     the next suspect. Run with DIAG=1 and read the copy-engine profile.
#
# HSA_ENABLE_SDMA=0 has no clean Intel equivalent, and you probably do not want
# one. The nearest knob is SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0, left unset:
#
#   * It is SYCL-only. The OpenMP plugin has its own LIBOMPTARGET_* controls.
#     Setting it would change the SYCL numbers and not the OpenMP ones --
#     silently biasing the exact comparison this benchmark exists to make.
#   * Intel documents these as development/debugging variables whose semantics
#     may change between releases. (Newer stacks moved to UR_L0_* names.)
#
# If you want copy-engine sensitivity as a data point, run it as a separate
# labelled sweep rather than folding it into the baseline.
#
# MEMORY: 4.96e9 elements x 8 B = ~39.7 GB per buffer. A PVC 1550 card has
# 128 GB (64 GB per stack). Under COMPOSITE that fits; under FLAT a device is a
# 64 GB stack, so the top sizes may OOM if the code allocates several buffers
# of size N. That shows up as status=runtime_error rather than a wrong number,
# which is the good failure mode.
# =============================================================================
