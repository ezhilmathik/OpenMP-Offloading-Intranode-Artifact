#!/bin/bash -l
#SBATCH --job-name=heat-verify-hip
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465003145
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=verify-%j.out
#
# =============================================================================
#  verify.sh (HIP tree) -- CORRECTNESS ONLY. Counterpart to run.sh.
# =============================================================================
#
#   run.sh     WRITE_TXT=0, no CPU reference, timings are meaningful
#   verify.sh  WRITE_TXT=1, CPU reference, timings are MEANINGLESS (file I/O
#              sits inside the measured region). Never plot numbers from here.
#
#  INVARIANT: every variant must execute under the SAME runtime environment
#  here as it does in run.sh. WRITE_TXT is the only deliberate difference.
#  variant_env() below is a verbatim copy of the case block in run.sh -- keep
#  them in sync, or factor both into a shared env.sh.
#
#  Unlike the OpenMP tree, HIP variants use the stock runtime: no patched
#  OpenMP-target runtime, no P2P shim, no LD_PRELOAD. The -stream-omp and
#  -stream-omp-p2p variants still use host OpenMP threads to drive the
#  per-device streams, so the OMP_* settings below are still load-bearing.
#
#  The pass criterion is the same as compare.py's, so the two tools cannot
#  return opposite verdicts on the same pair of files.
#
#  Submit directly -- it is not part of the submit.sh pipeline:
#    sbatch verify.sh                          # defaults: N=256, 100 steps
#    sbatch --export=ALL,N=128,STEPS=50 verify.sh
#    sbatch --export=ALL,TOL=1e-9,RTOL=1e-9 verify.sh    # tighten
#
#  Writing solutions costs O(N^3) lines. Keep N small.
# =============================================================================

# Deliberately no `set -e`: one failing variant must not kill the sweep.
set -Euo pipefail

module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4
module load lumi-CrayPath

WORK_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORK_DIR"

# Clean inherited runtime settings. Each process receives an explicit, minimal
# environment in variant_env().
unset LD_PRELOAD
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset OMP_TARGET_OFFLOAD
unset HSA_ENABLE_SDMA
unset HSA_ENABLE_PEER_SDMA
unset OMP_NUM_THREADS
unset OMP_THREAD_LIMIT
unset OMP_NUM_HIDDEN_HELPER_THREADS

export WRITE_TXT=1                     # <-- the whole point of this script

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N="${N:-256}"
STEPS="${STEPS:-100}"
KAPPA_MAX="${KAPPA_MAX:-0.95}"
FAMILIES="${FAMILIES:-2,4}"
FAMILIES="${FAMILIES//,/ }"
FAMILIES="${FAMILIES//+/ }"
BASELINE="${BASELINE:-1}"

# Pass criterion:  |ref - gpu| <= TOL + RTOL*|ref|
#
# These are numpy.allclose's atol/rtol, and they are deliberately the same
# numbers compare.py defaults to. A bit-exact match is not achievable here:
# the variants decompose the domain differently, so ~100 timesteps of FMA
# contraction and differing summation order put the honest agreement floor
# several orders above the 1e-10 this script used to demand. That threshold
# failed correct code.
#
# A real defect -- wrong halo exchange, a race, a missing sync -- produces
# errors far larger than these bounds, so sensitivity is retained.
TOL="${TOL:-1e-6}"                     # absolute tolerance (compare.py --atol)
RTOL="${RTOL:-1e-5}"                   # relative tolerance (compare.py --rtol)

TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
SERIAL_TARGET="${SERIAL_TARGET:-serial-ref}"

# Keep this dt derivation identical to the one in submit.sh, otherwise you are
# verifying a different stability regime than you benchmark.
DT=$(awk -v N="$N" -v k="$KAPPA_MAX" -v s="$STEPS" \
      'BEGIN { h = 1.0/(N-1); dt = h*h/(12.0*k); printf "%.7g", s*dt }')

# ---------------------------------------------------------------------------
# Variant table.
#
#   VARIANT_SUFFIXES -> binary is "${fam}-${suffix}"
#   VARIANT_OUTNAMES -> expected file is "u_${outname}_${fam}gpu.txt"
#
# These match the write_u_values() literals in *-hip*.cpp. If you rename them
# again, update this table too -- a variant whose file is not found counts as
# a FAILURE, not a skip, and the script lists the u_*.txt files that did
# appear so the mismatch is obvious:
#   grep -n 'u_.*\.txt' *-hip*.cpp
# ---------------------------------------------------------------------------
VARIANT_SUFFIXES=(hip hip-stream hip-stream-omp hip-stream-omp-p2p)
VARIANT_OUTNAMES=(hip hip_stream hip_stream_omp hip_stream_omp_p2p)

BASELINE_BIN="1-hip"
BASELINE_OUT="u_hip_1gpu.txt"

REF="u_serial.txt"

# ---------------------------------------------------------------------------
valid_family() { [[ "$1" == "2" || "$1" == "4" ]]; }

for fam in $FAMILIES; do
    if ! valid_family "$fam"; then
        echo "ERROR: FAMILIES accepts only 2 and/or 4; got '$fam'."
        exit 1
    fi
done

echo "============================================================"
echo "HIP correctness check   N=${N}  STEPS=${STEPS}  dt=${DT}"
echo "Pass criterion: |ref - gpu| <= TOL + RTOL*|ref|"
echo "  TOL  = ${TOL}"
echo "  RTOL = ${RTOL}"
echo "Families     : ${FAMILIES}   baseline: ${BASELINE}"
echo "WRITE_TXT    : ${WRITE_TXT}"
echo "============================================================"

rocm-smi --showproductname || rocm-smi || true

trap 'echo "ERROR at line $LINENO"' ERR

# One GCD from each physical MI250X card:
#   1 GPU  -> 0
#   2 GPUs -> 0,2
#   4 GPUs -> 0,2,4,6
mask_for() {
    local n="$1" values=() i
    for ((i = 0; i < n; i++)); do values+=("$((2 * i))"); done
    (IFS=,; echo "${values[*]}")
}

# ---------------------------------------------------------------------------
# Per-variant runtime environment -- VERBATIM COPY of the case block in run.sh,
# with WRITE_TXT flipped to 1. Populates the global VARIANT_ENV array.
# ---------------------------------------------------------------------------
VARIANT_ENV=()
variant_env() {
    local binary="$1" gpus="$2" mask="$3"

    VARIANT_ENV=(
        env
        -u LD_PRELOAD
        -u HIP_VISIBLE_DEVICES
        -u CUDA_VISIBLE_DEVICES
        -u OMP_TARGET_OFFLOAD
        -u HSA_ENABLE_SDMA
        -u HSA_ENABLE_PEER_SDMA
        -u OMP_NUM_THREADS
        -u OMP_THREAD_LIMIT
        -u OMP_NUM_HIDDEN_HELPER_THREADS
        "ROCR_VISIBLE_DEVICES=${mask}"
        "WRITE_TXT=1"
        "HSA_ENABLE_SDMA=1"
    )

    case "$binary" in
        *-stream-omp-p2p)
            VARIANT_ENV+=(
                "OMP_NUM_THREADS=${gpus}"
                "OMP_THREAD_LIMIT=${gpus}"
                "OMP_NUM_HIDDEN_HELPER_THREADS=0"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=0"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=0"
                "HSA_ENABLE_SDMA=1"
                "HSA_ENABLE_PEER_SDMA=1"
            )
            ;;

        *-stream-omp)
            VARIANT_ENV+=(
                "OMP_NUM_THREADS=${gpus}"
                "OMP_THREAD_LIMIT=${gpus}"
                "OMP_NUM_HIDDEN_HELPER_THREADS=0"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=0"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=0"
                "HSA_ENABLE_SDMA=1"
            )
            ;;

        *-omp-stream)
            VARIANT_ENV+=(
                "OMP_NUM_THREADS=1"
                "OMP_NUM_HIDDEN_HELPER_THREADS=0"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=1"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"
                "HSA_ENABLE_SDMA=1"
            )
            ;;

        *)
            VARIANT_ENV+=(
                "OMP_NUM_THREADS=1"
                "OMP_NUM_HIDDEN_HELPER_THREADS=0"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=1"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"
                "HSA_ENABLE_SDMA=1"
            )
            ;;
    esac
}

# ---------------------------------------------------------------------------
# max |a-b| and max relative diff over the two solution files, plus the
# location of the worst cell and a count of cells outside tolerance.
#
# NaN/Inf are counted as violations explicitly: every awk comparison against
# NaN is false, so without this guard a file full of NaN reports zero cells
# over tolerance and PASSES.
# ---------------------------------------------------------------------------
compare() {
    awk -v tol="$TOL" -v rtol="$RTOL" '
        function nonfinite(s) { return (s ~ /[nN][aA][nN]/ || s ~ /[iI][nN][fF]/) }

        NR == FNR {
            if (nonfinite($NF)) refbad++
            a[FNR] = $NF + 0
            next
        }
        {
            if (nonfinite($NF)) {
                bad++; nf++
                if (badline == 0) badline = FNR
                next
            }

            r = a[FNR]; c = $NF + 0
            d = r - c; if (d < 0) d = -d
            ar = (r < 0 ? -r : r)

            if (d > wabs) { wabs = d; iabs = FNR }
            rel = (ar > 0 ? d / ar : 0)
            if (rel > wrel) { wrel = rel; irel = FNR }

            if (d > tol + rtol * ar) {
                bad++
                if (badline == 0) badline = FNR
            }
        }
        END {
            if (FNR != length(a)) {
                printf "LINE COUNT MISMATCH (ref %d vs gpu %d)\n", length(a), FNR
                exit 2
            }
            if (refbad > 0) {
                printf "REFERENCE HAS %d NON-FINITE VALUES -- the serial run is ", refbad
                printf "broken, not the GPU variant.\n"
                exit 2
            }
            printf "max|diff| = %.6e (line %d)   max rel = %.6e (line %d)\n",
                   wabs + 0, iabs + 0, wrel + 0, irel + 0
            printf "cells over tolerance: %d", bad + 0
            if (nf > 0)  printf "   (%d NaN/Inf)", nf
            if (bad > 0) printf "   first at line %d", badline
            printf "   -> %s\n", (bad > 0 ? "FAIL" : "PASS")
            exit (bad > 0 ? 1 : 0)
        }' "$1" "$2"
}

declare -a PASSED=() FAILED=() SKIPPED=()

# ---------------------------------------------------------------------------
# CPU reference (heat3D_variable_coeff.c). Not built by build.sh.
# ---------------------------------------------------------------------------
if [[ ! -x ./serial ]]; then
    echo
    echo "Building CPU reference (make ${SERIAL_TARGET})..."
    make "$SERIAL_TARGET" || {
        echo "ERROR: could not build serial reference."
        echo "       Check the target name in the Makefile and pass"
        echo "       --export=ALL,SERIAL_TARGET=<target> if it differs."
        exit 1
    }
fi

echo
echo ">>> CPU reference: ./serial ${N} ${DT}"
if ! timeout "$TIMEOUT_SEC" ./serial "$N" "$DT"; then
    echo "ERROR: serial reference failed or timed out"
    exit 1
fi
[[ -f "$REF" ]] || { echo "ERROR: ${REF} was not produced"; exit 1; }
echo "    reference: ${REF}  ($(wc -l < "$REF") lines)"

check_variant() {
    local binary="$1" gpus="$2" out="$3"
    local mask; mask=$(mask_for "$gpus")

    if [[ ! -x "./$binary" ]]; then
        echo
        echo ">>> ${binary}"
        echo "    SKIP (missing binary)"
        SKIPPED+=("$binary (missing binary)")
        return 0
    fi

    echo
    echo ">>> ${binary}  gpus=${gpus}  mask=${mask}"
    rm -f "$out"

    variant_env "$binary" "$gpus" "$mask"
    local cmd=("${VARIANT_ENV[@]}" "./$binary" "$N" "$DT")

    echo "    Runtime configuration:"
    local item
    for item in "${cmd[@]}"; do
        case "$item" in
            ROCR_VISIBLE_DEVICES=*|WRITE_TXT=*|OMP_NUM_THREADS=*|OMP_THREAD_LIMIT=*|\
            OMP_NUM_HIDDEN_HELPER_THREADS=*|LIBOMP_USE_HIDDEN_HELPER_TASK=*|\
            LIBOMP_NUM_HIDDEN_HELPER_THREADS=*|HSA_ENABLE_SDMA=*|HSA_ENABLE_PEER_SDMA=*)
                echo "      $item"
                ;;
        esac
    done

    printf '    command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    local exit_code=0
    timeout "$TIMEOUT_SEC" "${cmd[@]}" || exit_code=$?

    if [[ $exit_code -eq 124 ]]; then
        echo "    RUN TIMED OUT after ${TIMEOUT_SEC}s"
        FAILED+=("$binary (timeout)")
        return 0
    elif [[ $exit_code -ne 0 ]]; then
        echo "    RUN FAILED (exit ${exit_code})"
        FAILED+=("$binary (run failed)")
        return 0
    fi

    if [[ ! -f "$out" ]]; then
        echo "    ERROR: expected ${out}, not produced."
        echo "    Solution files currently present (fix VARIANT_OUTNAMES if the"
        echo "    binary writes a different name):"
        ls -1 u_*.txt 2>/dev/null | sed 's/^/      /' || echo "      (none)"
        FAILED+=("$binary (no output)")
        return 0
    fi

    local cmp_status=0
    compare "$REF" "$out" | sed 's/^/    /'
    cmp_status=${PIPESTATUS[0]}

    case "$cmp_status" in
        0) PASSED+=("$binary") ;;
        2) FAILED+=("$binary (malformed output)") ;;
        *) FAILED+=("$binary") ;;
    esac
}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
if [[ "$BASELINE" == "1" ]]; then
    check_variant "$BASELINE_BIN" 1 "$BASELINE_OUT"
fi

for fam in $FAMILIES; do
    for i in "${!VARIANT_SUFFIXES[@]}"; do
        check_variant \
            "${fam}-${VARIANT_SUFFIXES[$i]}" \
            "$fam" \
            "u_${VARIANT_OUTNAMES[$i]}_${fam}gpu.txt"
    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "PASSED  (${#PASSED[@]}): ${PASSED[*]:-none}"
echo "FAILED  (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "SKIPPED (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
echo "============================================================"
echo
echo "Solution .txt files kept for manual inspection. Remove with:"
echo "  ./clean.sh -f out"
echo "Cross-check with the Python tool (same criterion, same defaults):"
echo "  python3 compare.py --backend hip"

# A run in which nothing executed is not a pass.
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "RESULT: FAIL"
    exit 1
elif [[ ${#PASSED[@]} -eq 0 ]]; then
    echo "RESULT: INCONCLUSIVE -- no variant ran."
    exit 2
else
    echo "RESULT: PASS"
    exit 0
fi
