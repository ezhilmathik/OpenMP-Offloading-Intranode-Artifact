#!/bin/bash
#SBATCH --job-name=heat-verify
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --account=project_465003145
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=verify-%j.out
#
# =============================================================================
#  verify.sh -- CORRECTNESS ONLY. Counterpart to run.sh.
# =============================================================================
#
#   run.sh     WRITE_TXT=0, no CPU reference, timings are meaningful
#   verify.sh  WRITE_TXT=1, CPU reference, timings are MEANINGLESS (file I/O
#              sits inside the measured region). Never plot numbers from here.
#
#  INVARIANT: every variant must execute under the SAME runtime environment
#  here as it does in run.sh. WRITE_TXT is the only deliberate difference.
#  If verify.sh runs a variant with different thread counts, a different copy
#  engine, or a different LD_PRELOAD than run.sh does, then it is not
#  validating the code path you benchmark. The variant_env() block below is a
#  verbatim copy of the case block in run.sh -- keep them in sync (or factor
#  both into a shared env.sh).
#
#  Matches the run.sh in which:
#    - OMP_THREAD_LIMIT is never set (thread-limit-var is inherited by the
#      initial task on the DEVICE and throttles the target teams)
#    - OMP_NUM_HIDDEN_HELPER_THREADS is never set (it contradicted the
#      LIBOMP_NUM_HIDDEN_HELPER_THREADS request)
#    - every variant gets HELPERS hidden-helper threads, default 8
#
#  The *-stream-omp sources must have had their getenv() guards relaxed
#  (relax_hidden_helper_guard.py) or they will exit immediately. A
#  "WARNING: running without OMP_NUM_HIDDEN_HELPER_THREADS=0" line is expected
#  and harmless.
#
#  Submit directly -- it is not part of the submit.sh pipeline:
#    sbatch verify.sh                          # defaults: N=256, 100 steps
#    sbatch --export=ALL,N=128,STEPS=50 verify.sh
#    sbatch --export=ALL,TOL=1e-9,RTOL=1e-9 verify.sh    # tighten
#    sbatch --export=ALL,HELPERS=16 verify.sh
#
#  Writing solutions costs O(N^3) lines. Keep N small.
# =============================================================================

# Deliberately NOT -e: one failing variant must not kill the sweep.
set -Euo pipefail

module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4
module load lumi-CrayPath

# Slurm executes a copied script from its spool directory. Use the original
# submission directory, where the binaries and shim are located.
WORK_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORK_DIR"

# ---------------------------------------------------------------------------
# Environment common to every binary (mirrors run.sh).
# Transfer/runtime settings are applied per binary in variant_env().
# ---------------------------------------------------------------------------
unset LD_PRELOAD
unset HSA_ENABLE_SDMA
unset HSA_ENABLE_PEER_SDMA
unset OMP_NUM_HIDDEN_HELPER_THREADS
unset OMP_THREAD_LIMIT

export OMP_TARGET_OFFLOAD=mandatory
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export WRITE_TXT=1                     # <-- the whole point of this script

OMPTARGET_P2P_LIB="${OMPTARGET_P2P_LIB:-/scratch/project_465003145/omptarget-p2p/lib}"
P2P_SHIM="${P2P_SHIM:-${WORK_DIR}/libp2p_copy_shim.so}"
P2P_PRELOAD="${P2P_SHIM}:${OMPTARGET_P2P_LIB}/libomp.so:${OMPTARGET_P2P_LIB}/libomptarget.so.19.0git"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N="${N:-256}"
STEPS="${STEPS:-100}"
KAPPA_MAX="${KAPPA_MAX:-0.95}"
FAMILIES="${FAMILIES:-2 4}"
BASELINE="${BASELINE:-1}"              # verify the 1-GCD path by default
HELPERS="${HELPERS:-8}"                # must match run.sh

[[ "$HELPERS" =~ ^[0-9]+$ ]] || {
    echo "ERROR: HELPERS must be a non-negative integer, got '$HELPERS'." >&2; exit 1; }

# Pass criterion:  |ref - gpu| <= TOL + RTOL*|ref|
#
# These are numpy.allclose's atol/rtol, and deliberately the same numbers
# compare.py defaults to, so the two tools cannot disagree on the same files.
# A bit-exact match is not achievable: the variants decompose the domain
# differently, so accumulated FMA contraction and differing summation order
# put the agreement floor well above the 1e-10 this script used to demand.
# A real defect -- wrong halo exchange, a race, a missing sync -- produces
# errors far larger than these bounds.
TOL="${TOL:-1e-6}"                     # absolute tolerance (compare.py --atol)
RTOL="${RTOL:-1e-5}"                   # relative tolerance (compare.py --rtol)

TIMEOUT_SEC="${TIMEOUT_SEC:-900}"

# Keep this dt derivation identical to the one in submit.sh, otherwise you are
# verifying a different stability regime than you benchmark.
DT=$(awk -v N="$N" -v k="$KAPPA_MAX" -v s="$STEPS" \
      'BEGIN { h = 1.0/(N-1); dt = h*h/(12.0*k); printf "%.7g", s*dt }')

# ---------------------------------------------------------------------------
# Variant table. Swap these two arrays when porting to the HIP/CUDA tree --
# check the filenames the sources actually write before trusting them.
# ---------------------------------------------------------------------------
VARIANT_SUFFIXES=(omp omp-stream omp-stream-omp omp-stream-omp-p2p)
VARIANT_OUTNAMES=(openmp openmp_stream openmp_stream_omp openmp_stream_omp_p2p)

BASELINE_BIN="1-omp"
BASELINE_OUT="u_openmp_1gpu.txt"

REF="u_serial.txt"

echo "============================================================"
echo "Correctness check   N=${N}  STEPS=${STEPS}  dt=${DT}"
echo "Pass criterion: |ref - gpu| <= TOL + RTOL*|ref|"
echo "  TOL  = ${TOL}"
echo "  RTOL = ${RTOL}"
echo "Families     : ${FAMILIES}   baseline: ${BASELINE}"
echo "Hidden helper threads : ${HELPERS}"
echo "P2P runtime  : ${OMPTARGET_P2P_LIB}"
echo "P2P shim     : ${P2P_SHIM}"
echo "============================================================"

trap 'echo "ERROR at line $LINENO"' ERR

mask_for() {
    local n="$1" v=() i
    for ((i = 0; i < n; i++)); do v+=("$((2 * i))"); done
    (IFS=,; echo "${v[*]}")
}

check_p2p_files() {
    local missing=0 file
    for file in \
        "$P2P_SHIM" \
        "$OMPTARGET_P2P_LIB/libomp.so" \
        "$OMPTARGET_P2P_LIB/libomptarget.so.19.0git"
    do
        if [[ ! -r "$file" ]]; then
            echo "    P2P requirement missing: $file"
            missing=1
        fi
    done
    [[ $missing -eq 0 ]]
}

# ---------------------------------------------------------------------------
# Per-variant runtime environment -- VERBATIM COPY of the case block in run.sh,
# with WRITE_TXT flipped to 1. Populates the global VARIANT_ENV array.
#
# OMP_THREAD_LIMIT and OMP_NUM_HIDDEN_HELPER_THREADS are stripped and never
# re-added, exactly as in run.sh.
# ---------------------------------------------------------------------------
VARIANT_ENV=()
variant_env() {
    local binary="$1" gpus="$2" mask="$3"

    VARIANT_ENV=(
        env
        -u LD_PRELOAD
        -u HSA_ENABLE_SDMA
        -u HSA_ENABLE_PEER_SDMA
        -u OMP_NUM_THREADS
        -u OMP_THREAD_LIMIT
        -u OMP_NUM_HIDDEN_HELPER_THREADS
        -u LIBOMP_USE_HIDDEN_HELPER_TASK
        -u LIBOMP_NUM_HIDDEN_HELPER_THREADS
        "ROCR_VISIBLE_DEVICES=${mask}"
        "WRITE_TXT=1"
    )

    case "$binary" in
        *-stream-omp-p2p)
            # Version 4: one explicit host thread per GCD, plus runtime helpers.
            VARIANT_ENV+=(
                "OMP_NUM_THREADS=${gpus}"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=1"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"
                "HSA_ENABLE_SDMA=1"
                "HSA_ENABLE_PEER_SDMA=1"
                "LD_PRELOAD=${P2P_PRELOAD}"
            )
            ;;

        *-stream-omp)
            # Version 3: one explicit host thread per GCD, plus runtime helpers.
            VARIANT_ENV+=(
                "OMP_NUM_THREADS=${gpus}"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=1"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"
                "HSA_ENABLE_SDMA=1"
            )
            ;;

        *-omp-stream)
            # Version 2: one application host thread; runtime helpers progress
            # the asynchronous transfers and kernels.
            VARIANT_ENV+=(
                "OMP_NUM_THREADS=1"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=1"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"
                "HSA_ENABLE_SDMA=1"
            )
            ;;

        *)
            # Version 1 and the 1-GCD baseline.
            VARIANT_ENV+=(
                "OMP_NUM_THREADS=1"
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
# CPU reference. Not built by build.sh -- build it here if absent.
# ---------------------------------------------------------------------------
if [[ ! -x ./serial ]]; then
    echo ""
    echo "Building CPU reference..."
    make serial-ref || { echo "ERROR: could not build serial reference"; exit 1; }
fi

echo ""
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
        echo ""
        echo ">>> ${binary}"
        echo "    SKIP (missing binary)"
        SKIPPED+=("$binary (missing binary)")
        return 0
    fi

    if [[ "$binary" == *-stream-omp-p2p ]] && ! check_p2p_files; then
        echo ""
        echo ">>> ${binary}"
        echo "    SKIP (missing P2P runtime)"
        SKIPPED+=("$binary (missing p2p runtime)")
        return 0
    fi

    echo ""
    echo ">>> ${binary}  gpus=${gpus}  mask=${mask}"
    rm -f "$out"

    variant_env "$binary" "$gpus" "$mask"
    local cmd=("${VARIANT_ENV[@]}" "./$binary" "$N" "$DT")

    echo "    Runtime configuration:"
    local item
    for item in "${cmd[@]}"; do
        case "$item" in
            ROCR_VISIBLE_DEVICES=*|\
            OMP_NUM_THREADS=*|\
            OMP_THREAD_LIMIT=*|\
            OMP_NUM_HIDDEN_HELPER_THREADS=*|\
            LIBOMP_USE_HIDDEN_HELPER_TASK=*|\
            LIBOMP_NUM_HIDDEN_HELPER_THREADS=*|\
            HSA_ENABLE_SDMA=*|\
            HSA_ENABLE_PEER_SDMA=*|\
            WRITE_TXT=*|\
            LD_PRELOAD=*)
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
        echo "    If this is a *-stream-omp variant, check whether the"
        echo "    OMP_NUM_HIDDEN_HELPER_THREADS guard is still fatal:"
        echo "      python3 relax_hidden_helper_guard.py --dry-run *.c"
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
echo ""
echo "============================================================"
echo "PASSED  (${#PASSED[@]}): ${PASSED[*]:-none}"
echo "FAILED  (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "SKIPPED (${#SKIPPED[@]}): ${SKIPPED[*]:-none}"
echo "============================================================"
echo ""
echo "Solution .txt files kept for manual inspection. Remove with:"
echo "  ./clean.sh -f out"
echo "Cross-check with the Python tool (same criterion, same defaults):"
echo "  python3 compare.py --backend openmp"

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
