#!/bin/bash -l
#SBATCH --job-name=heat-verify
#SBATCH --account=pn67qe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --output=verify-%j.out
#
# CORRECTNESS ONLY. Counterpart to run.sh:
#   run.sh     WRITE_TXT=0, timings meaningful
#   verify.sh  WRITE_TXT=1, CPU reference, timings MEANINGLESS
#
# This script mirrors run.sh's Intel whole-card, per-variant helper policy:
#   1-omp:
#       one application host thread, no LLVM hidden helpers
#   2/4-omp and 2/4-omp-stream:
#       one application host thread, one LLVM runtime helper per GPU card
#   2/4-omp-stream-omp and P2P:
#       one explicit application host thread per GPU card, no hidden helpers
#
# Examples:
#   sbatch verify.sh
#   sbatch --export=ALL,N=128,STEPS=50 verify.sh
#
# Writing solutions costs O(N^3) lines. Keep N small.

set -uo pipefail

module load slurm_setup
module load intel-toolkit

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export OMP_TARGET_OFFLOAD=MANDATORY

# Required by the source-code checks in the explicit host-thread variants.
# LLVM runtime-helper controls are selected separately for each binary below.
export OMP_NUM_HIDDEN_HELPER_THREADS=0
unset LIBOMP_USE_HIDDEN_HELPER_TASK
unset LIBOMP_NUM_HIDDEN_HELPER_THREADS
unset OMP_THREAD_LIMIT

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
unset ONEAPI_DEVICE_SELECTOR
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export WRITE_TXT=1

HIER="COMPOSITE"
export ZE_FLAT_DEVICE_HIERARCHY="$HIER"

N="${N:-256}"
STEPS="${STEPS:-100}"
KAPPA_MAX="${KAPPA_MAX:-0.95}"
FAMILIES="${FAMILIES:-2 4}"
TOL="${TOL:-1e-10}"

DT=$(awk -v N="$N" -v k="$KAPPA_MAX" -v s="$STEPS" '
    BEGIN {
        h = 1.0 / (N - 1);
        dt = h * h / (12.0 * k);
        printf "%.17g", s * dt;
    }
')

echo "============================================================"
echo "Correctness check   N=${N}  STEPS=${STEPS}  DT=${DT}"
echo "Tolerance           ${TOL}"
echo "Hierarchy           COMPOSITE"
echo "Device unit         one whole Intel Max 1550 card"
echo "Compiler            $(icx --version | head -1)"
echo "============================================================"
sycl-ls 2>/dev/null || xpu-smi discovery 2>/dev/null || true

if [[ ! -x ./serial ]]; then
    echo "Building CPU reference..."
    make CC=icx serial-ref || {
        echo "ERROR: could not build serial reference"
        exit 1
    }
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    TARGETS=(1-omp)

    for fam in $FAMILIES; do
        case "$fam" in
            2|4) ;;
            *)
                echo "ERROR: unsupported family '$fam'; expected 2 or 4"
                exit 1
                ;;
        esac

        TARGETS+=(
            "${fam}-omp"
            "${fam}-omp-stream"
            "${fam}-omp-stream-omp"
            "${fam}-omp-stream-omp-p2p"
        )
    done

    echo "Building: ${TARGETS[*]}"
    make CC=icx "${TARGETS[@]}" || {
        echo "ERROR: build failed"
        exit 1
    }
    echo
fi

mask_for() {
    local n="$1"
    local values=()
    local i

    for ((i = 0; i < n; i++)); do
        values+=("$i")
    done

    (IFS=,; echo "${values[*]}")
}

print_runtime_config() {
    local item

    echo "    Runtime configuration:"
    for item in "$@"; do
        case "$item" in
            ZE_AFFINITY_MASK=*|\
            OMP_NUM_THREADS=*|\
            OMP_THREAD_LIMIT=*|\
            OMP_NUM_HIDDEN_HELPER_THREADS=*|\
            LIBOMP_USE_HIDDEN_HELPER_TASK=*|\
            LIBOMP_NUM_HIDDEN_HELPER_THREADS=*)
                echo "      $item"
                ;;
        esac
    done
}

REF="u_serial.txt"

echo
echo ">>> CPU reference: ./serial ${N} ${DT}"
rm -f "$REF"

if ! ./serial "$N" "$DT"; then
    echo "ERROR: serial reference failed"
    exit 1
fi

[[ -f "$REF" ]] || {
    echo "ERROR: ${REF} was not produced"
    exit 1
}

echo "    reference: ${REF} ($(wc -l < "$REF") lines)"

compare() {
    awk -v tol="$TOL" '
        NR == FNR {
            a[FNR] = $NF
            next
        }
        {
            d = a[FNR] - $NF
            if (d < 0) d = -d
            if (d > worst) {
                worst = d
                where = FNR
            }
        }
        END {
            if (FNR != length(a)) {
                printf "LINE COUNT MISMATCH (%d vs %d)\n", length(a), FNR
                exit 2
            }

            printf "max|diff| = %.6e at line %d -> %s\n",
                   worst, where, (worst <= tol ? "PASS" : "FAIL")

            exit (worst <= tol ? 0 : 1)
        }
    ' "$1" "$2"
}

declare -a PASSED=()
declare -a FAILED=()

check_variant() {
    local binary="$1"
    local gpus="$2"
    local out="$3"
    local mask

    mask=$(mask_for "$gpus")

    if [[ ! -x "./$binary" ]]; then
        echo "  SKIP (missing): $binary"
        FAILED+=("$binary (missing)")
        return 0
    fi

    echo
    echo ">>> ${binary}  cards=${gpus}  ZE_AFFINITY_MASK=${mask}"
    rm -f "$out"

    local run_cmd

    case "$binary" in
        1-omp)
            # One-card baseline: one application thread, no runtime helpers.
            run_cmd=(
                env
                "ZE_AFFINITY_MASK=${mask}"
                "OMP_NUM_THREADS=1"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=0"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=0"
                "./$binary" "$N" "$DT"
            )
            ;;

        *-omp-stream-omp-p2p|*-omp-stream-omp)
            # Versions 3 and 4:
            # one explicit application host thread per complete GPU card;
            # LLVM hidden-helper tasks are disabled.
            run_cmd=(
                env
                "ZE_AFFINITY_MASK=${mask}"
                "OMP_NUM_THREADS=${gpus}"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=0"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=0"
                "./$binary" "$N" "$DT"
            )
            ;;

        *-omp-stream|*-omp)
            # Versions 1 and 2:
            # one application host thread; one LLVM runtime helper per GPU card.
            # OMP_THREAD_LIMIT is deliberately removed because it is not part
            # of the runtime-managed configuration in the experiment table.
            run_cmd=(
                env
                -u OMP_THREAD_LIMIT
                "ZE_AFFINITY_MASK=${mask}"
                "OMP_NUM_THREADS=1"
                "LIBOMP_USE_HIDDEN_HELPER_TASK=1"
                "LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}"
                "./$binary" "$N" "$DT"
            )
            ;;

        *)
            echo "    ERROR: no runtime policy for ${binary}"
            FAILED+=("$binary (no runtime policy)")
            return 0
            ;;
    esac

    print_runtime_config "${run_cmd[@]}"

    if ! "${run_cmd[@]}"; then
        echo "    RUN FAILED"
        FAILED+=("$binary (run failed)")
        return 0
    fi

    if [[ ! -f "$out" ]]; then
        echo "    ERROR: expected ${out}, not produced"
        FAILED+=("$binary (no output)")
        return 0
    fi

    local result
    local compare_rc

    result=$(compare "$REF" "$out")
    compare_rc=$?
    echo "    $result"

    if [[ $compare_rc -eq 0 ]]; then
        PASSED+=("$binary")
    else
        FAILED+=("$binary")
    fi
}

check_variant 1-omp 1 u_openmp_1gpu.txt

for fam in $FAMILIES; do
    check_variant "${fam}-omp" \
        "$fam" "u_openmp_${fam}gpu.txt"

    check_variant "${fam}-omp-stream" \
        "$fam" "u_openmp_stream_${fam}gpu.txt"

    check_variant "${fam}-omp-stream-omp" \
        "$fam" "u_openmp_stream_omp_${fam}gpu.txt"

    check_variant "${fam}-omp-stream-omp-p2p" \
        "$fam" "u_openmp_stream_omp_p2p_${fam}gpu.txt"
done

echo
echo "============================================================"
echo "PASSED (${#PASSED[@]}): ${PASSED[*]:-none}"
echo "FAILED (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "============================================================"
echo
echo "Solution files were kept in the current working directory."
echo "Remove them with: ./clean.sh -f out"

[[ ${#FAILED[@]} -eq 0 ]]
