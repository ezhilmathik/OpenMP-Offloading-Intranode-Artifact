#!/bin/bash -l
#SBATCH --job-name=cg-verify
#SBATCH --account=pn67qe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=verify-%j.out
# ---------------------------------------------------------------------------
# verify.sh -- build from scratch, then check correctness on Intel PVC.
# Nothing here is a measurement.
#
#   ./verify.sh                 -> submits itself via sbatch
#   VERIFY_N=2000000 ./verify.sh
#
# The exit code is the gate: production is only worth submitting on 0.
#
# ---------------------------------------------------------------------------
# ONE DEVICE = ONE WHOLE CARD
#
# A Max 1550 is two stacks. ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE makes the Level
# Zero root device the whole card, and LIBOMPTARGET_DEVICES must not downgrade
# it to a stack -- both are required, either alone is not enough. So the mask
# indices ARE card indices:
#
#   1 card  -> ZE_AFFINITY_MASK=0
#   2 cards -> ZE_AFFINITY_MASK=0,1
#   4 cards -> ZE_AFFINITY_MASK=0,1,2,3
#
# This is the Intel analogue of one GCD per MI250X and one A100 per NVLink
# peer: every inter-device copy crosses a card boundary.
#
# ---------------------------------------------------------------------------
# SOURCE -> BINARY -> VARIANT
#
#   cg_omp.c        -> cg-cpu          -> cpu        OpenMP, host
#   1-omp.c         -> cg-ompgpu1      -> ompgpu1    OpenMP target, 1 card
#   2-omp.c         -> cg-ompgpu2      -> ompgpu2    OpenMP target, 2 cards
#   4-omp.c         -> cg-ompgpu4      -> ompgpu4    OpenMP target, 4 cards
#   2-omp-halo.c    -> cg-ompgpu2halo  -> packed halo, 2 cards
#   4-omp-halo.c    -> cg-ompgpu4halo  -> packed halo, 4 cards
#
# ---------------------------------------------------------------------------
# Environment:
#   VERIFY_N     rows for the check            default 500000
#   DEG          mean nnz per row              default 15
#   TOL          solver tolerance              default 1e-8
#   MAXIT        iteration cap                 default 2000
#   RELRES_MAX   worst true residual accepted  default 1e-7
#   ITER_SLACK   iteration count tolerance     default 2
#   VRUNS        solves per configuration      default 3
#   SPREAD_MAX   worst run-to-run spread       default 25 (%)
#   ORDERS       orderings to cross-check      default "natural rcm random"
#   HELPERS      hidden-helper threads         default 8 (libomp's own default)
#   CG_OVERLAP   passed through to the binaries (unset = each one's default)
#
# WHAT IS CHECKED
#
#   1. It builds.
#   2. True residual, recomputed as ||b - Ax||, not the drifting recursive r.
#   3. Cross-variant agreement against cg-cpu, within ITER_SLACK. This is what
#      catches a broken multi-device p exchange.
#   4. Ordering invariance: a symmetric permutation cannot move the count.
#   5. Multi-device halo path, reported not gated. Unlike the AMD build there
#      is no LD_PRELOAD here -- if cross-device P2P is not wired up, the Intel
#      runtime stages through the host, which shows up as roughly halved
#      bandwidth rather than an error. So read this line, it will not fail.
#   6. Run-to-run consistency.
# ---------------------------------------------------------------------------

set -uo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec sbatch --export=ALL "$0" "$@"
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"

VERIFY_N="${VERIFY_N:-500000}"
DEG="${DEG:-15}"
TOL="${TOL:-1e-8}"
MAXIT="${MAXIT:-2000}"
RELRES_MAX="${RELRES_MAX:-1e-7}"
ITER_SLACK="${ITER_SLACK:-2}"
VRUNS="${VRUNS:-3}"
SPREAD_MAX="${SPREAD_MAX:-25}"
ORDERS="${ORDERS:-natural rcm random}"
SHIFT="${SHIFT:-0.1}"
THREADS="${THREADS:-${SLURM_CPUS_ON_NODE:-96}}"
HELPERS="${HELPERS:-8}"
HIER="COMPOSITE"

echo "############################################################"
echo "# CG VERIFY   job ${SLURM_JOB_ID}   $(date)"
echo "# host ${SLURMD_NODENAME:-$(hostname)}"
echo "############################################################"
echo ""

module load slurm_setup
module load intel-toolkit

command -v icx >/dev/null || { echo "ERROR: icx not on PATH" >&2; exit 1; }

# ----- build from scratch ---------------------------------------------------
# Same flags as the Makefile. See the note there on -fno-fast-math.
CPU_FLAGS=(-O3 -fno-fast-math -fiopenmp -ffp-contract=off -Wl,--no-warn-execstack)
GPU_FLAGS=(-O3 -fno-fast-math -fiopenmp -fopenmp-targets=spir64 -ffp-contract=off
           -Wl,--no-warn-execstack)

echo "============================================================"
echo "BUILD"
echo "  compiler : $(icx --version 2>/dev/null | head -1)"
echo "  cpu      : $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')"
echo "============================================================"
sycl-ls 2>/dev/null || xpu-smi discovery 2>/dev/null || true
echo ""

rm -f cg-cpu cg-ompgpu1 cg-ompgpu2 cg-ompgpu4 cg-ompgpu2halo cg-ompgpu4halo

if ! icx "${CPU_FLAGS[@]}" -o cg-cpu cg_omp.c -lm; then
    echo ""; echo "VERIFY FAILED -- cg-cpu did not build"; exit 1
fi
echo "  OK : cg-cpu"

for k in 1 2 4; do
    if [[ -f "${k}-omp.c" ]]; then
        if ! icx "${GPU_FLAGS[@]}" -o "cg-ompgpu${k}" "${k}-omp.c" -lm; then
            echo ""; echo "VERIFY FAILED -- cg-ompgpu${k} did not build"; exit 1
        fi
        echo "  OK : cg-ompgpu${k}"
    fi
    if [[ -f "${k}-omp-halo.c" ]]; then
        if ! icx "${GPU_FLAGS[@]}" -o "cg-ompgpu${k}halo" "${k}-omp-halo.c" -lm; then
            echo ""; echo "VERIFY FAILED -- cg-ompgpu${k}halo did not build"; exit 1
        fi
        echo "  OK : cg-ompgpu${k}halo"
    fi
done
echo ""

# ----- runtime environment, identical for every binary ----------------------
export OMP_TARGET_OFFLOAD=MANDATORY
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1     # stable, PCI-ordered numbering
export ZE_FLAT_DEVICE_HIERARCHY="$HIER"    # 1 device = 1 whole card
unset ONEAPI_DEVICE_SELECTOR               # ZE_AFFINITY_MASK controls OpenMP devices
unset LIBOMPTARGET_DEVICES                 # must not downgrade a card to a stack
unset OMP_THREAD_LIMIT                     # would also cap threads INSIDE the
                                           # target regions and cost ~3x

# ----- device inventory -----------------------------------------------------
CARD_COUNT=$(sycl-ls 2>/dev/null | grep -c 'ext_oneapi_level_zero:gpu' || echo 0)
if (( CARD_COUNT == 0 )); then
    CARD_COUNT=$(xpu-smi discovery 2>/dev/null | grep -c 'Device ID' || echo 0)
fi
(( CARD_COUNT < 1 )) && CARD_COUNT=1

VARIANTS=()
[[ -x ./cg-cpu ]] && VARIANTS+=(cpu)
for k in 1 2 4; do
    [[ -x "./cg-ompgpu${k}"     ]] && (( CARD_COUNT >= k )) && VARIANTS+=("ompgpu${k}")
    [[ -x "./cg-ompgpu${k}halo" ]] && (( CARD_COUNT >= k )) && VARIANTS+=("ompgpu${k}halo")
done

declare -A ITERS RELRES SPREAD HALOPATH
FAIL=0
note_fail() { echo "    FAIL: $*"; FAIL=1; }

echo "============================================================"
echo "CORRECTNESS"
echo "  rows        : ${VERIFY_N}"
echo "  nnz/row     : ${DEG}"
echo "  tolerance   : ${TOL}   (true residual must be < ${RELRES_MAX})"
echo "  orderings   : ${ORDERS}"
echo "  variants    : ${VARIANTS[*]}"
echo "  cards seen  : ${CARD_COUNT}  (COMPOSITE: 1 device = 1 whole card)"
echo "  runs/config : ${VRUNS}"
echo "  helpers     : ${HELPERS}"
echo "============================================================"
echo ""

variant_ngpu() {
    case "$1" in
        cpu)                  echo 0 ;;
        ompgpu1)              echo 1 ;;
        ompgpu2|ompgpu2halo)  echo 2 ;;
        ompgpu4|ompgpu4halo)  echo 4 ;;
        *)                    echo -1 ;;
    esac
}

# COMPOSITE: mask index = card index.
mask_for() {
    local n="$1" v=() i
    for ((i = 0; i < n; i++)); do v+=("$i"); done
    (IFS=,; echo "${v[*]}")
}

run_one() {
    local variant="$1" order="$2"
    local binary out line cmd=() key="$1|$2" ngpu mask

    case "$variant" in
        cpu) binary=cg-cpu ;;
        *)   binary="cg-${variant}" ;;
    esac
    ngpu=$(variant_ngpu "$variant")

    if (( ngpu == 0 )); then
        cmd=(env -u ZE_AFFINITY_MASK -u OMP_THREAD_LIMIT
             "OMP_NUM_THREADS=${THREADS}" "OMP_PROC_BIND=spread" "OMP_PLACES=cores")
    else
        mask=$(mask_for "$ngpu")
        # One explicit host thread per card, as in the heat benchmark's
        # *-stream-omp variants: the CG sources drive each device from its own
        # host thread. Hidden helpers stay enabled at libomp's default count.
        cmd=(env -u OMP_THREAD_LIMIT -u LIBOMPTARGET_DEVICES
             "ZE_FLAT_DEVICE_HIERARCHY=${HIER}"
             "ZE_AFFINITY_MASK=${mask}"
             "LIBOMP_USE_HIDDEN_HELPER_TASK=1"
             "LIBOMP_NUM_HIDDEN_HELPER_THREADS=${HELPERS}")
    fi
    [[ -n "${CG_OVERLAP:-}" ]] && cmd+=("CG_OVERLAP=${CG_OVERLAP}")

    cmd+=("./$binary" -n "$VERIFY_N" -deg "$DEG" -shift "$SHIFT" -order "$order"
          -tol "$TOL" -maxit "$MAXIT" -runs "$VRUNS")

    out=$(mktemp "${TMPDIR:-/tmp}/cg_verify.XXXXXX")
    if ! "${cmd[@]}" > "$out" 2>&1; then
        note_fail "${variant} ${order}: non-zero exit"
        sed 's/^/      | /' "$out" | tail -20
        rm -f "$out"; return 1
    fi

    line=$(grep -m1 '^RESULT,' "$out")
    if [[ -z "$line" ]]; then
        note_fail "${variant} ${order}: no RESULT line"
        sed 's/^/      | /' "$out" | tail -20
        rm -f "$out"; return 1
    fi

    ITERS[$key]=$(echo "$line"  | awk -F, '{print $10}')
    RELRES[$key]=$(echo "$line" | awk -F, '{print $11}')
    SPREAD[$key]=$(echo "$line" | awk -F, '{print $18}')
    if (( ngpu >= 2 )); then
        HALOPATH[$key]=$(awk -F: '/halo path/ {sub(/^ +/,"",$2); print $2; exit}' "$out")
    fi
    rm -f "$out"; return 0
}

echo "--- running ------------------------------------------------"
for order in $ORDERS; do
    for variant in "${VARIANTS[@]}"; do
        printf "  %-13s %-8s " "$variant" "$order"
        if run_one "$variant" "$order"; then
            printf "iters=%-5s relres=%s\n" \
                   "${ITERS["$variant|$order"]}" "${RELRES["$variant|$order"]}"
        else
            printf "\n"
        fi
    done
done
echo ""

echo "--- check 1: true residual ---------------------------------"
C1=0
for order in $ORDERS; do
    for variant in "${VARIANTS[@]}"; do
        key="$variant|$order"
        [[ -n "${RELRES[$key]:-}" ]] || continue
        if (( $(awk -v r="${RELRES[$key]}" -v m="$RELRES_MAX" \
                    'BEGIN{print (r+0 > m+0) ? 1 : 0}') )); then
            note_fail "${variant} ${order}: true residual ${RELRES[$key]} > ${RELRES_MAX}"
            C1=1
        fi
    done
done
(( C1 == 0 )) && echo "  OK: every variant converged below ${RELRES_MAX}"
echo ""

echo "--- check 2: cross-variant agreement -----------------------"
C2=0
REF=""
for cand in cpu ompgpu1; do
    for v in "${VARIANTS[@]}"; do [[ "$v" == "$cand" ]] && REF="$cand"; done
    [[ -n "$REF" ]] && break
done
if [[ -z "$REF" ]]; then
    echo "  SKIP: no reference variant to compare against"
else
    echo "  reference: ${REF}"
    for order in $ORDERS; do
        ref_it="${ITERS["$REF|$order"]:-}"
        [[ -n "$ref_it" ]] || continue
        for variant in "${VARIANTS[@]}"; do
            [[ "$variant" == "$REF" ]] && continue
            it="${ITERS["$variant|$order"]:-}"
            [[ -n "$it" ]] || continue
            d=$(( it > ref_it ? it - ref_it : ref_it - it ))
            if (( d > ITER_SLACK )); then
                note_fail "${variant} ${order}: ${it} iters vs ${REF} ${ref_it} (differs by ${d})"
                C2=1
            fi
        done
    done
    (( C2 == 0 )) && echo "  OK: all variants agree with ${REF} to within ${ITER_SLACK} iterations"
fi
echo ""

echo "--- check 3: ordering invariance ---------------------------"
C3=0
NORD=$(echo "$ORDERS" | wc -w)
if (( NORD < 2 )); then
    echo "  SKIP: need at least two orderings (got: ${ORDERS})"
else
    for variant in "${VARIANTS[@]}"; do
        lo=""; hi=""
        for order in $ORDERS; do
            it="${ITERS["$variant|$order"]:-}"
            [[ -n "$it" ]] || continue
            [[ -z "$lo" || $it -lt $lo ]] && lo=$it
            [[ -z "$hi" || $it -gt $hi ]] && hi=$it
        done
        [[ -n "$lo" ]] || continue
        if (( hi - lo > ITER_SLACK )); then
            note_fail "${variant}: count moves with ordering (${lo}..${hi}) -- permutation code is wrong"
            C3=1
        fi
    done
    (( C3 == 0 )) && echo "  OK: iteration counts stable across ${ORDERS}"
fi
echo ""

echo "--- check 4: multi-device halo path ------------------------"
SAW_MULTI=0
for variant in ompgpu2 ompgpu4 ompgpu2halo ompgpu4halo; do
    for order in $ORDERS; do
        path="${HALOPATH["$variant|$order"]:-}"
        [[ -n "$path" ]] || continue
        SAW_MULTI=1
        if [[ "$path" == *"host staging"* ]]; then
            echo "  WARNING: ${variant} is using ${path}"
            echo "           Correct but will not scale."
        else
            echo "  OK: ${variant} -> ${path}"
        fi
        break
    done
done
(( SAW_MULTI == 0 )) && echo "  SKIP: no multi-device variant reported a halo path"
echo ""

echo "--- check 5: run-to-run consistency ------------------------"
C5=0
if (( VRUNS < 2 )); then
    echo "  SKIP: needs VRUNS >= 2 (got ${VRUNS})"
else
    for order in $ORDERS; do
        for variant in "${VARIANTS[@]}"; do
            key="$variant|$order"
            sp="${SPREAD[$key]:-}"
            [[ -n "$sp" ]] || continue
            if (( $(awk -v s="$sp" -v m="$SPREAD_MAX" 'BEGIN{print (s+0 > m+0) ? 1 : 0}') )); then
                note_fail "${variant} ${order}: spread ${sp}% across ${VRUNS} runs"
                C5=1
            fi
        done
    done
    (( C5 == 0 )) && echo "  OK: all variants reproduce within ${SPREAD_MAX}% across ${VRUNS} runs"
fi
echo ""

if (( FAIL == 0 )); then
    echo "############################################################"
    echo "# VERIFY PASSED"
    echo "#"
    echo "# The solution is correct and the variants agree."
    echo "#"
    echo "#     ./production.sh"
    echo "############################################################"
    exit 0
else
    echo "############################################################"
    echo "# VERIFY FAILED"
    echo "#"
    echo "# Do NOT run production. The numbers would be meaningless."
    echo "# Read the FAIL lines above."
    echo "############################################################"
    exit 1
fi
