#!/bin/bash -l
#SBATCH --job-name=cg-verify
#SBATCH --account=ehpc681
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=verify-%j.out
# ---------------------------------------------------------------------------
# verify.sh -- build from scratch, then check correctness. One file, no
# helpers. Nothing here is a measurement.
#
#   ./verify.sh                 -> submits itself via sbatch
#   VERIFY_N=2000000 ./verify.sh
#
# The exit code is the gate: production is only worth submitting on 0.
#
# ---------------------------------------------------------------------------
# SITE PORTABILITY
#
# Written for MareNostrum 5 (acc partition, 4x H100-64GB per node). The two
# site-specific things are environment variables:
#
#   CG_MODULES   modules to load, in order
#   ARCH         GPU arch; AUTO-DETECTED from nvidia-smi when unset
#
# The SBATCH header needs editing per site. On MeluXina that was:
# -A p201103, -p gpu, --qos default, no --gres.
#
# ---------------------------------------------------------------------------
# SOURCE -> BINARY -> VARIANT
#
#   cg_omp.c        -> cg-cpu          -> cpu        OpenMP, host
#   1-omp.c         -> cg-ompgpu1      -> ompgpu1    OpenMP target, 1 GPU
#   2-omp.c         -> cg-ompgpu2      -> ompgpu2    OpenMP target, 2 GPUs
#   4-omp.c         -> cg-ompgpu4      -> ompgpu4    OpenMP target, 4 GPUs
#   2-omp-halo.c    -> cg-ompgpu2halo  -> packed halo, 2 GPUs
#   4-omp-halo.c    -> cg-ompgpu4halo  -> packed halo, 4 GPUs
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
#   CG_OVERLAP   passed through (unset = each binary's own default)
#
# WHAT IS CHECKED
#
#   1. It builds.
#   2. True residual, recomputed as ||b - Ax||, not the drifting recursive r.
#   3. Cross-variant agreement against cg-cpu, within ITER_SLACK. This is what
#      catches a broken multi-GPU p exchange.
#   4. Ordering invariance: a symmetric permutation cannot move the count.
#   5. Multi-GPU halo path, reported not gated.
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
THREADS="${SLURM_CPUS_ON_NODE:-80}"
CG_MODULES="${CG_MODULES:-EB/apps GCC/13.2.0 cuda/12.8 clang/18.1.8-cuda12.8}"

echo "############################################################"
echo "# CG VERIFY   job ${SLURM_JOB_ID}   $(date)"
echo "# host ${SLURMD_NODENAME:-$(hostname)}"
echo "############################################################"
echo ""

module purge 2>/dev/null
for m in $CG_MODULES; do
    if ! module load "$m" 2>&1; then
        echo "ERROR: module load $m failed" >&2
        exit 1
    fi
done

for tool in clang nvidia-smi; do
    command -v "$tool" >/dev/null || { echo "ERROR: $tool not on PATH" >&2; exit 1; }
done

if [[ -z "${ARCH:-}" ]]; then
    CC_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
    if [[ -n "$CC_RAW" ]]; then
        ARCH="sm_${CC_RAW}"
    else
        echo "ERROR: could not detect compute capability; set ARCH=sm_XX by hand" >&2
        exit 1
    fi
fi

# ----- build from scratch ---------------------------------------------------
CPU_FLAGS=(-O3 -fopenmp -march=native -ffp-contract=off -Wl,--no-warn-execstack)
OMP_GPU_FLAGS=(-O3 -fopenmp "--offload-arch=${ARCH}" -ffp-contract=off -Wl,--no-warn-execstack)

echo "============================================================"
echo "BUILD"
echo "  modules : ${CG_MODULES}"
echo "  arch    : ${ARCH}  (auto-detected)"
echo "  cpu cc  : $(clang --version 2>/dev/null | head -1)"
echo "  cpu     : $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2); print $2; exit}')"
nvidia-smi -L 2>/dev/null || echo "  (no GPU visible)"
echo "============================================================"

rm -f cg-cpu cg-ompgpu1 cg-ompgpu2 cg-ompgpu4 cg-ompgpu2halo cg-ompgpu4halo

if ! clang "${CPU_FLAGS[@]}" -o cg-cpu cg_omp.c -lm; then
    echo ""; echo "VERIFY FAILED -- cg-cpu did not build"; exit 1
fi
echo "  OK : cg-cpu"

for k in 1 2 4; do
    if [[ -f "${k}-omp.c" ]]; then
        if ! clang "${OMP_GPU_FLAGS[@]}" -o "cg-ompgpu${k}" "${k}-omp.c" -lm; then
            echo ""; echo "VERIFY FAILED -- cg-ompgpu${k} did not build"; exit 1
        fi
        echo "  OK : cg-ompgpu${k}"
    fi
    if [[ -f "${k}-omp-halo.c" ]]; then
        if ! clang "${OMP_GPU_FLAGS[@]}" -o "cg-ompgpu${k}halo" "${k}-omp-halo.c" -lm; then
            echo ""; echo "VERIFY FAILED -- cg-ompgpu${k}halo did not build"; exit 1
        fi
        echo "  OK : cg-ompgpu${k}halo"
    fi
done
echo ""

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)

VARIANTS=()
[[ -x ./cg-cpu ]] && VARIANTS+=(cpu)
for k in 1 2 4; do
    [[ -x "./cg-ompgpu${k}"     ]] && (( GPU_COUNT >= k )) && VARIANTS+=("ompgpu${k}")
    [[ -x "./cg-ompgpu${k}halo" ]] && (( GPU_COUNT >= k )) && VARIANTS+=("ompgpu${k}halo")
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
echo "  GPUs seen   : ${GPU_COUNT}"
echo "  runs/config : ${VRUNS}"
echo "============================================================"
echo ""

echo "--- GPU topology -------------------------------------------"
nvidia-smi topo -m 2>/dev/null || echo "  (unavailable)"
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

visible_devices() {
    case "$1" in
        1) echo "0" ;;
        2) echo "0,1" ;;
        4) echo "0,1,2,3" ;;
        *) echo "" ;;
    esac
}

run_one() {
    local variant="$1" order="$2"
    local binary out line cmd=() key="$1|$2" ngpu devs

    case "$variant" in
        cpu) binary=cg-cpu ;;
        *)   binary="cg-${variant}" ;;
    esac
    ngpu=$(variant_ngpu "$variant")

    if (( ngpu == 0 )); then
        cmd=(env -u CUDA_VISIBLE_DEVICES "OMP_NUM_THREADS=${THREADS}"
             "OMP_PROC_BIND=close" "OMP_PLACES=cores")
    else
        devs=$(visible_devices "$ngpu")
        cmd=(env "CUDA_VISIBLE_DEVICES=${devs}")
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

echo "--- check 4: multi-GPU halo path ---------------------------"
SAW_MULTI=0
for variant in ompgpu2 ompgpu4 ompgpu2halo ompgpu4halo; do
    for order in $ORDERS; do
        path="${HALOPATH["$variant|$order"]:-}"
        [[ -n "$path" ]] || continue
        SAW_MULTI=1
        if [[ "$path" == *"host staging"* ]]; then
            echo "  WARNING: ${variant} is using ${path}"
            echo "           No peer access between these GPUs. Correct but will"
            echo "           not scale -- see the topology above."
        else
            echo "  OK: ${variant} -> ${path}"
        fi
        break
    done
done
(( SAW_MULTI == 0 )) && echo "  SKIP: no multi-GPU variant reported a halo path"
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
