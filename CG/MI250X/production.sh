#!/bin/bash -l
#SBATCH --job-name=cg-prod
#SBATCH --account=project_465003145
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=prod-%j.out
# ---------------------------------------------------------------------------
# ./production.sh                                  -> submits itself
# GBS="5 10 20" ORDERS="natural random" ./production.sh
# REP=2 ./production.sh                            -> second repetition
#
# Serialize repetitions, or they will delete each other's binaries:
#   for r in 1 2 3; do
#     REP=$r sbatch --dependency=singleton --job-name=cg-prod production.sh
#   done
# Or give each repetition its own directory and submit them in parallel.
#
# Run ./verify.sh first. This script does not check correctness.
#
# ---------------------------------------------------------------------------
# LUMI-G, AMD MI250X (gfx90a). ONE GCD PER CARD.
#
# A node has four cards = eight GCDs. ROCR_VISIBLE_DEVICES picks GCDs 0, 2, 4
# and 6, so "4 GPUs" is four separate cards and every peer copy crosses an
# Infinity Fabric link between packages. Using both GCDs of a card would
# measure the on-package link and would not compare with the NVLink numbers.
#
# NO CPU VARIANT IN THE SWEEP. cg-cpu exists as verify.sh's reference but is
# not timed here: at 50 GB it would dominate the walltime for a number that is
# not what this study is about.
#
# ---------------------------------------------------------------------------
# THREE THINGS THIS MACHINE NEEDS THAT NVIDIA AND INTEL DO NOT
#
#   1. -fno-openmp-target-xteam-reduction
#      The AMD XTeam reduction path allocates its scratch buffer against a
#      single device. Two host threads reducing concurrently on two devices
#      then fault inside the first dot product, before any exchange happens.
#
#   2. HSA_ENABLE_SDMA=1, HSA_ENABLE_PEER_SDMA=1
#
#   3. libp2p_copy_shim.so, LD_PRELOADed ahead of the patched runtime.
#      It interposes hsa_amd_memory_async_copy_on_engine, grants both agents
#      access to both allocations and picks an available SDMA engine. Without
#      it EVERY device-to-device omp_target_memcpy is rejected with
#      HSA_STATUS_ERROR_INVALID_ARGUMENT -- and the run does not abort, it
#      carries on with p never exchanged and converges to garbage.
#
# LD_PRELOAD, not LD_LIBRARY_PATH: the binaries carry an RPATH into the system
# ROCm install, which wins over LD_LIBRARY_PATH.
#
# ---------------------------------------------------------------------------
# NOTE ON CARD SIZE
#
# An MI250X GCD has 64 GB, against the A100's 40, so GBS runs to 50 here where
# the NVIDIA sweep stopped at 30.
#
# ---------------------------------------------------------------------------
# Knobs (environment, all optional):
#   GBS         footprint targets in GB       default "5 10 20 30 40 50"
#   NNZ_ROW     mean nonzeros per row         default 15
#   ORDERS      natural rcm random            default "natural"
#   RUNS        solves per configuration      default 3
#   MAXIT       iteration cap                 default 500
#   TOL         relative tolerance            default 1e-8
#   SHIFT       diagonal shift                default 0.1
#   REP         repetition index (a LABEL)    default 1
#   VARIANTS    subset of the list            default: whatever built
#   CG_P2P      1 = shim + patched runtime    default 1
#   CG_OVERLAP  1 = overlapped exchange       default 1
#
# FOOTPRINT MEANS TOTAL, NOT PER DEVICE
#   Every variant at "20GB" solves the SAME system. The multi-device variants
#   split the MATRIX but REPLICATE the five vectors, so per-device memory is
#       gb * ( (12*nnz_row + 8)/ngpu + 40 ) / (12*nnz_row + 48)
#   which at nnz_row=15 is 0.59*gb for 2 devices and 0.38*gb for 4.
# ---------------------------------------------------------------------------

set -uo pipefail
# Deliberately not -e: one failing size must not kill the rest of the sweep.

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    exec sbatch --export=ALL "$0" "$@"
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"

GBS="${GBS:-5 10 20 30 40 50}"
NNZ_ROW="${NNZ_ROW:-15}"
ORDERS="${ORDERS:-natural}"
RUNS="${RUNS:-3}"
MAXIT="${MAXIT:-500}"
TOL="${TOL:-1e-8}"
SHIFT="${SHIFT:-0.1}"
REP="${REP:-1}"
THREADS="${THREADS:-56}"
SLEEP_SEC="${SLEEP_SEC:-5}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
ARCH="${ARCH:-gfx90a}"
CG_P2P="${CG_P2P:-1}"
CG_OVERLAP="${CG_OVERLAP:-1}"
GPU_MEM_MB="${GPU_MEM_MB:-65536}"     # per GCD
OMPTARGET_P2P_LIB="${OMPTARGET_P2P_LIB:-/scratch/project_465003145/omptarget-p2p/lib}"
P2P_SHIM="${P2P_SHIM:-${PWD}/libp2p_copy_shim.so}"
CG_MODULES="${CG_MODULES:-LUMI/25.09 partition/G craype-accel-amd-gfx90a PrgEnv-amd rocm/6.4.4 lumi-CrayPath}"

module --force purge
for m in $CG_MODULES; do
    if ! module load "$m" 2>&1; then
        echo "ERROR: module load $m failed" >&2
        exit 1
    fi
done

command -v cc >/dev/null || { echo "ERROR: cc not on PATH" >&2; exit 1; }

# ----- build ---------------------------------------------------------------
CPU_FLAGS=(-O3 -fopenmp -ffp-contract=off)

# -fno-openmp-target-xteam-reduction: see item 1 in the header. Without it
# every multi-device run dies in the first dot product.
OMP_GPU_FLAGS=(-O3 -fopenmp "--offload-arch=${ARCH}" -ffp-contract=off
               -fno-openmp-target-xteam-reduction)

echo "=== BUILD  $(hostname)  $(date)  arch=${ARCH} ==="
rm -f cg-cpu cg-ompgpu1 cg-ompgpu2 cg-ompgpu4 cg-ompgpu2halo cg-ompgpu4halo

if ! cc "${CPU_FLAGS[@]}" -o cg-cpu cg_omp.c -lm; then
    echo "ERROR: cg-cpu did not build, aborting sweep" >&2; exit 1
fi
echo "  OK : cg-cpu  (built for reference; not timed in this sweep)"

for k in 1 2 4; do
    if [[ -f "${k}-omp.c" ]]; then
        if ! cc "${OMP_GPU_FLAGS[@]}" -o "cg-ompgpu${k}" "${k}-omp.c" -lm; then
            echo "ERROR: cg-ompgpu${k} did not build, aborting sweep" >&2; exit 1
        fi
        echo "  OK : cg-ompgpu${k}"
    fi
    if [[ -f "${k}-omp-halo.c" ]]; then
        if ! cc "${OMP_GPU_FLAGS[@]}" -o "cg-ompgpu${k}halo" "${k}-omp-halo.c" -lm; then
            echo "ERROR: cg-ompgpu${k}halo did not build, aborting sweep" >&2; exit 1
        fi
        echo "  OK : cg-ompgpu${k}halo"
    fi
done
echo ""

# ----- P2P offload runtime --------------------------------------------------
# Three pieces, all required together:
#   libp2p_copy_shim.so  interposes hsa_amd_memory_async_copy_on_engine, grants
#                        both agents access to both allocations, and selects an
#                        available SDMA engine
#   libomp.so            patched host runtime
#   libomptarget.so      patched offload runtime
# LD_PRELOAD, not LD_LIBRARY_PATH: the binaries carry an RPATH into the system
# ROCm install, which wins over LD_LIBRARY_PATH.
P2P_ENV=()
if (( CG_P2P == 1 )); then
    if [[ ! -f "$P2P_SHIM" ]]; then
        cc -O2 -fPIC -shared -o "$P2P_SHIM" p2p_copy_shim.c \
           -I"${ROCM_PATH:-/opt/rocm}/include" \
           -Wl,--version-script=p2p_copy_shim.map \
           -L"${ROCM_PATH:-/opt/rocm}/lib" -lhsa-runtime64 -ldl \
          || { echo "ERROR: could not build $P2P_SHIM" >&2; exit 1; }
    fi
    for f in "$P2P_SHIM" "${OMPTARGET_P2P_LIB}/libomp.so" \
             "${OMPTARGET_P2P_LIB}/libomptarget.so.19.0git"; do
        [[ -r "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
    done
    P2P_ENV=(
        "HSA_ENABLE_SDMA=1"
        "HSA_ENABLE_PEER_SDMA=1"
        "LD_PRELOAD=${P2P_SHIM}:${OMPTARGET_P2P_LIB}/libomp.so:${OMPTARGET_P2P_LIB}/libomptarget.so.19.0git"
    )
    echo "P2P runtime : ${OMPTARGET_P2P_LIB} + $(basename "$P2P_SHIM")"
else
    echo "P2P runtime : DISABLED (CG_P2P=0) -- peer copies will be rejected"
fi
echo ""

# ----- device inventory -----------------------------------------------------
GCD_COUNT=$(rocm-smi --showid --csv 2>/dev/null | grep -c '^card' || echo 0)
if (( GCD_COUNT == 0 )); then
    GCD_COUNT=$(ls -d /sys/class/kfd/kfd/topology/nodes/*/ 2>/dev/null | wc -l)
fi
CARD_COUNT=$(( GCD_COUNT / 2 ))
if (( CARD_COUNT < 1 )); then echo "ERROR: no visible GPU" >&2; exit 1; fi

DEFAULT_VARIANTS=""
for k in 1 2 4; do
    [[ -x "./cg-ompgpu${k}"     ]] && (( CARD_COUNT >= k )) && DEFAULT_VARIANTS="${DEFAULT_VARIANTS} ompgpu${k}"
    [[ -x "./cg-ompgpu${k}halo" ]] && (( CARD_COUNT >= k )) && DEFAULT_VARIANTS="${DEFAULT_VARIANTS} ompgpu${k}halo"
done
VARIANTS="${VARIANTS:-$DEFAULT_VARIANTS}"

export OMP_WAIT_POLICY=active

RESULTS_DIR="results_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
TIMING_CSV="${RESULTS_DIR}/timings.csv"
RUN_LOG="${RESULTS_DIR}/run.log"
SKIPPED_LOG="${RESULTS_DIR}/skipped.log"

echo "rep,job_id,variant,gb_target,impl,backend,order,rows,nnz,nnz_row,mean_bandwidth,footprint_gb,iters,relres,cycle_ms,spmv_ms,other_ms,spmv_share,gbs,best_ms,spread_pct,runs,par,halo_ms,wall_sec,status" \
    > "$TIMING_CSV"

exec > >(tee -a "$RUN_LOG") 2>&1

HOST_MEM_GB=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)

echo "============================================================"
echo "Host          : $(hostname)"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Repetition    : ${REP}"
echo "Results dir   : ${RESULTS_DIR}"
echo "Footprints GB : ${GBS}   (TOTAL problem size, not per device)"
echo "nnz per row   : ${NNZ_ROW}"
echo "Orderings     : ${ORDERS}"
echo "Runs / config : ${RUNS}"
echo "Variants      : ${VARIANTS}"
echo "GCDs seen     : ${GCD_COUNT}  ->  ${CARD_COUNT} cards (one GCD each)"
echo "GCD memory    : ${GPU_MEM_MB} MB    Host memory : ${HOST_MEM_GB} GB"
echo "Offload arch  : ${ARCH}"
echo "CG_OVERLAP    : ${CG_OVERLAP}"
echo "Precision     : fp64"
echo "============================================================"

echo "--- device topology ----------------------------------------"
rocm-smi --showtopo 2>/dev/null || echo "  (rocm-smi --showtopo unavailable)"
echo "-----------------------------------------------------------"
echo ""

rows_for_gb() {
    awk -v gb="$1" -v r="$2" 'BEGIN { bpr = 12*r + 48; printf "%d", int((gb*1e9)/(bpr*1.04)); }'
}

per_gpu_gb() {
    awk -v gb="$1" -v r="$2" -v g="$3" \
        'BEGIN { printf "%.2f", gb * ((12*r + 8)/g + 40) / (12*r + 48) }'
}

variant_ngpu() {
    case "$1" in
        cpu)                  echo 0 ;;
        ompgpu1)              echo 1 ;;
        ompgpu2|ompgpu2halo)  echo 2 ;;
        ompgpu4|ompgpu4halo)  echo 4 ;;
        *)                    echo -1 ;;
    esac
}

# ONE GCD PER CARD: GCDs 0,2,4,6 sit on cards 0,1,2,3.
visible_devices() {
    case "$1" in
        1) echo "0" ;;
        2) echo "0,2" ;;
        4) echo "0,2,4,6" ;;
        *) echo "" ;;
    esac
}

run_bench() {
    local variant="$1" gb="$2" order="$3"
    local binary ngpu n status wall t_start t_end exit_code payload halo out pergpu devs
    local cmd=()

    ngpu=$(variant_ngpu "$variant")
    if (( ngpu < 0 )); then echo "ERROR: unknown variant '$variant'" >&2; return 0; fi

    case "$variant" in
        cpu) binary=cg-cpu ;;
        *)   binary="cg-${variant}" ;;
    esac

    n=$(rows_for_gb "$gb" "$NNZ_ROW")

    if (( ngpu == 0 )); then
        cmd=(env -u ROCR_VISIBLE_DEVICES "OMP_NUM_THREADS=${THREADS}"
             "OMP_PROC_BIND=close" "OMP_PLACES=cores" "OMP_WAIT_POLICY=active")
    else
        devs=$(visible_devices "$ngpu")
        cmd=(env "ROCR_VISIBLE_DEVICES=${devs}" "CG_OVERLAP=${CG_OVERLAP}"
             "${P2P_ENV[@]}")
    fi

    if [[ ! -x "./$binary" ]]; then
        echo "  SKIP (no binary '$binary'): ${variant} ${gb}GB ${order}" | tee -a "$SKIPPED_LOG"
        return 0
    fi
    if (( ngpu > CARD_COUNT )); then
        echo "  SKIP (needs ${ngpu} cards, ${CARD_COUNT} visible): ${variant} ${gb}GB" | tee -a "$SKIPPED_LOG"
        return 0
    fi
    if (( n > 2147483000 )); then
        echo "  SKIP (n=${n} exceeds int32): ${variant} ${gb}GB" | tee -a "$SKIPPED_LOG"
        return 0
    fi
    if (( ngpu >= 1 )); then
        pergpu=$(per_gpu_gb "$gb" "$NNZ_ROW" "$ngpu")
        if (( $(awk -v p="$pergpu" -v m="$GPU_MEM_MB" 'BEGIN{print (p*1000 > m*0.92)}') )); then
            echo "  SKIP (${pergpu}GB/GCD over 92% of ${GPU_MEM_MB}MB): ${variant} ${gb}GB" | tee -a "$SKIPPED_LOG"
            return 0
        fi
    fi
    if (( $(awk -v g="$gb" -v h="$HOST_MEM_GB" 'BEGIN{print (g*1.3 > h*0.8)}') )); then
        echo "  SKIP (${gb}GB host peak over 80% of RAM): ${variant}" | tee -a "$SKIPPED_LOG"
        return 0
    fi

    echo ""
    if (( ngpu >= 2 )); then
        echo ">>> ${variant}  ${gb}GB total (${pergpu}GB/GCD)  order=${order}  n=${n}"
    else
        echo ">>> ${variant}  ${gb}GB  order=${order}  n=${n}"
    fi

    cmd+=("./$binary" -n "$n" -deg "$NNZ_ROW" -shift "$SHIFT" -order "$order"
          -tol "$TOL" -maxit "$MAXIT" -runs "$RUNS")

    out=$(mktemp "${TMPDIR:-/tmp}/cg_out.XXXXXX")
    t_start=$(date +%s%3N)
    if timeout "$TIMEOUT_SEC" "${cmd[@]}" > "$out" 2>&1; then
        status="ok"
    else
        exit_code=$?
        [[ $exit_code -eq 124 ]] && status="timeout" || status="runtime_error"
    fi
    t_end=$(date +%s%3N)
    wall=$(awk -v s="$t_start" -v e="$t_end" 'BEGIN{printf "%.4f",(e-s)/1000.0}')

    cat "$out"

    # A rejected peer copy does NOT abort the run: the plugin prints an error
    # and CG carries on with p unexchanged, converging to a wrong answer at
    # full speed. Catch it here so the row is not silently recorded as ok.
    if grep -q 'Failure to copy data from device' "$out"; then
        echo "  ERROR: peer copies were rejected -- is the shim loaded?" | tee -a "$SKIPPED_LOG"
        status="p2p_error"
    fi

    payload=$(grep -m1 '^RESULT,' "$out" | cut -d, -f2-)
    halo=$(grep -m1 '^HALO,' "$out" | awk -F, '{printf "%s", $5}')
    [[ -z "$halo" ]] && halo="N/A"
    rm -f "$out"

    if [[ -z "$payload" ]]; then
        payload="N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A"
        [[ "$status" == "ok" ]] && status="parse_error"
    else
        echo "    -> $(echo "$payload" | awk -F, '{printf "cycle=%s ms  spmv=%s ms  %s GB/s  spread=%s%%  iters=%s", $11,$12,$15,$17,$9}')  wall=${wall}s"
    fi

    echo "${REP},${SLURM_JOB_ID},${variant},${gb},${payload},${halo},${wall},${status}" >> "$TIMING_CSV"
    sleep "$SLEEP_SEC"
}

echo ""
echo "Starting sweep"
echo "==========================="

for order in $ORDERS; do
    for gb in $GBS; do
        for variant in $VARIANTS; do
            run_bench "$variant" "$gb" "$order"
        done
    done
done

# ---------------------------------------------------------------------------
# SUMMARY -- full CG cycle, not SpMV. The baseline is ompgpu1: the CPU variant
# is not part of this sweep.
# ---------------------------------------------------------------------------
echo ""
echo ""
echo "SUMMARY -- full CG cycle, ms/iter"
echo "==============================================================================="
awk -F, -v gbs="$GBS" -v orders="$ORDERS" -v vars="$VARIANTS" '
NR == 1 { next }
$NF != "ok" { next }
{ cyc[$7 "|" $4 "|" $3] = $15 }
END {
    no = split(orders, ords, " ")
    ng = split(gbs,    glist, " ")
    nv = split(vars,   vlist, " ")

    for (o = 1; o <= no; o++) {
        printf "\norder = %s\n\n", ords[o]

        printf "%6s", "GB"
        for (v = 1; v <= nv; v++) printf "%14s", vlist[v]
        printf "\n%s\n", "-------------------------------------------------------------------------------"
        for (i = 1; i <= ng; i++) {
            printf "%6s", glist[i]
            for (v = 1; v <= nv; v++) {
                t = cyc[ords[o] "|" glist[i] "|" vlist[v]]
                printf "%14s", (t == "" ? "-" : sprintf("%.3f", t))
            }
            printf "\n"
        }

        printf "\n  scaling vs ompgpu1 (ideal 2.00x / 4.00x)\n"
        printf "%6s %12s %12s %14s %14s\n",
               "GB", "ompgpu2/1", "ompgpu4/1", "halo2/1", "halo4/1"
        printf "%s\n", "-------------------------------------------------------------------------------"
        for (i = 1; i <= ng; i++) {
            g  = glist[i]
            o1 = cyc[ords[o] "|" g "|ompgpu1"]
            o2 = cyc[ords[o] "|" g "|ompgpu2"];  o4 = cyc[ords[o] "|" g "|ompgpu4"]
            h2 = cyc[ords[o] "|" g "|ompgpu2halo"]
            h4 = cyc[ords[o] "|" g "|ompgpu4halo"]
            printf "%6s", g
            printf "%13s", (o1 != "" && o2 != "" ? sprintf("%.2fx", o1/o2) : "-")
            printf "%13s", (o1 != "" && o4 != "" ? sprintf("%.2fx", o1/o4) : "-")
            printf "%14s", (o1 != "" && h2 != "" ? sprintf("%.2fx", o1/h2) : "-")
            printf "%15s", (o1 != "" && h4 != "" ? sprintf("%.2fx", o1/h4) : "-")
            printf "\n"
        }

        printf "\n  packed halo vs replicated exchange, same device count\n"
        printf "%6s %14s %14s\n", "GB", "halo2/ompgpu2", "halo4/ompgpu4"
        printf "%s\n", "-------------------------------------------------------------------------------"
        for (i = 1; i <= ng; i++) {
            g  = glist[i]
            o2 = cyc[ords[o] "|" g "|ompgpu2"];  h2 = cyc[ords[o] "|" g "|ompgpu2halo"]
            o4 = cyc[ords[o] "|" g "|ompgpu4"];  h4 = cyc[ords[o] "|" g "|ompgpu4halo"]
            printf "%6s", g
            printf "%15s", (o2 != "" && h2 != "" ? sprintf("%.2fx", o2/h2) : "-")
            printf "%15s", (o4 != "" && h4 != "" ? sprintf("%.2fx", o4/h4) : "-")
            printf "\n"
        }
    }
}' "$TIMING_CSV"
echo "==============================================================================="

echo ""
echo "All done."
echo "  Timing CSV : $TIMING_CSV"
echo "  Run log    : $RUN_LOG"
[[ -s "$SKIPPED_LOG" ]] && echo "  Skipped    : $SKIPPED_LOG ($(wc -l < "$SKIPPED_LOG") entries)"

exit 0
