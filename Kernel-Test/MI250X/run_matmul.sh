#!/bin/bash -l
# ===========================================================================
# run_matmul.sh -- native GPU model vs OpenMP offload, for three benchmarks
#                  (BENCH=matmul | vecadd | matvec) on NVIDIA and AMD GPUs.
#
#   NVIDIA (MeluXina A100, MareNostrum~5 H100): CUDA (nvcc) vs OpenMP (Clang),
#                  profiled with Nsight Systems and Nsight Compute.
#   AMD    (LUMI-G MI250X, one GCD):            HIP (hipcc) vs OpenMP (Cray/amdclang),
#                  profiled with rocprofv3 and the compiler's kernel-resource remarks.
#                  MeluXina (A100-40GB) and MareNostrum 5 ACC (H100-64GB)
# ---------------------------------------------------------------------------
# From a LOGIN node, submits one job per size and stage:
#   ./run_matmul.sh                           all default sizes and stages
#   SIZES="10 20" STAGES=time ./run_matmul.sh subset
#   REP=2 ./run_matmul.sh                     repetition label (own directories)
#   DRY_RUN=1 ./run_matmul.sh                 print the sbatch commands only
# Inside an INTERACTIVE allocation (salloc), it runs everything here, in turn.
#
# mat-cuda.cu and mat-omp.cpp must be in the directory you run this from.
# They are copied into every result directory, so each result keeps the exact
# source that produced it. The programs themselves are not modified.
#
# STAGES (each is its own job, directory and build)
#   time  build; OpenMP kernel-mode probe (LIBOMPTARGET_INFO, N=1024);
#         both programs under nsys with GPU-metrics sampling; CSV exports;
#         per-kernel averages of the GPU metrics
#   ncu   build; Nsight Compute metrics for CUDA naive 32x8, naive 128x1 and
#         tiled32, one launch each, warm-up launch skipped
#
# SIZE = total device footprint of A, B and C in GB (1e9 bytes), N rounded
#        down to a multiple of 128.  matmul: 3*N*N*8 bytes,  vecadd: 3*N*8 bytes,
#        matvec: (N*N + 2*N)*8 bytes.
#
# BENCH  matmul  mat-cuda.cu + mat-omp.cpp -> results/<site>/<GB>GB/...
#        vecadd  vec-cuda.cu + vec-omp.cpp -> results/<site>/vecadd-<GB>GB/...
#        matvec  mv-cuda.cu  + mv-omp.cpp  -> results/<site>/matvec-<GB>GB/...
#
# Knobs (environment, all optional)
#   BENCH       matmul | vecadd | matvec      matmul
#   SITE        meluxina | mn5                auto-detected
#   SIZES       footprints in GB              matmul: meluxina "10 20 30", mn5 "30 45 55"
#                                             vecadd: "0.001 0.01 0.1" + the matmul sizes
#                                             matvec: "0.1 1" + the matmul sizes
#   STAGES      "time ncu"
#   REPS        timed calls after warm-up     5 (one untimed warm-up call precedes
#                                             them, so 6 calls per kernel in total)
#   REP         repetition label              1
#   GPU_INDEX   GPU (NVIDIA) or GCD (AMD) used   0
#   GPU_MEM_MB  device memory, AMD fallback      65536
#   PMC_SETS_STR AMD counter sets, '|' separated  see the default above
#   PMC_OMP     1 = counters for OpenMP too       1
#   PEAK_BW_GBS theoretical DRAM bandwidth      only needed where the program does
#               of the device, in GB/s          not print it (e.g. 1638 for one MI250X GCD)
#   WALL_TIME   walltime of a time job        matmul 03:00:00, vecadd/matvec 00:30:00
#   WALL_NCU    walltime of an ncu job        matmul 03:00:00, vecadd/matvec 00:30:00
#   GM_FREQ     GPU-metrics sampling, Hz      matmul 200, vecadd/matvec 2000
#   NCU_OMP     1 = also try ncu on OpenMP    0   (fails with LLVM 18, see below)
#   KEEP_SQLITE 1 = keep nsys .sqlite files   0   (.nsys-rep is always kept)
#   BACKFILL    1 = no jobs: add kern_*.csv (exact kernel resources) to
#                   existing results of this site from their .nsys-rep files
#   ACCOUNT PARTITION QOS GPUS_OPT MODULES    override the site defaults
#
# Afterwards:  ./collect.sh     then     gnuplot plot.gp
#
# NCU AND OPENMP. LLVM 18's offload runtime initialises CUDA before main(),
# which Nsight Compute refuses ("Cuda is initialized before the tool").
# The OpenMP kernel's memory behaviour therefore comes from nsys GPU metrics,
# plus the CUDA 1-D replica (naive 128x1), which ncu can profile.
# ===========================================================================

set -uo pipefail
# Deliberately not -e: one failing size or stage must not stop the others.

# ----- site ------------------------------------------------------------------
detect_site() {
    if [[ -n "${SITE:-}" ]]; then echo "$SITE"; return; fi
    if [[ -d /apps/USE ]]; then echo meluxina; return; fi
    if [[ -n "${LUMI_STACK_VERSION:-}" || -d /appl/lumi ]]; then echo lumi; return; fi
    case "$(hostname -s)" in
        alogin*|as[0-9]*) echo mn5; return ;;
        uan*|nid[0-9]*)   echo lumi; return ;;
    esac
    echo unknown
}

SITE=$(detect_site)
case "$SITE" in
    meluxina)
        ACCOUNT="${ACCOUNT:-p201103}"
        PARTITION="${PARTITION:-gpu}"
        QOS="${QOS:-default}"
        MODULES="${MODULES:-env/release/2024.1 Clang/18.1.8-GCCcore-13.3.0-CUDA-12.6.0}"
        SIZES_MAT="10 20 30"; VENDOR="${VENDOR:-nvidia}"
        ;;
    mn5)
        ACCOUNT="${ACCOUNT:-ehpc681}"
        PARTITION="${PARTITION:-acc}"
        QOS="${QOS:-acc_ehpc}"
        MODULES="${MODULES:-EB/apps GCC/13.2.0 cuda/12.8 clang/18.1.8-cuda12.8}"
        SIZES_MAT="30 45 55"; VENDOR="${VENDOR:-nvidia}"
        ;;
    lumi)
        ACCOUNT="${ACCOUNT:-project_465003145}"
        PARTITION="${PARTITION:-standard-g}"
        QOS="${QOS:-}"
        MODULES="${MODULES:-LUMI/25.09 partition/G craype-accel-amd-gfx90a PrgEnv-amd rocm/6.4.4 lumi-CrayPath}"
        SIZES_MAT="30 45 55"; VENDOR="${VENDOR:-amd}"
        GPUS_OPT="${GPUS_OPT:---gpus-per-node=8}"     # the job owns the node; one GCD is used
        ;;
    *)
        echo "ERROR: cannot tell the site; run with SITE=meluxina, SITE=mn5 or SITE=lumi" >&2
        exit 1
        ;;
esac

# ----- benchmark --------------------------------------------------------------
# Native model and source suffix: CUDA (.cu) on NVIDIA, HIP (.cpp) on AMD.
if [[ $VENDOR == amd ]]; then NAT=hip; NATEXT=cpp; else NAT=cuda; NATEXT=cu; fi
BENCH="${BENCH:-matmul}"
case "$BENCH" in
    matmul)
        SRC_CU=mat-$NAT.$NATEXT; SRC_OMP=mat-omp.cpp; BIN_CU=matmul_$NAT; BIN_OMP=matmul_omp
        DIRPRE=""; JOBPRE=mm; REPS_DEF=5; GMF_DEF=200; WALL_DEF=03:00:00
        NCU_REGEX="dgemm_naive|dgemm_tiled32"; OMP_KERNEL=matrixMultiplyOpenMP
        SIZES="${SIZES:-$SIZES_MAT}"
        ;;
    vecadd)
        SRC_CU=vec-$NAT.$NATEXT; SRC_OMP=vec-omp.cpp; BIN_CU=vecadd_$NAT; BIN_OMP=vecadd_omp
        DIRPRE="vecadd-"; JOBPRE=va; REPS_DEF=5; GMF_DEF=2000; WALL_DEF=00:30:00
        NCU_REGEX="vadd_"; OMP_KERNEL=vectorAddOpenMP
        SIZES="${SIZES:-0.001 0.01 0.1 $SIZES_MAT}"
        ;;
    matvec)
        SRC_CU=mv-$NAT.$NATEXT; SRC_OMP=mv-omp.cpp; BIN_CU=matvec_$NAT; BIN_OMP=matvec_omp
        DIRPRE="matvec-"; JOBPRE="mv"; REPS_DEF=5; GMF_DEF=2000; WALL_DEF=00:30:00
        NCU_REGEX="mv_"; OMP_KERNEL=matrix_vector_multiply
        SIZES="${SIZES:-0.1 1 $SIZES_MAT}"
        ;;
    *)
        echo "ERROR: BENCH must be matmul, vecadd or matvec" >&2
        exit 1
        ;;
esac

GPUS_OPT="${GPUS_OPT:---gres=gpu:4}"
STAGES="${STAGES:-time ncu}"
REPS="${REPS:-$REPS_DEF}"
REP="${REP:-1}"
GPU_INDEX="${GPU_INDEX:-0}"
WALL_TIME="${WALL_TIME:-$WALL_DEF}"
WALL_NCU="${WALL_NCU:-$WALL_DEF}"
GM_FREQ="${GM_FREQ:-$GMF_DEF}"
NCU_OMP="${NCU_OMP:-0}"
KEEP_SQLITE="${KEEP_SQLITE:-0}"
RESULTS="${RESULTS:-results}"

# AMD counter sets: one rocprof run each (see stage_pmc). FETCH_SIZE/WRITE_SIZE
# are HBM traffic in KB per dispatch; the occupancy names vary between ROCm
# versions, so that set is allowed to fail.
PMC_OMP="${PMC_OMP:-1}"          # 1 = also collect counters for the OpenMP binary
if [[ -n ${PMC_SETS_STR:-} ]]; then
    IFS='|' read -r -a PMC_SETS <<< "$PMC_SETS_STR"     # sets separated by |
else
    PMC_SETS=("FETCH_SIZE WRITE_SIZE" "OccupancyPercent MeanOccupancyPerCU" "L2CacheHit MemUnitStalled")
fi

NCU_METRICS="gpu__time_duration.sum,launch__grid_size,launch__block_size,\
launch__registers_per_thread,launch__shared_mem_per_block_static,\
sm__maximum_warps_per_active_cycle_pct,sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum"

# ----- submission (login node) -------------------------------------------------
submit_all() {
    local f gb st d wall
    local -a gopt cmd
    for f in "$SRC_CU" "$SRC_OMP"; do
        [[ -f $f ]] || { echo "ERROR: $f not found in $PWD" >&2; exit 1; }
    done
    read -r -a gopt <<< "$GPUS_OPT"
    for gb in $SIZES; do
        for st in $STAGES; do
            d="$RESULTS/$SITE/${DIRPRE}${gb}GB/rep${REP}/$st"
            wall="$WALL_TIME"; [[ $st == ncu ]] && wall="$WALL_NCU"
            local -a qopt=()
            [[ -n $QOS ]] && qopt=(--qos="$QOS")
            cmd=(sbatch --job-name="${JOBPRE}-${SITE}-${gb}GB-${st}"
                 --account="$ACCOUNT" --partition="$PARTITION" ${qopt[@]+"${qopt[@]}"}
                 --nodes=1 --exclusive "${gopt[@]}" --time="$wall"
                 --output="$d/slurm-%j.out"
                 --export="ALL,MM_GB=$gb,MM_STAGE=$st,SITE=$SITE,REP=$REP,BENCH=$BENCH"
                 "$0")
            if [[ ${DRY_RUN:-0} == 1 ]]; then
                echo "${cmd[*]}"
            else
                mkdir -p "$d"
                "${cmd[@]}"
            fi
        done
    done
}

BACKFILL="${BACKFILL:-0}"
if [[ -z "${SLURM_JOB_ID:-}" && $BACKFILL != 1 ]]; then
    submit_all
    exit 0
fi

# ----- from here on: running on a compute node ---------------------------------
if [[ -n "${MM_GB:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1      # batch job from submit_all
fi
SRC_DIR=$PWD

module purge >/dev/null 2>&1
for m in $MODULES; do
    if ! module load "$m" >/dev/null 2>&1; then
        echo "ERROR: module load $m failed -- check 'module spider' on $SITE" >&2
        exit 1
    fi
done

# Find a tool on PATH, or inside the CUDA installation that provides nvcc.
find_tool() {
    local name=$1 p root c
    if p=$(command -v "$name" 2>/dev/null); then echo "$p"; return 0; fi
    p=$(command -v nvcc 2>/dev/null) || return 1
    root=$(dirname "$(dirname "$p")")
    for c in "$root/bin/$name" "$root"/nsight-systems-*/bin/"$name" "$root"/nsight-compute-*/"$name"; do
        [[ -x $c ]] && { echo "$c"; return 0; }
    done
    return 1
}

if [[ $VENDOR == amd ]]; then
    # Native: hipcc (or the Cray wrapper with -x hip). OpenMP: the Cray C++ wrapper
    # under PrgEnv-amd, else amdclang++.
    NVCC="${HIPCC:-}"
    if [[ -z $NVCC ]]; then
        if command -v hipcc >/dev/null; then NVCC=$(command -v hipcc)
        elif command -v CC  >/dev/null; then NVCC="$(command -v CC) -x hip"
        else echo "ERROR: neither hipcc nor CC found (load rocm / PrgEnv-amd)" >&2; exit 1; fi
    fi
    CLANGXX="${OMPCXX:-}"
    if [[ -z $CLANGXX ]]; then
        if   command -v CC          >/dev/null; then CLANGXX=$(command -v CC)
        elif command -v amdclang++  >/dev/null; then CLANGXX=$(command -v amdclang++)
        else echo "ERROR: neither CC nor amdclang++ found" >&2; exit 1; fi
    fi
    ROCPROF="${ROCPROF:-}"
    if [[ -z $ROCPROF ]]; then
        if   command -v rocprofv3 >/dev/null; then ROCPROF=$(command -v rocprofv3)
        elif command -v rocprof   >/dev/null; then ROCPROF=$(command -v rocprof)
        else echo "WARN: no rocprofv3/rocprof found; kernel traces will be missing" >&2; fi
    fi
    ROCPROF_V3=0
    if [[ -n $ROCPROF ]]; then
        [[ $(basename "$ROCPROF") == rocprofv3 ]] && ROCPROF_V3=1
        "$ROCPROF" --help 2>&1 | grep -q -- '--kernel-trace' && ROCPROF_V3=1
    fi
    command -v rocm-smi >/dev/null || echo "WARN: rocm-smi not found (device info incomplete)" >&2
else
NVCC=$(find_tool nvcc)       || { echo "ERROR: nvcc not found" >&2; exit 1; }
CLANGXX=$(find_tool clang++) || { echo "ERROR: clang++ not found" >&2; exit 1; }
NSYS="${NSYS:-$(find_tool nsys)}" || { echo "ERROR: nsys not found; set NSYS=/path/to/nsys" >&2; exit 1; }
if [[ $BACKFILL != 1 ]]; then
NCU="${NCU:-$(find_tool ncu)}"    || { echo "ERROR: ncu not found; set NCU=/path/to/ncu" >&2; exit 1; }
command -v nvidia-smi >/dev/null  || { echo "ERROR: nvidia-smi not found" >&2; exit 1; }

# GPU metrics are sampled on ALL GPUs of the node: nsys numbers GPUs its own
# way, so "device 0" may not be the GPU the programs use. gm_extract then
# keeps, for each kernel, the GPU that was busy (highest "SMs Active").
# nsys renamed --gpu-metrics-device to --gpu-metrics-devices; support both.
GM_OPT=""
nsys_help=$("$NSYS" profile --help 2>&1)
if   grep -q -- '--gpu-metrics-devices' <<< "$nsys_help"; then GM_OPT="--gpu-metrics-devices=all"
elif grep -q -- '--gpu-metrics-device'  <<< "$nsys_help"; then GM_OPT="--gpu-metrics-device=all"
fi
fi   # not BACKFILL
fi   # vendor

if [[ -z "${ARCH:-}" ]]; then
    if [[ $VENDOR == amd ]]; then
        ARCH=$(rocminfo 2>/dev/null | awk '/^ *Name: *gfx/ { print $2; exit }')
        [[ -n $ARCH ]] || ARCH=$(rocm_agent_enumerator 2>/dev/null | grep -m1 '^gfx')
        [[ -n $ARCH ]] || { echo "ERROR: cannot detect the GPU architecture; set ARCH=gfx90a" >&2; exit 1; }
    else
        cc=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
        [[ -n $cc ]] || { echo "ERROR: cannot detect compute capability; set ARCH=sm_XX" >&2; exit 1; }
        ARCH="sm_${cc}"
    fi
fi


# One GPU, addressed by its PCI order, so CUDA, OpenMP, nvidia-smi and the
# nsys GPU-metrics device all refer to the same physical GPU.
if [[ $VENDOR == amd ]]; then
    export ROCR_VISIBLE_DEVICES="$GPU_INDEX"   # one GCD (a GCD is a separate device)
    export HIP_VISIBLE_DEVICES=0
else
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_ON_NODE:-16}}"   # host init + CPU check
export OMP_PROC_BIND=close OMP_PLACES=cores

# ----- helpers -------------------------------------------------------------------
write_meta() {
    local d=$1 gb=$2 N=$3 stage=$4
    {
        echo "site: $SITE"
        echo "bench: $BENCH"
        echo "stage: $stage"
        echo "size_gb: $gb"
        echo "N: $N"
        echo "rep: $REP"
        echo "reps: $REPS"
        echo "date: $(date -Is)"
        echo "host: $(hostname)"
        echo "job: ${SLURM_JOB_ID:-none}"
        echo "gpu_index: $GPU_INDEX"
        echo "arch: $ARCH"
        if [[ $VENDOR == amd ]]; then
            echo "gpu: $(rocm-smi --showproductname --csv 2>/dev/null | awk -F, 'NR==2 {print $2}' | tr -d '\r')"
            echo "gpu_info: $(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2 {print $2" B VRAM"}' | tr -d '\r'), ROCm ${ROCM_VERSION:-${ROCM_PATH:-unknown}}"
        else
            echo "gpu: $(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader | head -1)"
            echo "gpu_info: $(nvidia-smi -i "$GPU_INDEX" --query-gpu=driver_version,memory.total,clocks.max.sm,clocks.max.mem --format=csv,noheader | head -1)"
        fi
        echo "cpu: $(lscpu 2>/dev/null | awk -F: '/Model name/ {sub(/^ +/, "", $2); print $2; exit}')"
        echo "omp_num_threads: $OMP_NUM_THREADS"
        echo "modules: $MODULES"
        echo "vendor: $VENDOR"
        [[ -n ${PEAK_BW_GBS:-} ]] && echo "peak_bw_gbs: $PEAK_BW_GBS"
        echo "native_compiler: $NVCC -- $($NVCC --version 2>&1 | head -1)"
        echo "omp_compiler: $CLANGXX -- $("$CLANGXX" --version 2>&1 | head -1)"
        if [[ $VENDOR == amd ]]; then
            echo "rocprof: ${ROCPROF:-none} $([[ -n ${ROCPROF:-} ]] && $ROCPROF --version 2>&1 | head -1)"
        else
            echo "nsys: $("$NSYS" --version 2>&1 | head -1)"
            echo "ncu: $("$NCU" --version 2>&1 | tail -1)"
        fi
        echo "gpu_metrics_option: ${GM_OPT:-none}"
        (cd "$d/src" && sha256sum "$SRC_CU" "$SRC_OMP") | sed 's/^/sha256: /'
        echo "--- module list"
        module list 2>&1
    } > "$d/meta.txt"
}

# CUDA is always built. The OpenMP binary is built only where it is used (the
# time stage, or ncu with NCU_OMP=1); if its build fails, the CUDA work still
# runs and only the OpenMP runs are skipped.
build() {
    local d=$1 stage=$2
    local -a nvflags ompflags
    if [[ $VENDOR == amd ]]; then
        # -ffp-contract=off is the AMD counterpart of nvcc's --fmad=false, so both
        # models round identically. -Rpass-analysis=kernel-resource-usage makes the
        # compiler report VGPRs/SGPRs, LDS, scratch and waves per SIMD per kernel;
        # that is where the register and occupancy rows come from on this vendor.
        nvflags=(-O3 --offload-arch="$ARCH" -ffp-contract=off -fopenmp
                 -Rpass-analysis=kernel-resource-usage -g1)
        ompflags=(-O3 -fopenmp --offload-arch="$ARCH" -ffp-contract=off
                  -Rpass-analysis=kernel-resource-usage -g1)
    else
        nvflags=(-O3 -arch="$ARCH" --fmad=false -lineinfo -Xptxas -v
                 -Xcompiler -fopenmp -Xcompiler -ffp-contract=off)
        # shellcheck disable=SC2054  # the comma belongs to -Wl,--no-warn-execstack
        ompflags=(-O3 -fopenmp --offload-arch="$ARCH" -ffp-contract=off
                  -gline-tables-only -Wl,--no-warn-execstack)
    fi
    local natlog="build_${NAT}.log" natlib=(-lgomp)
    [[ $VENDOR == amd ]] && natlib=(-lm)
    # shellcheck disable=SC2086  # $NVCC may be "CC -x hip"
    if ! ( cd "$d" && { echo "\$ $NVCC ${nvflags[*]} -o $BIN_CU src/$SRC_CU ${natlib[*]}"
                        $NVCC "${nvflags[@]}" -o "$BIN_CU" "src/$SRC_CU" "${natlib[@]}"; } > "$natlog" 2>&1 ); then
        echo "ERROR: native ($NAT) build failed, see $d/$natlog" >&2
        return 1
    fi
    if [[ $stage == time || $NCU_OMP == 1 || ( $VENDOR == amd && $PMC_OMP == 1 ) ]]; then
        if ! ( cd "$d" && { echo "\$ clang++ ${ompflags[*]} -o $BIN_OMP src/$SRC_OMP"
                            "$CLANGXX" "${ompflags[@]}" -o "$BIN_OMP" "src/$SRC_OMP"; } > build_omp.log 2>&1 ); then
            echo "ERROR: OpenMP build failed, see $d/build_omp.log -- OpenMP runs skipped" >&2
            rm -f "$d/$BIN_OMP"
        fi
    fi
    return 0
}

# Exact per-kernel launch resources as recorded by the driver: registers,
# static/dynamic shared memory in bytes, local memory per thread (spills).
# The nsys CSV rounds shared memory to 0.001 MB, hence this extra table.
kern_extract() {
    local db=$1 out=$2
    command -v python3 >/dev/null || return 0
    [[ -f $db ]] || return 0
    python3 - "$db" "$out" <<'PY'
import csv, sqlite3, sys
db, out = sys.argv[1], sys.argv[2]
cur = sqlite3.connect(db).cursor()
cols = {r[1] for r in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
name = "COALESCE(k.mangledName, k.shortName)" if "mangledName" in cols else "k.shortName"
loc = "k.localMemoryPerThread" if "localMemoryPerThread" in cols else "NULL"
rows = cur.execute(
    "SELECT s.value, k.gridX, k.gridY, k.blockX, k.blockY, k.registersPerThread, "
    "k.staticSharedMemory, k.dynamicSharedMemory, " + loc + ", COUNT(*) "
    "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = " + name + " "
    "GROUP BY s.value, k.gridX, k.gridY, k.blockX, k.blockY").fetchall()
with open(out, "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["name", "gridX", "gridY", "blockX", "blockY", "regs",
                "static_smem_B", "dyn_smem_B", "local_B", "launches"])
    w.writerows(rows)
print("kernel resources: %d kernel configurations written to %s" % (len(rows), out))
PY
}

# Average every sampled GPU metric over each kernel longer than 1 ms.
# awk cannot read SQLite, hence the few lines of Python.
gm_extract() {
    local db=$1 out=$2
    command -v python3 >/dev/null || { echo "WARN: python3 not found, GPU metrics not extracted"; return 0; }
    [[ -f $db ]] || { echo "WARN: $db not found, GPU metrics not extracted"; return 0; }
    python3 - "$db" "$out" <<'PY'
import csv, sqlite3, sys
db, out = sys.argv[1], sys.argv[2]
con = sqlite3.connect(db)
cur = con.cursor()
tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
with open(out, "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["start_ns", "end_ns", "gridX", "gridY", "blockX", "blockY",
                "name", "metric", "value", "samples", "gpu"])
    need = {"GPU_METRICS", "TARGET_INFO_GPU_METRICS", "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
    if not need <= tables:
        print("no GPU metrics in", db, "- missing tables:", sorted(need - tables))
        sys.exit(0)
    names = dict(cur.execute("SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
    kcols = {r[1] for r in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    namecol = "mangledName" if "mangledName" in kcols else "shortName"
    kernels = cur.execute(
        "SELECT k.start, k.end, k.gridX, k.gridY, k.blockX, k.blockY, s.value "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id = k." + namecol + " "
        "WHERE k.end - k.start > 1000000 ORDER BY k.start").fetchall()
    gcols = {r[1] for r in cur.execute("PRAGMA table_info(GPU_METRICS)")}
    src = "typeId" if "typeId" in gcols else "0"          # typeId separates the sampled GPUs
    cur.execute("CREATE TEMP TABLE gm AS SELECT timestamp, " + src + " AS gpu, metricId, value FROM GPU_METRICS")
    cur.execute("CREATE INDEX temp.gm_ts ON gm(timestamp)")
    busy_ids = [m for m, nm in names.items() if "sms active" in str(nm).lower()]
    chosen = {}
    for start, end, gx, gy, bx, by, name in kernels:
        rows = cur.execute("SELECT gpu, metricId, AVG(value), COUNT(*) FROM gm "
                           "WHERE timestamp BETWEEN ? AND ? GROUP BY gpu, metricId", (start, end)).fetchall()
        if not rows:
            continue
        score = {}
        for g, mid, avg, n in rows:                        # busiest GPU during this kernel
            if (mid in busy_ids) or (not busy_ids and "%" in str(names.get(mid, ""))):
                score[g] = score.get(g, 0.0) + avg
        best = max(score, key=score.get) if score else rows[0][0]
        chosen[best] = chosen.get(best, 0) + 1
        for g, mid, avg, n in rows:
            if g == best:
                w.writerow([start, end, gx, gy, bx, by, name, names.get(mid, mid), "%.4f" % avg, n, g])
print("GPU metrics: %d kernels written to %s; sampled GPU(s) used: %s" % (len(kernels), out, chosen))
PY
}

# Run one program under nsys (with GPU metrics if possible), export CSVs.
run_nsys() {
    local tag=$1; shift
    local -a base=(profile --trace=cuda --sample=none --cpuctxsw=none
                   --force-overwrite=true -o "nsys_$tag")
    local -a gm=()
    [[ -n $GM_OPT ]] && gm=("$GM_OPT" "--gpu-metrics-frequency=$GM_FREQ")

    "$NSYS" "${base[@]}" ${gm[@]+"${gm[@]}"} "$@" > "$tag.out" 2> "$tag.nsys.log"
    if ! grep -q '^launch overhead' "$tag.out" && (( ${#gm[@]} )); then
        echo "WARN: $tag did not complete with GPU metrics; retrying without" | tee -a "$tag.nsys.log" >&2
        "$NSYS" "${base[@]}" "$@" > "$tag.out" 2>> "$tag.nsys.log"
    fi
    if ! grep -q '^launch overhead' "$tag.out"; then
        echo "ERROR: $tag run did not complete, see $PWD/$tag.out and $tag.nsys.log" >&2
        return 1
    fi
    "$NSYS" stats --report cuda_gpu_trace --report cuda_api_sum --format csv \
            --force-export=true --output "$tag" "nsys_$tag.nsys-rep" >> "$tag.nsys.log" 2>&1
    kern_extract "nsys_$tag.sqlite" "kern_$tag.csv" >> "$tag.nsys.log" 2>&1
    gm_extract "nsys_$tag.sqlite" "gm_$tag.csv" >> "$tag.nsys.log" 2>&1
    [[ $KEEP_SQLITE == 1 ]] || rm -f "nsys_$tag.sqlite"
}

# --- AMD: one program run under rocprofv3, kernel trace to CSV -------------
# rocprofv3 writes <out>_kernel_trace.csv (rocprof v1: <out>.csv); both carry
# per-dispatch timestamps, grid and workgroup size, LDS and scratch.
run_rocprof() {
    local tag=$1; shift
    if [[ -z ${ROCPROF:-} ]]; then
        "$@" > "$tag.out" 2> "$tag.prof.log"
        echo "WARN: no rocprof, $tag has program output only" >> "$tag.prof.log"
        return
    fi
    rm -rf "prof_$tag"; mkdir -p "prof_$tag"
    if (( ROCPROF_V3 )); then
        "$ROCPROF" --kernel-trace --output-format csv -d "prof_$tag" -o "$tag" -- "$@" \
            > "$tag.out" 2> "$tag.prof.log"
    else
        "$ROCPROF" --timestamp on --stats -o "prof_$tag/$tag.csv" "$@" > "$tag.out" 2> "$tag.prof.log"
    fi
    # the CSV may sit in a subdirectory and carry a PID prefix: take the newest match
    local f
    f=$(find "prof_$tag" -name '*kernel_trace.csv' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    [[ -z $f ]] && f=$(find "prof_$tag" -name '*.csv' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    [[ -n $f ]] && cp "$f" "./rocp_${tag}_kernel_trace.csv"
    grep -q '^launch overhead' "$tag.out" || echo "ERROR: $tag run did not complete, see $PWD/$tag.out" >&2
}

# --- AMD: hardware counters for the native kernels (ncu counterpart) --------
# FETCH_SIZE/WRITE_SIZE (KB moved to and from HBM) give DRAM throughput;
# MeanOccupancyPerCU, where available, gives achieved occupancy.
stage_pmc() {
    local N=$1
    [[ -n ${ROCPROF:-} ]] || { echo "ERROR: no rocprof, PMC stage skipped" >&2; return 1; }
    (( ROCPROF_V3 )) && "$ROCPROF" --list-avail > pmc_available.txt 2>&1   # for diagnosis
    : > pmc.out
    pmc_one "$NAT" "./$BIN_CU" "$N"
    if [[ $PMC_OMP == 1 ]]; then
        if [[ -x $BIN_OMP ]]; then
            ( export OMP_TARGET_OFFLOAD=MANDATORY; pmc_one omp "./$BIN_OMP" "$N" )
        else
            echo "ERROR: no $BIN_OMP in the ncu stage; OpenMP counters skipped" >&2
        fi
    fi
}

# One program, one rocprof run per counter set; the CSVs are concatenated.
# The counter CSV also reports VGPR_Count, SGPR_Count, LDS_Block_Size and
# Scratch_Size per dispatch, which is where the AMD resource rows come from.
pmc_one() {
    local tag=$1 bin=$2 N=$3 set f ok=0 i=0
    # One set per run: hardware counter slots are limited, and an unknown name
    # makes the whole run fail, so the sets are tried independently.
    for set in "${PMC_SETS[@]}"; do
        i=$((i + 1))
        rm -rf "pmc_${tag}_run$i"; mkdir -p "pmc_${tag}_run$i"
        { echo "=== $tag counters: $set"
          if (( ROCPROF_V3 )); then
              # shellcheck disable=SC2086  # $set is a list of counter names
              "$ROCPROF" --pmc $set --output-format csv -d "pmc_${tag}_run$i" -o "pmc$i" -- "$bin" "$N" 1
          else
              printf 'pmc: %s\n' "$set" > "pmc_${tag}_run$i/in.txt"
              "$ROCPROF" -i "pmc_${tag}_run$i/in.txt" --timestamp on -o "pmc_${tag}_run$i/pmc$i.csv" "$bin" "$N" 1
          fi; } >> pmc.out 2>&1
        f=$(find "pmc_${tag}_run$i" -name '*counter_collection.csv' 2>/dev/null | head -1)
        [[ -z $f ]] && f=$(find "pmc_${tag}_run$i" -name '*.csv' ! -name '*kernel_trace.csv' \
                                ! -name '*agent_info.csv' 2>/dev/null | head -1)
        if [[ -n $f && -s $f ]]; then
            if (( ok )); then tail -n +2 "$f" >> "pmc_${tag}_counter_collection.csv"
            else cp "$f" "pmc_${tag}_counter_collection.csv"; ok=1; fi
        else
            echo "WARN: $tag counter set '$set' produced nothing (see pmc.out)" >&2
        fi
    done
    (( ok )) || echo "ERROR: no counters collected for $tag; check pmc.out and pmc_available.txt" >&2
}

stage_time_amd() {
    local N=$1 smi
    rocm-smi --showallinfo > smi_before.txt 2>&1
    rocm-smi --showuse --showmemuse --showpower --showtemp --csv -l 5 > clocks.csv 2>&1 &
    smi=$!
    if [[ -x $BIN_OMP ]]; then
        LIBOMPTARGET_INFO=16 OMP_TARGET_OFFLOAD=MANDATORY "./$BIN_OMP" 1024 1 \
            > omp_info_run.out 2> omp_info_raw.txt
        awk '/Launching kernel/ && !seen[$7]++' omp_info_raw.txt > omp_info.txt
        rm -f omp_info_raw.txt
    fi
    run_rocprof "$NAT" "./$BIN_CU" "$N" "$REPS"
    if [[ -x $BIN_OMP ]]; then
        ( export OMP_TARGET_OFFLOAD=MANDATORY; run_rocprof omp "./$BIN_OMP" "$N" "$REPS" )
    else
        echo "ERROR: no $BIN_OMP (OpenMP build failed); OpenMP runs skipped" >&2
    fi
    kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
    rocm-smi --showallinfo > smi_after.txt 2>&1
}

stage_time() {
    local N=$1 smi
    nvidia-smi -i "$GPU_INDEX" -q -d CLOCK,PERFORMANCE > smi_before.txt 2>&1
    nvidia-smi -i "$GPU_INDEX" --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw \
               --format=csv -l 5 > clocks.csv 2>&1 &
    smi=$!

    if [[ -x $BIN_OMP ]]; then
        # Kernel mode (SPMD/generic) and launch shape chosen by the OpenMP runtime.
        LIBOMPTARGET_INFO=16 OMP_TARGET_OFFLOAD=MANDATORY "./$BIN_OMP" 1024 1 \
            > omp_info_run.out 2> omp_info_raw.txt
        awk '/Launching kernel/ && !seen[$7]++' omp_info_raw.txt > omp_info.txt   # one line per kernel
        rm -f omp_info_raw.txt
    fi

    run_nsys cuda "./$BIN_CU" "$N" "$REPS"
    if [[ -x $BIN_OMP ]]; then
        ( export OMP_TARGET_OFFLOAD=MANDATORY; run_nsys omp "./$BIN_OMP" "$N" "$REPS" )
    else
        echo "ERROR: no $BIN_OMP (OpenMP build failed); OpenMP runs skipped" >&2
    fi

    kill "$smi" 2>/dev/null; wait "$smi" 2>/dev/null
    nvidia-smi -i "$GPU_INDEX" -q -d CLOCK,PERFORMANCE > smi_after.txt 2>&1
}

stage_ncu() {
    local N=$1
    # --filter-mode per-launch-config: the skip/count apply to each launch
    # configuration separately, so each of the three is profiled on its timed
    # launch (the warm-up launch is skipped). reps=1 keeps the unprofiled runs short.
    "$NCU" --metrics "$NCU_METRICS" -k "regex:$NCU_REGEX" \
           --filter-mode per-launch-config --launch-skip 1 --launch-count 1 \
           -f -o ncu_cuda "./$BIN_CU" "$N" 1 > ncu_cuda.out 2>&1
    if [[ -f ncu_cuda.ncu-rep ]]; then
        "$NCU" -i ncu_cuda.ncu-rep --page details --csv > ncu_cuda.csv 2>> ncu_cuda.out
    else
        echo "ERROR: ncu wrote no report, see $PWD/ncu_cuda.out" >&2
    fi
    if [[ $NCU_OMP == 1 && -x $BIN_OMP ]]; then
        ( export OMP_TARGET_OFFLOAD=MANDATORY
          "$NCU" --metrics "$NCU_METRICS" -k "regex:$OMP_KERNEL" \
                 --launch-skip 1 --launch-count 1 \
                 -f -o ncu_omp "./$BIN_OMP" "$N" 1 > ncu_omp.out 2>&1 )
        [[ -f ncu_omp.ncu-rep ]] && "$NCU" -i ncu_omp.ncu-rep --page details --csv > ncu_omp.csv
    fi
}

run_stage() {
    local gb=$1 stage=$2 d N total foot
    d="$RESULTS/$SITE/${DIRPRE}${gb}GB/rep${REP}/$stage"
    echo "=== $BENCH $SITE ${gb}GB rep$REP stage=$stage on $(hostname) at $(date) ==="

    if [[ $BENCH == matmul ]]; then
        N=$(awk -v g="$gb" 'BEGIN { printf "%d", int(sqrt(g * 1e9 / 24) / 128) * 128 }')
        foot=$(awk -v n="$N" 'BEGIN { printf "%.0f", 24 * n * n }')
    elif [[ $BENCH == matvec ]]; then
        N=$(awk -v g="$gb" 'BEGIN { printf "%d", int(sqrt(g * 1e9 / 8) / 128) * 128 }')
        foot=$(awk -v n="$N" 'BEGIN { printf "%.0f", 8 * n * n + 16 * n }')
    else
        N=$(awk -v g="$gb" 'BEGIN { printf "%d", int(g * 1e9 / 24 / 128) * 128 }')
        foot=$(awk -v n="$N" 'BEGIN { printf "%.0f", 24 * n }')
    fi
    (( N > 0 )) || { echo "ERROR: ${gb} GB gives N=0" >&2; return 1; }
    if [[ $VENDOR == amd ]]; then
        total=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2 {printf "%d", $2/1048576}')
        [[ -n $total && $total -gt 0 ]] || total="${GPU_MEM_MB:-65536}"     # MI250X GCD: 64 GB
    else
        total=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    fi
    if ! awk -v f="$foot" -v t="$total" 'BEGIN { exit !(f <= 0.92 * t * 1048576) }'; then
        echo "ERROR: ${gb} GB (N=$N) exceeds 92% of ${total} MiB; skipping" >&2
        return 1
    fi

    mkdir -p "$d/src" && d=$(cd "$d" && pwd)
    cp "$SRC_DIR/$SRC_CU" "$SRC_DIR/$SRC_OMP" "$d/src/" || return 1
    write_meta "$d" "$gb" "$N" "$stage"
    build "$d" "$stage" || return 1
    echo "  N=$N  arch=$ARCH  results in $d"

    ( cd "$d" || exit 1
      case "$stage" in
          time) if [[ $VENDOR == amd ]]; then stage_time_amd "$N"; else stage_time "$N"; fi ;;
          ncu)  if [[ $VENDOR == amd ]]; then stage_pmc "$N";      else stage_ncu  "$N"; fi ;;
          *)    echo "ERROR: unknown stage '$stage'" >&2; exit 1 ;;
      esac )
    echo "=== done ${gb}GB $stage at $(date) ==="
}

if [[ $BACKFILL == 1 ]]; then
    n=0
    for d in "$RESULTS/$SITE"/*GB/rep*/time; do
        for t in cuda omp; do
            [[ -f $d/nsys_$t.nsys-rep ]] || continue
            "$NSYS" export --type sqlite --force-overwrite=true -o "$d/nsys_$t.sqlite" \
                    "$d/nsys_$t.nsys-rep" > /dev/null 2>&1
            kern_extract "$d/nsys_$t.sqlite" "$d/kern_$t.csv" | sed "s#^#$d: #"
            [[ $KEEP_SQLITE == 1 ]] || rm -f "$d/nsys_$t.sqlite"
            n=$((n + 1))
        done
    done
    echo "Backfill done: $n reports under $RESULTS/$SITE. Now run ./collect.sh"
elif [[ -n "${MM_GB:-}" ]]; then
    run_stage "$MM_GB" "${MM_STAGE:-time}"
else
    echo "Interactive allocation detected: running all sizes and stages here."
    for gb in $SIZES; do
        for st in $STAGES; do
            run_stage "$gb" "$st"
        done
    done
fi
