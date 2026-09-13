#!/bin/bash -l
# ===========================================================================
# run_matmul.sh -- CUDA (nvcc) vs OpenMP offload (Clang): dense matmul, vector
#                  addition and matrix-vector (BENCH=matmul | vecadd | matvec)
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
#   REPS        timed calls after warm-up     matmul 3, vecadd 100, matvec 20
#   REP         repetition label              1
#   GPU_INDEX   GPU used on the node          0
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
    case "$(hostname -s)" in
        alogin*|as[0-9]*) echo mn5; return ;;
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
        SIZES_MAT="10 20 30"
        ;;
    mn5)
        ACCOUNT="${ACCOUNT:-ehpc681}"
        PARTITION="${PARTITION:-acc}"
        QOS="${QOS:-acc_ehpc}"
        MODULES="${MODULES:-EB/apps GCC/13.2.0 cuda/12.8 clang/18.1.8-cuda12.8}"
        SIZES_MAT="30 45 55"
        ;;
    *)
        echo "ERROR: cannot tell the site; run with SITE=meluxina or SITE=mn5" >&2
        exit 1
        ;;
esac

# ----- benchmark --------------------------------------------------------------
BENCH="${BENCH:-matmul}"
case "$BENCH" in
    matmul)
        SRC_CU=mat-cuda.cu; SRC_OMP=mat-omp.cpp; BIN_CU=matmul_cuda; BIN_OMP=matmul_omp
        DIRPRE=""; JOBPRE=mm; REPS_DEF=3; GMF_DEF=200; WALL_DEF=03:00:00
        NCU_REGEX="dgemm_naive|dgemm_tiled32"; OMP_KERNEL=matrixMultiplyOpenMP
        SIZES="${SIZES:-$SIZES_MAT}"
        ;;
    vecadd)
        SRC_CU=vec-cuda.cu; SRC_OMP=vec-omp.cpp; BIN_CU=vecadd_cuda; BIN_OMP=vecadd_omp
        DIRPRE="vecadd-"; JOBPRE=va; REPS_DEF=100; GMF_DEF=2000; WALL_DEF=00:30:00
        NCU_REGEX="vadd_"; OMP_KERNEL=vectorAddOpenMP
        SIZES="${SIZES:-0.001 0.01 0.1 $SIZES_MAT}"
        ;;
    matvec)
        SRC_CU=mv-cuda.cu; SRC_OMP=mv-omp.cpp; BIN_CU=matvec_cuda; BIN_OMP=matvec_omp
        DIRPRE="matvec-"; JOBPRE="mv"; REPS_DEF=20; GMF_DEF=2000; WALL_DEF=00:30:00
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
            cmd=(sbatch --job-name="${JOBPRE}-${SITE}-${gb}GB-${st}"
                 --account="$ACCOUNT" --partition="$PARTITION" --qos="$QOS"
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

NVCC=$(find_tool nvcc)       || { echo "ERROR: nvcc not found" >&2; exit 1; }
CLANGXX=$(find_tool clang++) || { echo "ERROR: clang++ not found" >&2; exit 1; }
NSYS="${NSYS:-$(find_tool nsys)}" || { echo "ERROR: nsys not found; set NSYS=/path/to/nsys" >&2; exit 1; }
if [[ $BACKFILL != 1 ]]; then
NCU="${NCU:-$(find_tool ncu)}"    || { echo "ERROR: ncu not found; set NCU=/path/to/ncu" >&2; exit 1; }
command -v nvidia-smi >/dev/null  || { echo "ERROR: nvidia-smi not found" >&2; exit 1; }

if [[ -z "${ARCH:-}" ]]; then
    cc=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
    [[ -n $cc ]] || { echo "ERROR: cannot detect compute capability; set ARCH=sm_XX" >&2; exit 1; }
    ARCH="sm_${cc}"
fi

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

# One GPU, addressed by its PCI order, so CUDA, OpenMP, nvidia-smi and the
# nsys GPU-metrics device all refer to the same physical GPU.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
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
        echo "gpu: $(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader | head -1)"
        echo "gpu_info: $(nvidia-smi -i "$GPU_INDEX" --query-gpu=driver_version,memory.total,clocks.max.sm,clocks.max.mem --format=csv,noheader | head -1)"
        echo "cpu: $(lscpu 2>/dev/null | awk -F: '/Model name/ {sub(/^ +/, "", $2); print $2; exit}')"
        echo "omp_num_threads: $OMP_NUM_THREADS"
        echo "modules: $MODULES"
        echo "nvcc: $("$NVCC" --version 2>&1 | tail -1)"
        echo "clang: $("$CLANGXX" --version 2>&1 | head -1)"
        echo "nsys: $("$NSYS" --version 2>&1 | head -1)"
        echo "ncu: $("$NCU" --version 2>&1 | tail -1)"
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
    local -a nvflags=(-O3 -arch="$ARCH" --fmad=false -lineinfo -Xptxas -v
                      -Xcompiler -fopenmp -Xcompiler -ffp-contract=off)
    # shellcheck disable=SC2054  # the comma belongs to -Wl,--no-warn-execstack
    local -a ompflags=(-O3 -fopenmp --offload-arch="$ARCH" -ffp-contract=off
                       -gline-tables-only -Wl,--no-warn-execstack)
    if ! ( cd "$d" && { echo "\$ nvcc ${nvflags[*]} -o $BIN_CU src/$SRC_CU -lgomp"
                        "$NVCC" "${nvflags[@]}" -o "$BIN_CU" "src/$SRC_CU" -lgomp; } > build_cuda.log 2>&1 ); then
        echo "ERROR: CUDA build failed, see $d/build_cuda.log" >&2
        return 1
    fi
    if [[ $stage == time || $NCU_OMP == 1 ]]; then
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
    total=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
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
          time) stage_time "$N" ;;
          ncu)  stage_ncu  "$N" ;;
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
