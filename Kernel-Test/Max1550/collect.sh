#!/bin/bash
# collect.sh -- turn results/ into one table per dataset, plus
#               analysis/summary.csv              every number, one row per kernel and dataset
#               analysis/overview.txt             CUDA vs OpenMP, one row per dataset, both benchmarks
#               analysis/plot_<site>.dat          matmul, one row per size, read by plot.gp
#               analysis/plot_<site>_vecadd.dat   vector addition
#               analysis/plot_<site>_matvec.dat   matrix-vector
#
# Usage:  ./collect.sh
#         RESULTS=results OUT=analysis PLOT_REP=1 ./collect.sh
#
# A dataset is results/<site>/<GB>GB/rep<R>/. Its table is printed and saved as
# table.txt in that directory; gm_summary.txt there lists every GPU metric.
# PLOT_REP picks which repetition goes into the plot files (summary.csv has all).
set -uo pipefail

RESULTS="${RESULTS:-results}"
OUT="${OUT:-analysis}"
PLOT_REP="${PLOT_REP:-1}"
AWK_PROG="$(cd "$(dirname "$0")" && pwd)/collect.awk"

awk --version 2>/dev/null | head -1 | grep -q 'GNU Awk' || {
    echo "ERROR: GNU awk (gawk) is required" >&2; exit 1; }
[[ -f $AWK_PROG ]] || { echo "ERROR: $AWK_PROG not found" >&2; exit 1; }

# Status of each stage: finished / cancelled / still running or failed,
# plus missing key files and any ERROR lines from the job log.
report_status() {
    local dir=$1 bin=$2 st so status f missing nat
    for st in time ncu; do
        [[ -d $dir/$st ]] || { echo "  [$st] not run"; continue; }
        so=$(find "$dir/$st" -maxdepth 1 -name 'slurm-*.out' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        if   [[ -z $so ]];                              then status="no Slurm log (interactive run)"
        elif grep -qi 'CANCELLED' "$so";                then status="CANCELLED: $(grep -i -m1 'CANCELLED' "$so" | sed 's/.*CANCELLED/CANCELLED/')"
        elif grep -q '^=== done' "$so";                 then status="finished"
        elif grep -q 'ERROR' "$so";                     then status="FAILED (errors below, log: $so)"
        else                                                 status="NOT finished: still running, or died (see $so)"
        fi
        nat=cuda
        [[ -e $dir/$st/${bin}_hip  || -s $dir/$st/build_hip.log  ]] && nat=hip
        [[ -e $dir/$st/${bin}_sycl || -s $dir/$st/build_sycl.log ]] && nat=sycl
        missing=""
        if [[ $st == time ]]; then
            if [[ $nat == sycl ]]; then
                for f in "${bin}_sycl" "${bin}_omp" sycl.out omp.out omp_info.txt omp_profile.txt; do
                    [[ -s $dir/$st/$f ]] || missing+=" $f"
                done
            elif [[ $nat == hip ]]; then
                for f in "${bin}_hip" "${bin}_omp" hip.out omp.out omp_info.txt \
                         rocp_hip_kernel_trace.csv rocp_omp_kernel_trace.csv; do
                    [[ -s $dir/$st/$f ]] || missing+=" $f"
                done
            else
                for f in "${bin}_cuda" "${bin}_omp" cuda.out omp.out omp_info.txt cuda_cuda_gpu_trace.csv \
                         omp_cuda_gpu_trace.csv gm_cuda.csv gm_omp.csv; do
                    [[ -s $dir/$st/$f ]] || missing+=" $f"
                done
            fi
        elif [[ $nat == sycl ]]; then
            for f in "${bin}_sycl" vtune_sycl.csv vtune_omp.csv; do [[ -s $dir/$st/$f ]] || missing+=" $f"; done
        elif [[ $nat == hip ]]; then
            for f in "${bin}_hip" pmc_hip_counter_collection.csv; do [[ -s $dir/$st/$f ]] || missing+=" $f"; done
        else
            for f in "${bin}_cuda" ncu_cuda.csv; do [[ -s $dir/$st/$f ]] || missing+=" $f"; done
        fi
        echo "  [$st] $status${missing:+ | missing:$missing}"
        [[ -n $so ]] && grep -h 'ERROR' "$so" | head -3 | sed 's/^/        /'
    done
}

mkdir -p "$OUT"
rm -f "$OUT/summary.csv" "$OUT"/plot_*.dat

found=0
for dir in "$RESULTS"/*/*GB/rep*; do
    [[ -f $dir/time/meta.txt ]] || continue
    found=1
    files=()
    for f in time/meta.txt time/build_cuda.log time/build_hip.log time/build_omp.log \
             time/cuda.out time/hip.out time/omp.out time/omp_info.txt \
             time/cuda_cuda_gpu_trace.csv time/omp_cuda_gpu_trace.csv \
             time/cuda_cuda_api_sum.csv time/omp_cuda_api_sum.csv \
             time/kern_cuda.csv time/kern_omp.csv \
             time/gm_cuda.csv time/gm_omp.csv \
             time/rocp_hip_kernel_trace.csv time/rocp_omp_kernel_trace.csv \
             time/sycl.out time/omp_profile.txt \
             ncu/ncu_cuda.csv ncu/ncu_omp.csv \
             ncu/build_hip.log ncu/pmc_hip_counter_collection.csv ncu/pmc_omp_counter_collection.csv \
             ncu/vtune_sycl.csv ncu/vtune_omp.csv \
             ncu/pmc_hip.csv; do
        [[ -s $dir/$f ]] && files+=("$dir/$f")
    done
    site=$(awk -F': ' '$1 == "site" { print $2; exit }' "$dir/time/meta.txt")
    bench=$(awk -F': ' '$1 == "bench" { print $2; exit }' "$dir/time/meta.txt")
    bench=${bench:-matmul}
    if [[ $bench == matmul ]]; then pf="$OUT/plot_${site}.dat"; else pf="$OUT/plot_${site}_${bench}.dat"; fi
    awk -f "$AWK_PROG" -v summary="$OUT/summary.csv" -v plotfile="$pf" \
        -v plotrep="$PLOT_REP" -v gmfile="$dir/gm_summary.txt" "${files[@]}" | tee "$dir/table.txt"
    case $bench in matmul) binpre=matmul ;; vecadd) binpre=vecadd ;; matvec) binpre=matvec ;; *) binpre=$bench ;; esac
    report_status "$dir" "$binpre"
    echo
done
(( found )) || { echo "No datasets under $RESULTS/ (expected $RESULTS/<site>/<GB>GB/rep<R>/time/meta.txt)" >&2; exit 1; }

# Plot files: header first, then rows sorted by size.
for p in "$OUT"/plot_*.dat; do
    [[ -f $p ]] || continue
    { head -1 "$p"; tail -n +2 "$p" | sort -g -k1,1; } > "$p.tmp" && mv "$p.tmp" "$p"
done
# Overview: CUDA vs OpenMP for every dataset of both benchmarks, from summary.csv.
# Reference CUDA kernel: naive for matmul (tiled32 as the second ratio), 1-D for vecadd.
awk -v FPAT='([^,]*)|("[^"]*")' '
    NR == 1 { for (i = 1; i <= NF; i++) H[$i] = i; next }
    function f(name) { v = $(H[name]); gsub(/"/, "", v); return v }
    {
        k = f("site") SUBSEP f("bench") SUBSEP f("size_gb") SUBSEP f("rep")
        if (!(k in seen)) { seen[k] = ++nk; KEY[nk] = k; GPU[k] = f("gpu"); NN[k] = f("N"); UNIT[k] = f("tp_unit") }
        im = f("impl")
        T[k, im] = f("t_med_ms"); TP[k, im] = f("throughput"); R[k, im] = f("regs")
        O[k, im] = f("occ_ach_nsys_pct"); D[k, im] = f("dram_nsys_pct"); L[k, im] = f("launch_wall_us")
    }
    function t(ms) { return (ms == "") ? "n/a" : (ms >= 1000) ? sprintf("%.2f s", ms / 1000) : (ms >= 1) ? sprintf("%.2f ms", ms) : sprintf("%.1f us", ms * 1000) }
    function p(a, b) { return (a == "" && b == "") ? "n/a" : sprintf("%s/%s", (a == "") ? "-" : sprintf("%.0f", a), (b == "") ? "-" : sprintf("%.0f", b)) }
    END {
        fmt = "%-9s %-7s %-22s %8s %11s %11s %11s %9s %9s %-8s %9s %9s %9s %11s\n"
        printf fmt, "site", "bench", "GPU", "size GB", "N", "CUDA time", "OMP time", "OMP/CUDA", "OMP/best", "unit", "tput C/O", "regs C/O", "occ% C/O", "DRAM% C/O"
        for (j = 1; j <= nk; j++) {                     # sort: site, bench, size, rep
            split(KEY[j], P, SUBSEP)
            SK[sprintf("%s|%s|%014.4f|%s", P[1], P[2], P[3], P[4])] = KEY[j]
        }
        ns = asorti(SK, ORD)
        for (j = 1; j <= ns; j++) {
            k = SK[ORD[j]]; split(k, P, SUBSEP)
            c = (P[2] == "vecadd") ? "cuda_vadd" : (P[2] == "matvec") ? "cuda_mvt" : "cuda_naive"; b = "cuda_tiled32"
            if (P[2] != "matmul") {                     # best = fastest CUDA kernel measured
                b = ""; nv = split((P[2] == "vecadd") ? "cuda_vadd cuda_v2d cuda_vvec" : "cuda_mvt cuda_mvw", VC, " ")
                for (q = 1; q <= nv; q++) if (T[k, VC[q]] > 0 && (b == "" || T[k, VC[q]] + 0 < T[k, b] + 0)) b = VC[q]
            }
            r1 = (T[k, c] > 0 && T[k, "omp"] != "") ? sprintf("%.2fx", T[k, "omp"] / T[k, c]) : "n/a"
            r2 = (T[k, b] > 0 && T[k, "omp"] != "") ? sprintf("%.2fx", T[k, "omp"] / T[k, b]) : "n/a"
            printf fmt, P[1], P[2], substr(GPU[k], 1, 22), P[3], NN[k], t(T[k, c]), t(T[k, "omp"]), r1, r2, UNIT[k],
                   p(TP[k, c], TP[k, "omp"]), p(R[k, c], R[k, "omp"]), p(O[k, c], O[k, "omp"]), p(D[k, c], D[k, "omp"])
        }
        print "CUDA = naive (matmul), 1-D (vecadd), thread/row (matvec): the counterpart of the OpenMP loop. OMP = your OpenMP kernel."
        print "best = tiled32 (matmul), fastest of 1-D / 2-D grid / double2 (vecadd), fastest of thread/row / warp/row (matvec)."
        print "Kernel times: profiler median (Nsight Systems on NVIDIA, rocprofv3 on AMD), warm-up excluded."
    }' "$OUT/summary.csv" > "$OUT/overview.txt"
echo "=== Overview: CUDA vs OpenMP ==="
cat "$OUT/overview.txt"
echo
echo "Wrote $OUT/summary.csv, $OUT/overview.txt and $(ls "$OUT"/plot_*.dat 2>/dev/null | tr '\n' ' ')"
