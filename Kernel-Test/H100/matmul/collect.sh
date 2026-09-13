#!/bin/bash
# collect.sh -- turn results/ into one table per dataset, plus
#               analysis/summary.csv       every number, one row per kernel and dataset
#               analysis/plot_<site>.dat   one row per size, read by plot.gp
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
    local dir=$1 st so status f missing
    for st in time ncu; do
        [[ -d $dir/$st ]] || { echo "  [$st] not run"; continue; }
        so=$(find "$dir/$st" -maxdepth 1 -name 'slurm-*.out' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
        if   [[ -z $so ]];                              then status="no Slurm log (interactive run)"
        elif grep -qi 'CANCELLED' "$so";                then status="CANCELLED: $(grep -i -m1 'CANCELLED' "$so" | sed 's/.*CANCELLED/CANCELLED/')"
        elif grep -q '^=== done' "$so";                 then status="finished"
        elif grep -q 'ERROR' "$so";                     then status="FAILED (errors below, log: $so)"
        else                                                 status="NOT finished: still running, or died (see $so)"
        fi
        missing=""
        if [[ $st == time ]]; then
            for f in matmul_cuda matmul_omp cuda.out omp.out omp_info.txt cuda_cuda_gpu_trace.csv \
                     omp_cuda_gpu_trace.csv gm_cuda.csv gm_omp.csv; do
                [[ -s $dir/$st/$f ]] || missing+=" $f"
            done
        else
            for f in matmul_cuda ncu_cuda.csv; do [[ -s $dir/$st/$f ]] || missing+=" $f"; done
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
    for f in time/meta.txt time/build_cuda.log time/cuda.out time/omp.out time/omp_info.txt \
             time/cuda_cuda_gpu_trace.csv time/omp_cuda_gpu_trace.csv \
             time/cuda_cuda_api_sum.csv time/omp_cuda_api_sum.csv \
             time/kern_cuda.csv time/kern_omp.csv \
             time/gm_cuda.csv time/gm_omp.csv ncu/ncu_cuda.csv ncu/ncu_omp.csv; do
        [[ -s $dir/$f ]] && files+=("$dir/$f")
    done
    site=$(awk -F': ' '$1 == "site" { print $2; exit }' "$dir/time/meta.txt")
    awk -f "$AWK_PROG" -v summary="$OUT/summary.csv" -v plotfile="$OUT/plot_${site}.dat" \
        -v plotrep="$PLOT_REP" -v gmfile="$dir/gm_summary.txt" "${files[@]}" | tee "$dir/table.txt"
    report_status "$dir"
    echo
done
(( found )) || { echo "No datasets under $RESULTS/ (expected $RESULTS/<site>/<GB>GB/rep<R>/time/meta.txt)" >&2; exit 1; }

# Plot files: header first, then rows sorted by size.
for p in "$OUT"/plot_*.dat; do
    [[ -f $p ]] || continue
    { head -1 "$p"; tail -n +2 "$p" | sort -g -k1,1; } > "$p.tmp" && mv "$p.tmp" "$p"
done
echo "Wrote $OUT/summary.csv and $(ls "$OUT"/plot_*.dat 2>/dev/null | tr '\n' ' ')"
