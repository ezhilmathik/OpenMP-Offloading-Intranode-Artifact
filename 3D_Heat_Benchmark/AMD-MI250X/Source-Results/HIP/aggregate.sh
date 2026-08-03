#!/bin/bash
# Merge HIP benchmark results into summary.csv.
#
# Default (recommended): use the solver's own "Time = ... seconds" value
# extracted directly from results_*/run.log.
#
#   ./aggregate.sh
#   ./aggregate.sh my-summary.csv
#   TIME_FIELD=wall ./aggregate.sh
#
# Output schema:
#   variant,gpus,N,samples,failed,min_sec,median_sec,max_sec,rel_spread_pct,steps
#
# The trailing "steps" column is what speedup.py reads to build the plot title
# ("~500 time steps"). Without it the title falls back to "time steps unknown".

set -euo pipefail

OUT="${1:-summary.csv}"
TIME_FIELD="${TIME_FIELD:-compute}"
shopt -s nullglob

if [[ "$TIME_FIELD" == "compute" ]]; then
    LOGS=(results_*/run.log)
    [[ ${#LOGS[@]} -gt 0 ]] || {
        echo "ERROR: no results_*/run.log files found." >&2
        exit 1
    }

    echo "Reading HIP solver times from ${#LOGS[@]} run.log file(s)..."
    awk '
    /^>>> Running:/ {
        variant=""; gpus=""; N=""; steps_done=""; steps_est=""; armed=0
        for (i=1; i<=NF; i++) {
            tok=$i
            sub(/^\.\//, "", tok)
            if (tok ~ /^[0-9]+-hip/) variant=tok
            if ($i ~ /^gpus=/) { split($i,a,"="); gpus=a[2] }
            if ($i ~ /^N=/)    { split($i,b,"="); N=b[2] }
        }
        if (variant != "" && gpus != "" && N != "") armed=1
        next
    }

    # "Estimated timesteps:     ~500"  -- the fallback, printed before the run.
    armed && /Estimated[[:space:]]+timesteps:/ {
        s=$NF; sub(/^~/, "", s); steps_est=s+0
        next
    }

    # "501 timesteps complete"  -- preferred: what the solver actually did.
    armed && /[0-9]+[[:space:]]+timesteps?[[:space:]]+complete/ {
        steps_done=$1+0
        next
    }

    armed && /^[[:space:]]*Time[[:space:]]*=/ {
        for (i=1; i<=NF; i++) {
            if ($i == "=" && (i+1) <= NF) {
                steps = (steps_done != "") ? steps_done : steps_est
                printf "%s,%s,%s,%s,%s\n", variant,gpus,N,$(i+1),steps
                armed=0
                break
            }
        }
    }
    ' "${LOGS[@]}" > ".compute.$$"

    ROWS=$(wc -l < ".compute.$$")
    [[ "$ROWS" -gt 0 ]] || {
        rm -f ".compute.$$"
        echo "ERROR: no solver 'Time = ... seconds' lines matched." >&2
        exit 1
    }
    echo "  timed runs  : $ROWS"

    NOSTEPS=$(awk -F, '$5 == "" { c++ } END { print c+0 }' ".compute.$$")
    [[ "$NOSTEPS" -eq 0 ]] || \
        echo "  WARNING     : $NOSTEPS run(s) had no detectable step count."

    awk -F, '
    function median(a,n,  i,j,t,m) {
        for (i=2; i<=n; i++) {
            t=a[i]
            for (j=i-1; j>=1 && a[j]>t; j--) a[j+1]=a[j]
            a[j+1]=t
        }
        m=int((n+1)/2)
        return (n%2) ? a[m] : (a[m]+a[m+1])/2.0
    }
    {
        key=$1 SUBSEP $2 SUBSEP $3
        variant[key]=$1; gpus[key]=$2; size[key]=$3
        if ($5 != "") steps[key]=$5
        count[key]++
        value[key SUBSEP count[key]]=$4+0
    }
    END {
        print "variant,gpus,N,samples,failed,min_sec,median_sec,max_sec,rel_spread_pct,steps"
        for (key in count) {
            n=count[key]
            delete a
            for (i=1; i<=n; i++) a[i]=value[key SUBSEP i]
            med=median(a,n); lo=a[1]; hi=a[n]
            spread=(med>0) ? 100.0*(hi-lo)/med : 0
            printf "%s,%s,%s,%d,0,%.6f,%.6f,%.6f,%.1f,%s\n", \
                   variant[key],gpus[key],size[key],n,lo,med,hi,spread,steps[key]
        }
    }
    ' ".compute.$$" > "$OUT.raw"

    rm -f ".compute.$$"
    head -n 1 "$OUT.raw" > "$OUT"
    tail -n +2 "$OUT.raw" | sort -t, -k1,1 -k3,3n >> "$OUT"
    rm -f "$OUT.raw"

    echo "  summary     : $OUT (compute)"
    echo
    if command -v column >/dev/null 2>&1; then
        column -s, -t < "$OUT"
    else
        cat "$OUT"
    fi
    exit 0
fi

if [[ "$TIME_FIELD" != "wall" && "$TIME_FIELD" != "reported" ]]; then
    echo "ERROR: TIME_FIELD must be compute, wall, or reported." >&2
    exit 1
fi

CSVS=(results_*/timings.csv)
[[ ${#CSVS[@]} -gt 0 ]] || {
    echo "ERROR: no results_*/timings.csv files found." >&2
    exit 1
}

echo "Merging ${#CSVS[@]} timings.csv file(s), TIME_FIELD=$TIME_FIELD..."
MERGED="all_timings.csv"
head -n 1 "${CSVS[0]}" > "$MERGED"
for file in "${CSVS[@]}"; do
    tail -n +2 "$file" >> "$MERGED"
done

awk -F, -v field="$TIME_FIELD" '
function median(a,n,  i,j,t,m) {
    for (i=2; i<=n; i++) {
        t=a[i]
        for (j=i-1; j>=1 && a[j]>t; j--) a[j+1]=a[j]
        a[j+1]=t
    }
    m=int((n+1)/2)
    return (n%2) ? a[m] : (a[m]+a[m+1])/2.0
}
NR==1 { next }
{
    # rocr_mask contains commas (for example 0,2,4,6), so a simple -F,
    # split produces extra fields. All fields after rocr_mask are shifted by
    # off=NF-12; recover their positions from the right-hand side.
    off=NF-12
    if (off < 0) off=0
    variant=$3; gpus=$5; N=$(7+off)
    nsteps=$(9+off); wall=$(10+off); reported=$(11+off); status=$NF
    key=variant SUBSEP gpus SUBSEP N
    name[key]=variant; ngpu[key]=gpus; size[key]=N; seen[key]=1
    if (nsteps != "" && nsteps != "unknown") steps[key]=nsteps

    if (status != "ok") {
        failed[key]++
        next
    }

    t=(field=="wall") ? wall : reported
    if (t=="N/A" || t=="") {
        failed[key]++
        next
    }

    count[key]++
    value[key SUBSEP count[key]]=t+0
}
END {
    print "variant,gpus,N,samples,failed,min_sec,median_sec,max_sec,rel_spread_pct,steps"
    for (key in seen) {
        n=count[key]+0
        if (n==0) {
            printf "%s,%s,%s,0,%d,N/A,N/A,N/A,N/A,%s\n", \
                   name[key],ngpu[key],size[key],failed[key]+0,steps[key]
            continue
        }
        delete a
        for (i=1; i<=n; i++) a[i]=value[key SUBSEP i]
        med=median(a,n); lo=a[1]; hi=a[n]
        spread=(med>0) ? 100.0*(hi-lo)/med : 0
        printf "%s,%s,%s,%d,%d,%.6f,%.6f,%.6f,%.1f,%s\n", \
               name[key],ngpu[key],size[key],n,failed[key]+0,lo,med,hi,spread,steps[key]
    }
}
' "$MERGED" > "$OUT.raw"

head -n 1 "$OUT.raw" > "$OUT"
tail -n +2 "$OUT.raw" | sort -t, -k1,1 -k3,3n >> "$OUT"
rm -f "$OUT.raw"

echo "  merged rows : $(($(wc -l < "$MERGED") - 1)) -> $MERGED"
echo "  summary     : $OUT ($TIME_FIELD)"
echo
if command -v column >/dev/null 2>&1; then
    column -s, -t < "$OUT"
else
    cat "$OUT"
fi
