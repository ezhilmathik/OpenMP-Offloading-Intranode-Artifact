#!/bin/bash
set -euo pipefail
OUT="${1:-summary.csv}"
TIME_FIELD="${TIME_FIELD:-compute}"
shopt -s nullglob

# Output schema:
#   variant,gpus,N,samples,failed,min_sec,median_sec,max_sec,rel_spread_pct,steps
# The trailing "steps" column is what speedup.py reads to build the plot title
# ("~500 time steps"). Without it the title falls back to "time steps unknown".

# =============================== compute mode ===============================
# The solver's own timer, straight out of run.log. reported_sec in timings.csv
# is unusable on old runs: the regex matched "Final time (T): 0.000135" from the
# parameter banner instead of "Time = 1.176864 seconds".
if [[ "$TIME_FIELD" == "compute" ]]; then
    LOGS=(results_*/run.log)
    [[ ${#LOGS[@]} -gt 0 ]] || { echo "ERROR: no results_*/run.log found." >&2; exit 1; }
    echo "Reading the solver's own timer from ${#LOGS[@]} run.log file(s)..."
    awk '
    /^>>> / {
        variant = ""; gpus = ""; N = ""; done_ = ""; est = ""; armed = 0
        for (i = 2; i <= NF; i++) {
            tok = $i; sub(/^\.\//, "", tok)
            if (tok ~ /^[0-9]+-omp/) variant = tok
            if ($i ~ /^gpus=/) { split($i, a, "="); gpus = a[2] }
            if ($i ~ /^N=/)    { split($i, b, "="); N    = b[2] }
        }
        if (variant != "" && gpus != "" && N != "") armed = 1
        next }
    # "Estimated timesteps:     ~500" -- fallback, printed before the run.
    armed && /Estimated[[:space:]]+timesteps:/ {
        s = $NF; sub(/^~/, "", s); est = s + 0; next }
    # "501 timesteps complete" -- preferred: what the solver actually did.
    armed && /[0-9]+[[:space:]]+timesteps?[[:space:]]+complete/ {
        done_ = $1 + 0; next }
    armed && /^Time = / {
        printf "%s,%s,%s,%s,%s\n", variant, gpus, N, $3,
               (done_ != "" ? done_ : est)
        armed = 0 }
    ' "${LOGS[@]}" > .compute.$$
    ROWS=$(wc -l < .compute.$$)
    [[ "$ROWS" -gt 0 ]] || { rm -f .compute.$$; echo "ERROR: no 'Time = ' lines matched." >&2; exit 1; }
    echo "  timed runs  : $ROWS"
    NOSTEPS=$(awk -F, '$5 == "" { c++ } END { print c+0 }' .compute.$$)
    [[ "$NOSTEPS" -eq 0 ]] || echo "  WARNING     : $NOSTEPS run(s) had no detectable step count."
    awk -F, '
    function median(arr, n,   i, j, t, m) {
        for (i = 2; i <= n; i++) { t = arr[i]
            for (j = i-1; j >= 1 && arr[j] > t; j--) arr[j+1] = arr[j]
            arr[j+1] = t }
        m = int((n+1)/2)
        return (n % 2) ? arr[m] : (arr[m] + arr[m+1]) / 2.0 }
    { key = $1 "," $2 "," $3; cnt[key]++; vals[key SUBSEP cnt[key]] = $4 + 0
      if ($5 != "") steps[key] = $5 }
    END {
        print "variant,gpus,N,samples,failed,min_sec,median_sec,max_sec,rel_spread_pct,steps"
        for (key in cnt) {
            n = cnt[key]; delete a
            for (i = 1; i <= n; i++) a[i] = vals[key SUBSEP i]
            med = median(a, n); lo = a[1]; hi = a[n]
            printf "%s,%d,0,%.4f,%.4f,%.4f,%.1f,%s\n", key, n, lo, med, hi,
                   (med > 0 ? 100.0*(hi-lo)/med : 0), steps[key] } }' .compute.$$ > "$OUT.raw"
    rm -f .compute.$$
    head -1 "$OUT.raw" > "$OUT"
    tail -n +2 "$OUT.raw" | sort -t, -k1,1 -k3,3n >> "$OUT"
    rm -f "$OUT.raw"
    echo "  summary     : $OUT  (compute)"
    echo ""
    if command -v column >/dev/null 2>&1; then column -s, -t < "$OUT"; else cat "$OUT"; fi
    exit 0
fi

CSVS=(results_*/timings.csv)
[[ ${#CSVS[@]} -gt 0 ]] || { echo "ERROR: no results_*/timings.csv here." >&2; exit 1; }
echo "Merging ${#CSVS[@]} file(s) on '${TIME_FIELD}' time..."
MERGED="all_timings.csv"
head -1 "${CSVS[0]}" > "$MERGED"
for f in "${CSVS[@]}"; do tail -n +2 "$f" >> "$MERGED"; done
awk -F, -v field="$TIME_FIELD" '
function median(arr, n,   i, j, t, m) {
    for (i = 2; i <= n; i++) { t = arr[i]
        for (j = i-1; j >= 1 && arr[j] > t; j--) arr[j+1] = arr[j]
        arr[j+1] = t }
    m = int((n+1)/2)
    return (n % 2) ? arr[m] : (arr[m] + arr[m+1]) / 2.0
}
NR == 1 { next }
{
    off = NF - 12; if (off < 0) off = 0
    variant = $3; gpus = $5
    N = $(7+off); nsteps = $(9+off); wall = $(10+off); reported = $(11+off); status = $NF
    key = variant "," gpus "," N
    seen[key] = 1
    if (nsteps != "" && nsteps != "unknown") steps[key] = nsteps
    if (status != "ok") { bad[key]++; why[key] = status; next }
    t = (field == "wall") ? wall : reported
    if (t == "N/A" || t == "") t = wall
    if (t == "N/A" || t == "") { bad[key]++; why[key] = "unparseable"; next }
    cnt[key]++; vals[key SUBSEP cnt[key]] = t + 0
}
END {
    print "variant,gpus,N,samples,failed,min_sec,median_sec,max_sec,rel_spread_pct,steps"
    for (key in seen) {
        n = cnt[key] + 0
        if (n == 0) { printf "%s,0,%d,N/A,N/A,N/A,%s,%s\n", key, bad[key]+0, why[key], steps[key]; continue }
        delete a
        for (i = 1; i <= n; i++) a[i] = vals[key SUBSEP i]
        med = median(a, n); lo = a[1]; hi = a[n]
        spread = (med > 0) ? 100.0*(hi-lo)/med : 0
        if (med < 0.01) suspect = 1
        printf "%s,%d,%d,%.4f,%.4f,%.4f,%.1f,%s\n", key, n, bad[key]+0, lo, med, hi, spread, steps[key]
    }
    if (suspect)
        print "WARNING: medians < 0.01 s -- time regex captured dt. Use TIME_FIELD=wall." > "/dev/stderr"
}' "$MERGED" > "$OUT.raw"
head -1 "$OUT.raw" > "$OUT"
tail -n +2 "$OUT.raw" | sort -t, -k1,1 -k3,3n >> "$OUT"
rm -f "$OUT.raw"
echo "  merged rows : $(($(wc -l < "$MERGED") - 1))  -> $MERGED"
echo "  summary     : $OUT"
echo ""
if command -v column >/dev/null 2>&1; then column -s, -t < "$OUT"; else cat "$OUT"; fi
