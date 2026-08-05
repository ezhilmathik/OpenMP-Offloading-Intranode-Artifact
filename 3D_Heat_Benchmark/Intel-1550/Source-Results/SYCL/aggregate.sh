#!/bin/bash
# ---------------------------------------------------------------------------
# Merge results_*/ into summary.csv. Identical to the OpenMP aggregate.sh
# except that variants are recognised as <n>-sycl* instead of <n>-omp*.
#
#   ./aggregate.sh                    # solver's own timer (default)
#   TIME_FIELD=wall ./aggregate.sh    # wall clock incl. startup/teardown
#
# For SYCL the compute-vs-wall gap is larger than on the OpenMP side when the
# binaries were built with AOT=0: JIT and queue/context setup sit in wall but
# outside the solver's timer. Compare like with like across the two pipelines.
# ---------------------------------------------------------------------------
set -euo pipefail
OUT="${1:-summary.csv}"
TIME_FIELD="${TIME_FIELD:-compute}"
shopt -s nullglob

# =============================== compute mode ===============================
# The solver's own timer, straight out of run.log.
if [[ "$TIME_FIELD" == "compute" ]]; then
    LOGS=(results_*/run.log)
    [[ ${#LOGS[@]} -gt 0 ]] || { echo "ERROR: no results_*/run.log found." >&2; exit 1; }
    echo "Reading the solver's own timer from ${#LOGS[@]} run.log file(s)..."
    awk '
    /^>>> / {
        variant = ""; gpus = ""; N = ""; steps = ""; armed = 0
        for (i = 2; i <= NF; i++) {
            tok = $i; sub(/^\.\//, "", tok)
            if (tok ~ /^[0-9]+-sycl/) variant = tok
            if ($i ~ /^gpus=/) { split($i, a, "="); gpus = a[2] }
            if ($i ~ /^N=/)    { split($i, b, "="); N    = b[2] }
        }
        if (variant != "" && gpus != "" && N != "") armed = 1
        next }
    armed && /^Estimated timesteps:/ { s = $3; sub(/^~/, "", s); steps = s }
    armed && /timesteps? complete/   { steps = $1 }
    armed && /^Time = / { printf "%s,%s,%s,%s,%s\n", variant, gpus, N, $3, steps; armed = 0 }
    ' "${LOGS[@]}" > .compute.$$
    ROWS=$(wc -l < .compute.$$)
    [[ "$ROWS" -gt 0 ]] || { rm -f .compute.$$; echo "ERROR: no 'Time = ' lines matched." >&2; exit 1; }
    echo "  timed runs  : $ROWS"
    awk -F, '
    function median(arr, n,   i, j, t, m) {
        for (i = 2; i <= n; i++) { t = arr[i]
            for (j = i-1; j >= 1 && arr[j] > t; j--) arr[j+1] = arr[j]
            arr[j+1] = t }
        m = int((n+1)/2)
        return (n % 2) ? arr[m] : (arr[m] + arr[m+1]) / 2.0 }
    { key = $1 "," $2 "," $3; cnt[key]++; vals[key SUBSEP cnt[key]] = $4 + 0
      if ($5 != "") st[key] = $5 }
    END {
        print "variant,gpus,N,samples,failed,min_sec,median_sec,max_sec,rel_spread_pct,steps"
        for (key in cnt) {
            n = cnt[key]; delete a
            for (i = 1; i <= n; i++) a[i] = vals[key SUBSEP i]
            med = median(a, n); lo = a[1]; hi = a[n]
            printf "%s,%d,0,%.4f,%.4f,%.4f,%.1f,%s\n", key, n, lo, med, hi,
                   (med > 0 ? 100.0*(hi-lo)/med : 0), st[key] } }' .compute.$$ > "$OUT.raw"
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
    # Fixed columns for the 13-field schema written by run.sh:
    #   1 rep  2 job_id  3 hier  4 variant  5 family  6 gpus  7 ze_mask
    #   8 N    9 dt      10 steps  11 wall_sec  12 reported_sec  13 status
    # The OpenMP aggregate.sh carries an "off = NF - 12" fudge meant to absorb
    # an older 12-field AMD schema; on the 13-field Intel files it lands
    # variant on hier and gpus on family, which is why its wall mode prints
    # rows keyed "COMPOSITE". Nothing to absorb here -- one schema, fixed.
    if (NF != 13) { skipped++; next }
    variant = $4; gpus = $6
    N = $8; steps = $10; wall = $11; reported = $12; status = $13
    key = variant "," gpus "," N
    seen[key] = 1
    if (steps != "" && steps != "N/A") st[key] = steps
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
        if (n == 0) { printf "%s,0,%d,N/A,N/A,N/A,%s,%s\n", key, bad[key]+0, why[key], st[key]; continue }
        delete a
        for (i = 1; i <= n; i++) a[i] = vals[key SUBSEP i]
        med = median(a, n); lo = a[1]; hi = a[n]
        spread = (med > 0) ? 100.0*(hi-lo)/med : 0
        if (med < 0.01) suspect = 1
        printf "%s,%d,%d,%.4f,%.4f,%.4f,%.1f,%s\n", key, n, bad[key]+0, lo, med, hi, spread, st[key]
    }
    if (skipped)
        printf "WARNING: %d row(s) did not have 13 fields and were ignored.\n", skipped > "/dev/stderr"
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
