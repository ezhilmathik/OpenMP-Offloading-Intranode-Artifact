#!/bin/bash
# ---------------------------------------------------------------------------
# Merge results_*/ -> summary.csv, one row per (variant, N).
#
#   ./aggregate.sh [outfile]
#
#   TIME_FIELD=compute    solver's own timer, read from run.log   <-- BEST
#   TIME_FIELD=wall       wall clock from timings.csv (incl. startup + map)
#   TIME_FIELD=reported   the reported_sec column as written
#
# Works on BOTH pipelines: column positions come from the CSV header, so the
# AMD schema (12 cols) and the Intel schema (13 cols, extra `hier`) both parse.
#
# Two data problems are handled automatically:
#   1. Unquoted comma masks ("0,2" / "0,1,2,3") added fields and shifted every
#      column after the mask. The shift is derived per row from the field count.
#   2. reported_sec written before the regex fix holds DT, not elapsed time:
#      the old pattern matched "Final time (T): 0.000135" from the parameter
#      banner. TIME_FIELD=compute bypasses the CSV entirely and reads
#      "Time = 1.176864 seconds" straight out of run.log.
#
# The trailing `steps` column carries the timestep count through to the plots.
# ---------------------------------------------------------------------------
set -euo pipefail

OUT="${1:-summary.csv}"
TIME_FIELD="${TIME_FIELD:-compute}"

shopt -s nullglob

# =============================== compute mode ===============================
if [[ "$TIME_FIELD" == "compute" ]]; then
    LOGS=(results_*/run.log)
    [[ ${#LOGS[@]} -gt 0 ]] || { echo "ERROR: no results_*/run.log found." >&2; exit 1; }
    echo "Reading the solver's own timer from ${#LOGS[@]} run.log file(s)..."

    awk '
    /^>>> / {
        # Position-independent: the AMD logs read ">>> Running: 2-omp  gpus=..."
        # while the Intel and CUDA logs read ">>> 2-cuda  gpus=...", so scan
        # every token for the one that looks like a binary name.
        variant = ""; gpus = ""; N = ""; steps = ""; armed = 0
        for (i = 2; i <= NF; i++) {
            tok = $i; sub(/^\.\//, "", tok)
            if (tok ~ /^[0-9]+-(omp|cuda|acc)/) variant = tok
            if ($i ~ /^gpus=/) { split($i, a, "="); gpus = a[2] }
            if ($i ~ /^N=/)    { split($i, b, "="); N    = b[2] }
        }
        if (variant != "" && gpus != "" && N != "") armed = 1
        next
    }
    # Prefer the actual completed-step count; the banner estimate is a fallback.
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

# ================================ CSV mode ==================================
else
    CSVS=(results_*/timings.csv)
    [[ ${#CSVS[@]} -gt 0 ]] || { echo "ERROR: no results_*/timings.csv found." >&2; exit 1; }
    echo "Merging ${#CSVS[@]} timings.csv on '${TIME_FIELD}' time..."

    MERGED="all_timings.csv"
    head -1 "${CSVS[0]}" > "$MERGED"
    for f in "${CSVS[@]}"; do tail -n +2 "$f" >> "$MERGED"; done

    awk -F, -v field="$TIME_FIELD" '
    function median(arr, n,   i, j, t, m) {
        for (i = 2; i <= n; i++) { t = arr[i]
            for (j = i-1; j >= 1 && arr[j] > t; j--) arr[j+1] = arr[j]
            arr[j+1] = t }
        m = int((n+1)/2)
        return (n % 2) ? arr[m] : (arr[m] + arr[m+1]) / 2.0 }
    # Header: locate columns by NAME, and remember where the mask sits.
    NR == 1 {
        HN = NF
        for (i = 1; i <= NF; i++) {
            gsub(/^[ \t]+|[ \t]+$/, "", $i)
            col[$i] = i
            if ($i ~ /_mask$/) maskcol = i }
        hashier = ("hier" in col)
        hassteps = ("steps" in col)
        next }
    # Data: columns after the mask shift right by (NF - HN).
    { off = NF - HN; if (off < 0) off = 0
      f_hier    = hashier ? $(col["hier"]) : ""
      f_variant = $(col["variant"])
      f_gpus    = $(col["gpus"])
      f_N       = $(col["N"]         > maskcol ? col["N"]         + off : col["N"])
      f_steps   = hassteps ? $(col["steps"] > maskcol ? col["steps"] + off : col["steps"]) : ""
      f_wall    = $(col["wall_sec"]  > maskcol ? col["wall_sec"]  + off : col["wall_sec"])
      f_rep     = $(col["reported_sec"] > maskcol ? col["reported_sec"] + off : col["reported_sec"])
      f_status  = $NF

      key = (hashier ? f_hier "," : "") f_variant "," f_gpus "," f_N
      seen[key] = 1
      if (f_steps != "" && f_steps != "N/A") st[key] = f_steps
      if (f_status != "ok") { bad[key]++; why[key] = f_status; next }
      t = (field == "wall") ? f_wall : f_rep
      if (t == "N/A" || t == "") t = f_wall
      if (t == "N/A" || t == "") { bad[key]++; why[key] = "unparseable"; next }
      cnt[key]++; vals[key SUBSEP cnt[key]] = t + 0 }
    END {
        printf "%svariant,gpus,N,samples,failed,min_sec,median_sec,max_sec,rel_spread_pct,steps\n",
               (hashier ? "hier," : "")
        for (key in seen) {
            n = cnt[key] + 0
            if (n == 0) { printf "%s,0,%d,N/A,N/A,N/A,%s,%s\n", key, bad[key]+0, why[key], st[key]; continue }
            delete a
            for (i = 1; i <= n; i++) a[i] = vals[key SUBSEP i]
            med = median(a, n); lo = a[1]; hi = a[n]
            if (med < 0.01) suspect = 1
            printf "%s,%d,%d,%.4f,%.4f,%.4f,%.1f,%s\n", key, n, bad[key]+0, lo, med, hi,
                   (med > 0 ? 100.0*(hi-lo)/med : 0), st[key] }
        if (suspect)
            print "WARNING: medians < 0.01 s -- reported_sec holds DT. Use TIME_FIELD=compute." > "/dev/stderr"
    }' "$MERGED" > "$OUT.raw"
    echo "  merged rows : $(($(wc -l < "$MERGED") - 1))  -> $MERGED"
fi

# ================================= output ===================================
# Detect the hier schema by the header NAME, not the column count: adding
# `steps` made the flat schema 10 columns, which the old count test mistook
# for the hier layout.
if head -1 "$OUT.raw" | grep -q '^hier,'; then
    SORTKEY="-k1,1 -k2,2 -k4,4n"
else
    SORTKEY="-k1,1 -k3,3n"
fi
head -1 "$OUT.raw" > "$OUT"
# shellcheck disable=SC2086
tail -n +2 "$OUT.raw" | sort -t, $SORTKEY >> "$OUT"
rm -f "$OUT.raw"

echo "  summary     : $OUT  (${TIME_FIELD})"
echo ""
if command -v column >/dev/null 2>&1; then column -s, -t < "$OUT"; else cat "$OUT"; fi
