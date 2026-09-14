#!/bin/bash
# spread.sh -- run-to-run variability across all configurations in summary.csv
# Usage: ./spread.sh [summary.csv ...]
set -uo pipefail
FILES=("${@:-summary.csv}")
awk -F, '
FNR == 1 {                                   # locate the columns by name
    for (i = 1; i <= NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i)
        if ($i == "variant")        cv = i
        if ($i == "gpus")           cg = i
        if ($i == "N")              cn = i
        if ($i == "samples")        cs = i
        if ($i == "median_sec")     cm = i
        if ($i == "rel_spread_pct") cp = i
    }
    next
}
$cm == "N/A" || $cp == "" { next }
{
    s = $cp + 0; n++
    sum += s
    if (s > worst) { worst = s; wrow = $cv " gpus=" $cg " N=" $cn }
    if ($cs + 0 < minrep || minrep == 0) minrep = $cs + 0
    if ($cs + 0 > maxrep) maxrep = $cs + 0
    if (s <= 1) le1++
    if (s <= 2) le2++
    if (s <= 5) le5++
    a[n] = s
}
END {
    if (!n) { print "no usable rows"; exit 1 }
    for (i = 2; i <= n; i++) { t = a[i]; for (j = i-1; j >= 1 && a[j] > t; j--) a[j+1] = a[j]; a[j+1] = t }
    med = (n % 2) ? a[int((n+1)/2)] : (a[n/2] + a[n/2+1]) / 2
    printf "configurations      : %d\n", n
    printf "repetitions each    : %d%s\n", minrep, (minrep == maxrep ? "" : sprintf("-%d", maxrep))
    printf "spread (max-min)/median:\n"
    printf "  median            : %.2f %%\n", med
    printf "  mean              : %.2f %%\n", sum / n
    printf "  90th percentile   : %.2f %%\n", a[int(0.9 * n + 0.5) ? int(0.9 * n + 0.5) : 1]
    printf "  maximum           : %.2f %%   (%s)\n", worst, wrow
    printf "  <= 1 %%            : %d of %d (%.0f%%)\n", le1, n, 100.0 * le1 / n
    printf "  <= 2 %%            : %d of %d (%.0f%%)\n", le2, n, 100.0 * le2 / n
    printf "  <= 5 %%            : %d of %d (%.0f%%)\n", le5, n, 100.0 * le5 / n
}' "${FILES[@]}"
