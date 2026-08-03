#!/bin/bash
# ---------------------------------------------------------------------------
# ASCII plots from summary.csv. No GUI, no matplotlib -- pure awk.
#
#   ./plot.sh                 bar chart of median time at the largest N
#   ./plot.sh 1024            bar chart at N=1024
#   ./plot.sh -g 2            2-GPU family: clustered bars per N (baseline + 4)
#   ./plot.sh -g 4            4-GPU family
#   ./plot.sh -s              speedup vs 1-omp, all grid sizes
#   ./plot.sh -a              all of the above
#
#   SUMMARY=other.csv ./plot.sh
#
# Column positions are read from the HEADER, so this works with both the AMD
# schema (variant,gpus,N,...) and the Intel schema (hier,variant,gpus,N,...).
# ---------------------------------------------------------------------------
set -euo pipefail

SUMMARY="${SUMMARY:-summary.csv}"
WIDTH="${WIDTH:-52}"
[[ -f "$SUMMARY" ]] || { echo "ERROR: $SUMMARY not found. Run ./aggregate.sh first." >&2; exit 1; }

MODE="bars"; ARG=""
case "${1:-}" in
    -g|--group)   MODE="group"; ARG="${2:-2}" ;;
    -s|--speedup) MODE="speedup" ;;
    -a|--all)     MODE="all" ;;
    -h|--help)    sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    "")           ;;
    *)            ARG="$1" ;;
esac

# Locate columns by name so both schemas work.
COLS=$(awk -F, 'NR==1 {
    for (i = 1; i <= NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i)
        if ($i == "variant")    v = i
        if ($i == "gpus")       g = i
        if ($i == "N")          n = i
        if ($i == "median_sec") m = i
    }
    if (!v || !g || !n || !m) { print "0 0 0 0"; exit }
    print v, g, n, m
}' "$SUMMARY")
read -r CV CG CN CM <<< "$COLS"
[[ "$CV" != "0" ]] || { echo "ERROR: could not find variant/gpus/N/median_sec columns in $SUMMARY header." >&2; exit 1; }

bars() {
    awk -F, -v want="$1" -v w="$WIDTH" -v cv="$CV" -v cg="$CG" -v cn="$CN" -v cm="$CM" '
    NR == 1 { next }
    $cm == "N/A" { next }
    { if (want == "" && $cn+0 > maxN) maxN = $cn+0
      n[NR]=$cn+0; v[NR]=$cv; g[NR]=$cg; t[NR]=$cm+0; rows=NR }
    END {
        target = (want == "") ? maxN : want+0
        for (i = 1; i <= rows; i++) if (n[i] == target) {
            if (t[i] > peak) peak = t[i]
            if (v[i] ~ /^1-/) base = t[i]
            keep[i] = 1; any = 1 }
        if (!any || peak <= 0) { printf "\n  No usable rows for N=%d\n", target; exit 0 }
        printf "\n  Median time, N=%d  (shorter is better)\n\n", target
        for (i = 1; i <= rows; i++) if (keep[i]) {
            len = int(w * t[i] / peak); if (len < 1) len = 1
            bar = ""; for (k = 0; k < len; k++) bar = bar "#"
            sp = (base > 0) ? base / t[i] : 0
            printf "  %-22s %-3s %-*s %8.1fs  %4.2fx\n", v[i], g[i], w, bar, t[i], sp }
        printf "\n  Speedup relative to the 1-GPU baseline at the same N.\n" }' "$SUMMARY"
}

group() {
    awk -F, -v fam="$1" -v w="$WIDTH" -v cv="$CV" -v cg="$CG" -v cn="$CN" -v cm="$CM" '
    # Naming-agnostic: works for 2-omp / 2-cuda / 2-acc alike. The baseline is
    # whatever starts with "1-"; within a family the suffix gives the order.
    function rank(x) {
        if (x ~ /^1-/)              return 1
        if (x !~ "^" fam "-")       return 0
        if (x ~ /-stream-omp-p2p$/) return 5
        if (x ~ /-stream-omp$/)     return 4
        if (x ~ /-stream$/)         return 3
        return 2 }
    NR == 1 { next }
    $cm == "N/A" { next }
    { r = rank($cv); if (r == 0) next
      N = $cn+0; Ns[N] = 1; t[N,r] = $cm+0; name[r] = $cv; gp[r] = $cg }
    END {
        printf "\n  ==================== %s-GPU family ====================\n", fam
        printf "  Cluster = baseline + the four %s-GPU variants. Shorter is better.\n", fam
        cnt = 0
        for (N in Ns) { cnt++; ord[cnt] = N+0 }
        for (i = 2; i <= cnt; i++) { key = ord[i]
            for (j = i-1; j >= 1 && ord[j] > key; j--) ord[j+1] = ord[j]
            ord[j+1] = key }
        for (o = 1; o <= cnt; o++) {
            N = ord[o]; peak = 0; best = 0; bestr = 0
            for (r = 1; r <= 5; r++) if ((N,r) in t) {
                if (t[N,r] > peak) peak = t[N,r]
                if (best == 0 || t[N,r] < best) { best = t[N,r]; bestr = r } }
            if (peak <= 0) continue
            base = ((N,1) in t) ? t[N,1] : 0
            printf "\n  N = %d\n", N
            for (r = 1; r <= 5; r++) {
                if (!((N,r) in t)) continue
                len = int(w * t[N,r] / peak); if (len < 1) len = 1
                bar = ""; for (k = 0; k < len; k++) bar = bar "#"
                sp = (base > 0) ? base / t[N,r] : 0
                printf "    %-22s %-3s %-*s %8.1fs %5.2fx%s\n", \
                       name[r], gp[r], w, bar, t[N,r], sp, (r == bestr ? "  <= best" : "") }
            if ((N,4) in t && (N,5) in t && t[N,4] > 0) {
                d = 100.0 * (t[N,4] - t[N,5]) / t[N,4]
                printf "    %-22s p2p vs stream-omp: %+.1f%% %s\n", "", d,
                       (d > 1 ? "(p2p faster)" : (d < -1 ? "(p2p SLOWER)" : "(no difference)")) } }
        printf "\n" }' "$SUMMARY"
}

speedup() {
    awk -F, -v w="$WIDTH" -v cv="$CV" -v cg="$CG" -v cn="$CN" -v cm="$CM" '
    NR == 1 { next }
    $cm == "N/A" { next }
    { n[NR]=$cn+0; v[NR]=$cv; g[NR]=$cg; t[NR]=$cm+0; rows=NR
      if (v[NR] ~ /^1-/) base[n[NR]] = t[NR]
      seenN[$cn+0] = 1 }
    END {
        printf "\n  Speedup vs the 1-GPU baseline  (ideal: 2.00x on 2 GPUs, 4.00x on 4)\n\n"
        cnt = 0
        for (N in seenN) { cnt++; ord[cnt] = N+0 }
        for (i = 2; i <= cnt; i++) { key = ord[i]
            for (j = i-1; j >= 1 && ord[j] > key; j--) ord[j+1] = ord[j]
            ord[j+1] = key }
        for (o = 1; o <= cnt; o++) { N = ord[o]
            if (!(N in base) || base[N] <= 0) continue
            printf "  N=%d\n", N
            for (i = 1; i <= rows; i++) if (n[i] == N && v[i] !~ /^1-/ && t[i] > 0) {
                sp = base[N] / t[i]
                len = int(20 * sp); if (len < 1) len = 1; if (len > w) len = w
                bar = ""; for (k = 0; k < len; k++) bar = bar "="
                printf "    %-22s %-3s %-*s %5.2fx\n", v[i], g[i], w, bar, sp }
            print "" } }' "$SUMMARY"
}

case "$MODE" in
    bars)    bars "$ARG" ;;
    group)   group "$ARG" ;;
    speedup) speedup ;;
    all)     bars ""; group 2; group 4; speedup ;;
esac
