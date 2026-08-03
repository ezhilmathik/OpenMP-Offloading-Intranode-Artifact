#!/bin/bash
# ASCII plots from a HIP summary.csv file.
#
#   ./plot.sh                 median-time bars at largest N
#   ./plot.sh 1024            bars at N=1024
#   ./plot.sh -g 2            2-GPU family by N
#   ./plot.sh -g 4            4-GPU family by N
#   ./plot.sh -s              speedup versus 1-hip
#   ./plot.sh -a              all views
#
#   SUMMARY=other.csv ./plot.sh

set -euo pipefail

SUMMARY="${SUMMARY:-summary.csv}"
WIDTH="${WIDTH:-52}"
[[ -f "$SUMMARY" ]] || {
    echo "ERROR: $SUMMARY not found. Run ./aggregate.sh first." >&2
    exit 1
}

MODE="bars"
ARG=""
case "${1:-}" in
    -g|--group)   MODE="group"; ARG="${2:-2}" ;;
    -s|--speedup) MODE="speedup" ;;
    -a|--all)     MODE="all" ;;
    -h|--help)    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    "")           ;;
    *)            ARG="$1" ;;
esac

COLS=$(awk -F, 'NR==1 {
    for (i=1; i<=NF; i++) {
        gsub(/^[ \t]+|[ \t]+$/, "", $i)
        if ($i=="variant") v=i
        if ($i=="gpus") g=i
        if ($i=="N") n=i
        if ($i=="median_sec") m=i
    }
    if (!v || !g || !n || !m) print "0 0 0 0"
    else print v,g,n,m
}' "$SUMMARY")
read -r CV CG CN CM <<< "$COLS"
[[ "$CV" != "0" ]] || {
    echo "ERROR: missing variant/gpus/N/median_sec columns in $SUMMARY." >&2
    exit 1
}

bars() {
    awk -F, -v want="$1" -v w="$WIDTH" -v cv="$CV" -v cg="$CG" -v cn="$CN" -v cm="$CM" '
    NR==1 || $cm=="N/A" { if (NR==1) next; if ($cm=="N/A") next }
    {
        if (want=="" && $cn+0>maxN) maxN=$cn+0
        n[++rows]=$cn+0; v[rows]=$cv; g[rows]=$cg; t[rows]=$cm+0
    }
    END {
        target=(want=="") ? maxN : want+0
        for (i=1; i<=rows; i++) if (n[i]==target) {
            keep[i]=1; any=1
            if (t[i]>peak) peak=t[i]
            if (v[i]=="1-hip") base=t[i]
        }
        if (!any || peak<=0) { printf "\n  No usable rows for N=%d\n",target; exit }
        printf "\n  Median HIP time, N=%d (shorter is better)\n\n",target
        for (i=1; i<=rows; i++) if (keep[i]) {
            len=int(w*t[i]/peak); if (len<1) len=1
            bar=""; for (k=0; k<len; k++) bar=bar "#"
            sp=(base>0) ? base/t[i] : 0
            printf "  %-25s %-3s %-*s %10.4fs  %5.2fx\n",v[i],g[i],w,bar,t[i],sp
        }
        printf "\n  Speedup is relative to 1-hip at the same N.\n"
    }' "$SUMMARY"
}

group() {
    awk -F, -v fam="$1" -v w="$WIDTH" -v cv="$CV" -v cg="$CG" -v cn="$CN" -v cm="$CM" '
    function rank(x) {
        if (x=="1-hip") return 1
        if (x==fam "-hip") return 2
        if (x==fam "-hip-stream") return 3
        if (x==fam "-hip-stream-omp") return 4
        if (x==fam "-hip-stream-omp-p2p") return 5
        return 0
    }
    NR==1 { next }
    $cm=="N/A" { next }
    {
        r=rank($cv); if (!r) next
        N=$cn+0; Ns[N]=1; t[N,r]=$cm+0; name[r]=$cv; gp[r]=$cg
    }
    END {
        printf "\n  ==================== %s-GPU HIP family ====================\n",fam
        cnt=0
        for (N in Ns) ord[++cnt]=N+0
        for (i=2; i<=cnt; i++) {
            key=ord[i]
            for (j=i-1; j>=1 && ord[j]>key; j--) ord[j+1]=ord[j]
            ord[j+1]=key
        }
        for (o=1; o<=cnt; o++) {
            N=ord[o]; peak=0; best=0; bestr=0
            for (r=1; r<=5; r++) if ((N SUBSEP r) in t) {
                if (t[N,r]>peak) peak=t[N,r]
                if (best==0 || t[N,r]<best) { best=t[N,r]; bestr=r }
            }
            if (peak<=0) continue
            base=((N SUBSEP 1) in t) ? t[N,1] : 0
            printf "\n  N = %d\n",N
            for (r=1; r<=5; r++) {
                if (!((N SUBSEP r) in t)) continue
                len=int(w*t[N,r]/peak); if (len<1) len=1
                bar=""; for (k=0; k<len; k++) bar=bar "#"
                sp=(base>0) ? base/t[N,r] : 0
                printf "    %-25s %-3s %-*s %10.4fs %5.2fx%s\n", \
                       name[r],gp[r],w,bar,t[N,r],sp,(r==bestr ? " <= best" : "")
            }
            if ((N SUBSEP 4) in t && (N SUBSEP 5) in t && t[N,4]>0) {
                d=100.0*(t[N,4]-t[N,5])/t[N,4]
                printf "    %-25s P2P versus stream-omp: %+.1f%%\n","",d
            }
        }
        printf "\n"
    }' "$SUMMARY"
}

speedup() {
    awk -F, -v w="$WIDTH" -v cv="$CV" -v cg="$CG" -v cn="$CN" -v cm="$CM" '
    NR==1 { next }
    $cm=="N/A" { next }
    {
        n[++rows]=$cn+0; v[rows]=$cv; g[rows]=$cg; t[rows]=$cm+0
        if (v[rows]=="1-hip") base[n[rows]]=t[rows]
        seen[n[rows]]=1
    }
    END {
        printf "\n  HIP speedup versus 1-hip (ideal: 2x on 2 GPUs, 4x on 4 GPUs)\n\n"
        cnt=0
        for (N in seen) ord[++cnt]=N+0
        for (i=2; i<=cnt; i++) {
            key=ord[i]
            for (j=i-1; j>=1 && ord[j]>key; j--) ord[j+1]=ord[j]
            ord[j+1]=key
        }
        for (o=1; o<=cnt; o++) {
            N=ord[o]
            if (!(N in base) || base[N]<=0) continue
            printf "  N=%d\n",N
            for (i=1; i<=rows; i++) if (n[i]==N && v[i]!="1-hip" && t[i]>0) {
                sp=base[N]/t[i]
                len=int(12*sp); if (len<1) len=1; if (len>w) len=w
                bar=""; for (k=0; k<len; k++) bar=bar "="
                printf "    %-25s %-3s %-*s %5.2fx\n",v[i],g[i],w,bar,sp
            }
            print ""
        }
    }' "$SUMMARY"
}

case "$MODE" in
    bars)    bars "$ARG" ;;
    group)   group "$ARG" ;;
    speedup) speedup ;;
    all)     bars ""; group 2; group 4; speedup ;;
esac
