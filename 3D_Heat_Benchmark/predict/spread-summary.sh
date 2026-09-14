#!/usr/bin/env bash
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"     # 3D_Heat_Benchmark
out="$(dirname "$0")/spread_summary.txt"     # keep the output next to this script
: > "$out"
for s in "$root"/*/Source-Results/*/spread.sh; do
    echo "=== ${s#$root/} ===" | tee -a "$out"
    (cd "$(dirname "$s")" && bash spread.sh) | tee -a "$out"
done
echo "Saved to $out"
