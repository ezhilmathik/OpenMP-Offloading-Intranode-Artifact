#!/usr/bin/env python3
"""
Average P2P bandwidth measurements across repetitions and report the effective
per-link, per-direction bandwidth to feed ../../model/plot_p2p.py --beta-p2p.

Mirrors the aggregate.sh + plot.sh workflow: it merges every results_*/ CSV,
takes the MEDIAN across repetitions at each (framework, version, N), then
reports both the mean over all sizes and the plateau (mean of the largest
--tail sizes, the same convention compute_gamma.py uses for gamma).

Version semantics (verified from the data: bandwidth_gbs * elapsed_sec / (8N)):
    v1  1 transfer   single link, one direction
    v2  2 transfers  one link pair, both directions at once
    v4  3 transfers  one source GPU to three peers at once

bandwidth_gbs is therefore an AGGREGATE over concurrent transfers. The model in
plot_p2p.py multiplies by `faces` itself, so it needs the PER-LINK, PER-DIRECTION
number -- that is the `per-link` column below, not the raw aggregate.

Usage:
    python3 p2p_average.py                          # globs results_*/ *.csv
    python3 p2p_average.py --glob 'results_*/p2p.csv'
    python3 p2p_average.py path/to/one.csv
    python3 p2p_average.py --hierarchy COMPOSITE
    python3 p2p_average.py --out beta_p2p.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from pathlib import Path

# Concurrent transfers aggregated into the bandwidth_gbs column, per version.
CONCURRENCY = {"v1": 1, "v2": 2, "v4": 3}

TOPOLOGY = {
    "v1": "1 link, unidirectional",
    "v2": "1 link, bidirectional",
    "v4": "3 links, unidirectional",
}


def load(paths: list[Path], hierarchy: str | None) -> dict:
    """Return {(hier, framework, version, N): [bandwidth samples]}."""
    samples: dict[tuple, list[float]] = {}
    used = 0

    for path in paths:
        try:
            with path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                required = {"framework", "version", "N", "bandwidth_gbs"}
                if not required.issubset(set(reader.fieldnames or [])):
                    print(f"[skip] {path}: missing columns", file=sys.stderr)
                    continue
                used += 1
                for row in reader:
                    if (row.get("status") or "ok").strip() != "ok":
                        continue
                    hier = (row.get("hierarchy") or "-").strip()
                    if hierarchy and hier != hierarchy:
                        continue
                    try:
                        key = (
                            hier,
                            row["framework"].strip(),
                            row["version"].strip(),
                            int(row["N"]),
                        )
                        bw = float(row["bandwidth_gbs"])
                    except (TypeError, ValueError):
                        continue
                    if bw > 0:
                        samples.setdefault(key, []).append(bw)
        except OSError as exc:
            print(f"[skip] {path}: {exc}", file=sys.stderr)

    if not used:
        sys.exit("no readable result CSVs found")
    print(f"merged {used} result file(s)")
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="explicit CSV paths")
    ap.add_argument("--glob", default="results_*/*.csv",
                    help="glob used when no paths are given")
    ap.add_argument("--hierarchy", default=None,
                    help="keep only this hierarchy (e.g. COMPOSITE, FLAT)")
    ap.add_argument("--tail", type=int, default=3,
                    help="largest-N rows averaged for the plateau (default 3)")
    ap.add_argument("--out", default=None,
                    help="write per-framework beta values to this CSV")
    args = ap.parse_args()

    paths = ([Path(p) for p in args.paths]
             if args.paths else sorted(Path(p) for p in glob.glob(args.glob)))
    if not paths:
        sys.exit(f"no files matched {args.glob!r}")

    samples = load(paths, args.hierarchy)

    # Median across repetitions, then group by (hier, framework, version).
    series: dict[tuple, list[tuple[int, float]]] = {}
    reps: dict[tuple, int] = {}
    for (hier, fw, ver, n), values in samples.items():
        series.setdefault((hier, fw, ver), []).append((n, statistics.median(values)))
        reps[(hier, fw, ver)] = max(reps.get((hier, fw, ver), 0), len(values))
    for v in series.values():
        v.sort()

    records = []
    print(f"\n  P2P bandwidth, median over repetitions  (plateau = mean of "
          f"{args.tail} largest sizes)\n")
    print(f"  {'hier':<10} {'framework':<10} {'ver':<4} {'topology':<24} "
          f"{'reps':>4} {'mean':>9} {'plateau':>9} {'per-link':>9}")
    print("  " + "-" * 92)

    for key in sorted(series):
        hier, fw, ver = key
        pts = series[key]
        bw = [b for _, b in pts]
        conc = CONCURRENCY.get(ver, 1)
        mean = statistics.mean(bw)
        plateau = statistics.mean(bw[-min(args.tail, len(bw)):])
        per_link = plateau / conc
        spread = 100.0 * (bw[-1] - bw[0]) / bw[0] if bw[0] else 0.0
        print(f"  {hier:<10} {fw:<10} {ver:<4} {TOPOLOGY.get(ver, '?'):<24} "
              f"{reps[key]:>4} {mean:>9.2f} {plateau:>9.2f} {per_link:>9.2f}")
        records.append({
            "hierarchy": hier, "framework": fw, "version": ver,
            "concurrent_transfers": conc, "sizes": len(pts),
            "mean_gbs": round(mean, 3), "plateau_gbs": round(plateau, 3),
            "per_link_per_dir_gbs": round(per_link, 3),
            "growth_pct_small_to_large": round(spread, 1),
        })

    print("\n  mean     = over all sizes;  plateau = asymptote;  "
          "per-link = plateau / concurrent transfers")

    # --- convergence warning -------------------------------------------------
    noisy = [r for r in records if r["growth_pct_small_to_large"] > 15.0]
    if noisy:
        print("\n  NOT CONVERGED -- bandwidth still rising at the largest size, so "
              "the plateau is a\n  lower bound and the mean is not meaningful:")
        for r in noisy:
            print(f"    {r['framework']:<10} {r['version']:<4} "
                  f"+{r['growth_pct_small_to_large']:.0f}% from smallest to largest size")

    # --- what to pass to plot_p2p.py ----------------------------------------
    print("\n  Suggested --beta-p2p (per-link, per-direction, from v1):")
    for key in sorted(series):
        hier, fw, ver = key
        if ver != "v1":
            continue
        bw = [b for _, b in series[key]]
        beta = statistics.mean(bw[-min(args.tail, len(bw)):])
        print(f"    {fw:<10} --beta-p2p {beta:.1f}")

    if args.out:
        out = Path(args.out)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
        print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
