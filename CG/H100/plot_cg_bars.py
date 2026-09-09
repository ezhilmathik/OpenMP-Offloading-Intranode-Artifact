#!/usr/bin/env python3
"""
plot_cg_bars.py — CG full-cycle time, CPU vs OpenMP offloading (H100, BSC)

Four series per footprint group:
    CPU (128 threads)          baseline, hatched
    OpenMP Off. 1 GPU          baseline, hatched
    OpenMP Off. halo, 2 GPUs
    OpenMP Off. halo, 4 GPUs

The CUDA variants and the replicated-exchange OpenMP variants are present in
the CSVs but not drawn here; this chart is about what the packed halo buys
over a single GPU.

INPUT
-----
Reads every results_*/timings.csv in the working directory and takes the
MEDIAN across repetitions per (variant, footprint). Median, not mean: cluster
timings are right-skewed, so one contended repetition drags a mean but not a
median.

timings.csv schema (header row present):
    rep,job_id,variant,gb_target,impl,backend,order,rows,nnz,nnz_row,
    mean_bandwidth,footprint_gb,iters,relres,cycle_ms,spmv_ms,other_ms,
    spmv_share,gbs,best_ms,spread_pct,runs,par,halo_ms,wall_sec,status

Examples:
    python3 plot_cg_bars.py
    python3 plot_cg_bars.py --gbs 10 20 30
    python3 plot_cg_bars.py --field best_ms --no-show
    python3 plot_cg_bars.py figs --log
"""

import os
import csv
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- fonts -----------------------------
MEDIUM_SIZE = 20
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=MEDIUM_SIZE)

# ========================= USER CONFIG =========================
CSV_GLOB    = "rep*/results_*/timings.csv"
DEFAULT_GBS = [5, 10, 20, 30, 40, 50]
TIME_COL      = "cycle_ms"      # "cycle_ms" (mean over runs) | "best_ms"
ORDER_TO_PLOT = "natural"

# (csv variant, legend label, colour, hatch)
# Hatch marks the reference bars, as the 1-GPU baselines are hatched in the
# diffusion charts.
SERIES = [
#    ("cpu",         "CPU (128 threads)",        "grey", "//"),
    ("ompgpu1",     "OpenMP Off. (1 GPU)",      "b",    "//"),
    ("ompgpu2halo", "OpenMP Off. (2 GPUs)", "g",   None),
    ("ompgpu4halo", "OpenMP Off. (4 GPUs)", "orange",   None),
]

# Bars annotated with their speedup over this series.
SPEEDUP_BASE  = "ompgpu1"
ANNOTATE      = ["ompgpu2halo", "ompgpu4halo"]

#TITLE         = "NVIDIA H100 - CG (fp64, natural ordering)"
TITLE = ("NVIDIA H100 - CG (fp64, CSR, mean 16 nnz/row, "
         r"$\|r\|/\|b\| \leq 10^{-8}$)")
#XLABEL        = "Problem footprint (GB, total across devices)."
XLABEL         ="Matrix and vector footprint (GB, total across devices)."
YLABEL        = "Full CG cycle in milliseconds per iteration."
GROUP_WIDTH   = 0.8
# ===============================================================


def collect(gbs):
    """values[variant][gb] = [times across repetitions]"""
    wanted = {v for v, _, _, _ in SERIES}
    gbs = set(float(g) for g in gbs)
    values = defaultdict(lambda: defaultdict(list))
    paths = sorted(glob.glob(CSV_GLOB))

    for path in paths:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            missing = {"variant", "gb_target", "order", "status", TIME_COL} \
                      - set(reader.fieldnames or [])
            if missing:
                print(f"[warn] {path}: missing column(s) {sorted(missing)}")
                continue
            for row in reader:
                if row["status"].strip() != "ok":
                    continue
                if row["order"].strip() != ORDER_TO_PLOT:
                    continue
                var = row["variant"].strip()
                if var not in wanted:
                    continue
                raw = (row.get(TIME_COL) or "").strip()
                if not raw or raw == "N/A":
                    continue
                try:
                    gb = float(row["gb_target"])
                    val = float(raw)
                except ValueError:
                    continue
                if gb not in gbs:
                    continue
                values[var][gb].append(val)
        print(f"[ok] read {path}")
    return values, paths


def medians(values):
    """med[variant][gb] = median across repetitions; cnt[variant][gb] = n"""
    med = defaultdict(dict)
    cnt = defaultdict(dict)
    for var, per_gb in values.items():
        for gb, vals in per_gb.items():
            med[var][gb] = float(np.median(vals))
            cnt[var][gb] = len(vals)
    return med, cnt


def plot_chart(med, gbs, out_pdf, logy=False, show=True):
    drawn = [s for s in SERIES if med.get(s[0])]
    if not drawn:
        print("[skip] no data for any series, no chart drawn.")
        return

    n_series = len(drawn)
    bar_w = GROUP_WIDTH / n_series
    x = np.arange(len(gbs))

    fig, ax = plt.subplots(figsize=(12, 8))
    positions = {}
    for i, (var, label, color, hatch) in enumerate(drawn):
        vals = [med.get(var, {}).get(float(g), np.nan) for g in gbs]
        pos = x + i * bar_w
        positions[var] = pos
        ax.bar(pos, vals, width=bar_w, color=color, edgecolor="grey",
               linewidth=0.01, label=label, hatch=hatch)

    # ---- annotate speedup over the 1-GPU OpenMP baseline ----
    for var in ANNOTATE:
        if var not in positions:
            continue
        for j, g in enumerate(gbs):
            base_t = med.get(SPEEDUP_BASE, {}).get(float(g))
            this_t = med.get(var, {}).get(float(g))
            if base_t and this_t:
                ax.annotate(f"{base_t / this_t:.2f}×",
                            xy=(positions[var][j], this_t),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=MEDIUM_SIZE - 3, rotation=90,
                            color="black")

    ax.set_xlabel(XLABEL, fontsize=MEDIUM_SIZE)
    ax.set_ylabel(YLABEL, fontsize=MEDIUM_SIZE)
    ax.set_title(TITLE)
    ax.set_xticks(x + (n_series - 1) / 2.0 * bar_w)
    ax.set_xticklabels([f"{g:g} GB" for g in gbs])

    if logy:
        ax.set_yscale("log")
    else:
        # headroom for the rotated speedup labels
        top = max(v for var, _, _, _ in drawn
                  for v in med.get(var, {}).values())
        ax.set_ylim(0, top * 1.12)

    ax.minorticks_on()
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, alpha=0.2)
    ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.5, alpha=0.2)

    ax.legend(ncol=1, loc="upper left", fontsize=MEDIUM_SIZE,
              framealpha=0.85, columnspacing=1.0, handletextpad=0.4,
              labelspacing=0.3)

    fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.10)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.10)
    plt.savefig(out_pdf, format="pdf")
    print(f"Wrote: {out_pdf}")
    if show:
        plt.show()
    plt.close(fig)


def print_table(med, cnt, gbs):
    print(f"\n===== CG FULL CYCLE, ms/iter ({TIME_COL}, median of repetitions) =====")
    for g in gbs:
        g = float(g)
        print(f"--- {g:g} GB ---")
        for var, label, _, _ in SERIES:
            if g in med.get(var, {}):
                print(f"{label:30s}: {med[var][g]:10.4f} ms  (n_reps={cnt[var][g]})")
            else:
                print(f"{label:30s}: MISSING")
        base_t = med.get(SPEEDUP_BASE, {}).get(g)
        for var in ANNOTATE:
            this_t = med.get(var, {}).get(g)
            if base_t and this_t:
                sp = base_t / this_t
                label = dict((v, l) for v, l, _, _ in SERIES)[var]
                print(f"{label + ' vs 1 GPU':30s}: {sp:10.2f}x  ({(sp - 1) * 100:+.1f}%)")
        print()


def write_csv(med, cnt, gbs, csv_path):
    rows = []
    for g in gbs:
        g = float(g)
        for var, label, _, _ in SERIES:
            if g in med.get(var, {}):
                rows.append([f"{g:g}", var, label,
                             f"{med[var][g]:.6f}", cnt[var][g]])
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["GB", "variant", "Method", "cycle_ms", "n_reps"])
        w.writerows(rows)
    print(f"Wrote: {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="CG bar chart, CPU vs OpenMP offloading.")
    p.add_argument("outdir", nargs="?", default=".",
                   help="folder for the PDF and cg_avg.csv (default: current directory)")
    p.add_argument("--gbs", type=int, nargs="+", default=DEFAULT_GBS,
                   help=f"footprints to plot (default: {DEFAULT_GBS})")
    p.add_argument("--field", default=TIME_COL, choices=["cycle_ms", "best_ms"],
                   help=f"which timing column to use (default: {TIME_COL})")
    p.add_argument("--log", action="store_true",
                   help="log y-axis (CPU is ~15x the fastest GPU variant)")
    p.add_argument("--no-show", action="store_true",
                   help="save the PDF without opening a plot window")
    return p.parse_args()


def main():
    global TIME_COL
    args = parse_args()
    TIME_COL = args.field

    os.makedirs(args.outdir, exist_ok=True)
    gbs = sorted(args.gbs)

    values, paths = collect(gbs)
    if not paths:
        print(f"No files matched '{CSV_GLOB}' in {os.getcwd()}")
        return
    med, cnt = medians(values)
    if not med:
        print("No valid timing data found — nothing to plot.")
        return

    print("\n===== DISCOVERY =====")
    print(f"Output folder      : {os.path.abspath(args.outdir)}")
    print(f"Footprints         : {gbs} GB")
    print(f"Time column        : {TIME_COL}")
    print(f"Ordering           : {ORDER_TO_PLOT}")
    print(f"CSV files read     : {len(paths)}")

    print_table(med, cnt, gbs)
    plot_chart(med, gbs, os.path.join(args.outdir, "CG_H100_halo.pdf"),
               logy=args.log, show=not args.no_show)
    write_csv(med, cnt, gbs, os.path.join(args.outdir, "cg_avg.csv"))


if __name__ == "__main__":
    main()
