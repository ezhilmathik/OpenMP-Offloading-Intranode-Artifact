#!/usr/bin/env python3
"""
Bar-chart averaging script for SYCL vs OpenMP Offloading (Intel GPU, 3D-Heat).

This is the Intel twin of the H100 plot_multi_gpu.py: same interleaved-bars
layout, same colour palette, same hatched 1-GPU baselines, and the same
inside-the-plot legend.

Values are read from the harness-produced `summary.csv` in each framework
folder (columns: variant,gpus,N,samples,failed,min_sec,median_sec,max_sec,
rel_spread_pct,steps). Searched per framework, first hit wins:

    <sub>/<hierarchy>/summary.csv     e.g. OpenMP/composite/summary.csv
    <sub>/summary_<hierarchy>.csv     e.g. OpenMP/summary_composite.csv
    <sub>/summary.csv                 e.g. OpenMP/summary.csv

If one framework's summary.csv is missing, its bar slots are still reserved so
the remaining bars keep their positions; the missing bars just show as gaps.

Pick the hierarchy with --mode (default: composite). Each program is split by
the leading number in its name (2-* -> 2-GPU chart, 4-* -> 4-GPU chart). The
1-GPU programs are carried into BOTH the 2-GPU and 4-GPU charts as a hatched
baseline reference.

The first positional argument is the output folder for the PDFs and the CSV
(default: plots). It is created if it does not exist.

Examples:
    python3 speedup.py                       # -> plots/
    python3 speedup.py plots                 # same, explicit
    python3 speedup.py .                     # write next to the script
    python3 speedup.py --mode tile           # tile hierarchy
    python3 speedup.py --ns 512 768 1024
    python3 speedup.py --gpus 4 --no-show
"""

import os
import re
import csv
import glob
import math
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- fonts -----------------------------
MEDIUM_SIZE = 16
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=MEDIUM_SIZE)

# ========================= USER CONFIG =========================
PLOTS_DIR = "."               # where the framework folders live (input)
OUT_DIR = "plots"             # where the PDFs / CSV are written (output)

# (legend display name, subfolder on disk)
FRAMEWORKS = [
    ("SYCL",         "SYCL"),
    ("OpenMP Off.",  "OpenMP"),
]

# Intel device-hierarchy mode: "composite" or "tile". Overridable with --mode.
HIERARCHY = "composite"

# summary.csv locations tried in order, first hit wins.
SUMMARY_CANDIDATES = [
    "{sub}/{hier}/summary.csv",
    "{sub}/summary_{hier}.csv",
    "{sub}/summary.csv",
]
STAT_COLUMN = "median_sec"    # or min_sec / max_sec

DEFAULT_NS = [512, 640, 768, 896, 1024, 1280]
DEFAULT_GPUS = [2, 4]

# Title text. {steps} is filled in automatically from the summary's steps column.
TITLE_SUFFIX_TEMPLATE = "({steps} time steps)."
GROUP_WIDTH = 0.8             # total width of one N's group of bars

PALETTE = ["b", "g", "r", "c", "m", "y", "k", "tab:pink",
           "tab:orange", "tab:brown", "tab:purple", "tab:olive"]
# ===============================================================


def gpu_count_of(prog: str):
    """Leading integer in the program name = number of GPUs (1-omp, 2-omp, ...)."""
    m = re.match(r"(\d+)", prog)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", prog)   # fallback: first integer anywhere
    return int(m.group(1)) if m else None


def find_summary(subdir: str, hierarchy: str):
    for pat in SUMMARY_CANDIDATES:
        p = os.path.join(PLOTS_DIR, pat.format(sub=subdir, hier=hierarchy))
        if os.path.isfile(p):
            return p
    return None


def collect(target_ns, hierarchy):
    """per_run[framework][N][prog][sample_key] = [time], read from summary.csv."""
    per_run = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    progs_seen = defaultdict(set)
    steps_by_n = defaultdict(set)           # N -> set of step counts seen
    scanned = used = skipped = 0
    target_ns = set(target_ns)

    for fw_name, subdir in FRAMEWORKS:
        path = find_summary(subdir, hierarchy)
        if path is None:
            print(f"[warn] no summary.csv for {fw_name} under {subdir}/ "
                  f"- its bars will be left empty")
            continue
        scanned += 1
        print(f"[info] {fw_name:12s} <- {path}")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if STAT_COLUMN not in (reader.fieldnames or []):
                print(f"[warn] no '{STAT_COLUMN}' column in {path}; "
                      f"columns are {reader.fieldnames}")
                continue
            for row in reader:
                try:
                    N = int(row["N"])
                    t = float(row[STAT_COLUMN])
                    samples = int(row.get("samples") or 1)
                except (KeyError, TypeError, ValueError):
                    skipped += 1
                    continue
                prog = (row.get("variant") or "").strip()
                if not prog or N not in target_ns or t <= 0 or samples <= 0:
                    skipped += 1
                    continue
                # one pseudo-run per sample, so n_runs in the table stays truthful
                for i in range(samples):
                    per_run[fw_name][N][prog][f"s{i + 1}"].append(t)
                progs_seen[fw_name].add(prog)
                if (row.get("steps") or "").strip().isdigit():
                    steps_by_n[N].add(int(row["steps"]))
                used += 1
    return per_run, progs_seen, steps_by_n, scanned, used, skipped


def build_series_for_gpu(progs_seen, gpu_count):
    """Multi-GPU variants for `gpu_count`, preceded by the 1-GPU baselines
    (SYCL and OpenMP) so every chart carries the single-GPU reference.
    A framework with no data still gets its slots, so bars stay aligned."""
    # 1-GPU baselines, one (or more) per framework
    baseline = []
    for fw_name, _ in FRAMEWORKS:
        ones = sorted(p for p in progs_seen[fw_name] if gpu_count_of(p) == 1)
        if not ones:                       # keep the empty slot
            baseline.append((f"{fw_name} 1-GPU Baseline", fw_name, None))
            continue
        for i, p in enumerate(ones):
            tag = f" {i + 1}" if len(ones) > 1 else ""
            baseline.append((f"{fw_name} 1-GPU Baseline{tag}", fw_name, p))

    # the actual multi-GPU variants, interleaved SYCL / OpenMP
    fw_progs = {fw: sorted(p for p in progs_seen[fw] if gpu_count_of(p) == gpu_count)
                for fw, _ in FRAMEWORKS}
    max_v = max((len(v) for v in fw_progs.values()), default=0)
    series = list(baseline)
    for v in range(max_v):
        for fw_name, _ in FRAMEWORKS:
            progs = fw_progs[fw_name]
            label = f"{fw_name} {gpu_count}-GPU Version {v + 1}"
            # prog=None -> no value -> gap in the chart, slot preserved
            series.append((label, fw_name, progs[v] if v < len(progs) else None))
    return series


def compute_averages(per_run, series, ns):
    """Average within each run folder, then across run folders."""
    avg = defaultdict(dict)
    counts = defaultdict(dict)
    for N in ns:
        for label, fw_name, prog in series:
            run_map = per_run[fw_name][N].get(prog, {})
            run_values = [float(np.mean(v)) for v in run_map.values() if v]
            if run_values:
                avg[N][label] = float(np.mean(run_values))
                counts[N][label] = len(run_values)
    return avg, counts


def nice_round(n):
    """Round a step count to a clean number (e.g. 501 -> 500)."""
    n = int(n)
    if n <= 0:
        return n
    digits = 2 - int(math.floor(math.log10(n))) - 1   # 2 significant figures
    return int(round(n, digits))


def steps_suffix(steps_by_n, ns):
    """Build the title suffix from detected step counts across the plotted N."""
    raw = {s for N in ns for s in steps_by_n.get(N, set())}
    if not raw:
        return "(time steps unknown)."
    rounded = sorted({nice_round(s) for s in raw})
    steps = f"~{rounded[0]}" if len(rounded) == 1 else f"{rounded[0]}\u2013{rounded[-1]}"
    return TITLE_SUFFIX_TEMPLATE.format(steps=steps)


def plot_chart(series, avg, ns, gpu_count, out_pdf, title_suffix, hierarchy, show=True):
    if not series:
        print(f"[skip] no {gpu_count}-GPU programs found, no chart drawn.")
        return
    n_series = len(series)
    barWidth = GROUP_WIDTH / n_series
    x = np.arange(len(ns))
    colors = {lab: PALETTE[i % len(PALETTE)] for i, (lab, _, _) in enumerate(series)}
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, (label, fw, prog) in enumerate(series):
        vals = [avg.get(N, {}).get(label, np.nan) for N in ns]
        hatch = "//" if "Baseline" in label else None
        ax.bar(x + i * barWidth, vals, width=barWidth, color=colors[label],
               edgecolor="grey", linewidth=0.01, label=label, hatch=hatch)

    # ---- annotate speedup (x) on the OpenMP Off. Version-4 bar ----
    base_label = "OpenMP Off. 1-GPU Baseline"
    v4_label   = f"OpenMP Off. {gpu_count}-GPU Version 4"
    v4_idx = next((i for i, (lab, _, _) in enumerate(series) if lab == v4_label), None)
    if v4_idx is not None:
        for j, N in enumerate(ns):
            base_t = avg.get(N, {}).get(base_label)
            v4_t   = avg.get(N, {}).get(v4_label)
            if base_t and v4_t:
                speedup = base_t / v4_t
                ax.annotate(f"{speedup:.2f}×",
                            xy=(x[j] + v4_idx * barWidth, v4_t),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=MEDIUM_SIZE - 3, rotation=90,
                            color="black")
    # ------------------------------------------------------------

    ax.set_xlabel("Grid Size ($N^3$).", fontsize=MEDIUM_SIZE)
    ax.set_ylabel("Solver Time in Seconds.", fontsize=MEDIUM_SIZE)
    ax.set_title(f"Intel 1550 - {gpu_count} GPUs ({hierarchy}) {title_suffix}")
    ax.set_xticks(x + (n_series - 1) / 2.0 * barWidth)
    ax.set_xticklabels([f"${n}^3$" for n in ns])
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth=0.5, alpha=0.2)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.2)
    # Legend inside the plot, matching the H100 chart: a couple of columns and
    # a slightly smaller font so all the variants fit without spilling out.
    ax.legend(ncol=2, loc="upper left", fontsize=MEDIUM_SIZE,
              framealpha=0.85, columnspacing=1.0, handletextpad=0.4,
              labelspacing=0.3)
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    print(f"Wrote: {out_pdf}")
    if show:
        plt.show()
    plt.close(fig)


def print_table(series, avg, counts, ns, gpu_count):
    print(f"\n===== {gpu_count}-GPU AVERAGES (with 1-GPU baseline) =====")
    base_label = "OpenMP Off. 1-GPU Baseline"
    v4_label   = f"OpenMP Off. {gpu_count}-GPU Version 4"
    for N in ns:
        print(f"--- N = {N} ---")
        for label, _, _ in series:
            if label in avg.get(N, {}):
                print(f"{label:38s}: {avg[N][label]:.6f} s  (n_runs={counts[N][label]})")
            else:
                print(f"{label:38s}: MISSING")
        base_t = avg.get(N, {}).get(base_label)
        v4_t   = avg.get(N, {}).get(v4_label)
        if base_t and v4_t:
            speedup = base_t / v4_t
            print(f"{'OpenMP Off. V4 speedup vs Baseline':38s}: {speedup:.2f}×")
        print()


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GPUs", "N", "Method", "Time_seconds", "n_runs"])
        w.writerows(rows)
    print(f"Wrote: {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="SYCL vs OpenMP Offloading bar charts (Intel GPU).")
    p.add_argument("outdir", nargs="?", default=OUT_DIR,
                   help=f"folder for the PDFs and CSV, created if missing "
                        f"(default: {OUT_DIR})")
    p.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS,
                   help=f"grid sizes to plot (default: {DEFAULT_NS})")
    p.add_argument("--gpus", type=int, nargs="+", default=DEFAULT_GPUS,
                   help=f"which GPU-count charts to draw (default: {DEFAULT_GPUS})")
    p.add_argument("--mode", choices=["composite", "tile"], default=HIERARCHY,
                   help=f"Intel device-hierarchy folder to read (default: {HIERARCHY})")
    p.add_argument("--stat", choices=["min_sec", "median_sec", "max_sec"],
                   default=STAT_COLUMN,
                   help=f"summary.csv column to plot (default: {STAT_COLUMN})")
    p.add_argument("--no-show", action="store_true",
                   help="save the PDFs without opening plot windows")
    return p.parse_args()


def main():
    global STAT_COLUMN
    args = parse_args()
    STAT_COLUMN = args.stat

    ns = sorted(args.ns)
    hierarchy = args.mode
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    per_run, progs_seen, steps_by_n, scanned, used, skipped = collect(ns, hierarchy)

    suffix = steps_suffix(steps_by_n, ns)

    print("\n===== DISCOVERY =====")
    print(f"Hierarchy mode     : {hierarchy}")
    print(f"N sizes requested  : {ns}")
    print(f"Detected timesteps : {suffix}")
    print(f"Statistic plotted  : {STAT_COLUMN}")
    print(f"Output folder      : {outdir}")
    for fw_name, _ in FRAMEWORKS:
        progs = sorted(progs_seen[fw_name])
        tagged = [f"{p}(gpu={gpu_count_of(p)})" for p in progs]
        print(f"{fw_name:18s} programs: {tagged}")
    print(f"Summary files read : {scanned}")
    print(f"Rows used          : {used} (skipped: {skipped})")

    csv_rows = []
    for gpu in args.gpus:
        series = build_series_for_gpu(progs_seen, gpu)
        avg, counts = compute_averages(per_run, series, ns)
        print_table(series, avg, counts, ns, gpu)
        plot_chart(series, avg, ns, gpu,
                   out_pdf=os.path.join(outdir,
                                        f"Diffusion_intel_{hierarchy}_{gpu}gpu.pdf"),
                   title_suffix=suffix, hierarchy=hierarchy, show=not args.no_show)
        for N in ns:
            for label, fw, prog in series:
                if label in avg.get(N, {}):
                    # use the program's REAL gpu count so baseline rows stay truthful
                    csv_rows.append([gpu_count_of(prog), N, label,
                                     f"{avg[N][label]:.6f}", counts[N][label]])

    write_csv(csv_rows, os.path.join(outdir,
                                     f"measurement_avg_intel_{hierarchy}.csv"))


if __name__ == "__main__":
    main()
