#!/usr/bin/env python3
"""
Bar-chart averaging script for CUDA vs OpenMP Offloading.

Draws TWO charts: one for the 2-GPU program variants and one for the 4-GPU
variants. Each program in a slurm file is split by the leading number in its
name (2-gpu* -> 2-GPU chart, 4-gpu* -> 4-GPU chart).

The single-GPU programs (1-gpu*) are no longer ignored: their timings are
added to BOTH the 2-GPU and 4-GPU charts as a single-GPU *baseline* reference
(drawn with a hatch so they stand out from the multi-GPU variants).

INPUT PATH
----------
The timings are read from the per-framework summary.csv written by
aggregate.sh, NOT from the slurm .out files:

    <PLOTS_DIR>/CUDA/summary.csv
    <PLOTS_DIR>/OpenMP/summary.csv

Run this script from the directory that contains the CUDA/ and OpenMP/
folders (i.e. NVIDIA-H100/), because PLOTS_DIR is ".".

summary.csv schema:
    variant,gpus,N,samples,failed,min_sec,median_sec,max_sec,rel_spread_pct

Examples:
    python3 plot_multi_gpu.py                      # default N sizes, both charts
    python3 plot_multi_gpu.py --ns 512 768 1024    # pick N sizes on the command line
    python3 plot_multi_gpu.py --gpus 4             # only the 4-GPU chart
    python3 plot_multi_gpu.py --ns 1024 1280 --no-show

The framework (CUDA vs OpenMP Offloading) is decided by which folder a
summary.csv is in, so program names do not need to be hard-coded; they are
auto-discovered from the "variant" column.
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
MEDIUM_SIZE = 14
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=MEDIUM_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=MEDIUM_SIZE)

# ========================= USER CONFIG =========================
PLOTS_DIR = "."

# (legend display name, subfolder on disk)
FRAMEWORKS = [
    ("CUDA",              "CUDA"),
    ("OpenMP Off.", "OpenMP"),
]

# ---- input file (produced by aggregate.sh) ----
SUMMARY_NAME = "summary.csv"   # looked up as <PLOTS_DIR>/<subdir>/summary.csv
TIME_COL     = "median_sec"    # "min_sec" / "median_sec" / "max_sec"

# Legend name for a framework folder named on the command line. Anything not
# listed here is used verbatim, so "speedup.py plots OpenMP CUDA" still labels
# the OpenMP bars "OpenMP Off." (which the speedup annotation looks for).
FRAMEWORK_ALIASES = {
    "OpenMP": "OpenMP Off.",
    "CUDA":   "CUDA",
}

RUN_GLOB = "run-*"      # auto-detect run-1, run-2, ... run-N
SLURM_GLOB = "*.out"    # parse every .out file inside each run folder

DEFAULT_NS = [512, 640, 768, 896, 1024, 1152, 1280]
DEFAULT_GPUS = [2, 4]

# Filename "mode" tokens to keep (your files are *_allmode_*).
# None means keep every mode. Overridable with --modes on the command line.
INCLUDE_MODES = {"allmode"}
INCLUDE_UNKNOWN_MODE = True   # keep files with no recognizable mode token

# Title text. {steps} is filled in automatically from the .out files.
TITLE_SUFFIX_TEMPLATE = "({steps} time steps)."
GROUP_WIDTH = 0.8             # total width of one N's group of bars

PALETTE = ["b", "g", "r", "c", "m", "y", "k", "tab:pink",
           "tab:orange", "tab:brown", "tab:purple", "tab:olive"]
# ===============================================================

RUNNING_RE = re.compile(r"Running:\s*\./(?P<prog>\S+)\s+(?P<N>\d+)")
TIME_RE    = re.compile(r"Time\s*=\s*(?P<t>[0-9]*\.?[0-9eE+-]+)\s*seconds")
MODE_TOKENS = ("2mode", "4mode", "8mode", "allmode")

# Auto-detect the number of time steps.
STEPS_DONE_RE = re.compile(r"(\d+)\s+timesteps?\s+complete", re.IGNORECASE)
STEPS_EST_RE  = re.compile(r"Estimated\s+timesteps:\s*~?\s*(\d+)")


def normalize_prog(p: str) -> str:
    p = os.path.basename(p)
    for suf in (".exe", ".x", ".out"):
        if p.endswith(suf):
            p = p[:-len(suf)]
    return p


def gpu_count_of(prog: str):
    """Leading integer in the program name = number of GPUs (1-gpu, 2-gpu, ...)."""
    m = re.match(r"(\d+)", prog)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", prog)   # fallback: first integer anywhere
    return int(m.group(1)) if m else None


def detect_mode(filename: str):
    for m in MODE_TOKENS:
        if m in filename:
            return m
    return None


def mode_ok(mode) -> bool:
    if mode is None:
        return INCLUDE_UNKNOWN_MODE
    if INCLUDE_MODES is None:
        return True
    return mode in INCLUDE_MODES


def run_sort_key(path: str):
    m = re.search(r"run-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 1_000_000


def find_run_folders(framework_base: str):
    dirs = [d for d in glob.glob(os.path.join(framework_base, RUN_GLOB)) if os.path.isdir(d)]
    return sorted(dirs, key=run_sort_key)


def parse_slurm_file(path: str):
    with open(path, "r", errors="ignore") as f:
        text = f.read()
    runs = list(RUNNING_RE.finditer(text))
    out = []
    for i, m in enumerate(runs):
        prog = normalize_prog(m.group("prog"))
        N = int(m.group("N"))
        start = m.start()
        end = runs[i + 1].start() if i + 1 < len(runs) else len(text)
        segment = text[start:end]
        tm = TIME_RE.search(segment)
        if not tm:
            continue                       # e.g. a run that failed / "Permission denied"
        # Prefer the actual completed-step count; fall back to the estimate.
        sm = STEPS_DONE_RE.search(segment) or STEPS_EST_RE.search(segment)
        steps = int(sm.group(1)) if sm else None
        out.append((prog, N, float(tm.group("t")), steps))
    return out


def collect(target_ns):
    """per_run[framework][N][prog][run_folder] = [times]

    Reads <PLOTS_DIR>/<subdir>/summary.csv for each framework. aggregate.sh
    has already averaged the repeats, so each (prog, N) contributes exactly
    one value under the pseudo run key "summary"; the downstream averaging in
    compute_averages() then passes it straight through.
    """
    per_run = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
    progs_seen = defaultdict(set)
    steps_by_n = defaultdict(set)           # N -> set of step counts seen
    scanned = used = skipped_mode = 0
    target_ns = set(target_ns)

    for fw_name, subdir in FRAMEWORKS:
        path = os.path.join(PLOTS_DIR, subdir, SUMMARY_NAME)
        if not os.path.isfile(path):
            print(f"[warn] missing summary file: {path}")
            continue
        scanned += 1
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            missing = {"variant", "N", TIME_COL} - set(reader.fieldnames or [])
            if missing:
                print(f"[warn] {path}: missing column(s) {sorted(missing)}; "
                      f"found {reader.fieldnames}")
                continue
            for row in reader:
                prog = normalize_prog(row["variant"].strip())
                N = int(row["N"])
                if N not in target_ns:
                    continue
                per_run[fw_name][N][prog]["summary"].append(float(row[TIME_COL]))
                progs_seen[fw_name].add(prog)
                # optional "steps" column (added by aggregate.sh) feeds the title
                if row.get("steps"):
                    steps_by_n[N].add(int(float(row["steps"])))
                used += 1
        print(f"[ok] read {path}")
    return per_run, progs_seen, steps_by_n, scanned, used, skipped_mode


def build_series_for_gpu(progs_seen, gpu_count):
    """Multi-GPU variants for `gpu_count`, preceded by the 1-GPU baselines
    (CUDA and OpenMP) so every chart carries the single-GPU reference.

    Returns a list of (label, framework, prog) tuples. Baseline entries have
    "Baseline" in their label so plot_chart can render them with a hatch.
    """
    # 1-GPU baselines, one (or more) per framework
    baseline = []
    for fw_name, _ in FRAMEWORKS:
        ones = sorted(p for p in progs_seen[fw_name] if gpu_count_of(p) == 1)
        for i, p in enumerate(ones):
            tag = f" {i + 1}" if len(ones) > 1 else ""
            baseline.append((f"{fw_name} 1-GPU Baseline{tag}", fw_name, p))

    # the actual multi-GPU variants, interleaved CUDA / OpenMP as before
    fw_progs = {fw: sorted(p for p in progs_seen[fw] if gpu_count_of(p) == gpu_count)
                for fw, _ in FRAMEWORKS}
    max_v = max((len(v) for v in fw_progs.values()), default=0)
    series = list(baseline)
    for v in range(max_v):
        for fw_name, _ in FRAMEWORKS:
            progs = fw_progs[fw_name]
            if v < len(progs):
                label = f"{fw_name} {gpu_count}-GPU Version {v + 1}"
                series.append((label, fw_name, progs[v]))
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
    """Round a step count to a clean number (e.g. 501 -> 500, 4999 -> 5000).

    Rounds to 2 significant figures so off-by-one counts like 500/501 collapse
    to one value, while 100, 500, 1000, 5000 etc. stay as themselves.
    """
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

'''
def plot_chart(series, avg, ns, gpu_count, out_pdf, title_suffix, show=True):
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

    ax.set_xlabel("Grid Size ($N^3$).", fontsize=MEDIUM_SIZE)
    ax.set_ylabel("Solver Time in Seconds.", fontsize=MEDIUM_SIZE)
    ax.set_title(f"NVIDIA H100 - {gpu_count} GPUs {title_suffix}")
    ax.set_xticks(x + (n_series - 1) / 2.0 * barWidth)
    ax.set_xticklabels([f"${n}^3$" for n in ns])

    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth=0.5, alpha=0.2)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.2)

    # Legend below the axes so it never overflows the plot box, however many
    # variants there are. Two columns per framework keeps CUDA/OpenMP aligned.
    ncol = min(len(FRAMEWORKS), n_series) or 1
    ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, columnspacing=1.2, handletextpad=0.5)

    # Legend inside the plot. Use a couple of columns and a slightly smaller
    # font so all the variants fit without spilling out of the axes box.
    ax.legend(ncol=2, loc="upper left", fontsize=MEDIUM_SIZE,
              framealpha=0.85, columnspacing=1.0, handletextpad=0.4,
              labelspacing=0.3)
    
    # bbox_inches="tight" expands the saved figure to include the outside legend.
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    print(f"Wrote: {out_pdf}")
    if show:
        plt.show()
    plt.close(fig)
'''


def plot_chart(series, avg, ns, gpu_count, out_pdf, title_suffix, show=True):
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

    # ---- annotate speedup (x faster) on the OpenMP Off. Version-4 bar ----
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
    ax.set_title(f"NVIDIA H100 - {gpu_count} GPUs {title_suffix}")
    ax.set_xticks(x + (n_series - 1) / 2.0 * barWidth)
    ax.set_xticklabels([f"${n}^3$" for n in ns])

    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", linewidth=0.5, alpha=0.2)
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.2)

    ncol = min(len(FRAMEWORKS), n_series) or 1
    ax.legend(ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, columnspacing=1.2, handletextpad=0.5)

    ax.legend(ncol=2, loc="upper left", fontsize=MEDIUM_SIZE,
              framealpha=0.85, columnspacing=1.0, handletextpad=0.4,
              labelspacing=0.3)

    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    print(f"Wrote: {out_pdf}")
    if show:
        plt.show()
    plt.close(fig)
    

def print_table(series, avg, counts, ns, gpu_count):
    print(f"\n===== {gpu_count}-GPU AVERAGES (with 1-GPU baseline) =====")
    for N in ns:
        print(f"--- N = {N} ---")
        for label, _, _ in series:
            if label in avg.get(N, {}):
                print(f"{label:38s}: {avg[N][label]:.6f} s  (n_runs={counts[N][label]})")
            else:
                print(f"{label:38s}: MISSING")
        print()


def write_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GPUs", "N", "Method", "Time_seconds", "n_runs"])
        w.writerows(rows)
    print(f"Wrote: {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description="CUDA vs OpenMP Offloading bar charts.")
    p.add_argument("outdir", nargs="?", default=".",
                   help="folder for the PDFs and measurement_avg.csv "
                        "(created if needed; default: current directory)")
    p.add_argument("frameworks", nargs="*",
                   help="framework folders to read summary.csv from, in bar "
                        f"order (default: {[s for _, s in FRAMEWORKS]})")
    p.add_argument("--ns", type=int, nargs="+", default=DEFAULT_NS,
                   help=f"grid sizes to plot (default: {DEFAULT_NS})")
    p.add_argument("--gpus", type=int, nargs="+", default=DEFAULT_GPUS,
                   help=f"which GPU-count charts to draw (default: {DEFAULT_GPUS})")
    p.add_argument("--modes", nargs="+", default=None,
                   help="filename mode tokens to include, or 'all' (default: allmode)")
    p.add_argument("--no-show", action="store_true",
                   help="save the PDFs without opening plot windows")
    return p.parse_args()


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
        # Speedup of OpenMP Off. Version-4 vs OpenMP Off. 1-GPU Baseline
        base_t = avg.get(N, {}).get(base_label)
        v4_t   = avg.get(N, {}).get(v4_label)
        if base_t and v4_t:
            speedup = base_t / v4_t
            pct = (speedup - 1.0) * 100.0
            print(f"{'OpenMP Off. V4 speedup vs Baseline':38s}: "
                  f"{speedup:.2f}x  ({pct:+.1f}%)")
        print()

        
def main():
    global INCLUDE_MODES, FRAMEWORKS
    args = parse_args()
    if args.modes is not None:
        INCLUDE_MODES = None if args.modes == ["all"] else set(args.modes)
    if args.frameworks:
        FRAMEWORKS = [(FRAMEWORK_ALIASES.get(d, d), d) for d in args.frameworks]

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    ns = sorted(args.ns)
    per_run, progs_seen, steps_by_n, scanned, used, skipped_mode = collect(ns)

    suffix = steps_suffix(steps_by_n, ns)

    print("===== DISCOVERY =====")
    print(f"Output folder      : {os.path.abspath(outdir)}")
    print(f"N sizes requested  : {ns}")
    print(f"Detected timesteps : {suffix}")
    print(f"Time column        : {TIME_COL}")
    for fw_name, _ in FRAMEWORKS:
        progs = sorted(progs_seen[fw_name])
        tagged = [f"{p}(gpu={gpu_count_of(p)})" for p in progs]
        print(f"{fw_name:18s} programs: {tagged}")
    print(f"Summary files read : {scanned}")
    print(f"Used (prog,N,time) : {used}")

    csv_rows = []
    for gpu in args.gpus:
        series = build_series_for_gpu(progs_seen, gpu)
        avg, counts = compute_averages(per_run, series, ns)
        print_table(series, avg, counts, ns, gpu)
        plot_chart(series, avg, ns, gpu,
                   out_pdf=os.path.join(outdir, f"Diffusion_H100_{gpu}gpu.pdf"),
                   title_suffix=suffix, show=not args.no_show)
        for N in ns:
            for label, fw, prog in series:
                if label in avg.get(N, {}):
                    # use the program's REAL gpu count so baseline rows stay truthful
                    csv_rows.append([gpu_count_of(prog), N, label,
                                     f"{avg[N][label]:.6f}", counts[N][label]])

    write_csv(csv_rows, os.path.join(outdir, "measurement_avg.csv"))


if __name__ == "__main__":
    main()
