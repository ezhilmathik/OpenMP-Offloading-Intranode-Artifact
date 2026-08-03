#!/usr/bin/env python3
"""
plot_p2p.py — Bandwidth vs N for versions v1, v2, v4 (P2P transfer, H100)

Usage:
    python3 plot_p2p.py

Automatically finds all results_*/timings.csv files in the current directory,
averages bandwidth across them, and writes the output to plots_final/.

Handles both row layouts:
    framework,version,N,elapsed_sec,bandwidth_gbs,status
    framework,version,N,,elapsed_sec,bandwidth_gbs,status
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Shared plot style — KEEP THIS BLOCK IDENTICAL IN EVERY PANEL SCRIPT
# so that panels placed side by side in one LaTeX figure match exactly.
#
# Effective on-page font size = matplotlib size x (panel width / FIGSIZE width).
# Panels go in at width=0.32\fulllength ~= 6.9 cm = 2.72 in, so:
#     FIGSIZE width 10 in -> scale 0.27 -> 22pt renders as  6.0pt  (too small)
#     FIGSIZE width  6 in -> scale 0.45 -> 22pt renders as 10.0pt  (good)
# MDPI wants >= ~8pt in figures. Shrink FIGSIZE rather than raise the fonts.
# ─────────────────────────────────────────────────────────────────────────────
FIGSIZE          = (10, 6.5)    # bigger = more plot area, smaller-looking text
                              # keep this IDENTICAL in all three scripts
FS_AXIS_LABEL    = 22
FS_TICK          = 18
FS_LEGEND        = 16
FS_THEORETICAL   = 16
XTICK_ROTATION   = 10
LINEWIDTH        = 2
MARKERSIZE       = 8
THEORETICAL_LW   = 1.5
LEGEND_ANCHOR    = (0.125, 0.9)
XLABEL           = "Transferred Size (GB, type: double)"
YLABEL           = "Bandwidth (GB/s)"

# ─────────────────────────────────────────────────────────────────────────────
# Config — per-script
# ─────────────────────────────────────────────────────────────────────────────
CSV_GLOB         = "results_*/timings.csv"
OUTPUT_DIR       = "plots_final"
OUTPUT_NAME      = "h100_p2p.pdf"
VERSIONS_TO_PLOT = ["v1", "v2"]

# Theoretical peer-link bandwidth (GB/s).
#   v1 = 1 GPU, v2 = 2 GPUs, v4 = 4 GPUs
# Map version -> (bandwidth, label text). Set to {} or None to disable.
THEORETICAL_BW = {
    "v1": (150, "1 GPU"),
    "v2": (300, "2 GPUs"),
    "v4": (450, "4 GPUs"),
}

# Where to anchor the theoretical-line labels: "right" or "left".
THEORETICAL_LABEL_SIDE = "right"

# Colour the theoretical lines per version instead of gray.
# Set to False to match the htod panel's gray dashed lines exactly.
THEORETICAL_PER_VERSION_COLOR = True

VERSION_COLOR = {"v1": "blue", "v2": "green", "v4": "red"}

# ── CUDA-vs-OpenMP gap box ───────────────────────────────────────────────────
# Prints the mean absolute bandwidth difference between CUDA and OpenMP Off.
# per version, in a small corner box. No shading; just the numbers.
SHOW_GAP_TEXTBOX = True
GAP_TEXTBOX_LOC  = (0.02, 0.12)   # axis fraction; (x, y), bottom-left origin
GAP_DIRECTION    = "P2P"          # label shown in the box header

SERIES_ORDER = [
    ("cuda",    "v1"),
    ("omp_off", "v1"),
    ("cuda",    "v2"),
    ("omp_off", "v2"),
    ("cuda",    "v4"),
    ("omp_off", "v4"),
]

SERIES = {
    ("cuda",    "v1"): ("CUDA (1 GPU)",     "blue",  "o"),
    ("omp_off", "v1"): ("OpenMP Off.  (1 GPU)",  "blue",  "^"),
    ("cuda",    "v2"): ("CUDA (2 GPUs)",    "green", "^"),
    ("omp_off", "v2"): ("OpenMP Off.  (2 GPUs)", "green", "D"),
    ("cuda",    "v4"): ("CUDA (4 GPUs)",    "red",   "D"),
    ("omp_off", "v4"): ("OpenMP Off.  (4 GPUs)", "red",   "P"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
DTYPE_BYTES = 8          # double


def gb_label(n, pos=None):
    """N doubles -> size in GB (decimal, the same convention as bandwidth_gbs)."""
    return f"{n * DTYPE_BYTES / 1e9:.2f}"


def parse_row(fields):
    """
    Accept:
      6 fields: framework, version, N, elapsed_sec, bandwidth_gbs, status
      7 fields: framework, version, N, '', elapsed_sec, bandwidth_gbs, status
    """
    if len(fields) == 6:
        fw, ver, n_str, _elapsed, bw_str, status = fields
    elif len(fields) == 7:
        fw, ver, n_str, _empty, _elapsed, bw_str, status = fields
    else:
        return None

    fw = fw.strip()
    ver = ver.strip()
    status = status.strip()
    bw_str = bw_str.strip()

    if fw == "framework":
        return None
    if ver not in VERSIONS_TO_PLOT:
        return None
    if status != "ok":
        return None
    if not bw_str or bw_str == "N/A":
        return None

    try:
        N = int(n_str.strip())
        bw = float(bw_str)
    except ValueError:
        return None

    return fw, ver, N, bw


def load_csv(path):
    data = defaultdict(list)
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        for fields in reader:
            result = parse_row(fields)
            if result is None:
                continue
            fw, ver, N, bw = result
            data[(fw, ver, N)].append(bw)
    return data


def load_and_average(csv_paths):
    combined = defaultdict(list)
    for path in csv_paths:
        for (fw, ver, N), bws in load_csv(path).items():
            combined[(fw, ver, N)].extend(bws)

    averaged = defaultdict(dict)
    for (fw, ver, N), bws in combined.items():
        averaged[(fw, ver)][N] = float(np.mean(bws))
    return averaged


def gap_stats(cuda_nd, omp_nd):
    """Mean absolute CUDA-vs-OMP bandwidth difference at sizes measured by BOTH.
    Returns None if the two frameworks share no transfer sizes."""
    common_N = sorted(set(cuda_nd) & set(omp_nd))
    if not common_N:
        return None

    cuda_bws = np.array([cuda_nd[n] for n in common_N])
    omp_bws  = np.array([omp_nd[n]  for n in common_N])
    abs_gap  = np.abs(cuda_bws - omp_bws)          # GB/s, per size

    return {
        "mean_abs_gap": float(np.mean(abs_gap)),   # GB/s
        "max_abs_gap":  float(np.max(abs_gap)),    # GB/s
    }


def active_theoretical_levels():
    """Theoretical levels for the versions actually being plotted, low to high."""
    if not THEORETICAL_BW:
        return []
    levels = [(bw, what, ver) for ver, (bw, what) in THEORETICAL_BW.items()
              if ver in VERSIONS_TO_PLOT]
    return sorted(levels, key=lambda item: item[0])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    csv_paths = sorted(glob.glob(CSV_GLOB))
    if not csv_paths:
        print(f"No files matched '{CSV_GLOB}' in {os.getcwd()}")
        return

    print(f"Found {len(csv_paths)} file(s):")
    for p in csv_paths:
        print(f"  {p}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_and_average(csv_paths)

    all_N = sorted({
        n
        for (fw, ver), nd in data.items()
        if ver in VERSIONS_TO_PLOT
        for n in nd
    })

    if not all_N:
        print("No valid bandwidth data found — nothing to plot.")
        return

    x_pos = list(range(len(all_N)))
    n_to_x = {n: i for i, n in enumerate(all_N)}

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax1.set_title("NVIDIA H100 - P2P", fontsize=16, fontweight="bold", pad=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [gb_label(n) for n in all_N],
        rotation=XTICK_ROTATION,
        fontsize=FS_TICK
    )
    ax1.set_xlabel(XLABEL, fontsize=FS_AXIS_LABEL)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.tick_params(axis="y", left=False, labelleft=False)

    ax2 = ax1.twinx()

    series_handles = []
    max_measured = 0.0
    for key in SERIES_ORDER:
        label, color, marker = SERIES[key]
        nd = data.get(key, {})
        if not nd:
            continue

        Ns_sorted = sorted(nd)
        xs = [n_to_x[n] for n in Ns_sorted]
        bws = [nd[n] for n in Ns_sorted]

        line, = ax2.plot(xs, bws, marker=marker, color=color,
                         linewidth=LINEWIDTH, markersize=MARKERSIZE, label=label)
        series_handles.append(line)
        max_measured = max(max_measured, max(bws))

    # ── CUDA vs OpenMP gap: quantify only, no shading ──────────────────────
    gap_lines = [f"Mean absolute {GAP_DIRECTION} gap"]
    for ver in VERSIONS_TO_PLOT:
        st = gap_stats(data.get(("cuda", ver), {}),
                       data.get(("omp_off", ver), {}))
        if st is None:
            continue
        gpu_label = SERIES[("omp_off", ver)][0].split("(")[1].rstrip(")").strip()
        gap_lines.append(f"CUDA vs. OpenMP Off. ({gpu_label}):  "
                         f"{st['mean_abs_gap']:.1f} GB/s")
        print(f"{ver}: mean |CUDA-OMP| = {st['mean_abs_gap']:.1f} GB/s "
              f"(max = {st['max_abs_gap']:.1f} GB/s)")

    # ── Theoretical bandwidth: one dashed line per version, NOT in legend ──
    #    v1 -> 150 GB/s, v2 -> 300 GB/s, v4 -> 450 GB/s
    levels = active_theoretical_levels()
    ymax = max_measured * 1.10
    if levels:
        if THEORETICAL_LABEL_SIDE == "left":
            label_x, label_ha = x_pos[0], "left"
        else:
            label_x, label_ha = x_pos[-1], "right"

        ymax = max(max_measured * 1.10, max(bw for bw, _, _ in levels) * 1.08)
        for bw, what, ver in levels:
            color = VERSION_COLOR[ver] if THEORETICAL_PER_VERSION_COLOR else "gray"
            ax2.axhline(bw, linestyle="--", color=color, linewidth=THEORETICAL_LW)
            ax2.text(
                label_x,
                bw + 0.01 * ymax,
                f"Theoretical BW, {what} ({bw} GB/s)",
                color=color,
                ha=label_ha,
                va="bottom",
                fontsize=FS_THEORETICAL
            )

    # ── Mean-gap summary box ─────────────────────────────────────────────────
    if SHOW_GAP_TEXTBOX and len(gap_lines) > 1:
        ax2.text(
            GAP_TEXTBOX_LOC[0], GAP_TEXTBOX_LOC[1], "\n".join(gap_lines),
            transform=ax2.transAxes, fontsize=FS_LEGEND,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85)
        )

    ax2.set_ylabel(YLABEL, fontsize=FS_AXIS_LABEL)
    ax2.tick_params(axis="y", labelsize=FS_TICK)

    # axhline does not autoscale, so set the range explicitly.
    ax2.set_ylim(0, ymax)

    fig.legend(
        handles=series_handles,
        loc="upper left",
        bbox_to_anchor=LEGEND_ANCHOR,
        fontsize=FS_LEGEND
    )

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
