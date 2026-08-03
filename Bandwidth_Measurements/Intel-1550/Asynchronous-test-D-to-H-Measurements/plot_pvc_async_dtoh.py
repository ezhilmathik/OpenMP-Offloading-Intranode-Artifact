#!/usr/bin/env python3
"""
plot_htod.py — Bandwidth vs vector size for versions v1, v2
               (Async H-to-D transfer, Intel PVC)

Usage:
    python3 plot_htod.py

Automatically finds all results_*/timings.csv files in the current directory,
averages bandwidth across them, and writes the output to plots_final/.

The x-axis reports the vector size, derived from N:
    bytes = N * DTYPE_BYTES
Note this is the vector, not the total bytes moved: v2 transfers 2x
that amount at the same N.

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
XLABEL_TEMPLATE  = "Transferred size ({unit}, type: double)"
YLABEL           = "Bandwidth (GB/s)"

# ── X-axis size conversion — KEEP IDENTICAL IN EVERY PANEL SCRIPT ────────────
# The axis reports the VECTOR size, N * DTYPE_BYTES — the same quantity for
# every version, so one tick means one thing.  It is deliberately NOT the total
# bytes moved: at a given N, v2 moves 2x the vector and v4 moves 3x (verified
# against the CSVs: reported bandwidth / (N*8/elapsed) = 1, 2, 3 exactly).
# A shared x tick cannot carry three different totals, hence "Vector size".
#
# UNIT_BASE 1000 = decimal GB, matching a bandwidth computed as bytes/1e9/sec
# (confirmed: 940e6 / 0.024599 = 38.213 GB/s, as reported).  Use 1024 for GiB
# only if the benchmark divided by 2**30 instead, or the axes disagree.
#
# XAXIS_UNIT: "auto" picks one unit for the whole axis from the largest size.
# Pin it to "MB" or "GB" explicitly if the panels cover different N ranges,
# otherwise one panel may come out in MB and its neighbour in GB.
DTYPE_BYTES      = 8            # double
UNIT_BASE        = 1000         # 1000 -> KB/MB/GB, 1024 -> KiB/MiB/GiB
XAXIS_UNIT       = "GB"         # "auto" | "KB" | "MB" | "GB"

# ─────────────────────────────────────────────────────────────────────────────
# Config — per-script
# ─────────────────────────────────────────────────────────────────────────────
CSV_GLOB         = "results_*/timings.csv"
OUTPUT_DIR       = "plots_final"
OUTPUT_NAME      = "pvc_async_dtoh.pdf"
VERSIONS_TO_PLOT = ["v1", "v2"]

# Theoretical host-link bandwidth, one direction (GB/s).
# PCIe Gen5 x16 per GPU -> 64 GB/s each way.  VERIFY on your host.
#   v1 = 1 GPU  -> a single H-to-D link
#   v2 = 2 GPUs -> two independent H-to-D links driven in parallel
# Map version -> (bandwidth, label text). Set to {} or None to disable.
THEORETICAL_BW = {
    "v1": (64, "1 GPU"),
    "v2": (128, "2 GPUs"),
}

# Where to anchor the theoretical-line labels: "right" or "left".
THEORETICAL_LABEL_SIDE = "right"

# One colour per version: every series of that version, and its theoretical
# line, are drawn in it. Frameworks are told apart by marker, not colour.
VERSION_COLOR = {
    "v1": "blue",
    "v2": "green",
}

# Colour the theoretical lines per version instead of gray.
THEORETICAL_PER_VERSION_COLOR = True

# ── SYCL-vs-OpenMP gap box ───────────────────────────────────────────────────
# Prints the mean absolute bandwidth difference between SYCL and OpenMP Off.
# per version, in a small corner box. No shading; just the numbers.
SHOW_GAP_TEXTBOX = True
GAP_TEXTBOX_LOC  = (0.02, 0.06)   # axis fraction; (x, y), bottom-left origin
GAP_DIRECTION    = "DtoH"         # label shown in the box header
GAP_BASELINE_FW  = "sycl"         # native framework compared against omp_off
GAP_BASELINE_LBL = "SYCL"         # how that framework is named in the box

SERIES_ORDER = [
    ("sycl",    "v1"),
    ("omp_off", "v1"),
    ("sycl",    "v2"),
    ("omp_off", "v2"),
]

SERIES = {
    ("sycl",    "v1"): ("SYCL (1 GPU)",          "o"),
    ("omp_off", "v1"): ("OpenMP Off. (1 GPU)",   "s"),
    ("sycl",    "v2"): ("SYCL (2 GPUs)",         "^"),
    ("omp_off", "v2"): ("OpenMP Off. (2 GPUs)",  "v"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def n_to_bytes(n):
    return n * DTYPE_BYTES


def unit_suffix(unit):
    """'MB' -> 'MiB' when UNIT_BASE is 1024."""
    return unit if UNIT_BASE == 1000 else unit.replace("B", "iB")


def pick_unit(max_bytes):
    """One unit for the whole axis, chosen from the largest value on it."""
    if XAXIS_UNIT != "auto":
        return XAXIS_UNIT
    if max_bytes >= UNIT_BASE ** 3:
        return "GB"
    if max_bytes >= UNIT_BASE ** 2:
        return "MB"
    return "KB"


def unit_divisor(unit):
    return {"KB": UNIT_BASE, "MB": UNIT_BASE ** 2, "GB": UNIT_BASE ** 3}[unit]


def size_label(n, unit):
    """Tick label for N, in `unit`, with just enough decimals to stay distinct."""
    value = n_to_bytes(n) / unit_divisor(unit)
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def parse_row(fields):
    """
    Accept:
      6 fields: framework, version, N, elapsed_sec, bandwidth_gbs, status
      7 fields, blank 4th:   framework, version, N, '', elapsed, bw, status
      7 fields, trailing tag: framework, version, N, elapsed, bw, status, tag
    The two 7-field layouts are told apart by whether field 4 is empty.
    """
    fields = [f.strip() for f in fields]

    if len(fields) == 6:
        fw, ver, n_str, _elapsed, bw_str, status = fields
    elif len(fields) == 7 and fields[3] == "":
        fw, ver, n_str, _empty, _elapsed, bw_str, status = fields
    elif len(fields) == 7:
        fw, ver, n_str, _elapsed, bw_str, status, _tag = fields
    else:
        return None

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


def gap_stats(base_nd, omp_nd):
    """Mean absolute baseline-vs-OMP bandwidth difference at sizes measured by BOTH.
    Returns None if the two frameworks share no transfer sizes."""
    common_N = sorted(set(base_nd) & set(omp_nd))
    if not common_N:
        return None

    base_bws = np.array([base_nd[n] for n in common_N])
    omp_bws  = np.array([omp_nd[n]  for n in common_N])
    abs_gap  = np.abs(base_bws - omp_bws)          # GB/s, per size

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

    # Equal-spaced positions so uneven size gaps don't distort the plot
    x_pos = list(range(len(all_N)))
    n_to_x = {n: i for i, n in enumerate(all_N)}

    unit = pick_unit(n_to_bytes(all_N[-1]))
    print(f"X-axis unit: {unit_suffix(unit)} "
          f"({size_label(all_N[0], unit)} .. {size_label(all_N[-1], unit)})")

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax1.set_title("Intel 1550 (composite) - DtoH: asynchronous", fontsize=16, fontweight="bold", pad=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [size_label(n, unit) for n in all_N],
        rotation=XTICK_ROTATION,
        fontsize=FS_TICK
    )
    ax1.set_xlabel(XLABEL_TEMPLATE.format(unit=unit_suffix(unit)),
                   fontsize=FS_AXIS_LABEL)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.tick_params(axis="y", left=False, labelleft=False)

    ax2 = ax1.twinx()

    series_handles = []
    max_measured = 0.0
    for key in SERIES_ORDER:
        label, marker = SERIES[key]
        _fw, ver = key
        color = VERSION_COLOR.get(ver, "black")
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

    # ── SYCL vs OpenMP gap: quantify only, no shading ──────────────────────
    gap_lines = [f"Mean absolute {GAP_DIRECTION} gap"]
    for ver in VERSIONS_TO_PLOT:
        st = gap_stats(data.get((GAP_BASELINE_FW, ver), {}),
                       data.get(("omp_off", ver), {}))
        if st is None:
            continue
        gpu_label = SERIES[("omp_off", ver)][0].split("(")[1].rstrip(")").strip()
        gap_lines.append(f"{GAP_BASELINE_LBL} vs. OpenMP Off. ({gpu_label}):  "
                         f"{st['mean_abs_gap']:.1f} GB/s")
        print(f"{ver}: mean |{GAP_BASELINE_LBL}-OMP| = {st['mean_abs_gap']:.1f} GB/s "
              f"(max = {st['max_abs_gap']:.1f} GB/s)")

    # ── Theoretical bandwidth: one dashed line per version, NOT in legend ──
    #    v1 ->  64 GB/s   (one PCIe Gen5 x16 H-to-D link)   blue
    #    v2 -> 128 GB/s   (two H-to-D links in parallel)     green
    levels = active_theoretical_levels()
    ymax = max_measured * 1.10
    if levels:
        if THEORETICAL_LABEL_SIDE == "left":
            label_x, label_ha = x_pos[0], "left"
        else:
            label_x, label_ha = x_pos[-1], "right"

        ymax = max(max_measured * 1.10, max(bw for bw, _, _ in levels) * 1.08)
        for bw, what, ver in levels:
            color = VERSION_COLOR.get(ver, "black") \
                if THEORETICAL_PER_VERSION_COLOR else "gray"
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
