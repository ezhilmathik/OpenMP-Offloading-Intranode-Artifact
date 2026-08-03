#!/usr/bin/env python3
"""
plot_p2p.py — Bandwidth vs vector size for versions v1, v2, v4
              (P2P transfer, Intel PVC / Data Center GPU Max)

Usage:
    python3 plot_p2p.py

Automatically finds all results_*/timings.csv files in the current directory,
averages bandwidth across them, and writes the output to plots_final/.
Only frameworks actually present in the CSVs are plotted.

The x-axis reports the vector size, derived from N:
    bytes = N * DTYPE_BYTES
Note this is the vector, not the total bytes moved: v2 transfers 2x and
v4 transfers 3x that amount at the same N.

Handles these row layouts:
    framework,version,N,elapsed_sec,bandwidth_gbs,status
    framework,version,N,,elapsed_sec,bandwidth_gbs,status
    framework,version,N,elapsed_sec,bandwidth_gbs,status,hierarchy
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
XLABEL_TEMPLATE  = "Transferred Size ({unit}, type: double)"
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
XAXIS_UNIT       = "auto"       # "auto" | "KB" | "MB" | "GB"

# ─────────────────────────────────────────────────────────────────────────────
# Config — per-script
# ─────────────────────────────────────────────────────────────────────────────
CSV_GLOB         = "results_*/timings.csv"
OUTPUT_DIR       = "plots_final"
OUTPUT_NAME      = "pvc_p2p.pdf"
VERSIONS_TO_PLOT = ["v1", "v2"]

# Theoretical peak bandwidth (GB/s) per version.
# Map version -> bandwidth. Use None to skip one; set to {} to disable all.
#
# Xe Link peer bandwidth, PER DIRECTION (these transfers are one-way).
#
# One link per pair = 106 GB/s each way. Each version's ceiling is
# (number of concurrent transfers) x (slowest path):
#   v1 -> 1 transfer  x 106 = 106
#   v2 -> 2 transfers x 106 = 212
#   v4 -> 3 transfers x 106 = 318   <- 3, not 4: v4 is a 3-sender gather into
#                                      dev0. Verified against the CSV:
#                                      bw / (N*8/elapsed) = 3.0000 exactly.
# NOT the sum of every link's capacity: elapsed is set by the slowest transfer,
# the faster ones finish early and idle.
THEORETICAL_BW = {
    "v1": 106,
    "v2": 212,
    "v4": 318,
}

# Where to anchor the theoretical-line labels: "right" or "left".
THEORETICAL_LABEL_SIDE = "right"

# Colour the theoretical lines per version instead of gray.
# Set to False to match the htod panel's gray dashed lines exactly.
THEORETICAL_PER_VERSION_COLOR = True

# Only plot rows from this device hierarchy, or None to accept any.
# COMPOSITE = one device per card (2 stacks fused); FLAT = one device per
# stack. The two are not comparable — averaging them would be silently wrong.
HIERARCHY_FILTER = "COMPOSITE"

# ── Baseline-vs-OpenMP gap box ───────────────────────────────────────────────
# Prints the mean absolute bandwidth difference between the native framework
# and OpenMP Off. per version, in a small corner box. No shading; just numbers.
# GAP_BASELINE_FW must match a framework key present in the CSVs ("sycl" on
# PVC, "hip" on MI250X, "cuda" on H100). If it isn't present for a version,
# that line is skipped automatically.
SHOW_GAP_TEXTBOX = True
GAP_TEXTBOX_LOC  = (0.02, 0.45)   # axis fraction; (x, y), bottom-left origin
GAP_DIRECTION    = "P2P"          # label shown in the box header
GAP_BASELINE_FW  = "sycl"         # native framework compared against omp_off
GAP_BASELINE_LBL = "SYCL"         # how that framework is named in the box

# Prefer these display styles when frameworks are present
FRAMEWORK_LABELS = {
    "cuda":    "CUDA",
    "hip":     "HIP",
    "sycl":    "SYCL",
    "acc":     "OpenACC",
    "omp_off": "OpenMP Off.",
}

FRAMEWORK_MARKERS = {
    "cuda":    "o",
    "hip":     "D",
    "sycl":    "o",
    "acc":     "s",
    "omp_off": "^",
}

VERSION_COLOR = {
    "v1": "blue",
    "v2": "green",
    "v4": "red",
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


def gpu_label(ver):
    """'v1' -> '1 GPU', 'v2' -> '2 GPUs'."""
    count = ver.lstrip("v")
    return f"{count} GPU" if count == "1" else f"{count} GPUs"


def parse_row(fields):
    """
    Accept:
      6 fields: framework, version, N, elapsed_sec, bandwidth_gbs, status
      7, blank 4th:    framework, version, N, '', elapsed, bw, status
      7, trailing tag: framework, version, N, elapsed, bw, status, hierarchy
    The two 7-field layouts are told apart by whether field 4 is empty.
    """
    fields = [f.strip() for f in fields]
    tag = None

    if len(fields) == 6:
        fw, ver, n_str, _elapsed, bw_str, status = fields
    elif len(fields) == 7 and fields[3] == "":
        fw, ver, n_str, _empty, _elapsed, bw_str, status = fields
    elif len(fields) == 7:
        fw, ver, n_str, _elapsed, bw_str, status, tag = fields
    else:
        return None

    if fw == "framework":
        return None
    if not fw or ver not in VERSIONS_TO_PLOT:
        return None
    if HIERARCHY_FILTER and tag and tag != HIERARCHY_FILTER:
        return None
    if status != "ok":
        return None
    if not bw_str or bw_str == "N/A":
        return None

    try:
        N = int(n_str)
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
    levels = [(bw, gpu_label(ver), ver)
              for ver, bw in THEORETICAL_BW.items()
              if ver in VERSIONS_TO_PLOT and bw is not None]
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

    available_frameworks = sorted({
        fw for (fw, ver) in data if ver in VERSIONS_TO_PLOT
    })

    if not available_frameworks:
        print("No frameworks found for requested versions.")
        return

    # Equal-spaced positions so uneven size gaps don't distort the plot
    x_pos = list(range(len(all_N)))
    n_to_x = {n: i for i, n in enumerate(all_N)}

    unit = pick_unit(n_to_bytes(all_N[-1]))
    print(f"X-axis unit: {unit_suffix(unit)} "
          f"({size_label(all_N[0], unit)} .. {size_label(all_N[-1], unit)})")

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    ax1.set_title("Intel 1550 (composite - P2P)", fontsize=16, fontweight="bold", pad=12)
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

    # ── Measured data: only frameworks that actually exist in the CSVs ──
    series_handles = []
    max_measured = 0.0
    for ver in VERSIONS_TO_PLOT:
        color = VERSION_COLOR.get(ver, "black")
        for fw in available_frameworks:
            nd = data.get((fw, ver), {})
            if not nd:
                continue

            Ns_sorted = sorted(nd)
            xs = [n_to_x[n] for n in Ns_sorted]
            bws = [nd[n] for n in Ns_sorted]

            label = f"{FRAMEWORK_LABELS.get(fw, fw)} ({gpu_label(ver)})"
            marker = FRAMEWORK_MARKERS.get(fw, "o")

            line, = ax2.plot(xs, bws, marker=marker, color=color,
                             linewidth=LINEWIDTH, markersize=MARKERSIZE,
                             label=label)
            series_handles.append(line)
            max_measured = max(max_measured, max(bws))

    # ── Native framework vs OpenMP gap: quantify only, no shading ──────────
    gap_lines = [f"Mean absolute {GAP_DIRECTION} gap"]
    for ver in VERSIONS_TO_PLOT:
        st = gap_stats(data.get((GAP_BASELINE_FW, ver), {}),
                       data.get(("omp_off", ver), {}))
        if st is None:
            continue
        gap_lines.append(f"{GAP_BASELINE_LBL} vs. OpenMP Off. ({gpu_label(ver)}):  "
                         f"{st['mean_abs_gap']:.1f} GB/s")
        print(f"{ver}: mean |{GAP_BASELINE_LBL}-OMP| = {st['mean_abs_gap']:.1f} GB/s "
              f"(max = {st['max_abs_gap']:.1f} GB/s)")

    # ── Theoretical bandwidth: one dashed line per version, NOT in legend ──
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
