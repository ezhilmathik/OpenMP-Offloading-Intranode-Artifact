#!/usr/bin/env python3
"""
plot_sync_htod.py — Bandwidth vs transferred size for versions v1, v2
                    (Sync H-to-D transfer, H100)
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ── Shared plot style (unchanged) ───────────────────────────────────────────
FIGSIZE          = (10, 6.5)
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

DTYPE_BYTES      = 8
UNIT_BASE        = 1000
XAXIS_UNIT       = "auto"

CSV_GLOB         = "results_*/timings.csv"
OUTPUT_DIR       = "plots_final"
OUTPUT_NAME      = "h100_sync_htod.pdf"
VERSIONS_TO_PLOT = ["v1", "v2"]

THEORETICAL_BW = {
    "v1": (64, "1 GPU"),
    "v2": (128, "2 GPUs"),
}
THEORETICAL_LABEL_SIDE = "right"

VERSION_COLOR = {
    "v1": "blue",
    "v2": "green",
}
THEORETICAL_PER_VERSION_COLOR = True

# ── CUDA-vs-OpenMP gap band (NEW) ────────────────────────────────────────────
# GAP_REL_DENOMINATOR: "midpoint" -> gap/((CUDA+OMP)/2)  (neither privileged)
#                      "omp"      -> gap/OMP   ("CUDA is X% faster than OMP")
#                      "cuda"     -> gap/CUDA
SHOW_GAP_BAND        = True
GAP_ALPHA            = 0.15
GAP_REL_DENOMINATOR  = "midpoint"
SHOW_GAP_TEXTBOX     = True
####GAP_TEXTBOX_LOC      = (0.02, 0.02)   # axis fraction; bottom-left
GAP_TEXTBOX_LOC      = (0.02, 0.18) 

SERIES_ORDER = [
    ("cuda",    "v1"),
    ("omp_off", "v1"),
    ("cuda",    "v2"),
    ("omp_off", "v2"),
]

SERIES = {
    ("cuda",    "v1"): ("CUDA (1 GPU)",          "o"),
    ("omp_off", "v1"): ("OpenMP Off. (1 GPU)",   "s"),
    ("cuda",    "v2"): ("CUDA (2 GPUs)",         "^"),
    ("omp_off", "v2"): ("OpenMP Off. (2 GPUs)",  "v"),
}


# ── Helpers (unchanged) ──────────────────────────────────────────────────────
def n_to_bytes(n):
    return n * DTYPE_BYTES


def unit_suffix(unit):
    return unit if UNIT_BASE == 1000 else unit.replace("B", "iB")


def pick_unit(max_bytes):
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
    value = n_to_bytes(n) / unit_divisor(unit)
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def parse_row(fields):
    if len(fields) == 6:
        fw, ver, n_str, _elapsed, bw_str, status = fields
    elif len(fields) == 7:
        fw, ver, n_str, _empty, _elapsed, bw_str, status = fields
    else:
        return None

    fw = fw.strip(); ver = ver.strip()
    status = status.strip(); bw_str = bw_str.strip()

    if fw == "framework":
        return None
    if ver not in VERSIONS_TO_PLOT:
        return None
    if status != "ok":
        return None
    if not bw_str or bw_str == "N/A":
        return None

    try:
        N = int(n_str.strip()); bw = float(bw_str)
    except ValueError:
        return None

    return fw, ver, N, bw


def load_csv(path):
    data = defaultdict(list)
    with open(path, newline="") as fh:
        for fields in csv.reader(fh):
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


# ── NEW: CUDA-vs-OMP gap statistics ──────────────────────────────────────────
def gap_stats(cuda_nd, omp_nd):
    """Gap between CUDA and OMP at transfer sizes measured by BOTH.
    Returns None if they share no sizes."""
    common_N = sorted(set(cuda_nd) & set(omp_nd))
    if not common_N:
        return None

    cuda_bws = np.array([cuda_nd[n] for n in common_N])
    omp_bws  = np.array([omp_nd[n]  for n in common_N])
    abs_gap  = np.abs(cuda_bws - omp_bws)          # GB/s

    if GAP_REL_DENOMINATOR == "omp":
        ref = omp_bws
    elif GAP_REL_DENOMINATOR == "cuda":
        ref = cuda_bws
    else:                                          # "midpoint"
        ref = 0.5 * (cuda_bws + omp_bws)
    rel_gap = 100.0 * abs_gap / ref                # %

    return {
        "common_N":     common_N,
        "cuda_bws":     cuda_bws,
        "omp_bws":      omp_bws,
        "mean_abs_gap": float(np.mean(abs_gap)),
        "mean_rel_gap": float(np.mean(rel_gap)),
        "max_abs_gap":  float(np.max(abs_gap)),
    }


def active_theoretical_levels():
    if not THEORETICAL_BW:
        return []
    levels = [(bw, what, ver) for ver, (bw, what) in THEORETICAL_BW.items()
              if ver in VERSIONS_TO_PLOT]
    return sorted(levels, key=lambda item: item[0])


# ── Main ─────────────────────────────────────────────────────────────────────
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
        n for (fw, ver), nd in data.items()
        if ver in VERSIONS_TO_PLOT for n in nd
    })
    if not all_N:
        print("No valid bandwidth data found — nothing to plot.")
        return

    x_pos = list(range(len(all_N)))
    n_to_x = {n: i for i, n in enumerate(all_N)}

    unit = pick_unit(n_to_bytes(all_N[-1]))
    print(f"X-axis unit: {unit_suffix(unit)} "
          f"({size_label(all_N[0], unit)} .. {size_label(all_N[-1], unit)})")

    fig, ax1 = plt.subplots(figsize=FIGSIZE)
#    ax1.set_title("NVIDIA H100 - HtoD:data transfer", fontsize=16, fontweight="bold", pad=12)
    ax1.set_title("NVIDIA H100 - HtoD: synchronous", fontsize=16, fontweight="bold", pad=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [size_label(n, unit) for n in all_N],
        rotation=XTICK_ROTATION, fontsize=FS_TICK
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
        

   # ── CUDA vs OMP gap: quantify only, no shading ──────────────────────
    gap_lines = ["Mean absolute HtoD gap"]
    for ver in VERSIONS_TO_PLOT:
        st = gap_stats(data.get(("cuda", ver), {}),
                       data.get(("omp_off", ver), {}))
        if st is None:
            continue
        label = SERIES[("omp_off", ver)][0].split("(")[1].rstrip(")")  # "1 GPU" / "2 GPUs"
        gap_lines.append(f"CUDA vs. OpenMP Off. ({label}):  "
                         f"{st['mean_abs_gap']:.1f} GB/s")
        print(f"{ver}: mean |CUDA-OMP| = {st['mean_abs_gap']:.1f} GB/s "
              f"(max = {st['max_abs_gap']:.1f} GB/s)")
                 
    # ── Theoretical bandwidth (unchanged) ──
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
            ax2.text(label_x, bw + 0.01 * ymax,
                     f"Theoretical BW, {what} ({bw} GB/s)",
                     color=color, ha=label_ha, va="bottom",
                     fontsize=FS_THEORETICAL)

    # ── NEW: mean-gap summary box ──
    if SHOW_GAP_TEXTBOX and gap_lines:
        ax2.text(GAP_TEXTBOX_LOC[0], GAP_TEXTBOX_LOC[1], "\n".join(gap_lines),
                 transform=ax2.transAxes, fontsize=FS_LEGEND,
                 va="bottom", ha="left",
                 bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))

    ax2.set_ylabel(YLABEL, fontsize=FS_AXIS_LABEL)
    ax2.tick_params(axis="y", labelsize=FS_TICK)
    ax2.set_ylim(0, ymax)

    fig.legend(handles=series_handles, loc="upper left",
               bbox_to_anchor=LEGEND_ANCHOR, fontsize=FS_LEGEND)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
