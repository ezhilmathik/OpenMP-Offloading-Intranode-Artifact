#!/usr/bin/env python3
"""
Plot theoretical (model) vs measured direct-P2P wall time from summary.csv,
for 2 or 4 GPUs, in the shared panel style.

The machine gamma value is loaded from gamma_single_gpu.csv. Both CSV paths
default to ../OpenMP relative to this script, so the program can be run from
any working directory.

  --gpus 2  -> ./2-omp-stream-omp-p2p ; interior /2 ; 1 halo face
  --gpus 4  -> ./4-omp-stream-omp-p2p ; interior /4 ; 2 halo faces (interior GPU)

Model (overlapped, P GPUs):
    M       = n^2 * 8
    t_int   = gamma * (n-2)^3 / P
    t_bnd   = 2 * gamma * (n-2)^2
    faces   = 1 if P == 2 else 2            (interior GPU is the bottleneck)
    t_comm  = faces * (alpha_p2p + M / beta_p2p)
    T_pred  = N_s * [ t_bnd + max(t_int, t_comm) ]

x-axis: grid size N^3 (the 3D domain is N x N x N points).
y-axis: wall-clock time (s), log scale.

Model-vs-measurement agreement is reported as relative deviation, normalised
by the MEASURED time (the ground truth). Because gamma is calibrated from real
single-GPU runs, this is a predictive model (not a peak/roofline lower bound),
so the deviation can legitimately go both ways -- there is no expectation that
measured < estimated. We therefore report:
    MAPE  = mean(|T_pred - T_meas| / T_meas)      (unsigned accuracy)
    bias  = mean( (T_pred - T_meas) / T_meas )     (signed; + => over-predicts)
    worst = max(|T_pred - T_meas| / T_meas)        (largest single miss)

Usage:
    python3 plot_p2p.py --gamma 6.4042e-11 --gpus 2 --beta-p2p 253.11 --residuals
    python3 plot_p2p.py --gamma 6.4042e-11 --gpus 4 --beta-p2p 253.11 --residuals

Outputs: p2p_<P>gpu.png and .pdf   (override with --out)

Add --residuals to draw a per-size relative-error sub-panel under the main plot.
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

# -- shared panel style (keep identical to the bandwidth-panel scripts) ------
FIGSIZE       = (10, 6.5)
FS_AXIS_LABEL = 24
FS_TICK       = 20
FS_LEGEND     = 20
LINEWIDTH     = 2
MARKERSIZE    = 8
XTICK_ROTATION = 0

COLOR_THEORY   = "tab:blue"
COLOR_MEASURED = "tab:green"
COLOR_RESIDUAL = "tab:red"

RE_N_FILE = re.compile(r"_N(\d+)_")
RE_N_GRID = re.compile(r"Grid size \(n\):\s*(\d+)")
RE_STEPS  = re.compile(r"(\d+)\s+timesteps complete")
RE_TIME   = re.compile(r"Time\s*=\s*([-\d.eE+]+)\s*seconds")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUMMARY_CSV = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "OpenMP", "summary.csv"))
DEFAULT_GAMMA_CSV = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "OpenMP", "gamma_single_gpu.csv"))


def load_gamma(path, tail_count=3):
    """Return the plateau gamma from the largest N rows in the gamma CSV."""
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        sys.exit(f"cannot read gamma CSV {path!r}: {exc}")

    values = []
    for row in rows:
        try:
            n = int(row["N"])
            gamma = float(row["gamma_s_per_point"])
        except (KeyError, TypeError, ValueError):
            continue
        values.append((n, gamma))

    if not values:
        sys.exit(
            f"no usable N/gamma_s_per_point rows found in gamma CSV {path!r}")

    values.sort(key=lambda item: item[0])
    tail = values[-min(tail_count, len(values)):]
    gamma = sum(value for _, value in tail) / len(tail)
    used_n = ", ".join(str(n) for n, _ in tail)
    print(f"gamma CSV: {path}")
    print(f"gamma = {gamma:.4e} s/point-update "
          f"(mean of N = {used_n})")
    return gamma


def load_p2p_rows(path, variant, time_column, nmin):
    """Load measured P2P median times and timestep counts from summary.csv."""
    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = set(reader.fieldnames or [])
            required = {"variant", "N", "steps", time_column}
            missing = required - fieldnames
            if missing:
                sys.exit(
                    f"summary CSV {path!r} is missing columns: "
                    + ", ".join(sorted(missing)))

            rows = []
            for row in reader:
                if row.get("variant") != variant:
                    continue
                try:
                    n = int(row["N"])
                    nsteps = int(row["steps"])
                    measured = float(row[time_column])
                except (TypeError, ValueError):
                    continue
                if n >= nmin:
                    rows.append({
                        "n": n,
                        "nsteps": nsteps,
                        "time": measured,
                    })
    except OSError as exc:
        sys.exit(f"cannot read summary CSV {path!r}: {exc}")

    rows.sort(key=lambda row: row["n"])
    if not rows:
        sys.exit(
            f"no {variant!r} rows found at/above nmin={nmin} in {path!r}")

    print(f"summary CSV: {path}")
    print(f"measured column: {time_column}")
    return rows


def parse_block(text, exe):
    for b in text.split("Running: ./"):
        if b.startswith(exe + " "):
            mt = RE_TIME.search(b)
            if not mt:
                return None
            ms = RE_STEPS.search(b)
            mn = RE_N_GRID.search(b)
            return {"time": float(mt.group(1)),
                    "nsteps": int(ms.group(1)) if ms else None,
                    "n": int(mn.group(1)) if mn else None}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV,
                    help="path to OpenMP summary.csv")
    ap.add_argument("--gamma-csv", default=DEFAULT_GAMMA_CSV,
                    help="path to gamma_single_gpu.csv")
    ap.add_argument("--time-column", default="median_sec",
                    choices=["min_sec", "median_sec", "max_sec"],
                    help="measured P2P time column from summary.csv")
    ap.add_argument("--gamma", type=float, default=None,
                    help="optional gamma override; default loads --gamma-csv")
    ap.add_argument("--gamma-tail", type=int, default=3,
                    help="largest-N gamma rows used for plateau mean")
    ap.add_argument("--gpus", type=int, choices=[2, 4], default=2,
                    help="number of GPUs (selects exe AND model, default 2)")
    ap.add_argument("--beta-p2p", type=float, required=True, help="GB/s")
    ap.add_argument("--alpha-p2p", type=float, default=0.0, help="seconds")
    ap.add_argument("--exe", default=None,
                    help="override CSV variant (default: <P>-omp-stream-omp-p2p)")
    ap.add_argument("--out", default=None,
                    help="output basename (default: p2p_<P>gpu)")
    ap.add_argument("--nmin", type=int, default=512,
                    help="skip grid sizes below this (default 512)")
    ap.add_argument("--linear-y", action="store_true",
                    help="use a linear y-axis instead of log")
    ap.add_argument("--residuals", action="store_true",
                    help="add a per-size relative-error sub-panel below the main plot")
    ap.add_argument("--no-annotate", action="store_true",
                    help="do not draw the MAPE/bias text box on the figure")
    ap.add_argument("--xlabel", default=r"Grid Size ($N^3$)",
                    help=r"x-axis title")
    args = ap.parse_args()

    P = args.gpus
    exe = args.exe or f"{P}-omp-stream-omp-p2p"
    out = args.out or f"h100_p2p_{P}gpu"
    faces = 1 if P == 2 else 2          # interior GPU exchanges 2 faces for P>=3
    beta = args.beta_p2p * 1e9
    if args.gamma_tail <= 0:
        sys.exit("--gamma-tail must be positive")
    if args.beta_p2p <= 0:
        sys.exit("--beta-p2p must be positive")

    g = (args.gamma if args.gamma is not None
         else load_gamma(args.gamma_csv, args.gamma_tail))
    if args.gamma is not None:
        print(f"gamma override: {g:.4e} s/point-update")

    measured_rows = load_p2p_rows(
        args.summary_csv, exe, args.time_column, args.nmin)

    N, Tpred, Tmeas = [], [], []
    for d in measured_rows:
        n = d["n"]
        nsteps = d["nsteps"]

        M = n * n * 8.0
        t_int = g * (n - 2) ** 3 / P
        t_bnd = 2.0 * g * (n - 2) ** 2
        t_comm = faces * (args.alpha_p2p + M / beta)
        Tp = nsteps * (t_bnd + max(t_int, t_comm))
        N.append(n)
        Tpred.append(Tp)
        Tmeas.append(d["time"])

    order = sorted(range(len(N)), key=lambda i: N[i])
    N     = [N[i] for i in order]
    Tpred = [Tpred[i] for i in order]
    Tmeas = [Tmeas[i] for i in order]

    # --- model-vs-measurement agreement (relative to measured time) ---------
    rel   = [(tp - tm) / tm for tp, tm in zip(Tpred, Tmeas)]   # signed fraction
    mape  = 100.0 * sum(abs(r) for r in rel) / len(rel)         # unsigned %
    bias  = 100.0 * sum(rel) / len(rel)                         # signed %
    worst = 100.0 * max(abs(r) for r in rel)                    # largest miss %

    print(f"[{P} GPUs] per-size relative deviation (T_pred vs T_meas):")
    for n, r in zip(N, rel):
        print(f"    N={n:>5}^3   {100.0 * r:+6.1f}%")
    print(f"[{P} GPUs] MAPE={mape:.1f}%  bias={bias:+.1f}%  "
          f"worst={worst:.1f}%  (n={len(rel)} sizes)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not available")

    x = list(range(len(N)))

    # Layout: single panel, or main panel + residual sub-panel sharing the x-axis.
    if args.residuals:
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(FIGSIZE[0], FIGSIZE[1] + 2.2), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        axr = None
        
    ax.plot(x, Tpred, "--o", color=COLOR_THEORY, lw=LINEWIDTH, ms=MARKERSIZE,
            #            label=r"Estimated $T_{total,%d, GPUs}^{\mathrm{P2P,overlap}}$" % P)
            label=r"Estimated $T_{total,%d,\mathrm{GPUs}}^{\mathrm{P2P,overlap}}$" % P)
    ax.plot(x, Tmeas, "-s", color=COLOR_MEASURED, lw=LINEWIDTH, ms=MARKERSIZE,
            label=r"Measured $T_{total,%d, GPUs}^{\mathrm{P2P,overlap}}$" % P)
    
    if not args.linear_y:
        ax.set_yscale("log")
        # Force labeled intermediate log ticks (2,3,4,6 x 10^n) so panels look
        # the same regardless of whether the range straddles a decade line.
        from matplotlib.ticker import (LogLocator, LogFormatterSciNotation,
                                       FuncFormatter)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        ax.yaxis.set_minor_locator(
            LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9)))

        def _minor_label(y, _pos):
            if y <= 0:
                return ""
            e = int(math.floor(math.log10(y) + 1e-9))
            m = int(round(y / 10.0 ** e))
            return (r"$%d\times10^{%d}$" % (m, e)) if m in (2, 3, 4, 6) else ""

        ax.yaxis.set_minor_formatter(FuncFormatter(_minor_label))

    ax.set_ylabel("Wall-clock time (s)", fontsize=FS_AXIS_LABEL)
    ax.tick_params(axis="y", which="both", labelsize=FS_TICK)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(fontsize=FS_LEGEND, loc="upper left")

    # On-figure accuracy annotation (MAPE + signed bias).
    if not args.no_annotate:
        ax.text(0.98, 0.02,
                rf"MAPE $= {mape:.1f}\%$" "\n" rf"bias $= {bias:+.1f}\%$",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS_LEGEND,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    # --- residual sub-panel: per-size relative error with +/-10/20% guides --
    if axr is not None:
        rel_pct = [100.0 * r for r in rel]
        axr.axhline(0.0, color="0.4", lw=1)
        for lvl in (10, 20):
            for s in (+1, -1):
                axr.axhline(s * lvl, color="0.7", lw=1, ls=":")
        axr.plot(x, rel_pct, "-o", color=COLOR_RESIDUAL,
                 lw=LINEWIDTH, ms=MARKERSIZE)

        # symmetric y-limits with a little headroom beyond the data / 20% guide
        ymax = max(20.0, worst) * 1.25
        axr.set_ylim(-ymax, ymax)
        axr.set_ylabel("Rel. err. (%)", fontsize=FS_TICK)
        axr.tick_params(axis="y", which="both", labelsize=FS_TICK - 4)
        axr.grid(True, which="major", axis="x", linestyle="--", alpha=0.4)
        xaxis_owner = axr
    else:
        xaxis_owner = ax

    xaxis_owner.set_xticks(x)
    xaxis_owner.set_xticklabels([f"${n}^3$" for n in N],
                                rotation=XTICK_ROTATION, fontsize=FS_TICK)
    xaxis_owner.set_xlabel(args.xlabel, fontsize=FS_AXIS_LABEL)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
        print(f"wrote {out}.{ext}")


if __name__ == "__main__":
    main()
