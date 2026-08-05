#!/usr/bin/env python3
"""
Version 1 (synchronous, host-staged) measured vs modelled wall time.

WHY THIS PLOT EXISTS
--------------------
plot_p2p.py consumes beta_p2p, but in the measured regime the overlapped model

    T = N_s * [ t_bnd + max(t_int, t_comm) ]

has t_int larger than t_comm by roughly 300-500x, so max() always selects
t_int and beta_p2p never reaches the prediction. That figure demonstrates a
well calibrated gamma and a compute-bound solver; it does not exercise the
interconnect, and the separately measured H2D and D2H bandwidths have no role
in it at all.

Version 1 is where they earn their place. It stages every halo plane through
host memory and overlaps nothing, so both directions enter the prediction
directly and the microbenchmark constants are actually load bearing:

    M       = 8 N^2                                     one halo plane, fp64
    t_plane = (alpha_d2h + M/beta_d2h) + (alpha_h2d + M/beta_h2d)
    planes  = 2 (P - 1)          two planes per internal interface
    t_comm  = planes * t_plane   serialised: one host thread issues them all
    t_comp  = gamma (n-2)^3 / P
    T_pred  = N_s * [ t_comp + t_comm ]

Nothing here is fitted. gamma comes from single-GPU runs (gamma_single_gpu.csv)
and the four transfer constants come from ../../../../Bandwidth_Measurements/.
The figure is therefore a prediction, not a regression, which is a stronger
claim than the P2P panel can make.

TWO ASSUMPTIONS TO CHECK AGAINST THE SOURCE
-------------------------------------------
Both are exposed as flags because they depend on what 2-openmp.c actually does,
and getting them wrong silently changes the curve rather than raising an error.

  --planes N        Default 2(P-1) assumes every plane transfer serialises on
                    the single host thread. If the implementation overlaps the
                    two directions, or a device pair transfers concurrently,
                    the effective count is lower. Read the timestep loop.

  --serial-compute  Default assumes the P kernels run concurrently, so the
                    interior term is divided by P. If Version 1 issues blocking
                    target regions in sequence, they do not overlap and the
                    term is the full gamma (n-2)^3. A quick check: if measured
                    2-omp is not faster than 1-omp, the kernels are serialised.

Usage
-----
    python3 plot_sync.py --gpus 2 --beta-d2h 24.1 --beta-h2d 26.3 --residuals
    python3 plot_sync.py --gpus 4 --beta-d2h 24.1 --beta-h2d 26.3 \
                         --alpha-d2h 1.2e-5 --alpha-h2d 1.1e-5
    python3 plot_sync.py --gpus 4 --beta-d2h 24.1 --beta-h2d 26.3 --breakdown


python3 plot_sync.py --gamma 6.4042e-11 --gpus 2 --beta-d2h <D2H_GBs> --beta-h2d <H2D_GBs> --residuals --breakdown
python3 plot_sync.py --gamma 6.4042e-11 --gpus 4 --beta-d2h <D2H_GBs> --beta-h2d <H2D_GBs> --residuals --breakdown


Outputs: sync_<P>gpu.png and .pdf   (override with --out)
"""
import argparse
import csv
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUMMARY_CSV = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "OpenMP", "summary.csv"))
DEFAULT_GAMMA_CSV = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "OpenMP", "gamma_single_gpu.csv"))

FIGSIZE       = (10, 6.5)
FS_AXIS_LABEL = 24
FS_TICK       = 20
FS_LEGEND     = 20
LINEWIDTH     = 2
MARKERSIZE    = 8

COLOR_THEORY   = "tab:blue"
COLOR_MEASURED = "tab:green"
COLOR_RESIDUAL = "tab:red"


def load_gamma(path, tail_count=3):
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        sys.exit(f"cannot read gamma CSV {path!r}: {exc}")
    values = []
    for row in rows:
        try:
            values.append((int(row["N"]), float(row["gamma_s_per_point"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        sys.exit(f"no usable N/gamma_s_per_point rows in {path!r}")
    values.sort(key=lambda item: item[0])
    tail = values[-min(tail_count, len(values)):]
    gamma = sum(v for _, v in tail) / len(tail)
    print(f"gamma CSV: {path}")
    print(f"gamma = {gamma:.4e} s/point-update "
          f"(mean of N = {', '.join(str(n) for n, _ in tail)})")
    return gamma


def load_variant(path, variant, time_column, nmin):
    try:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            missing = {"variant", "N", "steps", time_column} - set(reader.fieldnames or [])
            if missing:
                sys.exit(f"{path!r} missing columns: {', '.join(sorted(missing))}")
            rows = []
            for row in reader:
                if row.get("variant") != variant:
                    continue
                try:
                    n = int(row["N"])
                    rec = {"n": n, "nsteps": int(row["steps"]),
                           "time": float(row[time_column])}
                except (TypeError, ValueError):
                    continue
                if n >= nmin:
                    rows.append(rec)
    except OSError as exc:
        sys.exit(f"cannot read {path!r}: {exc}")
    if not rows:
        sys.exit(f"no {variant!r} rows at/above N={nmin} in {path!r}")
    rows.sort(key=lambda r: r["n"])
    print(f"summary CSV: {path}")
    print(f"variant: {variant}   measured column: {time_column}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, choices=[2, 4], default=2)
    ap.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    ap.add_argument("--gamma-csv", default=DEFAULT_GAMMA_CSV)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--gamma-tail", type=int, default=3)
    ap.add_argument("--time-column", default="median_sec",
                    choices=["min_sec", "median_sec", "max_sec"])
    ap.add_argument("--beta-d2h", type=float, required=True,
                    help="measured device-to-host bandwidth, GB/s")
    ap.add_argument("--beta-h2d", type=float, required=True,
                    help="measured host-to-device bandwidth, GB/s")
    ap.add_argument("--alpha-d2h", type=float, default=0.0, help="seconds")
    ap.add_argument("--alpha-h2d", type=float, default=0.0, help="seconds")
    ap.add_argument("--planes", type=int, default=None,
                    help="halo planes on the critical path per step "
                         "(default 2*(P-1), fully serialised)")
    ap.add_argument("--serial-compute", action="store_true",
                    help="kernels do not overlap: use gamma(n-2)^3, not /P")
    ap.add_argument("--exe", default=None, help="default: <P>-omp")
    ap.add_argument("--out", default=None)
    ap.add_argument("--nmin", type=int, default=512)
    ap.add_argument("--linear-y", action="store_true")
    ap.add_argument("--residuals", action="store_true")
    ap.add_argument("--breakdown", action="store_true",
                    help="print the compute/communication split per size")
    ap.add_argument("--no-annotate", action="store_true")
    ap.add_argument("--xlabel", default=r"Grid Size ($N^3$)")
    args = ap.parse_args()

    P = args.gpus
    exe = args.exe or f"{P}-omp"
    out = args.out or f"sync_{P}gpu"
    planes = args.planes if args.planes is not None else 2 * (P - 1)

    for name, val in (("--beta-d2h", args.beta_d2h), ("--beta-h2d", args.beta_h2d)):
        if val <= 0:
            sys.exit(f"{name} must be positive")
    if args.gamma_tail <= 0:
        sys.exit("--gamma-tail must be positive")

    b_d2h = args.beta_d2h * 1e9
    b_h2d = args.beta_h2d * 1e9

    g = (args.gamma if args.gamma is not None
         else load_gamma(args.gamma_csv, args.gamma_tail))
    if args.gamma is not None:
        print(f"gamma override: {g:.4e} s/point-update")

    print(f"host path: beta_d2h={args.beta_d2h} GB/s  beta_h2d={args.beta_h2d} GB/s  "
          f"alpha_d2h={args.alpha_d2h:.3e} s  alpha_h2d={args.alpha_h2d:.3e} s")
    print(f"planes on the critical path: {planes}"
          f"{' (default 2(P-1))' if args.planes is None else ' (user override)'}")
    print(f"compute term: gamma(n-2)^3"
          f"{'' if args.serial_compute else ' / %d' % P}")

    rows = load_variant(args.summary_csv, exe, args.time_column, args.nmin)

    N, Tpred, Tmeas, share = [], [], [], []
    for d in rows:
        n, nsteps = d["n"], d["nsteps"]
        M = 8.0 * n * n
        t_plane = ((args.alpha_d2h + M / b_d2h) + (args.alpha_h2d + M / b_h2d))
        t_comm = planes * t_plane
        t_comp = g * (n - 2) ** 3 / (1 if args.serial_compute else P)
        N.append(n)
        Tpred.append(nsteps * (t_comp + t_comm))
        Tmeas.append(d["time"])
        share.append(100.0 * t_comm / (t_comp + t_comm))

    rel   = [(tp - tm) / tm for tp, tm in zip(Tpred, Tmeas)]
    mape  = 100.0 * sum(abs(r) for r in rel) / len(rel)
    bias  = 100.0 * sum(rel) / len(rel)
    worst = 100.0 * max(abs(r) for r in rel)

    print(f"\n[{P} GPUs, Version 1 synchronous] relative deviation "
          f"(T_pred vs T_meas):")
    for n, r in zip(N, rel):
        print(f"    N={n:>5}^3   {100.0 * r:+6.1f}%")
    print(f"[{P} GPUs] MAPE={mape:.1f}%  bias={bias:+.1f}%  "
          f"worst={worst:.1f}%  (n={len(rel)} sizes)")

    if args.breakdown:
        print(f"\ncommunication share of the prediction:")
        for n, s in zip(N, share):
            print(f"    N={n:>5}^3   {s:5.1f}%")
    lo, hi = min(share), max(share)
    print(f"\ncommunication is {lo:.1f}-{hi:.1f}% of T_pred across the sweep.")
    if hi < 5.0:
        print("  NOTE: below ~5% the transfer constants barely move the curve,")
        print("  so this figure does not constrain beta_d2h/beta_h2d either.")
        print("  Report that honestly rather than presenting it as validation.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not available")

    x = list(range(len(N)))
    if args.residuals:
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(FIGSIZE[0], FIGSIZE[1] + 2.2), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    else:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        axr = None

    ax.plot(x, Tpred, "--o", color=COLOR_THEORY, lw=LINEWIDTH, ms=MARKERSIZE,
            label=r"Estimated $T_{total,%d,\mathrm{GPUs}}^{\mathrm{host-staged}}$" % P)
    ax.plot(x, Tmeas, "-s", color=COLOR_MEASURED, lw=LINEWIDTH, ms=MARKERSIZE,
            label=r"Measured $T_{total,%d,\mathrm{GPUs}}^{\mathrm{host-staged}}$" % P)

    if not args.linear_y:
        ax.set_yscale("log")
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

    if not args.no_annotate:
        ax.text(0.98, 0.02,
                rf"MAPE $= {mape:.1f}\%$" "\n" rf"bias $= {bias:+.1f}\%$",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=FS_LEGEND,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    if axr is not None:
        axr.axhline(0.0, color="0.4", lw=1)
        for lvl in (10, 20):
            for s in (+1, -1):
                axr.axhline(s * lvl, color="0.7", lw=1, ls=":")
        axr.plot(x, [100.0 * r for r in rel], "-o", color=COLOR_RESIDUAL,
                 lw=LINEWIDTH, ms=MARKERSIZE)
        ymax = max(20.0, worst) * 1.25
        axr.set_ylim(-ymax, ymax)
        axr.set_ylabel("Rel. err. (%)", fontsize=FS_TICK)
        axr.tick_params(axis="y", which="both", labelsize=FS_TICK - 4)
        axr.grid(True, which="major", axis="x", linestyle="--", alpha=0.4)
        owner = axr
    else:
        owner = ax

    owner.set_xticks(x)
    owner.set_xticklabels([f"${n}^3$" for n in N], fontsize=FS_TICK)
    owner.set_xlabel(args.xlabel, fontsize=FS_AXIS_LABEL)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
        print(f"wrote {out}.{ext}")


if __name__ == "__main__":
    main()
