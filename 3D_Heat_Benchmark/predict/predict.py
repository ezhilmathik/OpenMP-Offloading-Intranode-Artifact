#!/usr/bin/env python3
"""
predict.py -- hold-out validation table for the 3D-heat P2P performance model.

Reads one aggregate.sh summary.csv per machine (kept in this directory),
applies the machine's calibrated gamma and beta_p2p, and prints measured vs
predicted whole-run time (seconds, as in summary.csv) with the signed deviation. No fitting is done here:
gamma and beta are the values used for the calibration figures.

Model (overlapped, P GPUs), identical to plot_p2p.py:
    M       = n^2 * 8                      bytes per halo plane
    t_int   = gamma * (n-2)^3 / P
    t_bnd   = 2 * gamma * (n-2)^2
    faces   = 1 if P == 2 else 2
    t_comm  = faces * (alpha + M / beta)
    T_pred  = N_s * [ t_bnd + max(t_int, t_comm) ]

Deviation = (T_pred - T_meas) / T_meas   (+ => model over-predicts)

Expected files (override with --csv MACHINE=PATH):
    H100/summary.csv   mi250x/summary.csv   max1550/summary.csv

Usage:
    python3 predict.py                 # table for N >= 1408 (hold-out)
    python3 predict.py --latex         # LaTeX rows for the paper
    python3 predict.py --nmin 512      # include calibration sizes too
    python3 predict.py --only h100
"""
import argparse
import csv
import os
import sys

# --------------------------------------------------------------------------
# Calibrated inputs. gamma in s/point-update, beta in GB/s. Edit here only.
# --------------------------------------------------------------------------
MACHINES = {
    "h100":    {"label": "H100",     "gamma": 6.4042e-11, "beta": 253.11,
                "csv": "H100/summary.csv"},
    "mi250x":  {"label": "MI250X",   "gamma": 13.584e-11, "beta": 61.28,
                "csv": "mi250x/summary.csv"},
    "max1550": {"label": "Max 1550", "gamma": 4.6676e-11, "beta": 62.85,
                "csv": "max1550/summary.csv"},
}
VARIANT = "{P}-omp-stream-omp-p2p"


def predict_step(n, P, gamma, beta_gbs, alpha=0.0):
    """Model time per step in seconds."""
    beta = beta_gbs * 1e9
    M = n * n * 8.0
    t_int = gamma * (n - 2) ** 3 / P
    t_bnd = 2.0 * gamma * (n - 2) ** 2
    faces = 1 if P == 2 else 2
    t_comm = faces * (alpha + M / beta)
    return t_bnd + max(t_int, t_comm)


def load_rows(path, nmin, time_col):
    """Return {(P, n): (median_sec, steps, spread_pct, samples)}."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                P = int(row["gpus"])
                n = int(row["N"])
                steps = int(row["steps"])
                t = float(row[time_col])
                spread = float(row.get("rel_spread_pct", "nan"))
                samples = int(row.get("samples", 0))
            except (KeyError, TypeError, ValueError):
                continue
            if row.get("variant") != VARIANT.format(P=P) or n < nmin:
                continue
            out[(P, n)] = (t, steps, spread, samples)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nmin", type=int, default=1408,
                    help="smallest N to include (default 1408 = hold-out only)")
    ap.add_argument("--time-column", default="median_sec",
                    choices=["min_sec", "median_sec", "max_sec"])
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="P2P latency term in seconds (default 0)")
    ap.add_argument("--only", nargs="*", choices=sorted(MACHINES),
                    help="restrict to these machines")
    ap.add_argument("--csv", nargs="*", default=[], metavar="MACHINE=PATH",
                    help="override CSV path, e.g. --csv h100=../foo.csv")
    ap.add_argument("--latex", action="store_true",
                    help="print LaTeX table rows instead of the plain table")
    args = ap.parse_args()

    for item in args.csv:
        key, _, path = item.partition("=")
        if key not in MACHINES or not path:
            sys.exit(f"bad --csv entry {item!r}")
        MACHINES[key]["csv"] = path

    here = os.path.dirname(os.path.abspath(__file__))
    machines = args.only or list(MACHINES)

    results = []   # (label, n, P, meas_s, pred_s, dev_pct, spread, samples)
    for key in machines:
        m = MACHINES[key]
        path = m["csv"] if os.path.isabs(m["csv"]) else os.path.join(here, m["csv"])
        if not os.path.exists(path):
            print(f"# {m['label']}: {path} not found, skipping", file=sys.stderr)
            continue
        rows = load_rows(path, args.nmin, args.time_column)
        if not rows:
            print(f"# {m['label']}: no P2P rows with N >= {args.nmin}", file=sys.stderr)
            continue
        for (P, n), (t, steps, spread, samples) in sorted(rows.items(),
                                                         key=lambda kv: (kv[0][1], kv[0][0])):
            meas = t                                            # whole run, s
            pred = steps * predict_step(n, P, m["gamma"], m["beta"], args.alpha)
            dev = 100.0 * (pred - meas) / meas
            results.append((m["label"], n, P, meas, pred, dev, spread, samples))

    if not results:
        sys.exit("nothing to report")

    if args.latex:
        print("% GPU & N & GPUs & meas (s) & spread (%) & pred (s) & dev (%)")
        for label, n, P, meas, pred, dev, spread, _ in results:
            sign = f"{dev:+.1f}" if abs(dev) >= 0.05 else r"\phantom{+}0.0"
            print(f"{label:<9} & ${n}^3$ & {P} & {meas:7.2f} & {spread:4.1f} & "
                  f"{pred:7.2f} & ${sign}$ \\\\")
    else:
        hdr = f"{'GPU':<9} {'N':>6} {'GPUs':>4} {'meas (s)':>10} " \
              f"{'spread%':>8} {'pred (s)':>10} {'dev%':>7} {'n':>3}"
        print(hdr)
        print("-" * len(hdr))
        for label, n, P, meas, pred, dev, spread, samples in results:
            print(f"{label:<9} {n:>5}^3 {P:>4} {meas:>10.2f} {spread:>8.1f} "
                  f"{pred:>10.2f} {dev:>+7.1f} {samples:>3}")

    # per-machine summary
    print()
    for key in machines:
        label = MACHINES[key]["label"]
        devs = [r[5] for r in results if r[0] == label]
        if not devs:
            continue
        mape = sum(abs(d) for d in devs) / len(devs)
        bias = sum(devs) / len(devs)
        worst = max(abs(d) for d in devs)
        pre = "% " if args.latex else "# "
        print(f"{pre}{label}: MAPE={mape:.1f}%  bias={bias:+.1f}%  "
              f"worst={worst:.1f}%  (n={len(devs)})")


if __name__ == "__main__":
    main()
