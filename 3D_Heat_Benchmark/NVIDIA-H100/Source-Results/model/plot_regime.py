#!/usr/bin/env python3
"""
Why the solver is compute bound, shown rather than asserted.

When t_int exceeds t_comm by two orders of magnitude, a model-vs-measurement
panel cannot say anything about the interconnect: the max() selects t_int and
beta cancels. That is a result, but it needs a figure that displays the margin
instead of a fit that conceals it.

  --mode regime     The model's two terms against N, log-log. t_int ~ N^3/P and
                    t_comm ~ N^2, so they diverge; extrapolating back gives the
                    crossover N* where the solver would become transfer bound.
                    The measured bandwidths set the height of the t_comm lines,
                    which is what makes "compute bound" a quantitative claim.

  --mode decompose  Stacked predicted time per grid size: interior, boundary
                    planes, exposed communication. Version 1 exposes all of its
                    transfers; Versions 2-4 hide them, which is why their
                    predictions collapse onto each other.

Both modes are predictions from gamma (single-GPU calibration) and the measured
transfer constants. Nothing is fitted.

Usage
-----
    python3 plot_regime.py --mode regime --gpus 4 --gamma 5.507e-11 \
        --beta-p2p 253.11 --beta-d2h 24 --beta-h2d 26

    python3 plot_regime.py --mode decompose --gpus 4 --gamma 5.507e-11 \
        --beta-p2p 253.11 --beta-d2h 24 --beta-h2d 26
"""
import argparse
import math
import sys

FIGSIZE       = (10, 6.5)
FS_AXIS_LABEL = 22
FS_TICK       = 18
FS_LEGEND     = 16
LINEWIDTH     = 2
MARKERSIZE    = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["regime", "decompose"], default="regime")
    ap.add_argument("--gpus", type=int, choices=[2, 4], default=4)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--beta-p2p", type=float, required=True, help="GB/s")
    ap.add_argument("--beta-d2h", type=float, required=True, help="GB/s")
    ap.add_argument("--beta-h2d", type=float, required=True, help="GB/s")
    ap.add_argument("--alpha-p2p", type=float, default=0.0)
    ap.add_argument("--alpha-d2h", type=float, default=0.0)
    ap.add_argument("--alpha-h2d", type=float, default=0.0)
    ap.add_argument("--sizes", default="512,640,768,896,1024,1152,1280")
    ap.add_argument("--planes", type=int, default=None,
                    help="critical-path planes for the sync variant "
                         "(default 2(P-1))")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    P = args.gpus
    g = args.gamma
    faces = 1 if P == 2 else 2
    planes = args.planes if args.planes is not None else 2 * (P - 1)
    b_p2p = args.beta_p2p * 1e9
    b_d2h = args.beta_d2h * 1e9
    b_h2d = args.beta_h2d * 1e9
    sizes = [int(s) for s in args.sizes.split(",")]

    def t_int(n):   return g * (n - 2) ** 3 / P
    def t_bnd(n):   return 2.0 * g * (n - 2) ** 2
    def t_p2p(n):
        return faces * (args.alpha_p2p + 8.0 * n * n / b_p2p)
    def t_host1(n):
        M = 8.0 * n * n
        return (args.alpha_d2h + M / b_d2h) + (args.alpha_h2d + M / b_h2d)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not available")

    # ------------------------------------------------------------ regime ----
    if args.mode == "regime":
        # crossover: gamma n^3 / P == c * 8 n^2 / beta  ->  n* = 8 c P / (gamma beta)
        def crossover(beta, count):
            return 8.0 * count * P / (g * beta)

        n_star_p2p = crossover(b_p2p, faces)
        n_star_async = crossover((b_d2h * b_h2d) / (b_d2h + b_h2d), faces)
        n_star_sync = crossover((b_d2h * b_h2d) / (b_d2h + b_h2d), planes)

        print(f"crossover N* (t_int == t_comm), {P} GPUs:")
        print(f"  direct P2P            N* = {n_star_p2p:8.1f}")
        print(f"  host-staged, async    N* = {n_star_async:8.1f}")
        print(f"  host-staged, sync     N* = {n_star_sync:8.1f}")
        print(f"  smallest measured N   = {min(sizes)}")
        print(f"\nmargin at the smallest measured size:")
        n0 = min(sizes)
        print(f"  t_int/t_comm (P2P)        = {t_int(n0)/t_p2p(n0):8.0f}x")
        print(f"  t_int/t_comm (host,async) = {t_int(n0)/(faces*t_host1(n0)):8.0f}x")
        print(f"  t_int/t_comm (host,sync)  = {t_int(n0)/(planes*t_host1(n0)):8.1f}x")

        lo = max(4, int(min(n_star_p2p, n_star_async) / 3))
        ns = [lo * (max(sizes) * 3 / lo) ** (i / 199.0) for i in range(200)]

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(ns, [t_int(n) for n in ns], "-", color="tab:green",
                lw=LINEWIDTH + 1,
                label=r"$t_{\mathrm{int}} = \gamma (N-2)^3 / P$")
        ax.plot(ns, [planes * t_host1(n) for n in ns], "--", color="tab:red",
                lw=LINEWIDTH, label=r"$t_{\mathrm{comm}}$ host-staged, exposed")
        ax.plot(ns, [faces * t_host1(n) for n in ns], "-.", color="tab:orange",
                lw=LINEWIDTH, label=r"$t_{\mathrm{comm}}$ host-staged, overlapped")
        ax.plot(ns, [t_p2p(n) for n in ns], ":", color="tab:blue",
                lw=LINEWIDTH + 1, label=r"$t_{\mathrm{comm}}$ direct P2P")

        ax.axvspan(min(sizes), max(sizes), color="0.85", alpha=0.6, zorder=0)
        ax.text(math.sqrt(min(sizes) * max(sizes)), ax.get_ylim()[0],
                "measured range", ha="center", va="bottom",
                fontsize=FS_LEGEND, color="0.3")

        for n_star, c in ((n_star_p2p, "tab:blue"), (n_star_async, "tab:orange")):
            if n_star > lo:
                ax.axvline(n_star, color=c, lw=1, ls="-", alpha=0.5)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Grid size $N$", fontsize=FS_AXIS_LABEL)
        ax.set_ylabel("Time per timestep (s)", fontsize=FS_AXIS_LABEL)
        ax.tick_params(axis="both", which="both", labelsize=FS_TICK)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=FS_LEGEND, loc="upper left")
        out = args.out or f"regime_{P}gpu"

    # --------------------------------------------------------- decompose ----
    else:
        labels = [f"${n}^3$" for n in sizes]
        interior = [t_int(n) for n in sizes]
        boundary = [t_bnd(n) for n in sizes]
        exposed_sync = [planes * t_host1(n) for n in sizes]

        print(f"predicted per-step decomposition, {P} GPUs "
              f"(fractions of the synchronous total):")
        print(f"{'N':>6}  {'interior':>9}  {'boundary':>9}  {'exposed comm':>13}")
        for n, ti, tb, tc in zip(sizes, interior, boundary, exposed_sync):
            tot = ti + tc
            print(f"{n:>6}  {100*ti/tot:8.1f}%  {100*tb/tot:8.1f}%  "
                  f"{100*tc/tot:12.1f}%")

        x = list(range(len(sizes)))
        fig, ax = plt.subplots(figsize=FIGSIZE)
        w = 0.36
        xs_s = [i - w / 2 - 0.02 for i in x]
        xs_p = [i + w / 2 + 0.02 for i in x]

        ax.bar(xs_s, interior, w, color="tab:green", label="interior compute")
        ax.bar(xs_s, exposed_sync, w, bottom=interior, color="tab:red",
               label="exposed communication (sync)")
        ax.bar(xs_p, [i for i in interior], w, color="tab:green")
        ax.bar(xs_p, boundary, w, bottom=interior, color="tab:blue",
               label="boundary planes (overlapped variants)")

        for i in x:
            ax.text(xs_s[i], 0, "V1", ha="center", va="top",
                    fontsize=FS_LEGEND - 2, color="0.3")
            ax.text(xs_p[i], 0, "P2P", ha="center", va="top",
                    fontsize=FS_LEGEND - 2, color="0.3")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FS_TICK)
        ax.set_xlabel(r"Grid Size ($N^3$)", fontsize=FS_AXIS_LABEL)
        ax.set_ylabel("Predicted time per step (s)", fontsize=FS_AXIS_LABEL)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=FS_LEGEND, loc="upper left")
        out = args.out or f"decompose_{P}gpu"

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
        print(f"wrote {out}.{ext}")


if __name__ == "__main__":
    main()
