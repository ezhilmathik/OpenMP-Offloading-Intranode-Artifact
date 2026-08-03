#!/usr/bin/env python3
"""
plot_approach_compare_original_colors_thicker_spaced.py -- how do the three APPROACHES compare, and does the
ranking hold on more than one machine?

The question is not "which GPU is faster" and not "how well does it scale".
It is: approach 02 beat approach 01 on the H100 -- is that a property of the
code, or of the H100?

So every number here is a RATIO taken within a single
(machine, variant, N) cell:

        gain = t(reference approach) / t(this approach)

    gain > 1  ->  this approach is faster than the reference here
    gain = 1  ->  identical
    gain < 1  ->  slower

Hardware cancels in the ratio, so a gain of 1.15 means the same thing on an
H100 as on an MI250X. If an approach's gain is ~equal on all machines, the
improvement is portable. If it is 1.4 on one and 0.9 on the other, the
improvement is an artifact of one vendor's runtime and should not be reported
as a general result.

Machines currently wired up: NVIDIA H100, Intel PVC Max 1550, AMD MI250X --
each with approaches 01 / 02 / 03.

Outputs
-------
  N####_gain_<model>.png      per N: x=variant, bars=machine x approach
  portability_<model>.png     scatter: gain on machine A vs gain on machine B
  heatmap_<model>.png         variant x N grid of gain, one panel per machine/approach
  efficiency_<model>.png      SANITY: within-approach parallel efficiency. Read
                              this one FIRST. A cell pinned at 1/gpus means the
                              extra GPUs did nothing, so any gain measured
                              against it is a bug fix, not an optimisation.
  decision_<model>.png        geo-mean gain vs N, one panel per machine
  gains_<model>.csv           the full table

Usage
-----
  cd /path/to/HHT
  python3 gain.py --list              # check current-directory discovery
  python3 gain.py
  python3 gain.py --ref gmean
  python3 gain.py --variants 2-omp 4-omp 4-omp-stream-omp-p2p
  python3 gain.py --format pdf
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd

# =========================== CONFIG: EDIT THIS ==============================
MACHINES = {
    # "patterns" are globs relative to --root. The approach label is taken from
    # the trailing 2 digits of the folder name ("Intel-1550-02" -> "02"); a
    # folder with no trailing digits is treated as approach "01".
    # Run with --list to see exactly what is found on disk.
    "H100": dict(color="tab:blue", patterns=["NVIDIA-H100-*"]),
    "PVC 1550": dict(color="tab:orange", patterns=["Intel-1550-*"]),
    "MI250X": dict(color="tab:green", patterns=["AMD-MI250X-*"]),
}

# Folders matching any of these are ignored even if a pattern above matches.
# NOTE: matching is done on the LOWERCASED folder name, so patterns here must
# be written in lower case.
#
#   *comparison*  -- e.g. "NVIDIA-H100-Heat-comparison-01-02-03" would otherwise
#                    have its trailing "-03" mistaken for an approach label.
#   *mi250x-heat* -- "AMD-MI250X-Heat" has no 2-digit suffix, so it would be
#                    labelled approach "01" and COLLIDE with "AMD-MI250X-01".
#                    If that older folder is in fact the run you want as 01,
#                    remove this line and rename AMD-MI250X-01 instead.
EXCLUDE = ["*comparison*", "*backup*", "*-old*", "*plots*", "*test*",
           "*mi250x-heat*"]

# Output formats. Override with --format.
FORMATS = ["png"]

# --- MDPI page geometry -----------------------------------------------------
# A two-column-spanning MDPI figure is 170 mm wide. Drawing the figure at
# exactly that size means the typesetter places it 1:1 and nothing is scaled
# down, so the point sizes set in plot_per_n() are the point sizes that reach
# the reader. Previously the per-N figure was ~36 in wide and had to be shrunk
# by ~5.4x to fit the page, which is what made every label unreadable.
MDPI_WIDTH_IN = 170.0 / 25.4          # 6.69 in
MDPI_HEIGHT_IN = 4.3                  # ~109 mm; keeps a sane 1.55:1 aspect
# Usable text height of an MDPI page is ~247 mm. 9.0 in (229 mm) leaves room
# for the caption underneath, so the whole grid lands on one page.
MDPI_PAGE_HEIGHT_IN = 9.75

# Short x-axis names for the one-page grid, where the full variant names do not
# fit. Leave empty to auto-generate "2 Ver. 1", "2 Ver. 2", ... numbering the
# variants in plotting order within each GPU count. Fill it in to control the
# numbering by hand, e.g.
#   VARIANT_LABEL = {"2-omp": "2 Ver. 1", "2-omp-stream": "2 Ver. 2", ...}
# Whatever you put here is what the paper caption has to spell out.
VARIANT_LABEL = {}

# Expansion of each short label, printed as a key beside the legend on the
# one-page grid. Leave empty to auto-generate "OpenMP Off. 2-GPU Version 1"
# from the GPU count and the numbering. Fill it in to say what each version
# actually does, e.g.
#   VARIANT_DESC = {"2-omp": "OpenMP Off. 2-GPU, blocking copies", ...}
VARIANT_DESC = {}

# How each machine is named in the legends. Keys are the MACHINES keys above.
MACHINE_LEGEND = {"H100": "H100", "PVC 1550": "Intel-1550",
                  "MI250X": "AMD-MI250X"}

# Display names only: the internal approach identifiers remain 01/02/03.
APPROACH_LEGEND = {"01": "HHT-0", "02": "HHT-GPU", "03": "HHT-8"}

# Keep every bar in its machine colour, with distinct hatch patterns:
# HHT-0 is solid, HHT-GPU is dotted, and HHT-8 has light diagonal strokes.
APPROACH_HATCH = {"01": "", "02": "..", "03": "//"}

# Make individual bars easier to read while preserving a narrow white gap
# between adjacent machine/approach series inside each variant group.
BAR_WIDTH_FRACTION = 0.86

# Distance between the centres of neighbouring variant groups. 1.0 reproduces
# the original spacing; 1.22 adds a modest inter-group gap without changing
# the figure dimensions or panel layout.
GROUP_SPACING = 1.22
# Used by the decision view, where the panel is the machine and colour is the approach.
APPROACH_COLORS = {"01": "tab:blue", "02": "tab:orange", "03": "tab:green"}
# ============================================================================

def save(fig, outdir, stem, **kw):
    for ext in FORMATS:
        fig.savefig(outdir / f"{stem}.{ext}", **kw)


#plt.rcParams.update({
#    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10,
#    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
#    "axes.spines.top": False, "axes.spines.right": False,
#    "legend.frameon": False,
#})


plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.14, "grid.linewidth": 0.55,
    "grid.linestyle": ":", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    # Fine hatch marks remain readable at MDPI print size.
    "hatch.linewidth": 0.35,
})



# ------------------------------------------------------------------ load ----
def discover(root: Path, model: str):
    """Return [(machine, approach, csv_path, exists)] for everything matched."""
    import fnmatch
    found, seen = [], set()
    for machine, cfg in MACHINES.items():
        for pat in cfg.get("patterns", []):
            for d in sorted(root.glob(pat)):
                if not d.is_dir() or d in seen:
                    continue
                if any(fnmatch.fnmatch(d.name.lower(), e) for e in EXCLUDE):
                    continue
                seen.add(d)
                # require a hyphen-delimited 2-digit suffix; no suffix -> "01"
                m = re.search(r"-(\d{2})$", d.name)
                approach = m.group(1) if m else "01"
                csv = d / model / "summary.csv"
                if not csv.is_file() and (d / "summary.csv").is_file():
                    csv = d / "summary.csv"
                found.append((machine, approach, csv, csv.is_file()))

    # a real (machine, approach) collision would silently drop data
    keys = {}
    for machine, approach, csv, exists in found:
        if not exists:
            continue
        keys.setdefault((machine, approach), []).append(csv)
    for (machine, approach), paths in keys.items():
        if len(paths) > 1:
            print(f"  [COLLISION] {machine} approach {approach} matched "
                  f"{len(paths)} folders:", file=sys.stderr)
            for q in paths:
                print(f"              {q}", file=sys.stderr)
            print("              Add one to EXCLUDE or rename it.", file=sys.stderr)
    return found


def load(root: Path, model: str) -> pd.DataFrame:
    found = discover(root, model)
    if not found:
        sys.exit(f"No folders under {root} matched any pattern in MACHINES.\n"
                 f"Run:  ls {root}   and update the patterns.")

    frames = []
    for machine, approach, csv, exists in found:
        if not exists:
            print(f"  [MISSING] {machine:9s} {approach}  {csv}\n"
                  f"            folder exists but no summary.csv -- run ")
#                  f"./aggregate.sh in it", file=sys.stderr)
            continue
        d = pd.read_csv(csv)
        d["machine"] = machine
        d["approach"] = approach
        frames.append(d)
        print(f"  [ok]      {machine:9s} {approach}  {csv}")
    if not frames:
        sys.exit("Matched folders, but none contained a summary.csv.")
    return pd.concat(frames, ignore_index=True)


def report_coverage(df):
    """A ratio needs BOTH cells. Print the (variant, N) cells that are not
    measured on every machine x approach, because those silently become gaps
    in every figure -- e.g. 4-omp-stream-omp-p2p is missing N=512 and N=1024
    in at least one of the MI250X runs."""
    series = sorted(set(zip(df["machine"], df["approach"])))
    if len(series) < 2:
        return
    cells = df.groupby(["variant", "N"]).apply(
        lambda g: len(set(zip(g["machine"], g["approach"]))), include_groups=False)
    holes = cells[cells < len(series)]
    if holes.empty:
        print("  coverage: complete -- every variant x N present on all "
              f"{len(series)} machine x approach series")
        return
    print(f"  [coverage] {len(holes)} of {len(cells)} (variant, N) cells are not "
          f"present on all {len(series)} series:")
    for (v, N), k in holes.items():
        have = sorted(set(zip(df[(df.variant == v) & (df.N == N)]["machine"],
                              df[(df.variant == v) & (df.N == N)]["approach"])))
        missing = [f"{m}/{a}" for m, a in series if (m, a) not in have]
        n_text = str(int(N)) if float(N).is_integer() else f"{float(N):g}"
        print(f"             {v:<24s} N={n_text:<5s} missing: {', '.join(missing)}")


def add_efficiency(df):
    """Parallel efficiency WITHIN one (machine, approach, N): how much did the
    extra GPUs actually buy?

        speedup    = t(1 GPU) / t(this variant)
        efficiency = speedup / gpus          1.00 = perfect, 1/gpus = nothing

    This is a within-approach quantity, so unlike `gain` it cannot be distorted
    by a broken run in some OTHER approach. It is the check that tells you
    whether a cell is a legitimate baseline at all.

    An efficiency pinned at exactly 1/gpus (0.50 on 2 GPUs, 0.25 on 4) means the
    run took the same wall time as the single-GPU run: the additional devices
    contributed nothing, and any ratio taken against that cell is measuring the
    defect, not an optimisation.
    """
    df = df.copy()
    if "t_per_step" not in df.columns:
        df["t_per_step"] = df["median_sec"] / df["steps"]
    key = ["machine", "approach", "N"]
    one = df[df["gpus"] == 1]
    if one.empty:
        print("  [WARN] no 1-GPU variant found; cannot compute parallel "
              "efficiency", file=sys.stderr)
        df["efficiency"] = np.nan
        return df
    base = one.groupby(key)["t_per_step"].min().rename("t_1gpu")
    df = df.join(base, on=key)
    df["speedup"] = df["t_1gpu"] / df["t_per_step"]
    df["efficiency"] = df["speedup"] / df["gpus"]
    return df


def mark_baseline(df, ref, thresh):
    """Attach the efficiency of the cell the gain is measured AGAINST."""
    key = ["machine", "variant", "N"]
    if ref in ("best", "gmean"):
        # the reference is a blend of all approaches, so any broken approach
        # in the cell contaminates it
        b = df.groupby(key)["efficiency"].min().rename("baseline_eff")
    else:
        b = (df[df["approach"] == ref].set_index(key)["efficiency"]
               .rename("baseline_eff"))
    df = df.join(b, on=key)
    df["baseline_ok"] = ~(df["baseline_eff"] < thresh)
    return df


def report_efficiency(df, thresh):
    """Print the efficiency table and shout about cells that did not scale."""
    print(f"\n{'='*72}\nSANITY: parallel efficiency within each approach "
          f"(1.00 = perfect)\n{'='*72}")
    for (m, a), g in df.groupby(["machine", "approach"]):
        piv = g.pivot_table(index="variant", columns="N", values="efficiency")
        print(f"\n  {m} · approach {a}")
        print("    " + piv.round(2).to_string().replace("\n", "\n    "))

    bad = df[(df["gpus"] > 1) & (df["efficiency"] < thresh)]
    if bad.empty:
        print(f"\n  All multi-GPU cells scale above {thresh:.2f}. "
              f"Cross-approach ratios are meaningful.")
        return
    print(f"\n  [!] {len(bad)} multi-GPU cells have efficiency < {thresh:.2f}.")
    flat = bad[np.isclose(bad["efficiency"], 1.0 / bad["gpus"], atol=0.04)]
    if not flat.empty:
        print(f"  [!] {len(flat)} of them sit at exactly 1/gpus, i.e. the extra "
              f"GPUs did NOTHING.")
        print(f"      Those runs are not a valid baseline. A 'gain' measured "
              f"against them is\n      a bug fix in the denominator, not a "
              f"speedup in the numerator.")
    for (m, a), g in bad.groupby(["machine", "approach"]):
        vs = sorted(set(g["variant"]))
        print(f"      {m:<10} approach {a}: {', '.join(vs)}")
    print(f"\n  Re-run with --min-baseline-efficiency {thresh} to drop cells "
          f"whose reference is broken.")


def plot_efficiency(df, variants, outdir, model, thresh):
    """Same panel layout as the gain heatmap, but showing the within-approach
    efficiency, so a broken baseline is visible rather than inferred."""
    pairs = series_order(df)
    if not pairs:
        return
    Ns = sorted(df["N"].unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.0 * len(pairs) + 1.6,
                                                     0.42 * len(variants) + 2.8),
                             squeeze=False, sharey=True)
    im = None
    for ax, (m, a) in zip(axes[0], pairs):
        g = np.full((len(variants), len(Ns)), np.nan)
        for i, v in enumerate(variants):
            for j, N in enumerate(Ns):
                r = df[(df.machine == m) & (df.approach == a) &
                       (df.variant == v) & (df.N == N)]
                if not r.empty:
                    g[i, j] = r.iloc[0]["efficiency"]
        im = ax.imshow(g, cmap="RdYlGn", vmin=0.0, vmax=1.1, aspect="auto")
        for i in range(len(variants)):
            for j in range(len(Ns)):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f"{g[i,j]:.2f}", ha="center", va="center",
                            fontsize=6.5, color="#111")
        ax.set_xticks(range(len(Ns)))
        ax.set_xticklabels(Ns, rotation=45, fontsize=7)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants, fontsize=7)
        ax.set_title(f"{m} · {a}", fontsize=10)
        ax.set_xlabel("N")
        ax.grid(False)
    if im is not None:
        cb = fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)
        cb.set_label("parallel efficiency (speedup / gpus)")
    fig.suptitle(f"{model} — did the extra GPUs do anything? "
                 f"(red ≈ no; 1/gpus means the run ignored them)", fontsize=11)
    save(fig, outdir, f"efficiency_{model}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote efficiency_{model}.png")


def add_gain(df, ref):
    """gain = t_ref / t, within each (machine, variant, N).

    ref may be an approach label ("01"), or:
      "best"   -> ratio against the fastest approach in that cell
      "gmean"  -> ratio against the geometric mean of all approaches in that cell

    With ref="01" the reference approach is plotted too, as a flat 1.00 control.
    With "best"/"gmean" every approach including 01 gets a distinct value, which
    is usually the more readable way to show all three at once.

    Times are normalised per timestep first, so a 500-vs-501 step difference
    between approaches cannot leak into the ratio.
    """
    df = df.copy()
    key = ["machine", "variant", "N"]
    df["t_per_step"] = df["median_sec"] / df["steps"]

    if ref == "best":
        base = df.groupby(key)["t_per_step"].min().rename("t_ref")
    elif ref == "gmean":
        base = (df.groupby(key)["t_per_step"]
                  .apply(lambda s: float(np.exp(np.log(s).mean()))).rename("t_ref"))
    else:
        if ref not in set(df["approach"]):
            sys.exit(f"Reference '{ref}' not present. Have: "
                     f"{sorted(set(df['approach']))} (or use best / gmean)")
        base = (df[df["approach"] == ref].set_index(key)["t_per_step"]
                  .rename("t_ref"))

    df = df.join(base, on=key)
    df["gain"] = df["t_ref"] / df["t_per_step"]

    if ref not in ("best", "gmean"):
        print(f"  [NOTE] reference is approach {ref}, so approach {ref} is drawn "
              f"as exactly 1.00 in\n         every group. Its bars carry no "
              f"information, and a group where {ref}\n         scaled badly looks "
              f"identical to one where it scaled perfectly.\n"
              f"         Use --ref gmean to let every approach take a real value.",
              file=sys.stderr)

    # A machine carrying only one approach has no within-machine comparison to
    # make: gmean/best of a single value is that value, so gain == 1.00 by
    # definition and the bars are a tautology, not a result.
    counts = df.groupby("machine")["approach"].nunique()
    df.attrs["degenerate"] = sorted(counts[counts < 2].index)
    if df.attrs["degenerate"] and ref in ("best", "gmean"):
        for m in df.attrs["degenerate"]:
            print(f"  [WARN] {m}: only 1 approach present -> gain is identically "
                  f"1.00 and carries no information.", file=sys.stderr)
        print(f"  [WARN] Use --min-approaches 2 to drop such machines, or "
              f"--machines to select explicitly.", file=sys.stderr)
    return df


def series_order(df):
    return [(m, a) for m in MACHINES
            for a in sorted(set(df[df["machine"] == m]["approach"]))
            if not df[(df["machine"] == m) & (df["approach"] == a)].empty]


def variant_order(df, requested):
    if requested:
        return [v for v in requested if v in set(df["variant"])]
    seen = df[["variant", "gpus"]].drop_duplicates()
    return list(seen.sort_values(["gpus", "variant"])["variant"])


def approach_facecolor(machine, approach):
    """All approaches retain the machine colour as their background."""
    return MACHINES[machine]["color"]


def approach_edgecolor(approach):
    """Use a clear dark outline for hatch strokes and bar separation."""
    return "none"


def approach_display_name(approach):
    """Return the publication-facing approach name without changing data keys."""
    return APPROACH_LEGEND.get(approach, approach)


def handles(pairs):
    return [Patch(facecolor=approach_facecolor(m, a),
                  hatch=APPROACH_HATCH.get(a, ""),
                  edgecolor=approach_edgecolor(a), linewidth=0.35,
                  label=f"{MACHINE_LEGEND.get(m, m)}-{approach_display_name(a)}")
            for m, a in pairs]


def variant_labels(variants, gpus_of):
    """Short tick names for the one-page grid. VARIANT_LABEL wins if set."""
    if VARIANT_LABEL:
        return [VARIANT_LABEL.get(v, v) for v in variants]
    out, seen = [], {}
    for v in variants:
        g = gpus_of.get(v, 0)
        seen[g] = seen.get(g, 0) + 1
        out.append(f"{g} Ver. {seen[g]}")
    return out


def variant_key(variants, labels, gpus_of, model):
    """"2 Ver. 1 = OpenMP Off. 2-GPU Version 1" lines for the legend panel."""
    out, seen = [], {}
    for v, lab in zip(variants, labels):
        g = gpus_of.get(v, 0)
        seen[g] = seen.get(g, 0) + 1
        desc = VARIANT_DESC.get(
            v, f"{model} Off. {g}-GPU Version {seen[g]}")
        out.append(f"{lab} = {desc}")
    return out


# ------------------------------------------------------------ one page ------
def plot_all_n_grid(df, variants, outdir, model, ref, ncols=2):
    """Every N as a panel of ONE figure, sized to fall on a single MDPI page.

    Same content as the per-N figures: speedup over each machine's own 1-GPU
    run, dashed line at 1.0, dotted lines at perfect scaling. What changes is
    what a 85 mm panel can carry -- the per-bar value labels are dropped (63
    bars in 85 mm leaves ~1.3 mm each, so no font size makes them legible) and
    the variant names are shortened. The y-axis is shared across panels, so
    panel-to-panel bar heights are directly comparable.

    The spare grid slot holds the legend, which keeps it out of the panels.
    """
    pairs = series_order(df)
    n = len(pairs)
    if n == 0:
        return
    width = 0.95 / n
    machines = [m for m in MACHINES if m in {p[0] for p in pairs}]

    gpus_of = df.drop_duplicates("variant").set_index("variant")["gpus"].to_dict()
    variants = [v for v in variants if gpus_of.get(v, 2) > 1]
    if not variants:
        print("  [skip] no multi-GPU variants to draw")
        return
    labels = variant_labels(variants, gpus_of)
    # contiguous blocks of equal GPU count, so an ideal-scaling line can be
    # drawn across its own block only
    runs, start = [], 0
    for i in range(1, len(variants) + 1):
        if (i == len(variants)
                or gpus_of.get(variants[i], 0) != gpus_of.get(variants[start], 0)):
            g = gpus_of.get(variants[start], 0)
            if g > 1:
                runs.append((g, start, i - 1))
            start = i
    ideal = sorted({g for g, _, _ in runs})

    Ns = sorted(df["N"].unique())
    # one extra slot for the legend
    nrows = int(np.ceil((len(Ns) + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharey=True,
                             figsize=(MDPI_WIDTH_IN, MDPI_PAGE_HEIGHT_IN))
    flat = axes.flatten()
    x = np.arange(len(variants), dtype=float) * GROUP_SPACING
    group_half_width = ((n - 1) / 2) * width + 0.5 * width * BAR_WIDTH_FRACTION

    top = max(ideal) if ideal else 1.0
    for idx, N in enumerate(Ns):
        ax = flat[idx]
        sub = df[df["N"] == N]

        t1, wall1 = {}, {}
        for m in machines:
            one = sub[(sub["machine"] == m) & (sub["gpus"] == 1)]
            if one.empty:
                continue
            t1[m] = float(one["t_per_step"].median())
            wall1[m] = float(one["median_sec"].median())

        for i, (m, a) in enumerate(pairs):
            vals = []
            for v in variants:
                r = sub[(sub["machine"] == m) & (sub["approach"] == a) &
                        (sub["variant"] == v)]
                vals.append(t1[m] / r.iloc[0]["t_per_step"]
                            if (not r.empty and m in t1) else np.nan)
            off = (i - (n - 1) / 2) * width
            ax.bar(x + off, vals, width * BAR_WIDTH_FRACTION,
                   facecolor=approach_facecolor(m, a),
                   hatch=APPROACH_HATCH.get(a, ""),
                   edgecolor=approach_edgecolor(a), linewidth=0.35)
            if np.any(np.isfinite(vals)):
                top = max(top, float(np.nanmax(vals)))

        # zorder above the bars: 63 bars fill a 85 mm panel almost solidly, so
        # a reference line drawn behind them is invisible
        ax.axhline(1.0, color="k", ls="--", lw=0.8, zorder=4)
        # perfect scaling is 2x for a 2-GPU variant and 4x for a 4-GPU one, so
        # each line is drawn ONLY over the variants it applies to. A single
        # line spanning the whole panel would claim 2x is the target for the
        # 4-GPU group as well, which is wrong.
        for g, i0, i1 in runs:
            x0 = x[i0] - group_half_width
            x1 = x[i1] + group_half_width
            ax.hlines(g, x0, x1, color="#555", ls=":", lw=0.9, zorder=4)
            ax.text(x0 + 0.02, g, f"{g}\u00d7", ha="left", va="bottom",
                    fontsize=5, color="#555", zorder=5)
        ax.set_title(f"N = {N}\u00b3", fontsize=7, pad=11)
        # the 1-GPU wall time every bar in THIS panel was divided by; it grows
        # with N, so it has to be stated per panel and not once for the figure
        if wall1:
            ax.text(0.5, 1.012,
                    "1 GPU:  " + "   ".join(f"{m.split()[0]} {wall1[m]:.1f} s"
                                            for m in machines if m in wall1),
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=4.5, color="#555")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=5)
        ax.tick_params(axis="y", labelsize=6)
        
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=5)
        ax.tick_params(axis="y", labelsize=6)
        for sp in ax.spines.values():          # <- add these four
            sp.set_visible(True)
            sp.set_linewidth(0.6)
            sp.set_color("#999")
            
    for ax in flat[:len(Ns)]:
        ax.set_ylim(0, top * 1.10)

    # legend in the spare slot; the remaining slots (if any) are removed
    spare = flat[len(Ns)]
    spare.axis("off")
    # One column per machine, one row per approach. matplotlib fills a legend
    # column-major and `pairs` is already grouped by machine, so ncol=3 lays
    # the swatches out as the machine x approach grid without reordering.
    mh = [Patch(facecolor=approach_facecolor(m, a),
                hatch=APPROACH_HATCH.get(a, ""),
                edgecolor=approach_edgecolor(a), linewidth=0.35,
                label=f"{MACHINE_LEGEND.get(m, m)}-{approach_display_name(a)}")
          for m, a in pairs]
#    leg = spare.legend(handles=mh, loc="upper center", ncol=len(machines),
#                       fontsize=5., handlelength=1.3, handletextpad=0.4,
#                       columnspacing=0.9, labelspacing=0.35, borderpad=0.2)

#    leg = spare.legend(handles=mh, loc="upper center", ncol=len(machines),
#                       fontsize=5.5, handlelength=1.3, handletextpad=0.4,
#                       columnspacing=0.9, labelspacing=0.35, borderpad=0.2)

    leg = spare.legend(handles=mh, loc="upper center", ncol=3, fontsize=6,
                       handlelength=2.0, handleheight=1.15,
                       handletextpad=0.45, columnspacing=1.0)
    spare.add_artist(leg)


#    def _below(artist, gap=0.03):
    def _below(artist, gap=0.02):     # new
        """Bottom of an already-drawn artist, in axes coordinates."""
        fig.canvas.draw()
        return (artist.get_window_extent(fig.canvas.get_renderer())
                      .transformed(spare.transAxes.inverted()).y0 - gap)

    # the two reference lines, full width underneath
    leg2 = spare.legend(
        handles=[Line2D([0], [0], color="k", ls="--", lw=0.8,
                        label="1 GPU (no gain)"),
                 Line2D([0], [0], color="#555", ls=":", lw=0.9,
                        label="perfect scaling ("
                              + ", ".join(f"{g}\u00d7" for g in ideal)
                              + " w.r.t. each GPU group)")],
#        loc="upper left", bbox_to_anchor=(0.0, _below(leg)),
        loc="upper center", bbox_to_anchor=(0.5, _below(leg)),        # new        
        ncol=1, fontsize=5.5, handlelength=1.8, handletextpad=0.4,
        labelspacing=0.35, borderpad=0.2)
    # leg2 is the axes' own legend, so it is drawn already -- adding it as an
    # artist too would stroke the text twice in vector output

    # spell out the abbreviated x-axis names, so the figure is readable without
    # the caption. Placed by measuring what is above it, not by guessing.
    key = variant_key(variants, labels, gpus_of, model)
#    spare.text(0.0, _below(leg2), "\n".join(key), transform=spare.transAxes,
    spare.text(0.5, _below(leg2), "\n".join(key), transform=spare.transAxes,   # new               
               ha="center", va="top", fontsize=5.5, color="#333",
               linespacing=1.6)
    for ax in flat[len(Ns) + 1:]:
        ax.remove()

    fig.supylabel("speedup over 1 GPU  (t_1GPU / t).", fontsize=8)
    # fig.supxlabel("variant", fontsize=8)
    fig.supxlabel(
    "OpenMP Offloading {2,4}-GPU Versions on H100, MI250X, and Intel-1550.",
    fontsize=7)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save(fig, outdir, f"allN_gain_{model}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote allN_gain_{model} ({len(Ns)} panels on one page)")


# --------------------------------------------------------------- per-N ------
def plot_per_n(df, variants, outdir, model, ref):
    """ONE merged panel per N: every machine x approach side by side.

    Colour = machine, hatch = approach, so all series live in a single axes and
    can be compared directly instead of being split across sub-panels.

    Bars are speedup over that machine's OWN single-GPU run:

        speedup = t_1GPU(machine, N) / t(machine, approach, variant, N)

    The denominator of every bar is a fixed, physically meaningful quantity, so
    unlike a gmean reference a bar cannot rise because some other approach got
    slower. Each machine is divided by its own 1-GPU time, which puts a fast and
    a slow machine on the same axis instead of letting the slowest one set the
    scale and squash the rest.

    Read it as: 1.0 = the extra GPUs bought nothing, 2.0 on a 2-GPU variant =
    perfect scaling. The dotted lines mark those ideal values, so the gap
    between a bar and its dotted line is the scaling that was left on the table.
    Absolute performance is not on the axis any more; the 1-GPU wall times that
    were divided out are printed under the title so it is not lost.

    Times are per timestep, so a differing step count between runs cannot leak
    into the ratio.

    The figure is drawn at the MDPI 170 mm column width so it is placed in the
    paper at 1:1 and the point sizes below survive typesetting.
    """
    pairs = series_order(df)
    n = len(pairs)
    if n == 0:
        return
    width = 0.82 / n
    machines = [m for m in MACHINES if m in {p[0] for p in pairs}]

    # the 1-GPU runs become the reference line, so they are not bars any more
    gpus_of = df.drop_duplicates("variant").set_index("variant")["gpus"].to_dict()
    variants = [v for v in variants if gpus_of.get(v, 2) > 1]
    if not variants:
        print("  [skip] no multi-GPU variants left to draw")
        return

    group_half_width = ((n - 1) / 2) * width + 0.5 * width * BAR_WIDTH_FRACTION

    for N in sorted(df["N"].unique()):
        sub = df[df["N"] == N]
        x = np.arange(len(variants), dtype=float) * GROUP_SPACING
        fig, ax = plt.subplots(figsize=(MDPI_WIDTH_IN, MDPI_HEIGHT_IN))

        # one reference per machine: its 1-GPU time per step, median over the
        # three approaches. They agree to within a couple of percent at 1 GPU,
        # so collapsing them loses nothing and gives every machine a single,
        # stable denominator.
        t1, wall1 = {}, {}
        for m in machines:
            one = sub[(sub["machine"] == m) & (sub["gpus"] == 1)]
            if one.empty:
                continue
            t1[m] = float(one["t_per_step"].median())
            wall1[m] = float(one["median_sec"].median())

        top = 0.0
        for i, (m, a) in enumerate(pairs):
            vals = []
            for v in variants:
                r = sub[(sub["machine"] == m) & (sub["approach"] == a) &
                        (sub["variant"] == v)]
                vals.append(t1[m] / r.iloc[0]["t_per_step"]
                            if (not r.empty and m in t1) else np.nan)
            off = (i - (n - 1) / 2) * width
            ax.bar(x + off, vals, width * BAR_WIDTH_FRACTION,
                   facecolor=approach_facecolor(m, a),
                   hatch=APPROACH_HATCH.get(a, ""),
                   edgecolor=approach_edgecolor(a), linewidth=0.35)
            for xi, val in zip(x + off, vals):
                if not np.isnan(val):
                    ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom",
                            rotation=90, fontsize=4.0, color="#222")
                    top = max(top, val)

        # 1.0 is the single-GPU time itself: a bar at or below it means the
        # extra devices bought nothing
        ax.axhline(1.0, color="k", ls="--", lw=1.0, zorder=1)
        # and one dotted line per GPU count actually plotted, at perfect scaling
        for g in sorted({gpus_of.get(v, 0) for v in variants} - {0}):
            ax.axhline(g, color="#999", ls=":", lw=0.8, zorder=1)
            ax.text(x[-1] + group_half_width + 0.06, g, f" ideal {g}×",
                    ha="left", va="center", fontsize=5.5, color="#999",
                    clip_on=False)
            top = max(top, g)

        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=25, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        # a multi-GPU variant whose extra devices did nothing still gets a
        # bar; flag it so a flat 1-vs-4 GPU comparison is not read as a result
        suspect = set()
        if "baseline_ok" in sub.columns:
            for v in variants:
                r = sub[(sub["variant"] == v) & (~sub["baseline_ok"])]
                if not r.empty:
                    suspect.add(v)
        for lbl in ax.get_xticklabels():
            if lbl.get_text() in suspect:
                lbl.set_color("#C92A2A")
                lbl.set_fontweight("bold")

        ax.set_ylabel("speedup over 1 GPU  (t_1GPU / t)", fontsize=8)
        ax.set_xlabel("variant", fontsize=8)
        ax.set_ylim(0, (top if top > 0 else 1.0) * 1.18)
        # the absolute scale the ratio divides out, given back in one line
        divided_out = "   ·   ".join(f"{m} {wall1[m]:.3g} s"
                                     for m in machines if m in wall1)
        ax.set_title(f"{model} — N = {N}³ — speedup over each machine's own "
                     f"single-GPU run. Higher is better;\n1.0 = the extra GPUs "
                     f"bought nothing."
                     + (f"    [1-GPU wall time: {divided_out}]"
                        if divided_out else ""),
                     fontsize=8, pad=6)
        ax.legend(handles=handles(pairs)
                  + [Line2D([0], [0], color="k", ls="--", lw=1.0,
                            label="1 GPU (no gain)"),
                     Line2D([0], [0], color="#999", ls=":", lw=0.8,
                            label="perfect scaling")],
                  ncol=min(n, 3), fontsize=6, handlelength=1.8,
                  handleheight=1.0, handletextpad=0.45, columnspacing=0.9,
                  loc="upper left", bbox_to_anchor=(0.0, 0.86))
        caption = []
        deg = [m for m in df.attrs.get("degenerate", []) if m in machines]
        if deg:
            caption.append(", ".join(deg) + ": single approach only")
        if caption:
            ax.text(0.5, -0.28, "  |  ".join(caption), transform=ax.transAxes,
                    ha="center", va="top", fontsize=5.5, color="#C92A2A")
        fig.tight_layout()
        n_value = float(N)
        n_stem = (f"{int(n_value):04d}" if n_value.is_integer()
                  else str(N).replace(".", "p"))
        save(fig, outdir, f"N{n_stem}_gain_{model}")
        plt.close(fig)
    print(f"  wrote {len(df['N'].unique())} merged per-N gain figures")


# --------------------------------------------------------- portability ------
def plot_portability(df, variants, outdir, model, ref):
    machines = [m for m in MACHINES if m in set(df["machine"])]
    if len(machines) < 2:
        print("  [skip] portability scatter needs >=2 machines")
        return
    others = sorted(set(df["approach"]))          # include the reference too
    if len(others) < 2:
        print("  [skip] portability scatter needs >=2 approaches")
        return

    npair = len(machines) * (len(machines) - 1) // 2
    # with 3 machines this is 3 panels (H100-PVC, H100-MI250X, PVC-MI250X)
    fig, axes = plt.subplots(1, npair, figsize=(5.0 * npair, 5.2), squeeze=False)
    k = 0
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ap_color = {
        a: APPROACH_COLORS.get(a, default_cycle[i % len(default_cycle)])
        for i, a in enumerate(others)
    }

    for ia in range(len(machines)):
        for ib in range(ia + 1, len(machines)):
            ax = axes[0][k]; k += 1
            mA, mB = machines[ia], machines[ib]
            for a in others:
                gA, gB = [], []
                for v in variants:
                    for N in sorted(df["N"].unique()):
                        ra = df[(df.machine == mA) & (df.approach == a) &
                                (df.variant == v) & (df.N == N)]
                        rb = df[(df.machine == mB) & (df.approach == a) &
                                (df.variant == v) & (df.N == N)]
                        if ra.empty or rb.empty:
                            continue
                        gA.append(ra.iloc[0]["gain"])
                        gB.append(rb.iloc[0]["gain"])
                if not gA:
                    continue
                ax.scatter(gA, gB, s=26, alpha=0.75, color=ap_color[a],
                           edgecolor="white", lw=0.5,
                           label=f"{approach_display_name(a)}  (n={len(gA)})")
            lim = [0, max(1.2, *(ax.get_xlim() + ax.get_ylim()))]
            ax.plot(lim, lim, "k--", lw=0.9, zorder=0)
            ax.axhline(1, color="#999", lw=0.7, zorder=0)
            ax.axvline(1, color="#999", lw=0.7, zorder=0)
            ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.set_xlabel(f"gain on {mA}")
            ax.set_ylabel(f"gain on {mB}")
            ax.set_title(f"portable if on the diagonal\n({mA} vs {mB})", fontsize=10)
            ax.legend(fontsize=8)
    fig.suptitle(f"{model} — is the approach gain over {ref} portable? "
                 f"(one point per variant × N)", fontsize=11)
    fig.tight_layout()
    save(fig, outdir, f"portability_{model}")
    plt.close(fig)
    print(f"  wrote portability_{model}.png")


def plot_portability_3d(df, variants, outdir, model, ref):
    """One 3D scatter replacing the three pairwise panels.

    Axes are the gain on each of the three machines, so a single point is one
    (approach, variant, N) cell measured everywhere. The 2D version asked "is
    this point on the diagonal" three times; here there is one diagonal, the
    line x=y=z, and a point sitting on it means the approach did the same thing
    on all three machines -- i.e. the improvement is a property of the code.

    Everything is drawn in log2, because these are ratios: a cell that is 2x on
    one machine and 0.5x on another is equally far from the diagonal as one
    that is 4x and 1x, and only log spacing shows that honestly. Tick labels are
    converted back to plain ratios.

    Faint grey shadows are cast onto the three walls, which recovers the
    pairwise views the 2D version showed, without needing separate panels.
    """
    machines = [m for m in MACHINES if m in set(df["machine"])]
    if len(machines) < 3:
        print(f"  [skip] 3D portability needs 3 machines, have {len(machines)}; "
              f"falling back to pairwise")
        return plot_portability(df, variants, outdir, model, ref)
    if len(machines) > 3:
        print(f"  [note] {len(machines)} machines; 3D plot uses the first three: "
              f"{machines[:3]}")
    mx, my, mz = machines[:3]

    piv = (df[df["variant"].isin(variants)]
             .pivot_table(index=["approach", "variant", "N"], columns="machine",
                          values="gain"))
    for m in (mx, my, mz):
        if m not in piv.columns:
            print("  [skip] a machine has no overlapping cells")
            return
    piv = piv.dropna(subset=[mx, my, mz])
    piv = piv[(piv[[mx, my, mz]] > 0).all(axis=1)]
    if len(piv) < 3:
        print("  [skip] fewer than 3 cells measured on all three machines")
        return

    L = np.log2(piv[[mx, my, mz]])
    lo, hi = float(L.values.min()), float(L.values.max())
    pad = max(0.12 * (hi - lo), 0.15)
    lo, hi = lo - pad, hi + pad

    fig = plt.figure(figsize=(9.0, 8.0))
    ax = fig.add_subplot(111, projection="3d")

    # the portable line, plus its shadow on each wall
    ax.plot([lo, hi], [lo, hi], [lo, hi], "k--", lw=1.4, zorder=1)
    ax.plot([lo, hi], [lo, hi], [lo, lo], color="#bbb", ls="--", lw=0.8, zorder=0)
    ax.plot([lo, lo], [lo, hi], [lo, hi], color="#bbb", ls="--", lw=0.8, zorder=0)
    ax.plot([lo, hi], [hi, hi], [lo, hi], color="#bbb", ls="--", lw=0.8, zorder=0)

    for a in sorted(set(piv.index.get_level_values("approach"))):
        sel = L[L.index.get_level_values("approach") == a]
        if sel.empty:
            continue
        gx, gy, gz = sel[mx].values, sel[my].values, sel[mz].values
        # spread across machines, as a plain factor: 1.00 = perfectly portable
        spread = 2 ** (sel.max(axis=1) - sel.min(axis=1))
        col = APPROACH_COLORS.get(a, "#666")
        ax.scatter(gx, gy, gz, s=34, alpha=0.85, color=col, edgecolor="white",
                   lw=0.4, depthshade=False, zorder=5,
                   label=f"{approach_display_name(a)}  (median spread {spread.median():.2f}x, "
                         f"worst {spread.max():.2f}x)")
        # pairwise shadows on the three walls
        ax.scatter(gx, gy, zs=lo, zdir="z", s=9, color=col, alpha=0.16, zorder=0)
        ax.scatter(gy, gz, zs=lo, zdir="x", s=9, color=col, alpha=0.16, zorder=0)
        ax.scatter(gx, gz, zs=hi, zdir="y", s=9, color=col, alpha=0.16, zorder=0)

    # mark "no change anywhere"
    if lo < 0 < hi:
        ax.scatter([0], [0], [0], marker="+", s=90, color="k", lw=1.2, zorder=6)

    ticks = [k for k in range(int(np.floor(lo)), int(np.ceil(hi)) + 1)]
    labels = [f"{2.0**k:g}" if k >= 0 else f"{2.0**k:.2f}".rstrip("0")
              for k in ticks]
    for setter, tset, tlab in ((ax.set_xlim, ax.set_xticks, ax.set_xticklabels),
                               (ax.set_ylim, ax.set_yticks, ax.set_yticklabels),
                               (ax.set_zlim, ax.set_zticks, ax.set_zticklabels)):
        setter(lo, hi); tset(ticks); tlab(labels, fontsize=8)

    ax.set_xlabel(f"gain on {mx}", labelpad=8)
    ax.set_ylabel(f"gain on {my}", labelpad=8)
    ax.set_zlabel(f"gain on {mz}", labelpad=8)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=-58)
    ax.grid(True)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.96))
    fig.suptitle(f"{model} — is the gain over {ref} portable?\n"
                 f"one point per approach x variant x N; on the dashed diagonal "
                 f"= same everywhere = a property of the code",
                 fontsize=11)
    save(fig, outdir, f"portability3d_{model}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote portability3d_{model}.png")


# ------------------------------------------------------------- heatmap ------
def plot_heatmap(df, variants, outdir, model, ref):
    pairs = series_order(df)                      # include the reference too
    if not pairs:
        print("  [skip] nothing to plot")
        return
    Ns = sorted(df["N"].unique())
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.0 * len(pairs) + 1.6,
                                                     0.42 * len(variants) + 2.8),
                             squeeze=False, sharey=True)
    grids = []
    for m, a in pairs:
        g = np.full((len(variants), len(Ns)), np.nan)
        for i, v in enumerate(variants):
            for j, N in enumerate(Ns):
                r = df[(df.machine == m) & (df.approach == a) &
                       (df.variant == v) & (df.N == N)]
                if not r.empty:
                    g[i, j] = r.iloc[0]["gain"]
        grids.append(g)

    # A panel can be entirely empty (machine did not run this approach), and
    # nanmax over an all-NaN array returns NaN, which poisons the colour range.
    finite = np.concatenate([np.abs(np.log2(g[np.isfinite(g) & (g > 0)]))
                             for g in grids]) if grids else np.array([])
    vmax = max(float(finite.max()) if finite.size else 0.0, 0.05)

    im = None
    for ax, (m, a), g in zip(axes[0], pairs, grids):
        with np.errstate(divide="ignore", invalid="ignore"):
            im = ax.imshow(np.log2(g), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           aspect="auto")
        for i in range(len(variants)):
            for j in range(len(Ns)):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f"{g[i,j]:.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if abs(np.log2(g[i, j])) > vmax*0.55 else "#222")
        ax.set_xticks(range(len(Ns)))
        ax.set_xticklabels(Ns, rotation=45, fontsize=7)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants, fontsize=7)
        ax.set_title(f"{m} · {a}", fontsize=10)
        ax.set_xlabel("N")
        ax.grid(False)
    if im is not None:
        cb = fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)
        cb.set_label(f"log2 gain over approach {ref}")
    fig.suptitle(f"{model} — approach gain over {ref}   (red = faster)", fontsize=11)
    save(fig, outdir, f"heatmap_{model}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote heatmap_{model}.png")


# ------------------------------------------------------------ decision ------
def plot_decision(df, variants, outdir, model, ref):
    """One panel per machine, x = N, one line per approach.

    The approach to recommend is the one whose line sits above 1.0 in EVERY
    panel at EVERY N. Gains are aggregated across the selected variants with a
    geometric mean, which is the correct average for ratios.
    """
    agg = (df[df["variant"].isin(variants)]
             .groupby(["machine", "approach", "N"])["gain"]
             .apply(lambda s: float(np.exp(np.log(s).mean())))
             .reset_index())
    machines = [m for m in MACHINES if m in set(agg["machine"])]
    approaches = sorted(set(agg["approach"]))
    Ns = sorted(agg["N"].unique())

    fig, axes = plt.subplots(1, len(machines), figsize=(4.6 * len(machines) + 1, 4.8),
                             squeeze=False, sharey=True)
    for ax, m in zip(axes[0], machines):
        for a in approaches:
            r = agg[(agg["machine"] == m) & (agg["approach"] == a)].sort_values("N")
            if r.empty:
                continue
            ax.plot(r["N"], r["gain"], "o-", ms=5, lw=1.9,
                    color=APPROACH_COLORS.get(a, "#666"), label=approach_display_name(a))
        ax.axhline(1.0, color="k", ls="--", lw=1.0)
        ax.set_xticks(Ns); ax.set_xticklabels(Ns, rotation=45, fontsize=8)
        ax.set_xlabel("N")
        ax.set_title(m, fontsize=11)
        if len(set(df[df["machine"] == m]["approach"])) < 2:
            ax.text(0.5, 0.5, "single approach\nno comparison", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#C92A2A", alpha=0.6)
    axes[0][0].set_ylabel(f"gain vs {ref}  (geo-mean over {len(variants)} variant(s))")
    axes[0][0].legend(fontsize=9)
    fig.suptitle(f"{model} — which approach is above 1.0 on every machine, at every N?",
                 fontsize=12)
    fig.tight_layout()
    save(fig, outdir, f"decision_{model}")
    plt.close(fig)
    print(f"  wrote decision_{model}.png")
    return agg


def decision_table(df, agg, variants, ref):
    """Maximin summary: the safe choice is the approach with the best WORST cell."""
    machines = sorted(set(agg["machine"]))
    approaches = sorted(set(agg["approach"]))
    usable = [m for m in machines
              if len(set(df[df["machine"] == m]["approach"])) >= 2]

    print(f"\n{'='*72}\nDECISION SUMMARY  (variants: {', '.join(variants)})\n{'='*72}")
    if not usable:
        print("No machine has >=2 approaches. Cannot conclude anything yet.")
        return
    if len(usable) < len(MACHINES):
        missing = [m for m in MACHINES if m not in usable]
        print(f"INCOMPLETE: no multi-approach data for {', '.join(missing)}.")
        print("Any conclusion below covers only:", ", ".join(usable), "\n")

    print(f"  {'approach':<9} {'worst cell':>11} {'geo-mean':>9} {'best cell':>10}   per-machine geo-mean")
    print("  " + "-" * 78)
    rows = {}
    for a in approaches:
        sel = agg[(agg["approach"] == a) & (agg["machine"].isin(usable))]["gain"]
        if sel.empty:
            continue
        per_m = []
        for m in usable:
            g = agg[(agg["machine"] == m) & (agg["approach"] == a)]["gain"]
            per_m.append(f"{m}={float(np.exp(np.log(g).mean())):.2f}" if len(g) else f"{m}=--")
        rows[a] = (sel.min(), float(np.exp(np.log(sel).mean())), sel.max())
        print(f"  {a:<9} {sel.min():11.3f} {rows[a][1]:9.3f} {sel.max():10.3f}   "
              + "  ".join(per_m))

    if rows:
        maximin = max(rows, key=lambda a: rows[a][0])
        best_avg = max(rows, key=lambda a: rows[a][1])
        print(f"\n  Safest choice (best worst-case)  : approach {maximin}  "
              f"(never below {rows[maximin][0]:.2f})")
        print(f"  Best on average                  : approach {best_avg}")
        if maximin != best_avg:
            print("  -> these disagree: one approach is faster on average but has a "
                  "worse floor.")

    # does the ranking agree across machines?
    print("\n  Ranking per machine (best first):")
    orders = {}
    for m in usable:
        g = agg[agg["machine"] == m].groupby("approach")["gain"].apply(
            lambda s: float(np.exp(np.log(s).mean())))
        orders[m] = list(g.sort_values(ascending=False).index)
        print(f"    {m:<10} {' > '.join(orders[m])}")
    if len(set(map(tuple, orders.values()))) == 1 and len(orders) > 1:
        print("    -> ranking is CONSISTENT across machines; the winner is portable.")
    elif len(orders) > 1:
        print("    -> ranking DISAGREES across machines; there is no single winner.")

    # flag ties that are inside measurement noise
    print("\n  Note: your run-to-run spread is 1-6%, so gains within ~1.06 of each")
    print("  other are not distinguishable. Prefer the simpler code in that case.")


# ----------------------------------------------------------------- main -----
def main():
    ap = argparse.ArgumentParser()
    # The script is intentionally locked to the directory from which it is run.
    # --root and an old-style positional root are accepted only so older command
    # lines do not fail; their values are ignored.
    ap.add_argument("ignored_root", nargs="?", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--root", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--model", default="OpenMP")
    ap.add_argument("--ref", default="gmean",
                    help="reference: 'gmean' (default), 'best', or an approach "
                         "label such as 01. NOTE: naming an approach pins that "
                         "approach at exactly 1.00 in every group, which HIDES "
                         "how good or bad its baseline actually was. Prefer "
                         "gmean unless you specifically want 01 as the control.")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--machines", nargs="*", default=None)
    ap.add_argument("--decision-variants", nargs="*", default=None,
                    help="variants the recommendation should be based on; "
                         "default is all. Usually you want the variant you would "
                         "actually ship, e.g. 4-omp-stream-omp-p2p")
    ap.add_argument("--min-approaches", type=int, default=1,
                    help="drop machines with fewer than this many approaches "
                         "(use 2 to exclude machines that cannot be compared)")
    ap.add_argument("--min-baseline-efficiency", type=float, default=None,
                    help="drop (machine, variant, N) cells whose REFERENCE run "
                         "had parallel efficiency below this. Use ~0.6 to "
                         "exclude variants that did not scale in the reference "
                         "approach, whose 'gain' is a bug fix, not a speedup")
    ap.add_argument("--efficiency-warn", type=float, default=0.6,
                    help="efficiency below which a cell is reported as suspect "
                         "(default 0.6); reporting only, does not filter")
    ap.add_argument("--complete-cells-only", action="store_true",
                    help="keep only (variant, N) cells measured on every "
                         "machine x approach, so all panels compare like with like")
    ap.add_argument("--portability", default="auto",
                    choices=["auto", "2d", "3d", "both"],
                    help="auto/3d: single 3D scatter (needs 3 machines, falls "
                         "back to pairwise); 2d: the three pairwise panels; "
                         "both: write each")
    ap.add_argument("--outdir", default="plots-approach")
    ap.add_argument("--list", action="store_true",
                    help="DIAGNOSTIC ONLY: show which folders were matched, "
                         "then exit WITHOUT writing any plots")
    ap.add_argument("--format", nargs="*", default=["png"],
                    choices=["png", "pdf", "svg"],
                    help="output formats, e.g. --format png pdf")
    args = ap.parse_args()

    global FORMATS
    FORMATS = args.format

    root = Path.cwd().resolve()
    if args.root is not None or args.ignored_root is not None:
        supplied = args.root if args.root is not None else args.ignored_root
        print(
            f"  [NOTE] ignoring supplied root {supplied!r}; "
            f"using current directory {root}",
            file=sys.stderr,
        )
    if args.list:
        print(f"Scanning {root} for */{args.model}/summary.csv\n")
        rows = discover(root, args.model)
        if not rows:
            print("  nothing matched. Contents of root:")
            for d in sorted(p for p in root.iterdir() if p.is_dir()):
                print(f"    {d.name}")
            return
        for machine, approach, csv, exists in rows:
            print(f"  {'OK ' if exists else 'NO '} {machine:9s} approach {approach}  {csv}")
        by = {}
        for machine, approach, _, exists in rows:
            if exists:
                by.setdefault(machine, []).append(approach)
        print("\n(--list is diagnostic only: no plots were written. "
              "Re-run without --list.)")
        print("\nUsable:")
        for m, aps in by.items():
            print(f"  {m:9s} approaches {sorted(aps)}")
        return
    outdir = Path(args.outdir).expanduser() / args.model
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading from {root} ({args.model}):")
    df = load(root, args.model)
    if args.machines:
        df = df[df["machine"].isin(args.machines)]

    report_coverage(df)
    if args.complete_cells_only:
        nseries = len(set(zip(df["machine"], df["approach"])))
        cnt = df.groupby(["variant", "N"]).apply(
            lambda g: len(set(zip(g["machine"], g["approach"]))),
            include_groups=False)
        good = set(cnt[cnt == nseries].index)
        before = len(df)
        df = df[[(v, n) in good for v, n in zip(df["variant"], df["N"])]]
        print(f"  --complete-cells-only: kept {len(df)}/{before} rows")
        if df.empty:
            sys.exit("Nothing left after --complete-cells-only.")

    df = add_efficiency(df)
    report_efficiency(df, args.efficiency_warn)

    df = add_gain(df, args.ref)
    df = mark_baseline(df, args.ref, args.efficiency_warn)

    if args.min_baseline_efficiency is not None:
        before = len(df)
        keep = ~(df["baseline_eff"] < args.min_baseline_efficiency)
        dropped = df[~keep][["machine", "variant", "N"]].drop_duplicates()
        df = df[keep]
        print(f"\n  --min-baseline-efficiency {args.min_baseline_efficiency}: "
              f"dropped {before - len(df)} rows over "
              f"{len(dropped)} (machine, variant, N) cells")
        for v in sorted(set(dropped["variant"])):
            print(f"      {v}")
        if df.empty:
            sys.exit("Nothing left after --min-baseline-efficiency filtering.")

    if args.min_approaches > 1:
        keep = df.groupby("machine")["approach"].nunique()
        keep = keep[keep >= args.min_approaches].index
        dropped = sorted(set(df["machine"]) - set(keep))
        if dropped:
            print(f"  dropped (fewer than {args.min_approaches} approaches): "
                  f"{', '.join(dropped)}")
        df = df[df["machine"].isin(keep)]
        if df.empty:
            sys.exit("Nothing left after --min-approaches filtering.")

    variants = variant_order(df, args.variants)

    print(f"\nMachines  : {sorted(set(df['machine']))}")
    print(f"Approaches: {sorted(set(df['approach']))}   (reference = {args.ref})")
    print(f"Variants  : {len(variants)}\n")

    df.to_csv(outdir / f"gains_{args.model}.csv", index=False)
    plot_per_n(df, variants, outdir, args.model, args.ref)
    plot_all_n_grid(df, variants, outdir, args.model, args.ref)
    if args.portability in ('3d', 'auto'):
        plot_portability_3d(df, variants, outdir, args.model, args.ref)
    if args.portability in ('2d', 'both'):
        plot_portability(df, variants, outdir, args.model, args.ref)
    plot_heatmap(df, variants, outdir, args.model, args.ref)
    plot_efficiency(df, variants, outdir, args.model, args.efficiency_warn)

    dec_variants = args.decision_variants or variants
    dec_variants = [v for v in dec_variants if v in set(df["variant"])]
    agg = plot_decision(df, dec_variants, outdir, args.model, args.ref)
    decision_table(df, agg, dec_variants, args.ref)

    # console: geometric-mean gain per machine x approach, and agreement
    print(f"\nGeometric-mean gain over approach {args.ref} (all variants x N):")
    print(f"  {'machine':<10} {'approach':<9} {'gmean':>7}  {'min':>6} {'max':>6}")
    print("  " + "-" * 43)
    for (m, a), g in df.groupby(["machine", "approach"]):
        gm = np.exp(np.log(g["gain"].dropna()).mean())
        print(f"  {m:<10} {a:<9} {gm:7.3f}  {g['gain'].min():6.2f} {g['gain'].max():6.2f}")

    machines = sorted(set(df["machine"]))
    if len(machines) >= 2:
        print("\nPortability check (correlation of gain across machines):")
        for a in sorted(set(df["approach"])):
            piv = (df[df["approach"] == a]
                   .pivot_table(index=["variant", "N"], columns="machine",
                                values="gain"))
            piv = piv.dropna()
            if piv.shape[1] < 2 or len(piv) < 3:
                continue
            spread = (piv.max(axis=1) / piv.min(axis=1)).max()
            if piv.std().min() == 0:      # the reference: all 1.00 by definition
                print(f"  approach {a}: reference, gain == 1.00 everywhere")
                continue
            # with 3 machines, report every pair rather than just the first two
            corr = piv.corr()
            cols = list(piv.columns)
            pairs_txt = "  ".join(
                f"{cols[i]}~{cols[j]}: r={corr.iloc[i, j]:5.2f}"
                for i in range(len(cols)) for j in range(i + 1, len(cols)))
            print(f"  approach {a}: {pairs_txt}   worst cross-machine "
                  f"disagreement = {spread:.2f}x")

    print(f"\nAll output in {outdir}")


if __name__ == "__main__":
    main()
