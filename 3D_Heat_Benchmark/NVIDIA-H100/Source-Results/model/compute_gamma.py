#!/usr/bin/env python3
"""
Compute the machine constant gamma from an OpenMP summary.csv file.

Model:
    T1 = N_s * gamma * (n - 2)^3
    gamma = T1 / (N_s * (n - 2)^3)

By default, the script:
  1. Finds summary.csv automatically.
  2. Selects rows whose variant is "1-omp".
  3. Uses median_sec as the single-GPU wall time.
  4. Reports the mean gamma from the largest three grids.

Examples:
    # From Source-Results/model:
    python3 compute_gamma.py

    # Explicit summary path:
    python3 compute_gamma.py --summary ../OpenMP/summary.csv

    # Use minimum runtime instead of median:
    python3 compute_gamma.py --time-column min_sec

    # Also create gamma_vs_N.png:
    python3 compute_gamma.py --plot
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Return paths in order, without duplicates."""
    result: list[Path] = []
    seen: set[Path] = set()

    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)

    return result


def find_summary(explicit: str | None, directory: str | None) -> Path:
    """Find summary.csv using explicit and common project-relative locations."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Summary file does not exist: {path}")
        return path

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    candidates: list[Path] = []

    if directory:
        base = Path(directory).expanduser()
        candidates.extend(
            [
                base / "summary.csv",
                base / "OpenMP" / "summary.csv",
                base / ".." / "OpenMP" / "summary.csv",
            ]
        )

    candidates.extend(
        [
            cwd / "summary.csv",
            cwd / "OpenMP" / "summary.csv",
            cwd / ".." / "OpenMP" / "summary.csv",
            script_dir / "summary.csv",
            script_dir / "OpenMP" / "summary.csv",
            script_dir / ".." / "OpenMP" / "summary.csv",
        ]
    )

    checked = unique_paths(candidates)
    for path in checked:
        if path.is_file():
            return path

    locations = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not find summary.csv. Checked:\n"
        f"{locations}\n"
        "Pass its location explicitly with --summary PATH."
    )


def parse_int(value: str | None, field: str, row_number: int) -> int:
    if value is None or not value.strip():
        raise ValueError(f"row {row_number}: missing {field}")
    return int(value)


def parse_float(value: str | None, field: str, row_number: int) -> float:
    if value is None or not value.strip():
        raise ValueError(f"row {row_number}: missing {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: non-finite {field}")
    return result


def read_summary(
    summary_path: Path,
    variant: str,
    time_column: str,
) -> list[dict[str, int | float]]:
    """Read matching single-GPU rows and compute gamma for each grid."""
    rows: list[dict[str, int | float]] = []

    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        required = {"variant", "N", "steps", time_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"{summary_path} is missing required column(s): {missing_text}"
            )

        for row_number, row in enumerate(reader, start=2):
            if (row.get("variant") or "").strip() != variant:
                continue

            try:
                n = parse_int(row.get("N"), "N", row_number)
                steps = parse_int(row.get("steps"), "steps", row_number)
                runtime = parse_float(
                    row.get(time_column), time_column, row_number
                )
            except ValueError as error:
                print(f"[skip] {error}", file=sys.stderr)
                continue

            if n <= 2:
                print(
                    f"[skip] row {row_number}: N must be greater than 2",
                    file=sys.stderr,
                )
                continue
            if steps <= 0:
                print(
                    f"[skip] row {row_number}: steps must be positive",
                    file=sys.stderr,
                )
                continue
            if runtime <= 0:
                print(
                    f"[skip] row {row_number}: {time_column} must be positive",
                    file=sys.stderr,
                )
                continue

            interior = (n - 2) ** 3
            gamma = runtime / (steps * interior)

            rows.append(
                {
                    "N": n,
                    "interior_points": interior,
                    "timesteps": steps,
                    "T1_seconds": runtime,
                    "gamma_s_per_point": gamma,
                }
            )

    rows.sort(key=lambda item: int(item["N"]))
    return rows


def print_table(rows: list[dict[str, int | float]]) -> None:
    print(
        f"\n{'N':>6} {'interior=(n-2)^3':>18} {'steps':>7} "
        f"{'T1 [s]':>12} {'gamma [s/pt]':>15} {'gamma [ps/pt]':>15}"
    )
    print("-" * 78)

    for row in rows:
        gamma = float(row["gamma_s_per_point"])
        print(
            f"{int(row['N']):>6} "
            f"{int(row['interior_points']):>18} "
            f"{int(row['timesteps']):>7} "
            f"{float(row['T1_seconds']):>12.6f} "
            f"{gamma:>15.4e} "
            f"{gamma * 1e12:>15.3f}"
        )


def write_csv(
    rows: list[dict[str, int | float]],
    output_path: Path,
) -> None:
    fieldnames = [
        "N",
        "interior_points",
        "timesteps",
        "T1_seconds",
        "gamma_s_per_point",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plot(
    rows: list[dict[str, int | float]],
    plateau_gamma: float,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping plot", file=sys.stderr)
        return

    xs = [int(row["N"]) for row in rows]
    ys = [float(row["gamma_s_per_point"]) * 1e12 for row in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, "o-")
    plt.axhline(
        plateau_gamma * 1e12,
        linestyle="--",
        label=f"plateau = {plateau_gamma * 1e12:.2f} ps/pt",
    )
    plt.xlabel("grid size N")
    plt.ylabel(r"$\gamma$ [ps / point-update]")
    plt.title("Machine constant from single-GPU summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="optional project/results directory used while searching for summary.csv",
    )
    parser.add_argument(
        "--summary",
        help="explicit path to summary.csv",
    )
    parser.add_argument(
        "--exe",
        default="1-omp",
        help="variant to select from summary.csv (default: 1-omp)",
    )
    parser.add_argument(
        "--time-column",
        default="median_sec",
        choices=("min_sec", "median_sec", "max_sec"),
        help="runtime column used as T1 (default: median_sec)",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=3,
        help="number of largest grids used for plateau mean (default: 3)",
    )
    parser.add_argument(
        "--output",
        help="output CSV path; default is gamma_single_gpu.csv beside summary.csv",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="save gamma_vs_N.png beside the output CSV",
    )
    args = parser.parse_args()

    if args.tail <= 0:
        parser.error("--tail must be positive")

    try:
        summary_path = find_summary(args.summary, args.directory)
        rows = read_summary(summary_path, args.exe, args.time_column)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(str(error))

    if not rows:
        sys.exit(
            f"No usable rows found for variant {args.exe!r} in {summary_path}"
        )

    print(f"Reading: {summary_path}")
    print(f"Variant: {args.exe}")
    print(f"Runtime column: {args.time_column}")

    print_table(rows)

    tail_count = min(args.tail, len(rows))
    tail = rows[-tail_count:]
    plateau_gamma = sum(
        float(row["gamma_s_per_point"]) for row in tail
    ) / tail_count
    plateau_ns = ", ".join(str(int(row["N"])) for row in tail)

    print("-" * 78)
    print(
        f"Plateau gamma (mean of N = {plateau_ns}): "
        f"{plateau_gamma:.4e} s/point-update "
        f"({plateau_gamma * 1e12:.2f} ps/point-update)"
    )
    print("Use this value for gamma in the performance model.\n")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else summary_path.parent / "gamma_single_gpu.csv"
    )
    write_csv(rows, output_path)
    print(f"Wrote {output_path}")

    if args.plot:
        plot_path = output_path.parent / "gamma_vs_N.png"
        make_plot(rows, plateau_gamma, plot_path)
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
