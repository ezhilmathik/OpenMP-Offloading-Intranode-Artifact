#!/usr/bin/env bash
#
# Regenerate every figure for the NVIDIA H100 benchmarks.
#
# Runs each directory's plot script against the raw results_*/timings.csv data
# already present in the repository. No GPU is required — this reproduces the
# published figures from the shipped measurements.
#
# Usage:  ./all-plots.sh
#
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# Prefer python3; fall back to python if that is what exists.
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "ERROR: no python interpreter found on PATH." >&2
    exit 1
fi

DIRS=(
    Synchronous-test-H-to-D-Measurements
    Synchronous-test-D-to-H-Measurements
    Asynchronous-test-H-to-D-Measurements
    Asynchronous-test-D-to-H-Measurements
    P2P-Measurements
)

failed=()

for dir in "${DIRS[@]}"; do
    echo "============================================================"
    echo "$dir"
    echo "============================================================"

    if [[ ! -d "$dir" ]]; then
        echo "  ERROR: directory not found"
        failed+=("$dir (missing directory)")
        continue
    fi

    # Exactly one plot script per directory, named plot_*.py
    mapfile -t scripts < <(find "$dir" -maxdepth 1 -name 'plot_*.py' -printf '%f\n' | sort)

    if (( ${#scripts[@]} == 0 )); then
        echo "  ERROR: no plot_*.py found"
        failed+=("$dir (no plot script)")
        continue
    fi

    for script in "${scripts[@]}"; do
        echo "  Running $script"
        if ( cd "$dir" && "$PY" "$script" ); then
            :
        else
            echo "  ERROR: $script exited non-zero"
            failed+=("$dir/$script")
        fi
    done
    echo ""
done

echo "============================================================"
if (( ${#failed[@]} == 0 )); then
    echo "All figures regenerated successfully."
    echo ""
    echo "Output:"
    find . -path '*/plots_final/*' -name '*.pdf' -printf '  %p\n' | sort
    echo "============================================================"
    exit 0
else
    echo "FAILED (${#failed[@]}):"
    printf '  %s\n' "${failed[@]}"
    echo "============================================================"
    exit 1
fi
