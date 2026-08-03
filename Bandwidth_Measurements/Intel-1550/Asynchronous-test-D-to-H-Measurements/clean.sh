#!/bin/bash
# Remove binaries, build/run logs and (optionally) result directories.
#   ./clean.sh          -> binaries + logs
#   ./clean.sh --all    -> also delete results_* directories

make clean 2>/dev/null
rm -f build-*.out run-*.out icpx_build_*.log icx_build_*.log

if [[ "${1:-}" == "--all" ]]; then
  rm -rf results_*
  echo "Removed binaries, logs and results_* directories."
else
  echo "Removed binaries and logs. (Use --all to also delete results_* dirs.)"
fi
