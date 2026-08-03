#!/usr/bin/env bash
# Remove build and editor debris from the benchmark tree.
# Usage:  ./clean.sh          list what would be removed
#         ./clean.sh --force  actually remove it

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# Editor backups and build leftovers, matched by name.
mapfile -t JUNK < <(find . -type f \( \
      -name '*~' -o -name '*.o' -o -name '*.mod' \
   -o -name '*.bak' -o -name '.DS_Store' -o -name 'core.*' \) )

# Compiled executables, matched by content rather than name.
mapfile -t BINS < <(find . -type f -exec file --mime-type {} + \
   | grep -E 'x-(pie-)?executable|x-sharedlib' | cut -d: -f1)

TARGETS=( "${JUNK[@]}" "${BINS[@]}" )

if (( ${#TARGETS[@]} == 0 )); then
   echo "Nothing to clean."
   exit 0
fi

printf '%s\n' "${TARGETS[@]}"
echo "---"

if (( FORCE )); then
   printf '%s\0' "${TARGETS[@]}" | xargs -0 rm -v --
   echo "Removed ${#TARGETS[@]} file(s)."
else
   echo "${#TARGETS[@]} file(s) would be removed. Re-run with --force."
fi
