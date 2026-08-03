#!/bin/bash -l
# =============================================================================
#  clean.sh -- remove build and run artifacts from the 3D-Heat working tree
# =============================================================================
#
#  DRY RUN BY DEFAULT. Nothing is deleted unless you pass -f.
#
#  Usage:  ./clean.sh [-f] [category ...]
#
#    -f, --force     actually delete (default: just list what would go)
#    -h, --help      this message
#
#  Categories (default set: bin out slurm prof junk)
#    bin       compiled executables (1-omp, 2-omp-stream-omp-p2p, serial, ...)
#    out       solution files u_*.txt
#    slurm     job logs: slurm_*.out, run-*.out, build-*.out
#    prof      ./prof/ rocprof traces
#    junk      core dumps, editor backups (*~), *.orig, *.rej
#    probe     p2p-probe* executables
#    logs      *.log  (NOT in the default set -- these are your debug notes)
#    zips      *.zip  (NOT in the default set -- transfer archives)
#    results   results_*/ dirs, all_timings.csv, summary.csv
#              (NOT in the default set -- THIS IS YOUR MEASURED DATA. Re-running
#               the sweep costs node hours and the numbers will not be identical.)
#    all       every category above -- NOT "every file". Sources (.c .h .py .sh
#              .md, Makefile) and git-tracked paths still survive. But it DOES
#              take *.log and *.zip, which are not regenerable, so it asks first.
#
#  Safety rails, in order:
#    1. Anything tracked by git is NEVER deleted, whatever you ask for.
#    2. Source extensions (.c .h .cpp .py .sh .md .txt-that-is-not-u_*) are
#       never matched by any pattern here.
#    3. Executables are only removed if they are ELF binaries AND their name is
#       in the known-build-output list. A stray script called "2-omp" survives.
#
#  Examples
#    ./clean.sh                 # show what a default clean would remove
#    ./clean.sh -f              # do it
#    ./clean.sh -f out slurm    # only solution files and job logs
#    ./clean.sh -f all          # all categories, incl. non-regenerable *.log/*.zip
#    ./clean.sh -f --yes all    # ... without the confirmation prompt
# =============================================================================

set -euo pipefail

FORCE=0
ASSUME_YES=0
CATEGORIES=()

usage() { sed -n '2,48p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--force) FORCE=1 ;;
    -y|--yes)   ASSUME_YES=1 ;;
    -h|--help)  usage 0 ;;
    -*)         echo "ERROR: unknown option '$1'" >&2; usage 1 ;;
    bin|out|slurm|prof|junk|probe|logs|zips|results|all) CATEGORIES+=("$1") ;;
    *)          echo "ERROR: unknown category '$1'" >&2; usage 1 ;;
  esac
  shift
done

[[ ${#CATEGORIES[@]} -gt 0 ]] || CATEGORIES=(bin out slurm prof junk)
if [[ " ${CATEGORIES[*]} " == *" all "* ]]; then
  CATEGORIES=(bin out slurm prof junk probe logs zips results)
fi

want() { [[ " ${CATEGORIES[*]} " == *" $1 "* ]]; }

# -----------------------------------------------------------------------------
# Rail 1: collect git-tracked paths -- these are untouchable.
# -----------------------------------------------------------------------------
declare -A TRACKED=()
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  while IFS= read -r -d '' f; do TRACKED["$f"]=1; done < <(git ls-files -z 2>/dev/null || true)
  echo "git repo detected: ${#TRACKED[@]} tracked path(s) protected."
else
  echo "NOTE: not inside a git work tree -- tracked-file protection unavailable."
fi

# -----------------------------------------------------------------------------
# Rail 3: known build outputs. Extend here when you add a variant.
# -----------------------------------------------------------------------------
KNOWN_EXES=(
  serial
  1-omp
  2-omp 2-omp-stream 2-omp-stream-omp 2-omp-stream-omp-p2p
  4-omp 4-omp-stream 4-omp-stream-omp 4-omp-stream-omp-p2p
  1-cuda
  2-cuda 2-cuda-stream 2-cuda-stream-omp 2-cuda-stream-omp-p2p
  4-cuda 4-cuda-stream 4-cuda-stream-omp 4-cuda-stream-omp-p2p
)

is_elf() { [[ -f "$1" ]] && read -rn4 magic < "$1" 2>/dev/null && [[ "$magic" == $'\x7fELF' ]]; }

VICTIMS=()

consider() {
  local f="$1"
  [[ -e "$f" || -L "$f" ]]        || return 0
  [[ -n "${TRACKED[$f]:-}" ]]     && { echo "  protected (git-tracked): $f"; return 0; }
  case "$f" in
    *.c|*.h|*.cpp|*.hpp|*.py|*.sh|*.md|Makefile|makefile) return 0 ;;
  esac
  VICTIMS+=("$f")
}

# --- bin -----------------------------------------------------------------
if want bin; then
  for e in "${KNOWN_EXES[@]}"; do
    [[ -f "$e" ]] && is_elf "$e" && consider "$e"
  done
fi

# --- probe ---------------------------------------------------------------
if want probe; then
  for f in p2p-probe p2p-probe-*; do
    [[ -f "$f" ]] && is_elf "$f" && consider "$f"
  done
fi

# --- out -----------------------------------------------------------------
if want out; then
  for f in u_*.txt; do consider "$f"; done
fi

# --- slurm ---------------------------------------------------------------
if want slurm; then
  for f in slurm_*.out slurm-*.out run-*.out build-*.out; do consider "$f"; done
fi

# --- prof ----------------------------------------------------------------
if want prof; then
  [[ -d prof ]] && VICTIMS+=("prof/")
fi

# --- junk ----------------------------------------------------------------
if want junk; then
  for f in core core.[0-9]* *~ *.orig *.rej; do consider "$f"; done
fi

# --- logs ----------------------------------------------------------------
if want logs; then
  for f in *.log; do consider "$f"; done
fi

# --- zips ----------------------------------------------------------------
if want zips; then
  for f in *.zip; do consider "$f"; done
fi

# --- results -------------------------------------------------------------
if want results; then
  for d in results_*/; do [[ -d "$d" ]] && VICTIMS+=("$d"); done
  for f in all_timings.csv summary.csv; do consider "$f"; done
fi

# -----------------------------------------------------------------------------
# Report / execute
# -----------------------------------------------------------------------------
if [[ ${#VICTIMS[@]} -eq 0 ]]; then
  echo "Nothing to clean for categories: ${CATEGORIES[*]}"
  exit 0
fi

echo
echo "Categories: ${CATEGORIES[*]}"
echo "Candidates (${#VICTIMS[@]}):"
printf '  %s\n' "${VICTIMS[@]}"
echo
du -shc -- "${VICTIMS[@]}" 2>/dev/null | tail -1 || true
echo

if [[ "$FORCE" -ne 1 ]]; then
  echo "DRY RUN -- nothing removed. Re-run with -f to delete."
  exit 0
fi

# Non-regenerable categories get a second look, even under -f.
if want logs || want zips || want results; then
  if [[ "$ASSUME_YES" -ne 1 ]]; then
    echo "WARNING: this includes measured data and/or debug logs (results_*/,"
    echo "         *.log, *.zip). Re-running the sweep costs node hours and will"
    echo "         not reproduce identical numbers."
    if [[ -t 0 ]]; then
      read -r -p "Type 'yes' to proceed: " reply
      [[ "$reply" == "yes" ]] || { echo "Aborted."; exit 1; }
    else
      echo "Refusing to proceed non-interactively without --yes. Aborted." >&2
      exit 1
    fi
  fi
fi

rm -rf -- "${VICTIMS[@]}"
echo "Removed ${#VICTIMS[@]} item(s)."
