#!/bin/bash -l
# =============================================================================
#  clean.sh -- remove build and run artifacts from the SYCL working tree
# =============================================================================
#
#  DRY RUN BY DEFAULT. Nothing is deleted unless you pass -f.
#
#  Usage:  ./clean.sh [-f] [-y] [category ...]
#
#    -f, --force     actually delete (default: just list what would go)
#    -y, --yes       skip the confirmation for non-regenerable categories
#    -h, --help      this message
#
#  Categories (default set: bin out slurm junk)
#    bin       compiled executables (1-sycl, 4-sycl-stream-omp-p2p, serial...)
#    out       solution files u_*.txt
#    slurm     job logs: build-sycl-*.out run-sycl-*.out verify-sycl-*.out
#              verify_*.out probe-*.out slurm-*.out
#    junk      core dumps, editor backups (*~), *.rej
#    probe     the probe_p2p executable
#    presweep  convenience alias for: out slurm junk
#              (what you want between production sweeps)
#
#  NOT in the default set -- each is irreplaceable in a different way:
#    logs      *.log  -- includes dpct-conversion.log, the record of what the
#              migration did. Regenerating it means re-running DPCT.
#    zips      *.zip  -- transfer archives
#    backups   *.dpct.orig *.hardguard *.prename -- the pre-patch sources.
#              These are your only local record of what DPCT emitted before
#              the P2P shim, the guard relaxation and the filename rename.
#    results   results_*/ all_timings.csv summary.csv -- MEASURED DATA. Re-
#              running the sweep costs node hours and will not reproduce
#              identical numbers.
#    all       every category above. Asks first.
#
#  ** RESULTS AND MIXED BINARIES **
#  aggregate.sh merges EVERY results_*/ directory it finds. If you change a
#  source and re-run the sweep, old and new results are silently pooled into
#  one median. Before a sweep that follows a code change, either archive the
#  old data:
#      mkdir -p archive/pre-p2p-fix && mv results_* archive/pre-p2p-fix/
#  or delete it with `./clean.sh -f results`. Archiving is preferred: a median
#  over two different binaries is not a number you can correct later, and you
#  cannot tell it happened by looking at summary.csv.
#
#  Safety rails, in order:
#    1. Anything tracked by git is NEVER deleted, whatever you ask for.
#    2. Source extensions (.c .h .cpp .hpp .py .sh .md, Makefile) are never
#       matched. Note this does NOT cover .cpp.dpct.orig -- hence `backups`
#       being its own opt-in category rather than part of `junk`, which is
#       where an *.orig pattern would normally live.
#    3. dpct_output/ is never touched by any category.
#    4. Executables are only removed if they are ELF binaries AND their name
#       is in the known-build-output list. A stray script called "2-sycl"
#       survives.
#
#  Examples
#    ./clean.sh                      # show what a default clean would remove
#    ./clean.sh -f                   # do it
#    ./clean.sh -f presweep          # clear outputs and logs, keep binaries
#    ./clean.sh -f out slurm         # only solution files and job logs
#    ./clean.sh -f -y results        # drop measured data, no prompt
# =============================================================================

set -euo pipefail

FORCE=0
ASSUME_YES=0
CATEGORIES=()

usage() { sed -n '2,72p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--force) FORCE=1 ;;
    -y|--yes)   ASSUME_YES=1 ;;
    -h|--help)  usage 0 ;;
    -*)         echo "ERROR: unknown option '$1'" >&2; usage 1 ;;
    bin|out|slurm|junk|probe|logs|zips|backups|results|presweep|all)
                CATEGORIES+=("$1") ;;
    *)          echo "ERROR: unknown category '$1'" >&2; usage 1 ;;
  esac
  shift
done

[[ ${#CATEGORIES[@]} -gt 0 ]] || CATEGORIES=(bin out slurm junk)

if [[ " ${CATEGORIES[*]} " == *" presweep "* ]]; then
  CATEGORIES=(out slurm junk)
fi
if [[ " ${CATEGORIES[*]} " == *" all "* ]]; then
  CATEGORIES=(bin out slurm junk probe logs zips backups results)
fi

want() { [[ " ${CATEGORIES[*]} " == *" $1 "* ]]; }

# -----------------------------------------------------------------------------
# Rail 1: git-tracked paths are untouchable.
# -----------------------------------------------------------------------------
declare -A TRACKED=()
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  while IFS= read -r -d '' f; do TRACKED["$f"]=1; done < <(git ls-files -z 2>/dev/null || true)
  echo "git repo detected: ${#TRACKED[@]} tracked path(s) protected."
else
  echo "NOTE: not inside a git work tree -- tracked-file protection unavailable."
fi

# -----------------------------------------------------------------------------
# Rail 4: known build outputs. Extend here when you add a variant.
# -----------------------------------------------------------------------------
KNOWN_EXES=(
  serial
  1-sycl
  2-sycl 2-sycl-stream 2-sycl-stream-omp 2-sycl-stream-omp-p2p
  4-sycl 4-sycl-stream 4-sycl-stream-omp 4-sycl-stream-omp-p2p
)

is_elf() {
  [[ -f "$1" ]] || return 1
  [[ "$(head -c4 -- "$1" 2>/dev/null)" == $'\177ELF' ]]
}

VICTIMS=()

consider() {
  local f="$1"
  [[ -e "$f" || -L "$f" ]]    || return 0
  [[ "$f" == dpct_output* ]]  && return 0          # rail 3
  [[ -n "${TRACKED[$f]:-}" ]] && { echo "  protected (git-tracked): $f"; return 0; }
  case "$f" in
    *.c|*.h|*.cpp|*.hpp|*.py|*.sh|*.md|Makefile|makefile) return 0 ;;
  esac
  VICTIMS+=("$f")
}

# --- bin ---------------------------------------------------------------------
if want bin; then
  for e in "${KNOWN_EXES[@]}"; do
    [[ -f "$e" ]] && is_elf "$e" && consider "$e"
  done
fi

# --- probe -------------------------------------------------------------------
if want probe; then
  for f in probe_p2p p2p-probe p2p-probe-*; do
    [[ -f "$f" ]] && is_elf "$f" && consider "$f"
  done
fi

# --- out ---------------------------------------------------------------------
if want out; then
  for f in u_*.txt; do consider "$f"; done
fi

# --- slurm -------------------------------------------------------------------
if want slurm; then
  for f in build-sycl-*.out run-sycl-*.out verify-sycl-*.out verify_*.out \
           probe-*.out smoke-*.out slurm-*.out slurm_*.out; do
    consider "$f"
  done
fi

# --- junk --------------------------------------------------------------------
# Deliberately no *.orig here: *.dpct.orig are the pre-migration sources.
if want junk; then
  for f in core core.[0-9]* *~ *.rej; do consider "$f"; done
fi

# --- logs --------------------------------------------------------------------
if want logs; then
  for f in *.log; do consider "$f"; done
fi

# --- zips --------------------------------------------------------------------
if want zips; then
  for f in *.zip; do consider "$f"; done
fi

# --- backups -----------------------------------------------------------------
if want backups; then
  for f in *.dpct.orig *.hardguard *.prename; do consider "$f"; done
fi

# --- results -----------------------------------------------------------------
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

if want logs || want zips || want backups || want results; then
  if [[ "$ASSUME_YES" -ne 1 ]]; then
    echo "WARNING: this includes non-regenerable material (results_*/, *.log,"
    echo "         *.zip, or the pre-patch source backups). Re-running the"
    echo "         sweep costs node hours and will not reproduce identical"
    echo "         numbers; the .dpct.orig files are your record of what the"
    echo "         migration emitted before it was patched by hand."
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
