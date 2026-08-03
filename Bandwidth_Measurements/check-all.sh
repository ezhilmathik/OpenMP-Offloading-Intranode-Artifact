#!/usr/bin/env bash
#
# Top-level artifact check for the whole benchmark tree.
#
# Checks what lives above the vendor directories, verifies the three vendors
# are structurally parallel, then runs each vendor's own check-artifact.sh.
#
# Read-only: reports problems, changes nothing.
#
# Usage:  ./check-all.sh
#
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; BLD=$'\033[1m'; OFF=$'\033[0m'
problems=0

ok()   { echo "  ${GRN}ok${OFF}    $*"; }
bad()  { echo "  ${RED}FAIL${OFF}  $*"; problems=$((problems+1)); }
warn() { echo "  ${YEL}warn${OFF}  $*"; }
hdr()  { echo ""; echo "${BLD}############################################################${OFF}"
         echo "${BLD}# $*${OFF}"
         echo "${BLD}############################################################${OFF}"; }

VENDORS=(NVIDIA-H100 AMD-MI250X Intel-1550)
SUBDIRS=(
    Synchronous-test-H-to-D-Measurements
    Synchronous-test-D-to-H-Measurements
    Asynchronous-test-H-to-D-Measurements
    Asynchronous-test-D-to-H-Measurements
    P2P-Measurements
)

# ===========================================================================
hdr "Root-level files"
# ===========================================================================

for f in README.md LICENSE requirements.txt clean.sh; do
    [[ -f "$f" ]] && ok "$f present" || bad "$f MISSING"
done
for f in NOTICE CITATION.cff .gitignore; do
    [[ -f "$f" ]] && ok "$f present" || warn "$f missing (recommended)"
done

[[ -x clean.sh ]] && ok "clean.sh is executable" || warn "clean.sh is not executable"

if [[ -f requirements.txt ]]; then
    echo "        requirements.txt: $(tr '\n' ' ' < requirements.txt)"
fi

# Anything at root that should not be there
extra=$(ls -A | grep -vE "^(NVIDIA-H100|AMD-MI250X|Intel-1550|README\.md|LICENSE|NOTICE|CITATION\.cff|\.gitignore|requirements\.txt|clean\.sh|check-all\.sh)$")
[[ -z "$extra" ]] && ok "no unexpected files at root" \
                  || { bad "unexpected at root:"; echo "$extra" | sed 's/^/          /'; }

# ===========================================================================
hdr "Tree-wide hygiene"
# ===========================================================================

venvs=$(find . -type d \( -name '.venv' -o -name 'venv' -o -name '__pycache__' \) -print)
[[ -z "$venvs" ]] && ok "no virtualenvs or __pycache__ anywhere" \
                  || { bad "$(echo "$venvs" | wc -l) virtualenv/__pycache__ directories:"
                       echo "$venvs" | head -5 | sed 's/^/          /'; }

debris=$(find . \( -name '*~' -o -name '*.o' -o -name '*.bak' -o -name '*.out' \
                -o -name '.DS_Store' -o -name '*.pyc' -o -name 'core.*' \) -print)
[[ -z "$debris" ]] && ok "no editor backups, object files, Slurm .out or .pyc" \
                   || { bad "debris:"; echo "$debris" | head -8 | sed 's/^/          /'; }

bins=$(find . -type f -exec file --mime-type {} + 2>/dev/null \
       | grep -E 'x-(pie-)?executable|x-sharedlib' | cut -d: -f1)
[[ -z "$bins" ]] && ok "no compiled binaries" \
                 || { bad "compiled binaries:"; echo "$bins" | head -5 | sed 's/^/          /'; }

stale=$(ls -d to-compress old-timings 2>/dev/null)
[[ -z "$stale" ]] && ok "no stale snapshot directories" \
                  || bad "stale directories present: $stale"

echo "        total size: $(du -sh . | cut -f1)"

# ===========================================================================
hdr "Cross-vendor structure"
# ===========================================================================

for v in "${VENDORS[@]}"; do
    [[ -d "$v" ]] || { bad "$v MISSING"; continue; }
    miss=0
    for f in README.md all-plots.sh check-artifact.sh; do
        [[ -f "$v/$f" ]] || { bad "$v/$f missing"; miss=1; }
    done
    for s in "${SUBDIRS[@]}"; do
        [[ -d "$v/$s" ]] || { bad "$v/$s missing"; miss=1; }
    done
    (( miss == 0 )) && ok "$v: README, all-plots.sh, check-artifact.sh and 5 benchmark directories"
done

# One plot script and one matching figure per benchmark directory
echo ""
printf '  %-46s %-26s %s\n' "DIRECTORY" "SCRIPT" "FIGURE"
figs=0; figbad=0
for v in "${VENDORS[@]}"; do
  for s in "${SUBDIRS[@]}"; do
    d="$v/$s"; [[ -d "$d" ]] || continue
    scr=$(ls "$d"/plot_*.py 2>/dev/null | head -1)
    want=$(grep -m1 'OUTPUT_NAME' "$scr" 2>/dev/null | sed 's/.*=//; s/[" '"'"']//g')
    npdf=$(ls "$d"/plots_final/*.pdf 2>/dev/null | wc -l)
    have=$(ls "$d"/plots_final/ 2>/dev/null | tr '\n' ' ')
    printf '  %-46s %-26s %s\n' "$d" "$(basename "${scr:-NONE}")" "$have"
    if [[ -z "$scr" ]]; then figbad=$((figbad+1))
    elif (( npdf != 1 )) || [[ ! -f "$d/plots_final/$want" ]]; then figbad=$((figbad+1))
    else figs=$((figs+1)); fi
  done
done
echo ""
(( figbad == 0 )) && ok "$figs directories each have one plot script and one matching figure" \
                  || bad "$figbad directories have a script/figure mismatch"

# Every directory has a README with the standard 11 sections
badsec=0
for v in "${VENDORS[@]}"; do
  for s in "${SUBDIRS[@]}"; do
    f="$v/$s/README.md"
    [[ -f "$f" ]] || { badsec=$((badsec+1)); continue; }
    n=$(grep -cE '^## [0-9]+\. ' "$f")
    (( n == 11 )) || { bad "$f has $n numbered sections, expected 11"; badsec=$((badsec+1)); }
  done
done
(( badsec == 0 )) && ok "all 15 directory READMEs have the standard 11 sections"

# Relative paths referenced from the READMEs actually resolve
badpath=0
for v in "${VENDORS[@]}"; do
  for s in "${SUBDIRS[@]}"; do
    d="$v/$s"; f="$d/README.md"; [[ -f "$f" ]] || continue
    grep -q '\.\./\.\./requirements\.txt' "$f" && [[ ! -f "$d/../../requirements.txt" ]] \
        && { bad "$f: ../../requirements.txt does not resolve"; badpath=1; }
    grep -q '\.\./\.\./clean\.sh' "$f" && [[ ! -f "$d/../../clean.sh" ]] \
        && { bad "$f: ../../clean.sh does not resolve"; badpath=1; }
    grep -q '\.\./all-plots\.sh' "$f" && [[ ! -f "$d/../all-plots.sh" ]] \
        && { bad "$f: ../all-plots.sh does not resolve"; badpath=1; }
  done
done
(( badpath == 0 )) && ok "relative paths in READMEs resolve"

# ===========================================================================
hdr "Outstanding placeholders"
# ===========================================================================

tot=0
for f in README.md */README.md */*/README.md; do
    [[ -f "$f" ]] || continue
    n=$(grep -cE '\[PAPER TITLE\]|\[Figure N\]|zenodo\.XXXXXXX|REQUIRED BEFORE SUBMISSION|\[MIT / Apache' "$f")
    (( n > 0 )) && { tot=$((tot+n)); printf '  %-58s %s\n' "$f" "$n"; }
done
if (( tot == 0 )); then ok "no placeholders remaining"
else warn "$tot placeholder markers across the tree — expected until submission"; fi

# ===========================================================================
hdr "Per-vendor checks"
# ===========================================================================

for v in "${VENDORS[@]}"; do
    [[ -x "$v/check-artifact.sh" ]] || { warn "$v/check-artifact.sh not executable, skipping"; continue; }
    out=$( cd "$v" && ./check-artifact.sh 2>&1 )
    n=$(grep -c 'FAIL' <<< "$out")
    w=$(grep -c 'warn' <<< "$out")
    if (( n == 0 )); then ok "$v: clean ($w warnings)"
    else bad "$v: $n failures"; grep 'FAIL' <<< "$out" | sed 's/^/          /'; fi
done

# ===========================================================================
echo ""
echo "############################################################"
if (( problems == 0 )); then
    echo "${GRN}No problems found.${OFF}"
else
    echo "${RED}${problems} problem(s) found.${OFF}"
fi
echo "############################################################"
exit $(( problems > 0 ))
