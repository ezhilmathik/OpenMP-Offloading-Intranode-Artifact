#!/usr/bin/env bash
#
# Artifact consistency check for the NVIDIA H100 benchmark directories.
# Read-only: reports problems, changes nothing.
#
# Usage:  ./check-artifact.sh
#
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; OFF=$'\033[0m'
problems=0

ok()   { echo "  ${GRN}ok${OFF}    $*"; }
bad()  { echo "  ${RED}FAIL${OFF}  $*"; problems=$((problems+1)); }
warn() { echo "  ${YEL}warn${OFF}  $*"; }

# Read a shell flag assignment, ignoring any trailing comment.
flagval() {  # flagval <file> <NAME>
    grep -m1 "^$2=" "$1" 2>/dev/null | sed 's/#.*//' | cut -d= -f2 | tr -d '[:space:]'
}

DIRS=(
    Synchronous-test-H-to-D-Measurements
    Synchronous-test-D-to-H-Measurements
    Asynchronous-test-H-to-D-Measurements
    Asynchronous-test-D-to-H-Measurements
    P2P-Measurements
)

SYNC_NS="940000000 1740000000 2550000000 3350000000 4160000000 4960000000"

echo "############################################################"
echo "# Repository-wide"
echo "############################################################"

stray=$(find . -not -path './.venv/*' \
        \( -name '*~' -o -name '*.o' -o -name '*.bak' -o -name '.DS_Store' \) -print)
if [[ -z "$stray" ]]; then ok "no editor backups or build debris"
else bad "stray files:"; echo "$stray" | sed 's/^/          /'; fi

if [[ -d .venv ]]; then
    bad ".venv is inside the artifact tree — move it outside before archiving"
fi

bins=$(find . -not -path './.venv/*' -type f -exec file --mime-type {} + 2>/dev/null \
       | grep -E 'x-(pie-)?executable|x-sharedlib' | cut -d: -f1)
if [[ -z "$bins" ]]; then ok "no compiled binaries present"
else bad "compiled binaries present:"; echo "$bins" | sed 's/^/          /'; fi

for f in README.md all-plots.sh; do
    [[ -f "$f" ]] && ok "$f present" || bad "$f missing"
done
[[ -x all-plots.sh ]] && ok "all-plots.sh is executable" \
                      || warn "all-plots.sh is not executable (chmod +x)"

if [[ -f requirements.txt || -f ../requirements.txt ]]; then
    ok "requirements.txt found"
else
    bad "requirements.txt missing"
fi

echo ""

for d in "${DIRS[@]}"; do
    echo "############################################################"
    echo "# $d"
    echo "############################################################"

    if [[ ! -d "$d" ]]; then bad "directory missing"; echo ""; continue; fi
    pushd "$d" >/dev/null || continue

    # ---------------------------------------------------------------- files
    missing_files=0
    for f in openmp-1.cc openmp-2.cc cuda-1.cu cuda-2.cu \
             Makefile build.sh run.sh submit.sh README.md; do
        [[ -f "$f" ]] || { bad "$f MISSING"; missing_files=1; }
    done
    (( missing_files == 0 )) && ok "all required files present"

    old=$(ls version-*.cu version-*.cc 2>/dev/null)
    if [[ -z "$old" ]]; then ok "no un-renamed version-*.cu / version-*.cc"
    else bad "un-renamed sources: $(echo $old | tr '\n' ' ')"; fi

    acc_srcs=$(ls openacc-*.cc 2>/dev/null | wc -l)
    (( acc_srcs > 0 )) && ok "openacc sources retained ($acc_srcs)" \
                       || warn "no openacc-*.cc present"

    # ------------------------------------------------------------- plotting
    scripts=$(ls plot_*.py 2>/dev/null)
    n_scripts=$(echo "$scripts" | grep -c . || true)
    if (( n_scripts == 1 )); then ok "one plot script: $scripts"
    else bad "expected 1 plot_*.py, found $n_scripts"; fi

    if [[ -n "$scripts" ]]; then
        vtp=$(grep -m1 'VERSIONS_TO_PLOT' $scripts | sed 's/.*= *//')
        if [[ "$vtp" == *'v4'* ]]; then bad "VERSIONS_TO_PLOT includes v4: $vtp"
        else ok "VERSIONS_TO_PLOT = $vtp"; fi
    fi

    n_pdf=$(ls plots_final/*.pdf 2>/dev/null | wc -l)
    if (( n_pdf == 1 )); then ok "one figure: $(ls plots_final/*.pdf)"
    elif (( n_pdf == 0 )); then bad "no figure in plots_final/"
    else bad "$n_pdf PDFs in plots_final/ (stale output?):"
         ls plots_final/*.pdf | sed 's/^/          /'; fi

    n_res=$(ls -d results_*/ 2>/dev/null | wc -l)
    (( n_res > 0 )) && ok "$n_res results directories" || bad "no results_* data"

    # --------------------------------------------------------------- run.sh
    run_acc=$(flagval run.sh RUN_ACC)
    run_fut=$(flagval run.sh RUN_FUTURE)

    if [[ -n "$run_acc" ]]; then
        [[ "$run_acc" == 0 ]] && ok "run.sh: OpenACC gated off (RUN_ACC=0)" \
                              || bad "run.sh: RUN_ACC=$run_acc"
    elif grep -qE 'run_bench +"acc"' run.sh; then
        bad "run.sh runs OpenACC and has no RUN_ACC flag"
    else
        ok "run.sh does not run OpenACC"
    fi

    if [[ -n "$run_fut" ]]; then
        [[ "$run_fut" == 0 ]] && ok "run.sh: v4 gated off (RUN_FUTURE=0)" \
                              || bad "run.sh: RUN_FUTURE=$run_fut"
    elif grep -qE 'run_bench.*"v4' run.sh; then
        bad "run.sh runs v4 and has no RUN_FUTURE flag"
    else
        ok "run.sh does not run v4"
    fi

    ns=$(grep -m1 '^Ns=(' run.sh | sed 's/^Ns=(//; s/)$//')
    if [[ "$d" == P2P-Measurements ]]; then
        ok "Ns (P2P): $ns"
    elif [[ "$ns" == "$SYNC_NS" ]]; then
        ok "Ns matches the standard sweep"
    else
        bad "Ns differs from the standard sweep:"
        echo "          run.sh: $ns"
        echo "          expect: $SYNC_NS"
    fi

    # Every size in run.sh must appear in the shipped data. The data may
    # contain extra sizes (e.g. the P2P 4-to-4 sweep), which is fine.
    data_ns=$(cut -d, -f3 results_*/timings.csv 2>/dev/null \
              | grep -E '^[0-9]+$' | sort -un)
    unmatched=""
    for n in $ns; do
        grep -qx "$n" <<<"$data_ns" || unmatched="$unmatched $n"
    done
    if [[ -z "$unmatched" ]]; then ok "every run.sh size appears in the shipped data"
    else bad "sizes in run.sh with no shipped data:$unmatched"; fi

    # ------------------------------------------------------------- build.sh
    bld_acc=$(flagval build.sh BUILD_ACC)
    bld_fut=$(flagval build.sh BUILD_FUTURE)

    if [[ -n "$bld_acc" ]]; then
        [[ "$bld_acc" == 0 ]] && ok "build.sh: OpenACC gated off (BUILD_ACC=0)" \
                              || bad "build.sh: BUILD_ACC=$bld_acc"
    elif grep -qE '^\s*make acc' build.sh; then
        bad "build.sh runs 'make acc' and has no BUILD_ACC flag"
    else
        ok "build.sh does not build OpenACC"
    fi

    if [[ -n "$bld_fut" ]]; then
        [[ "$bld_fut" == 0 ]] && ok "build.sh: v4 gated off (BUILD_FUTURE=0)" \
                              || bad "build.sh: BUILD_FUTURE=$bld_fut"
    elif grep -qE 'version-4[^ ]*-(off|cuda|acc)' build.sh; then
        bad "build.sh expects v4 binaries and has no BUILD_FUTURE flag"
    else
        ok "build.sh expects only v1/v2 binaries"
    fi

    # build.sh and run.sh flags should agree
    if [[ -n "$bld_acc" && -n "$run_acc" && "$bld_acc" != "$run_acc" ]]; then
        bad "BUILD_ACC=$bld_acc but RUN_ACC=$run_acc — these should match"
    fi
    if [[ -n "$bld_fut" && -n "$run_fut" && "$bld_fut" != "$run_fut" ]]; then
        bad "BUILD_FUTURE=$bld_fut but RUN_FUTURE=$run_fut — these should match"
    fi

    # -------------------------------------------------------------- README
    for f in $(grep -oE '`(openmp|cuda|openacc|plot)[a-z0-9_.*-]+`' README.md \
               | tr -d '`' | grep -v '\*' | sort -u); do
        [[ -e "$f" ]] || bad "README references missing file: $f"
    done

    if grep -q '\[PAPER TITLE\]\|\[Figure N\]\|zenodo.XXXXXXX\|\[N\]' README.md; then
        warn "README still has placeholders to fill before submission"
    fi

    popd >/dev/null
    echo ""
done

echo "############################################################"
if (( problems == 0 )); then
    echo "${GRN}No problems found.${OFF}"
else
    echo "${RED}${problems} problem(s) found.${OFF}"
fi
echo "############################################################"
exit $(( problems > 0 ))
