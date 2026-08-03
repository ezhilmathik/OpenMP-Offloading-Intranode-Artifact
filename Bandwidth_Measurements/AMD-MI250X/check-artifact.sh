#!/usr/bin/env bash
#
# Artifact consistency check for the AMD MI250X benchmark directories.
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

DIRS=(
    Synchronous-test-H-to-D-Measurements
    Synchronous-test-D-to-H-Measurements
    Asynchronous-test-H-to-D-Measurements
    Asynchronous-test-D-to-H-Measurements
    P2P-Measurements
)

HOST_NS="940000000 1740000000 2550000000 3350000000 4160000000 4960000000"
P2P_NS="117500000 217500000 318750000 418750000 520000000 1240000000"

echo "############################################################"
echo "# Repository-wide"
echo "############################################################"

stray=$(find . -not -path './.venv/*' \
        \( -name '*~' -o -name '*.o' -o -name '*.bak' -o -name '*.out' \
           -o -name '.DS_Store' \) -print)
if [[ -z "$stray" ]]; then ok "no editor backups, build debris or Slurm .out files"
else bad "stray files:"; echo "$stray" | sed 's/^/          /'; fi

[[ -d .venv ]] && bad ".venv is inside the artifact tree — move it outside"

bins=$(find . -not -path './.venv/*' -type f -exec file --mime-type {} + 2>/dev/null \
       | grep -E 'x-(pie-)?executable|x-sharedlib' | cut -d: -f1)
if [[ -z "$bins" ]]; then ok "no compiled binaries present"
else bad "compiled binaries present:"; echo "$bins" | sed 's/^/          /'; fi

for f in README.md all-plots.sh; do
    [[ -f "$f" ]] && ok "$f present" || bad "$f missing"
done
[[ -x all-plots.sh ]] && ok "all-plots.sh is executable" \
                      || warn "all-plots.sh is not executable (chmod +x)"
grep -q 'set -uo pipefail' all-plots.sh 2>/dev/null \
    && ok "all-plots.sh is the hardened version" \
    || warn "all-plots.sh looks like the original loose version"

[[ -f ../requirements.txt || -f requirements.txt ]] \
    && ok "requirements.txt found" || bad "requirements.txt missing"

echo ""

for d in "${DIRS[@]}"; do
    echo "############################################################"
    echo "# $d"
    echo "############################################################"

    if [[ ! -d "$d" ]]; then bad "directory missing"; echo ""; continue; fi
    pushd "$d" >/dev/null || continue

    # ---------------------------------------------------------------- files
    miss=0
    for f in openmp-1.cc openmp-2.cc hip-1.hip hip-2.hip \
             Makefile build.sh run.sh submit.sh README.md; do
        [[ -f "$f" ]] || { bad "$f MISSING"; miss=1; }
    done
    (( miss == 0 )) && ok "all required files present"

    old=$(ls version-*.hip version-*.cu version-*.cc 2>/dev/null)
    if [[ -z "$old" ]]; then ok "no un-renamed version-* sources"
    else bad "un-renamed sources: $(echo $old | tr '\n' ' ')"; fi

    # v4 sources are deliberately retained
    fut=$(ls openmp-4*.cc hip-4*.hip 2>/dev/null | wc -l)
    (( fut > 0 )) && ok "future-work sources retained ($fut)" \
                  || warn "no v4 sources present"

    # --------------------------------------------------------------- plots
    scripts=$(ls plot_mi250x_*.py 2>/dev/null)
    n=$(echo "$scripts" | grep -c . || true)
    if (( n == 1 )); then ok "one plot script: $scripts"
    else bad "expected 1 plot_mi250x_*.py, found $n"; fi

    if [[ -n "$scripts" ]]; then
        vtp=$(grep -m1 'VERSIONS_TO_PLOT' $scripts | sed 's/.*= *//')
        [[ "$vtp" == *'v4'* ]] && bad "VERSIONS_TO_PLOT includes v4: $vtp" \
                               || ok "VERSIONS_TO_PLOT = $vtp"

        keep=$(grep -m1 'OUTPUT_NAME' $scripts | sed 's/.*=//; s/[" '"'"']//g')
        pdfs=$(ls plots_final/*.pdf 2>/dev/null | wc -l)
        if (( pdfs == 1 )) && [[ -f "plots_final/$keep" ]]; then
            ok "one figure, matching the script: $keep"
        elif (( pdfs == 0 )); then
            bad "no figure in plots_final/ (expected $keep)"
        else
            bad "$pdfs PDFs in plots_final/, expected only $keep:"
            ls plots_final/*.pdf | sed 's/^/          /'
        fi
    fi

    nres=$(ls -d results_*/ 2>/dev/null | wc -l)
    (( nres > 0 )) && ok "$nres results directories" || bad "no results_* data"

    # -------------------------------------------------------------- run.sh
    if grep -qE 'run_bench.*"v4' run.sh; then
        bad "run.sh still runs v4 or v4_4"
    else ok "run.sh runs only v1 and v2"; fi

    nrb=$(grep -cE '^\s*run_bench ' run.sh)
    (( nrb == 4 )) && ok "4 run_bench calls (2 OpenMP, 2 HIP)" \
                   || warn "$nrb run_bench calls, expected 4"

    ns=$(grep -m1 '^Ns=(' run.sh | sed 's/^Ns=(//; s/)$//')
    if [[ "$d" == P2P-Measurements ]]; then
        [[ "$ns" == "$P2P_NS" ]] && ok "Ns matches the P2P sweep" \
                                 || bad "Ns differs from the P2P sweep: $ns"
    else
        [[ "$ns" == "$HOST_NS" ]] && ok "Ns matches the host-transfer sweep" \
                                  || bad "Ns differs from the host sweep: $ns"
    fi

    data_ns=$(cut -d, -f3 results_*/timings.csv 2>/dev/null | grep -E '^[0-9]+$' | sort -un)
    unmatched=""
    for x in $ns; do grep -qx "$x" <<<"$data_ns" || unmatched="$unmatched $x"; done
    [[ -z "$unmatched" ]] && ok "every run.sh size appears in the shipped data" \
                          || bad "sizes with no shipped data:$unmatched"

    # OpenMP runtime handling
    if grep -q 'OMPTARGET_P2P_LIB:-' run.sh; then
        ok "OMPTARGET_P2P_LIB is overridable"
        grep -q 'project_465003145' run.sh \
            && ok "runtime path points at project_465003145" \
            || warn "runtime default path is not project_465003145"
    else
        bad "OMPTARGET_P2P_LIB is hardcoded (no \${VAR:-default})"
    fi
    grep -q 'LD_PRELOAD' run.sh && ok "OpenMP runtime is preloaded" \
                                || bad "no LD_PRELOAD — OpenMP would use the module runtime"

    # Environment settings that affect the measurement
    for v in HSA_ENABLE_SDMA OMP_TARGET_OFFLOAD ROCR_VISIBLE_DEVICES; do
        grep -q "export $v" run.sh && ok "$v is set" || bad "$v is not set"
    done

    # ------------------------------------------------------------ build.sh
    if grep -qE 'version-4[^ ]*-(hip|off)' build.sh; then
        bad "build.sh EXPECTED still lists v4 binaries"
    else ok "build.sh expects only v1/v2 binaries"; fi

    # ------------------------------------------------------------ Makefile
    if [[ "$d" == P2P-Measurements ]]; then
        grep -qE '^OMPTARGET_P2P_LIB[[:space:]]*\?=' Makefile \
            && ok "Makefile defines OMPTARGET_P2P_LIB (overridable)" \
            || bad "Makefile should define OMPTARGET_P2P_LIB with ?="
    fi
    
    badprereq=0
    while read -r src; do
        [[ -f "$src" ]] || { bad "Makefile prerequisite missing: $src"; badprereq=1; }
    done < <(grep -oE '^[a-z0-9-]+: [a-z0-9.-]+\.(hip|cc)$' Makefile | awk '{print $2}')
    (( badprereq == 0 )) && ok "all Makefile prerequisites exist"

    # -------------------------------------------------------------- README
    for f in $(grep -oE '`(openmp|hip|plot)[a-z0-9_.*-]+`' README.md \
               | tr -d '`' | grep -v '\*' | sort -u); do
        [[ -e "$f" ]] || bad "README references missing file: $f"
    done

    grep -q '\[PAPER TITLE\]\|\[Figure N\]\|zenodo.XXXXXXX\|REQUIRED BEFORE SUBMISSION' README.md \
        && warn "README still has placeholders to fill before submission"

    popd >/dev/null
    echo ""
done

echo "############################################################"
if (( problems == 0 )); then echo "${GRN}No problems found.${OFF}"
else echo "${RED}${problems} problem(s) found.${OFF}"; fi
echo "############################################################"
exit $(( problems > 0 ))
ekrishna@uan03:/scratch/project_465003145/Git/Bandwidth/AMD-MI250X> cat check-artifact.sh 
#!/usr/bin/env bash
#
# Artifact consistency check for the AMD MI250X benchmark directories.
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

DIRS=(
    Synchronous-test-H-to-D-Measurements
    Synchronous-test-D-to-H-Measurements
    Asynchronous-test-H-to-D-Measurements
    Asynchronous-test-D-to-H-Measurements
    P2P-Measurements
)

HOST_NS="940000000 1740000000 2550000000 3350000000 4160000000 4960000000"
P2P_NS="117500000 217500000 318750000 418750000 520000000 1240000000"

echo "############################################################"
echo "# Repository-wide"
echo "############################################################"

stray=$(find . -not -path './.venv/*' \
        \( -name '*~' -o -name '*.o' -o -name '*.bak' -o -name '*.out' \
           -o -name '.DS_Store' \) -print)
if [[ -z "$stray" ]]; then ok "no editor backups, build debris or Slurm .out files"
else bad "stray files:"; echo "$stray" | sed 's/^/          /'; fi

[[ -d .venv ]] && bad ".venv is inside the artifact tree — move it outside"

bins=$(find . -not -path './.venv/*' -type f -exec file --mime-type {} + 2>/dev/null \
       | grep -E 'x-(pie-)?executable|x-sharedlib' | cut -d: -f1)
if [[ -z "$bins" ]]; then ok "no compiled binaries present"
else bad "compiled binaries present:"; echo "$bins" | sed 's/^/          /'; fi

for f in README.md all-plots.sh; do
    [[ -f "$f" ]] && ok "$f present" || bad "$f missing"
done
[[ -x all-plots.sh ]] && ok "all-plots.sh is executable" \
                      || warn "all-plots.sh is not executable (chmod +x)"
grep -q 'set -uo pipefail' all-plots.sh 2>/dev/null \
    && ok "all-plots.sh is the hardened version" \
    || warn "all-plots.sh looks like the original loose version"

[[ -f ../requirements.txt || -f requirements.txt ]] \
    && ok "requirements.txt found" || bad "requirements.txt missing"

echo ""

for d in "${DIRS[@]}"; do
    echo "############################################################"
    echo "# $d"
    echo "############################################################"

    if [[ ! -d "$d" ]]; then bad "directory missing"; echo ""; continue; fi
    pushd "$d" >/dev/null || continue

    # ---------------------------------------------------------------- files
    miss=0
    for f in openmp-1.cc openmp-2.cc hip-1.hip hip-2.hip \
             Makefile build.sh run.sh submit.sh README.md; do
        [[ -f "$f" ]] || { bad "$f MISSING"; miss=1; }
    done
    (( miss == 0 )) && ok "all required files present"

    old=$(ls version-*.hip version-*.cu version-*.cc 2>/dev/null)
    if [[ -z "$old" ]]; then ok "no un-renamed version-* sources"
    else bad "un-renamed sources: $(echo $old | tr '\n' ' ')"; fi

    # v4 sources are deliberately retained
    fut=$(ls openmp-4*.cc hip-4*.hip 2>/dev/null | wc -l)
    (( fut > 0 )) && ok "future-work sources retained ($fut)" \
                  || warn "no v4 sources present"

    # --------------------------------------------------------------- plots
    scripts=$(ls plot_mi250x_*.py 2>/dev/null)
    n=$(echo "$scripts" | grep -c . || true)
    if (( n == 1 )); then ok "one plot script: $scripts"
    else bad "expected 1 plot_mi250x_*.py, found $n"; fi

    if [[ -n "$scripts" ]]; then
        vtp=$(grep -m1 'VERSIONS_TO_PLOT' $scripts | sed 's/.*= *//')
        [[ "$vtp" == *'v4'* ]] && bad "VERSIONS_TO_PLOT includes v4: $vtp" \
                               || ok "VERSIONS_TO_PLOT = $vtp"

        keep=$(grep -m1 'OUTPUT_NAME' $scripts | sed 's/.*=//; s/[" '"'"']//g')
        pdfs=$(ls plots_final/*.pdf 2>/dev/null | wc -l)
        if (( pdfs == 1 )) && [[ -f "plots_final/$keep" ]]; then
            ok "one figure, matching the script: $keep"
        elif (( pdfs == 0 )); then
            bad "no figure in plots_final/ (expected $keep)"
        else
            bad "$pdfs PDFs in plots_final/, expected only $keep:"
            ls plots_final/*.pdf | sed 's/^/          /'
        fi
    fi

    nres=$(ls -d results_*/ 2>/dev/null | wc -l)
    (( nres > 0 )) && ok "$nres results directories" || bad "no results_* data"

    # -------------------------------------------------------------- run.sh
    if grep -qE 'run_bench.*"v4' run.sh; then
        bad "run.sh still runs v4 or v4_4"
    else ok "run.sh runs only v1 and v2"; fi

    nrb=$(grep -cE '^\s*run_bench ' run.sh)
    (( nrb == 4 )) && ok "4 run_bench calls (2 OpenMP, 2 HIP)" \
                   || warn "$nrb run_bench calls, expected 4"

    ns=$(grep -m1 '^Ns=(' run.sh | sed 's/^Ns=(//; s/)$//')
    if [[ "$d" == P2P-Measurements ]]; then
        [[ "$ns" == "$P2P_NS" ]] && ok "Ns matches the P2P sweep" \
                                 || bad "Ns differs from the P2P sweep: $ns"
    else
        [[ "$ns" == "$HOST_NS" ]] && ok "Ns matches the host-transfer sweep" \
                                  || bad "Ns differs from the host sweep: $ns"
    fi

    data_ns=$(cut -d, -f3 results_*/timings.csv 2>/dev/null | grep -E '^[0-9]+$' | sort -un)
    unmatched=""
    for x in $ns; do grep -qx "$x" <<<"$data_ns" || unmatched="$unmatched $x"; done
    [[ -z "$unmatched" ]] && ok "every run.sh size appears in the shipped data" \
                          || bad "sizes with no shipped data:$unmatched"

    # OpenMP runtime handling
    if grep -q 'OMPTARGET_P2P_LIB:-' run.sh; then
        ok "OMPTARGET_P2P_LIB is overridable"
        grep -q 'project_465003145' run.sh \
            && ok "runtime path points at project_465003145" \
            || warn "runtime default path is not project_465003145"
    else
        bad "OMPTARGET_P2P_LIB is hardcoded (no \${VAR:-default})"
    fi
    grep -q 'LD_PRELOAD' run.sh && ok "OpenMP runtime is preloaded" \
                                || bad "no LD_PRELOAD — OpenMP would use the module runtime"

    # Environment settings that affect the measurement
    for v in HSA_ENABLE_SDMA OMP_TARGET_OFFLOAD ROCR_VISIBLE_DEVICES; do
        grep -q "export $v" run.sh && ok "$v is set" || bad "$v is not set"
    done

    # ------------------------------------------------------------ build.sh
    if grep -qE 'version-4[^ ]*-(hip|off)' build.sh; then
        bad "build.sh EXPECTED still lists v4 binaries"
    else ok "build.sh expects only v1/v2 binaries"; fi

    # ------------------------------------------------------------ Makefile
    grep -q 'OMPTARGET_P2P_LIB' Makefile \
        && bad "Makefile still defines OMPTARGET_P2P_LIB (dead)" \
        || ok "Makefile has no stale runtime path"

    badprereq=0
    while read -r src; do
        [[ -f "$src" ]] || { bad "Makefile prerequisite missing: $src"; badprereq=1; }
    done < <(grep -oE '^[a-z0-9-]+: [a-z0-9.-]+\.(hip|cc)$' Makefile | awk '{print $2}')
    (( badprereq == 0 )) && ok "all Makefile prerequisites exist"

    # -------------------------------------------------------------- README
    for f in $(grep -oE '`(openmp|hip|plot)[a-z0-9_.*-]+`' README.md \
               | tr -d '`' | grep -v '\*' | sort -u); do
        [[ -e "$f" ]] || bad "README references missing file: $f"
    done

    grep -q '\[PAPER TITLE\]\|\[Figure N\]\|zenodo.XXXXXXX\|REQUIRED BEFORE SUBMISSION' README.md \
        && warn "README still has placeholders to fill before submission"

    popd >/dev/null
    echo ""
done

echo "############################################################"
if (( problems == 0 )); then echo "${GRN}No problems found.${OFF}"
else echo "${RED}${problems} problem(s) found.${OFF}"; fi
echo "############################################################"
exit $(( problems > 0 ))
