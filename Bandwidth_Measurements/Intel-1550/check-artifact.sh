#!/usr/bin/env bash
#
# Artifact consistency check for the Intel Data Center GPU Max 1550 directories.
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
           -o -name '.DS_Store' -o -name '*.dp.cpp' \) -print)
[[ -z "$stray" ]] && ok "no editor backups, build debris, .out or .dp.cpp files" \
                  || { bad "stray files:"; echo "$stray" | sed 's/^/          /'; }

[[ -d .venv ]] && bad ".venv is inside the artifact tree — move it outside"

bins=$(find . -not -path './.venv/*' -type f -exec file --mime-type {} + 2>/dev/null \
       | grep -E 'x-(pie-)?executable|x-sharedlib' | cut -d: -f1)
[[ -z "$bins" ]] && ok "no compiled binaries present" \
                 || { bad "compiled binaries present:"; echo "$bins" | sed 's/^/          /'; }

# Wrong-vendor contamination, ignoring explanatory comments
contam=$(grep -rn 'rocm\|ROCR_\|amdclang\|gfx90a\|HSA_ENABLE\|hipcc\|nvcc\|sm_[0-9]' \
         --include='*.sh' --include='Makefile' --exclude='check-artifact.sh' . 2>/dev/null | grep -vE ':[0-9]+: *#')
[[ -z "$contam" ]] && ok "no AMD/NVIDIA settings in executable lines" \
                   || { bad "wrong-vendor settings:"; echo "$contam" | sed 's/^/          /'; }

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

    [[ -d "$d" ]] || { bad "directory missing"; echo ""; continue; }
    pushd "$d" >/dev/null || continue

    # ---------------------------------------------------------------- files
    miss=0
    for f in openmp-1.cc openmp-2.cc sycl-1.cpp sycl-2.cpp \
             Makefile build.sh run.sh submit.sh README.md; do
        [[ -f "$f" ]] || { bad "$f MISSING"; miss=1; }
    done
    (( miss == 0 )) && ok "all required files present"

    old=$(ls version-*.cpp version-*.hip version-*.cu 2>/dev/null)
    [[ -z "$old" ]] && ok "no un-renamed version-* sources" \
                    || bad "un-renamed sources: $(echo $old | tr '\n' ' ')"

    fut=$(ls openmp-4*.cc sycl-4*.cpp 2>/dev/null | wc -l)
    (( fut > 0 )) && ok "future-work sources retained ($fut)" \
                  || warn "no v4 sources present"

    # --------------------------------------------------------------- plots
    scripts=$(ls plot_pvc_*.py 2>/dev/null)
    n=$(echo "$scripts" | grep -c . || true)
    (( n == 1 )) && ok "one plot script: $scripts" \
                 || bad "expected 1 plot_pvc_*.py, found $n"

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

    # Results directories must be hierarchy-labelled
    badname=$(ls -d results_*/ 2>/dev/null | grep -vE '^results_(COMPOSITE|FLAT)_')
    [[ -z "$badname" ]] && ok "results directories are hierarchy-labelled" \
                        || bad "results directories missing hierarchy label: $badname"

    # env.txt provenance
    noenv=$(for r in results_*/; do [[ -f "$r/env.txt" ]] || echo "$r"; done)
    [[ -z "$noenv" ]] && ok "every results directory has env.txt" \
                      || bad "missing env.txt in: $(echo $noenv | tr '\n' ' ')"

    # CSV schema
    hdr=$(head -1 results_*/timings.csv 2>/dev/null | grep -m1 'framework')
    [[ "$hdr" == "framework,version,N,elapsed_sec,bandwidth_gbs,status,hierarchy" ]] \
        && ok "CSV has the 7-column schema with hierarchy" \
        || bad "unexpected CSV header: $hdr"

    # -------------------------------------------------------------- run.sh
    grep -qE 'run_bench.*"v4' run.sh && bad "run.sh still runs v4" \
                                     || ok "run.sh runs only v1 and v2"

    nrb=$(grep -cE '^ *run_bench "' run.sh)
    (( nrb == 4 )) && ok "4 run_bench calls (2 OpenMP, 2 SYCL)" \
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

    # Level Zero / OpenMP environment
    for v in OMP_TARGET_OFFLOAD ZE_FLAT_DEVICE_HIERARCHY ZE_AFFINITY_MASK \
             ZE_ENABLE_PCI_ID_DEVICE_ORDER ONEAPI_DEVICE_SELECTOR SYCL_CACHE_PERSISTENT; do
        grep -q "export $v" run.sh && ok "$v is set" || bad "$v is not set"
    done

    grep -q 'HIER="${1:-COMPOSITE}"' run.sh \
        && ok "hierarchy selectable via argument, defaulting to COMPOSITE" \
        || bad "run.sh does not take a hierarchy argument"

    grep -q 'SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE' run.sh \
        && grep -qE '^ *# *export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE' run.sh \
        && ok "copy-engine control documented and left unset" \
        || warn "copy-engine control not documented, or actually set"

    # P2P-only expectations
    if [[ "$d" == P2P-Measurements ]]; then
        grep -q 'LIBOMPTARGET_DEVICES' run.sh \
            && ok "LIBOMPTARGET_DEVICES set (device-model agreement)" \
            || bad "LIBOMPTARGET_DEVICES not set — OpenMP and SYCL may disagree"
        grep -q 'Preflight' run.sh && ok "preflight device check present" \
                                   || bad "preflight device check missing"
        grep -q 'intel-toolkit/' run.sh && ok "toolkit version pinned" \
                                        || warn "toolkit module is not pinned"
        noprobe=$(for r in results_*/; do [[ -f "$r/probe.txt" ]] || echo "$r"; done)
        [[ -z "$noprobe" ]] && ok "every results directory has probe.txt" \
                            || warn "missing probe.txt in: $(echo $noprobe | tr '\n' ' ')"
        [[ -f topology-test.cc ]] && ok "topology-test.cc present" \
                                  || warn "topology-test.cc missing"
    fi

    # ------------------------------------------------------------ build.sh
    grep -qE 'version-4[^ ]*-(sycl|off)' build.sh \
        && bad "build.sh EXPECTED still lists v4 binaries" \
        || ok "build.sh expects only v1/v2 binaries"

    # ------------------------------------------------------------ Makefile
    badprereq=0
    while read -r src; do
        [[ -f "$src" ]] || { bad "Makefile prerequisite missing: $src"; badprereq=1; }
    done < <(grep -oE '^[a-z0-9-]+: [a-z0-9.-]+\.(cpp|cc)$' Makefile | awk '{print $2}')
    (( badprereq == 0 )) && ok "all Makefile prerequisites exist"

    # -------------------------------------------------------------- README
    for f in $(grep -oE '`(openmp|sycl|plot|topology)[a-z0-9_.-]+\.(cc|cpp|py)`' README.md \
               | tr -d '`' | grep -v '\*' | sort -u); do
        [[ -e "$f" ]] || bad "README references missing file: $f"
    done

    grep -q '\[PAPER TITLE\]\|\[Figure N\]\|zenodo.XXXXXXX' README.md \
        && warn "README still has placeholders to fill before submission"

    popd >/dev/null
    echo ""
done

echo "############################################################"
(( problems == 0 )) && echo "${GRN}No problems found.${OFF}" \
                    || echo "${RED}${problems} problem(s) found.${OFF}"
echo "############################################################"
exit $(( problems > 0 ))
