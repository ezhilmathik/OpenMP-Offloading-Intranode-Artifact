#!/bin/bash -l
#SBATCH --job-name=heat-sycl-verify
#SBATCH --account=pn67qe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --output=verify-sycl-%j.out
#
# Correctness only -- never mixed into the timing sweep. Every binary here runs
# with WRITE_TXT=1, so each writes its own u_*.txt, and each is diffed against
# the serial CPU reference. This is what the old sycl-batch.sh MODE=test did,
# minus the build ladder: the flags are fixed by the Makefile now.
#
#   sbatch verify.sh              # N=128, dt from the same formula as submit.sh
#   sbatch --export=ALL,N=256 verify.sh
#
# TOL is a relative L-inf tolerance. -ffp-contract=off + -fno-fast-math on both
# sides makes agreement tight, but SYCL reduction/work-group ordering still
# differs from a serial loop, so exact equality is not the bar.

set -uo pipefail

module load slurm_setup
module load intel-toolkit
module load intel-dpct 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$PWD}"

N="${N:-128}"
STEPS="${STEPS:-500}"
KAPPA_MAX="${KAPPA_MAX:-0.95}"
DT="${DT:-$(awk -v N="$N" -v k="$KAPPA_MAX" -v s="$STEPS" \
      'BEGIN { h = 1.0/(N-1); dt = h*h/(12.0*k); printf "%.7g", s*dt }')}"
TOL="${TOL:-1e-10}"

export WRITE_TXT=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export SYCL_CACHE_PERSISTENT=1

echo "Verification run: N=${N} DT=${DT} TOL=${TOL}"
echo ""

make serial-ref || exit 1
make all gpu-baseline || exit 1

echo ""
echo "Serial reference..."
OMP_NUM_THREADS=1 ./serial "$N" "$DT" > verify_serial.out 2>&1 || {
    echo "ERROR: serial reference failed:"; cat verify_serial.out; exit 1; }
REF=$(ls -1 u_serial*.txt 2>/dev/null | head -1)
[[ -n "$REF" ]] || { echo "ERROR: serial run wrote no u_serial*.txt"; exit 1; }
echo "  reference: $REF ($(wc -l < "$REF") lines)"

mask_for() { local n="$1" v=() i; for ((i=0;i<n;i++)); do v+=("$i"); done; (IFS=,; echo "${v[*]}"); }

FAIL=0
check() {
    local binary="$1" gpus="$2" threads="$3"
    [[ -x "./$binary" ]] || { echo "  SKIP (missing): $binary"; return; }
    rm -f u_sycl_*.txt
    env "ZE_AFFINITY_MASK=$(mask_for "$gpus")" "OMP_NUM_THREADS=${threads}" \
        ./"$binary" "$N" "$DT" > "verify_${binary}.out" 2>&1
    local rc=$?
    local out; out=$(ls -1t u_sycl_*.txt 2>/dev/null | head -1)
    if [[ $rc -ne 0 || -z "$out" ]]; then
        echo "  FAIL  $binary (exit=$rc, output='${out:-none}') -- see verify_${binary}.out"
        FAIL=$((FAIL+1)); return
    fi
    # Layout-independent: compare every whitespace-separated numeric token in
    # file order, so this does not depend on how write_u_values() lays the
    # grid out (one value per line, a row per line, index+value pairs...).
    awk -v tol="$TOL" -v bin="$binary" '
        NR == FNR { for (i = 1; i <= NF; i++) { na++; a[na] = $i + 0 } next }
        { for (i = 1; i <= NF; i++) {
              nb++
              if (nb > na) next
              d = a[nb] - ($i + 0); if (d < 0) d = -d
              s = (a[nb] < 0 ? -a[nb] : a[nb]); if (s < 1) s = 1
              r = d / s
              if (r > worst) { worst = r; where = nb } } }
        END {
            if (na != nb) { printf "  FAIL  %s: %d vs %d values\n", bin, na, nb; exit 1 }
            if (worst > tol) { printf "  FAIL  %s: max rel diff %.3e at value %d (tol %g)\n", bin, worst, where, tol; exit 1 }
            printf "  OK    %s: max rel diff %.3e over %d values\n", bin, worst, na
        }' "$REF" "$out" || FAIL=$((FAIL+1))
}

echo ""
echo "GPU variants vs reference:"
check 1-sycl 1 1
for fam in 2 4; do
    check "${fam}-sycl"                "$fam" 1
    check "${fam}-sycl-stream"         "$fam" 1
    check "${fam}-sycl-stream-omp"     "$fam" "$fam"
    check "${fam}-sycl-stream-omp-p2p" "$fam" "$fam"
done

echo ""
if [[ $FAIL -eq 0 ]]; then
    echo "All variants agree with the serial reference within ${TOL}."
else
    echo "${FAIL} variant(s) FAILED -- do not trust their timings."
    exit 1
fi
