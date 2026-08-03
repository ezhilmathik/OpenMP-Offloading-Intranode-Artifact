#!/bin/bash -l
#SBATCH --job-name=heat-verify
#SBATCH --account=ehpc681
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=verify-%j.out
#
# CORRECTNESS ONLY. Counterpart to run.sh:
#   run.sh     WRITE_TXT=0, no CPU reference, timings meaningful
#   verify.sh  WRITE_TXT=1, CPU reference, timings MEANINGLESS (file I/O is
#              inside the measured region). Never plot numbers from here.
#
#   sbatch verify.sh                                  # N=256, 100 steps
#   sbatch --export=ALL,N=128,STEPS=50 verify.sh
#   sbatch --export=ALL,N=256,TOL=1e-12 verify.sh
#
# Writing solutions costs O(N^3) lines. Keep N small.
#
# The ground truth is `serial`, built with nvc -Mnofma so its floating-point
# contraction matches the CUDA --fmad=false. If this comparison fails at a
# tolerance near machine precision, suspect a missing --fmad=false before
# suspecting the algorithm.

set -uo pipefail

module purge
module load EB/apps
module load NVHPC/25.3-CUDA-12.8.0

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=active
export WRITE_TXT=1                     # <-- the whole point of this script

N="${N:-256}"
STEPS="${STEPS:-100}"
KAPPA_MAX="${KAPPA_MAX:-0.95}"
FAMILIES="${FAMILIES:-2 4}"
TOL="${TOL:-1e-10}"

DT=$(awk -v N="$N" -v k="$KAPPA_MAX" -v s="$STEPS" \
      'BEGIN { h = 1.0/(N-1); dt = h*h/(12.0*k); printf "%.7g", s*dt }')

echo "============================================================"
echo "Correctness check   N=${N}  STEPS=${STEPS}  dt=${DT}"
echo "Tolerance (max abs difference vs CPU reference): ${TOL}"
echo "nvcc     : $(nvcc --version | tail -2 | head -1)"
echo "============================================================"
nvidia-smi -L || true

# The CPU reference is not built by build.sh -- build it here if absent.
if [[ ! -x ./serial ]]; then
    echo "Building CPU reference (nvc -Mnofma)..."
    make serial-ref || { echo "ERROR: could not build serial reference"; exit 1; }
fi

mask_for() {
    local n="$1" v=() i
    for ((i = 0; i < n; i++)); do v+=("$i"); done
    (IFS=,; echo "${v[*]}")
}

REF="u_serial.txt"

echo ""
echo ">>> CPU reference: ./serial ${N} ${DT}"
./serial "$N" "$DT" || { echo "ERROR: serial reference failed"; exit 1; }
[[ -f "$REF" ]] || { echo "ERROR: ${REF} was not produced"; exit 1; }
echo "    reference: ${REF}  ($(wc -l < "$REF") lines)"

compare() {
    awk -v tol="$TOL" '
        NR == FNR { a[FNR] = $NF; next }
        {
            d = a[FNR] - $NF
            if (d < 0) d = -d
            if (d > worst) { worst = d; where = FNR }
        }
        END {
            if (FNR != length(a)) {
                printf "LINE COUNT MISMATCH (%d vs %d)\n", length(a), FNR
                exit 2
            }
            printf "max|diff| = %.6e  at line %d  -> %s\n", worst, where,
                   (worst <= tol ? "PASS" : "FAIL")
            exit (worst <= tol ? 0 : 1)
        }' "$1" "$2"
}

declare -a PASSED=() FAILED=()

check_variant() {
    local binary="$1" gpus="$2" out="$3"
    local mask; mask=$(mask_for "$gpus")

    if [[ ! -x "./$binary" ]]; then
        echo "  SKIP (missing): $binary"
        return 0
    fi

    echo ""
    echo ">>> ${binary}  gpus=${gpus}  CUDA_VISIBLE_DEVICES=${mask}"
    rm -f "$out"

    if ! env "CUDA_VISIBLE_DEVICES=${mask}" "OMP_NUM_THREADS=${gpus}" ./"$binary" "$N" "$DT"; then
        echo "    RUN FAILED"
        FAILED+=("$binary (run failed)")
        return 0
    fi

    if [[ ! -f "$out" ]]; then
        echo "    ERROR: expected ${out}, not produced"
        FAILED+=("$binary (no output)")
        return 0
    fi

    local result
    result=$(compare "$REF" "$out")
    echo "    $result"
    if [[ "$result" == *PASS* ]]; then PASSED+=("$binary"); else FAILED+=("$binary"); fi
}

if [[ -x ./1-cuda ]]; then
    check_variant 1-cuda 1 u_cuda_1gpu.txt
fi

for fam in $FAMILIES; do
    check_variant "${fam}-cuda"                "$fam" "u_cuda_${fam}gpu.txt"
    check_variant "${fam}-cuda-stream"         "$fam" "u_cuda_stream_${fam}gpu.txt"
    check_variant "${fam}-cuda-stream-omp"     "$fam" "u_cuda_stream_omp_${fam}gpu.txt"
    check_variant "${fam}-cuda-stream-omp-p2p" "$fam" "u_cuda_stream_omp_p2p_${fam}gpu.txt"
done

echo ""
echo "============================================================"
echo "PASSED (${#PASSED[@]}): ${PASSED[*]:-none}"
echo "FAILED (${#FAILED[@]}): ${FAILED[*]:-none}"
echo "============================================================"
echo ""
echo "Solution .txt files kept for inspection. Remove with: ./clean.sh -f out"

[[ ${#FAILED[@]} -eq 0 ]]
