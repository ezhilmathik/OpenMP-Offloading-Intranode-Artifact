#!/usr/bin/env python3
"""
compare.py — diff the serial (CPU) heat-solver result against every available
OpenMP-offloading GPU output.

Each *.txt file is one value per line ("%.17g"), i.e. a flat dump of u[k][j][i]
of length n^3, exactly as written by write_u_values() in the solvers.

Usage:
    python3 compare.py                         # ref=u_serial.txt, all known GPU files
    python3 compare.py --ref u_serial.txt      # explicit reference
    python3 compare.py --atol 1e-9 --rtol 1e-7 # tighten tolerances
    python3 compare.py file_a.txt file_b.txt   # compare these specific files vs ref

Exit code is 0 if every compared file PASSES, 1 otherwise (handy in scripts).
"""

import argparse
import os
import sys
import numpy as np

# Known GPU outputs (label -> filename). Only those that exist are compared,
# so this works for 1-, 2-, and 4-GPU runs without changes.
KNOWN = [
    ("1-GPU OpenMP",            "1_openmp_gpu.txt"),
    ("2-GPU",                   "u_openmp_2gpu.txt"),
    ("2-GPU stream",            "u_openmp_stream_2gpu.txt"),
    ("2-GPU stream-omp",        "u_openmp_stream_omp_2gpu.txt"),
    ("2-GPU stream-omp-p2p",    "u_openmp_stream_omp_p2p_2gpu.txt"),
    ("4-GPU",                   "u_openmp_4gpu.txt"),
    ("4-GPU stream",            "u_openmp_stream_4gpu.txt"),
    ("4-GPU stream-omp",        "u_openmp_stream_omp_4gpu.txt"),
    ("4-GPU stream-omp-p2p",    "u_openmp_stream_omp_p2p_4gpu.txt"),
]


def load(path):
    """Load a flat column of doubles; raise with a clear message on failure."""
    try:
        a = np.loadtxt(path, dtype=np.float64)
    except Exception as e:
        raise RuntimeError(f"could not read '{path}': {e}")
    return np.atleast_1d(a).ravel()


def metrics(ref, cand):
    """Return a dict of error metrics between reference and candidate arrays."""
    diff = cand - ref
    adiff = np.abs(diff)
    aref = np.abs(ref)

    # element-wise relative error where the reference is not ~0
    denom = np.maximum(aref, 1e-30)
    rel = adiff / denom
    # only trust relative error where the reference magnitude is meaningful
    mask = aref > 1e-12
    max_rel = float(rel[mask].max()) if np.any(mask) else 0.0

    ref_norm = float(np.linalg.norm(ref))
    rel_l2 = float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0 else float(np.linalg.norm(diff))

    i = int(np.argmax(adiff))
    return {
        "n":        ref.size,
        "max_abs":  float(adiff.max()),
        "mean_abs": float(adiff.mean()),
        "rms":      float(np.sqrt(np.mean(diff**2))),
        "rel_l2":   rel_l2,
        "max_rel":  max_rel,
        "argmax":   i,
        "ref_at":   float(ref[i]),
        "cand_at":  float(cand[i]),
        "n_nan":    int(np.isnan(cand).sum()),
        "n_inf":    int(np.isinf(cand).sum()),
    }


def compare_one(label, ref, path, atol, rtol):
    if not os.path.isfile(path):
        return None  # not generated; skip silently
    try:
        cand = load(path)
    except RuntimeError as e:
        print(f"  [ERROR] {label:<22} {path:<34} {e}")
        return False

    if cand.size != ref.size:
        print(f"  [FAIL ] {label:<22} {path:<34} "
              f"size mismatch: ref={ref.size} cand={cand.size}")
        return False

    m = metrics(ref, cand)
    ok = (m["n_nan"] == 0 and m["n_inf"] == 0
          and np.allclose(cand, ref, rtol=rtol, atol=atol, equal_nan=False))
    tag = "PASS" if ok else "FAIL"

    print(f"  [{tag} ] {label:<22} {os.path.basename(path):<34} "
          f"maxabs={m['max_abs']:.3e}  rms={m['rms']:.3e}  "
          f"relL2={m['rel_l2']:.3e}  maxrel={m['max_rel']:.3e}")

    if not ok:
        extra = ""
        if m["n_nan"] or m["n_inf"]:
            extra = f"  NaN={m['n_nan']} Inf={m['n_inf']}"
        print(f"           worst @ idx {m['argmax']}: "
              f"ref={m['ref_at']:.6e}  cand={m['cand_at']:.6e}{extra}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Compare serial CPU result vs GPU outputs.")
    ap.add_argument("files", nargs="*",
                    help="specific candidate files (default: all known GPU outputs that exist)")
    ap.add_argument("--ref", default="u_serial.txt",
                    help="reference (CPU/serial) file [default: u_serial.txt]")
    ap.add_argument("--atol", type=float, default=1e-6, help="absolute tolerance [1e-6]")
    ap.add_argument("--rtol", type=float, default=1e-5, help="relative tolerance [1e-5]")
    args = ap.parse_args()

    if not os.path.isfile(args.ref):
        print(f"ERROR: reference file '{args.ref}' not found.")
        print("Run the batch script in 'write' mode first to produce u_serial.txt.")
        sys.exit(2)

    ref = load(args.ref)
    print(f"Reference: {args.ref}  ({ref.size} values)")
    print(f"Tolerance: atol={args.atol:g}, rtol={args.rtol:g}\n")

    if args.files:
        targets = [(os.path.basename(f), f) for f in args.files]
    else:
        targets = KNOWN

    results = []
    any_found = False
    for label, path in targets:
        r = compare_one(label, ref, path, args.atol, args.rtol)
        if r is not None:
            any_found = True
            results.append((label, r))

    if not any_found:
        print("  (no candidate GPU output files found in this directory)")
        sys.exit(2)

    n_pass = sum(1 for _, r in results if r)
    n_total = len(results)
    print(f"\nSummary: {n_pass}/{n_total} passed "
          f"(atol={args.atol:g}, rtol={args.rtol:g})")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
