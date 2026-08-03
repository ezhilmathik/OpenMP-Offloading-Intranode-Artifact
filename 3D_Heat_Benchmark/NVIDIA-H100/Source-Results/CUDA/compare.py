#!/usr/bin/env python3
"""
compare.py — diff a serial (CPU) reference result against every other solution
dump in the current directory, regardless of how the candidate files are named.

Each *.txt file is one value per line ("%.17g"), i.e. a flat dump of u[k][j][i]
of length n^3, exactly as written by write_u_values() in the solvers.

GENERAL MODE (default):
    The reference is u_serial.txt. Every file matching the candidate pattern
    (default 'u_*.txt') EXCEPT the reference is loaded and compared against it.
    No hardcoded filename list -> works for CUDA, OpenMP, OpenACC, ... unchanged.

Usage:
    python3 compare.py                          # ref=u_serial.txt, all u_*.txt candidates
    python3 compare.py --ref u_serial.txt       # explicit reference
    python3 compare.py --glob 'u_cuda_*.txt'    # restrict candidates by pattern
    python3 compare.py --atol 1e-9 --rtol 1e-7  # tighten tolerances
    python3 compare.py a.txt b.txt              # compare these specific files vs ref

Exit code is 0 if every compared file PASSES, 1 otherwise (handy in scripts).
"""

import argparse
import glob
import os
import sys
import numpy as np


def load(path):
    """Load a flat column of doubles; raise with a clear message on failure."""
    try:
        a = np.loadtxt(path, dtype=np.float64)
    except Exception as e:
        raise RuntimeError(f"could not read '{path}': {e}")
    return np.atleast_1d(a).ravel()


def label_for(path):
    """Human-friendly label derived from the filename (no hardcoding)."""
    base = os.path.basename(path)
    name = base[:-4] if base.endswith(".txt") else base
    if name.startswith("u_"):
        name = name[2:]
    return name.replace("_", " ") or base


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


def compare_one(label, ref, path, atol, rtol, strict_size):
    """Compare one candidate file vs ref.
    Returns True/False (pass/fail), or None to skip (counts toward neither)."""
    try:
        cand = load(path)
    except RuntimeError as e:
        print(f"  [ERROR] {label:<22} {os.path.basename(path):<34} {e}")
        return False

    if cand.size != ref.size:
        # In auto-discovery mode a size mismatch usually means a different-N
        # run was left in the folder -> skip it rather than flag a false FAIL.
        # When files are passed explicitly, treat it as a real failure.
        tag = "FAIL" if strict_size else "SKIP"
        print(f"  [{tag} ] {label:<22} {os.path.basename(path):<34} "
              f"size mismatch: ref={ref.size} cand={cand.size}")
        return False if strict_size else None

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


def discover(pattern, ref_path):
    """All files matching pattern, excluding the reference, sorted by name."""
    ref_abs = os.path.abspath(ref_path)
    found = [p for p in glob.glob(pattern)
             if os.path.abspath(p) != ref_abs]
    return sorted(found)


def main():
    ap = argparse.ArgumentParser(description="Compare a serial CPU result vs every other solution dump.")
    ap.add_argument("files", nargs="*",
                    help="specific candidate files (default: auto-discover via --glob)")
    ap.add_argument("--ref", default="u_serial.txt",
                    help="reference (CPU/serial) file [default: u_serial.txt]")
    ap.add_argument("--glob", default="u_*.txt",
                    help="candidate filename pattern for auto-discovery [default: 'u_*.txt']")
    ap.add_argument("--atol", type=float, default=1e-6, help="absolute tolerance [1e-6]")
    ap.add_argument("--rtol", type=float, default=1e-5, help="relative tolerance [1e-5]")
    args = ap.parse_args()

    if not os.path.isfile(args.ref):
        print(f"ERROR: reference file '{args.ref}' not found.")
        print("Run the batch script in 'write' mode first to produce the reference.")
        sys.exit(2)

    ref = load(args.ref)
    print(f"Reference: {args.ref}  ({ref.size} values)")
    print(f"Tolerance: atol={args.atol:g}, rtol={args.rtol:g}\n")

    if args.files:
        targets = args.files
        strict_size = True          # explicit list -> size mismatch is a real FAIL
    else:
        targets = discover(args.glob, args.ref)
        strict_size = False         # auto mode -> skip stray different-N files

    if not targets:
        print(f"  (no candidate files found matching '{args.glob}' besides the reference)")
        sys.exit(2)

    results = []
    for path in targets:
        if not os.path.isfile(path):
            print(f"  [ERROR] {label_for(path):<22} {os.path.basename(path):<34} not found")
            results.append(False)
            continue
        r = compare_one(label_for(path), ref, path, args.atol, args.rtol, strict_size)
        if r is not None:
            results.append(r)

    if not results:
        print("\n  (nothing comparable — all candidates skipped)")
        sys.exit(2)

    n_pass = sum(1 for r in results if r)
    n_total = len(results)
    print(f"\nSummary: {n_pass}/{n_total} passed "
          f"(atol={args.atol:g}, rtol={args.rtol:g})")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
