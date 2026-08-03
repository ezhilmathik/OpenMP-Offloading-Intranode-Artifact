#!/usr/bin/env python3
"""
compare.py — diff the serial (CPU) heat-solver result against every available
GPU output.

Each *.txt file is one value per line ("%.17g"), i.e. a flat dump of u[k][j][i]
of length n^3, exactly as written by write_u_values() in the solvers.

The expected filenames are derived from a backend prefix, so the same script
serves the HIP, CUDA and OpenMP trees:

    u_<backend>_1gpu.txt
    u_<backend>_<N>gpu.txt
    u_<backend>_stream_<N>gpu.txt
    u_<backend>_stream_omp_<N>gpu.txt
    u_<backend>_stream_omp_p2p_<N>gpu.txt

Usage:
    python3 compare.py                          # backend=hip, all families
    python3 compare.py --backend openmp         # the OpenMP tree
    python3 compare.py --backend auto           # infer from what is on disk
    python3 compare.py --ref u_serial.txt       # explicit reference
    python3 compare.py --atol 1e-9 --rtol 1e-7  # tighten tolerances
    python3 compare.py file_a.txt file_b.txt    # compare these specific files

Pass criterion is numpy's:  |cand - ref| <= atol + rtol * |ref|

Exit codes:
    0  every compared file PASSED
    1  at least one FAILED
    2  nothing to compare (no reference, or no candidate files found)
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

BACKENDS = ("hip", "cuda", "openmp")

# (filename infix, human label suffix) in reporting order.
VARIANTS = [
    ("",                ""),
    ("stream",          " stream"),
    ("stream_omp",      " stream-omp"),
    ("stream_omp_p2p",  " stream-omp-p2p"),
]


def known_targets(backend, families=(2, 4), baseline=True):
    """(label, filename) pairs for one backend, in reporting order."""
    out = []
    if baseline:
        out.append((f"1-GPU {backend}", f"u_{backend}_1gpu.txt"))
    for fam in families:
        for infix, label in VARIANTS:
            if infix:
                name = f"u_{backend}_{infix}_{fam}gpu.txt"
            else:
                name = f"u_{backend}_{fam}gpu.txt"
            out.append((f"{fam}-GPU{label}", name))
    return out


def detect_backend(ref):
    """Guess the backend from the u_*.txt files present in the cwd."""
    found = set()
    for path in glob.glob("u_*.txt"):
        if os.path.basename(path) == os.path.basename(ref):
            continue
        m = re.match(r"u_([a-z]+)_", os.path.basename(path))
        if m and m.group(1) in BACKENDS:
            found.add(m.group(1))
    if len(found) == 1:
        return found.pop()
    if len(found) > 1:
        print(f"ERROR: outputs from more than one backend present: "
              f"{', '.join(sorted(found))}")
        print("       They would be compared against the same reference and the")
        print("       result would be ambiguous. Clean up, or pass --backend.")
        sys.exit(2)
    return None


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
    rel_l2 = (float(np.linalg.norm(diff) / ref_norm) if ref_norm > 0
              else float(np.linalg.norm(diff)))

    i = int(np.argmax(adiff))
    return {
        "n":        ref.size,
        "max_abs":  float(adiff.max()),
        "mean_abs": float(adiff.mean()),
        "rms":      float(np.sqrt(np.mean(diff ** 2))),
        "rel_l2":   rel_l2,
        "max_rel":  max_rel,
        "argmax":   i,
        "ref_at":   float(ref[i]),
        "cand_at":  float(cand[i]),
        "n_nan":    int(np.isnan(cand).sum()),
        "n_inf":    int(np.isinf(cand).sum()),
    }


def compare_one(label, ref, path, atol, rtol):
    """True = pass, False = fail, None = file absent."""
    if not os.path.isfile(path):
        return None
    try:
        cand = load(path)
    except RuntimeError as e:
        print(f"  [ERROR] {label:<22} {os.path.basename(path):<34} {e}")
        return False

    if cand.size != ref.size:
        print(f"  [FAIL ] {label:<22} {os.path.basename(path):<34} "
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
    ap = argparse.ArgumentParser(
        description="Compare serial CPU result vs GPU outputs.")
    ap.add_argument("files", nargs="*",
                    help="specific candidate files "
                         "(default: all known outputs for --backend that exist)")
    ap.add_argument("--ref", default="u_serial.txt",
                    help="reference (CPU/serial) file [default: u_serial.txt]")
    ap.add_argument("--backend", default="hip",
                    choices=(*BACKENDS, "auto"),
                    help="filename prefix to look for [default: hip]")
    ap.add_argument("--families", default="2,4",
                    help="GPU counts to look for [default: 2,4]")
    ap.add_argument("--atol", type=float, default=1e-6,
                    help="absolute tolerance [1e-6]")
    ap.add_argument("--rtol", type=float, default=1e-5,
                    help="relative tolerance [1e-5]")
    args = ap.parse_args()

    if not os.path.isfile(args.ref):
        print(f"ERROR: reference file '{args.ref}' not found.")
        print("Run verify.sh (WRITE_TXT=1) first to produce u_serial.txt.")
        sys.exit(2)

    backend = args.backend
    if backend == "auto":
        backend = detect_backend(args.ref)
        if backend is None:
            print("ERROR: no recognisable u_<backend>_*.txt files in this "
                  "directory.")
            sys.exit(2)
        print(f"Detected backend: {backend}")

    families = tuple(int(f) for f in args.families.replace("+", ",").split(",")
                     if f.strip())

    ref = load(args.ref)
    print(f"Reference: {args.ref}  ({ref.size} values)")
    print(f"Criterion: |cand - ref| <= atol + rtol*|ref|  "
          f"(atol={args.atol:g}, rtol={args.rtol:g})\n")

    if args.files:
        targets = [(os.path.basename(f), f) for f in args.files]
    else:
        targets = known_targets(backend, families)

    results = []
    missing = []
    for label, path in targets:
        r = compare_one(label, ref, path, args.atol, args.rtol)
        if r is None:
            missing.append(path)
        else:
            results.append((label, r))

    # Anything on disk that the table did not account for. A silently ignored
    # output file is how a rename slips through unnoticed.
    if not args.files:
        expected = {os.path.basename(p) for _, p in targets}
        expected.add(os.path.basename(args.ref))
        extras = sorted(os.path.basename(p) for p in glob.glob("u_*.txt")
                        if os.path.basename(p) not in expected)
    else:
        extras = []

    if missing:
        print(f"\n  Not present ({len(missing)}): {', '.join(missing)}")
    if extras:
        print(f"  Not in the table ({len(extras)}): {', '.join(extras)}")
        print("  -> wrong --backend, or the solvers write different names now.")

    if not results:
        print("\n  (no candidate output files found for backend "
              f"'{backend}' in this directory)")
        sys.exit(2)

    n_pass = sum(1 for _, r in results if r)
    n_total = len(results)
    print(f"\nSummary: {n_pass}/{n_total} passed, {len(missing)} missing "
          f"(atol={args.atol:g}, rtol={args.rtol:g})")
    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
