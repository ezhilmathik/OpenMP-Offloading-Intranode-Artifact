#!/usr/bin/env python3
"""
Compare solutions from CPU and parallel versions of the heat equation solver.
"""

import numpy as np
import sys

def read_solution(filename):
    """Read solution values from text file."""
    try:
        data = np.loadtxt(filename)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        sys.exit(1)

def compare_solutions(file1, file2):
    """Compare two solution files and report differences."""
    
    print("=" * 60)
    print("SOLUTION COMPARISON")
    print("=" * 60)
    
    # Read both files
    u1 = read_solution(file1)
    u2 = read_solution(file2)
    
    print(f"\n{file1}: {len(u1)} values")
    print(f"{file2}: {len(u2)} values")
    
    # Check if sizes match
    if len(u1) != len(u2):
        print(f"\n⚠️  WARNING: Different array sizes!")
        print(f"   {file1}: {len(u1)} values")
        print(f"   {file2}: {len(u2)} values")
        return
    
    # Compute differences
    diff = u1 - u2
    abs_diff = np.abs(diff)
    
    # Compute relative error (avoiding division by zero)
    rel_error = np.zeros_like(diff)
    mask = np.abs(u1) > 1e-15
    rel_error[mask] = abs_diff[mask] / np.abs(u1[mask])
    
    # Statistics
    print(f"\n" + "-" * 60)
    print("STATISTICS")
    print("-" * 60)
    print(f"Max absolute difference:  {np.max(abs_diff):.6e}")
    print(f"Mean absolute difference: {np.mean(abs_diff):.6e}")
    print(f"RMS difference:           {np.sqrt(np.mean(diff**2)):.6e}")
    
    if np.any(mask):
        print(f"Max relative error:       {np.max(rel_error):.6e}")
        print(f"Mean relative error:      {np.mean(rel_error[mask]):.6e}")
    
    # Check for exact match
    if np.allclose(u1, u2, rtol=1e-14, atol=1e-14):
        print(f"\n✅ Solutions match within machine precision!")
    elif np.allclose(u1, u2, rtol=1e-10, atol=1e-10):
        print(f"\n✅ Solutions match to high precision (rtol=1e-10)")
    elif np.allclose(u1, u2, rtol=1e-6, atol=1e-6):
        print(f"\n⚠️  Solutions match to moderate precision (rtol=1e-6)")
    else:
        print(f"\n❌ Solutions differ significantly!")
    
    # Find worst mismatches
    worst_indices = np.argsort(abs_diff)[-5:][::-1]
    print(f"\n" + "-" * 60)
    print("5 LARGEST DIFFERENCES")
    print("-" * 60)
    print(f"{'Index':<10} {'File1':<20} {'File2':<20} {'Abs Diff':<15}")
    print("-" * 60)
    for idx in worst_indices:
        print(f"{idx:<10} {u1[idx]:<20.12e} {u2[idx]:<20.12e} {abs_diff[idx]:<15.6e}")
    
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_solutions.py <file1> <file2>")
        print("Example: python compare_solutions.py u_cpu.txt u_parallel.txt")
        sys.exit(1)
    
    compare_solutions(sys.argv[1], sys.argv[2])
