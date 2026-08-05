# SYCL pipeline — 3D heat equation, Intel Data Center GPU Max 1550

Drop these into `~/3D_Heat_Benchmark/Final-3/SYCL/` alongside the `.cpp` sources.

```
./submit.sh 5          # build once (AOT), then 5 reps per grid size
./aggregate.sh         # medians -> summary.csv
./plot.sh -a           # ASCII bars / speedups
sbatch verify.sh       # correctness only, separate from timing
```

## What the old `sycl-batch.sh` did that this does not

The batch script chose flags **per binary** by walking a ladder (`-O3`, then
`-O3 -fno-fast-math`, then `-O2 …`) and kept whichever compiled first, with
non-stream binaries on the fast ladder and stream binaries on the safe one.
That is fine for "make it run", and fatal for "measure the difference between
these two versions": a `2-sycl` vs `2-sycl-stream` comparison then confounds
the algorithmic change with an optimisation and floating-point change. The
Makefile uses one flag set for all nine binaries, exactly as your OpenMP
Makefile does, and if one source ever fails to build the instruction is to drop
the whole `OPT` line rather than that one file.

## Flags

| | OpenMP | SYCL |
|---|---|---|
| compiler | `icx` | `icpx` |
| offload | `-fiopenmp -fopenmp-targets=spir64` | `-fsycl` |
| host OpenMP | (same flag) | `-fiopenmp` — still needed |
| FP | `-O3 -fno-fast-math -ffp-contract=off` | identical |
| link | `-Wl,--no-warn-execstack` | identical |

`-fiopenmp` stays in the SYCL build for two reasons: every source calls
`omp_get_wtime()`, and the `*-stream-omp` / `*-stream-omp-p2p` variants drive
one device per host OpenMP thread. It is host OpenMP only — no
`-fopenmp-targets`, no libomptarget.

**AOT is the default** (`-fsycl-targets=spir64_gen -Xs "-device pvc"`). Under
JIT, the SPIR-V→PVC translation happens at the first kernel submission, which
is *inside* the region the solver times itself, so it lands on the first
repetition of every variant. `AOT=0` falls back to JIT if `ocloc` is missing;
`run.sh` sets `SYCL_CACHE_PERSISTENT=1` as insurance either way, but treat rep 1
as a warm-up in that case.

## Runtime environment — the one deliberate inversion

`run.sh` **sets** `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, where the OpenMP
`run.sh` unsets it. Without it the OpenCL backend exposes the same four cards a
second time and a "4 GPU" run can put two ranks on one card through two
different backends. `libomptarget` does its own discovery and the variable only
confuses it, hence the difference.

Everything hidden-helper-related is gone: `LIBOMP_NUM_HIDDEN_HELPER_THREADS`,
`OMP_NUM_HIDDEN_HELPER_THREADS`, `LIBOMP_USE_HIDDEN_HELPER_TASK`,
`LIBOMPTARGET_DEVICES` are all libomptarget knobs. SYCL gets its asynchrony
from the queues themselves, so there is no pool to size and no `HELPERS` sweep
to run — the knob is dropped rather than carried over as a no-op.

`ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` is locked, same as your OpenMP `run.sh`:
one device = one whole card, both stacks. Keep it that way or the SYCL and
OpenMP numbers are measuring different hardware units.

Host threads per binary: `1` for `1-sycl`, `*-sycl` and `*-sycl-stream`;
`$gpus` for `*-sycl-stream-omp` and `*-sycl-stream-omp-p2p`.

Before each run, `run.sh` counts the Level Zero devices actually visible under
the mask and records `device_mismatch` instead of launching — that turns the
old "Need 4 GPUs, found 1" mid-run abort into one line in `skipped.log`.

## Two bugs carried over from the OpenMP scripts, fixed here

1. **`CC ?= icx` is a no-op.** GNU make predefines `CC=cc` and `CXX=g++`, and
   `?=` only assigns when a variable is *undefined* — origin `default` counts
   as defined. Run `make flags` in your OpenMP directory; if it prints
   `CC = cc`, that build has not been using `icx` unless the `intel-toolkit`
   module exports `CC`. This Makefile tests `$(origin CXX)` instead, which
   overrides the built-in default while still honouring `make CXX=…` or an
   exported `CXX`.

2. **`aggregate.sh` wall mode keys on the wrong columns.** Its `off = NF - 12`
   fudge targets an older 12-field schema; on the 13-field Intel CSV it puts
   `variant` on `hier` and `gpus` on `family`, so `TIME_FIELD=wall` produces
   rows labelled `COMPOSITE`. The compute path (the default) is unaffected,
   which is presumably why it went unnoticed. Fixed with fixed indices and a
   field-count guard.

## Things to check against your actual sources

- `verify.sh` globs for `u_serial*.txt` and `u_sycl_*.txt`. If
  `write_u_values()` in the SYCL sources uses different filenames, adjust the
  two globs. The numeric comparison itself is layout-independent — it compares
  every whitespace-separated token in file order — so the grid dump format does
  not matter.
- If the sources are still DPCT-migrated and `<dpct/dpct.hpp>` is not on the
  default include path, the Makefile picks up `$DPCT_BUNDLE_ROOT/include` when
  the `intel-dpct` module exports it.
- `NS`, `STEPS` and `KAPPA_MAX` in `submit.sh` must match the OpenMP sweep
  exactly, or the two sets of timings are for different amounts of work.
