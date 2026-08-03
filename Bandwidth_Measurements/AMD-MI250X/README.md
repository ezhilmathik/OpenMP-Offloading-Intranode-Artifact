# AMD MI250X — Intra-node Data Movement Benchmarks

Benchmarks accompanying the paper *[PAPER TITLE]*.

This directory contains the AMD Instinct MI250X measurements: host-to-device,
device-to-host, and peer-to-peer transfer bandwidth, comparing **OpenMP target
offload** against **native HIP**. Measurements were taken on **LUMI**.

Companion vendor directories: `../NVIDIA-H100/` and `../Intel-1550/`.

---

## Quick start — reproduce every figure, no GPU required

The raw measurement data from our runs is included, so all five figures can be
regenerated on any machine with Python:

```bash
python3 -m venv ~/venv-openmp-offloading
source ~/venv-openmp-offloading/bin/activate
pip install -r ../requirements.txt

./all-plots.sh
```

Runs in seconds. Each figure is written to the corresponding
`<directory>/plots_final/`, and the script reports which succeeded and which
failed.

Re-measuring from scratch requires an MI250X node, a Slurm scheduler, **and a
specific OpenMP offload runtime** — see below.

---

## Directories

| Directory | Measures | Figure |
|---|---|---|
| `Synchronous-test-H-to-D-Measurements/` | Blocking host → device | `mi250x_sync_htod.pdf` |
| `Synchronous-test-D-to-H-Measurements/` | Blocking device → host | `mi250x_sync_dtoh.pdf` |
| `Asynchronous-test-H-to-D-Measurements/` | Non-blocking host → device | `mi250x_async_htod.pdf` |
| `Asynchronous-test-D-to-H-Measurements/` | Non-blocking device → host | `mi250x_async_dtoh.pdf` |
| `P2P-Measurements/` | GCD ↔ GCD over Infinity Fabric | `mi250x_p2p.pdf` |

Each directory has its own README covering what it measures, how to build it,
how to re-run it, and how its output maps to the paper.

The four host-transfer directories sweep identical buffer sizes, so their
figures are directly comparable. P2P uses smaller buffers because both
endpoints are device-resident — see its README.

---

## OpenMP offload runtime — required for re-measurement

**All** OpenMP measurements in this directory were taken against an OpenMP
offload runtime built separately from the one supplied by LUMI's `rocm/6.4.4`
module. `run.sh` injects it via `LD_PRELOAD`:

```
$OMPTARGET_P2P_LIB/libomp.so
$OMPTARGET_P2P_LIB/libomptarget.so.19.0git
```

The library reports itself as:

```
AMD clang version 19.0.0git
(https://github.com/RadeonOpenCompute/llvm-project roc-6.4.4 25224
 d366fa84f3fdcbd4b10847ebd5db572ae12a34fb)
```

*[REQUIRED BEFORE SUBMISSION — describe how to obtain this runtime. It reports
the same ROCm version as the loaded module, so it is not a stock 6.4.4 build.
State whether it is patched, what the patch provides, and either include the
patch here or give a precise build recipe.]*

Point the scripts at your copy with:

```bash
OMPTARGET_P2P_LIB=/path/to/lib sbatch run.sh
```

**HIP measurements do not use this runtime** and need only a standard ROCm
installation. Regenerating the figures from the included data needs nothing
beyond Python.

---

## Common structure

Every benchmark directory follows the same layout:

| File | Purpose |
|---|---|
| `openmp-*.cc` | OpenMP target offload implementations |
| `hip-*.hip` | Native HIP implementations |
| `Makefile` | Build rules (`amdclang++`, `--offload-arch=gfx90a`) |
| `build.sh` | Slurm job: compile and verify |
| `run.sh` | Slurm job: run the sweep, write `timings.csv` |
| `submit.sh` | Submit build + N run jobs with an `afterok` dependency |
| `plot_mi250x_*.py` | Aggregate the runs, produce the figure |
| `results_<jobid>_<timestamp>/` | Raw output, 5 repetitions per directory |
| `plots_final/` | Generated figure |

Version numbering is consistent throughout: `v1` is the single-GCD case, `v2`
the two-GCD case. In P2P these denote transfer direction rather than device
count — see that directory's README.

`v4` and `v4_4` sources and build rules are retained for future work. They are
compiled by `make all` but are not executed by `run.sh` and not plotted.

---

## Environment settings that affect the results

`run.sh` exports three settings in every directory:

| Setting | Effect |
|---|---|
| `HSA_ENABLE_SDMA=0` | Disables the DMA engines; transfers use blit kernels instead. Changes measured bandwidth substantially. |
| `OMP_TARGET_OFFLOAD=mandatory` | OpenMP fails rather than silently falling back to the host. |
| `ROCR_VISIBLE_DEVICES=0,2,4,6` | One GCD per physical MI250X card, so multi-device cases use separate packages. |

Altering any of these — or using a different OpenMP runtime — produces
different numbers and does not constitute a reproduction.

---

## Software environment

```bash
module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4
```

The Slurm directives in `build.sh` and `run.sh` — account and partition — are
specific to LUMI and must be changed before submitting elsewhere.

---

## License and citation

*[Complete before submission.]* See `../LICENSE` and `../CITATION.cff`.
Archived at https://doi.org/10.5281/zenodo.XXXXXXX
