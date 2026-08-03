# Asynchronous Host-to-Device Bandwidth — AMD MI250X

Benchmarks accompanying the paper *[PAPER TITLE]*.

This directory measures **asynchronous host-to-device (H-to-D) transfer
bandwidth** on an AMD Instinct MI250X node, comparing **OpenMP target
offload** against **native HIP**.

Companion directories in `../`: `Synchronous-test-H-to-D-Measurements/`,
`Asynchronous-test-H-to-D-Measurements/`,
`Asynchronous-test-D-to-H-Measurements/`, `P2P-Measurements/`.
Other vendors: `../../NVIDIA-H100/` and `../../Intel-1550/`.

---

## 1. What is measured

| Version | Configuration | Theoretical ceiling |
|---|---|---|
| `v1` | Asynchronous transfer to a single GCD | 36 GB/s |
| `v2` | Concurrent asynchronous transfer to two GCDs on separate cards | 72 GB/s |

Transfers are issued without blocking, so data movement can overlap rather
than serialise. Each configuration is implemented twice — once in OpenMP
target offload, once in HIP — giving four measured series. Buffers are arrays
of `double` (8 bytes per element).

Buffer sizes match the synchronous directories, so the synchronous and
asynchronous figures are directly comparable.

The ceilings are the **host-link (Infinity Fabric) bandwidths** in one
direction: 36 GB/s per GCD, doubled when two independent links are driven in
parallel. They are configured in `THEORETICAL_BW` at the top of
`plot_mi250x_async_htod.py` and drawn as dashed reference lines.

### GCD selection

Each MI250X package contains **two GCDs**, each presented to the runtime as a
separate device. `run.sh` sets

```bash
export ROCR_VISIBLE_DEVICES=0,2,4,6
```

so that only one GCD per physical card is visible. `v2` therefore uses GCDs on
two **different** cards, each with its own host link — which is what makes
72 GB/s the correct ceiling. Selecting two GCDs of the same package would
share a host link and give a different result.

> A four-GPU variant (`v4`) exists in the repository and is left to future
> work. Its source and build rule remain in place, but it is not measured or
> plotted.

---

## 2. Contents

| File | Description |
|---|---|
| `openmp-1.cc`, `openmp-2.cc` | OpenMP target offload implementations |
| `hip-1.hip`, `hip-2.hip` | Equivalent native HIP implementations |
| `Makefile` | Build rules for both toolchains |
| `build.sh` | Slurm job: compiles the binaries and verifies they exist |
| `run.sh` | Slurm job: executes the buffer-size sweep, writes `timings.csv` |
| `submit.sh` | Submits `build.sh`, then N run jobs with an `afterok` dependency |
| `plot_mi250x_async_htod.py` | Aggregates all runs and produces the paper figure |
| `clean.sh` | Removes compiled binaries |
| `results_<jobid>_<timestamp>/` | Raw output from our runs (5 repetitions) |
| `plots_final/` | Generated figure, as it appears in the paper |
| `openmp-4.cc`, `hip-4.hip` | Four-GPU variant — future work, not used |

Compiled executables are not included; `build.sh` produces them.

---

## 3. Requirements

### Hardware

- An AMD Instinct MI250X node (`gfx90a`) with at least two cards, so that two
  GCDs on separate packages are available.
- Approximately 40 GB of host RAM — the largest buffer is 4.96e9 doubles
  (~39.7 GB).
- Exclusive node access. The scripts pass `--exclusive`.

### Software

Measurements were taken on **LUMI**. The environment `run.sh` sets up:

```bash
module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4
```

Figure generation needs Python 3 with numpy and matplotlib
(`../../requirements.txt`).

### Runtime environment settings

`run.sh` exports three settings that materially affect the measurements:

| Setting | Effect |
|---|---|
| `HSA_ENABLE_SDMA=0` | Disables the DMA engines, so transfers go through blit kernels instead. This changes measured bandwidth substantially; results are not comparable to runs without it. |
| `OMP_TARGET_OFFLOAD=mandatory` | Makes OpenMP fail rather than silently fall back to host execution. |
| `ROCR_VISIBLE_DEVICES=0,2,4,6` | One GCD per physical card — see Section 1. |

*[Add one line on why SDMA is disabled — reviewers will ask, and it is a
methodological choice rather than an incidental one.]*

### OpenMP offload runtime — required

The OpenMP measurements do **not** use the runtime supplied by the
`rocm/6.4.4` module. `run.sh` preloads a separate build:

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

*[REQUIRED BEFORE SUBMISSION — describe how to obtain this runtime.
It reports the same ROCm version as the loaded module, so it is not simply a
stock 6.4.4 build. State whether it is patched, what the patch does, and
either include the patch in this repository or give a precise recipe. Without
this, the OpenMP results cannot be reproduced.]*

Point the scripts at it with:

```bash
OMPTARGET_P2P_LIB=/path/to/lib sbatch run.sh
```

or edit the default near the top of `run.sh`.

**The HIP measurements do not use this runtime** and need only the standard
ROCm installation.

### Site-specific settings that must be changed

```bash
#SBATCH --account=project_465002427   # replace with your allocation
#SBATCH --partition=standard-g        # replace with your GPU partition
```

The `module load` lines are LUMI-specific and need local equivalents
elsewhere.

---

## 4. Build

```bash
sbatch build.sh          # on the cluster
```

or directly, with ROCm available:

```bash
make            # all four binaries
make omp        # OpenMP offload only
make hip        # HIP only
```

Produces:

```
version-1-off   version-2-off      # OpenMP target offload
version-1-hip   version-2-hip      # HIP
```

---

## 5. Smoke test (~2 minutes)

```bash
make omp
export HSA_ENABLE_SDMA=0
export ROCR_VISIBLE_DEVICES=0,2,4,6
LD_PRELOAD="$OMPTARGET_P2P_LIB/libomp.so:$OMPTARGET_P2P_LIB/libomptarget.so.19.0git" \
    ./version-1-off 100000000
```

Expected: one line reporting elapsed time and bandwidth in GB/s, below the
36 GB/s host-link ceiling. Failure here is an environment problem — most often
the OpenMP runtime path.

---

## 6. Reproducing the results

### Python environment

```bash
python3 -m venv ~/venv-openmp-offloading
source ~/venv-openmp-offloading/bin/activate
pip install -r ../../requirements.txt
```

### Regenerating the figure from the included data

```bash
python3 plot_mi250x_async_htod.py
```

Globs all `results_*/timings.csv`, averages each (framework, version, size)
triple across the five repetitions, writes
`plots_final/mi250x_async_htod.pdf`, and prints the mean and maximum
HIP-versus-OpenMP gap per version.

Because the raw results are included, this needs **no GPU and no cluster
access**. `../all-plots.sh` regenerates every MI250X figure at once.

### Re-measuring from scratch

```bash
./submit.sh
```

One build job followed by five independent run jobs, each depending on the
build succeeding. Adjust `N_RUNS` at the top of `submit.sh`.

Buffer sizes swept, in elements (`double`, 8 bytes each):

```
940000000  1740000000  2550000000  3350000000  4160000000  4960000000
```

These match the H100 and Intel directories, so the vendor figures are
directly comparable.

---

## 7. Expected output

Each run job writes `results_<jobid>_<timestamp>/` containing `timings.csv`,
`run.log`, and `skipped.log`.

`timings.csv` schema:

```
framework,version,N,elapsed_sec,bandwidth_gbs,status
```

`framework` is `omp_off` or `hip`; `version` is `v1` or `v2`; `status` is
`ok`, `timeout`, `runtime_error`, or `missing_binary`. Every row should read
`ok` in a healthy run — the plot script silently discards any row that does
not.

Some rows in the included data carry an extra empty field between `N` and
`elapsed_sec`, an artifact of an earlier version of `run.sh`. The plot script
accepts both layouts.

### Claim map

| Paper claim | Command | Cost | Confirming output |
|---|---|---|---|
| *[Figure N]*: OpenMP offload vs HIP, async H-to-D on MI250X | `python3 plot_mi250x_async_htod.py` | seconds, no GPU needed | `plots_final/mi250x_async_htod.pdf` |
| *[Figure N]*, measured from scratch | `./submit.sh`, then the above | ~5 node-h | same |
| *[Claim about the mean HIP vs OpenMP gap]* | as above | — | summary box in the figure; also printed to stdout |
| *[Any cross-vendor claim]* | also run `../../NVIDIA-H100/` and `../../Intel-1550/` | — | the respective figures |

*[Replace the bracketed labels with the actual figure and table numbers.]*

### Tolerance

Bandwidth varies run to run and machine to machine. Deviations of a few
percent from the published values are normal — the claim under test is the
relationship between OpenMP offload and HIP, not the absolute figures.

Two settings change the numbers enough that altering them does not constitute
a reproduction: the OpenMP runtime described in Section 3, and
`HSA_ENABLE_SDMA=0`.

Asynchronous measurements are inherently noisier than synchronous ones, so
expect somewhat wider run-to-run spread here.

---

## 8. Troubleshooting

**OpenMP results differ markedly from the published figures.** Either the
wrong OpenMP runtime is in use — check `OMPTARGET_P2P_LIB` — or
`HSA_ENABLE_SDMA` is not 0.

**`libomptarget.so.19.0git` not found.** See Section 3. Note that
`LD_PRELOAD` fails silently on a missing file, so the run would otherwise
proceed against the module's runtime and produce different numbers.

**OpenMP reports host execution or aborts.** `OMP_TARGET_OFFLOAD=mandatory`
turns a silent host fallback into a failure. Check that the binary was built
with `--offload-arch=gfx90a` and that a GPU is visible.

**Asynchronous bandwidth matches the synchronous figures.** The transfers are
not overlapping. Note that `HSA_ENABLE_SDMA=0` disables the DMA engines, which
constrains how much overlap is achievable — see Section 3.

**Every row reads `missing_binary`.** The build stage failed; inspect
`build-<jobid>.out`.

**Rows read `timeout`.** The per-measurement limit is `TIMEOUT_SEC` in
`run.sh`.

**Out of memory at large sizes.** Trim the `Ns` array in `run.sh` and state
which sizes were omitted in any comparison.

**`v2` results are missing or match `v1`.** Fewer than two GCDs were visible.
Check `ROCR_VISIBLE_DEVICES` and the allocation.

**The plot script reports no data.** It expects `results_*/timings.csv`
relative to the current directory — run it from this directory.

---

## 9. Cleaning up

```bash
./clean.sh              # binaries in this directory
make clean              # equivalent
../../clean.sh      # whole tree: binaries, editor backups, build debris
```

---

## 10. License and availability

*[Complete before submission.]*

- **Code:** [MIT / Apache-2.0 / BSD-3-Clause] — see `LICENSE`.
- **Measurement data** (`results_*/`, `plots_final/`): [CC BY 4.0].
- **Archived version:** https://doi.org/10.5281/zenodo.XXXXXXX
- **Third-party:** the OpenMP offload runtime derives from AMD's LLVM fork,
  licensed Apache-2.0 with LLVM exceptions. See Section 3.

## 11. Citation

*[BibTeX entry here; mirror it in `CITATION.cff` at the repository root.]*
