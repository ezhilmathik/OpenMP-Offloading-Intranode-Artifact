# Peer-to-Peer Bandwidth — AMD MI250X

Benchmarks accompanying the paper *[PAPER TITLE]*.

This directory measures **GCD-to-GCD peer-to-peer (P2P) transfer bandwidth**
on an AMD Instinct MI250X node, comparing **OpenMP target offload** against
**native HIP**. Unlike the host-transfer directories, transfers here traverse
the inter-GPU Infinity Fabric links rather than the host link.

Companion directories in `../`: `Synchronous-test-H-to-D-Measurements/`,
`Synchronous-test-D-to-H-Measurements/`,
`Asynchronous-test-H-to-D-Measurements/`,
`Asynchronous-test-D-to-H-Measurements/`.
Other vendors: `../../NVIDIA-H100/` and `../../Intel-1550/`.

---

## 1. What is measured

Peer-to-peer transfers between two GCDs, in two configurations:

| Version | Configuration | Theoretical ceiling |
|---|---|---|
| `v1` | Unidirectional: GCD A → GCD B | 100 GB/s |
| `v2` | Bidirectional: GCD A ↔ GCD B, both directions driven concurrently | 200 GB/s |

Both configurations use the same pair of devices; `v2` differs from `v1` in
direction, not in the number of GCDs involved. Each is implemented twice —
once in OpenMP target offload, once in HIP — giving four measured series.
Buffers are arrays of `double` (8 bytes per element).

The ceilings are **peer-link (Infinity Fabric) bandwidths**, not host-link
ones. They are configured in `THEORETICAL_BW` at the top of
`plot_mi250x_p2p.py` and drawn as dashed reference lines.

### Device selection — read this before comparing numbers

`run.sh` sets

```bash
export ROCR_VISIBLE_DEVICES=0,2,4,6
```

so only one GCD per physical MI250X card is visible. The measured pair
therefore sits on **two different packages**, and the transfer crosses the
inter-package Infinity Fabric rather than the fast on-package GCD-to-GCD link.
Measuring two GCDs of the same package would give substantially different
numbers.

*[Confirm and state the number of Infinity Fabric links between the selected
pair on the node used — the 100 GB/s figure follows from it, and a node with a
different link count has a different ceiling.]*

> **Not part of this study.** Two further configurations exist in the
> repository and are left to future work:
> - `v4` — gather: one GCD receives concurrently from three peers
>   (theoretical ceiling 300 GB/s).
> - `v4_4` — all-to-all: every GCD sends to and receives from all three peers
>   simultaneously.
>
> Their sources (`openmp-4.cc`, `hip-4.hip`, `openmp-4-4.cc`, `hip-4-4.hip`)
> and build rules remain in place, but they are not executed by `run.sh` and
> not plotted. To measure them, restore the corresponding `run_bench` calls in
> `run.sh` and add the versions to `VERSIONS_TO_PLOT` in `plot_mi250x_p2p.py`.

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
| `plot_mi250x_p2p.py` | Aggregates all runs and produces the paper figure |
| `clean.sh` | Removes compiled binaries |
| `results_<jobid>_<timestamp>/` | Raw output from our runs (5 repetitions) |
| `plots_final/` | Generated figure, as it appears in the paper |
| `openmp-4.cc`, `hip-4.hip`, `openmp-4-4.cc`, `hip-4-4.hip` | Four-GPU configurations — future work, not used |

Compiled executables are not included; `build.sh` produces them.

---

## 3. Requirements

### Hardware

- An AMD Instinct MI250X node (`gfx90a`) with at least two cards connected by
  Infinity Fabric peer links.
- Approximately 10 GB of host RAM for buffer setup; the transfers themselves
  are device-resident. The largest buffer is 1.24e9 doubles (~9.92 GB) per
  device.
- Exclusive node access. Peer-link contention from other jobs would
  invalidate the measurement.

### Software

Measurements were taken on **LUMI**:

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

| Setting | Effect |
|---|---|
| `HSA_ENABLE_SDMA=0` | Disables the DMA engines, so transfers go through blit kernels instead. This changes measured bandwidth substantially; results are not comparable to runs without it. |
| `OMP_TARGET_OFFLOAD=mandatory` | Makes OpenMP fail rather than silently fall back to host execution. |
| `ROCR_VISIBLE_DEVICES=0,2,4,6` | One GCD per physical card — see Section 1. |

*[Add one line on why SDMA is disabled — it is a methodological choice and
reviewers will ask.]*

### OpenMP offload runtime — required

The OpenMP measurements do **not** use the runtime supplied by the
`rocm/6.4.4` module. `run.sh` preloads a separate build:

```
$OMPTARGET_P2P_LIB/libomp.so
$OMPTARGET_P2P_LIB/libomptarget.so.19.0git
```

which reports itself as:

```
AMD clang version 19.0.0git
(https://github.com/RadeonOpenCompute/llvm-project roc-6.4.4 25224
 d366fa84f3fdcbd4b10847ebd5db572ae12a34fb)
```

*[REQUIRED BEFORE SUBMISSION — describe how to obtain this runtime. It reports
the same ROCm version as the loaded module, so it is not simply a stock 6.4.4
build. State whether it is patched, what the patch does, and either include
the patch here or give a precise recipe. This matters most in this directory:
the runtime's directory is named `omptarget-p2p`, so peer-to-peer support is
likely exactly what it provides. Without it the OpenMP P2P results cannot be
reproduced.]*

Point the scripts at it with:

```bash
OMPTARGET_P2P_LIB=/path/to/lib sbatch run.sh
```

**The HIP measurements do not use this runtime** and need only the standard
ROCm installation.

### Site-specific settings that must be changed

```bash
#SBATCH --account=project_465003145   # replace with your allocation
#SBATCH --partition=standard-g        # replace with your GPU partition
```

---

## 4. Build

```bash
sbatch build.sh          # on the cluster
```

or directly, with ROCm available:

```bash
make            # all binaries
make omp        # OpenMP offload only
make hip        # HIP only
```

`build.sh` verifies that `version-1-off`, `version-2-off`, `version-1-hip` and
`version-2-hip` exist, and exits non-zero otherwise, which blocks dependent
run jobs via the `afterok` dependency.

---

## 5. Smoke test (~2 minutes)

```bash
make omp
export HSA_ENABLE_SDMA=0
export ROCR_VISIBLE_DEVICES=0,2,4,6
LD_PRELOAD="$OMPTARGET_P2P_LIB/libomp.so:$OMPTARGET_P2P_LIB/libomptarget.so.19.0git" \
    ./version-1-off 100000000
```

Expected: one line reporting elapsed time and bandwidth in GB/s, well above
the 36 GB/s host-link ceiling if the peer link is being used. A result near or
below that suggests the transfer was staged through the host.

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
python3 plot_mi250x_p2p.py
```

Globs all `results_*/timings.csv`, averages each (framework, version, size)
triple across the five repetitions, writes `plots_final/mi250x_p2p.pdf`, and
prints the mean and maximum HIP-versus-OpenMP gap per version.

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
117500000  217500000  318750000  418750000  520000000  1240000000
```

These are smaller than the host-transfer sweep because both endpoints are
device-resident and must fit alongside each other in GCD memory. They match
the NVIDIA H100 P2P directory, so the two vendors' P2P figures are directly
comparable.

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

The included data also contains `v4` and `v4_4` rows from earlier runs. These
are not plotted and are not part of this study; see Section 1.

### Claim map

| Paper claim | Command | Cost | Confirming output |
|---|---|---|---|
| *[Figure N]*: OpenMP offload vs HIP, P2P on MI250X | `python3 plot_mi250x_p2p.py` | seconds, no GPU needed | `plots_final/mi250x_p2p.pdf` |
| *[Figure N]*, measured from scratch | `./submit.sh`, then the above | ~5 node-h | same |
| *[Claim about the mean HIP vs OpenMP P2P gap]* | as above | — | summary box in the figure; also printed to stdout |
| *[Any cross-vendor P2P claim]* | also run `../../NVIDIA-H100/P2P-Measurements/` | — | both figures |

*[Replace the bracketed labels with the actual figure and table numbers.]*

### Tolerance

Bandwidth varies run to run and machine to machine. Deviations of a few
percent from the published values are normal — the claim under test is the
relationship between OpenMP offload and HIP, not the absolute figures.

P2P results are more topology-sensitive than host transfers: the ceiling
depends on how many Infinity Fabric links connect the selected GCD pair, which
varies between nodes and between GCD selections on the same node. Report the
topology alongside any comparison.

Three things change the numbers enough that altering them does not constitute
a reproduction: the OpenMP runtime described in Section 3,
`HSA_ENABLE_SDMA=0`, and the `ROCR_VISIBLE_DEVICES` selection.

---

## 8. Troubleshooting

**Bandwidth is near the host-link rate rather than the peer-link rate.** Peer
access was not used and transfers were staged through the host. For OpenMP,
this is the most likely symptom of the wrong runtime being loaded — see
Section 3.

**OpenMP results differ markedly from the published figures.** Check
`OMPTARGET_P2P_LIB`, and that `HSA_ENABLE_SDMA` is 0.

**`libomptarget.so.19.0git` not found.** `LD_PRELOAD` fails silently on a
missing file, so the run would proceed against the module's runtime and
produce different numbers. Verify the path before submitting.

**OpenMP reports host execution or aborts.** `OMP_TARGET_OFFLOAD=mandatory`
turns a silent host fallback into a failure. Check that the binary was built
with `--offload-arch=gfx90a` and that GPUs are visible.

**Every row reads `missing_binary`.** The build stage failed; inspect
`build-<jobid>.out`.

**`v2` results are missing or match `v1`.** The two directions are not running
concurrently, or fewer than two GCDs were visible. Check
`ROCR_VISIBLE_DEVICES` and the allocation.

**The plot script reports no data.** It expects `results_*/timings.csv`
relative to the current directory — run it from this directory.

---

## 9. Cleaning up

```bash
./clean.sh              # binaries in this directory
make clean              # equivalent
../../clean.sh          # whole tree: binaries, editor backups, build debris
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
