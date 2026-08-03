# Asynchronous Device-to-Host Bandwidth — NVIDIA H100

Benchmarks accompanying the paper *[PAPER TITLE]*.

This directory measures **asynchronous device-to-host (D-to-H) transfer
bandwidth** on an NVIDIA H100 node, comparing **OpenMP target offload**
against **native CUDA**, for transfers originating from one GPU and from two
GPUs concurrently. Transfers are issued without blocking, so that data
movement can overlap rather than serialise.

Companion directories: `../Asynchronous-test-H-to-D-Measurements/` measures
the opposite direction; `../Synchronous-test-D-to-H-Measurements/` measures
the blocking equivalent. Other vendors: `../../AMD-MI250X/` and
`../../Intel-1550/`.

---

## 1. What is measured

Two implementations of the same transfer, at two levels of GPU concurrency:

| Version | Meaning | Theoretical ceiling |
|---|---|---|
| `v1` | Asynchronous transfer from a single GPU to host | 64 GB/s (PCIe Gen5 x16) |
| `v2` | Concurrent asynchronous transfer from two GPUs to host | 128 GB/s |

Each is implemented twice — once in OpenMP target offload, once in CUDA —
giving four measured series. Buffers are arrays of `double` (8 bytes per
element).

> A four-GPU variant exists in the development history but is not part of this
> study; it is left to future work.

**Buffer sizes differ from the synchronous tests.** The asynchronous sweep
uses buffers one eighth the size of the synchronous one, so the two figures
should not be read as point-by-point comparable — see Section 6.

---

## 2. Contents

| File | Description |
|---|---|
| `openmp-1.cc`, `openmp-2.cc` | OpenMP target offload implementations (1 GPU, 2 GPUs) |
| `cuda-1.cu`, `cuda-2.cu` | Equivalent native CUDA implementations |
| `Makefile` | Build rules for both toolchains |
| `build.sh` | Slurm job: compiles all binaries and verifies they exist |
| `run.sh` | Slurm job: executes the buffer-size sweep, writes `timings.csv` |
| `submit.sh` | Submits `build.sh`, then N run jobs with an `afterok` dependency |
| `plot_h100_async_dtoh.py` | Aggregates all runs and produces the paper figure |
| `clean.sh` | Removes compiled binaries |
| `results_<jobid>_<timestamp>/` | Raw output from our runs (5 repetitions) |
| `plots_final/` | Generated figure, as it appears in the paper |

Compiled executables are not included; `build.sh` produces them. Source files
are named by framework; the binaries they produce keep the `version-N-off` /
`version-N-cuda` names used in `run.sh` and in the CSV output.

---

## 3. Requirements

### Hardware

- A node with **at least two NVIDIA H100 GPUs** (compute capability 9.0,
  `sm_90`). The `v2` measurements require two; `v1` alone needs one.
- Approximately 10 GB of host RAM — the largest buffer is 6.2e8 doubles
  (~4.96 GB), with headroom for the staging buffers used by the asynchronous
  path.
- Exclusive node access. The scripts pass `--exclusive`; shared nodes give
  unstable bandwidth figures, and asynchronous measurements are more sensitive
  to interference than blocking ones.

### Software

Versions used for the published measurements:

| Component | Version |
|---|---|
| GCC | 13.2.0 |
| CUDA | 12.8 |
| Clang/LLVM (NVPTX offload) | 18.1.8-cuda12.8 |
| Python | 3.x with `numpy`, `matplotlib` |

Clang must have NVPTX offload support compiled in — most distribution builds
do not. Verify:

```bash
clang++ --print-file-name=libomptarget.so
```

An absolute path means offload is available; a bare filename means it is not.

### Site-specific settings that must be changed

`build.sh` and `run.sh` carry Slurm directives from the machine used for the
paper:

```bash
#SBATCH --account=ehpc229      # replace with your allocation
#SBATCH --partition=acc        # replace with your GPU partition
#SBATCH --qos=acc_ehpc         # replace, or delete this line
```

The `module load` lines reflect one site's module tree and will need local
equivalents. On a system without environment modules, ensure `clang++`,
`nvcc` and the CUDA toolkit are on `PATH` and remove the `module` lines.

---

## 4. Build

```bash
sbatch build.sh          # on the cluster
```

or directly, with the toolchain already available:

```bash
make            # all four binaries
make omp        # OpenMP offload only
make cuda       # CUDA only
```

Produces:

```
version-1-off   version-2-off      # OpenMP target offload
version-1-cuda  version-2-cuda     # CUDA
```

`build.sh` checks all four exist and exits non-zero otherwise, which blocks
dependent run jobs via the `afterok` dependency.

---

## 5. Smoke test (~2 minutes)

Confirm the environment is sane before committing queue time:

```bash
make omp
./version-1-off 100000000
```

Expected: one line reporting elapsed time and bandwidth in GB/s, with a value
in the tens of GB/s. Failure here indicates an environment problem rather than
a benchmark problem — see Troubleshooting.

---

## 6. Reproducing the results

### Full pipeline

```bash
./submit.sh
```

Submits one build job followed by **five independent run jobs**, each
depending on the build succeeding. Five repetitions is what the paper reports;
adjust `N_RUNS` at the top of `submit.sh`.

Monitor with `squeue -u $USER`.

Buffer sizes swept, in elements (`double`, 8 bytes each):

```
117500000  217500000  318750000  418750000  520000000  620000000
```

These are one eighth of the sizes used in the synchronous directories, chosen
so the asynchronous path is exercised in the regime where overlap is
measurable. Absolute bandwidth figures are therefore not directly comparable
between the synchronous and asynchronous tests at a given index; compare the
OpenMP-versus-CUDA relationship within each figure instead.

Each run job is capped at 30 minutes of wall clock, so the full pipeline costs
roughly 2.5 node-hours plus queue wait.

### Figure generation

```bash
python plot_h100_async_dtoh.py
```

This globs **all** `results_*/timings.csv` directories, averages each
(framework, version, size) triple across repetitions, and writes the figure to
`plots_final/`.

Because the raw `results_*` directories are included in this artifact, the
figure can be regenerated **without access to H100 hardware** — running the
plot script alone reproduces the published figure.

To regenerate every H100 figure at once, run `../all-plots.sh`.

---

## 7. Expected output

Each run job writes `results_<jobid>_<timestamp>/` containing:

| File | Contents |
|---|---|
| `timings.csv` | One row per measurement |
| `run.log` | Full job stdout |
| `skipped.log` | Missing binaries and errored runs |

`timings.csv` schema:

```
framework,version,N,elapsed_sec,bandwidth_gbs,status
```

`framework` is `omp_off` or `cuda`; `version` is `v1` or `v2`; `status` is
`ok`, `timeout`, `runtime_error`, or `missing_binary`. Every row should read
`ok` in a healthy run — the plot script silently discards any row that does
not.

### Claim map

| Paper claim | Command | Cost | Confirming output |
|---|---|---|---|
| *[Figure N]*: OpenMP offload vs CUDA, async D-to-H on H100 | `python plot_h100_async_dtoh.py` | seconds, no GPU needed | `plots_final/[filename].pdf` |
| *[Figure N]*, measured from scratch | `./submit.sh`, then the above | ~2.5 node-h on 2×H100 | same |
| *[Claim about the mean CUDA vs OpenMP gap]* | as above | — | summary box in the figure, in GB/s |
| *[Any claim comparing D-to-H against H-to-D, or async against sync]* | also run the corresponding directories | ~5–10 node-h total | the respective figures |

*[Replace the bracketed labels with the actual figure and table numbers.]*

### Tolerance

Bandwidth varies run to run and machine to machine. Deviations of a few
percent from the published values are normal and do not constitute a failed
reproduction — the claim under test is the relationship between OpenMP offload
and CUDA, not the absolute figures. Different GPU generations, PCIe
topologies, or CUDA/Clang versions will shift the numbers substantially.
Asynchronous measurements are inherently noisier than synchronous ones, so
expect somewhat wider run-to-run spread here.

Regenerating the figure from the included `results_*` data is deterministic
and should match the published figure.

---

## 8. Troubleshooting

**Every row reads `missing_binary`.** The build stage failed; inspect
`build-<jobid>.out`.

**`libomptarget.so` not found at runtime.** The Clang runtime path is not
resolvable. The `Makefile` embeds an rpath, so this usually means a later
`module load` overwrote the environment.

**Rows read `timeout`.** The per-measurement limit is `TIMEOUT_SEC=300` in
`run.sh`. Slower host interconnects may need more at the largest sizes.

**Out of memory at large sizes.** Trim the `Ns` array in `run.sh` and state
which sizes were omitted in any comparison.

**`v2` results are missing or match `v1`.** Fewer than two GPUs were visible.
Check `CUDA_VISIBLE_DEVICES` and the node allocation.

**Asynchronous bandwidth matches the synchronous figures.** The transfers are
not actually overlapping. Check that the host destination buffer is
page-locked, since transfers into pageable memory fall back to blocking
behaviour.

**Bandwidth is low or erratic.** Confirm exclusive allocation and that nothing
else shared the GPUs or the PCIe root complex.

**The plot script reports no data.** It expects `results_*/timings.csv`
relative to the current directory — run it from this directory.

---

## 9. Cleaning up

```bash
./clean.sh              # binaries in this directory
make clean              # equivalent
../../clean.sh   # whole tree: binaries, editor backups, build debris
```

---

## 10. License and availability

*[Complete before submission.]*

- **Code:** [MIT / Apache-2.0 / BSD-3-Clause] — see `LICENSE`.
- **Measurement data** (`results_*/`, `plots_final/`): [CC BY 4.0].
- **Archived version:** https://doi.org/10.5281/zenodo.XXXXXXX
- **Access restrictions:** none.

## 11. Citation

*[BibTeX entry here; mirror it in `CITATION.cff` at the repository root.]*
