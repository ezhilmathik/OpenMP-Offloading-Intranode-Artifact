# Asynchronous Device-to-Host Bandwidth — Intel Data Center GPU Max 1550

Benchmarks accompanying the paper *[PAPER TITLE]*.

This directory measures **asynchronous device-to-host (D-to-H) transfer
bandwidth** on an Intel Data Center GPU Max 1550 (Ponte Vecchio) node,
comparing **OpenMP target offload** against **native SYCL**.

Companion directories in `../`: `Synchronous-test-H-to-D-Measurements/`,
`Asynchronous-test-H-to-D-Measurements/`,
`Asynchronous-test-D-to-H-Measurements/`, `P2P-Measurements/`.
Other vendors: `../../NVIDIA-H100/` and `../../AMD-MI250X/`.

---

## 1. What is measured

| Version | Configuration | Theoretical ceiling |
|---|---|---|
| `v1` | Asynchronous transfer from a single device to host | 64 GB/s |
| `v2` | Concurrent asynchronous transfer from two devices to host | 128 GB/s |

Transfers are issued without blocking, so data movement can overlap rather
than serialise. Each configuration is implemented twice — once in OpenMP
target offload, once in SYCL — giving four measured series. Buffers are arrays
of `double` (8 bytes per element).

Buffer sizes match the synchronous directories, so the synchronous and
asynchronous figures are directly comparable.

The ceilings assume PCIe Gen5 x16 per card, 64 GB/s each way. *[Verify against
the host used; the plot script's `THEORETICAL_BW` carries the same caveat.]*

### What counts as a device

A Max 1550 card contains **two stacks**. Level Zero can present either the
card or the stack as a device, controlled by `ZE_FLAT_DEVICE_HIERARCHY`:

| Hierarchy | A device is | Mask used | Meaning |
|---|---|---|---|
| `COMPOSITE` (default) | a whole card | `0,1,2,3` | 4 cards, one full host link each |
| `FLAT` | one stack | `0,2,4,6` | one stack from each of 4 cards |

`COMPOSITE` is the default and is what the published figures use. It is the
intent-preserving analogue of the AMD runs, where the mask took one GCD per
physical card so no two devices shared a host link. Select the other with:

```bash
sbatch run.sh FLAT
```

Because "four devices" means different silicon under the two hierarchies, the
CSV carries a seventh column recording which was used — see Section 7.

> A four-device variant (`v4`) exists in the repository and is left to future
> work. Its source and build rule remain in place, but it is not measured or
> plotted.

---

## 2. Contents

| File | Description |
|---|---|
| `openmp-1.cc`, `openmp-2.cc` | OpenMP target offload implementations |
| `sycl-1.cpp`, `sycl-2.cpp` | Equivalent native SYCL implementations |
| `Makefile` | Build rules for both toolchains (`icpx`) |
| `build.sh` | Slurm job: compiles the binaries and verifies they exist |
| `run.sh` | Slurm job: executes the buffer-size sweep, writes `timings.csv` |
| `submit.sh` | Submits `build.sh`, then N run jobs with an `afterok` dependency |
| `plot_pvc_async_dtoh.py` | Aggregates all runs and produces the paper figure |
| `clean.sh` | Removes compiled binaries |
| `results_<HIER>_<jobid>_<timestamp>/` | Raw output from our runs (5 repetitions) |
| `plots_final/` | Generated figure, as it appears in the paper |
| `openmp-4.cc`, `sycl-4.cpp` | Four-device variant — future work, not used |

Compiled executables are not included; `build.sh` produces them.

---

## 3. Requirements

### Hardware

- A node with Intel Data Center GPU Max 1550 cards; at least two are needed
  for the `v2` measurements. Our runs used four.
- Approximately 40 GB of host RAM — the largest buffer is 4.96e9 doubles
  (~39.7 GB).
- Exclusive node access.

A 1550 card has 128 GB of HBM (64 GB per stack). Under `COMPOSITE` the largest
buffers fit comfortably; under `FLAT` a device is a 64 GB stack, so the top one
or two sizes may exhaust memory. That surfaces as `status=runtime_error` rather
than a wrong number.

### Software

Measurements were taken at **LRZ**:

```bash
module load slurm_setup
module load intel-toolkit
```

| Component | Version |
|---|---|
| Intel oneAPI DPC++/C++ Compiler | 2024.1.0 (2024.1.0.20240308) |
| Level Zero driver | 1.3.27642 |
| Python | 3 with numpy and matplotlib (`../../requirements.txt`) |

Unlike the AMD MI250X directories, **no custom OpenMP runtime is required**.
Level Zero exposes peer access natively and `icx`'s `libomptarget` uses it, so
there is nothing to preload.

### Runtime environment settings

`run.sh` exports the following:

| Setting | Effect |
|---|---|
| `OMP_TARGET_OFFLOAD=mandatory` | OpenMP fails rather than silently falling back to the host. |
| `ZE_FLAT_DEVICE_HIERARCHY` | `COMPOSITE` or `FLAT` — see Section 1. |
| `ZE_AFFINITY_MASK` | One device per physical card, set according to the hierarchy. |
| `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1` | Stable PCI-ordered device numbering. |
| `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` | Hides the duplicate OpenCL GPU entries. |
| `SYCL_CACHE_PERSISTENT=1` | Persists the SPIR-V → PVC JIT cache, so the first run of a binary does not pay codegen inside the timed region. |

**Deliberately not set:** `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0`, the nearest
analogue of the `HSA_ENABLE_SDMA=0` used on MI250X. It affects only SYCL, not
the OpenMP offload plugin, so setting it would bias the very comparison this
benchmark exists to make; and Intel documents these variables as
development-only with semantics subject to change between releases. The
reasoning is recorded in full at the bottom of `run.sh`.

### Site-specific settings that must be changed

```bash
#SBATCH --account=pn67qe      # replace with your allocation
#SBATCH --partition=general   # replace with your GPU partition
```

---

## 4. Build

```bash
sbatch build.sh          # on the cluster
```

or directly, with oneAPI available:

```bash
make            # all four binaries
make omp        # OpenMP offload only
make sycl       # SYCL only
```

Produces:

```
version-1-off    version-2-off      # OpenMP target offload
version-1-sycl   version-2-sycl     # SYCL
```

SYCL builds JIT to SPIR-V by default. For ahead-of-time codegen — slower build,
faster startup — uncomment the `-fsycl-targets=spir64_gen -Xs "-device pvc"`
line in the `Makefile`.

---

## 5. Smoke test (~2 minutes)

```bash
make omp
export OMP_TARGET_OFFLOAD=mandatory
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
./version-1-off 100000000
```

Expected: one line reporting elapsed time and bandwidth in GB/s. With
`OMP_TARGET_OFFLOAD=mandatory` a missing or misconfigured device produces a
hard failure rather than a quiet host-side result.

`sycl-ls` should list the Max 1550 devices under `ext_oneapi_level_zero:gpu`.

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
python3 plot_pvc_async_dtoh.py
```

Globs all `results_*/timings.csv`, averages each (framework, version, size)
triple across the five repetitions, writes `plots_final/pvc_async_dtoh.pdf`, and
prints the mean and maximum SYCL-versus-OpenMP gap per version.

Because the raw results are included, this needs **no GPU and no cluster
access**. `../all-plots.sh` regenerates every PVC figure at once.

### Re-measuring from scratch

```bash
./submit.sh
```

One build job followed by five independent run jobs, each depending on the
build succeeding. Adjust `N_RUNS` at the top of `submit.sh`.

To measure the other device hierarchy:

```bash
sbatch run.sh FLAT
```

Results land in a separately named directory, so the two hierarchies never mix.

Buffer sizes swept, in elements (`double`, 8 bytes each):

```
940000000  1740000000  2550000000  3350000000  4160000000  4960000000
```

These match the H100 and MI250X directories, so the vendor figures are
directly comparable.

Note that the AMD and NVIDIA versions of this array originally contained
commas, which bash does not treat as separators; the resulting trailing commas
produced a phantom extra field in those CSVs. The Intel scripts never had this
problem, and it has since been corrected in the other vendors' scripts.

---

## 7. Expected output

Each run job writes `results_<HIER>_<jobid>_<timestamp>/` containing:

| File | Contents |
|---|---|
| `timings.csv` | One row per measurement |
| `run.log` | Full job stdout |
| `skipped.log` | Missing binaries and errored runs |
| `env.txt` | Host, date, compiler version, hierarchy, affinity mask, and `sycl-ls` output |

`env.txt` records exactly what produced each result, which is what makes a
given `results_*` directory self-describing. A typical entry:

```
host=i20r08c02s02
hierarchy=COMPOSITE
ze_affinity_mask=0,1,2,3
compiler=Intel(R) oneAPI DPC++/C++ Compiler 2024.1.0 (2024.1.0.20240308)
```

`timings.csv` schema — note the **seventh column**, which the other two vendors
do not have:

```
framework,version,N,elapsed_sec,bandwidth_gbs,status,hierarchy
```

`framework` is `omp_off` or `sycl`; `version` is `v1` or `v2`; `status` is
`ok`, `timeout`, `runtime_error`, or `missing_binary`; `hierarchy` is
`COMPOSITE` or `FLAT`. The hierarchy column exists because the same device
count refers to different silicon under the two settings, so a bandwidth number
is not interpretable without it.

Every row should read `ok` in a healthy run — the plot script silently discards
any row that does not.

### Claim map

| Paper claim | Command | Cost | Confirming output |
|---|---|---|---|
| *[Figure N]*: OpenMP offload vs SYCL, async D-to-H on PVC | `python3 plot_pvc_async_dtoh.py` | seconds, no GPU needed | `plots_final/pvc_async_dtoh.pdf` |
| *[Figure N]*, measured from scratch | `./submit.sh`, then the above | ~5 node-h | same |
| *[Claim about the mean SYCL vs OpenMP gap]* | as above | — | summary box in the figure; also printed to stdout |
| *[Any cross-vendor claim]* | also run `../../NVIDIA-H100/` and `../../AMD-MI250X/` | — | the respective figures |

*[Replace the bracketed labels with the actual figure and table numbers.]*

### Tolerance

Bandwidth varies run to run and machine to machine. Deviations of a few
percent from the published values are normal — the claim under test is the
relationship between OpenMP offload and SYCL, not the absolute figures.

Comparing results taken under different `ZE_FLAT_DEVICE_HIERARCHY` settings is
not a reproduction; check the `hierarchy` column before comparing anything.

Asynchronous measurements are inherently noisier than synchronous ones, so
expect somewhat wider run-to-run spread here. Note also that these runs use
Level Zero's default copy-engine behaviour, whereas the AMD MI250X
asynchronous runs were taken with the DMA engines disabled
(`HSA_ENABLE_SDMA=0`); this asymmetry is deliberate and is explained in the
notes at the bottom of `run.sh`.

---

## 8. Troubleshooting

**OpenMP aborts or reports host execution.** `OMP_TARGET_OFFLOAD=mandatory`
turns a silent host fallback into a failure. Check that the binary was built
with `-fopenmp-targets=spir64` and that `sycl-ls` sees the GPUs.

**Numbers differ from the published figures by a large factor.** Check the
`hierarchy` column: `FLAT` and `COMPOSITE` results are not comparable.

**Rows read `runtime_error` at the largest sizes under `FLAT`.** A device is a
64 GB stack in that mode, so the top sizes may not fit. Expected; either use
`COMPOSITE` or trim the `Ns` array and say which sizes were omitted.

**The first SYCL run of a binary is slow.** JIT codegen. `SYCL_CACHE_PERSISTENT=1`
handles this after the first run; building AOT avoids it entirely.

**Asynchronous bandwidth matches the synchronous figures.** The transfers are
not overlapping. Note that no copy-engine controls are set here — see
Section 3 — so this reflects the Level Zero default behaviour.

**Every row reads `missing_binary`.** The build stage failed; inspect
`build-<jobid>.out`.

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
- **Access restrictions:** none.

## 11. Citation

*[BibTeX entry here; mirror it in `CITATION.cff` at the repository root.]*
