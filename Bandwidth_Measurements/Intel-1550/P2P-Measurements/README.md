# Peer-to-Peer Bandwidth — Intel Data Center GPU Max 1550

Benchmarks accompanying the paper *[PAPER TITLE]*.

This directory measures **GPU-to-GPU peer-to-peer (P2P) transfer bandwidth**
on an Intel Data Center GPU Max 1550 (Ponte Vecchio) node, comparing **OpenMP
target offload** against **native SYCL**. Unlike the host-transfer
directories, transfers here traverse the inter-GPU Xe Link fabric rather than
PCIe to the host.

Companion directories in `../`: `Synchronous-test-H-to-D-Measurements/`,
`Synchronous-test-D-to-H-Measurements/`,
`Asynchronous-test-H-to-D-Measurements/`,
`Asynchronous-test-D-to-H-Measurements/`.
Other vendors: `../../NVIDIA-H100/` and `../../AMD-MI250X/`.

---

## 1. What is measured

Peer-to-peer transfers between two devices, in two configurations:

| Version | Configuration | Theoretical ceiling |
|---|---|---|
| `v1` | Unidirectional: device A → device B | 106 GB/s |
| `v2` | Bidirectional: device A ↔ device B, both directions concurrently | 212 GB/s |

Both configurations use the same pair of devices; `v2` differs from `v1` in
direction, not in the number of devices involved. Each is implemented twice —
once in OpenMP target offload, once in SYCL — giving four measured series.
Buffers are arrays of `double` (8 bytes per element).

The ceilings are **peer-link (Xe Link) bandwidths**, not host-link ones. They
are configured in `THEORETICAL_BW` at the top of `plot_pvc_p2p.py` and drawn
as dashed reference lines.

### What counts as a device — and why both runtimes must agree

A Max 1550 card contains **two stacks**. Two independent controls decide
whether a "device" means a card or a stack, and **they must be set
consistently**:

| Control | Applies to | `COMPOSITE` | `FLAT` |
|---|---|---|---|
| `ZE_FLAT_DEVICE_HIERARCHY` + `ZE_AFFINITY_MASK` | Level Zero, i.e. SYCL | card, mask `0,1,2,3` | stack, mask `0,2,4,6` |
| `LIBOMPTARGET_DEVICES` | OpenMP offload only | `device` | `subdevice` |

`ZE_*` configures the Level Zero driver, which is what SYCL sees. OpenMP has
its own mapping control that the `ZE_*` variables do not affect. If OpenMP
resolved to stacks while SYCL used cards, `omp_off v1` and `sycl v1` would be
copying over different links and the comparison would be meaningless. Both are
set explicitly in `run.sh`, and a preflight check enforces agreement.

`COMPOSITE` is the default and is what the published figures use. Select the
other with `sbatch run.sh FLAT`.

> **Not part of this study.** Two further configurations exist in the
> repository and are left to future work:
> - `v4` — one device receiving concurrently from three peers (theoretical
>   ceiling 318 GB/s).
> - A 4-to-4 all-to-all OpenMP variant (`openmp-4-4.cc`); no SYCL counterpart
>   has been written.
>
> Their sources and build rules remain in place, but they are not executed by
> `run.sh` and not plotted. Note that `openmp-4.cc` has a **known bug**: its
> three transfers serialise rather than overlapping, so it reports
> approximately the same bandwidth as `v1`. This is documented in the notes at
> the bottom of `run.sh` and must be fixed before `v4` is used for anything.

---

## 2. Contents

| File | Description |
|---|---|
| `openmp-1.cc`, `openmp-2.cc` | OpenMP target offload implementations |
| `sycl-1.cpp`, `sycl-2.cpp` | Equivalent native SYCL implementations |
| `Makefile` | Build rules for both toolchains (`icpx`) |
| `build.sh` | Slurm job: compiles the binaries and verifies they exist |
| `run.sh` | Slurm job: preflight check, then the buffer-size sweep |
| `submit.sh` | Submits `build.sh`, then N run jobs with an `afterok` dependency |
| `plot_pvc_p2p.py` | Aggregates all runs and produces the paper figure |
| `topology-test.cc` | Diagnostic: reports the device topology and peer relationships |
| `clean.sh` | Removes compiled binaries |
| `results_<HIER>_<jobid>_<timestamp>/` | Raw output from our runs (5 repetitions) |
| `plots_final/` | Generated figure, as it appears in the paper |
| `openmp-4.cc`, `sycl-4.cpp`, `openmp-4-4.cc` | Future work — see Section 1 |

Compiled executables are not included; `build.sh` produces them.

`topology-test.cc` is worth building and running first on any new machine: it
reports what the runtime thinks the device layout is, which determines whether
your node is comparable to the one used here.

---

## 3. Requirements

### Hardware

- A node with at least two Intel Data Center GPU Max 1550 cards connected by
  Xe Link. Our runs used four.
- Approximately 10 GB of host RAM for buffer setup; the transfers themselves
  are device-resident. The largest buffer is 1.24e9 doubles (~9.92 GB) per
  device.
- Exclusive node access. Peer-link contention from other jobs would invalidate
  the measurement.

### Software

Measurements were taken at **LRZ**:

```bash
module load slurm_setup
module load intel-toolkit/2024.1.0
```

| Component | Version |
|---|---|
| Intel oneAPI DPC++/C++ Compiler | 2024.1.0 (2024.1.0.20240308) |
| Level Zero driver | 1.3.27642 |
| Python | 3 with numpy and matplotlib (`../../requirements.txt`) |

**The toolkit version is pinned deliberately.** The default device-hierarchy
behaviour has changed between oneAPI releases — older drivers defaulted to
`COMPOSITE` with implicit scaling, newer ones to `FLAT` — so a floating
`module load intel-toolkit` would make runs non-reproducible in precisely the
dimension under investigation.

Unlike the AMD MI250X directories, **no custom OpenMP runtime is preloaded**.
Level Zero exposes peer access natively and `icx`'s `libomptarget` is expected
to use it.

### Runtime environment settings

| Setting | Effect |
|---|---|
| `OMP_TARGET_OFFLOAD=mandatory` | OpenMP fails rather than silently falling back to the host. |
| `ZE_FLAT_DEVICE_HIERARCHY` | `COMPOSITE` or `FLAT` — see Section 1. |
| `ZE_AFFINITY_MASK` | One device per physical card, set according to the hierarchy. |
| `LIBOMPTARGET_DEVICES` | `device` or `subdevice`, kept consistent with the hierarchy. |
| `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1` | Stable PCI-ordered device numbering. |
| `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` | Hides the duplicate OpenCL GPU entries. |
| `OMP_NUM_THREADS=8` | Set explicitly rather than left to the runtime under `--exclusive`. |
| `SYCL_CACHE_PERSISTENT=1` | Persists the SPIR-V → PVC JIT cache. |

**Deliberately not set:** `SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=0`, the nearest
analogue of the `HSA_ENABLE_SDMA=0` used on MI250X. It affects only SYCL, not
the OpenMP offload plugin, so setting it would bias the very comparison this
benchmark exists to make; and Intel documents these variables as
development-only with semantics subject to change between releases. The
reasoning is recorded in full at the bottom of `run.sh`.

### Preflight device check

Before measuring, `run.sh` compiles two small probes — one calling
`omp_get_num_devices()`, one enumerating SYCL GPU devices — and refuses to run
the sweep unless they agree with each other **and** with the number of entries
in `ZE_AFFINITY_MASK`. If OpenMP reports eight devices while SYCL reports
four, OpenMP is treating stacks as devices and every "N devices" label would
mean something different for the two frameworks.

The probe results are written to `probe.txt` in each results directory.

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
make            # all binaries
make omp        # OpenMP offload only
make sycl       # SYCL only
```

`build.sh` verifies that `version-1-off`, `version-2-off`, `version-1-sycl`
and `version-2-sycl` exist, and exits non-zero otherwise, which blocks
dependent run jobs via the `afterok` dependency.

SYCL builds JIT to SPIR-V by default. For ahead-of-time codegen, uncomment the
`-fsycl-targets=spir64_gen -Xs "-device pvc"` line in the `Makefile`.

---

## 5. Smoke test (~2 minutes)

```bash
make omp
export OMP_TARGET_OFFLOAD=mandatory
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export ZE_AFFINITY_MASK=0,1,2,3
export LIBOMPTARGET_DEVICES=device
./version-1-off 100000000
```

Expected: one line reporting elapsed time and bandwidth in GB/s, above the
64 GB/s host-link ceiling if the peer link is being used.

`sycl-ls` should list the Max 1550 devices under `ext_oneapi_level_zero:gpu`,
and the count should match the affinity mask.

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
python3 plot_pvc_p2p.py
```

Globs all `results_*/timings.csv`, averages each (framework, version, size)
triple across the five repetitions, writes `plots_final/pvc_p2p.pdf`, and
prints the mean and maximum SYCL-versus-OpenMP gap per version.

Because the raw results are included, this needs **no GPU and no cluster
access**. `../all-plots.sh` regenerates every PVC figure at once.

### Re-measuring from scratch

```bash
./submit.sh
```

One build job followed by five independent run jobs, each depending on the
build succeeding. Adjust `N_RUNS` at the top of `submit.sh`.

`run.sh` takes two optional positional arguments:

```bash
sbatch run.sh [HIER] [DIAG]
#   HIER = COMPOSITE (default) | FLAT
#   DIAG = 0 (default) | 1
```

Results land in a directory named after the hierarchy, so the two never mix.

Buffer sizes swept, in elements (`double`, 8 bytes each):

```
117500000  217500000  318750000  418750000  520000000  1240000000
```

These are smaller than the host-transfer sweep because both endpoints are
device-resident. They match the H100 and MI250X P2P directories, so the three
vendors' P2P figures are directly comparable.

### Diagnostics

`DIAG=1` enables `LIBOMPTARGET_PLUGIN_PROFILE` and `LIBOMPTARGET_INFO`, giving
a per-engine copy profile and the runtime's data-transfer decisions. Profiling
perturbs the timings, so **runs taken with `DIAG=1` must not be used for
reported numbers**; `run.sh` prints a warning to that effect.

---

## 7. Expected output

Each run job writes `results_<HIER>_<jobid>_<timestamp>/` containing:

| File | Contents |
|---|---|
| `timings.csv` | One row per measurement |
| `run.log` | Full job stdout, including the preflight result |
| `skipped.log` | Missing binaries and errored runs |
| `env.txt` | Host, date, compiler version, hierarchy, affinity mask, `LIBOMPTARGET_DEVICES`, thread count, and `sycl-ls` output |
| `probe.txt` | Device counts reported by OpenMP and by SYCL |

Together, `env.txt` and `probe.txt` make each results directory
self-describing: they record not only what was run but that both runtimes
agreed on the hardware they were running on.

`timings.csv` schema — note the **seventh column**, which the other two
vendors do not have:

```
framework,version,N,elapsed_sec,bandwidth_gbs,status,hierarchy
```

`framework` is `omp_off` or `sycl`; `version` is `v1` or `v2`; `hierarchy` is
`COMPOSITE` or `FLAT`; `status` is one of:

| Status | Meaning |
|---|---|
| `ok` | Measurement completed and verified |
| `timeout` | Exceeded `TIMEOUT_SEC` |
| `runtime_error` | Non-zero exit |
| `missing_binary` | Binary not built |
| `verify_failed` | The binary reported a verification failure — the transfer did not do what it claims, so the bandwidth is measuring the wrong thing |

`verify_failed` exists only in this directory; rows carrying it are excluded
from the figure rather than being treated as valid measurements.

### Claim map

| Paper claim | Command | Cost | Confirming output |
|---|---|---|---|
| *[Figure N]*: OpenMP offload vs SYCL, P2P on PVC | `python3 plot_pvc_p2p.py` | seconds, no GPU needed | `plots_final/pvc_p2p.pdf` |
| *[Figure N]*, measured from scratch | `./submit.sh`, then the above | ~5 node-h | same |
| *[Claim about the SYCL vs OpenMP P2P gap]* | as above | — | summary box in the figure; also printed to stdout |
| *[Any cross-vendor P2P claim]* | also run the H100 and MI250X P2P directories | — | the respective figures |

*[Replace the bracketed labels with the actual figure and table numbers.]*

### Tolerance

Bandwidth varies run to run and machine to machine. Deviations of a few
percent from the published values are normal — the claim under test is the
relationship between OpenMP offload and SYCL, not the absolute figures.

Three things change the numbers enough that altering them does not constitute
a reproduction: the device hierarchy (`COMPOSITE` vs `FLAT`), the oneAPI
version, and the node's Xe Link topology. Check the `hierarchy` column and
`env.txt` before comparing anything.

---

## 8. Troubleshooting

**The run aborts during preflight.** Either OpenMP and SYCL disagree on the
device count, or neither matches `ZE_AFFINITY_MASK`. This is intentional — fix
the environment rather than bypassing the check. The usual cause is
`LIBOMPTARGET_DEVICES` not matching `ZE_FLAT_DEVICE_HIERARCHY`.

**Bandwidth is near the host-link rate rather than the peer-link rate.** The
transfer is being staged through the host rather than using the peer link.
Run with `DIAG=1` and read the copy-engine profile.

**Rows read `verify_failed`.** The binary's own correctness check failed, so
the reported bandwidth is not measuring the intended transfer. Investigate
before using any number from that run.

**OpenMP aborts or reports host execution.** `OMP_TARGET_OFFLOAD=mandatory`
turns a silent host fallback into a failure. Check the build used
`-fopenmp-targets=spir64` and that `sycl-ls` sees the GPUs.

**Numbers differ from the published figures by a large factor.** Check the
`hierarchy` column and the oneAPI version in `env.txt`.

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
