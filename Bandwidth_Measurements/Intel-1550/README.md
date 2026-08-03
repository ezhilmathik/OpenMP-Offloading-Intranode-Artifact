# Intel Data Center GPU Max 1550 — Intra-node Data Movement Benchmarks

Benchmarks accompanying the paper *[Evaluating OpenMP Offloading for Intranode Multi-GPU Programming on NVIDIA, AMD, and Intel GPUs: A 3D Heat Transfer Case Study]*.

This directory contains the Intel Data Center GPU Max 1550 (Ponte Vecchio)
measurements: host-to-device, device-to-host, and peer-to-peer transfer
bandwidth, comparing **OpenMP target offload** against **native SYCL**.
Measurements were taken at **LRZ**.

Companion vendor directories: `../NVIDIA-H100/` and `../AMD-MI250X/`.

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

Re-measuring from scratch requires a multi-card PVC node and a Slurm
scheduler.

---

## Directories

| Directory | Measures | Figure |
|---|---|---|
| `Synchronous-test-H-to-D-Measurements/` | Blocking host → device | `pvc_sync_htod.pdf` |
| `Synchronous-test-D-to-H-Measurements/` | Blocking device → host | `pvc_sync_dtoh.pdf` |
| `Asynchronous-test-H-to-D-Measurements/` | Non-blocking host → device | `pvc_async_htod.pdf` |
| `Asynchronous-test-D-to-H-Measurements/` | Non-blocking device → host | `pvc_async_dtoh.pdf` |
| `P2P-Measurements/` | GPU ↔ GPU over Xe Link | `pvc_p2p.pdf` |

Each directory has its own README covering what it measures, how to build it,
how to re-run it, and how its output maps to the paper.

The four host-transfer directories sweep identical buffer sizes, so their
figures are directly comparable. P2P uses smaller buffers because both
endpoints are device-resident.

---

## Device hierarchy — read this before comparing any numbers

A Max 1550 card contains **two stacks**, and the runtime can present either
the card or the stack as a device. Every `run.sh` takes the choice as an
argument:

```bash
sbatch run.sh            # COMPOSITE (default): a device is a whole card
sbatch run.sh FLAT       # FLAT: a device is one stack
```

| Hierarchy | `ZE_AFFINITY_MASK` | `LIBOMPTARGET_DEVICES` | Meaning |
|---|---|---|---|
| `COMPOSITE` | `0,1,2,3` | `device` | 4 whole cards, one full host link each |
| `FLAT` | `0,2,4,6` | `subdevice` | one stack from each of 4 cards |

The two columns matter equally. `ZE_*` configures the Level Zero driver, which
is what SYCL sees; `LIBOMPTARGET_DEVICES` is OpenMP's own mapping control and
is not affected by the `ZE_*` variables. If the two disagree, OpenMP and SYCL
benchmark different hardware under the same label. The P2P directory enforces
agreement with a preflight check that aborts the run rather than producing an
indefensible comparison.

Because the same device count means different silicon under the two
hierarchies, every `timings.csv` carries a seventh column recording which was
used, and every results directory is named after it. **Results from different
hierarchies are not comparable.**

The published figures use `COMPOSITE`.

---

## Common structure

Every benchmark directory follows the same layout:

| File | Purpose |
|---|---|
| `openmp-*.cc` | OpenMP target offload implementations |
| `sycl-*.cpp` | Native SYCL implementations |
| `Makefile` | Build rules (`icpx`, `-fsycl` / `-fopenmp-targets=spir64`) |
| `build.sh` | Slurm job: compile and verify |
| `run.sh` | Slurm job: run the sweep, write `timings.csv` |
| `submit.sh` | Submit build + N run jobs with an `afterok` dependency |
| `plot_pvc_*.py` | Aggregate the runs, produce the figure |
| `results_<HIER>_<jobid>_<timestamp>/` | Raw output, 5 repetitions per directory |
| `plots_final/` | Generated figure |

Version numbering is consistent throughout: `v1` is the single-device case,
`v2` the two-device case. In P2P these denote transfer direction rather than
device count — see that directory's README.

`v4` sources and build rules are retained for future work. They are not
executed by `run.sh` and not plotted. Note that `P2P-Measurements/openmp-4.cc`
has a known serialization bug, documented in the notes at the bottom of that
directory's `run.sh`.

---

## Provenance

Unlike the other two vendors, each results directory here contains an
`env.txt` recording the host, date, compiler version, device hierarchy,
affinity mask and `sycl-ls` output that produced it. The P2P directory adds a
`probe.txt` recording the device counts both runtimes reported. A results
directory is therefore self-describing: it says not only what was run, but on
what, and that both frameworks agreed about the hardware.

---

## Software environment

```bash
module load slurm_setup
module load intel-toolkit/2024.1.0
```

| Component | Version |
|---|---|
| Intel oneAPI DPC++/C++ Compiler | 2024.1.0 (2024.1.0.20240308) |
| Level Zero driver | 1.3.27642 |

The toolkit version is pinned in `P2P-Measurements/run.sh` deliberately: the
default device-hierarchy behaviour has changed between oneAPI releases, so a
floating module would make runs non-reproducible in exactly the dimension
under investigation.

No custom OpenMP runtime is required. This differs from the AMD MI250X
directories, which preload a separately built `libomptarget`.

The Slurm directives in `build.sh` and `run.sh` — account and partition — are
specific to LRZ and must be changed before submitting elsewhere.

---

## License and citation

*[Complete before submission.]* See `../LICENSE` and `../CITATION.cff`.
Archived at https://doi.org/10.5281/zenodo.XXXXXXX
