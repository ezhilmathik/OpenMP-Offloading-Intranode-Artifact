# NVIDIA H100 — Intra-node Data Movement Benchmarks

Benchmarks accompanying the paper *[PAPER TITLE]*.

This directory contains the NVIDIA H100 measurements: host-to-device,
device-to-host, and peer-to-peer transfer bandwidth, comparing **OpenMP target
offload** against **native CUDA**.

Companion vendor directories: `../AMD-MI250X/` and `../Intel-1550/`.

---

## Quick start — reproduce every figure, no GPU required

The raw measurement data from our runs is included, so all five figures can be
regenerated on any machine with Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt

./all-plots.sh
```

Runs in seconds. Each figure is written to the corresponding
`<directory>/plots_final/`, and the script reports which succeeded and which
failed.

Re-measuring from scratch requires a multi-GPU H100 node and a Slurm
scheduler; see the individual directory READMEs.

---

## Directories

| Directory | Measures | Figure |
|---|---|---|
| `Synchronous-test-H-to-D-Measurements/` | Blocking host → device | `h100_sync_htod.pdf` |
| `Synchronous-test-D-to-H-Measurements/` | Blocking device → host | `h100_sync_dtoh.pdf` |
| `Asynchronous-test-H-to-D-Measurements/` | Non-blocking host → device | `h100_async_htod.pdf` |
| `Asynchronous-test-D-to-H-Measurements/` | Non-blocking device → host | `h100_async_dtoh.pdf` |
| `P2P-Measurements/` | GPU ↔ GPU over NVLink | `h100_p2p.pdf` |

Each directory has its own README covering what it measures, how to build it,
how to re-run it, and how its output maps to the paper.

The four host-transfer directories sweep identical buffer sizes, so their
figures are directly comparable. P2P uses smaller buffers because both
endpoints are device-resident — see its README.

`all-plots.sh` regenerates every figure in one pass.

---

## Common structure

Every benchmark directory follows the same layout:

| File | Purpose |
|---|---|
| `openmp-*.cc` | OpenMP target offload implementations |
| `cuda-*.cu` | Native CUDA implementations |
| `openacc-*.cc` | OpenACC implementations — retained for reference, not used in this study |
| `Makefile` | Build rules |
| `build.sh` | Slurm job: compile and verify |
| `run.sh` | Slurm job: run the sweep, write `timings.csv` |
| `submit.sh` | Submit build + N run jobs with an `afterok` dependency |
| `plot_h100_*.py` | Aggregate the runs, produce the figure |
| `results_<jobid>_<timestamp>/` | Raw output, 5 repetitions per directory |
| `plots_final/` | Generated figure |

Version numbering is consistent throughout: `v1` is the single-GPU case, `v2`
the two-GPU case. In P2P these denote transfer direction rather than GPU count
— see that directory's README.

---

## Software environment

Measurements were taken with GCC 13.2.0, CUDA 12.8, and Clang 18.1.8 with
NVPTX offload support. Figure generation needs Python 3 with numpy and
matplotlib (`../requirements.txt`).

**No special runtime configuration is required.** Unlike the AMD MI250X
directories, which disable the DMA engines and preload a separately built
OpenMP offload runtime, and the Intel directories, which pin the Level Zero
device hierarchy and its OpenMP counterpart, the H100 measurements run against
the stock CUDA and Clang offload runtimes with no environment overrides. This
makes them the most straightforward of the three vendors to reproduce.

The Slurm directives in `build.sh` and `run.sh` — account, partition, QoS —
are specific to the machine used for the paper and must be changed before
submitting anywhere else. The `module load` lines likewise reflect one site's
module tree.

---

## License and citation

*[Complete before submission.]* See `../LICENSE` and `../CITATION.cff`.
Archived at https://doi.org/10.5281/zenodo.XXXXXXX
