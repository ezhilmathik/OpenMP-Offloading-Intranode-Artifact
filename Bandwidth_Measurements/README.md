# OpenMP Offloading Intra-node Bandwidth Benchmarks

Artifact accompanying the paper *[PAPER TITLE]*.

This repository measures **intra-node data movement bandwidth** on three GPU
architectures, comparing **OpenMP target offload** against each vendor's native
programming model.

Five transfer scenarios are measured on each: synchronous and asynchronous
host-to-device, synchronous and asynchronous device-to-host, and peer-to-peer
between devices — 15 benchmark directories in total.

---

## Contact

*[Complete before submission.]*

| | |
|---|---|
| Maintainer | *[NAME]* |
| Email | *[EMAIL]* |
| Affiliation | *[INSTITUTION]* |
| ORCID | *[ORCID]* |
| Repository | *[https://github.com/ORG/REPO]* |
| Paper version | tag *[vX.Y]*, commit *[SHA]* |
| Archive | https://doi.org/10.5281/zenodo.XXXXXXX |

Please open an issue on the repository for problems with the artifact, or mail
the maintainer if the repository is unavailable.

---

## Toolchains

Each platform is built with two toolchains: the vendor's native model, and
OpenMP target offload. On AMD and Intel a single compiler drives both; on
NVIDIA the two paths use different compilers entirely, which is worth keeping
in mind when reading the H100 figures.

| Vendor | Device | Native model | Native compiler | OpenMP offload compiler | Offload target |
|---|---|---|---|---|---|
| NVIDIA | H100 | CUDA | `nvcc` (CUDA 12.8) | `clang++` 18.1.8 | `nvptx64`, `sm_90` |
| AMD | Instinct MI250X | HIP | `amdclang++` (ROCm 6.4.4) | `amdclang++` (ROCm 6.4.4) | `amdgcn-amd-amdhsa`, `gfx90a` |
| Intel | Data Center GPU Max 1550 | SYCL | `icpx` (oneAPI 2024.1.0) | `icpx` (oneAPI 2024.1.0) | `spir64` (JIT to SPIR-V) |

Additional detail:

| Vendor | Host compiler | Runtime notes |
|---|---|---|
| NVIDIA | GCC 13.2.0 | Stock CUDA and Clang offload runtimes; no overrides |
| AMD | PrgEnv-amd (LUMI 25.09) | OpenMP runs preload a `libomptarget` built from AMD's LLVM fork, **not** the site ROCm module |
| Intel | oneAPI 2024.1.0 | Level Zero driver 1.3.27642; device hierarchy pinned explicitly |

OpenACC implementations (`openacc-*.cc`, NVIDIA only, built with `nvc++` from
NVHPC 23.7) are retained for reference. They are not built by default, not
measured, and not plotted.

Four-device variants (`v4`, `v4_4`) are likewise retained as future work and
excluded from the default build.

---

## Systems used

Each vendor's measurements were taken on a different European supercomputer:

| GPU | System | Operator | Location |
|---|---|---|---|
| NVIDIA H100 | MareNostrum 5 (ACC partition) | BSC | Barcelona, Spain |
| AMD MI250X | LUMI (LUMI-G partition) | CSC / EuroHPC | Kajaani, Finland |
| Intel Max 1550 | SuperMUC-NG | LRZ | Garching, Germany |

Because each vendor's results come from a different machine, the Slurm
directives, module trees and site-specific paths differ between the three
vendor directories. Each directory README lists exactly which lines need
changing to run elsewhere.

Cross-vendor comparisons in the paper are of the OpenMP-versus-native
relationship within each platform, not of absolute bandwidths between
platforms, since the host interconnects and node designs differ.

---

## Reproducing the figures — no GPU required

**The raw measurement data from every run is included in this repository.**
All 15 figures can therefore be regenerated on any machine with Python, in
seconds, without a GPU, a cluster account or a vendor toolchain:

```bash
python3 -m venv ~/venv-openmp-offloading
source ~/venv-openmp-offloading/bin/activate
pip install -r requirements.txt

for v in NVIDIA-H100 AMD-MI250X Intel-1550; do (cd "$v" && ./all-plots.sh); done
```

Each vendor's `all-plots.sh` reports which figures succeeded and which failed,
and writes each one to the corresponding `<directory>/plots_final/`.

This is the recommended path for artifact evaluation. Re-measuring from
scratch requires the specific hardware described below and roughly 67 node-
hours in total.

---

## Layout

```
.
├── NVIDIA-H100/          ├── AMD-MI250X/          ├── Intel-1550/
│   ├── README.md         │   ├── README.md        │   ├── README.md
│   ├── all-plots.sh      │   ├── all-plots.sh     │   ├── all-plots.sh
│   ├── check-artifact.sh │   ├── check-artifact.sh│   ├── check-artifact.sh
│   ├── submit-all.sh     │   ├── submit-all.sh    │   ├── submit-all.sh
│   ├── clean.sh          │   ├── clean.sh         │   ├── clean.sh
│   └── <5 benchmark directories, identical structure in each vendor>
├── requirements.txt      Python dependencies for figure generation
├── clean.sh              Tree-wide removal of binaries and build debris
├── check-all.sh          Consistency check across the whole artifact
├── appendix-artifact.tex Artifact appendix as submitted with the paper
├── LICENSE  NOTICE  CITATION.cff
```

The five benchmark directories, present under every vendor:

| Directory | Measures |
|---|---|
| `Synchronous-test-H-to-D-Measurements/` | Blocking host → device |
| `Synchronous-test-D-to-H-Measurements/` | Blocking device → host |
| `Asynchronous-test-H-to-D-Measurements/` | Non-blocking host → device |
| `Asynchronous-test-D-to-H-Measurements/` | Non-blocking device → host |
| `P2P-Measurements/` | Device ↔ device over the vendor's peer link |

Each contains its own README covering what it measures, the theoretical
ceilings, how to build and re-run it, and how its output maps to the paper.
**Start with the vendor README, then the directory README.**

Every directory follows the same conventions:

- `openmp-N.cc` — OpenMP target offload implementation
- `cuda-N.cu` / `hip-N.hip` / `sycl-N.cpp` — native implementation
- `openacc-N.cc` — OpenACC implementation, NVIDIA only, reference material
- `Makefile` — build rules for both toolchains
- `build.sh`, `run.sh`, `submit.sh` — Slurm build and measurement pipeline
- `plot_<vendor>_<scenario>.py` — aggregates the runs into one figure
- `results_*/` — raw output, five independent repetitions
- `plots_final/` — the generated figure, as it appears in the paper

`v1` denotes the single-device case and `v2` the two-device case. In the
peer-to-peer directories these denote transfer *direction* rather than device
count — see the respective READMEs.

---

## What differs between vendors

The three ports are deliberately structurally identical, but the platforms are
not, and three differences affect how results should be read:

**AMD MI250X requires a separately built OpenMP runtime.** The measurements
preload an OpenMP offload runtime built from AMD's LLVM fork rather than using
the one supplied by the site's ROCm module. Without it the OpenMP numbers are
not reproducible. See `AMD-MI250X/README.md`.

**AMD MI250X runs with the DMA engines disabled** (`HSA_ENABLE_SDMA=0`). This
materially changes measured bandwidth. The Intel port deliberately does *not*
set the nearest equivalent, because that control affects SYCL but not OpenMP
and would bias the comparison; the reasoning is recorded in the Intel
`run.sh` files.

**Intel requires two device-mapping controls to agree.** Level Zero and the
OpenMP offload plugin have independent notions of what a "device" is on a
two-stack card. The Intel scripts set both explicitly and abort if they
disagree, and every Intel `timings.csv` carries an extra `hierarchy` column
recording which mapping was used. The published figures use `COMPOSITE`.

**NVIDIA H100 needs none of the above** — it runs against the stock CUDA and
Clang offload runtimes with no environment overrides, which makes it the most
straightforward of the three to reproduce.

---

## Claim map

*[Complete before submission: replace the bracketed figure numbers.]*

| Paper figure | Directory | Regenerate with |
|---|---|---|
| *[Fig. N]* | `NVIDIA-H100/Synchronous-test-H-to-D-Measurements/` | `plot_h100_sync_htod.py` |
| *[Fig. N]* | `NVIDIA-H100/Synchronous-test-D-to-H-Measurements/` | `plot_h100_sync_dtoh.py` |
| *[Fig. N]* | `NVIDIA-H100/Asynchronous-test-H-to-D-Measurements/` | `plot_h100_async_htod.py` |
| *[Fig. N]* | `NVIDIA-H100/Asynchronous-test-D-to-H-Measurements/` | `plot_h100_async_dtoh.py` |
| *[Fig. N]* | `NVIDIA-H100/P2P-Measurements/` | `plot_h100_p2p.py` |
| *[Fig. N]* | `AMD-MI250X/Synchronous-test-H-to-D-Measurements/` | `plot_mi250x_sync_htod.py` |
| *[Fig. N]* | `AMD-MI250X/Synchronous-test-D-to-H-Measurements/` | `plot_mi250x_sync_dtoh.py` |
| *[Fig. N]* | `AMD-MI250X/Asynchronous-test-H-to-D-Measurements/` | `plot_mi250x_async_htod.py` |
| *[Fig. N]* | `AMD-MI250X/Asynchronous-test-D-to-H-Measurements/` | `plot_mi250x_async_dtoh.py` |
| *[Fig. N]* | `AMD-MI250X/P2P-Measurements/` | `plot_mi250x_p2p.py` |
| *[Fig. N]* | `Intel-1550/Synchronous-test-H-to-D-Measurements/` | `plot_pvc_sync_htod.py` |
| *[Fig. N]* | `Intel-1550/Synchronous-test-D-to-H-Measurements/` | `plot_pvc_sync_dtoh.py` |
| *[Fig. N]* | `Intel-1550/Asynchronous-test-H-to-D-Measurements/` | `plot_pvc_async_htod.py` |
| *[Fig. N]* | `Intel-1550/Asynchronous-test-D-to-H-Measurements/` | `plot_pvc_async_dtoh.py` |
| *[Fig. N]* | `Intel-1550/P2P-Measurements/` | `plot_pvc_p2p.py` |

Each directory README carries a more detailed claim map, including cost and
the specific output that confirms each claim.

---

## Re-measuring from scratch

Only needed to verify the measurements themselves rather than the analysis.
Each vendor needs its own hardware and a Slurm scheduler:

| Vendor | Hardware | Approximate cost |
|---|---|---|
| NVIDIA H100 | ≥2 H100 GPUs, NVLink for P2P | ~17 node-hours |
| AMD MI250X | ≥2 MI250X cards, Infinity Fabric for P2P | ~25 node-hours |
| Intel Max 1550 | ≥2 Max 1550 cards, Xe Link for P2P | ~25 node-hours |

The Slurm directives in every `build.sh` and `run.sh` — account, partition,
QoS — and the `module load` lines are specific to the machines used for the
paper, and must be replaced before submitting anywhere else. Each directory
README lists exactly which lines to change.

From within a benchmark directory:

```bash
./submit.sh          # one build job, then five independent run jobs
```

Or for all five directories of one vendor:

```bash
cd NVIDIA-H100 && ./submit-all.sh
```

The build job verifies that every expected binary exists before exiting, and
the run jobs are chained to it with an `afterok` dependency, so a failed build
blocks measurement rather than producing partial results.

---

## Checking the artifact

```bash
./check-all.sh                       # whole tree
(cd NVIDIA-H100 && ./check-artifact.sh)   # one vendor
```

`check-all.sh` verifies the root files, tree-wide hygiene, cross-vendor
structural parallelism, README consistency and relative-path resolution, then
runs each vendor's own checker.

```bash
./clean.sh                # list build debris that would be removed
./clean.sh --force        # remove it
```

`clean.sh` requires bash 4 or later and GNU `find`/`file`.

---

## Data and provenance

Every benchmark directory ships five independent repetitions. Each
`results_*/` directory contains:

| File | Contents |
|---|---|
| `timings.csv` | One row per measurement |
| `run.log` | Full job output |
| `skipped.log` | Any skipped or errored runs (absent if none) |
| `env.txt` | Intel only: host, compiler, device hierarchy, `sycl-ls` output |
| `probe.txt` | Intel P2P only: device counts reported by each runtime |

CSV schema is `framework,version,N,elapsed_sec,bandwidth_gbs,status` for
NVIDIA and AMD, with an additional `hierarchy` column on Intel. Intel results
directories are named `results_<HIERARCHY>_<jobid>_<timestamp>/`; NVIDIA and
AMD omit the hierarchy field.

---

## License

- **Code** — Apache License 2.0, see `LICENSE`.
- **Measurement data** (`results_*/`) and **figures** (`plots_final/`) —
  CC BY 4.0.
- **Third-party** — see `NOTICE`.

## Citation

*[Complete before submission.]*

Archived at https://doi.org/10.5281/zenodo.XXXXXXX

Citation metadata is in `CITATION.cff`. Please cite both the artifact and the
accompanying paper.
