# Peer-to-Peer Bandwidth — NVIDIA H100

Benchmarks accompanying the paper *[PAPER TITLE]*.

This directory measures **GPU-to-GPU peer-to-peer (P2P) transfer bandwidth**
on an NVIDIA H100 node, comparing **OpenMP target offload** against **native
CUDA**. Unlike the host-to-device directories, transfers here traverse the
inter-GPU peer link rather than PCIe to the host.

Companion directories in `../`: `Synchronous-test-H-to-D-Measurements/`,
`Synchronous-test-D-to-H-Measurements/`,
`Asynchronous-test-H-to-D-Measurements/`,
`Asynchronous-test-D-to-H-Measurements/`. Other vendors:
`../../AMD-MI250X/` and `../../Intel-1550/`.

---

## 1. What is measured

Peer-to-peer transfers between **GPU 0 and GPU 1**, in two configurations:

| Version | Configuration | Theoretical ceiling |
|---|---|---|
| `v1` | Unidirectional: GPU 0 → GPU 1 | 150 GB/s |
| `v2` | Bidirectional: GPU 0 ↔ GPU 1, both directions driven concurrently by two host threads | 300 GB/s |

Both configurations use the same pair of devices; `v2` differs from `v1` in
direction, not in the number of GPUs involved. Each is implemented twice —
once in OpenMP target offload, once in CUDA — giving four measured series.
Buffers are arrays of `double` (8 bytes per element).

The theoretical values are **peer-link (NVLink) bandwidths**, not PCIe. On the
node used, each H100's NVLink lanes are divided among its peers, giving
150 GB/s per peer pair in one direction and 300 GB/s when both directions are
saturated. They are configured in `THEORETICAL_BW` at the top of
`plot_h100_p2p.py` and appear as dashed reference lines in the figure.

> **Not part of this study.** Two further configurations exist in the
> repository and are left to future work:
> - `v4` — gather: GPU 0 receives concurrently from GPUs 1, 2 and 3
>   (theoretical ceiling 450 GB/s, i.e. three peer links inbound).
> - `v4_4` — all-to-all: every GPU sends to and receives from all three
>   peers simultaneously.
>
> Their sources and build rules remain in place, but they are disabled by
> default. To measure them: set `RUN_FUTURE=1` near the top of `run.sh`, add
> the corresponding binaries to `OMP_BINS` / `CUDA_BINS` in the `Makefile` and
> to `EXPECTED` in `build.sh`, and add `"v4"` (and `"v4_4"`) to
> `VERSIONS_TO_PLOT` in `plot_h100_p2p.py`.

> **OpenACC.** `openacc-*.cc` implement the same transfers in OpenACC, built
> with `nvc++` from NVHPC. They are retained for reference and comparison but
> are outside the scope of this paper: they are not compiled by `build.sh`,
> not executed by `run.sh`, and not plotted. To measure them, set `RUN_ACC=1`
> in `run.sh` (which also loads the NVHPC module) and run `make acc`.

---

## 2. Contents

### Used in this study

| File | Description |
|---|---|
| `openmp-1.cc`, `openmp-2.cc` | OpenMP target offload implementations (unidirectional, bidirectional) |
| `cuda-1.cu`, `cuda-2.cu` | Equivalent native CUDA implementations |
| `Makefile` | Build rules for all toolchains |
| `build.sh` | Slurm job: compiles the binaries and verifies they exist |
| `run.sh` | Slurm job: executes the buffer-size sweep, writes `timings.csv` |
| `submit.sh` | Submits `build.sh`, then N run jobs with an `afterok` dependency |
| `plot_h100_p2p.py` | Aggregates all runs and produces the paper figure |
| `clean.sh` | Removes compiled binaries |
| `results_<jobid>_<timestamp>/` | Raw output from our runs (5 repetitions) |
| `plots_final/` | Generated figure, as it appears in the paper |

### Retained but not used

| File | Description |
|---|---|
| `openacc-1.cc`, `openacc-2.cc`, `openacc-4.cc`, `openacc-4-4.cc` | OpenACC implementations — reference only, see Section 1 |
| `openmp-4.cc`, `cuda-4.cu` | Four-GPU gather (`v4`) — future work |
| `openmp-4-4.cc`, `cuda-4-4.cu` | All-to-all (`v4_4`) — future work |

Compiled executables are not included; `build.sh` produces them. Source files
are named by framework; the binaries they produce keep the `version-N-off` /
`version-N-cuda` names used in `run.sh` and in the CSV output.

---

## 3. Requirements

### Hardware

- A node with **at least two NVIDIA H100 GPUs** (compute capability 9.0,
  `sm_90`) joined by a **direct NVLink connection**. The benchmark uses GPU 0
  and GPU 1. P2P falling back over PCIe produces substantially lower numbers
  that are not comparable to those reported here. The node used for the
  published measurements had *[N]* H100 GPUs in an all-to-all NVLink
  arrangement, with 6 lanes (150 GB/s) between each pair.
- Approximately 10 GB of host RAM for buffer setup; the transfers themselves
  are device-resident. The largest buffer is 1.24e9 doubles (~9.92 GB) per
  device.
- Exclusive node access. The scripts pass `--exclusive`; peer-link contention
  from other jobs would invalidate the measurement.
- `v2` drives both directions from **two concurrent host threads**, so at
  least two CPU cores must be available to the job and `OMP_NUM_THREADS` must
  be 2 or more.

Confirm peer access is available before running:

```bash
nvidia-smi topo -m
```

Pairs should report `NV#` rather than `PIX` / `SYS`.

### Software

Versions used for the published measurements:

| Component | Version |
|---|---|
| GCC | 13.2.0 |
| CUDA | 12.8 |
| Clang/LLVM (NVPTX offload) | 18.1.8-cuda12.8 |
| Python | 3.12 |
| numpy | 2.5.1 |
| matplotlib | 3.11.1 |
| NVHPC (OpenACC only, optional) | 23.7-CUDA-12.2.0 |

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
make            # the four binaries used in this study
make omp        # OpenMP offload only
make cuda       # CUDA only
make acc        # OpenACC — requires NVHPC, not used in this study
make future     # four-GPU configurations — not used in this study
```

A bare `make` produces:

```
version-1-off   version-2-off      # OpenMP target offload
version-1-cuda  version-2-cuda     # CUDA
```

and needs no NVHPC. `build.sh` checks all four exist and exits non-zero
otherwise, which blocks dependent run jobs via the `afterok` dependency.

---

## 5. Smoke test (~2 minutes)

Confirm the environment is sane before committing queue time:

```bash
make omp
./version-1-off 100000000
```

Expected: one line reporting elapsed time and bandwidth in GB/s, at a value
well above the PCIe ceiling if the peer link is being used. A result in the
tens of GB/s suggests P2P fell back to routing through the host — see
Troubleshooting.

---

## 6. Reproducing the results

### Python environment

Needed for figure generation, on the cluster or on a laptop:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../../requirements.txt
```

### Regenerating the figure from the included data

```bash
python plot_h100_p2p.py
```

This globs **all** `results_*/timings.csv` directories, averages each
(framework, version, size) triple across repetitions, and writes
`plots_final/h100_p2p.pdf`. It also prints the mean and maximum
CUDA-versus-OpenMP bandwidth gap per version to stdout.

Because the raw `results_*` directories are included in this artifact, this
step needs **no GPU and no cluster access** — it reproduces the published
figure in seconds on any machine with Python.

To regenerate every H100 figure at once, run `../all-plots.sh`.

### Re-measuring from scratch

```bash
./submit.sh
```

Submits one build job followed by **five independent run jobs**, each
depending on the build succeeding. Five repetitions is what the paper reports;
adjust `N_RUNS` at the top of `submit.sh`.

Monitor with `squeue -u $USER`.

Buffer sizes swept, in elements (`double`, 8 bytes each):

```
117500000  217500000  318750000  418750000  520000000  1240000000
```

These are smaller than the sizes used in the host-transfer directories,
because both buffers are device-resident and must fit alongside each other in
GPU memory. Note that the final size is roughly double the previous one; the
x-axis uses categorical positions, so the points are drawn evenly spaced
regardless of this gap.

Each run job is capped at **one hour** of wall clock, so the full pipeline
costs roughly 5 node-hours plus queue wait — longer than the host-to-device
directories.

### Plot style

The style block at the top of the script is deliberately identical across all
panel scripts in this repository, so that panels placed side by side in a
single LaTeX figure have matching font sizes. Do not change `FIGSIZE` or the
`FS_*` values in one script without changing all of them.

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

Some rows in the included `results_*` data carry an extra empty field between
`N` and `elapsed_sec`, an artifact of an earlier version of `run.sh`. The plot
script accepts both the six- and seven-field layouts, so this does not affect
reproduction.

### Claim map

| Paper claim | Command | Cost | Confirming output |
|---|---|---|---|
| *[Figure N]*: OpenMP offload vs CUDA, P2P on H100 | `python plot_h100_p2p.py` | seconds, no GPU needed | `plots_final/h100_p2p.pdf` |
| *[Figure N]*, measured from scratch | `./submit.sh`, then the above | ~5 node-h | same |
| *[Claim about the mean CUDA vs OpenMP P2P gap]* | as above | — | summary box in the figure, in GB/s; also printed to stdout |
| *[Any claim comparing P2P against host transfers]* | also run the H-to-D and D-to-H directories | ~15 node-h total | the respective figures |

*[Replace the bracketed labels with the actual figure and table numbers.]*

### Tolerance

Bandwidth varies run to run and machine to machine. Deviations of a few
percent from the published values are normal and do not constitute a failed
reproduction — the claim under test is the relationship between OpenMP offload
and CUDA, not the absolute figures.

P2P results are more topology-sensitive than host transfers. The 150 GB/s and
300 GB/s reference lines follow from GPU 0 and GPU 1 sharing 6 NVLink lanes on
the node used; a node where that pair shares a different number of lanes has a
different ceiling, and `THEORETICAL_BW` in the plot script would need adjusting
to match. Report the topology alongside any comparison.

Regenerating the figure from the included `results_*` data is deterministic
and should match the published figure.

---

## 8. Troubleshooting

**Bandwidth is close to PCIe rates rather than NVLink rates.** Peer access was
not enabled and transfers were staged through the host. Check
`nvidia-smi topo -m` and confirm the selected device pair has a direct link.

**Every row reads `missing_binary`.** The build stage failed; inspect
`build-<jobid>.out`.

**`libomptarget.so` not found at runtime.** The Clang runtime path is not
resolvable. The `Makefile` embeds an rpath, so this usually means a later
`module load` overwrote the environment.

**Rows read `timeout`.** The per-measurement limit is `TIMEOUT_SEC=300` in
`run.sh`.

**Out of memory at large sizes.** Trim the `Ns` array in `run.sh` and state
which sizes were omitted in any comparison.

**`v2` results are missing or match `v1`.** The two directions are not running
concurrently. Confirm the host has at least two threads available
(`OMP_NUM_THREADS` ≥ 2, and the Slurm allocation is not restricting cores),
and that both GPUs are visible via `CUDA_VISIBLE_DEVICES`.

**Bandwidth is low or erratic.** Confirm exclusive allocation and that nothing
else was using the peer links.

**The plot script reports no data.** It expects `results_*/timings.csv`
relative to the current directory — run it from this directory.

**`ModuleNotFoundError: numpy`.** The Python environment is not set up; see
Section 6.

---

## 9. Cleaning up

```bash
./clean.sh              # binaries in this directory
make clean              # equivalent, including the unused configurations
../../clean.sh      # whole tree: binaries, editor backups, build debris
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
