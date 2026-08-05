# 3D Heat Benchmark — Intra-node Multi-GPU Offloading

A 3D variable-coefficient heat-equation solver used to compare **OpenMP target
offloading** against **native vendor models** (HIP, CUDA, SYCL) when a single
domain is decomposed across the GPUs *inside one node*.

The benchmark answers one question: **how much of the achievable intra-node
multi-GPU performance can portable OpenMP offloading actually deliver, and what
does the halo-exchange strategy cost?**

Every directory ships both the **sources** and the **results they produced**, so
a reader can re-run the pipeline or inspect the recorded data directly.

---

## 1. Machines

| Directory | Hardware | Site | Native model | Setup & commands |
|---|---|---|---|---|
| `AMD-MI250X/` | MI250X (gfx90a), 1 GCD per module | LUMI-G | HIP | [README](AMD-MI250X/Source-Results/README.md) |
| `NVIDIA-H100/` | H100 (sm_90) | MareNostrum 5 | CUDA | [README](NVIDIA-H100/Source-Results/README.md) |
| `Intel-1550/` | Data Center GPU Max 1550 (PVC) | LRZ | SYCL | [README](Intel-1550/Source-Results/README.md) |

**Everything machine-specific — modules, compiler flags, device masking,
per-variant runtime environment, the P2P runtime, correctness tolerances — is
documented in the per-machine README.** This file covers only what is common to
all three.

Each machine directory has the same shape:

```
<MACHINE>/Source-Results/
├── OpenMP/              # OpenMP target-offload variants + full run pipeline
├── HIP/ | CUDA/ | SYCL/ # native-model counterpart, same pipeline & CSV schema
├── model/               # analytical performance model + model-vs-measurement plots
├── plots/               # final figures
└── speedup.py           # cross-model speedup figures from the summary.csv files
```

The **unmodified, machine-independent sources** live one level up:

```
../OpenMP_Single_Source/
├── heat3D_variable_coeff.c     # serial CPU reference (correctness only)
├── 1-openmp.c                  # single-GPU baseline
├── 2-openmp.c                  2-openmp.c
├── 2-openmp-stream.c           2-openmp-stream.c
├── 2-openmp-stream-omp.c       2-openmp-stream-omp.c
├── 2-openmp-stream-omp-p2p.c   2-openmp-stream-omp-p2p.c
├── 4-openmp.c                  4-openmp.c
|── 4-openmp-stream.c           4-openmp-stream.c
├── 4-openmp-stream-omp.c       4-openmp-stream-omp.c
└── 4-openmp-stream-omp-p2p.c   4-openmp-stream-omp-p2p.c
```

These nine files are the *single source* compiled on **all three vendors**
without modification — only the compiler, the module environment, and the
device-selection variable differ. Each `OpenMP/` directory holds a copy next to
the scripts that build it.

---

## 2. The nine OpenMP variants

| Binary | GPUs | Halo exchange strategy |
|---|---|---|
| `1-omp` | 1 | none — single-device baseline, the speedup denominator |
| `2-omp` | 2 | synchronous, host-staged |
| `2-omp-stream` | 2 | asynchronous, host-staged |
| `2-omp-stream-omp` | 2 | one host thread per device, asynchronous, host-staged |
| `2-omp-stream-omp-p2p` | 2 | one host thread per device, asynchronous, direct P2P |
| `4-omp` | 4 | synchronous, host-staged |
| `4-omp-stream` | 4 | asynchronous, host-staged |
| `4-omp-stream-omp` | 4 | one host thread per device, asynchronous, host-staged |
| `4-omp-stream-omp-p2p` | 4 | one host thread per device, asynchronous, direct P2P |

The progression is deliberate: each step removes one serialisation point.
`-omp-stream-omp-p2p` is the only variant that needs anything outside a stock
toolchain, and only on AMD — see the
[MI250X README](AMD-MI250X/Source-Results/README.md#p2p-runtime-required-by-the--p2p-variants-only).

The `HIP/`, `CUDA/`, and `SYCL/` trees mirror this naming
(`2-hip-stream-omp-p2p`, `4-cuda-stream`, …) so a variant can be compared
model-to-model row by row.

---

## 3. Problem definition

Explicit finite-difference solve of

> ∂u/∂t = ∇·(κ(x,y,z) ∇u)

on an `N × N × N` grid with a spatially varying coefficient, Dirichlet
boundaries, decomposed in the slowest-varying dimension. Each timestep needs one
halo-face exchange per internal boundary.

**Timestep derivation** (identical in `submit.sh` and `verify.sh` on every
machine — keep them in sync or you verify a different stability regime than you
benchmark):

```
h  = 1 / (N - 1)
dt = h² / (12 · κ_max)          κ_max = 0.95
DT = STEPS · dt                 # the value passed to the binary
```

**Production sweep:** `N ∈ {512, 640, 768, 896, 1024, 1152, 1280}`, `STEPS=500`,
**5 independent repetitions** per size (each a separate Slurm job), medians
reported.

All variants are compiled with `-ffp-contract=off` so FMA contraction does not
make comparison across models meaningless.

---

## 4. Reproducing

The pipeline is the same everywhere; only the environment differs. From inside
`<MACHINE>/Source-Results/OpenMP/` (and identically in the native-model tree):

```bash
./submit.sh 5        # build + 5 reps × 7 grid sizes
squeue -u $USER      # wait
./aggregate.sh       # merge results_*/ -> summary.csv (medians)
./plot.sh -a         # ASCII bars, per-family clusters, speedups
```

Narrowing the sweep:

```bash
NS="1024" ./submit.sh 1                    # one size, one repetition
FAMILIES="4" BASELINE=0 ./submit.sh 3      # 4-GPU family only
STEPS=1000 ./submit.sh 5                   # longer runs
SKIP_BUILD=1 ./submit.sh 1                 # reuse binaries (build.sh runs make clean)
```

> `submit.sh`'s first argument is the **repetition count, not the grid size**.
> `./submit.sh 512` is caught and rejected; use `NS="512" ./submit.sh 1`.
> Override with `FORCE_RUNS=1` if you really want >20 repetitions.

**Read the machine README before submitting** — accounts, partitions, the
device-selection variable, and (on AMD) a source patch that must be applied
first all differ.

### Script reference

| Script | Role |
|---|---|
| `Makefile` | `all` (2- and 4-GPU families), `gpu-baseline` (`1-omp`), `serial-ref` (CPU reference), `clean` |
| `build.sh` | Slurm build job: loads the toolchain, builds, then **verifies every expected binary exists** and fails loudly if not |
| `submit.sh` | Orchestrator. Computes `DT` per `N`, submits one `run.sh` job per (N, repetition) |
| `run.sh` | One job = one `N`, one repetition, all variants. Sets the per-variant environment, times each binary, writes `results_*/timings.csv` |
| `aggregate.sh` | Merges all `results_*/timings.csv` → `summary.csv` with `min_sec` / `median_sec` / `max_sec` |
| `verify.sh` | **Correctness only.** Every variant against the CPU reference |
| `plot.sh` | ASCII charts from `summary.csv` — no GUI, no matplotlib |
| `clean.sh` | Removes binaries, `.txt` solutions (`-f out`), and result directories |

`plot.sh` locates its columns **by header name**, so it reads both the
AMD/NVIDIA schema (`variant,gpus,N,…`) and the Intel schema (extra `hier`
column). Modes: default bar chart at the largest `N`, `-g 2` / `-g 4` for a
clustered per-size view of one family, `-s` for speedup vs `1-omp`, `-a` for all.

---

## 5. Output schema

`results_*/timings.csv` — one row per variant execution:

```
rep, job_id, [hier,] variant, family, gpus, <mask>, N, dt, steps, wall_sec, reported_sec, status
```

The mask column is machine-specific (`rocr_mask`, `cuda_mask`, `ze_mask`), and
the Intel tree carries an extra leading `hier` column. Result directory names
also encode the machine's sweep knobs (`results_h8_…`,
`results_COMPOSITE_h8_…`).

* `wall_sec` — measured by the harness around the whole process.
* `reported_sec` — the binary's own `Time = … seconds` line; excludes setup and
  allocation. **This is the number behind every figure.**
* `status` — `ok`, `timeout`, `runtime_error`, `missing_binary`,
  `missing_p2p_runtime`.

A failing variant never aborts the repetition (`run.sh` is deliberately not
`set -e`); it is recorded with a non-`ok` status and the sweep continues. Skipped
work is also listed in `results_*/skipped.log`.

`aggregate.sh` defaults to `TIME_FIELD=compute`, reading the solver's timer
straight out of `run.log`; `reported_sec` in older `timings.csv` files is
unreliable because the regex once matched `dt` from the parameter banner. Use
`TIME_FIELD=wall` to cross-check.

---

## 6. Correctness

Timing runs never write solution files (`WRITE_TXT=0`). Verification is a
separate job:

```bash
sbatch verify.sh                              # defaults: N=256, STEPS=100
sbatch --export=ALL,N=128,STEPS=50 verify.sh
```

`verify.sh` builds the serial CPU reference, runs it to produce `u_serial.txt`,
then runs every variant with `WRITE_TXT=1` and compares element-wise. It reports
the worst absolute and relative difference, their locations, and the count of
cells outside tolerance, and exits non-zero on any failure — including the
"nothing ran" case, reported as `INCONCLUSIVE`, not as a pass.

Tolerances differ by machine; see the per-machine README.

> **Never plot numbers from `verify.sh`.** File I/O sits inside the measured
> region. The only intended difference between `run.sh` and `verify.sh` is
> `WRITE_TXT`; if the two per-variant environment blocks drift apart,
> verification stops validating the code path that was benchmarked.

---

## 7. Performance model

`<MACHINE>/Source-Results/model/` contains a calibrated analytical model, not a
roofline bound.

**Step 1 — calibrate the machine constant** from real single-GPU runs:

```bash
cd model
python3 compute_gamma.py                 # finds ../OpenMP/summary.csv
python3 compute_gamma.py --plot          # also writes gamma_vs_N.png
```

γ = T₁ / (N_s · (n−2)³), i.e. seconds per interior point-update. The reported
value is the plateau — the mean over the three largest grids — written to
`gamma_single_gpu.csv`.

**Step 2 — predict and compare** against the measured direct-P2P times:

```bash
python3 plot_p2p.py --gpus 2 --beta-p2p 230
python3 plot_p2p.py --gpus 4 --beta-p2p 230 --residuals
```

Overlapped model for P GPUs:

```
M      = 8 N²                                  # halo face, bytes (fp64)
t_int  = γ (N−2)³ / P                           # interior, split across devices
t_bnd  = 2 γ (N−2)²                             # boundary planes, not overlapped
faces  = 1 if P == 2 else 2                     # interior GPU is the bottleneck
T_pred = N_s · [ t_bnd + max(t_int, faces · (α_p2p + M / β_p2p)) ]
```

`β_p2p` comes from the companion `../Bandwidth_Measurements/` results. Because γ
is calibrated from measurement, the prediction can err in **either** direction;
agreement is reported as MAPE, signed bias, and worst single-size miss, all
normalised by the measured time.

---

## 8. Reading the results without re-running

Nothing needs to be executed to inspect the data:

* `<MACHINE>/Source-Results/*/summary.csv` — the aggregated medians behind every figure.
* `<MACHINE>/Source-Results/*/results_*/` — per-job raw timings and logs, untouched.
* `<MACHINE>/Source-Results/plots/` and `model/*.pdf` — the final figures.
* `run-<jobid>.out` — complete Slurm stdout, including the exact command line and
  environment printed for each variant before it ran.

Every run log echoes the resolved environment (device mask, thread counts,
transfer settings, `LD_PRELOAD`) and the quoted command line, so any single
measurement in the CSVs traces back to the exact configuration that produced it.

---


## 9. Citation and License


See the [repository root README](../README.md#citation). The artifact is
licensed under EUPL-1.2 ([`LICENSE`](../LICENSE)); recorded measurement data
under CC BY 4.0.
