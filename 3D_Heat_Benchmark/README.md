# 3D Heat Benchmark — Intra-node Multi-GPU Offloading

A 3D variable-coefficient heat-equation solver used to compare **OpenMP target
offloading** against **native vendor models** (HIP, CUDA, SYCL) when a single
domain is decomposed across the GPUs *inside one node*.

The benchmark answers one question: **how much of the achievable intra-node
multi-GPU performance can portable OpenMP offloading actually deliver, and what
does the halo-exchange strategy cost?**

Every directory in this tree ships both the **sources** and the **results they
produced**, so a reader can either re-run the pipeline or inspect the recorded
data directly.

---

## 1. Contents

```
3D_Heat_Benchmark/
├── AMD-MI250X/          # LUMI-G, MI250X (gfx90a)  — OpenMP + HIP (+ CUDA sources)
├── NVIDIA-H100/         # H100                     — OpenMP + CUDA
└── Intel-1550/          # Data Center GPU Max 1550 — OpenMP + SYCL
```

Each machine directory has the same shape:

```
<MACHINE>/Source-Results/
├── OpenMP/              # OpenMP target-offload variants + full run pipeline
│   ├── Makefile  build.sh  submit.sh  run.sh
│   ├── aggregate.sh  verify.sh  plot.sh  clean.sh  compare.py
│   ├── results_<jobid>_<timestamp>/   # one directory per job: timings.csv, run.log
│   ├── run-<jobid>.out                # raw Slurm stdout for that job
│   ├── summary.csv                    # aggregated medians (the file everything reads)
│   └── gamma_single_gpu.csv           # calibrated machine constant
├── HIP/ | CUDA/ | SYCL/ # native-model counterpart, identical pipeline & CSV schema
├── model/               # analytical performance model + model-vs-measurement plots
│   ├── compute_gamma.py
│   └── plot_p2p.py
├── plots/               # final figures
└── speedup.py           # cross-model speedup figures from the summary.csv files
```

The **unmodified, machine-independent sources** live one level up, outside this
directory:

```
../OpenMP_Single_Source/
├── heat3D_variable_coeff.c     # serial CPU reference (correctness only)
├── 1-openmp.c                  # single-GPU baseline
├── 2-openmp.c                  2-openmp-stream.c
├── 2-openmp-stream-omp.c       2-openmp-stream-omp-p2p.c
├── 4-openmp.c                  4-openmp-stream.c
└── 4-openmp-stream-omp.c       4-openmp-stream-omp-p2p.c
```

These nine files are the *single source* compiled on **all three vendors**
without modification — only the compiler, the module environment, and the
device-selection environment variable differ. The per-machine `OpenMP/`
directories contain copies of these sources next to the scripts that build them.

---

## 2. The nine OpenMP variants

| Binary | GPUs | Halo exchange strategy |
|---|---|---|
| `1-omp` | 1 | none — single-device baseline, the speedup denominator |
| `2-omp` | 2 | blocking exchange: `target update` on the default stream |
| `2-omp-stream` | 2 | one host thread, `target ... nowait`, hidden-helper tasks |
| `2-omp-stream-omp` | 2 | one host thread **per device**, explicit `nowait` + `taskwait` |
| `2-omp-stream-omp-p2p` | 2 | as above, on a patched runtime doing **direct GPU→GPU** copies |
| `4-omp` | 4 | blocking |
| `4-omp-stream` | 4 | one host thread, `nowait` |
| `4-omp-stream-omp` | 4 | one host thread per device |
| `4-omp-stream-omp-p2p` | 4 | one host thread per device, direct P2P |

The progression is deliberate: each step removes one serialisation point.
`-omp-stream-omp-p2p` is the only variant that requires anything outside a stock
toolchain (see §6).

The native `HIP/`, `CUDA/`, and `SYCL/` trees mirror this naming (`2-hip-stream-omp-p2p`,
`4-cuda-stream`, …) so a variant can be compared model-to-model row by row.

---

## 3. Problem definition

Explicit finite-difference solve of

> ∂u/∂t = ∇·(κ(x,y,z) ∇u)

on an `N × N × N` grid with a spatially varying coefficient, Dirichlet
boundaries, decomposed in the slowest-varying dimension. Each timestep needs one
halo-face exchange per internal boundary.

**Timestep derivation** (identical in `submit.sh` and `verify.sh` — keep them in
sync or you are verifying a different stability regime than you benchmark):

```
h  = 1 / (N - 1)
dt = h² / (12 · κ_max)          κ_max = 0.95
DT = STEPS · dt                 # the value passed to the binary
```

**Production sweep:** `N ∈ {512, 640, 768, 896, 1024, 1152, 1280}`, `STEPS=500`,
**5 independent repetitions** per size (each a separate Slurm job), medians
reported.

---

## 4. Reproducing a machine

Everything runs from inside `<MACHINE>/Source-Results/OpenMP/` (and identically
in the native-model directory).

```bash
cd AMD-MI250X/Source-Results/OpenMP

# 1. Build + submit the full sweep (1 build job + N_RUNS × |NS| run jobs).
#    The run jobs carry an afterok dependency on the build.
./submit.sh 5

# 2. Watch it.
squeue -u $USER

# 3. Merge every results_*/timings.csv into summary.csv (medians over reps).
./aggregate.sh

# 4. Look at it without leaving the terminal (pure awk, no matplotlib).
./plot.sh -a
```

Narrowing the sweep:

```bash
NS="1024" ./submit.sh 1                    # one size, one repetition
FAMILIES="4" BASELINE=0 ./submit.sh 3      # 4-GPU family only
STEPS=1000 ./submit.sh 5                   # longer runs
```

> `submit.sh`'s first argument is the **repetition count, not the grid size**.
> `./submit.sh 512` is caught and rejected; use `NS="512" ./submit.sh 1`.
> Override with `FORCE_RUNS=1` if you really do want >20 repetitions.

### Script reference

| Script | Role |
|---|---|
| `Makefile` | Targets: `all` (2- and 4-GPU families), `gpu-baseline` (`1-omp`), `serial-ref` (CPU reference), `check-p2p-lib`, `clean` |
| `build.sh` | Slurm build job: purges modules, loads the toolchain, `make all gpu-baseline`, then **verifies every expected binary exists** and fails loudly if not |
| `submit.sh` | Orchestrator. Computes `DT` per `N`, submits one `run.sh` job per (N, repetition) with `afterok` on the build |
| `run.sh` | One job = one `N`, one repetition, all variants. Sets the per-variant environment, times each binary, writes `results_<jobid>_<ts>/timings.csv` |
| `aggregate.sh` | Merges all `results_*/timings.csv` → `summary.csv` with `min_sec` / `median_sec` / `max_sec` |
| `verify.sh` | **Correctness only.** Runs every variant against the CPU reference |
| `compare.py` | Cross-variant / cross-model comparison from the summary CSVs |
| `plot.sh` | ASCII charts from `summary.csv` — no GUI, no matplotlib |
| `clean.sh` | Removes binaries, `.txt` solutions (`-f out`), and result directories |

`plot.sh` locates its columns **by header name**, so it works on both the AMD/NVIDIA
schema (`variant,gpus,N,…`) and the Intel schema (which carries an extra `hier`
column). Modes: default bar chart at the largest `N`, `-g 2` / `-g 4` for a
clustered per-size view of one family, `-s` for speedup vs `1-omp`, `-a` for all.

---

## 5. Output schema

`results_<jobid>_<timestamp>/timings.csv` — one row per variant execution:

```
rep, job_id, variant, family, gpus, rocr_mask, N, dt, steps, wall_sec, reported_sec, status
```

* `wall_sec` — measured by the harness around the whole process.
* `reported_sec` — parsed from the binary's own `Time = … seconds` line. **This is
  the number used in every figure**; it excludes setup and allocation.
* `status` — `ok`, `timeout`, `runtime_error`, `missing_binary`, `missing_p2p_runtime`.

A failing variant never aborts the repetition (`run.sh` is deliberately not
`set -e`); it is recorded with a non-`ok` status and the sweep continues. Skipped
work is also listed in `results_*/skipped.log`.

`summary.csv` adds `min_sec`, `median_sec`, `max_sec` across repetitions. `N/A`
rows survive aggregation and are filtered by the plotting scripts.

---

## 6. Device selection and the P2P runtime

**Device masking.** On MI250X one GCD is used per physical module, so the mask
is `0,2` for two GPUs and `0,2,4,6` for four — pairing both GCDs of one module
would measure the on-package link rather than the inter-module fabric. The masks
are generated by `mask_for()` in `run.sh`; the NVIDIA and Intel trees apply the
equivalent selection through their own variables.

**Per-variant environment** (`run.sh` sets this per binary; `verify.sh` carries a
verbatim copy of the same block):

| Variant suffix | Host threads | Hidden helper threads | Notes |
|---|---|---|---|
| *(plain)* and `-omp-stream` | `OMP_NUM_THREADS=1` | `= gpus`, helper tasks **on** | `HSA_ENABLE_SDMA=1` |
| `-stream-omp` | `= gpus` | `0`, helper tasks **off** | `HSA_ENABLE_SDMA=1` |
| `-stream-omp-p2p` | `= gpus` | `0`, helper tasks **off** | `HSA_ENABLE_PEER_SDMA=1` + `LD_PRELOAD` |

**The P2P runtime.** The `-p2p` variants require a patched OpenMP offload runtime
that performs direct device-to-device transfers instead of staging through host
memory, plus a small shim that selects the copy engine and enables peer access:

```
OMPTARGET_P2P_LIB=/scratch/project_465003145/omptarget-p2p/lib
  ├── libomp.so
  └── libomptarget.so.19.0git
libp2p_copy_shim.so      # built from p2p_copy_shim.map, LD_PRELOADed ahead of the runtime
```

They are preloaded at run time only — nothing links against them. `make
check-p2p-lib` asserts their presence before the run jobs start, and both
`run.sh` and `verify.sh` re-check and **skip rather than silently mis-measure**
if a file is missing.

<!-- TODO: state whether the patched runtime is included in this artifact, or
     give the upstream revision + patch needed to rebuild it. -->

---

## 7. Correctness

Timing runs never write solution files (`WRITE_TXT=0`). Verification is a
separate job:

```bash
sbatch verify.sh                              # defaults: N=256, STEPS=100
sbatch --export=ALL,N=128,STEPS=50 verify.sh
sbatch --export=ALL,TOL=1e-9,RTOL=1e-12 verify.sh
```

`verify.sh` builds the serial CPU reference if needed, runs it to produce
`u_serial.txt`, then runs every variant with `WRITE_TXT=1` and compares:

```
pass if   |u_ref − u_gpu|  ≤  TOL + RTOL · |u_ref|      (default TOL=1e-10, RTOL=0)
```

It reports the worst absolute and relative difference, their locations, and the
count of cells outside tolerance, and exits non-zero on any failure — including
the "nothing ran" case, which is reported as `INCONCLUSIVE`, not as a pass.

> **Never plot numbers from `verify.sh`.** File I/O sits inside the measured
> region there. The only intended difference between `run.sh` and `verify.sh` is
> `WRITE_TXT`; if the two per-variant environment blocks ever drift apart,
> verification stops validating the code path that was benchmarked.

All variants are compiled with `-ffp-contract=off` so that FMA contraction does
not make bitwise comparison across models meaningless.

---

## 8. Performance model

`model/` contains a calibrated analytical model, not a roofline bound.

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
t_comm = faces · (α_p2p + M / β_p2p)
T_pred = N_s · [ t_bnd + max(t_int, t_comm) ]
```

`β_p2p` comes from the companion `../Bandwidth_Measurements/` results. Because γ
is calibrated from measurement, the prediction can err in **either** direction;
agreement is reported as MAPE, signed bias, and worst single-size miss, all
normalised by the measured time.

---

## 9. Software environment

### AMD MI250X (LUMI-G) — reference configuration

```bash
module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4
module load lumi-CrayPath
```

`CC=cc`, `-O3 -fopenmp -ffp-contract=off`. Slurm: `--partition=standard-g
--nodes=1 --gpus-per-node=8 --exclusive`; 30 min for the build, 90 min per run
job.

### NVIDIA H100

<!-- TODO: modules / compiler + version, offload arch flag, CUDA toolkit
     version, Slurm partition and account, device-selection variable
     (CUDA_VISIBLE_DEVICES mask), and how the P2P variant is enabled. -->

### Intel Data Center GPU Max 1550

<!-- TODO: oneAPI version, icx flags (-fiopenmp -fopenmp-targets=spir64),
     ZE_AFFINITY_MASK (note tile-vs-card selection — the analogue of the
     GCD-pairing caveat in §6), and the meaning of the extra `hier` column in
     this tree's summary.csv schema. -->

---

## 10. Reading the results without re-running

Nothing needs to be executed to inspect the data:

* `<MACHINE>/Source-Results/*/summary.csv` — the aggregated medians behind every figure.
* `<MACHINE>/Source-Results/*/results_*/` — per-job raw timings and logs, untouched.
* `<MACHINE>/Source-Results/plots/` and `model/*.pdf` — the final figures.
* `run-<jobid>.out` — complete Slurm stdout, including the exact command line and
  environment printed for each variant before it ran.

Every run log echoes the resolved environment (`ROCR_VISIBLE_DEVICES`, thread
counts, SDMA settings, `LD_PRELOAD`) and the quoted command line, so any single
measurement in the CSVs can be traced back to the exact configuration that
produced it.

---

## 11. Citation

<!-- TODO: paper / preprint reference and BibTeX entry. -->

## 12. License

<!-- TODO. -->
