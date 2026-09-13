# OpenMP offloading versus native GPU programming models

Benchmarks and measurement pipeline for a kernel-level comparison of OpenMP
target offloading against each vendor's native GPU programming model, on three
accelerators:

| Platform | Device used | Native model | OpenMP compiler |
|---|---|---|---|
| MeluXina, LuxProvide | 1 × NVIDIA A100-SXM4-40GB | CUDA (nvcc 12.6) | Clang 18.1.8 |
| MareNostrum 5 ACC, BSC | 1 × NVIDIA H100 64GB | CUDA (nvcc 12.8) | Clang 18.1.8 |
| LUMI-G, CSC | 1 GCD of an AMD MI250X (64 GB) | HIP (ROCm 6.4.4) | Cray `CC` / amdclang++ |
| SuperMUC-NG Phase 2, LRZ | 1 tile of an Intel Max 1550 (64 GB) | SYCL (icpx 2024.1) | icpx 2024.1 |

Three benchmarks are measured at identical memory footprints on every platform:
dense matrix multiplication, dense matrix-vector product and vector addition.
Each is written twice — once as an OpenMP `target` region and once in the native
model — with the same arithmetic in the same order, so that the two differ only
in how the iteration space is mapped onto threads.

## Files

### Benchmark sources

Each program allocates its data on the device explicitly (`omp_target_alloc`,
`cudaMalloc`, `hipMalloc`, `sycl::malloc_device`), runs one untimed warm-up call
followed by `REPS` timed calls, verifies the result against a sequential CPU
reference, and measures launch overhead with an empty kernel.

| Benchmark | OpenMP | CUDA | HIP | SYCL |
|---|---|---|---|---|
| Matrix multiplication | `mat-omp.cpp` | `mat-cuda.cu` | `mat-hip.cpp` | `mat-sycl.cpp` |
| Matrix-vector product | `mv-omp.cpp` | `mv-cuda.cu` | `mv-hip.cpp` | `mv-sycl.cpp` |
| Vector addition | `vec-omp.cpp` | `vec-cuda.cu` | `vec-hip.cpp` | `vec-sycl.cpp` |

The same three OpenMP sources are compiled unchanged on all four machines.
The HIP sources are produced from the CUDA sources with `hipify`, followed by
`patch_mv.py` (see *Helper scripts*).

Native kernels per benchmark:

* **Matrix multiplication** — `naive` (one thread per element of C, 32×8 block),
  `naive 1-D` (the same kernel with a 128×1 block: the OpenMP layout, used as a
  control), `tiled32` (32×32 tiles staged in shared/local memory) and two
  register-tiled variants kept as references.
* **Matrix-vector** — `thread/row` (the counterpart of the OpenMP loop) and
  `warp/row` (one warp, wavefront or sub-group per row, with a shuffle or
  sub-group reduction).
* **Vector addition** — `1-D` (one thread per element), a 2-D grid variant,
  a 128-bit (`double2` / `vec<double,2>`) variant, and a grid-stride variant
  that reproduces the OpenMP launch shape as a control.

OpenMP kernels per benchmark: `teams distribute parallel for collapse(2)` for
matrix multiplication; `teams distribute` over rows with an inner
`parallel for reduction` for matrix-vector, alongside a plain thread-per-row
variant; `teams distribute parallel for` for vector addition.

### Pipeline

| File | Purpose |
|---|---|
| `run_matmul.sh` | builds and runs one benchmark at one footprint; submits Slurm jobs from a login node, or runs in place inside an allocation |
| `collect.sh` | turns the raw results into one table per dataset, plus `analysis/summary.csv`, `analysis/overview.txt` and gnuplot data files |
| `collect.awk` | the parser behind `collect.sh` |
| `plot.gp` | optional gnuplot figures |

### Helper scripts

| File | Purpose |
|---|---|
| `patch_mv.py` | makes `mv-hip.cpp` wavefront-aware (64 lanes on CDNA instead of 32) and reports the AMD architecture name |
| `patch_print.py` | reports the AMD architecture name in `mat-hip.cpp` and `vec-hip.cpp` (cosmetic) |
| `patch_kernels.py` | adds the `KERNELS` selector to `mat-hip.cpp` or `mat-cuda.cu` |
| `gm_check.py` | diagnoses gaps in Nsight Systems GPU-metric sampling |

## Requirements

The pipeline detects the site automatically and loads the modules below. Use
`SITE=` to override the detection and `MODULES=` to override the module list.

| Site | Slurm | Modules |
|---|---|---|
| `meluxina` | account `p201103`, partition `gpu`, QoS `default` | `env/release/2024.1`, `Clang/18.1.8-GCCcore-13.3.0-CUDA-12.6.0` |
| `mn5` | account `ehpc681`, partition `acc`, QoS `acc_ehpc` | `EB/apps`, `GCC/13.2.0`, `cuda/12.8`, `clang/18.1.8-cuda12.8` |
| `lumi` | account `project_465003145`, partition `standard-g` | `LUMI/25.09`, `partition/G`, `craype-accel-amd-gfx90a`, `PrgEnv-amd`, `rocm/6.4.4`, `lumi-CrayPath` |
| `lrz` | account `pn67qe`, partition `general` | `slurm_setup`, `intel-toolkit` |

Substitute your own account. Every job requests one exclusive node and uses one
device: GPU 0 on NVIDIA (`CUDA_VISIBLE_DEVICES`), GCD 0 on AMD
(`ROCR_VISIBLE_DEVICES`) and tile 0 on Intel (`ZE_FLAT_DEVICE_HIERARCHY=FLAT`
with `ZE_AFFINITY_MASK`). `GPU_INDEX=` selects a different one.

Compiler flags are chosen so that all models round identically:
`--fmad=false` for nvcc, `-ffp-contract=off` for Clang, hipcc and icpx, plus
`-fno-fast-math` and `-fno-sycl-id-queries-fit-in-int` on Intel. Every result is
therefore expected to be **bitwise identical** to a sequential CPU reference,
which each program checks and reports.

## Running

Copy the sources and the four pipeline files into one directory and run, from a
login node:

```bash
BENCH=matmul ./run_matmul.sh     # dense matrix multiplication
BENCH=matvec ./run_matmul.sh     # dense matrix-vector product
BENCH=vecadd ./run_matmul.sh     # vector addition
./collect.sh                     # once the queue is empty
```

Each command submits six jobs: three memory footprints (30, 45 and 55 GB; 10, 20
and 30 GB on the 40 GB A100) times two stages.

* The **`time` stage** measures performance. Each program runs once, undisturbed,
  under a kernel trace (Nsight Systems on NVIDIA, rocprofv3 on AMD; on Intel the
  programs report their own device time through SYCL event profiling and
  `LIBOMPTARGET_PLUGIN_PROFILE`).
* The **`ncu` stage** collects hardware counters with Nsight Compute (NVIDIA),
  rocprofv3 (AMD) or VTune (Intel). Profiling distorts timing, so its durations
  are never used; the name is historical.

Useful knobs (all optional):

```bash
REPS=1                    # timed calls after the warm-up (default 3)
SIZES="30 45"             # memory footprints in GB
STAGES=time               # or "ncu"; default is both
KERNELS=tiled32           # matrix multiplication: run only these native kernels
WALL_TIME=06:00:00        # walltime of a time job
WALL_NCU=05:00:00         # walltime of a counter job
DRY_RUN=1                 # print the sbatch commands without submitting
```

Inside an interactive allocation the script runs the work in place instead of
submitting.

## Results

`collect.sh` writes, per dataset (site × benchmark × footprint × repetition):

* a printed table, also saved as `table.txt` in the dataset directory,
* `analysis/summary.csv` — every metric, one row per kernel and dataset,
* `analysis/overview.txt` — native versus OpenMP, one row per dataset,
* `analysis/plot_<site>[_<bench>].dat` — input for `plot.gp`.

It also prints a status line per stage (`finished`, `CANCELLED`, `FAILED`,
`NOT finished`) and lists missing files, so incomplete runs are visible.

Every dataset keeps the exact sources that produced it (`src/`), their
checksums, the loaded modules, compiler and profiler versions, the GPU model and
driver, and the launch settings, in `meta.txt`.

## Known limitations

These affect which metrics are available, and are stated in the paper:

* **Nsight Compute counters overflow** in kernels longer than roughly 25 s,
  producing zero or impossible values. `collect.awk` flags them as
  `invalid (ncu)` rather than reporting them.
* **Nsight Compute cannot profile LLVM 18 OpenMP binaries** ("Cuda is
  initialized before the tool"), so on NVIDIA the OpenMP kernels have no L1/L2
  hit rates. Occupancy and DRAM throughput for them come from Nsight Systems
  GPU-metric sampling instead.
* **Nsight Systems GPU-metric sampling stops** partway through some very long
  runs. `gm_check.py` diagnoses it; affected cells are reported from the
  Nsight Compute run instead.
* **The AMD `MemUnitStalled` counter** reports 0 for these kernels; the HBM
  counters (`FETCH_SIZE`, `WRITE_SIZE`) occasionally fail in a given run.
* **The Intel toolchain exposes neither per-kernel register usage nor a
  theoretical occupancy model**, so those rows are `n/a` for that platform.
  VTune supplies achieved occupancy, HBM and L3 bandwidth and XVE activity.
* **VTune's raw data is several GB per collection** and scales with kernel
  duration; collection can exhaust a home quota. The pipeline deletes the raw
  directory after exporting the CSV (`KEEP_VTUNE=1` keeps it). For the longest
  matrix multiplication kernels, use `KERNELS=tiled32` to keep collection
  feasible.
* **SYCL event timestamps wrap** for kernels longer than roughly 90 s on the
  Max 1550. `collect.awk` detects this by comparing with the wall time per call
  and reports the wall time instead, marked `wall*`.
* **Host memory**: at the largest footprint the host arrays reach about 55 GB,
  so jobs request the node's full memory (`--mem=0`). Without it the job is
  killed by the OOM killer.

## Reproducing the published tables

The tables report a 30 GB footprint in the main text and 45 GB in the appendix,
with the fastest native kernel per platform (tiled 32×32 for matrix
multiplication, one warp/wavefront/sub-group per row for matrix-vector, one
thread per element for vector addition) against the fastest OpenMP variant.

```bash
# on each machine
BENCH=matmul ./run_matmul.sh
BENCH=matvec ./run_matmul.sh
BENCH=vecadd ./run_matmul.sh
./collect.sh
grep -E '^\s+\[(time|ncu)\]' <(./collect.sh) | grep -v 'finished$'   # anything incomplete
```

On the Intel platform the matrix multiplication counter stage needs
`REPS=1 KERNELS=tiled32` to fit the available time and disk.
