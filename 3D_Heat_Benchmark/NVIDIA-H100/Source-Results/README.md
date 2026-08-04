# NVIDIA H100 — 3D Heat Equation, OpenMP Target Offload

Multi-GPU OpenMP offload variants of the 3D variable-coefficient heat equation,
measured on **NVIDIA H100 (sm_90)**, MareNostrum 5 `acc` partition (4 GPUs/node).

## Layout

```
NVIDIA-H100/
└── Source-Results/
    └── OpenMP/
        ├── Makefile                  # one flag set for every variant
        ├── build.sh                  # Slurm: compile + check all 9 binaries
        ├── submit.sh                 # driver: build once, then N_RUNS reps per N
        ├── run.sh                    # Slurm: one grid size, one repetition
        ├── verify.sh                 # Slurm: correctness vs CPU reference
        ├── aggregate.sh              # results_*/ -> summary.csv (medians)
        ├── plot.sh                   # ASCII bar / speedup plots from summary.csv
        ├── heat3D_variable_coeff.c   # serial CPU reference
        ├── 1-openmp.c                # 1-GPU baseline
        ├── 2-openmp{,-stream,-stream-omp,-stream-omp-p2p}.c
        ├── 4-openmp{,-stream,-stream-omp,-stream-omp-p2p}.c
        └── results_<jobid>_<timestamp>/
            ├── timings.csv           # one row per variant run
            ├── run.log               # full stdout (source of the compute timer)
            └── skipped.log
```

## Environment

```bash
module purge
module load EB/apps GCC/13.2.0 cuda/12.8 clang/18.1.8-cuda12.8
```

Compiler is **clang 18.1.8**, not NVHPC. All binaries use the same flags:
`-O3 -fopenmp --offload-arch=sm_90 -ffp-contract=off`
(FMA contraction off, so GPU results are bit-reproducible against the CPU reference).

Update `--account`, `--partition` and `--qos` in the `#SBATCH` headers for your site.

## Reproducing

```bash
cd Source-Results/OpenMP

./submit.sh 5              # build + 5 reps x 7 grid sizes (512...1280)
squeue -u $USER            # wait
./aggregate.sh             # -> summary.csv
./plot.sh -a               # bars, per-family clusters, speedups
```

Useful knobs: `NS="1024 1280"`, `FAMILIES="2"`, `BASELINE=0`, `STEPS=500`,
`SKIP_BUILD=1` (reuse binaries — `build.sh` runs `make clean`).

Single job:

```bash
sbatch --export=ALL,N=640,DT=0.0001074146,STEPS=500 run.sh
```

`dt = h²/(12·κ_max)` with `h = 1/(N-1)`, `κ_max = 0.95`; `DT = STEPS·dt`.

## Correctness

```bash
sbatch --export=ALL,N=128,STEPS=50 verify.sh
```

Builds the CPU reference, runs every variant with `WRITE_TXT=1`, and compares
solutions element-wise (tolerance `1e-10`). **Timings from `verify.sh` are
meaningless** — writing `O(N³)` lines dominates. Keep `N` small.

## Notes

- Timing runs set `WRITE_TXT=0` and never touch the CPU reference.
- `aggregate.sh` defaults to `TIME_FIELD=compute`, i.e. the solver's own
  `Time = ... seconds` line from `run.log`. `reported_sec` in `timings.csv` is
  unreliable on older runs (the regex captured `dt` from the parameter banner).
  Use `TIME_FIELD=wall` to cross-check.
- Device selection is via `CUDA_VISIBLE_DEVICES` with `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
  masks are `0`, `0,1`, `0,1,2,3`.
- Host-thread policy is fixed per variant in `run.sh`: `*-stream-omp*` get one
  application thread per GPU; all others get one, with LLVM hidden helpers.
