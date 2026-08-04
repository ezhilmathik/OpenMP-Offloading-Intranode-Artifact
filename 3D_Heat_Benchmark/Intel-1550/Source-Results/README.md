# Intel Max 1550 — 3D Heat Equation, OpenMP Target Offload

Multi-GPU OpenMP offload variants of the 3D variable-coefficient heat equation,
measured on **Intel Data Center GPU Max 1550 (Ponte Vecchio)**, 4 cards per node.

## Layout

```
Intel-1550/
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
        ├── 1-openmp.c                # 1-card baseline
        ├── 2-openmp{,-stream,-stream-omp,-stream-omp-p2p}.c
        ├── 4-openmp{,-stream,-stream-omp,-stream-omp-p2p}.c
        └── results_COMPOSITE_h<helpers>_<jobid>_<timestamp>/
            ├── timings.csv           # one row per variant run
            ├── run.log               # full stdout (source of the compute timer)
            └── skipped.log
```

## Environment

```bash
module load slurm_setup intel-toolkit
```

Compiler is **`icx`** from the oneAPI toolkit; all binaries are built with the
same optimisation flags so variants are directly comparable.

Update `--account` and `--partition` in the `#SBATCH` headers for your site.

## Device model

One OpenMP device is **one whole Max 1550 card (both stacks)**:

```
ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
ZE_AFFINITY_MASK=0  |  0,1  |  0,1,2,3
LIBOMPTARGET_DEVICES   unset   # must not downgrade a device to a single stack
ONEAPI_DEVICE_SELECTOR unset   # ZE_AFFINITY_MASK owns device selection
```

Both `COMPOSITE` and an unset `LIBOMPTARGET_DEVICES` are required; either alone
is not enough. `run.sh` is locked to `COMPOSITE`.

## Variants

| Binary | Halo exchange |
|---|---|
| `1-omp` | single-card baseline (speedup denominator) |
| `N-omp` | blocking |
| `N-omp-stream` | `target nowait`, one host thread |
| `N-omp-stream-omp` | one host thread per card |
| `N-omp-stream-omp-p2p` | as above, direct device-to-device |

`N` is 2 or 4. The `-p2p` binaries run on the stock runtime; if cross-device P2P
is not wired up, the runtime silently stages through the host, which shows up as
roughly halved effective bandwidth rather than an error.

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
sbatch --export=ALL,N=1280,DT=...,STEPS=500,HELPERS=8 run.sh
```

`dt = h²/(12·κ_max)` with `h = 1/(N-1)`, `κ_max = 0.95`; `DT = STEPS·dt`.

## Hidden-helper threads (`HELPERS`, default 8)

`N-omp` and `N-omp-stream` are driven by one application thread plus LLVM runtime
helper threads. The helper count is **not** tied to the device count: earlier runs
used `LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}` (2 or 4), which is *below*
libomp's own default of 8 and throttled the runtime. On Max 1550, going back to 8
cut 4-GPU `N=1280` from 23.9 s to 18.0 s (`-omp-stream`) and 29.5 s to 19.7 s
(`-omp`); the `*-stream-omp` variants, which disable helpers, barely moved.

`HELPERS=8` should reproduce the stock-default timings. Sweep it with
`HELPERS=16`, `HELPERS=32`; each value lands in its own `results_COMPOSITE_h*/`
directory.

`OMP_THREAD_LIMIT` is deliberately **not** set for `*-stream-omp*`: it also caps
threads per team inside the target regions, costing roughly 3x.

## Correctness

```bash
sbatch --export=ALL,N=128,STEPS=50 verify.sh
```

Builds the CPU reference (`make CC=icx serial-ref`), runs every variant with
`WRITE_TXT=1`, and compares solutions element-wise (tolerance `1e-10`).
**Timings from `verify.sh` are meaningless** — writing `O(N³)` lines dominates.
Keep `N` small.

## Notes

- Timing runs set `WRITE_TXT=0` and never touch the CPU reference.
- `aggregate.sh` defaults to `TIME_FIELD=compute`, i.e. the solver's own
  `Time = ... seconds` line from `run.log`. Use `TIME_FIELD=wall` to cross-check.
- `plot.sh` locates columns by header name, so it reads both the AMD/NVIDIA
  schema (`variant,gpus,N,...`) and the Intel one (`hier,variant,gpus,N,...`).