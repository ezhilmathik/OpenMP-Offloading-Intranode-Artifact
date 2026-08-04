# AMD MI250X — 3D Heat Equation, OpenMP Target Offload

Multi-GPU OpenMP offload variants of the 3D variable-coefficient heat equation,
measured on **AMD MI250X (gfx90a)**, LUMI `standard-g` partition.
One OpenMP device = **one GCD**, one GCD per physical MI250X module.

## Layout

```
AMD-MI250X/
└── Source-Results/
    └── OpenMP/
        ├── Makefile                  # one flag set for every variant
        ├── build.sh                  # Slurm: compile + check binaries + P2P runtime
        ├── submit.sh                 # driver: build once, then N_RUNS reps per N
        ├── run.sh                    # Slurm: one grid size, one repetition
        ├── verify.sh                 # Slurm: correctness vs CPU reference
        ├── aggregate.sh              # results_*/ -> summary.csv (medians)
        ├── plot.sh                   # ASCII bar / speedup plots from summary.csv
        ├── compare.py                # standalone solution diff (same criterion)
        ├── clean.sh                  # remove binaries / solutions / results
        ├── heat3D_variable_coeff.c   # serial CPU reference
        ├── 1-openmp.c                # 1-GCD baseline
        ├── 2-openmp{,-stream,-stream-omp,-stream-omp-p2p}.c
        ├── 4-openmp{,-stream,-stream-omp,-stream-omp-p2p}.c
        ├── libp2p_copy_shim.so       # engine-selection / peer-access shim
        ├── p2p_copy_shim.map         # symbol version script for the shim
        ├── summary.csv               # aggregated medians (committed results)
        ├── run-<jobid>.out           # Slurm stdout per job
        └── results_h<helpers>_<jobid>_<timestamp>/
            ├── timings.csv           # one row per variant run
            ├── run.log               # full stdout (source of the compute timer)
            └── skipped.log
```

## Environment

```bash
module --force purge
module load LUMI/25.09 partition/G
module load craype-accel-amd-gfx90a
module load PrgEnv-amd
module load rocm/6.4.4
module load lumi-CrayPath
```

Compiler is the Cray wrapper `cc` (PrgEnv-amd). All binaries use the same flags:
`-O3 -fopenmp -ffp-contract=off`. Offload arch comes from
`craype-accel-amd-gfx90a`, not from an explicit `--offload-arch`.

Update `--account` and `--partition` in the `#SBATCH` headers for your site.

## Device mapping

The 2- and 4-GPU codes use **one GCD per module**, so the masks skip odd GCDs:

```
ROCR_VISIBLE_DEVICES=0        # 1 GCD (baseline)
ROCR_VISIBLE_DEVICES=0,2      # 2 GCDs, 2 modules
ROCR_VISIBLE_DEVICES=0,2,4,6  # 4 GCDs, 4 modules
```

Jobs request `--gpus-per-node=8 --exclusive` (a whole node) and then mask down.

## Variants

| Binary | Halo exchange | Host threads | Extra runtime |
|---|---|---|---|
| `1-omp` | — (baseline) | 1 | `HSA_ENABLE_SDMA=1` |
| `N-omp` | blocking | 1 | `HSA_ENABLE_SDMA=1` |
| `N-omp-stream` | `target nowait` | 1 | `HSA_ENABLE_SDMA=1` |
| `N-omp-stream-omp` | one thread per GCD | `N` | `HSA_ENABLE_SDMA=1` |
| `N-omp-stream-omp-p2p` | direct GCD-to-GCD | `N` | `+HSA_ENABLE_PEER_SDMA=1`, `LD_PRELOAD` |

`N` is 2 or 4. Every variant gets `LIBOMP_NUM_HIDDEN_HELPER_THREADS=$HELPERS`
(default 8, libomp's own default).

## P2P runtime (required by the `-p2p` variants only)

The `-p2p` binaries run on a **patched OpenMP offload runtime** that performs
direct GPU-to-GPU transfers, preloaded at run time:

```
LD_PRELOAD = libp2p_copy_shim.so
           : $OMPTARGET_P2P_LIB/libomp.so
           : $OMPTARGET_P2P_LIB/libomptarget.so.19.0git

OMPTARGET_P2P_LIB   default /scratch/project_465003145/omptarget-p2p/lib
P2P_SHIM            default ./libp2p_copy_shim.so
```

`make check-p2p-lib` (run by `build.sh`) fails early if the runtime is missing.
`run.sh` and `verify.sh` re-check per variant and record
`missing_p2p_runtime` rather than aborting the sweep. Point `OMPTARGET_P2P_LIB`
at your own build if you are not on this project's scratch.

## Prerequisite: relax the hidden-helper guard

The `*-stream-omp` and `*-stream-omp-p2p` sources read
`OMP_NUM_HIDDEN_HELPER_THREADS` and `exit(EXIT_FAILURE)` unless it is `0`.
The current scripts never set it, so those four variants abort immediately and
land in the CSV as `runtime_error`. Relax the guards first:

```bash
python3 relax_hidden_helper_guard.py --dry-run *.c   # inspect
python3 relax_hidden_helper_guard.py *.c             # apply
```

A `WARNING: running without OMP_NUM_HIDDEN_HELPER_THREADS=0` line is then
expected and harmless. The guard documents a real fault: concurrent
`omp_target_memcpy_async` from several host threads corrupting task-team state
and faulting inside `__kmpc_barrier` on libomp 18.1.8 — close to the runtime in
use here, and closer still for the `-p2p` variant. **Prove one job survives
before committing to a full sweep.**

## Reproducing

```bash
cd Source-Results/OpenMP

./submit.sh 5              # build + 5 reps x 7 grid sizes (512...1280)
squeue -u $USER            # wait
./aggregate.sh             # -> summary.csv
./plot.sh -a               # bars, per-family clusters, speedups
```

Useful knobs: `NS="1024 1280"`, `FAMILIES="2"`, `BASELINE=0`, `STEPS=500`.

Single job:

```bash
sbatch --export=ALL,N=1280,DT=...,STEPS=500,HELPERS=8 run.sh
```

`dt = h²/(12·κ_max)` with `h = 1/(N-1)`, `κ_max = 0.95`; `DT = STEPS·dt`.
`verify.sh` derives `dt` the same way — keep the two in sync or you verify a
different stability regime than you benchmark.

## Hidden-helper threads (`HELPERS`, default 8)

Earlier revisions set `LIBOMP_NUM_HIDDEN_HELPER_THREADS=${gpus}` (2 or 4),
*below* libomp's default of 8: a helper blocked on a transfer cannot submit the
next task, so one helper per device starves the runtime. Measured on Max 1550,
raising it to 8 cut the 4-GPU `N=1280` `-omp-stream` time from 23.9 s to 18.0 s;
re-check on MI250X. Two related changes:

- A global `OMP_NUM_HIDDEN_HELPER_THREADS=0` was in the environment while
  `LIBOMP_NUM_HIDDEN_HELPER_THREADS` asked for helpers. Contradictory; removed.
- `OMP_THREAD_LIMIT=${gpus}` on the `*-stream-omp*` variants: thread-limit-var
  is inherited by the initial task **on the device**, so it capped each team
  inside the target regions at 2–4 threads (~3x cost on Intel). Removed.

Neither variable is set anywhere now; both are stripped with `env -u`.

Each `HELPERS` value lands in its own `results_h<helpers>_*/` directory, so
sweeps do not mix.

## Correctness

```bash
sbatch verify.sh                                  # N=256, 100 steps
sbatch --export=ALL,N=128,STEPS=50 verify.sh
sbatch --export=ALL,TOL=1e-9,RTOL=1e-9 verify.sh  # tighten
```

Pass criterion is `|ref - gpu| <= TOL + RTOL*|ref|` with `TOL=1e-6`,
`RTOL=1e-5` — numpy's `atol`/`rtol`, and the same defaults as `compare.py`, so
the two tools cannot disagree on the same files. A bit-exact match is not
achievable: the variants decompose the domain differently, so FMA contraction
and summation order put the agreement floor above `1e-10`. A real defect (wrong
halo exchange, a race, a missing sync) produces errors far larger than these
bounds. NaN/Inf are counted as violations explicitly.

`verify.sh` runs each variant under a **verbatim copy** of `run.sh`'s
per-variant environment, with `WRITE_TXT` flipped to 1. If you change the case
block in one, change it in the other, or you are no longer validating the code
path you benchmark. **Timings from `verify.sh` are meaningless** — file I/O sits
inside the measured region. Keep `N` small; solutions cost `O(N³)` lines.

Cross-check outside Slurm:

```bash
python3 compare.py --backend openmp
```

## Notes

- Timing runs set `WRITE_TXT=0` and never build or execute the CPU reference.
  Build it on demand with `make serial-ref`.
- `run.sh` defaults to `BASELINE=0`; `submit.sh` passes `BASELINE=1` so the
  1-GCD denominator is timed in every job.
- `aggregate.sh` merges `results_*/` into `summary.csv` (per-variant medians
  over repetitions); `TIME_FIELD=wall` cross-checks against the solver's timer.
- `plot.sh` locates columns by header name, so it reads both this schema
  (`variant,gpus,N,...`) and the Intel one (`hier,variant,gpus,N,...`).
- Remove generated files with `./clean.sh` (`./clean.sh -f out` for solutions).