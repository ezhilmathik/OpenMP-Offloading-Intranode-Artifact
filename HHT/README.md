# HHT — Hidden-Helper-Thread Sensitivity Study

How much does the LLVM OpenMP runtime's **hidden-helper-thread count** change
the result? The same nine benchmark variants are run on all three machines under
three helper configurations, and every number reported here is a **ratio taken
within a single (machine, variant, N) cell**, so the hardware cancels out.

The question is not which GPU is faster. It is: when one configuration beats
another on the H100, is that a property of the **code** or of the **H100**?

Main results live in [`../3D_Heat_Benchmark/`](../3D_Heat_Benchmark/README.md);
this folder is the sensitivity study behind the helper-thread choice made there.

## The three configurations

| Folder suffix | Label | Hidden helper threads |
|---|---|---|
| `-01` | **HHT-0** | disabled |
| `-02` | **HHT-GPU** | one per GPU (2 or 4) |
| `-03` | **HHT-8** | fixed total of 8 (libomp's own default) |

Host threads are a property of the variant, not of the configuration:

| Variant | Version | Host threads |
|---|---|---|
| `{2,4}-omp` | 1 | 1 |
| `{2,4}-omp-stream` | 2 | 1 |
| `{2,4}-omp-stream-omp` | 3 | = GPUs |
| `{2,4}-omp-stream-omp-p2p` | 4 | = GPUs |

## Layout

```
HHT/
├── AMD-MI250X-{01,02,03}/OpenMP/     # one full pipeline copy per configuration
├── NVIDIA-H100-{01,02,03}/OpenMP/
├── Intel-1550-{01,02,03}/OpenMP/
├── gain.py                           # cross-machine comparison + figures
└── plots-approach/OpenMP/            # generated output
    ├── allN_gain_OpenMP.pdf          # <- used in the report
    ├── decision_OpenMP.pdf           # <- used in the report
    ├── efficiency_OpenMP.pdf         # sanity check — read this first
    ├── heatmap_OpenMP.pdf
    ├── portability3d_OpenMP.pdf
    ├── N{0512..1280}_gain_OpenMP.pdf
    └── gains_OpenMP.csv              # the full table behind every figure
```

Each `<MACHINE>-<NN>/OpenMP/` is a complete, self-contained copy of the run
pipeline, and `Makefile`, `build.sh`, `submit.sh`, `aggregate.sh`, `plot.sh`,
and `clean.sh` are identical to the corresponding machine in
`../3D_Heat_Benchmark/`. **Only `run.sh` and `verify.sh` differ** — they carry
the per-variant hidden-helper-thread policy, which is the whole point of this
study. Build instructions, modules, device masking, and the AMD P2P runtime are
unchanged; see that machine's README.

To confirm nothing else has drifted:

    diff -qr AMD-MI250X-01/OpenMP \
             ../3D_Heat_Benchmark/AMD-MI250X/Source-Results/OpenMP \
        --exclude='results_*' --exclude='run-*.out' --exclude='*.csv' \
        --exclude='*.so' --exclude='*.o'

## Regenerating the figures

`gain.py` reads `<folder>/OpenMP/summary.csv` from each of the nine directories,
so run `./aggregate.sh` inside any directory whose `results_*/` have changed
first. Needs `numpy`, `pandas`, `matplotlib`.

```bash
cd HHT

python3 gain.py --list          # check discovery only — writes nothing
python3 gain.py --format pdf    # regenerate everything into plots-approach/
```

The script is locked to the current working directory: `--root` and a positional
root are accepted for backward compatibility but ignored. Run it from `HHT/`.

Useful options:

```bash
python3 gain.py --ref 01                    # HHT-0 as the control (pins it to 1.00)
python3 gain.py --variants 4-omp-stream-omp-p2p
python3 gain.py --complete-cells-only       # only cells measured on all 9 series
python3 gain.py --min-baseline-efficiency 0.6
```

## How to read it

**Gain**, within each (machine, variant, N):

```
gain = t(reference) / t(this configuration)      # times normalised per timestep

gain > 1  faster than the reference here
gain = 1  identical
gain < 1  slower
```

The default reference is `gmean` — the geometric mean of all three
configurations in that cell — so every configuration takes a real value.
Naming a configuration instead (`--ref 01`) pins it at exactly 1.00 everywhere,
which hides how good or bad its baseline actually was.

Times are divided by the step count before the ratio, so a 500-vs-501 step
difference cannot leak in.

**Read `efficiency_OpenMP.pdf` before anything else.** It shows within-configuration
parallel efficiency (`speedup / gpus`). A cell pinned at exactly `1/gpus` — 0.50
on 2 GPUs, 0.25 on 4 — means the extra GPUs contributed nothing, so any "gain"
measured against that cell is a **bug fix in the denominator, not a speedup**.
`gain.py` prints these to the console and flags the affected x-axis labels in
red.

**The two report figures:**

- `allN_gain_OpenMP.pdf` — every N as a panel on one pdf page. Bars are speedup
  over each machine's *own* single-GPU run, so a fast and a slow machine share
  an axis. Colour = machine, hatch = configuration. Dashed line at 1.0 (extra
  GPUs bought nothing); dotted lines at perfect scaling, drawn only across the
  variants they apply to. The 1-GPU wall times that were divided out are printed
  above each panel.
- `decision_OpenMP.pdf` — one panel per machine, x = N, one line per
  configuration, aggregated across variants with a geometric mean. The
  configuration to recommend is the one above 1.0 in **every** panel at **every**
  N. The console prints the matching maximin table: safest choice (best
  worst-case) versus best on average, and whether the ranking agrees across
  machines.

Run-to-run spread is 1–6%, so configurations within ~1.06 of each other are not
distinguishable — prefer the simpler code in that case.

## Notes

- `gain.py` reports coverage holes: `(variant, N)` cells missing from some
  series become silent gaps in every figure. Fill them, or use
  `--complete-cells-only`.
- The `results_*/` directory names do **not** reliably encode the configuration
  (they derive from the `HELPERS` default in `run.sh`). Confirm a run's actual
  helper settings from the resolved environment echoed in its `run.log`.
- `AMD-MI250X-01/OpenMP/` contains `p2p_copy_shim.c`, which is missing from
  `../3D_Heat_Benchmark/AMD-MI250X/Source-Results/OpenMP/` — copy it there so
  the P2P shim can be rebuilt from source.
