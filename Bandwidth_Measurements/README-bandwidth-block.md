## 1. `Bandwidth_Measurements/` — intra-node data movement

Measures **intra-node transfer bandwidth** on three GPU architectures,
comparing **OpenMP target offload** against each vendor's native programming
model. Five transfer scenarios per platform — synchronous and asynchronous
host-to-device, synchronous and asynchronous device-to-host, and peer-to-peer
between devices — 15 benchmark directories in total.

| Vendor | Device | Native model | OpenMP offload compiler | Offload target |
|---|---|---|---|---|
| NVIDIA | H100 | CUDA — `nvcc` (CUDA 12.8) | `clang++` 18.1.8 | `nvptx64`, `sm_90` |
| AMD | Instinct MI250X | HIP — `amdclang++` (ROCm 6.4.4) | `amdclang++` (ROCm 6.4.4) | `amdgcn-amd-amdhsa`, `gfx90a` |
| Intel | Data Center GPU Max 1550 | SYCL — `icpx` (oneAPI 2024.1.0) | `icpx` (oneAPI 2024.1.0) | `spir64` (JIT to SPIR-V) |

On AMD and Intel a single compiler drives both paths; on NVIDIA the two paths
use different compilers entirely, which matters when reading the H100 figures.

Measurements were taken on MareNostrum 5 (BSC), LUMI (CSC/EuroHPC) and
SuperMUC-NG (LRZ) respectively, so the Slurm directives and module trees differ
between vendor directories. Cross-vendor comparisons in the paper are of the
**OpenMP-versus-native relationship within each platform**, not of absolute
bandwidths between platforms.

### Reproducing the figures — no GPU required

The raw measurement data from every run is included, so all 15 figures
regenerate on any machine with Python, in seconds:

```bash
cd Bandwidth_Measurements
python3 -m venv ~/venv-openmp-offloading
source ~/venv-openmp-offloading/bin/activate
pip install -r requirements.txt

for v in NVIDIA-H100 AMD-MI250X Intel-1550; do (cd "$v" && ./all-plots.sh); done
```

Each figure is written to the corresponding `<directory>/plots_final/`. This is
the recommended path for artifact evaluation; re-measuring from scratch needs
the specific hardware above and roughly 67 node-hours in total.

### Layout

```
Bandwidth_Measurements/
├── NVIDIA-H100/  AMD-MI250X/  Intel-1550/     one directory per vendor,
│   ├── README.md                              structurally identical
│   ├── all-plots.sh  check-artifact.sh
│   ├── submit-all.sh  clean.sh
│   └── Synchronous-test-H-to-D-Measurements/  blocking   host → device
│       Synchronous-test-D-to-H-Measurements/  blocking   device → host
│       Asynchronous-test-H-to-D-Measurements/ non-block. host → device
│       Asynchronous-test-D-to-H-Measurements/ non-block. device → host
│       P2P-Measurements/                      device ↔ device, peer link
├── check-all.sh          consistency check across the whole artifact
├── clean.sh              tree-wide removal of binaries and build debris
└── requirements.txt      Python dependencies for figure generation
```

Every benchmark directory carries the OpenMP and native sources, a `Makefile`,
the `build.sh`/`run.sh`/`submit.sh` Slurm pipeline, five independent
repetitions under `results_*/`, and the plotting script that turns them into
the published figure.

### Platform caveats

Three differences affect how the numbers should be read, and are documented in
full in the vendor READMEs:

- **AMD MI250X** preloads an OpenMP offload runtime built from AMD's LLVM fork
  rather than the site ROCm module, and runs with the DMA engines disabled
  (`HSA_ENABLE_SDMA=0`). Both materially change the measured bandwidth.
- **Intel Max 1550** requires the Level Zero and OpenMP device-mapping controls
  to agree; the scripts set both and abort if they disagree. Published figures
  use `COMPOSITE`.
- **NVIDIA H100** needs no environment overrides at all, making it the most
  straightforward of the three to reproduce.

> **Full detail:** [`Bandwidth_Measurements/README.md`](Bandwidth_Measurements/README.md)
> — contact and citation metadata, complete toolchain table, claim map, data
> schema, provenance and licensing.
