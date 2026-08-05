# OpenMP Offloading Intra-node

Artifact accompanying the paper *Evaluating OpenMP Offloading for Intra-node Multi-GPU Programming across NVIDIA, AMD, and Intel Architectures: A 3D Heat Transfer Case Study*.

How much of the achievable **intra-node multi-GPU** performance can portable
**OpenMP target offloading** deliver, and what does the halo-exchange strategy
cost? A 3D variable-coefficient heat-equation solver is decomposed across the
GPUs inside a single node and compared against native vendor models on **AMD
MI250X**, **NVIDIA H100**, and **Intel Max 1550**.

Every directory ships both the sources and the results they produced, so a
reader can re-run the pipeline or inspect the recorded data directly.

---

## Contents

| Folder | Purpose |
|---|---|
| [`OpenMP_Single_Source/`](OpenMP_Single_Source/README.md) | The nine OpenMP offload sources plus the serial CPU reference. Compiled on all three vendors **without modification** — only the compiler, modules, and device-selection variable differ. Start here to see what is actually being measured. |
| [`3D_Heat_Benchmark/`](3D_Heat_Benchmark/README.md) | **The main results.** Per-machine build/run/verify/aggregate pipelines, the recorded timings, the analytical performance model, and the paper's figures. One subfolder per machine, each with its own README. |
| [`HHT/`](HHT/README.md) | Sensitivity study: how much does the LLVM runtime's **hidden-helper-thread count** change the outcome? The same nine variants on all three machines under three helper configurations (HHT-0, HHT-GPU, HHT-8), compared as within-cell ratios so the hardware cancels out. |
| [`Bandwidth_Measurements/`](Bandwidth_Measurements/README.md) | Measured device-to-device bandwidth and latency per machine. Supplies the `β_p2p` and `α_p2p` constants the performance model in `3D_Heat_Benchmark/*/model/` is evaluated with. |

## Where to start

- **Just reading the numbers?** No execution needed — every
  `3D_Heat_Benchmark/<MACHINE>/Source-Results/*/summary.csv` holds the
  aggregated medians behind the figures, with the raw per-job logs beside them.
- **Re-running on your own machine?** Read the machine README under
  `3D_Heat_Benchmark/` first: accounts, partitions, modules, device masking, and
  (on AMD) a patched P2P runtime all differ per site.
- **Reproducing a specific figure?** Each folder README names the script that
  generates it.

All jobs assume **Slurm** and a node with at least four GPUs.

---

## Contact

| | |
|---|---|
| Maintainer | Ezhilmathi Krishnasamy |
| Email | ezhilmathi.krishnasamy@rudolfovo.eu |
| Affiliation | Rudolfovo Science and Technology, Novo Mesto, Slovenia<br>University of Ljubljana, Ljubljana, Slovenia<br>University of Luxembourg, Belval, Luxembourg |
| ORCID | [0000-0002-1971-4973](https://orcid.org/0000-0002-1971-4973) |
| Repository | https://github.com/ezhilmathi/OpenMP-Offloading-Intranode-Artifact |
| Paper version | tag `vX.Y`, commit `SHA` |
| Archive | https://doi.org/10.5281/zenodo.XXXXXXX |

Please open an issue on the repository for problems with the artifact, or mail
the maintainer if the repository is unavailable.

## Citation

If you use this artifact, please cite **both** the software and the paper.

`CITATION.cff` in the repository root is machine-readable — GitHub renders a
"Cite this repository" button from it, and Zenodo reads it when minting the DOI.

**Software**

```bibtex
@software{krishnasamy_openmp_intranode_artifact_2026,
  author    = {Krishnasamy, Ezhilmathi},
  title     = {{OpenMP Offloading Intra-node: Artifact for "Evaluating OpenMP
               Offloading for Intra-node Multi-GPU Programming across NVIDIA,
               AMD, and Intel Architectures: A 3D Heat Transfer Case Study"}},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/ezhilmathi/OpenMP-Offloading-Intranode-Artifact}
}
```

**Paper**

```bibtex
@article{krishnasamy_openmp_intranode_2026,
  author  = {Krishnasamy, Ezhilmathi},
  title   = {Evaluating OpenMP Offloading for Intra-node Multi-GPU Programming
             across NVIDIA, AMD, and Intel Architectures: A 3D Heat Transfer
             Case Study},
  journal = {[JOURNAL]},
  year    = {2026},
  doi     = {[DOI]}
}
```

## License

Licensed under the **European Union Public Licence v1.2 (EUPL-1.2)**. Full text
in [`LICENSE`](LICENSE); attribution and third-party notices in
[`NOTICE`](NOTICE).

    Copyright (c) 2026 Ezhilmathi Krishnasamy
    Licensed under the EUPL-1.2

EUPL-1.2 is a copyleft licence: distributed derivative works must be released
under the EUPL or one of the compatible licences named in its appendix (GPL-2.0
and -3.0, AGPL-3.0, LGPL, MPL-2.0, EPL-1.0, CeCILL, OSL, and CC BY-SA 3.0 for
non-software). Reading, running, and citing the artifact carry no such
condition.

The recorded measurement data under `3D_Heat_Benchmark/*/Source-Results/` and
`Bandwidth_Measurements/` is released under **CC BY 4.0**.
