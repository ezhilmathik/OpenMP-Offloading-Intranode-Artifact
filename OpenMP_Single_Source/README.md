# OpenMP_Single_Source

The canonical, machine-independent OpenMP target-offload sources. These are the
**single set of files compiled on all three vendors without modification** —
only the compiler, the module environment, and the device-selection variable
differ between machines.

```
OpenMP_Single_Source/
├── heat3D_variable_coeff.c     
├── 1-openmp.c                 
├── 2-openmp.c                  
├── 2-openmp-stream.c          
├── 2-openmp-stream-omp.c      
├── 2-openmp-stream-omp-p2p.c   
├── 4-openmp.c
├── 4-openmp-stream.c
├── 4-openmp-stream-omp.c
└── 4-openmp-stream-omp-p2p.c
```

Nothing is built here — there is no Makefile. See
[`../3D_Heat_Benchmark/`](../3D_Heat_Benchmark/Source-Results/README.md) for the variant
descriptions, the problem definition, and the run pipeline.

## Use

Each machine's `OpenMP/` directory already contains a copy of these files next
to the scripts that build them. Refresh them before reproducing:

```bash
cd OpenMP_Single_Source

for m in AMD-MI250X NVIDIA-H100 Intel-1550; do
    cp *.c "../3D_Heat_Benchmark/$m/Source-Results/OpenMP/"
done
```

Then follow the per-machine README:
[AMD-MI250X](../3D_Heat_Benchmark/AMD-MI250X/Source_Results/README.md) ·
[NVIDIA-H100](../3D_Heat_Benchmark/NVIDIA-H100/Source_Results/README.md) ·
[Intel-1550](../3D_Heat_Benchmark/Intel-1550/Source_Results/README.md)

To confirm the deployed copies have not drifted:

```bash
for m in AMD-MI250X NVIDIA-H100 Intel-1550; do
    echo "== $m"
    diff -q . "../3D_Heat_Benchmark/$m/Source-Results/OpenMP" 2>/dev/null |
        grep '\.c$' || echo "   identical"
done
```
