# CG: packed-halo conjugate gradient, OpenMP target offload

Unpreconditioned CG in fp64 on 1, 2 and 4 devices within a single node,
comparing a replicated exchange of the search direction p against a packed
halo that moves only the entries each device's rows actually read.

## Layout

    source/          the six CG variants, the matrix generator, and the
                     AMD peer-copy shim
    H100/            MareNostrum 5, 4x H100
    MI250X/          LUMI-G, 4x MI250X (one GCD per module)
    LRZ/             4x Intel Max 1550
                     rep4-6 are FLAT hierarchy runs; COMPOSITE was
                     measured first and discarded, see below

Each machine directory has its own Makefile and Slurm scripts with that
site's modules and account, three repetitions of the sweep, and the
plotting script that produced the figure.

## Reproducing

    cd <machine>
    ./verify.sh                 # correctness gate, exit 0 required
    GBS="5 10 20 30 40 50" REP=1 ./production.sh
    python3 plot_cg_bars.py --no-show

## Platform notes

**AMD MI250X (ROCm 6.4.4, AMD Clang 19).** Three workarounds, none needed
on NVIDIA or Intel:

1. `-fno-openmp-target-xteam-reduction`. The XTeam reduction path allocates
   its scratch against a single device, so two host threads reducing
   concurrently on two devices fault in the first dot product. Costs ~30%
   on the vector-operation time, ~1% of the full cycle.
2. `HSA_ENABLE_SDMA=1 HSA_ENABLE_PEER_SDMA=1`.
3. `libp2p_copy_shim.so`, LD_PRELOADed ahead of the patched runtime. It
   interposes `hsa_amd_memory_async_copy_on_engine`. Without it every
   device-to-device `omp_target_memcpy` is rejected with
   HSA_STATUS_ERROR_INVALID_ARGUMENT, and the run does not abort: it
   carries on with p unexchanged and converges to a wrong answer.

