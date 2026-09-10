# CG: Scalability test for OpenMP Offloading for the conjugate gradient method

CG is solved with fp64 on 1, 2, and 4 devices within a single node.

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
