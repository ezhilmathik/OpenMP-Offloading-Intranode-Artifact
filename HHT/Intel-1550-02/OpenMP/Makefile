# ---------------------------------------------------------------------------
# 3D heat equation, OpenMP target offload, Intel Data Center GPU Max 1550 (PVC).
# Compilers come from the modules loaded in build.sh (intel-toolkit).
# ---------------------------------------------------------------------------

CC ?= icx

# ---------------------------------------------------------------------------
# ONE flag set for EVERY variant. This is deliberate.
#
# The old openmp-batch.sh built non-stream binaries at -O3 (fast-math ON) and
# stream binaries at -O3 -fno-fast-math. That makes a 2-omp vs 2-omp-stream
# comparison confound the algorithmic change with a floating-point and
# optimisation change. For a measurement study every binary must be compiled
# identically.
#
# -fno-fast-math is the setting that survives the icx 2024.1.0 issues:
#   * host x86 "Register Coalescer" ICE at -O3 and -O2 with fast-math on
#   * stream kernels that BUILD at -O3 fast-math but abort at runtime with
#     "LEVEL_ZERO error: Failed to invoke deleted kernel"
# It also gives reproducible FP for diffing against the serial reference.
#
# If a source ever fails to compile here, do NOT quietly drop it to -O2 for
# that file alone -- drop the whole OPT line, so the comparison stays fair.
# ---------------------------------------------------------------------------
OPT       := -O3 -fno-fast-math
OMP_FLAGS := -fiopenmp -fopenmp-targets=spir64 -ffp-contract=off
LDFLAGS   := -Wl,--no-warn-execstack
LIBS      := -lm

GPU_CFLAGS := $(OPT) $(OMP_FLAGS)
CPU_CFLAGS := $(OPT) -fiopenmp -ffp-contract=off

# ---------------------------------------------------------------------------
# Targets: the 2-GPU and 4-GPU families.
# ---------------------------------------------------------------------------
BINS_2 := 2-omp 2-omp-stream 2-omp-stream-omp 2-omp-stream-omp-p2p
BINS_4 := 4-omp 4-omp-stream 4-omp-stream-omp 4-omp-stream-omp-p2p

BASELINE_BINS := 1-omp
REF_BINS      := serial

ALL_BINS := $(BINS_2) $(BINS_4) $(BASELINE_BINS) $(REF_BINS)

.PHONY: all two four baseline gpu-baseline serial-ref clean

all: two four

# ---------------------------------------------------------------------------
# 2-GPU family
# ---------------------------------------------------------------------------
two: $(BINS_2)

2-omp: 2-openmp.c
	$(CC) $(GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

2-omp-stream: 2-openmp-stream.c
	$(CC) $(GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

2-omp-stream-omp: 2-openmp-stream-omp.c
	$(CC) $(GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

2-omp-stream-omp-p2p: 2-openmp-stream-omp-p2p.c
	$(CC) $(GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# 4-GPU family
# ---------------------------------------------------------------------------
four: $(BINS_4)

4-omp: 4-openmp.c
	$(CC) $(GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

4-omp-stream: 4-openmp-stream.c
	$(CC) $(GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

4-omp-stream-omp: 4-openmp-stream-omp.c
	$(CC) $(GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

4-omp-stream-omp-p2p: 4-openmp-stream-omp-p2p.c
	$(CC) $(GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# Single-card baseline (production: speedup denominator)
# ---------------------------------------------------------------------------
gpu-baseline: $(BASELINE_BINS)

1-omp: 1-openmp.c
	$(CC) $(GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# CPU reference -- correctness only. Not built by build.sh, not run by run.sh.
# ---------------------------------------------------------------------------
serial-ref: $(REF_BINS)

serial: heat3D_variable_coeff.c
	$(CC) $(CPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

baseline: gpu-baseline serial-ref

clean:
	rm -f $(ALL_BINS) icx_build_*.log
