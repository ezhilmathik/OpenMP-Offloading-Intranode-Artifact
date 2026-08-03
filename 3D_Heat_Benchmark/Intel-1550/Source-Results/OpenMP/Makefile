# ---------------------------------------------------------------------------
# 3D heat equation, OpenMP target offload, NVIDIA H100 (sm_90).
# MareNostrum 5 / acc partition. Compilers come from build.sh's modules
# (GCC 13.2.0 + cuda/12.8 + clang/18.1.8-cuda12.8 -- NOT NVHPC).
# ---------------------------------------------------------------------------

CC ?= clang

# One flag set for EVERY variant, so a 2-omp vs 2-omp-stream comparison is not
# confounded by a compiler difference.
#   --offload-arch=sm_90     H100
#   -ffp-contract=off        no FMA contraction (== NVHPC -Mnofma); bit-repro
#                            against the CPU reference
OPT       := -O3
OMP_FLAGS := -fopenmp --offload-arch=sm_90 -ffp-contract=off
LDFLAGS   := -Wl,--no-warn-execstack
LIBS      := -lm

GPU_CFLAGS := $(OPT) $(OMP_FLAGS)
CPU_CFLAGS := $(OPT) -fopenmp -ffp-contract=off

# ---------------------------------------------------------------------------
# Targets: the 2-GPU and 4-GPU families.
#   plain          : blocking halo exchange
#   stream         : single host thread, target nowait
#   stream-omp     : one host thread per device
#   stream-omp-p2p : ditto, using direct device-to-device transfers
#
# NOTE: unlike the AMD build there is no patched runtime and no LD_PRELOAD --
# the *-p2p binaries are ordinary executables here.
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
# Single-GPU baseline (production: speedup denominator)
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
	rm -f $(ALL_BINS)
