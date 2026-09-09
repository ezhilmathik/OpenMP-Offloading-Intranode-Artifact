# ---------------------------------------------------------------------------
# Unstructured CG benchmark -- Intel Data Center GPU Max 1550 (PVC).
# Compilers come from the modules loaded in build.sh (intel-toolkit).
# ---------------------------------------------------------------------------
#
# NO CUDA TARGETS. The 1/2/4-cuda.cu sources belong to the NVIDIA study; nvcc
# does not exist here and they are simply not referenced.
#
# ONE FLAG SET FOR EVERY VARIANT. This is deliberate: a cpu-vs-ompgpu2 or an
# ompgpu2-vs-ompgpu2halo comparison must not confound the algorithmic change
# with a floating-point or optimisation change.
#
# -fno-fast-math is the setting that survives the icx issues seen in the heat
# benchmark on this machine:
#   * host x86 "Register Coalescer" ICE at -O3 and -O2 with fast-math on
#   * kernels that BUILD at -O3 fast-math but abort at runtime with
#     "LEVEL_ZERO error: Failed to invoke deleted kernel"
# It also gives reproducible FP, which is what verify.sh checks 2 and 3 need.
#
# If a source ever fails to compile, do NOT quietly drop that one file to -O2
# -- drop the whole OPT line, so the comparison stays fair.
#
# -ffp-contract=off on top: no FMA contraction, so every variant does the
# arithmetic in the same order and the iteration counts are comparable.
#
# fp64 only: unpreconditioned CG on an ill-conditioned matrix stagnates in
# fp32, so an fp32 "result" would not be comparable.
#
# ---------------------------------------------------------------------------
# ONE DEVICE = ONE WHOLE CARD
#
# A Max 1550 is two stacks. Under ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE the Level
# Zero root device is the whole card and both stacks are available to the
# runtime, so "4 GPUs" means four complete cards. That is selected at RUN time
# (see production.sh); the binaries are identical either way.
#
# ---------------------------------------------------------------------------
# SOURCE -> BINARY
#
#   cg_omp.c        -> cg-cpu
#   <k>-omp.c       -> cg-ompgpu<k>       for k = 1, 2, 4
#   <k>-omp-halo.c  -> cg-ompgpu<k>halo   for k = 2, 4
# ---------------------------------------------------------------------------

CC ?= icx

OPT       := -O3 -fno-fast-math
OMP_FLAGS := -fiopenmp -fopenmp-targets=spir64 -ffp-contract=off
LDFLAGS   := -Wl,--no-warn-execstack
LIBS      := -lm

GPU_CFLAGS := $(OPT) $(OMP_FLAGS)
CPU_CFLAGS := $(OPT) -fiopenmp -ffp-contract=off

.DELETE_ON_ERROR:

OMP_SRCS := $(wildcard 1-omp.c 2-omp.c 4-omp.c)
OMP_BINS := $(patsubst %-omp.c,cg-ompgpu%,$(OMP_SRCS))

HALO_SRCS := $(wildcard 2-omp-halo.c 4-omp-halo.c)
HALO_BINS := $(patsubst %-omp-halo.c,cg-ompgpu%halo,$(HALO_SRCS))

ALL_BINS := cg-cpu $(OMP_BINS) $(HALO_BINS)

.PHONY: all cpu ompgpu halo clean list

all: $(ALL_BINS)

cpu:    cg-cpu
ompgpu: $(OMP_BINS)
halo:   $(HALO_BINS)

# ---------------------------------------------------------------------------
# CPU reference. Built so verify.sh has a cross-variant reference for the
# iteration counts; NOT part of the production sweep on this machine.
# ---------------------------------------------------------------------------
cg-cpu: cg_omp.c matgen.h
	$(CC) $(CPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# OpenMP target offload: cg-ompgpu1, cg-ompgpu2, cg-ompgpu4
# ---------------------------------------------------------------------------
cg-ompgpu%: %-omp.c matgen.h
	$(CC) $(GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# OpenMP target offload, packed halo: cg-ompgpu2halo, cg-ompgpu4halo
#
# Its own rule: the stem in cg-ompgpu%: %-omp.c would come out as "2halo" and
# go looking for 2halo-omp.c.
# ---------------------------------------------------------------------------
cg-ompgpu%halo: %-omp-halo.c matgen.h
	$(CC) $(GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

list:
	@echo "compiler: $(CC)"
	@echo "flags   : $(GPU_CFLAGS)"
	@echo "sources : cg_omp.c $(OMP_SRCS) $(HALO_SRCS)"
	@echo "targets : $(ALL_BINS)"

clean:
	rm -f cg-cpu cg-ompgpu1 cg-ompgpu2 cg-ompgpu4 cg-ompgpu2halo cg-ompgpu4halo \
	      icx_build_*.log
