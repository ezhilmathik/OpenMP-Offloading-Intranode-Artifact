# ---------------------------------------------------------------------------
# Unstructured CG benchmark. NVIDIA H100 (sm_90), MareNostrum 5 acc partition.
# Compilers come from the modules loaded by build.sh / verify.sh /
# production.sh (GCC 13.2.0 + cuda/12.8 + clang 18.1.8).
#
# CPU        : clang -fopenmp -march=native   (reference only, not timed)
# GPU OpenMP : clang -fopenmp --offload-arch=sm_90
#
# NO CUDA TARGETS. The 1/2/4-cuda.cu sources belong to the separate CUDA-vs-
# OpenMP study; this directory is the packed-halo work and builds only the
# OpenMP target-offload family, so the A100, MI250X, PVC and H100 results are
# all the same code path.
#
# fp64 only, deliberately: unpreconditioned CG on an ill-conditioned matrix
# stagnates in fp32 -- the residual stops decreasing and the iteration count
# becomes meaningless, so an fp32 "result" would not be comparable.
#
# ---------------------------------------------------------------------------
# SOURCE -> BINARY
#
#   cg_omp.c        -> cg-cpu
#   <k>-omp.c       -> cg-ompgpu<k>       for k = 1, 2, 4
#   <k>-omp-halo.c  -> cg-ompgpu<k>halo   for k = 2, 4
#
# Picked up by wildcard, so dropping in a new file needs no edit here, nor in
# verify.sh or production.sh.
# ---------------------------------------------------------------------------

CC   := clang
ARCH ?= sm_90

.DELETE_ON_ERROR:

OPT := -O3

# -ffp-contract=off: no FMA contraction, so every variant does the arithmetic
# in the same order and their iteration counts are directly comparable.
# Without this they drift apart and you cannot tell a real bug from a rounding
# difference. verify.sh checks 2 and 3 depend on it.
CPU_CFLAGS     := $(OPT) -fopenmp -march=native -ffp-contract=off
OMP_GPU_CFLAGS := $(OPT) -fopenmp --offload-arch=$(ARCH) -ffp-contract=off

LDFLAGS      := -Wl,--no-warn-execstack
CPU_LIBS     := -lm
OMP_GPU_LIBS := -lm

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
# iteration counts; NOT part of the production sweep.
#
# -march=native is safe only because the build runs inside the allocation, on
# the same node type the timing runs on. Drop it if you build elsewhere.
# ---------------------------------------------------------------------------
cg-cpu: cg_omp.c matgen.h
	$(CC) $(CPU_CFLAGS) $(LDFLAGS) -o $@ $< $(CPU_LIBS)

# ---------------------------------------------------------------------------
# OpenMP target offload: cg-ompgpu1, cg-ompgpu2, cg-ompgpu4
# ---------------------------------------------------------------------------
cg-ompgpu%: %-omp.c matgen.h
	$(CC) $(OMP_GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(OMP_GPU_LIBS)

# ---------------------------------------------------------------------------
# OpenMP target offload, packed halo: cg-ompgpu2halo, cg-ompgpu4halo
#
# Its own rule: the stem in cg-ompgpu%: %-omp.c would come out as "2halo" and
# go looking for 2halo-omp.c.
# ---------------------------------------------------------------------------
cg-ompgpu%halo: %-omp-halo.c matgen.h
	$(CC) $(OMP_GPU_CFLAGS) $(LDFLAGS) -o $@ $< $(OMP_GPU_LIBS)

list:
	@echo "arch    : $(ARCH)"
	@echo "sources : cg_omp.c $(OMP_SRCS) $(HALO_SRCS)"
	@echo "targets : $(ALL_BINS)"

clean:
	rm -f cg-cpu cg-ompgpu1 cg-ompgpu2 cg-ompgpu4 cg-ompgpu2halo cg-ompgpu4halo
