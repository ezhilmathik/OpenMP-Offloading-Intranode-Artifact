# ---------------------------------------------------------------------------
# Unstructured CG benchmark -- LUMI-G (AMD MI250X, gfx90a).
#
# Load the toolchain first (build.sh does this for you):
#
#     module load LUMI/25.09 partition/G craype-accel-amd-gfx90a
#     module load PrgEnv-amd rocm/6.4.4 lumi-CrayPath
#
# CPU        : cc -fopenmp                (correctness reference only)
# GPU OpenMP : cc -fopenmp --offload-arch=gfx90a
#
# NO CUDA TARGETS. The 1/2/4-cuda.cu sources are part of the NVIDIA study and
# are simply not referenced here; nvcc does not exist on this machine.
#
# fp64 only, deliberately: unpreconditioned CG on an ill-conditioned matrix
# stagnates in fp32 -- the residual stops decreasing and the iteration count
# becomes meaningless, so an fp32 "result" would not be comparable.
#
# ---------------------------------------------------------------------------
# ONE GCD PER CARD
#
# An MI250X is two GCDs and a LUMI-G node holds four cards, so the OpenMP
# runtime sees EIGHT devices. This study uses one GCD per card, so "4 GPUs"
# means four separate cards, not two cards' worth of GCDs. That is selected at
# RUN time through ROCR_VISIBLE_DEVICES (see production.sh), not here -- the
# binaries are identical either way.
#
# ---------------------------------------------------------------------------
# SOURCE -> BINARY
#
#   cg_omp.c        -> cg-cpu
#   <k>-omp.c       -> cg-ompgpu<k>       for k = 1, 2, 4
#   <k>-omp-halo.c  -> cg-ompgpu<k>halo   for k = 2, 4
# ---------------------------------------------------------------------------

CC   ?= cc
ARCH ?= gfx90a

P2P_SHIM  := libp2p_copy_shim.so
ROCM_PATH ?= /opt/rocm

# Patched OpenMP offload runtime providing direct GCD-to-GCD transfers.
# Not needed at compile time; the run scripts put it on LD_LIBRARY_PATH.
OMPTARGET_P2P_LIB ?= /scratch/project_465003145/omptarget-p2p/lib


check-p2p-lib:
	@test -f "$(P2P_SHIM)" || \
	  { echo "ERROR: $(P2P_SHIM) not built -- run 'make $(P2P_SHIM)'"; exit 1; }
	@test -f "$(OMPTARGET_P2P_LIB)/libomp.so" || \
	  { echo "ERROR: libomp.so not found in $(OMPTARGET_P2P_LIB)"; exit 1; }
	@ls "$(OMPTARGET_P2P_LIB)"/libomptarget.so* >/dev/null 2>&1 || \
	  { echo "ERROR: libomptarget runtime not found in $(OMPTARGET_P2P_LIB)"; exit 1; }
	@echo "P2P runtime OK: $(OMPTARGET_P2P_LIB) + $(P2P_SHIM)"


.DELETE_ON_ERROR:

OPT := -O3

# -ffp-contract=off: no FMA contraction, so every variant does the arithmetic
# in the same order and their iteration counts are directly comparable.
# Without this they drift apart and you cannot tell a real bug from a rounding
# difference. This is what verify.sh check 2 relies on.
CPU_CFLAGS     := $(OPT) -fopenmp -ffp-contract=off
OMP_GPU_CFLAGS := $(OPT) -fopenmp --offload-arch=$(ARCH) -ffp-contract=off \
                  -fno-openmp-target-xteam-reduction

LIBS := -lm

OMP_SRCS := $(wildcard 1-omp.c 2-omp.c 4-omp.c)
OMP_BINS := $(patsubst %-omp.c,cg-ompgpu%,$(OMP_SRCS))

HALO_SRCS := $(wildcard 2-omp-halo.c 4-omp-halo.c)
HALO_BINS := $(patsubst %-omp-halo.c,cg-ompgpu%halo,$(HALO_SRCS))

ALL_BINS := cg-cpu $(OMP_BINS) $(HALO_BINS) $(P2P_SHIM)

# ---------------------------------------------------------------------------
# Peer-copy shim. Interposes hsa_amd_memory_async_copy_on_engine to grant both
# agents access to both allocations and select an available SDMA engine.
# Without it the stock runtime rejects every device-to-device omp_target_memcpy
# with HSA_STATUS_ERROR_INVALID_ARGUMENT and CG silently converges to garbage.
# ---------------------------------------------------------------------------
$(P2P_SHIM): p2p_copy_shim.c p2p_copy_shim.map
	$(CC) -O2 -fPIC -shared -o $@ $< \
	  -I$(ROCM_PATH)/include \
	  -Wl,--version-script=p2p_copy_shim.map \
	  -L$(ROCM_PATH)/lib -lhsa-runtime64 -ldl

.PHONY: all cpu ompgpu halo clean list check-p2p-lib

all: $(ALL_BINS)

cpu:    cg-cpu
ompgpu: $(OMP_BINS)
halo:   $(HALO_BINS)

# ---------------------------------------------------------------------------
# CPU reference. Built so verify.sh has a cross-variant reference to compare
# iteration counts against; NOT part of the production sweep on this machine.
# No -march=native: the Cray wrapper already targets the node's Trento cores,
# and production never runs this binary anyway.
# ---------------------------------------------------------------------------
cg-cpu: cg_omp.c matgen.h
	$(CC) $(CPU_CFLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# OpenMP target offload: cg-ompgpu1, cg-ompgpu2, cg-ompgpu4
# ---------------------------------------------------------------------------
cg-ompgpu%: %-omp.c matgen.h
	$(CC) $(OMP_GPU_CFLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# OpenMP target offload, packed halo: cg-ompgpu2halo, cg-ompgpu4halo
#
# Its own rule: the stem in cg-ompgpu%: %-omp.c would come out as "2halo"
# and go looking for 2halo-omp.c.
# ---------------------------------------------------------------------------
cg-ompgpu%halo: %-omp-halo.c matgen.h
	$(CC) $(OMP_GPU_CFLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# Confirm the P2P runtime is present before the run jobs need it.
# ---------------------------------------------------------------------------
check-p2p-lib:
	@test -f "$(OMPTARGET_P2P_LIB)/libomp.so" || \
	  { echo "ERROR: libomp.so not found in $(OMPTARGET_P2P_LIB)"; exit 1; }
	@ls "$(OMPTARGET_P2P_LIB)"/libomptarget.so* >/dev/null 2>&1 || \
	  { echo "ERROR: libomptarget runtime not found in $(OMPTARGET_P2P_LIB)"; exit 1; }
	@echo "P2P runtime OK: $(OMPTARGET_P2P_LIB)"

list:
	@echo "arch    : $(ARCH)"
	@echo "compiler: $(CC)"
	@echo "sources : cg_omp.c $(OMP_SRCS) $(HALO_SRCS)"
	@echo "targets : $(ALL_BINS)"

clean:
	rm -f cg-cpu cg-ompgpu1 cg-ompgpu2 cg-ompgpu4 cg-ompgpu2halo cg-ompgpu4halo \
	      $(P2P_SHIM)
