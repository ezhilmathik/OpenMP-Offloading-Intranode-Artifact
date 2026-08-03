# ---------------------------------------------------------------------------
# P2P bandwidth benchmarks, AMD MI250X (gfx90a) on LUMI.
# Compilers come from the modules loaded in build.sh (PrgEnv-amd + rocm).
# ---------------------------------------------------------------------------

# Patched OpenMP offload runtime providing direct GPU-to-GPU transfers.
# Not needed at compile time; run.sh LD_PRELOADs it for the omp_off runs.
OMPTARGET_P2P_LIB ?= /scratch/project_465003145/omptarget-p2p/lib

HIP_FLAGS := -O3 -fopenmp -x hip --offload-arch=gfx90a -Wno-unused-result

OMP_FLAGS := \
    -fopenmp \
    -fopenmp-targets=amdgcn-amd-amdhsa \
    -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a \
    -O3

# ---------------------------------------------------------------------------
# Targets built for this study: v1 (unidirectional) and v2 (bidirectional).
# ---------------------------------------------------------------------------
HIP_BINS := version-1-hip version-2-hip
OMP_BINS := version-1-off version-2-off

# Four-GPU configurations: future work. Build with `make future`.
FUTURE_BINS := version-4-hip version-4-4-hip \
               version-4-off version-4-4-off

ALL_BINS := $(HIP_BINS) $(OMP_BINS) $(FUTURE_BINS)

.PHONY: all hip omp future clean check-p2p-lib

all: hip omp

# ---------------------------------------------------------------------------
# HIP (amdclang++)
# ---------------------------------------------------------------------------
hip: $(HIP_BINS)

version-1-hip: hip-1.hip
	amdclang++ $(HIP_FLAGS) -o $@ $<

version-2-hip: hip-2.hip
	amdclang++ $(HIP_FLAGS) -o $@ $<

version-4-hip: hip-4.hip
	amdclang++ $(HIP_FLAGS) -o $@ $<

version-4-4-hip: hip-4-4.hip
	amdclang++ $(HIP_FLAGS) -o $@ $<

# ---------------------------------------------------------------------------
# OpenMP target offload (amdclang++)
# ---------------------------------------------------------------------------
omp: $(OMP_BINS)

version-1-off: openmp-1.cc
	amdclang++ $(OMP_FLAGS) -o $@ $<

version-2-off: openmp-2.cc
	amdclang++ $(OMP_FLAGS) -o $@ $<

version-4-off: openmp-4.cc
	amdclang++ $(OMP_FLAGS) -o $@ $<

version-4-4-off: openmp-4-4.cc
	amdclang++ $(OMP_FLAGS) -o $@ $<

# ---------------------------------------------------------------------------
# Four-GPU configurations, both frameworks
# ---------------------------------------------------------------------------
future: $(FUTURE_BINS)

# ---------------------------------------------------------------------------
# Confirm the P2P runtime is present before the run jobs need it.
# ---------------------------------------------------------------------------
check-p2p-lib:
	@test -f "$(OMPTARGET_P2P_LIB)/libomp.so" || { echo "ERROR: libomp.so not found in $(OMPTARGET_P2P_LIB)"; exit 1; }
	@test -f "$(OMPTARGET_P2P_LIB)/libomptarget.so.19.0git" || { echo "ERROR: libomptarget runtime not found in $(OMPTARGET_P2P_LIB)"; exit 1; }
	@echo "P2P runtime OK: $(OMPTARGET_P2P_LIB)"

clean:
	rm -f $(ALL_BINS)
