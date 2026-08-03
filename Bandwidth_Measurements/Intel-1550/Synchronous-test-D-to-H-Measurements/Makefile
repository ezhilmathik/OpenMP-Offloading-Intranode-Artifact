# =============================================================================
# Intel PVC (Ponte Vecchio) counterpart of the LUMI/MI250X Makefile.
#
#   LUMI                          ->  Intel
#   amdclang++ -x hip             ->  icpx -fsycl
#   --offload-arch=gfx90a         ->  -fopenmp-targets=spir64  (JIT to SPIR-V)
#                                     or -fsycl-targets=spir64_gen -Xs "-device pvc" (AOT)
#   version-N.hip  (HIP)          ->  sycl-N.cpp     (SYCL)
#   openmp-N.cc    (OMP offload)  ->  openmp-N.cc    (unchanged source)
#
# =============================================================================

CXX      := icpx
CC       := icx

# --- SYCL (replaces the HIP path) -------------------------------------------
#   -fsycl            : enable SYCL
#   -fiopenmp         : host OpenMP (omp_get_wtime + host-thread driven variants)
#   -ffp-contract=off : no FMA -> reproducible FP
#   -fno-fast-math    : icx enables fast-math by default; off = fewer surprises
SYCL_FLAGS := \
    -fsycl \
    -fiopenmp \
    -O3 \
    -ffp-contract=off \
    -fno-fast-math \
    -Wno-unused-result \
    -Wl,--no-warn-execstack

# Ahead-of-time codegen for PVC (much faster startup, longer build).
# Uncomment to enable; JIT to SPIR-V is the default above.
# SYCL_FLAGS += -fsycl-targets=spir64_gen -Xs "-device pvc"

# --- OpenMP target offload --------------------------------------------------
#   -fiopenmp -fopenmp-targets=spir64  is the Intel analogue of
#   -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -march=gfx90a
OMP_FLAGS := \
    -fiopenmp \
    -fopenmp-targets=spir64 \
    -O3 \
    -ffp-contract=off \
    -fno-fast-math \
    -Wno-unused-result \
    -Wl,--no-warn-execstack

LDLIBS := -lm

SYCL_BINS := version-1-sycl version-2-sycl version-4-sycl
OMP_BINS  := version-1-off  version-2-off  version-4-off
ALL_BINS  := $(SYCL_BINS) $(OMP_BINS)

.PHONY: all sycl omp clean

all: sycl omp

sycl: $(SYCL_BINS)

version-1-sycl: sycl-1.cpp
	$(CXX) $(SYCL_FLAGS) -o $@ $< $(LDLIBS)

version-2-sycl: sycl-2.cpp
	$(CXX) $(SYCL_FLAGS) -o $@ $< $(LDLIBS)

version-4-sycl: sycl-4.cpp
	$(CXX) $(SYCL_FLAGS) -o $@ $< $(LDLIBS)

omp: $(OMP_BINS)

version-1-off: openmp-1.cc
	$(CXX) $(OMP_FLAGS) -o $@ $< $(LDLIBS)

version-2-off: openmp-2.cc
	$(CXX) $(OMP_FLAGS) -o $@ $< $(LDLIBS)

version-4-off: openmp-4.cc
	$(CXX) $(OMP_FLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f $(ALL_BINS)
