# ---------------------------------------------------------------------------
# 3D heat equation, SYCL (icpx -fsycl), Intel Data Center GPU Max 1550 (PVC).
# Compilers come from the modules loaded in build.sh (intel-toolkit).
#
# Counterpart of the OpenMP Makefile. Same discipline: ONE flag set for EVERY
# variant, so a 2-sycl vs 2-sycl-stream comparison measures the algorithmic
# change and nothing else.
# ---------------------------------------------------------------------------

# NOTE: `CXX ?= icpx` does NOT work here. GNU make predefines CXX=g++ and
# CC=cc as built-in variables, and ?= only assigns when a variable is
# *undefined* -- origin 'default' counts as defined, so ?= is a no-op and the
# whole build silently goes to g++/cc. (`make flags` on the OpenMP Makefile
# will show `CC = cc` for the same reason; it only works there if the
# intel-toolkit module happens to export CC.) Testing origin overrides the
# built-in default while still honouring CXX=... from the environment or the
# command line.
ifeq ($(origin CXX),default)
  CXX := icpx
endif
ifeq ($(origin CC),default)
  CC := icx
endif

# ---------------------------------------------------------------------------
# AOT vs JIT.  AOT=1 is the default and it matters for measurement, not just
# for convenience: with JIT the SPIR-V -> PVC translation happens at the FIRST
# kernel submission, which is inside the region the solver times itself. That
# puts a one-off compile (0.5-3 s, and it scales with kernel count) into the
# first repetition of every variant. -fsycl-targets=spir64_gen removes it.
#
# Set AOT=0 only if ocloc is missing from the toolkit module; then also export
# SYCL_CACHE_PERSISTENT=1 (run.sh does) and treat rep 1 as a warm-up.
# ---------------------------------------------------------------------------
AOT ?= 1

# ---------------------------------------------------------------------------
# ONE flag set for EVERY variant.
#
#   -fsycl              enable SYCL
#   -fiopenmp           host OpenMP: omp_get_wtime() in every source, and the
#                       *-stream-omp / *-stream-omp-p2p variants drive their
#                       devices from explicit host OpenMP threads. This is HOST
#                       OpenMP only -- no -fopenmp-targets, no offload runtime.
#   -ffp-contract=off   no FMA contraction -> bitwise-comparable output when
#                       diffing against the serial reference
#   -fno-fast-math      same reason as the OpenMP build: reproducible FP, and
#                       it is the setting that survives the icx/icpx 2024.x
#                       codegen issues. If one source ever fails to compile,
#                       do NOT drop that file alone to -O2 -- drop the whole
#                       OPT line, so the comparison stays fair.
# ---------------------------------------------------------------------------
OPT        := -O3 -fno-fast-math
SYCL_FLAGS := -fsycl -fiopenmp -ffp-contract=off
LDFLAGS    := -Wl,--no-warn-execstack
LIBS       := -lm

ifeq ($(AOT),1)
SYCL_FLAGS += -fsycl-targets=spir64_gen -Xs "-device pvc"
endif

# DPCT-migrated sources include <dpct/dpct.hpp>. intel-dpct usually puts the
# headers on the default include path; if the module exports a root, use it.
ifneq ($(DPCT_BUNDLE_ROOT),)
SYCL_FLAGS += -I$(DPCT_BUNDLE_ROOT)/include
endif

GPU_CXXFLAGS := $(OPT) $(SYCL_FLAGS)
CPU_CFLAGS   := $(OPT) -fiopenmp -ffp-contract=off

# ---------------------------------------------------------------------------
# Targets: the 2-GPU and 4-GPU families.
# Binary name == source name minus .cpp, matching the OpenMP layout.
# ---------------------------------------------------------------------------
BINS_2 := 2-sycl 2-sycl-stream 2-sycl-stream-omp 2-sycl-stream-omp-p2p
BINS_4 := 4-sycl 4-sycl-stream 4-sycl-stream-omp 4-sycl-stream-omp-p2p

BASELINE_BINS := 1-sycl
REF_BINS      := serial

ALL_BINS := $(BINS_2) $(BINS_4) $(BASELINE_BINS) $(REF_BINS)

.PHONY: all two four baseline gpu-baseline serial-ref clean flags

all: two four

flags:
	@echo "CXX          = $(CXX)"
	@echo "AOT          = $(AOT)"
	@echo "GPU_CXXFLAGS = $(GPU_CXXFLAGS)"
	@echo "CPU_CFLAGS   = $(CPU_CFLAGS)"

# ---------------------------------------------------------------------------
# 2-GPU family
# ---------------------------------------------------------------------------
two: $(BINS_2)

2-sycl: 2-sycl.cpp
	$(CXX) $(GPU_CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

2-sycl-stream: 2-sycl-stream.cpp
	$(CXX) $(GPU_CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

2-sycl-stream-omp: 2-sycl-stream-omp.cpp
	$(CXX) $(GPU_CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

2-sycl-stream-omp-p2p: 2-sycl-stream-omp-p2p.cpp p2p_shim.hpp
	$(CXX) $(GPU_CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# 4-GPU family
# ---------------------------------------------------------------------------
four: $(BINS_4)

4-sycl: 4-sycl.cpp
	$(CXX) $(GPU_CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

4-sycl-stream: 4-sycl-stream.cpp
	$(CXX) $(GPU_CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

4-sycl-stream-omp: 4-sycl-stream-omp.cpp
	$(CXX) $(GPU_CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

4-sycl-stream-omp-p2p: 4-sycl-stream-omp-p2p.cpp p2p_shim.hpp
	$(CXX) $(GPU_CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# Single-card baseline (production: speedup denominator)
# ---------------------------------------------------------------------------
gpu-baseline: $(BASELINE_BINS)

1-sycl: 1-sycl.cpp
	$(CXX) $(GPU_CXXFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# CPU reference -- correctness only. Not built by build.sh, not run by run.sh.
# Plain C, no SYCL: built with icx.
# ---------------------------------------------------------------------------
serial-ref: $(REF_BINS)

serial: heat3D_variable_coeff.c
	$(CC) $(CPU_CFLAGS) $(LDFLAGS) -o $@ $< $(LIBS)

baseline: gpu-baseline serial-ref

clean:
	rm -f $(ALL_BINS) icpx_build_*.log
