# ---------------------------------------------------------------------------
# 3D heat equation, native HIP, AMD MI250X (gfx90a) on LUMI.
#
# The Makefile is invoked by build.sh. The normal production entry point is:
#
#     ./submit 5
#
# submit first submits build.sh, then submits all run.sh jobs with an afterok
# dependency on the build job. Therefore, the run jobs start only after this
# Makefile has built all requested executables successfully.
# ---------------------------------------------------------------------------

HIPCXX    ?= amdclang++
HIP_FLAGS ?= -O3 -std=c++17 -fopenmp -ffp-contract=off -x hip --offload-arch=gfx90a -Wno-unused-result

CPUCC     ?= cc
CPU_FLAGS ?= -O3 -fopenmp -ffp-contract=off

LIBS      ?= -lm

# ---------------------------------------------------------------------------
# Targets built for this study.
# ---------------------------------------------------------------------------
BINS_2 := 2-hip 2-hip-stream 2-hip-stream-omp 2-hip-stream-omp-p2p
BINS_4 := 4-hip 4-hip-stream 4-hip-stream-omp 4-hip-stream-omp-p2p

BASELINE_BINS := 1-hip
REF_BINS      := serial

ALL_HIP_BINS := $(BINS_2) $(BINS_4) $(BASELINE_BINS)
ALL_BINS     := $(ALL_HIP_BINS) $(REF_BINS)

.PHONY: all two four baseline gpu-baseline serial-ref clean list

all: two four

# ---------------------------------------------------------------------------
# 2-GPU family
# ---------------------------------------------------------------------------
two: $(BINS_2)

2-hip: 2-hip.cpp
	$(HIPCXX) $(HIP_FLAGS) -o $@ $< $(LIBS)

2-hip-stream: 2-hip-stream.cpp
	$(HIPCXX) $(HIP_FLAGS) -o $@ $< $(LIBS)

2-hip-stream-omp: 2-hip-stream-omp.cpp
	$(HIPCXX) $(HIP_FLAGS) -o $@ $< $(LIBS)

2-hip-stream-omp-p2p: 2-hip-stream-omp-p2p.cpp
	$(HIPCXX) $(HIP_FLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# 4-GPU family
# ---------------------------------------------------------------------------
four: $(BINS_4)

4-hip: 4-hip.cpp
	$(HIPCXX) $(HIP_FLAGS) -o $@ $< $(LIBS)

4-hip-stream: 4-hip-stream.cpp
	$(HIPCXX) $(HIP_FLAGS) -o $@ $< $(LIBS)

4-hip-stream-omp: 4-hip-stream-omp.cpp
	$(HIPCXX) $(HIP_FLAGS) -o $@ $< $(LIBS)

4-hip-stream-omp-p2p: 4-hip-stream-omp-p2p.cpp
	$(HIPCXX) $(HIP_FLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# Single-GCD baseline.
# ---------------------------------------------------------------------------
gpu-baseline: $(BASELINE_BINS)

1-hip: 1-hip.cpp
	$(HIPCXX) $(HIP_FLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# CPU reference: manual correctness work only, not production timing.
# ---------------------------------------------------------------------------
serial-ref: $(REF_BINS)

serial: heat3D_variable_coeff.c
	$(CPUCC) $(CPU_FLAGS) -o $@ $< $(LIBS)

baseline: gpu-baseline serial-ref

list:
	@echo "HIP compiler : $(HIPCXX)"
	@echo "HIP flags    : $(HIP_FLAGS)"
	@echo "HIP binaries : $(ALL_HIP_BINS)"

clean:
	rm -f $(ALL_BINS)
