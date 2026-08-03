# ---------------------------------------------------------------------------
# 3D heat equation, OpenMP target offload, AMD MI250X (gfx90a) on LUMI.
# Compilers come from the modules loaded in build.sh (PrgEnv-amd + rocm).
# ---------------------------------------------------------------------------

# Patched OpenMP offload runtime providing direct GPU-to-GPU transfers.
# Not needed at compile time; run.sh LD_PRELOADs it for the *-p2p runs.
OMPTARGET_P2P_LIB ?= /scratch/project_465003145/omptarget-p2p/lib

CC        ?= cc
OMP_FLAGS := -O3 -fopenmp -ffp-contract=off
CPU_FLAGS := -O3 -fopenmp -ffp-contract=off
LIBS      := -lm

# ---------------------------------------------------------------------------
# Targets built for this study: the 2-GPU and 4-GPU families.
#   plain    : blocking halo exchange
#   stream   : single host thread, target nowait
#   stream-omp   : one host thread per device
#   stream-omp-p2p : ditto, on the patched P2P runtime
# ---------------------------------------------------------------------------
BINS_2 := 2-omp 2-omp-stream 2-omp-stream-omp 2-omp-stream-omp-p2p
BINS_4 := 4-omp 4-omp-stream 4-omp-stream-omp 4-omp-stream-omp-p2p

# Single-GCD baseline: the speedup denominator, built by `make gpu-baseline`.
# CPU reference: correctness only, NOT part of production timing. `make serial`.
BASELINE_BINS := 1-omp
REF_BINS      := serial

ALL_BINS := $(BINS_2) $(BINS_4) $(BASELINE_BINS) $(REF_BINS)

.PHONY: all two four baseline gpu-baseline serial-ref clean check-p2p-lib

all: two four

# ---------------------------------------------------------------------------
# 2-GPU family
# ---------------------------------------------------------------------------
two: $(BINS_2)

2-omp: 2-openmp.c
	$(CC) $(OMP_FLAGS) -o $@ $< $(LIBS)

2-omp-stream: 2-openmp-stream.c
	$(CC) $(OMP_FLAGS) -o $@ $< $(LIBS)

2-omp-stream-omp: 2-openmp-stream-omp.c
	$(CC) $(OMP_FLAGS) -o $@ $< $(LIBS)

2-omp-stream-omp-p2p: 2-openmp-stream-omp-p2p.c
	$(CC) $(OMP_FLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# 4-GPU family
# ---------------------------------------------------------------------------
four: $(BINS_4)

4-omp: 4-openmp.c
	$(CC) $(OMP_FLAGS) -o $@ $< $(LIBS)

4-omp-stream: 4-openmp-stream.c
	$(CC) $(OMP_FLAGS) -o $@ $< $(LIBS)

4-omp-stream-omp: 4-openmp-stream-omp.c
	$(CC) $(OMP_FLAGS) -o $@ $< $(LIBS)

4-omp-stream-omp-p2p: 4-openmp-stream-omp-p2p.c
	$(CC) $(OMP_FLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# Single-GCD baseline (production: speedup denominator)
# ---------------------------------------------------------------------------
gpu-baseline: $(BASELINE_BINS)

1-omp: 1-openmp.c
	$(CC) $(OMP_FLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# CPU reference -- correctness checking only. Not built by build.sh, not run by
# run.sh. Build it by hand when you want to regenerate u_serial.txt to diff.
# ---------------------------------------------------------------------------
serial-ref: $(REF_BINS)

serial: heat3D_variable_coeff.c
	$(CC) $(CPU_FLAGS) -o $@ $< $(LIBS)

# Convenience: everything, for the correctness workflow.
baseline: gpu-baseline serial-ref

# ---------------------------------------------------------------------------
# Confirm the P2P runtime is present before the run jobs need it.
# ---------------------------------------------------------------------------
check-p2p-lib:
	@test -f "$(OMPTARGET_P2P_LIB)/libomp.so" || { echo "ERROR: libomp.so not found in $(OMPTARGET_P2P_LIB)"; exit 1; }
	@test -f "$(OMPTARGET_P2P_LIB)/libomptarget.so.19.0git" || { echo "ERROR: libomptarget runtime not found in $(OMPTARGET_P2P_LIB)"; exit 1; }
	@echo "P2P runtime OK: $(OMPTARGET_P2P_LIB)"

clean:
	rm -f $(ALL_BINS)
