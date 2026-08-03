# ---------------------------------------------------------------------------
# 3D heat equation, CUDA, NVIDIA H100 (sm_90).
# MareNostrum 5 / acc partition. Compilers come from build.sh's modules
# (EB/apps + NVHPC/25.3-CUDA-12.8.0, which provides both nvcc and nvc).
# ---------------------------------------------------------------------------

NVCC      ?= nvcc
SERIAL_CC ?= nvc

ARCH := sm_90
OPT  := -O3

# ---------------------------------------------------------------------------
# ONE flag set for EVERY variant, so a 2-cuda vs 2-cuda-stream comparison is
# not confounded by a compiler difference. These are the flags from the working
# cuda-batch.sh; do not change them for one binary only.
#
#   --fmad=false      nvcc contracts a*b+c into FMA by default. Equivalent to
#                     clang's -ffp-contract=off and nvc's -Mnofma. WITHOUT IT
#                     the CUDA results do not match the serial reference
#                     bit-for-bit and verify.sh fails at any sane tolerance.
#   --prec-div=true   full-precision division (not the fast approximation)
#   --prec-sqrt=true  full-precision sqrt
#   --ftz=false       keep denormals; do not flush to zero
#   -Xcompiler -fopenmp
#                     needed for omp_get_wtime() in every file, and for the
#                     host threading in the *-stream-omp* variants. It is the
#                     SAME timer function the OpenMP codes use, which is the
#                     point -- see the header of any .cu file.
#
# NOTE on -O2: this is the host-compiler optimisation level. Device code is
# optimised by ptxas independently (default -O3), so the kernels are not
# affected. The OpenMP directory builds at -O3; that difference lives on the
# host side only.
# ---------------------------------------------------------------------------
NVCC_FLAGS := $(OPT) --fmad=false --prec-div=true --prec-sqrt=true --ftz=false \
              -arch=$(ARCH) -Xcompiler -fopenmp
LIBS       := -lm

# Serial reference: -Mnofma == nvcc --fmad=false, so the CPU ground truth and
# the GPU codes use the same (non-fused) contraction. -mp links the OpenMP
# runtime the code uses for omp_get_wtime().
SERIAL_CFLAGS := $(OPT) -Mnofma -mp

# ---------------------------------------------------------------------------
# Targets: the 2-GPU and 4-GPU families.
#   plain          : blocking halo exchange, staged through the host
#   stream         : async halo exchange on dedicated streams
#   stream-omp     : one host thread per device
#   stream-omp-p2p : ditto, cudaMemcpyPeerAsync, no host staging at all
# ---------------------------------------------------------------------------
BINS_2 := 2-cuda 2-cuda-stream 2-cuda-stream-omp 2-cuda-stream-omp-p2p
BINS_4 := 4-cuda 4-cuda-stream 4-cuda-stream-omp 4-cuda-stream-omp-p2p

BASELINE_BINS := 1-cuda
REF_BINS      := serial

GPU_BINS := $(BINS_2) $(BINS_4) $(BASELINE_BINS)
ALL_BINS := $(GPU_BINS) $(REF_BINS)

.PHONY: all two four baseline gpu-baseline serial-ref clean

all: two four

two:  $(BINS_2)
four: $(BINS_4)

# Every GPU binary is built from the .cu file of the same name.
$(GPU_BINS): %: %.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

# ---------------------------------------------------------------------------
# Single-GPU baseline (production: speedup denominator)
# ---------------------------------------------------------------------------
gpu-baseline: $(BASELINE_BINS)

# ---------------------------------------------------------------------------
# CPU reference -- correctness only. Not built by build.sh, not run by run.sh.
# ---------------------------------------------------------------------------
serial-ref: $(REF_BINS)

serial: heat3D_variable_coeff.c
	$(SERIAL_CC) $(SERIAL_CFLAGS) -o $@ $< $(LIBS)

baseline: gpu-baseline serial-ref

clean:
	rm -f $(ALL_BINS)
