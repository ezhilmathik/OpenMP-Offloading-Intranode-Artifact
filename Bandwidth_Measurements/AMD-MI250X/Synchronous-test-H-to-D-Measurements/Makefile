
HIP_FLAGS := -O3 -fopenmp -x hip --offload-arch=gfx90a -Wno-unused-result

OMP_FLAGS := \
    -fopenmp \
    -fopenmp-targets=amdgcn-amd-amdhsa \
    -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a \
    -O3

HIP_BINS := version-1-hip version-2-hip version-4-hip
OMP_BINS := version-1-off version-2-off version-4-off 
ALL_BINS := $(HIP_BINS) $(OMP_BINS)

.PHONY: all hip omp clean

all: hip omp

hip: $(HIP_BINS)

version-1-hip: hip-1.hip
	amdclang++ $(HIP_FLAGS) -o $@ $<

version-2-hip: hip-2.hip
	amdclang++ $(HIP_FLAGS) -o $@ $<

version-4-hip: hip-4.hip
	amdclang++ $(HIP_FLAGS) -o $@ $<


omp: $(OMP_BINS)

version-1-off: openmp-1.cc
	amdclang++ $(OMP_FLAGS) -o $@ $<

version-2-off: openmp-2.cc
	amdclang++ $(OMP_FLAGS) -o $@ $<

version-4-off: openmp-4.cc
	amdclang++ $(OMP_FLAGS) -o $@ $<

clean:
	rm -f $(ALL_BINS)
