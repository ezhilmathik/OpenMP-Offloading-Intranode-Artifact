#include "hip/hip_runtime.h"
// mv-cuda.cu -- dense matrix-vector product y = A x  (N x N, row-major, double), CUDA
//
// Build (nvcc only):
//   nvcc -O3 -arch=sm_80 --fmad=false -lineinfo -Xptxas -v \
//        -Xcompiler -fopenmp -Xcompiler -ffp-contract=off \
//        -o matvec_cuda mv-cuda.cu -lgomp
//
// Run:   ./matvec_cuda <N> [reps=20]
//
// Counterpart of mv-omp.cpp: same N, same data, same timing, and every entry of y
// checked against the CPU. Kernels:
//   mv_thread_row   your kernel: one thread per row, x staged in shared memory in
//                   chunks of 256 (only change: size_t indices, so N*N cannot overflow).
//                   Same mapping as the OpenMP thread-per-row loop: the fair counterpart.
//   mv_warp_row     optimized: one warp per row, the 32 lanes read consecutive columns
//                   (coalesced), partial sums combined with warp shuffles.
//

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <hip/hip_runtime.h>

#define CUDA_CHECK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) {          \
    fprintf(stderr, "CUDA error: %s at %s:%d: %s\n", #call, __FILE__, __LINE__,        \
            hipGetErrorString(e_)); exit(1); } } while (0)

#define BLOCK_SIZE 256

// Your kernel, with size_t indices (row * cols overflowed int for N > 46340).
__global__ void mv_thread_row(const double *matrix, const double *vector, double *result,
                              size_t rows, size_t cols)
{
  __shared__ double vecShared[BLOCK_SIZE];                  // shared memory for the vector block

  size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  double temp = 0.0;

  for (size_t block = 0; block < (cols + BLOCK_SIZE - 1) / BLOCK_SIZE; ++block) {
    size_t vecIndex = block * BLOCK_SIZE + threadIdx.x;     // load the vector block
    if (vecIndex < cols)
      vecShared[threadIdx.x] = vector[vecIndex];
    else
      vecShared[threadIdx.x] = 0.0;
    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; ++i) {                  // row x vector block
      size_t col = block * BLOCK_SIZE + i;
      if (row < rows && col < cols)
        temp += matrix[row * cols + col] * vecShared[i];
    }
    __syncthreads();
  }
  if (row < rows)
    result[row] = temp;
}

// One wavefront (AMD, 64 lanes) or warp (NVIDIA, 32 lanes) per row: coalesced
// reads of A, two accumulators for more loads in flight, shuffle reduction.
// The grid-stride loop makes any grid size correct.
#ifdef __HIP_PLATFORM_AMD__
#define WF 64                    // CDNA wavefront
#else
#define WF 32                    // NVIDIA warp
#endif

__device__ __forceinline__ double shfl_down_d(double v, int off)
{
#ifdef __HIP_PLATFORM_AMD__
    return __shfl_down(v, off, WF);
#else
    return __shfl_down_sync(0xffffffffu, v, off);
#endif
}

__global__ void mv_warp_row(const double* __restrict__ A, const double* __restrict__ x,
                            double* __restrict__ y, size_t rows, size_t cols)
{
    const int    lane   = threadIdx.x & (WF - 1);
    const size_t warp   = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / WF;
    const size_t nwarps = ((size_t)gridDim.x * blockDim.x) / WF;
    for (size_t row = warp; row < rows; row += nwarps) {    // uniform within a wavefront
        const double* __restrict__ a = A + row * cols;
        double s0 = 0.0, s1 = 0.0;
        size_t j = lane;
        for (; j + WF < cols; j += 2 * WF) {
            s0 += a[j]      * x[j];
            s1 += a[j + WF] * x[j + WF];
        }
        if (j < cols) s0 += a[j] * x[j];
        double sum = s0 + s1;
        for (int off = WF / 2; off > 0; off >>= 1)
            sum += shfl_down_d(sum, off);
        if (lane == 0) y[row] = sum;
    }
}

// Launch-overhead probe: the counterpart of an empty OpenMP target region.
__global__ void empty_kernel() {}

static double peak_bw_GBs(int dev)
{
    int clk_khz = 0, bus_bits = 0;
    CUDA_CHECK(hipDeviceGetAttribute(&clk_khz,  hipDeviceAttributeMemoryClockRate,      dev));
    CUDA_CHECK(hipDeviceGetAttribute(&bus_bits, hipDeviceAttributeMemoryBusWidth, dev));
    return 2.0 * clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;
}

// One warm-up call, then `reps` calls of launch + wait; mean wall ms per call.
template <class F>
static double time_calls_ms(F launch, int reps)
{
    launch();
    CUDA_CHECK(hipGetLastError());
    CUDA_CHECK(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) { launch(); hipDeviceSynchronize(); }
    const auto t1 = std::chrono::steady_clock::now();
    CUDA_CHECK(hipGetLastError());
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <N> [reps=20]\n", argv[0]); return 1; }
    const size_t N    = strtoull(argv[1], nullptr, 10);
    const int    reps = argc > 2 ? atoi(argv[2]) : 20;
    if (N == 0 || reps <= 0) { fprintf(stderr, "N and reps must be positive\n"); return 1; }

    const size_t bytesA = N * N * sizeof(double);
    const size_t bytesV = N * sizeof(double);
    const double moved  = (double)bytesA + 2.0 * (double)bytesV;   // A and x read, y written

    int dev = 0;
    CUDA_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t prop;
    CUDA_CHECK(hipGetDeviceProperties(&prop, dev));
#ifdef __HIP_PLATFORM_AMD__
    printf("GPU : %s (%s, %d CUs), theoretical DRAM bandwidth %.0f GB/s\n",
           prop.name, prop.gcnArchName, prop.multiProcessorCount, peak_bw_GBs(dev));
#else
    printf("GPU : %s (sm_%d%d, %d SMs), theoretical DRAM bandwidth %.0f GB/s\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, peak_bw_GBs(dev));
#endif
    printf("N = %zu, A = %.3f GB, reps = %d, %.3f GB moved per call\n",
           N, bytesA / 1e9, reps, moved / 1e9);

    double* hA   = (double*)malloc(bytesA);
    double* hx   = (double*)malloc(bytesV);
    double* hy   = (double*)malloc(bytesV);
    double* href = (double*)malloc(bytesV);
    if (!hA || !hx || !hy || !href) { fprintf(stderr, "host malloc failed\n"); return 1; }
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; ++i)
        for (size_t j = 0; j < N; ++j)
            hA[(size_t)i * N + j] = (double)((size_t)i * N + j + 1);   // your "count"
    for (size_t j = 0; j < N; ++j) hx[j] = (double)(1 + j % 3);

    #pragma omp parallel for schedule(static)                          // CPU reference (exact)
    for (long long i = 0; i < (long long)N; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < N; ++j) s += hA[(size_t)i * N + j] * hx[j];
        href[i] = s;
    }

    double *dA, *dx, *dy;
    CUDA_CHECK(hipMalloc(&dA, bytesA));
    CUDA_CHECK(hipMalloc(&dx, bytesV));
    CUDA_CHECK(hipMalloc(&dy, bytesV));

    hipEvent_t e0, e1;
    CUDA_CHECK(hipEventCreate(&e0));
    CUDA_CHECK(hipEventCreate(&e1));
    CUDA_CHECK(hipEventRecord(e0));
    CUDA_CHECK(hipMemcpy(dA, hA, bytesA, hipMemcpyHostToDevice));
    CUDA_CHECK(hipMemcpy(dx, hx, bytesV, hipMemcpyHostToDevice));
    CUDA_CHECK(hipEventRecord(e1));
    CUDA_CHECK(hipEventSynchronize(e1));
    float h2d_ms = 0.0f;
    CUDA_CHECK(hipEventElapsedTime(&h2d_ms, e0, e1));
    printf("%-26s %11.3f ms  (A and x, not included in kernel times)\n", "H2D copy", h2d_ms);
    printf("%-26s %11s     %10s   %s\n", "kernel", "time/call", "GB/s", "check vs CPU (all of y)");

    auto run = [&](const char* name, auto launch) {
        CUDA_CHECK(hipMemset(dy, 0xFF, bytesV));        // NaN: unwritten entries fail the check
        const double ms = time_calls_ms(launch, reps);
        CUDA_CHECK(hipMemcpy(hy, dy, bytesV, hipMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < N; ++i) if (!(hy[i] == href[i])) ++bad;
        printf("%-26s %11.4f ms  %10.1f   %s (%zu of %zu wrong)\n", name, ms,
               moved / (ms / 1e3) / 1e9, bad ? "FAIL" : "bitwise exact", bad, N);
    };

    run("mv thread/row (yours)", [&] {
        mv_thread_row<<<(unsigned)((N + BLOCK_SIZE - 1) / BLOCK_SIZE), BLOCK_SIZE>>>(dA, dx, dy, N, N);
    });
    run("mv warp/row", [&] {
        const size_t rows_per_block = 256 / WF;            // 4 rows (AMD) or 8 rows (NVIDIA)
        const size_t blocks = (N + rows_per_block - 1) / rows_per_block;
        mv_warp_row<<<(unsigned)blocks, 256>>>(dA, dx, dy, N, N);
    });

    {   // launch overhead: empty kernel, launch + wait, mean of L calls
        const int L = 1000;
        empty_kernel<<<1, 1>>>();
        CUDA_CHECK(hipDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < L; ++i) { empty_kernel<<<1, 1>>>(); hipDeviceSynchronize(); }
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0).count() / L;
        CUDA_CHECK(hipGetLastError());
        printf("%-26s %11.2f us  (empty kernel, launch + synchronize, mean of %d)\n",
               "launch overhead", us, L);
    }

    CUDA_CHECK(hipEventDestroy(e0));
    CUDA_CHECK(hipEventDestroy(e1));
    CUDA_CHECK(hipFree(dA));
    CUDA_CHECK(hipFree(dx));
    CUDA_CHECK(hipFree(dy));
    free(hA); free(hx); free(hy); free(href);
    return 0;
}
