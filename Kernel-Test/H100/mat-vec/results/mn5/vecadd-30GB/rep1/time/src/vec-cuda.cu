// vec-cuda.cu -- vector addition C = A + B (double precision), CUDA
//
// Build (nvcc only):
//   nvcc -O3 -arch=sm_80 --fmad=false -lineinfo -Xptxas -v \
//        -Xcompiler -fopenmp -Xcompiler -ffp-contract=off \
//        -o vecadd_cuda vec-cuda.cu -lgomp
//
// Run:   ./vecadd_cuda <N elements> [reps=100]
//
// Counterpart of vec-omp.cpp: same N, same values (A = 1, B = i), same timing,
// and every element checked against the CPU. Kernels:
//   vadd_1d          one thread per element, 256 threads per block: the usual CUDA way
//                    and the direct counterpart of the OpenMP loop
//   vadd_2d          your layout: 16x16 blocks on a square 2-D grid, flattened to a
//                    1-D index (the shared-memory staging is left out: without reuse
//                    between threads it only adds a round trip and a barrier)
//   vadd_vec2        optimized: 128-bit loads (double2), 4 elements per thread in two
//                    independent loads, so more memory requests are in flight
//   vadd_gridstride  the OpenMP runtime's layout in CUDA: min(3200, N/128) blocks x 128
//                    threads, each thread striding through the vector (reference)
//
// Every timed call is launch + wait (cudaDeviceSynchronize), exactly like a
// blocking OpenMP target region, so the wall time per call compares directly
// with vec-omp.cpp. Kernel-only time comes from the nsys trace.
//
// Vector addition moves 24 bytes per element (read A and B, write C) and does
// one addition, so it is limited by DRAM bandwidth: the figure of merit is GB/s.
// A double addition is correctly rounded on CPU and GPU alike, so the check
// demands bitwise equality for all N elements.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) {          \
    fprintf(stderr, "CUDA error: %s at %s:%d: %s\n", #call, __FILE__, __LINE__,        \
            cudaGetErrorString(e_)); exit(1); } } while (0)

// One thread per element.
__global__ void vadd_1d(const double* __restrict__ A, const double* __restrict__ B,
                        double* __restrict__ C, size_t N)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// Your 2-D layout: square grid of 16x16 blocks, global (x, y) flattened to 1-D.
// Neighbouring threads in x read neighbouring addresses, so loads stay coalesced.
__global__ void vadd_2d(const double* __restrict__ A, const double* __restrict__ B,
                        double* __restrict__ C, size_t N)
{
    const size_t x = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    const size_t i = y * ((size_t)gridDim.x * blockDim.x) + x;
    if (i < N) C[i] = A[i] + B[i];
}

// 128-bit loads: each thread handles two double2 (4 elements), blockDim apart so
// that every load instruction of a warp is fully coalesced. cudaMalloc returns
// 256-byte aligned memory, so the double2 accesses are aligned. An odd last
// element is done by one thread.
__global__ void vadd_vec2(const double* __restrict__ A, const double* __restrict__ B,
                          double* __restrict__ C, size_t N)
{
    const size_t n2 = N / 2;
    const double2* __restrict__ A2 = reinterpret_cast<const double2*>(A);
    const double2* __restrict__ B2 = reinterpret_cast<const double2*>(B);
    double2* __restrict__ C2 = reinterpret_cast<double2*>(C);
    const size_t base = (size_t)blockIdx.x * blockDim.x * 2 + threadIdx.x;
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
        const size_t j = base + (size_t)u * blockDim.x;
        if (j < n2) {
            const double2 a = A2[j], b = B2[j];
            C2[j] = make_double2(a.x + b.x, a.y + b.y);
        }
    }
    if ((N & 1) && blockIdx.x == 0 && threadIdx.x == 0) C[N - 1] = A[N - 1] + B[N - 1];
}

// OpenMP layout: thread k of block t handles t*128 + k, then strides by the
// total thread count -- the static schedule the LLVM runtime uses for
// "teams distribute parallel for".
__global__ void vadd_gridstride(const double* __restrict__ A, const double* __restrict__ B,
                                double* __restrict__ C, size_t N)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride)
        C[i] = A[i] + B[i];
}

// Launch-overhead probe: the counterpart of an empty OpenMP target region.
__global__ void empty_kernel() {}

static double peak_bw_GBs(int dev)
{
    int clk_khz = 0, bus_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz,  cudaDevAttrMemoryClockRate,      dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
    return 2.0 * clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;   // double data rate
}

// One warm-up call, then `reps` calls of launch + wait; mean wall ms per call.
template <class F>
static double time_calls_ms(F launch, int reps)
{
    launch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        launch();
        cudaDeviceSynchronize();
    }
    const auto t1 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaGetLastError());
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

// Every element against the CPU; returns the number of wrong (or NaN) entries.
static size_t check_all(const double* A, const double* B, const double* C, size_t N)
{
    size_t bad = 0;
    #pragma omp parallel for reduction(+:bad) schedule(static)
    for (long long i = 0; i < (long long)N; ++i)
        if (!(C[i] == A[i] + B[i])) ++bad;
    return bad;
}

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <N elements> [reps=100]\n", argv[0]); return 1; }
    const size_t N    = strtoull(argv[1], nullptr, 10);   // size_t: 55 GB is 2.3e9 elements
    const int    reps = argc > 2 ? atoi(argv[2]) : 100;
    if (N == 0 || reps <= 0) { fprintf(stderr, "N and reps must be positive\n"); return 1; }

    const size_t bytes = N * sizeof(double);
    const double moved = 3.0 * (double)bytes;             // bytes per call: A, B read, C written

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU : %s (sm_%d%d, %d SMs), theoretical DRAM bandwidth %.0f GB/s\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount, peak_bw_GBs(dev));
    printf("N = %zu elements, each vector %.3f GB, reps = %d, %.3f GB moved per call\n",
           N, bytes / 1e9, reps, moved / 1e9);

    double* hA = (double*)malloc(bytes);
    double* hB = (double*)malloc(bytes);
    double* hC = (double*)malloc(bytes);
    if (!hA || !hB || !hC) { fprintf(stderr, "host malloc failed\n"); return 1; }
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)N; ++i) { hA[i] = 1.0; hB[i] = (double)i; }

    double *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, e0, e1));
    printf("%-26s %11.3f ms  (A and B, not included in kernel times)\n", "H2D copy", h2d_ms);
    printf("%-26s %11s     %10s   %s\n", "kernel", "time/call", "GB/s", "check vs CPU (all elements)");

    auto run = [&](const char* name, auto launch) {
        CUDA_CHECK(cudaMemset(dC, 0xFF, bytes));        // NaN: unwritten entries fail the check
        const double ms = time_calls_ms(launch, reps);
        CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
        const size_t bad = check_all(hA, hB, hC, N);
        printf("%-26s %11.4f ms  %10.1f   %s (%zu of %zu wrong)\n", name, ms,
               moved / (ms / 1e3) / 1e9, bad ? "FAIL" : "bitwise exact", bad, N);
    };

    run("vadd 1-D (256/block)", [&] {
        const size_t blocks = (N + 255) / 256;
        vadd_1d<<<(unsigned)blocks, 256>>>(dA, dB, dC, N);
    });
    run("vadd 2-D grid (16x16)", [&] {
        size_t g = (size_t)std::ceil(std::sqrt(N / 256.0));   // your formula
        while (g * g * 256 < N) ++g;                          // guard against rounding
        if (g == 0) g = 1;
        vadd_2d<<<dim3((unsigned)g, (unsigned)g), dim3(16, 16)>>>(dA, dB, dC, N);
    });
    run("vadd vectorized (double2)", [&] {
        size_t blocks = (N / 2 + 511) / 512;              // 256 threads x 2 double2 each
        if (blocks == 0) blocks = 1;
        vadd_vec2<<<(unsigned)blocks, 256>>>(dA, dB, dC, N);
    });
    run("vadd grid-stride (OMP)", [&] {
        size_t blocks = (N + 127) / 128;
        if (blocks > 3200) blocks = 3200;
        vadd_gridstride<<<(unsigned)blocks, 128>>>(dA, dB, dC, N);
    });

    {   // launch overhead: empty kernel, launch + wait, mean of L calls
        const int L = 1000;
        empty_kernel<<<1, 1>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < L; ++i) { empty_kernel<<<1, 1>>>(); cudaDeviceSynchronize(); }
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0).count() / L;
        CUDA_CHECK(cudaGetLastError());
        printf("%-26s %11.2f us  (empty kernel, launch + synchronize, mean of %d)\n",
               "launch overhead", us, L);
    }

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA); free(hB); free(hC);
    return 0;
}