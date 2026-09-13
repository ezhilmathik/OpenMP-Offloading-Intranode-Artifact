// mat-cuda.cu -- dense C = A * B  (N x N, row-major, double precision)
//
// Build (nvcc only):
//   nvcc -O3 -arch=sm_80 --fmad=false -lineinfo -Xptxas -v \
//        -Xcompiler -fopenmp -Xcompiler -ffp-contract=off \
//        -o matmul_cuda mat-cuda.cu -lgomp
//   Optional vendor ceiling: add  -DWITH_CUBLAS ... -lcublas
//
// Run:   ./matmul_cuda <N> [reps=5]
//
// Kernels, from simplest to fastest:
//   naive      one thread per C element; the direct counterpart of an
//              "omp target teams distribute parallel for collapse(2)" loop nest
//   tiled32    your original 32x32 shared-memory tiling
//   regtile_*  shared-memory tiles + each thread computes a TM x TN block of C
//              in registers (the standard way to make hand-written GEMM fast)
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#ifdef WITH_CUBLAS
#include <cublas_v2.h>
#endif

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) {          \
    fprintf(stderr, "CUDA error: %s at %s:%d: %s\n", #call, __FILE__, __LINE__,        \
            cudaGetErrorString(e_)); exit(1); } } while (0)

// ---------------------------------------------------------------------------
// 1. Naive: one thread per C(row, col). threadIdx.x runs along a row of C, so
//    reads of B and writes of C are coalesced; A(row, k) is a warp broadcast.
// ---------------------------------------------------------------------------
__global__ void dgemm_naive(const double* __restrict__ A, const double* __restrict__ B,
                            double* __restrict__ C, int N)
{
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= N || col >= N) return;
    double sum = 0.0;
    for (int k = 0; k < N; ++k)
        sum += A[(size_t)row * N + k] * B[(size_t)k * N + col];
    C[(size_t)row * N + col] = sum;
}

// ---------------------------------------------------------------------------
// 2. Your original kernel: 32x32 shared-memory tiles, one C element per thread.
// ---------------------------------------------------------------------------
constexpr int TILE = 32;

__global__ void dgemm_tiled32(const double* __restrict__ A, const double* __restrict__ B,
                              double* __restrict__ C, int N)
{
    __shared__ double sA[TILE][TILE];
    __shared__ double sB[TILE][TILE];
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    double value = 0.0;

    for (int t = 0; t < N; t += TILE) {
        sA[threadIdx.y][threadIdx.x] = (row < N && t + (int)threadIdx.x < N)
                                       ? A[(size_t)row * N + t + threadIdx.x] : 0.0;
        sB[threadIdx.y][threadIdx.x] = (col < N && t + (int)threadIdx.y < N)
                                       ? B[(size_t)(t + threadIdx.y) * N + col] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; ++i)
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[(size_t)row * N + col] = value;
}

// ---------------------------------------------------------------------------
// 3. Register-tiled: a block computes a BM x BN tile of C, stepping through K
//    in slices of BK. Each of the (BM/TM)*(BN/TN) threads keeps a TM x TN block
//    of C in registers, so every value loaded from shared memory is reused
//    TM or TN times instead of once. Thread (tx, ty) owns rows ty + i*RY and
//    columns tx + j*RX (strided), which keeps shared-memory reads free of bank
//    conflicts and global stores coalesced. sA is padded by one column so the
//    two distinct rows read by a warp fall in different banks.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
dgemm_regtile(const double* __restrict__ A, const double* __restrict__ B,
              double* __restrict__ C, int N)
{
    constexpr int RX = BN / TN;            // threads along a row of the tile
    constexpr int RY = BM / TM;            // threads along a column of the tile
    constexpr int NT = RX * RY;            // threads per block
    static_assert((BM * BK) % NT == 0 && (BK * BN) % NT == 0, "tile loads must divide evenly");

    __shared__ double sA[BM][BK + 1];
    __shared__ double sB[BK][BN];

    const int tid  = threadIdx.x;
    const int tx   = tid % RX;
    const int ty   = tid / RX;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    double acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    for (int t = 0; t < N; t += BK) {
        // Load the A slice (BM x BK): consecutive threads read consecutive k.
        #pragma unroll
        for (int p = 0; p < (BM * BK) / NT; ++p) {
            const int e = tid + p * NT;
            const int m = e / BK, k = e % BK;
            const int r = row0 + m, c = t + k;
            sA[m][k] = (r < N && c < N) ? A[(size_t)r * N + c] : 0.0;
        }
        // Load the B slice (BK x BN): consecutive threads read consecutive columns.
        #pragma unroll
        for (int p = 0; p < (BK * BN) / NT; ++p) {
            const int e = tid + p * NT;
            const int k = e / BN, n = e % BN;
            const int r = t + k, c = col0 + n;
            sB[k][n] = (r < N && c < N) ? B[(size_t)r * N + c] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = sA[ty + i * RY][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = sB[k][tx + j * RX];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int r = row0 + ty + i * RY;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int c = col0 + tx + j * RX;
            if (r < N && c < N) C[(size_t)r * N + c] = acc[i][j];
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
static void launch_regtile(const double* A, const double* B, double* C, int N)
{
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    dgemm_regtile<BM, BN, BK, TM, TN><<<grid, (BM / TM) * (BN / TN)>>>(A, B, C, N);
}

// Launch-overhead probe: the CUDA counterpart of an empty OpenMP target region.
__global__ void empty_kernel() {}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

// One warm-up call (lazy module loading, clock ramp-up), then the mean of `reps`.
template <class F>
static float time_ms(F f, int reps, cudaEvent_t a, cudaEvent_t b)
{
    f();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(a));
    for (int r = 0; r < reps; ++r) f();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms / reps;
}

// Compare ~512 entries of C (including all four corners) against a sequential
// CPU dot product in the same k order. Full CPU GEMM would take far too long.
static void check_sample(const double* A, const double* B, const double* C, int N,
                         double* max_rel, int* n_exact, int* n_checked)
{
    const int S = N < 23 ? N * N : 512;
    *max_rel = 0.0; *n_exact = 0; *n_checked = S;
    for (int s = 0; s < S; ++s) {
        int i, j;
        if (N < 23)      { i = s / N; j = s % N; }
        else if (s < 4)  { i = (s & 1) ? N - 1 : 0; j = (s & 2) ? N - 1 : 0; }
        else             { i = (int)((1103515245u * (unsigned)s + 12345u) % (unsigned)N);
                           j = (int)((2654435761u * (unsigned)s + 7u)     % (unsigned)N); }
        double ref = 0.0;
        for (int k = 0; k < N; ++k) ref += A[(size_t)i * N + k] * B[(size_t)k * N + j];
        const double got = C[(size_t)i * N + j];
        if (got == ref) ++*n_exact;
        const double d = std::fabs(got - ref) / std::fmax(std::fabs(ref), 1e-300);
        if (!(d <= *max_rel)) *max_rel = d;          // also catches NaN
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <N> [reps=5]\n", argv[0]); return 1; }
    const int N    = atoi(argv[1]);
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    if (N <= 0 || reps <= 0) { fprintf(stderr, "N and reps must be positive\n"); return 1; }

    const size_t nn    = (size_t)N * N;            // size_t: N*N overflows int above 46340
    const size_t bytes = nn * sizeof(double);
    const double flops = 2.0 * (double)N * N * N;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU : %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("N = %d, each matrix %.3f GB, reps = %d, %.3f GFLOP per multiply\n",
           N, bytes / 1e9, reps, flops / 1e9);

    // Host matrices: malloc + parallel initialisation (fast even for large N).
    // Non-constant values so that indexing mistakes cannot hide.
    double* hA = (double*)malloc(bytes);
    double* hB = (double*)malloc(bytes);
    double* hC = (double*)malloc(bytes);
    if (!hA || !hB || !hC) { fprintf(stderr, "host malloc failed\n"); return 1; }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            hA[(size_t)i * N + j] = ((i * 131 + j * 17) % 100) / 100.0;
            hB[(size_t)i * N + j] = ((i * 7 + j * 29) % 10 + 1) / 10.0;
        }

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
    printf("%-26s %11s     %10s   %s\n", "kernel", "time/call", "GFLOP/s", "check vs CPU");

    auto run = [&](const char* name, auto launch) {
        CUDA_CHECK(cudaMemset(dC, 0xFF, bytes));   // fill with NaN: unwritten entries fail the check
        const float ms = time_ms(launch, reps, e0, e1);
        CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
        double max_rel; int n_exact, n_checked;
        check_sample(hA, hB, hC, N, &max_rel, &n_exact, &n_checked);
        const char* verdict = (n_exact == n_checked) ? "bitwise exact"
                            : (max_rel < 1e-12)      ? "OK (rounding differs)" : "FAIL";
        printf("%-26s %11.3f ms  %10.1f   %s (%d/%d exact, max rel err %.1e)\n",
               name, ms, flops / (ms / 1e3) / 1e9, verdict, n_exact, n_checked, max_rel);
    };

    run("naive (32x8 threads)", [&] {
        dim3 block(32, 8), grid((N + 31) / 32, (N + 7) / 8);
        dgemm_naive<<<grid, block>>>(dA, dB, dC, N);
    });
    run("naive (128x1, OMP-like)", [&] {
        dim3 block(128, 1), grid((N + 127) / 128, N);
        dgemm_naive<<<grid, block>>>(dA, dB, dC, N);
    });
    run("tiled32 (your original)", [&] {
        dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
        dgemm_tiled32<<<grid, block>>>(dA, dB, dC, N);
    });
    run("regtile 64x64x16, 4x4",  [&] { launch_regtile< 64,  64, 16, 4, 4>(dA, dB, dC, N); });
    run("regtile 128x128x8, 8x8", [&] { launch_regtile<128, 128,  8, 8, 8>(dA, dB, dC, N); });

#ifdef WITH_CUBLAS
    // Vendor library as an upper bound. Row-major C = A*B is column-major C^T = B^T*A^T,
    // so pass B first. cuBLAS may use FP64 tensor cores and a different summation
    // order, and --fmad=false does not apply to it, so expect "OK", not "exact".
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return 1; }
    const double one = 1.0, zero = 0.0;
    run("cublasDgemm (reference)", [&] {
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &one, dB, N, dA, N, &zero, dC, N);
    });
    cublasDestroy(handle);
#endif

    // Launch overhead: empty kernel, launch + wait, averaged over L calls. The
    // synchronize matches a blocking OpenMP target region (see mat-omp.cpp).
    {
        const int L = 1000;
        empty_kernel<<<1, 1>>>();
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < L; ++i) {
            empty_kernel<<<1, 1>>>();
            cudaDeviceSynchronize();
        }
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
