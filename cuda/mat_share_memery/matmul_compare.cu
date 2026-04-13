#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

constexpr int N = 2048;
constexpr int TILE = 32;

// -----------------------------
// naive kernel: 不用 shared memory
// 每个 thread 计算 C[row, col]
// -----------------------------
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// -----------------------------
// tiled kernel: 使用 shared memory
// 每个 block 处理一个 TILE x TILE 子块
// -----------------------------
__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    // 沿着 K 维分块
    for (int t = 0; t < n; t += TILE) {
        // 从 HBM 加载到 shared memory
        if (row < n && (t + tx) < n) {
            As[ty][tx] = A[row * n + (t + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((t + ty) < n && col < n) {
            Bs[ty][tx] = B[(t + ty) * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // 在 shared memory 上反复复用
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// -----------------------------
// CPU 参考实现，用于验证正确性
// -----------------------------
void matmul_cpu(const float* A, const float* B, float* C, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

void init_matrix(float* M, int n) {
    for (int i = 0; i < n * n; ++i) {
        M[i] = (float)(rand() % 100) / 100.0f;
    }
}

bool check_result(const float* ref, const float* out, int n, float eps = 1e-2f) {
    for (int i = 0; i < n * n; ++i) {
        float diff = fabsf(ref[i] - out[i]);
        if (diff > eps) {
            printf("Mismatch at %d: ref=%f, out=%f, diff=%f\n",
                   i, ref[i], out[i], diff);
            return false;
        }
    }
    return true;
}

float run_kernel_and_time(void (*launch)(const float*, const float*, float*, int),
                          const float* dA, const float* dB, float* dC, int n) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    launch(dA, dB, dC, n);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

// 用普通函数包装 kernel launch，方便复用计时逻辑
void launch_naive(const float* dA, const float* dB, float* dC, int n) {
    dim3 block(TILE, TILE);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);
    matmul_naive<<<grid, block>>>(dA, dB, dC, n);
    CHECK_CUDA(cudaGetLastError());
}

void launch_tiled(const float* dA, const float* dB, float* dC, int n) {
    dim3 block(TILE, TILE);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y);
    matmul_tiled<<<grid, block>>>(dA, dB, dC, n);
    CHECK_CUDA(cudaGetLastError());
}

int main() {
    srand(42);

    const size_t bytes = (size_t)N * N * sizeof(float);

    float* hA = (float*)malloc(bytes);
    float* hB = (float*)malloc(bytes);
    float* hC_naive = (float*)malloc(bytes);
    float* hC_tiled = (float*)malloc(bytes);
    float* hC_ref = (float*)malloc(bytes);

    if (!hA || !hB || !hC_naive || !hC_tiled || !hC_ref) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }

    init_matrix(hA, N);
    init_matrix(hB, N);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // 预热，避免首次启动影响测量
    launch_naive(dA, dB, dC, N);
    launch_tiled(dA, dB, dC, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 计时 naive
    float naive_ms = run_kernel_and_time(launch_naive, dA, dB, dC, N);
    CHECK_CUDA(cudaMemcpy(hC_naive, dC, bytes, cudaMemcpyDeviceToHost));

    // 计时 tiled
    float tiled_ms = run_kernel_and_time(launch_tiled, dA, dB, dC, N);
    CHECK_CUDA(cudaMemcpy(hC_tiled, dC, bytes, cudaMemcpyDeviceToHost));

    // CPU 参考验证
    printf("Running CPU reference, this may take a while...\n");
    matmul_cpu(hA, hB, hC_ref, N);

    bool ok_naive = check_result(hC_ref, hC_naive, N);
    bool ok_tiled = check_result(hC_ref, hC_tiled, N);

    printf("\nMatrix size: %d x %d\n", N, N);
    printf("Naive kernel time: %.3f ms, correctness: %s\n",
           naive_ms, ok_naive ? "PASS" : "FAIL");
    printf("Tiled kernel time: %.3f ms, correctness: %s\n",
           tiled_ms, ok_tiled ? "PASS" : "FAIL");

    if (tiled_ms > 0.0f) {
        printf("Speedup (naive / tiled): %.2fx\n", naive_ms / tiled_ms);
    }

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    free(hA);
    free(hB);
    free(hC_naive);
    free(hC_tiled);
    free(hC_ref);

    return 0;
}
