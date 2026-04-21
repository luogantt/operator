#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>

// 这个宏的作用：
// 1. 调用任意 CUDA Runtime API
// 2. 如果返回值不是 cudaSuccess，就把错误打印出来并退出
// 3. 这样可以避免“程序静默失败”，方便调试
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << static_cast<int>(err) \
                  << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// 打印矩阵，方便我们直接观察输入输出
static void print_matrix(const std::vector<float>& mat, int rows, int cols, const char* name) {
    std::cout << name << " =" << std::endl;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2)
                      << mat[r * cols + c] << " ";
        }
        std::cout << std::endl;
    }
}

// 用简单规律初始化矩阵
// 这样既不是全 0，也不是随机到不好观察
static void init_matrix(std::vector<float>& mat, int rows, int cols, float start = 1.0f) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            mat[r * cols + c] = start + (r * cols + c) % 13;
        }
    }
}

// CPU 版本 GEMM，作为“标准答案”
// C[i,j] = sum_k A[i,k] * B[k,j]
static void cpu_gemm(const std::vector<float>& A,
                     const std::vector<float>& B,
                     std::vector<float>& C,
                     int M, int N, int K) {
    for (int i = 0; i < M; ++i) {          // 遍历 A 的行 / C 的行
        for (int j = 0; j < N; ++j) {      // 遍历 B 的列 / C 的列
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {  // K 是乘加累积维
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 检查 GPU 结果和 CPU 结果是否一致
static bool nearly_equal(const std::vector<float>& a,
                         const std::vector<float>& b,
                         float eps = 1e-3f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > eps) {
            std::cerr << "Mismatch at " << i
                      << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}


// ===============================
// 04. Shared Memory Tiling 版 GEMM
// ===============================
//
// 核心思想：
//   不要每次乘法都直接去 HBM(Global Memory) 拿数据。
//   先把一小块 A 和一小块 B 搬到 shared memory，
//   然后在 SM 内部反复复用。
//
// 这就是 tile / block GEMM 的基本思想。
//
// 为什么会快？
//   因为 global memory 很慢，shared memory 快得多。
//   通过“先搬一次，再用很多次”，减少 HBM 访问。
//
constexpr int TILE = 16;

__global__ void gemm_tiled(const float* A,
                           const float* B,
                           float* C,
                           int M, int N, int K) {
    // As 表示 A 的一个 tile
    __shared__ float As[TILE][TILE];

    // Bs 表示 B 的一个 tile
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;  // tile 内局部列号
    int ty = threadIdx.y;  // tile 内局部行号

    // 当前线程最终负责的全局输出坐标
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    // K 维不是一次算完，而是分成多个 TILE 宽度的小段
    // 每一轮 t，都加载一块 A 和一块 B
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        // 当前线程负责把 A 的一个元素搬进 shared memory
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // 当前线程负责把 B 的一个元素搬进 shared memory
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // 等待整个 block 把 tile 都装满
        __syncthreads();

        // 现在 As 和 Bs 都在 shared memory 里
        // 做这一个 tile 内部的乘加
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 这一轮用完了，准备下一轮覆盖 shared memory
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 256, N = 256, K = 256;

    std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.0f), hRef(M * N, 0.0f);
    init_matrix(hA, M, K, 1.0f);
    init_matrix(hB, K, N, 2.0f);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, hC.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    gemm_tiled<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cpu_gemm(hA, hB, hRef, M, N, K);

    std::cout << "Tiled GEMM time: " << ms << " ms\n";
    std::cout << (nearly_equal(hC, hRef) ? "PASS" : "FAIL") << std::endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
