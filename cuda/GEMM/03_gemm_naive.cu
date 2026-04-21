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
// 03. 最朴素的 GEMM
// ===============================
//
// 这是最经典的“一个线程算一个输出元素”的 CUDA GEMM。
//
// 优点：
//   1. 容易理解
//   2. 数学公式和代码几乎一一对应
//
// 缺点：
//   1. 对 A / B 的全局内存访问非常多
//   2. 数据几乎没有复用
//   3. 性能会很差
//
// 你可以把它理解为：
//   “功能正确，但没有做 memory optimization”
//
__global__ void gemm_naive(const float* A,
                           const float* B,
                           float* C,
                           int M, int N, int K) {
    // 当前线程负责的输出坐标
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;

        // 沿 K 维不断取 A 的一行 和 B 的一列 做乘加
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 128, N = 128, K = 128;

    std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.0f), hRef(M * N, 0.0f);
    init_matrix(hA, M, K, 1.0f);
    init_matrix(hB, K, N, 2.0f);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, hC.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    // 用 event 计时，看看这个 naive 版本要多久
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    gemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU 参考结果
    cpu_gemm(hA, hB, hRef, M, N, K);

    std::cout << "Naive GEMM time: " << ms << " ms\n";
    std::cout << (nearly_equal(hC, hRef) ? "PASS" : "FAIL") << std::endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
