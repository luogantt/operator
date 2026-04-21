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
// 01. GEMM 的数学定义 + 最小 CUDA 版本
// ===============================
//
// 我们先用最简单的方式理解 GEMM：
//   C = A * B
//
// 若 A 形状是 M x K，B 形状是 K x N，
// 那么 C 形状就是 M x N。
//
// 元素级公式：
//   C[row, col] = sum_{k=0}^{K-1} A[row, k] * B[k, col]
//
// 这个 kernel 的设计：
//   一个线程负责一个 C[row, col]
//
__global__ void gemm_1thread_per_element(const float* A,
                                         const float* B,
                                         float* C,
                                         int M, int N, int K) {
    // 线程在 x 方向上负责列号
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 线程在 y 方向上负责行号
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查，避免多开的线程访问越界
    if (row < M && col < N) {
        float sum = 0.0f;

        // 沿着 K 维做内积
        for (int k = 0; k < K; ++k) {
            // A[row, k] 在 row-major 内存中的地址
            float a = A[row * K + k];

            // B[k, col] 在 row-major 内存中的地址
            float b = B[k * N + col];

            sum += a * b;
        }

        // 写回 C[row, col]
        C[row * N + col] = sum;
    }
}

int main() {
    // 这里故意使用一个很小的矩阵，便于人工验证
    const int M = 2, K = 3, N = 2;

    // A: 2x3
    std::vector<float> hA = {
        1, 2, 3,
        4, 5, 6
    };

    // B: 3x2
    std::vector<float> hB = {
        7,  8,
        9, 10,
        11, 12
    };

    // CPU 和 GPU 的输出
    std::vector<float> hC_cpu(M * N, 0.0f), hC_gpu(M * N, 0.0f);

    // 先在 CPU 上算一遍，作为参考答案
    cpu_gemm(hA, hB, hC_cpu, M, N, K);

    // 设备端指针
    float *dA, *dB, *dC;

    // 在 GPU 上申请显存
    CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, hC_gpu.size() * sizeof(float)));

    // Host -> Device 拷贝
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    // 每个 block 16x16 个线程
    dim3 block(16, 16);

    // grid 需要覆盖整个 C 矩阵
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // 启动 kernel
    gemm_1thread_per_element<<<grid, block>>>(dA, dB, dC, M, N, K);

    // 检查 launch 是否报错
    CHECK_CUDA(cudaGetLastError());

    // 等待 kernel 结束
    CHECK_CUDA(cudaDeviceSynchronize());

    // Device -> Host 拷贝
    CHECK_CUDA(cudaMemcpy(hC_gpu.data(), dC, hC_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // 打印矩阵，直观看结果
    print_matrix(hA, M, K, "A");
    print_matrix(hB, K, N, "B");
    print_matrix(hC_cpu, M, N, "C_cpu");
    print_matrix(hC_gpu, M, N, "C_gpu");

    // 检查 GPU 和 CPU 是否一致
    std::cout << (nearly_equal(hC_cpu, hC_gpu) ? "PASS" : "FAIL") << std::endl;

    // 释放显存
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
