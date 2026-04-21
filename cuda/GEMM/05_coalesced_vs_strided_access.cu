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
// 05. Coalesced Access vs Strided Access
// ===============================
//
// 这个例子不是 GEMM 本身，而是专门演示：
// “为什么 CUDA 特别强调合并访存(coalesced memory access)”。
//
// coalesced：
//   一个 warp 中 32 个线程，访问一段连续地址。
//   GPU 可以把它合并成较少的 memory transaction。
//
// strided：
//   一个 warp 中 32 个线程，每个线程隔很远访问一次。
//   这样 transaction 会变多，带宽利用率差很多。
//
__global__ void copy_coalesced(const float* in, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 连续访问：线程 0 访问 in[0], 线程 1 访问 in[1], ...
    if (idx < N) {
        out[idx] = in[idx] * 2.0f;
    }
}

__global__ void copy_strided(const float* in, float* out, int N, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 稀疏访问：线程 idx 实际访问 idx * stride
    int pos = idx * stride;

    if (pos < N) {
        out[pos] = in[pos] * 2.0f;
    }
}

int main() {
    const int N = 1 << 24;
    const int stride = 32;

    std::vector<float> hIn(N, 1.0f), hOut(N, 0.0f);

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dIn, hIn.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t s1, e1, s2, e2;
    CHECK_CUDA(cudaEventCreate(&s1));
    CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventCreate(&s2));
    CHECK_CUDA(cudaEventCreate(&e2));

    int threads = 256;

    // 连续访问时，需要覆盖全部 N 个元素
    int blocks_coalesced = (N + threads - 1) / threads;

    // stride 版本只有 N/stride 个有效访问点
    int useful = (N + stride - 1) / stride;
    int blocks_strided = (useful + threads - 1) / threads;

    CHECK_CUDA(cudaEventRecord(s1));
    copy_coalesced<<<blocks_coalesced, threads>>>(dIn, dOut, N);
    CHECK_CUDA(cudaEventRecord(e1));
    CHECK_CUDA(cudaEventSynchronize(e1));

    CHECK_CUDA(cudaEventRecord(s2));
    copy_strided<<<blocks_strided, threads>>>(dIn, dOut, N, stride);
    CHECK_CUDA(cudaEventRecord(e2));
    CHECK_CUDA(cudaEventSynchronize(e2));

    float ms1 = 0.0f, ms2 = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms1, s1, e1));
    CHECK_CUDA(cudaEventElapsedTime(&ms2, s2, e2));

    std::cout << "Coalesced copy time: " << ms1 << " ms\n";
    std::cout << "Strided copy time   : " << ms2 << " ms\n";
    std::cout << "The strided version is usually slower because a warp touches scattered addresses.\n";

    cudaFree(dIn);
    cudaFree(dOut);
    cudaEventDestroy(s1);
    cudaEventDestroy(e1);
    cudaEventDestroy(s2);
    cudaEventDestroy(e2);
    return 0;
}
