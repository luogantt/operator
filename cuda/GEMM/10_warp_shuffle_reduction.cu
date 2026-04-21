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
// 10. Warp Shuffle Reduction
// ===============================
//
// __shfl_down_sync 是 warp-level primitive。
// 它允许一个线程直接读取同一个 warp 内另一个线程寄存器里的值。
//
// 这意味着：
//   某些 warp 内通信，不必经过 shared memory。
//   可以减少 shared memory 访问和同步成本。
//
// 这里做一个最简单的例子：
//   每个 warp 对 32 个数做求和。
//   输入全是 1，所以每个 warp 的输出都应是 32。
//
__inline__ __device__ float warp_reduce_sum(float val) {
    // offset = 16, 8, 4, 2, 1
    // 逐步把 warp 中后面的值累加到前面来
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warp_reduce_kernel(const float* in, float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程先读一个值到寄存器
    float val = in[idx];

    // warp 内归约
    val = warp_reduce_sum(val);

    // 只有每个 warp 的 lane 0 负责写结果
    if ((threadIdx.x & 31) == 0) {
        out[(blockIdx.x * blockDim.x + threadIdx.x) / 32] = val;
    }
}

int main() {
    const int N = 128;  // 4 个 warp

    std::vector<float> hIn(N, 1.0f), hOut(N / 32, 0.0f);

    float *dIn, *dOut;
    CHECK_CUDA(cudaMalloc(&dIn, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, (N / 32) * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dIn, hIn.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // 一个 block 里 128 线程 = 4 个 warp
    warp_reduce_kernel<<<1, 128>>>(dIn, dOut);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hOut.data(), dOut, (N / 32) * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Each warp sums 32 ones, so every output should be 32:\n";
    for (float x : hOut) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    cudaFree(dIn);
    cudaFree(dOut);
    return 0;
}
