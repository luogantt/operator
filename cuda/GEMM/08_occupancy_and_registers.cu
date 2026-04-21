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
// 08. Occupancy 和 Registers 的关系
// ===============================
//
// Occupancy 的直觉：
//   一个 SM 上，能同时驻留多少 warps / threads。
//
// 影响 occupancy 的资源主要有：
//   1. registers
//   2. shared memory
//   3. threads per block
//
// 这个例子里我们比较两个 kernel：
//   - low_register_kernel：寄存器压力较小
//   - high_register_kernel：寄存器压力较大
//
// 然后用 cudaOccupancyMaxActiveBlocksPerMultiprocessor()
// 估算它们的 occupancy 差异。
//
__global__ void low_register_kernel(float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 只有一个主要寄存器变量 x
    float x = static_cast<float>(idx);

    // 这里做一点计算，防止编译器把整个 kernel 优化掉
    #pragma unroll 16
    for (int i = 0; i < 64; ++i) {
        x = x * 1.0001f + 0.1f;
    }

    out[idx] = x;
}

__global__ void high_register_kernel(float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 故意声明很多临时变量，增加寄存器压力
    float r0=idx,  r1=idx+1,  r2=idx+2,  r3=idx+3;
    float r4=idx+4, r5=idx+5, r6=idx+6,  r7=idx+7;
    float r8=idx+8, r9=idx+9, r10=idx+10, r11=idx+11;
    float r12=idx+12, r13=idx+13, r14=idx+14, r15=idx+15;

    #pragma unroll 32
    for (int i = 0; i < 128; ++i) {
        r0  += r8  * 0.0001f; r1  += r9  * 0.0001f;
        r2  += r10 * 0.0001f; r3  += r11 * 0.0001f;
        r4  += r12 * 0.0001f; r5  += r13 * 0.0001f;
        r6  += r14 * 0.0001f; r7  += r15 * 0.0001f;

        r8  += r0  * 0.0001f; r9  += r1  * 0.0001f;
        r10 += r2  * 0.0001f; r11 += r3  * 0.0001f;
        r12 += r4  * 0.0001f; r13 += r5  * 0.0001f;
        r14 += r6  * 0.0001f; r15 += r7  * 0.0001f;
    }

    out[idx] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15;
}

int main() {
    int device = 0;
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    std::cout << "GPU: " << prop.name << "\n";

    int blockSize = 256;
    int numBlocksLow = 0, numBlocksHigh = 0;

    // 估算：在一个 SM 上，最多能同时挂多少个这种 block
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksLow, low_register_kernel, blockSize, 0));

    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksHigh, high_register_kernel, blockSize, 0));

    int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;

    // occupancy = 当前可驻留线程数 / SM 理论最大线程数
    float occLow  = (numBlocksLow  * blockSize) / static_cast<float>(maxThreadsPerSM);
    float occHigh = (numBlocksHigh * blockSize) / static_cast<float>(maxThreadsPerSM);

    std::cout << "Estimated occupancy (low-register kernel) : "
              << occLow * 100.0f << "%\n";
    std::cout << "Estimated occupancy (high-register kernel): "
              << occHigh * 100.0f << "%\n";
    std::cout << "This shows how heavier register usage can reduce occupancy.\n";

    // 再实际跑一下，确保这两个 kernel 可以正常执行
    const int N = 1 << 20;
    float* dOut;
    CHECK_CUDA(cudaMalloc(&dOut, N * sizeof(float)));

    low_register_kernel<<<(N + blockSize - 1) / blockSize, blockSize>>>(dOut);
    high_register_kernel<<<(N + blockSize - 1) / blockSize, blockSize>>>(dOut);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaFree(dOut);
    return 0;
}
