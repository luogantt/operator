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
// 06. Shared Memory Bank Conflict
// ===============================
//
// shared memory 很快，但它不是“无限并发”的。
// 它内部被划分成多个 bank。
//
// 如果一个 warp 中多个线程，同时访问落在同一个 bank 的不同地址，
// 就会发生 bank conflict，原本一次并行访问会被拆成多次顺序访问。
//
// 经典案例：
//   float tile[32][32];
//
// 当你按“列”访问时，可能出现严重 bank conflict。
// 解决手段：padding 一列，写成
//   float tile[32][33];
//
// 这样地址模 32 的分布被打散了。
//
constexpr int REPEAT = 20000;

// 不加 padding 的版本
__global__ void bank_conflict_kernel(float* out) {
    __shared__ float tile[32][32];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 先写入，确保 shared memory 中有数据
    tile[ty][tx] = static_cast<float>(ty * 32 + tx);
    __syncthreads();

    float sum = 0.0f;

    // 这里故意做大量重复访问，把 bank conflict 的时间差放大
    #pragma unroll 4
    for (int r = 0; r < REPEAT; ++r) {
        // 关键点：读 tile[tx][ty]，相当于“按列读”
        // 这在 32x32 布局下很容易触发冲突
        sum += tile[tx][ty];
    }

    if (tx == 0 && ty == 0) out[0] = sum;
}

// 加 padding 的版本
__global__ void no_bank_conflict_kernel(float* out) {
    // 多出的一列就是 padding
    __shared__ float tile[32][33];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    tile[ty][tx] = static_cast<float>(ty * 32 + tx);
    __syncthreads();

    float sum = 0.0f;

    #pragma unroll 4
    for (int r = 0; r < REPEAT; ++r) {
        // 访问模式和上面一样
        // 但由于每行长度不是 32，而是 33，
        // 地址映射到 bank 的方式被打散
        sum += tile[tx][ty];
    }

    if (tx == 0 && ty == 0) out[0] = sum;
}

int main() {
    float* dOut;
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));

    // 一个 32x32 block，方便让一个 warp 群完整打在 shared memory 上
    dim3 block(32, 32);

    cudaEvent_t s1, e1, s2, e2;
    CHECK_CUDA(cudaEventCreate(&s1));
    CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventCreate(&s2));
    CHECK_CUDA(cudaEventCreate(&e2));

    CHECK_CUDA(cudaEventRecord(s1));
    bank_conflict_kernel<<<1, block>>>(dOut);
    CHECK_CUDA(cudaEventRecord(e1));
    CHECK_CUDA(cudaEventSynchronize(e1));

    CHECK_CUDA(cudaEventRecord(s2));
    no_bank_conflict_kernel<<<1, block>>>(dOut);
    CHECK_CUDA(cudaEventRecord(e2));
    CHECK_CUDA(cudaEventSynchronize(e2));

    float ms1 = 0.0f, ms2 = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms1, s1, e1));
    CHECK_CUDA(cudaEventElapsedTime(&ms2, s2, e2));

    std::cout << "Without padding (32x32): " << ms1 << " ms\n";
    std::cout << "With padding    (32x33): " << ms2 << " ms\n";
    std::cout << "Padding usually reduces bank conflicts.\n";

    cudaFree(dOut);
    cudaEventDestroy(s1);
    cudaEventDestroy(e1);
    cudaEventDestroy(s2);
    cudaEventDestroy(e2);
    return 0;
}
