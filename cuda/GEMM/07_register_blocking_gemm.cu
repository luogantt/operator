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
// 07. Register Blocking GEMM
// ===============================
//
// 前面 tiled GEMM 里，一个线程通常只算一个 C 元素。
// 这里进一步优化：一个线程一次算多个输出元素。
//
// 为什么这样做？
//   因为寄存器(register)比 shared memory 还快。
//   如果一个线程能把多个累加器 acc[i] 放在寄存器里，
//   就可以减少 shared memory / global memory 的重复访问。
//
// 这里的设计：
//   一个线程负责同一行上的 4 个输出值。
//   所以 acc[4] 这 4 个累加器都放在寄存器中。
//
// 代价：
//   register 用量增加，occupancy 可能下降。
//   这就是“寄存器换吞吐”的典型做法。
//
constexpr int TILE_RB = 16;

__global__ void gemm_register_blocking(const float* A,
                                       const float* B,
                                       float* C,
                                       int M, int N, int K) {
    // A 的 tile 还是 16x16
    __shared__ float As[TILE_RB][TILE_RB];

    // B 的 tile 扩展为 16 x (16*4)
    // 因为每个线程要计算 4 个输出列
    __shared__ float Bs[TILE_RB][TILE_RB * 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 一个线程对应一个 row
    int row = blockIdx.y * TILE_RB + ty;

    // 但它要负责 4 个连续输出列
    int col0 = blockIdx.x * (TILE_RB * 4) + tx * 4;

    // 4 个累加器放进寄存器
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int t = 0; t < (K + TILE_RB - 1) / TILE_RB; ++t) {
        int a_col = t * TILE_RB + tx;

        // 加载 A tile
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // 同一个线程一次加载 B 的 4 个元素
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int b_col = col0 + i;
            int b_row = t * TILE_RB + ty;
            Bs[ty][tx * 4 + i] =
                (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }

        __syncthreads();

        // 用 As 的一个值，去乘 B 的 4 个值
        // 结果累积到 acc[0..3]
        #pragma unroll
        for (int k = 0; k < TILE_RB; ++k) {
            float a = As[ty][k];

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i] += a * Bs[k][tx * 4 + i];
            }
        }

        __syncthreads();
    }

    // 写回这 4 个输出
    if (row < M) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int col = col0 + i;
            if (col < N) {
                C[row * N + col] = acc[i];
            }
        }
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

    dim3 block(TILE_RB, TILE_RB);

    // x 方向每个 block 负责 TILE_RB * 4 列输出
    dim3 grid((N + TILE_RB * 4 - 1) / (TILE_RB * 4),
              (M + TILE_RB - 1) / TILE_RB);

    gemm_register_blocking<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
    cpu_gemm(hA, hB, hRef, M, N, K);

    std::cout << (nearly_equal(hC, hRef) ? "PASS" : "FAIL") << std::endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
