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
// 09. Double Buffering / Ping-Pong Buffer GEMM
// ===============================
//
// 前面 tiled GEMM 的流程大致是：
//   1. 加载 tile t
//   2. 计算 tile t
//   3. 再加载 tile t+1
//
// double buffering 的思想是：
//   在计算当前 tile 的时候，为下一轮 tile 准备缓冲区。
//   于是 shared memory 里有两个 buffer：
//     buffer 0, buffer 1
//
// 当前算一个，另一个准备下一轮。
// 这就像“乒乓缓冲”(ping-pong buffer)。
//
// 这个示例主要是讲结构，方便你理解调度思想。
//
constexpr int TILE_DB = 16;

__global__ void gemm_double_buffered(const float* A,
                                     const float* B,
                                     float* C,
                                     int M, int N, int K) {
    // 两组 shared memory buffer
    __shared__ float As[2][TILE_DB][TILE_DB];
    __shared__ float Bs[2][TILE_DB][TILE_DB];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DB + ty;
    int col = blockIdx.x * TILE_DB + tx;

    float sum = 0.0f;

    int stages = (K + TILE_DB - 1) / TILE_DB;

    // 先把第 0 个 stage 预加载到 buffer 0
    int a_col = tx;
    int b_row = ty;
    As[0][ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[0][ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads();

    for (int t = 0; t < stages; ++t) {
        // 当前 buffer
        int cur = t & 1;

        // 下一轮要写入的 buffer
        int nxt = cur ^ 1;

        // 如果还有下一轮，就提前把下一轮数据放进另一个 buffer
        if (t + 1 < stages) {
            int next_a_col = (t + 1) * TILE_DB + tx;
            int next_b_row = (t + 1) * TILE_DB + ty;

            As[nxt][ty][tx] =
                (row < M && next_a_col < K) ? A[row * K + next_a_col] : 0.0f;

            Bs[nxt][ty][tx] =
                (next_b_row < K && col < N) ? B[next_b_row * N + col] : 0.0f;
        }

        // 用当前 buffer 做乘加
        #pragma unroll
        for (int k = 0; k < TILE_DB; ++k) {
            sum += As[cur][ty][k] * Bs[cur][k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
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

    dim3 block(TILE_DB, TILE_DB);
    dim3 grid((N + TILE_DB - 1) / TILE_DB, (M + TILE_DB - 1) / TILE_DB);

    gemm_double_buffered<<<grid, block>>>(dA, dB, dC, M, N, K);
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
