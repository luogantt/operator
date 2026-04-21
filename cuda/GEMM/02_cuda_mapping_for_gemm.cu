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
// 02. GEMM 中 row / col 是如何映射到 CUDA 线程的
// ===============================
//
// 这个文件不做矩阵乘法，只做“映射观察”。
// 目的：让你看清楚
//
//   row = blockIdx.y * blockDim.y + threadIdx.y
//   col = blockIdx.x * blockDim.x + threadIdx.x
//
// 这两个公式到底是什么意思。
//
// 也就是说：
//   CUDA 是把“输出空间 C”切成小块(block)，
//   每个 block 再切成线程(thread)。
//
// 最终每个线程都拿到一个输出坐标 (row, col)。
//
__global__ void mapping_kernel(int* out_row, int* out_col, int M, int N) {
    // 当前线程计算出的列号
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 当前线程计算出的行号
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界保护
    if (row < M && col < N) {
        // 将二维坐标展开成一维下标
        int idx = row * N + col;

        // 记录这个位置对应的 row / col
        out_row[idx] = row;
        out_col[idx] = col;
    }
}

int main() {
    const int M = 4, N = 5;

    // 用来接收每个位置的 row / col
    std::vector<int> hRow(M * N, -1), hCol(M * N, -1);

    int *dRow, *dCol;
    CHECK_CUDA(cudaMalloc(&dRow, M * N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dCol, M * N * sizeof(int)));

    // 这里故意选一个不整齐的 block 大小，便于你观察
    dim3 block(3, 2);

    // 让 grid 覆盖整个 4x5 输出空间
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    mapping_kernel<<<grid, block>>>(dRow, dCol, M, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hRow.data(), dRow, M * N * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hCol.data(), dCol, M * N * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "Each output element C[row, col] is mapped to one thread:\n";
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int idx = r * N + c;
            std::cout << "(" << hRow[idx] << "," << hCol[idx] << ") ";
        }
        std::cout << "\n";
    }

    cudaFree(dRow);
    cudaFree(dCol);
    return 0;
}
