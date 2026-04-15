#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
do {                                                                        \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                  << " code=" << err                                         \
                  << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
} while (0)

// 打印矩阵
void printMatrix(const std::vector<float>& mat, int rows, int cols, const char* name) {
    std::cout << name << " =" << std::endl;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            std::cout << mat[r * cols + c] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// CUDA Kernel: 矩阵加法
__global__ void matAddKernel(const float* A, const float* B, float* C, int rows, int cols) {
    // block 内线程坐标 + block 坐标 -> 全局矩阵坐标
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];

        // 打印映射关系
        printf("block=(%d,%d), thread=(%d,%d) -> matrix(%d,%d), A=%.1f, B=%.1f, C=%.1f\n",
               blockIdx.x, blockIdx.y,
               threadIdx.x, threadIdx.y,
               row, col,
               A[idx], B[idx], C[idx]);
    }
}

int main() {
    // 矩阵大小
    const int rows = 8;
    const int cols = 8;
    const int size = rows * cols;
    const size_t bytes = size * sizeof(float);

    // Host 端矩阵
    std::vector<float> h_A(size);
    std::vector<float> h_B(size);
    std::vector<float> h_C(size, 0.0f);

    // 初始化矩阵
    // A: 0,1,2,3...
    // B: 全部 100
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int idx = r * cols + c;
            h_A[idx] = static_cast<float>(idx);
            h_B[idx] = 100.0f;
        }
    }

    printMatrix(h_A, rows, cols, "A");
    printMatrix(h_B, rows, cols, "B");

    // Device 指针
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // 拷贝到 GPU
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // 配置 block 和 grid
    dim3 block(4, 4);  // 每个 block 4x4 个线程
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    std::cout << "Launch config:" << std::endl;
    std::cout << "block = (" << block.x << ", " << block.y << ", " << block.z << ")" << std::endl;
    std::cout << "grid  = (" << grid.x  << ", " << grid.y  << ", " << grid.z  << ")" << std::endl;
    std::cout << std::endl;

    // 启动 Kernel
    matAddKernel<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 拷回 Host
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    printMatrix(h_C, rows, cols, "C = A + B");

    // 验证结果
    bool ok = true;
    for (int i = 0; i < size; i++) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            std::cerr << "Mismatch at index " << i
                      << ", got " << h_C[i]
                      << ", expected " << expected << std::endl;
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "Result check passed." << std::endl;
    } else {
        std::cout << "Result check failed." << std::endl;
    }

    // 释放显存
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
