#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <vector>
#include <cmath>

// WMMA 示例和前面不同，因为它依赖 Tensor Core 的 mma API
// 所以单独写一个最小文件，避免无关代码干扰理解。

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << static_cast<int>(err) \
                  << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

using namespace nvcuda;

// ===============================
// 11. Tensor Core / WMMA
// ===============================
//
// WMMA = Warp Matrix Multiply Accumulate
//
// 这是 NVIDIA 给 Tensor Core 提供的编程接口之一。
// 本质上：
//   让一个 warp 去驱动硬件里的矩阵乘加单元。
//
// 这里用最小例子：
//   16x16 * 16x16 -> 16x16
//
// A 和 B 用 half
// C 的累加用 float
//
__global__ void wmma_gemm_kernel(const half* A, const half* B, float* C) {
#if __CUDA_ARCH__ >= 700
    // 定义 A fragment
    // matrix_a 表示这是左乘矩阵 A
    // row_major 表示 A 按行主序读
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;

    // 定义 B fragment
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

    // 定义累加 fragment
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // 先把 C fragment 清零
    wmma::fill_fragment(c_frag, 0.0f);

    // 从显存读入 A / B
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);

    // 核心：让 Tensor Core 做矩阵乘加
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 把结果写回 C
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
#endif
}

int main() {
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    // WMMA 需要 Volta(sm_70) 及以上
    if (prop.major < 7) {
        std::cout << "This GPU does not support WMMA/Tensor Core sample (need sm_70+)." << std::endl;
        return 0;
    }

    std::vector<half> hA(16 * 16), hB(16 * 16);
    std::vector<float> hC(16 * 16, 0.0f);

    // 全部初始化成 1
    // 这样 16 维内积的结果应该大约是 16
    for (int i = 0; i < 16 * 16; ++i) {
        hA[i] = __float2half(1.0f);
        hB[i] = __float2half(1.0f);
    }

    half *dA, *dB;
    float *dC;
    CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&dC, hC.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(half), cudaMemcpyHostToDevice));

    // 一个 warp 就够了
    wmma_gemm_kernel<<<1, 32>>>(dA, dB, dC);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "C[0] = " << hC[0] << " (expected about 16)\n";

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
