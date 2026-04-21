#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>

// ===============================
// 12. cuBLAS GEMM
// ===============================
//
// 前面我们手写了很多 GEMM kernel。
// 但工业界真正常用的是 cuBLAS / CUTLASS 这样的高性能库。
//
// 这个例子演示：
//   如何直接调用 cuBLAS 的 cublasSgemm
//
// 你可以把它理解成：
//   “让 NVIDIA 已经写好的超强 GEMM 替你算”
//
// 注意：
//   cuBLAS 默认按 column-major 理解矩阵。
//   而我们前面的示例都用的是 row-major。
//   所以这里要做一点转置关系处理：
//
//   对 row-major 的 C = A * B
//   可利用
//      (A * B)^T = B^T * A^T
//
//   然后借 cuBLAS 的 column-major 语义间接完成。
//
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << static_cast<int>(err) \
                  << " \"" << cudaGetErrorString(err) << "\"" << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << static_cast<int>(st) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// 还是保留一个 CPU 版本，作为结果对照
static void cpu_gemm_row_major(const std::vector<float>& A,
                               const std::vector<float>& B,
                               std::vector<float>& C,
                               int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    const int M = 64, N = 64, K = 64;

    std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.0f), hRef(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) hA[i] = 1.0f + (i % 7);
    for (int i = 0; i < K * N; ++i) hB[i] = 2.0f + (i % 5);

    cpu_gemm_row_major(hA, hB, hRef, M, N, K);

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, hA.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, hB.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, hC.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    const float alpha = 1.0f, beta = 0.0f;

    // cublasSgemm 参数解释：
    //   C = alpha * op(A) * op(B) + beta * C
    //
    // 但注意这里是 column-major 解释方式。
    // 为了兼容我们 row-major 数据，
    // 这里实际上调用的是：
    //   C^T = B^T * A^T
    //
    // 所以把 dB 放前面、dA 放后面。
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             dB, N,
                             dA, K,
                             &beta,
                             dC, N));

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (size_t i = 0; i < hC.size(); ++i) {
        if (std::fabs(hC[i] - hRef[i]) > 1e-3f) {
            ok = false;
            std::cerr << "Mismatch at " << i
                      << ": " << hC[i] << " vs " << hRef[i] << "\n";
            break;
        }
    }

    std::cout << (ok ? "PASS" : "FAIL") << std::endl;

    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
