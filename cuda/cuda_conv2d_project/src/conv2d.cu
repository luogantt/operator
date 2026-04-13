#include "conv2d.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

namespace {

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error at " << file << ":" << line << " for " << expr
            << " -> " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

int output_size(int in, int kernel, int pad, int stride) {
    return (in + 2 * pad - kernel) / stride + 1;
}

}  // namespace

void check_conv2d_shapes(
    const Tensor4D& input,
    const Tensor4D& weight,
    const std::vector<float>& bias,
    const Conv2DParams& params) {
    if (input.n <= 0 || input.c <= 0 || input.h <= 0 || input.w <= 0) {
        throw std::invalid_argument("input shape must be positive");
    }
    if (weight.n <= 0 || weight.c <= 0 || weight.h <= 0 || weight.w <= 0) {
        throw std::invalid_argument("weight shape must be positive");
    }
    if (input.c != weight.c) {
        throw std::invalid_argument("input channels must equal weight channels");
    }
    if (!bias.empty() && static_cast<int>(bias.size()) != weight.n) {
        throw std::invalid_argument("bias size must equal out_channels");
    }
    if (params.stride_h <= 0 || params.stride_w <= 0) {
        throw std::invalid_argument("stride must be positive");
    }
    if (params.pad_h < 0 || params.pad_w < 0) {
        throw std::invalid_argument("padding cannot be negative");
    }
    const int out_h = output_size(input.h, weight.h, params.pad_h, params.stride_h);
    const int out_w = output_size(input.w, weight.w, params.pad_w, params.stride_w);
    if (out_h <= 0 || out_w <= 0) {
        throw std::invalid_argument("invalid output shape, kernel/pad/stride mismatch");
    }
}

Tensor4D conv2d_forward_cpu(
    const Tensor4D& input,
    const Tensor4D& weight,
    const std::vector<float>& bias,
    const Conv2DParams& params) {
    check_conv2d_shapes(input, weight, bias, params);

    const int out_h = output_size(input.h, weight.h, params.pad_h, params.stride_h);
    const int out_w = output_size(input.w, weight.w, params.pad_w, params.stride_w);
    Tensor4D output(input.n, weight.n, out_h, out_w);

    for (int n = 0; n < input.n; ++n) {
        for (int oc = 0; oc < weight.n; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = bias.empty() ? 0.0f : bias[oc];
                    for (int ic = 0; ic < input.c; ++ic) {
                        for (int kh = 0; kh < weight.h; ++kh) {
                            for (int kw = 0; kw < weight.w; ++kw) {
                                const int ih = oh * params.stride_h - params.pad_h + kh;
                                const int iw = ow * params.stride_w - params.pad_w + kw;
                                if (ih >= 0 && ih < input.h && iw >= 0 && iw < input.w) {
                                    sum += input(n, ic, ih, iw) * weight(oc, ic, kh, kw);
                                }
                            }
                        }
                    }
                    output(n, oc, oh, ow) = sum;
                }
            }
        }
    }
    return output;
}

__global__ void conv2d_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int out_h, int out_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K * out_h * out_w;
    if (idx >= total) {
        return;
    }

    int t = idx;
    const int ow = t % out_w; t /= out_w;
    const int oh = t % out_h; t /= out_h;
    const int oc = t % K; t /= K;
    const int n = t;

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < R; ++kh) {
            for (int kw = 0; kw < S; ++kw) {
                const int ih = oh * stride_h - pad_h + kh;
                const int iw = ow * stride_w - pad_w + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    const std::size_t in_idx = ((static_cast<std::size_t>(n) * C + ic) * H + ih) * W + iw;
                    const std::size_t w_idx = ((static_cast<std::size_t>(oc) * C + ic) * R + kh) * S + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    const std::size_t out_idx = ((static_cast<std::size_t>(n) * K + oc) * out_h + oh) * out_w + ow;
    output[out_idx] = sum;
}

Tensor4D conv2d_forward_cuda(
    const Tensor4D& input,
    const Tensor4D& weight,
    const std::vector<float>& bias,
    const Conv2DParams& params) {
    check_conv2d_shapes(input, weight, bias, params);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA device found");
    }

    const int out_h = output_size(input.h, weight.h, params.pad_h, params.stride_h);
    const int out_w = output_size(input.w, weight.w, params.pad_w, params.stride_w);
    Tensor4D output(input.n, weight.n, out_h, out_w);

    float* d_input = nullptr;
    float* d_weight = nullptr;
    float* d_bias = nullptr;
    float* d_output = nullptr;

    const std::size_t input_bytes = input.numel() * sizeof(float);
    const std::size_t weight_bytes = weight.numel() * sizeof(float);
    const std::size_t bias_bytes = bias.size() * sizeof(float);
    const std::size_t output_bytes = output.numel() * sizeof(float);

    try {
        CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
        CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));
        CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
        if (!bias.empty()) {
            CUDA_CHECK(cudaMalloc(&d_bias, bias_bytes));
        }

        CUDA_CHECK(cudaMemcpy(d_input, input.data.data(), input_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weight, weight.data.data(), weight_bytes, cudaMemcpyHostToDevice));
        if (!bias.empty()) {
            CUDA_CHECK(cudaMemcpy(d_bias, bias.data(), bias_bytes, cudaMemcpyHostToDevice));
        }

        const int total = static_cast<int>(output.numel());
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        conv2d_forward_kernel<<<blocks, threads>>>(
            d_input,
            d_weight,
            bias.empty() ? nullptr : d_bias,
            d_output,
            input.n, input.c, input.h, input.w,
            weight.n, weight.h, weight.w,
            out_h, out_w,
            params.stride_h, params.stride_w,
            params.pad_h, params.pad_w);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(output.data.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_bias);
        cudaFree(d_output);
        throw;
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
    return output;
}
