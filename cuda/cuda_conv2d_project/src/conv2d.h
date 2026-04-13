#pragma once

#include <vector>
#include <string>
#include <stdexcept>
#include <cstddef>

struct Conv2DParams {
    int stride_h{1};
    int stride_w{1};
    int pad_h{0};
    int pad_w{0};
};

struct Tensor4D {
    int n{0};
    int c{0};
    int h{0};
    int w{0};
    std::vector<float> data;

    Tensor4D() = default;
    Tensor4D(int n_, int c_, int h_, int w_)
        : n(n_), c(c_), h(h_), w(w_), data(static_cast<std::size_t>(n_) * c_ * h_ * w_, 0.0f) {}

    std::size_t numel() const {
        return static_cast<std::size_t>(n) * c * h * w;
    }

    float& operator()(int ni, int ci, int hi, int wi) {
        return data[((static_cast<std::size_t>(ni) * c + ci) * h + hi) * w + wi];
    }

    const float& operator()(int ni, int ci, int hi, int wi) const {
        return data[((static_cast<std::size_t>(ni) * c + ci) * h + hi) * w + wi];
    }
};

Tensor4D conv2d_forward_cpu(
    const Tensor4D& input,
    const Tensor4D& weight,
    const std::vector<float>& bias,
    const Conv2DParams& params);

Tensor4D conv2d_forward_cuda(
    const Tensor4D& input,
    const Tensor4D& weight,
    const std::vector<float>& bias,
    const Conv2DParams& params);

void check_conv2d_shapes(
    const Tensor4D& input,
    const Tensor4D& weight,
    const std::vector<float>& bias,
    const Conv2DParams& params);
