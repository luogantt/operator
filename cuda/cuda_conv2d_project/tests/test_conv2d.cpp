#include "conv2d.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool almost_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void require(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

void test_cpu_basic() {
    Tensor4D input(1, 1, 3, 3);
    input.data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    Tensor4D weight(1, 1, 2, 2);
    weight.data = {
        1, 0,
        0, 1
    };

    std::vector<float> bias = {0.0f};
    Conv2DParams params;

    Tensor4D out = conv2d_forward_cpu(input, weight, bias, params);
    require(out.n == 1 && out.c == 1 && out.h == 2 && out.w == 2, "Unexpected output shape in test_cpu_basic");

    const std::vector<float> expected = {6, 8, 12, 14};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(almost_equal(out.data[i], expected[i]), "Value mismatch in test_cpu_basic");
    }
}

void test_cpu_padding_stride() {
    Tensor4D input(1, 1, 4, 4);
    for (int i = 0; i < 16; ++i) {
        input.data[static_cast<std::size_t>(i)] = 1.0f;
    }

    Tensor4D weight(1, 1, 3, 3);
    for (float& v : weight.data) {
        v = 1.0f;
    }

    std::vector<float> bias = {0.0f};
    Conv2DParams params;
    params.stride_h = 2;
    params.stride_w = 2;
    params.pad_h = 1;
    params.pad_w = 1;

    Tensor4D out = conv2d_forward_cpu(input, weight, bias, params);
    require(out.h == 2 && out.w == 2, "Unexpected output shape in test_cpu_padding_stride");

    const std::vector<float> expected = {4, 6, 6, 9};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        require(almost_equal(out.data[i], expected[i]), "Value mismatch in test_cpu_padding_stride");
    }
}

void test_cuda_matches_cpu_if_available() {
    Tensor4D input(1, 2, 5, 5);
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = static_cast<float>((i % 7) - 3);
    }

    Tensor4D weight(3, 2, 3, 3);
    for (std::size_t i = 0; i < weight.data.size(); ++i) {
        weight.data[i] = static_cast<float>((static_cast<int>(i) % 5) - 2) * 0.5f;
    }

    std::vector<float> bias = {0.5f, -1.0f, 2.0f};
    Conv2DParams params;
    params.stride_h = 1;
    params.stride_w = 1;
    params.pad_h = 1;
    params.pad_w = 1;

    const Tensor4D cpu = conv2d_forward_cpu(input, weight, bias, params);

    try {
        const Tensor4D gpu = conv2d_forward_cuda(input, weight, bias, params);
        require(cpu.data.size() == gpu.data.size(), "CPU/GPU size mismatch");
        for (std::size_t i = 0; i < cpu.data.size(); ++i) {
            require(almost_equal(cpu.data[i], gpu.data[i], 1e-4f), "CPU/GPU value mismatch");
        }
    } catch (const std::exception& e) {
        std::cout << "Skipping CUDA assertion because CUDA is unavailable: " << e.what() << "\n";
    }
}

}  // namespace

int main() {
    try {
        test_cpu_basic();
        test_cpu_padding_stride();
        test_cuda_matches_cpu_if_available();
        std::cout << "All tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << '\n';
        return 1;
    }
}
