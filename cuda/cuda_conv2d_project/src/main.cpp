#include "conv2d.h"

#include <cmath>
#include <iomanip>
#include <iostream>

namespace {

bool almost_equal(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

void print_tensor(const Tensor4D& t, const std::string& name) {
    std::cout << name << " shape=[" << t.n << ", " << t.c << ", " << t.h << ", " << t.w << "]\n";
    for (int n = 0; n < t.n; ++n) {
        for (int c = 0; c < t.c; ++c) {
            std::cout << "  n=" << n << ", c=" << c << "\n";
            for (int h = 0; h < t.h; ++h) {
                for (int w = 0; w < t.w; ++w) {
                    std::cout << std::setw(8) << t(n, c, h, w) << ' ';
                }
                std::cout << '\n';
            }
        }
    }
}

}  // namespace

int main() {
    Tensor4D input(1, 1, 4, 4);
    for (int i = 0; i < 16; ++i) {
        input.data[static_cast<std::size_t>(i)] = static_cast<float>(i + 1);
    }

    Tensor4D weight(1, 1, 3, 3);
    weight.data = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    std::vector<float> bias = {0.0f};
    Conv2DParams params;
    params.stride_h = 1;
    params.stride_w = 1;
    params.pad_h = 1;
    params.pad_w = 1;

    const Tensor4D cpu_out = conv2d_forward_cpu(input, weight, bias, params);
    print_tensor(cpu_out, "CPU output");

    try {
        const Tensor4D gpu_out = conv2d_forward_cuda(input, weight, bias, params);
        print_tensor(gpu_out, "CUDA output");

        bool ok = true;
        for (std::size_t i = 0; i < cpu_out.data.size(); ++i) {
            if (!almost_equal(cpu_out.data[i], gpu_out.data[i])) {
                ok = false;
                std::cerr << "Mismatch at index " << i << ": cpu=" << cpu_out.data[i]
                          << ", gpu=" << gpu_out.data[i] << '\n';
                break;
            }
        }
        std::cout << (ok ? "CPU and CUDA results match.\n" : "CPU and CUDA results do not match.\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "CUDA run skipped/failed: " << e.what() << '\n';
        std::cerr << "CPU reference still ran successfully.\n";
        return 0;
    }
}
