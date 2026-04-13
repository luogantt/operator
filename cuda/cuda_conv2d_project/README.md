# CUDA Conv2D Minimal Project

这是一个最小可扩展的 CUDA `Conv2D` 项目，目标不是极致性能，而是先把一条完整链路打通：

- `Tensor4D` 数据结构
- CPU reference 实现
- CUDA kernel 实现
- CMake 构建
- demo 运行
- test 校验 CPU / CUDA 一致性

## 目录结构

```text
cuda_conv2d_project/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── conv2d.h
│   ├── conv2d.cu
│   └── main.cpp
└── tests/
    └── test_conv2d.cpp
```

## 支持的功能

当前版本支持：

- NCHW layout
- batch 输入
- 多输入通道 / 多输出通道
- stride
- padding
- bias
- forward only

当前不支持：

- dilation
- group convolution
- backward
- shared memory / tiling 优化
- Tensor Core / WMMA 优化

## 构建方法

你需要本机具备：

- NVIDIA GPU
- CUDA Toolkit
- `nvcc`
- CMake >= 3.18

示例：

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

## 运行 demo

```bash
./conv2d_demo
```

它会：

1. 先跑 CPU 参考实现
2. 再尝试跑 CUDA 实现
3. 对比两者输出是否一致

## 运行测试

```bash
ctest --output-on-failure
```

或者直接：

```bash
./test_conv2d
```

## 核函数设计

当前 kernel 采用最直白的 one-thread-per-output-element 映射：

- 一个线程负责一个 `output[n, oc, oh, ow]`
- 在线程内部遍历 `ic, kh, kw`
- 直接从 global memory 读取 input 和 weight

优点：

- 容易理解
- 容易验证正确性
- 便于后续逐步优化

缺点：

- 没有 shared memory 复用
- global memory 访问效率一般
- 不适合大规模性能测试

## 后续升级方向

你后面可以沿着这个项目继续做：

1. 加 `dilation`
2. 加 `group/depthwise convolution`
3. 改成 block-level tiling
4. 用 shared memory 缓存 input tile / weight tile
5. 做 `im2col + GEMM`
6. 接 cuBLAS / CUTLASS
7. 加 benchmark
8. 加 PyTorch extension 封装

## 说明

当前对话环境里没有安装 `nvcc`，也没有可见 GPU，所以我无法在这里实际编译和运行 CUDA 目标。

我已经把项目写成了标准 CUDA CMake 工程，你拿到有 CUDA Toolkit 的机器上可以直接编译。如果你愿意，下一步我可以继续把它升级成：

- `shared memory` 优化版
- `im2col + GEMM` 版
- `PyTorch CUDA extension` 版
- `benchmark` 版
