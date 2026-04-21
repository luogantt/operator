# CUDA GEMM 12 个示例教材版 README

这份 README 不是普通说明文件，而是配合 12 个 `.cu` 示例一起使用的“学习讲义”。

这一版在讲义版基础上，进一步补充了：

- 每个 `.cu` 文件的**执行流程图**
- 每个文件的**关键变量表**
- 每个文件的**memory 路径**
- 每个文件的**你应该观察什么**
- 每个文件和整个 CUDA GEMM 主线的关系

你可以把这份 README 当作：

> **源码旁白 + 路线图 + 动态执行说明**

---

# 一、总览：CUDA GEMM 到底在解决什么

GEMM：

```text
C = A × B
```

设：

- `A` 是 `M × K`
- `B` 是 `K × N`
- `C` 是 `M × N`

元素级公式：

```text
C[row, col] = Σ_{k=0}^{K-1} A[row, k] * B[k, col]
```

如果只从数学上看，这只是一个内积。

但在 CUDA 中，它变成了三个层次的问题：

## 1. 并行映射问题
谁来计算 `C[row, col]`？

## 2. 数据搬运问题
A、B 从哪里读？global memory、shared memory、register 各扮演什么角色？

## 3. 吞吐优化问题
如何尽量减少慢内存访问，并让 SM 一直有事做？

所以你可以把整个 CUDA GEMM 学习过程理解成一句话：

> **同一个数学公式，在 GPU 上被不断重新组织，以适应并行结构和内存层次。**

---

# 二、建议阅读路线

建议顺序：

```text
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10 -> 11 -> 12
```

逻辑路线：

```text
数学定义
  -> 线程映射
  -> naive kernel
  -> shared memory tiling
  -> coalesced access
  -> bank conflict
  -> register blocking
  -> occupancy tradeoff
  -> double buffering
  -> warp shuffle
  -> Tensor Core
  -> cuBLAS
```

---

# 三、每个文件的详细说明

---

## 01. `01_gemm_math_and_cpu_cuda.cu`

## 这个文件在整套体系里的位置

这是起点。  
它做的事情最简单，但极其重要：

> 先把 GEMM 从“数学公式”变成“一个线程负责一个输出元素”的 CUDA 写法。

如果这个文件没有吃透，后面的所有优化都会变成“背代码”。

---

### 算法目标

完成最基础的矩阵乘法：

```text
C[row, col] = Σ A[row, k] * B[k, col]
```

并且同时给出：

- CPU 参考结果
- GPU 结果
- 二者是否一致

---

### 执行流程图

```text
Host:
  准备 A, B
    ↓
  CPU 先算一遍 C_cpu
    ↓
  cudaMalloc 申请 dA, dB, dC
    ↓
  cudaMemcpy 把 A, B 送到 GPU
    ↓
  启动 kernel
    ↓
Device kernel:
  每个线程先计算 (row, col)
    ↓
  沿 K 维循环
    ↓
  累加 sum += A[row,k] * B[k,col]
    ↓
  写回 C[row,col]
    ↓
Host:
  cudaMemcpy 把 dC 拉回 hC_gpu
    ↓
  与 CPU 结果比较
```

---

### 关键变量表

| 变量 | 含义 | 你要怎么理解 |
|---|---|---|
| `M` | A 的行数 / C 的行数 | 输出矩阵高度 |
| `K` | A 的列数 / B 的行数 | 内积长度 |
| `N` | B 的列数 / C 的列数 | 输出矩阵宽度 |
| `row` | 当前线程负责的输出行 | 来自 y 方向映射 |
| `col` | 当前线程负责的输出列 | 来自 x 方向映射 |
| `sum` | 当前输出元素的累加器 | 一个线程的局部结果 |

---

### memory 路径

```text
A, B 在 host vector 中
  ↓ cudaMemcpy
A, B 进入 GPU global memory
  ↓ kernel 中每次直接读取
线程寄存器里的 sum 累加
  ↓
写回 global memory 中的 C
  ↓ cudaMemcpy
Host 读回 hC_gpu
```

这里还没有 shared memory。

---

### 你应该重点观察什么

1. `row` 和 `col` 是怎么来的  
2. 为什么 `A[row * K + k]` 是“取 A 的一行”  
3. 为什么 `B[k * N + col]` 是“取 B 的一列”  
4. GPU 结果和 CPU 结果为什么应该一致

---

### 学完这一节，你应该得到的理解

> GEMM 在最原始的 CUDA 形式下，就是“一个线程算一个输出元素”。

---

## 02. `02_cuda_mapping_for_gemm.cu`

## 这个文件在整套体系里的位置

这是“线程几何学”。  
它不做矩阵乘法，而是专门把 CUDA 的二维映射逻辑拆开讲。

---

### 算法目标

回答这个问题：

> `blockIdx / threadIdx / blockDim` 是怎样组合成矩阵坐标 `(row, col)` 的？

---

### 执行流程图

```text
Host:
  申请 out_row, out_col
    ↓
  启动 mapping_kernel
    ↓
Device kernel:
  根据 blockIdx / threadIdx 计算 row, col
    ↓
  把 row, col 存到对应位置
    ↓
Host:
  拷回结果
    ↓
  打印每个输出位置的 (row, col)
```

---

### 关键变量表

| 变量 | 含义 |
|---|---|
| `blockIdx.x` | 当前 block 的列编号 |
| `blockIdx.y` | 当前 block 的行编号 |
| `threadIdx.x` | block 内线程的列偏移 |
| `threadIdx.y` | block 内线程的行偏移 |
| `blockDim.x` | 一个 block 的宽度 |
| `blockDim.y` | 一个 block 的高度 |
| `row` | 全局输出行 |
| `col` | 全局输出列 |

---

### 核心映射公式

```cpp
col = blockIdx.x * blockDim.x + threadIdx.x;
row = blockIdx.y * blockDim.y + threadIdx.y;
```

你应该把它理解成：

- 先找到这个线程属于哪个 block
- 再加上它在 block 内部的偏移
- 于是得到它在全局矩阵中的位置

---

### memory 路径

这个例子几乎不涉及复杂内存层次，重点在“空间映射”。

---

### 你应该重点观察什么

1. 为什么 CUDA 的 `x` 常对应列、`y` 常对应行  
2. 为什么 block 像“矩阵上的小方块”  
3. 为什么 grid 要覆盖整个输出平面  
4. 为什么会有边界判断

---

### 学完这一节，你应该得到的理解

> CUDA 在二维问题上的本质，就是把输出平面切成 block，再切成线程。

---

## 03. `03_gemm_naive.cu`

## 这个文件在整套体系里的位置

这是“功能正确但没有优化”的版本。  
后面所有高级版本，都是对它的改造。

---

### 算法目标

用最朴素的方法在 GPU 上实现 GEMM，并统计时间。

---

### 执行流程图

```text
Host:
  初始化 A, B
    ↓
  拷到 GPU
    ↓
  启动 naive kernel
    ↓
Device kernel:
  一个线程负责一个 C[row,col]
    ↓
  每次都直接从 global memory 取 A[row,k] 和 B[k,col]
    ↓
  做 K 次乘加
    ↓
  写回 C[row,col]
    ↓
Host:
  计时并拉回结果
```

---

### 关键变量表

| 变量 | 含义 |
|---|---|
| `row, col` | 当前线程负责的输出位置 |
| `k` | K 维循环索引 |
| `sum` | 输出元素累加值 |
| `grid` | 覆盖整个输出矩阵 |
| `block(16,16)` | 一个 block 负责 16×16 个输出元素 |

---

### memory 路径

```text
global memory A
global memory B
   ↓ 反复读取
thread register: sum
   ↓
global memory C
```

关键问题：

- `A` 和 `B` 的访问高度重复
- 没有缓存式复用
- 很容易被 global memory 带宽卡住

---

### 你应该重点观察什么

1. 为什么这个 kernel 逻辑和数学公式最像  
2. 为什么它虽然正确，但通常性能不高  
3. K 越大时，它为什么会越来越依赖 global memory 带宽

---

### 学完这一节，你应该得到的理解

> naive GEMM 的核心问题，是同一数据被反复从慢内存读取。

---

## 04. `04_gemm_tiled_shared_memory.cu`

## 这个文件在整套体系里的位置

这是第一层真正重要的优化。  
它让你第一次看到 CUDA GEMM 的“块状复用”思想。

---

### 算法目标

通过 shared memory tile，减少 global memory 重复读取。

---

### 算法直觉

naive GEMM 中：

- 每个线程都自己去读 A 和 B
- 相邻线程明明需要很多相同数据，但没共享

tiled GEMM 中：

- 一个 block 协作加载 A 的一小块
- 一个 block 协作加载 B 的一小块
- 整个 block 共享这些数据做很多次乘法

---

### 执行流程图

```text
for each block:
  确定这个 block 负责 C 的哪一个 tile
    ↓
  for each t in K dimension tiles:
      每个线程加载 A 的一个元素到 As
      每个线程加载 B 的一个元素到 Bs
      ↓
      __syncthreads()
      ↓
      在 shared memory 中做 tile 内乘加
      ↓
      __syncthreads()
    ↓
  把 sum 写回 C
```

---

### 关键变量表

| 变量 | 含义 |
|---|---|
| `TILE` | tile 边长 |
| `As` | A 的 shared memory tile |
| `Bs` | B 的 shared memory tile |
| `t` | 当前处理的是 K 维上的第几个 tile |
| `tx, ty` | 当前线程在 tile 内的位置 |
| `row, col` | 当前线程最终输出坐标 |

---

### memory 路径

```text
global A/B
   ↓（协作加载一次）
shared As / Bs
   ↓（多次复用）
thread register sum
   ↓
global C
```

这就是 tiled GEMM 的本质。

---

### 你应该重点观察什么

1. 为什么 `As` / `Bs` 要定义在 shared memory  
2. 为什么每轮都要两次 `__syncthreads()`  
3. 为什么 `t` 是在 K 维分块，而不是 M/N 维  
4. 为什么同一块 shared data 能被多个线程反复使用

---

### 学完这一节，你应该得到的理解

> GEMM 优化的第一原则是：把会被重复使用的数据先搬到更快的存储层。

---

## 05. `05_coalesced_vs_strided_access.cu`

## 这个文件在整套体系里的位置

这是 global memory 访问模式课。  
它解决的是“同样读 global memory，为什么有人快有人慢”。

---

### 算法目标

对比两种 global memory 访问方式：

- warp 连续访问
- warp 跨步访问

---

### 执行流程图

```text
coalesced kernel:
  线程 i 读 in[i]
    ↓
  连续地址访问

strided kernel:
  线程 i 读 in[i * stride]
    ↓
  分散地址访问

比较两者时间
```

---

### 关键变量表

| 变量 | 含义 |
|---|---|
| `idx` | 线程全局索引 |
| `stride` | 线程访问的步长 |
| `pos` | strided 访问时的真实地址 |

---

### memory 路径

都是：

```text
global in -> thread register -> global out
```

差别不在“读多少数据”，而在“地址模式”。

---

### 你应该重点观察什么

1. 为什么 coalesced 访问更利于合并 transaction  
2. 为什么 stride=32 这样的大步长容易拖慢性能  
3. 为什么 warp 内访问模式在 CUDA 中如此关键

---

### 学完这一节，你应该得到的理解

> global memory 的性能不只看数据量，更看一个 warp 的地址分布是否连续。

---

## 06. `06_shared_memory_bank_conflict.cu`

## 这个文件在整套体系里的位置

这是 shared memory 访问模式课。  
前一课讲的是 global memory， 这一课讲的是 shared memory。

---

### 算法目标

演示：

- `32x32` 布局为什么容易 bank conflict
- `32x33` padding 为什么能缓解它

---

### 执行流程图

```text
kernel A:
  定义 tile[32][32]
    ↓
  按列访问很多次
    ↓
  测时间

kernel B:
  定义 tile[32][33]
    ↓
  按列访问很多次
    ↓
  测时间

比较两者
```

---

### 关键变量表

| 变量 | 含义 |
|---|---|
| `tile[32][32]` | 无 padding 布局 |
| `tile[32][33]` | 加一列 padding 的布局 |
| `tx, ty` | 当前线程二维索引 |
| `REPEAT` | 重复次数，用于放大时间差 |

---

### memory 路径

```text
thread 把值写进 shared memory tile
   ↓
thread 反复从 shared memory 读
```

关键矛盾在 shared memory 内部的 bank 分布，不在 global memory。

---

### 你应该重点观察什么

1. 两个 kernel 的访问模式几乎一样  
2. 唯一实质区别是数组宽度  
3. 为什么“多一列”会改变 bank 映射关系

---

### 学完这一节，你应该得到的理解

> shared memory 也需要“布局设计”，否则快存储也会被冲突拖慢。

---

## 07. `07_register_blocking_gemm.cu`

## 这个文件在整套体系里的位置

这是从 shared memory 优化进一步走向 register 优化。

---

### 算法目标

让一个线程一次计算多个输出元素，并把多个累加器放到寄存器中。

---

### 执行流程图

```text
每个线程负责 4 个输出列
  ↓
加载 A 的一个 tile 到 shared memory
加载 B 的更宽 tile 到 shared memory
  ↓
从 As 中取一个 a
  ↓
同时更新 acc[0], acc[1], acc[2], acc[3]
  ↓
多轮 tile 后写回 4 个输出
```

---

### 关键变量表

| 变量 | 含义 |
|---|---|
| `acc[4]` | 4 个寄存器累加器 |
| `col0` | 当前线程负责的第一个输出列 |
| `Bs[TILE][TILE*4]` | 为了让每线程一次处理 4 列而扩展的 B tile |

---

### memory 路径

```text
global A/B
   ↓
shared As/Bs
   ↓
register acc[4]
   ↓
global C
```

这是第一次明确引入“多个寄存器累加器”这个概念。

---

### 你应该重点观察什么

1. 为什么一个线程算多个输出更划算  
2. 为什么 acc 数组能提高计算密度  
3. 为什么 B tile 要变宽  
4. 为什么这会增加 register 使用量

---

### 学完这一节，你应该得到的理解

> 高性能 GEMM 不只是让更多线程参与，还要让单线程内部的计算更“厚”。

---

## 08. `08_occupancy_and_registers.cu`

## 这个文件在整套体系里的位置

这是专门讨论优化代价的文件。  
07 让你看到寄存器的好处，08 让你看到寄存器的代价。

---

### 算法目标

用 occupancy API 观察：

- 低寄存器 kernel
- 高寄存器 kernel

的驻留能力差异。

---

### 执行流程图

```text
读取 GPU 属性
  ↓
用 occupancy API 估算 low_register_kernel 的最大驻留 block 数
  ↓
估算 high_register_kernel 的最大驻留 block 数
  ↓
换算 occupancy
  ↓
打印结果
```

---

### 关键变量表

| 变量 | 含义 |
|---|---|
| `numBlocksLow` | 一个 SM 最多能挂多少个 low block |
| `numBlocksHigh` | 一个 SM 最多能挂多少个 high block |
| `maxThreadsPerSM` | SM 的最大线程容量 |
| `occLow/occHigh` | 估算 occupancy |

---

### memory 路径

这个文件的重点不在 memory hierarchy，而在资源约束。

---

### 你应该重点观察什么

1. 高寄存器 kernel 的 occupancy 是否更低  
2. 为什么 occupancy 下降是“资源结果”，不是“性能判决”  
3. 为什么高性能 kernel 常常不追求满 occupancy

---

### 学完这一节，你应该得到的理解

> register 是宝贵资源。用多了会减少并发，但有时换来更高吞吐，必须做 tradeoff。

---

## 09. `09_double_buffering_gemm.cu`

## 这个文件在整套体系里的位置

这是从“静态分块”迈向“流水线调度”。

---

### 算法目标

使用双缓冲 shared memory，让 tile 的加载和计算形成交替节拍。

---

### 执行流程图

```text
预加载 stage 0 到 buffer 0
  ↓
for each stage t:
    当前用 cur buffer 计算
    同时把下一轮数据写进 nxt buffer
    ↓
    cur / nxt 交换
  ↓
写回 C
```

---

### 关键变量表

| 变量 | 含义 |
|---|---|
| `As[2]` / `Bs[2]` | 两套缓冲区 |
| `cur` | 当前计算使用的 buffer |
| `nxt` | 下一轮要写入的 buffer |
| `stages` | K 维总共分成几轮 tile |

---

### memory 路径

```text
global A/B
   ↓
shared buffer 0 / buffer 1
   ↓
register sum
   ↓
global C
```

这里的重点不是层级变化，而是“时间重叠”。

---

### 你应该重点观察什么

1. 为什么要有两套 buffer  
2. `cur` / `nxt` 怎样交替  
3. 它相比普通 tiled GEMM，多出来的思想是什么

---

### 学完这一节，你应该得到的理解

> 真正高性能 kernel 不只是优化“空间复用”，还要优化“时间调度”。

---

## 10. `10_warp_shuffle_reduction.cu`

## 这个文件在整套体系里的位置

这是 warp 级原语入门。

---

### 算法目标

演示如何用 `__shfl_down_sync` 在 warp 内做归约，而不经过 shared memory。

---

### 执行流程图

```text
每个线程拿一个值到寄存器
  ↓
offset = 16
offset = 8
offset = 4
offset = 2
offset = 1
  ↓
逐步把后半部分值加到前半部分
  ↓
lane 0 得到 warp 总和
  ↓
lane 0 写结果
```

---

### 关键变量表

| 变量 | 含义 |
|---|---|
| `val` | 当前线程寄存器中的值 |
| `offset` | 本轮 shuffle 的跨度 |
| `threadIdx.x & 31` | 当前线程在 warp 内的 lane 编号 |

---

### memory 路径

```text
global in
   ↓
register val
   ↓ shuffle in warp registers
register val
   ↓
global out
```

这个例子几乎绕过了 shared memory。

---

### 你应该重点观察什么

1. 为什么一个 warp 内不需要 `__syncthreads()`  
2. 为什么 offset 是 16,8,4,2,1  
3. 为什么 lane 0 是最终输出者

---

### 学完这一节，你应该得到的理解

> warp 是比 block 更细的协作层，很多优化直接发生在 warp 内。

---

## 11. `11_tensor_core_wmma.cu`

## 这个文件在整套体系里的位置

这是从“自己写 FMA 循环”进入“调用矩阵硬件单元”的入口。

---

### 算法目标

使用 WMMA API 驱动 Tensor Core 完成一个 16×16×16 的矩阵乘加。

---

### 执行流程图

```text
定义 A fragment
定义 B fragment
定义 C accumulator fragment
  ↓
load_matrix_sync 读入 A/B
  ↓
mma_sync 调用 Tensor Core
  ↓
store_matrix_sync 写回 C
```

---

### 关键变量表

| 变量 | 含义 |
|---|---|
| `a_frag` | A 的矩阵片段 |
| `b_frag` | B 的矩阵片段 |
| `c_frag` | C 的累加片段 |
| `half` | 输入低精度类型 |
| `float` | 累加精度 |

---

### memory 路径

```text
global A/B
   ↓
WMMA fragments（寄存器/硬件友好片段）
   ↓
Tensor Core mma
   ↓
global C
```

---

### 你应该重点观察什么

1. fragment 不是普通二维数组  
2. `mma_sync` 才是核心  
3. 为什么一个 warp 就能驱动一次小矩阵乘加

---

### 学完这一节，你应该得到的理解

> 在现代 GPU 上，真正高吞吐 GEMM 往往依赖专门的矩阵硬件单元，而不只是普通 CUDA Core。

---

## 12. `12_cublas_gemm_compare.cu`

## 这个文件在整套体系里的位置

这是整套学习的“落地终点”。

---

### 算法目标

直接调用 cuBLAS 做 GEMM，并与 CPU 结果比对。

---

### 执行流程图

```text
Host:
  准备 A, B
    ↓
  CPU 先算参考结果
    ↓
  把 A, B 送入 GPU
    ↓
  创建 cuBLAS handle
    ↓
  调用 cublasSgemm
    ↓
  拷回结果
    ↓
  与 CPU 对比
```

---

### 关键变量表

| 变量 | 含义 |
|---|---|
| `handle` | cuBLAS 上下文 |
| `alpha, beta` | GEMM 系数 |
| `dA, dB, dC` | GPU 矩阵 |
| `CUBLAS_OP_N` | 不转置选项 |

---

### memory 路径

```text
host A/B
   ↓
global dA/dB
   ↓
cuBLAS 内部优化 kernel / Tensor Core / 调度逻辑
   ↓
global dC
   ↓
host hC
```

你看不到内部细节，但它背后正是前面那些原理的高度工程化实现。

---

### 你应该重点观察什么

1. 为什么 row-major 和 column-major 会有差异  
2. 为什么传参顺序看起来和直觉不一样  
3. 为什么工业库通常比手写基础 kernel 快很多

---

### 学完这一节，你应该得到的理解

> 学懂手写 GEMM 是为了掌握原理，真正工程里往往会把标准 GEMM 交给高度优化库完成。

---

# 四、整套代码的 memory hierarchy 视角总结

把 12 个文件按内存层次整理，可以看得更清楚：

## 1. 只用 global memory
- 01
- 03
- 05

## 2. 引入 shared memory
- 04
- 06
- 09

## 3. 强调 register
- 07
- 08
- 10

## 4. 上升到专门硬件矩阵单元
- 11

## 5. 上升到工业级库
- 12

所以本质路径是：

```text
global memory
  -> shared memory
  -> register
  -> warp primitive
  -> Tensor Core
  -> library abstraction
```

---

# 五、整套代码的“思维跃迁”总结

你可以把学习过程看成 5 次跃迁：

## 跃迁 1：从数学到线程
01, 02

## 跃迁 2：从正确到优化
03 -> 04

## 跃迁 3：从“会搬数据”到“会设计访存模式”
05, 06

## 跃迁 4：从 block 级优化到线程 / warp / pipeline 级优化
07, 08, 09, 10

## 跃迁 5：从手写 kernel 到硬件矩阵单元和工业库
11, 12

---

# 六、真正跑代码时建议你怎么做实验

## 实验 1
比较 `03` 和 `04`

你看的是：

- naive GEMM
- tiled GEMM

差多少倍，为什么。

## 实验 2
比较 `05`

你看的是：

- coalesced
- strided

差多少，理解 warp 访问模式。

## 实验 3
比较 `06`

你看的是：

- 32×32
- 32×33

确认 padding 的实际效果。

## 实验 4
运行 `08`

你看的是：

- 低寄存器 occupancy
- 高寄存器 occupancy

理解资源约束。

## 实验 5
运行 `11`

你看的是：

- 你的 GPU 是否支持 WMMA
- Tensor Core 基础接口长什么样

## 实验 6
运行 `12`

你看的是：

- cuBLAS 与手写代码的关系
- row-major / column-major 的坑

---

# 七、编译说明

## 普通文件

```bash
nvcc -O2 文件名.cu -o demo
./demo
```

例如：

```bash
nvcc -O2 04_gemm_tiled_shared_memory.cu -o demo04
./demo04
```

## WMMA

需要 `sm_70+`：

```bash
nvcc -O2 -arch=sm_70 11_tensor_core_wmma.cu -o demo11
./demo11
```

## cuBLAS

需要链接库：

```bash
nvcc -O2 12_cublas_gemm_compare.cu -lcublas -o demo12
./demo12
```

---

# 八、你下一步最自然该学什么

如果这套你已经理解得比较顺了，接下来最自然的 4 个方向是：

## 方向 1：多级 tiling
不只是 block tile，还要有 warp tile、thread tile。

## 方向 2：FlashAttention
看它怎样把 QK^T、softmax、PV 融合成“不落 HBM”的流式计算。

## 方向 3：CUTLASS
看工业级模板库如何把 GEMM 分层组织成可复用框架。

## 方向 4：PTX / SASS
看编译器最后把你的 kernel 编译成什么指令，寄存器用了多少。

---

# 九、最后的总总结

这 12 个 `.cu` 文件，其实都在回答同一个问题：

> **为什么 GEMM 在 GPU 上不是“把三重循环搬过去”那么简单，而是一整套关于并行映射、内存层次、资源约束和硬件单元的系统设计。**

如果你把这套资料真正吃透，你对 CUDA GEMM 的理解就会从：

- 会写一个 kernel

变成：

- 能解释一个 kernel 为什么快
- 能判断一个 kernel 卡在哪一层
- 能理解 CUTLASS / FlashAttention / Tensor Core 这类代码到底在干什么
