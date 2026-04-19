# GPU/Tensor 化与连接矩阵规模问题

把预处理代码重构为 GPU/PyTorch 风格可以缓解连接生成时间，但不能单独根本解决连接矩阵过大的问题。核心瓶颈不是 CPU 算不动，而是连接数量本身太大。

以 `250 x 400 = 100000` 个网格位置为例，Chen/Gong 风格默认 `Drange = 45`，每个 neuron 的局部邻域面积大约是：

```text
pi * 45^2 ~= 6360
```

即使考虑 E/I mask，四组连接加起来仍可能达到数亿级 synapses。每条 synapse 当前要写：

```text
I: pre index
J: post index
K: weight
D: delay
```

如果使用 `int64 + int64 + float64 + float64`，每条连接约 32 bytes：

```text
3e8 条连接 ~= 9.6 GB
6e8 条连接 ~= 19.2 GB
```

这还不包括 Python 临时数组、HDF5 元数据、chunk buffer 和 simulator 读取时的内存。

## GPU 能解决什么

GPU/PyTorch 可以明显加速：

```text
1. 分 chunk 计算距离矩阵
2. 快速生成局部连接 mask
3. 快速计算 K = W * exp(-DM^2 / sigma)
4. 多 GPU 并行处理不同 pop pair 或不同 post-neuron chunk
```

但最终仍然要把 `I/J/K/D` 写入 HDF5。HDF5 写入通常还是 CPU/磁盘瓶颈。GPU 生成完也要搬回 CPU 才能写文件，除非 simulator 直接读取 GPU tensor 或隐式连接规则。

## 更实际的解决方向

### 1. Chunked Streaming 写 HDF5

不要一次性生成完整 `I/J/K/D`，而是：

```text
post neurons 分块
-> GPU 生成该块连接
-> 立即 append/write 到 HDF5 extendable dataset
-> 释放临时 tensor
```

这能避免构建时内存爆炸，但最终 HDF5 文件仍然巨大。

### 2. 更紧凑的数据类型

如果 simulator 支持，可以改成：

```text
I/J: int64 -> int32
K/D: float64 -> float32
```

每条连接从约 32 bytes 降到 16 bytes，文件和内存约减半。

### 3. 降低连接范围

连接数近似随半径平方增长：

```text
连接数 ∝ Drange^2
```

例如：

```text
Drange 45 -> 邻域约 6360
Drange 20 -> 邻域约 1256
Drange 10 -> 邻域约 314
```

从 45 降到 20，连接数约变成 `(20/45)^2 ~= 0.20`。

### 4. 隐式连接

Chen/Gong 连接是规则距离函数：

```text
K(i,j) = W * exp(-dist(i,j)^2 / sigma), dist <= Drange
```

这种连接可以不显式存成几亿条 synapse，而是在 simulator 侧按规则或 convolution/local stencil 计算。HDF5 只保存：

```text
grid_h, grid_w
Drange
sigma
W
pbc
E/I mask rule
```

这是最适合 Python/PyTorch simulator 的长期方向。

### 5. ROI 或下采样

如果目标是替代 attention 层，不一定要对整幅 `250x400` 建巨大 recurrent 网络。可以先用 STP/LIF 活动或粗检测区域裁出 ROI：

```text
full lif_spk: 250x400
-> active bbox / connected component
-> local patch，例如 64x64 或 80x80
-> SpikeNet simulator
-> 输出局部 detection
-> 映射回全图坐标
```

## 建议路线

短期建议：

```text
矩形网格 + 下采样 / ROI + chunked streaming
```

长期建议：

```text
矩形网格 + simulator 侧隐式连接 / GPU tensor kernel
```

所以，单纯把现有预处理改成 PyTorch 多 GPU不能彻底解决连接矩阵过大。只有同时改变连接生成、存储或 simulator 表达方式，才能让 `250x400` 原始分辨率变得工程上可行。

