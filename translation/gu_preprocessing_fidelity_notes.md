# GU Preprocessing Fidelity Notes

当前 `translation/` 已经移除 GU approx / non-strict builder，只保留 strict GU 路线。

## 当前结论

正式网络生成入口是：

```python
build_gu_2018_strict_network(...)
```

它生成内存中的 strict GU 网络结构。两个上层接口复用同一个生成结果：

```text
direct-to-btorch:
    build_gu_2018_strict_btorch_bundle(...)

from-h5 debug/replay:
    build_gu_2018_strict_input(...)
    -> load_btorch_input_bundle(...)
```

因此现在不存在“strict 和 approx 两条网络生成逻辑不一致”的问题。

## Matlab 对齐范围

strict 路线对齐 GU Matlab 主逻辑：

```text
hybrid_degree
generate_IJ_2D
common-neighbor iteration
Lattice2Lattice
E/E inverse pool
I/E 根据 E/E incoming weight 做平衡
E/I 和 I/I 常数权重
```

默认参数也按 GU Matlab 设置：

```text
lattice_shape = 63 x 63
n_i = 1000
P_mat = [[0.16, 0.2],
         [0.2,  0.4]]
g_ee_mu = 4e-3
g_ee_std = 1e-3
g_ie = 5e-3
g_ii = 25e-3
zeta = 27/8
```

但它不是 Matlab bit-level 复刻：

```text
1. NumPy/PyTorch 随机数不会和 Matlab 完全一致
2. 距离矩阵和 common-neighbor 计算可用 PyTorch/GPU 实现
3. g_to_epsp_linear 是工程近似
4. 输出数组使用 Python 0-based index
```

## inverse pool 范围

当前 strict 路线只对 E/E 使用 inverse pool。I/E、E/I、I/I 不扩展 inverse pool。

原因：

```text
E/E:
    原始 GU 设计中使用 inverse-pool / lognormal 风格权重分配

I/E:
    原始 GU 设计中按每个 E target 的 E/E incoming weight 平衡 inhibitory input

E/I 和 I/I:
    原始 GU 设计中为常数权重
```

给所有连接都加 inverse pool 会改变 GU 原始 E/I balance，应作为另一个实验分支，而不是当前 strict 翻译。

## 规模注意

默认 `63x63`：

```text
N_e = 3969
E/E 期望连接数约 2.5M
```

这个规模可以显式生成和调试。

全尺寸视觉输入例如 `250x400`：

```text
N_e = 100000
```

不建议直接按 GU 默认概率显式生成所有连接，应先 resize / ROI / 降采样，或设计隐式/稀疏策略。
