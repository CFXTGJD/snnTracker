# Translation Pre/Post-Processing Guide

当前 `translation/` 只保留 GU strict 路线和 btorch adapter。非 strict / approx 输入路径已经移除。

## 当前链路

主路径：

```text
event HDF5
  -> build_gu_2018_strict_btorch_bundle(...)
  -> BtorchInputBundle
  -> external btorch/python simulator
  -> simulator output spikes
  -> postprocess_btorch_simulator_output(...)
```

调试/复现路径：

```text
event HDF5
  -> build_gu_2018_strict_input(...)
  -> *_in.h5
  -> load_btorch_input_bundle(...)
  -> BtorchInputBundle
  -> external btorch/python simulator
  -> simulator output spikes
  -> postprocessing
```

两条路径使用同一个 strict GU 网络生成逻辑：

```python
build_gu_2018_strict_network(...)
```

区别只是是否把中间网络配置写成 SpikeNet-style `*_in.h5`。

## 核心文件

```text
translation/spikenet_preprocessing.py
translation/btorch_interface.py
translation/spikenet_postprocessing.py
translation/detection_readout.py
translation/debug_gu_preprocessing_flow.py
translation/debug_btorch_interface_flow.py
translation/debug_postprocessing_flow.py
translation/debug_h5_utils.py
```

## GU Preprocessing

主要接口：

```python
from translation.spikenet_preprocessing import (
    GU2018Config,
    build_gu_2018_strict_network,
    build_gu_2018_strict_input,
)
```

`GU2018Config.lattice_shape` 可自行设置：

```python
GU2018Config(lattice_shape=(63, 63), n_i=1000)
GU2018Config(lattice_shape=(50, 80), n_i=1000)
GU2018Config(lattice_shape=(25, 40), n_i=250)
```

语义：

```text
pop0 excitatory lattice = H x W
N_e = H * W
```

输入 event HDF5 的 `frames_shape=(T,H,W)` 应与 `lattice_shape=(H,W)` 一致；不一致时需要先 resize/bin/ROI。

当前 strict 生成逻辑：

```text
E/E:
    hybrid_degree_strict
    generate_ij_2d_strict
    common-neighbor iteration
    inverse pool 权重

I/E:
    lattice_to_lattice_strict
    根据每个 E neuron 的 E/E incoming weight 平衡 inhibitory input

E/I:
    lattice_to_lattice_strict
    常数权重 g_ie

I/I:
    lattice_to_lattice_strict
    常数权重 g_ii
```

`GU2018Config` 默认值按 GU Matlab 主逻辑设置，但不是 Matlab bit-level 复刻。随机数、GPU/PyTorch 实现和部分数值近似会导致逐数值不完全一致。

## Direct-To-btorch 接口

这是现在推荐的主接口，不生成 `*_in.h5`：

```python
from translation.spikenet_preprocessing import GU2018Config
from translation.btorch_interface import build_gu_2018_strict_btorch_bundle

cfg = GU2018Config(
    lattice_shape=(50, 80),
    n_i=1000,
    step_tot=200,
    connection_device="cuda",
)

bundle = build_gu_2018_strict_btorch_bundle(
    "events_50x80.h5",
    config=cfg,
    input_pop=0,
    mapping="gu_column_major",
    batch_size=1,
)

simulator_input = bundle.simulator_kwargs()
```

返回 `BtorchInputBundle`：

```text
in_h5: None
n_by_pop: [num_pop]
offsets: [num_pop]
dt: float
step_tot: int
synapses: tuple[BtorchSynapseBlock, ...]
connectivity: scipy.sparse.coo_array, shape=(N_total, N_total)
input_current: [T, B, N_total] float32
input_event_files: tuple[str, ...]
grid_shape: (H, W)
mapping: "gu_column_major"
```

外部 btorch/python simulator 使用的核心字段：

```python
bundle.connectivity
bundle.input_current
bundle.n_by_pop
bundle.offsets
bundle.dt
bundle.step_tot
```

外部 simulator 可以按 btorch 的 SNN 写法消费这些字段：

```text
SparseConn(bundle.connectivity)
  -> AlphaPSC / ExponentialPSC
  -> LIF / ALIF / ELIF
  -> RecurrentNN
```

连接矩阵语义：

```text
row = pre_global_id
col = post_global_id
data = signed_weight
shape = (N_total, N_total)
```

输入电流语义：

```text
input_current = [T, batch, N_total]
```

GU 默认 event 映射：

```text
local_E_id = y + x * height
global_id = offset[0] + local_E_id
```

## From-HDF5 调试/复现接口

如果需要保存中间配置用于检查或复现实验，先生成 `*_in.h5`：

```python
from translation.spikenet_preprocessing import GU2018Config, build_gu_2018_strict_input

cfg = GU2018Config(lattice_shape=(50, 80), n_i=1000, step_tot=200)

in_h5 = build_gu_2018_strict_input(
    "gu_in.h5",
    ext_input_e="events_50x80.h5",
    ext_input_i=None,
    config=cfg,
)
```

再转成 btorch bundle：

```python
from translation.btorch_interface import load_btorch_input_bundle

bundle = load_btorch_input_bundle(
    "gu_in.h5",
    input_pop=0,
    mapping="gu_column_major",
    batch_size=1,
)
```

`*_in.h5` 是 SpikeNet-style 配置文件，保存：

```text
/config/Net/INIT001/N
/config/Net/INIT002/dt
/config/Net/INIT002/step_tot
/config/pops/pop*/file_current_input/fname
/config/syns/syn*/INIT006/I
/config/syns/syn*/INIT006/J
/config/syns/syn*/INIT006/K
/config/syns/syn*/INIT006/D
```

它不是仿真结果，只是“网络配置 + 输入文件引用”。

## btorch 输出到后处理

后处理核心不要求 HDF5，直接接收外部 simulator 输出。simulator 输出可以是：

```text
spikes tensor/ndarray: [T,N] or [T,batch,N]
(spikes, states, ...)
{"spikes": spikes}
{"z": spikes}
```

内部会转成：

```text
activity frames [T,H,W]
```

推荐 tensor 直连：

```python
from translation.btorch_interface import postprocess_btorch_simulator_output

activity, detections = postprocess_btorch_simulator_output(
    simulator_output,
    bundle,
    window=5,
    stride=5,
    threshold=1.0,
    min_area=4,
)
```

如果需要保存 btorch 输出给 `spikenet_postprocessing.py` 读，可以写 HDF5：

```python
from translation.btorch_interface import write_btorch_output_h5_from_bundle

write_btorch_output_h5_from_bundle(
    "btorch_out.h5",
    spikes,
    bundle,
    grid_shape=bundle.grid_shape,
    pop_index=0,
    mapping=bundle.mapping,
)
```

写出的关键路径：

```text
/activity
/pop_result_0/activity
/config/Net/INIT001/N
/config/Net/INIT002/dt
/config/Net/INIT002/step_tot
```

## GPU / PyTorch

当前 PyTorch/GPU 用于 strict GU 网络生成中的：

```text
periodic distance
common-neighbor A.T @ A
lattice-to-lattice pairwise distance
```

配置：

```python
GU2018Config(connection_device="cuda")
```

或命令行：

```bash
--connection-device cuda
```

HDF5 写入仍是 CPU I/O。外部 btorch/python simulator 决定是否把 `connectivity` 和 `input_current` 放到 GPU；`translation` 只保证输出接口和后处理输入接口。

## 调试命令

GU strict `*_in.h5` 调试：

```bash
conda run -n snntracker_py311 python translation/debug_gu_preprocessing_flow.py \
  --make-synthetic \
  --lattice-shape 8 10 \
  --n-i 20 \
  --step-tot 120 \
  --p-scale 0.15 \
  --common-neighbor-iterations 2 \
  --connection-device cuda \
  --output-dir /tmp/snntracker_gu_8x10_debug
```

btorch 接口合同调试：

```bash
conda run -n snntracker_py311 python translation/debug_btorch_interface_flow.py \
  --lattice-shape 8 10 \
  --frames 12 \
  --p-scale 0.15 \
  --common-neighbor-iterations 2 \
  --connection-device cuda \
  --output-dir /tmp/snntracker_btorch_direct_debug
```

CLI 直接生成 bundle 摘要：

```bash
conda run -n snntracker_py311 python translation/btorch_interface.py direct-gu \
  /path/to/events_50x80.h5 \
  --lattice-shape 50 80 \
  --n-i 1000 \
  --step-tot 200 \
  --summary-json /tmp/direct_bundle_summary.json
```

```bash
conda run -n snntracker_py311 python translation/btorch_interface.py from-h5 \
  /path/to/gu_in.h5 \
  --summary-json /tmp/from_h5_bundle_summary.json
```

## 当前状态

已完成：

```text
GU strict-only preprocessing
direct-to-btorch 主接口
from-h5 调试/复现接口
inverse pool 接口复用
btorch 输入输出适配
postprocessing tensor/HDF5 双入口
```

已移除：

```text
Chen/Gong builder
Chen/Gong preprocessing debug 入口
GU approx / non-strict builder
独立 Python simulator adapter 文档
```
