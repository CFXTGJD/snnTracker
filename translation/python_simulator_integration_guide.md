# Python Simulator Integration Guide

本文档给外部 python simulator 使用。假设 simulator 的接口语义和 btorch 一致：

```text
connectivity: scipy sparse matrix, shape=(N_total, N_total)
input_current: [T, batch, N_total]
output spikes: [T, batch, N_total] or [T, N_total]
```

`translation` 只负责两件事：

```text
preprocessing: event HDF5 -> btorch-style simulator input
postprocessing: simulator output spikes -> detections
```

`translation` 不运行 simulator。

## 推荐链路

```text
event HDF5
  -> build_gu_2018_strict_btorch_bundle(...)
  -> bundle.simulator_kwargs()
  -> your_python_simulator(...)
  -> postprocess_btorch_simulator_output(...)
  -> activity windows, detections
```

## Preprocessing 接口

主接口：

```python
from translation.spikenet_preprocessing import GU2018Config
from translation.btorch_interface import build_gu_2018_strict_btorch_bundle

cfg = GU2018Config(
    lattice_shape=(50, 80),   # (height, width)
    n_i=1000,
    step_tot=200,
    connection_device="cuda", # or "cpu"
)

bundle = build_gu_2018_strict_btorch_bundle(
    "events_50x80.h5",
    config=cfg,
    input_pop=0,
    mapping="gu_column_major",
    batch_size=1,
    current_scale=1.0,
    seed=1,
)

sim_input = bundle.simulator_kwargs()
```

`sim_input` 字段：

```text
connectivity     scipy.sparse.coo_array, shape=(N_total, N_total)
input_current    np.float32, shape=[T, batch, N_total]
dt               float
step_tot         int
n_by_pop         np.ndarray, population sizes, e.g. [N_E, N_I]
offsets          np.ndarray, global id offset for each population
grid_shape       (height, width)
mapping          usually "gu_column_major"
```

连接矩阵语义和 btorch `SparseConn` 一致：

```text
output = input @ connectivity
row = pre_global_id
col = post_global_id
data = signed_weight
```

权重符号：

```text
excitatory synapse type 0 -> positive weight
inhibitory synapse type 1 -> negative weight
```

输入电流语义：

```text
input_current[t, b, global_neuron_id]
```

默认只把事件输入注入 `pop0` excitatory lattice：

```text
local_E_id = y + x * height
global_id = offsets[0] + local_E_id
```

## Simulator 侧最小消费方式

你的 simulator 可以直接接收 `sim_input`：

```python
spikes = your_python_simulator(
    connectivity=sim_input["connectivity"],
    input_current=sim_input["input_current"],
    dt=sim_input["dt"],
    step_tot=sim_input["step_tot"],
    n_by_pop=sim_input["n_by_pop"],
    offsets=sim_input["offsets"],
)
```

如果你的 simulator 是 btorch 风格，通常会做类似：

```python
from btorch.models.linear import SparseConn

conn = SparseConn(conn=sim_input["connectivity"])
x = sim_input["input_current"]  # [T, batch, N_total]
```

具体 neuron/synapse/RecurrentNN 的构造属于 simulator 仓库，不在 `translation` 内。

## Simulator 输出要求

postprocessing 接受以下任一形式：

```text
spikes
(spikes, states, ...)
{"spikes": spikes}
{"z": spikes}
{"output": spikes}
{"activity": spikes}
```

其中 `spikes` 必须是：

```text
[T, N_total]
```

或：

```text
[T, batch, N_total]
```

可以是 `torch.Tensor` 或 `np.ndarray`。如果是 torch tensor，postprocessing 会自动 `detach().cpu().numpy()`。

## Postprocessing 接口

主接口：

```python
from translation.btorch_interface import postprocess_btorch_simulator_output

activity_windows, detections = postprocess_btorch_simulator_output(
    simulator_output,
    bundle,
    pop_index=0,
    batch_index=0,
    window=5,
    stride=5,
    threshold=1.0,
    min_area=4,
)
```

返回：

```text
activity_windows: np.ndarray, shape=[num_windows, H, W]
detections: list[BBoxDetection]
```

`BBoxDetection` 字段：

```text
window_index
t_start
t_end
x1, y1, x2, y2
score
area
label
```

坐标约定：

```text
x = width / column
y = height / row
x2, y2 是右开边界
```

## 完整示例

```python
from translation.spikenet_preprocessing import GU2018Config
from translation.btorch_interface import (
    build_gu_2018_strict_btorch_bundle,
    postprocess_btorch_simulator_output,
)
from translation.detection_readout import save_detections_json

cfg = GU2018Config(
    lattice_shape=(50, 80),
    n_i=1000,
    step_tot=200,
    connection_device="cuda",
)

bundle = build_gu_2018_strict_btorch_bundle(
    "events_50x80.h5",
    config=cfg,
    batch_size=1,
    current_scale=1.0,
)

sim_input = bundle.simulator_kwargs()

simulator_output = your_python_simulator(
    connectivity=sim_input["connectivity"],
    input_current=sim_input["input_current"],
    dt=sim_input["dt"],
    step_tot=sim_input["step_tot"],
    n_by_pop=sim_input["n_by_pop"],
    offsets=sim_input["offsets"],
)

activity, detections = postprocess_btorch_simulator_output(
    simulator_output,
    bundle,
    window=5,
    stride=5,
    threshold=1.0,
    min_area=4,
)

save_detections_json(detections, "detections.json")
```

## 可选：HDF5 调试路径

主路径不需要 `*_in.h5`。如果需要保存中间配置用于复查，可以用：

```python
from translation.spikenet_preprocessing import build_gu_2018_strict_input
from translation.btorch_interface import load_btorch_input_bundle

build_gu_2018_strict_input(
    "gu_in.h5",
    ext_input_e="events_50x80.h5",
    ext_input_i=None,
    config=cfg,
)

bundle = load_btorch_input_bundle(
    "gu_in.h5",
    input_pop=0,
    mapping="gu_column_major",
    batch_size=1,
)
```

这条路径只用于调试/复现，不是推荐主链路。

## 快速合同测试

不运行 simulator，只验证 preprocessing 输出合同和 postprocessing 输入合同：

```bash
conda run -n snntracker_py311 python translation/debug_btorch_interface_flow.py \
  --lattice-shape 6 8 \
  --frames 8 \
  --n-i 12 \
  --p-scale 0.08 \
  --common-neighbor-iterations 2 \
  --connection-device cpu \
  --output-dir /tmp/snntracker_btorch_contract_verify
```

如果已有 simulator 输出 `.npy`，可以测试它是否能接上后处理：

```bash
conda run -n snntracker_py311 python translation/debug_btorch_interface_flow.py \
  --lattice-shape 6 8 \
  --frames 8 \
  --n-i 12 \
  --simulator-output-npy /path/to/spikes.npy \
  --output-dir /tmp/snntracker_btorch_postprocess_verify
```
