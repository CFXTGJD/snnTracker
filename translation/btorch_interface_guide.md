# btorch Interface Guide

当前主接口是 strict GU preprocessing 输出 btorch/python simulator 输入合同，不写 `*_in.h5`，也不在 `translation` 里运行 simulator。

## Direct GU 主接口

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
    mapping="gu_column_major",
    batch_size=1,
)

simulator_input = bundle.simulator_kwargs()
```

返回：

```text
bundle.connectivity      scipy COO, shape=(N_total,N_total)
bundle.input_current    float32 [T,batch,N_total]
bundle.n_by_pop
bundle.offsets
bundle.dt
bundle.step_tot
bundle.synapses
bundle.grid_shape
```

外部 python simulator 应消费：

```text
simulator_input["connectivity"]    scipy COO, shape=(N_total,N_total)
simulator_input["input_current"]   float32 [T,batch,N_total]
simulator_input["dt"]
simulator_input["step_tot"]
simulator_input["n_by_pop"]
simulator_input["offsets"]
```

## btorch 连接矩阵语义

btorch 的 `SparseConn` 语义是：

```text
output = input @ conn
```

因此：

```text
conn row = pre_global_id
conn col = post_global_id
conn data = signed_weight
conn shape = (N_total, N_total)
```

SpikeNet/GU 中的局部连接：

```text
I -> population-local pre id
J -> population-local post id
K -> raw positive weight magnitude
D -> delay
```

适配层会：

```text
pre_global  = offset[pop_pre]  + I
post_global = offset[pop_post] + J
```

默认权重符号：

```text
syn_type == 0 -> positive
syn_type == 1 -> negative
```

## 输入 current 语义

event HDF5 schema：

```text
x
y
t
pol
attrs["frames_shape"] = (T,H,W)
attrs["dt"]
attrs["start_time"]
```

适配层输出：

```text
input_current = [T, batch, N_total]
```

GU 默认映射：

```text
local_E_id = y + x * height
global_id = offset[0] + local_E_id
```

默认只把图像事件注入 `pop0`。`pop1` 是 inhibitory quasi-lattice，不是完整图像网格。

## 外部 Simulator 接口

`translation` 不构造 btorch 模型。外部 python simulator 可以按 btorch 结构消费上面的输入：

```text
SparseConn(bundle.connectivity)
  -> AlphaPSC 或 ExponentialPSC
  -> LIF / ALIF / ELIF
  -> RecurrentNN
```

simulator 输入：

```text
bundle.input_current: [T, batch, N_total]
```

simulator 输出应返回：

```text
spikes: [T, batch, N_total]
```

## 输出到后处理

不要求写 HDF5。后处理直接接 python simulator 输出：

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

如需调试落盘：

```python
from translation.btorch_interface import write_btorch_output_h5_from_bundle

write_btorch_output_h5_from_bundle("btorch_out.h5", spikes, bundle)
```

## CLI

direct 主路径：

```bash
conda run -n snntracker_py311 python translation/btorch_interface.py direct-gu \
  /path/to/events_50x80.h5 \
  --lattice-shape 50 80 \
  --n-i 1000 \
  --step-tot 200 \
  --summary-json /tmp/direct_bundle_summary.json
```

接口合同调试：

```bash
conda run -n snntracker_py311 python translation/debug_btorch_interface_flow.py \
  --lattice-shape 8 10 \
  --frames 12 \
  --p-scale 0.15 \
  --connection-device cuda \
  --output-dir /tmp/snntracker_btorch_direct_debug
```
