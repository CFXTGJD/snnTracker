# Python Simulator Interface Contract

本文档说明如果 SpikeNet C++ simulator 已经被翻译成 Python，推荐暴露什么接口，以及如何和当前 `translation` 里的 preprocessing / postprocessing 接起来。

## 总体链路

推荐保持三段式：

```text
preprocessing
  -> *_in.h5 + event/input HDF5
python simulator
  -> *_out.h5 或 dense activity frames
postprocessing
  -> activity windows + bbox detections
```

也就是说，Python simulator 不需要直接知道 snnTracker 的 dataset loader、attention 代码或 bbox 代码。它只需要：

```text
读取 preprocessing 生成的配置
读取外部输入事件/电流
运行网络动力学
写出 activity/spike 结果
```

## 推荐的 Python Simulator API

最小推荐接口：

```python
from pathlib import Path

def run_spikenet_simulation(
    input_h5: str | Path,
    output_h5: str | Path,
    *,
    input_mode: str = "events",
    grid_shape: tuple[int, int] | None = None,
    pop_index_for_activity: int = 0,
    device: str = "cuda",
    dtype: str = "float32",
) -> Path:
    ...
```

其中：

```text
input_h5:
    preprocessing 生成的 *_in.h5

output_h5:
    simulator 写出的 *_out.h5

input_mode:
    events: 从 event HDF5 读 x/y/t/pol
    current: 从 current HDF5 读 current/neurons/frame_rate 等
    dense: 从 dense [T,H,W] 或 [T,N] 输入读

grid_shape:
    用于把 event x/y 映射到 pop-local neuron id，也用于输出 activity frame

pop_index_for_activity:
    通常为 0，即 excitatory population

device:
    cuda, cuda:0, cpu

dtype:
    float32 默认即可
```

也可以拆成面向对象形式：

```python
class PythonSpikeNetSimulator:
    @classmethod
    def from_h5(cls, input_h5: str | Path, *, device: str = "cuda"):
        ...

    def load_external_inputs(self):
        ...

    def run(self):
        ...

    def write_h5(self, output_h5: str | Path):
        ...
```

但从当前 pipeline 看，一个 `run_spikenet_simulation(...)` 函数最容易接。

## Preprocessing 给 Simulator 的输入

当前 Python preprocessing 主要由：

```text
translation/spikenet_preprocessing.py
```

生成 `*_in.h5`。核心字段是 C++ simulator 风格：

```text
/config/Net/INIT001/N
/config/Net/INIT002/dt
/config/Net/INIT002/step_tot
/config/pops/pop0/...
/config/pops/pop1/...
/config/pops/pop0/file_current_input/fname
/config/pops/pop1/file_current_input/fname
/config/syns/n_syns
/config/syns/syn*/INIT006/type
/config/syns/syn*/INIT006/i_pre
/config/syns/syn*/INIT006/j_post
/config/syns/syn*/INIT006/I
/config/syns/syn*/INIT006/J
/config/syns/syn*/INIT006/K
/config/syns/syn*/INIT006/D
```

当前约定已经改成 0-based：

```text
population: pop0, pop1
synapse type: 0=AMPA, 1=GABA, 2=NMDA
I/J: pop-local 0-based neuron id
x/y: 0-based image/grid coordinates
```

Python simulator 应直接按 0-based 读取，不要再做 Matlab 风格的 `-1`。

## 外部输入接口的选择

### C++ 原版的两个输入路径

C++ 原版里有两个不同输入 reader。

`file_current_input` 调用：

```text
NeuroPop::load_file_current_input(fname)
```

它读取的 external input HDF5 schema 是：

```text
/neurons
/current
/frame_rate
/mean_curr
/start_step
/end_step
```

这更像“每个 neuron 随时间变化的外部电流矩阵”，不是 event stream。

`file_spike_input` 调用：

```text
NeuroPop::load_file_spike_input(fname)
```

它读取：

```text
/x
/y
/t
/max_x
/max_y
```

然后用：

```text
neuron_id = x + max_x * y
```

把事件映射成 spike neuron id。

### 当前 translation 的事件文件

当前 snnTracker 侧导出的 event HDF5 包含：

```text
/x
/y
/t
/pol
attrs:
    frames_shape = (T, H, W)
    dt
    start_time
```

所以如果 Python simulator 要接当前 preprocessing，推荐支持这个 event schema：

```python
def read_event_h5(path):
    x = h5["x"][:]
    y = h5["y"][:]
    t = h5["t"][:]
    pol = h5["pol"][:] if "pol" in h5 else None
    frames_shape = h5.attrs.get("frames_shape")
    dt = h5.attrs.get("dt", 1)
    start_time = h5.attrs.get("start_time", 0)
```

然后根据 `grid_shape=(H,W)` 做映射。

## Event 到 Population 的映射

如果使用 Chen/Gong 当前矩形 grid：

```text
grid_shape = (H, W)
grid[y, x] = 0 for excitatory cells
grid[1::2, 1::2] = 1 for inhibitory cells
```

当前 preprocessing 的 population index map 是 column-major 顺序：

```python
flat = np.flatnonzero(grid.ravel(order="F") == pop_index)
rows, cols = np.unravel_index(flat, grid.shape, order="F")
pop_local_id[rows, cols] = np.arange(flat.size)
```

因此 Python simulator 必须用同一个 mapping，否则输入 spike 会打到错误 neuron。

推荐直接复用当前 postprocessing/preprocessing 的 mapping 逻辑，或者把它抽成共享函数：

```text
translation.spikenet_postprocessing.make_population_index_maps
```

事件映射伪代码：

```python
maps = make_population_index_maps(grid_shape)
pop0_map, pop1_map = maps

event_neuron_ids_pop0 = pop0_map[y, x]
valid = event_neuron_ids_pop0 >= 0
event_neuron_ids_pop0 = event_neuron_ids_pop0[valid]
event_t = t[valid]
```

如果同一个事件文件同时给 pop0 和 pop1，需要分别用 `pop0_map` 和 `pop1_map` 映射。当前 debug/preprocessing 中通常把同一个 event HDF5 引给 pop0/pop1，但语义上 simulator 可以选择：

```text
只把视觉输入注入 pop0
或者 pop0/pop1 都注入
或者 pop0 注入事件，pop1 用背景噪声
```

这需要根据模型实验设计明确。

## Python Simulator 应该读哪些配置

最小必读：

```text
/config/Net/INIT001/N
/config/Net/INIT002/dt
/config/Net/INIT002/step_tot
/config/syns/n_syns
/config/syns/syn*/INIT006/type
/config/syns/syn*/INIT006/i_pre
/config/syns/syn*/INIT006/j_post
/config/syns/syn*/INIT006/I
/config/syns/syn*/INIT006/J
/config/syns/syn*/INIT006/K
/config/syns/syn*/INIT006/D
```

建议也读：

```text
/config/pops/pop*/PARA001/para_str_ascii
/config/pops/pop*/ELIF/...
/config/pops/pop*/INIT011/r_V0
/config/pops/pop*/INIT011/p_fire
/config/pops/pop*/INIT012/mean
/config/pops/pop*/INIT012/std
/config/pops/pop*/file_current_input/fname
```

如果第一版 Python simulator 只是为了打通 object detection pipeline，可以先支持最小 LIF/ELIF 参数集，不支持全部 SpikeNet C++ 功能。关键是输入输出 schema 先稳定。

## Simulator 输出接口

当前 postprocessing 支持两类输出。

### 推荐输出：dense activity frames

最简单、最适合目标检测的是直接写：

```text
/activity
```

shape:

```text
[T, H, W]
```

含义：

```text
activity[t, y, x] = 该时间步或该 bin 下输出活动
```

可以是：

```text
0/1 spike frame
spike count frame
filtered firing rate frame
membrane activity proxy
```

只要 postprocessing 知道它代表什么即可。

当前 reader 已支持这些路径：

```text
/activity
/spikes
/spike_frames
/output/activity
/output/spikes
/pop_result_{pop}/activity
/pop_result_{pop}/spike_frames
```

所以 Python simulator 如果能直接写 `/activity`，接 postprocessing 最省事。

调用方式：

```bash
conda run -n snntracker_py311 python translation/spikenet_postprocessing.py \
  /path/to/python_sim_out.h5 \
  --grid-shape 250 400 \
  --window 5 \
  --stride 5 \
  --threshold 1 \
  --min-area 4 \
  --output-json /tmp/detections.json \
  --output-mot /tmp/detections_mot.txt
```

### 兼容 C++ 输出：compressed spike history

如果想尽量模拟 C++ 原版输出，就写：

```text
/pop_result_0/spike_hist_tot
/pop_result_0/num_spikes_pop
/pop_result_1/spike_hist_tot
/pop_result_1/num_spikes_pop
```

其中：

```text
num_spikes_pop[t] = 第 t 步 spike 数
spike_hist_tot = 所有时间步 spike neuron id 串接的一维数组
```

postprocessing 会按 `num_spikes_pop` 切开 `spike_hist_tot`，再还原成：

```text
dense_spike_hist: [T, N_pop]
```

然后通过 population map 转成：

```text
frames: [T, H, W]
```

这种方式更接近 C++，但 postprocessing 必须知道 `grid_shape`，并且 pop-local neuron id 必须和 preprocessing 的 population map 一致。

### 推荐同时写的 metadata

为了后处理更稳，建议 Python simulator 在 out.h5 里写：

```text
/config_filename/config_filename
/config/Net/INIT001/N
/config/Net/INIT002/dt
/config/Net/INIT002/step_tot
attrs/grid_shape = (H, W)
```

当前 postprocessing 可以从 `--config-h5` 读取 config；如果 out.h5 自己也带一份 config，调试更方便。

## 和当前 Preprocessing/Postprocessing 的接法

### Step 1: preprocessing

生成 `*_in.h5`：

```python
from translation.spikenet_preprocessing import ChenGong2019Config, build_chen_gong_2019_input

cfg = ChenGong2019Config(
    grid_shape=(250, 400),
    step_tot=200,
    connection_backend="torch",
    connection_device="cuda",
)

build_chen_gong_2019_input(
    "/tmp/debug_in.h5",
    "/tmp/events_pop0.h5",
    "/tmp/events_pop1.h5",
    config=cfg,
)
```

对于全尺寸 `250x400`，不建议第一版就生成完整显式连接表。可以先用 `25x40`、`50x80`、ROI 或较小 `drange` 调试。

### Step 2: Python simulator

推荐第一版：

```python
from python_spikenet import run_spikenet_simulation

run_spikenet_simulation(
    input_h5="/tmp/debug_in.h5",
    output_h5="/tmp/debug_out.h5",
    input_mode="events",
    grid_shape=(250, 400),
    pop_index_for_activity=0,
    device="cuda",
)
```

内部应该做：

```text
read config N/dt/step_tot
read synapse blocks I/J/K/D
read pop input file paths
read event HDF5 x/y/t/pol
map event x/y -> pop-local neuron ids
run simulation
write /activity or /pop_result_*/spike_hist_tot + num_spikes_pop
```

### Step 3: postprocessing

如果 simulator 写了 `/activity`：

```bash
conda run -n snntracker_py311 python translation/spikenet_postprocessing.py \
  /tmp/debug_out.h5 \
  --grid-shape 250 400 \
  --window 5 \
  --stride 5 \
  --threshold 1 \
  --min-area 4 \
  --output-json /tmp/detections.json \
  --output-mot /tmp/detections_mot.txt
```

如果 simulator 写的是 C++ 风格 compressed spikes：

```bash
conda run -n snntracker_py311 python translation/spikenet_postprocessing.py \
  /tmp/debug_out.h5 \
  --config-h5 /tmp/debug_in.h5 \
  --grid-shape 250 400 \
  --window 5 \
  --stride 5 \
  --threshold 1 \
  --min-area 4 \
  --output-json /tmp/detections.json \
  --output-mot /tmp/detections_mot.txt
```

## 当前最推荐的接口选择

为了尽快打通 object detection，我建议 Python simulator 第一版采用：

```text
输入:
    *_in.h5
    file_current_input/fname 指向的 event HDF5
    event HDF5 schema: x/y/t/pol + attrs frames_shape/dt/start_time

内部:
    使用 preprocessing/postprocessing 共享的 population map
    x/y -> pop-local neuron id

输出:
    /activity: [T,H,W]
```

这样优点是：

```text
preprocessing 仍然保持 SpikeNet-style config
simulator 不必一开始完全复刻 C++ out.h5
postprocessing 可以直接读 /activity
bbox debug 最直接
```

等模型动力学确认后，再补充 C++ 兼容输出：

```text
/pop_result_*/spike_hist_tot
/pop_result_*/num_spikes_pop
```

## 需要特别避免的坑

1. 不要混用 Matlab 1-based 和 Python 0-based。

当前 translation 已经改成 0-based。Python simulator 读 `I/J/type/pop` 时不要再减 1。

2. 不要让 event x/y 用错宽高。

约定是：

```text
frames: [T,H,W]
x: col, 0..W-1
y: row, 0..H-1
```

3. 不要让输入 map 和输出 map 不一致。

如果输入用 `make_population_index_maps(grid_shape)` 映射，输出也必须用同一套 map 反映射，否则 bbox 会错位。

4. 不要把 C++ 原版 `file_current_input` 当成 event reader。

C++ 原版 `file_current_input` 读的是 `/current` 和 `/neurons`。当前 event HDF5 是 `/x,/y,/t,/pol`。如果 Python simulator 选择复用 `file_current_input/fname` 指向 event 文件，要在 Python reader 里明确这是 event schema，而不是 C++ 原版 current schema。

5. 第一版不要强行全尺寸全连接。

`250x400` 全尺寸显式 `I/J/K/D` 会非常大。建议先用：

```text
25x40
50x80
ROI
较小 drange
或 simulator 侧隐式连接
```

打通接口和 bbox，再扩大规模。
