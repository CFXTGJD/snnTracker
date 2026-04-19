# Translation Pre/Post-Processing Guide

本文档记录 `translation` 下 preprocessing 和 post-processing 两部分的代码逻辑、调试方法和关键注意事项。

## Pre-Processing

### 目标

Pre-processing 的目标是替代 SpikeNet Matlab 侧生成 `*_in.h5` 的流程，把原本送入 snnTracker attention 层的输入写成 simulator 可以读取的配置和外部输入引用。

当前主要代码：

```text
translation/spikenet_preprocessing.py
translation/debug_preprocessing_flow.py
translation/debug_mot_attention_preprocessing.py
```

### 核心逻辑

`spikenet_preprocessing.py` 翻译了 SpikeNet Matlab `matlab_interface/write*HDF5.m` 的核心协议层：

```text
new_ygin_file
write_basic_para
write_pop_para
write_chemical_connection
write_ext_current_pop
write_syn_para
write_init_cond
write_elif_neuron_model
write_ext_conductance_settings
write_neuron_sampling
write_expl_var
```

这些函数现在全部使用 Python/C++ simulator 约定的 0-based 索引：

```text
population: pop0, pop1, ...
synapse type: 0=AMPA, 1=GABA, 2=NMDA
neuron id: 0..N_pop-1
event x/y: x=0..W-1, y=0..H-1
```

写入 HDF5 时不再做 Matlab 风格的 `-1` 转换。需要重点检查的字段包括：

```text
/config/syns/syn*/INIT006/type
/config/syns/syn*/INIT006/i_pre
/config/syns/syn*/INIT006/j_post
/config/syns/syn*/INIT006/I
/config/syns/syn*/INIT006/J
/config/pops/pop*/SAMP001/neurons
```

`build_chen_gong_2019_input(...)` 是 `models/main_Chen_and_Gong_2019.m` 的 Python 版本入口。它负责：

```text
1. 创建 *_in.h5
2. 写入 N/dt/step_tot
3. 写入 E/I population 参数
4. 写入 ELIF 参数
5. 写入外部 current 输入文件路径
6. 根据距离生成 E/E、E/I、I/E、I/I 连接
7. 写入 synapse 参数
8. 写入采样配置
```

E/I population 参数本身是小规模 HDF5 元数据写入，主要耗时不在这里。当前实现中网格 mask、population 数量、外部 conductance 向量都用 NumPy 数组生成：

```python
grid = np.zeros((height, width), dtype=np.int8)
grid[1::2, 1::2] = 1
n = np.bincount(grid.ravel(), minlength=2)
ext_mean_std = [np.full(pop_n, f_ext, dtype=float) for pop_n in n]
```

真正的热点是距离连接生成，因此这部分已经改为 PyTorch/GPU 优先。

现在支持矩形网格：

```python
ChenGong2019Config(grid_shape=(height, width))
```

如果不传 `grid_shape`，则沿用旧的方形：

```python
ChenGong2019Config(gsize=250)
```

### GPU 距离连接生成

`generate_chen_gong_connections(...)` 现在支持三个 backend：

```text
auto: 优先 PyTorch，PyTorch 不可用时回退 NumPy
torch: 强制 PyTorch，适合 GPU 调试和正式生成
numpy: 纯 CPU fallback，用于等价性对照
```

配置方式：

```python
ChenGong2019Config(
    grid_shape=(height, width),
    connection_backend="torch",
    connection_device="cuda",   # 也可以是 cuda:0, cuda:1, cpu
)
```

PyTorch 路径把每个 population pair 的距离矩阵按 post-synaptic chunk 计算：

```text
pre_coords: [N_pre, 2]
post_chunk: [chunk, 2]
dist2: [N_pre, chunk]
mask = dist2 <= drange^2
torch.nonzero(mask) -> I/J
```

这样避免一次性构造完整 `[N_pre, N_post]` 距离矩阵。最终 `I/J/K/D` 仍然会回到 CPU NumPy 数组，因为 HDF5 写入是 CPU I/O。

注意：GPU/tensor 化能显著加速距离计算和筛选，但不能消除最终连接表本身的大小。如果 `drange` 很大、网格又是 `250x400`，最终 `I/J/K/D` 仍可能非常大，需要继续依靠 chunk、ROI、稀疏策略或 simulator 侧隐式连接。

### Attention 前输入调试

`debug_mot_attention_preprocessing.py` 复用 `test_snntracker.py` attention 层之前的路径：

```text
SpikeStream.get_block_spikes
-> STPFilter.update_dynamics
-> STPFilter.local_connect
-> stp_filter.lif_spk
```

`lif_spk` 就是原来送入 attention 的输入：

```python
self.object_detection.update_dnf(self.stp_filter.lif_spk)
attentionBox, attentionInput = self.object_detection.get_attention_location(self.stp_filter.lif_spk)
```

调试命令示例：

```bash
conda run -n snntracker_py311 python translation/debug_mot_attention_preprocessing.py \
  --data-path /home/hanruoshui/snnTracker/motVidarReal2025 \
  --scene spike59 \
  --block-len 220 \
  --calibration-time 150 \
  --export-frames 40 \
  --attention-size 15 \
  --scale-w 10 \
  --scale-h 10 \
  --output-dir /tmp/snntracker_mot_rect_debug \
  --device cpu \
  --connection-backend torch \
  --connection-device cuda
```

该命令会输出：

```text
pre-attention event HDF5
preview PNG
SpikeNet-style *_in.h5
HDF5 tree dump
JSON report
```

更小的 synthetic GPU preprocessing 验证：

```bash
conda run -n snntracker_py311 python translation/debug_preprocessing_flow.py \
  --make-synthetic \
  --output-dir /tmp/snntracker_zero_based_cuda_debug \
  --synthetic-height 12 \
  --synthetic-width 16 \
  --synthetic-frames 6 \
  --step-tot 200 \
  --connection-backend torch \
  --connection-device cuda \
  --preview-frames 4
```

这个脚本会检查：

```text
event x/y 是否为 0-based 且落在 grid 内
pop0/pop1 file_current_input/fname 是否指向真实事件文件
N/dt/step_tot 是否写入
n_syns 是否为 4
每个 syn*/INIT006/I,J,K,D 长度是否一致
I/J 是否满足 0 <= I < N_pre, 0 <= J < N_post
```

### 矩形输入注意事项

MOT 原始数据是：

```text
spikes[t, y, x]
shape = [T, 250, 400]
```

其中：

```text
y: 0..249
x: 0..399
```

`export_hdf5_events.py` 写出：

```text
x = col index
y = row index
t = frame timestamp
```

预处理现在可以接矩形 `250x400`，但完整 Chen/Gong 距离连接在全尺寸下会非常大。调试脚本默认对大网格跳过完整连接构建，需要用 `--scale-w/--scale-h`、ROI 或后续 chunked streaming/隐式连接策略处理。

### Inverse Pool

inverse pool 属于 preprocessing 内部的权重分配方法，不属于 post-processing。它只在 GU 风格 E-to-E 权重生成中需要，用于从全局权重池按入度和目标总输入强度分配权重。

当前代码提供可选适配：

```python
assign_inverse_pool_weights_from_repo_b(...)
```

它复用仓库 B：

```text
network_generator.sampling.weight_assign.inversepool.InversePoolWeightAssign
```

Chen/Gong 距离连接路径默认不需要 inverse pool。

## Post-Processing

### 目标

Post-processing 的目标是替代 SpikeNet Matlab `PostProcessYG -> ReadH5/ReadYG -> AnalyseYG -> SaveRYG` 的最小必要路径，并为 object detection 增加任务读出层。

当前主要代码：

```text
translation/spikenet_postprocessing.py
translation/debug_postprocessing_flow.py
translation/detection_readout.py
translation/debug_detection_readout.py
```

### 输出读取

`read_spikenet_out_h5(...)` 读取 SpikeNet C++ out.h5 的典型路径：

```text
/config_filename/config_filename
/pop_result_0/spike_hist_tot
/pop_result_0/num_spikes_pop
/pop_result_1/spike_hist_tot
/pop_result_1/num_spikes_pop
```

其中：

```text
spike_hist_tot: 每个时间步 spike neuron id 串接后的压缩向量
num_spikes_pop: 每个时间步的 spike 数量
```

`compressed_spikes_to_dense(...)` 把压缩格式还原为：

```text
dense_spike_hist: [T, N_pop]
```

这里也按 0-based spike neuron id 处理。也就是说 `spike_hist_tot` 中的 neuron id 应该满足：

```text
0 <= id < N_pop
```

`dense_pop_to_grid_frames(...)` 再根据 preprocessing 的 E/I grid mask 把 pop-local neuron index 映射回：

```text
frames: [T, H, W]
```

### Task Readout: Detection BBox

任务读出层已经从 HDF5 解析里拆出来，集中在：

```text
translation/detection_readout.py
```

它的输入不是 HDF5 文件，而是 post-processing 或 Python simulator 直接产生的 dense activity：

```text
frames: [T, H, W]
```

这使它可以同时服务两种情况：

```text
1. SpikeNet-style out.h5 -> spikenet_postprocessing.py -> frames
2. Python/PyTorch simulator -> tensor/ndarray frames
```

完整任务读出流程：

```text
activity frames [T,H,W]
-> integrate_activity_windows(window, stride)
-> threshold 或 threshold_quantile
-> optional binary_closing
-> remove_small_objects
-> connected component labeling
-> bbox list
-> optional per-window NMS
-> JSON / MOT txt
```

核心函数：

```python
readout_bboxes(...)
detect_bboxes_from_windows(...)
integrate_activity_windows(...)
nms_detections(...)
scale_detections(...)
save_detections_json(...)
save_mot_txt(...)
```

输出 bbox 坐标约定：

```text
x1, y1, x2, y2
x 对应 width / col
y 对应 height / row
```

`BBoxDetection` 字段：

```text
window_index: 第几个时间窗
t_start/t_end: 该 bbox 对应的原始时间帧范围，左闭右开
x1/y1/x2/y2: bbox 坐标
score: activity 强度分数
area: 连通域面积
label: 类别，占位为 1
```

如果 preprocessing 或 simulator 对输入做了下采样、ROI crop 或 resize，可以用：

```python
scale_detections(...)
```

把 simulator 坐标映射回原始 MOT 图像坐标。例如 `25x40` 调试结果映射回 `250x400`：

```python
scale_detections(dets, scale_x=10, scale_y=10)
```

`spikenet_postprocessing.py` 现在只负责读取 out.h5 和重建 activity frames，bbox 读出会委托 `detection_readout.py`。因此真实链路是：

```text
out.h5
-> read_activity_frames(...)
-> readout_bboxes(...)
-> detections.json / detections_mot.txt
```

### Task Readout 调试

单独调试 bbox 读出层：

```bash
conda run -n snntracker_py311 python translation/debug_detection_readout.py \
  --frames 20 \
  --height 48 \
  --width 64 \
  --window 4 \
  --stride 4 \
  --threshold 1 \
  --min-area 8 \
  --nms-iou 0.3 \
  --output-dir /tmp/snntracker_detection_readout_debug
```

该调试会生成：

```text
synthetic_activity.npy
detections.json
detections_mot.txt
activity_windows_preview.png
detection_readout_report.json
```

也可以直接从 `.npy` activity frames 调用读出层 CLI：

```bash
conda run -n snntracker_py311 python translation/detection_readout.py \
  /tmp/snntracker_detection_readout_debug/synthetic_activity.npy \
  --window 4 \
  --stride 4 \
  --threshold 1 \
  --min-area 8 \
  --nms-iou 0.3 \
  --output-json /tmp/readout_detections.json \
  --output-mot /tmp/readout_detections_mot.txt
```

### Post-Processing 调试

仓库当前没有真实 simulator out.h5，所以 `debug_postprocessing_flow.py` 支持合成 SpikeNet-style 输出：

```bash
conda run -n snntracker_py311 python translation/debug_postprocessing_flow.py \
  --make-synthetic \
  --synthetic-height 24 \
  --synthetic-width 32 \
  --synthetic-frames 20 \
  --window 4 \
  --stride 4 \
  --threshold 1 \
  --min-area 4 \
  --output-dir /tmp/snntracker_post_debug
```

该调试会生成：

```text
synthetic_in.h5
synthetic_out.h5
detections.json
detections_mot.txt
activity_windows_preview.png
postprocessing_debug_report.json
```

真实 simulator 输出可用：

```bash
conda run -n snntracker_py311 python translation/spikenet_postprocessing.py \
  /path/to/simulator_out.h5 \
  --config-h5 /path/to/input_in.h5 \
  --grid-shape 250 400 \
  --window 5 \
  --stride 5 \
  --threshold 1 \
  --min-area 4 \
  --output-json /tmp/detections.json \
  --output-mot /tmp/detections_mot.txt
```

### 与 GT 指标

当前后处理已输出 MOT txt：

```text
frame,id,x,y,width,height,score,-1,-1,-1
```

可以接现有 `tracking_mot.py` 的 `TrackingMetrics` 或后续加入 detection-only IoU/mAP 评估。当前实现重点是把 simulator 输出读出成可视化和 bbox，而不是替代完整 MOT 评估。
