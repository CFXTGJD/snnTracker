# Purpose

本文档说明两个辅助脚本的用途和判读标准：`visualization/plot_spike_dataset.py` 用于快速巡检 spike 数据集可视化结果，`tools/toy_validate_modules.py` 用于用可控 toy examples 验证动态适应模块和运动估计模块是否按预期工作。建议在接入新数据集、修改 STP 参数、优化 motion estimation、或评测前做数据质量检查时运行。

# What is Verified

`plot_spike_dataset.py` 验证数据读取和可视化是否正常：能否从 `scene_dir/spikes.dat` 按 `config.yaml` 的 `spike_h/spike_w` 正确解码；能否生成采样 spike 帧、每帧活跃像素曲线、空间累积热力图、时间投影图；如果存在 `spikes_gt.txt`，还能检查 GT 框和 spike 底图是否对齐。

`toy_validate_modules.py` 验证模块级行为：STP toy 检查静态高频闪烁是否被适应抑制、运动或突发目标是否被保留；motion toy 检查简单运动目标是否形成稳定方向；direction sweep 建立输入方向到 `motion ID` 的实测映射；complex motion toy 检查 8 个方向、2 个速度、带稀疏随机噪声条件下的方向恢复鲁棒性。`tools/motion_parameter_sweep.py` 进一步扫描 `speed/object size/noise probability`，用于定位方向检测失败边界。`tools/motion_multi_object_no_overlap_sweep.py` 验证同帧多目标但轨迹不重叠时的局部方向估计；`tools/motion_multi_object_overlap_sweep.py` 验证目标轨迹交叉/重叠时的方向保持能力。当前 `motion_direction_sweep` 的实测映射为：`right -> 7`，`left -> 3`，`down -> 1`，`up -> 5`，`down_right -> 8`，`down_left -> 2`，`up_right -> 6`，`up_left -> 4`。

`motion_complex_toy_validation` 的构造方式是 16 个独立 case，而不是一个多目标混合场景：8 个方向分别为 `right/left/down/up/down_right/down_left/up_right/up_left`，每个方向测试 `speed 1` 和 `speed 2` 两种速度；每个 case 中有一个 `5x5` 移动方块目标，从画面中心附近出发，同时叠加概率为 `0.001` 的稀疏随机噪声。该设计用于检查单目标方向估计在速度变化和轻噪声下是否稳定，不验证多目标分离能力。

多目标无重叠 sweep 在同一帧放置 4 个互不重叠目标：`right/left/down_right/up_left`。多目标重叠 sweep 使用 3 个两目标交叉场景：`horizontal_cross` 为 `right vs left`，`vertical_cross` 为 `down vs up`，`diagonal_cross` 为 `down_right vs up_left`。两者的扫参范围由单目标失败边界收窄得到：`speed=1/2`，`box_size=3/5/7`，`noise_prob=0/0.00025/0.0005`。

# Mapping to Original Code

| Validation Item | Script Location | Original Module / Function |
|---|---|---|
| Spike `.dat` 解码 | `visualization/plot_spike_dataset.py` `main()` | `spkData/load_dat.py` `SpikeStream.get_block_spikes()` |
| 采样帧 montage | `save_montage()` | 数据集 spike frame 可视化巡检 |
| 活跃像素曲线 | `save_activity_curve()` | spike activity / frame 统计 |
| 空间累积热力图 | `save_spatial_heatmap()` | spatial spike accumulation |
| 时间投影图 | `save_temporal_projection()` | x-time / y-time 运动痕迹检查 |
| GT 框叠加 | `load_gt_boxes()`, `save_gt_overlay()` | `spikes_gt.txt` MOT 格式标注检查 |
| STP 简单验证 | `run_stp_toy()` | `spkProc/filters/stp_filters_torch.py` `STPFilter.update_dynamics()` |
| STP 复杂验证 | `run_stp_complex_toy()` | `STPFilter.update_dynamics()`, `filter_spk` |
| 运动估计简单验证 | `run_motion_toy()` | `spkProc/motion/motion_detection.py` `stdp_tracking()`, `local_wta()` |
| 方向 ID 映射 | `run_motion_direction_sweep()` | `motion_estimation.local_wta()` |
| 复杂方向压力测试 | `run_motion_complex_toy()` | `motion_weight`, `motion_id`, `motion_vector_max` |
| 方向检测参数扫参 | `tools/motion_parameter_sweep.py` | `motion_estimation.stdp_tracking()`, `motion_estimation.local_wta()` |
| 多目标无重叠扫参 | `tools/motion_multi_object_no_overlap_sweep.py` | `motion_id` per object mask |
| 多目标轨迹重叠扫参 | `tools/motion_multi_object_overlap_sweep.py` | `motion_id` per object mask, overlap / non-overlap metrics |

# How to Run

```bash
python visualization/plot_spike_dataset.py --scene_dir motVidarReal2025/spike59 --begin_idx 0 --num_frames 300 --num_samples 8 --output_dir results/spike_dataset_viz
python tools/toy_validate_modules.py --output_dir results/toy_module_validation
python tools/motion_parameter_sweep.py --output_dir results/motion_parameter_sweep
python tools/motion_multi_object_no_overlap_sweep.py --output_dir results/motion_multi_object_no_overlap_sweep
python tools/motion_multi_object_overlap_sweep.py --output_dir results/motion_multi_object_overlap_sweep
```

| Parameter | Script | Default | Description |
|---|---|---|---|
| `--scene_dir` | `plot_spike_dataset.py` | required | 场景目录，必须包含 `spikes.dat` |
| `--begin_idx` | `plot_spike_dataset.py` | `0` | 起始帧 |
| `--num_frames` | `plot_spike_dataset.py` | `300` | 读取帧数 |
| `--num_samples` | `plot_spike_dataset.py` | `8` | montage / GT overlay 中展示的采样帧数 |
| `--output_dir` | both | script-specific | 输出目录 |
| `--gt_path` | `plot_spike_dataset.py` | `scene_dir/spikes_gt.txt` | 可选 GT 文件路径 |
| `--speeds` | `motion_parameter_sweep.py` | `"1 2 3"` | 扫描的目标速度，单位为 pixels/frame |
| `--box_sizes` | `motion_parameter_sweep.py` | `"3 5 7"` | 方块目标边长 |
| `--noise_probs` | `motion_parameter_sweep.py` | `"0 0.001 0.003"` | 每像素随机 spike 概率 |
| `--correct_thr` | `motion_parameter_sweep.py` | `0.70` | 每个方向正确率通过阈值 |
| `--hit_thr` | `motion_parameter_sweep.py` | `0.70` | 每个方向有效 motion 输出通过阈值 |
| `--speeds` | multi-object sweep | `"1 2"` | 多目标测试速度范围，聚焦已知失败边界 |
| `--noise_probs` | multi-object sweep | `"0 0.00025 0.0005"` | 多目标测试噪声范围，聚焦低噪声边界 |

# Outputs

```text
results/
├── spike_dataset_viz/<scene>/
│   ├── <scene>_montage.png              # 均匀采样 spike 帧，用于检查解码和局部目标形态。
│   ├── <scene>_activity_curve.png       # 每帧 active pixels 数量，用于定位运动活跃时间段。
│   ├── <scene>_spatial_heatmap.png      # 时间维累积热力图，用于观察目标经过区域和固定噪声。
│   ├── <scene>_temporal_projection.png  # x-time / y-time 投影，用于观察时空运动轨迹。
│   └── <scene>_gt_overlay.png           # GT 框叠加图，仅在存在 GT 时生成，用于检查标注对齐。
└── toy_module_validation/
    ├── stp_toy_validation.png           # 简单 STP sanity check。
    ├── stp_complex_toy_validation.png   # 静态、噪声、运动、突发目标混合的 STP 压力检查。
    ├── motion_toy_validation.png        # 单方向运动估计稳定性检查。
    ├── motion_direction_sweep.png       # 输入方向到 motion ID 的实测映射。
    ├── motion_complex_toy_validation.png# 多方向、多速度、带噪声的 motion 压力检查。
    └── summary.txt                      # 所有 toy validation 的数值指标和 pass/fail。
results/motion_parameter_sweep/
├── motion_parameter_sweep_cases.csv     # 每个 direction/speed/box/noise case 的 expected_id、stable_id、correct_rate、hit_rate。
├── motion_parameter_sweep_aggregate.csv # 每个 speed/box/noise 组合在 8 个方向上的最小/平均正确率和失败方向。
├── motion_sweep_noise_0.png             # noise=0 时 speed 与 box_size 的通过边界热力图。
├── motion_sweep_noise_0.001.png         # noise=0.001 时的方向检测鲁棒性热力图。
├── motion_sweep_noise_0.003.png         # noise=0.003 时的方向检测鲁棒性热力图。
└── summary.txt                          # 参数范围、通过组合、失败组合和失败方向摘要。
results/motion_multi_object_no_overlap_sweep/
├── multi_no_overlap_cases.csv           # 每个目标的 expected_id、stable_id、correct_rate、hit_rate。
├── multi_no_overlap_aggregate.csv       # 每个 speed/box/noise 组合的最小正确率、失败目标和 pass/fail。
├── multi_no_overlap_noise_*.png         # 无重叠多目标在不同噪声下的 speed/box 热力图。
└── summary.txt                          # 无重叠多目标通过组合和失败组合摘要。
results/motion_multi_object_overlap_sweep/
├── multi_overlap_cases.csv              # 每个交叉场景目标的全帧与非重叠区域指标。
├── multi_overlap_aggregate.csv          # 每个 speed/box/noise 组合的重叠测试汇总。
├── multi_overlap_noise_*.png            # 轨迹重叠场景在不同噪声下的 speed/box 热力图。
└── summary.txt                          # 重叠多目标通过组合和失败组合摘要。
```

当前已运行所有本地数据集可视化：`badminton/cpl1/cplCam/pingpong/rotTrans/spike59` 均生成 montage、activity curve、spatial heatmap 和 temporal projection；其中 `rotTrans` 与 `spike59` 存在 `spikes_gt.txt`，额外生成 GT overlay。`tools/toy_validate_modules.py` 当前输出显示 STP 简单与复杂验证通过，motion 简单验证和 direction sweep 通过，motion complex 压力测试未通过。

`toy_module_validation` 中各图的读法如下：`stp_toy_validation.png` 左上/右上是输入 spike 帧，横轴为 width、纵轴为 height；左下 `Filtered Spike Response` 横轴为 frame，纵轴为 binary pass，静态点曲线应后期接近 0，运动目标曲线应在出现后保持较高；右下 `Adaptive Threshold at Static Pixel` 横轴为 frame，纵轴为静态点处自适应阈值。`stp_complex_toy_validation.png` 第一行是复杂输入帧；第二行左图横轴为 frame、纵轴为 pass/hit rate，用于比较静态结构、随机噪声、两个运动目标的通过情况；第二行右图横轴为 frame、纵轴为静态区域平均阈值；第三行左图横轴为 frame、纵轴为 spike pixels，比较输入和过滤后数量；第三行右图横轴为 frame、纵轴为 burst mask pass rate，检查短暂突发目标是否响应。

`motion_toy_validation.png` 左上/右上是运动方块起止帧；左下 `Dominant Motion ID per Frame` 横轴为 frame，纵轴为 motion ID，稳定水平线表示方向估计稳定；右下 `Recovered Motion Statistics` 横轴为 frame，纵轴为数值，`mean dx/mean dy` 表示有效运动像素的平均运动向量分量，`active motion pixels` 表示有效运动像素数量。`motion_direction_sweep.png` 横轴为输入方向，纵轴为 recovered motion ID；每个方向应恢复出非零且互不重复的 ID。`motion_complex_toy_validation.png` 三个热力图横轴均为 `speed 1/speed 2`，纵轴均为 8 个输入方向；左图颜色和值为 `correct_rate`，中图为 `hit_rate`，右图为稳定恢复出的 `motion ID`。

`motion_parameter_sweep` 当前默认扫描 `speed=1/2/3`、`box_size=3/5/7`、`noise_prob=0/0.001/0.003`，共 27 个组合，每个组合跑 8 个方向；另做了 speed=1 低噪声细扫：`noise_prob=0/0.00025/0.0005/0.001`。扫参图中横轴为 speed，纵轴为 box size；左图为 8 个方向中的最小 `correct_rate`，中图为最小 `hit_rate`，右图为该组合是否整体通过。`noise_prob=0.001` 对应 `72x96` 图像中约 `6.9` 个随机 spike/frame，`0.003` 约 `20.7` 个随机 spike/frame。

多目标无重叠图 `multi_no_overlap_noise_*.png` 的横轴为 speed，纵轴为 box size；左图是 4 个目标中的最小 `correct_rate`，中图是最小 `hit_rate`，右图是该组合是否所有目标通过。多目标重叠图 `multi_overlap_noise_*.png` 的横轴和纵轴相同；左图是所有交叉目标的全帧最小 `correct_rate`，中图是排除物理重叠像素后的最小 `non_overlap_correct_rate`，右图是整体 pass mask。若全帧和非重叠区域都低，说明不是单纯遮挡导致，而是交叉场景下方向估计本身不稳定。

# How to Interpret Results

正常情况下，montage 中 spike 不应全黑或全屏异常发亮；activity curve 应能反映运动强弱变化；spatial heatmap 应显示目标经过区域；temporal projection 应出现连续斜线或轨迹；GT overlay 中绿色框应覆盖 spike 目标。若 GT 框整体偏移，通常是帧号、坐标系、上下翻转或分辨率配置问题。

STP 结果中，`static_late_mean` 和 `noise_mean` 应低，`moving*_mean`、`burst_peak` 应高，`suppression_ratio` 应明显小于 1。当前结果为：`static_late_mean=0.0000`，`noise_mean=0.0380`，`moving1_mean=1.0000`，`moving2_mean=0.8500`，`burst_peak=1.0000`，`suppression_ratio=0.0875`，说明静态结构和随机噪声被有效抑制，两个运动目标和短暂突发目标被保留。若静态通过率高，说明适应抑制不足；若运动目标通过率低，说明阈值过严或目标响应太弱。

Motion 结果中，`motion_direction_sweep` 的 mapping 是判读 `motion ID` 的依据；例如简单向右 toy 输出 `stable_dir=7`，结合映射可判定为 `right`，当前该项通过。复杂 motion toy 中 `hit_rate` 高表示目标被检测到，`correct_rate` 高表示方向分类正确。当前 `min_hit_rate=1.0000` 但 `min_correct_rate=0.1667`，因此 `passed=False`：目标区域都能形成有效 motion 输出，但部分方向被估成相邻 ID。典型异常包括 `left speed_1` 期望 `3` 但稳定为 `2`，`down_right speed_2` 期望 `8` 但稳定为 `7`，`up_left speed_2` 期望 `4` 但稳定为 `3`。这通常表示方向分类鲁棒性不足，尤其在对角方向、速度变化或噪声干扰下容易发生混淆。

`summary.txt` 的指标含义和判别标准：`device` 表示运行设备；`stp_toy.static_late_mean` 是第 20 帧后静态点平均通过率，越低越好；`stp_toy.moving_late_mean` 是第 25 帧后运动目标通过率，需比静态点高至少 `0.4` 才通过。`stp_complex_toy.static_late_mean < 0.15`、`noise_mean < 0.35`、`moving1_mean > 0.55`、`moving2_mean > 0.55`、`burst_peak > 0.5`、`suppression_ratio < 0.8` 时判为通过。`motion_toy.stable_dir` 是 warmup 后最常见 motion ID，`dominant_dirs` 应非空且所有值一致；`mean_dx/mean_dy` 用于辅助判断方向向量，不直接作为 pass 条件。`motion_direction_sweep.mapping` 是方向到 ID 的校准表，`unique_nonzero` 应等于 8。`motion_complex_toy.cases` 中每个 case 记录 `expected_id/stable_id/correct_rate/hit_rate`；脚本要求所有 case 的 `correct_rate >= 0.70` 且 `hit_rate >= 0.70`，否则 `passed=False`。

参数扫参结果显示当前运动方向检测的可靠区间较窄：主 sweep 中 27 个组合仅 3 个通过，均为 `noise=0` 且 `speed=1`，对应 `box_size=3/5/7`。无噪声时，`speed=1` 基本稳定；`speed>=2` 后主要失败在对角方向，典型失败方向为 `down_right/down_left/up_right/up_left`，但水平和竖直方向仍多能保持正确。加入噪声后边界明显收缩：speed=1 低噪声细扫显示，`noise_prob=0.00025` 时部分 box size 仍可通过，`noise_prob=0.0005` 时 `box_size=3/5/7` 全部失败，说明噪声失败边界大致在 `0.00025-0.0005` 之间，即约 `1.7-3.5` 个随机 spike/frame。总体结论是：当前 motion 模块能稳定检测干净、单目标、低速运动；高速对角运动和稀疏随机噪声会显著降低方向分类正确率。

在 box_size=5、speed=[1,2]、noise_prob=0.001 的设置下，motion complex 压力测试失败；即使 noise_prob=0，speed 2 的部分对角方向仍然存在方向混淆。

多目标无重叠 sweep 共 18 个组合，其中 7 个通过、11 个失败。无噪声且 `speed=1` 的 `box_size=3/5/7` 全部通过；`speed=2` 全部失败，主要失败对象为 `obj_right`、`obj_down_right`、`obj_up_left`。在 `noise_prob=0.00025` 时，`box_size=3/7` 仍通过，`box_size=5` 因 `obj_down_right` 失败；在 `noise_prob=0.0005` 时，`box_size=5/7` 通过但 `box_size=3` 失败。结论是：同帧多目标但无重叠时，低速仍可工作，但对角方向和 speed=2 仍是主要边界。

多目标重叠 sweep 共 18 个组合，其中 4 个通过、14 个失败。无噪声且 `speed=1` 的 `box_size=3/5/7` 全部通过；`speed=2` 下水平、垂直、对角交叉全部失败。加噪声后，`noise_prob=0.00025` 下全部失败，常见失败目标包括 `vertical_cross:up_obj` 与 `diagonal_cross` 两个目标；`noise_prob=0.0005` 下仅 `speed=1, box_size=7` 通过。该结果说明轨迹交叉比无重叠更严格，轻噪声会明显放大方向混淆；但在干净、低速条件下，交叉场景仍有可用区间。
