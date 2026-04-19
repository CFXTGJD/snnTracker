# Input, Network, and Simulator Interface

本文档解释三个容易混淆的层次：

```text
1. 外部输入文件
2. preprocessing 生成的 *_in.h5 配置文件
3. C++ / Python simulator 读取配置并运行
```

重点回答：

```text
GU 风格 = 网络拓扑/权重生成方式
writeExtCurrentPopHDF5 = 外部输入文件引用方式
```

它们是什么关系，谁先谁后，以及数据格式一路如何接到 simulator。

## 一句话结论

`GU 风格` 和 `writeExtCurrentPopHDF5` 不是同一层。

```text
GU 风格:
    决定网络里有哪些 neuron population，
    哪些 neuron 之间有连接，
    连接权重 K 和 delay D 是多少。

writeExtCurrentPopHDF5:
    只是在 *_in.h5 里写一个外部输入文件路径，
    告诉 simulator 某个 population 的外部输入从哪个文件读。
```

它们都发生在 preprocessing 阶段，但写的是 `*_in.h5` 的不同部分：

```text
GU / ChenGong 风格 -> 写 /config/syns/syn*/INIT006/I,J,K,D
writeExtCurrentPopHDF5 -> 写 /config/pops/pop*/file_current_input/fname
```

所以不是“第二层在第一层之前还是之后”的前后级联关系。更准确地说：

```text
它们是同一个 *_in.h5 里的两个模块：
    一个模块描述网络结构；
    一个模块描述外部输入文件在哪里。
```

simulator 运行时会同时读这两部分。

## 总流程

完整流程可以画成：

```text
原始数据 / lif_spk / event stream
    |
    |  A. export_hdf5_events.py 或其他输入转换器
    v
外部输入文件 input_events.h5 或 input_current.h5

    同时：

preprocessing builder
    |
    |  B1. 写网络结构：N, pop 参数, syn I/J/K/D
    |  B2. 写外部输入引用：file_current_input/fname 或 file_spike_input/fname
    v
SpikeNet input config: *_in.h5

    |
    |  C. simulator import_HDF5
    v
C++ / Python simulator
    |
    |  D. 读取 *_in.h5 中的网络结构
    |  E. 按 *_in.h5 中的 fname 打开外部输入文件
    |  F. 运行神经元/突触动力学
    v
simulator output: *_out.h5

    |
    |  G. postprocessing
    v
activity frames / bbox detections
```

## 两类文件：不要混在一起

整个链路里至少有两个 HDF5 文件。

### 1. 外部输入文件

这是数据文件，里面放的是外部输入本身。

可能是 event-style：

```text
input_events.h5
    /x
    /y
    /t
    /pol
    attrs:
        frames_shape = (T, H, W)
        dt
        start_time
```

也可能是 C++ 原版 current-style：

```text
input_current.h5
    /current
    /neurons
    /frame_rate
    /mean_curr
    /start_step
    /end_step
```

这类文件不是网络配置，它只是“外部刺激/输入数据”。

### 2. `*_in.h5` 配置文件

这是 simulator 的主输入配置，里面描述：

```text
网络有多少 population
每个 population 有多少 neurons
神经元参数是什么
突触连接是什么
外部输入文件路径是什么
仿真 dt 和 step_tot 是多少
```

典型字段：

```text
*_in.h5
    /config/Net/INIT001/N
    /config/Net/INIT002/dt
    /config/Net/INIT002/step_tot

    /config/pops/pop0/...
    /config/pops/pop1/...

    /config/pops/pop0/file_current_input/fname
    /config/pops/pop1/file_current_input/fname

    /config/syns/n_syns
    /config/syns/syn0/INIT006/type
    /config/syns/syn0/INIT006/i_pre
    /config/syns/syn0/INIT006/j_post
    /config/syns/syn0/INIT006/I
    /config/syns/syn0/INIT006/J
    /config/syns/syn0/INIT006/K
    /config/syns/syn0/INIT006/D
```

`*_in.h5` 里通常不直接存完整输入事件，只存输入文件路径。

## GU 风格写的是哪部分？

GU 风格主要写 `*_in.h5` 的网络结构部分：

```text
/config/Net/INIT001/N
/config/pops/pop*/...
/config/syns/syn*/INIT006/I,J,K,D
```

比如 GU 默认：

```text
N_e = 3969
N_i = 1000
pop0 = excitatory
pop1 = inhibitory
```

GU 风格会决定：

```text
E/E 哪些 neuron 相连
I/E 哪些 neuron 相连
E/I 哪些 neuron 相连
I/I 哪些 neuron 相连
E/E 权重是否使用 inverse pool
I/E 权重是否按 E/E 入权重做平衡
delay D 怎么采样
```

最终写成：

```text
/config/syns/syn0/INIT006/I
/config/syns/syn0/INIT006/J
/config/syns/syn0/INIT006/K
/config/syns/syn0/INIT006/D
...
```

C++ simulator 不知道这些连接是 GU 生成的、Chen/Gong 生成的，还是手写的。它只读最终 `I/J/K/D`。

## writeExtCurrentPopHDF5 写的是哪部分？

`writeExtCurrentPopHDF5` 不生成网络连接。

它只写：

```text
/config/pops/popX/file_current_input/fname
```

含义是：

```text
popX 这个 population 有一个外部输入文件；
外部输入文件路径是 fname；
simulator 启动时需要打开这个 fname。
```

例如：

```text
/config/pops/pop0/file_current_input/fname = "/tmp/spike59_pre_attention_native.h5"
```

这不会改变：

```text
N
I/J/K/D
synapse type
网络拓扑
```

它只告诉 simulator：

```text
pop0 的外部输入从这个文件读。
```

所以它和 GU 风格的关系是并列关系：

```text
GU 风格:
    给 pop0/pop1 建网络连接。

writeExtCurrentPopHDF5:
    给 pop0/pop1 指定外部输入文件。
```

## 谁先谁后？

从“代码执行顺序”看，builder 里可以先写网络，也可以先写输入引用。

例如：

```python
write_basic_para(...)
write_pop_para(...)
write_ext_current_pop(...)
write_chemical_connection(...)
```

或者：

```python
write_basic_para(...)
write_pop_para(...)
write_chemical_connection(...)
write_ext_current_pop(...)
```

都可以，因为它们只是往 `*_in.h5` 的不同路径写数据。

真正的依赖关系是：

```text
1. 外部输入文件必须先存在，或者至少路径最终要有效。
2. *_in.h5 必须写好网络结构和外部输入引用。
3. simulator 再读取 *_in.h5。
```

所以按数据生命周期看：

```text
先准备外部输入文件
再生成 *_in.h5，里面引用外部输入文件
最后 simulator 读 *_in.h5，并进一步打开外部输入文件
```

但在 `*_in.h5` 内部，GU 连接字段和 external input fname 字段没有先后关系。

## 用 GU 风格时数据会怎么走？

如果采用 GU-style preprocessing，完整格式可以是：

```text
external event file:
    /tmp/events_50x80.h5
        /x
        /y
        /t
        /pol
        attrs/frames_shape = (T, 50, 80)

GU-style *_in.h5:
    /config/Net/INIT001/N = [N_e, N_i]
    /config/Net/INIT002/dt = 0.1
    /config/Net/INIT002/step_tot = ...

    /config/pops/pop0/file_current_input/fname = "/tmp/events_50x80.h5"

    /config/syns/syn0/INIT006/type = 0
    /config/syns/syn0/INIT006/i_pre = 0
    /config/syns/syn0/INIT006/j_post = 0
    /config/syns/syn0/INIT006/I = E/E pre neuron ids
    /config/syns/syn0/INIT006/J = E/E post neuron ids
    /config/syns/syn0/INIT006/K = E/E weights
    /config/syns/syn0/INIT006/D = E/E delays

    /config/syns/syn1/INIT006/type = 1
    /config/syns/syn1/INIT006/i_pre = 1
    /config/syns/syn1/INIT006/j_post = 0
    /config/syns/syn1/INIT006/I = I/E pre neuron ids
    /config/syns/syn1/INIT006/J = I/E post neuron ids
    /config/syns/syn1/INIT006/K = I/E weights
    /config/syns/syn1/INIT006/D = I/E delays

    ...
```

simulator 读到这个 `*_in.h5` 后：

```text
1. 创建 pop0 和 pop1
2. 创建 syn0/syn1/syn2/syn3
3. 打开 pop0 的外部输入文件 /tmp/events_50x80.h5
4. 把 event 映射到 pop0 neuron
5. 每个 timestep 更新外部输入、突触电流、膜电位、spike
6. 写出 out.h5
```

## 和 C++ 原版的关键区别

这里必须特别小心。

SpikeNet C++ 原版中：

```text
file_current_input
```

实际读的是 current-style 文件：

```text
/current
/neurons
/frame_rate
/mean_curr
/start_step
/end_step
```

而不是 event-style 文件：

```text
/x
/y
/t
/pol
```

event-style 更接近 C++ 原版的：

```text
file_spike_input
```

它读：

```text
/x
/y
/t
/max_x
/max_y
```

并用：

```text
neuron_id = x + max_x * y
```

把事件映射成 spike neuron。

所以如果你要严格接 C++ 原版，有两种正确做法。

### 做法 A：用 file_spike_input 接 event 文件

`*_in.h5` 写：

```text
/config/pops/pop0/file_spike_input/fname = "/tmp/events.h5"
```

外部 event 文件写：

```text
/x
/y
/t
/max_x
/max_y
```

优点：

```text
语义和 C++ 原版一致
event 直接作为 spike 输入
```

问题：

```text
当前 translation 主要写的是 file_current_input/fname
需要补一个 write_spike_file_input_pop(...)
event 文件也要补 max_x/max_y
```

### 做法 B：把 event 转成 current-style HDF5

`*_in.h5` 继续写：

```text
/config/pops/pop0/file_current_input/fname = "/tmp/current_input.h5"
```

但外部文件必须是：

```text
/current
/neurons
/frame_rate
/mean_curr
/start_step
/end_step
```

优点：

```text
兼容 C++ 原版 file_current_input
```

问题：

```text
需要把 x/y/t/pol event 先 bin 到 neuron current matrix
这会改变 event/spike 输入语义
```

### 做法 C：Python simulator 自定义 file_current_input 语义

如果用的是翻译后的 Python simulator，而不是 C++ 原版，可以约定：

```text
/config/pops/pop0/file_current_input/fname
```

虽然名字叫 `file_current_input`，但 Python reader 把它当 event-style HDF5 读：

```text
/x
/y
/t
/pol
attrs/frames_shape
```

优点：

```text
当前 preprocessing 改动最小
最容易和现有 debug_preprocessing_flow.py 接上
```

缺点：

```text
名字和 C++ 原版语义不一致
需要在 Python simulator 接口文档里写清楚
```

## 输入映射发生在哪里？

输入映射不是 GU 风格本身的一部分，也不是 `writeExtCurrentPopHDF5` 本身的一部分。

输入映射发生在：

```text
simulator 读取外部输入文件之后，
把外部输入坐标或 neuron id 转换成 population-local neuron id 的时候。
```

例如 event-style：

```text
event file: x, y, t
```

需要映射成：

```text
pop0 neuron id
```

如果网络是 Chen/Gong 矩形 grid：

```text
grid_shape = (250, 400)
event (y, x) -> pop0_map[y, x]
```

如果网络是 GU 默认：

```text
GU E lattice = 63x63
event 原图 = 250x400
```

那就不能直接用原始 `x/y`。必须先做：

```text
250x400 -> 63x63
```

或者：

```text
250x400 -> 50x80
```

然后再：

```text
event (y, x) -> GU pop0 neuron id
```

因此 GU 风格的输入问题是：

```text
网络拓扑定义的 neuron 坐标空间
和
外部输入 event 的图像坐标空间
是否一致。
```

如果一致，映射简单；如果不一致，就需要 resize / projection / ROI。

## 输出映射发生在哪里？

输出映射发生在 simulator 之后、postprocessing 之前。

simulator 输出通常是：

```text
pop0 spike neuron ids
```

或者：

```text
pop0 dense activity [T, N_pop0]
```

postprocessing 要把它变成：

```text
activity frames [T, H, W]
```

如果网络就是图像 grid：

```text
pop0 neuron id -> (y, x)
```

很直接。

如果网络是 GU 默认 `63x63`，但原始图像是 `250x400`，输出要么：

```text
先得到 [T,63,63]
再 scale bbox 回 [250,400]
```

要么：

```text
用 projection inverse / splatting
把 GU neuron activity 投回原图坐标
```

## 当前最推荐的工程接法

为了让概念最清楚，我建议按下面的接口定义走。

### 如果接 Python simulator

第一版推荐：

```text
preprocessing:
    生成 *_in.h5
    写网络结构 I/J/K/D
    写 /config/pops/pop0/file_current_input/fname = event_h5_path

event_h5:
    /x, /y, /t, /pol
    attrs/frames_shape

python simulator:
    明确把 file_current_input/fname 当 event HDF5 读
    用 grid_shape 把 x/y 映射到 pop0 neuron
    输出 /activity: [T,H,W]

postprocessing:
    直接读 /activity
    bbox readout
```

这条链路的优点是最少改当前代码，最快打通 object detection。

### 如果接 C++ 原版 simulator

不要把当前 `x/y/t/pol` event 文件直接塞给 `writeExtCurrentPopHDF5`。

应该二选一：

```text
方案 1:
    增加 writeSpikeFileInputPopHDF5 的 Python 翻译
    event 文件补 /max_x, /max_y
    C++ 走 file_spike_input

方案 2:
    把 event bin 成 current-style HDF5
    C++ 继续走 file_current_input
```

严格说，方案 1 更符合 event/spike 输入语义。

## 一个具体例子

假设要用 GU-like 矩形网络 `50x80`。

### 外部输入文件

先把原始 `250x400` lif_spk resize/bin 到 `50x80`，写成：

```text
/tmp/spike59_50x80_events.h5
    /x: 0..79
    /y: 0..49
    /t
    /pol
    attrs/frames_shape = (T, 50, 80)
```

### Preprocessing 生成 config

生成：

```text
/tmp/gu_50x80_in.h5
```

里面同时有：

```text
/config/Net/INIT001/N = [N_e, N_i]
/config/pops/pop0/file_current_input/fname = "/tmp/spike59_50x80_events.h5"
/config/syns/syn*/INIT006/I,J,K,D = GU-style generated connections
```

这里：

```text
file_current_input/fname 只是指向输入文件
GU-style generated connections 只是网络拓扑
```

### Simulator

Python simulator 读：

```text
/tmp/gu_50x80_in.h5
```

然后：

```text
读 N/dt/step_tot
读 syn I/J/K/D
读 pop0 的 fname
打开 /tmp/spike59_50x80_events.h5
把 event x/y 映射到 pop0 neuron
运行
写 /tmp/gu_50x80_out.h5
```

### 输出

推荐写：

```text
/tmp/gu_50x80_out.h5
    /activity: [T, 50, 80]
```

postprocessing 得到 bbox 后再 scale：

```text
scale_x = 400 / 80 = 5
scale_y = 250 / 50 = 5
```

映射回原始图像坐标。

## 最核心的区分

可以这样记：

```text
GU / ChenGong:
    决定网络内部怎么连。
    输出 I/J/K/D。

writeExtCurrentPopHDF5:
    决定外部输入文件在哪里。
    输出 fname。

输入映射:
    决定外部输入怎么变成某个 population 的 neuron activity。
    发生在 simulator reader 里，或者预先转换输入文件时。

输出映射:
    决定 simulator 的 neuron activity 怎么变回图像 activity。
    发生在 postprocessing 里。
```

