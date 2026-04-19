# SpikeNet 代码逻辑讲解

本文档基于本地仓库：

```text
/home/hanruoshui/SpikeNet
```

重点解释 SpikeNet 的输入输出、网络结构、神经元模型、连接方式、前向动力学和学习/可塑性机制。

## 整体架构

SpikeNet 分三层：

```text
Matlab preprocessing
-> C++ simulator
-> Matlab post-processing
```

README 中也明确说明它有三个独立组件：

```text
1. 用于配置 spiking network 的用户接口
2. C++ simulator
3. 用于解析和分析仿真结果的用户接口
```

对应代码大致是：

```text
main_demo.m / models/main_*.m
matlab_interface/write*HDF5.m
    ↓ 生成 *_in.h5

cpp_sources/main.cpp
cpp_sources/SimuInterface.cpp
cpp_sources/NeuroNet.cpp
cpp_sources/NeuroPop.cpp
cpp_sources/ChemSyn.cpp
    ↓ 生成 *_out.h5 + *_neurosamp.h5 + restart_*.h5

post_processing/PostProcessYG.m
matlab_interface/ReadH5.m
matlab_interface/SaveRYG.m
```

C++ 入口在：

```text
/home/hanruoshui/SpikeNet/cpp_sources/main.cpp
```

它读取命令行里的每个 `.h5` 输入文件，然后：

```cpp
SimuInterface simulator;
success = simulator.import_HDF5(argv[i]);
simulator.simulate();
```

所以 C++ simulator 的直接输入是：

```text
*_in.h5
```

直接输出是：

```text
*_out.h5
*_restart_*.h5
可选 *_neurosamp.h5
```

## 输入是什么

主要输入是 Matlab/Python preprocessing 写出的 HDF5 配置文件。C++ 读取逻辑在：

```text
/home/hanruoshui/SpikeNet/cpp_sources/SimuInterface.cpp
```

核心字段包括：

```text
/config/Net/INIT001/N
/config/Net/INIT002/dt
/config/Net/INIT002/step_tot
```

含义：

```text
N: 每个 population 的神经元数量
dt: 仿真时间步长，单位 ms
step_tot: 总仿真步数
```

每个 population 的配置在：

```text
/config/pops/pop0/...
/config/pops/pop1/...
```

常见字段：

```text
/config/pops/pop*/PARA001/para_str_ascii
/config/pops/pop*/INIT011/r_V0
/config/pops/pop*/INIT011/p_fire
/config/pops/pop*/INIT012/mean
/config/pops/pop*/INIT012/std
/config/pops/pop*/file_current_input/fname
/config/pops/pop*/file_spike_input/fname
/config/pops/pop*/SAMP001/...
/config/pops/pop*/neuron_model
/config/pops/pop*/ELIF/...
```

突触连接配置在：

```text
/config/syns/n_syns
/config/syns/syn0/INIT006/type
/config/syns/syn0/INIT006/i_pre
/config/syns/syn0/INIT006/j_post
/config/syns/syn0/INIT006/I
/config/syns/syn0/INIT006/J
/config/syns/syn0/INIT006/K
/config/syns/syn0/INIT006/D
```

含义：

```text
type: synapse type, C++ 内部 0=AMPA（兴奋）, 1=GABA（抑制）, 2=NMDA（慢兴奋）
i_pre: pre population index 哪个population发信号
j_post: post population index 哪个population接收
边列表（只存有的连接-稀疏）不同于权重矩阵
I: pre neuron indices
J: post neuron indices
K: synaptic coupling strengths 对应每一条连接的权重
D: conduction delays, ms spike从I到J需要的时间
```

Matlab 写入函数会把 Matlab 1-based index 转成 C++ 0-based index。

## 一个重要输入接口问题

需要特别注意：

`writeExtCurrentPopHDF5.m` 的注释说它是 “DVS input event/spike data”，但 C++ 里 `file_current_input` 实际调用的是：

```cpp
NeuroPop::load_file_current_input(fname)
```

代码在：

```text
/home/hanruoshui/SpikeNet/cpp_sources/NeuroPop.cpp
```

它读取的是：

```text
/current
/neurons
/frame_rate
/mean_curr
/start_step
/end_step
```

也就是 `ForExtCurrent/ImageProcess_DOG.m` 生成的那种 current 文件。

而真正读取事件流 `x/y/t` 的函数是：

```cpp
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

并映射：

```cpp
neuron_id = x[i] + max_x * y[i]
```

所以当前 SpikeNet C++ 原版里：

```text
file_current_input -> /current, /neurons, /frame_rate ...
file_spike_input   -> /x, /y, /t, /max_x, /max_y
```

这点和 `export_hdf5_events.py` 的 `x/y/t/pol` 接口需要核对。如果 Python simulator 已经改成 `file_current_input` 接 `x/y/t/pol`，那没问题；但如果直接跑 SpikeNet C++ 原版，事件流更应该走 `writeSpikeFileInputPopHDF5` 风格，而不是 `writeExtCurrentPopHDF5` 风格。

这是后续对接 simulator 时必须确认的接口差异。

## 输出是什么

C++ simulator 的主输出由：

```cpp
SimuInterface::output_results_HDF5()
```

生成，位置在：

```text
/home/hanruoshui/SpikeNet/cpp_sources/SimuInterface.cpp
```

它创建：

```text
*_out.h5
```

并写：

```text
/config_filename/config_filename
/run_away_killed/step
/pop_result_0/...
/pop_result_1/...
/syn_result_0/...
/syn_result_1/...
```

population 输出由：

```cpp
NeuroPop::output_results(...)
```

写入，主要字段：

```text
/pop_result_*/spike_hist_tot
/pop_result_*/num_spikes_pop
/pop_result_*/num_ref_pop
/pop_result_*/pop_para
/pop_result_*/stats_V_mean
/pop_result_*/stats_V_std
/pop_result_*/stats_I_input_mean
/pop_result_*/stats_I_input_std
...
```

其中最核心的是：

```text
spike_hist_tot
num_spikes_pop
```

它们是压缩 spike history：

```text
num_spikes_pop[t] = 第 t 步 spike 数
spike_hist_tot = 所有时间步 spiking neuron id 串接在一起
```

要重建 dense spike matrix，需要按 `num_spikes_pop` 把 `spike_hist_tot` 切回每个时间步。

synapse 输出由：

```cpp
ChemSyn::output_results(...)
```

写入：

```text
/syn_result_*/syn_para
/syn_result_*/sample_data
/syn_result_*/stats_I_mean
/syn_result_*/stats_I_std
/syn_result_*/stats_s_time_mean
/syn_result_*/stats_s_time_var
...
```

如果开启 neuron sampling，另有：

```text
*_0_neurosamp.h5
*_1_neurosamp.h5
```

里面可能有：

```text
V
I_leak
I_AMPA
I_GABA
I_NMDA
I_ext
I_K
rhat
```

## 网络结构是什么

核心网络类是：

```text
NeuroNet
```

代码在：

```text
/home/hanruoshui/SpikeNet/cpp_sources/NeuroNet.cpp
```

它包含：

```cpp
vector<int> N_array;
vector<NeuroPop*> NeuroPopArray;
vector<ChemSyn*> ChemSynArray;
```

也就是：

```text
一个 network = 多个 neuron populations + 多个 chemical synapse blocks
```

population 是：

```text
NeuroPop
```

每个 `NeuroPop` 是同一类神经元的集合，例如 excitatory population 和 inhibitory population。

synapse block 是：

```text
ChemSyn
```

每个 `ChemSyn` 表示一组从某个 pre population 到某个 post population 的连接，例如：

```text
pop0 -> pop0 AMPA
pop0 -> pop1 AMPA
pop1 -> pop0 GABA
pop1 -> pop1 GABA
```

以 `main_Chen_and_Gong_2019.m` 为例，网络是二维 grid 上的 E/I population：

```matlab
Grid = sparse(gsize,gsize);
Grid(2:2:gsize,2:2:gsize) = true; % 0 Exc, 1 Inh

N(1) = 3/4*gsize^2;
N(2) = 1/4*gsize^2;
```

当前 Python preprocessing 已将这一逻辑扩展成矩形 `H x W`，但 C++ 原版只看到 population size 和连接矩阵，不直接知道 grid 形状。

## 神经元模型是什么

神经元类是：

```text
NeuroPop
```

定义在：

```text
/home/hanruoshui/SpikeNet/cpp_sources/NeuroPop.h
```

实现主要在：

```text
/home/hanruoshui/SpikeNet/cpp_sources/NeuroPop.cpp
```

默认模型是：

```text
LIF: leaky integrate-and-fire
```

支持：

```text
ELIF: exponential leaky integrate-and-fire
```

主要参数：

```text
Cm: membrane capacitance
tau_ref: refractory time
V_rt: reset potential
V_lk: leak reversal potential
V_th: threshold
g_lk: leak conductance
V_ext: external conductance reversal potential
```

默认参数在 Matlab `writePopParaHDF5.m` 注释里也有：

```text
Cm = 0.25
tau_ref = 2.0
V_rt = -60.0
V_lk = -70.0
V_th = -50.0
g_lk = 0.0167
V_ext = 0.0
```

实际可通过：

```text
/config/pops/pop*/PARA001/para_str_ascii
```

覆盖。

膜电位更新在：

```cpp
NeuroPop::update_V(...)
```

核心公式是：

```cpp
I_input = I_AMPA + I_GABA + I_NMDA + I_GJ + I_ext + I_K
```

如果是 LIF：

```cpp
I_leak = -g_lk * (V - V_lk)
```

如果是 ELIF：

```cpp
I_leak = -g_lk * (V - V_lk)
       + g_lk * delT * exp((V - V_T) / delT)
```

然后 Euler 更新：

```cpp
Vdot = (I_leak + I_input) / Cm
V += Vdot * dt
```

只更新非 refractory neuron：

```cpp
if (ref_step_left[i] == 0) update V
```

spike 判定在：

```cpp
NeuroPop::update_spikes(...)
```

如果：

```cpp
ref_step_left[i] == 0 && V[i] >= V_th
```

则：

```text
记录 spike
V reset 到 V_rt
进入 refractory
```

## 外部输入有哪些

SpikeNet 支持几类外部输入。

### Gaussian external current

```text
/config/pops/pop*/INIT004
/config/pops/pop*/INIT014
/config/pops/pop*/INIT019
```

对应：

```cpp
set_gaussian_I_ext(...)
```

### Gaussian external conductance

```text
/config/pops/pop*/INIT012
/config/pops/pop*/INIT018
```

对应：

```cpp
set_gaussian_g_ext(...)
```

### Current file input

```text
/config/pops/pop*/file_current_input/fname
```

对应：

```cpp
load_file_current_input(...)
```

读取：

```text
/current
/neurons
/frame_rate
/mean_curr
/start_step
/end_step
```

每步把 current 加到 `I_ext`：

```cpp
I_ext[i] += mean_curr * current[current_ind][i]
```

### Spike file input

```text
/config/pops/pop*/file_spike_input/fname
```

对应：

```cpp
load_file_spike_input(...)
```

读取：

```text
/x
/y
/t
/max_x
/max_y
```

将事件映射为：

```cpp
neuron_id = x + max_x * y
```

### External Poisson synapse population

```text
/config/syns/syn*/INIT005
/config/syns/syn*/INIT017
```

这不是直接进入 `NeuroPop::I_ext`，而是作为 `ChemSyn` 的外部 noisy pre-population 注入。

## 连接方式是怎样的

连接由 preprocessing 直接写成稀疏边表：

```text
I: pre neuron ids
J: post neuron ids
K: weights
D: delays
```

C++ 读取后在：

```cpp
ChemSyn::init(...)
```

中转成按 pre neuron 分组的结构：

```cpp
C[i_pre].push_back(j_post)
K[i_pre].push_back(K_ij)
D[i_pre].push_back(round(D_ij / dt))
```

所以内部结构是：

```text
对每个 pre neuron i：
    C[i] = 所有 post neuron j
    K[i] = 对应权重
    D[i] = 对应延迟步数
```

连接本身怎么生成，不在 C++ simulator 里决定，而是在 Matlab/Python preprocessing 里决定。

例如 Chen/Gong 2019 是距离连接：

```matlab
A = (DM <= Drange) .* W .* exp(-DM.^2 / sigma)
```

GU 2018 则有更复杂的 degree / lattice / inverse pool 权重分配。C++ 不关心这些来源，只接收最终的 `I/J/K/D`。

## 突触模型是什么

突触类是：

```text
ChemSyn
```

定义在：

```text
/home/hanruoshui/SpikeNet/cpp_sources/ChemSyn.h
```

实现主要在：

```text
/home/hanruoshui/SpikeNet/cpp_sources/ChemSyn.cpp
```

支持三类化学突触：

```text
0 = AMPA
1 = GABA
2 = NMDA
```

Matlab 侧通常传：

```text
1 = AMPA
2 = GABA
3 = NMDA
```

写入 HDF5 时减 1。

突触是 conductance-based。电流公式在：

```cpp
ChemSyn::calc_I()
```

AMPA：

```cpp
I[j] = -gs_sum[j] * (V_post[j] - V_ex)
```

GABA：

```cpp
I[j] = -gs_sum[j] * (V_post[j] - V_in)
```

NMDA：

```cpp
I[j] = -gs_sum[j] * B(V_post[j]) * (V_post[j] - V_ex)
```

NMDA 有电压依赖 Mg block：

```cpp
B = 1 / (1 + miuMg_NMDA * exp(-gamma_NMDA * V))
```

突触动力学有两个模型：

```text
synapse_model = 0 默认，Gu/Gong 2016 风格
synapse_model = 1 Keane/Gong 2015 风格
```

模型 0 里，每个 pre spike 会开启 transmitter release，经过 delay 后写入 ring buffer：

```cpp
d_gs_sum_buffer[t_ring][j_post] += K_trans[i_pre] * (1 - s[i_pre]) * K[i_pre][syn_ind]
```

然后每步：

```cpp
gs_sum[j] += buffer[t][j]
gs_sum[j] *= exp(-dt / tau_decay)
```

模型 1 里用 rise/decay 两个变量：

```cpp
gs_sum = (gs_decay_sum - gs_rise_sum) / (tau_decay - tau_rise)
```

## 前向传播是怎么做的

这里的“前向传播”不是深度学习里的矩阵层前向，而是 spiking network 的时间步动力学。

主循环在：

```cpp
SimuInterface::simulate()
```

每个时间步调用：

```cpp
network.update(step_current)
```

`NeuroNet::update(...)` 的顺序是：

### 1. 每个 population 先更新 spikes

```cpp
NeuroPopArray[pop]->update_spikes(step_current)
```

这一步根据上一时刻的膜电位判断是否发放 spike，同时清空上一时刻的突触/外部电流缓存。

### 2. 每个 synapse block 读取 pre/post population 当前数据

```cpp
ChemSyn::recv_pop_data(NeuroPopArray)
```

读取：

```text
spikes_pre
spikes_post
V_post
```

### 3. 每个 synapse block 更新突触状态和 post current

```cpp
ChemSyn::update(step_current)
```

内部包括：

```text
STD / SP / inhibitory STDP
update_gs_sum
calc_I
sample / stats
```

### 4. synapse block 把电流发回 post population

```cpp
ChemSyn::send_pop_data(NeuroPopArray)
```

内部调用：

```cpp
NeuroPop::recv_I(I, pop_ind_pre, syn_type)
```

把 `I` 加到 post population 的：

```text
I_AMPA / I_GABA / I_NMDA
```

### 5. 可选 JH learning 更新

如果配置了 JH learning，会执行一组在线学习/可塑性更新。

### 6. 每个 population 更新膜电位

```cpp
NeuroPopArray[pop]->update_V(step_current)
```

这一步加入：

```text
I_AMPA
I_GABA
I_NMDA
I_ext
I_K
I_leak
```

并用 Euler 更新 `V`。

所以一个时间步的数据流是：

```text
V(t)
-> spike(t)
-> synapse conductance/current(t)
-> I_input(t)
-> V(t+dt)
```

## 反向传播是怎么做的

严格说，SpikeNet 没有深度学习意义上的 backpropagation。

没有：

```text
loss
gradient
autograd
反向传播穿过层
optimizer.step()
```

它是生物物理 SNN 仿真器，主要是手写动力学更新。

但它有几类在线突触可塑性，容易被误认为“反向传播”：

### 1. Short-term depression, STD

```cpp
ChemSyn::update_STD(...)
```

pre spike 时降低 vesicle availability：

```cpp
f_ves *= 1 - p_ves
```

然后逐步恢复到 1。

### 2. Synaptic plasticity, SP

```cpp
ChemSyn::update_SP(...)
```

更新 `u` 和 `x`，影响 `K_trans`。

### 3. Inhibitory STDP

```cpp
ChemSyn::update_inh_STDP(...)
```

维护 pre/post trace：

```cpp
x_trace_pre
x_trace_post
```

根据 spike timing 修改 inhibitory weights `K`。

### 4. JH Learning

在 `NeuroNet::update(...)` 中有一段 JH learning scheme：

```text
record_V_post_JH_Learn
update_post_spike_hist_JH_Learn
new_post_spikes_JH_Learn
old_pre_spikes
get_rhat_spiking
wchange_non_Hebbian_outgoing
update_Vint_JH_Learn
wchange_Hebbian_outgoing
new_pre_spikes_JH_Learn
get_all_rhat_JHLearn
```

这是一套在线学习/可塑性规则，不是 autograd backprop。

所以如果用深度学习术语类比：

```text
forward = 时间步动力学仿真
backward = 没有标准反向传播
learning = 局部突触可塑性规则
```

## 参数模型汇总

神经元参数：

```text
Cm
tau_ref
V_rt
V_lk
V_th
g_lk
V_ext
ELIF_delT
ELIF_VT
SFA: dg_K, tau_K, V_K
```

突触参数：

```text
V_ex
V_in
Dt_trans_AMPA
Dt_trans_GABA
Dt_trans_NMDA
tau_decay_AMPA
tau_decay_GABA
tau_decay_NMDA
tau_rise / tau_decay for model 1
NMDA Mg block parameters
STD parameters
STDP parameters
JH learning parameters
```

连接参数：

```text
I, J
K
D
syn_type
pop_ind_pre
pop_ind_post
```

仿真参数：

```text
N
dt
step_tot
random seeds
sampling windows
runaway killer
```

## 对当前 translation 工作的意义

当前 Python translation 实际是在替代 SpikeNet 的 Matlab 用户接口层：

```text
Matlab preprocessing -> Python preprocessing
Matlab post-processing -> Python post-processing
```

但 C++ simulator 的核心逻辑仍然是：

```text
*_in.h5
-> SimuInterface.import_HDF5
-> NeuroNet.update loop
-> *_out.h5
```

后续最需要确认的是输入接口：

```text
如果 simulator 用 C++ 原版：
    event x/y/t 应该更接近 file_spike_input
    current matrix 应该走 file_current_input

如果 Python simulator 已改接口：
    需要以 Python simulator 的 reader 为准
```

也就是说，网络结构和动力学本身已经清楚，但 “attention 前 lif_spk 到 simulator population 的映射” 必须和 simulator 实际读取代码保持一致。

## GU 风格 Preprocessing 逻辑

这里的 GU 风格指 SpikeNet 中的：

```text
models/main_GU_et_al_2018.m
```

它也是一个 Matlab preprocessing 入口，最终目标同样是生成 C++ simulator 能读取的 `*_in.h5`。它和当前 Python 已实现的 Chen/Gong 2019 路径不一样：Chen/Gong 2019 主要用二维网格距离规则直接生成四类 E/I 连接；GU 2018 则先生成带目标 degree 分布和空间结构的 E/E 网络，再根据 E/E 的入权重去约束 I/E 权重。

### 输入和输出

GU preprocessing 的输入不是 event 文件本身，而是一组网络和仿真参数：

```text
dt
step_tot
N_e
N_i
tau_ref
delay_max
dg_K
external spike rate
connection probability matrix
degree distribution parameters
weight distribution parameters
sampling settings
```

输出仍然是一个 SpikeNet input HDF5：

```text
*_in.h5
```

主要写入：

```text
/config/Net/INIT001/N
/config/Net/INIT002/dt
/config/Net/INIT002/step_tot
/config/pops/pop*/PARA001/...
/config/pops/pop*/SFA/...
/config/syns/PARA002/...
/config/syns/syn*/INIT006/type
/config/syns/syn*/INIT006/i_pre
/config/syns/syn*/INIT006/j_post
/config/syns/syn*/INIT006/I
/config/syns/syn*/INIT006/J
/config/syns/syn*/INIT006/K
/config/syns/syn*/INIT006/D
```

在原 Matlab 代码里，调用 helper 时使用 Matlab 1-based 索引；helper 写入 HDF5 前会统一减 1。现在 Python translation 的约定已经改成 0-based，所以如果后续实现 GU builder，`pop0/pop1`、`I/J`、`syn_type` 应该直接按 0-based 写入 HDF5。

### 网络规模

原始 GU 2018 默认：

```text
hw = 31
N_e = (2*hw + 1)^2 = 3969
N_i = 1000
N = [3969, 1000]
```

E population 被放在一个二维正方形 lattice 上，边长是：

```text
2*hw + 1 = 63
```

I population 不是规则网格上的完整 population，而是通过：

```text
quasi_lattice_2D(N_i, hw)
```

随机放置到和 E population 相同空间尺度的 quasi-lattice 上。也就是说 GU 2018 的空间结构是一个约 `63x63` 的神经元空间，不是当前 MOT 输入的 `250x400` 像素网格。

### Population 和神经元模型设置

GU 2018 只有两个 population：

```text
pop0: excitatory, N_e = 3969
pop1: inhibitory, N_i = 1000
```

Matlab 原代码：

```matlab
N = [N_e, N_i];
Type_mat = ones(Num_pop);
Type_mat(end, :) = 2;
```

含义是：

```text
pre population 是 E 时，synapse type = AMPA
pre population 是 I 时，synapse type = GABA
```

在 Python/C++ 0-based 约定下应写成：

```text
E pre: type = 0
I pre: type = 1
```

神经元参数方面，GU 入口只显式写了：

```text
tau_ref = 4 ms
```

其他膜电位参数使用 C++ `NeuroPop` 的默认参数。GU 还开启了 excitatory population 的 spike frequency adaptation：

```matlab
writeSpikeFreqAdptHDF5(FID, 1, dg_K);
```

其含义是给 E population 加 SFA/K-current 相关参数：

```text
dg_K = 0.01 uS
```

### 外部输入

GU 2018 使用的是外部 Poisson spike 输入，而不是当前 translation 中 attention/pre-attention event HDF5 文件输入。

原 Matlab 逻辑：

```matlab
rate_ext_I = 1*10^3;
rate_ext_E = 0.85*10^3;
g_ext = 2*10^-3;
writeExtSpikeSettingsHDF5(FID, 1, 1, g_ext, 1, rate_ext_E*ones(1, step_tot), ones(1, N(1)));
writeExtSpikeSettingsHDF5(FID, 2, 1, g_ext, 1, rate_ext_I*ones(1, step_tot), ones(1, N(2)));
```

注意变量名有点容易误读：第一行写给 population 1，也就是 E population；第二行写给 population 2，也就是 I population。按当前 0-based Python 约定，应是：

```text
pop0 receives external AMPA-like spike input at rate_ext_E
pop1 receives external AMPA-like spike input at rate_ext_I
```

这和我们现在接 snnTracker attention 前输入的目标不同。当前目标是把 `lif_spk` 或 event HDF5 作为 simulator 输入，因此更接近 `file_spike_input` 或定制 reader，而不是 GU 默认的 homogeneous external Poisson drive。

### E/E 连接生成

GU 的核心复杂度在 E/E 网络。流程是：

1. 先设定 E/E 的平均连接概率：

```matlab
P_mat = [0.16 0.2; 0.2 0.4];
deg_mean = N_e * P_mat(1,1);
```

2. 生成 E population 的入度和出度目标分布：

```matlab
q = 0.4;
degree_CV = 0.2;
in_out_r = 0.13;
[deg_in_0, deg_out_0] = hybrid_degree(N_e, deg_mean, deg_std_logn, in_out_r, q);
```

这里 `hybrid_degree(...)` 的作用是生成带指定均值、变异系数、入/出度相关性的 degree 序列。它不是简单的 Erdős-Rényi 随机图，而是让每个 E neuron 有自己的目标 in-degree/out-degree。

3. 根据空间距离、degree 分布和 common-neighbor 规则生成 E/E 边：

```matlab
a_Gamma = 2;
iter_num = 10;
[I_ee, J_ee, ~, ~, Lattice_E] = generate_IJ_2D(deg_in_0, deg_out_0, tau_c_EE, a_Gamma, iter_num);
```

这里输出的：

```text
I_ee: E/E pre-synaptic neuron ids
J_ee: E/E post-synaptic neuron ids
Lattice_E: E neuron 的二维空间位置
```

逻辑上，它在满足 degree target 的同时，让连接更偏向空间近邻，并通过 common-neighbor 规则让网络形成更真实的局部结构。这个网络拓扑比 Chen/Gong 2019 的“距离阈值 + 权重按距离衰减”复杂得多。

### E/E 权重和 inverse pool

GU 2018 中 inverse pool 只用于 E/E 权重生成。

原始流程：

1. 把目标平均 conductance 转换到 EPSP 参数空间：

```matlab
[fit_g_2_EPSP, ~] = g_EPSP_conversion();
EPSP_mu = fit_g_2_EPSP(g_EE_mu);
EPSP_sigma = 1;
```

2. 根据 EPSP 均值和方差拟合 lognormal 参数：

```matlab
mu_p = log((EPSP_mu^2)/sqrt(EPSP_sigma^2+EPSP_mu^2));
s_p = sqrt(log(EPSP_sigma^2/(EPSP_mu^2)+1));
mu_p = mu_p + s_p^2;
g_pool_generator_hld = @(N)g_pool_generator(N, mu_p, s_p);
```

3. 计算每个 E neuron 的 E/E in-degree：

```matlab
in_degree = full(sum(sparse(I_ee,J_ee,ones(size(I_ee))), 1));
```

4. 设定入权重总量的缩放目标：

```matlab
K_scale = sqrt(in_degree);
```

5. 调用 inverse pool：

```matlab
K_cell = inverse_pool(in_degree, K_scale, g_pool_generator_hld);
```

`inverse_pool(...)` 的作用是：先从一个全局 lognormal 权重池中采样一批候选权重，然后为每个 post-synaptic E neuron 分配一个子池。每个子池的元素个数等于该 neuron 的 E/E in-degree，同时子池权重和满足 `K_scale` 指定的相对比例。

也就是说，inverse pool 不是生成连接拓扑的函数，而是在已有 `I_ee/J_ee` 之后，给每个 post neuron 的 incoming E/E synapses 分配权重。

最后重排为和 `I_ee/J_ee` 对齐的一维 `K_ee`：

```matlab
K_ee = zeros(size(J_ee));
for j = 1:N_e
    K_ee(J_ee==j) = K_cell{j}';
end
```

在当前 Python translation 中，保留的适配函数是：

```python
assign_inverse_pool_weights_from_repo_b(...)
```

它复用仓库 B 的：

```text
network_generator.sampling.weight_assign.inversepool.InversePoolWeightAssign
```

如果以后实现 GU builder，合理结构是：

```text
generate GU E/E topology -> compute in_degree -> assign_inverse_pool_weights_from_repo_b -> write_chemical_connection
```

### I/E、E/I、I/I 连接

E/E 之后，GU 生成 I population 的 quasi-lattice：

```matlab
Lattice_I = quasi_lattice_2D(N(2), hw);
```

然后用 `Lattice2Lattice(...)` 生成另外三类连接：

```matlab
I/E: Lattice_I -> Lattice_E, tau_c_I,  P_mat(2,1)
E/I: Lattice_E -> Lattice_I, tau_c_IE, P_mat(1,2)
I/I: Lattice_I -> Lattice_I, tau_c_I,  P_mat(2,2)
```

其中：

```text
I/E 权重不是常数
E/I 权重是常数 g_IE
I/I 权重是常数 g_II
```

I/E 权重依赖目标 E neuron 已经收到的 E/E 总入权重：

```matlab
in_weight_EE = full(sum(sparse(I_ee,J_ee,K_ee),1));
for ind_E = 1:N(1)
    mu_K_tmp = in_weight_EE(ind_E)/sum(J==ind_E)*(g_EI_mu/g_EE_mu);
    K(J==ind_E) = abs(randn([1 sum(J==ind_E)])*(mu_K_tmp*0.25) + mu_K_tmp);
end
```

含义是：每个 E neuron 收到的 inhibitory input 强度要和它收到的 E/E input 成比例，从而维持 I/E ratio。这个逻辑是 GU 风格和简单距离连接最大的差别之一。

### 写入 HDF5

GU 最终还是调用同一个化学突触写入接口：

```matlab
writeChemicalConnectionHDF5(FID, Type_mat(1,1), 1, 1, I_ee, J_ee, K_ee, D);
writeChemicalConnectionHDF5(FID, Type_mat(2,1), 2, 1, I,    J,    K,    D);
writeChemicalConnectionHDF5(FID, Type_mat(1,2), 1, 2, I,    J,    K,    D);
writeChemicalConnectionHDF5(FID, Type_mat(2,2), 2, 2, I,    J,    K,    D);
```

在 Python 0-based 约定下，对应关系应是：

```text
E/E: type=0, pop_pre=0, pop_post=0
I/E: type=1, pop_pre=1, pop_post=0
E/I: type=0, pop_pre=0, pop_post=1
I/I: type=1, pop_pre=1, pop_post=1
```

`I/J` 也应该已经是 pop-local 0-based neuron ids。C++ simulator 不知道这些连接来自 GU、Chen/Gong 还是其他生成器；它只读取最终的：

```text
type
i_pre
j_post
I
J
K
D
```

### 和当前 object detection 任务的关系

GU 2018 默认网络是一个内生随机网络模型，用外部 Poisson spike drive 驱动；它不是为 event camera object detection 设计的视觉前端。

如果把 GU 风格用于当前任务，需要额外解决两个映射问题：

1. 输入映射：`250x400` event/lif_spk 如何映射到 GU 的 `3969` 个 E neurons。默认 GU lattice 是 `63x63`，不是 `250x400`。
2. 输出映射：simulator 输出的 `pop0` spike/activity 如何映射回图像坐标并读出 bbox。

因此，在当前目标检测链路中：

```text
input events -> preprocessing -> simulator -> postprocessing -> bbox
```

GU 风格可以作为一种网络连接和权重生成方案，但不是直接替代矩形视觉输入接口的完整方案。要真正使用 GU 风格，需要把它改造成矩形 lattice 或增加输入/输出坐标映射层。

### 和 Chen/Gong 2019 路径的对比

Chen/Gong 2019 当前 Python builder 的特点：

```text
E/I population 覆盖输入 grid
支持矩形 grid_shape=(H,W)
连接按距离阈值生成
E pre 的权重按距离衰减
I pre 的权重为常数
不需要 inverse pool
更容易把 event x/y 映射到 neuron grid
```

GU 2018 的特点：

```text
默认 E population 是 63x63 lattice
I population 是 quasi-lattice
E/E 拓扑由 degree + distance + common-neighbor 共同决定
E/E 权重使用 inverse pool
I/E 权重根据每个 E neuron 的 E/E 入权重归一化
默认外部输入是 Poisson spike drive
更像网络动力学复现实验，不是现成视觉输入接口
```

所以，对于当前 snnTracker 目标检测流程，Chen/Gong 2019 风格更容易先打通端到端；GU 风格更适合作为后续“更复杂连接统计和权重分布”的替代网络生成器。
