现在有三个仓库：1.当前工作目录（仓库A）——snnTracker 2.同级仓库（仓库B）——network_generator 3.同级仓库（仓库C）——SpikeNet
你需要读取仓库B中inverse_pool相关的接口和函数，以及仓库C中的matlab代码，在仓库A的translation文件夹里将仓库C的matlab相应代码改写成python，同时注意如果可以的话，调用仓库B中inverse pool的python代码，尽量复用。
！！！绝对不能修改仓库B和C的代码！！！只能在仓库A中修改！！！
仓库C的工作流程见SpikeNet/documentation/work_flow.png。
改写的核心目的是让snnTracker中的attention层接收的输入不再输入到attention层，而是输入到仓库C的c++ simulator里。你不需要完成c++ simulator部分代码的翻译（这个由其他合作者完成），只需要翻译相关的matlab代码为python，把翻译后的代码写成新的python文件，放到snnTracker/translation文件夹中。之前在snnTracker/spkData/export_hdf5_events.py中已经将输入转换成了SpikeNet中writeExtCurrentPopHDF5可以接收的形式。
更详细的解释如下：
1) 图里的 Pre-processing 对应哪些 Matlab 文件？各自作用是什么？
在 SpikeNet 的工作流里，Pre-processing 就是“在 Matlab 侧生成 A-T1_in.h5 输入配置”的阶段。文档里明确是 Matlab 预处理→in.h5→C++ simulator。.

A. 顶层“案例脚本”（你真正会改的入口）
main_demo.m：演示如何把网络参数、连接、输入、采样等写入 in.h5。.

models/main_*.m（如 main_Chen_and_Gong_2019.m）：具体任务脚本，负责定义网络、调用各类 write*HDF5、接刺激文件。
其中 main_Chen_and_Gong_2019.m 已经在用 writeExtCurrentPopHDF5 接入文件刺激。.

B. Matlab I/O 写入函数（matlab_interface/）
这些是 Pre-processing 的“协议层”：

new_ygin_files_and_randseedHDF5.m：创建新 *_in.h5 并设随机种子。.

writeBasicParaHDF5.m：写 N/dt/step_tot。.

writePopParaHDF5.m：写神经元参数串。.

writeChemicalConnectionHDF5.m：写连接索引、权重、延迟。.

writeExtCurrentPopHDF5.m：写外部文件输入路径（你现在要用的接口）。.

C. 外部刺激生成（可选）
ForExtCurrent/ImageProcess_DOG.m：把图像加工成 current 文件（/current, /neurons, /frame_rate ...）。这属于 Pre-processing 的“数据准备”。.

2) 为了最终做 object detection，除了预处理还要把哪些 Matlab 部分转成 Python？
你至少还要转两段（不然流程断在 Matlab）：

A. Post-processing 解析层
图里 PostProcessYG() 负责把 simulator 输出读入并保存成可分析结构。.

Python 对应：实现一个 read_out_h5.py（等价 ReadH5/ReadYG + SaveRYG 的最小子集），把你关心的 spike/activity 序列取出来。

B. More analysis（任务读出层）
图里最后一块是 “More analysis”，这部分在当前仓库主要是神经动力学分析，不是目标检测 bbox 读出。.
所以你需要自己在 Python 里实现：

时间窗积分/聚类/连通域；

目标级实例分离；

bbox 输出（x1,y1,x2,y2）；

与 GT 的 detection 指标计算。

简单说：
Matlab→Python 不仅是写 in.h5，还要把 out.h5 读出 + detection 读出头一起迁移。

3) inverse pool 在图中对应哪个模块？具体功能是什么？
在图中的位置
它属于 Pre-processing（Matlab）内部，不是独立方块。
因为它发生在“构图/配权”阶段，然后结果通过 writeChemicalConnectionHDF5 写入 in.h5 给 C++ 用。.

具体功能
根据每个 post neuron 的入度 K_num 与目标缩放 K_scale，

从全局权重池反向分配每个神经元收到的权重集合，

使“每个神经元权重数量=入度”，并尽量满足目标总输入强度。
核心在 simu_config_tools/inverse_pool.m。.

在 main_GU_et_al_2018.m 中，K_scale = sqrt(in_degree) 后调用 inverse_pool(...)，然后把生成的 K_ee 写入化学连接。.

