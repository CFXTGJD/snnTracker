任务说明（重写版）
你现在有 4 个仓库：

仓库A（当前工作目录）：snnTracker

仓库B（同级）：network_generator

仓库C（同级）：SpikeNet

仓库D（同级）：btorch

总目标
在不修改仓库B仓库C仓库D的前提下，只在仓库A内完成从 SpikeNet Matlab 到 Python 的迁移代码，实现：

snnTracker 中 attention 层的输入（已由 snnTracker/spkData/export_hdf5_events.py 转换）不再进入 attention，而是进入 SpikeNet 风格预处理流程并喂给 C++ simulator（Python 端接口准备即可，C++ simulator 的 Python 翻译由其他协作者负责）。

全流程仅保留 GU 风格：

使用 inverse pool 机制；

保留对应的 preprocessing / postprocessing / debug 代码；

删除 Chen/Gong 路线相关翻译产物。

强约束
严禁修改仓库B、仓库C、仓库D代码（只读参考）。

只允许修改仓库A（snnTracker），并将新代码放在 snnTracker/translation/。

停止使用 main_Chen_and_Gong_2019.m 风格：

删除此前由其翻译产生的 Python 文件；

删除对应 debug 文件；

不再引入该风格的逻辑分支。

全部采用 GU 风格：

对齐构图/配权/写入流程；

优先复用仓库B中 inverse pool 的 Python 实现（接口调用，不复制改写）。

需要交付的内容
A. 代码迁移与实现（仓库A）
在 translation/ 下新增或重构如下能力（文件名可合理设计）：

Preprocessing（GU风格）

生成 SpikeNet-compatible *_in.h5 配置；

对接 export_hdf5_events.py 产物；

调用仓库B inverse pool Python 接口进行权重分配；

输出可被“翻译后的 Python C++ simulator 接口层”读取的数据格式。

Postprocessing

实现最小可用的 out.h5 解析；

输出可直接接入仓库D（btorch）的结构化结果；

保留必要 debug 可视化/统计信息（仅 GU 风格相关）。

Adapter（A↔B↔D）

A 调用 B（inverse pool）接口适配层；

A 输出到 D（btorch）输入适配层；

明确输入输出 schema（字段、shape、dtype、单位、时间维语义）。

与仓库D（btorch）对接要求
预处理输出必须可接入 btorch（给出接口函数和数据格式）。

后处理输出（输入？）必须可接入 btorch（给出接口函数和数据格式）。

若 btorch 需要特定张量组织方式（如 [T,H,W]、事件列表、batch 结构），需在 adapter 中显式转换并文档化。

代码审查与验收标准
1) 逻辑一致性审查（Matlab ↔ Python）
检查 Python 与对应 Matlab（GU风格）在算法逻辑上的等价性：

参数含义是否一致；

索引语义（Matlab 1-based vs Python 0-based）是否正确；

时间步定义（dt, step_tot）是否一致；

inverse pool 输入输出统计约束是否一致。

给出逐模块映射表（Matlab函数 → Python函数）。

2) 运行与调试审查
验证“输入→预处理→（模拟器接口）→后处理→btorch接入”可完整跑通；

检查关键输出是否正确（shape、数值范围、必要统计）；

验证可接入“翻译后的 Python C++ simulator 代码”；

提供最小可复现实验（命令、配置、期望输出）。

文档要求（必须详细）
请提供完整文档，至少包括：

使用说明（step-by-step）

环境依赖、安装步骤、运行命令；

从原始输入到最终输出的完整命令链。

代码解释

每个核心模块的职责；

关键算法说明（尤其 inverse pool 调用链）。

接口清单（必须完整）

与 network_generator 的接口（函数名、参数、返回值、异常）；

与 btorch 的接口（函数名、参数、返回值、异常）；

输入/输出数据格式（字段、shape、dtype、单位）。

调试说明

常见错误与定位方式；

如何判断输出正确；

如何验证与 Matlab 逻辑一致。