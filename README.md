
 计算电磁学的扩散模型 (Diffusion-Based Electromagnetic Inverse Design)


【 1. 项目简介 (Introduction) 】
本项目致力于解决计算电磁学中的“逆向设计 (Inverse Design)”问题。
传统的电磁器件设计通常依赖于经验直觉或耗时的启发式优化算法（如遗传算法、拓扑优化等），需要反复调用正向物理求解器
本项目引入了生成式人工智能中的“条件去噪扩散概率模型”(Conditional DDPM)，实现了“端到端”的物理结构反推。
只需输入期望的二维电磁场分布图，模型即可在数秒钟内从纯噪声中直接生成对应的三维物理结构参数。

【 2. 核心算法与数据流 (Core Algorithm & Data Pipeline) 】
正向物理引擎：基于 PyTorch 加速的 FDTD (时域有限差分) 求解器，支持 CUDA 硬件加速。
逆向扩散模型：
设计条件 (Condition) : 目标二维电磁场强度图 (120x120 像素)。
生成目标 (Target)    : 5 维物理结构参数向量 [中心X, 中心Y, 中心Z, 半径r, 介电常数ε]。
数据处理 (Norm)      : 为了保障扩散模型的数学稳定性，所有物理参数在输入网络前均经过严格的 Min-Max 归一化 (压缩至 [-1, 1])，在逆向采样输出后再进行反归一化还原。
网络架构 (Network)   : 包含 FieldEncoder (特征提取) 和 DenoisingNet (时间步条件 MLP) 的定制化条件扩散模型。

 核心文件架构
项目采用高度解耦的面向对象设计 (OOP)：

├── fdtd_em/
│   ├── config/
│   │   └── generation_config.py        全局控制台：网格、PML、光源、探测器坐标
│   ├── data/
│   │   └── generator.py                调用 FDTD 引擎批量生成双通道物理数据集
│   ├── train/
│   │   ├── dataset.py                  解析 numpy 数据并转化为 Tensor
│   │   ├── model.py                    包含 ConditionalUNet 与 DDPM 核心算法
│   │   └── train.py                    执行扩散模型训练与 Loss 监控
│   └── examples/
│       ├── preview_setup.py            场景预览：快速渲染 3D 物理空间与传感器位置直观
│       └── test_forward_surrogate.py   加载权重，进行正向物理预测与可视化对比
├── run_pipeline.py                     总控开关
└── README.md                        

【 3. 快速运行指南 (Quick Start) 】

全局控制台参数generation_config.py


请在项目根目录下（激活虚拟环境后），按照以下 1 -> 4 的顺序运行脚本：

cd D:\eletromagnetism_FDTD\计算电磁学的扩散模型

▶ 步骤 0：物理环境自检 (Preview Setup)
    可单独运行，不在总控之中

▶ 步骤 1：物理引擎单次测试 (验证 FDTD 求解器与 CUDA 加速)
  命令： python -m fdtd_em.examples.run_field_plane
  说明： 运行单次 120x120x120 网格的 FDTD 仿真，验证本地 GPU 是否正确接管计算，并输出电场强度的极值。

▶ 步骤 2：生成训练数据集 (构建扩散模型的“知识库”)
  命令： python -m fdtd_em.data.generator
  说明： 根据 `config/generation_config.py` 中的 `num_samples` 参数，利用 GPU 批量运行 FDTD 仿真，自动保存生成的物理结构参数 (structure.json) 和对应的场图分布 (field_map.pt)。

▶ 步骤 3：训练条件扩散模型 (核心算法训练)
  命令： python -m fdtd_em.train.train
  说明： 自动读取最新的数据集，将数据归一化后送入 Conditional DDPM 进行加噪与去噪训练。训练结束后，模型权重和配置将自动保存在 `checkpoints` 文件夹中。

▶ 步骤 4：逆向设计推理测试 (验证逆向生成能力)
  命令： python -m fdtd_em.examples.forward_surrogate
  说明： 加载最新训练的扩散模型，抽取一张目标场图作为条件输入。模型将从纯正态分布噪声开始，历经 1000 步去噪，最终输出反归一化后的预测物理参数，并与真实参数(Ground Truth)进行 MAE 误差对比。


【 4. 显存策略与环境 (VRAM Policy & Environment) 】

PowerShell 虚拟环境激活命令：
> cd D:\eletromagnetism_FDTD\计算电磁学的扩散模型
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1

本项目支持两种执行模式，可通过配置文件统一切换，无需修改底层物理代码：

1. LOCAL MODE (本地开发模式)
   目标环境: 个人工作站 / 笔记本 (例如 8GB 显存 GPU)。
   特性: 在分配显存前进行解析性的显存限制检查，防止由于网格过大导致 CUDA Out Of Memory (OOM) 崩溃。适合代码调试、算法验证。

2. SERVER MODE (服务器模式)
   目标环境: HPC 集群 / 大显存服务器 GPU。
   特性: 解除严格的显存限制，允许进行高分辨率的大规模 3D 仿真和海量数据集生成，是训练高精度扩散模型的理想模式。

提示：环境模式的切换不会改变电磁场的物理方程或数值求解逻辑，仅影响显存的安全管理策略。
