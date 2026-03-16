其他语言阅读：[English](README.md) | [简体中文](README_zh.md)

# 强化学习期末项目：隐式Q学习与特征解耦

## 简介

本项目主要探索和评估了在离线强化学习（Offline RL）场景下，**隐式Q学习 (Implicit Q-Learning, IQL)** 算法结合各种**特征解耦 (Representation Disentanglement)** 方法的表现。通过对比不同的对比学习和正则化手段（如 Barlow Twins, HSIC, InfoNCE, 距离相关系数 dCor 等），我们在包含噪声的复杂状态空间中验证了编码器 (Encoder) 的鲁棒性与特征提取能力。

---

## 系统要求

- **操作系统:** Linux (Ubuntu 20.04/22.04) 或 Windows Subsystem for Linux (WSL2)。
  > *强烈建议不要在纯 Windows 下运行，因为 `mujoco-py` 等物理引擎库在 Windows 上极难配置编译环境。*
- **显卡:** 推荐使用 NVIDIA GPU 以加速训练。
- **复现方式:** 提供 Docker 容器和 Conda 本地环境两种快速配置方式。

---

## 安装指南

为了兼顾本地开发与跨设备的代码复现，我们提供两种环境配置方式：

### 选项 1：使用 Docker
如果您在计算集群上运行，或者没有 `sudo` 权限来安装系统级 C++ 依赖，我们强烈建议使用 Docker 获得一键开箱即用的复现环境：

```bash
# 1. 构建 Docker 镜像
# 【方案 A】直接加载预打包镜像 (如果您已下载本项目的 iql_project.tar)
sudo docker load -i iql_project.tar

# 【方案 B】从 Dockerfile 重新构建镜像 (约需 20 分钟)
sudo docker build -t iql_project:latest .

# 2. 运行容器并挂载 GPU
# 自动映射宿主机 GPU 和当前项目目录
sudo docker run --gpus all -it --rm -v "$(pwd)":/app iql_project:latest /bin/bash

# 此时您已经进入了包含所有代码和正确环境的容器中，可以直接运行训练脚本或启动 Jupyter
```

---

### 选项 2：算力集群部署 (HPC Cluster Deployment) 🌌



---

### 选项 3：使用 Conda (适合本地开发)

**步骤 3.1: 安装系统底层依赖**
由于本项目使用了 `mujoco-py` 和 `d4rl`，必须先在系统中安装必要的 C++ 编译环境和 OpenGL 图形库，否则后续安装会报错。

```bash
sudo apt-get update -q
sudo apt-get install -y build-essential libosmesa6-dev libgl1-mesa-glx libglfw3 libglfw3-dev patchelf
```

**步骤 3.2: 创建 Conda 环境**
我们提供了一个精简版的依赖文件 `environment_clean.yml` 以最大程度确保跨设备兼容性。

```bash
conda env create -f environment_clean.yml
conda activate rl_env
```

**步骤 3.3: 安装 PyTorch 与高危依赖**
请根据您本机的 CUDA 版本安装对应的 PyTorch（本项目默认基于 CUDA 12.1 测试）。

```bash
# 安装 PyTorch
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 安装 D4RL (从源码安装以防报错)
pip install git+[https://github.com/Farama-Foundation/d4rl@master#egg=d4rl](https://github.com/Farama-Foundation/d4rl@master#egg=d4rl)
```

---

## 运行代码

### 自动化运行
我们在根目录下提供了执行脚本，您可以直接运行它们来进行一键本地训练或提交集群任务：

```bash
# 运行所有本地实验
bash run_all.sh

# 提交集群任务
bash submit_all.sh
```

### 交互式实验
本项目按不同的解耦方法（如 Barlow Twins, HSIC 等）将实验拆分成了多个 Jupyter Notebook。启动 Jupyter 查看详细的对比与可视化分析：

```bash
jupyter lab
```
在左侧目录中，您可以打开类似 `exp_disentangled_barlow.ipynb` 或 `exp_raw_noisy.ipynb` 的文件进行单独运行和调试。

---

## 核心目录结构

```text
FINAL_PROJECT/
├── src/                            # 核心源代码
│   ├── dataset.py                  # D4RL 数据集加载与预处理
│   ├── encoder.py                  # 各类特征提取与解耦编码器
│   ├── iql.py                      # 隐式Q学习核心算法实现
│   ├── train_eval.py               # 训练与评估主循环
│   └── vis.py                      # 可视化与绘图工具
├── results/                        # 实验结果与模型权重输出目录
├── logs/                           # TensorBoard 等训练日志
├── exp_*.ipynb                     # 针对不同解耦方法的对比实验 Notebook
├── run_all.sh                      # 本地自动化批量运行脚本
├── submit_all.sh                   # 计算集群任务提交脚本
├── environment_clean.yml           # 精简版跨平台环境依赖
├── Dockerfile                      # 容器化部署文件 (用于一键复现)
├── README.md                       # 英文版项目说明
└── README_zh.md                    # 中文版项目说明 (本文档)
```