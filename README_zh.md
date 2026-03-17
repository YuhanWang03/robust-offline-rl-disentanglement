# 鲁棒离线强化学习中的解耦表征

## 项目概述

本项目研究**离线强化学习（offline reinforcement learning）在合成观测扰动下的表征鲁棒性**。实验基于 **IQL** 管线，在三类观测扰动 family 下比较多种编码器设置与解耦正则方法：

- **concat**：将无关噪声直接拼接到干净观测后
- **project**：在拼接后施加线性混合
- **nonlinear**：在拼接后施加非线性混合

整个仓库同时面向**课程项目复现**和 **GitHub 项目展示**进行组织。训练 notebook 尽量保持可读性，而重复使用的工具逻辑（如评估、保存指标、冻结编码器后的 IQL 训练）统一放入 `src/`。

---

## 当前研究范围

当前项目比较以下表征设置与基线：

- `true_only`
- `raw_noisy`
- `plain`
- `disentangled_cov`
- `disentangled_barlow`
- `disentangled_hsic`
- `disentangled_dcor`
- `disentangled_infonce`
- `disentangled_l1`

主要环境为 D4RL locomotion 任务，例如：

- `halfcheetah-medium-v2`
- `hopper-medium-v2`
- `walker2d-medium-v2`

---

## 前置条件

- **操作系统：** 推荐使用 Linux 或 WSL2。  
  不推荐直接在原生 Windows 环境下运行，因为 MuJoCo / D4RL 相关依赖在 Linux 系环境中更容易配置和维护。
- **GPU：** 推荐使用 NVIDIA GPU 以提升训练速度。
- **Python 环境：** 推荐使用 Conda 环境或 Docker 容器，以保证可复现性。

---

## 环境安装

支持两种配置方式。

### 方式一：Docker

如果你希望使用可复现环境，或者不想手动安装系统级依赖，推荐使用 Docker。

```bash
# 构建镜像
docker build -t robust_offline_rl:latest .

# 使用 GPU 启动容器
docker run --gpus all -it --rm robust_offline_rl:latest
```

进入容器后即可直接运行 notebook 或脚本。

### 方式二：Conda（推荐本地开发使用）

#### 1. 安装系统依赖

这些包通常是 MuJoCo、OpenGL 和 D4RL 相关组件所需要的。

```bash
sudo apt-get update -q
sudo apt-get install -y \
    build-essential \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglfw3-dev \
    patchelf
```

#### 2. 创建 Conda 环境

使用仓库中提供的环境文件：

```bash
conda env create -f environment.yml
conda activate rl_env
```

如果你本地使用的是精简版环境文件，也可以将 `environment.yml` 替换成你自己的环境配置文件。

#### 3. 安装 PyTorch

请安装与你本地 CUDA 版本匹配的 PyTorch。下面是 CUDA 12.1 的示例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4. 安装 D4RL

```bash
pip install "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl"
```

---

## 仓库结构

```text
robust-offline-rl-disentanglement/
├── README.md
├── README_zh.md
├── .gitignore
├── Dockerfile
├── environment.yml
├── docs/
│   ├── proposal.pdf
│   ├── final_report.pdf
│   └── project_overview.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── experiment_config.py
│   ├── dataset.py
│   ├── encoder.py
│   ├── iql.py
│   ├── train_eval.py
│   ├── utils.py
│   └── visualization.py
├── scripts/
│   ├── run_all.sh
│   └── submit_all.sh
├── notebooks/
│   ├── 01_true_only.ipynb
│   ├── 02_raw_noisy.ipynb
│   ├── 03_plain_encoder.ipynb
│   ├── 04_disentangled_cov.ipynb
│   ├── 05_disentangled_barlow.ipynb
│   ├── 06_disentangled_hsic.ipynb
│   ├── 07_disentangled_dcor.ipynb
│   ├── 08_disentangled_infonce.ipynb
│   ├── 09_disentangled_l1.ipynb
│   └── 10_eval_all.ipynb
├── artifacts/
│   ├── checkpoints/
│   ├── executed/
│   ├── obs_stats/
│   └── slurm_jobs/
├── results/
│   ├── figures/
│   ├── tables/
│   └── raw_metrics/
└── logs/
```

---

## 输出目录说明

### noisy / disentangled / plain 方法

对于依赖噪声设定的方法（例如 `disentangled_barlow`、`plain` 或 `raw_noisy`），输出目录组织为：

```text
artifacts/
├── checkpoints/
│   └── <method>/
│       └── <env_name>/
│           └── <noise_tag>/
│               └── seed_<n>/
│                   ├── encoder_epoch_50.pth
│                   ├── iql_epoch_10.pth
│                   ├── ...
│                   └── iql_epoch_100.pth
└── obs_stats/
    └── <method>/
        └── <env_name>/
            └── <noise_tag>/
                └── seed_<n>/
                    └── obs_stats.npz

results/
└── raw_metrics/
    └── <method>/
        └── <env_name>/
            └── <noise_tag>/
                └── seed_<n>/
                    └── metrics.json
```

其中，一个典型的 `noise_tag` 形式如下：

```text
nd5_ns0p5_nonlinear
```

### `true_only` 基线

`true_only` **不参与噪声 sweep**，因此目录中没有 `noise_tag` 这一层。

```text
artifacts/
├── checkpoints/
│   └── true_only/
│       └── <env_name>/
│           └── seed_<n>/
│               ├── iql_epoch_10.pth
│               ├── ...
│               └── iql_epoch_100.pth
└── obs_stats/
    └── true_only/
        └── <env_name>/
            └── seed_<n>/
                └── obs_stats.npz

results/
└── raw_metrics/
    └── true_only/
        └── <env_name>/
            └── seed_<n>/
                └── metrics.json
```

---

## 如何运行实验

### 本地运行

在项目根目录下使用本地脚本：

```bash
bash scripts/run_all.sh
```

也可以在运行时覆盖实验变量：

```bash
KERNEL=python3 \
ENV_NAME=halfcheetah-medium-v2 \
SEED=1 \
NOISE_DIM=5 \
NOISE_SCALE=2.0 \
NOISE_TYPE=nonlinear \
bash scripts/run_all.sh
```

### 集群运行

使用提交脚本：

```bash
bash scripts/submit_all.sh
```

该脚本面向学校算力集群，自动在 `artifacts/slurm_jobs/` 下生成 Slurm 任务脚本。

### 交互式 notebook 运行

在项目根目录启动 Jupyter：

```bash
jupyter lab
```

然后打开 `notebooks/` 目录下的 notebook。

---

## 推荐工作流

一个典型工作流如下：

1. 运行一个或多个训练 notebook。
2. 自动保存 checkpoint、观测统计量和 metrics。
3. 使用 `notebooks/10_eval_all.ipynb` 汇总 `results/raw_metrics/`。
4. 将图保存到 `results/figures/`，将汇总表保存到 `results/tables/`。

---

## 可复现性说明

- 训练 seed 由 `src/experiment_config.py` 和 notebook 层参数共同控制。
- `artifacts/checkpoints/` 主要用于本地训练输出，通常不纳入 Git 跟踪。
- `results/raw_metrics/` 是后续聚合分析和绘图的主要数据来源。
- 在 GitHub / 作品集展示中，通常上传代表性图表和总结表格比上传所有原始 checkpoint 更重要。

---

## 项目当前状态

当前项目已经包含：

- proposal 阶段的问题定义，
- 中期实验 benchmark 管线，
- 已重构的训练 / 评估 notebook，
- 可复现的 checkpoint、metrics 和绘图输出目录结构。

当前的主要研究方向是分析：**在结构化扰动下，解耦方法何时有效、何时无效，以及鲁棒性边界在何处出现。**

---

## 引用 / 致谢

本仓库使用 D4RL locomotion 数据集和基于 IQL 的 offline RL 设定作为实验基础。如果你基于本代码继续开展研究，请同时按需引用相关上游库和 benchmark 论文。
