# 鲁棒离线强化学习中的解耦表征

## 项目概述

本项目研究**离线强化学习（Offline RL）在合成观测扰动下的表征鲁棒性**。核心思路是**特权预训练框架（Privileged Pretraining Framework，PPF）**：在编码器预训练阶段使用干净状态作为监督目标（一种特权信息），推理时仅使用带噪声的观测。下游策略（IQL、TD3+BC 或 BC）在冻结编码器的输出上训练。

实验在三类观测扰动 family 下，比较多种编码器设置与解耦正则方法：

- **concat**：将无关噪声直接拼接到干净观测后
- **project**：拼接后施加随机正交线性混合
- **nonlinear**：拼接后施加两层非线性混合

整个仓库同时面向**课程项目复现**和**会议论文准备**进行组织。训练 notebook 保持可读性，重复使用的工具逻辑（评估、指标保存、冻结编码器后的训练循环）统一放入 `src/`。

---

## 表征方法

### 主实验方法（PPF 框架）

| 方法 | 说明 |
|---|---|
| `true_only` | 上界——直接在干净状态上训练策略 |
| `raw_noisy` | 下界——直接在原始带噪观测上训练策略 |
| `plain` | PPF 编码器，仅含动力学 + 奖励目标，无解耦正则 |
| `disentangled_barlow` | PPF + Barlow Twins 互相关惩罚 |
| `disentangled_cov` | PPF + 协方差白化惩罚 |
| `disentangled_hsic` | PPF + HSIC 独立性准则 |
| `disentangled_dcor` | PPF + 距离相关惩罚 |
| `disentangled_infonce` | PPF + InfoNCE 对比惩罚 |
| `disentangled_l1` | PPF + L1 互相关惩罚 |

### 外部基线

| 方法 | 说明 |
|---|---|
| `pca` | PCA-IQL——将带噪观测投影到前 k 个主成分（无神经网络编码器，无特权信息） |

---

## 消融实验

| 组别 | 说明 |
|---|---|
| **B1 — 去除特权监督** | 移除干净状态监督，编码器改为在带噪下一观测上做预测 |
| **B2 — 仅奖励预训练** | 移除动力学损失，仅保留奖励预测 + 解耦正则 |
| **A — 算法消融** | 将 IQL 替换为 TD3+BC 或 BC，编码器预训练不变 |

---

## 实验环境

- `halfcheetah-medium-v2`
- `hopper-medium-v2`
- `walker2d-medium-v2`
- `ant-medium-v2`

---

## 前置条件

- **操作系统：** 推荐 Linux 或 WSL2；原生 Windows 不支持（MuJoCo / D4RL 依赖要求）。
- **GPU：** 推荐 NVIDIA GPU 以加速训练。
- **Python 环境：** 推荐使用 Conda 或 Docker 保证可复现性。

---

## 环境安装

### 方式一：Docker

```bash
docker build -t robust_offline_rl:latest .
docker run --gpus all -it --rm robust_offline_rl:latest
```

### 方式二：Conda（推荐本地开发使用）

#### 1. 安装系统依赖

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

```bash
conda env create -f environment.yml
conda activate rl_env
```

#### 3. 安装 PyTorch

请安装与本地 CUDA 版本匹配的 PyTorch。CUDA 12.1 示例：

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
│   ├── config.py              # 全局路径常量
│   ├── experiment_config.py   # 读取环境变量覆盖（ENV_NAME、SEED 等）
│   ├── dataset.py             # NoisyOfflineRLDataset
│   ├── encoder.py             # DisentangledEncoder
│   ├── pca_encoder.py         # PCAEncoder（外部基线）
│   ├── iql.py                 # IQLAgent
│   ├── td3bc.py               # TD3BCAgent
│   ├── bc.py                  # BCAgent
│   ├── train_eval.py          # 训练循环 + 评估工具函数
│   ├── utils.py
│   └── visualization.py
├── scripts/
│   ├── run_all.sh                      # 本地执行（编辑 NOTEBOOKS 数组）
│   ├── submit_all.sh                   # Slurm：主 IQL 实验
│   ├── submit_true_only.sh             # Slurm：true_only 基线
│   ├── submit_ablation_reward_only.sh  # Slurm：B2 仅奖励消融
│   ├── submit_ablation_td3bc.sh        # Slurm：算法消融 A（TD3+BC）
│   ├── submit_ablation_bc.sh           # Slurm：算法消融 A（BC）
│   └── submit_ablation_baselines.sh    # Slurm：外部基线（PCA-IQL）
├── notebooks/
│   ├── main/                      # PPF 主实验（IQL）
│   │   ├── exp_true_only.ipynb
│   │   ├── exp_raw_noisy.ipynb
│   │   ├── exp_plain_encoder.ipynb
│   │   ├── exp_disentangled_barlow.ipynb
│   │   ├── exp_disentangled_cov.ipynb
│   │   ├── exp_disentangled_hsic.ipynb
│   │   ├── exp_disentangled_dcor.ipynb
│   │   ├── exp_disentangled_infonce.ipynb
│   │   └── exp_disentangled_l1.ipynb
│   ├── ablation_noisy_target/     # B1：去除特权监督
│   │   ├── exp_plain_encoder_no_priv.ipynb
│   │   ├── exp_disentangled_barlow_no_priv.ipynb
│   │   ├── exp_disentangled_cov_no_priv.ipynb
│   │   ├── exp_disentangled_hsic_no_priv.ipynb
│   │   ├── exp_disentangled_dcor_no_priv.ipynb
│   │   ├── exp_disentangled_infonce_no_priv.ipynb
│   │   └── exp_disentangled_l1_no_priv.ipynb
│   ├── ablation_reward_only/      # B2：去除动力学损失
│   │   ├── exp_plain_encoder_reward_only.ipynb
│   │   ├── exp_disentangled_barlow_reward_only.ipynb
│   │   ├── exp_disentangled_cov_reward_only.ipynb
│   │   ├── exp_disentangled_hsic_reward_only.ipynb
│   │   ├── exp_disentangled_dcor_reward_only.ipynb
│   │   ├── exp_disentangled_infonce_reward_only.ipynb
│   │   └── exp_disentangled_l1_reward_only.ipynb
│   ├── ablation_td3bc/            # 消融 A：TD3+BC 策略
│   │   ├── exp_true_only_td3bc.ipynb
│   │   ├── exp_raw_noisy_td3bc.ipynb
│   │   ├── exp_plain_encoder_td3bc.ipynb
│   │   ├── exp_disentangled_barlow_td3bc.ipynb
│   │   ├── exp_disentangled_cov_td3bc.ipynb
│   │   ├── exp_disentangled_hsic_td3bc.ipynb
│   │   ├── exp_disentangled_dcor_td3bc.ipynb
│   │   ├── exp_disentangled_infonce_td3bc.ipynb
│   │   └── exp_disentangled_l1_td3bc.ipynb
│   ├── ablation_bc/               # 消融 A：BC 策略
│   │   ├── exp_true_only_bc.ipynb
│   │   ├── exp_raw_noisy_bc.ipynb
│   │   ├── exp_plain_encoder_bc.ipynb
│   │   ├── exp_disentangled_barlow_bc.ipynb
│   │   ├── exp_disentangled_cov_bc.ipynb
│   │   ├── exp_disentangled_hsic_bc.ipynb
│   │   ├── exp_disentangled_dcor_bc.ipynb
│   │   ├── exp_disentangled_infonce_bc.ipynb
│   │   └── exp_disentangled_l1_bc.ipynb
│   ├── baselines/                 # 外部基线
│   │   └── exp_pca_iql.ipynb
│   └── analysis/                  # 汇总分析与可视化
│       ├── eval_all.ipynb
│       ├── eval_no_priv_ablation.ipynb
│       ├── select_best_methods.ipynb
│       ├── exp_lambda_sensitivity.ipynb
│       └── visualization.ipynb
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

所有输出按照方法、环境、噪声配置和 seed 组织成统一的目录层级。

### 含噪声 sweep 的方法（除 `true_only` 外均含）

```text
artifacts/
├── checkpoints/
│   └── <method>/<env_name>/<noise_tag>/seed_<n>/
│       ├── encoder_epoch_50.pth   # 仅 PPF 方法
│       ├── iql_epoch_10.pth
│       └── iql_epoch_100.pth
└── obs_stats/
    └── <method>/<env_name>/<noise_tag>/seed_<n>/
        └── obs_stats.npz

results/
└── raw_metrics/
    └── <method>/<env_name>/<noise_tag>/seed_<n>/
        └── metrics.json
```

典型的 `noise_tag` 形如 `nd11_ns2p0_nonlinear`（noise_dim=11，noise_scale=2.0，noise_type=nonlinear）。

### `true_only` 基线

`true_only` 不参与噪声 sweep，因此没有 `noise_tag` 这一层：

```text
artifacts/checkpoints/true_only/<env_name>/seed_<n>/
results/raw_metrics/true_only/<env_name>/seed_<n>/metrics.json
```

---

## 如何运行实验

### 本地运行

编辑 `run_all.sh` 中的 `NOTEBOOKS` 数组，选择要运行的 notebook，然后：

```bash
bash scripts/run_all.sh
```

运行时覆盖实验变量：

```bash
ENV_NAME=halfcheetah-medium-v2 SEED=1 NOISE_DIM=11 NOISE_SCALE=2.0 NOISE_TYPE=nonlinear \
bash scripts/run_all.sh
```

### 集群运行（Slurm）

每个脚本提交完整的 job 网格（seeds × noise dims × noise scales × noise types）。在提交前修改脚本顶部的 `ENV_NAME` 变量。

| 脚本 | 用途 |
|---|---|
| `submit_all.sh` | 主 IQL 实验（PPF 方法） |
| `submit_true_only.sh` | `true_only` 基线（仅 seed sweep） |
| `submit_ablation_reward_only.sh` | B2：仅奖励预训练消融 |
| `submit_ablation_td3bc.sh` | 消融 A：TD3+BC 策略 |
| `submit_ablation_bc.sh` | 消融 A：BC 策略 |
| `submit_ablation_baselines.sh` | 外部基线（PCA-IQL） |

```bash
bash scripts/submit_all.sh
```

### 交互式 notebook 运行

```bash
jupyter lab
```

打开 `notebooks/` 对应子目录下的 notebook。

---

## 推荐工作流

1. 运行 `notebooks/main/` 中的主实验（目标环境 × 噪声配置）。
2. 运行各消融组（`ablation_noisy_target/`、`ablation_reward_only/`、`ablation_td3bc/`、`ablation_bc/`）和外部基线（`baselines/`）。
3. 使用 `notebooks/analysis/eval_all.ipynb` 汇总 `results/raw_metrics/`。
4. 将图保存到 `results/figures/`，将汇总表保存到 `results/tables/`。

---

## 可复现性说明

- 训练 seed 通过 `src/experiment_config.py` 读取 `SEED` 环境变量控制，可在运行时覆盖。
- `artifacts/checkpoints/` 通常不纳入 Git 跟踪；`results/raw_metrics/` 是后续聚合分析的主要数据来源。
- PCA 主成分保存为 `.npz` 文件，存放于 `artifacts/checkpoints/pca/`，保证结果可精确复现。

---

## 实验规划

### 阶段一

重点：单环境、核心方法对比、基础消融。

- **环境：** `halfcheetah-medium-v2`
- **噪声类型：** nonlinear
- **噪声网格：** dims ∈ {3, 6, 8, 11}，scales ∈ {0.5, 1.0, 1.5, 2.0}
- **Seeds：** 3

**实验内容：**
- [ ] 主 IQL 实验——全部 9 种方法（`true_only`、`raw_noisy`、`plain`、6 种解耦变体）
- [ ] 消融 B1——去除特权监督（`*_no_priv`）
- [ ] 消融 B2——仅奖励预训练（`*_reward_only`）
- [ ] 分析——汇总、维度 sweep 图、方法对比表

---

### 阶段二

重点：多环境泛化、算法鲁棒性、外部基线。

- **环境：** `halfcheetah-medium-v2`、`hopper-medium-v2`、`walker2d-medium-v2`、`ant-medium-v2`
- **噪声类型：** nonlinear
- **噪声网格：** 与阶段一相同
- **Seeds：** 5

**实验内容：**
- [ ] 主 IQL 实验——全环境 × 全方法
- [ ] 消融 A——算法消融：TD3+BC 和 BC 策略
- [ ] 消融 B1 + B2——扩展至全环境
- [ ] 外部基线——PCA-IQL（全环境）
- [ ] 分析——lambda 敏感性分析、跨环境汇总表、publication-ready 图表

---

## 引用 / 致谢

本仓库使用 D4RL locomotion 数据集和基于 IQL 的 offline RL 管线作为实验基础。如果你基于本代码继续开展研究，请按需引用相关上游库和 benchmark 论文（D4RL、IQL、TD3+BC）。
