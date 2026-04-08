# 中期报告：鲁棒离线强化学习中的解耦表征

**课程项目 · 中期进展报告**
**日期：2026年4月**

---

## 1. 问题定义与研究动机

### 1.1 背景与现实场景

离线强化学习（Offline RL）依赖固定的历史数据集进行策略学习，部署阶段无法与环境交互。然而，实际系统（如自动驾驶、工业机器人）中存在一个普遍的**训练-部署感知差异**：

- **数据收集阶段**：离线数据集往往在高度受控的实验室或仿真环境中采集，此时真实干净状态 $s_{\text{true}}$ 完全可观测并被完整记录。
- **实际部署阶段**：智能体只能访问廉价、受损或高度嘈杂的车载传感器，获得的是受污染的观测 $\tilde{x}_{\text{noisy}}$，而非干净状态。

这一差异使得**在数据收集阶段利用 $s_{\text{true}}$ 作为特权监督信号**成为一种高度现实且自然的选择：花费少量额外标注成本，换取部署阶段更强的观测鲁棒性。

### 1.2 特权预训练框架（PPF）

本项目围绕一个**特权预训练框架（Privileged Pretraining Framework, PPF）**展开研究：

> 在表征预训练阶段，利用数据收集时记录的干净状态 $s_{\text{true}}$ 作为特权预测目标，引导编码器从带噪观测 $\tilde{x}$ 中提取任务语义表征；预训练完成后冻结编码器，在噪声条件下训练和部署 offline RL 策略。

PPF 的核心假设是：$s_{\text{true}}$ 在训练集中**可用但在部署时不可用**。这与现有特权蒸馏（privileged distillation）研究一脉相承，但我们将其引入 offline RL 的解耦表征学习场景，并系统研究不同解耦正则方法在该框架下的有效性边界。

### 1.3 核心研究问题

本项目围绕以下核心问题展开：

> 1. **有效性**：在 PPF 框架下，解耦表征正则化相比无正则编码器和直接使用噪声观测，能带来多大的鲁棒性提升？
> 2. **边界条件**：在何种噪声类型、维度、强度下解耦方法有效，边界在哪里？
> 3. **特权依赖性**：鲁棒性收益主要来自解耦正则本身，还是来自 $s_{\text{true}}$ 提供的特权监督？取消特权信息时方法是否仍然有效？

实验框架基于 **IQL（Implicit Q-Learning）**，通过先预训练编码器、再冻结编码器训练策略的两阶段流程，系统性地对比多种解耦正则方法在不同噪声设定下的性能表现。

---

## 2. 方法框架

### 2.1 观测噪声模型

对干净观测 $s \in \mathbb{R}^{d_s}$，引入 $d_n$ 维无关噪声 $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$，构造三种结构化扰动：

| 噪声类型 | 构造方式 | 描述 |
|----------|----------|------|
| **concat** | $\tilde{x} = [s;\; \epsilon]$ | 直接拼接，噪声维度与信号完全独立 |
| **project** | $\tilde{x} = Q\,[s;\; \epsilon]$，$Q$ 为随机正交矩阵 | 线性混合，信号与噪声维度相互耦合 |
| **nonlinear** | $\tilde{x} = W_2\,\tanh(W_1\,[s;\; \epsilon])$，$W_1, W_2$ 为随机正交矩阵 | 非线性混合，最难分离 |

#### 实验参数：两阶段设计

实验参数的确定分两个阶段进行：

**第一阶段（粗粒度参数扫描）**：采用较大的参数范围探索噪声影响的整体规律。

| 参数 | 取值 |
|------|------|
| 噪声维度 $d_n$ | $\{5,\; 10,\; 20,\; 40,\; 80\}$ |
| 噪声强度 $\sigma$ | $\{0.5,\; 1.0,\; 2.0,\; 4.0,\; 8.0\}$ |

第一阶段的关键发现：concat 和 project 噪声在所有参数范围内几乎不影响性能；nonlinear 噪声在高维（$d_n \geq 20$，halfcheetah obs_dim=17）、高强度（$\sigma \geq 4.0$）时所有方法全面崩溃，且各方法差异已趋于一致，无法区分优劣。因此，非线性噪声的"有效实验区间"集中在 $d_n \leq 17$、$\sigma \leq 2.0$ 范围内。

**第二阶段（精细化扫描，当前主要实验）**：根据第一阶段结果，将 nonlinear 噪声的参数范围收窄至方法差异最显著的区间：

| 参数 | 取值（halfcheetah） | 取值（hopper） |
|------|---------------------|----------------|
| 噪声维度 $d_n$ | $\{4,\; 8,\; 13,\; 17\}$ | $\{2,\; 4,\; 6,\; 8\}$ |
| 噪声强度 $\sigma$ | $\{0.5,\; 1.0,\; 1.5,\; 2.0\}$ | $\{0.5,\; 1.0,\; 1.5,\; 2.0\}$ |

维度上限对齐各环境的真实观测维度（halfcheetah: 17 维，hopper: 11 维），旨在覆盖从"噪声比例较低"到"噪声维度与观测维度相当"的全压力区间。

### 2.2 编码器方法对比

共对比 **9 种设置**：

| 类别 | 方法 | 简称 | 参与后续环境扫描 |
|------|------|------|-----------------|
| 上界基线 | 无噪声观测直接训练 IQL | `true_only` | 是 |
| 下界基线 | 直接使用噪声观测 | `raw_noisy` | 是 |
| 无正则编码器 | MLP 编码器（无解耦约束） | `plain` | 是 |
| 解耦编码器 | Barlow Twins 正则 | `Barlow` | 是 |
| 解耦编码器 | 协方差惩罚 | `Cov` | 否（halfcheetah 已充分评估） |
| 解耦编码器 | HSIC 最小化 | `HSIC` | 是 |
| 解耦编码器 | 距离相关性（dCor） | `dCor` | 是 |
| 解耦编码器 | InfoNCE 对比学习 | `InfoNCE` | 否（halfcheetah 已充分评估） |
| 解耦编码器 | L1 稀疏约束 | `L1` | 否（halfcheetah 已充分评估） |

后续 hopper 和 walker2d 实验仅运行标注"是"的 6 种方法（true_only、raw_noisy、plain、Barlow、HSIC、dCor），理由如下：Cov、InfoNCE、L1 在 halfcheetah 的综合排名中分别位列第 4–6，且在高噪声下的行为模式已经明确；而 HSIC、Barlow、dCor 覆盖了前三名，加上两个基线和 plain，足以在新环境中验证方法排名的跨任务一致性，同时节省约 1/3 的计算开销。

#### 各解耦方法的正则损失

设 $Z = z_{\text{task}} \in \mathbb{R}^{B \times d}$，$\tilde{Z} = z_{\text{irrel}} \in \mathbb{R}^{B \times d}$，$B$ 为 batch size，$d$ 为 latent 维度。六种解耦正则的损失形式如下：

**Barlow Twins**：最小化 $z_{\text{task}}$ 与 $z_{\text{irrel}}$ 的跨分支互相关矩阵偏离单位阵的程度：

$$\mathcal{L}_{\text{BT}} = \sum_i (1 - C_{ii})^2 + \lambda \sum_{i \neq j} C_{ij}^2, \quad C_{ij} = \frac{\sum_b \bar{z}^{\text{task}}_{b,i}\, \bar{z}^{\text{irrel}}_{b,j}}{\left\|\bar{z}^{\text{task}}_{\cdot,i}\right\|_2 \left\|\bar{z}^{\text{irrel}}_{\cdot,j}\right\|_2}$$

其中 $\bar{z}$ 表示 batch 内均值归一化后的向量。对角项迫使两分支对齐，非对角项惩罚跨分支相关。

**协方差惩罚（Cov）**：直接惩罚两分支的跨协方差矩阵的 Frobenius 范数：

$$\mathcal{L}_{\text{Cov}} = \frac{1}{d^2} \left\|\text{Cov}(z_{\text{task}},\, z_{\text{irrel}})\right\|_F^2$$

**HSIC**：使用希尔伯特–施密特独立准则（Hilbert-Schmidt Independence Criterion）度量两分支的核空间依赖性：

$$\mathcal{L}_{\text{HSIC}} = \frac{1}{(B-1)^2} \operatorname{tr}(K_{\text{task}}\, H\, K_{\text{irrel}}\, H)$$

其中 $K_{\text{task}}, K_{\text{irrel}}$ 分别为两分支的 Gram 矩阵（线性核），$H = I - \frac{1}{B}\mathbf{1}\mathbf{1}^\top$ 为中心化矩阵。$\mathcal{L}_{\text{HSIC}} = 0$ 当且仅当两分支统计独立。

**距离相关性（dCor）**：最小化两分支的距离相关系数：

$$\mathcal{L}_{\text{dCor}} = \mathcal{R}^2(z_{\text{task}},\, z_{\text{irrel}}) = \frac{\text{dCov}^2(z_{\text{task}},\, z_{\text{irrel}})}{\sqrt{\text{dVar}(z_{\text{task}}) \cdot \text{dVar}(z_{\text{irrel}})}}$$

其中 dCov 基于 pairwise 距离矩阵的双重中心化计算，$\mathcal{R}^2 = 0$ 当且仅当两变量独立。

**InfoNCE**：将两分支的配对样本视为负样本，通过对比损失最大化两分支的区分度（即最小化互信息）：

$$\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{B}\sum_b \log \frac{\exp(-\text{sim}(z^{\text{task}}_b,\, z^{\text{irrel}}_b) / \tau)}{\sum_{b'} \exp(-\text{sim}(z^{\text{task}}_b,\, z^{\text{irrel}}_{b'}) / \tau)}$$

其中 $\text{sim}(\cdot,\cdot)$ 为余弦相似度，$\tau$ 为温度参数。

**L1 稀疏约束**：对无关分支施加 L1 惩罚，鼓励 $z_{\text{irrel}}$ 稀疏化：

$$\mathcal{L}_{\text{L1}} = \frac{1}{B \cdot d} \left\|z_{\text{irrel}}\right\|_1$$

### 2.3 训练设置

整个训练流程分为两个阶段，编码器与策略解耦训练：

**阶段一：编码器预训练（50 epochs，batch size 512）**

编码器接受带噪观测 $\tilde{x}$ 作为输入，预训练损失由三项组成：

$$\mathcal{L}_{\text{pretrain}} = \mathcal{L}_{\text{next-state}} + \lambda_r \mathcal{L}_{\text{reward}} + \lambda_d \mathcal{L}_{\text{disentangle}}$$

- $\mathcal{L}_{\text{next-state}}$：用任务编码 $z_{\text{task}}$ 和动作 $a$ 预测**下一步干净状态** $s'_{\text{true}}$（MSE 损失）。这是特权信息：训练时已知噪声维度边界，预测目标为纯净观测。
- $\mathcal{L}_{\text{reward}}$：用 $z_{\text{task}}$ 预测标量奖励 $r$（MSE 损失）。
- $\mathcal{L}_{\text{disentangle}}$：不同方法采用不同正则形式（Barlow Twins、HSIC、dCor 等），约束 $z_{\text{task}}$ 与 $z_{\text{irrel}}$ 之间的统计独立性。

编码器采用双分支结构：`task_encoder` 提取任务相关表征 $z_{\text{task}}$，`irrel_encoder` 提取噪声/无关表征 $z_{\text{irrel}}$。隐藏层宽度随输入维度自动缩放（$\max(256, d_{\text{in}} \times 4)$）以保持各噪声设定下的模型容量可比性。

**阶段二：策略训练（100 epochs，batch size 256）**

编码器权重完全冻结，IQL 在编码后特征 $z_{\text{task}}$ 上进行标准 offline RL 训练（Value / Q-network / Actor 三网络结构）。

- **评估指标**：D4RL normalized score，20 episode 均值；跨 3 seeds 报告均值 ± 标准差
- **环境**：`halfcheetah-medium-v2`（主要），`hopper-medium-v2`（次要）

---

## 3. 实验结果

### 3.1 上界基线

| 环境 | true_only 均值 | std |
|------|---------------|-----|
| halfcheetah-medium-v2 | **40.22** | 0.32 |

这是没有任何噪声时的 IQL 性能上界，作为所有对比的参考。

### 3.2 Nonlinear 噪声：主要实验（halfcheetah-medium-v2，seeds 1/2/3）

Nonlinear 是三种噪声中最难应对的，因为噪声通过随机非线性网络与信号完全耦合。

**各方法在不同噪声强度下的性能（dim=4，最低噪声维度）：**

| 方法 | scale=0.5 | scale=1.0 | scale=1.5 | scale=2.0 |
|------|-----------|-----------|-----------|-----------|
| HSIC | 40.03 | 39.90 | 38.60 | 36.87 |
| Barlow | 38.44 | 37.40 | 38.38 | 35.69 |
| dCor | 39.52 | 38.65 | 37.00 | 36.84 |
| Cov | 38.03 | 37.52 | 36.75 | 36.97 |
| InfoNCE | 38.80 | 37.33 | 36.41 | 35.85 |
| L1 | 37.76 | 36.47 | 37.75 | 34.18 |
| Plain | 38.42 | 37.98 | 36.31 | 32.28 |
| Raw Noisy | 36.76 | 34.64 | 34.36 | 29.52 |

**各方法在高噪声维度下的性能（dim=17，最高压力）：**

| 方法 | scale=0.5 | scale=1.0 | scale=1.5 | scale=2.0 |
|------|-----------|-----------|-----------|-----------|
| HSIC | 38.03 | 37.03 | 29.74 | **18.78** |
| Barlow | 38.14 | 36.98 | 29.78 | **20.02** |
| Cov | 38.76 | 35.12 | 28.53 | **15.01** |
| dCor | 37.24 | 36.02 | 28.46 | **18.90** |
| InfoNCE | 37.73 | 33.59 | 27.46 | **16.43** |
| L1 | 37.37 | 32.78 | 26.97 | **14.07** |
| Plain | 37.45 | 35.09 | 26.88 | **14.76** |
| Raw Noisy | 37.31 | 33.30 | 23.59 | **13.19** |

**关键观察**：
- 在 dim=17, scale=2.0 极端条件下，所有方法（包括解耦方法）均出现严重性能崩溃，降幅超过 50%
- HSIC 和 Barlow 在高压力区域仍能保持相对最高的绝对分数
- Plain Encoder 在低噪声时表现接近解耦方法，但高噪声时快速崩溃
- Raw Noisy 始终垫底，说明编码器预训练本身有价值

### 3.3 方法综合排名（halfcheetah-medium-v2，nonlinear）

基于全局均值、高压力条件均值、性能下降率、胜率四个维度的综合评分：

| 排名 | 方法 | 全局均值 | 高压力均值 | 性能下降% | 胜率 | 综合分 |
|------|------|---------|-----------|----------|------|--------|
| 1 | **HSIC** | 35.53 | 28.62 | 24.4% | 75.0% | 1.00 |
| 2 | **Barlow** | 34.73 | 28.01 | 24.8% | 68.8% | 0.83 |
| 3 | dCor | 33.98 | 25.97 | 28.0% | 50.0% | 0.46 |
| 4 | Cov | 33.86 | 24.70 | 30.0% | 56.3% | 0.35 |
| 5 | InfoNCE | 33.62 | 24.75 | 26.3% | 43.8% | 0.32 |
| 6 | L1 | 32.79 | 23.92 | 37.6% | 43.8% | 0.00 |

**HSIC 和 Barlow Twins 是目前表现最稳定的两种解耦方法。**

### 3.4 Concat 噪声：halfcheetah-medium-v2

Concat 噪声仅将无关维度直接拼接，线性编码器理论上可以通过权重置零将其滤除。

**结论**：所有方法（含 raw_noisy）在 concat 噪声下均接近 true_only 上界（~40），差异在 1~2 分以内。解耦正则对 concat 噪声几乎无额外收益，这与理论预期一致。

典型数据（dim=80，scale=4.0，各方法最苛刻条件）：

| 方法 | 分数 |
|------|------|
| Barlow | 31.97 |
| L1 | 28.68（不稳定） |
| 其余方法 | ~39–41 |

注：L1 在 concat 高维时出现严重不稳定，Barlow 在极端维度也有下降，其余方法接近上界。

### 3.5 Project 噪声：halfcheetah-medium-v2

Project 噪声使用随机线性矩阵混合信号与噪声，理论上可被线性解耦。

**结论**：大多数方法表现稳定（~39–41），与 concat 噪声类似。L1 在高维（dim=80, scale=2.0）下出现崩溃（~19），不稳定性更严重。其他解耦方法均保持接近 true_only 的性能。

### 3.6 特权信息消融（No-Privilege Ablation）

研究去除噪声维度标签（即不告知编码器哪些维度是噪声）时的性能损失：

| 方法 | 有特权信息 | 无特权信息 | 性能差距 |
|------|-----------|-----------|---------|
| Plain | 25.18 | 11.36 | 13.83 |
| Barlow | 28.01 | 13.71 | 14.30 |
| HSIC | 28.62 | 13.41 | 15.22 |

**发现**：即使最好的解耦方法（HSIC），在不提供噪声维度标签时也会出现约 15 分的性能下降。说明当前解耦方法对特权信息依赖较强，盲解耦（无监督噪声检测）仍是挑战。

### 3.7 Hopper-medium-v2（部分结果）

对 Hopper 环境（nonlinear 噪声，dim ∈ {2,4,6,8}，seeds 1/2/3）运行了 4 种方法（HSIC, Barlow, Plain, Raw Noisy）：

| 方法 | dim=2 均值 | dim=8，scale=2.0 |
|------|-----------|-----------------|
| HSIC | ~45.4–48.0 | 41.57 |
| Barlow | ~43.4–45.8 | 41.12 |
| Plain | ~45.9–48.0 | 39.29 |
| Raw Noisy | ~46.2–47.1 | 36.21 |

**初步观察**：Hopper 对非线性噪声的鲁棒性整体优于 Halfcheetah，在 dim=8, scale=2.0 时各方法仍保持在 36–42 之间，尚未出现 Halfcheetah 那样的完全崩溃。

---

## 4. 规律总结与分析

### 4.1 噪声类型决定难度层级

```
concat  <  project  <<  nonlinear
（最易）                  （最难）
```

- **concat**：所有编码器（含无编码器的 raw_noisy）均能保持接近上界，解耦正则几乎没有必要
- **project**：绝大多数方法仍然鲁棒，L1 是唯一显著不稳定的方法
- **nonlinear**：方法间差异显著，高噪声区域所有方法均崩溃，但 HSIC/Barlow 崩溃更慢

### 4.2 解耦方法的边界条件

解耦正则在以下条件下**有效**：
- 噪声维度较小（dim ≤ 8）
- 噪声强度中等（scale ≤ 1.0）
- 噪声类型为 nonlinear（concat/project 时解耦几乎无益）

解耦正则在以下条件下**失效**：
- 噪声维度超过观测维度的 ~50%（halfcheetah obs_dim=17，噪声 dim=17 时崩溃）
- 噪声强度 ≥ 1.5 且维度 ≥ 13 的组合压力下
- 无特权信息（不知道噪声维度）时

### 4.3 L1 的不稳定性

L1 稀疏正则在所有三种噪声类型的高维场景下均表现出显著不稳定性（标准差大、偶发崩溃），综合排名最末。推测是因为 L1 在高维稀疏化时容易陷入不良局部最优，梯度信号不稳定。

### 4.4 Plain Encoder 的竞争力

无解耦正则的普通 MLP 编码器在低-中等噪声条件下与解耦方法差距不大，说明**编码器本身的容量和预训练本身就带来了一定的噪声过滤能力**，解耦正则的边际收益在轻噪声下有限。

---

## 5. 下一步计划

### 5.1 扩展实验覆盖

1. **补全环境覆盖**：hopper 精简方法集（目前仅 4 种）+ concat/project 噪声；启动 walker2d 全量实验
2. **跨环境汇总分析**：三环境统一对比，验证方法排名是否跨任务一致
3. **增加 seeds**：nonlinear 主实验从 3 seeds 扩展到 5 seeds，提升统计显著性

### 5.2 消融实验（计划中）

#### 消融 A：替换下游 RL 算法

当前框架将 IQL 作为唯一的下游策略学习算法。计划将其替换为其他 offline RL 方法（如 TD3+BC、CQL），验证解耦编码器的增益是否对下游算法具有通用性，还是 IQL 特有的性质。

> **研究问题**：解耦预训练的鲁棒性收益是否与下游 RL 算法解耦？HSIC/Barlow 的优势是否在不同算法下一致？

#### 消融 B：PPF 中的特权监督信号消融

PPF 框架的核心是以 $s'_{\text{true}}$ 作为特权预测目标。本消融系统性地弱化或移除这一特权，量化其对最终鲁棒性的贡献。

| 变体 | Next-State 预测目标 | 特权程度 | 动机 |
|------|---------------------|---------|------|
| **基准（当前）** | $s'_{\text{true}}$（干净下一状态） | 完全特权 | PPF 框架的标准设定 |
| **变体 B1：Noisy Target** | $\tilde{x}'_{\text{noisy}}$（带噪下一观测） | 无特权 | 模拟 $s_{\text{true}}$ 不可记录的场景；与"无特权信息消融"（4.2节）的关联：后者去除噪声维度标签，本变体进一步去除下一状态的干净目标 |
| **变体 B2：Reward Only** | 删除该损失项，仅保留 $\mathcal{L}_{\text{reward}}$ | 最弱监督 | 测试 reward-only 预训练能否维持解耦效果，以及奖励信号是否足够驱动任务相关表征 |

> **研究问题**：鲁棒性收益有多大比例来自 $s'_{\text{true}}$ 的强监督，有多大比例来自解耦正则本身？当特权信息完全不可用时（B1/B2），HSIC/Barlow 的优势是否仍然保持？

这组消融与研究问题 3 直接对应，将回答 PPF 框架中特权信息与解耦正则对鲁棒性的各自贡献。

### 5.3 分析工作

4. **解耦质量度量**：结合已保存的 `obs_stats`，量化 $z_{\text{task}}$ 与 $z_{\text{irrel}}$ 的实际统计独立性，分析解耦质量与下游性能的相关性

---

## 6. 当前完成情况

后续环境（hopper、walker2d）采用**精简方法集**：`true_only`、`raw_noisy`、`plain`、`Barlow`、`HSIC`、`dCor`（共 6 种），省略在 halfcheetah 上排名靠后且行为已充分刻画的 Cov、InfoNCE、L1。

| 实验类别 | 环境 | 噪声类型 | 方法数 | 状态 |
|----------|------|---------|--------|------|
| 全量方法系统性扫描（第一阶段，粗粒度） | halfcheetah | nonlinear | 9 种 | 完成（dim: 5/10/20/40/80，scale: 0.5–8.0） |
| 全量方法系统性扫描（第一阶段，粗粒度） | halfcheetah | concat | 9 种 | 完成（dim: 5/10/20/40/80，scale: 0.5–4.0） |
| 全量方法系统性扫描（第一阶段，粗粒度） | halfcheetah | project | 9 种 | 完成（dim: 10/20/40/80，scale: 1.0–8.0） |
| 精细化主实验（第二阶段） | halfcheetah | nonlinear | 9 种 | 完成（seeds 1/2/3，dim: 4/8/13/17，scale: 0.5–2.0） |
| 特权信息消融（no-privilege） | halfcheetah | nonlinear | 3 种 | 完成（Plain/Barlow/HSIC，dim=13/17，scale=1.5/2.0） |
| 方法综合排名 | halfcheetah | nonlinear | 6 种解耦 | 完成 |
| 精简方法扫描 | hopper | nonlinear | 4 种（HSIC/Barlow/Plain/Raw） | 部分完成（seeds 1/2/3，dim: 2/4/6/8） |
| 精简方法扫描 | hopper | nonlinear | 6 种 | 待完成（补全 true_only/dCor） |
| 精简方法扫描 | hopper | concat/project | 6 种 | 待完成 |
| 精简方法扫描 | walker2d | nonlinear | 6 种 | 待完成 |
| 精简方法扫描 | walker2d | concat/project | 6 种 | 待完成 |
| 消融 A：替换下游 RL 算法 | halfcheetah | nonlinear | 3–4 种 | 计划中 |
| 消融 B：PPF 特权信号强度 | halfcheetah | nonlinear | 3–4 种 | 计划中 |

---

## 7. 参考框架与工具

- **Offline RL 基础**：IQL (Kostrikov et al., 2021)
- **数据集**：D4RL locomotion benchmark (Fu et al., 2020)
- **解耦方法**：Barlow Twins (Zbontar et al., 2021)，HSIC (Gretton et al., 2005)，dCor (Székely et al., 2007)，InfoNCE (van den Oord et al., 2018)
- **实验环境**：MuJoCo + D4RL，PyTorch，GPU 加速训练
