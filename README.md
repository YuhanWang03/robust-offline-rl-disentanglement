# Robust Offline RL Disentanglement

## Overview

This project studies **representation robustness in offline reinforcement learning** under **synthetic observation corruption**. The core idea is a **Privileged Pretraining Framework (PPF)**: the encoder is trained with access to clean states as supervision targets (a form of privileged information), then deployed at inference time on corrupted observations only. The downstream policy (IQL, TD3+BC, or BC) is trained on the frozen encoder's output.

Experiments compare multiple encoder settings and disentanglement regularizers under three corruption families:

- **concat**: append nuisance noise directly to the clean observation
- **project**: apply a random orthogonal linear mixing after concatenation
- **nonlinear**: apply a two-layer nonlinear mixing after concatenation

The repository is organized for both **course-project reproducibility** and **conference paper preparation**. Training notebooks are kept readable, while repeated utility logic (evaluation, metric saving, frozen-encoder training) is factored into `src/`.

---

## Representation Methods

### Main methods (PPF-based)

| Method | Description |
|---|---|
| `true_only` | Upper bound — policy trained on clean states directly |
| `raw_noisy` | Lower bound — policy trained on raw corrupted observations |
| `plain` | PPF encoder with dynamics + reward objectives only, no disentanglement |
| `disentangled_barlow` | PPF + Barlow Twins cross-correlation penalty |
| `disentangled_cov` | PPF + covariance whitening penalty |
| `disentangled_hsic` | PPF + HSIC independence criterion |
| `disentangled_dcor` | PPF + distance correlation penalty |
| `disentangled_infonce` | PPF + InfoNCE contrastive penalty |
| `disentangled_l1` | PPF + L1 cross-correlation penalty |

### External baseline

| Method | Description |
|---|---|
| `pca` | PCA-IQL — projects noisy observations onto top-k PCA components (no neural encoder, no privileged information) |

---

## Ablation Experiments

| Group | Description |
|---|---|
| **B1 — no privileged target** | Remove clean-state supervision; encoder trained on noisy next-obs prediction |
| **B2 — reward only** | Remove dynamics loss; encoder trained on reward prediction + disentanglement only |
| **A — algorithm** | Replace IQL with TD3+BC or BC; encoder pretraining unchanged |

---

## Environments

- `halfcheetah-medium-v2`
- `hopper-medium-v2`
- `walker2d-medium-v2`
- `ant-medium-v2`

---

## Prerequisites

- **OS:** Linux or WSL2 is recommended. Native Windows is not supported due to MuJoCo / D4RL dependency requirements.
- **GPU:** NVIDIA GPU recommended for training speed.
- **Python environment:** Conda or Docker for reproducibility.

---

## Installation

### Option 1: Docker

```bash
docker build -t robust_offline_rl:latest .
docker run --gpus all -it --rm robust_offline_rl:latest
```

### Option 2: Conda (recommended for local development)

#### 1. Install system dependencies

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

#### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate rl_env
```

#### 3. Install PyTorch

Install the build matching your local CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Install D4RL

```bash
pip install "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl"
```

---

## Repository Structure

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
│   ├── config.py              # global path constants
│   ├── experiment_config.py   # reads env-var overrides (ENV_NAME, SEED, etc.)
│   ├── dataset.py             # NoisyOfflineRLDataset
│   ├── encoder.py             # DisentangledEncoder
│   ├── pca_encoder.py         # PCAEncoder (external baseline)
│   ├── iql.py                 # IQLAgent
│   ├── td3bc.py               # TD3BCAgent
│   ├── bc.py                  # BCAgent
│   ├── train_eval.py          # training loops + evaluation utilities
│   ├── utils.py
│   └── visualization.py
├── scripts/
│   ├── run_all.sh                      # local execution (edit NOTEBOOKS array)
│   ├── submit_all.sh                   # Slurm: main IQL methods
│   ├── submit_true_only.sh             # Slurm: true_only baseline
│   ├── submit_ablation_reward_only.sh  # Slurm: B2 reward-only ablation
│   ├── submit_ablation_td3bc.sh        # Slurm: ablation A with TD3+BC
│   ├── submit_ablation_bc.sh           # Slurm: ablation A with BC
│   └── submit_ablation_baselines.sh    # Slurm: external baselines (PCA-IQL)
├── notebooks/
│   ├── main/                      # PPF main experiments (IQL)
│   │   ├── exp_true_only.ipynb
│   │   ├── exp_raw_noisy.ipynb
│   │   ├── exp_plain_encoder.ipynb
│   │   ├── exp_disentangled_barlow.ipynb
│   │   ├── exp_disentangled_cov.ipynb
│   │   ├── exp_disentangled_hsic.ipynb
│   │   ├── exp_disentangled_dcor.ipynb
│   │   ├── exp_disentangled_infonce.ipynb
│   │   └── exp_disentangled_l1.ipynb
│   ├── ablation_noisy_target/     # B1: remove privileged supervision
│   │   ├── exp_plain_encoder_no_priv.ipynb
│   │   ├── exp_disentangled_barlow_no_priv.ipynb
│   │   ├── exp_disentangled_cov_no_priv.ipynb
│   │   ├── exp_disentangled_hsic_no_priv.ipynb
│   │   ├── exp_disentangled_dcor_no_priv.ipynb
│   │   ├── exp_disentangled_infonce_no_priv.ipynb
│   │   └── exp_disentangled_l1_no_priv.ipynb
│   ├── ablation_reward_only/      # B2: remove dynamics loss
│   │   ├── exp_plain_encoder_reward_only.ipynb
│   │   ├── exp_disentangled_barlow_reward_only.ipynb
│   │   ├── exp_disentangled_cov_reward_only.ipynb
│   │   ├── exp_disentangled_hsic_reward_only.ipynb
│   │   ├── exp_disentangled_dcor_reward_only.ipynb
│   │   ├── exp_disentangled_infonce_reward_only.ipynb
│   │   └── exp_disentangled_l1_reward_only.ipynb
│   ├── ablation_td3bc/            # Ablation A: TD3+BC policy
│   │   ├── exp_true_only_td3bc.ipynb
│   │   ├── exp_raw_noisy_td3bc.ipynb
│   │   ├── exp_plain_encoder_td3bc.ipynb
│   │   ├── exp_disentangled_barlow_td3bc.ipynb
│   │   ├── exp_disentangled_cov_td3bc.ipynb
│   │   ├── exp_disentangled_hsic_td3bc.ipynb
│   │   ├── exp_disentangled_dcor_td3bc.ipynb
│   │   ├── exp_disentangled_infonce_td3bc.ipynb
│   │   └── exp_disentangled_l1_td3bc.ipynb
│   ├── ablation_bc/               # Ablation A: BC policy
│   │   ├── exp_true_only_bc.ipynb
│   │   ├── exp_raw_noisy_bc.ipynb
│   │   ├── exp_plain_encoder_bc.ipynb
│   │   ├── exp_disentangled_barlow_bc.ipynb
│   │   ├── exp_disentangled_cov_bc.ipynb
│   │   ├── exp_disentangled_hsic_bc.ipynb
│   │   ├── exp_disentangled_dcor_bc.ipynb
│   │   ├── exp_disentangled_infonce_bc.ipynb
│   │   └── exp_disentangled_l1_bc.ipynb
│   ├── baselines/                 # External baselines
│   │   └── exp_pca_iql.ipynb
│   └── analysis/                  # Aggregation and visualization
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

## Output Layout

All outputs follow a consistent directory hierarchy keyed by method, environment, noise configuration, and seed.

### Methods with noise sweep (all except `true_only`)

```text
artifacts/
├── checkpoints/
│   └── <method>/<env_name>/<noise_tag>/seed_<n>/
│       ├── encoder_epoch_50.pth   # for PPF methods only
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

A typical `noise_tag` looks like `nd11_ns2p0_nonlinear` (noise_dim=11, noise_scale=2.0, noise_type=nonlinear).

### `true_only` baseline

`true_only` does not sweep noise parameters, so there is no `noise_tag` directory level:

```text
artifacts/checkpoints/true_only/<env_name>/seed_<n>/
results/raw_metrics/true_only/<env_name>/seed_<n>/metrics.json
```

---

## Running Experiments

### Local execution

Edit the `NOTEBOOKS` array in `run_all.sh` to select which notebooks to run, then:

```bash
bash scripts/run_all.sh
```

Override experiment variables at runtime:

```bash
ENV_NAME=halfcheetah-medium-v2 SEED=1 NOISE_DIM=11 NOISE_SCALE=2.0 NOISE_TYPE=nonlinear \
bash scripts/run_all.sh
```

### Cluster execution (Slurm)

Each script submits a full job grid (seeds × noise dims × noise scales × noise types). Set `ENV_NAME` at the top of the script before submitting.

| Script | Purpose |
|---|---|
| `submit_all.sh` | Main IQL experiments (PPF methods) |
| `submit_true_only.sh` | `true_only` baseline (seed sweep only) |
| `submit_ablation_reward_only.sh` | B2: reward-only pretraining ablation |
| `submit_ablation_td3bc.sh` | Ablation A: TD3+BC policy |
| `submit_ablation_bc.sh` | Ablation A: BC policy |
| `submit_ablation_baselines.sh` | External baselines (PCA-IQL) |

```bash
bash scripts/submit_all.sh
```

### Interactive notebook execution

```bash
jupyter lab
```

Then open notebooks under the relevant subdirectory of `notebooks/`.

---

## Recommended Workflow

1. Run `notebooks/main/` experiments for target environments and noise configurations.
2. Run ablation groups (`ablation_noisy_target/`, `ablation_reward_only/`, `ablation_td3bc/`, `ablation_bc/`) and baseline (`baselines/`).
3. Use `notebooks/analysis/eval_all.ipynb` to aggregate `results/raw_metrics/`.
4. Save plots to `results/figures/` and summary tables to `results/tables/`.

---

## Notes on Reproducibility

- Training seeds are controlled via `src/experiment_config.py` (reads `SEED` env-var) and can be overridden at runtime.
- `artifacts/checkpoints/` is excluded from Git tracking; `results/raw_metrics/` is the primary artifact for aggregation.
- PCA components are saved as `.npz` files under `artifacts/checkpoints/pca/` for exact reproducibility.

---

## Experiment Roadmap

### Phase 1

Focus: single environment, core method comparison, basic ablations.

- **Environment:** `halfcheetah-medium-v2`
- **Noise type:** nonlinear
- **Noise grid:** dims ∈ {3, 6, 8, 11}, scales ∈ {0.5, 1.0, 1.5, 2.0}
- **Seeds:** 3

**Experiments:**
- [ ] Main IQL experiments — all 9 methods (`true_only`, `raw_noisy`, `plain`, 6 disentangled variants)
- [ ] Ablation B1 — remove privileged supervision (`*_no_priv`)
- [ ] Ablation B2 — reward-only pretraining (`*_reward_only`)
- [ ] Analysis — aggregation, dimension sweep plots, method comparison tables

---

### Phase 2

Focus: multi-environment generalization, algorithm robustness, external baseline.

- **Environments:** `halfcheetah-medium-v2`, `hopper-medium-v2`, `walker2d-medium-v2`, `ant-medium-v2`
- **Noise type:** nonlinear
- **Noise grid:** same as Phase 1
- **Seeds:** 5

**Experiments:**
- [ ] Main IQL experiments — all environments × all methods
- [ ] Ablation A — algorithm: TD3+BC and BC policies
- [ ] Ablation B1 + B2 — all environments
- [ ] External baseline — PCA-IQL across all environments
- [ ] Analysis — lambda sensitivity, cross-environment summary tables, publication-ready figures

---

## Citation / Acknowledgment

This repository uses D4RL locomotion datasets and an IQL-based offline RL pipeline as its experimental foundation. If you build on this codebase, please also cite the relevant upstream libraries and benchmark papers (D4RL, IQL, TD3+BC) as appropriate.
