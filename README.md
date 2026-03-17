# Robust Offline RL Disentanglement

## Overview

This project studies **representation robustness in offline reinforcement learning** under **synthetic observation corruption**. The experiments use an **IQL-based pipeline** and compare multiple encoder settings and disentanglement regularizers under three corruption families:

- **concat**: append nuisance noise directly to the clean observation
- **project**: apply linear mixing after concatenation
- **nonlinear**: apply nonlinear mixing after concatenation

The repository is organized for both **course-project reproducibility** and **GitHub portfolio presentation**. Training notebooks are kept readable, while repeated utility logic (evaluation, metric saving, frozen-encoder IQL training) is moved into `src/`.

---

## Current Scope

The project compares the following representation settings and baselines:

- `true_only`
- `raw_noisy`
- `plain`
- `disentangled_cov`
- `disentangled_barlow`
- `disentangled_hsic`
- `disentangled_dcor`
- `disentangled_infonce`
- `disentangled_l1`

The main environments are D4RL locomotion tasks such as:

- `halfcheetah-medium-v2`
- `hopper-medium-v2`
- `walker2d-medium-v2`

---

## Prerequisites

- **OS:** Linux or WSL2 is recommended.
  Native Windows is not recommended because MuJoCo / D4RL-related dependencies are significantly easier to manage in Linux-based environments.
- **GPU:** An NVIDIA GPU is recommended for faster training.
- **Python environment:** A Conda environment or Docker container is recommended for reproducibility.

---

## Installation

Two setup paths are supported.

### Option 1: Docker

Use Docker if you want a reproducible environment or do not want to manually install system-level dependencies.

```bash
# Build the image
docker build -t robust_offline_rl:latest .

# Run the container with GPU access
docker run --gpus all -it --rm robust_offline_rl:latest
```

After entering the container, you can run notebooks or scripts directly.

### Option 2: Conda (recommended for local development)

#### 1. Install system dependencies

These packages are typically needed for MuJoCo, OpenGL, and D4RL-related components.

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

Use the environment file provided in the repository.

```bash
conda env create -f environment.yml
conda activate rl_env
```

If your local setup uses a cleaned environment file instead, replace `environment.yml` with your preferred environment specification.

#### 3. Install PyTorch

Install the PyTorch build that matches your local CUDA version. Example for CUDA 12.1:

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

## Output Layout

### Noisy / disentangled / plain methods

For methods that depend on corruption settings (for example `disentangled_barlow`, `plain`, or `raw_noisy`), outputs are organized as:

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

where a typical `noise_tag` looks like:

```text
nd5_ns0p5_nonlinear
```

### `true_only` baseline

`true_only` does **not** use corruption sweeps, so it does not include a `noise_tag` directory level.

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

## Running Experiments

### Local execution

Use the local runner script from the project root:

```bash
bash scripts/run_all.sh
```

You can also override experiment variables at runtime:

```bash
KERNEL=python3 \
ENV_NAME=halfcheetah-medium-v2 \
SEED=1 \
NOISE_DIM=5 \
NOISE_SCALE=2.0 \
NOISE_TYPE=nonlinear \
bash scripts/run_all.sh
```

### Cluster execution

Use the submission script:

```bash
bash scripts/submit_all.sh
```

This script is intended for the school compute cluster and generates Slurm jobs under `artifacts/slurm_jobs/`.

### Interactive notebook execution

Launch Jupyter from the project root:

```bash
jupyter lab
```

Then open notebooks under `notebooks/`.

---

## Recommended Workflow

A typical workflow is:

1. Run training notebooks for one or more methods.
2. Save checkpoints, observation statistics, and metrics automatically.
3. Use `notebooks/10_eval_all.ipynb` to aggregate `results/raw_metrics/`.
4. Save plots to `results/figures/` and summary tables to `results/tables/`.

---

## Notes on Reproducibility

- Training seeds are controlled through `src/experiment_config.py` and notebook-level parameters.
- `artifacts/checkpoints/` is intended for local training outputs and is usually excluded from Git tracking.
- `results/raw_metrics/` is the main source used for later aggregation and plotting.
- For portfolio/GitHub presentation, representative figures and summary tables are more important than uploading all raw checkpoints.

---

## Project Status

This project currently contains:

- a proposal-stage problem formulation,
- a mid-stage experimental benchmark pipeline,
- reorganized training/evaluation notebooks,
- reproducible output directories for checkpoints, metrics, and plotting.

The main research direction is to study **when disentanglement helps, when it does not, and where robustness boundaries emerge under structured corruption**.

---

## Citation / Acknowledgment

This repository uses D4RL locomotion datasets and an IQL-based offline RL setup as the experimental foundation. If you build on this codebase for research use, please also cite the corresponding upstream libraries and benchmark papers where appropriate.
