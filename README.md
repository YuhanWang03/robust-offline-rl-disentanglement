# Final RL Project: Implicit Q-Learning & Representation Disentanglement

## Introduction

This project explores and evaluates the performance of **Implicit Q-Learning (IQL)** combined with various **Representation Disentanglement** methods in Offline Reinforcement Learning scenarios. By comparing different contrastive learning and regularization techniques (e.g., Barlow Twins, HSIC, InfoNCE, distance correlation/dCor), we validate the robustness and feature extraction capabilities of the encoder in complex, noisy state spaces.

---

## Prerequisites

- **OS:** Linux (Ubuntu 20.04/22.04) or Windows Subsystem for Linux (WSL2). 
  > *Not recommended for native Windows as physics engine libraries like `mujoco-py` are extremely difficult to compile on it.*
- **GPU:** NVIDIA GPU recommended.
- **Reproducibility:** Both Conda local environment and Docker container methods are provided.

---

## Installation Guide

To accommodate both local development and code reproducibility, we provide two setup methods:

### Option 1: Docker
If you are running on a cluster or lack `sudo` privileges to install system-level C++ dependencies, we highly recommend using Docker for a one-click reproducible environment:

```bash
# 1. Build the Docker image
# This automatically sets up CUDA 12.1, OS underlying dependencies, PyTorch, and D4RL
docker build -t iql_project:latest .

# 2. Run the container with GPU access
docker run --gpus all -it --rm iql_project:latest

# Now you are inside the container with everything set up, you can directly run scripts or Jupyter.
```

### Option 2: Conda (For Local Development)

**Step 2.1: Install System Dependencies**
Due to the usage of `mujoco-py` and `d4rl`, underlying C++ compilers and OpenGL graphics libraries must be installed first.

```bash
sudo apt-get update -q
sudo apt-get install -y build-essential libosmesa6-dev libgl1-mesa-glx libglfw3 libglfw3-dev patchelf
```

**Step 2.2: Create the Conda Environment**
We provide a cleaned-up dependency file `environment_clean.yml` to ensure cross-device compatibility.

```bash
conda env create -f environment_clean.yml
conda activate rl_env
```

**Step 2.3: Install PyTorch & D4RL**
Please install the appropriate PyTorch version based on your local CUDA version (tested with CUDA 12.1).

```bash
# Install PyTorch
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install D4RL
pip install git+[https://github.com/Farama-Foundation/d4rl@master#egg=d4rl](https://github.com/Farama-Foundation/d4rl@master#egg=d4rl)
```

---

## Usage

### Automated Execution
Execution scripts are provided for one-click training or cluster submission:

```bash
# Run all local experiments
bash run_all.sh

# Submit tasks to a cluster
bash submit_all.sh
```

### Interactive Experiments
Experiments are divided into multiple Jupyter Notebooks based on the disentanglement methods. Start Jupyter to view detailed comparisons and visualizations:

```bash
jupyter lab
```
In the left directory panel, you can open files like `exp_disentangled_barlow.ipynb` or `exp_raw_noisy.ipynb` to run specific experiments.

---

## Core Project Structure

```text
FINAL_PROJECT/
├── src/                            # Core source code
│   ├── dataset.py                  # D4RL dataset loading
│   ├── encoder.py                  # Feature encoders
│   ├── iql.py                      # IQL implementation
│   ├── train_eval.py               # Training loop
│   └── vis.py                      # Visualization tools
├── results/                        # Experiment results output
├── logs/                           # Training logs
├── exp_*.ipynb                     # Experiments for different methods
├── run_all.sh                      # Local automation script
├── submit_all.sh                   # Cluster submission script
├── environment_clean.yml           # Cleaned environment dependencies
└── Dockerfile                      # Dockerfile for reproducibility
```