# Base image: CUDA 12.1 + cuDNN + build toolchain (Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for mujoco/d4rl and common tooling
RUN apt-get update -q && apt-get install -y \
    build-essential \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglew-dev \
    mesa-common-dev \
    libglfw3 \
    libglfw3-dev \
    patchelf \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"

# Set working directory
WORKDIR /app

# Create Conda environment from the provided spec
COPY environment_clean.yml .
# RUN conda env create -f environment_clean.yml
# RUN conda config --remove channels defaults || true && conda env create -f environment_clean.yml
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && conda env create -f environment_clean.yml

# Install PyTorch (CUDA 12.1) and D4RL inside rl_env
RUN conda run -n rl_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN conda run -n rl_env pip install "Cython<3"
RUN conda run -n rl_env pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl

# 1. 下载并解压 MuJoCo 2.1.0
RUN mkdir -p /root/.mujoco && \
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
    tar -xzf mujoco210-linux-x86_64.tar.gz -C /root/.mujoco/ && \
    rm mujoco210-linux-x86_64.tar.gz

# 2. 设置环境变量
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

# Copy project files
COPY . .

# Make rl_env the default Python environment
ENV CONDA_DEFAULT_ENV=rl_env
ENV PATH="/opt/conda/envs/rl_env/bin:$PATH"
RUN echo "conda activate rl_env" >> ~/.bashrc
RUN conda run -n rl_env python -c "import mujoco_py"

# Default entry: open a bash shell
CMD ["/bin/bash"]