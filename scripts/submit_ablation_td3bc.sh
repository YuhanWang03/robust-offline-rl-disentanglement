#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Ablation A (algorithm): TD3+BC policy for all encoder methods.
# Covers 6 disentangled + plain (with noise sweep) and
# true_only + raw_noisy baselines (separate sections).
#
# Target environment — change this line to switch environments:
#   halfcheetah-medium-v2 | hopper-medium-v2 | walker2d-medium-v2 | ant-medium-v2
# ------------------------------------------------------------
ENV_NAME="halfcheetah-medium-v2"

# ------------------------------------------------------------
# Cluster paths and conda environment
# ------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/robust-offline-rl-disentanglement}"
NOTEBOOK_DIR="$PROJECT_ROOT/notebooks"
JOB_DIR="$PROJECT_ROOT/artifacts/slurm_jobs"
EXEC_DIR="$PROJECT_ROOT/artifacts/executed/slurm"
LOG_DIR="$PROJECT_ROOT/logs/slurm"

CONDA_INIT="${CONDA_INIT:-/project/engineering/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rl_env}"
JUPYTER_KERNEL="${JUPYTER_KERNEL:-python3}"

mkdir -p "$JOB_DIR" "$EXEC_DIR" "$LOG_DIR"

# ------------------------------------------------------------
# Slurm resource configuration
# ------------------------------------------------------------
SLURM_PARTITION="${SLURM_PARTITION:-gpu-linuxlab}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-engr-class-any}"
SLURM_TIME="${SLURM_TIME:-03:30:00}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_MEM="${SLURM_MEM:-32G}"
SLURM_GPUS="${SLURM_GPUS:-1}"

# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------
# Notebooks that require noise parameters (disentangled + plain + raw_noisy).
NOISY_NOTEBOOKS=(
  "exp_disentangled_barlow_td3bc.ipynb"
  "exp_disentangled_cov_td3bc.ipynb"
  "exp_disentangled_dcor_td3bc.ipynb"
  "exp_disentangled_hsic_td3bc.ipynb"
  "exp_disentangled_infonce_td3bc.ipynb"
  "exp_disentangled_l1_td3bc.ipynb"
  "exp_plain_encoder_td3bc.ipynb"
  "exp_raw_noisy_td3bc.ipynb"
)

# true_only uses clean states only — no noise sweep needed.
TRUE_ONLY_NB="exp_true_only_td3bc.ipynb"

SEEDS=(1 2 3)
NOISE_DIMS=(3 6 8 11)
NOISE_SCALES=(0.5 1.0 1.5 2.0)
NOISE_TYPES=("nonlinear")

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
scale_to_tag() {
  local value="$1"
  echo "${value//./p}"
}

submit_noisy_job() {
  local notebook="$1"
  local env_name="$2"
  local seed="$3"
  local ndim="$4"
  local nscale="$5"
  local ntype="$6"

  local method="${notebook%.ipynb}"
  local scale_tag
  scale_tag="$(scale_to_tag "$nscale")"

  local job_name="${method}_${env_name}_s${seed}_nd${ndim}_ns${scale_tag}_nt${ntype}"
  local job_script="$JOB_DIR/${job_name}.slurm"
  local notebook_path="$NOTEBOOK_DIR/ablation_td3bc/$notebook"

  cat > "$job_script" <<EOT
#!/usr/bin/env bash
#SBATCH -J ${job_name}
#SBATCH -o ${LOG_DIR}/%x_%j.out
#SBATCH -e ${LOG_DIR}/%x_%j.err
#SBATCH -t ${SLURM_TIME}
#SBATCH -p ${SLURM_PARTITION}
#SBATCH -c ${SLURM_CPUS}
#SBATCH -A ${SLURM_ACCOUNT}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --gres=gpu:${SLURM_GPUS}

set -eo pipefail

cd "${PROJECT_ROOT}"
set +u
source "${CONDA_INIT}"
conda activate "${CONDA_ENV_NAME}"
set -u

export CPATH="\$CONDA_PREFIX/include"
export C_INCLUDE_PATH="\$CONDA_PREFIX/include"
export LIBRARY_PATH="\$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib:\$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

export ENV_NAME="${env_name}"
export SEED="${seed}"
export NOISE_DIM="${ndim}"
export NOISE_SCALE="${nscale}"
export NOISE_TYPE="${ntype}"

echo "Running: ${notebook}"
echo "ENV_NAME=\$ENV_NAME | SEED=\$SEED | NOISE_DIM=\$NOISE_DIM | NOISE_SCALE=\$NOISE_SCALE | NOISE_TYPE=\$NOISE_TYPE"

python -m jupyter nbconvert \\
  --to notebook \\
  --execute "${notebook_path}" \\
  --ExecutePreprocessor.timeout=-1 \\
  --ExecutePreprocessor.kernel_name="${JUPYTER_KERNEL}" \\
  --output "${job_name}.out.ipynb" \\
  --output-dir "${EXEC_DIR}"
EOT

  sbatch "$job_script"
  echo "✅ Submitted: $job_name"
}

submit_true_only_job() {
  local seed="$1"
  local method="${TRUE_ONLY_NB%.ipynb}"
  local job_name="${method}_${ENV_NAME}_s${seed}"
  local job_script="$JOB_DIR/${job_name}.slurm"
  local notebook_path="$NOTEBOOK_DIR/ablation_td3bc/$TRUE_ONLY_NB"

  cat > "$job_script" <<EOT
#!/usr/bin/env bash
#SBATCH -J ${job_name}
#SBATCH -o ${LOG_DIR}/%x_%j.out
#SBATCH -e ${LOG_DIR}/%x_%j.err
#SBATCH -t ${SLURM_TIME}
#SBATCH -p ${SLURM_PARTITION}
#SBATCH -c ${SLURM_CPUS}
#SBATCH -A ${SLURM_ACCOUNT}
#SBATCH --mem=${SLURM_MEM}
#SBATCH --gres=gpu:${SLURM_GPUS}

set -eo pipefail

cd "${PROJECT_ROOT}"
set +u
source "${CONDA_INIT}"
conda activate "${CONDA_ENV_NAME}"
set -u

export CPATH="\$CONDA_PREFIX/include"
export C_INCLUDE_PATH="\$CONDA_PREFIX/include"
export LIBRARY_PATH="\$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib:\$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

export ENV_NAME="${ENV_NAME}"
export SEED="${seed}"

echo "Running true_only baseline: ${TRUE_ONLY_NB}"
echo "ENV_NAME=\$ENV_NAME | SEED=\$SEED"

python -m jupyter nbconvert \\
  --to notebook \\
  --execute "${notebook_path}" \\
  --ExecutePreprocessor.timeout=-1 \\
  --ExecutePreprocessor.kernel_name="${JUPYTER_KERNEL}" \\
  --output "${job_name}.out.ipynb" \\
  --output-dir "${EXEC_DIR}"
EOT

  sbatch "$job_script"
  echo "✅ Submitted: $job_name"
}

# ------------------------------------------------------------
# Submit noisy notebooks over the full noise grid
# ------------------------------------------------------------
for notebook in "${NOISY_NOTEBOOKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for ndim in "${NOISE_DIMS[@]}"; do
      for nscale in "${NOISE_SCALES[@]}"; do
        for ntype in "${NOISE_TYPES[@]}"; do
          submit_noisy_job "$notebook" "$ENV_NAME" "$seed" "$ndim" "$nscale" "$ntype"
        done
      done
    done
  done
done

# ------------------------------------------------------------
# Submit true_only (seed sweep only, no noise parameters)
# ------------------------------------------------------------
echo
echo "--- Submitting true_only_td3bc ---"
for seed in "${SEEDS[@]}"; do
  submit_true_only_job "$seed"
done

echo
echo "All td3bc ablation jobs submitted for environment: $ENV_NAME"
echo "Use: squeue -u \$(whoami)"
