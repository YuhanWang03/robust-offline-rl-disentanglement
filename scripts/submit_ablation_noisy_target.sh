#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# B1 ablation: remove privileged supervision.
# Encoder is trained without clean-state targets; next-state
# prediction uses noisy observations only.
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
NOTEBOOKS=(
  "exp_plain_encoder_no_priv.ipynb"
  "exp_disentangled_barlow_no_priv.ipynb"
  "exp_disentangled_cov_no_priv.ipynb"
  "exp_disentangled_hsic_no_priv.ipynb"
  "exp_disentangled_dcor_no_priv.ipynb"
  "exp_disentangled_infonce_no_priv.ipynb"
  "exp_disentangled_l1_no_priv.ipynb"
)

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

submit_job() {
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
  local notebook_path="$NOTEBOOK_DIR/ablation_noisy_target/$notebook"

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

# ------------------------------------------------------------
# Submit all B1 notebooks over the noise grid
# ------------------------------------------------------------
for notebook in "${NOTEBOOKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for ndim in "${NOISE_DIMS[@]}"; do
      for nscale in "${NOISE_SCALES[@]}"; do
        for ntype in "${NOISE_TYPES[@]}"; do
          submit_job "$notebook" "$ENV_NAME" "$seed" "$ndim" "$nscale" "$ntype"
        done
      done
    done
  done
done

echo
echo "All B1 (no_priv) jobs submitted for environment: $ENV_NAME"
echo "Use: squeue -u \$(whoami)"
