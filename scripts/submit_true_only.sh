#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Target environment — change this line to switch environments:
#   halfcheetah-medium-v2 | hopper-medium-v2 | walker2d-medium-v2 | ant-medium-v2
# ------------------------------------------------------------
ENV_NAME="ant-medium-v2"

# ------------------------------------------------------------
# User-configurable cluster paths and environment
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
SLURM_CPUS="${SLURM_CPUS:-4}"
SLURM_MEM="${SLURM_MEM:-16G}"
SLURM_GPUS="${SLURM_GPUS:-1}"

# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------
TRUE_ONLY_NOTEBOOK="exp_true_only.ipynb"

SEEDS=(1 2 3 4)

# ------------------------------------------------------------
# Submit
# ------------------------------------------------------------
for seed in "${SEEDS[@]}"; do
  local_method="${TRUE_ONLY_NOTEBOOK%.ipynb}"
  job_name="${local_method}_${ENV_NAME}_s${seed}"
  job_script="$JOB_DIR/${job_name}.slurm"
  notebook_path="$NOTEBOOK_DIR/$TRUE_ONLY_NOTEBOOK"

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

echo "Running true_only baseline: ${TRUE_ONLY_NOTEBOOK}"
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
  echo "✅ Submitted true_only job: $job_name"
done

echo
echo "true_only job submitted for environment: $ENV_NAME"
echo "Use: squeue -u \$(whoami)"
