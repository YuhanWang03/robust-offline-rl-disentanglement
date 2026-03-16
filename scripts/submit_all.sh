#!/usr/bin/env bash
set -euo pipefail

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
NOISY_NOTEBOOKS=(
  "02_raw_noisy.ipynb"
  "03_plain_encoder.ipynb"
  "04_disentangled_cov.ipynb"
  "05_disentangled_barlow.ipynb"
  "06_disentangled_hsic.ipynb"
  "07_disentangled_dcor.ipynb"
  "08_disentangled_infonce.ipynb"
  "09_disentangled_l1.ipynb"
)

TRUE_ONLY_NOTEBOOK="01_true_only.ipynb"

ENV_NAMES=(
  "halfcheetah-medium-v2"
  "hopper-medium-v2"
  "walker2d-medium-v2"
)

SEEDS=(2)

NOISE_DIMS=(5 10 20 40 80)
NOISE_SCALES=(1.0 2.0 4.0)
NOISE_TYPES=("nonlinear")

RUN_TRUE_ONLY=true

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
  local notebook_path="$NOTEBOOK_DIR/$notebook"

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

set -euo pipefail

cd "${PROJECT_ROOT}"
source "${CONDA_INIT}"
conda activate "${CONDA_ENV_NAME}"

export CPATH="\$CONDA_PREFIX/include"
export C_INCLUDE_PATH="\$CONDA_PREFIX/include"
export LIBRARY_PATH="\$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib:\$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

export ENV_NAME="${env_name}"
export SEED="${seed}"
export NOISE_DIM="${ndim}"
export NOISE_SCALE="${nscale}"
export NOISE_TYPE="${ntype}"

echo "Running notebook: ${notebook}"
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
  echo "✅ Submitted noisy job: $job_name"
}

submit_true_only_job() {
  local env_name="$1"
  local seed="$2"

  local method="${TRUE_ONLY_NOTEBOOK%.ipynb}"
  local job_name="${method}_${env_name}_s${seed}"
  local job_script="$JOB_DIR/${job_name}.slurm"
  local notebook_path="$NOTEBOOK_DIR/$TRUE_ONLY_NOTEBOOK"

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

set -euo pipefail

cd "${PROJECT_ROOT}"
source "${CONDA_INIT}"
conda activate "${CONDA_ENV_NAME}"

export CPATH="\$CONDA_PREFIX/include"
export C_INCLUDE_PATH="\$CONDA_PREFIX/include"
export LIBRARY_PATH="\$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:\$CONDA_PREFIX/lib:\$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"

export ENV_NAME="${env_name}"
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
}

# ------------------------------------------------------------
# Submit noisy methods
# ------------------------------------------------------------
for notebook in "${NOISY_NOTEBOOKS[@]}"; do
  for env_name in "${ENV_NAMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for ndim in "${NOISE_DIMS[@]}"; do
        for nscale in "${NOISE_SCALES[@]}"; do
          for ntype in "${NOISE_TYPES[@]}"; do
            submit_noisy_job "$notebook" "$env_name" "$seed" "$ndim" "$nscale" "$ntype"
          done
        done
      done
    done
  done
done

# ------------------------------------------------------------
# Submit true_only baselines
# ------------------------------------------------------------
if [[ "$RUN_TRUE_ONLY" == "true" ]]; then
  for env_name in "${ENV_NAMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      submit_true_only_job "$env_name" "$seed"
    done
  done
fi

echo
echo "All jobs have been submitted."
echo "Use: squeue -u \$(whoami)"