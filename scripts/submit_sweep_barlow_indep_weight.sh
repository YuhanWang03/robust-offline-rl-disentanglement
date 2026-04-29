#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Sweep: Barlow indep_weight on Ant nonlinear noise
# Fix (env, noise_dim, noise_scale, noise_type, seed), vary INDEP_WEIGHT
# ------------------------------------------------------------

# ── Fixed experiment parameters ──────────────────────────────
ENV_NAME="ant-medium-v2"
NOISE_TYPE="nonlinear"
NOISE_DIM="${NOISE_DIM:-56}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
SEED="${SEED:-1}"

# ── INDEP_WEIGHT sweep values ─────────────────────────────────
INDEP_WEIGHTS=(
  0.0005 0.001 0.002 0.003 0.005 0.007
  0.01 0.015 0.02 0.03 0.05 0.07
  0.1 0.15 0.2
)

# ── Cluster / path configuration ──────────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/robust-offline-rl-disentanglement}"
NOTEBOOK_PATH="$PROJECT_ROOT/notebooks/ablation_indep_weight/exp_disentangled_barlow_indep_sweep.ipynb"
JOB_DIR="$PROJECT_ROOT/artifacts/slurm_jobs"
EXEC_DIR="$PROJECT_ROOT/artifacts/executed/slurm"
LOG_DIR="$PROJECT_ROOT/logs/slurm"

CONDA_INIT="${CONDA_INIT:-/project/engineering/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rl_env}"
JUPYTER_KERNEL="${JUPYTER_KERNEL:-python3}"

mkdir -p "$JOB_DIR" "$EXEC_DIR" "$LOG_DIR"

# ── Slurm resource configuration ──────────────────────────────
SLURM_PARTITION="${SLURM_PARTITION:-gpu-linuxlab}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-engr-class-any}"
SLURM_TIME="${SLURM_TIME:-03:30:00}"
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_MEM="${SLURM_MEM:-32G}"
SLURM_GPUS="${SLURM_GPUS:-1}"

# ── Helper ────────────────────────────────────────────────────
scale_to_tag() { echo "${1//./p}"; }

# ── Submit one job per indep_weight ───────────────────────────
for iw in "${INDEP_WEIGHTS[@]}"; do
  iw_tag="iw_$(scale_to_tag "$iw")"
  scale_tag="$(scale_to_tag "$NOISE_SCALE")"

  job_name="barlow_sweep_${ENV_NAME}_nd${NOISE_DIM}_ns${scale_tag}_s${SEED}_${iw_tag}"
  job_script="$JOB_DIR/${job_name}.slurm"

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
export NOISE_TYPE="${NOISE_TYPE}"
export NOISE_DIM="${NOISE_DIM}"
export NOISE_SCALE="${NOISE_SCALE}"
export SEED="${SEED}"
export INDEP_WEIGHT="${iw}"

echo "=== Barlow indep_weight sweep ==="
echo "ENV_NAME=\$ENV_NAME | NOISE_TYPE=\$NOISE_TYPE | NOISE_DIM=\$NOISE_DIM | NOISE_SCALE=\$NOISE_SCALE | SEED=\$SEED | INDEP_WEIGHT=\$INDEP_WEIGHT"

python -m jupyter nbconvert \\
  --to notebook \\
  --execute "${NOTEBOOK_PATH}" \\
  --ExecutePreprocessor.timeout=-1 \\
  --ExecutePreprocessor.kernel_name="${JUPYTER_KERNEL}" \\
  --output "${job_name}.out.ipynb" \\
  --output-dir "${EXEC_DIR}"
EOT

  sbatch "$job_script"
  echo "✅ Submitted: $job_name"
done

echo
echo "Submitted ${#INDEP_WEIGHTS[@]} jobs for Barlow indep_weight sweep."
echo "Use: squeue -u \$(whoami)"
