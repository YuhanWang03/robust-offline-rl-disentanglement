#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# RunPod: B2 ablation — reward-only pretraining.
# Mirrors submit_ablation_reward_only.sh but runs directly
# on the pod without Slurm / sbatch.
#
# Target environment — change this line to switch environments:
#   halfcheetah-medium-v2 | hopper-medium-v2 | walker2d-medium-v2 | ant-medium-v2
# ============================================================
ENV_NAME="${ENV_NAME:-halfcheetah-medium-v2}"

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/robust-offline-rl-disentanglement}"
NOTEBOOK_DIR="$PROJECT_ROOT/notebooks/ablation_reward_only"
EXEC_DIR="${EXEC_DIR:-$PROJECT_ROOT/artifacts/executed/runpod}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/runpod}"

CONDA_SH="${CONDA_SH:-/workspace/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rl_env}"
JUPYTER_KERNEL="${JUPYTER_KERNEL:-rl_env}"

GPU_ID="${GPU_ID:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

mkdir -p "$EXEC_DIR" "$LOG_DIR"

# ------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------
NOTEBOOKS=(
  "exp_disentangled_barlow_reward_only.ipynb"
  "exp_disentangled_dcor_reward_only.ipynb"
  "exp_disentangled_hsic_reward_only.ipynb"
  "exp_plain_encoder_reward_only.ipynb"
)

SEEDS=(2 3)
if [[ -n "${SEED:-}" ]]; then SEEDS=("$SEED"); fi  # single-seed override for multi-GPU runs
NOISE_DIMS=(4 8 13 17)
if [[ -n "${NOISE_DIMS_STR:-}" ]]; then read -ra NOISE_DIMS <<< "$NOISE_DIMS_STR"; fi  # e.g. NOISE_DIMS_STR="6"
NOISE_SCALES=(0.5 1.0 1.5 2.0)
NOISE_TYPES=("nonlinear")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
scale_to_tag() { echo "${1//./p}"; }
timestamp()    { date "+%Y-%m-%d %H:%M:%S"; }

# ------------------------------------------------------------
# Environment bootstrap
# ------------------------------------------------------------
if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda init script not found: $CONDA_SH"
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

export JUPYTER_CONFIG_DIR="${JUPYTER_CONFIG_DIR:-/tmp/jupyter_clean_config}"
mkdir -p "$JUPYTER_CONFIG_DIR"

python -m pip install -q ipykernel >/dev/null 2>&1 || true
if ! jupyter kernelspec list 2>/dev/null | grep -qE "^[[:space:]]*${JUPYTER_KERNEL}[[:space:]]"; then
  python -m ipykernel install --user --name "$JUPYTER_KERNEL" --display-name "Python ($JUPYTER_KERNEL)"
fi

mkdir -p /workspace/.d4rl/datasets
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/workspace/.d4rl/datasets}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

if [[ -d /workspace/.mujoco/mujoco210 ]]; then
  MUJOCO_HOME="/workspace/.mujoco/mujoco210"
elif [[ -d /root/.mujoco/mujoco210 ]]; then
  MUJOCO_HOME="/root/.mujoco/mujoco210"
else
  echo "ERROR: MuJoCo 2.1.0 not found in /workspace/.mujoco or /root/.mujoco"
  exit 1
fi

export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_HOME"
export CPATH="$CONDA_PREFIX/include"
export C_INCLUDE_PATH="$CONDA_PREFIX/include"
export LIBRARY_PATH="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$CONDA_PREFIX/lib:$MUJOCO_HOME/bin:/usr/lib/nvidia"

echo "[$(timestamp)] PROJECT_ROOT=$PROJECT_ROOT"
echo "[$(timestamp)] ENV_NAME=$ENV_NAME"
echo "[$(timestamp)] JUPYTER_KERNEL=$JUPYTER_KERNEL"
echo "[$(timestamp)] GPU_ID=$GPU_ID"
echo "[$(timestamp)] MUJOCO_HOME=$MUJOCO_HOME"
echo "[$(timestamp)] D4RL_DATASET_DIR=$D4RL_DATASET_DIR"
echo

# ------------------------------------------------------------
# Run one job
# ------------------------------------------------------------
run_job() {
  local notebook="$1"
  local env_name="$2"
  local seed="$3"
  local ndim="$4"
  local nscale="$5"
  local ntype="$6"

  # derive METHOD name from notebook filename: exp_disentangled_barlow_reward_only.ipynb -> disentangled_barlow_reward_only
  local method
  method="${notebook%.ipynb}"
  method="${method#exp_}"
  local scale_tag
  scale_tag="$(scale_to_tag "$nscale")"

  local job_name="${method}_${env_name}_s${seed}_nd${ndim}_ns${scale_tag}_nt${ntype}"
  local done_flag="$EXEC_DIR/${job_name}.done"
  local out_log="$LOG_DIR/${job_name}.log"

  if [[ -f "$done_flag" && "$FORCE_RERUN" != "1" ]]; then
    echo "[$(timestamp)] SKIP: $job_name (already completed)"
    return 0
  fi

  echo "============================================================" | tee "$out_log"
  echo "[$(timestamp)] START: $job_name" | tee -a "$out_log"
  echo "METHOD=$method | ENV_NAME=$env_name | SEED=$seed | NOISE_DIM=$ndim | NOISE_SCALE=$nscale | NOISE_TYPE=$ntype" | tee -a "$out_log"
  echo "CUDA_VISIBLE_DEVICES=$GPU_ID" | tee -a "$out_log"
  echo "============================================================" | tee -a "$out_log"

  (
    export METHOD="$method"
    export ENV_NAME="$env_name"
    export SEED="$seed"
    export NOISE_DIM="$ndim"
    export NOISE_SCALE="$nscale"
    export NOISE_TYPE="$ntype"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    export MUJOCO_GL="osmesa"
    export D4RL_SUPPRESS_IMPORT_ERROR="1"
    export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/workspace/.d4rl/datasets}"

    cd "$PROJECT_ROOT"
    python scripts/run_reward_only.py
  ) 2>&1 | tee -a "$out_log"

  local exit_code=${PIPESTATUS[0]}
  if [[ "$exit_code" -ne 0 ]]; then
    echo "[$(timestamp)] FAILED: $job_name" | tee -a "$out_log"
    return "$exit_code"
  fi

  touch "$done_flag"
  echo "[$(timestamp)] DONE: $job_name" | tee -a "$out_log"
  echo | tee -a "$out_log"
}

# ------------------------------------------------------------
# Main loop
# ------------------------------------------------------------
for notebook in "${NOTEBOOKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for ndim in "${NOISE_DIMS[@]}"; do
      for nscale in "${NOISE_SCALES[@]}"; do
        for ntype in "${NOISE_TYPES[@]}"; do
          run_job "$notebook" "$ENV_NAME" "$seed" "$ndim" "$nscale" "$ntype"
        done
      done
    done
  done
done

echo
echo "[$(timestamp)] All reward_only jobs finished for ENV_NAME=$ENV_NAME"
