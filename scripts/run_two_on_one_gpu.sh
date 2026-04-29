#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# RunPod test: run 2 notebook jobs concurrently on ONE GPU
# ============================================================

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/robust-offline-rl-disentanglement}"
NOTEBOOK_DIR="${NOTEBOOK_DIR:-$PROJECT_ROOT/notebooks/main}"
EXEC_DIR="${EXEC_DIR:-$PROJECT_ROOT/artifacts/executed/runpod_parallel_test}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/runpod_parallel_test}"

CONDA_SH="${CONDA_SH:-/workspace/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rl_env}"
JUPYTER_KERNEL="${JUPYTER_KERNEL:-rl_env}"

GPU_ID="${GPU_ID:-0}"
ENV_NAME="${ENV_NAME:-halfcheetah-medium-v2}"
NOTEBOOK_NAME="${NOTEBOOK_NAME:-exp_disentangled_dcor.ipynb}"

mkdir -p "$EXEC_DIR" "$LOG_DIR"

scale_to_tag() {
  local value="$1"
  echo "${value//./p}"
}

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

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
  echo "ERROR: MuJoCo 2.1.0 not found"
  exit 1
fi

export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_HOME"
export CPATH="$CONDA_PREFIX/include"
export C_INCLUDE_PATH="$CONDA_PREFIX/include"
export LIBRARY_PATH="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$CONDA_PREFIX/lib:$MUJOCO_HOME/bin:/usr/lib/nvidia"

NOTEBOOK_PATH="$NOTEBOOK_DIR/$NOTEBOOK_NAME"
if [[ ! -f "$NOTEBOOK_PATH" ]]; then
  echo "ERROR: notebook not found: $NOTEBOOK_PATH"
  exit 1
fi

run_job() {
  local seed="$1"
  local ndim="$2"
  local nscale="$3"
  local ntype="$4"

  local method="${NOTEBOOK_NAME%.ipynb}"
  local scale_tag
  scale_tag="$(scale_to_tag "$nscale")"

  local job_name="${method}_${ENV_NAME}_s${seed}_nd${ndim}_ns${scale_tag}_nt${ntype}"
  local out_nb="$EXEC_DIR/${job_name}.out.ipynb"
  local out_log="$LOG_DIR/${job_name}.log"

  echo "[$(timestamp)] START $job_name on GPU $GPU_ID" | tee "$out_log"

  (
    export ENV_NAME="$ENV_NAME"
    export SEED="$seed"
    export NOISE_DIM="$ndim"
    export NOISE_SCALE="$nscale"
    export NOISE_TYPE="$ntype"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    cd "$PROJECT_ROOT"

    python -m jupyter nbconvert \
      --to notebook \
      --execute "$NOTEBOOK_PATH" \
      --ExecutePreprocessor.timeout=-1 \
      --ExecutePreprocessor.kernel_name="$JUPYTER_KERNEL" \
      --output "$(basename "$out_nb")" \
      --output-dir "$EXEC_DIR"
  ) >> "$out_log" 2>&1

  local code=$?
  if [[ $code -eq 0 ]]; then
    echo "[$(timestamp)] DONE  $job_name" | tee -a "$out_log"
  else
    echo "[$(timestamp)] FAIL  $job_name (exit=$code)" | tee -a "$out_log"
  fi
  return $code
}

# ----------------------------
# Pick exactly two test jobs
# ----------------------------
run_job 1 28 0.5 nonlinear &
PID1=$!

run_job 2 28 0.5 nonlinear &
PID2=$!

echo "Launched two jobs on GPU $GPU_ID"
echo "PID1=$PID1"
echo "PID2=$PID2"

wait $PID1
STATUS1=$?

wait $PID2
STATUS2=$?

echo
echo "================ FINAL STATUS ================"
echo "Job 1 exit code: $STATUS1"
echo "Job 2 exit code: $STATUS2"

if [[ $STATUS1 -eq 0 && $STATUS2 -eq 0 ]]; then
  echo "Both jobs finished successfully."
else
  echo "At least one job failed."
  exit 1
fi