#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# RunPod: Barlow indep_weight sweep on Ant nonlinear noise
# One A40 GPU — runs 2 notebooks concurrently, then waits
# before launching the next pair (8 rounds for 15 values).
# ============================================================

# ── Fixed experiment parameters ──────────────────────────────
ENV_NAME="ant-medium-v2"
NOISE_TYPE="nonlinear"
NOISE_DIM="${NOISE_DIM:-83}"
NOISE_SCALE="${NOISE_SCALE:-1.0}"
SEED="${SEED:-1}"

# ── INDEP_WEIGHT sweep values (15 points, log-scale) ─────────
INDEP_WEIGHTS=(
  0.0005 0.001 0.002 0.003 0.005 0.007
  0.01 0.015 0.02 0.03 0.05 0.07
  0.1 0.15 0.2
)

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/robust-offline-rl-disentanglement}"
NOTEBOOK_PATH="$PROJECT_ROOT/notebooks/ablation_indep_weight/exp_disentangled_barlow_indep_sweep.ipynb"
EXEC_DIR="${EXEC_DIR:-$PROJECT_ROOT/artifacts/executed/runpod}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/runpod}"

CONDA_SH="${CONDA_SH:-/workspace/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rl_env}"
JUPYTER_KERNEL="${JUPYTER_KERNEL:-rl_env}"
GPU_ID="${GPU_ID:-0}"

mkdir -p "$EXEC_DIR" "$LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────
scale_to_tag() { echo "${1//./p}"; }
timestamp()    { date "+%Y-%m-%d %H:%M:%S"; }

# ── Environment bootstrap ─────────────────────────────────────
if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda init script not found: $CONDA_SH"
  exit 1
fi

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
  echo "ERROR: MuJoCo 2.1.0 not found in /workspace/.mujoco or /root/.mujoco"
  exit 1
fi

export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_HOME"
export CPATH="$CONDA_PREFIX/include"
export C_INCLUDE_PATH="$CONDA_PREFIX/include"
export LIBRARY_PATH="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$CONDA_PREFIX/lib:$MUJOCO_HOME/bin:/usr/lib/nvidia"

if [[ ! -f "$NOTEBOOK_PATH" ]]; then
  echo "ERROR: notebook not found: $NOTEBOOK_PATH"
  exit 1
fi

echo "[$(timestamp)] PROJECT_ROOT=$PROJECT_ROOT"
echo "[$(timestamp)] ENV_NAME=$ENV_NAME | NOISE_TYPE=$NOISE_TYPE | NOISE_DIM=$NOISE_DIM | NOISE_SCALE=$NOISE_SCALE | SEED=$SEED"
echo "[$(timestamp)] GPU_ID=$GPU_ID | total sweep points=${#INDEP_WEIGHTS[@]}"
echo

# ── Run one sweep point in the background ────────────────────
run_one() {
  local iw="$1"
  local iw_tag
  iw_tag="iw_$(scale_to_tag "$iw")"
  local scale_tag
  scale_tag="$(scale_to_tag "$NOISE_SCALE")"

  local job_name="barlow_sweep_${ENV_NAME}_nd${NOISE_DIM}_ns${scale_tag}_s${SEED}_${iw_tag}"
  local out_nb="$EXEC_DIR/${job_name}.out.ipynb"
  local out_log="$LOG_DIR/${job_name}.log"

  echo "[$(timestamp)] START $job_name" | tee "$out_log"

  (
    export ENV_NAME="$ENV_NAME"
    export NOISE_TYPE="$NOISE_TYPE"
    export NOISE_DIM="$NOISE_DIM"
    export NOISE_SCALE="$NOISE_SCALE"
    export SEED="$SEED"
    export INDEP_WEIGHT="$iw"
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

# ── Main loop: launch in pairs, wait after each pair ─────────
FAILED=0
n=${#INDEP_WEIGHTS[@]}
i=0

while [[ $i -lt $n ]]; do
  iw1="${INDEP_WEIGHTS[$i]}"

  if [[ $((i + 1)) -lt $n ]]; then
    iw2="${INDEP_WEIGHTS[$((i + 1))]}"
    echo "[$(timestamp)] === Round $((i / 2 + 1)): launching pair INDEP_WEIGHT=$iw1 and $iw2 ==="

    run_one "$iw1" &
    PID1=$!
    run_one "$iw2" &
    PID2=$!

    wait $PID1 || { echo "[$(timestamp)] FAILED: iw=$iw1"; FAILED=$((FAILED + 1)); }
    wait $PID2 || { echo "[$(timestamp)] FAILED: iw=$iw2"; FAILED=$((FAILED + 1)); }

    i=$((i + 2))
  else
    echo "[$(timestamp)] === Final round: launching single INDEP_WEIGHT=$iw1 ==="
    run_one "$iw1" || { echo "[$(timestamp)] FAILED: iw=$iw1"; FAILED=$((FAILED + 1)); }
    i=$((i + 1))
  fi

  echo
done

# ── Summary ───────────────────────────────────────────────────
echo "================================================"
echo "[$(timestamp)] Sweep complete."
echo "Total sweep points : $n"
echo "Failed             : $FAILED"
echo "================================================"

[[ $FAILED -eq 0 ]] || exit 1
