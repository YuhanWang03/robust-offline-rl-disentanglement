#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Resolve project paths
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_DIR="$PROJECT_ROOT/notebooks"
EXEC_DIR="$PROJECT_ROOT/artifacts/executed/local"
LOG_DIR="$PROJECT_ROOT/logs/local"

mkdir -p "$EXEC_DIR" "$LOG_DIR"

# ------------------------------------------------------------
# User-configurable parameters
# ------------------------------------------------------------
# Leave empty to use the kernel recorded inside each notebook.
KERNEL="${KERNEL:-rl_env}"

# These environment variables are consumed by src/experiment_config.py
# and the updated notebooks.
export ENV_NAME="${ENV_NAME:-halfcheetah-medium-v2}"
export SEED="${SEED:-1}"
export NOISE_DIM="${NOISE_DIM:-17}"
export NOISE_SCALE="${NOISE_SCALE:-2.0}"
export NOISE_TYPE="${NOISE_TYPE:-nonlinear}"

# Select notebooks for local execution.
# Use paths relative to notebooks/ (include subdirectory).
NOTEBOOKS=(
  "ablation_noisy_target/exp_plain_encoder_no_priv.ipynb"
  "ablation_noisy_target/exp_disentangled_barlow_no_priv.ipynb"
  "ablation_noisy_target/exp_disentangled_hsic_no_priv.ipynb"
)

# ------------------------------------------------------------
# Environment summary
# ------------------------------------------------------------
echo "==== $(date) ===="
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "NOTEBOOK_DIR: $NOTEBOOK_DIR"
echo "EXEC_DIR:     $EXEC_DIR"
echo "LOG_DIR:      $LOG_DIR"
echo "Python:       $(which python || true)"
python --version || true
echo "Kernel:       ${KERNEL:-<from notebook metadata>}"
echo "ENV_NAME:     $ENV_NAME"
echo "SEED:         $SEED"
echo "NOISE_DIM:    $NOISE_DIM"
echo "NOISE_SCALE:  $NOISE_SCALE"
echo "NOISE_TYPE:   $NOISE_TYPE"
echo "================="

# ------------------------------------------------------------
# Execute notebooks
# ------------------------------------------------------------
for nb in "${NOTEBOOKS[@]}"; do
  nb_path="$NOTEBOOK_DIR/$nb"

  if [[ ! -f "$nb_path" ]]; then
    echo "❌ Notebook not found: $nb_path"
    exit 1
  fi

  base="$(basename "${nb%.ipynb}")"
  out_nb="${base}.out.ipynb"
  log="$LOG_DIR/${base}.log"

  echo
  echo "==> [$(date)] Running: $nb"
  echo "    Output notebook: $EXEC_DIR/$out_nb"
  echo "    Log file:        $log"

  : > "$log"

  cmd=(
    python -m jupyter nbconvert
    --to notebook
    --execute "$nb_path"
    --ExecutePreprocessor.timeout=-1
    --output "$out_nb"
    --output-dir "$EXEC_DIR"
  )

  if [[ -n "$KERNEL" ]]; then
    cmd+=(--ExecutePreprocessor.kernel_name="$KERNEL")
  fi

  if ! "${cmd[@]}" >> "$log" 2>&1; then
    echo "❌ FAILED: $nb"
    echo "   Check log: $log"
    echo "   Last 80 lines:"
    tail -n 80 "$log" || true
    exit 1
  fi

  echo "✅ [$(date)] Done: $nb"
done

echo
echo "🎉 ALL DONE: $(date)"