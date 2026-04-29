#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 3-GPU parallel launcher for RunPod.
# Assigns one seed per GPU and runs a target script in parallel.
#
# Usage:
#   bash scripts/runpod_3gpu_launcher.sh <target_script>
#
# Examples:
#   bash scripts/runpod_3gpu_launcher.sh scripts/runpod_submit_ablation_noisy_target.sh
#   bash scripts/runpod_3gpu_launcher.sh scripts/runpod_submit_all.sh
#   ENV_NAME=hopper-medium-v2 bash scripts/runpod_3gpu_launcher.sh scripts/runpod_submit_all.sh
#
# Requirements:
#   The target script must support the SEED and GPU_ID env vars.
#   All runpod_submit_*.sh scripts satisfy this requirement.
# ============================================================

TARGET_SCRIPT="${1:-}"
if [[ -z "$TARGET_SCRIPT" ]]; then
  echo "Usage: bash $(basename "$0") <target_script>"
  echo ""
  echo "Available targets:"
  ls "$(dirname "$0")"/runpod_submit_*.sh 2>/dev/null | xargs -I{} basename {}
  exit 1
fi

if [[ ! -f "$TARGET_SCRIPT" ]]; then
  echo "ERROR: script not found: $TARGET_SCRIPT"
  exit 1
fi

LOG_BASE="${LOG_BASE:-/workspace/robust-offline-rl-disentanglement/logs/runpod}"
mkdir -p "$LOG_BASE"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(timestamp)] Launching 3-GPU parallel run"
echo "  Script:   $TARGET_SCRIPT"
echo "  GPU 0  →  seed 1"
echo "  GPU 1  →  seed 2"
echo "  GPU 2  →  seed 3"
echo "  ENV_NAME: ${ENV_NAME:-<default>}"
echo

# Launch one instance per GPU, each handling a single seed
SEED=1 GPU_ID=0 bash "$TARGET_SCRIPT" \
  > "$LOG_BASE/launcher_gpu0_seed1.log" 2>&1 &
PID0=$!

SEED=2 GPU_ID=1 bash "$TARGET_SCRIPT" \
  > "$LOG_BASE/launcher_gpu1_seed2.log" 2>&1 &
PID1=$!

SEED=3 GPU_ID=2 bash "$TARGET_SCRIPT" \
  > "$LOG_BASE/launcher_gpu2_seed3.log" 2>&1 &
PID2=$!

echo "[$(timestamp)] All three instances launched"
echo "  PID (GPU 0, seed 1): $PID0  →  $LOG_BASE/launcher_gpu0_seed1.log"
echo "  PID (GPU 1, seed 2): $PID1  →  $LOG_BASE/launcher_gpu1_seed2.log"
echo "  PID (GPU 2, seed 3): $PID2  →  $LOG_BASE/launcher_gpu2_seed3.log"
echo ""
echo "Monitor progress with:"
echo "  tail -f $LOG_BASE/launcher_gpu0_seed1.log"
echo "  tail -f $LOG_BASE/launcher_gpu1_seed2.log"
echo "  tail -f $LOG_BASE/launcher_gpu2_seed3.log"
echo ""

STATUS0=0; STATUS1=0; STATUS2=0
wait "$PID0" || STATUS0=$?
wait "$PID1" || STATUS1=$?
wait "$PID2" || STATUS2=$?

echo
echo "================ FINAL STATUS ================"
echo "  GPU 0  seed 1  exit=$STATUS0"
echo "  GPU 1  seed 2  exit=$STATUS1"
echo "  GPU 2  seed 3  exit=$STATUS2"

if [[ $STATUS0 -eq 0 && $STATUS1 -eq 0 && $STATUS2 -eq 0 ]]; then
  echo "[$(timestamp)] All seeds finished successfully."
else
  echo "[$(timestamp)] At least one seed failed — check logs above."
  exit 1
fi
