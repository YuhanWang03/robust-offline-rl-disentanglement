#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 4-GPU launcher: 每张卡跑 2 个并发任务。
#
# 分配策略：
#   GPU 0  seed=1  task1: NOISE_DIM_A  task2: NOISE_DIM_B
#   GPU 1  seed=2  task1: NOISE_DIM_A  task2: NOISE_DIM_B
#   GPU 2  seed=3  task1: NOISE_DIM_A  task2: NOISE_DIM_B
#   GPU 3  seed=1,2,3  运行第二个目标脚本（可选，不传则跳过）
#
# 用法：
#   bash scripts/runpod_4gpu_launcher.sh <target_script> [target_script_gpu3]
#
# 示例（B1 消融，GPU3 同时跑主实验）：
#   bash scripts/runpod_4gpu_launcher.sh \
#       scripts/runpod_submit_ablation_noisy_target.sh \
#       scripts/runpod_submit_all.sh
#
# 切换环境：
#   ENV_NAME=hopper-medium-v2 bash scripts/runpod_4gpu_launcher.sh \
#       scripts/runpod_submit_ablation_noisy_target.sh
# ============================================================

TARGET_SCRIPT="${1:-}"
TARGET_SCRIPT_GPU3="${2:-}"   # 可选：GPU 3 跑的另一个实验

if [[ -z "$TARGET_SCRIPT" ]]; then
  echo "Usage: bash $(basename "$0") <target_script> [target_script_gpu3]"
  echo ""
  echo "Available targets:"
  ls "$(dirname "$0")"/runpod_submit_*.sh 2>/dev/null | xargs -I{} basename {}
  exit 1
fi

if [[ ! -f "$TARGET_SCRIPT" ]]; then
  echo "ERROR: script not found: $TARGET_SCRIPT"
  exit 1
fi

# ------------------------------------------------------------
# 每张卡的 2 个并发任务分别跑哪个 noise_dim
# 根据你在脚本里设置的 NOISE_DIMS 来填写
# ------------------------------------------------------------
NOISE_DIM_TASK1="${NOISE_DIM_TASK1:-6}"    # GPU 上第 1 个并发任务的 noise_dim
NOISE_DIM_TASK2="${NOISE_DIM_TASK2:-11}"   # GPU 上第 2 个并发任务的 noise_dim

LOG_BASE="${LOG_BASE:-/workspace/robust-offline-rl-disentanglement/logs/runpod}"
mkdir -p "$LOG_BASE"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(timestamp)] 4-GPU 启动器"
echo "  主脚本:      $TARGET_SCRIPT"
echo "  GPU3 脚本:   ${TARGET_SCRIPT_GPU3:-(不使用)}"
echo "  ENV_NAME:    ${ENV_NAME:-<脚本默认>}"
echo "  每卡任务分配:"
echo "    GPU 0  seed=1  noise_dim=$NOISE_DIM_TASK1 & noise_dim=$NOISE_DIM_TASK2"
echo "    GPU 1  seed=2  noise_dim=$NOISE_DIM_TASK1 & noise_dim=$NOISE_DIM_TASK2"
echo "    GPU 2  seed=3  noise_dim=$NOISE_DIM_TASK1 & noise_dim=$NOISE_DIM_TASK2"
if [[ -n "$TARGET_SCRIPT_GPU3" ]]; then
  echo "    GPU 3  seeds=1,2,3  $TARGET_SCRIPT_GPU3"
else
  echo "    GPU 3  (闲置)"
fi
echo

# ============================================================
# GPU 0 — seed 1，两个 noise_dim 并发
# ============================================================
SEED=1 GPU_ID=0 NOISE_DIMS_STR="$NOISE_DIM_TASK1" bash "$TARGET_SCRIPT" \
  > "$LOG_BASE/gpu0_seed1_nd${NOISE_DIM_TASK1}.log" 2>&1 &
PID_0A=$!

SEED=1 GPU_ID=0 NOISE_DIMS_STR="$NOISE_DIM_TASK2" bash "$TARGET_SCRIPT" \
  > "$LOG_BASE/gpu0_seed1_nd${NOISE_DIM_TASK2}.log" 2>&1 &
PID_0B=$!

# ============================================================
# GPU 1 — seed 2，两个 noise_dim 并发
# ============================================================
SEED=2 GPU_ID=1 NOISE_DIMS_STR="$NOISE_DIM_TASK1" bash "$TARGET_SCRIPT" \
  > "$LOG_BASE/gpu1_seed2_nd${NOISE_DIM_TASK1}.log" 2>&1 &
PID_1A=$!

SEED=2 GPU_ID=1 NOISE_DIMS_STR="$NOISE_DIM_TASK2" bash "$TARGET_SCRIPT" \
  > "$LOG_BASE/gpu1_seed2_nd${NOISE_DIM_TASK2}.log" 2>&1 &
PID_1B=$!

# ============================================================
# GPU 2 — seed 3，两个 noise_dim 并发
# ============================================================
SEED=3 GPU_ID=2 NOISE_DIMS_STR="$NOISE_DIM_TASK1" bash "$TARGET_SCRIPT" \
  > "$LOG_BASE/gpu2_seed3_nd${NOISE_DIM_TASK1}.log" 2>&1 &
PID_2A=$!

SEED=3 GPU_ID=2 NOISE_DIMS_STR="$NOISE_DIM_TASK2" bash "$TARGET_SCRIPT" \
  > "$LOG_BASE/gpu2_seed3_nd${NOISE_DIM_TASK2}.log" 2>&1 &
PID_2B=$!

# ============================================================
# GPU 3 — 可选：跑另一个实验脚本（全 seeds）
# ============================================================
PID_3=""
if [[ -n "$TARGET_SCRIPT_GPU3" && -f "$TARGET_SCRIPT_GPU3" ]]; then
  GPU_ID=3 bash "$TARGET_SCRIPT_GPU3" \
    > "$LOG_BASE/gpu3_extra.log" 2>&1 &
  PID_3=$!
fi

# ============================================================
# 输出所有 PID 和日志路径
# ============================================================
echo "[$(timestamp)] 已启动所有后台进程："
echo "  GPU 0 task1 (seed=1, nd=$NOISE_DIM_TASK1)  PID=$PID_0A  → $LOG_BASE/gpu0_seed1_nd${NOISE_DIM_TASK1}.log"
echo "  GPU 0 task2 (seed=1, nd=$NOISE_DIM_TASK2)  PID=$PID_0B  → $LOG_BASE/gpu0_seed1_nd${NOISE_DIM_TASK2}.log"
echo "  GPU 1 task1 (seed=2, nd=$NOISE_DIM_TASK1)  PID=$PID_1A  → $LOG_BASE/gpu1_seed2_nd${NOISE_DIM_TASK1}.log"
echo "  GPU 1 task2 (seed=2, nd=$NOISE_DIM_TASK2)  PID=$PID_1B  → $LOG_BASE/gpu1_seed2_nd${NOISE_DIM_TASK2}.log"
echo "  GPU 2 task1 (seed=3, nd=$NOISE_DIM_TASK1)  PID=$PID_2A  → $LOG_BASE/gpu2_seed3_nd${NOISE_DIM_TASK1}.log"
echo "  GPU 2 task2 (seed=3, nd=$NOISE_DIM_TASK2)  PID=$PID_2B  → $LOG_BASE/gpu2_seed3_nd${NOISE_DIM_TASK2}.log"
[[ -n "$PID_3" ]] && echo "  GPU 3 extra                              PID=$PID_3  → $LOG_BASE/gpu3_extra.log"
echo ""
echo "实时查看进度（示例）："
echo "  tail -f $LOG_BASE/gpu0_seed1_nd${NOISE_DIM_TASK1}.log"
echo ""

# ============================================================
# 等待所有进程结束
# ============================================================
PIDS=("$PID_0A" "$PID_0B" "$PID_1A" "$PID_1B" "$PID_2A" "$PID_2B")
[[ -n "$PID_3" ]] && PIDS+=("$PID_3")

FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || { echo "[$(timestamp)] WARN: PID $pid 退出异常"; FAILED=$((FAILED+1)); }
done

echo
echo "================ FINAL STATUS ================"
if [[ $FAILED -eq 0 ]]; then
  echo "[$(timestamp)] 所有任务完成。"
else
  echo "[$(timestamp)] 有 $FAILED 个进程异常退出，请检查对应日志。"
  exit 1
fi
