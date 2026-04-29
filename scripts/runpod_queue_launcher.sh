#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# RunPod 任务队列启动器
#
# 把所有参数组合生成任务队列，由 4 GPU × 2 slot = 8 个 Worker
# 并发消费，任意一个 Worker 完成后立即领取下一个任务。
#
# 用法：
#   bash scripts/runpod_queue_launcher.sh
#
# 切换实验只需修改下方 ===== 用户配置区 ===== 内的参数。
# ============================================================

# ===================================================================
# 用户配置区 —— 按需修改
# ===================================================================

ENV_NAME="${ENV_NAME:-halfcheetah-medium-v2}"

# 要运行的方法名（对应 run_reward_only.py 中的 METHOD）
METHODS=(
  "plain_reward_only"
  "disentangled_barlow_reward_only"
  "disentangled_dcor_reward_only"
  "disentangled_hsic_reward_only"
)

SEEDS=(2 3)
NOISE_DIMS=(4 8 13 17)
NOISE_SCALES=(0.5 1.0 1.5 2.0)
NOISE_TYPES=("nonlinear")

# GPU 数量和每张卡的并发任务数
NUM_GPUS=4
TASKS_PER_GPU=2

# ===================================================================
# 路径配置
# ===================================================================

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/robust-offline-rl-disentanglement}"
EXEC_DIR="${EXEC_DIR:-$PROJECT_ROOT/artifacts/executed/runpod}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/runpod}"

CONDA_SH="${CONDA_SH:-/workspace/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rl_env}"

FORCE_RERUN="${FORCE_RERUN:-0}"

mkdir -p "$EXEC_DIR" "$LOG_DIR"

# ===================================================================
# 环境 Bootstrap
# ===================================================================

if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda init script not found: $CONDA_SH"; exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

mkdir -p /workspace/.d4rl/datasets
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/workspace/.d4rl/datasets}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

if [[ -d /workspace/.mujoco/mujoco210 ]]; then
  MUJOCO_HOME="/workspace/.mujoco/mujoco210"
elif [[ -d /root/.mujoco/mujoco210 ]]; then
  MUJOCO_HOME="/root/.mujoco/mujoco210"
else
  echo "ERROR: MuJoCo 2.1.0 not found"; exit 1
fi

export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_HOME"
export CPATH="$CONDA_PREFIX/include"
export C_INCLUDE_PATH="$CONDA_PREFIX/include"
export LIBRARY_PATH="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$CONDA_PREFIX/lib:$MUJOCO_HOME/bin:/usr/lib/nvidia"

# ===================================================================
# 生成任务队列
# ===================================================================

QUEUE_DIR=$(mktemp -d /tmp/runpod_queue_XXXXXX)
trap 'rm -rf "$QUEUE_DIR"' EXIT

scale_to_tag() { echo "${1//./p}"; }
timestamp()    { date "+%Y-%m-%d %H:%M:%S"; }

JOB_ID=0
for method in "${METHODS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for ndim in "${NOISE_DIMS[@]}"; do
      for nscale in "${NOISE_SCALES[@]}"; do
        for ntype in "${NOISE_TYPES[@]}"; do
          printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$method" "$ENV_NAME" "$seed" "$ndim" "$nscale" "$ntype" \
            > "$(printf '%s/%08d.job' "$QUEUE_DIR" $JOB_ID)"
          JOB_ID=$((JOB_ID + 1))
        done
      done
    done
  done
done

TOTAL_JOBS=$JOB_ID
TOTAL_WORKERS=$((NUM_GPUS * TASKS_PER_GPU))

echo "[$(timestamp)] ================================================"
echo "[$(timestamp)] 任务队列已生成"
echo "  总任务数:   $TOTAL_JOBS"
echo "  Worker 数:  $TOTAL_WORKERS  (${NUM_GPUS} GPU × ${TASKS_PER_GPU} slot)"
echo "  ENV_NAME:   $ENV_NAME"
echo "[$(timestamp)] ================================================"
echo

# ===================================================================
# 单个任务执行函数
# ===================================================================

run_one_job() {
  local gpu_id="$1"
  local worker_id="$2"
  local method="$3"
  local env_name="$4"
  local seed="$5"
  local ndim="$6"
  local nscale="$7"
  local ntype="$8"

  local scale_tag
  scale_tag="$(scale_to_tag "$nscale")"

  local job_name="${method}_${env_name}_s${seed}_nd${ndim}_ns${scale_tag}_nt${ntype}"
  local done_flag="$EXEC_DIR/${job_name}.done"
  local out_log="$LOG_DIR/${job_name}.log"

  if [[ -f "$done_flag" && "$FORCE_RERUN" != "1" ]]; then
    echo "[$(timestamp)][$worker_id] SKIP: $job_name"
    return 0
  fi

  echo "============================================================" | tee "$out_log"
  echo "[$(timestamp)][$worker_id] START: $job_name  GPU=$gpu_id" | tee -a "$out_log"
  echo "METHOD=$method | ENV=$env_name | SEED=$seed | ND=$ndim | NS=$nscale | NT=$ntype" | tee -a "$out_log"
  echo "============================================================" | tee -a "$out_log"

  (
    export METHOD="$method"
    export ENV_NAME="$env_name"
    export SEED="$seed"
    export NOISE_DIM="$ndim"
    export NOISE_SCALE="$nscale"
    export NOISE_TYPE="$ntype"
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export MUJOCO_GL="osmesa"
    export D4RL_SUPPRESS_IMPORT_ERROR="1"
    export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/workspace/.d4rl/datasets}"
    cd "$PROJECT_ROOT"
    python scripts/run_reward_only.py
  ) 2>&1 | tee -a "$out_log"

  local exit_code=${PIPESTATUS[0]}
  if [[ "$exit_code" -ne 0 ]]; then
    echo "[$(timestamp)][$worker_id] FAILED: $job_name" | tee -a "$out_log"
    return "$exit_code"
  fi

  touch "$done_flag"
  echo "[$(timestamp)][$worker_id] DONE: $job_name" | tee -a "$out_log"
}

# ===================================================================
# Worker 函数：循环从队列抢任务，直到队列为空
# ===================================================================

worker() {
  local gpu_id="$1"
  local worker_id="$2"

  # 取消继承主脚本的 EXIT trap，防止 Worker 退出时误删队列目录
  trap - EXIT

  echo "[$(timestamp)][$worker_id] Worker 启动，GPU=$gpu_id"

  local completed=0

  while true; do
    # 原子抢占：用 mv 重命名，只有成功 mv 的 worker 才能执行该任务
    local job_file=""
    for f in "$QUEUE_DIR"/*.job; do
      [[ -f "$f" ]] || continue
      local claimed="${f%.job}.running.${worker_id}"
      mv "$f" "$claimed" 2>/dev/null && job_file="$claimed" && break
    done

    # 队列已空
    if [[ -z "$job_file" ]]; then
      echo "[$(timestamp)][$worker_id] 队列已空，Worker 退出（完成 $completed 个任务）"
      break
    fi

    # 读取任务参数（tab 分隔）
    local notebook env_name seed ndim nscale ntype
    IFS=$'\t' read -r method env_name seed ndim nscale ntype < "$job_file"
    rm -f "$job_file"

    # 执行任务（失败不中断 Worker，继续领下一个）
    run_one_job "$gpu_id" "$worker_id" \
      "$method" "$env_name" "$seed" "$ndim" "$nscale" "$ntype" || true

    completed=$((completed + 1))
  done
}

# ===================================================================
# 启动 NUM_GPUS × TASKS_PER_GPU 个 Worker
# ===================================================================

PIDS=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
  for slot in $(seq 0 $((TASKS_PER_GPU - 1))); do
    wid="G${gpu}S${slot}"
    worker "$gpu" "$wid" > "$LOG_DIR/worker_${wid}.log" 2>&1 &
    PIDS+=($!)
    echo "[$(timestamp)] 启动 Worker $wid  PID=${PIDS[-1]}  GPU=$gpu  → $LOG_DIR/worker_${wid}.log"
  done
done

echo
echo "实时查看 Worker 日志："
echo "  tail -f $LOG_DIR/worker_G0S0.log"
echo "  tail -f $LOG_DIR/worker_G0S1.log"
echo "  # 以此类推..."
echo

# ===================================================================
# 等待所有 Worker 完成
# ===================================================================

FAILED=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || { echo "[$(timestamp)] WARN: PID $pid 异常退出"; FAILED=$((FAILED+1)); }
done

echo
echo "================ FINAL STATUS ================"
echo "总任务数: $TOTAL_JOBS"
if [[ $FAILED -eq 0 ]]; then
  echo "[$(timestamp)] 所有 Worker 正常退出。"
else
  echo "[$(timestamp)] 有 $FAILED 个 Worker 异常，请检查 $LOG_DIR/worker_*.log"
  exit 1
fi
