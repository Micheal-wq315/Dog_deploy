#!/usr/bin/env bash
# 使用独立的 CPU MuJoCo / ONNX 环境回放原版 W1W，图形测试避开训练。
set -euo pipefail

HIMLOCO_SOURCE="${HIMLOCO_SOURCE:-/media/rmer/data/rpwd_training/references/uw-himloco}"
HIMLOCO_PYTHON="${HIMLOCO_PYTHON:-/media/rmer/data/rpwd_training/envs/uw-himloco-sim2sim/bin/python}"
HIMLOCO_RUNTIME="${HIMLOCO_RUNTIME:-/media/rmer/data/rpwd_training/sim2sim/uw-himloco}"

if [[ ! -x "$HIMLOCO_PYTHON" || ! -f "$HIMLOCO_SOURCE/mujoco/w1w_2/play_gui.py" ]]; then
  echo "找不到独立 Python 或 W1W 播放器，请检查 HIMLOCO_PYTHON 和 HIMLOCO_SOURCE。" >&2
  exit 1
fi

# 无窗口模式使用 CPU 物理和 CPUExecutionProvider，不创建渲染器。
HIMLOCO_RENDER=1
for arg in "$@"; do
  case "$arg" in
    --headless|--help|-h) HIMLOCO_RENDER=0 ;;
  esac
done
if (( HIMLOCO_RENDER )); then
  HIMLOCO_TRAINING="$(pgrep -af '(^|[ /])train[^ /]*\.py([[:space:]]|$)' || true)"
  if [[ -n "$HIMLOCO_TRAINING" ]]; then
    echo "检测到训练进程，暂不启动图形回放；训练结束后重新运行。" >&2
    echo "$HIMLOCO_TRAINING" >&2
    exit 1
  fi
fi

mkdir -p "$HIMLOCO_RUNTIME/tmp"
export TMPDIR="$HIMLOCO_RUNTIME/tmp"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "$HIMLOCO_PYTHON" -u "$HIMLOCO_SOURCE/mujoco/w1w_2/play_gui.py" "$@"
