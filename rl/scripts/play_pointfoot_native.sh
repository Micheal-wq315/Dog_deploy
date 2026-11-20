#!/usr/bin/env bash
# 点足版原生 MuJoCo 键盘测试（CPU，不占用 GPU 训练）。
# 自动选择最新的 pointfoot checkpoint；可用 --checkpoint 覆盖。
# 例：./rl/scripts/play_pointfoot_native.sh --lane stones --scan-heatmap
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${POINTFOOT_MJLAB_PYTHON:-/media/rmer/data/rpwd_training/envs/go2w-supermimic/bin/python}"
DATA="${POINTFOOT_MJLAB_DATA:-/media/rmer/data/rpwd_training/pointfoot_mjlab}"
export PYTHONPATH="$ROOT/rl:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"

if [[ " $* " != *" --checkpoint "* ]]; then
  CKPT="$(find "$DATA" -maxdepth 3 -name 'model_*.pt' -printf '%T@ %p\n' 2>/dev/null \
          | sort -n | tail -1 | cut -d' ' -f2- || true)"
  if [[ -z "$CKPT" ]]; then
    echo "没有找到 pointfoot checkpoint：$DATA" >&2
    echo "请用 --checkpoint /绝对路径/model_xxxxx.pt 指定" >&2
    exit 1
  fi
  echo "[自动选择] $CKPT" >&2
  set -- --checkpoint "$CKPT" "$@"
fi

exec "$PY" "$SCRIPT_DIR/pointfoot_mujoco.py" "$@"
