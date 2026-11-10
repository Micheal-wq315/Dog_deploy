#!/usr/bin/env bash
# 原生MuJoCo CPU控制和评测；不创建第二个Warp GPU或Isaac Sim实例。
set -euo pipefail
export GO2W_CACHE="${GO2W_CACHE:-/media/rmer/系统/rpwd_training/go2w_supermimic/cache}"
source "$(cd "$(dirname "$0")" && pwd)/go2w_reference_common.sh"
export CUDA_VISIBLE_DEVICES=""
export MUJOCO_GL=glfw
exec "$GO2W_PYTHON" "$GO2W_SCRIPT_DIR/go2w_mujoco.py" "$@"
