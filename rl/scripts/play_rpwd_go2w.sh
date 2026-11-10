#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/rpwd_go2w_common.sh"
export CUDA_VISIBLE_DEVICES="" MUJOCO_GL=glfw
exec "$RPWD_GO2W_PYTHON" "$RPWD_GO2W_SCRIPT_DIR/play_rpwd_go2w.py" "$@"
