#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/pointfoot_mjlab_common.sh"
export CUDA_VISIBLE_DEVICES="" MUJOCO_GL=glfw
exec "$POINTFOOT_MJLAB_PYTHON" "$POINTFOOT_MJLAB_SCRIPT_DIR/play_pointfoot_mjlab.py" "$@"
