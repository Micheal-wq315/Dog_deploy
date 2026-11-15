#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/digdog_mjlab_common.sh"
export CUDA_VISIBLE_DEVICES="" MUJOCO_GL=glfw
exec "$DIGDOG_MJLAB_PYTHON" "$DIGDOG_MJLAB_SCRIPT_DIR/play_digdog_mjlab.py" "$@"
