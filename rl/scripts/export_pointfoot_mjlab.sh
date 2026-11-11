#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/pointfoot_mjlab_common.sh"
export CUDA_VISIBLE_DEVICES=''
exec "$POINTFOOT_MJLAB_PYTHON" "$POINTFOOT_MJLAB_SCRIPT_DIR/export_pointfoot_mjlab.py" "$@"
