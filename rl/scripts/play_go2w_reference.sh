#!/usr/bin/env bash
# 浏览器键盘控制；默认展示作者随仓库提供的v7/20000轮模型。
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/go2w_reference_common.sh"
ensure_go2w_idle
cd "$GO2W_SOURCE"
exec "$GO2W_PYTHON" scripts/play_gui.py \
  --checkpoint "$GO2W_SOURCE/logs/go2w_terrain_v7_20000/model_20000.pt" \
  --terrain-kind up --terrain-level 1 --device cpu "$@"
