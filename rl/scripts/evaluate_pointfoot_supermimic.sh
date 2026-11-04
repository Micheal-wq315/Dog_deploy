#!/usr/bin/env bash
# 单实例保护的确定性验收；所有case在同一个headless实例内顺序执行。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
export PYTHONPATH="$(cd "$SCRIPT_DIR/.." && pwd):${PYTHONPATH:-}"
ensure_isaac_sim_idle
exec "$ISAACLAB_PATH/isaaclab.sh" -p "$SCRIPT_DIR/evaluate_pointfoot_supermimic.py" --headless "$@"
