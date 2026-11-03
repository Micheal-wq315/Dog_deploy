#!/usr/bin/env bash
# 点足 SuperMimic GUI 回放；必须先确认没有训练或其它仿真实例。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_common.sh
source "$SCRIPT_DIR/_common.sh"
# 隔离 worktree 必须优先导入自己的训练包，不能落回主目录的 editable 安装。
export PYTHONPATH="$(cd "$SCRIPT_DIR/.." && pwd):${PYTHONPATH:-}"
ensure_isaac_sim_idle

"$ISAACLAB_PATH/isaaclab.sh" -p \
    "$SCRIPT_DIR/play_pointfoot_supermimic.py" "$@"
