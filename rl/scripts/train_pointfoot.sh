#!/usr/bin/env bash
# 启动 RP_wd Go2 点足版台阶/崎岖地形 PPO 训练。
set -euo pipefail

if [[ "${TERM:-dumb}" == "dumb" ]]; then
    export TERM=xterm-256color
fi

# shellcheck source=./_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
ensure_isaac_sim_idle

set -- --task Isaac-Velocity-Rough-RPWD-Pointfoot-v0 --headless "$@"

if command -v systemd-inhibit >/dev/null 2>&1; then
    exec systemd-inhibit --what=idle:sleep:handle-lid-switch \
        --why="大狗点足版 Isaac Sim 强化学习训练" \
        "${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/train.py" "$@"
fi
exec "${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/train.py" "$@"
