#!/usr/bin/env bash
# 训练 Odin1 深度避障策略 (相机 + 粗糙地形 + 腿部力矩观测).
# 默认 256 并行环境、5000 PPO 迭代、无头运行，并在训练期间阻止系统休眠。
set -euo pipefail

# shellcheck source=./_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

set -- --task Isaac-Velocity-Rough-RPWD-OdinDepth-v0 --headless "$@"

if command -v systemd-inhibit >/dev/null 2>&1; then
    exec systemd-inhibit --what=idle:sleep:handle-lid-switch \
        --why="RP_wd Odin1 深度感知强化学习训练" \
        "${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/train.py" "$@"
fi
exec "${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/train.py" "$@"
