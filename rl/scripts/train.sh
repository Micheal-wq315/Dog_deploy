#!/usr/bin/env bash
# 训练 RP_wd 平地速度跟踪策略 (便捷入口, 内部调用 isaaclab.sh -p)
#
#   ./rl/scripts/train.sh                     # 默认 2048 环境, 3000 迭代, headless
#   ./rl/scripts/train.sh --num_envs 4096     # 覆盖参数
#   ./rl/scripts/train.sh --resume            # 续训最新 checkpoint
set -euo pipefail

# shellcheck source=./_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

# 注: train.py 内部也会强制 headless, 这里显式写出只是为了让意图一眼可见
set -- --task Isaac-Velocity-Flat-RPWD-v0 --headless "$@"

# 训练期间**阻止系统休眠/待机**: Isaac Sim 在挂起-恢复后必挂 (2026-10-10 实测:
# 一次夜间休眠把跑着的训练直接冻死, GPU 空转 12 小时). systemd-inhibit 会在本进程存活期间生效.
if command -v systemd-inhibit >/dev/null 2>&1; then
    exec systemd-inhibit --what=idle:sleep:handle-lid-switch \
        --why="RP_wd 强化学习训练 (Isaac Sim 挂起后会死锁)" \
        "${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/train.py" "$@"
fi
exec "${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/train.py" "$@"
