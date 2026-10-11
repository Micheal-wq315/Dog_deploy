#!/usr/bin/env bash
# 测试训练好的策略 (固定指令 / 指令扫描 / 随机指令), 默认加载最新 checkpoint
#
#   ./rl/scripts/test_policy.sh                              # GUI, 随机指令, 实时打印跟踪指标
#   ./rl/scripts/test_policy.sh --cmd 0.5 0 0                # GUI, 固定 vx=0.5 m/s
#   ./rl/scripts/test_policy.sh --sweep                      # 扫指令表, 输出跟踪误差汇总
#   ./rl/scripts/test_policy.sh --sweep --headless           # 无窗口跑扫描 (适合脚本化)
set -euo pipefail

# shellcheck source=./_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

exec "${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/test_policy.py" "$@"
