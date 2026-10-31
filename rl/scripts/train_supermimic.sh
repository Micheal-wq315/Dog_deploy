#!/usr/bin/env bash
# RP_wd SuperMimic模型式训练入口，自动 headless、串行保护并阻止系统休眠。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
ensure_isaac_sim_idle

set -- --headless "$@"
if command -v systemd-inhibit >/dev/null 2>&1; then
    exec systemd-inhibit --what=idle:sleep:handle-lid-switch \
        --why="RP_wd SuperMimic世界模型强化学习" \
        "${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/train_supermimic.py" "$@"
fi
exec "${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/train_supermimic.py" "$@"
