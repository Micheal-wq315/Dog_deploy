#!/usr/bin/env bash
# 点足版 SuperMimic 教师训练：headless、单实例保护、休眠抑制。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./_common.sh
source "$SCRIPT_DIR/_common.sh"
ensure_isaac_sim_idle

set -- --headless "$@"
if command -v systemd-inhibit >/dev/null 2>&1; then
    exec systemd-inhibit --what=idle:sleep:handle-lid-switch \
        --why="RP_wd 点足 SuperMimic 世界模型训练" \
        "$ISAACLAB_PATH/isaaclab.sh" -p \
        "$SCRIPT_DIR/train_pointfoot_supermimic.py" "$@"
fi
exec "$ISAACLAB_PATH/isaaclab.sh" -p \
    "$SCRIPT_DIR/train_pointfoot_supermimic.py" "$@"
