#!/usr/bin/env bash
# 回放SuperMimic蒸馏的1740→16 Odin深度学生。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"
ensure_isaac_sim_idle
exec "${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/play_supermimic.py" "$@"
