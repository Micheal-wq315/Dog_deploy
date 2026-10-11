#!/usr/bin/env bash
# 回放训练好的策略并导出 ONNX (导出到 rl/exported/policy_flat.onnx)
#
#   ./rl/scripts/play.sh                                         # 图形界面回放 + 导出
#   ./rl/scripts/play.sh --headless --video --video_length 300    # 无显示器: 录一段视频
set -euo pipefail

# shellcheck source=./_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

exec "${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/play.py" "$@"
