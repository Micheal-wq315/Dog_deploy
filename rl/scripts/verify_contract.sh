#!/usr/bin/env bash
# sim2real 契约复核: 训练侧 (Isaac Lab) 与部署侧 (src/inference) 接口是否一致
# 改过奖励/观测/动作/部署 YAML 之后都应该跑一遍
#
#   ./rl/scripts/verify_contract.sh              # 静态 + 运行时检查 (~2 分钟)
#   ./rl/scripts/verify_contract.sh --no-sim     # 只跑静态检查 (秒级)
set -euo pipefail

# shellcheck source=./_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

exec "${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/verify_contract.py" "$@"
