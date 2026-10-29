#!/usr/bin/env bash
# sim2real 契约复核: 训练侧 (Isaac Lab) 与部署侧 (src/inference) 接口是否一致
# 改过奖励/观测/动作/部署 YAML 之后都应该跑一遍
#
#   ./rl/scripts/verify_contract.sh              # 静态 + 运行时检查 (~2 分钟)
#   ./rl/scripts/verify_contract.sh --no-sim     # 只跑静态检查 (秒级)
set -euo pipefail

# shellcheck source=./_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
if [[ " $* " != *" --no-sim "* ]]; then
    ensure_isaac_sim_idle
fi

VERIFY_LOG="$(mktemp "${TMPDIR:-/tmp}/rpwd-contract.XXXXXX.log")"
trap 'rm -f "${VERIFY_LOG}"' EXIT

# Isaac Lab's launcher can return status 0 after a Python traceback. Require the
# verifier's explicit PASS marker as well as a successful process exit.
set +e
"${ISAACLAB_PATH}/isaaclab.sh" -p "${SCRIPT_DIR}/verify_contract.py" "$@" 2>&1 | tee "${VERIFY_LOG}"
VERIFY_STATUS=${PIPESTATUS[0]}
set -e
if [[ "${VERIFY_STATUS}" -ne 0 ]]; then
    exit "${VERIFY_STATUS}"
fi
if ! grep -Fq "PASS: 全部检查通过" "${VERIFY_LOG}"; then
    echo "[FAIL] Contract verifier did not report PASS; see output above." >&2
    exit 1
fi
