#!/usr/bin/env bash
# 顺序执行8000轮清晰深度教师、2000轮含噪学生；任一阶段失败立即停止。
set -euo pipefail
if [[ "${RPWD_ODIN_PIPELINE_INHIBITED:-0}" != 1 ]] && command -v systemd-inhibit >/dev/null 2>&1; then
    exec systemd-inhibit --what=idle:sleep:handle-lid-switch --why="Odin教师、验收与学生完整训练流程" \
        env RPWD_ODIN_PIPELINE_INHIBITED=1 bash "$0" "$@"
fi
export TERM="${TERM:-xterm-256color}"
[[ "${TERM}" == dumb ]] && export TERM=xterm-256color
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
for arg in "$@"; do
    case "${arg}" in
        --resume|--load_run*|--checkpoint*|--teacher_checkpoint*|--task*|--max_iterations*|--experiment_name*|--run_name*)
            echo "完整流程固定8000+2000轮（合计10000）；分阶段启动/续训请使用train_odin_depth.sh，见rl/CURRICULUM.md。" >&2
            exit 1 ;;
    esac
done
RUN_LABEL="odin_v18_$(date +%Y%m%d_%H%M%S)"

# 前1000轮先验平地；3000轮再验6cm上下阶。未达标就停，避免盲跑满8000轮。
TEACHER_RESUME=()
for TARGET in 1000 3000 8000; do
    STAGE_LABEL="${RUN_LABEL}_teacher${TARGET}"
    "${SCRIPT_DIR}/train_odin_depth.sh" "$@" \
        --task Isaac-Velocity-Rough-RPWD-OdinDepth-Teacher-v0 \
        --max_iterations "${TARGET}" --run_name "${STAGE_LABEL}" "${TEACHER_RESUME[@]}"

# 只接受本次完整结束的教师记录，避免误拿旧实验或半写入的模型。
TEACHER_CHECKPOINT="$(python - "${SCRIPT_DIR}/../logs/rsl_rl/rp_wd_odin_teacher" "${STAGE_LABEL}" "${TARGET}" <<'PY'
import json
import sys
from pathlib import Path
records = list(Path(sys.argv[1]).glob(f"*_{sys.argv[2]}/training_result.json"))
if len(records) != 1:
    raise SystemExit("没有本次完整教师训练记录，停止学生阶段。")
result = json.loads(records[0].read_text())
if result["completed_iterations"] != int(sys.argv[3]) or not Path(result["checkpoint"]).is_file():
    raise SystemExit("教师训练未达到本阶段目标，停止后续阶段。")
print(result["checkpoint"])
PY
)"
TEACHER_RUN_DIR="$(dirname "${TEACHER_CHECKPOINT}")"
if [[ "${TARGET}" != 8000 ]]; then
    STAGE_EVAL="${TEACHER_RUN_DIR}/stage_gate_seed42"
    "${SCRIPT_DIR}/gui_test.sh" --obstacle-benchmark --headless \
        --checkpoint "${TEACHER_CHECKPOINT}" --seed 42 --num_envs 12 --save-dir "${STAGE_EVAL}"
    SCENARIOS=(flat_reference)
    [[ "${TARGET}" == 3000 ]] && SCENARIOS+=(step_up_06cm step_down_08cm)
    python "${SCRIPT_DIR}/qualify_odin_teacher.py" "${STAGE_EVAL}" \
        --scenarios "${SCENARIOS[@]}" --output "${TEACHER_RUN_DIR}/stage_gate.json"
fi
TEACHER_RESUME=(--resume --load_run "${TEACHER_RUN_DIR}" --checkpoint "$(basename "${TEACHER_CHECKPOINT}")")
done
EVAL_DIRS=()
for EVAL_SEED in 42 43 44; do
    EVAL_DIR="${TEACHER_RUN_DIR}/qualification_seed${EVAL_SEED}"
    "${SCRIPT_DIR}/gui_test.sh" --obstacle-benchmark --headless \
        --checkpoint "${TEACHER_CHECKPOINT}" --seed "${EVAL_SEED}" --num_envs 12 --save-dir "${EVAL_DIR}"
    EVAL_DIRS+=("${EVAL_DIR}")
done
# 平地/上阶/下阶/矮障碍首次尝试完成率>=80%、低接触完成率>=60%才进入蒸馏。
# 墙和交错绕行先保留诊断结果，不把局部速度策略宣称为完整导航系统。
python "${SCRIPT_DIR}/qualify_odin_teacher.py" "${EVAL_DIRS[@]}" --output "${TEACHER_RUN_DIR}/qualification.json"
"${SCRIPT_DIR}/train_odin_depth.sh" "$@" \
    --task Isaac-Velocity-Rough-RPWD-OdinDepth-Student-v0 \
    --max_iterations 2000 --run_name "${RUN_LABEL}" --teacher_checkpoint "${TEACHER_CHECKPOINT}"
