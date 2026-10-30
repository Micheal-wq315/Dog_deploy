#!/usr/bin/env bash
# 顺序执行18000轮基础地形教师、2000轮学生，支持中断恢复与阶段诊断模式。
set -euo pipefail
if [[ "${RPWD_ODIN_PIPELINE_INHIBITED:-0}" != 1 ]] && command -v systemd-inhibit >/dev/null 2>&1; then
    exec systemd-inhibit --what=idle:sleep:handle-lid-switch --why="Odin教师、验收与学生完整训练流程" \
        env RPWD_ODIN_PIPELINE_INHIBITED=1 bash "$0" "$@"
fi
export TERM="${TERM:-xterm-256color}"
[[ "${TERM}" == dumb ]] && export TERM=xterm-256color
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
RESUME_TEACHER=""
CONTINUE_STAGE_FAILURE=0
TRAIN_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --resume-teacher-checkpoint)
            [[ "$#" -ge 2 ]] || { echo "缺少教师checkpoint路径" >&2; exit 1; }
            RESUME_TEACHER="$2"; shift 2 ;;
        --continue-on-stage-failure)
            CONTINUE_STAGE_FAILURE=1; shift ;;
        --resume|--load_run*|--checkpoint*|--teacher_checkpoint*|--task*|--max_iterations*|--experiment_name*|--run_name*)
            echo "完整流程固定18000+2000轮（合计20000）；分阶段启动/续训请使用train_odin_depth.sh，见rl/CURRICULUM.md。" >&2
            exit 1 ;;
        *) TRAIN_ARGS+=("$1"); shift ;;
    esac
done
set -- "${TRAIN_ARGS[@]}"
RUN_LABEL="odin_basic_v20_$(date +%Y%m%d_%H%M%S)"

# 2000/6000轮验平地，12000轮加入上下阶；默认失败停止，诊断模式可继续。
TEACHER_RESUME=()
COMPLETED_TEACHER=0
if [[ -n "${RESUME_TEACHER}" ]]; then
    RESUME_TEACHER="$(realpath "${RESUME_TEACHER}")"
    COMPLETED_TEACHER="$(python - "${RESUME_TEACHER}" <<'PY'
import sys
from pathlib import Path
import torch
path = Path(sys.argv[1])
state = torch.load(path, map_location="cpu", weights_only=False)
if "actor_state_dict" not in state:
    raise SystemExit("续训必须使用PPO教师checkpoint。")
completed = int(state["iter"]) + 1
if not 0 < completed <= 18000 or path.name != f"model_{completed - 1}.pt":
    raise SystemExit("教师迭代记录与文件名不匹配或已超过本流程目标。")
print(completed)
PY
)"
    TEACHER_CHECKPOINT="${RESUME_TEACHER}"
    TEACHER_RUN_DIR="$(dirname "${TEACHER_CHECKPOINT}")"
    TEACHER_RESUME=(--resume --load_run "${TEACHER_RUN_DIR}" --checkpoint "$(basename "${TEACHER_CHECKPOINT}")")
    echo "[INFO] 从已完成${COMPLETED_TEACHER}轮的教师接续，总目标18000+2000=20000。"
fi
for TARGET in 2000 6000 12000 18000; do
    [[ "${TARGET}" -le "${COMPLETED_TEACHER}" ]] && continue
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
if [[ "${TARGET}" != 18000 ]]; then
    STAGE_EVAL="${TEACHER_RUN_DIR}/stage_gate_seed42"
    "${SCRIPT_DIR}/gui_test.sh" --obstacle-benchmark --headless \
        --checkpoint "${TEACHER_CHECKPOINT}" --seed 42 --num_envs 12 --save-dir "${STAGE_EVAL}"
    SCENARIOS=(flat_reference)
    [[ "${TARGET}" == 12000 ]] && SCENARIOS+=(step_up_06cm step_down_08cm)
    if ! python "${SCRIPT_DIR}/qualify_odin_teacher.py" "${STAGE_EVAL}" \
        --scenarios "${SCENARIOS[@]}" --output "${TEACHER_RUN_DIR}/stage_gate.json"; then
        # 只允许完整评测给出的性能不达标继续；异常/缺失结果仍停止。
        [[ "${CONTINUE_STAGE_FAILURE}" == 1 && -s "${TEACHER_RUN_DIR}/stage_gate.json" ]] || exit 1
        echo "[WARN] 阶段性能未达标，按长期训练模式保存诊断并继续；最终教师资格门槛仍生效。"
    fi
fi
TEACHER_RESUME=(--resume --load_run "${TEACHER_RUN_DIR}" --checkpoint "$(basename "${TEACHER_CHECKPOINT}")")
done
EVAL_DIRS=()
for EVAL_SEED in 42 43 44; do
    EVAL_DIR="${TEACHER_RUN_DIR}/qualification_${RUN_LABEL}_seed${EVAL_SEED}"
    "${SCRIPT_DIR}/gui_test.sh" --obstacle-benchmark --headless \
        --checkpoint "${TEACHER_CHECKPOINT}" --seed "${EVAL_SEED}" --num_envs 12 --save-dir "${EVAL_DIR}"
    EVAL_DIRS+=("${EVAL_DIR}")
done
# 基础专项：平地、6cm上阶、8cm下阶；12cm箱与绕行保留诊断，不作基础达标声明。
python "${SCRIPT_DIR}/qualify_odin_teacher.py" "${EVAL_DIRS[@]}" \
    --scenarios flat_reference step_up_06cm step_down_08cm --output "${TEACHER_RUN_DIR}/qualification.json"
"${SCRIPT_DIR}/train_odin_depth.sh" "$@" \
    --task Isaac-Velocity-Rough-RPWD-OdinDepth-Student-v0 \
    --max_iterations 2000 --run_name "${RUN_LABEL}" --teacher_checkpoint "${TEACHER_CHECKPOINT}"
