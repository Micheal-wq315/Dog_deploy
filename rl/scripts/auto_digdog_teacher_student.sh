#!/usr/bin/env bash
# 教师 20k 完成后自动短评测并接力 LiDAR 学生。
# 门槛：flat/4/6/8cm 不得 fallen_or_nonfinite，且 max_x 必须超过起点附近；
# 10/12/16cm、stones、tires 允许摔倒（教师本来就可能未攻克），但不允许整体评测失败。
# 用法:
#   TEACHER_RUN=/绝对路径/教师run ./rl/scripts/auto_digdog_teacher_student.sh
# 可选: TEACHER_STEP=20000 EVAL_SECONDS=20 GATE_MIN_X=2.0 STUDENT_ITERATIONS=20000
#       STUDENT_RUN=/绝对路径/学生run DRY_RUN=1
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/digdog_mjlab_common.sh"

TEACHER_RUN="${TEACHER_RUN:-$(ls -td "$DIGDOG_MJLAB_DATA"/*_digdog_scan_teacher 2>/dev/null | head -1 || true)}"
[[ -n "${TEACHER_RUN:-}" && -d "$TEACHER_RUN" ]] || { echo "找不到教师 run；用 TEACHER_RUN=... 指定" >&2; exit 1; }
TEACHER_STEP="${TEACHER_STEP:-20000}"
TEACHER_MODEL="$TEACHER_RUN/model_${TEACHER_STEP}.pt"
LOG="$TEACHER_RUN/auto_chain.log"
EVAL_LOG="$TEACHER_RUN/auto_eval.log"
STUDENT_LOG="$TEACHER_RUN/auto_student.log"
exec >>"$LOG" 2>&1

echo "[$(date '+%F %T')] 等待教师模型 $TEACHER_MODEL"
while [[ ! -f "$TEACHER_MODEL" ]]; do
  if ! pgrep -f "train_digdog_mjlab.py.*${TEACHER_RUN}" >/dev/null 2>&1; then
    echo "[$(date '+%F %T')] 教师进程已退出但没有 $TEACHER_MODEL，停止接力"
    exit 2
  fi
  sleep 60
done
echo "[$(date '+%F %T')] 教师模型已落盘，等待教师进程退出"
while pgrep -f "train_digdog_mjlab.py.*${TEACHER_RUN}" >/dev/null 2>&1; do
  sleep 10
done
echo "[$(date '+%F %T')] 教师进程已退出，开始短评测"

LANES=(flat stairs4 stairs6 stairs8 stairs10 stairs12 stairs16 stones tires)
if ! "$DIR/play_digdog_mjlab.sh" --mode evaluate --checkpoint "$TEACHER_MODEL" \
    --lanes "${LANES[@]}" --seconds "${EVAL_SECONDS:-20}" --seeds 0 >"$EVAL_LOG" 2>&1; then
  echo "[$(date '+%F %T')] 短评测命令失败，保留教师，不启动学生（见 $EVAL_LOG）"
  exit 3
fi

if ! "$DIGDOG_MJLAB_PYTHON" - "$EVAL_LOG" <<'PY'
import json, os, re, sys

rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    m = re.search(r"\[能力测试\]\s*(\{.*\})", line)
    if m:
        try:
            rows.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
if not rows:
    print("评测没有解析到任何 lane 结果")
    sys.exit(4)

min_x = float(os.environ.get("GATE_MIN_X", "2.0"))
gate_lanes = ("flat", "stairs4", "stairs6", "stairs8")
bad = []
for r in rows:
    lane, reason = r.get("lane"), r.get("reason")
    mx = float(r.get("max_x") or 0.0)
    print(f"{lane}: reason={reason} max_x={mx:.2f}")
    if lane in gate_lanes:
        if reason == "fallen_or_nonfinite":
            bad.append(f"{lane}:nonfinite")
        if mx < min_x:
            bad.append(f"{lane}:max_x={mx:.2f}")
if bad:
    print("门槛失败: " + ", ".join(bad))
    sys.exit(4)
print("门槛通过")
PY
then
  echo "[$(date '+%F %T')] 评测门槛失败，保留教师，不启动学生（见 $EVAL_LOG）"
  exit 4
fi

export TEACHER="$TEACHER_MODEL"
export DIGDOG_MJLAB_RUN="${STUDENT_RUN:-$DIGDOG_MJLAB_DATA/$(date +%Y-%m-%d_%H-%M-%S)_digdog_lidar_student}"
echo "[$(date '+%F %T')] 门槛通过，启动 LiDAR 学生: $DIGDOG_MJLAB_RUN"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[$(date '+%F %T')] DRY_RUN=1，不实际启动学生"
  exit 0
fi
if ! "$DIR/train_digdog_lidar_student.sh" --iterations "${STUDENT_ITERATIONS:-20000}" >"$STUDENT_LOG" 2>&1; then
  echo "[$(date '+%F %T')] LiDAR 学生训练失败（见 $STUDENT_LOG）"
  exit 5
fi
echo "[$(date '+%F %T')] LiDAR 学生训练完成: $DIGDOG_MJLAB_RUN"
