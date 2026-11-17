#!/usr/bin/env bash
# DigDogv1 Mid-360 LiDAR 学生：从特权教师热启动，冻结教师做动作蒸馏 + 自身 RL。
# 学生读取 273 维局部高度图；训练/部署均按 update_hz=10 保持，未知格填 -2.0。
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/digdog_mjlab_common.sh"
if [[ -z "${TEACHER:-}" ]]; then
  TEACHER="$(ls -t "$DIGDOG_MJLAB_DATA"/*digdog_scan_teacher/model_*.pt 2>/dev/null | head -n1 || true)"
fi
[[ -n "${TEACHER:-}" && -f "$TEACHER" ]] || {
  echo "找不到教师 checkpoint；用 TEACHER=/绝对路径/model_xxxxx.pt 指定" >&2
  exit 1
}
export DIGDOG_MJLAB_RUN="${DIGDOG_MJLAB_RUN:-$DIGDOG_MJLAB_DATA/$(date +%Y-%m-%d_%H-%M-%S)_digdog_lidar_student}"
exec "$DIR/train_digdog_mjlab.sh" \
  --scan lidar --scan-hidden 256 128 64 --teacher "$TEACHER" --warm-start "$TEACHER" \
  --distill-coef 1.0 --iterations 20000 --save-interval 500 --log-interval 50 "$@"
