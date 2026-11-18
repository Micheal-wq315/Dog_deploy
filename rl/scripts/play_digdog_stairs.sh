#!/usr/bin/env bash
# DigDog 全地形键盘测试：1=flat 2=4cm 3=6cm 4=8cm 5=10cm 6=12cm 7=16cm 8=ramp 9=stones 0=tires
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/digdog_mjlab_common.sh"
if [[ -n "${1:-}" ]]; then
  CKPT="$1"
else
  # 不自动选 LiDAR 学生的早期 checkpoint（自动接力可能留下未收敛存档）；
  # 要测学生必须显式传路径。
  CKPT="$(ls -t "$DIGDOG_MJLAB_DATA"/*/model_*.pt 2>/dev/null | grep -v '_digdog_lidar_student/' | head -1)"
fi
[[ -f "$CKPT" ]] || { echo "找不到模型：$CKPT" >&2; exit 1; }
echo "模型: $CKPT"
echo "按键: 1=flat 2=4cm 3=6cm 4=8cm 5=10cm 6=12cm 7=16cm 8=ramp 9=stones 0=tires"
echo "方向键前后/横移，Q/E转向，Space停止，R复位，K暂停，Esc退出"
exec "$SCRIPT_DIR/play_digdog_mjlab.sh" --mode play --checkpoint "$CKPT" --lane stairs4 --speed .2
