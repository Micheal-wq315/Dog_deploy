#!/usr/bin/env bash
# DigDog 全高度台阶键盘测试：1=flat 2=4cm 3=6cm 4=8cm 5=10cm 6=12cm 7=16cm 8=ramp
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/digdog_mjlab_common.sh"
RUN="/media/rmer/data/rpwd_training/digdog_mjlab/2026-11-16_大狗DigDogv1点足版_60k_seed0_v2"
if [[ -n "${1:-}" ]]; then
  CKPT="$1"
else
  CKPT="$(ls -t "$RUN"/model_*.pt | head -1)"
fi
echo "模型: $CKPT"
echo "按键: 1=flat 2=4cm 3=6cm 4=8cm 5=10cm 6=12cm 7=16cm 8=ramp"
echo "方向键前后/横移，Q/E转向，Space停止，R复位，K暂停，Esc退出"
exec "$SCRIPT_DIR/play_digdog_mjlab.sh" --mode play --checkpoint "$CKPT" --lane stairs4 --speed .2
