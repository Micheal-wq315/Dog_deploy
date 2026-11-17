#!/usr/bin/env bash
# DigDogv1 特权高度图教师：从 model_20000 热启动，Actor/Critic/世界模型沿用旧 checkpoint。
# 教师读取 TerrainField 的真实 273 维扫描；训练完成后供 LiDAR 学生蒸馏。
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/digdog_mjlab_common.sh"
BASE="${BASE:-$DIGDOG_MJLAB_DATA/2026-11-16_大狗DigDogv1点足版_60k_seed0_v2/model_20000.pt}"
[[ -f "$BASE" ]] || { echo "找不到教师热启动模型：$BASE（用 BASE=... 指定）" >&2; exit 1; }
export DIGDOG_MJLAB_RUN="${DIGDOG_MJLAB_RUN:-$DIGDOG_MJLAB_DATA/$(date +%Y-%m-%d_%H-%M-%S)_digdog_scan_teacher}"
# 8 类地形共 6958 个 geom；MuJoCo Warp 的静态 geom 缓冲约 25M，4096×6958
# 会静默丢掉地面接触（机器人自由落体）。3072×6958=21.4M 稳定。
exec "$DIR/train_digdog_mjlab.sh" \
  --num-envs 3072 --scan true --scan-hidden 256 128 64 --warm-start "$BASE" \
  --iterations 20000 --save-interval 500 --log-interval 50 "$@"
