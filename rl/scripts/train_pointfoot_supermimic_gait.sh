#!/usr/bin/env bash
# 低频完整步态：独立v4任务，默认从零学习，不继承旧碎步策略。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/train_pointfoot_supermimic_full.sh" \
  --task Isaac-Velocity-Rough-RPWD-Pointfoot-SuperMimic-Gait-v0 \
  --iterations 20000 --horizon 48 --wm_window 16 \
  --action_noise 0.08 --action_noise_final 0.015 --noise_decay_iterations 8000 \
  --output_dir /media/rmer/data/rpwd_training/pointfoot_supermimic_gait \
  --run_name 大狗点足版_低频完整步态_v4 "$@"
