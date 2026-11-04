#!/usr/bin/env bash
# 完整点足盲策略：多方向/停止、上下阶/台面/斜坡/碎石，按表现晋级。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/train_pointfoot_supermimic.sh" \
    --task Isaac-Velocity-Rough-RPWD-Pointfoot-SuperMimic-Full-v0 \
    --num_envs 1024 --iterations 40000 --steps_per_iteration 1 \
    --buffer_len 512 --wm_window 8 --wm_batch 1024 --wm_updates 4 \
    --imag_batch 1024 --horizon 32 --warmup 300 \
    --archive_windows 4096 --success_archive_windows 2048 \
    --wm_diagnostics_interval 500 --diagnostic_batch 64 \
    --critic_epochs 5 --critic_minibatches 4 \
    --actor_ramp_iterations 1000 --actor_hold_iterations 20000 \
    --lr_decay_iterations 20000 --lr_final_frac 0.2 \
    --save_interval 500 --log_interval 50 --seed 42 \
    --output_dir /media/rmer/data/rpwd_training/pointfoot_supermimic_full \
    --run_name 大狗点足版_完整任务_v3 "$@"
