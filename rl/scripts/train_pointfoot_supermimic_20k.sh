#!/usr/bin/env bash
# RTX 5070 Ti 16GB 的 Go2W 核心流程预设：12关节点足盲策略，20000轮。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/train_pointfoot_supermimic.sh" \
    --num_envs 1024 --iterations 20000 --steps_per_iteration 1 \
    --buffer_len 512 --wm_window 8 --wm_batch 1024 --wm_updates 4 \
    --imag_batch 1024 --horizon 32 --warmup 300 \
    --critic_epochs 5 --critic_minibatches 4 \
    --actor_ramp_iterations 1000 --actor_hold_iterations 20000 \
    --lr_decay_iterations 20000 --lr_final_frac 0.2 \
    --save_interval 500 --log_interval 50 --seed 42 \
    --output_dir /media/rmer/data/rpwd_training/pointfoot_supermimic \
    --run_name 大狗点足版_盲狗_Go2W对齐_20k "$@"
