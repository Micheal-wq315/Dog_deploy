#!/usr/bin/env bash
# 使用原Go2W v7源码从零训练，保留4096环境，只下调想象批量。
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/go2w_reference_common.sh"
ensure_go2w_idle
cd "$GO2W_SOURCE"
GO2W_RUN="${GO2W_RUN:-/media/rmer/data/rpwd_training/go2w_supermimic/$($GO2W_PYTHON -c 'from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S"))')_v7_seed0}"
set -- "$GO2W_PYTHON" scripts/train.py \
  --terrain --terrain-revision 7 --command-curriculum \
  --num-envs 4096 --iterations 20000 --num-steps-per-env 1 --buffer-len 512 \
  --wm-batch 2048 --wm-window 8 --wm-updates 4 \
  --imag-batch 4096 --horizon 32 --critic-epochs 5 --critic-minibatches 4 \
  --prioritized-replay --compile-imagination --warmup 300 \
  --actor-ramp-iterations 1000 --actor-hold-iterations 20000 \
  --lr-decay-iterations 20000 --lr-final-frac .2 \
  --save-interval 500 --log-interval 50 --seed 0 --output "$GO2W_RUN" "$@"
echo "[Go2W] 原始源码：$GO2W_SOURCE"
echo "[Go2W] Python：$GO2W_PYTHON"
echo "[Go2W] 默认输出：$GO2W_RUN；后续--output参数可覆盖"
if command -v systemd-inhibit >/dev/null 2>&1; then
  exec systemd-inhibit --what=idle:sleep:handle-lid-switch --why="Go2W原版SuperMimic训练" "$@"
fi
exec "$@"
