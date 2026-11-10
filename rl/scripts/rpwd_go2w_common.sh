#!/usr/bin/env bash
# 独立命名空间复用已安装依赖，输出与缓存均写可写数据盘。
set -euo pipefail
RPWD_GO2W_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RPWD_GO2W_ROOT="$(cd "$RPWD_GO2W_SCRIPT_DIR/../.." && pwd)"
export RPWD_GO2W_DATA="${RPWD_GO2W_DATA:-/media/rmer/系统/rpwd_training/rpwd_go2w_supermimic}"
export RPWD_GO2W_PYTHON="${RPWD_GO2W_PYTHON:-/media/rmer/data/rpwd_training/envs/go2w-supermimic/bin/python}"
export PYTHONPATH="$RPWD_GO2W_ROOT/rl:$RPWD_GO2W_SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
export WARP_CACHE_PATH="$RPWD_GO2W_DATA/cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$RPWD_GO2W_DATA/cache/inductor"
export TRITON_CACHE_DIR="$RPWD_GO2W_DATA/cache/triton"
export CUDA_CACHE_PATH="$RPWD_GO2W_DATA/cache/cuda"
export TMPDIR="$RPWD_GO2W_DATA/cache/tmp"
if [[ "$RPWD_GO2W_DATA" == /media/rmer/系统/* ]] && ! mountpoint -q /media/rmer/系统; then
  echo "输出磁盘尚未挂载：/media/rmer/系统；请先挂载或设置RPWD_GO2W_DATA。" >&2
  exit 1
fi
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
