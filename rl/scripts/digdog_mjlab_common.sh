#!/usr/bin/env bash
# 独立命名空间复用已安装依赖，输出与缓存均写可写数据盘。
set -euo pipefail
DIGDOG_MJLAB_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIGDOG_MJLAB_ROOT="$(cd "$DIGDOG_MJLAB_SCRIPT_DIR/../.." && pwd)"
export DIGDOG_MJLAB_DATA="${DIGDOG_MJLAB_DATA:-/media/rmer/data/rpwd_training/digdog_mjlab}"
export DIGDOG_MJLAB_PYTHON="${DIGDOG_MJLAB_PYTHON:-/media/rmer/data/rpwd_training/envs/go2w-supermimic/bin/python}"
export PYTHONPATH="$DIGDOG_MJLAB_ROOT/rl:$DIGDOG_MJLAB_SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
export WARP_CACHE_PATH="$DIGDOG_MJLAB_DATA/cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$DIGDOG_MJLAB_DATA/cache/inductor"
export TRITON_CACHE_DIR="$DIGDOG_MJLAB_DATA/cache/triton"
export CUDA_CACHE_PATH="$DIGDOG_MJLAB_DATA/cache/cuda"
export TMPDIR="$DIGDOG_MJLAB_DATA/cache/tmp"
if [[ "$DIGDOG_MJLAB_DATA" == /media/rmer/系统/* ]] && ! mountpoint -q /media/rmer/系统; then
  echo "输出磁盘尚未挂载：/media/rmer/系统；请先挂载或设置DIGDOG_MJLAB_DATA。" >&2
  exit 1
fi
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
