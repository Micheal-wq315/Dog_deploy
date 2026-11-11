#!/usr/bin/env bash
# 独立命名空间复用已安装依赖，输出与缓存均写可写数据盘。
set -euo pipefail
POINTFOOT_MJLAB_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POINTFOOT_MJLAB_ROOT="$(cd "$POINTFOOT_MJLAB_SCRIPT_DIR/../.." && pwd)"
export POINTFOOT_MJLAB_DATA="${POINTFOOT_MJLAB_DATA:-/media/rmer/系统/rpwd_training/pointfoot_mjlab}"
export POINTFOOT_MJLAB_PYTHON="${POINTFOOT_MJLAB_PYTHON:-/media/rmer/data/rpwd_training/envs/go2w-supermimic/bin/python}"
export PYTHONPATH="$POINTFOOT_MJLAB_ROOT/rl:$POINTFOOT_MJLAB_SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
export WARP_CACHE_PATH="$POINTFOOT_MJLAB_DATA/cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$POINTFOOT_MJLAB_DATA/cache/inductor"
export TRITON_CACHE_DIR="$POINTFOOT_MJLAB_DATA/cache/triton"
export CUDA_CACHE_PATH="$POINTFOOT_MJLAB_DATA/cache/cuda"
export TMPDIR="$POINTFOOT_MJLAB_DATA/cache/tmp"
if [[ "$POINTFOOT_MJLAB_DATA" == /media/rmer/系统/* ]] && ! mountpoint -q /media/rmer/系统; then
  echo "输出磁盘尚未挂载：/media/rmer/系统；请先挂载或设置POINTFOOT_MJLAB_DATA。" >&2
  exit 1
fi
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
