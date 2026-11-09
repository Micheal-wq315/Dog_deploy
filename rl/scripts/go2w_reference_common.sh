#!/usr/bin/env bash
# Go2W原仓库的独立环境与数据盘缓存；不激活或修改Isaac Lab环境。
set -euo pipefail
GO2W_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GO2W_WORKTREE="$(cd "$GO2W_SCRIPT_DIR/../.." && pwd)"
GO2W_MAIN="$(dirname "$(git -C "$GO2W_WORKTREE" rev-parse --path-format=absolute --git-common-dir)")"
export GO2W_SOURCE="${GO2W_SOURCE:-$GO2W_MAIN/go2w-supermimic}"
export GO2W_PYTHON="${GO2W_PYTHON:-/media/rmer/data/rpwd_training/envs/go2w-supermimic/bin/python}"
if [[ ! -f "$GO2W_SOURCE/scripts/train.py" || ! -x "$GO2W_PYTHON" ]]; then
  echo "缺少Go2W源码或独立Python环境；请检查GO2W_SOURCE和GO2W_PYTHON。" >&2
  exit 1
fi
export PYTHONPATH="$GO2W_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
GO2W_CACHE="${GO2W_CACHE:-/media/rmer/data/rpwd_training/go2w_supermimic/cache}"
export WARP_CACHE_PATH="$GO2W_CACHE/warp"
export TORCHINDUCTOR_CACHE_DIR="$GO2W_CACHE/inductor"
export TRITON_CACHE_DIR="$GO2W_CACHE/triton"
export CUDA_CACHE_PATH="$GO2W_CACHE/cuda"
export TMPDIR="$GO2W_CACHE/tmp"
mkdir -p "$WARP_CACHE_PATH" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"

ensure_go2w_idle() {
  "$GO2W_PYTHON" - <<'PY'
from pathlib import Path
scripts = {'train.py', 'train_supermimic.py', 'train_pointfoot_supermimic.py',
           'play.py', 'play_supermimic.py', 'play_pointfoot_supermimic.py',
           'evaluate_pointfoot_supermimic.py', 'verify_contract.py',
           'play_gui.py', 'play_keyboard.py', 'evaluate_go2w_reference.py'}
active = []
for directory in Path('/proc').glob('[0-9]*'):
    try:
        argv = directory.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
        executable = Path(argv[0]).name
        if executable == 'kit' or ('python' in executable and any(Path(a).name in scripts for a in argv[1:])):
            active.append((directory.name, ' '.join(argv[:3])))
    except (OSError, IndexError):
        continue
if active:
    raise SystemExit('已有仿真进程，请先关闭后再启动Go2W：' + str(active))
PY
}
