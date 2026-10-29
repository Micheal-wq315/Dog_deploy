#!/usr/bin/env bash
# 公共环境准备 (被 train.sh / play.sh / verify_contract.sh source)
#
# 说明: isaaclab.sh -p 用的是"当前激活的 conda 环境", 所以这里先确保切到 isaaclab 环境.
# 可用环境变量覆盖:
#   ISAACLAB_PATH  Isaac Lab 根目录 (默认 /media/rmer/data/IsaacLab)
#   CONDA_SH       conda 的 profile.d/conda.sh 路径 (默认 ~/miniconda3/etc/profile.d/conda.sh)
#   CONDA_ENV      conda 环境名 (默认 isaaclab)

ISAACLAB_PATH="${ISAACLAB_PATH:-/media/rmer/data/IsaacLab}"
if [[ "${TERM:-dumb}" == dumb ]]; then
    export TERM=xterm-256color
fi
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"

if [[ ! -x "${ISAACLAB_PATH}/isaaclab.sh" ]]; then
    echo "错误: 找不到 ${ISAACLAB_PATH}/isaaclab.sh (用 ISAACLAB_PATH 指定 Isaac Lab 目录)" >&2
    exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]]; then
    # shellcheck disable=SC1090
    # Isaac Lab 的 conda activate hook 会直接展开 PYTHONPATH; set -u 下先初始化空值。
    export PYTHONPATH="${PYTHONPATH:-}"
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ensure_isaac_sim_idle() {
    python - <<'PY'
from pathlib import Path
active = []
scripts = {"train.py", "test_policy.py", "play.py", "verify_contract.py"}
for directory in Path("/proc").glob("[0-9]*"):
    try:
        argv = directory.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0")
        executable = Path(argv[0]).name
        if executable == "kit" or ("python" in executable and any(Path(arg).name in scripts for arg in argv[1:])):
            active.append((directory.name, " ".join(argv[:4])))
    except (OSError, IndexError):
        continue
if active:
    raise SystemExit("已有 Isaac Sim 实例，停止启动第二个实例：" + str(active))
PY
}
