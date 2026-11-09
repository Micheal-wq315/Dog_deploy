#!/usr/bin/env bash
# 从零训练完成后串行测试本机新模型并录像，避免只展示作者预训练模型。
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/go2w_reference_common.sh"
export GO2W_RUN="${GO2W_RUN:-/media/rmer/data/rpwd_training/go2w_supermimic/$($GO2W_PYTHON -c 'from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S"))')_v7_seed0}"
cd "$GO2W_SOURCE"
GO2W_RUN="$("$GO2W_PYTHON" - "$GO2W_RUN" "$@" <<'PY'
import argparse
from pathlib import Path
import sys
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--output')
args, _ = parser.parse_known_args(sys.argv[2:])
print(Path(args.output or sys.argv[1]).expanduser().resolve())
PY
)"
export GO2W_RUN
"$GO2W_SCRIPT_DIR/train_go2w_reference.sh" "$@"
GO2W_MODEL="$($GO2W_PYTHON - "$GO2W_RUN" <<'PY'
from pathlib import Path
import sys
models = list(Path(sys.argv[1]).glob('model_*.pt'))
if not models:
    raise SystemExit('训练结束但没有checkpoint，不能开始效果测试')
print(max(models, key=lambda p: int(p.stem.split('_')[-1])))
PY
)"
echo "[Go2W] 本机训练完成，开始测试新模型：$GO2W_MODEL"
"$GO2W_SCRIPT_DIR/evaluate_go2w_reference.sh" --checkpoint "$GO2W_MODEL" \
  --kinds flat up down --levels 1 2 3 --seconds 18 --video --output "$GO2W_RUN/evaluation"
echo "[Go2W] 完整实验结束，报告和录像：$GO2W_RUN/evaluation"
