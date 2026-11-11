#!/usr/bin/env bash
# 单一GPU训练，完成后串行CPU测试和完整控制器导出。
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/pointfoot_mjlab_common.sh"
if [[ " ${*} " == *" --help "* ]]; then exec "$POINTFOOT_MJLAB_SCRIPT_DIR/train_pointfoot_mjlab.sh" "$@"; fi
mapfile -t POINTFOOT_OPTIONS < <("$POINTFOOT_MJLAB_PYTHON" - "$@" <<'PY'
import argparse,os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser(add_help=False);p.add_argument('--iterations',type=int,default=60000);p.add_argument('--output');a,_=p.parse_known_args()
stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S')
print(a.iterations);print(a.output or os.getenv('POINTFOOT_MJLAB_RUN') or str(Path(os.environ['POINTFOOT_MJLAB_DATA'])/(stamp+'_大狗点足版')))
PY
)
export POINTFOOT_MJLAB_RUN="${POINTFOOT_OPTIONS[1]}"
"$POINTFOOT_MJLAB_SCRIPT_DIR/train_pointfoot_mjlab.sh" "$@"
POINTFOOT_CHECKPOINT="$POINTFOOT_MJLAB_RUN/model_${POINTFOOT_OPTIONS[0]}.pt"
"$POINTFOOT_MJLAB_SCRIPT_DIR/play_pointfoot_mjlab.sh" --mode evaluate --checkpoint "$POINTFOOT_CHECKPOINT" --speed .2 --seconds 55 --seeds 0 1 2 --output "$POINTFOOT_MJLAB_RUN/evaluation_${POINTFOOT_OPTIONS[0]}"
"$POINTFOOT_MJLAB_SCRIPT_DIR/export_pointfoot_mjlab.sh" --checkpoint "$POINTFOOT_CHECKPOINT" --output "$POINTFOOT_MJLAB_RUN/exported_${POINTFOOT_OPTIONS[0]}"
"$POINTFOOT_MJLAB_PYTHON" "$POINTFOOT_MJLAB_SCRIPT_DIR/check_pointfoot_acceptance.py" --summary "$POINTFOOT_MJLAB_RUN/evaluation_${POINTFOOT_OPTIONS[0]}/summary.json" --output "$POINTFOOT_MJLAB_RUN/acceptance.json"
