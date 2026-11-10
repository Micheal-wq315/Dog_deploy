#!/usr/bin/env bash
# 同一机器人完成训练后，串行执行CPU能力评测和独立ONNX导出。
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/rpwd_go2w_common.sh"
if [[ " ${*} " == *" --help "* ]]; then
  exec "$RPWD_GO2W_SCRIPT_DIR/train_rpwd_go2w.sh" "$@"
fi
mapfile -t RPWD_EXPERIMENT_OPTIONS < <("$RPWD_GO2W_PYTHON" - "$@" <<'PY'
import argparse,os,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--robot',default='rpwd');p.add_argument('--iterations',type=int,default=60000);p.add_argument('--output')
a,_=p.parse_known_args(sys.argv[1:])
stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d_%H-%M-%S')
output=a.output or os.getenv('RPWD_GO2W_RUN') or str(Path(os.environ['RPWD_GO2W_DATA'])/f'{stamp}_{a.robot}')
print(a.robot);print(a.iterations);print(Path(output).expanduser().resolve())
PY
)
export RPWD_GO2W_RUN="${RPWD_EXPERIMENT_OPTIONS[2]}"
"$RPWD_GO2W_SCRIPT_DIR/train_rpwd_go2w.sh" "$@"
RPWD_FINAL_MODEL="$RPWD_GO2W_RUN/model_${RPWD_EXPERIMENT_OPTIONS[1]}.pt"
"$RPWD_GO2W_SCRIPT_DIR/play_rpwd_go2w.sh" --mode evaluate --robot "${RPWD_EXPERIMENT_OPTIONS[0]}" \
  --checkpoint "$RPWD_FINAL_MODEL" --output "$RPWD_GO2W_RUN/evaluation_${RPWD_EXPERIMENT_OPTIONS[1]}"
"$RPWD_GO2W_SCRIPT_DIR/export_rpwd_go2w.sh" --checkpoint "$RPWD_FINAL_MODEL" \
  --output "$RPWD_GO2W_RUN/exported_${RPWD_EXPERIMENT_OPTIONS[1]}"
