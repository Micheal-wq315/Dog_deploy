#!/usr/bin/env bash
# 默认RP_wd；追加--robot go2w可运行同一算法和修正版课程的Go2W对照。
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/rpwd_go2w_common.sh"
mapfile -t RPWD_GO2W_OPTIONS < <("$RPWD_GO2W_PYTHON" - "$@" <<'PY'
import argparse,sys
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--robot',default='rpwd');p.add_argument('--device',default='cuda:0')
a,_=p.parse_known_args(sys.argv[1:]);print(a.robot);print(a.device)
PY
)
if [[ " ${*} " != *" --help "* ]]; then
  if [[ "${RPWD_GO2W_OPTIONS[1]}" == cpu ]]; then
    export CUDA_VISIBLE_DEVICES=""
  else
    "$RPWD_GO2W_PYTHON" - <<'PY'
from pathlib import Path
active=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        a=p.read_bytes().decode(errors='replace').split('\0')
        if not a:continue
        if Path(a[0]).name=='kit' or ('python' in Path(a[0]).name and any(Path(x).name in ('train.py','train_rpwd_go2w.py','train_supermimic.py','train_pointfoot_supermimic.py','test_policy.py','verify_contract.py','play.py','play_supermimic.py','play_pointfoot_supermimic.py','evaluate_pointfoot_supermimic.py') for x in a[1:])):
            active.append(p.parent.name)
    except OSError:pass
if active:raise SystemExit('已有GPU训练/Isaac仿真进程，请先结束后再启动：'+str(active))
PY
  fi
fi
RPWD_GO2W_RUN="${RPWD_GO2W_RUN:-$RPWD_GO2W_DATA/$($RPWD_GO2W_PYTHON -c 'from datetime import datetime; from zoneinfo import ZoneInfo; print(datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S"))')_${RPWD_GO2W_OPTIONS[0]}}"
set -- "$RPWD_GO2W_PYTHON" "$RPWD_GO2W_SCRIPT_DIR/train_rpwd_go2w.py" \
  --robot rpwd --terrain --terrain-revision 7 --command-curriculum \
  --num-envs 4096 --iterations 60000 --num-steps-per-env 1 --buffer-len 512 \
  --wm-batch 2048 --wm-window 8 --wm-updates 4 --imag-batch 4096 --horizon 32 \
  --critic-epochs 5 --critic-minibatches 4 --prioritized-replay --compile-imagination \
  --warmup 300 --actor-ramp-iterations 1000 --actor-hold-iterations 20000 \
  --lr-decay-iterations 20000 --lr-final-frac .2 --save-interval 500 --log-interval 50 \
  --seed 0 --output "$RPWD_GO2W_RUN" "$@"
if command -v systemd-inhibit >/dev/null 2>&1; then
  exec systemd-inhibit --what=idle:sleep:handle-lid-switch --why="RP_wd/Go2W共享SuperMimic训练" "$@"
fi
exec "$@"
