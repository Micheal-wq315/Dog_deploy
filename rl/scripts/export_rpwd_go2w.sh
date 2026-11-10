#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/rpwd_go2w_common.sh"
export CUDA_VISIBLE_DEVICES=""
exec "$RPWD_GO2W_PYTHON" "$RPWD_GO2W_SCRIPT_DIR/export_rpwd_go2w.py" "$@"
