#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/go2w_reference_common.sh"
ensure_go2w_idle
cd "$GO2W_SOURCE"
exec "$GO2W_PYTHON" "$GO2W_SCRIPT_DIR/evaluate_go2w_reference.py" "$@"
