#!/usr/bin/env bash
# ============================================================================
# GUI 测试入口 —— 随时可用: 自动加载**最新** checkpoint, 开 Isaac Sim 窗口回放
# ============================================================================
#
# 用法:
#   ./rl/scripts/gui_test.sh                      # 开窗口, 随机指令 (训练分布), 实时打印跟踪指标
#   ./rl/scripts/gui_test.sh --cmd 0.5 0 0        # 固定 vx=0.5 m/s
#   ./rl/scripts/gui_test.sh --cmd 0 0 0.6        # 固定左转 ωz=0.6 rad/s  —— 演示效果最好
#   ./rl/scripts/gui_test.sh --sweep              # 开窗口跑指令扫描 (汇总表在终端)
#   ./rl/scripts/gui_test.sh --checkpoint <x.pt>  # 指定模型
#
# 与训练并发时的保护 (2026-10-10 实测: 三个 Isaac Sim 实例抢一张卡会把训练**冻死**):
#   (默认)      训练在跑 -> 拒绝启动并提示换用下面两个选项
#   --wait      等训练跑完再自动启动 (每 30s 报一次进度)
#   --force     无视风险直接启动 (只开这一个测试实例时通常没事, 但不保证)
#   --dry-run   只打印将要执行的命令, 不启动仿真 (用于确认参数)
#
# 其余参数原样传给 rl/scripts/test_policy.py
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
# 提示信息里显示的脚本名: teleop.sh 通过 exec 调用本脚本时 $0 会变成 gui_test.sh,
# 所以允许调用方用环境变量显式指定
SELF="${GUI_TEST_SELF:-rl/scripts/$(basename "$0")}"

WAIT=0
FORCE=0
DRY=0
PASSTHROUGH=()
for arg in "$@"; do
    case "${arg}" in
        --wait) WAIT=1 ;;
        --force) FORCE=1 ;;
        --dry-run) DRY=1 ;;
        *) PASSTHROUGH+=("${arg}") ;;
    esac
done

training_pid() { pgrep -f "rl/scripts/[t]rain\.py" | head -1; }

# ---- 找出将要加载的 checkpoint (与 test_policy.py 的选择一致: 最新写入的那个) ----
latest_ckpt="$(ls -t rl/logs/rsl_rl/*/*/model_*.pt 2>/dev/null | head -1)"
if [[ -z "${latest_ckpt}" ]]; then
    echo "错误: 找不到任何 checkpoint (rl/logs/rsl_rl/*/*/model_*.pt), 先训练一次再来。" >&2
    exit 1
fi

# ---- 训练进度 / 预计剩余时间 (粗算) ----
training_progress() {
    local log pid it el
    log="$(ls -t rl/logs/train_*.log 2>/dev/null | head -1)"
    [[ -z "${log}" ]] && return
    it="$(grep -ao 'Learning iteration [0-9]*/[0-9]*' "${log}" 2>/dev/null | tail -1 | grep -o '[0-9]*' | head -1)"
    [[ -z "${it}" ]] && return
    pid="$(training_pid)"
    el="$(ps -o etimes= -p "${pid}" 2>/dev/null | tr -d ' ')"
    [[ -z "${el}" ]] && return
    awk -v it="${it}" -v el="${el}" 'BEGIN{ r=el/(it>15?it-15:1); printf "  训练进度: %d/3000 迭代, %.1f s/迭代, 预计还需 %.0f 分钟\n", it, r, (3000-it)*r/60 }'
}

# ---- 并发保护 ----
if [[ -n "$(training_pid)" ]] && [[ ${FORCE} -eq 0 ]]; then
    if [[ ${WAIT} -eq 0 ]]; then
        cat >&2 <<EOF
检测到训练正在运行 (PID $(training_pid)) —— 默认不启动 GUI 测试。

原因: 多个 Isaac Sim 实例共用一张显卡时训练进程可能死锁 (2026-10-10 实测,
      训练空转 12 小时, 只跑到 101/3000)。

请三选一:
  ./${SELF} --wait      # 等训练结束后自动启动 (推荐)
  ./${SELF} --force     # 立即启动, 自担风险 (只多开这一个实例通常没事)
  ./${SELF} --dry-run --force    # 只看将要执行的命令, 不启动

EOF
        training_progress >&2
        exit 1
    fi
    echo "==> --wait: 等训练结束 (每 30s 报一次)"
    while [[ -n "$(training_pid)" ]]; do
        printf '\r    [%s] ' "$(date '+%H:%M:%S')"
        training_progress | tail -1 | sed 's/^ *//' | tr -d '\n'
        sleep 30
    done
    echo
    echo "==> 训练已结束, 开始 GUI 测试"
    latest_ckpt="$(ls -t rl/logs/rsl_rl/*/*/model_*.pt 2>/dev/null | head -1)"  # 训练会写出新模型, 重新取
fi

CMD=("${SCRIPT_DIR}/test_policy.sh" "${PASSTHROUGH[@]}")
echo "==> 模型:   ${latest_ckpt}"
echo "==> 命令:   ${CMD[*]}"
if [[ ${DRY} -eq 1 ]]; then
    echo "==> --dry-run: 不启动仿真"
    exit 0
fi

exec "${CMD[@]}"
