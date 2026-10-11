#!/usr/bin/env bash
# ============================================================================
# 遥控测试入口 —— 用键盘/手柄手动驾驶训练好的策略 (默认键盘: 本机没接手柄也能用)
# ============================================================================
#
# 用法:
#   ./rl/scripts/teleop.sh                    # 键盘, 开窗口, 加载最新模型, 平地
#   ./rl/scripts/teleop.sh --num_envs 4       # 少放几个机器人, 视野更清楚
#   ./rl/scripts/teleop.sh --teleop-scale 0.5 # 把最大速度减半, 更好操控 (新手建议)
#   ./rl/scripts/teleop.sh --teleop gamepad   # 插了手柄再切手柄
#   ./rl/scripts/teleop.sh --wait             # 训练在跑时, 等它结束再启动 (推荐)
#   ./rl/scripts/teleop.sh --force            # 训练在跑也立刻启动 (自担风险)
#
# 键盘操作 (按键前先**点一下 Isaac Sim 窗口**让它获得焦点):
#   W = 前进        S = 后退
#   A = 左移        D = 右移
#   E = 左转        Q = 右转          (方向键/小键盘 8-2-4-6-7-9 同样可用)
#   按住生效, 松开立即归零; 终端每 0.5s 打印当前指令与实际速度对照.
#
# 手柄操作: 左摇杆 ↑↓=前进/后退, 左摇杆 ←→=左右平移, 右摇杆 ←→=左转/右转
#
# 指令会被裁剪到**训练时的指令范围** (vx/vy ±2.0 m/s, wz ±1.0 rad/s) ——
# 超出这个范围策略没见过, 会失效, 那不是 bug.
#
# 注意: 实机遥控不需要这个脚本 —— 部署端 roboparty_inference 本来就支持手柄
# (`is_joy_control_` 默认开启, 手柄 axes[4]/axes[3] -> vx/vy, LT/RT -> ωz, 带 clip_cmd 限幅)。
# 本脚本是**仿真里**手动驾驶策略用的。
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 让 gui_test.sh 的提示信息显示本脚本名 (exec 之后 $0 会变成 gui_test.sh)
export GUI_TEST_SELF="rl/scripts/teleop.sh"

# 复用 gui_test.sh 的并发保护 (检测到训练在跑会拦住, 除非 --wait/--force)
exec "${SCRIPT_DIR}/gui_test.sh" --teleop keyboard "$@"
