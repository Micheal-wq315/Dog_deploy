"""测试训练好的 RP_wd 策略: 固定指令 / 指令扫描 / 随机指令.

与 play.py 的区别: play.py 是"看一眼"用的回放; 本脚本是**测量**用的 ——
直接对比"指令速度 vs 实际速度", 统计跟踪误差/姿态/打滑/摔倒, 并支持一次扫完整个指令表.

用法:
    ./rl/scripts/test_policy.sh                          # GUI, 随机指令 (训练分布), 实时打印指标
    ./rl/scripts/test_policy.sh --cmd 0.5 0 0            # GUI, 固定 vx=0.5 m/s
    ./rl/scripts/test_policy.sh --teleop gamepad         # ★ 手柄手动驾驶 (见 teleop.sh)
    ./rl/scripts/test_policy.sh --teleop keyboard        # ★ 键盘手动驾驶
    ./rl/scripts/test_policy.sh --sweep                  # 自动扫指令表, 输出跟踪误差汇总表
    ./rl/scripts/test_policy.sh --checkpoint <路径.pt> --export

默认加载**最新** checkpoint; 默认用 Play 配置 (关域随机化/观测噪声, 看纯策略表现),
加 --train-dist 则用训练配置 (带域随机化, 测鲁棒性)。GUI 下默认按真实 50 Hz 步进 (--no-real-time 可关)。
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="Test the latest RP_wd policy.")
parser.add_argument("--task", type=str, default=None, help="显式指定任务；深度模型使用 OdinDepth-Play 任务。")
parser.add_argument("--seed", type=int, default=42, help="测试随机种子。")
parser.add_argument("--save-dir", type=str, default=None, help="保存逐环境 CSV、汇总、测试图和配置。")
parser.add_argument("--save-depth", action="store_true", help="每0.5秒保存原始图、旋转校正图及16×12网格；需要 --save-dir。")
parser.add_argument("--eval-reset", action="store_true", help="固定出生朝向、零初速度，便于重复测试。")
parser.add_argument(
    "--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"),
    help="固定速度指令 [m/s, m/s, rad/s]; 不填则用训练分布的随机指令.",
)
parser.add_argument("--sweep", action="store_true", default=False, help="自动扫描一组典型指令并输出汇总表 (覆盖 --cmd).")
parser.add_argument("--checkpoint", type=str, default=None, help="模型文件 (.pt); 缺省用最新的.")
parser.add_argument("--experiment_name", type=str, default=None, help="实验名 (定位日志目录).")
parser.add_argument("--num_envs", type=int, default=16, help="并行环境数 (GUI 下建议 ≤32).")
parser.add_argument("--max_steps", type=int, default=0, help="总步数上限 (0 = 一直跑, 关窗/Ctrl+C 退出).")
parser.add_argument("--log_interval", type=int, default=25, help="每隔多少步打印一行指标 (25 步 = 0.5 s).")
parser.add_argument("--train-dist", action="store_true", default=False, help="用训练配置 (含域随机化+观测噪声) 测鲁棒性.")
parser.add_argument("--rough", action="store_true", default=False, help="在崎岖地形上测试/遥控 (随机地形).")
parser.add_argument("--terrain-level", type=int, default=None,
                    help="固定生成地形行号；普通任务0–5、OdinDepth任务0–11。")
parser.add_argument("--export", action="store_true", default=False, help="同时导出 ONNX 到 rl/exported/.")
parser.add_argument(
    "--teleop", type=str, default=None, choices=["gamepad", "keyboard", "udp"],
    help="遥控模式: 用手柄/键盘手动驾驶策略 (自动按真实 50Hz 步进, 指令裁剪到训练范围).",
)
parser.add_argument("--teleop-scale", type=float, default=1.0, help="遥控最大速度的缩放系数 (默认 1.0).")
parser.add_argument("--udp-port", type=int, default=5005,
                    help="--teleop udp 时监听的端口 (与 dji_pad_publisher 的 udp_port 一致).")
parser.add_argument(
    "--real-time", action=argparse.BooleanOptionalAction, default=None,
    help="按真实 50 Hz 步进 (GUI 观察用); 默认: GUI 下开, headless 下关.",
)
parser.add_argument("--settle_steps", type=int, default=100, help="扫描: 每组指令的稳定时间 [步].")
parser.add_argument("--measure_steps", type=int, default=150, help="扫描: 每组指令的测量时间 [步].")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task and "OdinDepth" in args_cli.task:
    args_cli.enable_cameras = True
if args_cli.save_depth and not args_cli.save_dir:
    parser.error("--save-depth 需要同时指定 --save-dir。")
if args_cli.save_depth and (not args_cli.task or "OdinDepth" not in args_cli.task):
    parser.error("--save-depth 仅适用于 OdinDepth 任务，请显式指定 --task Isaac-Velocity-Rough-RPWD-OdinDepth-Play-v0。")
if args_cli.num_envs < 1 or args_cli.measure_steps < 1 or args_cli.settle_steps < 0:
    parser.error("环境数和测量步数必须大于0，稳定步数不能小于0。")
if args_cli.real_time is None:
    # GUI 下按真实时间步进看着才自然; 遥控时必须实时 (否则机器人以 80 倍速跑, 根本没法操控)
    args_cli.real_time = (not args_cli.headless) or (args_cli.teleop is not None)
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Sim 启动后的导入
# ---------------------------------------------------------------------------
import importlib.metadata as metadata  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import rp_wd_rl.tasks  # noqa: E402, F401
from onnx_utils import export_and_verify  # noqa: E402
from rp_wd_rl.assets.rp_wd_contract import (  # noqa: E402
    CMD_VX_RANGE,
    CMD_VY_RANGE,
    CMD_WZ_RANGE,
    POLICY_DT,
    WHEEL_JOINT_INDICES,
    WHEEL_RADIUS,
)
from sim_shutdown import shutdown  # noqa: E402
from evaluation_recorder import EvaluationRecorder  # noqa: E402

_RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_ROOT = os.path.join(_RL_ROOT, "logs", "rsl_rl")

# 扫描用的典型指令 (覆盖前进/后退/侧移/自转/组合, 都在部署端 clip_cmd 范围内)
SWEEP_COMMANDS: list[tuple[float, float, float]] = [
    (0.0, 0.0, 0.0),    # 站立
    (0.5, 0.0, 0.0),    # 中速前进
    (CMD_VX_RANGE[1], 0.0, 0.0),
    (CMD_VX_RANGE[0], 0.0, 0.0),
    (0.0, CMD_VY_RANGE[1], 0.0),
    (0.0, CMD_VY_RANGE[0], 0.0),
    (0.0, 0.0, CMD_WZ_RANGE[1]),
    (0.0, 0.0, CMD_WZ_RANGE[0]),
    (0.4, 0.2, 0.5),    # 范围内的组合机动
]


def hold_command(env, cmd: tuple[float, float, float]) -> None:
    """把速度指令钉死为 cmd, 并阻止随机重采样."""
    term = env.command_manager.get_term("base_velocity")
    cmd_t = torch.tensor(cmd, dtype=torch.float32, device=env.device).expand(env.num_envs, 3)
    term.vel_command_b[:] = cmd_t
    # 指令为零 -> 标记 standing; 否则 _update_command() 会把非零指令当 standing 清零
    term.is_standing_env[:] = bool(torch.linalg.norm(cmd_t[0]) < 0.1)
    term.time_left[:] = 1.0e9  # 阻止 resample


class UdpCommandDevice:
    """从 dji_pad_publisher 的 UDP 包读取控制量.

    为什么用 UDP 而不是直接订阅 ROS 的 /joy: 仿真跑在 conda 的 python 3.11 下,
    系统 rclpy 只有 3.10 的扩展, 装不进这个环境. 发布者 (系统 python) 把控制量打成
    6 个 little-endian float32 发到 127.0.0.1:<port>, 这里非阻塞读取最新一包.

    包格式: (vx, vy, wz, pitch, roll, height_offset)
    超过 0.5 s 没收到包就归零 —— 手柄/发布者挂掉时不能让机器人继续跑.
    """

    def __init__(self, port: int):
        import socket

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", port))
        self._sock.setblocking(False)
        self._last = (0.0,) * 6
        self._last_time = 0.0

    def reset(self):
        self._last = (0.0,) * 6
        self._last_time = 0.0

    def advance(self):
        import struct
        import time

        while True:
            try:
                data, _ = self._sock.recvfrom(64)
            except BlockingIOError:
                break
            if len(data) >= 24:
                self._last = struct.unpack("<6f", data[:24])
                self._last_time = time.time()
        if time.time() - self._last_time > 0.5:
            return torch.zeros(6)
        return torch.tensor(self._last, dtype=torch.float32)


def hold_posture(env, posture) -> None:
    """把姿态/高度指令钉死 (pitch, roll, height_offset)."""
    term = env.command_manager.get_term("base_posture")
    val = torch.tensor(posture, dtype=torch.float32, device=env.device).expand(env.num_envs, 3)
    term.posture_command[:] = val
    term.time_left[:] = 1.0e9


def make_teleop_device(kind: str, device_str: str, scale: float = 1.0):
    """创建 SE(2) 遥操作设备 (手柄或键盘), 用于手动驾驶策略.

    灵敏度按**训练指令范围**设置, 并且每步还会再裁剪一次 —— 保证不会发出训练分布外的指令
    (超出分布的指令策略会直接失效, 那属于"边界测试", 不是 bug).

    键盘用的是自定义的 Se2KeyboardWASD (WASD 平移 + Q/E 转向), 同时保留自带的
    方向键/小键盘绑定.
    """
    from isaaclab.devices import Se2Gamepad, Se2GamepadCfg
    from keyboard_wasd import Se2KeyboardWASD, Se2KeyboardWASDCfg

    common = {
        "sim_device": device_str,
        "v_x_sensitivity": CMD_VX_RANGE[1] * scale,
        "v_y_sensitivity": CMD_VY_RANGE[1] * scale,
        "omega_z_sensitivity": CMD_WZ_RANGE[1] * scale,
    }
    if kind == "gamepad":
        device = Se2Gamepad(Se2GamepadCfg(**common))
    elif kind == "udp":
        device = UdpCommandDevice(args_cli.udp_port)
    else:
        device = Se2KeyboardWASD(Se2KeyboardWASDCfg(**common))
    device.reset()
    return device


TELEOP_HELP = {
    "gamepad": "左摇杆 ↑↓=前进/后退, 左摇杆 ←→=左右平移, 右摇杆 ←→=左转/右转",
    "keyboard": "W/S=前进/后退, A/D=左移/右移, E=左转, Q=右转 (也支持方向键/小键盘8-4-6-2-7-9)",
    "udp": "DJI 手柄经 UDP 桥 (先在另一终端跑 dji_pad_publisher.py; 右杆平移/左杆纵向转向/左杆横向pitch/拨轮roll/按键升降高度)",
}


def read_metrics(env) -> dict:
    """读一帧指标 (按环境求均值; 跟踪误差另给最大/最小值以看离散度)."""
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command("base_velocity")
    v_b = robot.data.root_lin_vel_b
    w_b = robot.data.root_ang_vel_b
    err_lin = torch.linalg.norm(cmd[:, :2] - v_b[:, :2], dim=1)
    err_yaw = (cmd[:, 2] - w_b[:, 2]).abs()
    wheel_lin = robot.data.joint_vel[:, WHEEL_JOINT_INDICES].abs().mean(dim=1) * WHEEL_RADIUS
    speed = torch.linalg.norm(v_b[:, :2], dim=1)
    return {
        "cmd": cmd.mean(dim=0),
        "v": v_b.mean(dim=0),
        "wz": w_b[:, 2].mean(),
        "err_lin": err_lin.mean(),
        "err_lin_max": err_lin.max(),
        "err_yaw": err_yaw.mean(),
        "tilt_deg": torch.rad2deg(torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))).mean(),
        "height": robot.data.root_pos_w[:, 2].mean(),
        "speed": speed.mean(),
        "wheel_lin": wheel_lin.mean(),
        "slip": (wheel_lin.mean() / torch.clamp(speed.mean(), min=1e-3)).item(),
    }


def format_line(step: int, m: dict, rew: float) -> str:
    c, v = m["cmd"], m["v"]
    return (
        f"[{step * POLICY_DT:6.1f}s] 指令({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) "
        f"实际({v[0]:+.2f},{v[1]:+.2f},{m['wz']:+.2f}) "
        f"|误差| 线速{m['err_lin']:.3f}(最大{m['err_lin_max']:.3f}) 偏航{m['err_yaw']:.3f} "
        f"| 姿态{m['tilt_deg']:.1f}° 高{m['height']:.3f}m "
        f"| 轮线速{m['wheel_lin']:.2f} 滑移比{m['slip']:.2f} | 回报{rew:+.4f}/步"
    )


def step_once(env, wrapped, policy) -> tuple[float, int]:
    """策略走一步，保存转移前状态及转移终止原因；超时单独计数。"""
    with torch.inference_mode():
        recorder = getattr(env, "evaluation_recorder", None)
        before = recorder.capture(env) if recorder is not None else None
        obs = wrapped.get_observations()
        act = policy(obs)
        _, rew, dones, _ = wrapped.step(act)
        policy.reset(dones)
        if recorder is not None:
            recorder.record(env, before)
    return rew.mean().item(), int(env.reset_terminated.sum())


def run_continuous(env, wrapped, policy) -> None:
    """连续测试: 遥控 / 固定指令 / 训练分布随机指令, 实时打印指标."""
    fixed = tuple(args_cli.cmd) if args_cli.cmd is not None else None
    device = None
    if args_cli.teleop is not None:
        device = make_teleop_device(args_cli.teleop, str(env.device), args_cli.teleop_scale)
        print(f"[INFO] 遥控模式 (设备: {args_cli.teleop}, 缩放 {args_cli.teleop_scale})")
        print(f"[INFO] 操作: {TELEOP_HELP[args_cli.teleop]}")
        print(f"[INFO] 指令上限 (训练范围): vx{CMD_VX_RANGE} vy{CMD_VY_RANGE} wz{CMD_WZ_RANGE}")
        print("[INFO] 按住生效, 松开 = 零指令 = 原地站定.")
        bound_keys = sorted(getattr(device, "_INPUT_KEY_MAPPING", {}).keys())
        if bound_keys:
            print(f"[INFO] 已绑定的按键: {' '.join(bound_keys)}\n")
    elif fixed is not None:
        print(f"[INFO] 固定指令: vx={fixed[0]}, vy={fixed[1]}, wz={fixed[2]}")
    else:
        print(f"[INFO] 随机指令 (训练分布): vx{CMD_VX_RANGE} vy{CMD_VY_RANGE} wz{CMD_WZ_RANGE}")
    print(f"[INFO] {'真实时间 50Hz' if args_cli.real_time else '全速'} 步进; 每 {args_cli.log_interval} 步打印一行.\n")

    wrapped.get_observations()
    step, deaths, rew_acc = 0, 0, 0.0
    start = time.time()
    while simulation_app.is_running():
        t0 = time.time()
        if device is not None:
            # 遥控: 读设备 -> 裁剪到训练指令范围 -> 钉进命令管理器
            cmd = device.advance().to(env.device).flatten()
            if cmd.numel() >= 6:
                # UDP 桥: 后 3 维是姿态/高度指令
                hold_posture(env, cmd[3:6].tolist())
                cmd = cmd[:3]
            elif "base_posture" in env.command_manager.active_terms:
                # 键盘/手柄只有 x/y/wz, 没有姿态通道 —— 把姿态指令钉 0,
                # 否则会按随机采样给出 pitch/roll/height, 驾驶时机器人会莫名倾斜/升降
                hold_posture(env, (0.0, 0.0, 0.0))
            cmd[0] = cmd[0].clamp(*CMD_VX_RANGE)
            cmd[1] = cmd[1].clamp(*CMD_VY_RANGE)
            cmd[2] = cmd[2].clamp(*CMD_WZ_RANGE)
            hold_command(env, cmd.tolist())
        elif fixed is not None:
            hold_command(env, fixed)
            if "base_posture" in env.command_manager.active_terms:
                hold_posture(env, (0.0, 0.0, 0.0))
        rew, n_done = step_once(env, wrapped, policy)
        rew_acc += rew
        deaths += n_done
        step += 1
        if step % args_cli.log_interval == 0:
            print(format_line(step, read_metrics(env), rew_acc / args_cli.log_interval))
            rew_acc = 0.0
        if args_cli.max_steps and step >= args_cli.max_steps:
            print(f"[INFO] 达到 --max_steps={args_cli.max_steps}, 停止.")
            break
        if args_cli.real_time:
            sleep = POLICY_DT - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)
    print(f"[INFO] 共 {step} 步 ({step * POLICY_DT:.1f}s 仿真时间), 非超时终止 {deaths} 次, 墙钟 {time.time() - start:.1f}s.")


def run_sweep(env, wrapped, policy) -> None:
    """指令扫描: 对每组指令先稳定再测量, 输出汇总表."""
    wrapped.get_observations()
    print(f"[INFO] 扫描 {len(SWEEP_COMMANDS)} 组指令 (每组: 稳定 {args_cli.settle_steps} 步 + 测量 {args_cli.measure_steps} 步)\n")
    header = (
        f"{'指令 vx/vy/wz':>20} | {'实际 vx/vy/wz':>20} | {'线速误差':>8} | {'偏航误差':>8} "
        f"| {'姿态°':>6} | {'高 m':>5} | {'滑移比':>6} | {'终止':>4}"
    )
    print(header)
    print("-" * len(header))
    for case_index, cmd in enumerate(SWEEP_COMMANDS):
        # 物理状态缓存在推理模式内生成，后续重置也必须使用相同模式。
        with torch.inference_mode():
            wrapped.reset()
        recorder = getattr(env, "evaluation_recorder", None)
        if recorder is not None:
            recorder.case = f"{case_index:02d}_{cmd[0]:+.2f}_{cmd[1]:+.2f}_{cmd[2]:+.2f}"
            recorder.phase = "settle"
        deaths = 0
        # 姿态/高度指令钉 0: 这一轮测的是**速度**跟踪, 随机的姿态指令会干扰读数
        if "base_posture" in env.command_manager.active_terms:
            hold_posture(env, (0.0, 0.0, 0.0))
        for _ in range(args_cli.settle_steps):
            tick = time.monotonic()
            hold_command(env, cmd)
            if "base_posture" in env.command_manager.active_terms:
                hold_posture(env, (0.0, 0.0, 0.0))
            _, n_done = step_once(env, wrapped, policy)
            deaths += n_done
            if args_cli.real_time:
                time.sleep(max(0, POLICY_DT - (time.monotonic() - tick)))
        if recorder is not None:
            recorder.phase = "measure"
        acc = {k: 0.0 for k in ("err_lin", "err_yaw", "tilt_deg", "height", "slip", "vx", "vy", "wz")}
        for _ in range(args_cli.measure_steps):
            tick = time.monotonic()
            hold_command(env, cmd)
            if "base_posture" in env.command_manager.active_terms:
                hold_posture(env, (0.0, 0.0, 0.0))
            _, n_done = step_once(env, wrapped, policy)
            deaths += n_done
            m = read_metrics(env)
            acc["err_lin"] += m["err_lin"].item()
            acc["err_yaw"] += m["err_yaw"].item()
            acc["tilt_deg"] += m["tilt_deg"].item()
            acc["height"] += m["height"].item()
            acc["slip"] += m["slip"]
            acc["vx"] += m["v"][0].item()
            acc["vy"] += m["v"][1].item()
            acc["wz"] += m["wz"].item()
            if args_cli.real_time:
                time.sleep(max(0, POLICY_DT - (time.monotonic() - tick)))
        n = max(args_cli.measure_steps, 1)
        print(
            f"{cmd[0]:+7.2f}/{cmd[1]:+5.2f}/{cmd[2]:+5.2f} | "
            f"{acc['vx'] / n:+7.2f}/{acc['vy'] / n:+5.2f}/{acc['wz'] / n:+5.2f} | "
            f"{acc['err_lin'] / n:8.3f} | {acc['err_yaw'] / n:8.3f} | "
            f"{acc['tilt_deg'] / n:6.1f} | {acc['height'] / n:5.3f} | {acc['slip'] / n:6.2f} | {deaths:4d}"
        )
    print(
        "\n[说明] 线速误差/偏航误差 = |实际-指令| 的均值; "
        "滑移比仅供参考（原地转向时无意义）；终止为该组含稳定阶段的非超时终止次数。"
    )


def main() -> None:
    if args_cli.sweep and args_cli.teleop is not None:
        print("[WARN] --teleop 与 --sweep 不能同时使用, 忽略 --sweep (进入遥控模式).")
        args_cli.sweep = False
    surface = "Rough" if args_cli.rough else "Flat"
    task = args_cli.task or (f"Isaac-Velocity-{surface}-RPWD-v0" if args_cli.train_dist else f"Isaac-Velocity-{surface}-RPWD-Play-v0")
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    if args_cli.terrain_level is not None:
        generator = env_cfg.scene.terrain.terrain_generator
        if generator is None or not 0 <= args_cli.terrain_level < generator.num_rows:
            raise ValueError("--terrain-level 必须对应生成地形中的有效行号。")
        # 按行生成才能使“等级”对应真实难度；max_init 本身只限定随机行号上限。
        generator.curriculum = True
        generator.seed = args_cli.seed
        env_cfg.scene.terrain.max_init_terrain_level = args_cli.terrain_level
        env_cfg.curriculum.terrain_levels = None
    if args_cli.eval_reset:
        env_cfg.events.reset_base.params.update(
            pose_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            velocity_range={},
        )
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ---- 定位 checkpoint (默认最新) ----
    if args_cli.checkpoint:
        ckpt = os.path.abspath(args_cli.checkpoint)
    else:
        ckpt = get_checkpoint_path(os.path.join(LOG_ROOT, agent_cfg.experiment_name), ".*", "model_.*.pt")
    print(f"[INFO] 模型: {ckpt}")
    _lvl = getattr(args_cli, "terrain_level", None)
    _lvl = _lvl if _lvl is not None else getattr(env_cfg.scene.terrain, "max_init_terrain_level", None)
    _info = f" (地形行号: {_lvl})" if env_cfg.scene.terrain.terrain_generator is not None else ""
    print(f"[INFO] 任务: {task}{_info}")
    env_cfg.log_dir = os.path.dirname(ckpt)

    # ---- 环境 + 策略 ----
    env = gym.make(task, cfg=env_cfg).unwrapped
    if args_cli.terrain_level is not None:
        terrain = env.scene.terrain
        terrain.terrain_levels[:] = args_cli.terrain_level
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        env.scene.env_origins[:] = terrain.env_origins
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    wrapped.reset()
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.device)
    if args_cli.save_dir:
        env.evaluation_recorder = EvaluationRecorder(args_cli.save_dir, env, {
            "checkpoint": ckpt, "task": task, "seed": args_cli.seed,
            "terrain_level": args_cli.terrain_level, "num_envs": env.num_envs,
            "eval_reset": args_cli.eval_reset, "settle_steps": args_cli.settle_steps,
            "measure_steps": args_cli.measure_steps, "arguments": vars(args_cli),
        }, save_depth=args_cli.save_depth)

    if args_cli.export:
        output_name = "policy_odin_depth.onnx" if "OdinDepth" in task else "policy_flat.onnx"
        dst = export_and_verify(runner, os.path.join(_RL_ROOT, "exported"),
            expected_input_dim=int(wrapped.get_observations()["policy"].shape[-1]),
            output_filename=output_name)
        print(f"[INFO] 部署: cp {dst} src/inference/robots/wheel_quad/models/{output_name}")

    completed = False
    try:
        if args_cli.sweep:
            run_sweep(env, wrapped, policy)
        else:
            run_continuous(env, wrapped, policy)
        completed = True
    finally:
        if args_cli.save_dir:
            env.evaluation_recorder.finish(completed=completed)
    print("\n[INFO] 测试结束.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        shutdown(simulation_app, exit_code=1)
    shutdown(simulation_app)
