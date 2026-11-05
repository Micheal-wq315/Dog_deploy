#!/usr/bin/env python3
"""回放点足 SuperMimic；支持最新模型、指定地形和 WASD 键盘控制。"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=None, help="省略时加载数据盘最新正式运行的模型")
parser.add_argument("--task", default=None, help="默认使用checkpoint记录的任务，兼容旧单向和完整任务")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--auto-reset", action="store_true",
                    help="恢复训练的自动复位条件；默认关闭自动复位，仅按R手动复位")
parser.add_argument("--teleop", choices=["keyboard"], default=None)
parser.add_argument("--terrain", choices=["mixed", "flat", "stairs", "up", "down", "platform", "slope", "rough", "rubble"], default="mixed")
parser.add_argument("--terrain_level", type=int, choices=range(6), default=0,
                    help="固定等级0..5；v4对应2/4/6/8/10/12cm，旧版2/4/8/12/16/20cm")
parser.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"))
parser.add_argument("--forward_speed", type=float, default=None)
parser.add_argument("--lateral_speed", type=float, default=None)
parser.add_argument("--yaw_speed", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
if args.teleop and args.headless:
    parser.error("键盘控制需要GUI；请去掉--headless")
if args.teleop and args.cmd is not None:
    parser.error("--teleop和--cmd只选一种")
if args.num_envs < 1 or args.max_steps < 0 or min((v for v in (args.forward_speed, args.lateral_speed, args.yaw_speed) if v is not None), default=0) < 0:
    parser.error("环境数必须为正，步数和速度幅值不得为负")
sys.argv = [sys.argv[0], *unknown]
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import rp_wd_rl.tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from rp_wd_rl.supermimic.pointfoot import FORMAT, FRAME_DIMS, HISTORY, PointfootActor  # noqa: E402
from pointfoot_runtime import LEGACY_TASK, resolve_pointfoot_checkpoint, actor_from_checkpoint, policy_action
from sim_shutdown import shutdown  # noqa: E402


def make_keyboard(device):
    import carb.input
    from keyboard_wasd import Se2KeyboardWASD, Se2KeyboardWASDCfg

    class HeldKeyboard(Se2KeyboardWASD):
        """按键状态重算指令，防止自动重复或停止后松键产生反向速度。"""
        def __init__(self, cfg):
            self.held = set()
            self.reset_robot = False
            super().__init__(cfg)

        def reset(self):
            self.held.clear()
            super().reset()

        def _on_keyboard_event(self, event, *unused, **kwargs):
            key = event.input.name
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if key in ("SPACE", "L", "R"):
                    self.reset()
                    self.reset_robot |= key == "R"
                elif key in self._INPUT_KEY_MAPPING:
                    self.held.add(key)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self.held.discard(key)
            self._base_command.fill(0)
            for pressed in self.held:
                self._base_command += self._INPUT_KEY_MAPPING[pressed]
            return True

    keyboard = HeldKeyboard(Se2KeyboardWASDCfg(
        sim_device=str(device), v_x_sensitivity=args.forward_speed,
        v_y_sensitivity=args.lateral_speed, omega_z_sensitivity=args.yaw_speed))
    keyboard.reset()
    return keyboard


def main() -> None:
    checkpoint_path = resolve_pointfoot_checkpoint(args.checkpoint)
    device = torch.device(args.device or "cuda:0")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("需要225维五帧点足SuperMimic v2模型")
    actor = actor_from_checkpoint(checkpoint, device)

    task_id = args.task or checkpoint.get("task", LEGACY_TASK)
    cfg = parse_env_cfg(task_id, device=str(device), num_envs=args.num_envs)
    full = getattr(cfg, "supermimic_revision", 2) >= 3
    if (getattr(cfg, "supermimic_revision", 2) >= 4) != bool(checkpoint.get("gait_enabled", False)):
        raise ValueError("模型与任务的步态版本不匹配，请使用checkpoint默认任务")
    if full:
        from rp_wd_rl.supermimic.pointfoot_courses import COMMAND_LIMITS
        stage = int(checkpoint.get("terrain_state", {}).get("full", {}).get("command_stage", 0))
        if not 0 <= stage < len(COMMAND_LIMITS):
            raise ValueError("checkpoint的命令课程阶段无效")
        limits = COMMAND_LIMITS[stage]
    else:
        limits = (.45, .05, .08)
    for name, preferred, bound in zip(("forward_speed", "lateral_speed", "yaw_speed"),
                                      (.35, .20, .60), limits):
        if getattr(args, name) is None:
            setattr(args, name, min(preferred, bound))
    cfg.seed = int(checkpoint.get("config", {}).get("seed", 42))
    cfg.curriculum = {}
    if not args.auto_reset:
        # 空终止配置同时关闭超时、赛道出口、触地/倾倒和低高度复位。
        # 用户可以持续驾驶，只有下方R键分支显式调用env.reset()。
        cfg.terminations = {}
    cfg.observations.policy.enable_corruption = False
    generator = cfg.scene.terrain.terrain_generator
    if args.terrain != "mixed":
        candidates = {"stairs": ("up", "stairs_up"), "up": ("up", "stairs_up"),
                      "rough": ("rubble", "rough"), "rubble": ("rubble", "rough")}.get(
                          args.terrain, (args.terrain,))
        name = next((key for key in candidates if key in generator.sub_terrains), None)
        if name is None:
            raise ValueError(f"此模型任务不包含{args.terrain}地形；可选{list(generator.sub_terrains)}")
        terrain = generator.sub_terrains[name]
        terrain.proportion = 1.0
        generator.sub_terrains = {name: terrain}
        generator.num_cols = 1
    cfg.scene.terrain.max_init_terrain_level = args.terrain_level
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (2.5, 2.5, 1.6)
    cfg.viewer.lookat = (0.0, 0.0, 0.1)
    manual = args.teleop is not None or args.cmd is not None
    if manual:
        command_cfg = cfg.commands.base_velocity
        if full:
            command_cfg.manual_control = True
        command_cfg.resampling_time_range = (1e9, 1e9)
        command_cfg.rel_standing_envs = 0.0
        command_cfg.heading_command = False
        command_cfg.ranges.lin_vel_x = (0.0, 0.0)
        command_cfg.ranges.lin_vel_y = (0.0, 0.0)
        command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    env = gym.make(task_id, cfg=cfg, render_mode=None if args.headless else "human")
    try:
        terrain = env.unwrapped.scene.terrain
        terrain.terrain_levels.fill_(args.terrain_level)
        terrain.env_origins.copy_(terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
        obs, _ = env.reset(seed=cfg.seed)
        keyboard = make_keyboard(device) if args.teleop else None
        term = env.unwrapped.command_manager.get_term("base_velocity")
        command_start = sum(FRAME_DIMS[:2]) * HISTORY + FRAME_DIMS[2] * (HISTORY - 1)
        print(f"[回放] checkpoint={checkpoint_path} GUI={not args.headless} "
              f"地形={args.terrain} 等级={args.terrain_level} "
              f"自动复位={'开启' if args.auto_reset else '关闭（仅R手动复位）'}", flush=True)
        if args.terrain in ("stairs", "up", "down"):
            heights = getattr(cfg, "pointfoot_step_heights", (.02, .04, .08, .12, .16, .20))
            print(f"[地形] 单级高度={100 * heights[args.terrain_level]:g}cm", flush=True)
        if keyboard:
            print("[键盘] 先点击仿真视口；W/S前后，A/D横移，E左转、Q右转；空格停止，R复位。按住生效，松开归零。", flush=True)
            print(f"[键盘] vx=±{args.forward_speed}m/s，vy=±{args.lateral_speed}m/s，wz=±{args.yaw_speed}rad/s。", flush=True)
            if not full:
                print("[键盘] 当前是旧单向上阶模型，倒退/停止/横移/转向尚未充分训练。", flush=True)
        step = 0
        while simulation_app.is_running() and (args.max_steps <= 0 or step < args.max_steps):
            began = time.monotonic()
            with torch.inference_mode():
                if keyboard and keyboard.reset_robot:
                    obs, _ = env.reset()
                    keyboard.reset_robot = False
                policy_input = obs["policy"].reshape(env.unwrapped.num_envs, -1).clone()
                if manual:
                    command = keyboard.advance() if keyboard else torch.tensor(args.cmd, device=device)
                    limits = command.new_tensor((args.forward_speed, args.lateral_speed, args.yaw_speed))
                    if keyboard:
                        command = command.clamp(-limits, limits)
                    term.vel_command_b[:] = command
                    term.is_standing_env[:] = False
                    term.time_left[:] = 1e9
                    # 只替换最新指令帧，历史仍按真实仿真步推进，避免多append一帧。
                    policy_input[:, command_start:command_start + 3] = command
                obs, _, terminated, _, _ = env.step(policy_action(actor, policy_input, env))
            step += 1
            if step == 1 or step % 100 == 0:
                robot = env.unwrapped.scene["robot"]
                actual = torch.cat((robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]), -1)
                print(f"[状态] step={step} 指令={term.command[0].tolist()} "
                      f"实际vx/vy/wz={actual[0].tolist()} 本步失败={int(terminated.sum())}", flush=True)
            if not args.headless:
                time.sleep(max(0.0, env.unwrapped.step_dt - (time.monotonic() - began)))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        shutdown(simulation_app, exit_code=1)
    else:
        shutdown(simulation_app)
