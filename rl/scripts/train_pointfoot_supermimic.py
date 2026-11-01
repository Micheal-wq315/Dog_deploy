#!/usr/bin/env python3
"""训练点足版 SuperMimic 世界模型教师；训练强制无 GUI。"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Velocity-Rough-RPWD-Pointfoot-SuperMimic-WM-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=40000)
parser.add_argument("--steps_per_iteration", type=int, default=1,
                    help="Go2W源码每次更新采集1个仿真步；保持此值可对齐其更新节奏")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", default="rl/logs/pointfoot_supermimic")
parser.add_argument("--run_name", default="大狗点足版")
parser.add_argument("--resume", default=None, help="恢复点足版 SuperMimic checkpoint")
parser.add_argument("--buffer_len", type=int, default=512)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batch", type=int, default=128)
parser.add_argument("--wm_updates", type=int, default=4)
parser.add_argument("--imag_batch", type=int, default=128)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--warmup", type=int, default=300)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--lr", type=float, default=2.0e-4)
parser.add_argument("--actor_lr_scale", type=float, default=0.3)
parser.add_argument("--actor_ramp_iterations", type=int, default=1000)
parser.add_argument("--actor_hold_iterations", type=int, default=20000)
parser.add_argument("--actor_schedule_start", type=int, default=0)
parser.add_argument("--lr_decay_iterations", type=int, default=20000)
parser.add_argument("--lr_final_frac", type=float, default=0.2)
parser.add_argument("--action_noise", type=float, default=0.15)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--save_interval", type=int, default=1000)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--wm_hidden", type=int, nargs="+", default=(512, 512, 512))
parser.add_argument("--actor_hidden", type=int, nargs="+", default=(256, 128, 64))
parser.add_argument("--critic_hidden", type=int, nargs="+", default=(256, 128, 64))
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
if min(args.num_envs, args.iterations, args.steps_per_iteration, args.buffer_len,
       args.wm_window, args.wm_batch, args.wm_updates, args.imag_batch, args.horizon,
       args.save_interval, args.log_interval, args.critic_epochs,
       args.critic_minibatches) < 1:
    parser.error("环境数、训练步数、回放和日志参数必须为正数")
if args.buffer_len < args.wm_window or args.warmup < args.wm_window:
    parser.error("buffer_len 和 warmup 必须不小于 wm_window")
if not (0.0 < args.gamma <= 1.0 and 0.0 <= args.lam <= 1.0 and args.lr > 0.0
        and args.actor_hold_iterations >= 0 and args.actor_schedule_start >= 0
        and 0.0 < args.lr_final_frac <= 1.0):
    parser.error("gamma、lambda 或学习率超出有效范围")
args.headless = True
sys.argv = [sys.argv[0], *unknown]
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import rp_wd_rl.tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from rp_wd_rl.supermimic.pointfoot import (  # noqa: E402
    ACTION_DIM,
    FORMAT,
    POLICY_DIM,
    PointfootSuperMimicLearner,
)
from sim_shutdown import shutdown  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


def _atomic_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _prune_checkpoints(run_dir: Path, keep: int = 5) -> None:
    checkpoints = sorted(
        run_dir.glob("teacher_iteration_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    for old in checkpoints[:-keep]:
        old.unlink(missing_ok=True)


class PointfootStateAdapter:
    """从仿真真值提取34维状态、273维高度图和点脚接触监督。"""

    def __init__(self, env):
        self.env = env.unwrapped
        self.robot = self.env.scene["robot"]
        action_term = self.env.action_manager.get_term("joint_pos")
        self.leg_ids = torch.as_tensor(
            action_term._joint_ids, device=self.env.device, dtype=torch.long
        )
        if len(self.leg_ids) != ACTION_DIM or self.env.action_manager.total_action_dim != ACTION_DIM:
            raise RuntimeError("点足 SuperMimic 要求12个主动关节、12维策略动作")
        foot_names = [f"{leg}_FOOT_LINK" for leg in ("LF", "RF", "LB", "RB")]
        self.foot_body_ids, _ = self.robot.find_bodies(foot_names, preserve_order=True)
        self.nonfoot_names = [name for name in self.robot.body_names if name not in foot_names]
        self.nonfoot_sensor_ids, _ = self.env.scene["contact_forces"].find_bodies(
            self.nonfoot_names, preserve_order=True
        )
        self.foot_sensor_ids, _ = self.env.scene["contact_forces"].find_bodies(
            foot_names, preserve_order=True
        )
        if len(self.foot_body_ids) != 4 or len(self.foot_sensor_ids) != 4:
            raise RuntimeError("ContactSensor 未解析到四个点脚刚体")

    def capture(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        env, robot = self.env, self.robot
        data = robot.data
        ground_z = env.scene.sensors["height_scanner"].data.ray_hits_w[:, 0, 2]
        ground_z = torch.where(torch.isfinite(ground_z), ground_z, env.scene.env_origins[:, 2])
        clearance = data.root_pos_w[:, 2] - ground_z
        joint_pos = data.joint_pos[:, self.leg_ids] - data.default_joint_pos[:, self.leg_ids]
        joint_vel = data.joint_vel[:, self.leg_ids]
        state = torch.cat((
            clearance[:, None], data.projected_gravity_b,
            data.root_lin_vel_b, data.root_ang_vel_b, joint_pos, joint_vel,
        ), dim=-1)
        if state.shape[-1] != 34:
            raise RuntimeError(f"点足物理状态应为34维，实际为{state.shape[-1]}")

        quat = data.root_quat_w
        qw, qx, qy, qz = quat.unbind(-1)
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy.square() + qz.square()))
        pose = torch.cat((data.root_pos_w[:, :2], yaw[:, None]), dim=-1)
        command = env.command_manager.get_command("base_velocity")
        scan = observations["supermimic_terrain"].reshape(env.num_envs, -1)
        if scan.shape[-1] != 273:
            raise RuntimeError(f"地形高度图应为273维，实际为{scan.shape[-1]}")

        contact_sensor = env.scene["contact_forces"]
        history = contact_sensor.data.net_forces_w_history
        foot_force = history[:, :, self.foot_sensor_ids].norm(dim=-1).amax(dim=1)
        foot_force = foot_force.clamp(0.0, 500.0) / 500.0
        contacts = (foot_force > (8.0 / 500.0)).float()
        nonfoot_force = history[:, :, self.nonfoot_sensor_ids].norm(dim=-1).amax(dim=(1, 2))
        collision = (nonfoot_force > 10.0).float()[:, None]
        policy_history = observations["policy"].reshape(env.num_envs, -1)
        if policy_history.shape[-1] != POLICY_DIM:
            raise RuntimeError(f"点足策略历史应为{POLICY_DIM}维，实际为{policy_history.shape[-1]}")
        return {
            "state": state,
            "pose": pose,
            "command": command,
            "scan": scan,
            "force": foot_force,
            "contacts": contacts,
            "collision": collision,
            "history": policy_history,
        }


def _make_env():
    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg, render_mode=None)
    print(f"[信息] 任务={args.task} 环境数={env.unwrapped.num_envs} headless=True", flush=True)
    return env


def _new_run_dir() -> Path:
    root = Path(args.output_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        name += f"_{args.run_name}"
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, ensure_ascii=False, indent=2, default=str)
    return run_dir


def _pose_delta(current: torch.Tensor, following: torch.Tensor) -> torch.Tensor:
    yaw_delta = following[:, 2] - current[:, 2]
    yaw_delta = torch.atan2(yaw_delta.sin(), yaw_delta.cos())
    displacement = following[:, :2] - current[:, :2]
    cosine, sine = current[:, 2].cos(), current[:, 2].sin()
    local_x = cosine * displacement[:, 0] + sine * displacement[:, 1]
    local_y = -sine * displacement[:, 0] + cosine * displacement[:, 1]
    return torch.stack((local_x, local_y, yaw_delta), dim=-1)


def _ensure_course_state(env) -> None:
    if hasattr(env, "_pointfoot_course_caps"):
        return
    terrain = env.scene.terrain
    env._pointfoot_course_caps = torch.zeros(
        terrain.terrain_generator.num_cols, dtype=torch.long, device=env.device
    )
    env._pointfoot_course_trials = torch.zeros(
        terrain.terrain_generator.num_cols, terrain.terrain_generator.num_rows, 64,
        device=env.device,
    )
    env._pointfoot_course_counts = torch.zeros(
        terrain.terrain_generator.num_cols, terrain.terrain_generator.num_rows,
        dtype=torch.long, device=env.device,
    )


def _main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    env = _make_env()
    run_dir = _new_run_dir()
    learner = PointfootSuperMimicLearner(
        args, torch.device(env.unwrapped.device), env.unwrapped.num_envs
    )
    adapter = PointfootStateAdapter(env)
    start_iteration = 0
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(
            args.resume, map_location=env.unwrapped.device, weights_only=True
        )
        start_iteration = learner.load_state_dict(resume_payload)
        if start_iteration >= args.iterations:
            raise ValueError("iterations 必须高于恢复 checkpoint 的轮数")

    try:
        observations, _ = env.reset(seed=args.seed)
        _ensure_course_state(env.unwrapped)
        simulation_steps = int((resume_payload or {}).get("simulation_steps", 0))
        env.unwrapped.common_step_counter = simulation_steps
        if resume_payload and "terrain_state" in resume_payload:
            terrain = env.unwrapped.scene.terrain
            terrain_state = resume_payload["terrain_state"]
            levels = terrain_state["levels"].to(env.unwrapped.device)
            types = terrain_state["types"].to(env.unwrapped.device)
            if levels.shape != terrain.terrain_levels.shape:
                raise ValueError("续训 checkpoint 的环境数与当前 --num_envs 不一致")
            terrain.terrain_levels.copy_(levels)
            terrain.terrain_types.copy_(types)
            terrain.env_origins.copy_(terrain.terrain_origins[levels, types])
            env.unwrapped._pointfoot_course_caps = terrain_state["caps"].to(env.unwrapped.device)
            env.unwrapped._pointfoot_course_trials = terrain_state["trials"].to(env.unwrapped.device)
            env.unwrapped._pointfoot_course_counts = terrain_state["counts"].to(env.unwrapped.device)
            observations, _ = env.reset(seed=args.seed)
        current = adapter.capture(observations)
        previous = torch.zeros(env.unwrapped.num_envs, ACTION_DIM, device=env.unwrapped.device)
        ou_noise = torch.zeros_like(previous)
        episode_return = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
        recent_returns: deque[float] = deque(maxlen=100)
        model_metrics: dict[str, float] = {}
        began = time.monotonic()
        replay_mib = sum(t.numel() * t.element_size() for t in learner.replay.data.values()) / 2**20
        print(f"[信息] 回放容量={args.buffer_len}步，显存约{replay_mib:.1f}MiB", flush=True)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) as log:
            for iteration in range(start_iteration, args.iterations):
                learner.actor_optimizer.param_groups[0]["lr"] = learner.actor_learning_rate(iteration)
                rewards, velocity_errors, reward_term_values, policy_metrics = [], [], [], {}
                for _ in range(args.steps_per_iteration):
                    with torch.no_grad():
                        learner.actor.norm.update(current["history"])
                        action_mean = learner.actor(current["history"])
                        rho = 0.95
                        ou_noise.mul_(rho).add_(
                            torch.randn_like(ou_noise), alpha=args.action_noise * (1.0 - rho**2) ** 0.5
                        )
                        action = (action_mean + ou_noise).clamp(-1.0, 1.0)
                        next_observations, reward, terminated, truncated, _ = env.step(action)
                        following = adapter.capture(next_observations)
                        terminated = terminated.reshape(-1).bool()
                        truncated = truncated.reshape(-1).bool()
                        done = terminated | truncated
                        valid = ~done
                        transition = {
                            "state": current["state"],
                            "action": action,
                            "previous": previous,
                            "command": current["command"],
                            "scan": current["scan"],
                            "force": current["force"],
                            "contacts": current["contacts"],
                            "history": current["history"],
                            "pose": _pose_delta(current["pose"], following["pose"]),
                            "next_state": following["state"],
                            "next_scan": following["scan"],
                            "next_force": following["force"],
                            "next_contacts": following["contacts"],
                            "reward": reward[:, None],
                            "collision": following["collision"],
                        }
                        event = (done | (following["collision"].squeeze(-1) > 0)
                                 | ((following["contacts"] - current["contacts"]).abs().sum(-1) > 0)
                                 | (following["force"].amax(-1) > 0.35))
                        learner.replay.add(transition, done, valid, event)
                        episode_return += reward
                        if done.any():
                            recent_returns.extend(episode_return[done].cpu().tolist())
                            episode_return[done] = 0.0
                            previous[done] = 0.0
                            ou_noise[done] = 0.0
                        previous[~done] = action[~done]
                        velocity_errors.append(
                            (following["state"][:, 4:6] - following["command"][:, :2])
                            .norm(dim=-1).mean()
                        )
                        current = following
                        observations = next_observations
                        rewards.append(reward)
                        reward_term_values.append(
                            env.unwrapped.reward_manager._step_reward.detach().mean(dim=0)
                        )
                        simulation_steps += 1

                if learner.replay.size >= args.wm_window:
                    for _ in range(args.wm_updates):
                        model_metrics = learner.fit_world_model()
                    if learner.replay.total_steps >= args.warmup:
                        policy_metrics = learner.improve()

                if (iteration + 1) % args.log_interval == 0 or iteration == start_iteration:
                    terrain = env.unwrapped.scene.terrain
                    row = {
                        "iteration": iteration + 1,
                        "simulation_steps": simulation_steps,
                        "real_transitions": simulation_steps * env.unwrapped.num_envs,
                        "reward_mean": float(torch.stack(rewards).mean().detach()),
                        "episode_return_mean": (
                            sum(recent_returns) / len(recent_returns) if recent_returns else None
                        ),
                        "velocity_error_xy": float(torch.stack(velocity_errors).mean().detach()),
                        "replay_steps": learner.replay.size,
                        "policy_updates": learner.policy_updates,
                        "skipped_updates": learner.skipped_updates,
                        "mean_terrain_level": float(terrain.terrain_levels.float().mean()),
                        "elapsed_seconds": round(time.monotonic() - began, 2),
                        **model_metrics,
                        **policy_metrics,
                    }
                    term_means = torch.stack(reward_term_values).mean(dim=0)
                    row["reward_terms_per_second"] = {
                        name: float(term_means[index].detach())
                        for index, name in enumerate(env.unwrapped.reward_manager.active_terms)
                    }
                    if hasattr(env.unwrapped, "_pointfoot_course_caps"):
                        row["terrain_level_caps"] = env.unwrapped._pointfoot_course_caps.tolist()
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print("[训练] " + json.dumps(row, ensure_ascii=False), flush=True)

                if (iteration + 1) % args.save_interval == 0 or iteration + 1 == args.iterations:
                    payload = learner.state_dict(iteration + 1)
                    payload["simulation_steps"] = simulation_steps
                    payload["task"] = args.task
                    terrain = env.unwrapped.scene.terrain
                    payload["terrain_state"] = {
                        "levels": terrain.terrain_levels.detach().cpu().clone(),
                        "types": terrain.terrain_types.detach().cpu().clone(),
                        "caps": env.unwrapped._pointfoot_course_caps.detach().cpu().clone(),
                        "trials": env.unwrapped._pointfoot_course_trials.detach().cpu().clone(),
                        "counts": env.unwrapped._pointfoot_course_counts.detach().cpu().clone(),
                    }
                    _atomic_save(payload, run_dir / "teacher_latest.pt")
                    _atomic_save(payload, run_dir / f"teacher_iteration_{iteration + 1}.pt")
                    _prune_checkpoints(run_dir)

        final = learner.state_dict(args.iterations)
        final["simulation_steps"] = simulation_steps
        final["task"] = args.task
        terrain = env.unwrapped.scene.terrain
        final["terrain_state"] = {
            "levels": terrain.terrain_levels.detach().cpu().clone(),
            "types": terrain.terrain_types.detach().cpu().clone(),
            "caps": env.unwrapped._pointfoot_course_caps.detach().cpu().clone(),
            "trials": env.unwrapped._pointfoot_course_trials.detach().cpu().clone(),
            "counts": env.unwrapped._pointfoot_course_counts.detach().cpu().clone(),
        }
        _atomic_save(final, run_dir / "teacher_final.pt")
        print(f"[完成] 点足 SuperMimic checkpoint：{run_dir / 'teacher_final.pt'}", flush=True)
        env.close()
    except BaseException:
        env.close()
        raise


if __name__ == "__main__":
    try:
        _main()
    finally:
        shutdown(simulation_app)
