#!/usr/bin/env python3
"""训练点足版 SuperMimic 世界模型教师；训练强制无 GUI。"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import hashlib
from pathlib import Path
import sys
import traceback
import time

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Velocity-Rough-RPWD-Pointfoot-SuperMimic-WM-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=20000)
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--steps_per_iteration", type=int, default=1,
                    help="Go2W源码每次更新采集1个仿真步；保持此值可对齐其更新节奏")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", default="rl/logs/pointfoot_supermimic")
parser.add_argument("--run_name", default="大狗点足版")
parser.add_argument("--resume", default=None, help="同任务同版本的完整续训")
parser.add_argument("--init_actor", default=None, help="只迁移225维Actor，世界模型/Critic/课程从零开始")
parser.add_argument("--archive_windows", type=int, default=0)
parser.add_argument("--success_archive_windows", type=int, default=0)
parser.add_argument("--wm_diagnostics_interval", type=int, default=500)
parser.add_argument("--diagnostic_batch", type=int, default=64)
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
parser.add_argument("--action_noise_final", type=float, default=None)
parser.add_argument("--noise_decay_iterations", type=int, default=8000)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--save_interval", type=int, default=500)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--wm_hidden", type=int, nargs="+", default=(512, 512, 512))
parser.add_argument("--actor_hidden", type=int, nargs="+", default=(256, 128, 64))
parser.add_argument("--critic_hidden", type=int, nargs="+", default=(256, 128, 64))
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
if unknown:
    parser.error("未识别的参数：" + " ".join(unknown))
if args.resume and args.init_actor:
    parser.error("--resume和--init_actor只选一种")
if args.action_noise < 0 or (args.action_noise_final is not None and args.action_noise_final < 0) or args.noise_decay_iterations < 1:
    parser.error("探索噪声不得为负，退火轮数必须为正")
if min(args.archive_windows, args.success_archive_windows) < 0 or min(args.wm_diagnostics_interval, args.diagnostic_batch) < 1:
    parser.error("归档容量不得为负，诊断周期和批次必须为正数")
if args.episode_length_s is not None and args.episode_length_s <= 0:
    parser.error("episode_length_s必须为正")
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
from pointfoot_runtime import PointfootStateAdapter, command_mode_ids, command_skills_ready, COMMAND_MODE_NAMES
from transition_capture import capture_before_reset
from pointfoot_recovery import recover_invalid_rows
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




def _make_env():
    cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.seed = args.seed
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s
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
    print(f"[信息] 运行目录={run_dir}", flush=True)
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
        terrain.cfg.terrain_generator.num_cols, dtype=torch.long, device=env.device
    )
    env._pointfoot_course_trials = torch.zeros(
        terrain.cfg.terrain_generator.num_cols, terrain.cfg.terrain_generator.num_rows, 64,
        device=env.device,
    )
    env._pointfoot_course_counts = torch.zeros(
        terrain.cfg.terrain_generator.num_cols, terrain.cfg.terrain_generator.num_rows,
        dtype=torch.long, device=env.device,
    )


def _training_contract(env, adapter):
    """保存会改变模型语义的配置，阻止旧奖励/旧地形被当作同一次续训。"""
    cfg = env.cfg
    gen = cfg.scene.terrain.terrain_generator
    action = env.action_manager.get_term("joint_pos")
    scale = torch.as_tensor(action._scale).reshape(-1)[:ACTION_DIM].cpu().tolist()
    return {
        "task": args.task, "task_revision": int(getattr(cfg, "supermimic_revision", 2)),
        "seed": args.seed, "num_envs": env.num_envs,
        "policy_dim": POLICY_DIM + (2 if adapter.gait_enabled else 0),
        "action_dim": ACTION_DIM, "policy_dt": env.step_dt,
        "episode_length_s": cfg.episode_length_s,
        "hard_min_height": adapter.task_rewards.hard_min_height,
        "fall_angle": adapter.task_rewards.fall_angle,
        "command_skill_gate_version": 1, "measurement_version": 3,
        "joint_names": [adapter.robot.joint_names[i] for i in adapter.leg_ids.cpu().tolist()],
        "action_scales": scale, "reward_weights": dict(adapter.task_rewards.weights),
        "terrain_size": list(gen.size), "terrain_rows": gen.num_rows, "terrain_cols": gen.num_cols,
        "terrains": {key: {"proportion": value.proportion, "kind": getattr(value, "kind", key)}
                     for key, value in gen.sub_terrains.items()},
    }


def _course_payload(env):
    terrain = env.scene.terrain
    result = {
        "levels": terrain.terrain_levels.detach().cpu().clone(),
        "types": terrain.terrain_types.detach().cpu().clone(),
        "caps": env._pointfoot_course_caps.detach().cpu().clone(),
        "trials": env._pointfoot_course_trials.detach().cpu().clone(),
        "counts": env._pointfoot_course_counts.detach().cpu().clone(),
    }
    if getattr(env.cfg, "supermimic_revision", 2) >= 3:
        from rp_wd_rl.tasks.locomotion.velocity.mdp.pointfoot_full import full_course_state_dict
        result["full"] = full_course_state_dict(env)
    return result


def _finite_rows(values):
    keys = ("state", "pose", "scan", "force", "history")
    return torch.stack([torch.isfinite(values[k]).flatten(1).all(-1) for k in keys]).all(0)


def _main() -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    env = _make_env()
    run_dir = _new_run_dir()
    adapter = PointfootStateAdapter(env)
    contract = _training_contract(env.unwrapped, adapter)
    if contract["task_revision"] >= 3:
        code_paths = ["rl/rp_wd_rl/supermimic/pointfoot_terrain.py",
                      "rl/rp_wd_rl/supermimic/pointfoot_courses.py",
                      "rl/rp_wd_rl/tasks/locomotion/velocity/mdp/pointfoot_full.py",
                      "rl/rp_wd_rl/tasks/locomotion/velocity/rp_wd_pointfoot_full_env_cfg.py"]
        if adapter.gait_enabled:
            code_paths += ["rl/rp_wd_rl/supermimic/pointfoot_gait.py",
                           "rl/rp_wd_rl/supermimic/pointfoot_gait_rewards.py",
                           "rl/rp_wd_rl/supermimic/pointfoot.py",
                           "rl/rp_wd_rl/tasks/locomotion/velocity/rp_wd_pointfoot_gait_env_cfg.py"]
            from rp_wd_rl.supermimic.pointfoot_gait import gait_parameters
            contract["gait"] = {**gait_parameters(), "terrain_pass_rate": env.unwrapped.cfg.terrain_pass_rate,
                                "minimum_trials": env.unwrapped.cfg.terrain_minimum_trials}
            contract["step_heights"] = list(env.unwrapped.cfg.pointfoot_step_heights)
            contract["exploration"] = {"initial": args.action_noise, "final": args.action_noise_final,
                                        "decay_iterations": args.noise_decay_iterations}
        contract["task_code_hash"] = hashlib.sha256(b"".join((REPO_ROOT / p).read_bytes() for p in code_paths)).hexdigest()
    with (run_dir / "training_contract.json").open("w", encoding="utf-8") as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2)
    with (run_dir / "env_config.json").open("w", encoding="utf-8") as stream:
        json.dump(env.unwrapped.cfg.to_dict(), stream, ensure_ascii=False, indent=2, default=str)
    learner = PointfootSuperMimicLearner(
        args, torch.device(env.unwrapped.device), env.unwrapped.num_envs,
        adapter.field, adapter.task_rewards
    )
    if args.init_actor:
        initial = torch.load(args.init_actor, map_location=env.unwrapped.device, weights_only=True)
        if initial.get("format") != FORMAT:
            raise ValueError("init_actor需要点足SuperMimic checkpoint")
        if bool(initial.get("gait_enabled", False)) != adapter.gait_enabled:
            raise ValueError("v4相位残差策略与旧225维碎步策略不兼容；请从零训练新步态")
        learner.actor.load_state_dict(initial["actor"])
        print(f"[信息] 只迁移Actor：{args.init_actor}；世界模型/价值/优化器/课程均从零开始", flush=True)
    start_iteration = 0
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(
            args.resume, map_location=env.unwrapped.device, weights_only=True
        )
        saved_contract = resume_payload.get("training_contract")
        if saved_contract is not None and saved_contract != contract:
            raise ValueError("续训的任务/奖励/地形/关节/环境数等配置已变化；请使用--init_actor开启新实验")
        if saved_contract is None and (contract["task_revision"] >= 3 or resume_payload.get("task") != args.task):
            raise ValueError("旧checkpoint没有相符的完整任务契约；请使用--init_actor迁移策略")
        if int(resume_payload.get("config", {}).get("seed", args.seed)) != args.seed:
            raise ValueError("续训必须保持相同seed以复现静态地形和物理随机化")
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
            env.unwrapped._pointfoot_course_caps.copy_(terrain_state["caps"].to(env.unwrapped.device))
            env.unwrapped._pointfoot_course_trials.copy_(terrain_state["trials"].to(env.unwrapped.device))
            env.unwrapped._pointfoot_course_counts.copy_(terrain_state["counts"].to(env.unwrapped.device))
            if "full" in terrain_state:
                from rp_wd_rl.tasks.locomotion.velocity.mdp.pointfoot_full import load_full_course_state_dict
                load_full_course_state_dict(env.unwrapped, terrain_state["full"])
            observations, _ = env.reset(seed=args.seed)
        current = adapter.capture(observations)
        previous = torch.zeros(env.unwrapped.num_envs, ACTION_DIM, device=env.unwrapped.device)
        ou_noise = torch.zeros_like(previous)
        episode_return = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
        recent_returns: deque[float] = deque(maxlen=100)
        model_metrics: dict[str, float] = {}
        interval_reward = torch.zeros((), device=env.unwrapped.device)
        interval_error = torch.zeros(3, device=env.unwrapped.device)
        interval_xy_error = torch.zeros((), device=env.unwrapped.device)
        interval_velocity = torch.zeros(3, device=env.unwrapped.device)
        interval_command = torch.zeros(3, device=env.unwrapped.device)
        interval_terms = torch.zeros(len(env.unwrapped.reward_manager.active_terms), device=env.unwrapped.device)
        interval_mode_error = torch.zeros(len(COMMAND_MODE_NAMES), 3, device=env.unwrapped.device)
        interval_mode_velocity = torch.zeros_like(interval_mode_error)
        interval_mode_command = torch.zeros_like(interval_mode_error)
        interval_mode_count = torch.zeros(len(COMMAND_MODE_NAMES), device=env.unwrapped.device)
        interval_mode_falls = torch.zeros(len(COMMAND_MODE_NAMES), device=env.unwrapped.device)
        interval_clearance = torch.zeros(2, device=env.unwrapped.device)
        interval_valid = interval_steps = interval_falls = invalid_env_resets = 0
        from rp_wd_rl.supermimic.pointfoot_gait import PointfootGaitMetrics
        gait_metrics = PointfootGaitMetrics(env.unwrapped.num_envs, env.unwrapped.device, env.unwrapped.step_dt)
        began = time.monotonic()
        replay_mib = sum(t.numel() * t.element_size() for t in learner.replay.data.values()) / 2**20
        print(f"[信息] 回放容量={args.buffer_len}步，显存约{replay_mib:.1f}MiB", flush=True)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) as log:
            for iteration in range(start_iteration, args.iterations):
                learner.actor_optimizer.param_groups[0]["lr"] = learner.actor_learning_rate(iteration)
                policy_metrics = {}
                for _ in range(args.steps_per_iteration):
                    with torch.no_grad():
                        actor_features = (learner.actor.features(current["history"], current["phase"])
                                          if adapter.gait_enabled else current["history"])
                        learner.actor.norm.update(actor_features)
                        rho = 0.95
                        noise_std = args.action_noise
                        if args.action_noise_final is not None:
                            fraction = min(1., iteration / args.noise_decay_iterations)
                            noise_std += fraction * (args.action_noise_final - args.action_noise)
                        ou_noise.mul_(rho).add_(
                            torch.randn_like(ou_noise), alpha=noise_std * (1.0 - rho**2) ** 0.5
                        )
                        if adapter.gait_enabled:
                            action = learner.actor(current["history"], current["phase"], noise=ou_noise)
                        else:
                            action = (learner.actor(current["history"]) + ou_noise).clamp(-1.0, 1.0)
                        current = {key: value.clone() for key, value in current.items()}
                        critic_real = torch.cat((current["state"], current["command"], current["context"],
                                                 previous, current["scan"]), -1)
                        if adapter.gait_enabled:
                            from rp_wd_rl.supermimic.pointfoot_gait import phase_features
                            critic_real = torch.cat((critic_real, phase_features(current["phase"])), -1)
                        learner.critic.norm.update(critic_real)
                        learner.target_critic.norm.load_state_dict(learner.critic.norm.state_dict())
                        def physical_snapshot():
                            snapshot = adapter.capture({"policy": current["history"]})
                            bad_physical = ~_finite_rows(snapshot)
                            if bad_physical.any():
                                # 必须在env.step自己的自动reset之前否决无效成功，不能等返回后补标。
                                env.unwrapped.termination_manager.terminated[bad_physical] = True
                                episode = getattr(env.unwrapped, "_pointfoot_full_episode", None)
                                if episode is not None:
                                    episode["passed"][bad_physical] = False
                                pulse = getattr(env.unwrapped, "_pointfoot_success_event", None)
                                if pulse is not None:
                                    pulse[bad_physical] = False
                            return (snapshot, env.unwrapped.termination_manager.terminated.clone(),
                                    env.unwrapped.termination_manager.time_outs.clone())
                        with capture_before_reset(env.unwrapped.termination_manager, physical_snapshot,
                                                  honor_captured_termination=True) as snapshots:
                            next_observations, reward, _, _, _ = env.step(action)
                        physical, terminated, truncated = snapshots[0]
                        following = adapter.capture(next_observations)
                        bad = ~_finite_rows(physical) | ~_finite_rows(following) | ~torch.isfinite(reward)
                        if bad.any():
                            invalid_env_resets += int(bad.sum())
                            # 已被物理快照标失败且自动reset成功的行不再重复reset。
                            recover = bad & (~terminated | ~_finite_rows(following))
                            if recover.any():
                                next_observations = recover_invalid_rows(env.unwrapped, recover)
                                following = adapter.capture(next_observations)
                            if not _finite_rows(following).all():
                                raise RuntimeError("无效环境复位后仍有非有限状态，停止以保护模型")
                            reward = torch.where(bad, 0.0, reward)
                        terminated = terminated.reshape(-1).bool() | bad
                        truncated = truncated.reshape(-1).bool()
                        done = terminated | truncated
                        valid = ~done
                        transition = {
                            "state": current["state"],
                            "context": current["context"], "bias": current["bias"],
                            "world_pose": current["pose"],
                            "action": action,
                            "previous": previous,
                            "command": current["command"],
                            "scan": current["scan"],
                            "force": current["force"],
                            "contacts": current["contacts"],
                            "history": current["history"],
                            "pose": _pose_delta(current["pose"], physical["pose"]),
                            "next_state": physical["state"],
                            "next_scan": physical["scan"],
                            "next_force": physical["force"],
                            "next_contacts": physical["contacts"],
                            "reward": reward[:, None],
                            "collision": physical["collision"],
                        }
                        if adapter.gait_enabled:
                            transition["phase"] = current["phase"]
                        event = (done | (physical["collision"].squeeze(-1) > 0)
                                 | ((physical["contacts"] - current["contacts"]).abs().sum(-1) > 0)
                                 | (physical["force"].amax(-1) > 0.35))
                        useful = getattr(env.unwrapped, "_pointfoot_success_event", torch.zeros_like(done))
                        ground_change = (adapter.field.sample(physical["pose"][:, :2])
                                         - adapter.field.sample(current["pose"][:, :2])).abs()
                        aligned = (physical["state"][:, 4:6] * current["command"][:, :2]).sum(-1) > 0
                        useful = (useful | ((ground_change > .005) & valid & aligned)) & ~bad
                        learner.model.update_stats(transition, valid)
                        learner.replay.add(transition, terminated, valid, event, success_event=useful)
                        gait_metrics.update(physical, done)
                        episode_return += reward
                        if done.any():
                            recent_returns.extend(episode_return[done].cpu().tolist())
                            episode_return[done] = 0.0
                            previous[done] = 0.0
                            ou_noise[done] = 0.0
                        previous[~done] = action[~done]
                        measured = torch.cat((physical["state"][:, 4:6], physical["state"][:, 9:10]), -1)
                        interval_error += (measured[valid] - current["command"][valid]).abs().sum(0)
                        interval_xy_error += (measured[valid, :2] - current["command"][valid, :2]).norm(dim=-1).sum()
                        interval_velocity += measured[valid].sum(0)
                        interval_command += current["command"][valid].sum(0)
                        modes = command_mode_ids(current["command"])
                        interval_mode_error.index_add_(0, modes[valid], (measured[valid] - current["command"][valid]).abs())
                        interval_mode_velocity.index_add_(0, modes[valid], measured[valid])
                        interval_mode_command.index_add_(0, modes[valid], current["command"][valid])
                        interval_mode_count.index_add_(0, modes[valid], torch.ones_like(modes[valid], dtype=torch.float))
                        interval_mode_falls.index_add_(0, modes[terminated], torch.ones_like(modes[terminated], dtype=torch.float))
                        support = adapter.task_rewards.support_clearance(
                            physical["state"], physical["scan"], physical["contacts"], physical["force"])
                        interval_clearance += torch.stack((physical["state"][valid, 0].sum(), support[valid].sum()))
                        interval_valid += int(valid.sum())
                        interval_falls += int(terminated.sum())
                        interval_reward += reward.mean()
                        safe_terms = env.unwrapped.reward_manager._step_reward.detach()[~bad]
                        if len(safe_terms):
                            interval_terms += safe_terms.mean(0)
                        interval_steps += 1
                        current = following
                        observations = next_observations
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
                        "actor_lr": learner.actor_optimizer.param_groups[0]["lr"],
                        "simulation_steps": simulation_steps,
                        "real_transitions": simulation_steps * env.unwrapped.num_envs,
                        "reward_mean": float(interval_reward / max(interval_steps, 1)),
                        "velocity_mae": (interval_error / max(interval_valid, 1)).tolist(),
                        "actual_velocity_mean": (interval_velocity / max(interval_valid, 1)).tolist(),
                        "command_mean": (interval_command / max(interval_valid, 1)).tolist(),
                        "falls": interval_falls, "interval_steps": interval_steps,
                        "episode_return_mean": (
                            sum(recent_returns) / len(recent_returns) if recent_returns else None
                        ),
                        "velocity_error_xy": float(interval_xy_error / max(interval_valid, 1)),
                        "replay_steps": learner.replay.size,
                        "invalid_env_resets": invalid_env_resets,
                        "root_clearance_mean": float(interval_clearance[0] / max(interval_valid, 1)),
                        "support_clearance_mean": float(interval_clearance[1] / max(interval_valid, 1)),
                        "command_modes": {name: {
                            "samples": int(interval_mode_count[i]), "falls": int(interval_mode_falls[i]),
                            "velocity_mae": (interval_mode_error[i] / interval_mode_count[i].clamp_min(1)).tolist(),
                            "actual_mean": (interval_mode_velocity[i] / interval_mode_count[i].clamp_min(1)).tolist(),
                            "command_mean": (interval_mode_command[i] / interval_mode_count[i].clamp_min(1)).tolist()
                        } for i, name in enumerate(COMMAND_MODE_NAMES)},
                        "policy_updates": learner.policy_updates,
                        "skipped_updates": learner.skipped_updates,
                        "mean_terrain_level": float(terrain.terrain_levels.float().mean()),
                        "elapsed_seconds": round(time.monotonic() - began, 2),
                        **model_metrics,
                        **policy_metrics,
                    }
                    term_means = interval_terms / max(interval_steps, 1)
                    row["reward_terms_per_second"] = {
                        name: float(term_means[index].detach())
                        for index, name in enumerate(env.unwrapped.reward_manager.active_terms)
                    }
                    row["gait_metrics"] = gait_metrics.summary()
                    if adapter.gait_enabled:
                        from rp_wd_rl.tasks.locomotion.velocity.mdp.pointfoot_full import update_gait_quality
                        row["gait_quality_ready"] = update_gait_quality(env.unwrapped, row["gait_metrics"])
                    row["exploration_std"] = noise_std
                    gait_metrics.clear_interval()
                    if hasattr(env.unwrapped, "_pointfoot_course_caps"):
                        row["terrain_level_caps"] = env.unwrapped._pointfoot_course_caps.tolist()
                    row["course_trials_by_type_level"] = env.unwrapped._pointfoot_course_counts.tolist()
                    row["course_recent_success_by_type_level"] = env.unwrapped._pointfoot_course_trials.mean(-1).tolist()
                    if (iteration + 1) % args.wm_diagnostics_interval == 0 and learner.replay.size >= args.wm_window:
                        row.update(learner.prediction_metrics(batch_size=args.diagnostic_batch))
                    row.update(learner.replay_stats())
                    if contract["task_revision"] >= 3:
                        from rp_wd_rl.tasks.locomotion.velocity.mdp.pointfoot_full import full_summary, update_full_command_curriculum
                        ready = command_skills_ready(row["command_modes"])
                        row["command_skills_ready"] = ready
                        update_full_command_curriculum(env.unwrapped, iteration + 1,
                                                       row["velocity_mae"] if ready else [float("inf")] * 3)
                        row["full_course"] = full_summary(env.unwrapped)
                    log.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print("[训练] " + json.dumps(row, ensure_ascii=False), flush=True)
                    interval_reward.zero_(); interval_error.zero_(); interval_terms.zero_(); interval_xy_error.zero_()
                    interval_velocity.zero_(); interval_command.zero_()
                    interval_mode_error.zero_(); interval_mode_count.zero_(); interval_mode_falls.zero_()
                    interval_mode_velocity.zero_(); interval_mode_command.zero_()
                    interval_clearance.zero_()
                    interval_valid = interval_steps = interval_falls = 0

                if (iteration + 1) % args.save_interval == 0 or iteration + 1 == args.iterations:
                    payload = learner.state_dict(iteration + 1)
                    payload["simulation_steps"] = simulation_steps
                    payload["task"] = args.task
                    payload["training_contract"] = contract
                    payload["task_revision"] = contract["task_revision"]
                    payload["terrain_state"] = _course_payload(env.unwrapped)
                    _atomic_save(payload, run_dir / "teacher_latest.pt")
                    _atomic_save(payload, run_dir / f"teacher_iteration_{iteration + 1}.pt")
                    _prune_checkpoints(run_dir)

        final = learner.state_dict(args.iterations)
        final["simulation_steps"] = simulation_steps
        final["task"] = args.task
        final["training_contract"] = contract
        final["task_revision"] = contract["task_revision"]
        final["terrain_state"] = _course_payload(env.unwrapped)
        _atomic_save(final, run_dir / "teacher_final.pt")
        print(f"[完成] 点足 SuperMimic checkpoint：{run_dir / 'teacher_final.pt'}", flush=True)
        env.close()
    except BaseException:
        env.close()
        raise


if __name__ == "__main__":
    try:
        _main()
    except BaseException:
        traceback.print_exc()
        shutdown(simulation_app, exit_code=1)
    else:
        shutdown(simulation_app)
