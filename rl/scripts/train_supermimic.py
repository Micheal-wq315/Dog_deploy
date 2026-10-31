"""RP_wd SuperMimic首版：可微世界模型教师训练与Odin深度学生在线蒸馏。"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=("teacher", "student"), required=True,
                    help="teacher=世界模型想象训练；student=在线DAgger深度蒸馏")
parser.add_argument("--task", default=None, help="默认按阶段选择SuperMimic任务")
parser.add_argument("--teacher_checkpoint", default=None, help="student阶段需要的教师checkpoint")
parser.add_argument("--resume_teacher", default=None,
                    help="教师阶段从teacher_latest.pt续训；环境和回放池重新初始化")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=None,
                    help="默认teacher=100000、student=20000次学习更新")
parser.add_argument("--steps_per_iteration", type=int, default=24,
                    help="每次学习更新收集的50Hz真实仿真步，与PPO的24步rollout对齐")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", default="rl/logs/supermimic")
parser.add_argument("--run_name", default=None)
parser.add_argument("--warmup", type=int, default=512, help="教师开始想象更新前的真实采样步数")
parser.add_argument("--save_interval", type=int, default=500)
parser.add_argument("--log_interval", type=int, default=25)
parser.add_argument("--replay_steps", type=int, default=512, help="回放时间容量；内存按环境数线性增长")
parser.add_argument("--ensemble", type=int, default=3)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batch", type=int, default=64)
parser.add_argument("--wm_updates", type=int, default=2)
parser.add_argument("--wm_lr", type=float, default=2.0e-4)
parser.add_argument("--imag_batch", type=int, default=64)
parser.add_argument("--horizon", type=int, default=8)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--actor_lr", type=float, default=1.0e-4)
parser.add_argument("--critic_lr", type=float, default=3.0e-4)
parser.add_argument("--action_noise", type=float, default=0.45)
parser.add_argument("--uncertainty_coef", type=float, default=0.05)
parser.add_argument("--imagination_reward_clip", type=float, default=300.0,
                    help="想象奖励裁剪幅值；默认保留台阶成功的250分奖励信号")
parser.add_argument("--actor_hidden", type=int, nargs="+", default=(512, 512, 256))
parser.add_argument("--critic_hidden", type=int, nargs="+", default=(512, 256))
parser.add_argument("--wm_hidden", type=int, nargs="+", default=(384, 384, 384))
parser.add_argument("--student_lr", type=float, default=3.0e-4)
parser.add_argument("--student_batch", type=int, default=256)
parser.add_argument("--student_replay", type=int, default=8192)
parser.add_argument("--student_updates", type=int, default=1)
parser.add_argument("--dagger_beta_start", type=float, default=1.0)
parser.add_argument("--dagger_beta_end", type=float, default=0.1)
parser.add_argument("--dagger_anneal_steps", type=int, default=50000)
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()

if args.num_envs < 1 or args.steps_per_iteration < 1 or args.save_interval < 1 or args.log_interval < 1:
    parser.error("环境数与日志/保存间隔必须为正数")
if args.stage == "teacher":
    args.iterations = args.iterations or 100000
    if args.warmup < args.wm_window or args.replay_steps < args.wm_window:
        parser.error("教师 warmup 和 replay_steps 必须不小于 wm_window")
    if min(args.ensemble, args.wm_window, args.wm_batch, args.wm_updates,
           args.imag_batch, args.horizon) < 1:
        parser.error("世界模型、回放与想象参数必须为正数")
    if args.imagination_reward_clip <= 0 or args.uncertainty_coef < 0:
        parser.error("想象奖励裁剪必须为正数，不确定性惩罚不能为负")
else:
    args.iterations = args.iterations or 20000
    if not args.teacher_checkpoint:
        parser.error("student阶段需要 --teacher_checkpoint")
    if args.resume_teacher:
        parser.error("--resume_teacher 只用于 teacher 阶段")
    if args.student_replay < args.student_batch:
        parser.error("student_replay 必须不小于 student_batch")
args.headless = True
if args.stage == "student":
    args.enable_cameras = True
sys.argv = [sys.argv[0], *unknown]
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import rp_wd_rl.tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from rp_wd_rl.supermimic.learner import ModelBasedLearner  # noqa: E402
from rp_wd_rl.supermimic.model import (  # noqa: E402
    ACTION_LIMIT,
    DEPLOY_INPUT_DIM,
    STUDENT_FORMAT,
    TEACHER_FORMAT,
    DeterministicTeacher,
    OdinDepthStudent,
    make_actor_features,
    student_from_checkpoint,
    teacher_from_checkpoint,
)
from sim_shutdown import shutdown  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1).to(dtype=torch.bool)


def _atomic_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _prune_periodic_checkpoints(run_dir: Path, prefix: str, keep: int = 5) -> None:
    checkpoints = sorted(
        run_dir.glob(f"{prefix}_iteration_*.pt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    for stale in checkpoints[:-keep]:
        stale.unlink(missing_ok=True)


class RPWDStateAdapter:
    """把 Isaac Lab 真值映射到 38维状态、局部扫描及接触监督。"""

    def __init__(self, env):
        self.env = env.unwrapped
        self.robot = self.env.scene["robot"]
        self.leg_ids = torch.as_tensor(
            self.env.action_manager.get_term("leg_pos")._joint_ids,
            device=self.env.device, dtype=torch.long,
        )
        self.wheel_ids = torch.as_tensor(
            self.env.action_manager.get_term("wheel_vel")._joint_ids,
            device=self.env.device, dtype=torch.long,
        )
        if self.leg_ids.numel() != 12 or self.wheel_ids.numel() != 4:
            raise RuntimeError("SuperMimic状态适配要求12个腿关节和4个轮关节")
        if self.env.action_manager.total_action_dim != 16:
            raise RuntimeError("RP_wd动作接口应为16维")

    def capture(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        data = self.robot.data
        joint_pos = data.joint_pos - data.default_joint_pos
        leg_pos = joint_pos[:, self.leg_ids]
        joint_vel = data.joint_vel[:, torch.cat((self.leg_ids, self.wheel_ids))]
        state = torch.cat((
            data.root_pos_w[:, 2:3],
            data.projected_gravity_b,
            data.root_lin_vel_b,
            data.root_ang_vel_b,
            leg_pos,
            joint_vel,
        ), dim=-1)
        if state.shape[-1] != 38:
            raise RuntimeError(f"RP_wd世界状态应为38维，实际为{state.shape[-1]}")

        quat = data.root_quat_w
        qw, qx, qy, qz = quat.unbind(-1)
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy.square() + qz.square()))
        pose = torch.cat((data.root_pos_w[:, :2], yaw.unsqueeze(-1)), dim=-1)

        velocity_command = self.env.command_manager.get_command("base_velocity")
        posture_command = self.env.command_manager.get_command("base_posture")
        command = torch.cat((velocity_command, posture_command), dim=-1)

        scan = observations["supermimic_terrain"].reshape(self.env.num_envs, -1)
        privileged = observations["supermimic_state"].reshape(self.env.num_envs, -1)
        wheel_force = privileged[:, -4:].clamp(0.0, 1.0)
        contacts = (wheel_force > 0.01).float()
        torque = (data.applied_torque[:, self.leg_ids] / 42.0).clamp(-1.0, 1.0)
        if scan.shape[-1] != 192 or command.shape[-1] != 6:
            raise RuntimeError(
                f"SuperMimic观测维度不匹配: scan={scan.shape[-1]}, command={command.shape[-1]}"
            )
        return {
            "state": state,
            "pose": pose,
            "command": command,
            "scan": scan,
            "torque": torque,
            "contacts": contacts,
            "wheel_force": wheel_force,
        }


class StudentReplay:
    def __init__(self, capacity: int, device: torch.device):
        self.observation = torch.empty(capacity, DEPLOY_INPUT_DIM, device=device)
        self.action = torch.empty(capacity, 16, device=device)
        self.capacity = capacity
        self.position = 0
        self.size = 0

    def add(self, observation: torch.Tensor, action: torch.Tensor) -> None:
        count = observation.shape[0]
        indices = (torch.arange(count, device=observation.device) + self.position) % self.capacity
        self.observation[indices] = observation.detach()
        self.action[indices] = action.detach()
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(self.size, (batch_size,), device=self.observation.device)
        return self.observation[indices], self.action[indices]


def _task_for_stage(stage: str) -> str:
    return (
        "Isaac-Velocity-Rough-RPWD-OdinDepth-SuperMimic-WM-Teacher-v0"
        if stage == "teacher"
        else "Isaac-Velocity-Rough-RPWD-OdinDepth-SuperMimic-WM-Student-v0"
    )


def _make_env():
    task = args.task or _task_for_stage(args.stage)
    cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(task, cfg=cfg, render_mode=None)
    print(f"[信息] 阶段={args.stage} 任务={task} 环境数={env.unwrapped.num_envs}", flush=True)
    return env


def _new_run_dir() -> Path:
    root = Path(args.output_dir)
    if not root.is_absolute():
        root = REPO_ROOT / root
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_name:
        name += f"_{args.run_name}"
    run_dir = root / args.stage / name
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, ensure_ascii=False, indent=2, default=str)
    return run_dir


def _teacher_environment_step(env, adapter, learner, current, previous_action, ou_noise,
                             episode_return, recent_returns):
    num_envs = env.unwrapped.num_envs
    actor_features = make_actor_features(
        current["state"], current["command"], previous_action,
        current["scan"], current["torque"], current["contacts"],
    )
    with torch.no_grad():
        policy_action = learner.actor(actor_features)
        rho = 0.95
        innovation = args.action_noise * (1.0 - rho * rho) ** 0.5
        ou_noise.mul_(rho).add_(torch.randn_like(ou_noise), alpha=innovation)
        action = (policy_action + ou_noise).clamp(-ACTION_LIMIT, ACTION_LIMIT)
        observations, reward, terminated, truncated, _ = env.step(action)
        next_state = adapter.capture(observations)
        terminated = _as_bool(terminated)
        truncated = _as_bool(truncated)
        done = terminated | truncated
        valid_dynamics = ~done
        yaw_delta = next_state["pose"][:, 2] - current["pose"][:, 2]
        yaw_delta = torch.atan2(yaw_delta.sin(), yaw_delta.cos())
        displacement = next_state["pose"][:, :2] - current["pose"][:, :2]
        yaw = current["pose"][:, 2]
        cosine, sine = yaw.cos(), yaw.sin()
        local_x = cosine * displacement[:, 0] + sine * displacement[:, 1]
        local_y = -sine * displacement[:, 0] + cosine * displacement[:, 1]
        pose_delta = torch.stack((local_x, local_y, yaw_delta), dim=-1)
        event_mask = terminated | (next_state["wheel_force"].amax(-1) > 0.35) | (
            (next_state["contacts"] - current["contacts"]).abs().sum(-1) > 0.0
        )
        transition = {
            "state": current["state"], "action": action,
            "previous_action": previous_action, "command": current["command"],
            "scan": current["scan"], "pose": current["pose"],
            "torque": current["torque"], "contacts": current["contacts"],
            "wheel_force": current["wheel_force"],
            "next_state": next_state["state"], "pose_delta": pose_delta,
            "next_scan": next_state["scan"], "next_torque": next_state["torque"],
            "next_contacts": next_state["contacts"],
            "next_wheel_force": next_state["wheel_force"],
            "reward": reward.reshape(num_envs, 1),
        }
        learner.replay.add(transition, terminated, valid_dynamics, event_mask)
        episode_return += reward
        if done.any():
            recent_returns.extend(episode_return[done].detach().cpu().tolist())
            episode_return[done] = 0.0
            previous_action[done] = 0.0
            ou_noise[done] = 0.0
        previous_action[~done] = action[~done]
    return observations, next_state, previous_action, ou_noise, episode_return, reward


def _collect_teacher(env, adapter: RPWDStateAdapter, run_dir: Path) -> None:
    learner = ModelBasedLearner(args, torch.device(env.unwrapped.device), env.unwrapped.num_envs)
    start_iteration = 0
    simulation_steps = 0
    if args.resume_teacher:
        checkpoint = torch.load(args.resume_teacher, map_location=env.unwrapped.device, weights_only=True)
        start_iteration = learner.load_state_dict(checkpoint)
        simulation_steps = int(checkpoint.get("simulation_steps", 0))
        if start_iteration >= args.iterations:
            raise ValueError(
                f"续训起点{start_iteration}已达到目标轮数{args.iterations}；请提高 --iterations"
            )
    observations, _ = env.reset(seed=args.seed)
    # 恢复课程时钟；各环境实体和回放池从新仿真初始化，回放重新热身。
    env.unwrapped.common_step_counter = simulation_steps
    current = adapter.capture(observations)
    num_envs = env.unwrapped.num_envs
    previous_action = torch.zeros(num_envs, 16, device=env.unwrapped.device)
    ou_noise = torch.zeros_like(previous_action)
    episode_return = torch.zeros(num_envs, device=env.unwrapped.device)
    recent_returns: deque[float] = deque(maxlen=100)
    model_metrics: dict[str, float] = {}
    learner.actor.train()
    started = time.time()

    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) as metrics_file:
        for iteration in range(start_iteration, args.iterations):
            rewards = []
            for _ in range(args.steps_per_iteration):
                (observations, current, previous_action, ou_noise, episode_return,
                 reward) = _teacher_environment_step(
                    env, adapter, learner, current, previous_action, ou_noise,
                    episode_return, recent_returns,
                )
                rewards.append(reward)
                simulation_steps += 1

            if learner.replay.total_steps >= args.warmup:
                for _ in range(args.wm_updates):
                    model_metrics = learner.fit_world_model()
                policy_metrics = learner.imagine_and_improve()
            else:
                policy_metrics = {}

            if (iteration + 1) % args.log_interval == 0 or iteration == 0:
                row = {
                    "iteration": iteration + 1,
                    "simulation_steps": simulation_steps,
                    "real_transitions": simulation_steps * num_envs,
                    "reward_mean": float(torch.stack(rewards).mean().detach()),
                    "episode_return_mean": (sum(recent_returns) / len(recent_returns)
                                            if recent_returns else None),
                    "velocity_error_xy": float((current["state"][:, 4:6]
                                                 - current["command"][:, :2]).norm(dim=-1).mean()),
                    "replay_steps": learner.replay.total_steps,
                    **model_metrics,
                    **policy_metrics,
                    "elapsed_seconds": round(time.time() - started, 2),
                }
                metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print("[训练] " + json.dumps(row, ensure_ascii=False), flush=True)

            if (iteration + 1) % args.save_interval == 0 or iteration + 1 == args.iterations:
                payload = learner.state_dict(iteration + 1)
                payload["format"] = TEACHER_FORMAT
                payload["actor_hidden"] = learner.actor.hidden
                payload["simulation_steps"] = simulation_steps
                _atomic_save(payload, run_dir / "teacher_latest.pt")
                _atomic_save(payload, run_dir / f"teacher_iteration_{iteration + 1}.pt")
                _prune_periodic_checkpoints(run_dir, "teacher")

    final_path = run_dir / "teacher_final.pt"
    payload = learner.state_dict(args.iterations)
    payload["format"] = TEACHER_FORMAT
    payload["actor_hidden"] = learner.actor.hidden
    payload["simulation_steps"] = simulation_steps
    _atomic_save(payload, final_path)
    with (run_dir / "training_result.json").open("w", encoding="utf-8") as stream:
        json.dump({"stage": "teacher", "completed_iterations": args.iterations,
                   "simulation_steps": simulation_steps,
                   "real_transitions": simulation_steps * num_envs,
                   "checkpoint": str(final_path)}, stream, ensure_ascii=False, indent=2)
    print(f"[完成] 世界模型教师: {final_path}", flush=True)


def _student_observation(observations: dict[str, torch.Tensor]) -> torch.Tensor:
    policy = observations["policy"].reshape(observations["policy"].shape[0], -1)
    depth = observations["depth_memory"].reshape(observations["depth_memory"].shape[0], -1)
    result = torch.cat((policy, depth), dim=-1)
    if result.shape[-1] != DEPLOY_INPUT_DIM:
        raise RuntimeError(f"学生部署输入必须是{DEPLOY_INPUT_DIM}维，实际为{result.shape[-1]}")
    return result


def _export_student(student: OdinDepthStudent, path: Path) -> Path:
    from onnx_utils import consolidate_onnx, verify_onnx

    raw_path = path.with_name("policy_odin_supermimic_raw.onnx")
    torch.onnx.export(
        student.eval().cpu(), torch.zeros(1, DEPLOY_INPUT_DIM), str(raw_path),
        export_params=True, opset_version=18, input_names=["obs"], output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )
    consolidate_onnx(str(raw_path), str(path))
    raw_path.unlink(missing_ok=True)
    Path(f"{raw_path}.data").unlink(missing_ok=True)
    if not verify_onnx(str(path), expected_input_dim=DEPLOY_INPUT_DIM):
        raise RuntimeError(f"Odin SuperMimic ONNX 契约校验失败: {path}")
    return path


def _collect_student(env, adapter: RPWDStateAdapter, run_dir: Path) -> None:
    device = torch.device(env.unwrapped.device)
    teacher = teacher_from_checkpoint(args.teacher_checkpoint, device)
    student = OdinDepthStudent().to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.student_lr)
    replay = StudentReplay(args.student_replay, device)
    observations, _ = env.reset(seed=args.seed)
    current = adapter.capture(observations)
    student_observation = _student_observation(observations)
    previous_action = torch.zeros(env.unwrapped.num_envs, 16, device=device)
    episode_return = torch.zeros(env.unwrapped.num_envs, device=device)
    recent_returns: deque[float] = deque(maxlen=100)
    started = time.time()
    simulation_steps = 0

    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8", buffering=1) as metrics_file:
        for iteration in range(args.iterations):
            rewards = []
            beta_values = []
            for _ in range(args.steps_per_iteration):
                features = make_actor_features(
                    current["state"], current["command"], previous_action,
                    current["scan"], current["torque"], current["contacts"],
                )
                with torch.no_grad():
                    target_action = teacher(features)
                    student_action = student(student_observation)
                    progress = min(1.0, simulation_steps / max(args.dagger_anneal_steps, 1))
                    beta = args.dagger_beta_start + progress * (
                        args.dagger_beta_end - args.dagger_beta_start
                    )
                    executed_action = beta * target_action + (1.0 - beta) * student_action
                    replay.add(student_observation, target_action)

                    next_observations, reward, terminated, truncated, _ = env.step(executed_action)
                    next_state = adapter.capture(next_observations)
                    next_student_observation = _student_observation(next_observations)
                    done = _as_bool(terminated) | _as_bool(truncated)
                    episode_return += reward
                    if done.any():
                        recent_returns.extend(episode_return[done].detach().cpu().tolist())
                        episode_return[done] = 0.0
                        previous_action[done] = 0.0
                    previous_action[~done] = executed_action[~done]
                    current = next_state
                    observations = next_observations
                    student_observation = next_student_observation
                rewards.append(reward)
                beta_values.append(beta)
                simulation_steps += 1

            losses = []
            if replay.size >= args.student_batch:
                student.train()
                for _ in range(args.student_updates):
                    obs_batch, action_batch = replay.sample(args.student_batch)
                    prediction = student(obs_batch)
                    loss = F.smooth_l1_loss(prediction, action_batch, beta=0.1)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    losses.append((float(loss.detach()), float(grad_norm.detach())))
                student.eval()

            if (iteration + 1) % args.log_interval == 0 or iteration == 0:
                row = {
                    "iteration": iteration + 1,
                    "simulation_steps": simulation_steps,
                    "real_transitions": simulation_steps * env.unwrapped.num_envs,
                    "dagger_teacher_mix": beta_values[-1],
                    "student_loss": (sum(item[0] for item in losses) / len(losses) if losses else None),
                    "student_grad_norm": (sum(item[1] for item in losses) / len(losses) if losses else None),
                    "reward_mean": float(torch.stack(rewards).mean().detach()),
                    "episode_return_mean": (sum(recent_returns) / len(recent_returns)
                                            if recent_returns else None),
                    "replay_samples": replay.size,
                    "elapsed_seconds": round(time.time() - started, 2),
                }
                metrics_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                print("[蒸馏] " + json.dumps(row, ensure_ascii=False), flush=True)

            if (iteration + 1) % args.save_interval == 0 or iteration + 1 == args.iterations:
                payload = {
                    "format": STUDENT_FORMAT,
                    "step": iteration + 1,
                    "simulation_steps": simulation_steps,
                    "hidden": student.hidden,
                    "student": student.state_dict(),
                    "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
                    "input_dim": DEPLOY_INPUT_DIM,
                    "output_dim": 16,
                    "config": vars(args).copy(),
                }
                _atomic_save(payload, run_dir / "student_latest.pt")
                _atomic_save(payload, run_dir / f"student_iteration_{iteration + 1}.pt")
                _prune_periodic_checkpoints(run_dir, "student")

    final_checkpoint = run_dir / "student_final.pt"
    payload = {
        "format": STUDENT_FORMAT,
        "step": args.iterations,
        "simulation_steps": simulation_steps,
        "hidden": student.hidden,
        "student": student.state_dict(),
        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
        "input_dim": DEPLOY_INPUT_DIM,
        "output_dim": 16,
        "config": vars(args).copy(),
    }
    _atomic_save(payload, final_checkpoint)
    onnx_path = _export_student(student, run_dir / "policy_odin_supermimic.onnx")
    with (run_dir / "training_result.json").open("w", encoding="utf-8") as stream:
        json.dump({"stage": "student", "completed_iterations": args.iterations,
                   "simulation_steps": simulation_steps,
                   "real_transitions": simulation_steps * env.unwrapped.num_envs,
                   "checkpoint": str(final_checkpoint), "onnx": str(onnx_path),
                   "onnx_input": 1740, "onnx_output": 16}, stream,
                  ensure_ascii=False, indent=2)
    print(f"[完成] 深度学生: {final_checkpoint}", flush=True)
    print(f"[完成] 部署 ONNX: {onnx_path}", flush=True)


def main() -> None:
    if args.iterations < 1:
        raise ValueError("iterations 必须为正数")
    if args.stage == "teacher" and args.resume_teacher and not os.path.isfile(args.resume_teacher):
        raise FileNotFoundError(args.resume_teacher)
    if args.stage == "student" and not os.path.isfile(args.teacher_checkpoint):
        raise FileNotFoundError(args.teacher_checkpoint)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    env = _make_env()
    run_dir = _new_run_dir()
    adapter = RPWDStateAdapter(env)
    try:
        if args.stage == "teacher":
            _collect_teacher(env, adapter, run_dir)
        else:
            _collect_student(env, adapter, run_dir)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown(simulation_app)
