#!/usr/bin/env python3
"""点足盲策略确定性验收：八种运动指令、六类地形，保存复位前指标。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback
from zoneinfo import ZoneInfo

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--task", default=None, help="省略时使用checkpoint记录的任务")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--suite", choices=["commands", "terrain", "all"], default="all")
parser.add_argument("--cases", nargs="+", default=None, help="只验收指定场景，例如forward up_forward up_reverse")
parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2], choices=range(6))
parser.add_argument("--seeds", type=int, nargs="+", default=[42])
parser.add_argument("--seconds", type=float, default=8.0, help="运动指令验收时长")
parser.add_argument("--terrain_seconds", type=float, default=35.0)
parser.add_argument("--terrain_speed", type=float, default=.20, help="地形测试前后速度，默认匹配当前低速课程")
parser.add_argument("--settle_seconds", type=float, default=2.0)
parser.add_argument("--max_steps", type=int, default=0, help="仅用于工具冒烟，0表示完整验收")
parser.add_argument("--output", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or min(args.seconds, args.terrain_seconds) <= 0 or args.settle_seconds < 0 or args.max_steps < 0:
    parser.error("环境数/测量时长必须为正，稳定时间和max_steps不得为负")
if args.terrain_speed <= 0:
    parser.error("terrain_speed必须为正数")
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa:E402
import torch  # noqa:E402
import rp_wd_rl.tasks  # noqa:E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa:E402
from rp_wd_rl.supermimic.pointfoot import FORMAT, PointfootActor  # noqa:E402
from pointfoot_runtime import (  # noqa:E402
    LEGACY_TASK, PointfootStateAdapter, resolve_pointfoot_checkpoint, set_pointfoot_command,
    course_frame_pass, CoursePassTracker,
    actor_from_checkpoint, policy_action,
)
from transition_capture import capture_before_reset  # noqa:E402
from sim_shutdown import shutdown  # noqa:E402


COMMAND_CASES = [
    ("stand", (0., 0., 0.)), ("forward", (.3, 0., 0.)), ("reverse", (-.3, 0., 0.)),
    ("left", (0., .2, 0.)), ("right", (0., -.2, 0.)),
    ("yaw_left", (0., 0., .5)), ("yaw_right", (0., 0., -.5)),
    ("combined", (.25, .10, .35)),
]


def case_list(available):
    result = []
    if args.suite in ("commands", "all"):
        result += [(name, "flat", 0, cmd, args.seconds) for name, cmd in COMMAND_CASES]
    if args.suite in ("terrain", "all"):
        for kind in ("up", "down", "platform", "slope", "rubble"):
            if kind not in available:
                continue
            for level in args.levels:
                for direction in ((1, -1) if kind in ("up", "down") else (1,)):
                    result.append((f"{kind}_{'forward' if direction > 0 else 'reverse'}", kind,
                                   level, (getattr(args, "terrain_speed", .3) * direction, 0., 0.),
                                   args.terrain_seconds))
    selected = getattr(args, "cases", None)
    if selected:
        result = [case for case in result if case[0] in selected]
        missing = set(selected) - {case[0] for case in result}
        if missing:
            raise ValueError(f"当前任务/套件不包含验收场景：{sorted(missing)}")
    return result


def column_kinds(cfg):
    import numpy as np
    gen = cfg.scene.terrain.terrain_generator
    items = list(gen.sub_terrains.items())
    proportions = np.array([x.proportion for _, x in items])
    cumulative = (proportions / proportions.sum()).cumsum()
    result = []
    for column in range(gen.num_cols):
        i = int(np.where(column / gen.num_cols + .001 < cumulative)[0][0])
        name, sub = items[i]
        kind = getattr(sub, "kind", {"stairs_up": "up", "rough": "rubble"}.get(name, name))
        result.append(kind)
    return result


def main():
    checkpoint_path = resolve_pointfoot_checkpoint(args.checkpoint)
    device = args.device or "cuda:0"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("需要225→12点足世界模型策略")
    task = args.task or checkpoint.get("task", LEGACY_TASK)
    actor = actor_from_checkpoint(checkpoint, device)
    cfg = parse_env_cfg(task, device=device, num_envs=args.num_envs)
    if (getattr(cfg, "supermimic_revision", 2) >= 4) != bool(checkpoint.get("gait_enabled", False)):
        raise ValueError("模型与任务的步态版本不匹配")
    cfg.seed = int(checkpoint.get("config", {}).get("seed", 42))
    cfg.curriculum = {}
    cfg.observations.policy.enable_corruption = False
    cfg.episode_length_s = max(args.seconds, args.terrain_seconds) + args.settle_seconds + 10
    if getattr(cfg, "supermimic_revision", 2) >= 3:
        cfg.commands.base_velocity.manual_control = True
        cfg.events.reset_base.params["free_yaw_range"] = (-.08, .08)
    cfg.commands.base_velocity.resampling_time_range = (1e9, 1e9)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0., 0.)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0., 0.)
    kinds = column_kinds(cfg)
    if args.suite in ("commands", "all") and "flat" not in kinds:
        raise ValueError("指令验收需要flat地形")
    out = Path(args.output or (Path("/media/rmer/data/rpwd_training/pointfoot_evaluations") /
              datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")))
    if (out / "summary.json").exists() or (out / "trials.csv").exists():
        raise FileExistsError(f"避免覆盖已有评测：{out}")
    out.mkdir(parents=True, exist_ok=True)
    env = gym.make(task, cfg=cfg, render_mode=None)
    rows = []
    try:
        raw = env.unwrapped
        adapter = PointfootStateAdapter(env)
        for seed in args.seeds:
            for name, kind, level, command_values, duration in case_list(set(kinds)):
                column = kinds.index(kind)
                terrain = raw.scene.terrain
                terrain.terrain_types.fill_(column)
                terrain.terrain_levels.fill_(level)
                terrain.env_origins.copy_(terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
                origins = terrain.env_origins.clone()
                with torch.inference_mode():
                    obs, _ = env.reset(seed=seed)
                command = torch.tensor(command_values, device=device)
                count = torch.zeros(raw.num_envs, device=device)
                sums = torch.zeros(raw.num_envs, 3, device=device)
                errors = torch.zeros_like(sums)
                heights = torch.zeros(raw.num_envs, device=device)
                failed = torch.zeros(raw.num_envs, dtype=torch.bool, device=device)
                ended = torch.zeros_like(failed)
                tracker = CoursePassTracker(raw.num_envs, device, raw.step_dt)
                from rp_wd_rl.supermimic.pointfoot_gait import PointfootGaitMetrics
                gait_metrics = PointfootGaitMetrics(raw.num_envs, device, raw.step_dt)
                reached, progress = tracker.reached, tracker.progress
                limit = int((duration + args.settle_seconds) / raw.step_dt)
                settle = int(args.settle_seconds / raw.step_dt)
                if args.max_steps:
                    limit = min(limit, args.max_steps)
                    settle = min(settle, max(0, limit // 4))
                for step in range(limit):
                    with torch.inference_mode():
                        policy_input = obs["policy"].clone()
                        set_pointfoot_command(raw, policy_input, command)
                        def snapshot():
                            return (adapter.capture({"policy": policy_input}),
                                    raw.termination_manager.terminated.clone(),
                                    raw.termination_manager.time_outs.clone())
                        with capture_before_reset(raw.termination_manager, snapshot) as captured:
                            obs, _, _, _, _ = env.step(policy_action(actor, policy_input, env))
                        physical, terminated, truncated = captured[0]
                        state = physical["state"]
                        finite = torch.stack([torch.isfinite(physical[k]).flatten(1).all(-1)
                                              for k in ("state", "pose", "force", "foot_positions")]).all(0)
                        active = ~ended
                        failed |= active & (terminated | ~finite)
                        velocity = torch.cat((state[:, 4:6], state[:, 9:10]), -1)
                        support = adapter.task_rewards.support_clearance(
                            state, physical["scan"], physical["contacts"], physical["force"])
                        if step >= settle:
                            keep = active & finite
                            count += keep
                            sums[keep] += velocity[keep]
                            errors[keep] += (velocity[keep] - command).abs()
                            heights[keep] += support[keep]
                        delta = physical["pose"][:, :2] - origins[:, :2]
                        direction = 1 if command_values[0] >= 0 else -1
                        distance = direction * delta[:, 0]
                        from rp_wd_rl.supermimic.pointfoot_courses import COURSE_KINDS
                        frame_pass = course_frame_pass(
                            physical, origins, torch.full_like(terrain.terrain_levels, COURSE_KINDS.index(kind)),
                            torch.full_like(terrain.terrain_levels, level),
                            step_heights=getattr(cfg, "pointfoot_step_heights", None))
                        required = 2.8 if kind in ("up", "down") else 5.4
                        if getattr(cfg, "supermimic_revision", 2) < 3 and kind == "up":
                            required = 3.3
                        tracker.update(distance, frame_pass & (distance > required), active, failed)
                        ended |= terminated | truncated | ~finite
                        gait_metrics.update(physical, ended)
                    if ended.all():
                        break
                means = sums / count.clamp_min(1)[:, None]
                maes = errors / count.clamp_min(1)[:, None]
                height_means = heights / count.clamp_min(1)
                for i in range(raw.num_envs):
                    tracking = bool((maes[i, :2] < .15).all() and maes[i, 2] < .30)
                    for axis, target in enumerate(command_values):
                        if abs(target) > .05:
                            tracking &= float(means[i, axis]) * (1 if target > 0 else -1) >= .5 * abs(target)
                    if name == "stand":
                        tracking = bool(means[i, :2].norm() < .05 and maes[i, 2] < .1 and maes[i, :2].max() < .06)
                    passed = (bool(reached[i]) if name not in dict(COMMAND_CASES) else tracking)
                    passed &= not bool(failed[i]) and int(count[i]) > 0
                    rows.append({"case": name, "terrain": kind, "level": level, "seed": seed, "env": i,
                                 "command": list(command_values), "actual_mean": means[i].tolist(),
                                 "velocity_mae": maes[i].tolist(), "support_clearance_mean": float(height_means[i]),
                                 "max_forward_progress": float(progress[i]), "samples": int(count[i]),
                                 "gait_case_aggregate": gait_metrics.summary(),
                                 "failed": bool(failed[i]), "reached": bool(reached[i]), "pass": passed})
                subset = rows[-raw.num_envs:]
                print(f"[验收] {name} 等级={level} seed={seed} 通过={sum(r['pass'] for r in subset)}/{len(subset)}", flush=True)
        report = {"checkpoint": str(checkpoint_path), "checkpoint_step": checkpoint.get("step"),
                  "step_heights": list(getattr(cfg, "pointfoot_step_heights", (.02, .04, .08, .12, .16, .20))),
                  "task": task, "evaluation_protocol": 3, "landing_stability_seconds": .12,
                  "deterministic_actor": True, "observation_noise": False,
                  "smoke_only": bool(args.max_steps), "config": vars(args), "trials": rows}
        (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if rows:
            with (out / "trials.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        print(f"[完成] 确定性评测数据：{out}；smoke_only={bool(args.max_steps)}", flush=True)
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
