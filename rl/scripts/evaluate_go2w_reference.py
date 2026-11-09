#!/usr/bin/env python3
"""原版Go2W确定性测试与录像；不导入本仓库RP_wd训练任务。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--kinds", nargs="+", choices=("flat", "up", "down", "platform", "slope", "rubble"), default=["flat", "up"])
parser.add_argument("--levels", nargs="+", type=int, choices=range(6), default=[0, 1, 2])
parser.add_argument("--seeds", nargs="+", type=int, default=[0])
parser.add_argument("--vx", type=float, default=.3)
parser.add_argument("--seconds", type=float, default=20.)
parser.add_argument("--settle-seconds", type=float, default=2.)
parser.add_argument("--max-steps", type=int, default=0)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--video", action="store_true")
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()
if args.seconds <= 0 or args.settle_seconds < 0 or args.max_steps < 0 or not 0 < abs(args.vx) <= 1.:
    parser.error("时长必须为正、settle/max-steps不得为负，vx绝对值须在(0,1]内")


def main():
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from go2w_supermimic.core.bridge import Go2WBridge
    from go2w_supermimic.core.policy import load_actor, evaluation_env_cfg
    torch.set_num_threads(1)
    source = Path(os.environ["GO2W_SOURCE"])
    checkpoint_path = (args.checkpoint or source / "logs/go2w_terrain_v7_20000/model_20000.pt").resolve()
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    actor = load_actor(checkpoint, args.device)
    output = args.output or Path("/media/rmer/data/rpwd_training/go2w_supermimic/evaluations") / datetime.now(
        ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
    if (output / "summary.json").exists():
        raise FileExistsError(f"不覆盖已有测试：{output}")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    report = {"checkpoint": str(checkpoint_path), "step": checkpoint["step"],
              "model_origin": "随开源仓库提供" if checkpoint_path.is_relative_to(source / "logs") else "本机训练",
              "source_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
              "dependencies": {name: importlib.metadata.version(name) for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch")},
              "smoke_only": bool(args.max_steps), "observation_noise": False,
              "command": [args.vx, 0., 0.], "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "trials": rows}
    for kind in args.kinds:
        for level in ([0] if kind == "flat" else args.levels):
            for seed in args.seeds:
                cfg = evaluation_env_cfg(checkpoint, level, kind)
                cfg.scene.num_envs, cfg.seed = 1, seed
                cfg.auto_reset, cfg.terminations, cfg.curriculum = False, {}, {}
                cfg.rewards, cfg.metrics = {}, {}
                cfg.observations.pop("critic", None)
                cfg.episode_length_s = args.seconds + args.settle_seconds + 10
                term = cfg.commands["twist"]
                term.heading_command = False
                term.rel_heading_envs = term.rel_standing_envs = term.rel_forward_envs = 0.
                term.resampling_time_range = (1e9, 1e9)
                term.ranges.lin_vel_x = (args.vx, args.vx)
                term.ranges.lin_vel_y = (0., 0.)
                term.ranges.ang_vel_z = (0., 0.)
                term.debug_vis = False
                cfg.viewer.width, cfg.viewer.height = 800, 600
                cfg.viewer.enable_shadows = cfg.viewer.enable_reflections = False
                cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 2.5, -20., 140.
                with ExitStack() as stack:
                    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array" if args.video else None)
                    stack.callback(env.close)
                    bridge = Go2WBridge(env)
                    if checkpoint["env_action_order"] != bridge.to_env or abs(checkpoint["dt"] - env.step_dt) > 1e-8:
                        raise ValueError("模型动作顺序或控制周期与原环境不一致")
                    obs, _ = env.reset()
                    origins = env.scene.env_origins.clone()
                    steps = int((args.seconds + args.settle_seconds) / env.step_dt)
                    if args.max_steps:
                        steps = min(steps, args.max_steps)
                    settle = min(int(args.settle_seconds / env.step_dt), steps // 4)
                    writer = None
                    video = output / f"{kind}_level{level}_seed{seed}.mp4"
                    if args.video:
                        import mediapy
                        writer = stack.enter_context(mediapy.VideoWriter(str(video), shape=(600, 800), fps=25))
                    velocities = []
                    reached = False
                    failed = False
                    nonwheel_hit = False
                    stable_time = max_progress = 0.
                    expected = (1 if kind == "up" else -1 if kind == "down" else 0) * 5 * (.02, .04, .08, .12, .16, .20)[level]
                    with torch.inference_mode():
                        for i in range(steps):
                            obs, _, _, _, _ = env.step(bridge.env_action(actor(obs["actor"])))
                            d = bridge.robot.data
                            velocity = torch.cat((d.root_link_lin_vel_b[:, :2], d.root_link_ang_vel_b[:, 2:3]), -1)[0]
                            finite = bool(torch.isfinite(obs["actor"]).all() and torch.isfinite(velocity).all())
                            tilt = float(d.projected_gravity_b[0, :2].norm())
                            failed |= not finite or float(d.projected_gravity_b[0, 2]) > -.5
                            if checkpoint["config"].get("terrain_revision", 1) >= 7:
                                from go2w_supermimic.core.collision import illegal_nonwheel_contact
                                nonwheel_hit |= bool(illegal_nonwheel_contact(env)[0])
                                failed |= nonwheel_hit
                            offset = (d.root_link_pos_w - origins)[0]
                            progress = float(offset[0]) * (1 if args.vx > 0 else -1)
                            max_progress = max(max_progress, progress)
                            if kind in ("up", "down"):
                                contact = bool((bridge.contacts()[0, :4] > .5).sum() >= 2)
                                frame_pass = (progress > 2.8 and abs(float(offset[1])) < 2.5 and tilt < .4
                                              and expected + .20 < float(offset[2]) < expected + .60 and contact
                                              and abs(float(d.root_link_lin_vel_b[0, 2])) < .35 and not failed)
                                stable_time = stable_time + env.step_dt if frame_pass else 0.
                                reached |= stable_time >= .12 - 1e-6
                            if i >= settle:
                                velocities.append(velocity.cpu())
                            if writer is not None and i % 2 == 0:
                                writer.add_image(env.render())
                            if failed:
                                break
                    measured = torch.stack(velocities) if velocities else torch.zeros(1, 3)
                    command = measured.new_tensor((args.vx, 0., 0.))
                    row = {"terrain": kind, "level": level, "step_height": (.02, .04, .08, .12, .16, .20)[level],
                           "seed": seed, "samples": len(velocities), "failed": failed,
                           "nonwheel_collision": nonwheel_hit,
                           "actual_mean": measured.mean(0).tolist(), "velocity_mae": (measured - command).abs().mean(0).tolist(),
                           "max_forward_progress": max_progress, "reached_stairs": reached,
                           "video": str(video) if args.video else None}
                    rows.append(row)
                    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                    print("[Go2W测试] " + json.dumps(row, ensure_ascii=False), flush=True)
    print(f"[完成] Go2W效果数据：{output}", flush=True)


if __name__ == "__main__":
    main()
