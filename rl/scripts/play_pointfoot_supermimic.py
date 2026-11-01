#!/usr/bin/env python3
"""GUI 或 headless 回放点足版 SuperMimic checkpoint。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Isaac-Velocity-Rough-RPWD-Pointfoot-SuperMimic-WM-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps", type=int, default=0, help="0表示持续运行到关闭窗口")
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0], *unknown]
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import rp_wd_rl.tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from rp_wd_rl.supermimic.pointfoot import FORMAT, PointfootActor  # noqa: E402
from sim_shutdown import shutdown  # noqa: E402


def main() -> None:
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=args.device or "cuda:0", weights_only=True)
    if checkpoint.get("format") != FORMAT:
        raise ValueError("只支持点足版 SuperMimic checkpoint")
    device = torch.device(args.device or "cuda:0")
    actor = PointfootActor(tuple(checkpoint["actor_hidden"])).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    cfg = parse_env_cfg(args.task, device=str(device), num_envs=args.num_envs)
    cfg.seed = 42
    render_mode = None if args.headless else "human"
    env = gym.make(args.task, cfg=cfg, render_mode=render_mode)
    try:
        obs, _ = env.reset(seed=42)
        step = 0
        print(f"[回放] checkpoint={checkpoint_path} GUI={not args.headless}", flush=True)
        while simulation_app.is_running() and (args.max_steps <= 0 or step < args.max_steps):
            with torch.inference_mode():
                action = actor(obs["policy"].reshape(env.unwrapped.num_envs, -1))
                obs, _, _, _, _ = env.step(action)
            step += 1
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown(simulation_app)
