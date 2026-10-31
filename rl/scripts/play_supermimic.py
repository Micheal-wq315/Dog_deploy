"""在Isaac Lab中回放SuperMimic蒸馏后的Odin深度学生。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

sys.stdout.reconfigure(line_buffering=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, help="student_final.pt")
parser.add_argument("--task", default="Isaac-Velocity-Rough-RPWD-OdinDepth-Play-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_steps", type=int, default=0, help="0表示窗口关闭前持续运行")
parser.add_argument("--log_interval", type=int, default=25)
parser.add_argument("--real_time", action=argparse.BooleanOptionalAction, default=True)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import rp_wd_rl.tasks  # noqa: E402,F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402
from rp_wd_rl.supermimic.model import DEPLOY_INPUT_DIM, student_from_checkpoint  # noqa: E402
from sim_shutdown import shutdown  # noqa: E402


def main() -> None:
    device = torch.device(args.device or "cuda:0")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    student = student_from_checkpoint(str(checkpoint_path), device)
    cfg = parse_env_cfg(args.task, device=str(device), num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg, render_mode="human")
    try:
        observations, _ = env.reset(seed=args.seed)
        policy = observations["policy"].reshape(args.num_envs, -1)
        depth = observations["depth_memory"].reshape(args.num_envs, -1)
        if policy.shape[-1] + depth.shape[-1] != DEPLOY_INPUT_DIM:
            raise RuntimeError("回放观测与1740维部署输入契约不匹配")
        target_period = cfg.sim.dt * cfg.decimation
        step = 0
        while simulation_app.is_running() and (args.max_steps == 0 or step < args.max_steps):
            start = time.perf_counter()
            with torch.no_grad():
                observation = torch.cat((policy, depth), dim=-1)
                action = student(observation)
                observations, reward, terminated, truncated, _ = env.step(action)
                policy = observations["policy"].reshape(args.num_envs, -1)
                depth = observations["depth_memory"].reshape(args.num_envs, -1)
            step += 1
            if step % args.log_interval == 0:
                dones = (terminated | truncated).float().mean().item()
                print(f"[回放] step={step} reward={reward.mean().item():.3f} "
                      f"done比例={dones:.3f}", flush=True)
            if args.real_time:
                remaining = target_period - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        shutdown(simulation_app)
