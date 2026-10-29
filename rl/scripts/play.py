"""加载训练好的 RP_wd 策略: 仿真回放 + 导出 ONNX (部署用).

用法:
    # 回放 + 导出 (需要显示器; 无显示器时用 --headless + --video)
    /media/rmer/data/IsaacLab/isaaclab.sh -p rl/scripts/play.py --num_envs 16

    # 指定 checkpoint
    /media/rmer/data/IsaacLab/isaaclab.sh -p rl/scripts/play.py --checkpoint rl/logs/rsl_rl/rp_wd_flat/2026-.../model_2999.pt

导出结果: 普通策略为 policy_flat.onnx; OdinDepth 任务导出 policy_odin_depth.onnx.
导出时从当前任务观测推导输入维度, 并校验输出 16 维与 ONNX Runtime 推理.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# 输出重定向到文件时也实时刷新
sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description="Play RP_wd policy and export ONNX.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-RPWD-Play-v0", help="任务名 (Play 变体).")
parser.add_argument("--num_envs", type=int, default=None, help="并行环境数.")
parser.add_argument("--checkpoint", type=str, default=None, help="模型文件 (.pt); 缺省自动找最新.")
parser.add_argument("--experiment_name", type=str, default=None, help="实验名 (定位日志目录).")
parser.add_argument("--load_run", type=str, default=None, help="run 目录 (正则, 默认最新).")
parser.add_argument("--seed", type=int, default=None, help="随机种子.")
parser.add_argument("--video", action="store_true", default=False, help="录制视频.")
parser.add_argument("--video_length", type=int, default=200, help="视频长度 (步).")
parser.add_argument("--real-time", action="store_true", default=False, help="按真实时间步进 (50Hz).")
parser.add_argument("--max_steps", type=int, default=0, help="回放步数上限 (0 = 一直跑, 便于脚本自检).")
parser.add_argument("--no-export", action="store_true", default=False, help="跳过 ONNX 导出.")
parser.add_argument("--export_dir", type=str, default=None, help="导出目录 (默认 rl/exported).")
parser.add_argument("--depth-memory-stride", type=int, default=None, help="v16为3，旧1740维模型回放用1。")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video or "OdinDepth" in args_cli.task:
    args_cli.enable_cameras = True

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

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import rp_wd_rl.tasks  # noqa: E402, F401
from onnx_utils import export_and_verify  # noqa: E402
from sim_shutdown import shutdown  # noqa: E402  (同目录下的关闭兜底)

_RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_ROOT = os.path.join(_RL_ROOT, "logs", "rsl_rl")
DEFAULT_EXPORT_DIR = os.path.join(_RL_ROOT, "exported")

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    """回放 + 导出."""
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.depth_memory_stride is not None:
        if "OdinDepth" not in args_cli.task or args_cli.depth_memory_stride < 1:
            raise ValueError("--depth-memory-stride仅用于深度任务且必须大于0。")
        env_cfg.observations.depth_memory.depth_memory.params["memory_stride"] = args_cli.depth_memory_stride
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join(LOG_ROOT, agent_cfg.experiment_name)
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        print(f"[INFO] 在 {log_root_path} 中查找最新 checkpoint")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] 加载模型: {resume_path}")
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    from policy_checkpoint import load_inference_checkpoint
    load_inference_checkpoint(runner, resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- 导出 ONNX (部署端用) ----
    if not args_cli.no_export:
        export_dir = os.path.abspath(args_cli.export_dir) if args_cli.export_dir else DEFAULT_EXPORT_DIR
        current_obs = env.get_observations()
        expected_dim = int(current_obs["policy"].shape[-1])
        output_name = "policy_odin_depth.onnx" if "OdinDepth" in args_cli.task else "policy_flat.onnx"
        dst = export_and_verify(
            runner, export_dir, expected_input_dim=expected_dim, output_filename=output_name
        )
        print(f"[INFO] 部署: cp {dst} src/inference/robots/wheel_quad/models/{output_name}")

    # ---- 回放 ----
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    print(f"[INFO] 开始回放 (策略频率 {1.0 / dt:.1f} Hz). Ctrl+C 退出.")
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
        timestep += 1
        if args_cli.video and timestep == args_cli.video_length:
            break
        if args_cli.max_steps and timestep >= args_cli.max_steps:
            print(f"[INFO] 回放 {timestep} 步后退出 (--max_steps).")
            break
        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    # 关闭仿真并退出 (close() 偶发挂起, 见 sim_shutdown.py)
    shutdown(simulation_app)
