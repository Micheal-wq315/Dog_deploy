"""训练 RP_wd 策略 (Isaac Lab + RSL-RL PPO).

用法 (在 isaaclab conda 环境下):
    /media/rmer/data/IsaacLab/isaaclab.sh -p rl/scripts/train.py \
        --task Isaac-Velocity-Flat-RPWD-v0 --headless --num_envs 2048

日志写入 rl/logs/rsl_rl/<experiment_name>/<时间戳>_<run_name>/ (与 CWD 无关).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from isaaclab.app import AppLauncher

# 输出重定向到文件时也实时刷新 (便于 tail -f 看训练进度)
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# 命令行参数 (必须在启动 SimulationApp 之前解析)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train RP_wd policy with RSL-RL.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-RPWD-v0", help="注册的任务名.")
parser.add_argument("--num_envs", type=int, default=None, help="并行环境数 (默认取环境配置里的值).")
parser.add_argument("--max_iterations", type=int, default=None, help="训练迭代数.")
parser.add_argument("--seed", type=int, default=None, help="随机种子.")
parser.add_argument("--video", action="store_true", default=False, help="录制训练视频.")
parser.add_argument("--video_length", type=int, default=200, help="视频长度 (步).")
parser.add_argument("--video_interval", type=int, default=2000, help="录制间隔 (步).")
parser.add_argument("--experiment_name", type=str, default=None, help="实验名 (日志目录/续训定位).")
parser.add_argument("--run_name", type=str, default=None, help="本次运行的别名.")
parser.add_argument("--resume", action="store_true", default=False, help="从 checkpoint 续训.")
parser.add_argument("--load_run", type=str, default=None, help="要加载的 run 目录 (正则, 默认最新).")
parser.add_argument("--checkpoint", type=str, default=None, help="要加载的模型文件名 (正则, 默认 model_.*.pt).")
parser.add_argument("--logger", type=str, default=None, choices=["tensorboard", "wandb", "neptune"])
parser.add_argument("--log_project_name", type=str, default=None, help="wandb/neptune 项目名.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video or "OdinDepth" in args_cli.task:
    args_cli.enable_cameras = True

# 训练一律无头运行: 强制关掉 Isaac Sim 图形窗口.
# 不管从哪里启动 (./rl/scripts/train.sh / python rl/scripts/train.py), 都不会弹窗,
# 也就不需要显示器、不额外占显存. 录制训练视频用的是离屏渲染, 与 headless 不冲突.
args_cli.headless = True

# 清空 sys.argv 供 Hydra 使用
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Isaac Sim (必须先于任何 isaaclab.* 仿真模块导入)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 以下导入都必须在 SimulationApp 启动之后
# ---------------------------------------------------------------------------
import importlib.metadata as metadata  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import rp_wd_rl.tasks  # noqa: E402, F401  (导入即注册 gym 任务)
from sim_shutdown import shutdown  # noqa: E402  (同目录下的关闭兜底)

# 训练日志固定放在仓库 rl/logs 下
_RL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_ROOT = os.path.join(_RL_ROOT, "logs", "rsl_rl")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

RSL_RL_VERSION = metadata.version("rsl-rl-lib")



def resolve_resume_path(log_root_path: str, load_run: str, checkpoint: str) -> str:
    """定位要加载的 checkpoint.

    支持两种写法:
      - `--load_run` 是**实验目录内的 run 名** (正则): 走 Isaac Lab 的 get_checkpoint_path
      - `--load_run` 是**绝对路径**: 直接在该目录里匹配 checkpoint
        (用于跨实验热启动, 例如从平地的 run 热启动到崎岖地形任务)
    """
    if os.path.isabs(load_run):
        cands = sorted(f for f in os.listdir(load_run) if re.match(checkpoint, f))
        if not cands:
            raise FileNotFoundError(f"{load_run} 中没有匹配 {checkpoint} 的 checkpoint")
        return os.path.join(load_run, cands[-1])
    return get_checkpoint_path(log_root_path, load_run, checkpoint)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    """训练主流程."""
    # ---- 命令行覆盖配置 ----
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if args_cli.log_project_name is not None:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    if args_cli.resume:
        agent_cfg.resume = True
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint

    # rsl-rl 5.x: 旧式 policy= 配置会被迁移到 actor/critic
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, RSL_RL_VERSION)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ---- 日志目录 ----
    log_root_path = os.path.join(LOG_ROOT, agent_cfg.experiment_name)
    print(f"[INFO] 实验日志目录: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    # ---- 环境 ----
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 续训: 必须在创建新 log_dir 之前定位 checkpoint
    if agent_cfg.resume:
        # 实验目录可能还不存在 (新实验名 + 热启动), get_checkpoint_path 会 listdir 它
        os.makedirs(log_root_path, exist_ok=True)
        resume_path = resolve_resume_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 录制训练视频.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # clip_actions 必须与部署端 default.yaml 的 clip_actions 一致 (对原始动作裁剪)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    start_time = time.time()

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume:
        print(f"[INFO] 加载模型: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    # 关闭仿真并退出 (close() 偶发挂起, 见 sim_shutdown.py)
    shutdown(simulation_app)
