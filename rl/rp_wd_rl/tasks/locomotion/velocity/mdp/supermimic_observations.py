"""SuperMimic 教师专用的仿真特权观测。"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .terrain import terrain_height_under_root


def supermimic_terrain_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    grid_size: tuple[int, int] = (16, 12),
    height_clip: float = 0.5,
) -> torch.Tensor:
    """返回以机身高度为零点的真值高度图，形状 [N, 1, 16, 12]。"""
    sensor = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w
    root_z = env.scene["robot"].data.root_pos_w[:, 2]
    height = hits[..., 2] - root_z[:, None]
    height = torch.nan_to_num(height, nan=height_clip, posinf=height_clip, neginf=-height_clip)
    height = (height.clamp(-height_clip, height_clip) / height_clip).view(env.num_envs, *grid_size)
    return height.unsqueeze(1)


def supermimic_privileged_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_limit: float = 500.0,
) -> torch.Tensor:
    """真实机身速度、脚下净空及四轮法向力，输出11维并限制到稳定范围。"""
    robot = env.scene["robot"]
    velocity = torch.cat((robot.data.root_lin_vel_b, robot.data.root_ang_vel_b), dim=-1)
    clearance = robot.data.root_pos_w[:, 2] - terrain_height_under_root(env)
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    wheel_normal = forces[..., 2].clamp_min(0.0).amax(dim=1) / force_limit
    wheel_normal = wheel_normal.clamp(0.0, 1.0)
    return torch.cat((velocity, clearance.unsqueeze(-1), wheel_normal), dim=-1)
