"""崎岖地形几何量：使用机身正下方射线而非地块出生原点。"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


def terrain_height_under_root(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回机器人根部正下方的地面高度；平地任务回退到环境原点高度。"""
    fallback = env.scene.env_origins[:, 2]
    sensor = env.scene.sensors.get("height_scanner")
    if sensor is None:
        return fallback
    hit_z = sensor.data.ray_hits_w[:, 0, 2]
    return torch.where(torch.isfinite(hit_z), hit_z, fallback)


def root_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """机身根部相对本地支撑地面的高度。"""
    return env.scene["robot"].data.root_pos_w[:, 2] - terrain_height_under_root(env)
