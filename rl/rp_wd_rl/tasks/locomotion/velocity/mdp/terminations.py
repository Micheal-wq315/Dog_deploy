"""终止条件 (本任务自定义的项)."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from .terrain import terrain_height_under_root


def root_height_below_local_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机体低于脚下地面的净空阈值时终止；平地回退到环境原点。"""
    height_above_ground = env.scene[asset_cfg.name].data.root_pos_w[:, 2] - terrain_height_under_root(env)
    return height_above_ground < minimum_height
