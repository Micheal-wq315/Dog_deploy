"""RP_wd 观测项: 带常值零偏的版本 (对应实机传感器).

与 isaaclab.envs.mdp 的同名项相比, 这里叠加了 mdp/events.py 在 reset 时采样的
整回合常值零偏. 观测的 *维度与顺序* 与核心版本完全一致, 因此不影响 sim2real 契约.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp import base_ang_vel as _base_ang_vel
from isaaclab.envs.mdp import joint_pos_rel as _joint_pos_rel
from isaaclab.managers import SceneEntityCfg

from .events import GYRO_BIAS_ATTR, JOINT_OFFSET_ATTR


def base_ang_vel_biased(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """机体角速度 + 陀螺零偏 (rad/s, 机体系)."""
    obs = _base_ang_vel(env, asset_cfg)
    bias = getattr(env, GYRO_BIAS_ATTR, None)
    return obs if bias is None else obs + bias


def joint_pos_rel_biased(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """关节位置 (相对默认姿态) + 编码器零点误差 (rad)."""
    obs = _joint_pos_rel(env, asset_cfg)
    offset = getattr(env, JOINT_OFFSET_ATTR, None)
    if offset is None:
        return obs
    # 零偏按全长存储, 这里按该观测项实际使用的关节子集索引
    return obs + offset[:, asset_cfg.joint_ids]
