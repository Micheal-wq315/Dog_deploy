"""终止条件 (本任务自定义的项)."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def root_height_below_local_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机体高度低于**本环境地形原点之上**的阈值时终止.

    ⚠️ 崎岖地形上**不能用** `isaaclab.envs.mdp.root_height_below_minimum`
    (它判世界绝对高度): 生成地形的每块地块平台高度各不相同 (实测 36 块从 -0.36 到 +0.40 m),
    落在低处的环境会一出生就被判死。实测后果: Play 测试里 400 步终止 402 次
    (回合长度中位数 1 步, 终止时倾角 0° —— 根本没摔, 只是"在坑里"),
    训练日志里 `Episode_Termination/root_height` 占 14-16%, 相当于白白浪费约 1/3 的样本
    并扭曲地形课程。

    本函数把判据改成"相对本环境地形原点的高度", 与"机器人相对自己所处地形的高度"一致。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    height_above_origin = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return height_above_origin < minimum_height
