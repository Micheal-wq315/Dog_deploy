"""课程学习项 (随训练进度调整 MDP 参数)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from rp_wd_rl.assets.rp_wd_contract import (
    CMD_HEIGHT_RANGE,
    CMD_PITCH_RANGE,
    CMD_ROLL_RANGE,
    CMD_VX_RANGE,
    CMD_VY_RANGE,
    CMD_WZ_RANGE,
)


def command_lin_vel_curriculum(
    env: ManagerBasedRLEnv,
    env_ids,
    command_name: str = "base_velocity",
    start_scale: float = 0.35,
    end_scale: float = 1.0,
    warmup_steps: int = 240,
    ramp_steps: int = 24000,
    ramp_angular: bool = False,
) -> float:
    """把**直线**速度指令范围按训练进度从 ``start_scale`` 线性放大到 ``end_scale`` 倍.

    为什么要它: 直接把指令范围从 ±0.6 跳到 ±2.0 m/s 时, "宽指令分布 + 宽跟踪核" 会让
    "慢慢走"成为局部最优 —— 实测策略卡在跟踪误差 ~1.9 m/s 上平了 400+ 迭代.
    课程从策略**当前能力附近**起步, 学会了再放大, 是扩展指令范围的标准做法
    (最终仍要覆盖完整的契约范围, 否则实机手柄一推就出分布).

    默认**只放直线范围**, 角速度保持契约满额 —— 实测偏航跟踪一直很好(94% 上限),
    没必要把它一起降下来.

    Args:
        env_ids: 由管理器传入 (本项对所有环境生效, 不使用).
        start_scale: 起始比例 (0.35 → vx ±0.7, vy ±0.7).
        end_scale: 结束比例 (1.0 → 契约满额).
        warmup_steps: 起始比例保持多久 [策略步].
        ramp_steps: 从起始比例线性放大到结束比例所需的策略步数.
        ramp_angular: 是否也放大角速度范围 (默认否).

    Note:
        单位是**策略步** (= ``env.common_step_counter``, 每调用一次 ``env.step`` 计 1).
        默认配置下 1 迭代 = ``num_steps_per_env`` = 24 步, 所以
        ``warmup_steps=240`` ≈ 10 迭代, ``ramp_steps=24000`` ≈ 1000 迭代.

    Returns:
        当前比例 (会被 Isaac Lab 记到 ``Curriculum/<该项名字>`` 曲线上, 方便监控进度).
    """
    counter = env.common_step_counter
    if counter <= warmup_steps:
        scale = start_scale
    else:
        progress = min(1.0, (counter - warmup_steps) / max(ramp_steps, 1))
        scale = start_scale + (end_scale - start_scale) * progress

    term = env.command_manager.get_term(command_name)
    ranges = term.cfg.ranges
    ranges.lin_vel_x = (CMD_VX_RANGE[0] * scale, CMD_VX_RANGE[1] * scale)
    ranges.lin_vel_y = (CMD_VY_RANGE[0] * scale, CMD_VY_RANGE[1] * scale)
    if ramp_angular:
        ranges.ang_vel_z = (CMD_WZ_RANGE[0] * scale, CMD_WZ_RANGE[1] * scale)
    else:
        ranges.ang_vel_z = CMD_WZ_RANGE
    return scale


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """按机器人走出的距离升降**地形难度等级** (崎岖地形课程).

    走够半个地块宽度就升级, 走得不到指令距离的一半就降级.

    源码取自 Isaac Lab 2.3.2 的 velocity 任务
    (`isaaclab_tasks/manager_based/locomotion/velocity/mdp/curriculums.py:27-56`),
    复制到本仓库以便独立演进.

    .. note::
        只对 ``terrain_type="generator"`` 且 ``use_terrain_origins=True`` 有效
        (否则 ``terrain_levels`` 属性不存在, 会直接 AttributeError).

    Returns:
        当前平均地形等级 (会记到 ``Curriculum/terrain_levels`` 曲线上).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # 走出多远
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # 走够半个地块 -> 升级
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # 走不到指令距离的一半 -> 降级
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())

def posture_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids,
    command_name: str = "base_posture",
    start_scale: float = 0.0,
    end_scale: float = 1.0,
    warmup_steps: int = 4800,
    ramp_steps: int = 24000,
) -> float:
    """把姿态/高度指令幅度从 ``start_scale`` 逐步放大到 ``end_scale``.

    为什么要它: 一开始就让 ±10° 姿态 + ±7 cm 高度和速度同时学, 早期容易又摔又跟不准.
    先只学"站住 + 跑起来"(姿态指令=0), 等基础能力稳了再逐步加入姿态/高度目标 ——
    与速度课程同样的思路, 只是作用在另一个指令项上.

    单位同 `command_lin_vel_curriculum`: 策略步 (1 迭代 = num_steps_per_env = 24 步).
    默认 warmup 4800 步 ≈ 200 迭代, ramp 24000 步 ≈ 1000 迭代.
    """
    counter = env.common_step_counter
    if counter <= warmup_steps:
        scale = start_scale
    else:
        progress = min(1.0, (counter - warmup_steps) / max(ramp_steps, 1))
        scale = start_scale + (end_scale - start_scale) * progress

    term = env.command_manager.get_term(command_name)
    r = term.cfg.ranges
    r.pitch = (CMD_PITCH_RANGE[0] * scale, CMD_PITCH_RANGE[1] * scale)
    r.roll = (CMD_ROLL_RANGE[0] * scale, CMD_ROLL_RANGE[1] * scale)
    r.height = (CMD_HEIGHT_RANGE[0] * scale, CMD_HEIGHT_RANGE[1] * scale)
    return scale
