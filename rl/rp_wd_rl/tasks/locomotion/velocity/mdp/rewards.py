"""RP_wd 专用奖励项 (轮足特有, 标准四足奖励里没有的)."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def wheel_vel_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """轮子转速平方和 — 抑制空转/打滑 (权重需很小, 否则会压制正常行进)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def stand_still_wheel_vel(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
    max_wheel_speed: float = 2.0,
) -> torch.Tensor:
    """零指令 (站立) 时惩罚轮子转动 — 实机上表现为原地溜车/抖轮.

    只在速度指令接近 0 时生效; 有指令时轮子转是正常的, 不做惩罚.

    ⚠️ 这里必须**有界**: 早期版本用 Σω²(rad/s)² 且无上限, 轮子转到 5 rad/s 就是每轮 25、
    四轮 100, 乘以权重后单回合扣分达跟踪奖励的 7 倍, 结果策略学会"早点摔倒"来少扣分
    (自杀式局部最优). 现在改成四轮转速的 L2 范数并截断到 max_wheel_speed,
    单步惩罚上界 = weight × max_wheel_speed.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # 线速度与偏航角速度都要接近 0 才算"站立"
    is_standing = torch.linalg.norm(command, dim=1) < command_threshold
    wheel_speed = torch.linalg.vector_norm(asset.data.joint_vel[:, asset_cfg.joint_ids], dim=1)
    return torch.clamp(wheel_speed, max=max_wheel_speed) * is_standing.float()


def wheel_rolling_slip_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_joint_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_FOOT_JOINT"]),
    wheel_body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_FOOT_LINK"]),
    wheel_radius: float = 0.1025,
    max_mismatch: float = 0.5,
) -> torch.Tensor:
    """轮子滚动失配: mean_i clamp((ω_i·r − v_i)²), v_i 为轮心沿机体 x 轴的实际速度.

    纯滚动时 ω·r == v -> 0; 把轮子当脚"跑"(轮子基本不转) 时该项显著为正.

    为什么需要它: 只有跟踪奖励时, "四轮滚动前进" 与 "用腿蹬地、轮子当脚的小跑步态"
    都能拿到同样的跟踪奖励 —— 实测 (iter 1400) 策略收敛到了后者: 平均着地轮数 2.36/4,
    轮速只有应有值的 1/3. 小跑步态冲击大 (仿真的接触/冲击是最不准的部分, sim2real 风险最高),
    且完全浪费了轮子. 加入滚动失配惩罚后, 只有真正滚动才能低成本地拿到跟踪奖励.

    `max_mismatch` 用来**截断**失配 (默认 0.5 m/s): 不然早期乱动时该项会大到形成悬崖,
    把探索直接压死.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    omega = robot.data.joint_vel[:, wheel_joint_cfg.joint_ids] * wheel_radius  # (N, 4)
    # 轮心的世界系速度 -> 投影到机体 x 轴
    v_w = robot.data.body_lin_vel_w[:, wheel_body_cfg.body_ids, :]  # (N, 4, 3)
    quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, v_w.shape[1], -1)
    v_forward = math_utils.quat_apply_inverse(quat, v_w)[..., 0]  # (N, 4)
    mismatch_sq = torch.square(omega - v_forward).clamp(max=max_mismatch * max_mismatch)
    return mismatch_sq.mean(dim=1)


def wheels_in_contact_frac(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*_FOOT_LINK"]),
    threshold: float = 1.0,
) -> torch.Tensor:
    """着地轮子的比例 (0..1) —— 四轮滚动前进时应当恒为 1.

    ⚠️ 不要用 isaaclab.envs.mdp.desired_contacts: 它在 2.3.2 里的语义是
    "**所有**列出的刚体都**没**接触时返回 1", 当作正向奖励用会变成"奖励四轮离地".
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1)
    in_contact = forces.max(dim=1)[0] > threshold  # (N, num_bodies)
    return in_contact.float().mean(dim=1)


def wheel_torque_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """轮电机力矩平方和 — RS02 额定 6 N·m / 峰值 17 N·m, 抑制长时间大扭矩."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
