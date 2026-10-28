"""RP_wd 专用奖励项 (轮足特有, 标准四足奖励里没有的)."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .terrain import terrain_height_under_root


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
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=[".*_FOOT_LINK"]),
    wheel_radius: float = 0.1025,
    max_mismatch: float = 0.5,
) -> torch.Tensor:
    """轮子滚动失配: mean_i clamp((ω_i·r − v_i)²), v_i 为轮心沿机体 x 轴的实际速度.

    纯滚动时 ω·r == v -> 0; 把轮子当脚"跑"(轮子基本不转) 时该项显著为正.

    为什么需要它: 只有跟踪奖励时, "四轮滚动前进" 与 "用腿蹬地、轮子当脚的小跑步态"
    都能拿到同样的跟踪奖励 —— 实测 (iter 1400) 策略收敛到了后者: 平均着地轮数 2.36/4,
    轮速只有应有值的 1/3. 小跑步态冲击大 (仿真的接触/冲击是最不准的部分, sim2real 风险最高),
    且完全浪费了轮子. 加入滚动失配惩罚后, 只有真正滚动才能低成本地拿到跟踪奖励.

    仅惩罚接触地面的轮子，空中抬轮不再被当成打滑。`max_mismatch` 用来**截断**失配
    (默认 0.5 m/s): 不然早期乱动时该项会大到形成悬崖,
    把探索直接压死.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    omega = robot.data.joint_vel[:, wheel_joint_cfg.joint_ids] * wheel_radius  # (N, 4)
    # 轮心的世界系速度 -> 投影到机体 x 轴
    v_w = robot.data.body_lin_vel_w[:, wheel_body_cfg.body_ids, :]  # (N, 4, 3)
    quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, v_w.shape[1], -1)
    v_forward = math_utils.quat_apply_inverse(quat, v_w)[..., 0]  # (N, 4)
    mismatch_sq = torch.square(omega - v_forward).clamp(max=max_mismatch * max_mismatch)
    contact = env.scene.sensors[contact_sensor_cfg.name]
    forces = contact.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=-1)
    on_ground = forces.amax(dim=1) > 1.0
    weights = on_ground.to(mismatch_sq.dtype)
    return (mismatch_sq * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


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


def _pitch_roll(asset: Articulation) -> tuple[torch.Tensor, torch.Tensor]:
    """由机体系投影重力反算 (pitch, roll), 与契约符号约定一致."""
    g = asset.data.projected_gravity_b
    pitch = torch.asin(torch.clamp(g[:, 0], -1.0, 1.0))   # 抬头为正
    roll = torch.atan2(g[:, 1], -g[:, 2])                 # 右侧下沉为正
    return pitch, roll


def attitude_track_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_posture",
    std: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身姿态 (pitch/roll) 跟踪奖励: exp(-(e_pitch² + e_roll²) / std²).

    用 exp 核而不是 L2: 姿态误差本来就小 (0.02~0.05 rad), L2 加权后量级远小于速度奖励,
    根本"突出"不出来. exp 核的满额权重与速度跟踪同量级, 才能真正把控制精度顶上去.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    pitch, roll = _pitch_roll(asset)
    err_sq = torch.square(pitch - command[:, 0]) + torch.square(roll - command[:, 1])
    return torch.exp(-err_sq / std**2)


def attitude_limit_l2(
    env: ManagerBasedRLEnv,
    limit: float = 0.1745,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """超出允许倾角的部分的 L2 惩罚 (默认 10°).

    跟踪奖励管"跟准指令", 这一项管"别超过 10°": 允许运动时自然摆动到 10° 左右,
    再多就罚 —— 对应需求"减少前倾和左右倾斜, 允许约 10° 的运动倾斜".
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pitch, roll = _pitch_roll(asset)
    over_p = torch.clamp(pitch.abs() - limit, min=0.0)
    over_r = torch.clamp(roll.abs() - limit, min=0.0)
    return torch.square(over_p) + torch.square(over_r)


def height_track_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_posture",
    std: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身相对脚下地面的高度跟踪奖励，避免地块世界z变化影响站姿。"""
    from rp_wd_rl.assets.rp_wd_contract import STAND_HEIGHT

    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target = STAND_HEIGHT + command[:, 2]
    height_above_ground = asset.data.root_pos_w[:, 2] - terrain_height_under_root(env)
    err_sq = torch.square(height_above_ground - target)
    return torch.exp(-err_sq / std**2)


def leg_joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"]
    ),
) -> torch.Tensor:
    """腿部关节 (不含轮) 角速度平方和 —— 抑制"抬腿换位"式运动.

    针对实测问题: 左右横移时策略会用对角腿交替抬起、错开重心来侧挪 (对角腿错开).
    这种步态在仿真里可行, 但 sim2real 很差 (冲击、支撑面小、轮子侧向刮擦).
    惩罚腿部关节速度, 等于逼它"用轮子滚, 而不是用腿挪".
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
