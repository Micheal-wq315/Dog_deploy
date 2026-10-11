"""RP_wd 域随机化事件: 传感器零偏 / 标定误差.

部署端 wheel_quad 推理节点**不做** IMU 零偏标定 (obs_manager.cpp 直接使用陀螺原始读数),
电机编码器零点也只有人手标定的 motor_zero_offset. 因此实机观测里必然存在:
  - 陀螺零偏 (缓慢漂移, 每台上电不同)
  - 关节位置零点误差 (编码器标定残差)

这类误差是**整回合常值**而非白噪声, 光靠观测噪声 (Unoise) 覆盖不到, 所以单独建模:
在 reset 时按环境重采样, 存在 env 属性上, 由 mdp/observations.py 里的观测项叠加.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv

# env 上的属性名 (observations.py 按同样的名字取)
GYRO_BIAS_ATTR = "_rp_wd_gyro_bias"
JOINT_OFFSET_ATTR = "_rp_wd_joint_offset"


def randomize_sensor_bias(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    gyro_bias_range: tuple[float, float] = (-0.05, 0.05),
    joint_pos_offset_range: tuple[float, float] = (-0.015, 0.015),
) -> None:
    """每回合重采样 IMU 陀螺零偏与关节位置零点误差.

    Args:
        env: 环境实例.
        env_ids: 需要重采样的环境索引 (reset 事件自动传入).
        gyro_bias_range: 陀螺零偏范围 [rad/s], 三轴独立均匀采样.
        joint_pos_offset_range: 关节位置零点误差范围 [rad], 每关节独立均匀采样.
    """
    if not hasattr(env, GYRO_BIAS_ATTR):
        setattr(env, GYRO_BIAS_ATTR, torch.zeros(env.num_envs, 3, device=env.device))
    num_joints = env.scene["robot"].num_joints
    if not hasattr(env, JOINT_OFFSET_ATTR):
        setattr(env, JOINT_OFFSET_ATTR, torch.zeros(env.num_envs, num_joints, device=env.device))

    num_reset = len(env_ids)
    if num_reset == 0:
        return

    def _uniform(shape: tuple[int, ...], value_range: tuple[float, float]) -> torch.Tensor:
        low, high = value_range
        return torch.rand(shape, device=env.device) * (high - low) + low

    getattr(env, GYRO_BIAS_ATTR)[env_ids] = _uniform((num_reset, 3), gyro_bias_range)
    getattr(env, JOINT_OFFSET_ATTR)[env_ids] = _uniform((num_reset, num_joints), joint_pos_offset_range)
