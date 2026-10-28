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
from isaaclab.sensors import TiledCamera

from .events import GYRO_BIAS_ATTR, JOINT_OFFSET_ATTR


def leg_torque_normalized(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=[".*_ABAD_JOINT", ".*_HIP_JOINT", ".*_KENN_JOINT"]
    ),
    effort_limit: float = 42.0,
) -> torch.Tensor:
    """腿部执行器力矩反馈 (Nm), 按额定 URDF 力矩归一化并限幅到 [-1, 1]."""
    asset = env.scene[asset_cfg.name]
    return torch.clamp(asset.data.applied_torque[:, asset_cfg.joint_ids] / effort_limit, -1.0, 1.0)


def odin_depth_grid(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("odin_depth_camera"),
    data_type: str = "distance_to_camera",
    max_depth: float = 6.0,
    grid_size: tuple[int, int] = (12, 16),  # (height, width)
    noise_std: float = 0.03,
    dropout_prob: float = 0.01,
    crop_fraction_range: tuple[float, float] = (0.80, 1.0),
    random_crop: bool = True,
    augmentation_warmup_steps: int = 12_000,
    augmentation_ramp_steps: int = 60_000,
) -> torch.Tensor:
    """Odin1 风格深度观测: 随机裁切、深度噪声/缺测，再做最近障碍保留型池化.

    返回 [num_envs, H*W] 的 0..1 距离网格。无回波编码为 max_depth；网格池化
    取每格最近有效点，避免平均池化把细障碍抹掉。部署端 odin_depth_adapter
    使用同样的分格最小值和 max_depth 归一化。传感器增强从无增强开始，按训练步数
    线性渐入目标噪声、缺测率和裁切范围，让策略先学几何避障再适应传感器误差。
    """
    import torch.nn.functional as F

    sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output[data_type]
    if depth.ndim == 4:
        depth = depth[..., 0]
    depth = depth.to(dtype=torch.float32).clone()
    valid = torch.isfinite(depth) & (depth > 0.0)
    depth = torch.where(valid, depth, torch.full_like(depth, max_depth))
    depth.clamp_(min=0.2, max=max_depth)

    step = int(env.common_step_counter)
    aug_progress = min(
        1.0,
        max(0.0, (step - augmentation_warmup_steps) / max(augmentation_ramp_steps, 1)),
    )
    crop_low = 1.0 - (1.0 - crop_fraction_range[0]) * aug_progress
    crop_high = 1.0 - (1.0 - crop_fraction_range[1]) * aug_progress

    if random_crop and crop_low < 1.0:
        count = depth.shape[0]
        crop_h = torch.empty(count, device=depth.device).uniform_(crop_low, crop_high)
        crop_w = torch.empty(count, device=depth.device).uniform_(crop_low, crop_high)
        shift_x = (torch.rand(count, device=depth.device) * 2.0 - 1.0) * (1.0 - crop_w)
        shift_y = (torch.rand(count, device=depth.device) * 2.0 - 1.0) * (1.0 - crop_h)
        theta = torch.zeros(count, 2, 3, device=depth.device, dtype=depth.dtype)
        theta[:, 0, 0] = crop_w
        theta[:, 1, 1] = crop_h
        theta[:, 0, 2] = shift_x
        theta[:, 1, 2] = shift_y
        grid = F.affine_grid(theta, (count, 1, *depth.shape[-2:]), align_corners=False)
        depth = F.grid_sample(
            depth.unsqueeze(1), grid, mode="bilinear", padding_mode="border", align_corners=False
        ).squeeze(1)

    effective_dropout = dropout_prob * aug_progress
    if effective_dropout > 0.0:
        missing = torch.rand_like(depth) < effective_dropout
        depth = torch.where(missing, torch.full_like(depth, max_depth), depth)
    depth.clamp_(min=0.2, max=max_depth)

    # Adaptive max-pooling on negative distance is min-pooling: preserve nearby obstacles.
    grid = -F.adaptive_max_pool2d(-depth.unsqueeze(1), output_size=grid_size)
    grid = grid.squeeze(1)
    # 在降采样后加噪，避免 min-pooling 把大量像素噪声的极小值放大成系统性近距偏差。
    effective_noise = noise_std * aug_progress
    if effective_noise > 0.0:
        grid = grid + torch.randn_like(grid) * effective_noise
    grid = (grid / max_depth).clamp_(0.0, 1.0)
    return grid.flatten(start_dim=1)


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
