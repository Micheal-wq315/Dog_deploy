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

from rp_wd_rl.assets.rp_wd_contract import (
    ODIN_DEPTH_MEMORY_FRAMES,
    ODIN_DEPTH_MEMORY_STRIDE,
    ODIN_DEPTH_ROTATION_K,
)

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
    data_type: str = "distance_to_image_plane",
    max_depth: float = 6.0,
    grid_size: tuple[int, int] = (16, 12),  # (height, width), portrait after rotation
    rotation_k: int = ODIN_DEPTH_ROTATION_K,
    noise_std: float = 0.03,
    dropout_prob: float = 0.01,
    crop_fraction_range: tuple[float, float] = (0.80, 1.0),
    random_crop: bool = True,
    augmentation_warmup_steps: int = 12_000,
    augmentation_ramp_steps: int = 48_000,
    crop_transform: torch.Tensor | None = None,
) -> torch.Tensor:
    """Odin1 风格深度观测: 随机裁切、深度噪声/缺测，再做最近障碍保留型池化.

    返回 [num_envs, H, W] 的 0..1 距离网格。先按传感器校正角度旋转，再裁切和池化；
    无回波编码为 max_depth。网格池化取每格最近有效点，避免平均池化把细障碍抹掉。
    部署端 odin_depth_adapter
    使用同样的分格最小值和 max_depth 归一化。传感器增强从无增强开始，按训练步数
    线性渐入目标噪声、缺测率和裁切范围，让策略先学几何避障再适应传感器误差。
    """
    import torch.nn.functional as F

    sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output[data_type]
    if depth.ndim == 4:
        depth = depth[..., 0]
    depth = depth.to(dtype=torch.float32).clone()
    if rotation_k % 4:
        depth = torch.rot90(depth, k=rotation_k % 4, dims=(-2, -1))
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
        if crop_transform is not None:
            # 每回合固定几何偏差，避免逐帧随机缩放破坏障碍的时间连续性。
            identity = torch.zeros_like(theta)
            identity[:, 0, 0] = identity[:, 1, 1] = 1.0
            theta = identity + (crop_transform - identity) * aug_progress
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
    return grid


def odin_depth_memory(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("odin_depth_camera"),
    data_type: str = "distance_to_image_plane",
    max_depth: float = 6.0,
    grid_size: tuple[int, int] = (16, 12),
    noise_std: float = 0.03,
    dropout_prob: float = 0.01,
    crop_fraction_range: tuple[float, float] = (0.80, 1.0),
    random_crop: bool = True,
    augmentation_warmup_steps: int = 12_000,
    augmentation_ramp_steps: int = 48_000,
    memory_frames: int = ODIN_DEPTH_MEMORY_FRAMES,
    rotation_k: int = ODIN_DEPTH_ROTATION_K,
    memory_key: str = "policy",
    episode_bias_m: float = 0.0,
    scale_error: float = 0.0,
    occlusion_prob: float = 0.0,
    packet_loss_prob: float = 0.0,
    max_latency_frames: int = 0,
    memory_stride: int = ODIN_DEPTH_MEMORY_STRIDE,
) -> torch.Tensor:
    """输出按相机新帧更新的时间记忆，形状为 [N, frames, H, W].

    默认保留22帧并间隔3帧抽8张，约覆盖1.4秒；以camera.frame检测新图，只在新图到达时推进
    历史窗口。每回合首帧用当前图复制填满，和部署端适配器初始化规则一致。
    """
    if memory_frames < 1 or memory_stride < 1:
        raise ValueError("memory_frames和memory_stride必须大于0")
    sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]
    # SensorBase按需更新；先取data触发到期渲染，再检查frame，避免缓存阻止新帧生成。
    _ = sensor.data
    frame_id = sensor.frame.to(device=env.device, dtype=torch.long)
    attr = f"_rpwd_odin_memory_{memory_key}"
    state = getattr(env, attr, None)
    history_frames = (memory_frames - 1) * memory_stride + 1
    shape = (env.num_envs, history_frames, *grid_size)
    initialized = state is None or state["memory"].shape != shape
    if initialized:
        state = {
            "memory": torch.ones(shape, device=env.device),
            "queue": torch.ones(env.num_envs, max_latency_frames + 1, *grid_size, device=env.device),
            "frame": torch.full_like(frame_id, -1),
            "reset_step": torch.full_like(frame_id, -1),
            "profile": torch.zeros(env.num_envs, device=env.device),
            "bias": torch.zeros(env.num_envs, device=env.device),
            "scale": torch.ones(env.num_envs, device=env.device),
            "latency": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "crop": torch.zeros(env.num_envs, 2, 3, device=env.device),
        }
        state["crop"][:, 0, 0] = state["crop"][:, 1, 1] = 1.0
        setattr(env, attr, state)
    step = int(env.common_step_counter)
    reset_mask = ((env.episode_length_buf == 0) & (state["reset_step"] != step)) | initialized
    new_frame = frame_id != state["frame"]
    if not bool((new_frame | reset_mask).any()):
        return state["memory"][:, ::memory_stride]
    ids = reset_mask.nonzero(as_tuple=False).flatten()
    if len(ids):
        r = torch.rand(len(ids), device=env.device)
        # 60%正常、30%偏差、10%严重退化；这是待用真实 Odin 数据校准的初始配比。
        state["profile"][ids] = torch.where(r < 0.6, 0.0, torch.where(r < 0.9, 1.0, 2.0))
        state["bias"][ids] = torch.empty(len(ids), device=env.device).uniform_(-episode_bias_m, episode_bias_m)
        state["scale"][ids] = 1.0 + torch.empty(len(ids), device=env.device).uniform_(-scale_error, scale_error)
        state["latency"][ids] = torch.randint(max_latency_frames + 1, (len(ids),), device=env.device)
        crop = torch.empty(len(ids), 2, device=env.device).uniform_(*crop_fraction_range)
        state["crop"][ids, 0, 0] = crop[:, 1]
        state["crop"][ids, 1, 1] = crop[:, 0]
        shift = (torch.rand(len(ids), 2, device=env.device) * 2 - 1) * (1 - crop.flip(1))
        state["crop"][ids, :, 2] = shift
        state["reset_step"][ids] = step
    current = odin_depth_grid(
        env,
        sensor_cfg=sensor_cfg,
        data_type=data_type,
        max_depth=max_depth,
        grid_size=grid_size,
        rotation_k=rotation_k,
        noise_std=noise_std,
        dropout_prob=dropout_prob,
        crop_fraction_range=crop_fraction_range,
        random_crop=random_crop,
        augmentation_warmup_steps=augmentation_warmup_steps,
        augmentation_ramp_steps=augmentation_ramp_steps,
        crop_transform=state["crop"],
    )
    progress = min(1.0, max(0.0, (step - augmentation_warmup_steps) / max(augmentation_ramp_steps, 1)))
    biased = (state["profile"] > 0).float() * progress
    current = current * (1.0 + (state["scale"] - 1.0)[:, None, None] * biased[:, None, None])
    current += (state["bias"] / max_depth * biased)[:, None, None]
    current.clamp_(0.0, 1.0)
    severe = state["profile"] == 2
    blocked = (torch.rand(env.num_envs, device=env.device) < occlusion_prob * progress) & severe
    # 连片遮挡比独立噪点更接近镜头污渍、部分遮挡；沿用部署端无回波=最远距离的语义。
    yy = torch.arange(grid_size[0], device=env.device)[None, :, None]
    xx = torch.arange(grid_size[1], device=env.device)[None, None, :]
    cy = torch.randint(grid_size[0], (env.num_envs, 1, 1), device=env.device)
    cx = torch.randint(grid_size[1], (env.num_envs, 1, 1), device=env.device)
    patch = ((yy - cy).abs() <= 2) & ((xx - cx).abs() <= 2) & blocked[:, None, None]
    current = current.masked_fill(patch, 1.0)
    queue = torch.cat((state["queue"][:, 1:], current[:, None]), dim=1)
    queue = torch.where(new_frame[:, None, None, None], queue, state["queue"])
    queue = torch.where(reset_mask[:, None, None, None], current[:, None].expand_as(queue), queue)
    latency = (state["latency"].float() * progress).long()
    delivered = queue[torch.arange(env.num_envs, device=env.device), max_latency_frames - latency]
    dropped = torch.rand(env.num_envs, device=env.device) < packet_loss_prob * progress
    advance = new_frame & ~reset_mask & ~dropped
    shifted = torch.cat((state["memory"][:, 1:], delivered[:, None]), dim=1)
    memory = torch.where(advance[:, None, None, None], shifted, state["memory"])
    state["memory"] = torch.where(reset_mask[:, None, None, None], delivered[:, None].expand_as(memory), memory)
    state["queue"] = queue
    state["frame"] = frame_id.clone()
    return state["memory"][:, ::memory_stride]


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
