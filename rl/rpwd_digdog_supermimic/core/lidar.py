"""Livox Mid-360 局部高度图模拟（训练/评测/数据记录共用）。

真实地形由 :class:`TerrainField` 的高度图给出。这里把与特权扫描完全相同的
21x13 网格转换成学生策略可观测的 LiDAR 高度图：

- 网格：x in [-1, 1] m、y in [-0.6, 0.6] m，随机体 yaw 旋转，与
  ``TerrainField.scan`` 同格同符号（正数=该格地面低于机身）；
- Mid-360：水平 360 度，垂直 -7 度 ~ +52 度，默认安装下倾 20 度，
  最远 40 m，最近 0.15 m；
- 射线遮挡：从雷达到目标格做高度场射线步进，被更高地形挡住的格标为不可见；
- 测距噪声和随机丢点：只在训练/带噪评测打开，确定性评测只保留视场和遮挡；
- 不可见格填入 ``LIDAR['unknown']``（默认 -2.0，低于真实扫描的 ±1.5 范围）。

安装外参默认是 base_link 上方 7 cm、x/y 居中、下倾 20 度。实物安装改变时只改
``robot/config.py:LIDAR``，训练端和部署端必须使用同一份常量。该常量不进入
``robot_signature``，因此不会让已有的盲策略 checkpoint 失效。
"""

from dataclasses import dataclass

import torch

from .terrain import SCAN_DIM, TerrainField
from ..robot.config import LIDAR


@dataclass(frozen=True)
class LidarCfg:
  """Mid-360 安装、视场和噪声参数。"""

  mount: tuple[float, float, float]
  tilt_deg: float
  v_min_deg: float
  v_max_deg: float
  range_min: float
  range_max: float
  update_hz: float
  noise_std: float
  dropout: float
  unknown: float
  occlusion_margin: float
  occlusion_samples: int

  @classmethod
  def from_contract(cls, contract=None):
    c = LIDAR if contract is None else contract
    cfg = cls(
      mount=tuple(float(v) for v in c['mount']),
      tilt_deg=float(c['tilt_deg']),
      v_min_deg=float(c['v_min_deg']),
      v_max_deg=float(c['v_max_deg']),
      range_min=float(c['range_min']),
      range_max=float(c['range_max']),
      update_hz=float(c['update_hz']),
      noise_std=float(c['noise_std']),
      dropout=float(c['dropout']),
      unknown=float(c['unknown']),
      occlusion_margin=float(c['occlusion_margin']),
      occlusion_samples=int(c['occlusion_samples']),
    )
    if cfg.update_hz <= 0:
      raise ValueError('LiDAR update_hz 必须为正')
    if cfg.range_min < 0 or cfg.range_max <= cfg.range_min:
      raise ValueError('LiDAR 测距范围不合法')
    if not cfg.v_min_deg < cfg.v_max_deg:
      raise ValueError('LiDAR 垂直视场不合法')
    if cfg.dropout < 0 or cfg.dropout >= 1 or cfg.noise_std < 0:
      raise ValueError('LiDAR 噪声/丢点参数不合法')
    if cfg.occlusion_samples < 0:
      raise ValueError('LiDAR 遮挡采样数不能为负')
    return cfg

  def hold_steps(self, dt):
    """控制周期 dt 下，高度图保持多少个策略步（Mid-360 默认 10 Hz）。"""
    return max(1, int(round(1.0 / (self.update_hz * dt))))


class LidarHeightScan:
  """由真实高度场生成 Mid-360 风格的 273 维局部高度图。

  ``scan`` 返回学生策略输入；``true_scan`` 返回特权教师/世界模型使用的真实
  扫描。两者同网格、同符号，便于教师-学生蒸馏和部署端对齐。
  """

  def __init__(self, cfg=None):
    self.cfg = cfg or LidarCfg.from_contract()
    n = self.cfg.occlusion_samples
    # 端点不采样：s=0 是雷达自身，s=1 是目标格。
    self._ray_s = torch.linspace(0., 1., n + 2)[1:-1]

  # ------------------------------------------------------------------ 基础接口
  def true_scan(self, field, state, pose):
    """特权真实高度扫描，和世界模型/Critic 的 scan 完全一致。"""
    return field.scan(state, pose)

  def scan(self, field, state, pose, noise=True):
    """学生策略可见的 LiDAR 高度图；noise=False 为确定性评测。"""
    return self.scan_with_mask(field, state, pose, noise=noise)[0]

  @torch.no_grad()
  def scan_with_mask(self, field: TerrainField, state, pose, noise=True):
    """返回 (高度图 (B,273), 可见掩码 (B,273))。"""
    self._check(field, state, pose)
    device = state.device
    heights = self.true_scan(field, state, pose)
    # 目标地面相对机身的 z。TerrainField.scan 的正方向是“地面比机身低”，
    # 所以高度图里的地面 z 取负号。
    target_z = -heights
    base_abs = state[:, :1] + field.sample(pose[:, :2])[:, None]
    offsets = field.offsets.to(device)
    mount_xy = state.new_tensor(self.cfg.mount[:2])
    mount_z = float(self.cfg.mount[2])
    dx = offsets[:, 0] - mount_xy[0]
    dy = offsets[:, 1] - mount_xy[1]
    radius = torch.sqrt(dx.square() + dy.square()).clamp_min(1e-6)
    dz = target_z - mount_z
    vertical = torch.atan2(dz, radius[None, :]) + torch.deg2rad(
      state.new_tensor(self.cfg.tilt_deg))
    rho = torch.sqrt(radius[None, :].square() + dz.square())
    visible = ((vertical >= torch.deg2rad(state.new_tensor(self.cfg.v_min_deg))) &
               (vertical <= torch.deg2rad(state.new_tensor(self.cfg.v_max_deg))) &
               (rho >= self.cfg.range_min) & (rho <= self.cfg.range_max))
    if self.cfg.occlusion_samples > 0:
      visible = visible & ~self._occluded(field, pose, base_abs, offsets,
                                          target_z, mount_z)
    if noise:
      heights = heights + self.cfg.noise_std * torch.randn_like(heights)
      if self.cfg.dropout:
        visible = visible & (torch.rand_like(heights) >= self.cfg.dropout)
    heights = torch.where(visible, heights, torch.full_like(heights, self.cfg.unknown))
    return heights, visible

  # ------------------------------------------------------------------ 内部实现
  def _occluded(self, field, pose, base_abs, offsets, target_z, mount_z):
    """高度场射线步进：目标格之前有地形高于视线则不可见。"""
    s = self._ray_s.to(pose.device)                                  # (S,)
    mount_xy = pose.new_tensor(self.cfg.mount[:2])                  # (2,)
    # local: (S,273,2)，从雷达安装点插值到每个目标格。
    local = mount_xy[None, None, :] + s[:, None, None] * (
      offsets[None, :, :] - mount_xy[None, None, :])
    c, sn = pose[:, 2].cos()[:, None, None], pose[:, 2].sin()[:, None, None]
    world = torch.stack((pose[:, None, None, 0] + c * local[None, :, :, 0] -
                         sn * local[None, :, :, 1],
                         pose[:, None, None, 1] + sn * local[None, :, :, 0] +
                         c * local[None, :, :, 1]), -1)              # (B,S,273,2)
    ground = field.sample(world) - base_abs[:, None, :]              # (B,S,273)
    ray = mount_z + s[None, :, None] * (target_z[:, None, :] - mount_z)
    return (ground > ray + self.cfg.occlusion_margin).any(1)         # (B,273)

  def _check(self, field, state, pose):
    if field.offsets.shape[0] != SCAN_DIM:
      raise ValueError(f'LiDAR 网格必须是 {SCAN_DIM} 维，实际 {field.offsets.shape[0]}')
    if state.ndim != 2 or state.shape[1] != 36:
      raise ValueError(f'state 必须是 (B,36)，实际 {tuple(state.shape)}')
    if pose.shape != (state.shape[0], 3):
      raise ValueError(f'pose 必须是 ({state.shape[0]},3)，实际 {tuple(pose.shape)}')
    if field.offsets.device != state.device:
      raise ValueError('LiDAR 高度场 offsets 与 state 不在同一设备')
