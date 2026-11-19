"""Explicit straight/side/yaw command modes, with more stair traversal samples."""
from dataclasses import dataclass
import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg


def mix_commands(values, stairs, surface=None):
  result = values.clone()
  mode = torch.rand(len(values),device=values.device)
  # General terrain: 40% pure vx, 20% pure vy, 20% pure yaw, 20% combined.
  result[mode<.4,1:] = 0.
  result[(mode>=.4)&(mode<.6),0] = 0.
  result[(mode>=.4)&(mode<.6),2] = 0.
  result[(mode>=.6)&(mode<.8),:2] = 0.
  # Stairs: 80% low-speed pure forward/backward, 20% full-range commands.
  result[stairs] = values[stairs]
  traverse = stairs & (mode<.8)
  magnitude = .15 + .15*torch.rand(len(values),device=values.device)
  result[traverse,0] = torch.where(values[traverse,0]<0,-magnitude[traverse],magnitude[traverse])
  result[traverse,1:] = 0.
  # stones/tires 必须真的走完桩/胎：统一给低速纯前进/后退并保持整回合，
  # 否则横移/转向指令会让 surface_stall（3 秒无 x 进展）在起点就终止。
  if surface is not None:
    result[surface] = values[surface]
    result[surface,0] = torch.where(values[surface,0]<0,-magnitude[surface],magnitude[surface])
    result[surface,1:] = 0.
  return result


class MixedVelocityCommand(UniformVelocityCommand):
  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    gen = self._env.cfg.scene.terrain.terrain_generator
    kinds = [c.kind for c in gen.sub_terrains.values()]
    stair_columns = torch.tensor([k in ('up','down') for k in kinds],device=self.device)
    surface_columns = torch.tensor([k in ('stones','tires') for k in kinds],device=self.device)
    types = self._env.scene.terrain.terrain_types[env_ids]
    stairs = stair_columns[types]
    surface = surface_columns[types]
    mixed = mix_commands(self.vel_command_b[env_ids],stairs,surface)
    # Stair magnitudes must obey the current curriculum bounds, including reverse.
    for i, bounds in enumerate((self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y, self.cfg.ranges.ang_vel_z)):
      mixed[:,i].clamp_(*bounds)
    self.vel_command_b[env_ids] = mixed
    if self.cfg.hold_stair_command:
      self.time_left[env_ids[~torch.zeros_like(stairs)]] = self._env.cfg.episode_length_s + 1.



@dataclass(kw_only=True)
class MixedVelocityCommandCfg(UniformVelocityCommandCfg):
  hold_stair_command: bool = False
  def build(self, env):
    return MixedVelocityCommand(self,env)
