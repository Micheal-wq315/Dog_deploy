"""Non-wheel collision termination and pre-reset labels for imagined survival."""
import torch


def illegal_nonwheel_contact(env, force_threshold=10., surface_force_threshold=None):
  data = env.scene['nonwheel_ground_contact'].data
  force = data.force_history if data.force_history is not None else data.force
  finite = torch.isfinite(force).flatten(1).all(-1)
  magnitude = torch.nan_to_num(force, nan=0., posinf=0., neginf=0.).norm(dim=-1)
  peak = magnitude.flatten(1).amax(-1)
  threshold = peak.new_tensor(force_threshold)
  if surface_force_threshold is not None:
    # 梅花桩/轮胎是 1m 高的柱面，脚探进 3–5cm 窄缝时小腿必然会擦到桩壁；
    # 用通用的 15N 阈值会在起点边缘反复终止，策略只能学会不上桩。
    types = env.scene.terrain.terrain_types
    surface = (types == 6) | (types == 7)
    threshold = torch.where(surface, peak.new_tensor(surface_force_threshold),
                            threshold)
  hit = (peak > threshold) & finite
  # Scene/termination buffers are cleared by automatic reset. These copies
  # survive it, so the learner retains the transition that caused termination.
  env._nonwheel_hit = hit.detach().clone()
  env._nonwheel_label_valid = finite.detach().clone()
  env._nonwheel_peak = peak.detach().clone()
  return hit


def collision_events(batch):
  valid = batch['valid'].squeeze(-1).bool()
  prefix = torch.cat((torch.ones_like(valid[:,:1]), valid.cumprod(1).bool()[:,:-1]),1)
  return ((batch['collision'].squeeze(-1) > .5)
          & batch['collision_valid'].squeeze(-1).bool() & prefix).any(-1)
