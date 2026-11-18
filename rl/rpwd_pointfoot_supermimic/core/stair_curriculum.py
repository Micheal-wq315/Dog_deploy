"""V5 episode outcomes and performance-gated velocity curriculum.

All buffers are training-only metadata, never actor observations.
"""
import torch
from ..robot.config import STAND_HEIGHT

KINDS = ('flat', 'up', 'down', 'platform', 'slope', 'rubble', 'stones', 'tires')
HEIGHTS = (.02, .04, .06, .08, .10, .12, .14, .16)
# 20000 轮教师必须覆盖到 16cm：第 6 段（14cm 考核）在 15000 轮解锁，留 5000 轮
# 在无探针固定的最高档继续训练。若沿用 34000 轮的旧表，20k 只会走到 10cm。
MIN_ITERATIONS = (0, 2000, 5000, 8000, 11000, 13000, 15000)


def buffers(env):
  if not hasattr(env, '_stair_stats'):
    env._stair_stats = torch.zeros(len(KINDS), len(HEIGHTS), 4, device=env.device)  # trials, pass, fall, stuck
    env._stair_stall_time = torch.zeros(env.num_envs, device=env.device)
    env._stair_stuck = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  if not hasattr(env, '_stair_recent'):
    window = getattr(env.cfg, 'promotion_window', 256)
    env._stair_recent = torch.zeros(len(HEIGHTS) - 2, window, dtype=torch.uint8, device=env.device)
    env._stair_recent_count = torch.zeros(len(HEIGHTS) - 2, dtype=torch.long, device=env.device)
    env._stair_episode_stage = torch.full((env.num_envs,), getattr(env, '_stair_stage', 0), dtype=torch.long, device=env.device)
  return env._stair_stats


def stair_pass(offset, gravity, level, up):
  """Require body/rear clearance of the last riser, correct landing height and stability."""
  heights = offset.new_tensor(HEIGHTS)[level]
  expected = torch.where(up, 5 * heights, -5 * heights)
  # 机身高度窗口必须按本机器人0.16m站高，而不是点足0.34m的固定 .23/.46。
  lo = expected + STAND_HEIGHT - .10
  hi = expected + STAND_HEIGHT + .12
  return ((offset[:, 0].abs() > 2.8) & (offset[:, 1].abs() < 2.5)
          & (offset[:, 2] > lo) & (offset[:, 2] < hi)
          & (gravity[:, :2].norm(dim=-1) < .4))


def mark_surface_reached(env, env_ids=None):
  """锁存“已经踩在桩顶/胎面到达终点线”的瞬间。

  课程只在 reset 时统计成功，而 stones/tires 的出口在桩/胎区边缘，reset 时
  前脚往往已经踩到平地；不锁存会把真正踩过去的回合计成 0 pass。
  """
  bridge = getattr(env, '_go2w_posture_bridge', None)
  if bridge is None:
    return
  if not hasattr(env, '_surface_reached'):
    env._surface_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  if not hasattr(env, '_surface_feet'):
    env._surface_feet, _ = env.scene['robot'].find_bodies(
        tuple(f'{x}_foot_link' for x in ('FL', 'FR', 'RL', 'RR')), preserve_order=True)
  types = env.scene.terrain.terrain_types
  active = env.episode_length_buf > 0
  fallen = env.termination_manager.terminated
  offset = env.scene['robot'].data.root_link_pos_w - env.scene.env_origins
  feet = env.scene['robot'].data.body_link_pos_w[:, env._surface_feet]
  on_top = (bridge.field.sample(feet[:, :, :2]) > .01).float().mean(-1) > .5
  reached = (((types == 6) | (types == 7)) & active & ~fallen
             & (offset[:, 0].abs() > 5.4) & (offset[:, 1].abs() < 6.5) & on_top)
  env._surface_reached |= reached


def curriculum(env, env_ids):
  stats = buffers(env)
  terrain = env.scene.terrain
  d = env.scene['robot'].data
  offset = d.root_link_pos_w[env_ids] - env.scene.env_origins[env_ids]
  levels = terrain.terrain_levels[env_ids].clone()
  types = terrain.terrain_types[env_ids]
  # Eight fixed columns, generated in KINDS order.
  stairs = (types == 1) | (types == 2)
  active = env.episode_length_buf[env_ids] > 0
  fallen = env.termination_manager.terminated[env_ids]
  success = (offset[:, 0].abs() > 5.4) & (offset[:, 1].abs() < 6.5)
  top = stair_pass(offset, d.projected_gravity_b[env_ids], levels, types == 1)
  foot_ids,_=env.scene['robot'].find_bodies(tuple(f'{x}_foot_link' for x in ('FL','FR','RL','RR')),preserve_order=True)
  feet=d.body_link_pos_w[env_ids][:,foot_ids]-env.scene.env_origins[env_ids,None,:]
  # stones/tires 不允许在平地上绕过：至少一半支撑脚要落在桩顶/胎面(高地面)。
  # 出口在桩/胎区边缘，reset 瞬间前脚可能已落到平地；用 step 事件锁存的
  # “踩在桩顶/胎面越过终点线”补充判定，避免真正通过的回合被判 0 pass。
  surface = (types == 6) | (types == 7)
  on_top_frac = torch.zeros_like(offset[:, 0])
  bridge = getattr(env, '_go2w_posture_bridge', None)
  if bridge is not None:
    world_feet = feet + env.scene.env_origins[env_ids, None, :]
    on_top_frac = (bridge.field.sample(world_feet[:, :, :2]) > .01).float().mean(-1)
    on_top = on_top_frac > .5
    reached = getattr(env, '_surface_reached', None)
    if reached is not None:
      success = torch.where(surface, (success & on_top) | reached[env_ids], success)
      reached[env_ids] = False
    else:
      success = torch.where(surface, success & on_top, success)
  # stones/tires 回合诊断：终点 x、on_top 比例、掉落/出口/超时分类。累计值
  # 只用于日志，不进入观测；train.py 的 terrain_details 会把它写进 metrics。
  if not hasattr(env, '_surface_diag'):
    env._surface_diag = torch.zeros(2, 6, device=env.device)
  exited = (offset[:, 0].abs() > 6.5) | (offset[:, 1].abs() > 6.5)
  for index, kind in enumerate((6, 7)):
    selected = (types == kind) & active
    if not selected.any():
      continue
    diag = env._surface_diag[index]
    diag[0] += selected.sum()
    diag[1] += offset[selected, 0].sum()
    diag[2] += on_top_frac[selected].sum()
    diag[3] += fallen[selected].sum()
    diag[4] += exited[selected].sum()
    diag[5] += (~fallen[selected] & ~exited[selected]).sum()
  direction=offset[:,0].sign()
  rear_clear=(feet[:,:,0]*direction[:,None]).amin(-1)>2.65
  top=top & rear_clear
  # Landing contacts rule out simply jumping through the clearance test.
  contact = (env.scene['wheel_ground_contact'].data.force[env_ids].reshape(len(env_ids),-1,3)[:,:,2] < -8.).sum(-1) >= 2
  success = torch.where(stairs, top & contact, success) & active & ~fallen
  command = env.command_manager.get_command('twist')[env_ids]
  traversal = (command[:, 0].abs() > .1) & (command[:, 1:].abs().amax(-1) < .05)
  trials = (~stairs | traversal) & active & ((env.episode_length_buf[env_ids] * env.step_dt >= 5.) | fallen)
  values = torch.stack((trials, success & trials, fallen & trials,
                        env._stair_stuck[env_ids] & trials), -1).float()
  stats.view(-1, 4).index_add_(0, types * len(HEIGHTS) + levels, values)
  record_recent(env, levels, types, trials, values[:, 1], env._stair_episode_stage[env_ids])
  terrain.terrain_levels[env_ids] = (levels + success.long() - (active & fallen).long()).clamp(0, len(HEIGHTS) - 1)
  if hasattr(env, '_stair_probe_mask') and env._stair_stage < len(HEIGHTS) - 2:
    pinned = env._stair_probe_mask[env_ids]
    terrain.terrain_levels[env_ids[pinned]] = env._stair_stage + 1
  terrain.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], types]
  env._stair_episode_stage[env_ids] = getattr(env, '_stair_stage', 0)
  env._stair_stall_time[env_ids] = 0
  env._stair_stuck[env_ids] = False
  return terrain.terrain_levels.float().mean()


def append_recent(history, counts, index, outcomes):
  """批量写入环形窗口；只保留最近window条，避免重复索引造成不确定写入。"""
  n, window = len(outcomes), history.shape[1]
  take = min(n, window)
  if take:
    positions = (counts[index] + torch.arange(n - take, n, device=history.device)) % window
    history[index, positions] = outcomes[-take:].to(torch.uint8)
    counts[index] += n


def record_recent(env, levels, types, trials, passed, episode_stages):
  # 扩速前开始的旧回合不能作为新命令阶段的考核成绩。
  current = episode_stages == getattr(env, '_stair_stage', 0)
  for index in range(len(HEIGHTS) - 2):
    eligible = (types == 1) & (levels == index + 1) & trials & current
    append_recent(env._stair_recent, env._stair_recent_count, index, passed[eligible])


def recent_counts(env):
  buffers(env)
  return torch.stack((env._stair_recent_count.clamp_max(env._stair_recent.shape[1]),
                       env._stair_recent.sum(-1)), -1)


def next_stage(stage, iteration, recent, velocity_mae, min_trials=128, pass_rate=.60):
  if stage >= len(HEIGHTS) - 2 or iteration < MIN_ITERATIONS[stage + 1]:
    return stage
  trials, passed = recent[stage]
  accurate = max(velocity_mae[:2]) < .18 and velocity_mae[2] < .30
  return stage + int(trials >= min_trials and passed / max(trials, 1) >= pass_rate and accurate)


def set_probe_stage(env, stage, fraction=.2):
  """固定一部分上阶环境考核晋级高度；其余环境照常升降难度。"""
  env._stair_stage = stage
  ids = torch.where(env.scene.terrain.terrain_types == 1)[0]
  env._stair_probe_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  if stage < len(HEIGHTS) - 2 and len(ids):
    count = max(1, round(len(ids) * fraction))
    # 沿环境序号均匀抽取，避免只覆盖某一连续区域的随机参数。
    positions = torch.linspace(0, len(ids) - 1, count, device=env.device).long()
    probes = ids[positions]
    env._stair_probe_mask[probes] = True
    env.scene.terrain.terrain_levels[probes] = stage + 1
    terrain = env.scene.terrain
    terrain.env_origins[probes] = terrain.terrain_origins[terrain.terrain_levels[probes], terrain.terrain_types[probes]]


def snapshot_recent(env):
  return dict(version='recent-v1', window=env._stair_recent.shape[1],
              history=env._stair_recent.detach().cpu().clone(),
              count=env._stair_recent_count.detach().cpu().clone())


def restore_recent(env, saved):
  buffers(env)
  recent = (saved or {}).get('recent')
  if recent is None:
    return  # 旧累计统计仅保留作日志，不能伪装成最近表现。
  if recent.get('version') != 'recent-v1' or recent['window'] != env._stair_recent.shape[1]:
    raise ValueError('课程窗口版本或大小不匹配')
  env._stair_recent.copy_(recent['history'].to(env.device))
  env._stair_recent_count.copy_(recent['count'].to(env.device))


def promotion_report(env, stage):
  if stage >= len(HEIGHTS) - 2:
    return dict(stage=stage, complete=True)
  trials, passed = recent_counts(env)[stage].cpu().tolist()
  return dict(stage=stage, target_height_cm=(4, 6, 8, 10, 12, 14)[stage], trials=trials,
              passed=passed, pass_rate=passed / max(trials, 1),
              required_trials=getattr(env.cfg, 'promotion_min_trials', 128),
              required_pass_rate=getattr(env.cfg, 'promotion_pass_rate', .6),
              probe_envs=int(env._stair_probe_mask.sum()))


def summary(env):
  stats = buffers(env).detach().cpu().tolist()
  levels, types = env.scene.terrain.terrain_levels, env.scene.terrain.terrain_types
  result = {}
  for i, name in enumerate(KINDS):
    selected = levels[types == i]
    result[name] = dict(level_counts=torch.bincount(selected, minlength=len(HEIGHTS)).cpu().tolist(),
                       episodes_by_level=stats[i])
  diag = env._surface_diag.detach().cpu().tolist() if hasattr(env, '_surface_diag') else [[0.] * 6] * 2
  result['surface_diag'] = {
      name: dict(trials=diag[i][0], end_x_mean=diag[i][1] / max(diag[i][0], 1.),
                 on_top_mean=diag[i][2] / max(diag[i][0], 1.),
                 fall=diag[i][3], exit=diag[i][4], timeout=diag[i][5])
      for i, name in enumerate(('stones', 'tires'))}
  return result
