"""Static Go2W courses. Independent from the original Go2 terrain preset."""
from dataclasses import dataclass
import math
import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainOutput, TerrainGeometry, TerrainGeneratorCfg
from mjlab.managers import CurriculumTermCfg, TerminationTermCfg, RewardTermCfg, EventTermCfg
from .flat import go2w_velocity_flat_env_cfg

STEP_HEIGHTS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16)
TERRAIN_KINDS = ('flat', 'up', 'down', 'platform', 'slope', 'rubble', 'stones', 'tires')
TERRAIN_PROPORTIONS = (.28, .32, .14, .04, .04, .08, .05, .05)

@dataclass
class CourseCfg(SubTerrainCfg):
  kind: str = "rubble"
  bidirectional: bool = False
  stair_tread: float = .45

  def function(self, difficulty, spec, rng):
    level = min(len(STEP_HEIGHTS) - 1, max(0, round(difficulty * (len(STEP_HEIGHTS) - 1))))
    h = STEP_HEIGHTS[level]
    sx, sy = self.size
    if self.bidirectional:
      sx = 8.  # Mirror the original course into a 16 m square.
    body, geoms = spec.body("terrain"), []
    def box(x, y, z, a, b, c, yaw=0., pitch=0., shade=.45):
      # Trim the source half at its spawn plane before mirroring, avoiding
      # coincident contact surfaces in the central floor/down-stair landing.
      if self.bidirectional and yaw == 0. and pitch == 0.:
        left, right = max(1., x-a/2), x+a/2
        if right <= left:
          return
        x, a = (left+right)/2, right-left
      # Composition Rz(yaw) Ry(pitch); z is centre, dimensions are full lengths.
      cy, ss, cp, sp = math.cos(yaw/2), math.sin(yaw/2), math.cos(pitch/2), math.sin(pitch/2)
      placements = [(x, (cy*cp,-ss*sp,cy*sp,ss*cp))]
      if self.bidirectional:
        # Reflect about x=8: yaw and pitch change sign, roll stays unchanged.
        placements = [(x+7., (cy*cp,-ss*sp,cy*sp,ss*cp)),
                      (9.-x, (cy*cp,-ss*sp,-cy*sp,-ss*cp))]
      for xx, quat in placements:
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=(xx,y,z),
                             size=(a/2,b/2,c/2), quat=quat)
        geoms.append(TerrainGeometry(geom=geom, color=(shade,shade,.9*shade,1.)))
    def direct_box(x, y, z, a, b, c, yaw=0., shade=.5):
      # Start platform / unique geometry that must not be mirrored.
      cy, ss = math.cos(yaw/2), math.sin(yaw/2)
      geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=(x,y,z),
                           size=(a/2,b/2,c/2), quat=(cy,0.,0.,ss))
      geoms.append(TerrainGeometry(geom=geom, color=(shade,shade,.9*shade,1.)))
    box(sx/2,sy/2,-.1,sx,sy,.2,shade=.6)
    origin = np.array([1., sy/2, 0.])
    if self.kind in ("up", "down"):
      n, tread = 5, self.stair_tread
      if self.kind == "up":
        # Overlapping foundation boxes make solid risers and a long top landing.
        for k in range(n):
          start = 2. + k*tread
          box((start+sx)/2,sy/2,(k+1)*h/2,sx-start,sy,(k+1)*h)
      else:
        origin[2] = n*h
        for k in range(n):
          end = 2. + k*tread
          height = (n-k)*h
          box(end/2,sy/2,height/2,end,sy,height)
    elif self.kind == "platform":
      for x in (2.6,4.6):
        box(x,sy/2,h/2,.9,sy,h)
    elif self.kind == "slope":
      # A continuous wedge made by a tilted slab with its low edge buried.
      angle = math.radians(3+12*difficulty)
      length = 3.
      thickness = .16
      center_z = length/2*math.sin(angle)-thickness/2*math.cos(angle)
      box(3.5,sy/2,center_z,length,sy,thickness,pitch=-angle)
      end = 3.5+length/2*math.cos(angle)
      top = length*math.sin(angle)
      box((end+sx)/2,sy/2,top/2,sx-end,sy,top)
    elif self.kind == "rubble":
      # Staggered rows cover the lane. Static overlapping rocks, never loose bodies.
      # Heights of all layers together are capped by h (2--20 cm).
      # 只铺中央赛道（bidirectional 会镜像到两侧，16m 全宽铺满会浪费静态 geom）。
      for ix, x in enumerate(np.arange(2.15,5.9,.48)):
        for y in np.arange(sy/2-2.4,sy/2+2.45,.6):
          a,b = rng.uniform(.35,.65,2)
          height = h*rng.uniform(.35,.7)
          xx,yy = x+rng.uniform(-.12,.12),y+rng.uniform(-.12,.12)
          yaw = rng.uniform(-math.pi,math.pi)
          box(xx,yy,height/2,a,b,height,yaw=yaw,shade=rng.uniform(.3,.55))
          if rng.random() < .25+.35*difficulty:
            cap = h-height
            # Deliberately overlapping cap; bounded surface height even in stacks.
            pitch = rng.uniform(-.15,.15)*difficulty
            half_z = (cap*math.cos(pitch)+a*.65*abs(math.sin(pitch)))/2
            box(xx+.04,yy-.03,h-half_z,a*.65,b*.65,cap,
                yaw=yaw+.6,pitch=pitch,shade=.55)
    elif self.kind == "stones":
      # PDF 表 1 梅花桩：两排、每排 3 块、中心对称；上排首块距平台 100mm、
      # 下排 400mm，同排间距 400mm、两排间距 200mm。用户要求顶面恒定 1m 且与
      # 平台齐平：方块/平台是 0.9m 基座上的 100mm 薄块，缝隙深 100mm，脚不会卡死。
      # 课程从简单开始：方块 0.40→0.20，同排缝 5→40cm，两排缝 3→20cm，
      # 首块缝 5→10cm，两排错位 3→30cm；上排首块始终在前进方向的左脚侧。
      top, thick, plat = 1.0, .10, .80
      base = top - thick
      span = len(STEP_HEIGHTS) - 1
      t = level / span
      side = .40 - .20 * t
      gap_same = .05 + .35 * t
      gap_row = .03 + .17 * t
      gap_near = .05 + .05 * t
      delta = .03 + .27 * t
      field = 3 * side + 2 * gap_same + 2 * gap_near + delta
      module = field + plat
      # 基座只比桩阵略宽，避免在桩阵外侧留出可绕行的 0.9m 台阶。
      base_w = max(plat, 2 * side + gap_row + .02)
      # 中央起点平台按 PDF 做成 800×800，而不是整条 16m 宽的安全台，
      # 否则策略可以一直站在平台上刷 foothold / 速度跟踪。
      direct_box(8., sy/2, top/2, .8, .8, top, shade=.6)
      for d in (1., -1.):
        edge = 8. + d * .4
        n = 0
        while n < 3 and abs(edge + d * (n + 1) * module - 8.) <= 7.4:
          n += 1
        if n == 0:
          continue
        length = n * module
        direct_box(edge + d * length / 2, sy/2, base / 2, length, base_w, base, shade=.42)
        for m in range(n):
          start = edge + d * m * module
          # 上排首块在前进方向的左脚侧：d=+1 → +y；d=-1 → -y。
          row_a = sy/2 + d * (side + gap_row) / 2
          row_b = sy/2 - d * (side + gap_row) / 2
          for row, off in ((row_a, gap_near), (row_b, gap_near + delta)):
            for k in range(3):
              xc = start + d * (off + side / 2 + k * (side + gap_same))
              direct_box(xc, row, base + thick / 2, side, side, thick,
                         shade=rng.uniform(.35, .55))
          direct_box(start + d * (field + plat / 2), sy/2, top / 2, plat, plat, top,
                     shade=.6)
      origin[2] = top
    elif self.kind == "tires":
      # 按 PDF 165/70R13：外径 550mm、内径 330mm、胎面宽 165mm，8 胎平躺成 2×4 矩阵。
      # 训练赛道沿 x 排 8 列、y 排 2 行（4.4m×1.1m 的轮胎带），左右桩阵由 box() 双向镜像。
      # 12 段 box 近似圆环，胎面孔隙可见、可踩空；胎高固定 165mm，不再随课程升高。
      D, hole, w = .55, .33, .165
      rmid = (D + hole) / 4
      seg = (D - hole) / 2
      tang = 2 * math.pi * rmid / 12
      for ix, x in enumerate(np.arange(2.7 + D/2, 7.1, D)):
        for iy, y in enumerate((sy/2 - D/2, sy/2 + D/2)):
          cx, cy = x + rng.uniform(-.01, .01), y + rng.uniform(-.01, .01)
          for k in range(12):
            ang = 2 * math.pi * k / 12
            box(cx + rmid*math.cos(ang), cy + rmid*math.sin(ang), w/2,
                seg, tang, w, yaw=ang, shade=rng.uniform(.22, .38))
    elif self.kind != "flat":
      raise ValueError(self.kind)
    if self.bidirectional:
      origin[0] = 8.
    return TerrainOutput(origin=origin, geometries=geoms)


def terrain_linear_tracking(env):
  velocity = env.scene["robot"].data.root_link_lin_vel_b
  command = env.command_manager.get_command("twist")
  return torch.exp(-(velocity[:,:2]-command[:,:2]).square().sum(-1)/.25)


def terrain_yaw_tracking(env):
  velocity = env.scene["robot"].data.root_link_ang_vel_b
  command = env.command_manager.get_command("twist")
  return torch.exp(-(velocity[:,2]-command[:,2]).square()/.5)


def course_exit(env):
  offset = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
  return (offset[:,0] > 5.4) | (offset[:,0] < -.65) | (offset[:,1].abs() > 2.7)


def stone_fall(env):
  """梅花桩：机身降到起点平台以下（掉下桩/踩空）即终止。"""
  bridge = getattr(env, "_go2w_posture_bridge", None)
  if bridge is None:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  types = env.scene.terrain.terrain_types
  root = bridge.robot.data.root_link_pos_w
  below = root[:, 2] - env.scene.env_origins[:, 2] < .10
  return (types == 6) & below


def course_curriculum(env, env_ids):
  terrain = env.scene.terrain
  offset = env.scene["robot"].data.root_link_pos_w[env_ids] - env.scene.env_origins[env_ids]
  active = env.episode_length_buf[env_ids] > 0
  fallen = env.termination_manager.terminated[env_ids]
  success = active & ~fallen & (offset[:,0] > 5.4) & (offset[:,1].abs() < 2.7)
  failure = active & (fallen | (offset[:,0] < -.65) | (offset[:,1].abs() > 2.7))
  terrain.terrain_levels[env_ids] = (terrain.terrain_levels[env_ids] + success.long() - failure.long()).clamp(0,len(STEP_HEIGHTS)-1)
  terrain.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
  return terrain.terrain_levels.float().mean()


def bidirectional_exit(env):
  offset = env.scene["robot"].data.root_link_pos_w - env.scene.env_origins
  return (offset[:,0].abs() > 6.5) | (offset[:,1].abs() > 6.5)


def bidirectional_curriculum(env, env_ids):
  terrain = env.scene.terrain
  offset = env.scene["robot"].data.root_link_pos_w[env_ids] - env.scene.env_origins[env_ids]
  active = env.episode_length_buf[env_ids] > 0
  fallen = env.termination_manager.terminated[env_ids]
  # Both course directions count. Lateral exits are neutral, not evidence
  # of clearing steps; negative commands no longer count as failure.
  success = active & ~fallen & (offset[:,0].abs() > 5.4) & (offset[:,1].abs() < 6.5)
  failure = active & fallen
  terrain.terrain_levels[env_ids] = (terrain.terrain_levels[env_ids] + success.long() - failure.long()).clamp(0,len(STEP_HEIGHTS)-1)
  terrain.env_origins[env_ids] = terrain.terrain_origins[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]]
  return terrain.terrain_levels.float().mean()



def go2w_velocity_terrain_env_cfg(play=False,history=5,seed=0,revision=7,actor_wheel_vel=False,robot='rpwd_pointfoot'):
 from mjlab.sensor import ContactMatch,ContactSensorCfg
 from ..core.stair_curriculum import curriculum, mark_surface_reached, surface_stall
 from ..core.collision import illegal_nonwheel_contact
 from ..core.mixed_commands import MixedVelocityCommandCfg
 cfg=go2w_velocity_flat_env_cfg(play,history,False,robot)
 cfg.scene.terrain.terrain_type='generator'
 cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(seed=seed,size=(16.,16.),num_rows=len(STEP_HEIGHTS),num_cols=len(TERRAIN_KINDS),
  border_width=5.,curriculum=True,sub_terrains={kind:CourseCfg(kind=kind,bidirectional=True,stair_tread=.30,proportion=p)
  for kind,p in zip(TERRAIN_KINDS, TERRAIN_PROPORTIONS)})
 cfg.scene.terrain.max_init_terrain_level=0
 cfg.scene.sensors=(ContactSensorCfg(name='wheel_ground_contact',
  primary=ContactMatch(mode='body',pattern=tuple(f'{leg}_foot_link' for leg in ('FL','FR','RL','RR')),entity='robot'),
  secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='netforce',num_slots=1),
  ContactSensorCfg(name='nonwheel_ground_contact',
  primary=ContactMatch(mode='body',pattern='.*',entity='robot',
  exclude=tuple(f'{leg}_foot_link' for leg in ('FL','FR','RL','RR'))),
  secondary=ContactMatch(mode='body',pattern='terrain'),
  fields=('found','force'),reduce='netforce',num_slots=1,history_length=cfg.decimation))
 cfg.events['reset_base'].params['pose_range'].update(x=(-.1,.1),y=(-.1,.1),yaw=(-.08,.08))
 # 每一步锁存“踩在桩顶/胎面越过终点线”，reset 时课程再消费，避免出口踩空。
 cfg.events['surface_reached']=EventTermCfg(mode='step',func=mark_surface_reached)
 cfg.episode_length_s=35.
 cfg.terminations['course_exit']=TerminationTermCfg(func=bidirectional_exit,time_out=True)
 cfg.terminations['stone_fall']=TerminationTermCfg(func=stone_fall,time_out=False)
 # stones/tires 前进指令下 3 秒无 +0.1m 进展即判失败，不再等 35 秒超时。
 cfg.terminations['surface_stall']=TerminationTermCfg(func=surface_stall,time_out=False)
 cfg.terminations['illegal_nonwheel_contact']=TerminationTermCfg(func=illegal_nonwheel_contact,params={'force_threshold':15.,'surface_force_threshold':300.})
 if not play:
  cfg.curriculum={'course':CurriculumTermCfg(func=curriculum)}
  cfg.commands['twist']=MixedVelocityCommandCfg(**vars(cfg.commands['twist']))
  cfg.commands['twist'].hold_stair_command=True
 return cfg
