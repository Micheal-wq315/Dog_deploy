"""Velocity ranges scheduled in learner iterations, independent of sim substeps."""
# DigDogv1 腿长约0.4m、步态1.25Hz；0.2m/s时参考IK无截断，超过约0.3-0.35m/s
# 横移/转向组合开始撞关节限位，因此速度课程按本机器人缩放。
STAGES = (
  (0, (-.15,.25), (-.10,.10), (-.4,.4)),
  (3000, (-.20,.30), (-.12,.12), (-.5,.5)),
  (8000, (-.25,.35), (-.15,.15), (-.6,.6)),
  (15000, (-.30,.40), (-.18,.18), (-.7,.7)),
  (22000, (-.35,.45), (-.20,.20), (-.8,.8)),
)


def set_command_stage(cfg, iteration, stage=None):
  # 地形课程有 0..6 段（含 14/16cm 探针），命令速度表只有 0..4 五档；
  # 地形探针升到第 5/6 段时沿用最后一档，不能拿 5 去索引 STAGES。
  index = stage if stage is not None else max(i for i, item in enumerate(STAGES) if max(0,iteration) >= item[0])
  index = min(max(int(index), 0), len(STAGES) - 1)
  _, x, y, yaw = STAGES[index]
  cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z = x, y, yaw
  return index
