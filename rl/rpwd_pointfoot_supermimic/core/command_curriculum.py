"""Velocity ranges scheduled in learner iterations, independent of sim substeps."""
STAGES = (
  (0, (-.2,.35), (-.15,.15), (-.5,.5)),
  (3000, (-.3,.45), (-.2,.2), (-.6,.6)),
  (8000, (-.4,.6), (-.25,.25), (-.8,.8)),
  (15000, (-.5,.6), (-.3,.3), (-1.,1.)),
  (22000, (-.6,.6), (-.35,.35), (-1.,1.)),
)


def set_command_stage(cfg, iteration, stage=None):
  index = stage if stage is not None else max(i for i, item in enumerate(STAGES) if max(0,iteration) >= item[0])
  # 地形探针阶段可到 6（14/16cm），但命令速度只有 0–4 五档；越界时沿用最后一档，
  # 避免和 DigDog 13050 轮相同的 IndexError 中断训练。
  index = min(max(int(index), 0), len(STAGES) - 1)
  _, x, y, yaw = STAGES[index]
  cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z = x, y, yaw
  return index
