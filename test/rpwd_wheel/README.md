# RP_wd 轮足版大狗测试记录

预留目录，记录原部署机器人 RP_wd 轮足的测试数据。

- 训练/回放：
  - Isaac Lab：`rl/rp_wd_rl`（168→16，原 PPO/Odin/SuperMimic 链路）
  - MuJoCo SuperMimic：`rl/rpwd_go2w_supermimic --robot rpwd`（245→16）
- 部署端：`src/inference/robots/wheel_quad`（ROS2 推理节点）
- 数据目录：`runs/<时间>_model<轮次>/`

字段与 `../README.md` 第一版 schema 对齐；轮足版本额外需要 4 个轮子关节的目标/实际轮速、轮端接触力和轮力矩。记录脚本待补。
