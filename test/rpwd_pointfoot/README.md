# RP_wd 点足版大狗测试记录

预留目录，记录 17.038kg 大狗点足的测试数据。

- 训练/回放包：`rl/rpwd_pointfoot_supermimic`（235→12，1.5Hz）
- 原生入口：`rl/scripts/pointfoot_mujoco.py` / `rl/scripts/play_pointfoot_mjlab.sh`
- 历史模型：原 60k 训练在 `/media/rmer/系统/rpwd_training/pointfoot_mjlab/2026-11-11_大狗点足版_60k_seed0/`；该盘当前可能未挂载，测试前先确认
- 数据目录：`runs/<时间>_model<轮次>/`

字段与 `../README.md` 第一版 schema 对齐（12 关节、球脚接触力/峰值、机身 IMU、命令/相位）。记录脚本待补；如果该盘未挂载，先把模型/日志拷到数据盘再跑。
