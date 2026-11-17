# Go2W 轮足参考测试记录

预留目录，记录 Go2W 原版轮足的测试数据。

- 训练/回放包：`rl/rpwd_go2w_supermimic`（`--robot go2w`）
- 原生入口：`rl/scripts/play_rpwd_go2w.sh --robot go2w`（或 `rl/scripts/go2w_mujoco.py`）
- 模型：`/media/rmer/data/rpwd_training/go2w_supermimic/` 下的正式 run（按需指定 checkpoint）
- 数据目录：`runs/<时间>_model<轮次>/`

字段与 `../README.md` 第一版 schema 对齐（关节角/dq/目标角/动作/扭矩、足端力/峰值、机身 IMU、命令/相位、终止原因）。记录脚本待补，补好后把 `record_test.py` 放到本目录即可。
