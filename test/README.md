# 大狗测试数据记录

四条大狗各自预留独立目录，原始测试数据默认写在各自的 `runs/` 下（`.gitignore` 已忽略，避免把大 CSV 提交进 git）。

| 机器人 | 目录 | 训练/回放入口 | 当前状态 |
|---|---|---|---|
| DigDogv1 点足 | `test/dig_dog/` | `rl/rpwd_digdog_supermimic` + `rl/scripts/play_digdog_mjlab.sh` | **第一版 `record_test.py` 已就绪**，正向/反向 10/12cm 数据已记录 |
| RP_wd 轮足 | `test/rpwd_wheel/` | `rl/rp_wd_rl`（Isaac Lab）/ `rl/rpwd_go2w_supermimic --robot rpwd` | 目录/字段已预留，记录脚本待补 |
| Go2W 轮足参考 | `test/go2w/` | `rl/rpwd_go2w_supermimic --robot go2w` | 目录/字段已预留，记录脚本待补 |
| RP_wd 点足 | `test/rpwd_pointfoot/` | `rl/rpwd_pointfoot_supermimic` + `rl/scripts/pointfoot_mujoco.py` | 目录/字段已预留，记录脚本待补 |

## 数据格式（第一版，DigDog 已实现）

每个 lane × seed × direction 一个子目录：

```
<lane>_seed<k>_<direction>/
  trajectory.csv        # 50Hz 逐控制步全量数据
  observations.npy      # [N,235] Actor 本体历史观测（默认保存，可用 --no-obs 关闭）
  height_scan.npy       # [N,273] 高度图策略的当前扫描（仅 --scan true/lidar 模型）
  summary.json          # 该次测试的汇总
```

run 级还有 `meta.json`、`summary.json`（所有 lane/seed/direction 汇总）。

### trajectory.csv 主要字段

- **时间/命令/相位**：`t`、`t_since_start`、`cmd_vx/vy/wz`、`phase`、`phase_sin/cos`
- **机身**：`x/y/z`、`qw/qx/qy/qz`、世界/机身线速度、机身角速度、投影重力、`tilt_deg`、`clearance`
- **12 关节**（FL/FR/RL/RR × hip/thigh/calf）：
  `q_*`、`dq_*`、`q_target_*`、`action_*`、`tau_*`、`dq_delta_*`、`tau_delta_*`、`action_delta_*`
- **四足**：`foot_<leg>_pos_*`、`foot_<leg>_vel_*`、`foot_<leg>_ground`、`foot_<leg>_clearance`、`foot_<leg>_slip`、`foot_<leg>_force_xyz`、`foot_<leg>_peak`、`foot_<leg>_count`、`foot_<leg>_contact`
- **碰撞/终止**：`nonwheel_force`、`ncon`、`wheel_contacts`、`finite`

### summary.json 主要字段

`lane/seed/direction/success/reason/measured_seconds/max_x/max_tilt/peak_nonwheel_force/actual_mean/velocity_mae/rows/foot_peak_max/tau_abs_max/height_scan/scan_mode/scan_dim/specification`

### 高度图 / Mid-360 约定

- 高度图网格与训练特权扫描完全一致：**21×13=273**，机体 yaw 坐标，x∈[-1,1]m、y∈[-0.6,0.6]m，正数表示该格地面低于机身。
- 盲策略模型没有 `height_scan.npy`；`scan_mode=true` 是特权教师（真实高度场），`scan_mode=lidar` 是 Mid-360 学生（遮挡/视场/噪声/丢点，未知格 -2.0）。
- LiDAR 高度图按 **10Hz** 更新并保持 5 个 50Hz 控制步，所以相邻 CSV 行的扫描可能重复；`height_scan.npy` 与 `observations.npy` 逐行对齐。
- 原生回放的确定性评测使用 `noise=False`：仍保留视场和射线遮挡，但不加随机测距噪声/丢点；训练时打开噪声。

## 重要约定

- `success=false` 且 `max_x > goal_x` 表示**已经越过终点**，只是成功锁定条件（至少两足接触、离地间隙、竖直速度）没触发；`reason=timeout` 不等于“没上去”。判断爬阶能力优先看 `max_x`、`peak_nonwheel_force` 和足端 CSV。
- 正向/反向必须分开记录：反向用 `--directions backward`，脚本会把机身 yaw 设为 π、命令 vx 取负，让后背朝 +x 方向上台阶。
- 原生测试使用 CPU，不占训练 GPU。

## 第一版记录（2026-11-17）

`test/dig_dog/record_test.py` 已完成，支持：

```bash
cd /home/rmer/Dog_deploy-main/Dog_deploy-main/test/dig_dog
python record_test.py \
  --checkpoint /media/rmer/data/rpwd_training/digdog_mjlab/2026-11-16_大狗DigDogv1点足版_60k_seed0_v2/model_60000.pt \
  --lanes stairs10 stairs12 --directions forward backward --seconds 50 --seeds 0
```

其他三条大狗按同样的目录/字段预留，后续补各自的 `record_test.py` 即可。
