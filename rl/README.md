# RP_wd 强化学习训练工程

用 Isaac Lab (2.3.2 + Isaac Sim 5.1) + RSL-RL PPO 训练 RP_wd 四轮足机器人的速度跟踪策略，
产出的 ONNX 直接对接本仓库的部署端推理节点 (`src/inference/robots/wheel_quad`)。

```
        训练侧 (本目录)                                部署侧 (src/inference)
┌──────────────────────────────┐              ┌──────────────────────────────────┐
│ Isaac Lab 2.3.2              │              │ roboparty_inference (ROS2)       │
│  RP_wd USD (description/)    │  ONNX 168→16 │  obs 拼接 56×3 → ONNX → 动作后处理 │
│  PPO (rsl-rl) → policy.onnx  │ ───────────► │  → MIT 指令 (CAN) → 电机          │
└──────────────────────────────┘              └──────────────────────────────────┘
        rl/exported/policy_flat.onnx          src/inference/robots/wheel_quad/models/
```

## 目录结构

```
rl/
├── pyproject.toml                     # pip install -e ./rl
├── rp_wd_rl/
│   ├── assets/
│   │   ├── rp_wd_contract.py          # ★ sim2real 契约常量 (纯 Python, 不依赖 isaaclab)
│   │   └── rp_wd.py                   # ArticulationCfg (USD/驱动器/默认姿态)
│   └── tasks/locomotion/velocity/
│       ├── rp_wd_env_cfg.py           # ★ 环境配置 (观测/动作/奖励/域随机化/终止)
│       ├── mdp/{rewards,observations,events}.py   # 轮足专用奖励 + 传感器零偏 DR
│       └── agents/rsl_rl_ppo_cfg.py   # PPO 超参
└── scripts/
    ├── verify_contract.py             # ★ 契约复核 (训练侧 ↔ 部署侧)
    ├── train.py / train.sh            # 训练
    └── play.py / play.sh              # 回放 + 导出 ONNX
```

## 环境准备

```bash
conda activate isaaclab                       # torch 2.11+cu128 / isaaclab 2.3.2 / rsl-rl-lib 5.3.0
pip install -e ./rl                           # 只需一次

# Isaac Sim 在 G 盘(新加卷): /media/rmer/新加卷/isaac-sim-5.1.0
# Isaac Lab 已用软链指向它:    /media/rmer/data/IsaacLab/_isaac_sim -> 上述路径
export ISAACLAB_PATH=/media/rmer/data/IsaacLab
```

## 使用流程

```bash
# 0) 契约复核 (改过观测/动作/奖励/部署 YAML 后都该跑; ~1 分钟)
./rl/scripts/verify_contract.sh

# 1) 训练 (默认 2048 环境 / 3000 迭代 / headless)
./rl/scripts/train.sh
./rl/scripts/train.sh --num_envs 4096 --max_iterations 5000 --run_name dr_v1

# Odin1 深度避障策略 (独立 profile; headless 相机渲染; 默认256环境 / 8000教师+2000学生，共10000轮)
./rl/scripts/train_odin_pipeline.sh --num_envs 256 --seed 42

# 2) 回放 (需要图形界面) + 导出 ONNX
./rl/scripts/play.sh
./rl/scripts/play.sh --headless --video --video_length 300   # 无显示器: 录视频
# 普通策略: policy_flat.onnx (168 → 16); 深度策略使用 --task Isaac-Velocity-Rough-RPWD-OdinDepth-Play-v0
# 深度策略用 --task Isaac-Velocity-Rough-RPWD-OdinDepth-Play-v0 回放并导出 1740 → 16 的模型

# 3) 部署 (单文件, 直接覆盖即可)
cp rl/exported/policy_flat.onnx src/inference/robots/wheel_quad/models/policy_flat.onnx
```

> 导出说明: torch 2.11 的导出器默认把权重写成 `policy.onnx.data`，脚本会自动合并成**单文件**
> 并删除中间产物 —— 部署端只会拷这一个 `.onnx`，依赖 `.data` 会在实机上加载失败。

训练日志: `rl/logs/rsl_rl/<experiment_name>/<时间戳>_<run_name>/` (TensorBoard 可看)。
续训: `./rl/scripts/train.sh --resume`。

**无窗口说明**：训练 (`train.py`) 与契约校验 (`verify_contract.py`) 在脚本内**强制 headless**，
不管加不加 `--headless`、从哪个入口启动都不会弹 Isaac Sim 窗口（也就不需要显示器），
录制训练视频走离屏渲染。只有 `play.sh` 默认开窗口用于肉眼观察，无显示器时用
`./rl/scripts/play.sh --headless --video --video_length 300`。
脚本结束时若 Isaac Sim 关闭卡住，会在 5 秒后自动强制退出（见 `scripts/sim_shutdown.py`），
不必手动 Ctrl-C。

## ★ sim2real 契约表

**两端必须逐字一致**。本表由 `verify_contract.py` 自动核对，改动请同步改
`rl/rp_wd_rl/assets/rp_wd_contract.py` 与部署端 YAML/C++。

| 项目 | 训练侧 (Isaac Lab) | 部署侧 (wheel_quad) |
|---|---|---|
| 物理步长 / 降采样 | `sim.dt=0.0025`, `decimation=8` | `dt: 0.0025`, `decimation: 8` |
| 策略频率 | 50 Hz | 50 Hz (`inference_node.cpp:516`) |
| 普通策略观测（单帧 56 维） | `ang_vel(3) + projected_gravity(3) + cmd(3) + joint_pos(12,腿) + joint_vel(16) + last_action(16) + cmd_posture(3)` | `configs/default.yaml` 中的观测布局 |
| 帧栈 | `history_length=3`, term-major, 每分量旧→新 | `frame_stacks:[3]`, `obs_stack_orders:["obs_major"]` |
| 普通策略 ONNX 输入/输出 | 168 / 16 (`obs` / `actions`) | 168 / 16 |
| 关节位置观测 | 相对默认姿态 (`joint_pos_rel`) | 读值 − `joint_default_angle` |
| 关节速度观测 | 全部 16 关节（含轮） | `dof_vel:16`（含轮） |
| `last_action` | **原始**动作（未缩放/未偏置，经 `clip_actions` 裁剪） | ONNX 原始输出（裁剪后） |
| 位置动作 | 腿 12 关节，`scale=0.125/0.25/0.25`，offset=默认姿态 | `act = raw×scale + joint_default_angle` |
| 速度动作 | 轮 4 关节，`scale=5.0`，offset=0 | `act_vel = raw×5.0`（`kp=0` 纯阻尼） |
| 动作裁剪 | `clip_actions=10.0`（RslRlVecEnvWrapper，作用于原始动作） | `clip_actions: 10.0` |
| PD 增益 | 腿 `kp=100 kd=3`；轮 `kp=0 kd=3` | `robot.yaml` 的 `kp`/`kd` |
| 力矩限幅 | 腿 42 N·m；轮 17 N·m | — |
| 站立默认姿态 | ABAD 0 / HIP 0.72 / KENN -1.41 | `joint_default_angle` |
| 速度指令 | vx ±0.6 vy ±0.6 ωz ±1.0 | `clip_cmd` |
| 姿态/高度指令 | pitch ±10° roll ±10° height ±7 cm | `clip_posture` |
| 倾角保护 (2026-10-11 提速版) | `bad_orientation` 60° | `gravity_z_upper: -0.5`（acos(0.5)=60°）→ 急停 |
| 关节顺序 | articulation 顺序 = `[ABAD×4, HIP×4, KENN×4, FOOT×4]`（组内 LB/LF/RB/RF） | `usd2urdf: [8,0,12,4,…]`、`wheel_joint_indices: [12,13,14,15]` |

### Odin1 深度避障策略

深度输入使用独立任务、ONNX 和部署 YAML，不覆盖普通 168→16 策略：

| 项目 | 配置 |
|---|---|
| ONNX | `policy_odin_depth.onnx`，输入 1740 (=68×3 + 8×16×12)，输出仍为 16 |
| 本体观测 | 腿部力矩 12 维（仿真/实机均按 42 N·m 归一化），随其余 56 维观测堆叠3帧 |
| 深度记忆 | 相机新帧到达时更新22帧缓冲，间隔3帧抽8张（约1.4秒，旧→新），CNN输入为 `[8,16,12]` |
| 图像方向/网格 | 原始160×120横图先逆时针旋转90°，生成16行×12列竖向最近距离网格，特征归一化到0..1 |
| 深度增强 | 学生渐入3cm噪声、1%缺测、回合固定95–100%裁切、偏差/尺度、连片遮挡、8%丢帧、0–2帧延迟；仅新图更新历史 |
| 深度语义 | 仿真使用 `distance_to_image_plane`（光轴Z深度），与ROS深度图的米制/毫米制值对齐；驱动语义仍需实测确认 |
| 仿真相机 | 前向 160×120、约 120°×90°、0.2–6 m；安装位姿是待实测标定的初始估值 |
| 障碍课程 | 实体上下4级台阶、矮障碍、45cm绕行墙、起伏/坡道＋平地自由控制；安全到点升级，物理随机化沿用rough |
| 力矩处理 | 无电机力矩惩罚；12维腿部力矩只作为策略观测 |
| 任务 ID | `Isaac-Velocity-Rough-RPWD-OdinDepth-v0` / `...-OdinDepth-Play-v0` |
| ROS2 部署 | Odin1 官方深度图 → `odin_depth_adapter.launch.py` → `/odin1/depth_features` → `policy:=odin_depth` |

v18修复跨回合终点/碰撞状态重置，新增无效位置目标约束（非力矩惩罚），地形从单级小台阶起步；1000/3000轮加入阶段验收，达标才继续。设计与失败证据见`CURRICULUM.md`。

一键 headless 训练：`./rl/scripts/train_odin_pipeline.sh --num_envs 256 --seed 42`。
用 `./rl/scripts/play.sh --task Isaac-Velocity-Rough-RPWD-OdinDepth-Play-v0 --checkpoint <path>`
做仿真回放并导出专属 ONNX。策略使用独立 CNN 编码8帧深度记忆；
导出时会把本体观测和深度历史展平成部署端单输入 ONNX。

推荐8000轮教师PPO＋2000轮学生在线蒸馏；教师、学生独立日志，test/play都能读取。完整的10000轮课程、目标奖励、接触约束和续训规则见 [CURRICULUM.md](CURRICULUM.md)。

**最容易写错的两点**（都已在校验脚本里逐项检查）：

1. **关节顺序**：USD/Isaac Lab 是"按关节类型分组"，而 URDF/CAN 是"按腿分组"，
   两者用 `usd2urdf` 映射。实测确认 Isaac Lab 报告的顺序与部署端假设一致。
2. **帧栈排布**：term-major，每个分量的 3 帧按"旧→新"排列（最新在最后），
   且首帧是**复制当前观测**（`last_action` 首帧为全零）。写成"按帧拼接"或"新→旧"都会让策略在实机上失效。

## 观测/动作的具体来源

- 观测里**没有** `base_lin_vel`、`height_scan`：实机没有对应传感器（无全局定位、无高程图）。
- 轮子不参与 `dof_pos`（12 维），但参与 `dof_vel`（16 维）—— 与部署端 `dof_pos:12 / dof_vel:16` 一致。
- 轮子用速度控制（`JointVelocityAction`），因此不叠加默认角度。

## 域随机化（训练侧）

| 类别 | 范围 | 说明 |
|---|---|---|
| 地面/机身摩擦 | 静 0.6–1.4，动 0.4–1.2，恢复 0–0.1 | startup，按刚体分桶 |
| 机身质量 | +(-1.5, 2.5) kg | 负载（云台/电池差异） |
| 质心 | x±3 cm, y±2 cm, z±1 cm | |
| 电机增益 kp/kd | ×(0.7, 1.3) | 电压/温度导致的增益漂移 |
| 关节摩擦 / 电枢 | 0–0.08 N·m / 0–0.03 | 减速箱阻力与转子惯量 |
| **IMU 陀螺零偏** | ±0.05 rad/s，每回合常值 | 部署端**不做零偏标定** |
| **关节零点误差** | ±0.015 rad，每回合常值 | 编码器标定残差 |
| 观测噪声 | 各分量不同（见 `rp_wd_env_cfg.py`） | 训练开、Play 关 |
| 推力扰动 | 每 8–12 s 给 0.6 m/s 随机速度 | |
| 初始状态 | 位姿/速度/关节小幅随机 | 每回合 |

## 调参提示

- **奖励配比是第一优先级**：跟踪奖励必须明显大于各项惩罚之和，否则策略会选"站着不动"
  甚至"主动摔倒"（提前结束回合来少扣分）。判据：跑 Play 环境统计各 `Episode_Reward/*`，
  跟踪两项之和应至少是惩罚总和的数倍。改任何惩罚权重后都建议重新做一次这个检查。
- **必须奖励"滚动"**：只给速度跟踪奖励时，策略会收敛到"用轮子当脚跑小步"的
  退化解（实测 iter 1400：平均着地轮数 2.36/4、轮速只有应有值的 1/3）。
  已用 `wheel_rolling_slip`（轮速×半径 与 轮心速度的失配）+ `all_wheels_contact`
  两项把解拉回滚动。**改奖励后务必用 `./rl/scripts/test_policy.sh --sweep` 复核**
  着地轮数与滑移比。
- 陷阱：`isaaclab.envs.mdp.desired_contacts` 在 2.3.2 里的语义是"**所有**列出的刚体都没接触
  时返回 1"（不是接触比例），当正向奖励用会变成奖励离地。本工程用自定义的
  `mdp.wheels_in_contact_frac` 取着地比例。
- 学不会站立/行走：先把 `cmd` 范围减半、`rel_standing_envs` 调 0.2 起步；
- 轮子"溜冰"打滑：增大 `stand_still_wheel_vel` 权重或减小 `wheel_vel_l2`（注意它是负奖励）；
- 实机抖动/振荡：加大 `action_rate_l2`、`dof_acc_l2` 惩罚；确认 `act_alpha`（部署端 EMA 0.9）与 `kd`；
- 实机转向迟钝：提高 `track_ang_vel_z_exp` 权重；
- 训练速度实测（RTX 5070 Ti 16 GB，默认 2048 环境）：**约 0.86 s/迭代**
  → 1000 迭代 ≈ 15 分钟，3000 迭代 ≈ 43 分钟（含域随机化的平地收敛通常 3000–6000 迭代）。
  显存无压力；想更快可 `--num_envs 4096`。

## 已知待确认项 (TODO)

- [ ] **轮速上限**：`VELOCITY_LIMIT_SIM["FOOT"]=50.0` 取自部署端命令限幅，不是 RS02 手册值；
      拿到电机铭牌转速后改成实测值。
- [ ] **轮力矩**：17 N·m 为 RS02 峰值，连续工作建议按额定 6 N·m 复核散热。
- [ ] 部署端 `act_alpha=0.9` 的 EMA 在 400 Hz 控制线程上时间常数约 1.1 ms，训练侧未建模（影响可忽略）。
- [ ] 关节零点误差目前只加在观测上（近似）；若要更严格，需要同时偏移动作目标。
- [ ] Odin真实深度朝向/外参/深度语义标定；新课程完整训练与固定赛道多种子验收。粗糙地形与深度课程已经接入。
```
