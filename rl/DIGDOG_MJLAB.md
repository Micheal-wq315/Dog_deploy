# 大狗 DigDogv1：MuJoCo SuperMimic 过台阶适配

日期：2026-11-15。分支 `codex/rpwd-digdogv1-mjlab`，工作区
`/media/rmer/data/rpwd_training/worktrees/rpwd-digdogv1-mjlab`。

来源为 `Stackforce-simready-dig_dogv1-mjlab.zip`（MJLab 1.6.0 包），原始
XML/URDF/网格解压在 `/media/rmer/data/rpwd_training/references/dig_dogv1_20261115`。
本适配复用点足 `rpwd_pointfoot_supermimic` 世界模型、想象 BPTT、TD(λ) Critic、
台阶课程与验收结构，新建 `rpwd_digdog_supermimic`，不修改原点足分支。

## 机器人与坐标

- 质量 **3.495kg**；机身碰撞盒约 **0.2015 × 0.316 × 0.122m**；12 个腿关节，
  KFE 连杆为脚端。
- 原始 MJCF 是 CAD 坐标：X=左右、Y=上、Z=前后。`build_digdog_mjcf.py` 在
  `worldbody` 外套一个带 freejoint 的 `base_link` 根节点，并用四元数
  **`(0.5,0.5,0.5,0.5)`** 旋转旧模型，得到框架统一的 **X 前 / Y 左 / Z 上**。
  旧 `base_link` 改名 `body_link`，作为根节点的刚性孩子。
- 关节/脚端统一为 `FL/FR/RL/RR + hip/thigh/calf`（后右 BR→RR、后左 BL→RL），
  与点足框架的 `LEG_JOINT_NAMES`、动作顺序和接触传感器命名一致。
- 生成命令：

```bash
python3 rl/scripts/build_digdog_mjcf.py \
  /path/to/assets/robot.xml description/dig_dogv1/robot.xml
```

## 站姿、执行器与步态

- 额定站高 **0.16m**（固定足跟接触点相对 `base_link`），初始高度 **0.1692m**；
  膝角约 **0.33rad**。0.18m 站在 0.2m/s 时参考 IK 已撞关节限位（4–14 个采样点、
  9mm 误差）；0.16m 在 0.2m/s 下无限位截断、IK 误差为 0，6cm 抬脚也可达。
- 脚端是弧形小腿，不能用 KFE 原点或零位包围盒中心；固定接触点取额定站姿下
  足跟接触区质心（KFE 局部坐标，见 `robot/config.py:FOOT_LOCAL`），等效脚半径
  **0.0084m**。
- XML 位置执行器 **kp20 / kv0.5 / ±10N·m**；动作尺度 **0.5**，关节电枢 0.01，
  soft limit 0.95。
- 步态 **1.25Hz**，duty **0.6**，抬脚净空 **0.06m**，FL/RR 与 FR/RL 对角交替。
- 广义解析 IK：roll 绕 X、hip/knee 绕 Y，带每腿固定偏移和方向符号；Actor 仍为
  **235→12**（47×5 帧），世界模型 340 维输入不变。
- 奖励高度区间 **0.14–0.20m**（rough 0.12–0.22），非足端碰撞终止阈值 15N。


## 2026-11-16 关键修复（第二轮训练）

- 原生测试相位仍硬编码 1.5Hz，与训练 1.25Hz 不一致，已改读
  `kinematics.FREQUENCY`（提交 `469eabb`）；首次 5k/10k/15k CPU 抽查因此无效。
- 站高 0.18→**0.16m**、初始高度 0.189→**0.1692m**、抬脚净空 0.05→**0.06m**；
  参考 IK 扫描显示 0.18m/0.2m/s 有 4–14 个限位截断，0.16m 为 0。
- 台阶通过窗口原先按点足 0.34m 站高写死 `top+.23~+.46`，对 0.16m 站高不可达；
  改为 `top+STAND_HEIGHT-.10 ~ +.12`，4/12cm 上阶和 4cm 下阶合成检查通过。
- 新增直行 `yaw_rate_l2(-1)`、`lateral_vel_l2(-.5)` 和上阶 `stair_progress(1.0)`
  奖励；`track_angular_velocity` 权重 1→2。
- 速度课程按腿长缩放：stage0 `vx -.15~.25 / vy ±.1 / wz ±.4`，最高 stage4
  `vx -.35~.45 / vy ±.2 / wz ±.8`；步频门槛 0.8–2.5→**0.9–1.9Hz**。
- 第一轮 2026-11-15 22:44 启动，停在 model_19500；第二轮 2026-11-16 07:31
  从零启动，服务 `rpwd-digdog-mjlab-60k-20261116.service`，运行目录
  `/media/rmer/data/rpwd_training/digdog_mjlab/2026-11-16_大狗DigDogv1点足版_60k_seed0_v2/`。

## 训练

```bash
cd /media/rmer/data/rpwd_training/worktrees/rpwd-digdogv1-mjlab
# 默认 4096 环境、60000 轮、seed0；1.25Hz 已固化在步态模块
./rl/scripts/train_digdog_mjlab.sh
# 从本机器人检查点续训（不能加载点足/轮足 checkpoint）
./rl/scripts/train_digdog_mjlab.sh --resume /绝对路径/model_xxxxx.pt --iterations 60000
```

第一轮（2026-11-15 22:44 启动）已停止，停在 `model_19500`，其 0/256 上阶通过
不能作为能力结论（台阶窗口写死）。第二轮修复后于 2026-11-16 07:31 从零启动：
服务 `rpwd-digdog-mjlab-60k-20261116.service`，运行目录
`/media/rmer/data/rpwd_training/digdog_mjlab/2026-11-16_大狗DigDogv1点足版_60k_seed0_v2/`，
stdout `train_stdout.log`，指标 `metrics.jsonl`。4096 环境、60000 轮、seed0；
2048 世界模型批量×16 步、2048 想象批量×48 步、300 轮预热、每 500 轮保存、
每 5000 轮 CPU 确定性抽查。启动检查以 metrics.jsonl 为准；
**这不等于收敛或上阶能力**。

## 测试

```bash
# 原生 MuJoCo CPU 评测；训练内每5000轮抽查也走这个入口
./rl/scripts/play_digdog_mjlab.sh --mode evaluate \
  --checkpoint /media/rmer/data/rpwd_training/digdog_mjlab/<run>/model_05000.pt \
  --lanes flat stairs4 stairs8 stairs12 --speed .2 --seconds 55 --seeds 0 1 2
```

已验证：修复后 0.16m 站姿的 12 轮短模型平地 5 秒测试完成（前进约 1.53m、
均速约 0.206m/s、无翻倒/非足端碰撞）；4096 环境 2 轮冒烟显存峰值约 3.2GB；
8 环境 12 轮更新冒烟中新奖励项均有限、0 跳过更新、0 无效复位；步态扫描在
0.16m/0.2m/s 下零限位截断、实际脚网格抬升 6cm；台阶窗口合成检查通过。
第一轮 model_19000 在修正 1.25Hz 时钟后仍是平地 16.4s/3.11m 出界、4cm 止于
第一级、8cm 摔倒，因此不能把首轮问题归因于评估时钟；修复后的第二轮台阶通过率、
12cm 能力和 ONNX 导出尚待训练后验证。

## 边界

- 原始 MJCF 不能直接当 Z-up 使用；任何重新导出也必须经过包装旋转。
- 改变站高/脚点会同时影响 `DEFAULT_Q`、`INIT_HEIGHT`、`FOOT_LOCAL` 和
  `profile.height_band_*`，改后应重新跑原生平地与台阶短测。
- `robot_id=rpwd_digdog`，检查点格式 `rpwd-digdog-*`；禁止与
  `rpwd_pointfoot` / Go2W / 旧 Isaac 检查点交叉加载。
