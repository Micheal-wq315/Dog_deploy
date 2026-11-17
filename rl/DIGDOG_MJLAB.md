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

截至 2026-11-16 12:13（step 10050/60000，4.66 小时）：0 跳过、0 无效复位；
stage 0→1 在 step 4800（4cm），stage 1→2 在 step 8050（6cm）；当前 stage 2（8cm）
最近 **198 次考核 121 次通过（61.1%，已达 60% 门槛）**。stage 3 需要等课程下限
step 15000（`MIN_ITERATIONS`）才正式解锁，当前 command_stage 仍为 2；
步态约 1.4Hz，平均地形等级 1.92。

独立 CPU 抽查：
- `check_5000`：平地 **completed**、4cm **completed**、8cm stuck（停在第一级前）。
- `check_10000`：平地 timeout 但 max_x 10.21（终点 7.5，已越过）、4cm **completed**、
  8cm timeout 但 max_x 9.81（终点 7.5，已越过）。平地/8cm 是到达/越过终点后
  成功标志未触发（稳定通过条件对当前步态偏严），不是没走完；判据需后续复核。
- 10cm/12cm 和最终验收仍待训练后确认。

## 仓库当前的大狗（四条）

| 机器人 | 训练包 | 接口 | 当前状态 |
|---|---|---|---|
| RP_wd 大狗轮足版 | `rl/rp_wd_rl`（Isaac Lab）/ `rl/rpwd_go2w_supermimic --robot rpwd`（MuJoCo） | 168→16 / 245→16 | 原部署机器人，16关节（12腿+4轮）；PPO/Odin/SuperMimic链路保留；MuJoCo迁移已接入 |
| Go2W 轮足参考 | `rl/rpwd_go2w_supermimic --robot go2w` | 245→16 | 原版参考/原生测试链路保留 |
| RP_wd 大狗点足版 | `rl/rpwd_pointfoot_supermimic` | 235→12 | 17.038kg、1.5Hz；60k训练因重启停在约57600，55k抽查平地/4/8cm通过，12cm完整赛道未过 |
| DigDogv1 大狗点足版 | `rl/rpwd_digdog_supermimic` | 235→12 | 3.495kg、1.25Hz；第二轮训练中，4/6cm已晋级，8cm训练通过率61.1%（stage3按课程下限15000轮解锁） |


## 测试

```bash
# 原生 MuJoCo CPU 评测；训练内每5000轮抽查也走这个入口
./rl/scripts/play_digdog_mjlab.sh --mode evaluate \
  --checkpoint /media/rmer/data/rpwd_training/digdog_mjlab/<run>/model_05000.pt \
  --lanes flat stairs4 stairs8 stairs12 --speed .2 --seconds 55 --seeds 0 1 2
```

全高度键盘测试（4/6/8/10/12/16cm，自动取最新模型）：

```bash
./rl/scripts/play_digdog_stairs.sh
```

键位：1=flat、2=4cm、3=6cm、4=8cm、5=10cm、6=12cm、7=16cm、8=ramp；方向键/横移、Q/E 转向、Space 停止、R 复位、K 暂停、Esc 退出。
测试场从 x=0.5 出生、台阶从 x=2.2 开始，5 秒短测到不了台阶；至少留 30–50 秒或直接用键盘走过去。

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

## 2026-11-17 正向退化回归检查

用 `test/dig_dog/record_test.py` 对 model_20000~60000 做正向/反向 stairs12：
model_20000 正 9.99/反 9.75，model_30000 正 9.42/反 2.37（摔倒），
model_40000~60000 正向卡 2.3/反向 9.59~9.83。退化发生在 **30k→40k**。

定位：Actor LR 调度（20k 6e-5 → 30k 3.6e-5 → 40k+ 1.2e-5 地板）为主触发器；
前腿台阶 clearance 从 0.322/0.332 降到 0.224/0.224、前足 peak 111.6/93.1N
降到 63.6/64.0N，后腿保持，`swing_clearance` 奖励从 -0.0188 恶化到 -0.0339；
`stair_pass` 只看升段 + 双向地形使反向/下阶成为局部最优（up12 64.8% vs down12 89.9%）；
命令 stage4/地形等级 3.49→4.83 是放大因素。世界模型/critic/资产排除。

当前可用 **model_20000** 做正向/反向对照；下一版应从 model_20000 做
Actor LR 地板、命令上限、前腿抬脚和完整往返消融，不要从 60k 单调续训。

## 2026-11-17 Mid-360 局部高度图与教师-学生蒸馏

### 观测与契约

- **本体观测不变**：Actor 仍是 5 帧 × 47 = 235 维（无基座线速度、无力矩、无接触）。
- **Actor 新增 273 维局部高度图**：带高度图的 Actor 输入 **235 + 273 = 508**；
  Critic 仍为 331（36+3+7+12+273），世界模型仍为 340（36+7+12+12+273）。
- **教师/学生同结构**：`core/policy.py:ScanActor` 保留盲策略的 235→256→128→64 主干，
  另加 273→256→128→64 高度图编码器，拼接后输出 12 维残差；教师读真实
  `TerrainField.scan`，学生读 Mid-360 模拟扫描。
- **高度图网格**：与 `TerrainField.scan` 完全一致，**21×13=273**，机体 yaw 坐标，
  x∈[-1,1] m、y∈[-0.6,0.6] m，4cm 全局地图上双线性采样；正数表示该格地面低于机身，
  不是通用的“地面高度”，部署端必须保持同一符号约定。
- **未知格**：填 **-2.0**（低于真实扫描的 ±1.5 范围）；`scan_with_mask` 同时返回
  可见掩码，训练/记录只把高度值送进网络。

### Mid-360 模拟

`core/lidar.py:LidarHeightScan`，常量在 `robot/config.py:LIDAR`：

| 项 | 默认值 | 说明 |
|---|---|---|
| `mount` | `(0, 0, 0.07)` m | base_link 上方 7cm、x/y 居中；实物必须一致 |
| `tilt_deg` | `20`（下倾） | 有效机体垂直视场约 -27°~+32° |
| `v_min_deg/v_max_deg` | `-7/52` | Mid-360 原始垂直视场 |
| `range_min/range_max` | `0.15/40` m | 本地网格只需前 1.1m，远距不截断 |
| `update_hz` | `10` | 50Hz 控制下每 5 步刷新，其余步保持 |
| `noise_std/dropout` | `0.01m/5%` | 训练/带噪评测；确定性评测关闭随机项 |
| `occlusion_samples` | `6` | 高度场射线步进，被更高地形挡住的格标为未知 |

注意：0.16m 站高、雷达在 base 上方 7cm、下倾 20° 时，**机身附近约 0.4–0.5m 内是盲区**，
近场格会大量填入 -2.0。这是真实 Mid-360 的几何限制，不是 bug；教师能看到这些格，
学生必须靠本体历史/步态外推补足，正是蒸馏要学的部分。若实物把雷达装得更高/更前/
下倾更多，只改 `LIDAR`，训练和部署两端必须同步。

`LIDAR` 故意**不进入** `profile()/robot_signature`，所以旧盲策略 checkpoint 仍能加载；
带高度图 checkpoint 额外保存 `lidar` 和 `lidar_signature`，改安装/视场/噪声后禁止静默续训。

### 教师-学生训练

1. **教师（特权）**：`./rl/scripts/train_digdog_scan_teacher.sh`
   - 从 **model_20000** 热启动（`BASE=` 可换），Actor 主干和世界模型/Critic 原样恢复；
     高度图分支初始为零输出，第一步动作与旧盲策略逐元素一致（实测最大差 0.0）。
   - 默认 20000 轮，避开 30k→40k 的 Actor LR 地板/正向遗忘区间。
2. **学生（LiDAR）**：`./rl/scripts/train_digdog_lidar_student.sh`
   - 自动取最新教师 `model_*.pt`，或 `TEACHER=/绝对路径/model_xxxxx.pt` 指定。
   - 学生从教师完整热启动，教师冻结；学生动作驱动世界模型，损失为
     **RL 全回报 + `distill_coef` × BC(学生动作, 教师动作)**（默认 1.0）。
   - LiDAR 高度图在真实环境按 10Hz 保持，在想象 rollout 中也按 `lidar_hold=5` 保持；
     教师每个想象状态读真实扫描给出 BC 目标，学生只读 LiDAR 扫描。
3. **续训**：教师用 `--resume <teacher model> --scan true`；学生用
   `--resume <student model> --scan lidar --teacher <teacher>`，`--warm-start` 只用于新 run。

### 代码改动

- 新增 `core/lidar.py`、`ScanActor`、`Replay(scan=273)`、`Learner.warm_start/update_scan`。
- `core/imagine.py`：`make_imagined_step/rollout` 支持 `scan_actor`、教师 BC、LiDAR 保持；
  带高度图分支时禁用 `torch.compile`，走 eager。
- `core/terrain.py`：抽出 `TerrainField.from_model(..., bounds=...)`，原生测试场把
  plane 当作 z=0、box 作为障碍；训练环境的 `from_env` 行为不变。
- `rl/scripts/digdog_mujoco.py`：原生回放自动识别 `scan_mode`，教师读真实扫描、
  学生读确定性 LiDAR 扫描（保留视场/遮挡，不加随机噪声）；`record_test.py` 额外保存
  `height_scan.npy [N,273]`。

### 已验证（2026-11-17）

- `ScanActor.from_blind(model_20000)` 的初始输出与旧盲策略**逐元素 0.0 差**；
  高度图分支首步只更新末层，第二步起编码器/输出头均获得非零梯度，未出现零初始化死分支。
- 8 环境 CPU 烟测：`--scan true` 教师、`--scan lidar` + 冻结教师学生均 3 轮完成，
  WM/actor/critic loss 有限、0 跳过/0 无效复位；LiDAR 学生的 replay 增加 273 维扫描。
- 原生 MuJoCo 平地短测可加载教师/学生 checkpoint；`record_test.py` 输出
  `height_scan.npy (50,273)`、`observations.npy (50,235)`、`scan_mode=lidar`，
  扫描值范围 -2.0~0.17。
- 以上是**链路烟测**，不是能力结论；正式教师/学生训练已在 3072 环境启动（见下节，
  4096 会因 MuJoCo Warp 静态 geom 缓冲丢地面接触），12/16cm 与真实 Mid-360 部署仍待验证。

### 待办 / 风险

1. **实机安装外参**：确认 Mid-360 的实际 mount/tilt；如果装在头/背且倾角不同，
   先改 `LIDAR` 再训练，不要拿默认 7cm/20° 的模型直接上真机。
2. **真实高度图节点**：ROS2 侧需要订阅 Livox CustomMsg/PointCloud2，按 base TF 聚合到
   21×13 局部网格、10Hz 发布，未知格 -2.0；ONNX 输入变为 508，部署 YAML 同步；
   改完必须跑 `./rl/scripts/verify_contract.sh`。当前 `src/inference` 是轮足部署链路，
   DigDog 的 ONNX/ROS2 推理节点仍需新建。
3. **近场盲区**：如果实机近场任务（窄台阶、轮胎）依赖 0.5m 内高度，需提高安装高度/
   下倾角、或加入里程计累积的局部地图；当前单帧高度图不解决。
4. **教师回归**：教师仍沿用旧奖励/课程，30k 后可能重复正向 12cm 退化；默认只训 20k，
   学生选教师时应结合 `record_test.py` 的正/反向 `max_x`，不要只看最后一步。
5. **蒸馏权重**：`distill_coef=1.0` 是初值；若学生只会模仿教师、RL 跟踪变差，先降到
   0.2–0.5；若 LiDAR 噪声下动作抖动，先加大 `noise_std/dropout` 或教师数据覆盖。

## 2026-11-17 全地形扩展（stones/tires/16cm）与 3072 环境约束

### 地形与课程

- `STEP_HEIGHTS` 扩到 **8 档**：0.02–0.16m（新增 14/16cm）；`CourseCfg` 的 level 按 0..7 映射。
- **8 类地形**：`flat/up/down/platform/slope/rubble/stones/tires`，比例
  `.28/.32/.14/.04/.04/.08/.05/.05`。
- `stones`：交错箱体梅花桩，高度 2–16cm；`tires`：横放圆柱（轴沿车道横向），
  半径 5.5–13cm，宽度 9–13cm。训练用两者只铺**中央 4.8m 通道**（测试场仍全宽），
  控制 4096 级并行下的 geom 总数。
- `stair_curriculum` 同步为 8 类/8 档：`KINDS`、`HEIGHTS`、stats `(8,8,4)`、
  最近窗口 6 档、探针/晋级/报告全部按 8 档。
- 20k 教师课程下限改为 `(0,2000,5000,8000,11000,13000,15000)`：第 6 段（14cm 考核）
  在 15000 轮解锁，留 5000 轮训练 16cm；旧表 34000 轮在 20k 内到不了 16cm。
- 地形集合从 6 类扩到 8 类后，热启动只搬权重；旧 6×6×4 terrain/curriculum 统计与新
  8×8×4 不兼容，`Learner.warm_start` 重置 `terrain_state/curriculum_state`（同地形
  `--resume` 不受影响）。

### 关键坑：4096 环境丢失地面接触

- **现象**：8 类地形 4096 环境训练初始 reward 从 -8 迅速恶化到 -2e6，
  `weighted_reward_terms.base_height` 从 -399 到 -3.6e6，机器人整体自由落体；
  `ncollision=0`、`wheel_ground_contact=0`。64 环境 GPU、32 环境 CPU 均正常。
- **定位**：新地形模型有 **6958 个 geom**。MuJoCo Warp 的静态 geom 总缓冲约 25M
  （4096×6144）；`4096×6958=28.5M` 超限后 broadphase/narrowphase **静默不产地面接触**。
  对照：旧 6 类 4484×4096=18.4M 正常；6 类 8 档 5982×4096=24.5M 正常；
  7 类 6222×4096=25.5M 失效。`nconmax=512` 会在 CCD EPA 申请 6.4GB 直接 OOM；
  禁用 CCD、关闭 CUDA graph、切换 `sap_segmented` 均不能修复。
- **处理**：新的教师/学生脚本固定 `--num-envs 3072`（3072×6958=21.4M），
  保持 8 类地形完整；4096 仅适用于旧 6 类地形。
- **验证**：3072 零动作 10 步 `wheel_ground_contact=13587`、root z 0.178m；
  正式教师 step 50 reward 0.079、`base_height≈0`、`wm_loss=0.31`，
  8 类地形各 124–981 个环境（stones/tires 各 154）。

### 测试场与脚本

- `go2w_test_maps.py` 的 `LANES` 扩到 **13 条**：flat/stairs4/6/8/10/12/16/ramp/
  rubble/mixed/slalom/stones/tires；测试场 plane 扩为 18×30m、`stat.extent=30`。
  MuJoCo `mj_ray` 只覆盖 plane 的 size 半宽；原 24m 半宽下 y=24 的 tires lane
  一移动就 `nonfinite`。
- `digdog_mujoco.py` 键位 1–8 = LANES 前 8 条，9=stones，0=tires；查看器图例同步。
- `play_digdog_stairs.sh` 自动取最新 `model_*.pt`（跨 run）。
- `core/terrain.py` 的 `TerrainField.from_model` 支持圆柱高度场：按圆柱轴向求垂直射线
  与圆柱面的交点，取顶面；训练环境与原生测试共用。
- 原生验证：tires lane 3s 到 x=1.32m、无 nonfinite；stones 同样；两条 lane 的障碍从
  x=2.2 开始，短测不作为通过能力结论。

### 训练

- 教师 run：`/media/rmer/data/rpwd_training/digdog_mjlab/2026-11-17_23-22-42_digdog_scan_teacher`，
  3072 环境、20000 轮、seed0、从 `model_20000` 热启动；step 50 指标已恢复。
- 16cm 训练要等 stage 6（≥15000 轮）；当前只是训练配置，不是能力结论；教师/学生完成后
  用 `record_test.py`/原生评测做多 lane 回归。
