# 感知与落脚地形设计（草稿）：先点足大狗验证，再迁 DigDogv1

> 状态：点足大狗 P1/P2 已实施并在 3072 环境训练中；DigDog P3 待迁移。  
> 目标：训练真正的**感知 + 落脚准确性**——梅花桩细高桩阵、平躺轮胎紧密阵列；不是平地上的矮方块/稀疏横放圆柱。

## 0. 结论先行

1. **先在 RP_wd 点足（约 17kg、站高约 0.34m）验证地形/奖励/落脚**，再迁 DigDogv1（3.495kg、0.16m、1.25Hz）。
2. 旧的 DigDog `model_20000`（60k run）是 10/12cm 基线；新 20k 教师（8 类错误地形）**弃用**；自动起的 LiDAR 学生已停。
3. 两种地形：
   - **梅花桩**：细高、顶端小的竖直桩，桩顶高低不一，桩间是深空隙/地面；机器人必须踩桩顶，靠高度图/深度选落脚。
   - **轮胎**：多个**平躺圆环胎**（胎孔可见）紧紧挨着/叠成一簇；机器人在胎面走，踩进胎孔/空隙算失败。
4. 训练场、测试场、键盘测试使用同一套几何；true scan（教师）和 Mid-360 高度图（学生）必须能看到桩顶/胎面。
5. 台阶课程必须保留；每次晋级/存档先跑原生 10/12/14/16cm，**10/12cm 不过不许启学生**；教师/学生都以“台阶 + 新地形”联合门禁选模。

## 1. 现状与问题

- 我做错的两类地形：
  - `stones`：平地上的 2–16cm 矮方块，间隙是平地，机器人可以在平地上绕/踩，感知和落脚没有约束。
  - `tires`：稀疏、轴横向的圆柱（像立着/横放的轮子），不是平躺紧挨的胎环。
- 10/12cm 回归（DigDog 新 20k 教师，原生测试）：
  - stairs12：`stuck_8s`，max_x 2.30m
  - stairs16：`stuck_8s`，max_x 2.21m
  - stairs10：max_x 5.29m，未完成
  - 旧 `model_20000` 本来能上 12cm，新教师被 8 类错误地形/20k 训练训坏。
- 当前 GPU 已释放，自动接力不再启动；键盘测试先用旧 DigDog 模型。

## 2. 为什么先在点足大狗验证

- 17kg、站高约 0.34m，抗扰/承载余量大，桩顶/胎环的落脚误差容忍度更高；先验证地形、奖励、落脚、感知，成本低。
- DigDog 3.5kg、站高 0.16m、步频 1.25Hz，对桩顶尺寸/间隙/胎孔更敏感；等大狗跑通后再缩放参数、接 Mid-360。
- 两套包：`rl/rpwd_pointfoot_supermimic`（大狗点足）与 `rl/rpwd_digdog_supermimic`（DigDog）；地形/感知/奖励尽量抽成共享模块或同源实现，按机器人参数化。

## 3. 地形设计

### 3.1 梅花桩

- 结构：竖直方桩/圆柱，**桩顶小、桩身高、桩间是深空隙**；桩顶高度随机（课程分档），机器人只能踩桩顶。
- 训练场：
  - 中央通道铺 2D 桩阵（约 8–10 列 × 4–5 行），桩顶高度按课程 0.25–0.45m（DigDog）/0.35–0.55m（大狗）；
  - 起点加**出发平台**，机器人站在平台顶；平台前是第一根桩；
  - 桩间是低地面/坑；掉下桩、脚踩空、机身/腿撞桩都算失败；
  - 高度场用桩的 box 顶面，`TerrainField` 采样能看到桩顶和空隙；true scan / LiDAR 共用。
- 测试场：与训练同几何的 `stones` lane；起点、桩阵、目标点；键盘测试 9 键。
- 落脚：绿色落脚点对应“脚落在桩顶”的奖励；脚在空隙/低地面、机身低于桩顶、非轮碰撞 → 终止/大惩罚。

### 3.2 轮胎

- 结构：**平躺圆环胎**近似为 8 段 box 围成的圆环：外径 D、胎孔 d、胎宽/高 w；胎孔保留为可踩空的洞。段数从 12 降到 8 是为把每环境静态 geom 压到 Warp 缓冲以内（视觉仍是带孔圆环）；`yaw=ang` 让径向厚度用 `seg=(D-d)/2`、切向弦长用 `2π·rmid/8`。
- 训练场：
  - 轮胎中心距约等于外径（**紧挨/轻微重叠**），铺成 2–3 行 × 6–8 列的胎阵；机器人从胎面走；
  - 胎面孔隙是失败/惩罚点；胎面是落脚支撑；
  - 高度场是各段 box 的顶面；true scan / LiDAR 能看到胎面与孔。
- 测试场：同几何的 `tires` lane；键盘 0 键。
- 落脚：脚落在胎面环上 → 奖励；踩进胎孔/胎间空隙、非轮碰撞 → 失败/终止。

### 3.3 参数化与机器人比例（首版估计，待确认）

| 参数 | 大狗点足（17kg） | DigDog（3.5kg） | 说明 |
|---|---:|---:|---|
| 站高 | ~0.34m | 0.16m | 从 robot config 读取 |
| 步频 | 1.5Hz | 1.25Hz | 现有 |
| 足端半径 | 从 geometry 读取 | 0.0084m | 桩顶尺寸 ≥ 6–8×足径 |
| 桩截面 | 0.16–0.22m | 0.10–0.15m | ≥1–2 个高度图格子 |
| 桩高（课程） | 0.35–0.55m | 0.25–0.45m | 图1为高桩，非矮块 |
| 桩中心距 | 0.45–0.60m | 0.30–0.40m | 2–3×桩截面 |
| 轮胎外径 D | 0.55–0.65m | 0.40–0.45m | 图2平躺胎 |
| 胎孔 d | 0.28–0.32m | 0.20–0.24m | 保留可踩空孔 |
| 胎宽/高 w | 0.15–0.18m | 0.10–0.13m | 平躺后的高度 |
| 胎中心距 | ≈D | ≈D | 紧挨/轻微重叠 |

### 3.4 训练场 / 测试场统一

- 训练：`CourseCfg` 的 `stones`/`tires` 重建；`TERRAIN_KINDS` 仍 8 类；中央通道控制 geom 数。
- 测试：`go2w_test_maps.py` 的 `stones`/`tires` lane 重建；大狗原生测试入口同源；键盘键位 9/0。
- 高度场/障碍几何：`TerrainField.from_model` 支持 box（桩、胎环段）；竖直圆柱若使用需补竖直圆柱顶面；先优先 box，工程量小。
- Warp 静态 geom 上限：实测旧点足 8 类地形每环境 12552 geom，3072 环境约 38.6M，超过约 25M 静默丢接触；瘦身后（rubble 只铺中央、轮胎 8 段）每环境约 6.7–7.1k，3072 环境约 20.6–21.8M，接近已验证可用的 DigDog 21.4M。4096 环境需继续降到 <6144/环境，当前不做。

## 4. 观测、奖励、终止、课程

### 4.1 观测

- 本体 235 不变；带高度图 Actor 508（235+273）。
- 大狗先验证：true scan（教师）或同一套高度图；DigDog 再接 Mid-360 模拟高度图（10Hz、5 步保持、未知 -2.0）。
- **扫描分辨率风险**：21×13、x[-1,1]m/y[-0.6,0.6]m → 约 0.1m/格；桩顶 0.10–0.15m 只有 1–2 格、胎孔 0.20m 约 2 格。首版可保持 21×13；若落脚不稳，再评估 31×19 或 41×25（需同步 Actor/ONNX/契约/verify_contract）。

### 4.2 奖励

在现有 `task.py` 基础上新增/强化：
- **foothold**：脚在桩顶/胎面上的支撑奖励；脚落空隙/胎孔/低地面的惩罚；只在 `stones/tires` 生效（`kind∈{6,7}`），并用脚下 8 方向 0.14m 环形采样的“窄支撑”判定，排除 1.4m 起点平台刷分；
- `swing_clearance`/`foot_xy`/`foot_slip`/`landing_velocity` 保留；
- **nonwheel / body**：机身/腿撞桩、胎；沿用 15N 终止；
- **height / upright / track**：保留；降低高速/大转向权重，避免再次把 10/12 台阶训坏；
- **obstacle_progress/stair_progress**：桩阵/胎阵的推进奖励；
- 台阶项保持，按机器人缩放。

### 4.3 终止/成功

- 终止：掉下桩/胎面、脚踩进空隙/胎孔、机身低于桩顶、非轮碰撞、course_exit、超时。
- 成功：
  - 台阶：沿用 `stair_pass`（上阶+后脚过阶+落地高度+稳定）；
  - 梅花桩：从起点平台经桩顶走到目标，全程脚/机身不落低地面；
  - 轮胎：从胎阵一端走到另一端，至少 N 次胎面落脚、不踩胎孔/空隙。
- 训练统计：`stones`/`tires` 的 trials/pass/fall/stuck 重新定义，不再用“走完 5.4m 双向课程”这种可绕开障碍的宽松判据。

### 4.4 课程

- 8 类：flat/up/down/platform/slope/rubble/stones/tires；台阶比例不降。
- 高度：`STEP_HEIGHTS` 保留 2–16cm；stones 的桩高、tires 的胎高单独按等级映射。
- 晋级：地形探针 + 台阶 10/12 原生回归 + 步态门槛；`MIN_ITERATIONS` 不要为了 16cm 把 14cm 探针提前到 12cm 未稳之前。
- **存档选模**：每次 save 跑原生 flat/4/6/8/10/12/16 + stones/tires；按“台阶不退化 + 新地形通过”选最佳，不以最新为准。

## 5. 训练/验证流程与门禁

### 阶段

- **P0 设计确认**：尺寸、扫描分辨率、测试场几何、旧模型基线。
- **P1 大狗地形/测试**：新 stones/tires 几何 + 高度场 + 原生测试 + 键盘；旧大狗点足模型先走台阶基线。
- **P2 大狗教师**：true scan/高度图；新地形 + 台阶混合；20k 左右；每 N 轮原生 10/12；保存选模。
- **P3 DigDog 迁移**：同源地形/奖励；旧 `model_20000` 热启动；true scan 教师；10/12 门禁。
- **P4 DigDog LiDAR 学生**：P3 教师冻结 + BC/RL；Mid-360 高度图；10/12 + 新地形门禁。
- **P5 部署验证**：ROS2 高度图节点、ONNX 508、verify_contract、真实 Mid-360。

### 门禁

| 门禁 | 内容 | 不通过处理 |
|---|---|---|
| G0 | 地形高度场/8 环境/3072 geom/原生加载/scan | 修几何/分辨率 |
| G1 | 大狗旧模型台阶 10/12；新地形原生 | 不加新地形训练 |
| G2 | 大狗教师：10/12 不退化 + stones/tires 能过 | 选旧 checkpoint/调课程 |
| G3 | DigDog 教师：10/12 不退化 + 新地形 | 不启学生 |
| G4 | DigDog LiDAR 学生：教师 BC+RL；10/12 + 新地形 | 调 distill/继续 RL |
| G5 | 部署：ONNX/ROS2/verify_contract/真机 | 不上真机 |

## 6. 代码改动清单

- `rl/rpwd_digdog_supermimic/env/terrain.py`：stones/tires 重建；桩阵/胎环 helper；起点平台；通道/geom 数。
- `rl/rpwd_digdog_supermimic/core/terrain.py`：box 高度场（桩/胎环段）；如需竖直圆柱补顶面；scan。
- `rl/rpwd_digdog_supermimic/core/task.py`：foothold/gap/nonwheel 奖励与终止。
- `rl/rpwd_digdog_supermimic/core/stair_curriculum.py`：stones/tires 成功判据/stats/晋级；台阶保留。
- `rl/rpwd_digdog_supermimic/core/train.py`：save 选模/原生回归；旧模型热启动。
- `rl/scripts/go2w_test_maps.py`：测试场 stones/tires lane 重建。
- `rl/scripts/digdog_mujoco.py`：键位/图例/测试。
- 大狗点足对应文件：`rl/rpwd_pointfoot_supermimic/...` 与 `train_pointfoot/play_pointfoot` 同源改造；高度图/scan 若未接入，先接共享模块。
- 脚本：`train_*`/`auto_*` 门禁加入 10/12；旧模型路径；不再用坏教师。
- 文档：本设计、DEVELOPMENT 版本记录；实现后 commit。

## 7. 风险与开放问题

1. **桩顶/胎孔的扫描分辨率**：21×13 的 0.1m 格子可能不够；是否升到 31×19/41×25？会改 Actor/ONNX/契约，需 verify_contract。
2. **大狗的感知来源**：点足大狗没有 Mid-360；先 true scan/高度图验证，还是加深度相机/高度图？LiDAR 迁移只到 DigDog。
3. **size/比例**：表中尺寸是首版估计，需按图中实际桩/胎和机器人足端确认。
4. **10/12 回归**：新教师为什么把 12cm 训坏——20k 错误地形、命令阶段、scan、测试场差异；重训需保留台阶混合 + 原生回归 + 选模，必要时用旧模型做 KL/正则。
5. **Warp geom 上限**：桩/胎环段数量 × 8 类 × 8 档；3072 环境先验证，4096 需要 ngeom<6144。
6. **测试场深空隙**：测试地面是 plane；桩高出地面即可形成 0.3–0.5m 落差，但机器人仍可落到地面；测试成功/失败判据要明确（踩桩顶才算）。
7. **落脚奖励权重**：foothold 过大可能让机器人不敢走；先用小权重 + 终止，逐步加。
8. **sim2real**：真实梅花桩/轮胎的尺寸、摩擦、弹性与仿真差异；LiDAR 噪声、遮挡；部署前 verify_contract。

## 8. 里程碑

- M0：确认尺寸/扫描/测试场；旧模型基线复测（10/12）。
- M1：大狗点足新地形 + 高度场 + 原生/键盘；G0/G1。
- M2：大狗点足教师 20k 左右；G2；10/12 不退化。
- M3：DigDog 同源迁移 + true scan 教师；G3。
- M4：DigDog LiDAR 学生；G4。
- M5：部署/真机；G5。

## 9. 立即执行

1. 不再使用新 20k DigDog 教师；学生保持停止；自动接力不重启。
2. 旧 DigDog `model_20000`（60k run）作为台阶/键盘基线。
3. 按本设计先做大狗点足 P1：新 stones/tires 几何 + 高度场 + 原生测试 + 键盘，确认尺寸后进训练。

## 10. 实施记录（2026-11-18）

- 点足 `env/terrain.py`：8 类/8 档；stones 为细高桩 + 与桩顶齐平的起点平台，首排桩距起点 0.9m；tires 为 8 段平躺圆环紧密阵列；rubble 只铺中央 ±2.4m 赛道；3072 环境 geom 约 20.6M。
- 点足 `core/task.py`：`foothold=.5 / gap=-.5` 只在 stones(6)/tires(7) 生效；窄支撑判定排除平台；`core/replay.py` 新增 kind 字段并由 `train.py` 写入，`imagine.py` 把它透传给想象奖励。
- 首次 3072 环境训练（12552 geom/环境）在 step1150 表现为假训练：reward -5.5M、base_height -275M、velocity_mae 20–28m/s、stones 13705 回合全 fall；确认是 Warp 静态 geom 总缓冲超限导致自由落体，不是算法/奖励调参问题。
- 修复后 3072 环境训练 step50：reward=0.0634、falls=1、velocity_mae=0.074/0.050/0.116；step200：reward=0.0635、stones 50 次尝试（早期从盲走热启动掉桩属正常）、无自由落体/无 base_height 爆炸。
- 待办：点足 10/12 台阶与 stones/tires 落脚门禁；DigDog 按同源几何迁移（旧 `model_20000` 热启动，10/12 门禁）；`pointfoot_mujoco` 接 true scan 以加载新 508 维模型。

### 10.1 5000 步续训修复

- 现象：stairs/flat 持续进步（step5000 reward=0.0714、velocity_mae=0.038/0.032/0.063、level_mean=1.0），但 stones/tires 仍 0 pass。
- 根因：Actor 的想象 rollout 里 `survival` 只看倾角；真实 `stone_fall`（root z 相对起点平台下沉 >10cm）没有进入想象 continuation，Actor 在想象里走下桩/平台不会被惩罚。tires 入口普通地面还被 `gap` 持续扣分。
- 修复：Replay 增加 `origin_z`；`imagine.py` 按 `root_z - origin_z < .10` 把 stones 的 fall 乘入 `cont`；`task.py` 的 `gap` 增加 `high_near`（脚下 0.14m 内有 >0.12m 高面才计 gap）。
- 验证：8 环境 CPU 烟测 `gap=0`、actor_loss 有限、0 falls；从 `model_5000.pt` 续训，step5050 reward=0.0688、falls=1、velocity_mae=0.056/0.030/0.079、`gap≈0`；命令课程在 5000 晋级 stage2，统计窗口清零属设计行为。

### 10.2 简化地形（2026-11-18）

- 用户要求：停掉 20k，参考 extreme-parkour 的 stepping stones；保留现有平地；所有桩顶在同一平面；先大桩面，后续再减小；轮胎高度从 2cm 起步。
- stones：平地 + 同平面桩顶，无高起点平台；`side .40→.14`、`gap .04→.16`，桩高=课程 2–16cm。
- tires：轮胎高度=课程 2–16cm（2/4/6/8/10/12/14/16）。
- 阈值：`foothold/gap` 的 `on_top/high_near` 从 .12 降到 .01，去掉窄支撑判定；`stair_curriculum` 的 stones/tires 成功阈值同步 .01。
- 测试场：stones 等平面 .10m + 大桩面；tires 降到 .08m。
- 新 run：从 `model_18500` 热启动、3072 环境、20k；8 环境 CPU 烟测 falls=0、reward=0.0595、actor_loss 有限。
