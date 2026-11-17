# DigDogv1 测试记录

## 第一版脚本

`record_test.py`：原生 MuJoCo CPU 全量记录，逐 50Hz 控制步保存机身/IMU、12 关节角/角速度/目标角/动作/扭矩、四足位置/速度/接触力/撞击峰值/离地间隙、非足端碰撞、命令/相位和 235 维观测历史。

```bash
cd /home/rmer/Dog_deploy-main/Dog_deploy-main/test/dig_dog
python record_test.py \
  --checkpoint /media/rmer/data/rpwd_training/digdog_mjlab/2026-11-16_大狗DigDogv1点足版_60k_seed0_v2/model_60000.pt \
  --lanes stairs10 stairs12 --directions forward backward --seconds 50 --seeds 0
```

输出在 `runs/<时间>_model<轮次>/`；字段定义见 `../README.md`。

## 2026-11-17 正向/反向 12cm 检查结果

原生 stairs12，0.2m/s，seed0，50s（`max_x` 越过 7.5 即表示爬完整条上升段；`success=false/timeout` 是成功锁定条件未触发，不是没上去）：

| 模型 | 正向 12cm max_x | 反向 12cm max_x | 备注 |
|---|---|---|---|
| model_20000 | **9.99** | **9.75** | 正/反都能走完，最佳全能快照 |
| model_30000 | **9.42** | 2.37（fallen, tilt 63°） | 正向能走，反向反而摔 |
| model_40000 | 2.31（stuck_8s） | **9.59** | 正向卡第一级，反向能走 |
| model_50000 | 2.30（stuck_8s） | **9.67** | 同上 |
| model_60000 | 2.23（stuck_8s） | **9.83** | 同上；10cm 正向 2.32、反向 10.04 |

结论：

- **正向 12cm 能力在 30k→40k 之间退化**，不是一开始就不会；之前看到的“正面能上 12cm、有明显抬腿”对应 **model_20000/30000**。
- **反向 12cm 从 40k 起保持**；model_30000 反向反而摔倒，说明训练在正/反两个方向之间发生了策略切换/遗忘，不是简单单调提升。
- 正向卡点都在 **x≈2.2–2.3 的第一级台阶**；20k/30k 的 `foot_peak` 约 111/132N，40k+ 约 93–97N，抬腿/足端撞击模式明显不同。
- `success=false` 但 `max_x>7.5` 的 run 其实是越过终点的；成功锁定条件（≥2 足接触、0.15–0.70m 离地、竖直速度）对当前步态偏严，需和“能不能上去”分开看。

## Mid-360 高度图模型（2026-11-17 起）

`record_test.py` 会自动识别 checkpoint 的 `scan_mode`：

- `scan_mode=none`：旧 235 维盲策略，不写 `height_scan.npy`；
- `scan_mode=true`：特权教师，`height_scan.npy` 是真实 273 维高度扫描；
- `scan_mode=lidar`：Mid-360 学生，`height_scan.npy` 是带视场/遮挡/噪声/丢点、未知格 -2.0 的 273 维局部高度图。

```bash
# 教师/学生都可直接记录；--no-obs 可关闭 235 维观测，高度图仍会保存
python record_test.py \
  --checkpoint /media/rmer/data/rpwd_training/digdog_mjlab/<run>/model_02000.pt \
  --lanes stairs10 stairs12 --directions forward backward --seconds 50 --seeds 0
```

高度图按 10Hz 更新、保持 5 个控制步，`height_scan.npy` 与 `observations.npy` 逐行对齐。原生确定性回放不加随机噪声，但保留 Mid-360 的垂直视场（默认下倾 20°）和射线遮挡。

## 当前建议

1. 需要正向 12cm 现在可用 **model_20000**（正向 9.99 / 反向 9.75）；model_30000 只适合正向。
2. 训练后期（stage 4 / 命令范围扩到 vx -.35~.45 后）疑似出现正向策略遗忘；下一版应做**方向平衡 + 完整往返 + 低阶回放**，而不是继续从 60k 单调续训。
3. 前腿高抬与 40k+ 抬腿幅度下降需要结合 CSV 里的 `foot_*_clearance/peak`、`q/dq/tau` 和前腿自碰撞进一步确认。

## 训练代码层面的原因（2026-11-17 回归检查）

按主次：

1. **主触发器：Actor 学习率调度**
   - `rl/rpwd_digdog_supermimic/core/train.py:actor_learning_rate()`
   - `rl/scripts/train_digdog_mjlab.sh`：`--actor-hold-iterations 20000 --lr-decay-iterations 20000 --lr-final-frac .2 --actor-lr-scale .3`
   - Actor LR：20k 6e-5 → 30k 3.6e-5 → **40k+ 1.2e-5 地板**。20k/30k 正向 12cm 能走完，40k 起卡第一级；反向 30k 反而摔、40k+ 反而能走，说明低 LR 后策略从“正向优先”切到“反向/后腿优先”并被锁死。

2. **直接机制：前腿抬脚奖励/策略**
   - `core/task.py:reward_terms()` 的 `swing_clearance`、`stair_progress`、`foot_xy`
   - 20k→60k 正向 stairs12：前腿 FL/FR 台阶 clearance max **0.322/0.332 → 0.224/0.224**；前足 peak **111.6/93.1N → 63.6/64.0N**；后腿 RL/RR 仍 0.323/0.310、peak 93.4/74.8N。
   - `swing_clearance` 奖励从 -0.0188 恶化到 -0.0339；正向靠前腿领阶，反向靠后腿，所以反向保留、正向第一级过不去。

3. **制度因素：台阶判定只看升段 + 双向地形**
   - `core/stair_curriculum.py:stair_pass()` 只检查 `|x|>2.8` 和机身高度，不区分正向/反向，也不要求下阶。
   - `env/terrain.py` 的 `CourseCfg(bidirectional=True)` 把 up/down 在 x 两侧镜像。
   - 最终 up 12cm 64.8% vs **down 12cm 89.9%**；反向/下阶是更容易拿分的局部最优。

4. **放大因素：命令课程 stage 4 + 地形等级**
   - `core/command_curriculum.py:STAGES` 在 22000 轮进入 stage 4：`vx -0.35~0.45 / vy ±0.2 / wz ±0.8`。
   - `env/terrain.py`/`stair_curriculum` 的地形等级 20k 3.49 → 60k 4.83；低速度正向 12cm 样本占比下降。

**排除**：世界模型 loss 0.377→0.277、critic loss 0.0028→0.0038 均正常；机器人资产/契约指纹匹配；同一测试脚本下 model_20000/30000 能过 12cm，不是评估或资产问题。更像 Actor 策略坍缩/课程偏置。

## 消融与修复建议

- 从 **model_20000** 做消融（不要从 60k 续训）：
  1. Actor LR 地板抬到 3e-5、延长 hold，避免 40k 被锁死；
  2. 命令上限先压到 0.3，保留低速度正向样本；
  3. 对 FL/FR 台阶 clearance/peak 加专项奖励/约束，并检查前腿高抬时的大腿-机身自碰撞；
  4. `stair_pass` 改为正向/反向分别统计 + 完整往返（上→平台→下）；
  5. 方向平衡 + 低阶回放，避免策略只靠反向/下阶。
- 当前可直接使用 **model_20000**（正向 9.99 / 反向 9.75）做对照；model_60000 只适合反向。
