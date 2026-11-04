# 点足 SuperMimic 完整任务 v3

2026-11-04，分支 `codex/pointfoot-supermimic-wm`。本版本解决上一版只有正向上阶、无法完整遥控，以及高度/终止/评测口径不一致的问题。新增任务独立注册为：

`Isaac-Velocity-Rough-RPWD-Pointfoot-SuperMimic-Full-v0`

旧单向上阶任务仍对应旧模型；回放会根据 checkpoint 的 task 自动选择。Actor 的225维输入、12维位置动作、34维物理状态及50Hz控制频率保持兼容。本版本是本体盲策略，高度图只供世界模型和 Critic 使用。

## 源码参考与迁移范围

参考本地 `go2w-supermimic` 的 `a299ed5f5571da4492b8a286d94acfe4a86128e1`；保留其MIT版权与许可于 [third_party/go2w_supermimic_LICENSE](third_party/go2w_supermimic_LICENSE)：

| 源码 | 本版实现 |
|---|---|
| `core/mixed_commands.py` | 纯前后、纯横移、纯转向、组合指令混合；楼梯以低速双向直行为主 |
| `env/terrain.py` | 平地、上阶、下阶、台面、坡道、碎石，镜像双向赛道 |
| `core/stair_curriculum.py` | 真实稳定到达末级才算通过；按实际通过率和速度误差扩速 |
| `core/terrain.py` | 静态地形高度缓存，想象时根据预测位姿重新查询 |
| `core/terrain_world_model.py` | 残差世界模型、自回归窗口、接触/载荷/碰撞辅助监督 |
| `core/imagine.py` | 冻结模型权重，保留动作梯度，32步全回报＋末端目标Critic，TD(λ) |
| `core/replay.py` | WM混采在线、事件和成功窗口；Actor种子均匀采样 |

机器人仍是RP_wd机身与Go2小腿/固定球脚、12个腿关节，仿真使用Isaac Lab/PhysX。保留本机PD参数与力矩/速度限幅。轮子滚动专用项不适用于点足，使用支撑脚切向速度、局部越阶净空和关节姿态约束。物理上下文是四脚摩擦＋COM偏移7维，载荷辅助头仍为四个合力模长。**没有电机力矩惩罚。**

## 高度与姿态

上一版没有固定高度误差项，主要限制来自0.17m低净空终止和想象中的高度生存代理。本版统一实际环境与想象的支撑面定义：用脚下地面、当前足端接触和载荷估计支撑面；无有效支撑时回退到机身正下方净空。

| 项目 | 当前配置 |
|---|---|
| 典型默认站高 | 约0.339m |
| 平地零惩罚高度区间 | 0.22–0.42m |
| 明显起伏/台阶零惩罚区间 | 平滑放宽至0.20–0.44m |
| 区间外高度权重 | -0.10，按越界距离/0.10m的平方计费 |
| 紧急低高度终止 | 支撑面净空低于0.12m |
| 姿态奖励 | 10°内直立奖励不衰减，超过后平缓减弱 |
| 跌倒终止 | 倾角60° |
| 垂向速度权重 | 从-0.25减至-0.05 |
| 横向角速度权重 | 从-0.05减至-0.02 |
| 关节姿态权重 | 从-0.05减至-0.02，起伏处进一步放松髋/膝 |

高度范围内的代价和梯度均为零，不把身体拉回某一个固定站高。例如平地0.45m比上界高3cm，每秒高度代价仅约0.009。机身从低踏面跨到高踏面时，实际终止和想象生存都使用同一支撑面估计，避免仅因根部射线命中高台阶而误判趴下。

停止项只在接近零指令时抑制机身和脚端漂动；运动时的脚滑用URDF正运动学/雅可比、关节速度、机身速度计算，且仅作用于支撑脚。越阶净空以减少欠缺净空的代价引导抬脚，不持续给悬脚正奖励；进度来自有符号、有界的真实机身运动。

## 完整运动任务与课程

普通地形的非停止指令按40%纯vx、20%纯vy、20%纯wz、20%组合采样，额外独立覆盖10%停止。正负方向都采样，普通指令每3–6秒更换。楼梯的非停止样本80%为0.20–0.60m/s范围内、遵守当前课程上限的双向直行，20%为调整；直行指令保持到回合结束。

| 阶段 | vx上限 m/s | vy上限 m/s | wz上限 rad/s | 最早开放轮数 |
|---:|---:|---:|---:|---:|
| 0 | ±0.25 | ±0.15 | ±0.40 | 0 |
| 1 | ±0.32 | ±0.20 | ±0.50 | 1500 |
| 2 | ±0.40 | ±0.25 | ±0.65 | 4000 |
| 3 | ±0.48 | ±0.30 | ±0.80 | 8000 |
| 4 | ±0.55 | ±0.35 | ±0.90 | 12000 |
| 5 | ±0.60 | ±0.40 | ±1.00 | 16000 |

扩速同时要求：

- 最近的上下阶记录各至少64个有效回合，通过率均≥60%。
- 实测vx/vy/wz误差达到门槛。
- 停止、前进、倒退、左右横移、左右原地转向七类均有足够样本，且各方向有真实响应；纯粹站桩即使全局平均误差较小也不能升级。

地形为6行×10列、每块16×16m，中心出生，±X均有障碍；列数按1平地/3上阶/2下阶/1台面/1坡道/2碎石分配。列类型精确复用Isaac Lab的比例映射，避免把某个列编号误当成地形种类。

上、下阶各有五级踏步，踏面0.30m，单级高度2/4/8/12/16/20cm。坡道从约3°到15°，碎石随等级增大，台面保留落地段。所有环境从最低难度开始，先满足64回合、60%通过及时间门槛才开放更高等级；时间只是最早开放条件，未达质量门槛时继续练当前级。转向、停止和不沿赛道的纯横移为中性回合，不因为没有走过X方向终点就降级。

通过检查末级脚底高度、至少两个当前接触脚、机身高度/倾角、垂向速度，并要求连续稳定0.12秒。卡住、失败、真实通过、中性回合分开统计。赛道出口对正反方向对称，解决旧版后退0.35m就复位的问题。键盘回放仍默认仅R手动复位。

## 世界模型与稳定性

保持单个512×3残差模型、225→12盲Actor、329维特权Critic、8步世界模型自回归、32步想象和5轮/4小批次价值训练。世界模型输入338维；连续输出41维及6维接触/碰撞/终止辅助输出。

- 回放取消两份可由地形重建的273维扫描；1024环境、512步时约节省1.07GiB。
- 默认4096个事件窗口、2048个成功窗口持久保留；不足时回退在线样本。窗口跨reset会截断，旧环形数据不能被新转移冒充。
- Actor初始状态均匀抽样；优先回放只用于世界模型。
- 保存8步回放预测的速度、净空、姿态和接触误差；这些是回放诊断，不是独立验证集成绩。
- 无效物理状态在自动reset结算前标记失败，排除成功档案；部分reset保护健康环境历史，避免同一物理帧被插入两次。
- 保存任务、奖励、地形、种子、动作映射和代码摘要；配置变化需开启新实验。课程和命令阶段完整恢复，回放/档案重新预热。

## 训练、回放与验收

以下命令在隔离worktree执行：

```bash
cd /home/rmer/Dog_deploy-main/Dog_deploy-main-pointfoot-wm
# 默认1024环境、40000轮，headless、单实例保护、防休眠。
./rl/scripts/train_pointfoot_supermimic_full.sh
# 如需20000轮，覆盖累计目标轮数。
./rl/scripts/train_pointfoot_supermimic_full.sh --iterations 20000
# 同任务继续训练，保持原环境数与seed。
./rl/scripts/train_pointfoot_supermimic_full.sh --resume /path/to/teacher_latest.pt
# 可选：只迁移旧225维Actor；世界模型、价值、优化器、课程重新初始化。
./rl/scripts/train_pointfoot_supermimic_full.sh --init_actor /path/to/old_teacher_final.pt
```

完整任务默认从零训练。旧单向模型可选择迁移Actor，但迁移后需要重新学习未覆盖的停止、倒退、横移和转向；不能把旧单向训练的轮数当成完整任务已训练轮数。

```bash
# 自动识别checkpoint对应任务；默认无自动复位。
./rl/scripts/play_pointfoot_supermimic.sh --checkpoint /path/to/full_teacher_final.pt \
  --teleop keyboard --terrain flat
./rl/scripts/play_pointfoot_supermimic.sh --checkpoint /path/to/full_teacher_final.pt \
  --teleop keyboard --terrain down --terrain_level 2
# 八种运动指令确定性验收，保存JSON与CSV。
./rl/scripts/evaluate_pointfoot_supermimic.sh --checkpoint /path/to/full_teacher_final.pt \
  --suite commands --num_envs 8 --seeds 42 43 44
# 六类地形与2/4/8/12cm诊断；不是噪声探索成绩。
./rl/scripts/evaluate_pointfoot_supermimic.sh --checkpoint /path/to/full_teacher_final.pt \
  --suite all --levels 0 1 2 3 --num_envs 8 --seeds 42 43 44
```

正式日志/模型位于 `/media/rmer/data/rpwd_training/pointfoot_supermimic_full/`；验收默认位于 `/media/rmer/data/rpwd_training/pointfoot_evaluations/`。键盘默认幅值会读取checkpoint的已开放命令阶段，避免用超出当前课程的速度误测模型。自动选择最新模型只查看正式运行目录，测试产生的validation子目录不会冒充正式模型。

## 验证与结果边界

本次完成代码与集成验证，未启动新的长期训练：

- 83项CPU测试及52个子用例通过，覆盖高度区间/梯度、20cm跨阶支撑面、FK脚滑、方向分布、课程晋级/恢复、环形档案、NaN恢复与健康行历史保护、真实落地和自动reset后的计数隔离。
- 32环境20轮短回合训练完成13次Actor更新，跳过更新0、无效环境恢复0；从该checkpoint恢复到22轮通过。
- 确定性验收工具完成15个场景、2环境共30个工具冒烟试次，覆盖六类地形；输出标记smoke_only=true，不视作策略能力成绩。
- 旧20000轮单向模型的4cm台阶5步回放通过；完整verify_contract.sh通过。

验证日志位于数据盘的 `pointfoot_full_v3_*_20261104.log`，验收数据为 `/media/rmer/data/rpwd_training/pointfoot_evaluations/full_v3_validation_20261104/`。具体结果同时记录在 `DEVELOPMENT.md` 的版本表。短回合训练用于检查模型更新、reset与课程保存/恢复；评测的`--max_steps`仅用于工具冒烟，输出明确标记`smoke_only=true`。这些测试不证明新策略已学会运动或上阶。

上一轮20000轮单向模型可用于对照。完整任务需要新一轮训练后，再根据分方向误差、停止漂移、各高度的真实稳定通过率判定效果，不能只凭总奖励或平均地形等级下结论。
