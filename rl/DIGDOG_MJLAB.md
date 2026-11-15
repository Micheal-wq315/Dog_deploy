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

- 额定站高 **0.18m**（固定足跟接触点相对 `base_link`），初始高度 **0.189m**；
  膝角约 **0.51rad**，为摆动留出限位余量（0.19m 时膝角 0.60，已接近上限）。
- 脚端是弧形小腿，不能用 KFE 原点或零位包围盒中心；固定接触点取额定站姿下
  足跟接触区质心（KFE 局部坐标，见 `robot/config.py:FOOT_LOCAL`），等效脚半径
  **0.0084m**。
- XML 位置执行器 **kp20 / kv0.5 / ±10N·m**；动作尺度 **0.5**，关节电枢 0.01，
  soft limit 0.95。
- 步态 **1.25Hz**，duty **0.6**，抬脚净空 **0.05m**，FL/RR 与 FR/RL 对角交替。
- 广义解析 IK：roll 绕 X、hip/knee 绕 Y，带每腿固定偏移和方向符号；Actor 仍为
  **235→12**（47×5 帧），世界模型 340 维输入不变。
- 奖励高度区间 **0.17–0.22m**（rough 0.15–0.24），非足端碰撞终止阈值 15N。

## 训练

```bash
cd /media/rmer/data/rpwd_training/worktrees/rpwd-digdogv1-mjlab
# 默认 4096 环境、60000 轮、seed0；1.25Hz 已固化在步态模块
./rl/scripts/train_digdog_mjlab.sh
# 从本机器人检查点续训（不能加载点足/轮足 checkpoint）
./rl/scripts/train_digdog_mjlab.sh --resume /绝对路径/model_xxxxx.pt --iterations 60000
```

正式训练：服务 `rpwd-digdog-mjlab-60k-20261115.service`，运行目录
`/media/rmer/data/rpwd_training/digdog_mjlab/2026-11-15_大狗DigDogv1点足版_60k_seed0/`，
stdout `train_stdout.log`，指标 `metrics.jsonl`。超参与点足正式版一致：4096 环境、
2048 世界模型批量×16 步、2048 想象批量×48 步、300 轮预热、每 500 轮保存、
每 5000 轮 CPU 确定性抽查。启动检查到 100 轮时 WM loss 已下降、0 跳过、0 无效复位；
**这不等同于收敛或上阶能力**。

## 测试

```bash
# 原生 MuJoCo CPU 评测；训练内每5000轮抽查也走这个入口
./rl/scripts/play_digdog_mjlab.sh --mode evaluate \
  --checkpoint /media/rmer/data/rpwd_training/digdog_mjlab/<run>/model_05000.pt \
  --lanes flat stairs4 stairs8 stairs12 --speed .2 --seconds 55 --seeds 0 1 2
```

已验证：12 轮短模型的平地 5 秒测试完成（前进约 1.46m、均速约 0.205m/s、无翻倒、
无非足端碰撞）；4096 环境 2 轮冒烟显存峰值约 3.2GB；8 环境 12 轮更新冒烟中
world model / actor / critic loss 均为有限值、0 跳过更新、0 无效复位。
正式 60000 轮的台阶通过率、12cm 能力和 ONNX 导出尚待训练后验证。

## 边界

- 原始 MJCF 不能直接当 Z-up 使用；任何重新导出也必须经过包装旋转。
- 改变站高/脚点会同时影响 `DEFAULT_Q`、`INIT_HEIGHT`、`FOOT_LOCAL` 和
  `profile.height_band_*`，改后应重新跑原生平地与台阶短测。
- `robot_id=rpwd_digdog`，检查点格式 `rpwd-digdog-*`；禁止与
  `rpwd_pointfoot` / Go2W / 旧 Isaac 检查点交叉加载。
