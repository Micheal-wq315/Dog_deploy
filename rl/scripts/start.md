# RP_wd 训练 / 测试常用命令速查

> 所有命令都在**仓库根目录** `Dog_deploy-main/` 下执行。
> 脚本内部会自动激活 conda 环境 `isaaclab` 并定位 Isaac Lab（`/media/rmer/data/IsaacLab`）。

---

## ⚠️ 三条铁律

1. **不要同时开两个仿真** —— 多个 Isaac Sim 实例共用一张卡会把训练进程**死锁**
   （2026-10-10 实测：训练空转 12 小时，只跑到 101/3000）。GUI 测试有保护，会拦住你；
   测试脚本自己也要等训练结束再跑。
2. **训练永远无头**（脚本内强制），不需要显示器；只有 play / test / teleop 会开窗口。
3. **杀仿真进程要 `-9`** —— Isaac Sim 会吞掉 SIGTERM：
   ```bash
   pkill -9 -f "rl/scripts/[t]est_policy.py"    # [t] 是为了不误杀自己这条命令
   ```

---

## 1. 训练

```bash
./rl/scripts/train.sh                                    # 默认: 平地, 2048 环境, 3000 迭代, headless
./rl/scripts/train.sh --num_envs 4096                    # 更大并行 (实测 1.2 s/迭代, 3000 迭代约 57 分钟)
./rl/scripts/train.sh --max_iterations 6000 --run_name v2
./rl/scripts/train.sh --resume                           # 从最新 checkpoint 续训
./rl/scripts/train.sh --resume --load_run 2026-10-11_11-35-05_rolling_v4 --checkpoint model_1500.pt

# Odin1 v19：18000轮基础地形教师 → 三种子验收 → 2000轮含噪学生（共20000，headless）
./rl/scripts/train_odin_pipeline.sh --num_envs 256 --seed 42

# 从已保存教师checkpoint接续两万轮流程；阶段性能未达标仅记录，最终资格门槛仍保留
./rl/scripts/train_odin_pipeline.sh --num_envs 256 --seed 42 \
  --resume-teacher-checkpoint /home/rmer/Dog_deploy-main/Dog_deploy-main/rl/logs/rsl_rl/rp_wd_odin_teacher/2026-10-29_22-50-43_odin_basic_v19_20261029_225031_teacher2000/model_1999.pt \
  --continue-on-stage-failure
# 分阶段续训学生，到学生阶段总目标2000轮
./rl/scripts/train_odin_depth.sh --task Isaac-Velocity-Rough-RPWD-OdinDepth-Student-v0 \
  --resume --load_run <学生run名> --max_iterations 2000
# 直接PPO对照（20000轮，含深度增强）
./rl/scripts/train_odin_depth.sh --run_name odin_basic_direct_v19
```

- 日志：`rl/logs/train_*.log`（训练 stdout）＋ `rl/logs/rsl_rl/<实验名>/<时间戳>_<run_name>/`（TensorBoard + checkpoint）
- 训练期间**自动阻止系统休眠**（`systemd-inhibit`），夜间不会被冻死
- 关键指标在 TensorBoard 里看 `Episode_Reward/*`（跟踪 vs 惩罚的配比）和 `Episode_Termination/*`
- 深度相机使用离屏渲染；训练仍无 GUI，但比普通任务更占渲染资源
- 教师：2000–10000轮扩展速度，14000–17000轮加入平地姿态/高度，15000–18000轮渐入小幅水平推扰；学生前1000轮渐入深度退化
- 一键流程在2000/6000轮验平地并记录台阶诊断，12000轮验6cm上阶/8cm下阶，通过才续训；最终18000轮进行三种子基础验收。平地0.35m/s指令下实测vx需≥0.25m/s；默认未通过自动停止。显式加`--continue-on-stage-failure`仅允许阶段性能失败继续，最终教师验收仍需通过
- 教师地形上限按0/2000/5000/9000/13000轮解锁0/2/5/8/11，最高单级8cm；四分之一专项环境长期复习0–2级；到达终点且无严重非足端撞击才升级。完整设计见 [CURRICULUM.md](../CURRICULUM.md)
- 本轮专注平地、1–8cm上下阶/低障碍、起伏与缓坡，不混入绕墙训练；固定测试中的12cm箱与绕障只作诊断
- 已移除所有电机力矩惩罚；12维腿部力矩观测仍保留
- OdinDepth 当前输入契约为1740→16（68×3本体/力矩 + 8×16×12深度记忆）；旧780维 checkpoint 不兼容，需新模型
- 深度原图默认逆时针旋转90°，adapter由22帧历史间隔3帧抽8张（约1.4秒）；旧1740维模型回放用`--depth-memory-stride 1`、ROS用`memory_stride:=1`。实物方向相反时启动adapter时设 `rotation_deg:=270`
- 高度奖励与低高度终止改用机身正下方地面射线，按实际地面评估站姿净空
- `--max_iterations` 表示包含已完成轮数的总目标；续训恢复课程步数，从 checkpoint 下一轮开始

```bash
tail -f rl/logs/train_rolling_*.log                      # 实时看迭代/回报
tensorboard --logdir rl/logs/rsl_rl/rp_wd_flat           # 曲线
```

---

## 2. 验收 / 测试

```bash
# 契约回归: 训练侧 ↔ 部署端接口是否仍然一致 (改过观测/动作/奖励后必跑)
./rl/scripts/verify_contract.sh                          # 静态 + 运行时, ~25 秒
./rl/scripts/verify_contract.sh --no-sim                 # 只跑静态检查, 秒级

# 指令扫描: 10 组典型指令, 输出跟踪误差 / 姿态 / 滑移比汇总表 (最常用的量化验收)
./rl/scripts/test_policy.sh --sweep --headless
./rl/scripts/test_policy.sh --sweep                      # 开窗口, 边看边出表

# ★ 崎岖地形 (随机地形: 台阶/方格/粗糙/坡度 共 6 类, 全难度随机出现)
./rl/scripts/test_policy.sh --rough --sweep --headless   # 地形上量跟踪误差
./rl/scripts/test_policy.sh --rough --cmd 0.5 0 0        # 地形上跑固定指令
./rl/scripts/test_policy.sh --rough --train-dist --sweep # 训练分布(含域随机化)下的地形鲁棒性

# 固定指令: 一组指令跑到底
./rl/scripts/test_policy.sh --cmd 0.5 0 0 --headless     # 前进 0.5 m/s
./rl/scripts/test_policy.sh --cmd 0 0 0.6 --headless     # 原地左转
./rl/scripts/test_policy.sh --cmd 0 0.4 0                # 侧移

# 指定模型 / 测鲁棒性 / 顺带导出
./rl/scripts/test_policy.sh --checkpoint rl/logs/.../model_1500.pt
./rl/scripts/test_policy.sh --train-dist --sweep         # 用带域随机化的训练配置测
./rl/scripts/test_policy.sh --export                     # 测试同时导出 ONNX

# Odin1 深度策略 (训练完成后 GUI 回放并导出专属 ONNX)
./rl/scripts/play.sh --task Isaac-Velocity-Rough-RPWD-OdinDepth-Play-v0 --checkpoint rl/logs/rsl_rl/rp_wd_odin_student/<run>/model_1999.pt

# Odin1 深度模型量化测试：9组范围内指令，保存CSV、JSON、PNG和SVG
# 每次指定新的保存目录，已有 samples.csv 时会拒绝覆盖
./rl/scripts/gui_test.sh --task Isaac-Velocity-Rough-RPWD-OdinDepth-Play-v0 \
  --checkpoint rl/logs/rsl_rl/rp_wd_odin_student/<run>/model_1999.pt \
  --terrain-level 0 --eval-reset --seed 42 --num_envs 12 --sweep \
  --settle_steps 50 --measure_steps 200 --headless \
  --save-dir rl/logs/evaluation/odin_test_level0
# 去掉 --headless 可打开GUI；--terrain-level 2 可测试更高难度

# 保存 Odin 原图、90°/270°方向候选、16×12策略网格及PNG预览（每25步，约0.5秒一帧）
./rl/scripts/gui_test.sh --task Isaac-Velocity-Rough-RPWD-OdinDepth-Play-v0 \
  --checkpoint rl/logs/rsl_rl/rp_wd_odin_student/<run>/model_1999.pt \
  --cmd 0.3 0 0 --save-dir rl/logs/evaluation/odin_depth_check --save-depth

# 固定场景基准：平地、6cm上阶、8cm下阶、12cm矮箱、45cm绕行墙、45cm交错障碍
# 默认12环境、前进0.35m/s、1000步；保存每场景深度图与CSV/JSON/曲线
./rl/scripts/gui_test.sh --obstacle-benchmark \
  --checkpoint rl/logs/rsl_rl/rp_wd_odin_student/<run>/model_<iter>.pt \
  --save-dir rl/logs/evaluation/odin_obstacle_benchmark --save-depth --headless

# 保存完整训练数据与训练曲线：不启动仿真，可在训练期间使用
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
python rl/scripts/plot_training.py \
  --run rl/logs/rsl_rl/rp_wd_odin_student/<run> \
  --output-dir rl/logs/evaluation/odin_training
```

固定场景基准的 `terrain_summary.json` 记录逐回合完整完成率、非足端接触<20N的低接触完成率、未完成尾回合、停滞比例和轮端力P95。每个场景按越过最后障碍的距离阈值判定，不把1.5m靠近当成功；地形仍为合并mesh，非足端接触不能精确区分障碍与地面，不能报告精确障碍碰撞率。

**测试数据**：`samples.csv` 保存每环境、每策略步的状态和终止原因；`summary.csv/json`
只汇总测量阶段，`terrain_summary.json` 按地形类型汇总。`--save-depth` 会生成 `depth_frames/`
（原始深度、旋转校正图、16×12米制网格、归一化网格及PNG预览）；CSV含脚下地面高度及机身净空。
`test_curves.png/svg` 为测试图，
`training_curves.png/svg` 为训练图。固定等级会启用按难度行生成地形，并将所有环境放到指定行；
它是混合地形速度测试，不代表某一台阶高度的通过率。测试中的非超时终止与正常超时分别计数。

**看什么**：`线速误差` 越小越好；`滑移比` 仅是轮线速度与机身速度的粗略比值，原地转向时无意义，不能单独判断接触打滑。

---

## 3. GUI 演示 / 遥控

```bash
# GUI 测试 (带并发保护: 训练在跑会拦住)
./rl/scripts/gui_test.sh                                 # 随机指令
./rl/scripts/gui_test.sh --cmd 0 0 0.6                   # 原地转, 演示差速最直观
./rl/scripts/gui_test.sh --wait                          # 训练在跑: 等它结束再自动开
./rl/scripts/gui_test.sh --force                         # 训练在跑也立刻开 (自担风险)
./rl/scripts/gui_test.sh --dry-run --force                # 只看要执行什么, 不启动

# 键盘遥控 (手动驾驶)
./rl/scripts/teleop.sh                                   # 平地: 4 台机器人, 真实 50Hz
./rl/scripts/teleop.sh --rough                           # ★ 崎岖地形上手动驾驶
./rl/scripts/teleop.sh --num_envs 8 --teleop-scale 0.5    # 多放几台 + 速度减半
./rl/scripts/teleop.sh --teleop gamepad                  # 插了手柄再切手柄
```

**遥控键位**（先点一下 Isaac Sim 窗口获得焦点；按住生效，松开归零）：

| 按键 | 动作 | 按键 | 动作 |
|---|---|---|---|
| `W` / `S` | 前进 / 后退 | `E` | 左转 |
| `A` / `D` | 左移 / 右移 | `Q` | 右转 |

（方向键 / 小键盘 8-2-4-6-7-9 同样可用；改键位去 `rl/scripts/keyboard_wasd.py`）

---

## 4. 导出 ONNX → 上实机

```bash
./rl/scripts/play.sh                                     # 回放 + 导出 + 自动校验
./rl/scripts/play.sh --no-export --num_envs 16           # 只回放不导出

# 普通策略产物: rl/exported/policy_flat.onnx (输入 obs[1,168] → 输出 actions[1,16])
cp rl/exported/policy_flat.onnx src/inference/robots/wheel_quad/models/policy_flat.onnx

# 实机运行 (部署端)
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch roboparty_inference inference.launch.py robot:=wheel_quad policy:=default
```

Odin1 深度策略的独立部署顺序：

```bash
# 终端 1：启动官方 Odin1 ROS2 驱动，并按设备说明开启深度流
# 确认官方驱动 config/control_command.yaml 的 senddepth 已启用
ros2 launch odin_ros_driver odin1_ros2.launch.py
# 首次连接可检查深度消息 encoding (adapter 支持 16UC1 毫米或 32FC1 米)
ros2 topic echo --once /odin1/depth_img_competetion --field encoding
# 终端 2：旋转90°、生成16×12网格并维护8帧深度记忆（朝向相反时传 rotation_deg:=270）
ros2 launch roboparty_inference odin_depth_adapter.launch.py rotation_deg:=90
# 终端 3：加载 policy_odin_depth.onnx 与独立输入配置
ros2 launch roboparty_inference inference.launch.py robot:=wheel_quad policy:=odin_depth
```

深度策略输入为原有56维观测 + 12维归一化腿部力矩（共68维，堆叠3帧）+ 8×16×12深度记忆，
ONNX 为1740→16。学生阶段对深度渐入3cm噪声、1%缺测、回合固定95–100%裁切、偏差/遮挡、丢帧和延迟；仿真与adapter
都将160×120原图逆时针旋转90°，再按0.2–6m裁剪并生成16行×12列最近距离特征。实物朝向相反时将
`rotation_deg`设为270。上机前用 `--save-depth` 检查原图/校正图/网格，并校准相机安装位姿。

采集真实 Odin1 数据校准旋转/量程（先 source ROS 2 与工作空间，并启动官方深度节点）：

```bash
source /opt/ros/jazzy/setup.bash
cd ~/Dog_deploy-main/Dog_deploy-main
colcon build --packages-select roboparty_inference
source install/setup.bash
ros2 topic type /odin1/depth_img_competetion
ros2 run roboparty_inference odin_depth_calibrate.py \
  --topic /odin1/depth_img_competetion --frames 1 \
  --output rl/logs/evaluation/odin_depth_calibration
```

将已知障碍放在相机正前方，检查输出目录的 `rotation_candidates.png` 和 `grid_candidates.png`，确认障碍落在策略网格的前方区域，再决定 adapter 使用90°还是270°。工具会保存原始 `.npy`、encoding、图像尺寸、有效像素比例和深度统计；如果 `ros2 topic type` 不是 `sensor_msgs/msg/Image`，需按官方消息类型调整 adapter，不能直接套用此脚本。

实机手柄：`is_joy_control_` 默认开启，左摇杆 axes[4]/axes[3] → vx/vy，LT/RT → ωz，带 `clip_cmd` 限幅。

---

## 5. 状态排查

```bash
# 训练跑到哪了 / 还剩多久
grep -ao 'Learning iteration [0-9]*/[0-9]*' rl/logs/train_*.log | tail -1

# 现在有哪些仿真在跑
pgrep -af "isaaclab.sh|rl/scripts/.*\.py" | grep -v pgrep

# 显存占用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 最新模型是哪个
ls -t rl/logs/rsl_rl/*/*/model_*.pt | head -1
```

---

## 6. 常用参数一览

| 参数 | 作用 | 备注 |
|---|---|---|
| `--num_envs N` | 并行环境数 | 平地 4096 约 1.2 s/迭代；显存 5–8 GB |
| `--max_iterations N` | 训练迭代数 | 平地收敛约 3000 |
| `--run_name X` | 本次别名 | 出现在日志目录名里 |
| `--resume` | 续训 | 自动找最新 run + 最新 checkpoint |
| `--checkpoint PATH` | 指定模型 | 测试/回放时用 |
| `--cmd VX VY WZ` | 固定指令 | 单位 m/s、m/s、rad/s |
| `--teleop keyboard\|gamepad` | 遥控 | 自动实时 50Hz |
| `--teleop-scale S` | 遥控速度缩放 | 0.5 = 速度减半 |
| `--train-dist` | 用训练配置（含域随机化）测试 | 测鲁棒性 |
| `--headless` | 无窗口 | 测试/导出可加 |

**训练指令范围（契约，不能超）**：
- 速度：vx ∈ [−0.6, 0.6]、vy ∈ [−0.6, 0.6]、ωz ∈ [−1.0, 1.0]
- 姿态/高度：pitch ±10°、roll ±10°、height ±0.07 m（相对站立高度的偏移）

> 注意：横移（vy）靠的是轮子侧向刮擦 —— 轮子轴向固定、没有侧向驱动，
> 所以大侧向速度物理上很难跟准，是**预期行为**，不是策略坏了。

---

## 7. 相关文档

| 文件 | 内容 |
|---|---|
| `rl/README.md` | sim2real 契约表、域随机化范围、调参提示、已知待办 |
| `rl/rp_wd_rl/assets/rp_wd_contract.py` | 契约常量（唯一真相源） |
| `README.md`（仓库根） | 部署端（ROS2 推理节点）构建与运行 |
