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

# Odin1 深度避障 (默认 256 环境、总计 8000 iter、headless)
./rl/scripts/train_odin_depth.sh --run_name odin1_v1
# 续训到总目标 8000；checkpoint 使用指定 run 中的最新模型
./rl/scripts/train_odin_depth.sh --resume --max_iterations 8000 --run_name odin1_curriculum8k
```

- 日志：`rl/logs/train_*.log`（训练 stdout）＋ `rl/logs/rsl_rl/<实验名>/<时间戳>_<run_name>/`（TensorBoard + checkpoint）
- 训练期间**自动阻止系统休眠**（`systemd-inhibit`），夜间不会被冻死
- 关键指标在 TensorBoard 里看 `Episode_Reward/*`（跟踪 vs 惩罚的配比）和 `Episode_Termination/*`
- 深度相机使用离屏渲染；训练仍无 GUI，但比普通任务更占渲染资源
- 深度课程：100–2400 轮扩展速度、1500–4000 轮扩展姿态、1200–5000 轮加入推扰；5600–6800 轮渐增腿部力矩惩罚
- 地形最高等级按 0/1000/2500/4500/6000 轮解锁为 1/3/5/8/11；升级仍依赖实际行进表现，课程详见 `rl/CURRICULUM.md`
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
./rl/scripts/play.sh --task Isaac-Velocity-Rough-RPWD-OdinDepth-Play-v0 --checkpoint rl/logs/rsl_rl/rp_wd_odin_depth/<run>/model_7999.pt
```

**看什么**：`线速误差` 越小越好；`滑移比` 接近 1.0 = 四轮纯滚动（≈0.2 = 在用轮子当脚蹦）；`终止` = 期间摔倒次数。

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
# 终端 2：深度图转换为策略使用的 12×16 特征
ros2 launch roboparty_inference odin_depth_adapter.launch.py
# 终端 3：加载 policy_odin_depth.onnx 与独立输入配置
ros2 launch roboparty_inference inference.launch.py robot:=wheel_quad policy:=odin_depth
```

深度策略输入为原有 56 维观测 + 12 维归一化腿部力矩 + 192 维距离网格，帧栈 3，
ONNX 为 780→16。训练深度增强包含 0.03 m 高斯噪声、1% 像素缺测和 80–100% 随机裁切；
障碍地形混入 rough 课程。实机 adapter 读取 `/odin1/depth_img_competetion`，按 0.2–6 m
裁剪并生成 12×16 最近距离特征。目前仿真相机安装位姿是估计值，需按真实支架校准后再做实机评估。

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
