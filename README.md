# Dog_deploy — 四轮足机器人推理部署

基于 ROS2 Jazzy + ONNX Runtime 的机器人强化学习策略推理部署系统，支持 DAMIAO IMU 和 CAN/CAN-FD 电机驱动。

## 项目结构

```
Dog_deploy/
├── src/
│   ├── damiao_imu/       # DAMIAO IMU ROS2 驱动 (库 + 独立发布节点)
│   ├── motors/           # roboparty_motors — 电机驱动库 (CAN/CAN-FD)
│   └── inference/        # roboparty_inference — ONNX 推理节点
│       ├── include/      # RobotInterface 等公共头文件
│       ├── src/          # 推理节点主逻辑
│       ├── launch/       # launch 文件
│       ├── robots/       # 机器人配置文件 (robot.yaml + 策略 YAML)
│       └── thirdparty/   # onnxruntime, cnpy 等第三方库
├── scripts/              # 工具脚本
└── .vscode/              # IDE 配置
```

### 三个功能包

| 包名 | 路径 | CMake project | 说明 |
|------|------|---------------|------|
| `damiao_imu` | `src/damiao_imu/` | `damiao_imu` | IMU 驱动基类 + DAMIAO 串口实现 + ROS2 发布节点 |
| `roboparty_motors` | `src/motors/` | `roboparty_motors` | LRO/RS02 电机驱动 (SocketCAN/CAN-FD) |
| `roboparty_inference` | `src/inference/` | `roboparty_inference` | ONNX 推理节点, 机器人硬件抽象层 |

### 依赖关系

```
roboparty_inference
    ├── damiao_imu        (IMUDriver 基类 + 工厂)
    └── roboparty_motors  (MotorDriver 基类 + 工厂)
```

## 环境要求

| 依赖 | 版本 | 安装方式 |
|------|------|---------|
| Ubuntu | 24.04 | — |
| ROS2 | Jazzy | [官方安装](https://docs.ros.org/en/jazzy/Installation.html) |
| GCC | ≥11 | 系统自带 |
| CMake | ≥3.16 | 系统自带 |
| Eigen3 | ≥3.3 | `apt install libeigen3-dev` |
| fmt | ≥8.0 | `apt install libfmt-dev` |
| Boost | ≥1.74 | `apt install libboost-system-dev` |
| pybind11 | — | `apt install pybind11-dev` |
| Python3 | ≥3.10 | 系统自带 |
| ONNX Runtime | 1.21+ | 已内置在 `thirdparty/` 中 |

### 一键安装依赖

```bash
sudo apt install -y libeigen3-dev libfmt-dev libboost-system-dev pybind11-dev ccache
```

## 构建

### 1. 构建 damiao_imu 和 motors

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select damiao_imu roboparty_motors
```

### 2. 构建 inference (依赖前两步的产物)

```bash
source install/setup.bash
colcon build --packages-select roboparty_inference
```

### 3. 一键全量构建

```bash
source /opt/ros/jazzy/setup.bash
rm -rf build install log
colcon build --packages-select damiao_imu roboparty_motors
source install/setup.bash
colcon build --packages-select roboparty_inference
```

## 强化学习训练 (策略来源)

策略在本仓库的 [rl/](rl/) 目录里用 Isaac Lab 2.3.2 + RSL-RL PPO 训练，导出的 ONNX 与推理节点
逐项对齐（观测 159 维 / 动作 16 维 / 关节顺序 / 动作缩放 / PD 增益 / 控制周期），
由 `rl/scripts/verify_contract.py` 自动复核。

```bash
./rl/scripts/verify_contract.sh   # sim2real 契约复核 (训练侧 ↔ 部署侧, ~25 秒)
./rl/scripts/train.sh             # 训练 (强制 headless, 不弹窗口)
./rl/scripts/play.sh              # 回放 + 导出 ONNX
cp rl/exported/policy_flat.onnx src/inference/robots/wheel_quad/models/policy_flat.onnx
```

完整契约表、域随机化范围与调参提示见 [rl/README.md](rl/README.md)。

## 运行

### 推理节点 (机器人控制)

启动推理节点，直接读取 IMU 和电机硬件：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# wheel_quad 机器人 (默认)
ros2 launch roboparty_inference inference.launch.py

# rpo 机器人
ros2 launch roboparty_inference inference.launch.py robot:=rpo

# 指定策略
ros2 launch roboparty_inference inference.launch.py robot:=wheel_quad policy:=default
```

### IMU 独立调试

仅运行 IMU 数据发布（不启动推理）：

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch damiao_imu damiao_imu.launch.py device:=/dev/ttyACM0
ros2 topic echo /imu/data
```

### 手柄操作

| 按钮 | 功能 |
|------|------|
| **X** (buttons[1]) | 启动/暂停推理 |
| **B** (buttons[2]) | 电机初始化/反初始化 |
| **A** (buttons[0]) | 复位关节到默认姿态 |
| **Y** (buttons[3]) | 切换手柄控制 ↔ /cmd_vel 话题控制 |
| **左摇杆上下** (axes[4]) | 前进/后退 |
| **左摇杆左右** (axes[3]) | 左右平移 |
| **LT/RT** (axes[2]/axes[5]) | 左转/右转 |

### ROS2 Service 接口

推理节点提供以下 service 用于外部控制：

```bash
ros2 service call /init_motors    std_srvs/srv/Trigger  # 初始化电机
ros2 service call /deinit_motors  std_srvs/srv/Trigger  # 反初始化电机
ros2 service call /reset_joints   std_srvs/srv/Trigger  # 复位关节
ros2 service call /stand_up       std_srvs/srv/Trigger  # 缓慢站立
ros2 service call /set_zeros      std_srvs/srv/Trigger  # 设置零点
ros2 service call /clear_errors   std_srvs/srv/Trigger  # 清除故障
ros2 service call /read_joints    std_srvs/srv/Trigger  # 发布关节状态
ros2 service call /read_imu       std_srvs/srv/Trigger  # 发布IMU数据
ros2 service call /start_inference std_srvs/srv/Trigger # 启动推理
ros2 service call /stop_inference  std_srvs/srv/Trigger # 停止推理
```

### 话题

| 话题 | 类型 | 发布者 | 用途 |
|------|------|--------|------|
| `/action` | `sensor_msgs/JointState` | inference_node | 目标关节位置 (电机控制) |
| `/joint_states` | `sensor_msgs/JointState` | inference_node | 当前关节状态 |
| `/imu` | `sensor_msgs/Imu` | inference_node | IMU 姿态+角速度 |
| `/cmd_vel` | `geometry_msgs/Twist` | 外部节点 | 速度指令 (非手柄模式) |
| `/joy` | `sensor_msgs/Joy` | 手柄节点 | 手柄输入 |
| `/imu/data` | `sensor_msgs/Imu` | damiao_imu_node | IMU 调试数据 |
| `/imu/data_raw` | `damiao_imu/DamiaoImuData` | damiao_imu_node | IMU 原始数据 |

## 配置

### 机器人硬件配置 (robot.yaml)

`robots/<robot_name>/robot.yaml` 定义：

- **imu**: IMU 型号、接口类型、设备路径
- **motors**: 电机 CAN 总线拓扑、ID 分配、零位偏移
- **robot**: PD 增益、电机方向、URDF 索引映射、IMU 外参

### 策略配置 (configs/<policy>.yaml)

`robots/<robot_name>/configs/<policy>.yaml` 定义：

- **model_names**: ONNX 模型文件列表 (支持多策略)
- **obs_layouts**: 观测空间分量定义 (`ang_vel:3, gravity_b:3, ...`)
- **frame_stacks**: 历史帧数 (10 帧 FrameMajor 或 3 帧 ObsMajor)
- **控制参数**: 动作缩放、截断、降采样等

### 配置示例

当前项目包含两个机器人：

| 机器人 | 关节数 | 说明 |
|--------|--------|------|
| `wheel_quad` | 16 | 四轮足机器人 (LRO_CAN + RS02 电机) |
| `rpo` | 23 | 双足机器人 (LRO CAN-FD 电机) |

## 线程模型

```
main        SCHED_FIFO 50   ROS2 回调 (话题订阅 + Service)
executor₀   默认调度        手柄/cmd_vel/joint_state 回调
executor₁   默认调度        Service 回调
control     SCHED_FIFO 70   动作发布 250Hz
inference   SCHED_FIFO 70   ONNX 推理 50Hz
```

## 实时性要求

- 控制周期: `dt × decimation = 0.004 × 5 = 0.02s` (50Hz 推理)
- 动作发布: 250Hz (`1/dt`)
- `mlockall()` 锁定内存防止 swap 抖动
- 建议使用实时内核 (`linux-image-rt`) 以获得最佳实时性能

## RDK5 迁移注意事项

参见 [RDK5 兼容性分析](#) 了解从 x86_64/ROS2 Jazzy 迁移到 ARM64/ROS2 Humble 的注意事项：

- ONNX Runtime 需要替换为 ARM64 版本
- fmt 版本降级 (9→8)，不影响使用
- damiao_imu 需要在 Humble 下重新 colcon build
- `offline_threshold_` 建议调大以容忍 ARM 实时性差异
- CAN 接口可能不同于 x86_64 设备
