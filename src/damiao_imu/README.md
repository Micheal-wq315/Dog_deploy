# DAMIAO IMU ROS2 驱动

## 快速启动

### 1. 硬件连接

USB 线连接 DAMIAO IMU → 确认设备：

```bash
ls /dev/ttyACM*
```

### 2. 串口权限（一次性）

```bash
sudo usermod -a -G dialout $USER   # 然后重新登录
```

### 3. 构建

```bash
source /opt/ros/jazzy/setup.bash
cd /home/rp/imu
PATH=/usr/bin:$PATH colcon build --packages-select damiao_imu --symlink-install
```

### 4. 运行

```bash
source /opt/ros/jazzy/setup.bash
source /home/rp/imu/install/setup.bash

# 方式一
ros2 run damiao_imu damiao_imu_node /dev/ttyACM0

# 方式二
ros2 launch damiao_imu damiao_imu.launch.py device:=/dev/ttyACM0
```

### 5. 查看数据

```bash
ros2 topic echo /imu/data_raw     # 自定义 DamiaoImuData
ros2 topic echo /imu/data         # 标准 sensor_msgs/Imu
ros2 topic hz /imu/data_raw       # 发布频率
```

### 话题

| 话题 | 类型 | 用途 |
|------|------|------|
| `/imu/data_raw` | `damiao_imu/msg/DamiaoImuData` | 加速度、角速度、欧拉角、四元数、温度、设备ID |
| `/imu/data` | `sensor_msgs/msg/Imu` | 标准格式，兼容 imu_filter、robot_localization、rviz2 |

---

## 代码架构

```
damiao_imu/
├── launch/
│   └── damiao_imu.launch.py          # launch 文件
├── msg/
│   └── DamiaoImuData.msg             # 自定义消息定义
├── include/damiao_imu/
│   └── imu_driver.hpp                # 抽象基类 + SensorData 结构体
├── src/
│   └── imu_driver.cpp                # 工厂方法 create_imu()
├── src_node/
│   └── damiao_imu_node.cpp           # ROS2 节点：创建 driver、发布 topic
├── src/drivers/damiao/
│   ├── damiao_protocol.h             # 帧格式、状态机、CRC 函数声明
│   ├── damiao_imu_driver.hpp         # DamiaoIMUDriver 类声明
│   └── damiao_imu_driver.cpp         # 核心：串口配置、逐字节解帧、CRC 校验
└── src/protocol/damiao/
    ├── damiao_serial_port.hpp         # 串口类声明
    └── damiao_serial_port.cpp         # 串口：open/read/write、独立 RX 线程
```

### 分层结构

```
┌────────────────────────────────────────────┐
│  ROS2 层  (damiao_imu_node.cpp)            │
│  创建 DamiaoIMUDriver，注册 callback        │
│  发布 /imu/data_raw 和 /imu/data           │
└──────────────────┬─────────────────────────┘
                   │ callback(SensorData)
┌──────────────────┴─────────────────────────┐
│  驱动层  (DamiaoIMUDriver)                  │
│  - configure_device(): 启动时配置 IMU       │
│  - serial_rx_cbk(): 串口数据入口            │
│  - parse_byte(): 状态机逐字节拼帧            │
│  - process_frame(): CRC 校验 + 解析 float   │
│  - sensor_data_: 持久化，逐字段更新          │
└──────────────────┬─────────────────────────┘
                   │
┌──────────────────┴─────────────────────────┐
│  传输层  (DamiaoSerialPort)                 │
│  - 独立 RX 线程 (SCHED_FIFO)                │
│  - select() 1ms 轮询                        │
│  - 8N1, 460800bps, raw mode                │
└──────────────────┬─────────────────────────┘
                   │ USB CDC-ACM
┌──────────────────┴─────────────────────────┐
│  硬件  (DM-IMU-L1)                          │
│  主动推送四种数据帧                          │
└────────────────────────────────────────────┘
```

### 数据流

1. 硬件按设定频率推送数据帧，四种类型交替发送
2. 串口 RX 线程 `select()` 读取 → 回调 `serial_rx_cbk`
3. 状态机逐字节解析：`WAIT_55 → WAIT_AA → GET_ID → GET_TYPE → COLLECT_DATA → CRC_LO → CRC_HI → FOOTER`
4. `process_frame()`：CRC16 校验通过 → 按帧类型解包 float（小端序）→ 更新 `sensor_data_`
5. 回调通知 ROS2 节点，发布 `/imu/data_raw` 和 `/imu/data`

`sensor_data_` 是持久化的——每次只更新当前帧类型对应的字段（如加速度帧只更新 acc_x/y/z），其余字段保留上一次的值。

### 串口协议

帧格式（来自说明书 V1.2）：

```
55 AA  [ID]  [type]  [data]  [CRC16 LE]  0A
│  │    │      │       │        │         │
│  │    │      │       │        │         └─ 帧尾
│  │    │      │       │        └─ CRC-16 小端序
│  │    │      │       └─ float 数组 (accel/gyro/euler 各 12B, quat 16B)
│  │    │      └─ 帧类型: 01=加速度 02=角速度 03=欧拉角 04=四元数
│  │    └─ 设备 ID
│  └─ 帧头
```

### CRC-16

- **覆盖范围**：`55 AA + ID + type + data`（帧头到 data 末尾，共 4+N 字节）
- **算法**：多项式 `0x1021`，查表法，`crc << 1`，初始值 `0xFFFF`
- **字节序**：CRC 在帧中为小端序（低字节在前）

对应说明书附录四的 `Get_CRC16` 实现。帧中 CRC 不含自身和帧尾 `0A`。

### 设备配置

启动时自动执行配置流程（`configure_device()`）：

1. 进入设置模式
2. 依次使能加速度、角速度、欧拉角、四元数输出
3. 设置 USB 为输出接口
4. 保存参数到 Flash
5. 恢复正常模式开始推流

每步间有延时等待设备响应。
