# Odin1 + jie_3d_nav 集成方案（统一 ROS2 Humble）

> 目标：给 rpo / wheel_quad 机器人加「导航 + 感知」，前端用留形科技 Odin1 空间记忆模组，导航用 jie_3d_nav（OctoMap 3D 规划），执行端复用本仓库现有的 ONNX 推理节点。全链路统一在 ROS2 Humble / Ubuntu 22.04。

## 1. 总体架构

```
┌─────────────┐   point cloud / pose(≤1000Hz)   ┌──────────────────┐
│  Odin1      │ ───────────────────────────────▶ │ odin_ros_driver  │
│ (LiDAR+视觉) │                                  │ (Humble 驱动)     │
└─────────────┘                                  └────────┬─────────┘
                                                          │ /odom pose + /points + /depth
                                                          ▼
                                          ┌──────────────────────────────┐
                                          │ jie_octomap (OctoMap 建图/管理) │
                                          └──────────────┬───────────────┘
                                                         │ occupancy grid
                                                         ▼
                                          ┌──────────────────────────────┐
                                          │ octo_planner (3D 路径规划+跟踪) │
                                          └──────────────┬───────────────┘
                                                         │ cmd_vel (twist)
                                                         ▼
┌────────────────┐   MIT 指令(CAN)   ┌────────────────────────────────┐
│ roboparty_motors│ ◀─────────────── │ roboparty_inference (ONNX 推理) │
│  (DM/RS02 电机) │                  │  订阅 cmd_vel, 输出关节力矩指令   │
└────────────────┘                  └───────────────▲────────────────┘
                                                     │ /robot_0/elevation_data (187维)
                            elevation 转换节点 ◀────────┘  (从 Odin1 点云/深度生成)
```

三条链路：
1. **定位/建图**：Odin1 → odin_ros_driver → odom pose + 点云 → jie_octomap（OctoMap）。
2. **导航规划**：octo_planner 在 OctoMap 上做 3D 路径规划 + 跟踪 → 输出 `cmd_vel` → 现有 inference 节点（已订阅 cmd_vel，见 `ros_interface.cpp` cmd_vel 回调）。
3. **感知（可选，接 attn_enc）**：Odin1 点云/深度 → 高程图 → `/robot_0/elevation_data`（187 维 Float32MultiArray）→ attn_enc 策略。

## 2. 复用 / 替换 / 新建

| 类别 | 内容 | 说明 |
|---|---|---|
| **直接复用** | `odin_ros_driver` | Odin1 官方 ROS2 驱动（Humble） |
| **直接复用** | `jie_map_msgs` | 地图保存/加载/导出 service（机器人无关） |
| **直接复用** | `jie_octomap` | OctoMap 导入/管理/编辑 + Web/Qt GUI（机器人无关） |
| **直接复用** | `octo_planner` | 3D 路径规划 + 跟踪 + nav launch（机器人无关） |
| **替换** | `d1_bringup` / `d1_description` / `d1_controller` | jie_3d_nav 自带的智元 D1 专用包 → 换成本仓库 `roboparty_motors` + `roboparty_inference` + IMU |
| **替换** | `publish_d1_odom` 逻辑 | D1 用自己的 odom 发布 → 改成用 Odin1 pose 发布 odom→base_link |
| **新建** | `elevation 转换节点` | 点云/深度 → 187 维高程数组 → `/robot_0/elevation_data`（仅 attn_enc 需要） |
| **新建（可能）** | odom 桥接/重映射 | 若 octo_planner 期望的 odom topic 与 Odin1 驱动输出不一致，加一个轻量 remap/转发 |

## 3. 话题接线（topic 名以实际 odin_ros_driver / octo_planner 源码为准）

| 段 | 来源 | 话题（待确认） | 目的地 | 说明 |
|---|---|---|---|---|
| 定位 | odin_ros_driver | `/odom`（或 `/odin/odom`） | octo_planner + TF | Odin1 高频位姿，作为 odom→base_link |
| 定位 | odin_ros_driver | `/points` / 点云话题 | jie_octomap | OctoMap 建图输入 |
| 定位 | odin_ros_driver | `/depth` / `/rgb` | elevation 节点 | 生成高程图 |
| 导航 | octo_planner | `/cmd_vel` | roboparty_inference | Twist；inference 已按 `clip_cmd=[-0.4,0.6,-0.4,0.4,-0.8,0.8]` 裁剪 |
| 感知 | elevation 节点 | `/robot_0/elevation_data` | roboparty_inference | Float32MultiArray，187 维 |
| 地图 | octo_planner → jie_map_msgs | `*/save_map` 等 srv | jie_octomap | 空间记忆保存/重载（relocalization） |

**必须对照代码确认的项**：odin_ros_driver 实际发布的位姿/点云 topic 名与 frame_id；octo_planner 实际订阅的 odom topic 名、发布的 cmd_vel topic 名与消息类型（`geometry_msgs/Twist` 还是 `TwistStamped`）。这些决定 remap 怎么写，不确认无法直接 launch。

## 4. TF 与坐标系

```
map ──(Odin1 relocalize 后)──▶ odom ──▶ base_link ──▶ imu / lidar
```

- **SLAM 模式**（`odin1_slam.launch.py`）：只有 odom→base_link，无全局 map。
- **重定位模式**（`odin1_loc.launch.py`，`control_command.yaml` 里 `custom_map_mode: 2`）：加载空间记忆地图，Odin1 提供 map→odom（或 map→base_link）。
- **替换点**：jie_3d_nav 的 `use_static_odom_to_base` / `publish_d1_odom` 是给 D1 用的；我们改为「Odin1 pose → odom→base_link」，并保证与电机/IMU 的 base_link 定义一致（否则腿的姿态和导航位姿会打架）。

## 5. 关键配置改动

- `octo_planner/config/nav_params.yaml`：`relocalization_bin_file` / `map_package_dir` / `relocalization_pcd_file` 指向实际地图包；`publish_d1_odom: false`；`use_static_odom_to_base` 按实际 TF 源设置。
- `roboparty_inference` 的 config yaml：用 `default.yaml`（纯本体 proprio）跑导航；要接感知则用 `attn_enc.yaml`（`extra_obs_layouts: perception:187`）。
- **速度约定对齐**（重点）：现有 inference 的 cmd_vel 是**本体坐标系**的 `[vx, vy, ωz]`，且 vx 范围 [-0.4, 0.6]（前向 0.6 是主运动方向）。octo_planner 输出的速度是 map 系还是 base_link 系、单位是否 m/s·rad/s，必须核对；若在 map 系下给出，需转成 base_link 系（乘当前 yaw）再下发。RL 策略对「侧向速度 vy」的容忍很小（clip 到 ±0.4），规划端应尽量抑制侧向分量。

## 6. 高程感知（attn_enc）的落地难点

- `attn_enc.yaml` 订阅 `/robot_0/elevation_data`，`perception:187`，且 `frame_stacks: 5`、`frame_major`。意味着 elevation 节点要**按控制频率（约 50Hz，`dt×decimation = 0.004×5 = 0.02s`）持续发布** 187 维数组。
- **187 维的具体排布（多宽×多长网格、分辨率、单位）本部署仓库里没有**——它来自训练侧（IsaacLab 的高程/height-map obs 定义）。写 elevation 节点前必须先拿到训练侧对 `perception` obs 的精确定义，否则喂进去的 187 个数布局对不上，策略输出会是乱的。这是整个感知链路里风险最高的一项。

## 7. 落地步骤（checklist）

1. [ ] 系统统一 Ubuntu 22.04 + ROS2 Humble；本仓库三包（damiao_imu / roboparty_motors / roboparty_inference）迁 Humble 重编译（改 `/opt/ros/humble/` 路径即可，见迁移说明）。
2. [ ] 装 `odin_ros_driver`，先单独跑通 Odin1：确认位姿/点云/深度/RGB 的实际 topic 名、frame_id、频率。
3. [ ] 装 `jie_map_msgs` / `jie_octomap` / `octo_planner`（跳过 D1 依赖，参考其 `web_test.launch.py` 的做法把 D1 摘掉）。
4. [ ] 跑 `odin1_slam.launch.py` 建图：确认 OctoMap 能正常导入 Odin1 点云。
5. [ ] 接执行端：把 octo_planner 的 cmd_vel remap/桥接到 roboparty_inference 的 cmd_vel 订阅；核对速度帧与单位。
6. [ ] TF 对齐：Odin1 pose → odom→base_link，替代 `publish_d1_odom`；与 IMU/电机的 base_link 对齐。
7. [ ] 空间记忆：`odin1_loc.launch.py` + `custom_map_mode: 2` 跑重定位，验证建好的地图可复用。
8. [ ] （可选）attn_enc 感知：先拿到训练侧 perception obs 定义 → 写 elevation 转换节点 → 连 `/robot_0/elevation_data` → 用 `attn_enc.yaml` 实车/仿真验证。

## 8. 风险与待确认

| 风险 | 等级 | 说明 |
|---|---|---|
| perception 187 维布局未知 | 高 | 必须从训练侧拿 obs 定义，否则 attn_enc 不可用 |
| cmd_vel 坐标系/单位不一致 | 高 | map 系→base_link 需转，单位需 m/s·rad/s |
| topic 名/frame_id 未核对 | 中 | 直接决定 remap；先跑 driver 打印一遍 |
| Odin1 pose 与电机/IMU base_link 不一致 | 中 | 腿姿态与导航位姿冲突 |
| Humble EOL 2027-05 | 低 | 只剩约 7 个月支持，长期维护需计划 |

## 9. 参考

- 本仓库：`src/inference/robots/rpo/configs/attn_enc.yaml`（感知 obs 定义）、`src/inference/src/ros_interface.cpp`（cmd_vel / elevation 回调）、`src/inference/robots/rpo/robot.yaml`（kp/kd、电机映射）。
- jie_3d_nav：<https://github.com/6-robot/jie_3d_nav>
- Odin1 / 留形科技（Manifold Tech）：空间记忆模组，配套 `odin_ros_driver`。
