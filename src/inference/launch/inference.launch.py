# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2025-2026 Luo1imasi

# ============================================================================
# inference.launch.py — ROS2 launch 文件, 启动 inference_node
#
# 职责:
#   1. 接收命令行参数 (robot, policy)
#   2. 校验参数合法性 (防注入)
#   3. 定位 robot.yaml 和 policy.yaml 的路径
#   4. 将路径和参数注入 ROS2 参数服务器, 启动推理节点
#
# 使用示例:
#   ros2 launch roboparty_inference inference.launch.py
#   ros2 launch roboparty_inference inference.launch.py robot:=rpo policy:=beyondmimic
#   ros2 launch roboparty_inference inference.launch.py robot:=rpo policy:=default.yaml
# ============================================================================

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import re

# ---------------------------------------------------------------------------
# 正则: 限制 robot 和 policy 名称只能含字母/数字/下划线/点/连字符
# 防止路径遍历攻击 (如 robot:=../../etc/passwd)
# ---------------------------------------------------------------------------
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# ============================================================================
# launch_setup — 延迟执行的启动逻辑 (OpaqueFunction 回调)
#
# 为什么用 OpaqueFunction 而不是直接在 generate_launch_description 里写?
#   - LaunchConfiguration 的值在声明时还未解析 (只是占位符)
#   - .perform(context) 必须在 launch 系统准备好 context 之后才能调用
#   - OpaqueFunction 保证在参数值已就绪后才执行
#
# 参数:
#   context: ROS2 launch 上下文, 包含所有 launch 参数的值
# ============================================================================
def launch_setup(context, *args, **kwargs):

    # ── 步骤 1: 从 launch 上下文获取参数实际值 ──────────────────────────
    # .perform(context) 将 LaunchConfiguration 占位符替换为实际命令行参数
    robot = LaunchConfiguration("robot").perform(context)     # 如 "rpo"
    policy = LaunchConfiguration("policy").perform(context)   # 如 "beyondmimic"

    # ── 步骤 2: 安全校验 — 防路径遍历攻击 ───────────────────────────────
    # 如果有人传 robot:=../../etc, 拼接出的路径会指向系统目录
    # 正则限制只允许合法文件名, 防止读取任意文件
    if not NAME_PATTERN.fullmatch(robot):
        raise ValueError(f"Invalid robot name: {robot}")
    if not NAME_PATTERN.fullmatch(policy):
        raise ValueError(f"Invalid policy name: {policy}")

    # ── 步骤 3: 策略文件名规范化 ─────────────────────────────────────────
    # "beyondmimic" → "beyondmimic.yaml"
    # "default.yaml" → "default.yaml"  (已经带 .yaml 后缀的不重复加)
    policy_file = policy if policy.endswith(".yaml") else f"{policy}.yaml"

    # ── 步骤 4: 路径拼接 ─────────────────────────────────────────────────
    # 通过 ament_index 找到包的安装目录 (install/share/roboparty_inference)
    robot_dir = os.path.join(
        get_package_share_directory("roboparty_inference"),
        "robots",
        robot,                          # 如 "rpo"
    )
    # robot_dir = ".../share/roboparty_inference/robots/rpo"

    robot_config = os.path.join(robot_dir, "robot.yaml")
    # → "/path/to/share/roboparty_inference/robots/rpo/robot.yaml"

    policy_config = os.path.join(robot_dir, "configs", policy_file)
    # → "/path/to/share/roboparty_inference/robots/rpo/configs/beyondmimic.yaml"

    # ── 步骤 5: 文件存在性检查 ──────────────────────────────────────────
    # 提前失败, 给出清晰的错误信息 (而不是等节点启动后才发现文件不存在)
    if not os.path.isfile(robot_config):
        raise FileNotFoundError(f"Robot config not found: {robot_config}")
    if not os.path.isfile(policy_config):
        raise FileNotFoundError(f"Inference config not found: {policy_config}")

    # ── 步骤 6: 返回要启动的 ROS2 节点列表 ──────────────────────────────
    return [
        Node(
            package="roboparty_inference",      # ROS2 包名 (package.xml 中定义)
            executable="inference_node",        # 可执行文件名 (CMakeLists.txt 中 add_executable)
            name="inference_node",              # 节点实例名 (可在 ros2 node list 中看到)
            parameters=[
                # parameters 列表中的元素按顺序加载:
                #   ① 先加载 YAML 文件 (策略配置, 大部分参数在里面)
                #   ② 再加载内联 dict (这些参数覆盖 YAML 中的同名参数)
                #
                policy_config,                   # 如 robots/rpo/configs/beyondmimic.yaml
                {
                    # 以下参数由 launch 动态确定, 不在 YAML 里写死:
                    "robot_name": robot,         # → "rpo", 用于日志和文件查找
                    "policy_name": policy,       # → "beyondmimic", 用于日志
                    "robot_config": robot_config, # → robot.yaml 完整路径
                    "model_dir": os.path.join(robot_dir, "models"),
                    #              → .../robots/rpo/models/ (ONNX 文件目录)
                    "motion_dir": os.path.join(robot_dir, "motions"),
                    #              → .../robots/rpo/motions/ (NPZ 文件目录)
                },
            ],
            output="screen",                     # 节点 stdout/stderr 输出到终端
            # prefix=["xterm -e gdb -ex run --args"],  # 调试用: 在 gdb 中运行
        ),
    ]


# ============================================================================
# generate_launch_description — ROS2 launch 系统入口 (必须函数)
#
# LaunchDescription 包含:
#   ① DeclareLaunchArgument × 2: 声明两个命令行参数 (带默认值)
#   ② OpaqueFunction: 延迟执行 launch_setup, 等参数值解析完成后才运行
# ============================================================================
def generate_launch_description():
    return LaunchDescription(
        [
            # ── 声明 launch 参数 ────────────────────────────────────────
            # 格式: ros2 launch ... robot:=rpo policy:=beyondmimic
            DeclareLaunchArgument(
                "robot",
                default_value="wheel_quad",     # 默认机器人名称
                description="Robot name (must be a valid directory in robots/).",
            ),
            DeclareLaunchArgument(
                "policy",
                default_value="default",         # 默认策略名称
                description="Policy name (YAML file in robots/<robot>/configs/).",
            ),

            # ── 延迟执行 ────────────────────────────────────────────────
            # 此时 LaunchConfiguration 的值尚未解析 (只是字符串占位符)
            # OpaqueFunction 在 launch context 就绪后回调 launch_setup
            OpaqueFunction(function=launch_setup),
        ]
    )

# ============================================================================
# 参数解析链路总结:
#
#   命令行:
#     ros2 launch inference.launch.py robot:=rpo policy:=beyondmimic
#       ↓
#   DeclareLaunchArgument("robot"), DeclareLaunchArgument("policy")
#       ↓
#   launch_setup(context):
#     robot = LaunchConfiguration("robot").perform(context)   → "rpo"
#     policy = LaunchConfiguration("policy").perform(context) → "beyondmimic"
#       ↓
#     路径拼接:
#     robot_dir = .../share/roboparty_inference/robots/rpo/
#     policy_config = robot_dir/configs/beyondmimic.yaml
#       ↓
#   Node parameters = [policy_config, {"robot_config": ..., "model_dir": ...}]
#       ↓
#   inference_node 构造函数:
#     load_config() → get_parameter("robot_config") → RobotInterface(robot_config_path_)
# ============================================================================
