# SPDX-License-Identifier: GPL-3.0
# Copyright (C) 2025-2026 Luo1imasi

# ============================================================================
# dji_pad.launch.py — 一条命令启动 DJI 手柄 + 推理节点 (实机手柄遥控)
#
# 职责:
#   1. 启动 dji_pad_publisher.py (DJI 手柄 → /joy, Xbox 兼容布局)
#   2. 复用 inference.launch.py 启动推理节点 (订阅 /joy 驱动机器人)
#
# 使用示例:
#   ros2 launch roboparty_inference dji_pad.launch.py
#   ros2 launch roboparty_inference dji_pad.launch.py robot:=wheel_quad policy:=default
#   ros2 launch roboparty_inference dji_pad.launch.py wz_axis:=left_x calibrate_wz:=true
#   ros2 launch roboparty_inference dji_pad.launch.py start_inference:=false   # 只发 /joy 调试
#
# 为什么必须用 /usr/bin/python3:
#   本机 rclpy 只有系统 Python 3.10 的扩展, conda 的 3.13 会报 _rclpy_pybind11 缺失。
#
# 轴映射 (2026-10-12 实测, 详见 dji_pad_publisher.py 文件头):
#   右摇杆 纵/横 → vx / vy      左摇杆纵向 → wz (自回中)
#   左摇杆横向 ABS_Z 空闲 (该轴松手不回中, 静止读数 -0.36, 用它做转向会导致"没人碰也自转")
# ============================================================================

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# ---------------------------------------------------------------------------
# 手柄发布者用系统 Python (rclpy 只有 3.10 扩展)
# ---------------------------------------------------------------------------
SYSTEM_PYTHON = "/usr/bin/python3"


def generate_launch_description():
    share = get_package_share_directory("roboparty_inference")
    publisher_script = os.path.join(share, "scripts", "dji_pad_publisher.py")
    inference_launch = os.path.join(share, "launch", "inference.launch.py")

    declared = [
        DeclareLaunchArgument("robot", default_value="wheel_quad", description="机器人配置名"),
        DeclareLaunchArgument("policy", default_value="default", description="策略配置名"),
        DeclareLaunchArgument(
            "start_inference", default_value="true",
            description="是否同时启动推理节点 (false = 只发 /joy, 用于单独调试手柄)",
        ),
        DeclareLaunchArgument(
            "wz_axis", default_value="left_y",
            description="转向取自哪个摇杆轴: left_y(默认,自回中) / left_x / right_x / right_y / dial",
        ),
        DeclareLaunchArgument("deadzone", default_value="0.10", description="摇杆死区"),
        DeclareLaunchArgument(
            "calibrate_wz", default_value="true",
            description="启动时对转向/姿态轴做中点标定 (补偿机械偏置)",
        ),
        DeclareLaunchArgument(
            "pitch_axis", default_value="left_x",
            description="pitch 来源轴 (默认左摇杆横向; 该轴不回中, 语义=配平)",
        ),
        DeclareLaunchArgument(
            "roll_axis", default_value="dial",
            description="roll 来源轴 (默认拨轮; 不回中, 语义=配平)",
        ),
        DeclareLaunchArgument(
            "udp_port", default_value="5005",
            description="UDP 转发端口 (给仿真侧 test_policy.py --teleop udp; 0=关闭)",
        ),
    ]

    # ── 手柄发布者 ────────────────────────────────────────────────────────
    joy_publisher = ExecuteProcess(
        cmd=[
            SYSTEM_PYTHON, publisher_script,
            "--ros-args",
            "-p", ["wz_axis:=", LaunchConfiguration("wz_axis")],
            "-p", ["deadzone:=", LaunchConfiguration("deadzone")],
            "-p", ["calibrate_wz:=", LaunchConfiguration("calibrate_wz")],
            "-p", ["pitch_axis:=", LaunchConfiguration("pitch_axis")],
            "-p", ["roll_axis:=", LaunchConfiguration("roll_axis")],
            "-p", ["udp_port:=", LaunchConfiguration("udp_port")],
        ],
        name="dji_pad_publisher",
        output="screen",
    )

    # ── 推理节点 (复用既有 launch, 保证参数校验/路径解析只有一份实现) ──────
    inference = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(inference_launch),
        launch_arguments={
            "robot": LaunchConfiguration("robot"),
            "policy": LaunchConfiguration("policy"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("start_inference")),
    )

    return LaunchDescription(declared + [joy_publisher, inference])
