"""Start Odin1 depth-image to policy-feature adapter."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("input_topic", default_value="/odin1/depth_img_competetion"),
            DeclareLaunchArgument("output_topic", default_value="/odin1/depth_features"),
            DeclareLaunchArgument("rotation_deg", default_value="90"),
            DeclareLaunchArgument("memory_stride", default_value="3"),
            Node(
                package="roboparty_inference",
                executable="odin_depth_adapter.py",
                name="odin_depth_adapter",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("input_topic"),
                        "output_topic": LaunchConfiguration("output_topic"),
                        "grid_height": 16,
                        "grid_width": 12,
                        "rotation_deg": ParameterValue(LaunchConfiguration("rotation_deg"), value_type=int),
                        "memory_frames": 8,
                        "memory_stride": ParameterValue(LaunchConfiguration("memory_stride"), value_type=int),
                        "max_depth": 6.0,
                        "min_depth": 0.2,
                    }
                ],
                output="screen",
            ),
        ]
    )
