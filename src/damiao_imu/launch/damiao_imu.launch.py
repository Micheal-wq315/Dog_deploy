from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    device_arg = DeclareLaunchArgument(
        'device', default_value='/dev/ttyACM0',
        description='DAMIAO IMU USB serial device path')

    imu_node = Node(
        package='damiao_imu',
        executable='damiao_imu_node',
        name='damiao_imu_node',
        output='screen',
        arguments=[LaunchConfiguration('device')])

    return LaunchDescription([device_arg, imu_node])
