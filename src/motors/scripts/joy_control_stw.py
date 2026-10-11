#!/usr/bin/env python3
"""Joystick speed control for STW (伺泰威) motor via ROS2 joy_node."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

import motors_py
import sys
import time


class JoyControlNode(Node):
    def __init__(self):
        super().__init__("joy_control_stw")

        self.declare_parameter("motor_id", 1)
        self.declare_parameter("can_interface", "can0")
        self.declare_parameter("speed_axis", 1)      # left stick vertical
        self.declare_parameter("dead_zone", 0.05)     # dead zone threshold
        self.declare_parameter("max_speed", 50.0)     # max speed rad/s
        self.declare_parameter("enable_button", 0)    # A button to enable
        self.declare_parameter("disable_button", 1)   # B button to disable

        motor_id = self.get_parameter("motor_id").value
        can_iface = self.get_parameter("can_interface").value
        self.speed_axis = self.get_parameter("speed_axis").value
        self.dead_zone = self.get_parameter("dead_zone").value
        self.max_speed = self.get_parameter("max_speed").value
        self.enable_btn = self.get_parameter("enable_button").value
        self.disable_btn = self.get_parameter("disable_button").value

        self.motor = motors_py.MotorDriver.create_motor(
            motor_id=motor_id,
            interface_type="can",
            interface=can_iface,
            motor_type="STW",
            motor_model=0,
            master_id_offset=0,
            motor_zero_offset=0.0,
        )

        self.motor_enabled = False
        self.target_speed = 0.0
        self.last_cmd_time = time.time()
        self.cmd_timeout = 0.2  # disable motor if no joy msg for 200ms

        self.joy_sub = self.create_subscription(
            Joy, "/joy", self.joy_callback, 10
        )

        self.control_timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info(
            f"Joystick Control STW ready. motor_id={motor_id}, can={can_iface}"
        )
        self.get_logger().info(
            f"  Speed axis={self.speed_axis}, max={self.max_speed} rad/s"
        )
        self.get_logger().info(
            f"  Enable=Btn{self.enable_btn}, Disable=Btn{self.disable_btn}"
        )

    def joy_callback(self, msg: Joy):
        if len(msg.axes) <= self.speed_axis:
            return

        raw = msg.axes[self.speed_axis]
        # Apply dead zone
        if abs(raw) < self.dead_zone:
            raw = 0.0
        # Invert: pushing stick forward (negative) → positive speed
        self.target_speed = -raw * self.max_speed
        self.last_cmd_time = time.time()

        if len(msg.buttons) > self.enable_btn and msg.buttons[self.enable_btn]:
            self.enable_motor()
        if len(msg.buttons) > self.disable_btn and msg.buttons[self.disable_btn]:
            self.disable_motor()

    def enable_motor(self):
        if self.motor_enabled:
            return
        self.get_logger().info("Enabling motor...")
        self.motor.set_motor_control_mode(motors_py.MotorControlMode.SPD)
        time.sleep(0.05)
        self.motor.lock_motor()
        self.motor_enabled = True
        self.get_logger().info("Motor enabled (SPEED mode, closed-loop)")

    def disable_motor(self):
        if not self.motor_enabled:
            return
        self.get_logger().info("Disabling motor...")
        self.motor.unlock_motor()
        self.motor_enabled = False
        self.target_speed = 0.0
        self.get_logger().info("Motor disabled")

    def control_loop(self):
        if not self.motor_enabled:
            return

        now = time.time()
        if now - self.last_cmd_time > self.cmd_timeout:
            self.target_speed = 0.0
            self.get_logger().warn("Joystick timeout, speed set to 0")

        self.motor.motor_spd_cmd(self.target_speed)

    def shutdown(self):
        self.get_logger().info("Shutting down...")
        if self.motor_enabled:
            # Ramp down
            for _ in range(30):
                self.target_speed *= 0.8
                self.motor.motor_spd_cmd(self.target_speed)
                time.sleep(0.01)
            self.motor.unlock_motor()
        self.destroy_node()


def main():
    rclpy.init(args=sys.argv)
    node = JoyControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C pressed, exiting...")
    finally:
        node.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
