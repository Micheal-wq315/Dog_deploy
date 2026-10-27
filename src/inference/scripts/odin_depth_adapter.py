#!/usr/bin/env python3
"""Convert Odin1 ROS2 depth images to the trained 12x16 perception vector."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


def min_pool_depth(depth_m: np.ndarray, out_h: int, out_w: int, max_depth: float) -> np.ndarray:
    """Adaptive min-pool matching torch adaptive_max_pool2d(-depth) semantics."""
    h, w = depth_m.shape
    pooled = np.empty((out_h, out_w), dtype=np.float32)
    for row in range(out_h):
        y0 = math.floor(row * h / out_h)
        y1 = math.ceil((row + 1) * h / out_h)
        for col in range(out_w):
            x0 = math.floor(col * w / out_w)
            x1 = math.ceil((col + 1) * w / out_w)
            pooled[row, col] = float(np.min(depth_m[y0:y1, x0:x1]))
    return np.clip(pooled / max_depth, 0.0, 1.0)


class OdinDepthAdapter(Node):
    def __init__(self) -> None:
        super().__init__("odin_depth_adapter")
        self.declare_parameter("input_topic", "/odin1/depth_img_competetion")
        self.declare_parameter("output_topic", "/odin1/depth_features")
        self.declare_parameter("grid_height", 12)
        self.declare_parameter("grid_width", 16)
        self.declare_parameter("max_depth", 6.0)
        self.declare_parameter("min_depth", 0.2)
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.grid_height = int(self.get_parameter("grid_height").value)
        self.grid_width = int(self.get_parameter("grid_width").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.min_depth = float(self.get_parameter("min_depth").value)
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Float32MultiArray, self.output_topic, 1)
        self.subscription = self.create_subscription(
            Image, self.input_topic, self.depth_callback, qos_profile_sensor_data
        )
        self.get_logger().info(
            f"Depth {self.input_topic} -> {self.output_topic}, "
            f"grid={self.grid_height}x{self.grid_width}, range={self.min_depth:.1f}-{self.max_depth:.1f}m"
        )

    def depth_callback(self, msg: Image) -> None:
        try:
            image = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
            if image.ndim == 3:
                image = image[..., 0]
            if msg.encoding in ("16UC1", "mono16") and image.dtype == np.uint16:
                depth_m = image.astype(np.float32) * 0.001  # ROS 16UC1 convention: millimeters
            elif msg.encoding == "32FC1" and image.dtype == np.float32:
                depth_m = image.astype(np.float32, copy=False)  # 32FC1 convention: meters
            else:
                raise ValueError(
                    f"unsupported depth encoding {msg.encoding!r} with dtype {image.dtype}; "
                    "expected 16UC1 (mm) or 32FC1 (m)"
                )
            valid = np.isfinite(depth_m) & (depth_m > 0.0)
            depth_m = np.where(valid, depth_m, self.max_depth)
            depth_m = np.clip(depth_m, self.min_depth, self.max_depth)
            grid = min_pool_depth(depth_m, self.grid_height, self.grid_width, self.max_depth)
            out = Float32MultiArray()
            out.data = grid.reshape(-1).tolist()
            self.publisher.publish(out)
        except Exception as exc:  # malformed frame: keep previous features until timeout
            self.get_logger().warning(f"Could not convert Odin1 depth frame: {exc}")


def main() -> None:
    rclpy.init()
    node = OdinDepthAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
