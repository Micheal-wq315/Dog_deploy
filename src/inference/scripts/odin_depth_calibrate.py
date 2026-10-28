#!/usr/bin/env python3
"""采集一帧 Odin1 ROS 深度图，生成旋转方向和策略网格候选图。

先 source ROS 2 和工作空间，再运行：
    python3 src/inference/scripts/odin_depth_calibrate.py --topic /odin1/depth_img_competetion

采集前将已知障碍放在相机正前方。本工具不猜测相机安装方向，会保存四种候选供检查。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def decode_depth(msg: Image, bridge: CvBridge) -> np.ndarray:
    image = np.asarray(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))
    if image.ndim == 3:
        if image.shape[-1] != 1:
            raise ValueError(f"期望单通道深度图，收到 shape={image.shape}")
        image = image[..., 0]
    if msg.encoding in ("16UC1", "mono16") and image.dtype == np.uint16:
        depth_m = image.astype(np.float32) * 0.001
    elif msg.encoding == "32FC1" and image.dtype == np.float32:
        depth_m = image.astype(np.float32, copy=False)
    else:
        raise ValueError(
            f"不支持 encoding={msg.encoding!r}, dtype={image.dtype}; "
            "当前仅处理16UC1/mono16毫米或32FC1米"
        )
    return depth_m.copy()


class DepthCapture(Node):
    def __init__(self, topic: str, target_frames: int):
        super().__init__("odin_depth_calibrate")
        self.bridge = CvBridge()
        self.target_frames = target_frames
        self.frames: list[tuple[np.ndarray, dict]] = []
        self.subscription = self.create_subscription(
            Image, topic, self.callback, qos_profile_sensor_data
        )
        self.get_logger().info(f"等待深度图: {topic}")

    def callback(self, msg: Image) -> None:
        if len(self.frames) >= self.target_frames:
            return
        try:
            depth = decode_depth(msg, self.bridge)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            info = {
                "encoding": msg.encoding,
                "dtype": str(depth.dtype),
                "shape_hw": list(depth.shape),
                "step_bytes": int(msg.step),
                "frame_id": msg.header.frame_id,
                "stamp_sec": stamp,
            }
            self.frames.append((depth, info))
            self.get_logger().info(
                f"收到深度帧 {len(self.frames)}/{self.target_frames}: "
                f"{msg.encoding}, {depth.shape[1]}x{depth.shape[0]}"
            )
        except Exception as exc:
            self.get_logger().error(f"深度图解码失败: {exc}")


def min_pool(depth: np.ndarray, out_h: int, out_w: int, max_depth: float) -> np.ndarray:
    h, w = depth.shape
    result = np.empty((out_h, out_w), dtype=np.float32)
    for y in range(out_h):
        y0, y1 = math.floor(y * h / out_h), math.ceil((y + 1) * h / out_h)
        for x in range(out_w):
            x0, x1 = math.floor(x * w / out_w), math.ceil((x + 1) * w / out_w)
            result[y, x] = np.min(depth[y0:y1, x0:x1])
    return result


def colorize(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    safe = np.where(valid, depth, max_depth)
    normalized = np.clip((safe - min_depth) / (max_depth - min_depth), 0.0, 1.0)
    image = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = (0, 0, 0)
    return image


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def save_previews(depth: np.ndarray, output: Path, min_depth: float, max_depth: float) -> dict:
    rotations = ((0, 0), (90, 1), (180, 2), (270, 3))
    raw_bgr = colorize(depth, min_depth, max_depth)
    raw_path = output / "depth_raw.png"
    cv2.imwrite(str(raw_path), raw_bgr)
    valid = np.isfinite(depth) & (depth > 0.0)
    grid_source = np.where(valid, np.clip(depth, min_depth, max_depth), max_depth).astype(np.float32)

    orientation_panels = []
    grid_panels = []
    candidates = {}
    for degrees, k in rotations:
        corrected = np.rot90(depth, k=k).copy()
        corrected_grid_source = np.rot90(grid_source, k=k).copy()
        grid_m = min_pool(corrected_grid_source, 16, 12, max_depth)
        grid_norm = np.clip(grid_m / max_depth, 0.0, 1.0)
        np.save(output / f"grid_{degrees}deg_ccw_m.npy", grid_m)
        candidates[str(degrees)] = {
            "rotated_shape_hw": list(corrected.shape),
            "policy_grid_shape_hw": [16, 12],
            "grid_min_m": float(np.min(grid_m)),
            "grid_max_m": float(np.max(grid_m)),
        }

        image = colorize(corrected, min_depth, max_depth)
        panel = fit_panel(image, 420, 360)
        cv2.putText(panel, f"CCW {degrees} deg  {corrected.shape[1]}x{corrected.shape[0]}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        orientation_panels.append(panel)

        grid_img = cv2.applyColorMap(np.round(grid_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        grid_img = cv2.resize(grid_img, (240, 320), interpolation=cv2.INTER_NEAREST)
        grid_panel = np.zeros((360, 420, 3), dtype=np.uint8)
        grid_panel[28:348, 90:330] = grid_img
        cv2.putText(grid_panel, f"CCW {degrees} -> grid 12x16", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        grid_panels.append(grid_panel)

    orientation = np.vstack((np.hstack(orientation_panels[:2]), np.hstack(orientation_panels[2:])))
    grids = np.vstack((np.hstack(grid_panels[:2]), np.hstack(grid_panels[2:])))
    orientation_path = output / "rotation_candidates.png"
    grid_path = output / "grid_candidates.png"
    cv2.imwrite(str(orientation_path), orientation)
    cv2.imwrite(str(grid_path), grids)
    return {
        "raw_preview": str(raw_path),
        "rotation_candidates_preview": str(orientation_path),
        "grid_candidates_preview": str(grid_path),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="采集 Odin1 深度图并生成旋转/网格候选预览")
    parser.add_argument("--topic", default="/odin1/depth_img_competetion", help="sensor_msgs/msg/Image 深度话题")
    parser.add_argument("--output", default="rl/logs/evaluation/odin_depth_calibration", help="输出目录")
    parser.add_argument("--frames", type=int, default=1, help="保存帧数，默认1")
    parser.add_argument("--timeout", type=float, default=15.0, help="等待话题超时秒数")
    parser.add_argument("--min-depth", type=float, default=0.2, help="预览/网格近端，米")
    parser.add_argument("--max-depth", type=float, default=6.0, help="网格远端，米")
    args = parser.parse_args()
    if args.frames < 1 or args.timeout <= 0 or args.max_depth <= args.min_depth:
        parser.error("frames/timeout必须为正数，max-depth必须大于min-depth")

    rclpy.init()
    node = DepthCapture(args.topic, args.frames)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and len(node.frames) < args.frames and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not node.frames:
        print(f"[错误] {args.timeout:g}秒内没有收到深度图；请检查话题名与消息类型。")
        return 2

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (depth, info) in enumerate(node.frames):
        np.save(output / f"depth_raw_{index:03d}.npy", depth)
        valid = np.isfinite(depth) & (depth > 0.0)
        values = depth[valid]
        info.update({
            "valid_fraction": float(valid.mean()),
            "valid_min_m": float(values.min()) if values.size else None,
            "valid_p01_m": float(np.percentile(values, 1)) if values.size else None,
            "valid_median_m": float(np.median(values)) if values.size else None,
            "valid_p99_m": float(np.percentile(values, 99)) if values.size else None,
            "valid_max_m": float(values.max()) if values.size else None,
        })
        records.append(info)

    previews = save_previews(node.frames[-1][0], output, args.min_depth, args.max_depth)
    report = {
        "topic": args.topic,
        "message_type": "sensor_msgs/msg/Image",
        "frame_count": len(node.frames),
        "range_m": [args.min_depth, args.max_depth],
        "frames": records,
        "previews": previews,
        "review_note": (
            "将相机正前方放置一个已知障碍，查看 rotation_candidates.png 与 grid_candidates.png。"
            "选择能让障碍出现在策略前方区域的旋转角；不要根据原图横竖直接猜旋转方向。"
        ),
    }
    (output / "calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 保存到: {output}")
    for key, value in previews.items():
        if key.endswith("preview"):
            print(f"  {key}: {value}")
    for i, info in enumerate(records):
        print(f"  帧{i}: {info['encoding']} {info['shape_hw'][1]}x{info['shape_hw'][0]}, "
              f"有效{info['valid_fraction']:.1%}, 中值{info['valid_median_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
