#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""DJI 手柄 → ROS2 发布者。

把 DJI Virtual Joystick (USB 2ca3:1021) 的 evdev 数据转成 ROS2 话题，
既发布 Xbox 兼容布局的 /joy 给 inference_node 直接使用，也发布解析好的控制量。

★ 必须用系统 Python 运行 (rclpy 只有 3.10 的扩展):
      /usr/bin/python3 src/inference/scripts/dji_pad_publisher.py
  用 conda 的 python3 (3.13) 会报 _rclpy_pybind11 缺失。

★ 运行前 source 环境:
      source /opt/ros/humble/setup.bash

--------------------------------------------------------------------------
发布的接口
--------------------------------------------------------------------------
  /joy                sensor_msgs/Joy       Xbox 兼容布局, 8 轴 8 键
  /dji_pad/cmd_vel    geometry_msgs/Twist   控制量: vx, vy, wz (已按 max_* 限幅)
  /dji_pad/raw        sensor_msgs/Joy       原始 5 轴 + 8 键, 不做任何映射(调试用)

QoS 三个话题统一为 best_effort + KeepLast(1) + volatile —— 必须与
inference_node.hpp 中 `/joy` 订阅者的 QoS 一致, 否则收不到消息。

--------------------------------------------------------------------------
轴映射 (本机实测得出, 见下方"标定"一节)
--------------------------------------------------------------------------
  物理控件          原始轴        归一化后用途
  右摇杆 纵向       ABS_Y   (1)   vx   (前进/后退)
  右摇杆 横向       ABS_X   (0)   vy   (左右平移)
  左摇杆 纵向       ABS_RX  (3)   wz   (转向角速度) ← 默认, 自回中
  左摇杆 横向       ABS_Z   (2)   空闲 (该轴不回中, 见下方警告)
  拨轮              ABS_RY  (4)   油门 (0..1, 可锁定; 仅当 throttle_scales_speed=true 时生效)

  ★★ 为什么转向不用左摇杆「横向」(ABS_Z) ★★
  实测 (2026-10-12, 手柄平放不触碰): ABS_Z 静止读数 -0.36 —— 而其它三个摇杆轴
  静止都在 ±0.00。历史测量还见过 -279 / -299 / -302 / -306 / -46 (原始值) 的漂移,
  即归一化后 -0.42 ~ -0.07。若用它做转向, **没人碰手柄时机器人也会持续自转**;
  死区(默认 0.10)挡不住这么大的偏移, 而放大死区又会吃掉近半可用行程。
  故默认把转向放到**自回中的左摇杆纵向**(`wz_axis` 参数可改回 left_x)。
  ABS_Z 空闲不用; 如需起用, 打开 `calibrate_wz:=true` 中标定。

  8 个按键: HID usage 1..8 → evdev 0x130..0x137
  (描述符里声明了 24 个按键, 实测只有 8 个存在, 其余 16 个恒为 0)

/joy 的轴下标按 Xbox 惯例排布, 以匹配 inference_node 的读法:
      axes[2] → LT,  axes[3] → 右摇杆横(vy),  axes[4] → 右摇杆纵(vx),  axes[5] → RT

--------------------------------------------------------------------------
标定: 方向符号
--------------------------------------------------------------------------
本脚本无法自动判断"推摇杆向上"对应 +660 还是 -660, 因此提供 5 个 invert_*
参数。默认全部 false。如果方向反了, 用下面任一方式确认后改参数:

  a) 直接看实时面板 (能看到每个轴的原始值随操作变化):
         python3 tools/dji_pad_tui.py
  b) 启动本节点后看它自己的 2Hz 状态日志

典型情况: Linux HID 手柄推上通常给出 ABS_Y 负值, 所以 vx 反了就把
invert_right_y 置 true。其余同理。

--------------------------------------------------------------------------
示例
--------------------------------------------------------------------------
  # 默认参数运行
  /usr/bin/python3 src/inference/scripts/dji_pad_publisher.py

  # 交换左右摇杆(如果实测左右和上面相反)
  /usr/bin/python3 src/inference/scripts/dji_pad_publisher.py --ros-args \
      -p swap_sticks:=true -p invert_right_y:=true

  # 只看不改
  ros2 topic echo /joy
"""

import ctypes
import fcntl
import glob
import math
import os
import select
import struct
import sys
import threading
import time

# ──────────────────────────────────────────────────────────────────────────
# rclpy 引导: 给出可操作的错误信息, 而不是一条晦涩的 ImportError
# ──────────────────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Joy
    from geometry_msgs.msg import Twist
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        "\n[错误] 无法导入 rclpy: %s\n\n"
        "请确认:\n"
        "  1. 用系统 Python 而不是 conda:\n"
        "       /usr/bin/python3 %s\n"
        "     当前解释器: %s\n"
        "  2. 已 source ROS2 环境:\n"
        "       source /opt/ros/humble/setup.bash\n"
        "     当前 ROS_DISTRO=%r\n\n"
        % (exc, os.path.abspath(__file__), sys.executable, os.environ.get("ROS_DISTRO"))
    )
    raise SystemExit(2)


# ──────────────────────────────────────────────────────────────────────────
# evdev
# ──────────────────────────────────────────────────────────────────────────
EVENT_FORMAT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)
EV_SYN, EV_KEY, EV_ABS = 0x00, 0x01, 0x03

# 实测出的物理轴分配: 原始 evdev code -> 角色
RAW_LEFT_X = 2   # ABS_Z   左摇杆横向
RAW_LEFT_Y = 3   # ABS_RX  左摇杆纵向
RAW_RIGHT_X = 0  # ABS_X   右摇杆横向
RAW_RIGHT_Y = 1  # ABS_Y   右摇杆纵向
RAW_DIAL = 4     # ABS_RY  拨轮

RAW_AXIS_CODES = (0, 1, 2, 3, 4)

# 按键: HID usage 1..8 -> evdev code 0x130..0x137
BUTTON_CODES = tuple(0x130 + i for i in range(8))

AXIS_FULL_SCALE = 660.0  # 实测所有轴都是 -660..+660
TARGET_NAME = "DJI"


class input_absinfo(ctypes.Structure):
    _fields_ = [
        ("value", ctypes.c_int32), ("minimum", ctypes.c_int32),
        ("maximum", ctypes.c_int32), ("fuzz", ctypes.c_int32),
        ("flat", ctypes.c_int32), ("resolution", ctypes.c_int32),
    ]


def EVIOCGABS(axis):
    return 0x80184540 + axis


def find_device(forced=None):
    """动态查找 DJI 手柄的 event 节点。

    不写死 event 编号: USB 掉线重连后编号会变(实测从 event17 重连过)。
    """
    if forced:
        return (forced, "指定设备") if os.path.exists(forced) else (None, None)
    for sysdev in sorted(glob.glob("/sys/class/input/event*/device/name")):
        try:
            with open(sysdev) as fh:
                name = fh.read().strip()
        except OSError:
            continue
        if TARGET_NAME in name:
            path = "/dev/input/" + sysdev.split("/")[4]
            if os.path.exists(path):
                return path, name
    return None, None


class PadState:
    """线程安全的原始手柄状态。轴值已归一化到 -1..+1。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.axes = {c: 0.0 for c in RAW_AXIS_CODES}
        self.buttons = {c: 0 for c in BUTTON_CODES}
        self.connected = False
        self.device_path = None
        self.device_name = None
        self.reconnects = 0
        self.event_count = 0
        self.last_event_at = 0.0
        self.error = None

    def snapshot(self):
        with self.lock:
            return (
                dict(self.axes), dict(self.buttons),
                self.connected, self.device_path, self.device_name,
                self.reconnects, self.event_count, self.last_event_at, self.error,
            )


class PadReader(threading.Thread):
    """后台线程: 持续读 evdev, 掉线自动重连。

    独立线程而非挂在 ROS 定时器上, 是为了不丢事件 —— 手柄快速划动时
    上报率可达 100Hz 以上, 而发布定时器只有 50Hz。
    """

    def __init__(self, state, device, log_warn):
        super().__init__(daemon=True)
        self.state = state
        self.forced = device or None
        self.log_warn = log_warn
        self._stop = threading.Event()
        self._rescan_at = 0.0

    def stop(self):
        self._stop.set()

    def _open(self):
        path, name = find_device(self.forced)
        if path is None:
            with self.state.lock:
                self.state.connected = False
                self.state.error = "未找到 DJI 手柄"
            return None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            with self.state.lock:
                self.state.connected = False
                self.state.error = "打开 %s 失败: %s" % (path, exc)
            return None

        axes = {}
        for code in RAW_AXIS_CODES:
            info = input_absinfo()
            try:
                fcntl.ioctl(fd, EVIOCGABS(code), info)
                span = info.maximum - info.minimum
                axes[code] = (2.0 * (info.value - info.minimum) / span - 1.0) if span else 0.0
            except OSError:
                axes[code] = 0.0

        with self.state.lock:
            self.state.axes.update(axes)
            self.state.connected = True
            self.state.device_path = path
            self.state.device_name = name
            self.state.reconnects += 1
            self.state.error = None
        return fd

    def run(self):
        fd = None
        while not self._stop.is_set():
            if fd is None:
                now = time.time()
                if now >= self._rescan_at:
                    self._rescan_at = now + 1.0
                    fd = self._open()
                if fd is None:
                    self._stop.wait(0.05)
                    continue

            try:
                ready, _, _ = select.select([fd], [], [], 0.05)
            except (OSError, ValueError):
                ready = []
            if not ready:
                continue

            try:
                data = os.read(fd, EVENT_SIZE * 256)
            except BlockingIOError:
                continue
            except OSError as exc:
                with self.state.lock:
                    self.state.connected = False
                    self.state.error = "设备读取失败(已掉线): %s" % exc
                self.log_warn("手柄掉线: %s" % exc)
                try:
                    os.close(fd)
                except OSError:
                    pass
                fd = None
                self._rescan_at = 0.0
                continue

            changed = False
            now = time.time()
            with self.state.lock:
                for off in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
                    _, _, etype, code, value = struct.unpack(
                        EVENT_FORMAT, data[off:off + EVENT_SIZE])
                    if etype == EV_SYN:
                        continue
                    self.state.event_count += 1
                    changed = True
                    if etype == EV_ABS and code in self.state.axes:
                        v = value / AXIS_FULL_SCALE
                        self.state.axes[code] = 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)
                    elif etype == EV_KEY and code in self.state.buttons:
                        self.state.buttons[code] = 1 if value else 0
                if changed:
                    self.state.last_event_at = now

        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


# ──────────────────────────────────────────────────────────────────────────
# 信号处理
# ──────────────────────────────────────────────────────────────────────────
def apply_deadzone(v, dz):
    """死区: 死区内归零, 死区外重新拉伸到满量程, 避免跳变。"""
    if dz <= 0.0:
        return v
    a = abs(v)
    if a <= dz:
        return 0.0
    return math.copysign((a - dz) / (1.0 - dz), v)


def apply_expo(v, e):
    """指数曲线: e=0 线性, e=1 全三次方, 让小幅操作更精细。"""
    if e <= 0.0:
        return v
    return (1.0 - e) * v + e * (v ** 3)


# ──────────────────────────────────────────────────────────────────────────
# ROS2 节点
# ──────────────────────────────────────────────────────────────────────────
class DjiPadPublisher(Node):

    def __init__(self):
        super().__init__("dji_pad_publisher")

        # ── 参数 ──────────────────────────────────────────────────────
        self.declare_parameter("device", "")            # 空 = 自动查找 DJI
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("deadzone", 0.10)
        self.declare_parameter("expo", 0.0)
        self.declare_parameter("swap_sticks", False)    # 左右摇杆对调

        # 方向标定 (见文件头「标定」)
        self.declare_parameter("invert_left_x", False)   # ABS_Z  (空闲)
        self.declare_parameter("invert_left_y", False)   # ABS_RX (转向 wz)
        self.declare_parameter("invert_right_x", False)  # ABS_X  (平移 vy)
        self.declare_parameter("invert_right_y", False)  # ABS_Y  (前进 vx)
        self.declare_parameter("invert_dial", False)     # ABS_RY (油门)

        # ── 转向来源与中点标定 ────────────────────────────────────────
        # wz 取自哪个摇杆轴。默认 left_y (ABS_RX, 自回中);
        # 可改 left_x / right_x / right_y / dial —— 但只有自回中的轴适合做转向,
        # 见文件头「为什么转向不用左摇杆横向」。
        self.declare_parameter("wz_axis", "left_y")
        # 启动时对该轴做中点标定 (采样取中位数当零点)。对不回中/有偏置的轴是必需的;
        # 对自回中轴也能修掉小的机械偏置。
        self.declare_parameter("calibrate_wz", True)
        self.declare_parameter("calibrate_seconds", 1.0)
        # 标定出的零点偏移超过此值就判为"标定时有人在推杆", 不采纳并告警
        self.declare_parameter("calibrate_max_offset", 0.15)
        # 油门取自哪个轴 (拨轮 ABS_RY 天然"可锁定", 适合当油门)
        self.declare_parameter("throttle_axis", "dial")

        # 控制量限幅, 默认对齐 wheel_quad/configs/default.yaml 的 clip_cmd
        self.declare_parameter("max_vx", 1.0)
        self.declare_parameter("max_vy", 0.6)
        self.declare_parameter("max_wz", 1.0)

        self.declare_parameter("throttle_scales_speed", False)
        self.declare_parameter("yaw_trigger_compat", True)

        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("cmd_topic", "/dji_pad/cmd_vel")
        self.declare_parameter("raw_topic", "/dji_pad/raw")
        self.declare_parameter("publish_joy", True)
        self.declare_parameter("publish_cmd", True)
        self.declare_parameter("publish_raw", False)
        self.declare_parameter("status_log_period", 2.0)

        gp = self.get_parameter
        self.device = gp("device").value
        self.rate = max(1.0, float(gp("publish_rate").value))
        self.deadzone = min(0.95, max(0.0, float(gp("deadzone").value)))
        self.expo = min(1.0, max(0.0, float(gp("expo").value)))
        self.swap = bool(gp("swap_sticks").value)
        self.inv = {
            RAW_LEFT_X: bool(gp("invert_left_x").value),
            RAW_LEFT_Y: bool(gp("invert_left_y").value),
            RAW_RIGHT_X: bool(gp("invert_right_x").value),
            RAW_RIGHT_Y: bool(gp("invert_right_y").value),
            RAW_DIAL: bool(gp("invert_dial").value),
        }
        self.max_vx = float(gp("max_vx").value)
        self.max_vy = float(gp("max_vy").value)
        self.max_wz = float(gp("max_wz").value)
        self.throttle_scales = bool(gp("throttle_scales_speed").value)
        self.yaw_compat = bool(gp("yaw_trigger_compat").value)
        self.status_period = float(gp("status_log_period").value)

        # 转向来源 + 中点标定
        self.wz_axis = str(gp("wz_axis").value)
        if self.wz_axis not in ("left_x", "left_y", "right_x", "right_y", "dial"):
            self.get_logger().warn("wz_axis=%r 非法, 回退到 left_y" % self.wz_axis)
            self.wz_axis = "left_y"
        self.calibrate_wz = bool(gp("calibrate_wz").value)
        self.calibrate_seconds = max(0.0, float(gp("calibrate_seconds").value))
        self.calibrate_max_offset = abs(float(gp("calibrate_max_offset").value))
        self.throttle_axis = str(gp("throttle_axis").value)
        self._wz_offset = 0.0          # 标定出的零点偏移
        self._calib_samples = []       # 标定采样
        self._calib_done = not self.calibrate_wz

        # 左右摇杆分组: swap 后 (X,Y) 变左杆, (Z,RX) 变右杆
        if self.swap:
            self.c_left_x, self.c_left_y = RAW_RIGHT_X, RAW_RIGHT_Y
            self.c_right_x, self.c_right_y = RAW_LEFT_X, RAW_LEFT_Y
        else:
            self.c_left_x, self.c_left_y = RAW_LEFT_X, RAW_LEFT_Y
            self.c_right_x, self.c_right_y = RAW_RIGHT_X, RAW_RIGHT_Y

        # ── QoS: 必须与 inference_node 的 /joy 订阅者一致 ──────────────
        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub_joy = self.create_publisher(Joy, gp("joy_topic").value, qos) \
            if bool(gp("publish_joy").value) else None
        self.pub_cmd = self.create_publisher(Twist, gp("cmd_topic").value, qos) \
            if bool(gp("publish_cmd").value) else None
        self.pub_raw = self.create_publisher(Joy, gp("raw_topic").value, qos) \
            if bool(gp("publish_raw").value) else None

        # ── 读取线程 ──────────────────────────────────────────────────
        self.state = PadState()
        self._last_warn = 0.0
        self.reader = PadReader(self.state, self.device, self._warn_throttled)
        self.reader.start()

        self._was_connected = False
        self._last_status = 0.0
        self.timer = self.create_timer(1.0 / self.rate, self._tick)

        self.get_logger().info(
            f"DJI 手柄发布者已启动 | 映射: 右杆(横→vy, 纵→vx) 转向→{self.wz_axis} "
            f"油门→{self.throttle_axis} | {self.rate:g}Hz deadzone={self.deadzone:.2f} "
            f"expo={self.expo:.2f} swap={self.swap} 中点标定={'开' if self.calibrate_wz else '关'}"
        )
        self.get_logger().info(
            "控制量限幅: vx±%.2f m/s  vy±%.2f m/s  wz±%.2f rad/s"
            % (self.max_vx, self.max_vy, self.max_wz)
        )
        self.get_logger().info(
            f"轴分配: 右杆 ABS_Y(1)→vx / ABS_X(0)→vy | 转向→{self.wz_axis} | "
            f"油门→{self.throttle_axis} | 左横 ABS_Z(2) 空闲 | 拨轮 ABS_RY(4)"
        )
        if self.wz_axis == "left_x":
            self.get_logger().warn(
                "wz_axis=left_x: ABS_Z 在这台手柄上松手不回中 (实测静止 -0.36), "
                "转向不会自动归零 —— 只建议在 calibrate_wz:=true 且偏移稳定时使用。"
            )
        if self.pub_joy:
            self.get_logger().info("发布 /joy (Xbox 兼容布局, 8 轴 8 键)")
        if self.pub_cmd:
            self.get_logger().info("发布 %s" % gp("cmd_topic").value)
        if self.pub_raw:
            self.get_logger().info("发布 %s" % gp("raw_topic").value)

    def _warn_throttled(self, msg):
        now = time.time()
        if now - self._last_warn >= 2.0:
            self._last_warn = now
            self.get_logger().warn(msg)

    # ── 主循环 ────────────────────────────────────────────────────────
    def _tick(self):
        (raw, buttons, connected, path, name,
         reconnects, events, last_event, error) = self.state.snapshot()

        if connected != self._was_connected:
            self._was_connected = connected
            if connected:
                self.get_logger().info("手柄已连接: %s (%s)" % (name, path))
            else:
                self.get_logger().warn("手柄已断开 —— 已输出零速度指令")

        # ── 归一化 + 标定 ────────────────────────────────────────────
        def norm(code):
            v = raw.get(code, 0.0)
            if self.inv.get(code):
                v = -v
            return v

        left_x_raw = norm(self.c_left_x)
        left_y_raw = norm(self.c_left_y)
        right_x_raw = norm(self.c_right_x)
        right_y_raw = norm(self.c_right_y)
        dial_raw = norm(RAW_DIAL)

        # 摇杆(自回中轴): 死区 + 曲线
        left_x = apply_expo(apply_deadzone(left_x_raw, self.deadzone), self.expo)
        right_x = apply_expo(apply_deadzone(right_x_raw, self.deadzone), self.expo)
        right_y = apply_expo(apply_deadzone(right_y_raw, self.deadzone), self.expo)

        # ── 转向来源 + 启动中点标定 ───────────────────────────────────
        # 为什么要标定: 实测本手柄某轴静止时有偏置 (ABS_Z 静止 -0.36), 不补偿的话
        # "松手"就等于持续给转向指令。见文件头「为什么转向不用左摇杆横向」。
        wz_raw = {
            "left_x": left_x_raw,
            "left_y": left_y_raw,
            "right_x": right_x_raw,
            "right_y": right_y_raw,
            "dial": dial_raw,
        }[self.wz_axis]
        if not self._calib_done:
            self._calib_samples.append(wz_raw)
            if len(self._calib_samples) >= max(1, int(self.calibrate_seconds * self.rate)):
                # 取中位数 (对偶发跳变比均值稳)
                center = sorted(self._calib_samples)[len(self._calib_samples) // 2]
                if abs(center) <= self.calibrate_max_offset:
                    self._wz_offset = center
                    self.get_logger().info(
                        "转向轴 %s 中点标定: 偏移 %+.3f 已补偿 (标定 %d 帧)"
                        % (self.wz_axis, center, len(self._calib_samples)))
                else:
                    self.get_logger().warn(
                        "转向轴 %s 标定偏移 %+.3f 超过 %.2f —— 标定时是否在推杆? "
                        "本次不补偿; 松开摇杆重跑本节点即可"
                        % (self.wz_axis, center, self.calibrate_max_offset))
                self._calib_done = True
        # 先扣标定中点, 再过死区/曲线 —— 保证"松手 = 零指令"
        wz = apply_expo(apply_deadzone(wz_raw - self._wz_offset, self.deadzone), self.expo)

        # ── 油门 (默认拨轮: 天然可锁定, 适合当油门) ────────────────────
        throttle_raw = {
            "dial": dial_raw,
            "left_x": left_x_raw,
            "left_y": left_y_raw,
            "right_x": right_x_raw,
            "right_y": right_y_raw,
        }.get(self.throttle_axis, dial_raw)
        throttle = min(1.0, max(0.0, (throttle_raw + 1.0) * 0.5))

        # ── 控制量 ───────────────────────────────────────────────────
        # 右摇杆 -> 平移; 转向 -> wz (默认左摇杆纵向, 自回中)。
        vx, vy = right_y, right_x
        if self.throttle_scales:
            vx *= throttle
            vy *= throttle
            wz *= throttle

        if not connected:
            # 安全: 掉线时必须发零, 否则机器人会保持上一条指令继续跑
            vx = vy = wz = 0.0
            throttle = 0.0

        stamp = self.get_clock().now().to_msg()

        # ── /joy ─────────────────────────────────────────────────────
        if self.pub_joy is not None:
            msg = Joy()
            msg.header.stamp = stamp
            msg.header.frame_id = "dji_pad"
            # 长度固定 8: inference_node 无越界检查地读 axes[2..5] / buttons[0..5]
            msg.axes = [left_x, left_y_raw, 0.0, right_x, right_y, 0.0, dial_raw, 0.0]
            if self.yaw_compat:
                # inference_node 的转向逻辑: axes[2]<0 取 -axes[2], axes[5]<0 取 axes[5]
                # 把 wz 按符号拆成两半即可得到双向转向, 且静止时两路都为 0
                msg.axes[2] = -wz if wz > 0.0 else 0.0
                msg.axes[5] = wz if wz < 0.0 else 0.0
            else:
                msg.axes[2] = wz
            msg.buttons = [buttons.get(c, 0) for c in BUTTON_CODES]
            self.pub_joy.publish(msg)

        # ── /dji_pad/cmd_vel ─────────────────────────────────────────
        if self.pub_cmd is not None:
            cmd = Twist()
            cmd.linear.x = vx * self.max_vx
            cmd.linear.y = vy * self.max_vy
            cmd.angular.z = wz * self.max_wz
            self.pub_cmd.publish(cmd)

        # ── /dji_pad/raw ─────────────────────────────────────────────
        if self.pub_raw is not None:
            msg = Joy()
            msg.header.stamp = stamp
            msg.header.frame_id = "dji_pad_raw"
            msg.axes = [raw.get(c, 0.0) for c in RAW_AXIS_CODES]
            msg.buttons = [buttons.get(c, 0) for c in BUTTON_CODES]
            self.pub_raw.publish(msg)

        # ── 状态日志 ─────────────────────────────────────────────────
        now = time.time()
        if self.status_period > 0 and now - self._last_status >= self.status_period:
            self._last_status = now
            if connected:
                self.get_logger().info(
                    "vx=%+.2f vy=%+.2f wz=%+.2f | 油门=%.2f | 左(横%+.2f 纵%+.2f) "
                    "右(横%+.2f 纵%+.2f) 拨轮%+.2f | 键=%s | 事件%d 重连%d"
                    % (vx * self.max_vx, vy * self.max_vy, wz * self.max_wz, throttle,
                       left_x, left_y_raw, right_x, right_y, dial_raw,
                       "".join(str(buttons.get(c, 0)) for c in BUTTON_CODES),
                       events, reconnects))
            else:
                self.get_logger().warn("手柄未连接: %s" % (error or "等待设备"))

    def destroy_node(self):
        try:
            self.reader.stop()
            self.reader.join(timeout=1.0)
        finally:
            super().destroy_node()


def main(argv=None):
    rclpy.init(args=argv)
    node = None
    try:
        node = DjiPadPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
