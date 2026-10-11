"""WASD 键位的 SE(2) 键盘设备 (Isaac Lab 自带 Se2Keyboard 的扩展).

自带 Se2Keyboard 只用方向键/小键盘, 这里**追加** WASD + QE (原有键位保留):

    W / S = 前进 / 后退
    A / D = 左移 / 右移        (机体坐标系: 左为 +y)
    E / Q = 左转 / 右转        (ωz 正 / 负)

按键语义是"**按住生效**": 按下 W 持续前进, 松开立即归零; 同时按住 W 和 S 会互相抵消.
速度上限由 cfg 的 v_x/v_y/omega_z_sensitivity 决定 (本工程按训练指令范围设置).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isaaclab.devices import DeviceBase, Se2Keyboard, Se2KeyboardCfg


class Se2KeyboardWASD(Se2Keyboard):
    """在方向键/小键盘之外, 额外支持 WASD + QE 的 SE(2) 键盘设备."""

    def _create_key_bindings(self):
        # 先建立自带绑定 (方向键 / 小键盘), 再追加 WASD + QE
        super()._create_key_bindings()
        self._INPUT_KEY_MAPPING.update(
            {
                "W": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
                "S": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
                "A": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
                "D": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
                "E": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
                "Q": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
            }
        )


@dataclass
class Se2KeyboardWASDCfg(Se2KeyboardCfg):
    """WASD 键盘设备的配置 (字段与 Se2KeyboardCfg 相同)."""

    class_type: type[DeviceBase] = Se2KeyboardWASD
