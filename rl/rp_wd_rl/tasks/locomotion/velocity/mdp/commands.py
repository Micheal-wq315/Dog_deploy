"""自定义指令项: 机身姿态 + 高度 (pitch / roll / height).

轮足机器人除了速度, 还能靠腿部伸缩改变**机身姿态与高度** —— 这需要一个独立的指令通道。
单独做成一个 command term (而不是改写 UniformVelocityCommand) 的好处:
观测里它就是一个独立分量, 部署端只需在 obs_layouts 里加一项 `cmd_posture:3`,
既有分量的偏移全部不变。
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from rp_wd_rl.assets.rp_wd_contract import CMD_HEIGHT_RANGE, CMD_PITCH_RANGE, CMD_ROLL_RANGE


class UniformPostureCommand(CommandTerm):
    """均匀采样机身姿态+高度指令 (pitch, roll, height_offset).

    单位: pitch/roll 为 rad, height 为相对站立高度的偏移 [m]。

    符号约定 (与契约一致, 由机体系重力反算):
        pitch = asin(g_x)          抬头为正
        roll  = atan2(g_y, -g_z)   右侧下沉为正
        height                     相对 STAND_HEIGHT 的偏移, 负 = 蹲低
    """

    cfg: UniformPostureCommandCfg

    def __init__(self, cfg: UniformPostureCommandCfg, env):
        super().__init__(cfg, env)
        # (num_envs, 3) = (pitch, roll, height_offset)
        self.posture_command = torch.zeros(self.num_envs, 3, device=self.device)

    def __str__(self) -> str:
        r = self.cfg.ranges
        return f"UniformPostureCommand(pitch={r.pitch}, roll={r.roll}, height={r.height})"

    @property
    def command(self) -> torch.Tensor:
        """(num_envs, 3) 的 (pitch, roll, height_offset) 指令."""
        return self.posture_command

    def _update_metrics(self):
        # 训练日志里不额外记录 (姿态/高度误差由各自的跟踪奖励反映)
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        for col, value_range in enumerate(
            (self.cfg.ranges.pitch, self.cfg.ranges.roll, self.cfg.ranges.height)
        ):
            r = torch.empty(n, device=self.device)
            self.posture_command[env_ids, col] = r.uniform_(*value_range)
        if self.cfg.zero_on_course:
            from .odin_course import odin_course_mask
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self.posture_command[ids[odin_course_mask(self._env)[ids]]] = 0.0

    def _update_command(self):
        # 指令不随时间演化 (只有 resample 会改变它)
        pass


@configclass
class UniformPostureCommandCfg(CommandTermCfg):
    """机身姿态+高度指令配置."""

    class_type: type = UniformPostureCommand

    resampling_time_range: tuple[float, float] = (10.0, 10.0)
    zero_on_course: bool = False

    @configclass
    class Ranges:
        """采样范围, 取自契约常量."""

        pitch: tuple[float, float] = CMD_PITCH_RANGE
        roll: tuple[float, float] = CMD_ROLL_RANGE
        height: tuple[float, float] = CMD_HEIGHT_RANGE

    ranges: Ranges = Ranges()
