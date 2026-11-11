"""点足完整步态参考；从本仓库v4的URDF解析IK迁移，真实与想象共用。"""
import math
from .kinematics import safe_atan2
import torch
from torch import nn
GAIT_FREQUENCY=1.5
GAIT_DUTY=.60
GAIT_CLEARANCE=.065
class PointfootGaitReference(nn.Module):
    """LF/RB 与 RF/LB 对角交替；轨迹在抬脚/落脚边界位置与速度连续。"""

    def __init__(self, task):
        super().__init__()
        for name, value in (("default_q", task.default_q), ("ids", task.ids),
                            ("anchors", task.anchors), ("offsets", task.offsets)):
            self.register_buffer(name, value.detach().clone())
        state = task.default_q.new_zeros(1, 34)
        self.register_buffer("neutral", task.feet_body(state)[0].detach())
        scales = torch.ones_like(task.default_q)
        scales[task.ids] = scales.new_tensor((.25, .6, .8))
        self.register_buffer("scales", scales)
        self.register_buffer("limits", task.limits.detach().clone())
        if ((self.offsets[:, 0, (0, 2)].abs().max() > 1e-6)
                or (self.offsets[:, 1, 0].abs().max() > 1e-6)
                or not (self.offsets[:, 1, 2] < 0).all()):
            raise ValueError("v4解析IK需要当前RP_wd的横向髋偏置及负Z大腿结构")

    def trajectory(self, phase, command):
        phase = (phase.reshape(-1, 1) + phase.new_tensor((0., .5, .5, 0.))).remainder(1.)
        stance = phase < GAIT_DUTY
        swing = ((phase - GAIT_DUTY) / (1 - GAIT_DUTY)).clamp(0, 1)
        # 五次插值在边界一、二阶导数为零，避免参考轨迹自身制造冲击。
        blend = 10 * swing**3 - 15 * swing**4 + 6 * swing**5
        slope = -(1 - GAIT_DUTY) / GAIT_DUTY
        swing_reach = (1 - slope) * blend + slope * swing - .5
        reach = torch.where(stance, .5 - phase / GAIT_DUTY, swing_reach)
        velocity = command[:, None, :2].expand(-1, 4, -1).clone()
        velocity = velocity + command[:, None, 2:3] * torch.stack(
            (-self.neutral[:, 1], self.neutral[:, 0]), -1)
        # 低速时不强迫最小步长；停止/转向/横移都使用各脚实际应有的速度。
        travel = velocity * (GAIT_DUTY / GAIT_FREQUENCY)
        # 零指令下也保持可微；避免编译反传在零范数处形成0/0。
        active = ((command[:, :2].square().sum(-1)+1e-8).sqrt()-1e-4 + .25 * command[:, 2].abs())
        activation = (active / .08).clamp(0, 1)
        lift = torch.sin(math.pi * swing).square() * activation[:, None]
        xy = self.neutral[None, :, :2] + reach[..., None] * travel
        z = self.neutral[None, :, 2] + GAIT_CLEARANCE * lift
        return torch.cat((xy, z[..., None]), -1), stance, lift, activation

    def inverse(self, target):
        """针对实际URDF的X/Y/Y链解析IK；输出按训练关节顺序排列的绝对关节角。"""
        d = target - self.anchors
        lateral = self.offsets[:, :, 1].sum(-1)
        z = -(d[..., 1:].square().sum(-1) - lateral.square()).clamp_min(.005**2).sqrt()
        abad = safe_atan2(d[..., 1], -d[..., 2]) - safe_atan2(lateral, -z)
        upper = -self.offsets[:, 1, 2]
        lower = self.offsets[:, 2, (0, 2)].square().sum(-1).sqrt()
        bend = ((d[..., 0].square() + z.square() - upper.square() - lower.square())
                / (2 * upper * lower)).clamp(-.995, .995)
        knee_effective = -bend.acos()
        hip = safe_atan2(-d[..., 0], -z) - safe_atan2(
            lower * knee_effective.sin(), upper + lower * knee_effective.cos())
        knee = knee_effective - safe_atan2(-self.offsets[:, 2, 0], -self.offsets[:, 2, 2])
        q = target.new_zeros((len(target), 12))
        q[:, self.ids] = torch.stack((abad, hip, knee), -1)
        return q.clamp(self.limits[:, 0], self.limits[:, 1])

    def actions(self, phase, command):
        target, _, _, _ = self.trajectory(phase, command)
        return ((self.inverse(target) - self.default_q) / self.scales).clamp(-1, 1)

class PointfootGaitMetrics:
    """只统计真实转移：接触去抖后测完整落脚周期、步距和腾空时间。"""

    def __init__(self, count, device, dt=.02):
        self.dt = dt
        self.time = torch.zeros(count, device=device)
        self.contact = torch.ones(count, 4, dtype=torch.bool, device=device)
        self.debounce = torch.zeros(count, 4, dtype=torch.long, device=device)
        self.seen = torch.zeros_like(self.contact)
        self.last_time = torch.zeros(count, 4, device=device)
        self.air = torch.zeros_like(self.last_time)
        self.last_xy = torch.zeros(count, 4, 2, device=device)
        self.clear_interval()

    def clear_interval(self):
        self.period_sum = self.stride_sum = self.air_sum = self.cycles = 0.

    @torch.no_grad()
    def update(self, snapshot, done):
        raw = snapshot["contacts"] > .5
        self.time += self.dt
        self.debounce = torch.where(raw != self.contact, self.debounce + 1, 0)
        changed = (self.debounce >= 2) & ~done[:, None]
        touchdown = changed & raw
        self.contact = torch.where(changed, raw, self.contact)
        self.debounce[changed] = 0
        self.air += (~self.contact) * self.dt
        xy = snapshot["foot_positions"][..., :2]
        valid = touchdown & self.seen
        period = self.time[:, None] - self.last_time
        # 排除重复抖触地、停止后很久才开始迈步；不把它们伪装成正常周期。
        valid &= (period > .15) & (period < 2.)
        self.period_sum += float(period[valid].sum())
        self.stride_sum += float((xy - self.last_xy).norm(dim=-1)[valid].sum())
        self.air_sum += float(self.air[valid].sum())
        self.cycles += int(valid.sum())
        self.last_time = torch.where(touchdown, self.time[:, None], self.last_time)
        self.last_xy = torch.where(touchdown[..., None], xy, self.last_xy)
        self.seen |= touchdown
        self.air[touchdown] = 0
        self.seen[done] = False
        self.contact[done] = True
        self.debounce[done] = 0
        self.air[done] = 0

    def summary(self):
        n = max(self.cycles, 1)
        return {"cycles": self.cycles,
                "frequency_hz": self.cycles / max(self.period_sum, 1e-8),
                "stride_m": self.stride_sum / n, "air_time_s": self.air_sum / n}
