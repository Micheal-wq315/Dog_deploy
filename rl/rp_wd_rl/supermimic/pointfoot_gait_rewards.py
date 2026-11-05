"""v4 步态奖励：共享于真实仿真和可微想象，不向盲 Actor 泄露地形。"""
import torch

from .pointfoot_terrain import PointfootTaskRewards, FULL_REWARD_WEIGHTS
from .pointfoot_gait import PointfootGaitReference, GAIT_CLEARANCE

GAIT_REWARD_WEIGHTS = {
    **FULL_REWARD_WEIGHTS,
    "track_linear_velocity": 2., "track_angular_velocity": 1.,
    "leg_posture": -.01, "action_rate_l2": -.03,
    "gait_foot_path": -.4, "gait_swing_height": -.8,
    "gait_contact": -.15, "landing_velocity": -.03,
}


class GaitPointfootTaskRewards(PointfootTaskRewards):
    def __init__(self, joint_names, default_q, limits):
        super().__init__(joint_names, default_q, limits, profile="full")
        self.weights = dict(GAIT_REWARD_WEIGHTS)
        self.gait = PointfootGaitReference(self)

    def terms(self, state, action, previous, command, scan, collision, terminated,
              contacts=None, force=None, phase=None):
        if phase is None:
            raise ValueError("v4步态奖励必须使用真实相位，不能以零相位代替")
        terms = super().terms(state, action, previous, command, scan, collision,
                              terminated, contacts=contacts, force=force)
        target, stance, lift, active = self.gait.trajectory(phase, command)
        feet = self.feet(state)
        body = self.feet_body(state)
        below, valid = self._sample_ground(scan, feet[..., :2])
        # 扫描整个即将摆动的路径，代替仅前方16cm的单点判断；正反/横移/转向均适用。
        velocity = command[:, None, :2] + command[:, None, 2:3] * torch.stack(
            (-body[..., 1], body[..., 0]), -1)
        direction = velocity / velocity.norm(dim=-1, keepdim=True).clamp_min(.05)
        ground = [below]
        for distance in (.10, .20, .30):
            sample, sample_valid = self._sample_ground(scan, feet[..., :2] + distance * direction)
            ground.append(torch.where(sample_valid, sample, below))
        rise = (torch.stack(ground).amax(0) - below).clamp(0, .20)
        desired_lift = torch.maximum(torch.full_like(rise, GAIT_CLEARANCE), rise + .035)
        bottom = feet[..., 2] - self.radius
        actual_lift = bottom - below
        # 相位定义摆动脚，不能靠预测contact=1逃避抬脚，也不强迫支撑脚抬离地面。
        swing = (~stance).to(state.dtype) * active[:, None] * valid
        height_error = ((actual_lift - desired_lift * lift) / .06).square().clamp_max(4)
        deficit = ((desired_lift * lift - actual_lift).clamp_min(0) / .10)
        terms["stair_clearance"] = (swing * deficit / (1 + deficit)).mean(-1)
        terms["gait_swing_height"] = (swing * height_error).mean(-1)
        # 米制轨迹给出完整前后行程，抑制原地快速抖腿；上阶时放宽XY以允许落点调整。
        rough = self._local_roughness(scan)
        path = ((body[..., :2] - target[..., :2]) / .08).square().sum(-1).clamp_max(4)
        terms["gait_foot_path"] = active * (1 - .5 * rough) * path.mean(-1)
        if contacts is None:
            raise ValueError("v4奖励需要真实接触或世界模型接触概率")
        terms["gait_contact"] = active * (contacts - stance.to(contacts.dtype)).square().mean(-1)
        velocity = self.feet_velocity(state)
        near = valid & (actual_lift < .035)
        # 在触地前就限制向下速度；等contact=1后再罚会漏掉瞬间撞地的冲击。
        descent = (-velocity[..., 2]).clamp_min(0)
        terms["landing_velocity"] = (near * (descent / .5).square().clamp_max(4)).mean(-1)
        # 低速指令不能靠站桩获得85%以上的跟踪奖励。
        terms["track_linear_velocity"] = torch.exp(-(state[:, 4:6] - command[:, :2]).square().sum(-1) / .09)
        terms["track_angular_velocity"] = torch.exp(-(state[:, 9] - command[:, 2]).square() / .09)
        return terms
