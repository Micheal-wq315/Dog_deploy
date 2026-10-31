"""纯Torch台阶事件状态机；不依赖仿真，便于验证抬轮与落地奖励防投机。"""

import torch


class StairEvents:
    """每轮只对更高的新踏面给事件奖励；缓存由环境策略步管理。"""

    def __init__(self, count, device, wheels=4):
        shape = (count, wheels)
        self.valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.armed = torch.zeros(shape, dtype=torch.bool, device=device)
        self.crossed = torch.zeros_like(self.armed)
        self.airborne = torch.zeros_like(self.armed)
        self.bad_impact = torch.zeros_like(self.armed)
        self.contact_steps = torch.zeros(shape, dtype=torch.long, device=device)
        self.peak_fraction = torch.zeros(shape, device=device)
        self.target = torch.zeros(shape, device=device)
        self.base = torch.zeros_like(self.target)
        self.start_x = torch.zeros_like(self.target)
        self.completed = torch.zeros_like(self.target)
        self.highest_stage = torch.zeros(count, device=device)
        self.step = -1
        self.outputs = {}

    def reset(self, ids):
        self.valid[ids] = False
        self.armed[ids] = False
        self.crossed[ids] = False
        self.airborne[ids] = False
        self.bad_impact[ids] = False
        self.contact_steps[ids] = 0
        self.peak_fraction[ids] = 0.

    def update(self, *, ground, ahead, wheel_x, bottom, vertical_speed, contact, forward_speed,
               active, safe, alive, impact_ok):
        """ground[N,4]；ahead为轮端前方最近更高踏面，无阶为ground。"""
        finite = torch.isfinite(ground).all(dim=1) & torch.isfinite(ahead).all(dim=1)
        initialized = self.valid.clone()
        fresh = ~self.valid & finite
        self.completed[fresh] = ground[fresh]
        self.highest_stage[fresh] = ground[fresh].amin(dim=1)
        self.valid |= finite
        eligible = initialized & finite & active & alive
        support = contact.sum(dim=1) >= 2
        moving = (forward_speed / .25).clamp(0, 1)
        stable = eligible & safe & support
        rise = ahead - ground
        arm = (~self.armed & (rise > .008) & (rise <= .16)
               & (ahead > self.completed + .008) & eligible[:, None])
        self.armed |= arm
        self.target = torch.where(arm, ahead, self.target)
        self.base = torch.where(arm, ground, self.base)
        self.start_x = torch.where(arm, wheel_x, self.start_x)
        self.crossed[arm] = False
        self.airborne[arm] = False
        self.bad_impact[arm] = False
        self.contact_steps[arm] = 0
        self.peak_fraction[arm] = 0.
        self.airborne |= self.armed & ~contact
        self.bad_impact |= self.armed & ~impact_ok
        # 小阶允许直接滚上；高阶要求曾离地。正向运动门控阻止停车抬腿领奖。
        lifted = self.airborne | ((self.target - self.base) <= .025)
        gate = self.armed & stable[:, None]
        height_fraction = ((bottom - self.base) / (self.target - self.base + .02).clamp_min(.02)).clamp(0, 1)
        # 只奖励相对此前最高点的新增有效抬升；保持悬轮不再持续获得正奖励。
        clearance_gain = (height_fraction - self.peak_fraction).clamp_min(0.)
        clearance = (clearance_gain * gate * ~self.crossed * (vertical_speed > -.05)
                     * moving[:, None]).mean(dim=1)
        self.peak_fraction = torch.maximum(self.peak_fraction, height_fraction * self.armed)
        crossed_now = (gate & ~self.crossed & ~self.bad_impact & lifted & (bottom >= self.target - .008)
                       & (ground >= self.target - .008) & (wheel_x > self.start_x + .025)
                       & (moving[:, None] > .1))
        self.crossed |= crossed_now
        settled = (gate & ~self.bad_impact & self.crossed & lifted & contact & ((bottom-self.target).abs() < .035)
                   & (vertical_speed.abs() < .25) & (ground >= self.target - .008))
        self.contact_steps = torch.where(settled, self.contact_steps + 1, 0)
        landing = self.armed & (self.contact_steps >= 3)
        self.completed = torch.where(landing, self.target, self.completed)
        self.armed &= ~landing
        self.peak_fraction = torch.where(landing, torch.zeros_like(self.peak_fraction), self.peak_fraction)
        stage_height = self.completed.amin(dim=1)
        stage = stable & (stage_height > self.highest_stage + .008)
        self.highest_stage = torch.maximum(self.highest_stage, torch.where(stage, stage_height, self.highest_stage))
        self.outputs = {"clearance": clearance, "crossing": crossed_now.float().mean(dim=1),
                        "landing": landing.float().mean(dim=1), "stage": stage.float()}
        return self.outputs


class PerformanceGate:
    """按地形组维护当前难度的最近试验窗口，时钟只能限制上限而不能代替表现。"""

    def __init__(self, device, groups=3, window=64):
        self.cap = torch.zeros(groups, dtype=torch.long, device=device)
        self.count = torch.zeros_like(self.cap)
        self.success = torch.zeros(groups, window, device=device)
        self.clean = torch.zeros_like(self.success)
        self.error = torch.zeros_like(self.success)
        self.window = window

    def record(self, groups, levels, success, clean, error, time_cap):
        for group in range(len(self.cap)):
            indices = ((groups == group) & (levels == self.cap[group])).nonzero().flatten()
            for index in indices:
                slot = int(self.count[group] % self.window)
                self.success[group, slot] = success[index]
                self.clean[group, slot] = clean[index]
                self.error[group, slot] = error[index]
                self.count[group] += 1
            ready = self.count[group] >= self.window
            if (ready and self.cap[group] < time_cap and self.success[group].mean() >= .8
                    and self.clean[group].mean() >= .8 and self.error[group].mean() <= .18):
                self.cap[group] += 1
                self.count[group] = 0
        return self.cap

    def state_dict(self):
        return {key: getattr(self, key).clone() for key in ("cap", "count", "success", "clean", "error")}

    def load_state_dict(self, state):
        for key, value in state.items():
            target = getattr(self, key)
            if target.shape != value.shape:
                raise ValueError(f"课程窗口形状不匹配：{key}")
            target.copy_(value.to(target.device))
