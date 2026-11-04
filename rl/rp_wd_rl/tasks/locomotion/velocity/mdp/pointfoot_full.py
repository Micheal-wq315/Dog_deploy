"""full-v3 的 Isaac Lab 适配；直接导入本模块，不依赖 mdp.__init__ 导出。

父训练每轮调用 update_full_command_curriculum(env, iteration, velocity_mae)，
每个真实策略步调用 update_full_course_step(env)，checkpoint 保存/恢复使用
full_course_state_dict / load_full_course_state_dict；full_summary 可直接写 JSON。

持久状态：_pointfoot_full_course（FullCourseHistory），其中 caps/trials/counts
与既有 _pointfoot_course_* 三个张量共享；额外保存 stats、command_stage、
iteration、velocity_mae。_pointfoot_full_episode 为不随 checkpoint 恢复的回合
计时/锁存。_pointfoot_success_event 为 [N] bool 的单步脉冲，reset 不清除此脉冲。
"""

from __future__ import annotations

import math

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from rp_wd_rl.supermimic.pointfoot_courses import (
    COMMAND_LIMITS,
    COURSE_KINDS,
    EXIT_DISTANCE,
    FOOT_RADIUS,
    FullCourseHistory,
    STAIR_TREAD,
    column_kind_indices,
    pointfoot_full_course,
    sample_mixed_commands,
    stable_course_pass,
)


@configclass
class PointfootFullCourseCfg(SubTerrainBaseCfg):
    function = pointfoot_full_course
    kind: str = "flat"
    stair_tread: float = STAIR_TREAD


def _ids(env, env_ids):
    all_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is None:
        return all_ids
    if isinstance(env_ids, slice):
        return all_ids[env_ids]
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device)


def _history(env):
    history = getattr(env, "_pointfoot_full_course", None)
    if history is None:
        cfg = env.scene.terrain.cfg.terrain_generator
        history = FullCourseHistory(
            column_kind_indices(cfg.sub_terrains, cfg.num_cols), cfg.num_rows, env.device,
            caps=getattr(env, "_pointfoot_course_caps", None),
            trials=getattr(env, "_pointfoot_course_trials", None),
            counts=getattr(env, "_pointfoot_course_counts", None),
        )
        env._pointfoot_full_course = history
        env._pointfoot_course_caps = history.caps
        env._pointfoot_course_trials = history.trials
        env._pointfoot_course_counts = history.counts
    return history


def full_kind_indices(env):
    """返回逐环境 kind 索引；列分配精确复用 generator 配方。"""
    return _history(env).column_kinds[env.scene.terrain.terrain_types]


def full_stair_mask(env):
    kinds = full_kind_indices(env)
    return (kinds == COURSE_KINDS.index("up")) | (kinds == COURSE_KINDS.index("down"))


def _episode_state(env):
    state = getattr(env, "_pointfoot_full_episode", None)
    if state is None:
        def zeros(dtype=torch.float32):
            return torch.zeros(env.num_envs, device=env.device, dtype=dtype)
        state = {"passed": zeros(torch.bool), "stable_time": zeros(),
                 "traverse_time": zeros(), "stall_time": zeros(), "stuck": zeros(torch.bool),
                 "anchor_xy": env.scene["robot"].data.root_pos_w[:, :2].clone()}
        env._pointfoot_full_episode = state
        env._pointfoot_full_last_step = -1
        env._pointfoot_success_event = zeros(torch.bool)
    return state


@torch.no_grad()
def update_full_course_step(env):
    """每真实策略步幂等更新；可由终止项、真实奖励 wrapper 或父 snapshot 调用。

    在 reset 前采接触/末级高度。脉冲只在下一 common_step_counter 刷新，
    因而同一步自动 reset 后父训练仍可用它归档终止转移。
    """
    _history(env)
    state = _episode_state(env)
    step = int(env.common_step_counter)
    if env._pointfoot_full_last_step == step:
        return env._pointfoot_success_event
    env._pointfoot_full_last_step = step
    env._pointfoot_success_event.zero_()
    robot, sensor = env.scene["robot"], env.scene["contact_forces"]
    names = [f"{leg}_FOOT_LINK" for leg in ("LF", "RF", "LB", "RB")]
    body_ids, _ = robot.find_bodies(names, preserve_order=True)
    sensor_ids, _ = sensor.find_bodies(names, preserve_order=True)
    if len(body_ids) != 4 or len(sensor_ids) != 4:
        raise RuntimeError("full-v3 需要四个独立固定球脚刚体，不能使用轮子资产")
    # 使用当前力而非三帧最大值，防止把起跳前的旧接触当成当前着地。
    force = sensor.data.net_forces_w[:, sensor_ids].norm(dim=-1)
    contact = force > 8.
    origin = env.scene.env_origins
    offset = robot.data.root_pos_w - origin
    bottom = robot.data.body_pos_w[:, body_ids, 2] - FOOT_RADIUS - origin[:, 2:3]
    kinds = full_kind_indices(env)
    passed_now = stable_course_pass(
        offset, robot.data.projected_gravity_b, env.scene.terrain.terrain_levels, kinds,
        bottom, contact, robot.data.root_lin_vel_w[:, 2],
    )
    base_ids, _ = sensor.find_bodies("base_link", preserve_order=True)
    if len(base_ids):
        base_force = sensor.data.net_forces_w_history[:, :, base_ids].norm(dim=-1).amax(dim=(1, 2))
        passed_now &= base_force <= 1.
    alive_step = env.episode_length_buf > 0
    dt = float(env.step_dt)
    state["stable_time"] = torch.where(passed_now & alive_step, state["stable_time"] + dt, 0.)
    success = (state["stable_time"] >= .12 - 1e-6) & ~state["passed"] & alive_step
    env._pointfoot_success_event.copy_(success)
    state["passed"] |= success

    command = env.command_manager.get_command("base_velocity")
    stairs = full_stair_mask(env)
    # 楼梯只认纯双向直行；平台/坡/碎石只认带X分量的遍历，纯横移中性。
    # 平地的通过线覆盖X/Y两轴，因此任意平移均可计入。
    straight = (command[:, 0].abs() > .1) & (command[:, 1:].abs().amax(-1) < .05)
    flat = kinds == COURSE_KINDS.index("flat")
    travel = torch.where(flat, command[:, :2].norm(dim=-1) > .1,
                         torch.where(stairs, straight, command[:, 0].abs() > .1)) & alive_step
    state["traverse_time"] += travel * dt
    moved = (robot.data.root_pos_w[:, :2] - state["anchor_xy"]).norm(dim=-1) >= .12
    state["anchor_xy"][moved | ~travel] = robot.data.root_pos_w[moved | ~travel, :2]
    state["stall_time"] = torch.where(travel & ~moved, state["stall_time"] + dt, 0.)
    state["stuck"] |= (state["stall_time"] >= 8.) & ~state["passed"]
    return env._pointfoot_success_event


class PointfootFullVelocityCommand(UniformVelocityCommand):
    """普通环境 3–6 秒重采样，楼梯遍历保持到整个回合结束。"""

    def _resample_command(self, env_ids):
        ids = _ids(self._env, env_ids)
        if self.cfg.manual_control:
            self.vel_command_b[ids] = 0.
            self.is_standing_env[ids] = False
            self.is_heading_env[ids] = False
            self.time_left[ids] = 1e9
            return
        history = _history(self._env)
        values, traverse = sample_mixed_commands(
            len(ids), full_stair_mask(self._env)[ids], history.limits,
            stop_probability=self.cfg.rel_standing_envs,
        )
        self.vel_command_b[ids] = values
        self.is_standing_env[ids] = values.abs().amax(-1) == 0.
        self.is_heading_env[ids] = False
        # 只锁住遍历命令；调整/停止样本仍按普通时间重采样。
        self.time_left[ids[traverse]] = float(self._env.cfg.episode_length_s) + 1.


@configclass
class PointfootFullVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = PointfootFullVelocityCommand
    manual_control: bool = False


def _apply_command_ranges(env, history):
    term = env.command_manager.get_term("base_velocity")
    x, y, yaw = history.limits
    term.cfg.ranges.lin_vel_x = (-x, x)
    term.cfg.ranges.lin_vel_y = (-y, y)
    term.cfg.ranges.ang_vel_z = (-yaw, yaw)
    # 扩速只影响下一次采样，不改变当前已锁定的楼梯遍历指令。


def update_full_command_curriculum(env, iteration, velocity_mae):
    """父训练每轮调用一次；返回 0..5 的命令阶段，参数误差顺序为 vx/vy/wz。"""
    history = _history(env)
    stage = history.update_commands(iteration, velocity_mae)
    _apply_command_ranges(env, history)
    return stage


def full_course_state_dict(env):
    """保存累计课程和命令门控；当前回合的物理状态不属于 checkpoint。"""
    return _history(env).state_dict()


def load_full_course_state_dict(env, state):
    """原地恢复张量，保持与父脚本旧三个 _pointfoot_course_* 引用兼容。"""
    history = _history(env)
    history.load_state_dict(state)
    _apply_command_ranges(env, history)
    terrain = env.scene.terrain
    terrain.terrain_levels.clamp_min_(0)
    terrain.terrain_levels.copy_(torch.minimum(terrain.terrain_levels, history.caps[terrain.terrain_types]))
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    reset_full_course_state(env, None)


full_course_load_state_dict = load_full_course_state_dict


def full_summary(env):
    """六种地形×六等级统计、最近通过率、列上限和当前卡住数量。"""
    result = _history(env).summary()
    terrain = env.scene.terrain
    kinds = full_kind_indices(env)
    state = _episode_state(env)
    result["active_stuck"] = int(state["stuck"].sum())
    for index, kind in enumerate(COURSE_KINDS):
        selected = kinds == index
        result["terrain"][kind]["level_counts"] = torch.bincount(
            terrain.terrain_levels[selected], minlength=len(COMMAND_LIMITS)
        ).tolist()
        result["terrain"][kind]["active_stuck_by_level"] = torch.bincount(
            terrain.terrain_levels[selected & state["stuck"]], minlength=len(COMMAND_LIMITS)
        ).tolist()
    return result


def full_terrain_curriculum(env, env_ids):
    """真实通过才晋级，有效遍历失败降级，转向/停止保留原级且不污染窗口。"""
    history = _history(env)
    state = _episode_state(env)
    ids = _ids(env, env_ids)
    ids = ids[env.episode_length_buf[ids] > 0]
    terrain = env.scene.terrain
    if not len(ids):
        return terrain.terrain_levels.float().mean()
    update_full_course_step(env)
    columns = terrain.terrain_types[ids]
    levels = terrain.terrain_levels[ids].clone()
    fallen = env.termination_manager.terminated[ids]
    passed = state["passed"][ids] & ~fallen
    duration = env.episode_length_buf[ids] * float(env.step_dt)
    # 中性指令占主导的回合不因为没过阶而失败；真实通过仍然可以计入。
    attempted = (state["traverse_time"][ids] >= .5 * duration) & (state["traverse_time"][ids] > 0.)
    eligible = passed | (attempted & ((duration >= 5.) | fallen | state["stuck"][ids]))
    history.record(columns, levels, passed, eligible, fallen, state["stuck"][ids])
    next_level = levels + passed.long() - (eligible & ~passed).long()
    terrain.terrain_levels[ids] = torch.minimum(next_level.clamp_min(0), history.caps[columns])
    terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], columns]
    return terrain.terrain_levels.float().mean()


def full_course_exit(env):
    update_full_course_step(env)
    delta = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    return (delta[:, 0].abs() > EXIT_DISTANCE) | (delta[:, 1].abs() > EXIT_DISTANCE)


def full_course_stuck(env):
    """遍历指令连续8秒未移动12cm，结束无效回合并计入卡住统计。"""
    update_full_course_step(env)
    return _episode_state(env)["stuck"]


def full_low_support_height(env, minimum_height=.12):
    """共享脚接触/载荷/扫描支撑面，避免20cm台阶边缘的机身单射线误终止。"""
    from .pointfoot_stairs import pointfoot_support_clearance

    return pointfoot_support_clearance(env) < minimum_height


def reset_full_course_root(env, env_ids, free_yaw_range=(-math.pi, math.pi)):
    """仅 flat 默认随机朝向；评测可收窄 free_yaw_range，其余五类保持 ±.08。"""
    ids = _ids(env, env_ids)
    robot = env.scene["robot"]
    state = robot.data.default_root_state[ids].clone()
    state[:, :3] += env.scene.env_origins[ids]
    state[:, :2] += torch.empty(len(ids), 2, device=env.device).uniform_(-.08, .08)
    kinds = full_kind_indices(env)[ids]
    free = kinds == COURSE_KINDS.index("flat")
    fraction = torch.rand(len(ids), device=env.device)
    lower, upper = free_yaw_range
    yaw = torch.where(free, lower + (upper - lower) * fraction, -.08 + .16 * fraction)
    state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    amplitude = state.new_tensor((.05, .05, .03, .03, .03, .05))
    state[:, 7:13] = (2 * torch.rand(len(ids), 6, device=env.device) - 1) * amplitude
    robot.write_root_pose_to_sim(state[:, :7], env_ids=ids)
    robot.write_root_velocity_to_sim(state[:, 7:13], env_ids=ids)


def reset_full_course_state(env, env_ids):
    """复位回合计时和通过锁存；成功事件脉冲必须保留到下一真实策略步。"""
    state = _episode_state(env)
    ids = _ids(env, env_ids)
    for key in ("passed", "stable_time", "traverse_time", "stall_time", "stuck"):
        state[key][ids] = 0
    state["anchor_xy"][ids] = env.scene["robot"].data.root_pos_w[ids, :2]
    # 不改 _pointfoot_full_last_step / _pointfoot_success_event，避免同一步清除终止转移。
