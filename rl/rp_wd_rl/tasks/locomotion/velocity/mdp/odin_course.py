"""Odin 专项赛道：目标指令、真实进度、安全升级和有界接触代价。"""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz


def odin_course_mask(env):
    """地形列固定分配；平地列用于自由遥控，其余列用于定向通过。"""
    terrain = env.scene.terrain
    generator = terrain.cfg.terrain_generator
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if generator is None or not hasattr(terrain, "terrain_types"):
        return result
    cfgs = list(generator.sub_terrains.values())
    total = sum(cfg.proportion for cfg in cfgs)
    cumulative = 0.0
    for cfg in cfgs:
        lower = cumulative / total
        cumulative += cfg.proportion
        upper = cumulative / total
        if getattr(cfg, "course_kind", "manual") != "manual":
            column = (terrain.terrain_types.float() + 0.5) / generator.num_cols
            result |= (column >= lower) & (column < upper)
    return result


class OdinCourseVelocityCommand(UniformVelocityCommand):
    """将训练目标编码为既有 [vx, vy, wz]，专项赛道终点始终在出生点前方。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.course_mask = odin_course_mask(env)
        self.goal_w = torch.zeros(env.num_envs, 3, device=env.device)
        self.speed = torch.zeros(env.num_envs, device=env.device)
        self.previous_distance = torch.zeros_like(self.speed)
        self.progress_speed = torch.zeros_like(self.speed)
        self.max_nonfoot_force = torch.zeros_like(self.speed)
        self._state_step = -1
        self.nonfoot_ids, _ = env.scene["contact_forces"].find_bodies(
            ["base_link", ".*_ABAD_LINK", ".*_HIP_LINK", ".*_KENN_LINK"]
        )
        self.metrics["course_distance"] = torch.zeros_like(self.speed)
        self.metrics["course_success"] = torch.zeros_like(self.speed)

    def refresh_progress(self):
        """多个奖励/终止共享同一状态快照，每个策略步只推进一次。"""
        if self._state_step == self._env.common_step_counter:
            return
        distance = torch.linalg.vector_norm(self.goal_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.progress_speed = (self.previous_distance - distance) / self._env.step_dt
        self.previous_distance.copy_(distance)
        sensor = self._env.scene["contact_forces"]
        force = sensor.data.net_forces_w_history[:, :, self.nonfoot_ids].norm(dim=-1).amax(dim=(1, 2))
        self.max_nonfoot_force = torch.maximum(self.max_nonfoot_force, force)
        self._state_step = self._env.common_step_counter

    def reset(self, env_ids=None):
        # Isaac Lab先reset指令、后清零episode_length_buf，不能用步数识别新回合。
        extras = super().reset(env_ids)
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )
        self.goal_w[ids] = self._env.scene.env_origins[ids]
        self.goal_w[ids, 0] += self.cfg.goal_distance
        self.previous_distance[ids] = torch.linalg.vector_norm(
            self.goal_w[ids, :2] - self.robot.data.root_pos_w[ids, :2], dim=1
        )
        self.progress_speed[ids] = 0.0
        self.max_nonfoot_force[ids] = 0.0
        # 同步新指令，避免复位后首个观测携带上一地块的目标方向。
        self._update_command()
        return extras

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        maximum = max(abs(v) for v in self.cfg.ranges.lin_vel_x)
        self.speed[ids] = torch.empty(len(ids), device=self.device).uniform_(0.6 * maximum, maximum)
        self.is_standing_env[ids[self.course_mask[ids]]] = False

    def _update_metrics(self):
        super()._update_metrics()
        if not hasattr(self, "goal_w"):
            return
        self.refresh_progress()
        self.metrics["course_distance"] = self.previous_distance * self.course_mask
        self.metrics["course_success"] = (
            (self.previous_distance < self.cfg.goal_tolerance) & self.course_mask
        ).float()

    def _update_command(self):
        super()._update_command()
        delta_w = self.goal_w - self.robot.data.root_pos_w
        delta_w[:, 2] = 0.0
        delta_b = quat_apply_inverse(self.robot.data.root_quat_w, delta_w)
        distance = delta_w[:, :2].norm(dim=1).clamp_min(1e-6)
        velocity = delta_b[:, :2] / distance[:, None] * self.speed[:, None]
        velocity *= (distance / 0.7).clamp(max=1.0)[:, None]
        ids = self.course_mask
        self.vel_command_b[ids, :2] = velocity[ids]
        heading_error = torch.atan2(delta_b[:, 1], delta_b[:, 0])
        self.vel_command_b[ids, 2] = (0.8 * heading_error[ids]).clamp(*self.cfg.ranges.ang_vel_z)


@configclass
class OdinCourseVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = OdinCourseVelocityCommand
    goal_distance: float = 4.4
    goal_tolerance: float = 0.35


def reset_odin_course_root(env, env_ids, pose_range, velocity_range):
    """专项赛道朝 +X 小范围复位；平地保留随机朝向和自由指令训练。"""
    from isaaclab.envs.mdp import reset_root_state_uniform

    reset_root_state_uniform(env, env_ids, pose_range, velocity_range)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[odin_course_mask(env)[ids]]
    if not len(ids):
        return
    robot = env.scene["robot"]
    state = robot.data.default_root_state[ids].clone()
    state[:, :3] += env.scene.env_origins[ids]
    state[:, :2] += torch.empty(len(ids), 2, device=env.device).uniform_(-0.06, 0.06)
    yaw = torch.empty(len(ids), device=env.device).uniform_(-0.10, 0.10)
    state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    state[:, 7:13] = 0.0
    robot.write_root_pose_to_sim(state[:, :7], env_ids=ids)
    robot.write_root_velocity_to_sim(state[:, 7:13], env_ids=ids)


def odin_goal_reached(env):
    term = env.command_manager.get_term("base_velocity")
    term.refresh_progress()
    from .terrain import root_clearance
    stable = (-env.scene["robot"].data.projected_gravity_b[:, 2] > 0.9063) & (root_clearance(env) > 0.27)
    return term.course_mask & (term.previous_distance < term.cfg.goal_tolerance) & stable


def odin_course_out_of_bounds(env, half_width: float = 2.6):
    term = env.command_manager.get_term("base_velocity")
    offset = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    return term.course_mask & ((offset[:, 1].abs() > half_width) | (offset[:, 0] < -0.45))


def odin_goal_progress(env):
    """只奖励缩短终点距离的真实位移；倒退扣分，瞬时大跳不能获得无界奖励。"""
    term = env.command_manager.get_term("base_velocity")
    term.refresh_progress()
    return (term.progress_speed / term.speed.clamp_min(0.12)).clamp(-1.0, 1.0) * term.course_mask


def odin_failure(env):
    """到点是正常终态；仅摔倒、机身碰撞、净空不足和越界等失败扣分。"""
    manager = env.termination_manager
    failure = torch.zeros_like(manager.terminated)
    for name in manager.active_terms:
        if name != "goal_reached" and not manager.get_term_cfg(name).time_out:
            failure |= manager.get_term(name)
    return failure


def odin_goal_success(env):
    """成功与失败同时触发时不发到点奖励。"""
    return env.termination_manager.get_term("goal_reached") & ~odin_failure(env)


def odin_course_tracking(env, std: float = 0.25):
    """跨越赛道保留速度驱动；只在绕墙列放松直接跟踪。"""
    term = env.command_manager.get_term("base_velocity")
    error = (term.command[:, :2] - env.scene["robot"].data.root_lin_vel_b[:, :2]).square().sum(dim=1)
    return torch.exp(-error / std**2) * torch.where(odin_bypass_mask(env), 0.35, 1.0)


def odin_bypass_mask(env):
    """只标识绕行墙，其他地形不应整体削弱速度跟踪。"""
    terrain = env.scene.terrain
    generator = terrain.cfg.terrain_generator
    mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    cfgs = list(generator.sub_terrains.values())
    total = sum(cfg.proportion for cfg in cfgs)
    cumulative = 0.0
    column = (terrain.terrain_types.float() + 0.5) / generator.num_cols
    for cfg in cfgs:
        lower = cumulative / total
        cumulative += cfg.proportion
        if getattr(cfg, "course_kind", "manual") == "bypass":
            mask |= (column >= lower) & (column < cumulative / total)
    return mask


def odin_joint_target_excess(env, asset_cfg: SceneEntityCfg):
    """约束超出物理关节范围的无效位置目标，不对电机力矩加惩罚。"""
    robot = env.scene[asset_cfg.name]
    target = robot.data.joint_pos_target[:, asset_cfg.joint_ids]
    limits = robot.data.joint_pos_limits[:, asset_cfg.joint_ids]
    excess = (limits[..., 0] - target).clamp_min(0) + (target - limits[..., 1]).clamp_min(0)
    return excess.square().mean(dim=1)


def odin_course_yaw_tracking(env, std: float = 0.5):
    """绕行中允许临时偏离直指终点的转向指令，平地保留完整转向精度。"""
    term = env.command_manager.get_term("base_velocity")
    error = (term.command[:, 2] - env.scene["robot"].data.root_ang_vel_b[:, 2]).square()
    return torch.exp(-error / std**2) * torch.where(term.course_mask, 0.5, 1.0)


def contact_load_excess(env, sensor_cfg: SceneEntityCfg, threshold: float = 180.0, scale: float = 180.0):
    """有界冲击代价；正常轮地支撑不扣分，大力撞击与跌落冲击才增加代价。"""
    sensor = env.scene[sensor_cfg.name]
    force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).amax(dim=(1, 2))
    return ((force - threshold) / scale).clamp(0.0, 1.0)


def odin_safe_terrain_levels(env, env_ids, max_level_schedule, max_nonfoot_force: float = 120.0):
    """到达终点且未发生严重非足端撞击才升级；失败降级，平地始终为0级。"""
    term = env.command_manager.get_term("base_velocity")
    term.refresh_progress()
    terrain = env.scene.terrain
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    cap = max_level_schedule[0][1]
    for step, level in max_level_schedule:
        if env.common_step_counter >= step:
            cap = level
    cap = min(cap, terrain.terrain_origins.shape[0] - 1)
    success = odin_goal_success(env)[ids]
    safe = term.max_nonfoot_force[ids] < max_nonfoot_force
    up = success & safe & term.course_mask[ids] & (terrain.terrain_levels[ids] < cap)
    down = ~success & term.course_mask[ids] & (env.episode_length_buf[ids] > 0)
    terrain.update_env_origins(ids, up, down)
    levels = terrain.terrain_levels[ids].clamp(max=cap)
    terrain.terrain_levels[ids] = torch.where(term.course_mask[ids], levels, 0)
    terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
    return terrain.terrain_levels.float().mean()
