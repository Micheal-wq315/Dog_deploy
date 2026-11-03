"""点足直线楼梯课程、越阶抬脚/着陆奖励与表现门控。"""

from __future__ import annotations

import numpy as np
import torch
import trimesh

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.warp import raycast_mesh


POINTFOOT_STEP_HEIGHTS = (0.02, 0.04, 0.08, 0.12, 0.16, 0.20)


def pointfoot_directed_stairs(difficulty: float, cfg: SubTerrainBaseCfg):
    """沿 +X 方向生成实体楼梯，起点前保留平地，结构按 Go2W 台阶课程序列。"""
    size_x, size_y = cfg.size
    # TerrainGenerator 在 curriculum 模式下给每一行采样
    # difficulty=(row+U[0,1))/num_rows；用 floor(difficulty*num_rows)
    # 可以保证一行对应一个难度，不让行内随机数把低阶混成高阶。
    level = min(len(POINTFOOT_STEP_HEIGHTS) - 1,
                max(0, int(float(difficulty) * len(POINTFOOT_STEP_HEIGHTS))))
    step_height = POINTFOOT_STEP_HEIGHTS[level]
    num_steps = 5
    tread = 0.30
    start_x = min(2.0, size_x * 0.35)
    bottom = -0.20
    boxes: list[trimesh.Trimesh] = []

    def add_block(x0: float, x1: float, top: float) -> None:
        if x1 <= x0:
            return
        height = top - bottom
        block = trimesh.creation.box(extents=(x1 - x0, size_y, height))
        block.apply_translation(((x0 + x1) * 0.5, size_y * 0.5, (top + bottom) * 0.5))
        boxes.append(block)

    # 相邻实体块只共面接触，不重叠；每一级顶面高度单调增加。
    add_block(0.0, start_x, 0.0)
    previous_x = start_x
    for step in range(num_steps):
        next_x = min(size_x, start_x + (step + 1) * tread)
        add_block(previous_x, next_x, (step + 1) * step_height)
        previous_x = next_x
    add_block(previous_x, size_x, num_steps * step_height)

    mesh = trimesh.util.concatenate(boxes)
    # 机器人从局部原点前方约1.5m处遇到第一级，课程行进方向固定朝 +X。
    origin = np.asarray((0.5, size_y * 0.5, 0.0), dtype=np.float64)
    return [mesh], origin


@configclass
class PointfootDirectedStairsCfg(SubTerrainBaseCfg):
    """Isaac Lab 子地形适配器。"""

    function = pointfoot_directed_stairs


def pointfoot_stair_state(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """每策略步共享一次射线和接触快照，产生抬腿与更高踏面着陆事件。"""
    step = int(env.common_step_counter)
    cached = getattr(env, "_pointfoot_stair_state", None)
    if cached is not None and cached["step"] == step:
        return cached["outputs"]

    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    foot_body_ids, _ = robot.find_bodies(
        [f"{leg}_FOOT_LINK" for leg in ("LF", "RF", "LB", "RB")], preserve_order=True
    )
    foot_sensor_ids, _ = sensor.find_bodies(
        [f"{leg}_FOOT_LINK" for leg in ("LF", "RF", "LB", "RB")], preserve_order=True
    )
    if len(foot_body_ids) != 4 or len(foot_sensor_ids) != 4:
        raise RuntimeError("点足楼梯奖励需要按 LF/RF/LB/RB 解析到四个独立脚端")

    foot_pos = robot.data.body_pos_w[:, foot_body_ids]
    quat = robot.data.root_quat_w
    qw, qx, qy, qz = quat.unbind(-1)
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy.square() + qz.square()))
    offsets = foot_pos.new_tensor((0.0, 0.10, 0.20, 0.30, 0.40, 0.50))
    starts = foot_pos[:, :, None, :].expand(-1, -1, len(offsets), -1).clone()
    starts[..., 0] += yaw.cos()[:, None, None] * offsets
    starts[..., 1] += yaw.sin()[:, None, None] * offsets
    starts[..., 2] += 1.0
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1.0
    scanner = env.scene["height_scanner"]
    mesh = scanner.meshes[scanner.cfg.mesh_prim_paths[0]]
    hits = raycast_mesh(starts.flatten(1, 2), directions.flatten(1, 2), mesh, max_dist=4.0)[0]
    raw_heights = hits[..., 2].reshape(env.num_envs, 4, len(offsets))
    ground_valid = torch.isfinite(raw_heights[..., 0])
    heights = torch.nan_to_num(raw_heights, nan=-10.0, posinf=-10.0, neginf=-10.0)
    ground = torch.where(ground_valid, heights[..., 0], torch.zeros_like(heights[..., 0]))
    rise = heights[..., 1:] - ground[..., None]
    valid_rise = (ground_valid[..., None] & torch.isfinite(rise)
                  & (rise > 0.015) & (rise <= 0.205))
    first = valid_rise.float().argmax(-1)
    next_height = torch.gather(heights[..., 1:], -1, first[..., None]).squeeze(-1)
    has_step = valid_rise.any(-1)
    rise_height = torch.where(has_step, next_height - ground, torch.zeros_like(ground))

    forces = sensor.data.net_forces_w_history[:, :, foot_sensor_ids, :].norm(dim=-1).amax(dim=1)
    contact = forces > 8.0
    force_n = forces.clamp(0.0, 500.0) / 500.0
    is_stair = _pointfoot_stair_mask(env)
    command = env.command_manager.get_command("base_velocity")
    forward_gate = (command[:, 0] > 0.05).float()
    swing = (~contact) & has_step & ground_valid & is_stair[:, None]
    foot_height = foot_pos[..., 2] - ground
    target_clearance = rise_height + 0.055
    clearance_score = torch.exp(-torch.square((foot_height - target_clearance) / 0.055))

    if cached is None or cached["contact"].shape[0] != env.num_envs:
        cached = {
            "step": -1,
            "valid": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "contact": torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device),
            "last_ground": torch.zeros(env.num_envs, 4, device=env.device),
            "lift_peak": torch.zeros(env.num_envs, 4, device=env.device),
            "last_x": robot.data.root_pos_w[:, 0].detach().clone(),
            "outputs": {},
        }
    reset = env.episode_length_buf == 0
    cached["valid"][reset] = False
    fresh = ~cached["valid"] & ground_valid.all(-1)
    cached["contact"][fresh] = contact[fresh]
    cached["last_ground"][fresh] = ground[fresh]
    cached["lift_peak"][fresh] = 0.0
    cached["last_x"][fresh] = robot.data.root_pos_w[fresh, 0]
    cached["valid"] |= ground_valid.all(-1)

    # 只奖励摆动中首次接近目标抬脚高度的增量；达到峰值后悬空不再持续得分。
    lift_gain = (clearance_score - cached["lift_peak"]).clamp_min(0.0)
    lift = (lift_gain * swing.float()).mean(-1) * forward_gate
    cached["lift_peak"] = torch.where(
        swing, torch.maximum(cached["lift_peak"], clearance_score),
        torch.where(contact, torch.zeros_like(cached["lift_peak"]), cached["lift_peak"]),
    )

    landing_up = (cached["valid"][:, None] & contact & ~cached["contact"]
                  & ground_valid & ((ground - cached["last_ground"]) > 0.01) & is_stair[:, None]
                  & forward_gate[:, None].bool())
    projected = robot.data.projected_gravity_b
    upright = (-projected[:, 2] > 0.80) & (projected[:, :2].norm(dim=-1) < 0.60)
    landing = (landing_up & upright[:, None]).float().mean(-1)
    cached["last_ground"] = torch.where(contact, ground, cached["last_ground"])
    cached["contact"] = contact

    root_x = robot.data.root_pos_w[:, 0]
    progress = ((root_x - cached["last_x"]) / max(env.step_dt, 1e-6)).clamp(0.0, 0.6) / 0.6
    progress *= is_stair.float() * forward_gate
    cached["last_x"] = root_x.detach()
    outputs = {"lift": lift, "landing": landing, "progress": progress,
               "contacts": contact.float(), "forces": force_n,
               "upcoming_rise": rise_height.clamp(0.0, 0.25).mean(-1)}
    cached["step"] = step
    cached["outputs"] = outputs
    env._pointfoot_stair_state = cached
    return outputs


def _pointfoot_stair_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    terrain = env.scene.terrain
    if not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # 配置固定四列：平地、楼梯、楼梯、粗糙地形。
    return (terrain.terrain_types == 1) | (terrain.terrain_types == 2)


def pointfoot_stair_lift_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return pointfoot_stair_state(env)["lift"]


def pointfoot_stair_landing_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return pointfoot_stair_state(env)["landing"]


def pointfoot_stair_progress_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return pointfoot_stair_state(env)["progress"]


def reset_pointfoot_stair_state(env: ManagerBasedRLEnv, env_ids) -> None:
    """在 Isaac Lab 的 reset 阶段清除部分环境的脚端接触边沿状态。"""
    cached = getattr(env, "_pointfoot_stair_state", None)
    if cached is None:
        return
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    cached["valid"][ids] = False
    cached["contact"][ids] = False
    cached["last_ground"][ids] = 0.0
    cached["lift_peak"][ids] = 0.0
    cached["last_x"][ids] = env.scene["robot"].data.root_pos_w[ids, 0]
    cached["step"] = -1


def pointfoot_terrain_curriculum(
    env: ManagerBasedRLEnv,
    env_ids,
    performance_window: int = 64,
    pass_threshold: float = 0.60,
    minimum_steps: tuple[int, ...] = (0, 1500, 4000, 8000, 12000, 16000),
) -> torch.Tensor:
    """仿 Go2W 的64回合成功率门控；走得远但摔倒不算通过楼梯。"""
    terrain = env.scene.terrain
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not hasattr(env, "_pointfoot_course_caps"):
        env._pointfoot_course_caps = torch.zeros(terrain.cfg.terrain_generator.num_cols,
                                                dtype=torch.long, device=env.device)
        env._pointfoot_course_trials = torch.zeros(
            terrain.cfg.terrain_generator.num_cols, terrain.cfg.terrain_generator.num_rows,
            performance_window, device=env.device
        )
        env._pointfoot_course_counts = torch.zeros(
            terrain.cfg.terrain_generator.num_cols, terrain.cfg.terrain_generator.num_rows,
            dtype=torch.long, device=env.device
        )

    if len(ids):
        levels = terrain.terrain_levels[ids].clone()
        types = terrain.terrain_types[ids]
        active = env.episode_length_buf[ids] > 0
        failed = env.termination_manager.terminated[ids]
        robot = env.scene["robot"]
        offset = robot.data.root_pos_w[ids] - env.scene.env_origins[ids]
        stairs = (types == 1) | (types == 2)
        h = offset.new_tensor(POINTFOOT_STEP_HEIGHTS)[levels]
        expected_z = 0.32 + 5.0 * h
        stable = ((-robot.data.projected_gravity_b[ids, 2] > 0.80)
                  & (robot.data.projected_gravity_b[ids, :2].norm(dim=-1) < 0.60))
        stair_success = ((offset[:, 0] > 3.3) & (offset[:, 1].abs() < 2.5)
                         & (offset[:, 2] > expected_z - 0.08)
                         & (offset[:, 2] < expected_z + 0.30) & stable & ~failed)
        ordinary_success = (offset[:, 0] > 3.3) & stable & ~failed
        success = torch.where(stairs, stair_success, ordinary_success) & active

        for index in range(len(ids)):
            if not bool(active[index]):
                continue
            terrain_type = int(types[index])
            level = int(levels[index])
            count = int(env._pointfoot_course_counts[terrain_type, level])
            slot = count % performance_window
            env._pointfoot_course_trials[terrain_type, level, slot] = success[index].float()
            env._pointfoot_course_counts[terrain_type, level] += 1
            count += 1
            if (level == int(env._pointfoot_course_caps[terrain_type])
                    and level < terrain.cfg.terrain_generator.num_rows - 1
                    and count >= performance_window
                    and int(env.common_step_counter) >= minimum_steps[level + 1]
                    and env._pointfoot_course_trials[terrain_type, level].mean() >= pass_threshold):
                env._pointfoot_course_caps[terrain_type] += 1

        next_level = torch.where(success, levels + 1, (levels - 1).clamp_min(0))
        caps = env._pointfoot_course_caps[types]
        terrain.terrain_levels[ids] = torch.minimum(next_level, caps).clamp_max(
            terrain.cfg.terrain_generator.num_rows - 1
        )
        terrain.env_origins[ids] = terrain.terrain_origins[
            terrain.terrain_levels[ids], terrain.terrain_types[ids]
        ]

    return terrain.terrain_levels.float().mean()


def pointfoot_model_reward(env: ManagerBasedRLEnv, term: str) -> torch.Tensor:
    """真实奖励和想象奖励调用同一组解析函数；在reset前保留碰撞监督。"""
    cached = getattr(env, "_pointfoot_model_reward_cache", None)
    if cached is None or cached[0] != env.common_step_counter:
        from rp_wd_rl.supermimic.pointfoot_terrain import PointfootTaskRewards
        robot = env.scene["robot"]
        ids = env.action_manager.get_term("joint_pos")._joint_ids
        names = list(robot.joint_names)[ids] if isinstance(ids, slice) else [robot.joint_names[i] for i in ids]
        default = robot.data.default_joint_pos[0, ids]
        task = getattr(env, "_pointfoot_task_rewards", None)
        if task is None:
            task = PointfootTaskRewards(names, default, robot.data.soft_joint_pos_limits[0, ids])
            env._pointfoot_task_rewards = task
        data = robot.data
        field = getattr(env, "_pointfoot_field", None)
        ground = (field.sample(data.root_pos_w[:, :2]) if field is not None else
                  env.scene["height_scanner"].data.ray_hits_w[:, 0, 2])
        state = torch.cat(((data.root_pos_w[:, 2] - ground)[:, None], data.projected_gravity_b,
                           data.root_lin_vel_b, data.root_ang_vel_b,
                           data.joint_pos[:, ids] - data.default_joint_pos[:, ids],
                           data.joint_vel[:, ids]), -1)
        if field is not None:
            q = data.root_quat_w
            yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                              1 - 2 * (q[:, 2].square() + q[:, 3].square()))
            pose = torch.cat((data.root_pos_w[:, :2], yaw[:, None]), -1)
            scan = field.scan(state, pose)
        else:
            hits = env.scene["supermimic_terrain_scanner"].data.ray_hits_w[..., 2]
            scan = (data.root_pos_w[:, 2:3] - hits).clamp(-1.5, 1.5)
        sensor = env.scene["contact_forces"]
        nonfoot = [name for name in robot.body_names if "FOOT_LINK" not in name]
        sensor_ids, _ = sensor.find_bodies(nonfoot, preserve_order=True)
        force = sensor.data.net_forces_w_history[:, :, sensor_ids].norm(dim=-1).amax(dim=(1, 2))
        collision = (force > 10).float()
        # env.step返回时终止环境已reset，必须先缓存这个时刻的真实碰撞标签。
        env._pointfoot_transition_collision = collision[:, None].clone()
        terms = task.terms(state, env.action_manager.action, env.action_manager.prev_action,
                           env.command_manager.get_command("base_velocity"), scan, collision,
                           env.termination_manager.terminated.float())
        cached = (env.common_step_counter, terms)
        env._pointfoot_model_reward_cache = cached
    return cached[1][term]


def pointfoot_supermimic_scan(env: ManagerBasedRLEnv) -> torch.Tensor:
    """21×13米制特权扫描，按Go2W的x-major顺序。"""
    hits = env.scene["supermimic_terrain_scanner"].data.ray_hits_w[..., 2]
    heights = env.scene["robot"].data.root_pos_w[:, 2:3] - hits
    return torch.nan_to_num(heights, nan=1.5, posinf=1.5, neginf=-1.5).clamp(-1.5, 1.5)


def pointfoot_course_exit(env: ManagerBasedRLEnv) -> torch.Tensor:
    """到达顶层后结束回合，避免穿出本赛道后被错误评估或跑入相邻环境。"""
    delta = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    return (delta[:, 0] > 4.5) | (delta[:, 0] < -0.35) | (delta[:, 1].abs() > 2.7)
