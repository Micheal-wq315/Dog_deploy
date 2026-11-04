"""点足世界模型训练与确定性评测共享的状态适配；导入本模块不启动仿真。"""
from __future__ import annotations

from pathlib import Path
import torch

from rp_wd_rl.supermimic.pointfoot import ACTION_DIM, POLICY_DIM
from rp_wd_rl.supermimic.pointfoot_terrain import PointfootTerrainField, PointfootTaskRewards

LEGACY_TASK = "Isaac-Velocity-Rough-RPWD-Pointfoot-SuperMimic-WM-v0"
FULL_TASK = "Isaac-Velocity-Rough-RPWD-Pointfoot-SuperMimic-Full-v0"


def resolve_pointfoot_checkpoint(path=None):
    if path:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate
    roots = [Path("/media/rmer/data/rpwd_training") / name for name in
             ("pointfoot_supermimic", "pointfoot_supermimic_full")]
    candidates = [p for root in roots for p in root.glob("*/teacher_latest.pt")]
    if not candidates:
        raise FileNotFoundError("未找到点足正式模型，请用--checkpoint指定文件")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    final = latest.parent / "teacher_final.pt"
    return final if final.exists() else latest


class PointfootStateAdapter:
    """从仿真真值提取34维状态、273维高度图和点脚接触监督。"""

    def __init__(self, env):
        self.env = env.unwrapped
        self.robot = self.env.scene["robot"]
        action_term = self.env.action_manager.get_term("joint_pos")
        self.leg_ids = torch.as_tensor(
            list(range(self.robot.num_joints))[action_term._joint_ids]
            if isinstance(action_term._joint_ids, slice) else action_term._joint_ids,
            device=self.env.device, dtype=torch.long
        )
        if len(self.leg_ids) != ACTION_DIM or self.env.action_manager.total_action_dim != ACTION_DIM:
            raise RuntimeError("点足 SuperMimic 要求12个主动关节、12维策略动作")
        foot_names = [f"{leg}_FOOT_LINK" for leg in ("LF", "RF", "LB", "RB")]
        self.foot_body_ids, _ = self.robot.find_bodies(foot_names, preserve_order=True)
        self.nonfoot_names = [name for name in self.robot.body_names if name not in foot_names]
        self.nonfoot_sensor_ids, _ = self.env.scene["contact_forces"].find_bodies(
            self.nonfoot_names, preserve_order=True
        )
        self.foot_sensor_ids, _ = self.env.scene["contact_forces"].find_bodies(
            foot_names, preserve_order=True
        )
        if len(self.foot_body_ids) != 4 or len(self.foot_sensor_ids) != 4:
            raise RuntimeError("ContactSensor 未解析到四个点脚刚体")
        names = [self.robot.joint_names[i] for i in self.leg_ids.cpu().tolist()]
        self.task_rewards = PointfootTaskRewards(
            names, self.robot.data.default_joint_pos[0, self.leg_ids],
            self.robot.data.soft_joint_pos_limits[0, self.leg_ids],
            profile="full" if getattr(self.env.cfg, "supermimic_revision", 2) >= 3 else "legacy")
        self.field = PointfootTerrainField.from_env(self.env)
        self.env._pointfoot_field = self.field
        self.env._pointfoot_task_rewards = self.task_rewards
        materials = self.robot.root_physx_view.get_material_properties().to(self.env.device)
        # 此URDF四个脚端各只有一个球形shape；刚体顺序不等于shape顺序。
        shape_ids, offset = {}, 0
        for body, link_path in zip(self.robot.body_names, self.robot.root_physx_view.link_paths[0]):
            body_view = self.robot._physics_sim_view.create_rigid_body_view(link_path)
            shape_ids[body] = offset
            offset += body_view.max_shapes
        if offset != materials.shape[1]:
            raise RuntimeError(f"碰撞shape映射失败：USD={offset}, PhysX={materials.shape[1]}")
        friction = materials[:, [shape_ids[name] for name in foot_names], 1]
        base_id = self.robot.find_bodies("base_link")[0][0]
        com = self.robot.root_physx_view.get_coms().to(self.env.device)[:, base_id, :3]
        reference_com = com.new_tensor((-0.0064993, -0.0009295, 0.0021204))
        self.context = torch.cat((friction, com - reference_com), -1)
        print(f"[信息] 本地包={Path(__import__('rp_wd_rl').__file__).resolve()} "
              f"盲策略={POLICY_DIM}→12 地形缓存={self.field.h.numel()*4/2**20:.1f}MiB", flush=True)


    def capture(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        env, robot = self.env, self.robot
        data = robot.data
        ground_z = self.field.sample(data.root_pos_w[:, :2])
        ground_z = torch.where(torch.isfinite(ground_z), ground_z, env.scene.env_origins[:, 2])
        clearance = data.root_pos_w[:, 2] - ground_z
        joint_pos = data.joint_pos[:, self.leg_ids] - data.default_joint_pos[:, self.leg_ids]
        joint_vel = data.joint_vel[:, self.leg_ids]
        state = torch.cat((
            clearance[:, None], data.projected_gravity_b,
            data.root_lin_vel_b, data.root_ang_vel_b, joint_pos, joint_vel,
        ), dim=-1)
        if state.shape[-1] != 34:
            raise RuntimeError(f"点足物理状态应为34维，实际为{state.shape[-1]}")

        quat = data.root_quat_w
        qw, qx, qy, qz = quat.unbind(-1)
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy.square() + qz.square()))
        pose = torch.cat((data.root_pos_w[:, :2], yaw[:, None]), dim=-1)
        command = env.command_manager.get_command("base_velocity")
        scan = self.field.scan(state, pose)
        if scan.shape[-1] != 273:
            raise RuntimeError(f"地形高度图应为273维，实际为{scan.shape[-1]}")

        contact_sensor = env.scene["contact_forces"]
        history = contact_sensor.data.net_forces_w_history
        if getattr(env.cfg, "supermimic_revision", 2) >= 3:
            foot_force = contact_sensor.data.net_forces_w[:, self.foot_sensor_ids].norm(dim=-1)
        else:
            foot_force = history[:, :, self.foot_sensor_ids].norm(dim=-1).amax(dim=1)
        foot_force = foot_force.clamp(0.0, 500.0) / 500.0
        contacts = (foot_force > (8.0 / 500.0)).float()
        nonfoot_force = history[:, :, self.nonfoot_sensor_ids].norm(dim=-1).amax(dim=(1, 2))
        collision = (nonfoot_force > 10.0).float()[:, None]
        policy_history = observations["policy"].reshape(env.num_envs, -1)
        if policy_history.shape[-1] != POLICY_DIM:
            raise RuntimeError(f"点足策略历史应为{POLICY_DIM}维，实际为{policy_history.shape[-1]}")
        result = {
            "state": state,
            "root_pos": data.root_pos_w,
            "foot_positions": data.body_pos_w[:, self.foot_body_ids],
            "contacts_current": (contact_sensor.data.net_forces_w[:, self.foot_sensor_ids].norm(dim=-1) > 8.0).float(),
            "vertical_speed_world": data.root_lin_vel_w[:, 2],
            "pose": pose,
            "command": command,
            "scan": scan,
            "force": foot_force,
            "contacts": contacts,
            "collision": collision,
            "history": policy_history,
            "context": self.context,
            "bias": torch.cat((getattr(env, "_rp_wd_gyro_bias"),
                               getattr(env, "_rp_wd_joint_offset")[:, self.leg_ids]), -1),
        }
        return {key: value.clone() for key, value in result.items()}


COMMAND_MODE_NAMES = ("stand", "forward", "reverse", "left", "right", "yaw_left", "yaw_right", "combined")


def command_mode_ids(command):
    """按实际命令分组，防止正负速度均值互相抵消掩盖跟踪失败。"""
    active = command.abs() > 1e-5
    count = active.sum(-1)
    ids = torch.full((len(command),), 7, device=command.device, dtype=torch.long)
    ids[count == 0] = 0
    for component, positive, negative in ((0, 1, 2), (1, 3, 4), (2, 5, 6)):
        pure = (count == 1) & active[:, component]
        ids[pure & (command[:, component] > 0)] = positive
        ids[pure & (command[:, component] < 0)] = negative
    return ids


def set_pointfoot_command(env, policy_input, command):
    """钉住指令并更新最新一帧，历史由真实env.step推进。"""
    from rp_wd_rl.supermimic.pointfoot import FRAME_DIMS, HISTORY
    command = torch.as_tensor(command, dtype=torch.float32, device=env.device)
    term = env.command_manager.get_term("base_velocity")
    term.vel_command_b[:] = command
    term.is_standing_env[:] = command.square().sum(-1) < 1e-10
    term.time_left[:] = 1e9
    start = sum(FRAME_DIMS[:2]) * HISTORY + FRAME_DIMS[2] * (HISTORY - 1)
    policy_input[:, start:start + 3] = command


def command_skills_ready(metrics, minimum_samples=64):
    """扩速必须覆盖七种独立运动，避免小指令的全局平均误差掩盖站桩。"""
    import math
    for name in COMMAND_MODE_NAMES[:7]:
        row = metrics.get(name, {})
        if row.get("samples", 0) < minimum_samples:
            return False
        arrays = [row.get(key, []) for key in ("velocity_mae", "actual_mean", "command_mean")]
        if any(len(values) != 3 or not all(math.isfinite(float(v)) for v in values) for values in arrays):
            return False
        error, actual, command = arrays
        if name == "stand":
            if any(value > bound for value, bound in zip(error, (.06, .06, .12))):
                return False
        else:
            axis = 0 if name in ("forward", "reverse") else 1 if name in ("left", "right") else 2
            target = command[axis]
            if abs(target) < 1e-5 or actual[axis] * (1 if target > 0 else -1) < .5 * abs(target):
                return False
            if error[axis] > max(.05, .6 * abs(target)):
                return False
    return True


def course_frame_pass(snapshot, origins, kinds, levels):
    """验收复用训练的末级脚底/当前接触/垂向速度判定，避免掠过算通过。"""
    from rp_wd_rl.supermimic.pointfoot_courses import FOOT_RADIUS, stable_course_pass
    return stable_course_pass(
        snapshot["root_pos"] - origins, snapshot["state"][:, 1:4], levels, kinds,
        snapshot["foot_positions"][..., 2] - FOOT_RADIUS - origins[:, 2:3],
        snapshot["contacts_current"].bool(), snapshot["vertical_speed_world"],
    )


class CoursePassTracker:
    """单试次通过锁存；已结束环境的后续自动复位不能更新进度。"""
    def __init__(self, count, device, dt):
        self.dt = dt
        self.stable_time = torch.zeros(count, device=device)
        self.reached = torch.zeros(count, dtype=torch.bool, device=device)
        self.progress = torch.zeros(count, device=device)

    def update(self, distance, frame_pass, active, failed):
        self.progress.copy_(torch.where(active & torch.isfinite(distance),
                                       torch.maximum(self.progress, distance.nan_to_num(0.0)), self.progress))
        self.stable_time.copy_(torch.where(active & frame_pass & ~failed,
                                          self.stable_time + self.dt, 0.0))
        self.reached |= active & (self.stable_time >= .12 - 1e-6) & ~failed
