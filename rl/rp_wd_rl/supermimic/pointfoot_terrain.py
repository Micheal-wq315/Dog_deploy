"""与 Go2W 相同的静态地形查询、解析奖励和想象位姿推进，适配12关节点足。"""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch
import torch.nn.functional as F


# 保留点足任务的低速目标和无力矩惩罚约定；真实环境与想象共享这些项。
REWARD_WEIGHTS = {
    "track_linear_velocity": 1.5, "track_angular_velocity": 0.75,
    "upright": 0.3, "leg_posture": -0.05, "action_rate_l2": -0.01,
    "dof_pos_limits": -1.0, "lin_vel_z_l2": -0.25, "ang_vel_xy_l2": -0.05,
    "stair_clearance": -1.0, "obstacle_progress": 0.5,
    "nonfoot_collision": -0.5, "terminated": -100.0,
}

# full-v3 使用轻量净空区间与点足接触速度；不含轮子滚动、腿速或力矩代价。
FULL_REWARD_WEIGHTS = {
    **REWARD_WEIGHTS,
    "upright": 0.15, "leg_posture": -0.02,
    "lin_vel_z_l2": -0.05, "ang_vel_xy_l2": -0.02,
    "base_height": -0.10, "standstill": -1.0, "foot_slip": -0.15,
}


def advance_pose(pose, delta):
    c, s = pose[:, 2].cos(), pose[:, 2].sin()
    return pose + torch.stack((c * delta[:, 0] - s * delta[:, 1],
                               s * delta[:, 0] + c * delta[:, 1], delta[:, 2]), -1)


class PointfootTerrainField:
    """启动时将 PhysX 地形网格光栅化；想象时按位姿重新查询，不预测地形残差。"""

    def __init__(self, heights, origin, resolution=0.04):
        self.h, self.origin, self.res = heights, origin, resolution
        x, y = torch.meshgrid(torch.linspace(-1, 1, 21, device=heights.device),
                              torch.linspace(-0.6, 0.6, 13, device=heights.device), indexing="ij")
        self.offsets = torch.stack((x.flatten(), y.flatten()), -1)

    def sample(self, xy):
        finite = torch.isfinite(xy).all(-1)
        clean = torch.where(torch.isfinite(xy), xy, torch.zeros_like(xy))
        gx = ((clean[..., 0] - self.origin[0]) / self.res).clamp(0, self.h.shape[1] - 1)
        gy = ((clean[..., 1] - self.origin[1]) / self.res).clamp(0, self.h.shape[0] - 1)
        x = gx.floor().long().clamp(0, self.h.shape[1] - 2)
        y = gy.floor().long().clamp(0, self.h.shape[0] - 2)
        a, b = gx - x, gy - y
        value = ((1 - a) * self.h[y, x] + a * self.h[y, x + 1]) * (1 - b)
        value = value + ((1 - a) * self.h[y + 1, x] + a * self.h[y + 1, x + 1]) * b
        return torch.where(finite, value, torch.full_like(value, float("nan")))

    def scan(self, state, pose):
        c, s = pose[:, 2:3].cos(), pose[:, 2:3].sin()
        dx, dy = self.offsets[:, 0], self.offsets[:, 1]
        xy = pose[:, None, :2] + torch.stack((c * dx - s * dy, s * dx + c * dy), -1)
        root_z = state[:, :1] + self.sample(pose[:, :2])[:, None]
        return (root_z - self.sample(xy)).clamp(-1.5, 1.5)

    @classmethod
    @torch.no_grad()
    def from_env(cls, env, resolution=0.04):
        from isaaclab.utils.warp import raycast_mesh
        cfg = env.scene.terrain.cfg.terrain_generator
        # 地形生成器以世界原点为中心。额外覆盖边框，避免越过赛道后虚构台阶。
        half_x, half_y = cfg.num_rows * cfg.size[0] / 2 + 2, cfg.num_cols * cfg.size[1] / 2 + 2
        xs = torch.arange(-half_x, half_x + resolution, resolution, device=env.device)
        ys = torch.arange(-half_y, half_y + resolution, resolution, device=env.device)
        y, x = torch.meshgrid(ys, xs, indexing="ij")
        starts = torch.stack((x.flatten(), y.flatten(), torch.full_like(x.flatten(), 5.0)), -1)
        scanner = env.scene["height_scanner"]
        mesh = scanner.meshes[scanner.cfg.mesh_prim_paths[0]]
        chunks = []
        for chunk in starts.split(100_000):
            directions = torch.zeros_like(chunk)
            directions[:, 2] = -1
            hits = raycast_mesh(chunk, directions, mesh, max_dist=10.0)[0]
            chunks.append(torch.nan_to_num(hits[:, 2], nan=0.0, posinf=0.0, neginf=0.0))
        return cls(torch.cat(chunks).reshape(len(ys), len(xs)), (-half_x, -half_y), resolution)


def sample_local_scan(scan, xy):
    # 与Go2W相同的x-major 21x13米制扫描；值为base_z-地面z。
    gx = ((xy[..., 0] + 1) * 10).clamp(0, 20)
    gy = ((xy[..., 1] + 0.6) * 10).clamp(0, 12)
    ix, iy = gx.floor().long().clamp(0, 19), gy.floor().long().clamp(0, 11)
    a, b = gx - ix, gy - iy
    def at(dx, dy):
        return scan.gather(1, (ix + dx) * 13 + iy + dy)
    return -1.0 * (((1 - a) * at(0, 0) + a * at(1, 0)) * (1 - b)
                  + ((1 - a) * at(0, 1) + a * at(1, 1)) * b)


class PointfootTaskRewards:
    """从真实 URDF 读取运动学；同一解析奖励用于仿真与世界模型想象。"""

    def __init__(self, joint_names, default_q, limits, profile="legacy"):
        if profile not in ("legacy", "full"):
            raise ValueError(f"未知点足奖励 profile: {profile!r}")
        self.profile = profile
        self.weights = dict(REWARD_WEIGHTS if profile == "legacy" else FULL_REWARD_WEIGHTS)
        self.hard_min_height = 0.17 if profile == "legacy" else 0.12
        self.fall_angle = math.radians(55 if profile == "legacy" else 60)
        self.default_q, self.limits = default_q, limits
        root = ET.parse(Path(__file__).resolve().parents[3] /
                        "description/RP_wd_pointfoot/urdf/RP_wd_pointfoot.urdf").getroot()
        joints = {j.attrib["name"]: j for j in root.findall("joint")}
        indices, anchors, offsets = [], [], []
        radii = []
        for leg in ("LF", "RF", "LB", "RB"):
            names = [f"{leg}_{kind}_JOINT" for kind in ("ABAD", "HIP", "KENN")]
            indices.append([joint_names.index(name) for name in names])
            chain = [joints[name] for name in names] + [joints[f"{leg}_FOOT_FIXED_JOINT"]]
            values = []
            for j in chain:
                origin = j.find("origin")
                if any(abs(float(v)) > 1e-6 for v in origin.attrib.get("rpy", "0 0 0").split()):
                    raise ValueError("点足解析运动学需要零RPY关节原点")
                values.append([float(v) for v in origin.attrib["xyz"].split()])
            for j, expected in zip(chain[:3], ((1, 0, 0), (0, 1, 0), (0, 1, 0))):
                axis = tuple(float(v) for v in j.find("axis").attrib["xyz"].split())
                if axis != expected:
                    raise ValueError("点足解析 Jacobian 需要 URDF 中的 X/Y/Y 关节轴")
            collision = root.find(f"link[@name='{leg}_FOOT_LINK']/collision")
            radii.append(float(collision.find("geometry/sphere").attrib["radius"]))
            if profile == "full":
                # full 的 FK 跟踪真实球脚中心；legacy 保留历史的 FOOT_LINK 原点。
                center = [float(v) for v in collision.find("origin").attrib["xyz"].split()]
                values[-1] = [a + b for a, b in zip(values[-1], center)]
            anchors.append(values[0])
            offsets.append(values[1:])
        self.ids = torch.tensor(indices, device=default_q.device)
        self.anchors = default_q.new_tensor(anchors)
        self.offsets = default_q.new_tensor(offsets)
        if len(set(radii)) != 1:
            raise ValueError("四个点脚必须使用相同球形碰撞半径")
        self.radius = radii[0]

    @staticmethod
    def rotate_y(point, angle):
        c, s = angle.cos(), angle.sin()
        x, y, z = point.unbind(-1)
        return torch.stack((c * x + s * z, y.expand_as(c), -s * x + c * z), -1)

    @staticmethod
    def rotate_x(point, angle):
        c, s = angle.cos(), angle.sin()
        x, y, z = point.unbind(-1)
        return torch.stack((x.expand_as(c), c * y - s * z, s * y + c * z), -1)

    def feet_body(self, state):
        """机身坐标系下四脚位置，顺序为 LF/RF/LB/RB，单位米。"""
        q = (state[:, 10:22] + self.default_q)[:, self.ids]
        point = self.rotate_y(self.offsets[:, 2], q[:, :, 2]) + self.offsets[:, 1]
        point = self.rotate_y(point, q[:, :, 1]) + self.offsets[:, 0]
        return self.rotate_x(point, q[:, :, 0]) + self.anchors

    @staticmethod
    def _to_heading(state, point):
        # 由投影重力去除 roll/pitch，得到与 scan 一致的水平、随 yaw 转动的坐标系。
        up = -F.normalize(state[:, 1:4], dim=-1, eps=1e-6)
        cp = up[:, 1:].square().sum(-1).clamp_min(1e-12).sqrt()
        sr, cr, sp = up[:, 1] / cp, up[:, 2] / cp, -up[:, 0]
        x, y, z = point.unbind(-1)
        yr, zr = cr[:, None] * y - sr[:, None] * z, sr[:, None] * y + cr[:, None] * z
        return torch.stack((cp[:, None] * x + sp[:, None] * zr, yr,
                            -sp[:, None] * x + cp[:, None] * zr), -1)

    def feet(self, state):
        return self._to_heading(state, self.feet_body(state))

    def feet_jacobian(self, state, point_body=None):
        """返回 [B,4,3,3] 的解析位置 Jacobian；每腿三列对应 ABAD/HIP/KENN。"""
        q = (state[:, 10:22] + self.default_q)[:, self.ids]
        point = self.feet_body(state) if point_body is None else point_body
        hip = self.rotate_x(self.offsets[:, 0], q[:, :, 0]) + self.anchors
        knee = self.rotate_y(self.offsets[:, 1], q[:, :, 1]) + self.offsets[:, 0]
        knee = self.rotate_x(knee, q[:, :, 0]) + self.anchors
        axis_x = torch.zeros_like(point)
        axis_x[..., 0] = 1
        axis_y = torch.stack((torch.zeros_like(q[:, :, 0]),
                              q[:, :, 0].cos(), q[:, :, 0].sin()), -1)
        return torch.stack((torch.linalg.cross(axis_x, point - self.anchors),
                            torch.linalg.cross(axis_y, point - hip),
                            torch.linalg.cross(axis_y, point - knee)), -1)

    def feet_velocity(self, state):
        """水平随航向坐标系中的实际脚速：R(v_body + omega×p + J qdot)，单位 m/s。"""
        point = self.feet_body(state)
        jacobian = self.feet_jacobian(state, point)
        relative = (jacobian @ state[:, 22:34][:, self.ids].unsqueeze(-1)).squeeze(-1)
        velocity = (state[:, None, 4:7]
                    + torch.linalg.cross(state[:, None, 7:10].expand_as(point), point)
                    + relative)
        return self._to_heading(state, velocity)

    @staticmethod
    def _support_weights(state, contacts, force):
        # contacts 可为真实 0/1 或想象概率；force 为四脚非负合力载荷（现有 adapter 为 N/500）。
        weights = state.new_ones((state.shape[0], 4))
        if contacts is None and force is None:
            return weights * 0
        for value, is_contact in ((contacts, True), (force, False)):
            if value is not None:
                value = value.to(device=state.device, dtype=state.dtype)
                if value.shape != weights.shape:
                    raise ValueError("点足 contacts/force 必须是 [batch,4]，顺序 LF/RF/LB/RB")
                value = torch.where(torch.isfinite(value), value, torch.zeros_like(value)).clamp_min(0)
                weights = weights * (value.clamp_max(1) if is_contact else value)
        return weights

    def support_clearance(self, state, scan, contacts=None, force=None):
        """返回 [B] 支撑面相对机身净空（米），缺少有效支撑/足下扫描时回退 state[:,0]。

        scan 为 [B,273] 的 base_z-ground_z；按真实球脚水平位置采样地面，
        接触概率与载荷共同加权。只改变机身正下方的地面不会改变有支撑时的结果。
        此函数保持 Torch 梯度，不缓存时间状态，供真实、想象、日志和评测共用。
        """
        point = self.feet(state)
        below, valid = self._sample_ground(scan, point[..., :2])
        below = -below
        weights = self._support_weights(state, contacts, force)
        valid = valid & torch.isfinite(point).all(-1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        below = torch.where(valid, below, torch.zeros_like(below))
        total = weights.sum(-1)
        clearance = (weights * below).sum(-1) / total.clamp_min(1e-8)
        return torch.where(total > 1e-8, clearance, state[:, 0])

    @staticmethod
    def _sample_ground(scan, xy):
        # 先清理缺测再插值，避免无效脚的 NaN 在反向传播中通过 0*NaN 污染有效梯度。
        finite_scan, finite_xy = torch.isfinite(scan), torch.isfinite(xy).all(-1)
        clean_scan = torch.where(finite_scan, scan, torch.zeros_like(scan))
        clean_xy = torch.where(torch.isfinite(xy), xy, torch.zeros_like(xy))
        ground = sample_local_scan(clean_scan, clean_xy)
        coverage = -sample_local_scan(finite_scan.to(scan.dtype), clean_xy)
        return ground, finite_xy & (coverage >= 1 - 1e-6)

    @staticmethod
    def _local_roughness(scan):
        # 与参考 terrain_posture 一样只看支撑/即将接近的地面，远处障碍不放松姿态。
        local = scan.reshape(-1, 21, 13)[:, 5:16, 3:10].flatten(1)
        finite = torch.isfinite(local)
        high = local.masked_fill(~finite, -torch.inf).amax(-1)
        low = local.masked_fill(~finite, torch.inf).amin(-1)
        relief = torch.where(finite.any(-1), high - low, torch.zeros_like(high))
        return ((relief - 0.02) / 0.08).clamp(0, 1)

    def _full_terms(self, state, action, previous, command, scan, collision, terminated,
                    contacts, force):
        v, w = state[:, 4:7], state[:, 7:10]
        q = state[:, 10:22]
        absolute = q + self.default_q
        limits = ((self.limits[:, 0] - absolute).clamp_min(0)
                  + (absolute - self.limits[:, 1]).clamp_min(0)).sum(-1)
        rough = self._local_roughness(scan)
        clearance = self.support_clearance(state, scan, contacts, force)
        low, high = 0.22 - 0.02 * rough, 0.42 + 0.02 * rough
        # legacy 从未直接奖励高度；full 新增轻量区间代价，区间内严格为零。
        height_cost = (((low - clearance).clamp_min(0) / 0.10).square()
                       + ((clearance - high).clamp_min(0) / 0.10).square())
        legs = q[:, self.ids].square()
        posture = 2 * legs[:, :, 0].sum(-1) + (1 - 0.875 * rough) * legs[:, :, 1:].sum((1, 2))
        gravity = F.normalize(state[:, 1:4], dim=-1, eps=1e-6)
        # 约 10° 内不锁定水平姿态，超出后平缓下降；倒置不会获得 upright 满分。
        tilt_excess = (math.cos(math.radians(10)) + gravity[:, 2]).clamp_min(0)
        point = self.feet(state)
        below, valid_below = self._sample_ground(scan, point[..., :2])
        speed = command[:, :2].norm(dim=-1)
        direction = command[:, :2] / speed[:, None].clamp_min(0.1)
        ahead, valid_ahead = self._sample_ground(scan, point[..., :2] + 0.16 * direction[:, None])
        valid = valid_below & valid_ahead
        rise = torch.where(valid, ahead - below, torch.zeros_like(ahead))
        gate = ((rise - 0.01) / 0.04).clamp(0, 1) * (speed > 0.1)[:, None]
        deficit = (torch.where(valid, ahead, point[..., 2])
                   + self.radius + 0.025 - point[..., 2]).clamp_min(0) / 0.20
        along = (v[:, :2] * command[:, :2]).sum(-1) / speed.clamp_min(0.1)
        foot_velocity = self.feet_velocity(state)
        support = self._support_weights(state, contacts, force)
        slip = ((foot_velocity[..., :2].square().sum(-1) * support).sum(-1)
                / support.sum(-1).clamp_min(1e-8))
        standing = (command[:, :2].norm(dim=-1) < 0.05) & (command[:, 2].abs() < 0.05)
        # 仅零速指令约束停稳；不用腿部关节速度惩罚代替脚滑，运动时不压制摆腿。
        relative = (self.feet_jacobian(state) @ state[:, 22:34][:, self.ids].unsqueeze(-1)).squeeze(-1)
        standstill = standing * (v.square().sum(-1) + w.square().sum(-1)
                                + 0.25 * relative.square().sum(-1).mean(-1))
        return {
            "track_linear_velocity": torch.exp(-(v[:, :2] - command[:, :2]).square().sum(-1) / 0.25),
            "track_angular_velocity": torch.exp(-(w[:, 2] - command[:, 2]).square() / 0.5),
            "upright": torch.exp(-tilt_excess / 0.18), "leg_posture": posture,
            "action_rate_l2": (action - previous).square().sum(-1),
            "dof_pos_limits": limits, "base_height": height_cost,
            "lin_vel_z_l2": v[:, 2].square(), "ang_vel_xy_l2": w[:, :2].square().sum(-1),
            # 净空只减罚，不对抬脚/悬脚持续给正奖励；进度必须来自有符号机身运动。
            "stair_clearance": (gate * deficit / (1 + deficit)).mean(-1),
            "obstacle_progress": rough * (speed > 0.1) * (along / speed.clamp_min(0.2)).clamp(-1, 1),
            "standstill": standstill, "foot_slip": slip,
            "nonfoot_collision": collision.reshape(-1), "terminated": terminated.reshape(-1),
        }

    def terms(self, state, action, previous, command, scan, collision, terminated,
              contacts=None, force=None):
        if self.profile == "full":
            return self._full_terms(state, action, previous, command, scan, collision, terminated,
                                    contacts, force)
        v, w = state[:, 4:7], state[:, 7:10]
        q = state[:, 10:22]
        absolute = q + self.default_q
        limits = ((self.limits[:, 0] - absolute).clamp_min(0)
                  + (absolute - self.limits[:, 1]).clamp_min(0)).sum(-1)
        point = self.feet(state)
        below = sample_local_scan(scan, point[..., :2])
        speed = command[:, :2].norm(dim=-1)
        direction = command[:, :2] / speed[:, None].clamp_min(0.1)
        ahead = sample_local_scan(scan, point[..., :2] + 0.16 * direction[:, None])
        rise = ahead - below
        gate = ((rise - 0.01) / 0.04).clamp(0, 1) * (speed > 0.1)[:, None]
        deficit = (ahead + self.radius + 0.025 - point[..., 2]).clamp_min(0) / 0.20
        relief = scan.max(-1).values - scan.min(-1).values
        rough = ((relief - 0.02) / 0.08).clamp(0, 1)
        along = (v[:, :2] * command[:, :2]).sum(-1) / speed.clamp_min(0.1)
        return {
            "track_linear_velocity": torch.exp(-(v[:, :2] - command[:, :2]).square().sum(-1) / 0.25),
            "track_angular_velocity": torch.exp(-(w[:, 2] - command[:, 2]).square() / 0.5),
            "upright": torch.exp(-state[:, 1:3].square().sum(-1) / 0.6**2),
            "leg_posture": q.square().sum(-1) * (1 - 0.8 * rough),
            "action_rate_l2": (action - previous).square().sum(-1),
            "dof_pos_limits": limits,
            "lin_vel_z_l2": v[:, 2].square(), "ang_vel_xy_l2": w[:, :2].square().sum(-1),
            "stair_clearance": (gate * deficit / (1 + deficit)).mean(-1),
            "obstacle_progress": rough * (speed > 0.1) * (along / speed.clamp_min(0.2)).clamp(-1, 1),
            "nonfoot_collision": collision.reshape(-1), "terminated": terminated.reshape(-1),
        }

    def reward(self, *args, **kwargs):
        terms = self.terms(*args, **kwargs)
        return sum(self.weights[name] * value for name, value in terms.items())
