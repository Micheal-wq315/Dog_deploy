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

    def __init__(self, joint_names, default_q, limits):
        self.default_q, self.limits = default_q, limits
        root = ET.parse(Path(__file__).resolve().parents[3] /
                        "description/RP_wd_pointfoot/urdf/RP_wd_pointfoot.urdf").getroot()
        joints = {j.attrib["name"]: j for j in root.findall("joint")}
        indices, anchors, offsets = [], [], []
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
            anchors.append(values[0])
            offsets.append(values[1:])
        self.ids = torch.tensor(indices, device=default_q.device)
        self.anchors = default_q.new_tensor(anchors)
        self.offsets = default_q.new_tensor(offsets)
        self.radius = 0.022

    @staticmethod
    def rotate_y(point, angle):
        c, s = angle.cos(), angle.sin()
        x, y, z = point.unbind(-1)
        return torch.stack((c * x + s * z, y.expand_as(c), -s * x + c * z), -1)

    def feet(self, state):
        q = (state[:, 10:22] + self.default_q)[:, self.ids]
        point = self.rotate_y(self.offsets[:, 2], q[:, :, 2]) + self.offsets[:, 1]
        point = self.rotate_y(point, q[:, :, 1]) + self.offsets[:, 0]
        x, y, z = point.unbind(-1)
        c, s = q[:, :, 0].cos(), q[:, :, 0].sin()
        point = torch.stack((x, c * y - s * z, s * y + c * z), -1) + self.anchors
        up = -F.normalize(state[:, 1:4], dim=-1, eps=1e-6)
        cp = up[:, 1:].square().sum(-1).clamp_min(1e-12).sqrt()
        sr, cr, sp = up[:, 1] / cp, up[:, 2] / cp, -up[:, 0]
        x, y, z = point.unbind(-1)
        yr, zr = cr[:, None] * y - sr[:, None] * z, sr[:, None] * y + cr[:, None] * z
        return torch.stack((cp[:, None] * x + sp[:, None] * zr, yr,
                            -sp[:, None] * x + cp[:, None] * zr), -1)

    def terms(self, state, action, previous, command, scan, collision, terminated):
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

    def reward(self, *args):
        terms = self.terms(*args)
        return sum(REWARD_WEIGHTS[name] * value for name, value in terms.items())
