"""大狗 DigDogv1：3.495kg、12路腿位置驱动、四个KFE脚端。

适配点足的 SuperMimic 过台阶框架。原始 MJCF 是 CAD 的 Y-up/Z-forward 坐标，
由 build_digdog_mjcf.py 外层包一个四元数 (0.5,0.5,0.5,0.5) 的 base_link 根节点，
转换为框架统一的 X-forward / Y-left / Z-up 坐标。
"""
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
import hashlib
import json

import mujoco
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.utils.spec_config import CollisionCfg

REPO_ROOT = Path(__file__).resolve().parents[3]
DIGDOG_XML = REPO_ROOT / 'description/dig_dogv1/robot.xml'
GO2W_XML = DIGDOG_XML
DIGDOG_URDF = REPO_ROOT / 'description/dig_dogv1/robot.urdf'

LEGS = ('FL', 'FR', 'RL', 'RR')
LEG_JOINT_NAMES = tuple(f'{leg}_{kind}_joint' for leg in LEGS for kind in ('hip', 'thigh', 'calf'))
FOOT_BODIES = tuple(f'{leg}_foot_link' for leg in LEGS)
WHEEL_GEOM_NAMES = tuple(f'{leg}_foot_collision_{i}' for leg in LEGS for i in range(4))
WHEEL_JOINT_NAMES = ()

# 0.18m 站高（脚端接触点相对 base_link），由 contact-point IK 求得；膝角约 0.51rad。
DEFAULT_Q = (
    -0.01227221, -0.17743997, 0.50644177,   # FL
    0.01227224, 0.17743997, -0.50644176,    # FR
    0.01227221, 0.17743998, -0.50644182,    # RL
    -0.01227224, -0.17743997, 0.5064418,    # RR
)
STAND_HEIGHT = .18
INIT_HEIGHT = .189          # 默认站姿下脚网格最低点刚好触地
FOOT_RADIUS = .0084         # 接触点与实际网格最低点的竖向差

# KFE 局部坐标系下的固定足跟接触点（额定站姿下由网格最低接触区质心确定）。
FOOT_LOCAL = {
    'FL': (-.1255, .105, .0271),
    'FR': (-.1255, -.105, -.0075),
    'RL': (-.1255, -.105, -.0075),
    'RR': (-.1255, .105, -.0075),
}


@dataclass(kw_only=True)
class RobotEntityCfg(EntityCfg):
    robot_id: str = 'rpwd_digdog'


def _check(robot):
    if robot != 'rpwd_digdog':
        raise ValueError('此包只接受 DigDogv1，禁止交叉加载点足/轮足模型')


def get_spec(robot='rpwd_digdog'):
    _check(robot)
    return mujoco.MjSpec.from_file(str(DIGDOG_XML))


def get_go2w_robot_cfg(robot='rpwd_digdog'):
    _check(robot)
    return RobotEntityCfg(
        robot_id=robot,
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0., 0., INIT_HEIGHT),
            joint_pos={name: DEFAULT_Q[i] for i, name in enumerate(LEG_JOINT_NAMES)},
            joint_vel={'.*': 0.},
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=('.*_collision_.*',),
                contype=2,
                conaffinity=1,
                condim={'.*_foot_collision_.*': 3, '.*': 1},
                priority={'.*_foot_collision_.*': 1, '.*': 0},
            ),
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=tuple(f'^{n}$' for n in LEG_JOINT_NAMES)),),
            soft_joint_pos_limit_factor=.95,
        ),
    )


@lru_cache(None)
def profile(robot='rpwd_digdog'):
    _check(robot)
    return dict(
        robot_id=robot, kp=[20.] * 12, kd=[.5] * 12, effort=[10.] * 12,
        foot_radius=FOOT_RADIUS, stand_height=STAND_HEIGHT, default_q=list(DEFAULT_Q),
        self_collisions=False,
        source_sha256=hashlib.sha256(DIGDOG_URDF.read_bytes() + DIGDOG_XML.read_bytes()).hexdigest(),
        adapter_version=1,
        gait_frequency=1.25, gait_duty=.6, gait_clearance=.05, residual_scale=.6,
        collision_force=15., height_band_flat=[.17, .22], height_band_rough=[.15, .24],
        action_scale=[.5] * 12, rate_limits=[6., 8., 10.] * 4,
    )


def robot_signature(robot='rpwd_digdog'):
    return hashlib.sha256(json.dumps(profile(robot), sort_keys=True).encode()).hexdigest()


def robot_id(env):
    return env.cfg.scene.entities['robot'].robot_id


@lru_cache(None)
def rp_kinematic_constants(device='cpu'):
    """从适配后的 MJCF 读取零位腿链；脚点使用额定站姿下的固定足跟接触点。"""
    import numpy as np
    import torch
    m = mujoco.MjModel.from_xml_path(str(DIGDOG_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    base = d.xpos[m.body('base_link').id].copy()
    origins, limits, signs = [], [], []
    for leg in LEGS:
        hip = d.xanchor[m.joint(f'{leg}_hip_joint').id] - base
        thigh = d.xanchor[m.joint(f'{leg}_thigh_joint').id] - base
        calf = d.xanchor[m.joint(f'{leg}_calf_joint').id] - base
        bid = m.body(f'{leg}_foot_link').id
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.xquat[bid])
        foot = R.reshape(3, 3) @ np.asarray(FOOT_LOCAL[leg]) + d.xpos[bid] - base
        origins.append([hip, thigh - hip, calf - thigh, foot - calf])
        lim = []
        for kind in ('hip', 'thigh', 'calf'):
            lim.append(m.jnt_range[m.joint(f'{leg}_{kind}_joint').id].copy())
        limits.append(lim)
        a_roll = d.xaxis[m.joint(f'{leg}_hip_joint').id]
        a_thigh = d.xaxis[m.joint(f'{leg}_thigh_joint').id]
        a_calf = d.xaxis[m.joint(f'{leg}_calf_joint').id]
        s_roll = 1. if a_roll[0] >= 0 else -1.
        s_pitch = 1. if a_thigh[1] >= 0 else -1.
        s_calf = 1. if a_calf[1] >= 0 else -1.
        if abs(s_calf - s_pitch) > 1e-6:
            raise ValueError(f'{leg} 膝轴与髋轴方向不一致')
        signs.append([s_roll, s_pitch, s_pitch])
    return dict(
        origins=torch.tensor(np.asarray(origins), dtype=torch.float32, device=device),
        centers=torch.zeros(4, 3, device=device),
        radius=FOOT_RADIUS,
        limits=torch.tensor(np.asarray(limits), dtype=torch.float32, device=device).reshape(12, 2),
        default_q=torch.tensor(DEFAULT_Q, dtype=torch.float32, device=device),
        signs=torch.tensor(np.asarray(signs), dtype=torch.float32, device=device),
        height_band_flat=torch.tensor([.17, .22], device=device),
        height_band_rough=torch.tensor([.15, .24], device=device),
        clearance=.05,
    )
