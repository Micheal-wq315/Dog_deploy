"""两种机器人的物理配置；状态/动作仍统一为FL、FR、RL、RR逐腿排列。"""
from dataclasses import dataclass, fields
from functools import lru_cache, partial
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import CollisionCfg
from . import go2w

LEG_JOINT_NAMES = go2w.LEG_JOINT_NAMES
WHEEL_JOINT_NAMES = go2w.WHEEL_JOINT_NAMES
WHEEL_GEOM_NAMES = go2w.WHEEL_GEOM_NAMES
GO2W_XML = go2w.GO2W_XML
REPO_ROOT = Path(__file__).resolve().parents[3]
RPWD_XML = REPO_ROOT / 'description/RP_wd/mjcf/RP_wd.xml'
RPWD_URDF = REPO_ROOT / 'description/RP_wd/urdf/RP_wd.urdf'
LEGS = ('FL', 'FR', 'RL', 'RR')
RP_LEGS = ('LF', 'RF', 'LB', 'RB')
RP_KINDS = ('ABAD', 'HIP', 'KENN')
RP_TO_CANONICAL = {f'{leg}_{kind}_JOINT': f'{can}_{target}_joint'
                   for leg, can in zip(RP_LEGS, LEGS)
                   for kind, target in zip((*RP_KINDS, 'FOOT'), ('hip', 'thigh', 'calf', 'wheel'))}


@dataclass(kw_only=True)
class RobotEntityCfg(EntityCfg):
    robot_id: str = 'go2w'


@lru_cache(None)
def hardware_contract():
    path = REPO_ROOT / 'rl/rp_wd_rl/assets/rp_wd_contract.py'
    spec = importlib.util.spec_from_file_location('_rpwd_mjlab_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(None)
def wheel_shape():
    root = ET.parse(RPWD_URDF).getroot()
    shapes = []
    for leg in RP_LEGS:
        cylinder = root.find(f"link[@name='{leg}_FOOT_LINK']/collision/geometry/cylinder")
        shapes.append((float(cylinder.get('radius')), float(cylinder.get('length')) / 2))
    if any(abs(a - b) > 1e-8 for shape in shapes for a, b in zip(shape, shapes[0])):
        raise ValueError('四个轮子的几何不同，须显式支持逐轮尺寸')
    if abs(shapes[0][0] - hardware_contract().WHEEL_RADIUS) > 1e-8:
        raise ValueError('URDF轮半径与部署契约不一致')
    return shapes[0]


def get_spec(robot='go2w'):
    if robot == 'go2w':
        return go2w.get_spec()
    if robot != 'rpwd':
        raise ValueError(f'未知机器人 {robot}')
    spec = mujoco.MjSpec.from_file(str(RPWD_XML))
    spec.modelname = 'RP_wd_wheel_supermimic'
    for act in list(spec.actuators):
        spec.delete(act)
    spec.body('base_link').pos = (0., 0., hardware_contract().STAND_HEIGHT + .02)
    spec.body('base_link').add_freejoint(name='floating_base')
    for old, new in RP_TO_CANONICAL.items():
        spec.joint(old).name = new
    for old, new in zip(RP_LEGS, LEGS):
        foot = spec.body(f'{old}_FOOT_LINK')
        foot.name = f'{new}_wheel_link'
        for geom in foot.geoms:
            if geom.contype or geom.conaffinity:
                geom.name = f'{new}_wheel_collision'
                geom.friction = (.8, .02, .001)
    for i, geom in enumerate(spec.geoms):
        if geom.contype or geom.conaffinity:
            if not geom.name:
                geom.name = f'body_{i}_collision'
            geom.margin = 0.
            # 仅与地形接触，保持RP_wd既有的关闭自碰撞设置。
            geom.contype, geom.conaffinity = 2, 1
        else:
            geom.group = 2
    return spec


def get_go2w_robot_cfg(robot='go2w'):
    if robot == 'go2w':
        original = go2w.get_go2w_robot_cfg()
        return RobotEntityCfg(**{f.name: getattr(original, f.name) for f in fields(EntityCfg)}, robot_id=robot)
    c = hardware_contract()
    if robot != 'rpwd':
        raise ValueError(robot)
    return RobotEntityCfg(
        robot_id=robot, spec_fn=partial(get_spec, robot),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0., 0., c.STAND_HEIGHT + .02),
            joint_pos={'.*_hip_joint': c.DEFAULT_JOINT_POS['ABAD'],
                       '.*_thigh_joint': c.DEFAULT_JOINT_POS['HIP'],
                       '.*_calf_joint': c.DEFAULT_JOINT_POS['KENN'], '.*_wheel_joint': 0.},
            joint_vel={'.*': 0.}),
        collisions=(CollisionCfg(geom_names_expr=('.*_collision',), contype=2, conaffinity=1,
                    condim={'.*_wheel_collision': 3, 'body_.*_collision': 1},
                    priority={'.*_wheel_collision': 1, 'body_.*_collision': 0}),),
        articulation=EntityArticulationInfoCfg(actuators=(
            BuiltinPositionActuatorCfg(target_names_expr=LEG_JOINT_NAMES,
                stiffness=c.STIFFNESS['HIP'], damping=c.DAMPING['HIP'],
                effort_limit=c.EFFORT_LIMIT_SIM['HIP'], armature=.01),
            BuiltinVelocityActuatorCfg(target_names_expr=WHEEL_JOINT_NAMES,
                damping=c.DAMPING['FOOT'], effort_limit=c.EFFORT_LIMIT_SIM['FOOT'], armature=.01),
        ), soft_joint_pos_limit_factor=.9))


@lru_cache(None)
def profile(robot):
    if robot == 'go2w':
        return dict(robot_id=robot, kp=[20., 20., 40.] * 4, kd=[1., 1., 2.] * 4,
                    effort=[23.5, 23.5, 45.] * 4, wheel_kd=2., wheel_effort=45.,
                    source_sha256=hashlib.sha256(GO2W_XML.read_bytes()).hexdigest(),
                    wheel_radius=.086, wheel_half_width=.026, adapter_version=1, wheel_friction=[.8,.02,.001])
    c = hardware_contract()
    return dict(robot_id=robot, kp=[c.STIFFNESS[k] for k in RP_KINDS] * 4,
                kd=[c.DAMPING[k] for k in RP_KINDS] * 4,
                effort=[c.EFFORT_LIMIT_SIM[k] for k in RP_KINDS] * 4,
                wheel_kd=c.DAMPING['FOOT'], wheel_effort=c.EFFORT_LIMIT_SIM['FOOT'],
                source_sha256=hashlib.sha256(RPWD_URDF.read_bytes() + RPWD_XML.read_bytes() + (REPO_ROOT/'rl/rp_wd_rl/assets/rp_wd_contract.py').read_bytes()).hexdigest(),
                wheel_radius=wheel_shape()[0], wheel_half_width=wheel_shape()[1], adapter_version=1, self_collisions=False, wheel_friction=[.8,.02,.001])


def robot_signature(robot):
    # 路径不进入指纹，使同一模型在不同worktree或机器上仍可恢复。
    return hashlib.sha256(json.dumps(profile(robot), sort_keys=True).encode()).hexdigest()


def robot_id(env):
    return env.cfg.scene.entities['robot'].robot_id


def rp_kinematic_constants(device):
    """从URDF读取零位刚体变换；轮角不在WM状态中，径向偏心误差上界约0.08mm。"""
    import torch
    root = ET.parse(RPWD_URDF).getroot()
    joints = {j.attrib['name']: j for j in root.findall('joint')}
    links = {x.attrib['name']: x for x in root.findall('link')}
    origins, centers = [], []
    for leg in RP_LEGS:
        chain = []
        for kind, axis in zip((*RP_KINDS, 'FOOT'), ('1 0 0', '0 1 0', '0 1 0', '0 1 0')):
            joint = joints[f'{leg}_{kind}_JOINT']
            origin = joint.find('origin')
            if any(abs(float(v)) > 1e-8 for v in origin.get('rpy', '0 0 0').split()):
                raise ValueError('RP_wd FK目前要求关节零位rpy为零')
            actual = [float(v) for v in joint.find('axis').get('xyz').split()]
            if actual != [float(v) for v in axis.split()]:
                raise ValueError('URDF关节轴变化，须同步更新可微FK')
            chain.append([float(v) for v in origin.get('xyz').split()])
        origins.append(chain)
        centers.append([float(v) for v in links[f'{leg}_FOOT_LINK'].find('collision/origin').get('xyz').split()])
    return {'origins': torch.tensor(origins, device=device),
            'centers': torch.tensor(centers, device=device),
            'radius': wheel_shape()[0], 'half_width': wheel_shape()[1]}
