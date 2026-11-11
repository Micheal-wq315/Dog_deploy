"""大狗点足版：17.038kg、12路腿位置驱动、四个固定球脚，无轮电机。"""
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
import hashlib
import json
import xml.etree.ElementTree as E
import mujoco
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.utils.spec_config import CollisionCfg

REPO_ROOT=Path(__file__).resolve().parents[3]
URDF=REPO_ROOT/'description/RP_wd_pointfoot/urdf/RP_wd_pointfoot.urdf'
GO2W_XML=REPO_ROOT/'description/RP_wd_pointfoot/mjcf/RP_wd_pointfoot.xml'
RPWD_XML=GO2W_XML
RP_LEGS=('LF','RF','LB','RB')
RP_KINDS=('ABAD','HIP','KENN')
LEGS=('FL','FR','RL','RR')
LEG_JOINT_NAMES=tuple(f'{leg}_{kind}_joint' for leg in LEGS for kind in ('hip','thigh','calf'))
WHEEL_JOINT_NAMES=()
WHEEL_GEOM_NAMES=tuple(f'{leg}_foot_collision' for leg in LEGS)
DEFAULT_Q=(0.,.72,-1.41)*4
STAND_HEIGHT=.3403

@dataclass(kw_only=True)
class RobotEntityCfg(EntityCfg):
    robot_id: str='rpwd_pointfoot'

def get_spec(robot='rpwd_pointfoot'):
    if robot!='rpwd_pointfoot':raise ValueError('此包只接受大狗点足版，禁止交叉加载轮足模型')
    spec=mujoco.MjSpec.from_file(str(GO2W_XML))
    for old,new in zip(RP_LEGS,LEGS):spec.body(old+'_FOOT_LINK').name=new+'_foot_link'
    return spec

def get_go2w_robot_cfg(robot='rpwd_pointfoot'):
    if robot!='rpwd_pointfoot':raise ValueError(robot)
    return RobotEntityCfg(robot_id=robot,spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,STAND_HEIGHT+.02),
            joint_pos={'.*_hip_joint':0.,'.*_thigh_joint':.72,'.*_calf_joint':-1.41},joint_vel={'.*':0.}),
        collisions=(CollisionCfg(geom_names_expr=('.*collision.*',),contype=2,conaffinity=1,
            condim={'.*_foot_collision':3,'.*_collision_.*':1},priority={'.*_foot_collision':1,'.*_collision_.*':0}),),
        articulation=EntityArticulationInfoCfg(actuators=(BuiltinPositionActuatorCfg(
            target_names_expr=LEG_JOINT_NAMES,stiffness=100.,damping=3.,effort_limit=42.,armature=.01),),
            soft_joint_pos_limit_factor=.9))

@lru_cache(None)
def profile(robot='rpwd_pointfoot'):
    if robot!='rpwd_pointfoot':raise ValueError(robot)
    return dict(robot_id=robot,kp=[100.]*12,kd=[3.]*12,effort=[42.]*12,
        foot_radius=.022,stand_height=STAND_HEIGHT,default_q=list(DEFAULT_Q),self_collisions=False,
        source_sha256=hashlib.sha256(URDF.read_bytes()+GO2W_XML.read_bytes()).hexdigest(),adapter_version=1,
        gait_frequency=1.5,gait_duty=.6,gait_clearance=.065,residual_scale=.6,collision_force=15.,height_band_flat=[.28,.40],height_band_rough=[.24,.44],
        action_scale=[.25,.6,.8]*4,rate_limits=[6.,8.,10.]*4)

def robot_signature(robot='rpwd_pointfoot'):
    return hashlib.sha256(json.dumps(profile(robot),sort_keys=True).encode()).hexdigest()

def robot_id(env):return env.cfg.scene.entities['robot'].robot_id

@lru_cache(None)
def rp_kinematic_constants(device='cpu'):
    import torch
    root=E.parse(URDF).getroot();joints={j.get('name'):j for j in root.findall('joint')}
    origins=[];centers=[];limits=[]
    for leg in RP_LEGS:
        chain=[]
        for kind in (*RP_KINDS,'FOOT_FIXED'):
            j=joints[f'{leg}_{kind}_JOINT']
            if any(abs(float(v))>1e-8 for v in j.find('origin').get('rpy','0 0 0').split()):raise ValueError('点足FK要求当前URDF零位rpy为零')
            chain.append([float(x) for x in j.find('origin').get('xyz').split()])
            if kind in RP_KINDS:
                if j.find('axis').get('xyz')!=('1 0 0' if kind=='ABAD' else '0 1 0'):raise ValueError('关节轴不匹配')
                l=j.find('limit');a,b=float(l.get('lower')),float(l.get('upper'));m=(a+b)/2;limits.append((m+(a-m)*.9,m+(b-m)*.9))
        origins.append(chain)
        n=root.find(f"link[@name='{leg}_FOOT_LINK']/collision/origin")
        centers.append([float(x) for x in n.get('xyz').split()])
    return dict(origins=torch.tensor(origins,device=device),centers=torch.tensor(centers,device=device),
        radius=.022,limits=torch.tensor(limits,device=device),default_q=torch.tensor(DEFAULT_Q,device=device))
