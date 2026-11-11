"""点足专用环境基础配置，按独立观测项堆叠五帧历史。"""
import torch
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import RewardTermCfg,EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg
from ..robot.config import get_go2w_robot_cfg,LEG_JOINT_NAMES,WHEEL_GEOM_NAMES
from ..core.kinematics import clock,reset_clock
from ..core.task import WEIGHTS,env_reward

def canonical_last_action(env):
 names=list(env.action_manager.get_term('joint_pos').target_names)
 return env.action_manager.action[:,[names.index(n) for n in LEG_JOINT_NAMES]]

def low_height(env):return env.scene['robot'].data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]<.16

def go2w_velocity_flat_env_cfg(play=False,history=5,actor_wheel_vel=False,robot='rpwd_pointfoot'):
 cfg=make_velocity_env_cfg();cfg.scene.entities={'robot':get_go2w_robot_cfg(robot)}
 cfg.scene.num_envs=1;cfg.scene.sensors=();cfg.sim.mujoco.timestep=.005;cfg.decimation=4
 cfg.sim.mujoco.cone='elliptic';cfg.sim.mujoco.impratio=10;cfg.sim.mujoco.ccd_iterations=50
 cfg.viewer.body_name='base_link';cfg.viewer.distance=2.
 legs=SceneEntityCfg('robot',joint_names=LEG_JOINT_NAMES,preserve_order=True)
 cfg.actions={'joint_pos':JointPositionActionCfg(entity_name='robot',actuator_names=LEG_JOINT_NAMES,
  scale={'.*_hip_joint':.25,'.*_thigh_joint':.6,'.*_calf_joint':.8},use_default_offset=True)}
 terms={
  'base_ang_vel':ObservationTermCfg(func=mdp.base_ang_vel,noise=UniformNoiseCfg(n_min=-.2,n_max=.2)),
  'gravity':ObservationTermCfg(func=mdp.projected_gravity,noise=UniformNoiseCfg(n_min=-.05,n_max=.05)),
  'joint_pos':ObservationTermCfg(func=mdp.joint_pos_rel,params={'asset_cfg':legs,'biased':True},noise=UniformNoiseCfg(n_min=-.01,n_max=.01)),
  'joint_vel':ObservationTermCfg(func=mdp.joint_vel_rel,params={'asset_cfg':legs},scale=.05,noise=UniformNoiseCfg(n_min=-1.5,n_max=1.5)),
  'actions':ObservationTermCfg(func=canonical_last_action),
  'command':ObservationTermCfg(func=mdp.generated_commands,params={'command_name':'twist'}),
  'phase':ObservationTermCfg(func=clock)}
 cfg.observations={'actor':ObservationGroupCfg(terms=terms,concatenate_terms=True,enable_corruption=not play,history_length=history)}
 cfg.rewards={name:RewardTermCfg(func=env_reward,weight=w,params={'term':name}) for name,w in WEIGHTS.items()}
 cfg.events.pop('push_robot',None);cfg.events['base_com'].params['asset_cfg'].body_names=('base_link',)
 cfg.events['foot_friction'].params['asset_cfg'].geom_names=WHEEL_GEOM_NAMES;cfg.events['encoder_bias'].params['asset_cfg']=legs
 cfg.events['gait_phase']=EventTermCfg(func=reset_clock,mode='reset')
 cfg.terminations.pop('out_of_terrain_bounds',None)
 command=cfg.commands['twist'];command.heading_command=False;command.ranges.heading=None;command.rel_heading_envs=0.
 command.rel_standing_envs=.15;command.ranges.lin_vel_x=(-.2,.35);command.ranges.lin_vel_y=(-.15,.15);command.ranges.ang_vel_z=(-.5,.5)
 cfg.curriculum={}
 if play:
  for name in ('base_com','encoder_bias','foot_friction'):cfg.events.pop(name,None)
 return cfg
