"""按名称映射12个点足执行器；接触标签使用世界向上承载力。"""
import torch
from ..robot.config import LEG_JOINT_NAMES,WHEEL_GEOM_NAMES,get_spec,rp_kinematic_constants
from .kinematics import clock
class Go2WBridge:
 def __init__(self,env):
  self.env=env;self.robot=env.scene['robot'];self.robot_id='rpwd_pointfoot';self.geometry=rp_kinematic_constants(env.device)
  self.legs,_=self.robot.find_joints(LEG_JOINT_NAMES,preserve_order=True);self.wheels=[]
  if len(self.legs)!=12:raise ValueError('点足必须恰好12个驱动关节')
  names=list(env.action_manager.get_term('joint_pos').target_names)
  self.to_canonical=[names.index(n) for n in LEG_JOINT_NAMES];self.to_env=[LEG_JOINT_NAMES.index(n) for n in names]
  geoms,_=self.robot.find_geoms(WHEEL_GEOM_NAMES,preserve_order=True);self.geoms=self.robot.indexing.geom_ids[geoms]
  bodies,_=self.robot.find_bodies('base_link');self.base=self.robot.indexing.body_ids[bodies[0]]
  self.nominal_com=torch.as_tensor(get_spec().body('base_link').ipos.copy(),device=env.device,dtype=torch.float32)
 def canonical_action(self,action):return action[:,self.to_canonical]
 def env_action(self,action):return action[:,self.to_env]
 def state(self):
  d=self.robot.data
  return torch.cat((d.root_link_pos_w[:,2:3],d.projected_gravity_b,d.root_link_lin_vel_b,d.root_link_ang_vel_b,(d.joint_pos-d.default_joint_pos)[:,self.legs],d.joint_vel[:,self.legs],clock(self.env)),-1)
 def pose(self):
  d=self.robot.data;return torch.cat((d.root_link_pos_w[:,:2],d.heading_w[:,None]),-1)
 def context(self):
  m=self.env.sim.model;f=m.geom_friction;c=m.body_ipos
  mu=f[:,self.geoms,0] if f.ndim==3 else f[self.geoms,0].expand(self.env.num_envs,-1)
  com=c[:,self.base] if c.ndim==3 else c[self.base].expand(self.env.num_envs,-1)
  return torch.cat((mu,com-self.nominal_com),-1)
 def contacts(self):
  from mjlab.utils.lab_api.math import quat_apply_inverse
  sensor=self.env.scene['wheel_ground_contact'];names=[n.split('/')[-1] for n in sensor.primary_names]
  order=[names.index(f'{leg}_foot_link') for leg in ('FL','FR','RL','RR')]
  # netforce记录主匹配体施加给地面的力，取反才是地面对脚的承载力。
  force=-sensor.data.force[:,order].reshape(-1,4,3);found=(force[:,:,2]>8.).float()
  q=self.robot.data.root_link_quat_w[:,None,:].expand(-1,4,-1)
  return torch.cat((found,quat_apply_inverse(q.reshape(-1,4),force.reshape(-1,3)).reshape(-1,12)),-1)
def pose_delta(before,after):
 xy=after[:,:2]-before[:,:2];c,s=before[:,2].cos(),before[:,2].sin();y=after[:,2]-before[:,2]
 return torch.stack((c*xy[:,0]+s*xy[:,1],-s*xy[:,0]+c*xy[:,1],torch.atan2(y.sin(),y.cos())),-1)
