"""URDF球脚运动学与确定性步态时钟，不依赖Isaac运行时。"""
import math
import torch
from ..robot.config import rp_kinematic_constants
FREQUENCY=1.5

def phase_features(phase):
 a=2*math.pi*phase.reshape(-1,1);return torch.cat((a.sin(),a.cos()),-1)

def advance_clock(state,dt=.02):
 a=2*math.pi*FREQUENCY*dt;s,c=state[:,-2:].unbind(-1)
 return torch.cat((state[:,:34],torch.stack((s*math.cos(a)+c*math.sin(a),c*math.cos(a)-s*math.sin(a)),-1)),-1)

def clock(env):
 offset=getattr(env,'_foot_phase_offset',torch.zeros(env.num_envs,device=env.device))
 return phase_features(offset+env.episode_length_buf*env.step_dt*FREQUENCY)

def reset_clock(env,env_ids):
 if not hasattr(env,'_foot_phase_offset'):env._foot_phase_offset=torch.zeros(env.num_envs,device=env.device)
 env._foot_phase_offset[env_ids]=torch.rand(len(env_ids),device=env.device)

def feet_body(state,geometry=None):
 g=geometry if geometry is not None else rp_kinematic_constants(state.device)
 o=g['origins'];q=(state[:,10:22]+g['default_q']).reshape(-1,4,3)
 def ry(a,p):
  x,y,z=p.unbind(-1);return torch.stack((a.cos()*x+a.sin()*z,y.expand_as(a),-a.sin()*x+a.cos()*z),-1)
 p=ry(q[:,:,2],o[:,3]+g['centers'])+o[:,2];p=ry(q[:,:,1],p)+o[:,1]
 x,y,z=p.unbind(-1);a=q[:,:,0]
 return torch.stack((x,a.cos()*y-a.sin()*z,a.sin()*y+a.cos()*z),-1)+o[:,0]

def rotate_world(vector,state,pose):
 up=-state[:,1:4]/(state[:,1:4].square().sum(-1,keepdim=True)+1e-8).sqrt()
 pitch=torch.asin((-up[:,0]).clamp(-.99999,.99999));roll=safe_atan2(up[:,1],up[:,2])
 r,p,y=roll[:,None],pitch[:,None],pose[:,2:3];x0,y0,z0=vector.unbind(-1)
 y1=r.cos()*y0-r.sin()*z0;z1=r.sin()*y0+r.cos()*z0
 x2=p.cos()*x0+p.sin()*z1;z2=-p.sin()*x0+p.cos()*z1
 return torch.stack((y.cos()*x2-y.sin()*y1,y.sin()*x2+y.cos()*y1,z2),-1)

def foot_state(state,pose,field,geometry):
 body=feet_body(state,geometry);delta=torch.zeros_like(state);delta[:,10:22]=state[:,22:34]*.001
 velocity=(feet_body(state+delta,geometry)-body)/.001
 velocity=velocity+state[:,None,4:7]+torch.cross(state[:,None,7:10].expand_as(body),body,dim=-1)
 rotated=rotate_world(body,state,pose);ground=field.sample(pose[:,:2])
 world=torch.cat((pose[:,None,:2]+rotated[:,:,:2],(state[:,:1]+ground[:,None]+rotated[:,:,2]).unsqueeze(-1)),-1)
 return body,world,rotate_world(velocity,state,pose)


def safe_atan2(y,x):
 """对零分母显式保护，保持四象限；避免融合反传中的0/0。"""
 xx=torch.where(x>=0,x.clamp_min(1e-6),x.clamp_max(-1e-6))
 a=torch.atan(y/xx)
 return torch.where(x<0,a+torch.where(y>=0,torch.full_like(a,math.pi),torch.full_like(a,-math.pi)),a)
