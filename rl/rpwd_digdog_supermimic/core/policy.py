"""235维盲策略：URDF完整步态参考、有界残差及每步关节目标限速。"""
import math
from .kinematics import safe_atan2
from types import SimpleNamespace
import torch
from torch import nn
from .world_model import RunningNorm
from .kinematics import feet_body
from .gait import PointfootGaitReference
from .terrain import SCAN_DIM
from ..robot.config import rp_kinematic_constants,robot_signature,profile
TERMS=(3,3,12,12,12,3,2)
FRAME_DIM=sum(TERMS)
LEGACY_FRAME_DIM=FRAME_DIM
CRITIC_DIM=36+3+7+12
POLICY_FORMAT='rpwd-digdog-flat-v1'

def frame(state,last_env_action,command,bias,noise=False):
 values=[state[:,7:10],state[:,1:4],state[:,10:22]+bias,state[:,22:34],last_env_action,command,state[:,34:36]]
 widths=(.2,.05,.01,1.5,0.,0.,0.)
 if noise:values=[v+(torch.rand_like(v)*2-1)*w if w else v for v,w in zip(values,widths)]
 return torch.cat([v*(.05 if i==3 else 1.) for i,v in enumerate(values)],-1)

def push_history(history,latest,length):
 result=[];h=f=0
 for n in TERMS:
  result.append(torch.cat((history[:,h+n:h+n*length],latest[:,f:f+n]),-1));h+=n*length;f+=n
 return torch.cat(result,-1)

def critic_input(state,command,context,previous):return torch.cat((state,command,context,previous),-1)

class Network(nn.Module):
 def __init__(self,inputs,outputs,hidden):
  super().__init__();self.norm=RunningNorm(inputs);layers=[]
  for n in hidden:layers.extend((nn.Linear(inputs,n),nn.ELU()));inputs=n
  layers.append(nn.Linear(inputs,outputs));self.net=nn.Sequential(*layers)
 def forward(self,x):return self.net(self.norm(x))

class Actor(Network):
 def __init__(self,history=5,hidden=(256,128,64),actor_wheel_vel=False):
  if actor_wheel_vel:raise ValueError('点足没有轮速输入')
  super().__init__(FRAME_DIM*history,12,hidden)
  self.history=history;self.hidden=tuple(hidden);self.actor_wheel_vel=False;self.residual_scale=profile()["residual_scale"]
  g=rp_kinematic_constants()
  task=SimpleNamespace(default_q=g['default_q'],ids=torch.arange(12).reshape(4,3),anchors=g['origins'][:,0],offsets=g['origins'][:,1:].clone(),limits=g['limits'],signs=g['signs'],action_scale=torch.tensor(profile()['action_scale']))
  task.offsets[:,2]+=g['centers'];task.feet_body=lambda state:feet_body(state,g)
  self.reference=PointfootGaitReference(task)
  self.register_buffer('rate',torch.tensor(profile()['rate_limits'])*.02/torch.tensor(profile()['action_scale']))
  nn.init.uniform_(self.net[-1].weight,-.01,.01);nn.init.zeros_(self.net[-1].bias)
 def forward(self,obs,exploration=None):
  h=self.history;phase=safe_atan2(obs[:,-2],obs[:,-1])/(2*math.pi)
  command=obs[:,-2*h-3:-2*h];prev=obs[:,42*h-12:42*h]
  residual=self.residual_scale*super().forward(obs).tanh()
  if exploration is not None:residual=residual+exploration
  desired=(self.reference.actions(phase,command)+residual).clamp(-1,1)
  return (prev+self.rate*torch.tanh((desired-prev)/self.rate)).clamp(-1,1)

class ScanActor(Actor):
 """带 273 维局部高度图的策略。

 网络保留盲策略的 235→256→128→64 主干和原 ``net``（用于从旧 checkpoint
 无损热启动），额外加一个 273→...→64 的高度图编码器，再与主干特征拼接输出
 12 维残差。高度图编码器最后一层初始化为零、输出头的高度图半区初始化为很小
 随机数，因此从旧盲策略热启动时第一帧动作与旧策略完全一致，同时保留梯度通道；
 教师训练后再逐步学习使用高度图。教师和学生共用该结构，只是教师输入真实扫描、
 学生输入 Mid-360 扫描。
 """

 def __init__(self,history=5,hidden=(256,128,64),scan_dim=SCAN_DIM,scan_hidden=(256,128,64)):
  super().__init__(history,hidden)
  self.scan_dim=int(scan_dim);self.scan_hidden=tuple(scan_hidden)
  self.scan_norm=RunningNorm(self.scan_dim)
  layers=[];n=self.scan_dim
  for width in self.scan_hidden:
   layers.extend((nn.Linear(n,width),nn.ELU()));n=width
  layers.append(nn.Linear(n,self.scan_hidden[-1]))
  self.scan_net=nn.Sequential(*layers)
  self.head=nn.Linear(hidden[-1]+self.scan_hidden[-1],12)
  self._init_scan_head()

 def _init_scan_head(self):
  """精确复现盲策略的初始动作，同时给高度图分支留出学习通道。

  scan_net 最后一层置零使初始高度图特征为零，因此初始动作与旧盲策略完全一致；
  输出头的高度图半区用很小的随机数而不是全零，给 scan_net 末层提供非零梯度，
  否则零初始化会让整个高度图分支永远收不到梯度。
  """
  old=self.net[-1]
  with torch.no_grad():
   self.head.weight.zero_()
   self.head.weight[:,:old.in_features].copy_(old.weight)
   self.head.weight[:,old.in_features:].uniform_(-.01,.01)
   self.head.bias.copy_(old.bias)
   self.scan_net[-1].weight.zero_()
   self.scan_net[-1].bias.zero_()

 @classmethod
 def from_blind(cls,checkpoint,scan_dim=SCAN_DIM,scan_hidden=(256,128,64)):
  """从旧 235 维盲策略热启动；新高度图分支初始为零输出。"""
  actor=cls(checkpoint['history'],tuple(checkpoint['actor_hidden']),
            scan_dim=scan_dim,scan_hidden=scan_hidden)
  actor.load_state_dict(checkpoint['actor'],strict=False)
  actor._init_scan_head()
  return actor

 def forward(self,obs,exploration=None,scan=None):
  if scan is None:raise ValueError('ScanActor需要273维局部高度图')
  if scan.shape[-1]!=self.scan_dim:raise ValueError(f'高度图维度应为{self.scan_dim}，实际{scan.shape[-1]}')
  h=self.history;phase=safe_atan2(obs[:,-2],obs[:,-1])/(2*math.pi)
  command=obs[:,-2*h-3:-2*h];prev=obs[:,42*h-12:42*h]
  proprio=self.net[:-1](self.norm(obs))
  height=self.scan_net(self.scan_norm(scan))
  residual=self.residual_scale*self.head(torch.cat((proprio,height),-1)).tanh()
  if exploration is not None:residual=residual+exploration
  desired=(self.reference.actions(phase,command)+residual).clamp(-1,1)
  return (prev+self.rate*torch.tanh((desired-prev)/self.rate)).clamp(-1,1)


def checkpoint_wheel_vel(checkpoint):return False

def load_actor(checkpoint,device='cpu'):
 if checkpoint.get('format') not in (POLICY_FORMAT,'rpwd-digdog-terrain-v1') or checkpoint.get('robot_id')!='rpwd_digdog':raise ValueError('不是新点足MuJoCo策略，禁止混用轮足或旧Isaac模型')
 if checkpoint.get('robot_signature')!=robot_signature():raise ValueError('点足资产/步态契约变化')
 scan_dim=int(checkpoint.get('actor_scan_dim') or 0)
 if scan_dim:
  actor=ScanActor(checkpoint['history'],checkpoint['actor_hidden'],scan_dim=scan_dim,
                  scan_hidden=tuple(checkpoint.get('actor_scan_hidden',(256,128,64)))).to(device)
 else:
  actor=Actor(checkpoint['history'],checkpoint['actor_hidden']).to(device)
 actor.load_state_dict(checkpoint['actor']);return actor.eval()

def evaluation_env_cfg(checkpoint,level=0,kind=None):
 from ..env.terrain import go2w_velocity_terrain_env_cfg
 cfg=go2w_velocity_terrain_env_cfg(play=True,history=checkpoint['history'],seed=checkpoint['config']['seed'],revision=7,robot='rpwd_digdog')
 gen=cfg.scene.terrain.terrain_generator;gen.num_rows=1;gen.difficulty_range=(level/5,level/5)
 if kind is not None:gen.sub_terrains={kind:gen.sub_terrains[kind]}
 cfg.sim.nconmax=max(cfg.sim.nconmax or 0,512);cfg.sim.njmax=max(cfg.sim.njmax or 0,2048)
 return cfg
