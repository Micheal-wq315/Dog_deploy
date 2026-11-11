"""点足迁移的物理、控制、可微想象和课程回归。"""
import math
from pathlib import Path
import xml.etree.ElementTree as E
import mujoco
import numpy as np
import pytest
import torch
from rpwd_pointfoot_supermimic.robot.config import get_spec,rp_kinematic_constants,DEFAULT_Q,URDF,LEG_JOINT_NAMES
from rpwd_pointfoot_supermimic.core.kinematics import feet_body,phase_features,advance_clock
from rpwd_pointfoot_supermimic.core.policy import Actor,frame,push_history,TERMS,load_actor
from rpwd_pointfoot_supermimic.core.terrain import TerrainField
from rpwd_pointfoot_supermimic.core.task import reward_terms,WEIGHTS
from rpwd_pointfoot_supermimic.core.terrain_world_model import TerrainWorldModel
from rpwd_pointfoot_supermimic.core.imagine import make_imagined_step
from rpwd_pointfoot_supermimic.core.stair_curriculum import append_recent,next_stage

def state(n=4):
 s=torch.zeros(n,36);s[:,0]=.3403;s[:,3]=-1;s[:,35]=1;return s

def history(a,s,cmd=None):
 f=frame(s,torch.zeros(len(s),12),torch.tensor([[.2,0,0]]*len(s)) if cmd is None else cmd,torch.zeros(len(s),12))
 return torch.cat([t.repeat(1,a.history) for t in f.split(TERMS,-1)],-1)

def constants(a):
 g=rp_kinematic_constants();return dict(nonwheel_collision=True,geometry=g,default_q=g['default_q'],limits=g['limits'],reference=a.reference)

def test_asset_mass_axes_and_fixed_feet():
 m=get_spec().compile();r=E.parse(URDF).getroot()
 assert m.nq==19 and m.nv==18
 assert (m.jnt_type==mujoco.mjtJoint.mjJNT_HINGE).sum()==12
 assert m.body_mass.sum()==pytest.approx(sum(float(x.find('inertial/mass').get('value')) for x in r.findall('link')),abs=1e-8)
 for name in LEG_JOINT_NAMES:assert m.joint(name).id>=0
 for leg in ('FL','FR','RL','RR'):
  b=m.body(leg+'_foot_link').id
  assert m.body_jntnum[b]==0
  g=m.geom(leg+'_foot_collision').id
  assert m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE
  assert m.geom_size[g,0]==pytest.approx(.022)

def test_fk_matches_mujoco():
 m=get_spec().compile();d=mujoco.MjData(m);g=rp_kinematic_constants()
 rng=np.random.default_rng(3)
 for _ in range(12):
  s=state(1);s[:,10:22]=torch.tensor(rng.uniform(-.15,.15,(1,12)),dtype=torch.float32)
  d.qpos[:3]=(0,0,.34);d.qpos[3:7]=(1,0,0,0)
  for i,n in enumerate(LEG_JOINT_NAMES):d.qpos[m.jnt_qposadr[m.joint(n).id]]=DEFAULT_Q[i]+float(s[0,10+i])
  mujoco.mj_forward(m,d)
  actual=np.stack([d.geom_xpos[m.geom(leg+'_foot_collision').id]-d.xpos[m.body('base_link').id] for leg in ('FL','FR','RL','RR')])
  np.testing.assert_allclose(feet_body(s,g)[0].numpy(),actual,atol=1e-6)

def test_ik_and_clock():
 a=Actor();s=state(7);phase=torch.linspace(0,.9,7);cmd=torch.tensor([[.2,0,0]]*7)
 target,*_=a.reference.trajectory(phase,cmd);s[:,10:22]=a.reference.inverse(target)-a.reference.default_q
 torch.testing.assert_close(feet_body(s),target,atol=1e-5,rtol=1e-5)
 s=state();s[:,34:36]=phase_features(torch.tensor([0.,.2,.4,.9]))
 torch.testing.assert_close(advance_clock(s)[:,34:36],phase_features(torch.tensor([0.,.2,.4,.9])+.03),atol=2e-6,rtol=2e-6)

def test_shapes_and_rate_limit():
 a=Actor();s=state();h=history(a,s);u=a(h,torch.randn(4,12)*10)
 assert h.shape==(4,235) and u.shape==(4,12)
 assert (u.abs()<=a.rate+1e-6).all() and (u.abs()<=1).all()
 latest=frame(s,torch.zeros(4,12),torch.zeros(4,3),torch.zeros(4,12))
 assert push_history(h,latest,5).shape==h.shape

def test_stationary_reference_and_stride():
 a=Actor();phase=torch.linspace(0,1,101)
 target,stance,lift,active=a.reference.trajectory(phase,torch.zeros(101,3))
 assert lift.abs().max()==0 and active.abs().max()==0
 target,stance,lift,active=a.reference.trajectory(phase,torch.tensor([[.2,0,0]]*101))
 assert target[:,0,0].max()-target[:,0,0].min()>.075
 assert lift.max()>.99

def test_shared_reward_finite_gradients():
 a=Actor();s=state().requires_grad_();f=TerrainField(torch.zeros(80,80),(-1.6,-1.6))
 t=reward_terms(s,torch.zeros(4,12),torch.zeros(4,12),torch.tensor([[.2,0,0]]*4),constants(a),torch.ones(4),pose=torch.zeros(4,3),field=f)
 assert set(t)==set(WEIGHTS)
 assert all(v.shape==(4,) and torch.isfinite(v).all() for v in t.values())
 sum(v.mean() for v in t.values()).backward();assert torch.isfinite(s.grad).all()

def test_exact_clock_world_model():
 m=TerrainWorldModel((32,32),contacts=True,collisions=True);s=state();u=torch.zeros(4,12);c=torch.zeros(4,280)
 nxt,pose,contact,collision=m.forward_all(s,u,u,c)
 assert nxt.shape==(4,36) and pose.shape==(4,3) and contact.shape==(4,16)
 torch.testing.assert_close(nxt[:,34:36],advance_clock(s)[:,34:36])

def test_compiled_imagination_reaches_actor():
 torch.manual_seed(4);a=Actor(hidden=(32,32));m=TerrainWorldModel((32,32),contacts=True,collisions=True)
 f=TerrainField(torch.zeros(80,80),(-1.6,-1.6));s=state();p=torch.zeros(4,3);cmd=torch.tensor([[.2,0,0]]*4)
 step=make_imagined_step(m,a,constants(a),WEIGHTS,list(range(12)),field=f,noise=False)
 compiled=torch.compile(step,backend='eager',fullgraph=True)
 result=compiled(s,torch.zeros(4,12),history(a,s),p,f.scan(s,p),cmd,torch.zeros(4,7),torch.zeros(4,12))
 assert torch.isfinite(result[-2]).all();(-result[-2].mean()).backward()
 assert a.net[-1].weight.grad is not None and torch.isfinite(a.net[-1].weight.grad).all()
 assert a.net[-1].weight.grad.abs().sum()>0

def test_recent_gate_and_robot_rejection():
 h=torch.zeros(4,256,dtype=torch.uint8);c=torch.zeros(4,dtype=torch.long)
 append_recent(h,c,0,torch.zeros(1000));append_recent(h,c,0,torch.ones(256))
 assert h[0].sum()==256
 assert next_stage(0,2999,[[256,256]]*4,[.1,.1,.1])==0
 assert next_stage(0,3000,[[256,256]]*4,[.1,.1,.1])==1
 with pytest.raises(ValueError):load_actor({'format':'rpwd-go2w-terrain-v1','robot_id':'rpwd'})

def test_native_actuator_gains_and_limits():
 from mjlab.scene import Scene
 from rpwd_pointfoot_supermimic.env.terrain import go2w_velocity_terrain_env_cfg
 cfg=go2w_velocity_terrain_env_cfg(play=True);cfg.scene.num_envs=1;cfg.scene.terrain=None;cfg.scene.sensors=()
 scene=Scene(cfg.scene,device='cpu');m=scene.compile();cfg.sim.mujoco.apply(m)
 assert m.nu==12 and m.nv==18
 np.testing.assert_allclose(m.actuator_gainprm[:,0],100.)
 np.testing.assert_allclose(m.actuator_biasprm[:,1],-100.)
 np.testing.assert_allclose(m.actuator_biasprm[:,2],-3.)
 np.testing.assert_allclose(m.actuator_forcerange,np.tile([-42.,42.],(12,1)))
 assert m.opt.timestep==pytest.approx(.005)

def test_contact_force_direction_and_sidewall():
 from types import SimpleNamespace as S
 from rpwd_pointfoot_supermimic.core.bridge import Go2WBridge
 b=Go2WBridge.__new__(Go2WBridge)
 force=torch.tensor([[[0.,0.,-42.]]*4,[[90.,0.,0.]]*4])
 sensor=S(primary_names=[f'{x}_foot_link' for x in ('FL','FR','RL','RR')],data=S(force=force))
 b.env=S(scene={'wheel_ground_contact':sensor});b.robot=S(data=S(root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]*2)))
 result=b.contacts()
 torch.testing.assert_close(result[0,:4],torch.ones(4));torch.testing.assert_close(result[1,:4],torch.zeros(4))
 torch.testing.assert_close(result[0,4:].reshape(4,3)[:,2],torch.full((4,),42.))

def test_link_inertia_preserved():
 m=get_spec().compile();root=E.parse(URDF).getroot()
 mapping=dict(zip(('LF','RF','LB','RB'),('FL','FR','RL','RR')))
 for link in root.findall('link'):
  name=link.get('name')
  if name.endswith('FOOT_LINK'):name=mapping[name[:2]]+'_foot_link'
  i=m.body(name).id;n=link.find('inertial/inertia');R=np.zeros(9);mujoco.mju_quat2Mat(R,m.body_iquat[i]);R=R.reshape(3,3)
  actual=R@np.diag(m.body_inertia[i])@R.T
  expected=np.array([[float(n.get('ixx')),float(n.get('ixy')),float(n.get('ixz'))],[float(n.get('ixy')),float(n.get('iyy')),float(n.get('iyz'))],[float(n.get('ixz')),float(n.get('iyz')),float(n.get('izz'))]])
  np.testing.assert_allclose(actual,expected,atol=1e-8)

def test_full_body_rotation_matches_mujoco():
 from rpwd_pointfoot_supermimic.core.kinematics import rotate_world
 m=get_spec().compile();d=mujoco.MjData(m);rng=np.random.default_rng(11)
 for _ in range(8):
  quat=rng.normal(size=4);quat/=np.linalg.norm(quat);d.qpos[:3]=(0,0,.4);d.qpos[3:7]=quat
  for i,n in enumerate(LEG_JOINT_NAMES):d.qpos[m.jnt_qposadr[m.joint(n).id]]=DEFAULT_Q[i]
  mujoco.mj_forward(m,d);R=d.xmat[m.body('base_link').id].reshape(3,3)
  s=state(1);s[:,1:4]=torch.tensor(R.T@np.array([0,0,-1]),dtype=torch.float32)
  pose=torch.tensor([[0.,0.,math.atan2(R[1,0],R[0,0])]])
  predicted=rotate_world(feet_body(s),s,pose)[0].numpy()
  actual=np.stack([d.geom_xpos[m.geom(leg+'_foot_collision').id]-d.xpos[m.body('base_link').id] for leg in ('FL','FR','RL','RR')])
  np.testing.assert_allclose(predicted,actual,atol=2e-5)

def test_numerical_boundaries_and_height_band():
 from rpwd_pointfoot_supermimic.core.kinematics import safe_atan2
 y=torch.tensor([0.,1.,0.,-1.],requires_grad=True);x=torch.tensor([1.,0.,-1.,0.],requires_grad=True)
 q=safe_atan2(y,x);assert torch.isfinite(q).all();q.sum().backward();assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
 a=Actor();s=state(4).requires_grad_();cmd=torch.tensor([[0.,0.,0.],[0.,0.,.3],[0.,.15,0.],[.2,0.,0.]])
 f=TerrainField(torch.zeros(80,80),(-1.6,-1.6))
 t=reward_terms(s,torch.zeros(4,12),torch.zeros(4,12),cmd,constants(a),torch.ones(4),pose=torch.zeros(4,3),field=f)
 torch.testing.assert_close(t['base_height'],torch.zeros(4));sum(v.sum() for v in t.values()).backward();assert torch.isfinite(s.grad).all()
