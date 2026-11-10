"""轮足RP_wd物理适配、近期晋级窗口和上游算法一致性回归。"""
from types import SimpleNamespace as NS
import importlib
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import torch

from rpwd_go2w_supermimic.robot.config import (
    get_go2w_robot_cfg, RPWD_URDF, RP_LEGS, LEGS, rp_kinematic_constants,
)
from rpwd_go2w_supermimic.core.rolling_rewards import wheel_geometry
from rpwd_go2w_supermimic.core.stair_curriculum import (
    append_recent, buffers, next_stage, recent_counts, set_probe_stage,
    snapshot_recent, restore_recent,
)

@pytest.fixture(scope='module')
def model():
    return get_go2w_robot_cfg('rpwd').build().compile()


def test_urdf_mass_joint_limits_and_drive(model):
    root=ET.parse(RPWD_URDF).getroot()
    expected=sum(float(x.get('value')) for x in root.findall('link/inertial/mass'))
    assert abs(sum(model.body_mass)-expected)<1e-7
    assert (model.nq,model.nv,model.nu)==(23,22,16)
    assert int(model.jnt_type[0])==int(mujoco.mjtJoint.mjJNT_FREE)
    for i in range(model.nu):
        joint=model.joint(int(model.actuator_trnid[i,0])).name
        if 'wheel' in joint:
            np.testing.assert_allclose(model.actuator_gainprm[i,0],3.)
            np.testing.assert_allclose(model.actuator_biasprm[i,:3],[0,0,-3])
            np.testing.assert_allclose(model.actuator_forcerange[i],[-17,17])
        else:
            np.testing.assert_allclose(model.actuator_gainprm[i,0],100.)
            np.testing.assert_allclose(model.actuator_biasprm[i,:3],[0,-100,-3])
            np.testing.assert_allclose(model.actuator_forcerange[i],[-42,42])
    coll=[g for g in range(model.ngeom) if model.geom_contype[g]]
    assert all(model.geom_contype[g]==2 and model.geom_conaffinity[g]==1 for g in coll)


def test_differentiable_fk_matches_native_mujoco(model):
    data=mujoco.MjData(model)
    geometry=rp_kinematic_constants('cpu')
    rng=np.random.default_rng(4)
    default=torch.tensor([0.,.72,-1.41]*4)
    qadr=[int(model.joint(f'{leg}_{kind}_joint').qposadr[0]) for leg in LEGS for kind in ('hip','thigh','calf')]
    for _ in range(20):
        mujoco.mj_resetDataKeyframe(model,data,0)
        q=default.numpy()+rng.uniform(-.15,.15,12)
        data.qpos[qadr]=q
        quat=np.zeros(4);mujoco.mju_axisAngle2Quat(quat,np.array([1.,0.,0.]),float(rng.uniform(-.2,.2)))
        data.qpos[3:7]=quat;mujoco.mj_forward(model,data)
        R=data.xmat[model.body('base_link').id].reshape(3,3)
        state=torch.zeros(1,38);state[0,1:4]=torch.tensor(R.T@np.array([0.,0.,-1.]));state[0,10:22]=torch.tensor(q)-default
        point,radius=wheel_geometry(state,default,geometry)
        for i,leg in enumerate(LEGS):
            gid=model.geom(f'{leg}_wheel_collision').id
            expected=R.T@(data.geom_xpos[gid]-data.xpos[model.body('base_link').id])
            np.testing.assert_allclose(point[0,i].numpy(),expected,atol=2e-6)
            axis=data.geom_xmat[gid].reshape(3,3)[:,2]
            expected_r=.1025*np.sqrt(max(0.,1-axis[2]**2))+.02*abs(axis[2])
            np.testing.assert_allclose(radius[0,i],expected_r,atol=2e-6)
    state.requires_grad_();wheel_geometry(state,default,geometry)[0].square().sum().backward()
    assert torch.isfinite(state.grad).all() and state.grad[0,10:22].abs().sum()>0


def test_recent_window_ignores_ancient_failures():
    h=torch.zeros(4,256,dtype=torch.uint8);n=torch.zeros(4,dtype=torch.long)
    append_recent(h,n,1,torch.zeros(10000));append_recent(h,n,1,torch.ones(256))
    current=torch.stack((n.clamp_max(256),h.sum(-1)),-1).numpy()
    assert tuple(current[1])==(256,256)
    assert next_stage(1,50000,current,[.1,.07,.15])==2
    assert next_stage(1,7000,current,[.1,.07,.15])==1
    assert next_stage(1,50000,current,[.19,.07,.15])==1


@pytest.mark.parametrize('samples,passed,promote',[(127,127,False),(128,77,True),(256,153,False),(256,154,True)])
def test_gate_preserves_minimum_and_pass_threshold(samples,passed,promote):
    recent=np.zeros((4,2));recent[1]=[samples,passed]
    assert next_stage(1,10000,recent,[.1,.07,.15])==1+int(promote)


def fake_env():
    types=torch.tensor([0]*5+[1]*20+[2]*5)
    origins=torch.arange(6*6*3,dtype=torch.float).reshape(6,6,3)
    terrain=NS(terrain_types=types,terrain_levels=torch.zeros(30,dtype=torch.long),
               terrain_origins=origins,env_origins=torch.zeros(30,3))
    return NS(cfg=NS(promotion_window=256),device='cpu',num_envs=30,scene=NS(terrain=terrain))


def test_probe_pool_and_restore():
    env=fake_env();buffers(env);set_probe_stage(env,1,.2)
    assert env._stair_probe_mask.sum()==4
    assert (env.scene.terrain.terrain_levels[env._stair_probe_mask]==2).all()
    assert (env.scene.terrain.terrain_levels[~env._stair_probe_mask]==0).all()
    append_recent(env._stair_recent,env._stair_recent_count,1,torch.tensor([0,1,1]))
    saved={'recent':snapshot_recent(env)};other=fake_env();restore_recent(other,saved)
    torch.testing.assert_close(recent_counts(env),recent_counts(other))
    legacy=fake_env();restore_recent(legacy,{'stats':torch.ones(6,6,4)})
    assert recent_counts(legacy).sum()==0


def test_go2w_world_model_and_reward_match_upstream():
    source=Path(os.environ.get('GO2W_UPSTREAM_SOURCE','/home/rmer/Dog_deploy-main/Dog_deploy-main/go2w-supermimic'))
    if not source.exists():pytest.skip('上游只读对照源码不在本机')
    sys.path.insert(0,str(source))
    old=importlib.import_module('go2w_supermimic.core.terrain_world_model')
    new=importlib.import_module('rpwd_go2w_supermimic.core.terrain_world_model')
    torch.manual_seed(8);a=old.TerrainWorldModel((32,32),contacts=True,collisions=True)
    torch.manual_seed(8);b=new.TerrainWorldModel((32,32),contacts=True,collisions=True)
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    state=torch.randn(3,38)*.1;state[:,1:4]=torch.tensor([0.,0.,-1.]);state[:,0]=.4
    inputs=(state,torch.randn(3,16)*.1,torch.zeros(3,16),torch.randn(3,280)*.1)
    for x,y in zip(a.forward_all(*inputs),b.forward_all(*inputs)):torch.testing.assert_close(x,y,rtol=0,atol=0)
    oa=importlib.import_module('go2w_supermimic.core.imagine');ob=importlib.import_module('rpwd_go2w_supermimic.core.imagine')
    constants=dict(kp=torch.tensor([20.,20.,40.]*4),kd=torch.tensor([1.,1.,2.]*4),effort=torch.tensor([23.5,23.5,45.]*4),
        default_q=torch.tensor([0.,.9,-1.8]*4),limits=torch.tensor([[-3.,3.]]*12),leg_action_scale=torch.tensor([.25,.6,.8]*4),
        wheel_action_scale=35.,terrain_posture_v2=True,terrain_rewards=True,rolling_rewards=True,obstacle_progress=True)
    args=(state,inputs[1],inputs[2],torch.ones(3,3)*.1,constants,torch.ones(3))
    scan=torch.rand(3,273)*.05+.4
    x=oa.reward_terms(*args,scan=scan);y=ob.reward_terms(*args,scan=scan)
    assert x.keys()==y.keys()
    for k in x:torch.testing.assert_close(x[k],y[k],rtol=0,atol=0)


def test_checkpoint_robot_and_course_guards(tmp_path):
    from rpwd_go2w_supermimic.core.train import Learner,parser
    from rpwd_go2w_supermimic.core.policy import load_actor
    def args(robot):
        return parser().parse_args(['--robot',robot,'--terrain','--terrain-revision','7','--command-curriculum',
                                   '--device','cpu','--wm-hidden','32','32','--actor-hidden','32','--critic-hidden','32'])
    a=args('rpwd');x=Learner(a,field=object());path=tmp_path/'model.pt';x.save(path,7,list(range(16)),.02)
    y=Learner(args('rpwd'),field=object());assert y.load(path,list(range(16)),.02)==7
    with pytest.raises(ValueError,match='机器人类型'):
        Learner(args('go2w'),field=object()).load(path,list(range(16)),.02)
    wrong=args('rpwd');wrong.promotion_window=128
    with pytest.raises(ValueError,match='课程参数'):
        Learner(wrong,field=object()).load(path,list(range(16)),.02)
    c=torch.load(path,map_location='cpu',weights_only=True);c['robot_signature']='invalid'
    with pytest.raises(ValueError,match='指纹'):
        load_actor(c)


def test_export_permutation_is_per_frame():
    from export_rpwd_go2w import DeploymentActor
    class Sentinel(torch.nn.Module):
        history=5
        def forward(self,x):return x[:,214:230]
    order=[3,2,1,0]+list(range(4,16))
    model=DeploymentActor(Sentinel(),order)
    x=torch.arange(245,dtype=torch.float32)[None]
    torch.testing.assert_close(model(x),x[:,214:230][:,order])


def test_rpwd_imagination_compiles_and_reaches_actor_gradient():
    from rpwd_go2w_supermimic.core.terrain_world_model import TerrainWorldModel
    from rpwd_go2w_supermimic.core.policy import Actor
    from rpwd_go2w_supermimic.core.imagine import make_imagined_step
    class Field:
        def scan(self,state,pose):return state[:,:1].expand(-1,273)
    model=TerrainWorldModel((32,32),contacts=True,collisions=True);actor=Actor(hidden=(32,32))
    constants=dict(kp=torch.full((12,),100.),kd=torch.full((12,),3.),effort=torch.full((12,),42.),
        wheel_kd=3.,wheel_effort=17.,default_q=torch.tensor([0.,.72,-1.41]*4),
        limits=torch.tensor([[-3.,3.]]*12),leg_action_scale=torch.tensor([.25,.6,.8]*4),
        wheel_action_scale=35.,terrain_posture_v2=True,terrain_rewards=True,rolling_rewards=True,
        obstacle_progress=True,nonwheel_collision=True,kinematics=rp_kinematic_constants('cpu'))
    weights=dict(track_linear_velocity=1.5,track_angular_velocity=1.,upright=1.,leg_posture=-.4,
                 standstill=-2.,action_rate_l2=-.05,joint_torques_l2=-2e-5,dof_pos_limits=-1.,
                 terminated=-20.,unnecessary_wheel_lift=-.2,rolling_leg_motion=-.12,obstacle_progress=.5)
    step=make_imagined_step(model,actor,constants,weights,list(range(16)),noise=False,field=Field())
    compiled=torch.compile(step,backend='eager',fullgraph=True)
    state=torch.zeros(2,38);state[:,0]=.42;state[:,3]=-1.
    out=compiled(state,torch.zeros(2,16),torch.zeros(2,245),torch.zeros(2,3),
                 torch.full((2,273),.42),torch.ones(2,3)*.1,torch.zeros(2,7),torch.zeros(2,12))
    (-out[5].mean()).backward()
    grads=[p.grad for p in actor.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads)>0


def test_old_phase_episodes_do_not_promote_new_phase():
    from rpwd_go2w_supermimic.core.stair_curriculum import record_recent
    env=fake_env();buffers(env);env._stair_stage=1
    levels=torch.full((256,),2);types=torch.ones(256,dtype=torch.long)
    valid=torch.ones(256,dtype=torch.bool);passed=torch.ones(256)
    record_recent(env,levels,types,valid,passed,torch.zeros(256,dtype=torch.long))
    assert recent_counts(env).sum()==0
    record_recent(env,levels,types,valid,passed,torch.ones(256,dtype=torch.long))
    assert tuple(recent_counts(env)[1].tolist())==(256,256)


def test_each_link_inertia_and_joint_axis_match_urdf(model):
    from rpwd_go2w_supermimic.robot.config import RP_TO_CANONICAL
    root=ET.parse(RPWD_URDF).getroot()
    for link in root.findall('link'):
        old=link.get('name');name=old
        for rp,can in zip(RP_LEGS,LEGS):
            if old==f'{rp}_FOOT_LINK':name=f'{can}_wheel_link'
        b=model.body(name).id;i=link.find('inertial')
        np.testing.assert_allclose(model.body_mass[b],float(i.find('mass').get('value')),atol=1e-8)
        np.testing.assert_allclose(model.body_ipos[b],[float(x) for x in i.find('origin').get('xyz').split()],atol=1e-8)
        v=i.find('inertia');expected=np.array([[float(v.get(k)) for k in row] for row in
                            (('ixx','ixy','ixz'),('ixy','iyy','iyz'),('ixz','iyz','izz'))])
        rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,model.body_iquat[b]);R=rotation.reshape(3,3)
        np.testing.assert_allclose(R@np.diag(model.body_inertia[b])@R.T,expected,atol=1e-7)
    for joint in root.findall('joint'):
        j=model.joint(RP_TO_CANONICAL[joint.get('name')])
        np.testing.assert_allclose(j.axis,[float(x) for x in joint.find('axis').get('xyz').split()],atol=1e-8)
        if joint.get('type')=='revolute':
            limit=joint.find('limit');np.testing.assert_allclose(j.range,[float(limit.get('lower')),float(limit.get('upper'))],atol=1e-8)


def test_two_robots_use_same_task_and_randomization():
    from rpwd_go2w_supermimic.env.terrain import go2w_velocity_terrain_env_cfg
    a=go2w_velocity_terrain_env_cfg(revision=7,robot='go2w')
    b=go2w_velocity_terrain_env_cfg(revision=7,robot='rpwd')
    assert a.decimation==b.decimation==4
    assert vars(a.sim.mujoco)==vars(b.sim.mujoco)
    assert a.actions.keys()==b.actions.keys()
    for k in a.actions:assert a.actions[k].scale==b.actions[k].scale
    assert a.rewards.keys()==b.rewards.keys()
    for k in a.rewards:
        assert a.rewards[k].func is b.rewards[k].func
        assert a.rewards[k].weight==b.rewards[k].weight
    assert a.events.keys()==b.events.keys()
    for k in a.events:
        assert a.events[k].func is b.events[k].func
        assert a.events[k].params==b.events[k].params
    assert a.observations['actor'].history_length==b.observations['actor'].history_length==5


def test_export_yaml_joint_mapping_matches_hardware():
    from export_rpwd_go2w import deployment_config
    from rpwd_go2w_supermimic.robot.config import hardware_contract,RP_KINDS
    c=hardware_contract();p=deployment_config({'history':5,'curriculum_state':{'stage':1}})['inference_node']['ros__parameters']
    expected=[f'{leg}_{kind}_JOINT' for leg in RP_LEGS for kind in RP_KINDS]+[f'{leg}_FOOT_JOINT' for leg in RP_LEGS]
    assert [c.URDF_JOINT_ORDER[i] for i in p['usd2urdf']]==expected
    assert sum(int(x.split(':')[1]) for x in p['obs_layouts'][0].split(','))*p['frame_stacks'][0]==245
    assert p['action_scales']==[.25,.6,.8]*4+[35.]*4
    assert p['obs_scales_dof_vel']==.05 and p['clip_actions']==1.
