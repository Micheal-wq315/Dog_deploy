"""点足真实/想象共享奖励；摆脚、支撑、落地分别约束，无轮子或电机力矩代价。"""
import math
from .kinematics import safe_atan2
import torch
from .kinematics import foot_state

WEIGHTS=dict(track_linear_velocity=2.5,track_angular_velocity=1.,upright=.3,
 standstill=-1.,action_rate_l2=-.025,dof_pos_limits=-1.,terminated=-20.,
 base_height=-.3,foot_xy=-.12,swing_clearance=-.5,contact_phase=-.15,
 foot_slip=-.1,landing_velocity=-.05,body_motion=-.02,obstacle_progress=.5,
 foothold=.5,gap=-.5)

def constants(bridge,reference=None):
 from .policy import Actor
 g=bridge.geometry
 return dict(nonwheel_collision=True,geometry=g,default_q=g['default_q'],limits=g['limits'],reference=reference or Actor().reference.to(bridge.env.device))

def reward_terms(state,action,previous,command,c,survival_prob,scan=None,pose=None,field=None,contacts=None,kind=None):
 body,feet,velocity=foot_state(state,pose,field,c['geometry'])
 ground=field.sample(feet[:,:,:2]);clearance=feet[:,:,2]-ground-.022
 phase=safe_atan2(state[:,34],state[:,35])/(2*math.pi)
 target,stance,lift,activation=c['reference'].trajectory(phase,command)
 moving=activation[:,None];stance=torch.where(moving>.01,stance,torch.ones_like(stance))
 support=contacts[:,:4] if contacts is not None else torch.sigmoid((.015-clearance)/.008)
 # 只在梅花桩(6)/轮胎(7)启用落脚项，避免平地、台阶上出现常驻 gap 惩罚。
 active=torch.zeros_like(ground[:,0]) if kind is None else ((kind.reshape(-1)==6)|(kind.reshape(-1)==7)).float()
 # 桩顶/胎面算高地面(.12m以上)；再用窄支撑判定排除 1.4m 起点平台：
 # 脚周围 8 个方向 0.14m 内存在 >5cm 落差才算“踩在桩顶/胎面上”。
 on_top=(ground>.12).float()
 offsets=ground.new_tensor((( .14,0.),(-.14,0.),(0.,.14),(0.,-.14),
                            (.10,.10),(-.10,.10),(.10,-.10),(-.10,-.10)))
 ring=field.sample(feet[:,:,:2].unsqueeze(2)+offsets)
 narrow=((ground[:,:,None]-ring).amax(-1)>.05).float()
 foothold=((on_top*narrow*support).sum(-1)/support.sum(-1).clamp_min(1))*active
 gap=((1-on_top)*support).mean(-1)*active
 # 前方多点查障碍，只用于训练奖励；Actor不读取地形。
 speed=(command[:,:2].square().sum(-1)+1e-8).sqrt().clamp_min(.05)
 direction=command[:,:2]/speed[:,None]
 yaw=pose[:,2:3];dx=direction[:,0:1]*yaw.cos()-direction[:,1:2]*yaw.sin();dy=direction[:,0:1]*yaw.sin()+direction[:,1:2]*yaw.cos()
 ahead=feet[:,:,:2,None]+torch.stack((dx,dy),1)[:,None,:,:]*state.new_tensor((.08,.16,.24))[None,None,None,:]
 look=field.sample(ahead.permute(0,1,3,2))
 climb=(look.amax(-1)-ground).clamp(0,.12)
 required=(.065+climb)*lift
 swing=(~stance).float()*moving
 # 整个摆动轨迹必须抬起并回落；不能靠预测contact=0关闭净空损失。
 clear_error=((clearance-required)/.06).square().clamp_max(25)
 xy_error=(body[:,:,:2]-target[:,:,:2]).square().sum(-1)/.08**2
 rough=(ground.amax(-1)-ground.amin(-1)).div(.08).clamp(0,1)
 base_z=state[:,0]+field.sample(pose[:,:2])
 support_height=ground.mean(-1)
 height=base_z-support_height
 height_cost=((.28-.04*rough-height).clamp_min(0)+(height-.40-.04*rough).clamp_min(0)).square()/.08**2
 q=state[:,10:22]+c['default_q'];limits=c['limits']
 violation=((limits[:,0]-q).clamp_min(0)+(q-limits[:,1]).clamp_min(0)).sum(-1)
 near=(clearance<.04).float()
 forward=(state[:,4:6]*command[:,:2]).sum(-1)/speed
 return dict(
  track_linear_velocity=torch.exp(-(state[:,4:6]-command[:,:2]).square().sum(-1)/.09),
  track_angular_velocity=torch.exp(-(state[:,9]-command[:,2]).square()/.09),
  upright=torch.exp(-state[:,1:3].square().sum(-1)/.5**2),
  standstill=(command.square().sum(-1)<1e-6).float()*(state[:,4:10].square().sum(-1)+.01*state[:,22:34].square().sum(-1)),
  action_rate_l2=(action-previous).square().sum(-1),dof_pos_limits=violation,terminated=1-survival_prob,
  base_height=height_cost,foot_xy=xy_error.mean(-1)*(.5+.5*(1-rough))*activation,
  swing_clearance=(clear_error*swing).mean(-1),
  contact_phase=(support-stance.float()).square().mean(-1)*activation,
  foot_slip=(support*velocity[:,:,:2].square().sum(-1)).mean(-1),
  landing_velocity=(near*velocity[:,:,2].square()).mean(-1),
  body_motion=state[:,6].square()+.2*state[:,7:9].square().sum(-1),
  obstacle_progress=forward.clamp(-.6,.6)*(speed>.1).float()*rough,
  foothold=foothold,gap=gap)

def env_reward(env,term):
 if getattr(env,"_foot_reward_counter",-1)==env.common_step_counter:return env._foot_reward_cache[term].clone()
 from .terrain import TerrainBridge,TerrainField
 bridge=getattr(env,'_go2w_posture_bridge',None)
 if bridge is None:
  bridge=TerrainBridge(env,TerrainField.from_env(env));env._go2w_posture_bridge=bridge
 if not hasattr(env,'_pointfoot_reward_constants'):env._pointfoot_reward_constants=constants(bridge)
 action=bridge.canonical_action(env.action_manager.action)
 previous=bridge.canonical_action(env.action_manager.prev_action)
 state=bridge.state();pose=bridge.pose();command=env.command_manager.get_command('twist')
 terrain=getattr(env.scene,'terrain',None)
 kind=None if terrain is None or not hasattr(terrain,'terrain_types') else terrain.terrain_types.float()
 env._foot_reward_cache=reward_terms(state,action,previous,command,env._pointfoot_reward_constants,
  (~env.termination_manager.terminated).float(),bridge.field.scan(state,pose),pose,bridge.field,bridge.contacts(),kind)
 env._foot_reward_counter=env.common_step_counter
 return env._foot_reward_cache[term].clone()
