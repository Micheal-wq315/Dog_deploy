"""大狗 DigDogv1 点足版：真实采样、残差动力学、想象梯度Actor及独立确定性检查。"""

import argparse
import copy
import json
import math
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from .bridge import Go2WBridge, pose_delta
from ..robot.config import robot_signature
from .world_model import Go2WWorldModel
from .policy import Actor, Network, FRAME_DIM, CRITIC_DIM, POLICY_FORMAT, critic_input, checkpoint_wheel_vel
from .replay import Replay
from .imagine import frozen, rollout, make_imagined_step
from .recovery import recover_invalid_envs


def constants_from_bridge(bridge):
  from .task import constants
  return constants(bridge)


def actor_learning_rate(args, iteration):
  age = max(0, iteration - args.actor_schedule_start)
  progress = min(1., max(0, age - args.actor_hold_iterations) / max(1, args.lr_decay_iterations))
  decay = args.lr_final_frac + (1.-args.lr_final_frac)*.5*(1.+math.cos(math.pi*progress))
  ramp = min(1., (age+1) / max(1, args.actor_ramp_iterations))
  return args.lr * args.actor_lr_scale * decay * ramp


def update(optimizer, loss, parameters, element_clip=None):
  parameters = list(parameters)
  optimizer.zero_grad(set_to_none=True)
  if not torch.isfinite(loss):
    print("RECOVERY skipped non-finite loss; optimizer unchanged", flush=True)
    return False
  loss.backward()
  norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=False)
  if not torch.isfinite(norm):
    optimizer.zero_grad(set_to_none=True)
    print("RECOVERY skipped non-finite gradient; optimizer unchanged", flush=True)
    return False
  if element_clip is not None:
    torch.nn.utils.clip_grad_value_(parameters, element_clip)
  optimizer.step()
  return True


class Learner:
  """Pure-Torch learner; also usable in small CPU tests without mjlab."""
  def __init__(self, args, field=None):
    self.args, self.field = args, field
    self.format = POLICY_FORMAT
    self.model_class = Go2WWorldModel
    extra = 0
    if field is not None:
      from .terrain_world_model import TerrainWorldModel, POLICY_FORMAT as TERRAIN_FORMAT
      from .terrain import SCAN_DIM
      self.model_class, self.format, extra = TerrainWorldModel, TERRAIN_FORMAT, SCAN_DIM
    self.model = self.model_class(tuple(args.wm_hidden), **(
      {"contacts":args.terrain_revision >= 4, "collisions":args.terrain_revision >= 7} if field is not None else {})).to(args.device)
    self.actor = Actor(args.history, tuple(args.actor_hidden)).to(args.device)
    self.critic = Network(CRITIC_DIM + extra, 1, tuple(args.critic_hidden)).to(args.device)
    self.target = copy.deepcopy(self.critic)
    self.optimizers = [torch.optim.Adam(module.parameters(), lr=args.lr)
                       for module in (self.model, self.actor, self.critic)]
    self.curriculum_state = None
    self.policy_updates = 0
    self.skipped_updates = 0
    self.terrain_state = None
    self.env_steps = 0
    self.invalid_env_resets=0
    self.imagined_step = None
    self.compiler_fallbacks=0
    self.consecutive_actor_failures=0

  def critic_input(self, state, command, context, previous, pose):
    x = critic_input(state, command, context, previous)
    return x if self.field is None else torch.cat((x, self.field.scan(state, pose)), -1)

  def fit(self, b):
    if self.field is not None:
      loss = self.model.terrain_loss(b, self.field)
      if not update(self.optimizers[0], loss, self.model.parameters()):
        self.skipped_updates += 1
        return None
      return loss.item()
    loss = self.model.loss(b["state"][:, 0], b["action"], b["previous"], b["context"][:, 0],
                           b["next"], b["pose"], b["valid"].squeeze(-1))
    if not update(self.optimizers[0], loss, self.model.parameters()):
      self.skipped_updates += 1
      return None
    return loss.item()

  def improve(self, seed, constants, weights, env_order, dt):
    args = self.args
    if args.compile_imagination and self.imagined_step is None:
      step = make_imagined_step(self.model,self.actor,constants,weights,env_order,
                                dt=dt,field=self.field)
      self.imagined_step = torch.compile(step, backend="aot_eager", dynamic=False, fullgraph=True)
    with frozen(self.model, self.target):
      actor_loss, inputs, targets = rollout(
        self.model, self.actor, self.target, seed, constants, weights, env_order,
        horizon=args.horizon, dt=dt, gamma=args.gamma, lam=args.lam, field=self.field,
        step_fn=self.imagined_step)
      ok=update(self.optimizers[1],actor_loss,self.actor.parameters(),args.actor_grad_clip_element)
      if not ok and self.imagined_step is not None:
        # 模型梯度算法不变；编译反传异常时用普通Torch重算同一批种子。
        self.compiler_fallbacks+=1
        self.imagined_step=None
        args.compile_imagination=False
        print("编译想象反传异常，切换普通Torch并重算；保持完整模型梯度。",flush=True)
        actor_loss,inputs,targets=rollout(self.model,self.actor,self.target,seed,constants,weights,env_order,
            horizon=args.horizon,dt=dt,gamma=args.gamma,lam=args.lam,field=self.field,step_fn=None)
        ok=update(self.optimizers[1],actor_loss,self.actor.parameters(),args.actor_grad_clip_element)
      if not ok:
        self.skipped_updates+=1;self.consecutive_actor_failures+=1
        if not getattr(self,"_gradient_debug_saved",False):
          torch.save(dict(seed={k:v.detach().cpu() for k,v in seed.items()},model=self.model.checkpoint(),actor=self.actor.state_dict()),Path(args.output)/"gradient_debug.pt")
          self._gradient_debug_saved=True
        if self.consecutive_actor_failures>=10:raise RuntimeError("连续10次Actor梯度失败，停止以避免空转训练")
        return None,None
      self.consecutive_actor_failures=0
    x, y = inputs.flatten(0, 1), targets.flatten()
    critic_losses = []
    for _ in range(args.critic_epochs):
      order = torch.randperm(len(x),device=x.device)
      for indices in order.tensor_split(min(args.critic_minibatches,len(x))):
        value = self.critic(x[indices]).squeeze(-1)
        critic_loss = torch.nn.functional.mse_loss(value, y[indices])
        if not update(self.optimizers[2], critic_loss, self.critic.parameters()):
          self.skipped_updates += 1
          return actor_loss.item(), None
        critic_losses.append(critic_loss.detach())
    with torch.no_grad():
      for target, source in zip(self.target.parameters(), self.critic.parameters()):
        target.lerp_(source, 0.02)
      # Running normalization is a coordinate system, not an EMA weight.
      self.target.norm.load_state_dict(self.critic.norm.state_dict())
    self.policy_updates += 1
    return actor_loss.item(), torch.stack(critic_losses).mean().item()

  def save(self, path, step, env_order, dt):
    checkpoint = {
      "training_algorithm": "pointfoot-supermimic-full-return-v1",
      "robot_id": self.args.robot, "robot_signature": robot_signature(self.args.robot),
      "adaptation_version": 1, "display_name": "大狗 DigDogv1 点足版",
      "compiler_fallbacks":self.compiler_fallbacks,"skipped_updates":self.skipped_updates,"invalid_env_resets":self.invalid_env_resets,
      "format": self.format, "step": step, "history": self.actor.history,
      "actor_wheel_vel": False,
      "actor_hidden": self.actor.hidden, "actor": self.actor.state_dict(),
      "world_model": self.model.checkpoint(), "critic": self.critic.state_dict(),
      "target_critic": self.target.state_dict(), "config": vars(self.args),
      "optimizers": [o.state_dict() for o in self.optimizers],
      "terrain_state": self.terrain_state,
      "curriculum_state": self.curriculum_state,
      "env_steps": self.env_steps,
      "policy_updates": self.policy_updates, "env_action_order": list(env_order), "dt": dt,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)

  def load(self, path, env_order, dt):
    ckpt = torch.load(path, map_location=self.args.device, weights_only=True)
    legacy = ckpt.get("format") in ("go2w-flat-policy-v1", "go2w-terrain-policy-v1")
    if ckpt.get("robot_id", "go2w" if legacy else None) != self.args.robot:
      raise ValueError("机器人类型不匹配，不能把Go2W权重当作RP_wd模型恢复")
    if not legacy and ckpt.get("robot_signature") != robot_signature(self.args.robot):
      raise ValueError("机器人资产/驱动契约变化，禁止静默续训")
    if not legacy:
      for name in ('promotion_window', 'promotion_min_trials', 'promotion_pass_rate', 'probe_fraction'):
        if ckpt['config'].get(name) != getattr(self.args, name):
          raise ValueError('恢复时课程参数变化：' + name)
    if legacy:
      ckpt["format"] = self.format
    if ckpt.get("format") != self.format:
      raise ValueError("Resume requires a Go2W policy checkpoint")
    if ckpt["env_action_order"] != list(env_order) or abs(ckpt["dt"] - dt) > 1e-8:
      raise ValueError("Checkpoint action order/control timestep differs from this environment")
    if ckpt["history"] != self.args.history:
      raise ValueError("Checkpoint observation history differs")
    if self.field is not None and ckpt["config"]["seed"] != self.args.seed:
      raise ValueError("Terrain resume requires the same terrain seed")
    if ckpt["config"].get("terrain_revision", 1) != getattr(self.args, "terrain_revision", 1):
      raise ValueError("Terrain revision/action scale differs; start a new run for a different revision")
    if checkpoint_wheel_vel(ckpt):
      raise ValueError("Legacy actor uses measured wheel speeds; start a new run for the 245D actor")
    self.actor.load_state_dict(ckpt["actor"])
    self.model.load_state_dict(self.model_class.from_checkpoint(ckpt["world_model"]).state_dict())
    self.critic.load_state_dict(ckpt["critic"])
    self.target.load_state_dict(ckpt["target_critic"])
    for optimizer, data in zip(self.optimizers, ckpt["optimizers"]):
      optimizer.load_state_dict(data)
    self.policy_updates = ckpt["policy_updates"]
    self.skipped_updates=ckpt.get("skipped_updates",0)
    self.compiler_fallbacks=ckpt.get("compiler_fallbacks",0)
    self.invalid_env_resets=ckpt.get("invalid_env_resets",0)
    if self.compiler_fallbacks: self.args.compile_imagination=False
    self.terrain_state = ckpt.get("terrain_state")
    self.curriculum_state = ckpt.get("curriculum_state")
    self.env_steps = ckpt.get("env_steps", ckpt["step"] * ckpt["config"].get("num_steps_per_env", 1))
    return ckpt["step"]


def parser():
  p = argparse.ArgumentParser(description=__doc__)
  for name, default in (("num-envs", 64), ("iterations", 10000), ("buffer-len", 512),
                        ("history", 5), ("wm-window", 8), ("wm-batch", 128),
                        ("wm-updates", 2), ("imag-batch", 128), ("horizon", 16),
                        ("warmup", 300), ("critic-epochs", 5), ("save-interval", 1000),
                        ("log-interval", 50), ("seed", 0), ("num-steps-per-env", 1), ("critic-minibatches", 4),
                        ("archive-size", 100000), ("actor-ramp-iterations", 1000),
                        ("lr-decay-iterations", 30000), ("actor-hold-iterations", 0), ("actor-schedule-start", 0)):
    p.add_argument("--" + name, type=int, default=default)
  for name, default in (("lr", 2e-4), ("action-noise", 0.15), ("gamma", 0.99), ("lam", 0.95)):
    p.add_argument("--" + name, type=float, default=default)
  p.add_argument("--compile-imagination", action=argparse.BooleanOptionalAction, default=False)
  p.add_argument("--prioritized-replay", action="store_true")
  p.add_argument("--actor-lr-scale", type=float, default=.3)
  p.add_argument("--lr-final-frac", type=float, default=.03)
  p.add_argument("--actor-grad-clip-element", type=float, default=.1)
  p.add_argument("--wm-hidden", type=int, nargs="+", default=[512, 512, 512])
  p.add_argument("--actor-hidden", type=int, nargs="+", default=[256, 128, 64])
  p.add_argument("--critic-hidden", type=int, nargs="+", default=[256, 128, 64])
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--terrain", action="store_true", help="Blind static rough-terrain task")
  p.add_argument("--terrain-revision", type=int, choices=(7,), default=7,
                 help="2: larger leg actions; 3: bidirectional commands; 4: rolling rewards and 30cm stairs; 5: performance-gated stair curriculum; 6: wheel-local clearance; 7: v5 plus non-wheel collision termination; new run required")
  p.add_argument("--command-curriculum", action="store_true", help="Progressively widen velocity ranges")
  p.add_argument("--command-curriculum-start", type=int, default=0, help="Learner iteration where the velocity schedule starts; preserve on resume")
  p.add_argument("--init-actor", default="", help="Initialize only the actor from a flat policy; dynamics/critic start fresh")
  p.add_argument("--promotion-window", type=int, default=256)
  p.add_argument("--promotion-min-trials", type=int, default=128)
  p.add_argument("--promotion-pass-rate", type=float, default=.6)
  p.add_argument("--probe-fraction", type=float, default=.2)
  p.add_argument("--robot", choices=("rpwd_digdog",), default="rpwd_digdog")
  p.add_argument("--resume", default="")
  p.add_argument("--output", default="")
  return p


def main():
  p = parser()
  args = p.parse_args()
  if not args.terrain:p.error("点足新任务必须使用--terrain；正式入口train_pointfoot_mjlab.sh已配置")
  if args.init_actor:p.error("不接受旧flat或轮足Actor，点足版需从零训练或完整恢复对应checkpoint")
  if args.device.startswith("cuda"):torch.set_float32_matmul_precision("high")
  positive = (args.archive_size, args.critic_minibatches, args.lr_decay_iterations, args.num_steps_per_env, args.num_envs, args.iterations, args.buffer_len, args.history, args.wm_window,
              args.wm_batch, args.wm_updates, args.imag_batch, args.horizon, args.critic_epochs,
              args.save_interval, args.log_interval, *args.wm_hidden, *args.actor_hidden,
              *args.critic_hidden)
  if min(positive) < 1 or args.warmup < 0 or args.buffer_len < args.wm_window:
    p.error("Positive sizes required; buffer-len must be >= wm-window; warmup >= 0")
  if not (0 < args.gamma <= 1 and 0 <= args.lam <= 1 and args.lr > 0 and args.action_noise >= 0):
    p.error("Invalid optimizer, noise or return parameters")
  if args.actor_lr_scale <= 0 or not 0 < args.lr_final_frac <= 1 or args.actor_ramp_iterations < 0 or args.actor_grad_clip_element <= 0:
    p.error("Invalid actor learning-rate/gradient protection")
  if args.actor_hold_iterations < 0:
    p.error("actor-hold-iterations must be nonnegative")
  if args.terrain_revision >= 5 and (not args.command_curriculum or args.archive_size < 2):
    p.error("terrain v5 requires command-curriculum and archive-size >=2")
  if args.prioritized_replay and (not args.terrain or args.terrain_revision < 4):
    p.error("prioritized-replay requires measured contacts from terrain revision >=4")
  from mjlab.envs import ManagerBasedRlEnv
  from rpwd_digdog_supermimic.env.flat import go2w_velocity_flat_env_cfg

  torch.manual_seed(args.seed)
  if args.resume and args.init_actor:
    p.error("Choose resume or init-actor")
  if args.terrain_revision >= 2 and (not args.terrain or args.init_actor):
    p.error("terrain-revision >=2 requires --terrain and cannot initialize a flat actor")
  if args.terrain:
    from rpwd_digdog_supermimic.env.terrain import go2w_velocity_terrain_env_cfg
    cfg = go2w_velocity_terrain_env_cfg(history=args.history, seed=args.seed, revision=args.terrain_revision, robot=args.robot)
  else:
    cfg = go2w_velocity_flat_env_cfg(history=args.history, robot=args.robot)
  if args.command_curriculum:
    if not args.terrain or args.terrain_revision < 4:
      p.error("command-curriculum requires terrain revision >=4")
    from .command_curriculum import set_command_stage
    set_command_stage(cfg.commands["twist"], 0)
  if not 1 <= args.promotion_min_trials <= args.promotion_window or not 0 < args.promotion_pass_rate <= 1 or not 0 < args.probe_fraction <= .5:
    p.error("课程窗口/门槛/考核环境比例不合法")
  cfg.promotion_window = args.promotion_window
  cfg.promotion_min_trials = args.promotion_min_trials
  cfg.promotion_pass_rate = args.promotion_pass_rate
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    obs, _ = env.reset()
    field = None
    if args.terrain:
      from .terrain import TerrainField, TerrainBridge
      field = TerrainField.from_env(env)
      bridge = TerrainBridge(env, field)
      env._go2w_posture_bridge = bridge
      print(f"Shared terrain atlas: {field.h.numel()*4/2**20:.1f} MiB", flush=True)
    else:
      bridge = Go2WBridge(env)
    learner = Learner(args, field)
    if args.init_actor:
      ckpt = torch.load(args.init_actor, map_location=args.device, weights_only=True)
      if ckpt.get('robot_id', 'go2w') != args.robot:
        raise ValueError('init-actor机器人类型不匹配')
      if ckpt.get("format") not in (POLICY_FORMAT, 'go2w-flat-policy-v1') or ckpt["history"] != args.history:
        raise ValueError("init-actor requires a matching flat actor")
      if ckpt['format'] == POLICY_FORMAT and ckpt.get('robot_signature') != robot_signature(args.robot):
        raise ValueError('init-actor物理契约不匹配')
      if ckpt["env_action_order"] != list(bridge.to_env) or abs(ckpt["dt"]-env.step_dt)>1e-8:
        raise ValueError("Warm-start action contract differs")
      if checkpoint_wheel_vel(ckpt):
        raise ValueError("Cannot initialize a 245D actor from a legacy wheel-speed actor")
      learner.actor.load_state_dict(ckpt["actor"])
    constants = constants_from_bridge(bridge)
    constants["reference"] = learner.actor.reference
    weights = {name: term.weight for name, term in cfg.rewards.items()}
    contact_targets = args.terrain and args.terrain_revision >= 4
    replay = Replay(args.buffer_len, args.num_envs, args.device, FRAME_DIM * args.history,
                    terrain=args.terrain, contacts=contact_targets, courses=args.terrain_revision >= 5, collisions=args.terrain_revision >= 7)
    start = learner.load(args.resume, bridge.to_env, env.step_dt) if args.resume else 0
    if args.command_curriculum:
      set_command_stage(env.command_manager.get_term("twist").cfg,
                        start - args.command_curriculum_start,
                        stage=(int(learner.curriculum_state['stage']) if learner.curriculum_state else 0)
                        if args.terrain_revision >= 5 else None)
    if args.terrain and learner.terrain_state is not None:
      terrain = env.scene.terrain
      levels = learner.terrain_state["levels"].to(args.device)
      types = learner.terrain_state["types"].to(args.device)
      if levels.shape != terrain.terrain_levels.shape:
        raise ValueError("Terrain resume requires the same num-envs")
      terrain.terrain_levels.copy_(levels)
      terrain.terrain_types.copy_(types)
      terrain.env_origins.copy_(terrain.terrain_origins[levels, types])
      obs, _ = env.reset()
    if args.terrain_revision >= 5:
      from .stair_curriculum import buffers, next_stage, summary, recent_counts, set_probe_stage, snapshot_recent, restore_recent, promotion_report
      stats = buffers(env)
      saved = learner.curriculum_state
      command_stage = int(saved['stage']) if saved else 0
      velocity_ema = saved['velocity_ema'] if saved else [1., 1., 1.]
      if saved:
        stats.copy_(saved['stats'].to(env.device))
      set_command_stage(env.command_manager.get_term('twist').cfg, 0, stage=command_stage)
      restore_recent(env, saved)
      set_probe_stage(env, command_stage, args.probe_fraction)
      obs, _ = env.reset()
    if start >= args.iterations:
      raise ValueError("iterations must exceed the resumed checkpoint step")
    if obs["actor"].shape[-1] != FRAME_DIM * args.history:
      raise ValueError("Go2W actor observation dimension mismatch")
    output = Path(args.output or "logs/go2w/" + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S"))
    output.mkdir(parents=True, exist_ok=True)
    from ..robot.config import profile, REPO_ROOT
    import subprocess
    manifest = dict(display_name="大狗 DigDogv1 点足版", config=vars(args), robot=profile(args.robot), robot_signature=robot_signature(args.robot),
                    actor_input=FRAME_DIM*args.history, actor_output=12, world_model_input=67+273 if args.terrain else 67,
                    critic_input=CRITIC_DIM+273 if args.terrain else CRITIC_DIM, matmul_precision=torch.get_float32_matmul_precision(),
                    physics_dt=cfg.sim.mujoco.timestep, policy_dt=env.step_dt, curriculum_version='recent-v1',
                    weights=weights, resume_step=start,
                    git_commit=subprocess.check_output(['git','-C',str(REPO_ROOT),'rev-parse','HEAD'],text=True).strip(),
                    git_dirty=bool(subprocess.check_output(['git','-C',str(REPO_ROOT),'status','--porcelain'],text=True).strip()))
    (output / f'run_config_from_{start}.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    previous = torch.zeros(args.num_envs, 12, device=args.device)
    ou = torch.zeros_like(previous)
    totals = torch.zeros(6, device=args.device)
    reward_totals = torch.zeros(len(env.reward_manager.active_terms), device=args.device)
    interval_steps = 0
    nonwheel_terminations = 0
    invalid_env_resets = learner.invalid_env_resets
    from .gait import PointfootGaitMetrics
    gait_metrics=PointfootGaitMetrics(args.num_envs,args.device,env.step_dt)
    cpu_check=None
    began = time.monotonic()
    print(f"Go2W replay storage: {sum(t.numel() * t.element_size() for t in replay.data.values()) / 2**20:.1f} MiB", flush=True)
    with (output / "metrics.jsonl").open("a") as log:
      for step in range(start, args.iterations):
        for group in learner.optimizers[1].param_groups:
          group["lr"] = actor_learning_rate(args, step)
        if args.command_curriculum and args.terrain_revision < 5:
          command_stage = set_command_stage(env.command_manager.get_term("twist").cfg,
                                             step - args.command_curriculum_start)
        for _ in range(args.num_steps_per_env):
          with torch.no_grad():
            history = obs["actor"].clone()
            state, pose, context = bridge.state(), bridge.pose(), bridge.context()
            command = env.command_manager.get_command("twist").clone()
            bias = bridge.robot.data.encoder_bias[:, bridge.legs].clone()
            learner.actor.norm.update(history)
            learner.critic.norm.update(learner.critic_input(state, command, context, previous, pose))
            learner.target.norm.load_state_dict(learner.critic.norm.state_dict())
            ou = 0.95 * ou + (1 - 0.95**2)**0.5 * torch.randn_like(ou)
            noise_scale = args.action_noise * max(.15, 1. - (step+1)/12000.)
            action = learner.actor(history, noise_scale * ou)
            if args.terrain_revision >= 5:
              course_types = env.scene.terrain.terrain_types.clone()
              course_levels = env.scene.terrain.terrain_levels.clone()
              course_origins = env.scene.env_origins.clone()
            obs, real_reward, terminated, truncated, _ = env.step(bridge.env_action(action))
            terminated, truncated = terminated.clone(), truncated.clone()
            obs, bad = recover_invalid_envs(env, obs, real_reward)
            invalid_env_resets += int(bad.sum())
            learner.invalid_env_resets=invalid_env_resets
            valid = ~(terminated | truncated | bad)
            real_reward = torch.where(bad, torch.zeros_like(real_reward), real_reward)
            nxt, delta = bridge.state(), pose_delta(pose, bridge.pose())
            from .kinematics import foot_state
            _,feet,_=foot_state(nxt,bridge.pose(),field,bridge.geometry)
            gait_metrics.update({"contacts":bridge.contacts()[:,:4],"foot_positions":feet},~valid)
            measured = torch.cat((nxt[:, 4:6], nxt[:, 9:10]), -1)
            reward_totals += env.reward_manager._step_reward[~bad].sum(0)
            totals[0] += real_reward.sum()
            totals[1] += terminated.sum()
            totals[2] += valid.sum()
            totals[3:] += (measured[valid] - command[valid]).abs().sum(0)
            interval_steps += 1
            course_data = {}
            if args.terrain_revision >= 7:
              label_valid = env._nonwheel_label_valid & ~bad & torch.isfinite(state).all(-1)
              course_data.update(collision=env._nonwheel_hit[:,None].float(),
                                 collision_valid=label_valid[:,None].float())
              nonwheel_terminations += int((env._nonwheel_hit & label_valid).sum())
            if args.terrain_revision >= 5:
              new_pose = bridge.pose()
              ground_rise = field.sample(new_pose[:,:2]) - field.sample(pose[:,:2])
              forward_motion = (delta[:,:2] * command[:,:2]).sum(-1) > 0
              offset = bridge.robot.data.root_link_pos_w - course_origins
              from .stair_curriculum import stair_pass
              reached = stair_pass(offset, bridge.robot.data.projected_gravity_b, course_levels, course_types == 1)
              crossing = (pose[:,0] - course_origins[:,0]).abs() <= 2.8
              useful = (ground_rise > .005) & forward_motion & valid
              course_data['course'] = torch.stack((course_types.float(), course_levels.float(),
                                                   useful.float(), (reached & crossing & valid).float()), -1)
            replay.add(state=state, action=action, previous=previous, context=context,
                       next=nxt, pose=delta, valid=valid[:, None], history=history,
                       command=command, bias=bias, **({"world_pose": pose} if args.terrain else {}),
                       **({"next_contacts":bridge.contacts()} if contact_targets else {}), **course_data)
            if valid.any():
              model_context = context if field is None else torch.cat((context, field.scan(state, pose)), -1)
              learner.model.update_stats(state[valid], action[valid], previous[valid], model_context[valid],
                                          nxt[valid], delta[valid])
            previous = action.clone()
            previous[~valid], ou[~valid] = 0., 0.
          learner.env_steps += 1
        wm_loss = actor_loss = critic_loss = None
        if replay.size >= args.wm_window:
          for _ in range(args.wm_updates):
            batch = (replay.sample_world_model(args.wm_batch, args.wm_window, capacity=args.archive_size)
                     if args.prioritized_replay else replay.sample(args.wm_batch, args.wm_window))
            wm_loss = learner.fit(batch)
          # Replay is not checkpointed; refit it during warmup after every resume.
          if step - start >= args.warmup:
            seed = {k: v[:, 0] for k, v in replay.sample(args.imag_batch, 1).items()}
            actor_loss, critic_loss = learner.improve(seed, constants, weights, bridge.to_env, env.step_dt)
        if (step + 1) % args.log_interval == 0 or step + 1 == args.iterations:
          row = dict(robot_id=args.robot, step=step + 1, actor_lr=learner.optimizers[1].param_groups[0]["lr"],
                     archive_windows=0 if replay.archive is None else replay.archive.size, env_steps=learner.env_steps,
                     total_transitions=learner.env_steps * args.num_envs,
                     num_steps_per_env=args.num_steps_per_env, wm_loss=wm_loss, actor_loss=actor_loss,
                     critic_loss=critic_loss, reward=(totals[0] / (interval_steps * args.num_envs)).item(),
                     falls=int(totals[1]), interval_steps=interval_steps,
                     velocity_mae=(totals[3:] / totals[2].clamp_min(1)).tolist(),
                     policy_updates=learner.policy_updates, compiler_fallbacks=learner.compiler_fallbacks, imagination_backend="aot_eager" if args.compile_imagination else "eager",
                     skipped_updates=learner.skipped_updates, invalid_env_resets=invalid_env_resets,
                     elapsed_s=time.monotonic() - began)
          if args.device.startswith("cuda"):
            row.update(cuda_allocated_mib=torch.cuda.memory_allocated()/2**20,cuda_reserved_mib=torch.cuda.memory_reserved()/2**20,cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
          if args.terrain_revision >= 7:
            row['nonwheel_terminations'] = nonwheel_terminations
          if args.terrain:
            row.update({k:v.item() for k,v in learner.model.last_contact_metrics.items()})
            row["terrain_level_mean"] = env.scene.terrain.terrain_levels.float().mean().item()
            row["terrain_level_max"] = env.scene.terrain.terrain_levels.max().item()
          if args.terrain_revision >= 5:
            velocity_ema = [.9*a + .1*b for a,b in zip(velocity_ema, row['velocity_mae'])]
            row['terrain_details'] = summary(env)
            row['weighted_reward_terms'] = dict(zip(env.reward_manager.active_terms,
              (reward_totals / (interval_steps * args.num_envs)).tolist()))
            row['success_archive_windows'] = 0 if replay.success_archive is None else replay.success_archive.size
            promoted = next_stage(command_stage, step + 1 - args.command_curriculum_start,
                                  recent_counts(env).cpu().numpy(), velocity_ema, args.promotion_min_trials, args.promotion_pass_rate)
            row['promotion_window'] = promotion_report(env, command_stage)
            row['gait']=gait_metrics.summary()
            if row['gait']['cycles']<64 or not .9<=row['gait']['frequency_hz']<=1.9 or not .08<=row['gait']['air_time_s']<=.6 or row['gait']['stride_m']<.04:
              promoted=command_stage
            row['next_command_stage'] = promoted
            gait_metrics.clear_interval()
          if args.command_curriculum:
            ranges = env.command_manager.get_term("twist").cfg.ranges
            row.update(command_stage=command_stage, command_ranges=[ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z])
          if args.terrain_revision >= 3:
            row["command_negative_vx_fraction"] = (command[:,0] < 0).float().mean().item()
            row["command_abs_max"] = command.abs().amax(0).tolist()
          log.write(json.dumps(row) + "\n")
          log.flush()
          print(json.dumps(row), flush=True)
          if args.terrain_revision >= 5 and promoted != command_stage:
            command_stage = promoted
            env._stair_stats.zero_()
            env._stair_recent.zero_()
            env._stair_recent_count.zero_()
            # 改变考核高度只作用于池内下一次自然复位，避免中途瞬移污染转移。
            env._stair_stage = command_stage
            set_command_stage(env.command_manager.get_term('twist').cfg, 0, stage=command_stage)
          reward_totals.zero_()
          totals.zero_()
          interval_steps = 0
        if (step + 1) % args.save_interval == 0 or step + 1 == args.iterations:
          if args.terrain:
            learner.terrain_state = {"levels": env.scene.terrain.terrain_levels.cpu().clone(),
                                     "types": env.scene.terrain.terrain_types.cpu().clone()}
          if args.terrain_revision >= 5:
            learner.curriculum_state = dict(stage=command_stage, velocity_ema=velocity_ema,
                                             stats=env._stair_stats.detach().cpu().clone(), recent=snapshot_recent(env))
          learner.save(output / f"model_{step + 1}.pt", step + 1, bridge.to_env, env.step_dt)
          if (step+1)%5000==0 and step+1<args.iterations and (cpu_check is None or cpu_check.poll() is not None):
            from .validation import launch_cpu_check
            cpu_check=launch_cpu_check(output/f"model_{step+1}.pt",output/"checks"/f"check_{step+1}")
    print(f"{args.robot} SuperMimic training complete: {output}")
  finally:
    env.close()
