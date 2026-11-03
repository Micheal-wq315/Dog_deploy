"""点足 SuperMimic：真实转移拟合、前视地形模型与可微想象 Actor。"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .pointfoot_terrain import advance_pose
import torch.nn.functional as F


STATE_DIM = 34
ACTION_DIM = 12
COMMAND_DIM = 3
SCAN_DIM = 21 * 13
CONTACT_DIM = 4
FORCE_DIM = 4
POSE_DIM = 3
CONTEXT_DIM = 7
HISTORY = 5
FRAME_DIMS = (3, 3, 3, 12, 12, 12)
POLICY_DIM = sum(FRAME_DIMS) * HISTORY
MODEL_INPUT_DIM = STATE_DIM + 2 * ACTION_DIM + CONTEXT_DIM + SCAN_DIM
MODEL_CONT_DIM = STATE_DIM + POSE_DIM + FORCE_DIM
MODEL_AUX_DIM = CONTACT_DIM + 2  # 四脚接触、非脚碰撞、回合终止
FORMAT = "rp-wd-pointfoot-supermimic-v2"


class RunningNorm(nn.Module):
    """保存在 checkpoint 中的在线归一化统计。"""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1.0e-4))

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        value = value.reshape(-1, self.mean.numel()).float()
        if value.numel() == 0:
            return
        batch_mean = value.mean(0)
        batch_var = value.var(0, unbiased=False)
        count = value.shape[0]
        delta = batch_mean - self.mean
        total = self.count + count
        self.var.copy_(
            ((self.var * self.count + batch_var * count
              + delta.square() * self.count * count / total) / total).clamp_min(1e-8)
        )
        self.mean.add_(delta * count / total)
        self.count.copy_(total)

    @property
    def scale(self) -> torch.Tensor:
        return self.var.clamp_min(1e-6).sqrt()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return ((value - self.mean) / self.scale).clamp(-10.0, 10.0)


class PointfootActor(nn.Module):
    """盲策略，只接收 225 维本体历史并输出归一化12关节位置动作。"""

    def __init__(self, hidden: tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        width = POLICY_DIM
        for size in hidden:
            layers.extend((nn.Linear(width, size), nn.ELU()))
            width = size
        layers.append(nn.Linear(width, ACTION_DIM))
        self.net = nn.Sequential(*layers)
        self.norm = RunningNorm(POLICY_DIM)
        self.hidden = tuple(hidden)
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(history)).tanh()


class PointfootCritic(nn.Module):
    """特权价值函数：机身状态、动作、前视高度图、脚端接触/载荷。"""

    def __init__(self, hidden: tuple[int, ...] = (256, 128, 64)):
        super().__init__()
        self.input_dim = STATE_DIM + COMMAND_DIM + CONTEXT_DIM + ACTION_DIM + SCAN_DIM
        layers: list[nn.Module] = []
        width = self.input_dim
        for size in hidden:
            layers.extend((nn.Linear(width, size), nn.ELU()))
            width = size
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.norm = RunningNorm(self.input_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(features))


class PointfootDynamics(nn.Module):
    """预测残差状态、机身位姿、脚端载荷和接触/终止事件。"""

    def __init__(self, hidden: tuple[int, ...] = (512, 512, 512)):
        super().__init__()
        self.hidden = tuple(hidden)
        layers: list[nn.Module] = []
        width = MODEL_INPUT_DIM
        for size in hidden:
            layers.extend((nn.Linear(width, size), nn.SiLU()))
            width = size
        self.trunk = nn.Sequential(*layers)
        self.continuous_head = nn.Linear(width, MODEL_CONT_DIM)
        self.event_head = nn.Linear(width, MODEL_AUX_DIM)
        self.input_norm = RunningNorm(MODEL_INPUT_DIM)
        self.output_norm = RunningNorm(MODEL_CONT_DIM)
        nn.init.zeros_(self.continuous_head.bias)
        nn.init.zeros_(self.event_head.weight)
        nn.init.constant_(self.event_head.bias[-1], -4.0)
        nn.init.constant_(self.event_head.bias[CONTACT_DIM:CONTACT_DIM + 1], -4.6)

    @staticmethod
    def targets(batch):
        return torch.cat((batch["next_state"] - batch["state"], batch["pose"],
                          batch["next_force"]), dim=-1)

    @torch.no_grad()
    def update_stats(self, batch, valid):
        if valid.any():
            inputs = torch.cat((batch["state"], batch["action"], batch["previous"],
                                batch["context"], batch["scan"]), -1)
            self.input_norm.update(inputs[valid])
            self.output_norm.update(self.targets(batch)[valid])

    def forward(self, state, action, previous, context, scan):
        inputs = torch.cat((state, action, previous, context, scan), dim=-1)
        latent = self.trunk(self.input_norm(inputs))
        continuous = self.continuous_head(latent) * self.output_norm.scale + self.output_norm.mean
        event = self.event_head(latent)
        return {
            "continuous": continuous,
            "state": state + continuous[..., :STATE_DIM],
            "pose_delta": continuous[..., STATE_DIM:STATE_DIM + POSE_DIM],
            "force": continuous[..., -FORCE_DIM:].clamp(0, 1),
            "contact_logit": event[..., :CONTACT_DIM],
            "collision_logit": event[..., CONTACT_DIM:CONTACT_DIM + 1],
            "terminated_logit": event[..., CONTACT_DIM + 1:CONTACT_DIM + 2],
        }


class PointfootReplay:
    """按时间和环境保存短序列；楼梯接触/摔倒附近事件提高采样概率。"""

    SHAPES = {
        "state": STATE_DIM, "action": ACTION_DIM, "previous": ACTION_DIM,
        "context": CONTEXT_DIM, "world_pose": POSE_DIM, "bias": 15,
        "command": COMMAND_DIM, "scan": SCAN_DIM, "force": FORCE_DIM,
        "contacts": CONTACT_DIM, "history": POLICY_DIM, "pose": POSE_DIM,
        "next_state": STATE_DIM, "next_scan": SCAN_DIM, "next_force": FORCE_DIM,
        "next_contacts": CONTACT_DIM, "reward": 1, "collision": 1,
    }

    def __init__(self, capacity: int, num_envs: int, device: torch.device):
        self.capacity, self.num_envs, self.device = capacity, num_envs, device
        self.data = {
            key: torch.zeros(capacity, num_envs, width, device=device)
            for key, width in self.SHAPES.items()
        }
        self.terminated = torch.zeros(capacity, num_envs, 1, device=device)
        self.valid = torch.zeros_like(self.terminated)
        self.events = torch.zeros(capacity, num_envs, dtype=torch.bool, device=device)
        self.stamps = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.total_steps = 0

    @property
    def size(self) -> int:
        return min(self.total_steps, self.capacity)

    def add(
        self, transition: dict[str, torch.Tensor], terminated: torch.Tensor,
        valid: torch.Tensor, event: torch.Tensor,
    ) -> None:
        slot = self.total_steps % self.capacity
        for name, value in transition.items():
            self.data[name][slot].copy_(value.detach().reshape(self.num_envs, -1))
        self.terminated[slot].copy_(terminated.detach().reshape(self.num_envs, 1).float())
        self.valid[slot].copy_(valid.detach().reshape(self.num_envs, 1).float())
        self.events[slot].copy_(event.detach().reshape(self.num_envs).bool())
        self.stamps[slot] = self.total_steps
        self.total_steps += 1

    def _sample_indices(self, batch: int, window: int = 1):
        if self.total_steps < window:
            raise RuntimeError("SuperMimic 回放数据不足")
        oldest = max(0, self.total_steps - self.capacity)
        latest = self.total_steps - window
        event_slots, event_envs = self.events.nonzero(as_tuple=True)
        if event_slots.numel():
            event_steps = self.stamps[event_slots]
            keep = (event_steps >= oldest) & (event_steps <= latest + window - 1)
            event_slots, event_envs, event_steps = (
                event_slots[keep], event_envs[keep], event_steps[keep]
            )
        event_count = min(batch // 4, int(event_slots.numel()))
        uniform_count = batch - event_count
        starts = torch.randint(oldest, latest + 1, (uniform_count,), device=self.device)
        env_ids = torch.randint(self.num_envs, (uniform_count,), device=self.device)
        if event_count:
            selected = torch.randint(event_slots.numel(), (event_count,), device=self.device)
            event_steps = event_steps[selected]
            offsets = torch.randint(window, (event_count,), device=self.device)
            starts = torch.cat((starts, (event_steps - offsets).clamp(oldest, latest)))
            env_ids = torch.cat((env_ids, event_envs[selected]))
        order = torch.randperm(batch, device=self.device)
        stamps = starts[order, None] + torch.arange(window, device=self.device)[None, :]
        slots = stamps.remainder(self.capacity)
        env_ids = env_ids[order]
        sampled = {key: value[slots, env_ids[:, None]] for key, value in self.data.items()}
        sampled["terminated"] = self.terminated[slots, env_ids[:, None]]
        sampled["valid"] = self.valid[slots, env_ids[:, None]]
        sampled["sequence_valid"] = self.stamps[slots].eq(stamps).unsqueeze(-1).float()
        return sampled

    def sample_windows(self, batch: int, window: int) -> dict[str, torch.Tensor]:
        return self._sample_indices(batch, window)

    def sample_steps(self, batch: int) -> dict[str, torch.Tensor]:
        result = self._sample_indices(batch, 1)
        return {key: value[:, 0] for key, value in result.items()}


def _critic_features(state, command, previous, scan, context):
    return torch.cat((state, command, context, previous, scan), dim=-1)


def _push_history(history: torch.Tensor, latest: torch.Tensor) -> torch.Tensor:
    chunks = []
    history_offset = frame_offset = 0
    for width in FRAME_DIMS:
        chunks.append(torch.cat((
            history[:, history_offset + width:history_offset + width * HISTORY],
            latest[:, frame_offset:frame_offset + width],
        ), dim=-1))
        history_offset += width * HISTORY
        frame_offset += width
    return torch.cat(chunks, dim=-1)


def _imagined_frame(
    state: torch.Tensor, action: torch.Tensor, command: torch.Tensor,
    gyro_bias: torch.Tensor, joint_bias: torch.Tensor,
) -> torch.Tensor:
    """与点足策略项顺序一致，并复现训练观测中的传感器噪声幅值。"""
    n = state.shape[0]
    angular = state[:, 7:10] + gyro_bias + (torch.rand(n, 3, device=state.device) * 0.40 - 0.20)
    gravity = state[:, 1:4] + (torch.rand(n, 3, device=state.device) * 0.10 - 0.05)
    joint_pos = state[:, 10:22] + joint_bias + (torch.rand(n, 12, device=state.device) * 0.02 - 0.01)
    joint_vel = state[:, 22:34] + (torch.rand(n, 12, device=state.device) * 3.0 - 1.5)
    return torch.cat((angular, gravity, command, joint_pos, joint_vel * 0.05, action), dim=-1)


class PointfootSuperMimicLearner:
    """Go2W式更新：真实环境采样、模型拟合、想象回报 Actor 梯度和 TD(lambda) Critic。"""

    def __init__(self, args, device: torch.device, num_envs: int, field, task_rewards):
        self.args, self.device = args, device
        self.field, self.task_rewards = field, task_rewards
        self.model = PointfootDynamics(tuple(args.wm_hidden)).to(device)
        self.actor = PointfootActor(tuple(args.actor_hidden)).to(device)
        self.critic = PointfootCritic(tuple(args.critic_hidden)).to(device)
        self.target_critic = copy.deepcopy(self.critic).eval()
        self.target_critic.requires_grad_(False)
        self.replay = PointfootReplay(args.buffer_len, num_envs, device)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr * args.actor_lr_scale)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.policy_updates = 0
        self.skipped_updates = 0

    def fit_world_model(self) -> dict[str, float]:
        batch = self.replay.sample_windows(self.args.wm_batch, self.args.wm_window)
        seq = batch["sequence_valid"].squeeze(-1).bool()
        valid = batch["valid"].squeeze(-1).bool()
        state, pose = batch["state"][:, 0], batch["world_pose"][:, 0]
        active = seq[:, 0]
        accum = {key: [] for key in ("state", "pose", "force", "contact", "collision", "terminal")}
        scale = self.model.output_norm.scale
        for t in range(self.args.wm_window):
            active = active & seq[:, t]
            if not active.any():
                break
            # 原版流程：只处理终止之前的行，自动reset状态不进入自回归序列。
            keep = active.nonzero(as_tuple=True)[0]
            scan = self.field.scan(state[keep], pose[keep])
            pred = self.model(state[keep], batch["action"][keep, t], batch["previous"][keep, t],
                              batch["context"][keep, 0], scan)
            dynamics = valid[keep, t]
            if dynamics.any():
                indices = keep[dynamics]
                errors = {
                    "state": (pred["state"][dynamics] - batch["next_state"][indices, t]) / scale[:STATE_DIM],
                    "pose": (pred["pose_delta"][dynamics] - batch["pose"][indices, t]) / scale[STATE_DIM:STATE_DIM + POSE_DIM],
                    "force": (pred["continuous"][dynamics, -FORCE_DIM:] - batch["next_force"][indices, t]) / scale[-FORCE_DIM:],
                }
                for key, error in errors.items():
                    accum[key].append(error.square().mean() if key != "force" else
                                      F.smooth_l1_loss(error, torch.zeros_like(error)))
                accum["contact"].append(F.binary_cross_entropy_with_logits(
                    pred["contact_logit"][dynamics], batch["next_contacts"][indices, t]))
            accum["collision"].append(F.binary_cross_entropy_with_logits(
                pred["collision_logit"], batch["collision"][keep, t]))
            accum["terminal"].append(F.binary_cross_entropy_with_logits(
                pred["terminated_logit"], batch["terminated"][keep, t]))
            next_state = state.clone()
            next_pose = pose.clone()
            next_state[keep] = pred["state"]
            next_pose[keep] = advance_pose(pose[keep], pred["pose_delta"])
            state, pose = next_state, next_pose
            active = active & valid[:, t]
        coefficients = {"state": 1., "pose": 1., "force": .1, "contact": .1,
                        "collision": 1., "terminal": .2}
        if not any(accum.values()):
            return {"wm_loss": 0.0}
        means = {key: torch.stack(value).mean() for key, value in accum.items() if value}
        loss = sum(coefficients[key] * value for key, value in means.items())
        self.model_optimizer.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if torch.isfinite(grad):
                self.model_optimizer.step()
            else:
                self.skipped_updates += 1
        else:
            self.skipped_updates += 1
        return {"wm_loss": float(loss.detach()),
                **{f"wm_{key}": float(value.detach()) for key, value in means.items()}}

    def improve(self) -> dict[str, float]:
        args = self.args
        seed = self.replay.sample_steps(args.imag_batch)
        state = seed["state"].detach()
        history = seed["history"].detach()
        previous = seed["previous"].detach()
        command = seed["command"].detach()
        pose = seed["world_pose"].detach()
        context = seed["context"].detach()
        scan = self.field.scan(state, pose)
        contacts = seed["contacts"].detach()
        force = seed["force"].detach()
        # 保留每回合传感器零偏，仿真想象帧沿用当时可观测到的估计。
        gyro_bias = seed["bias"][:, :3]
        joint_bias = seed["bias"][:, 3:]
        critic_now = _critic_features(state, command, previous, scan, context)

        rewards, continuations, critic_inputs, next_critic_inputs = [], [], [], []
        alive = torch.ones(args.imag_batch, 1, device=self.device)
        discount = 1.0
        objective = torch.zeros_like(alive)
        # 与 Go2W 的 frozen(model, target) 一致：冻结模型权重，但保留
        # 想象状态对动作的导数，让梯度只更新 Actor。
        model_parameters = tuple(self.model.parameters())
        model_grad_state = tuple(parameter.requires_grad for parameter in model_parameters)
        self.model.zero_grad(set_to_none=True)
        self.model.requires_grad_(False)
        for _ in range(args.horizon):
            critic_inputs.append(_critic_features(
                state, command, previous, scan, context
            ).detach())
            action = self.actor(history)
            pred = self.model(state, action, previous, context, scan)
            collision = pred["collision_logit"].sigmoid()
            next_state = pred["state"]
            next_pose = advance_pose(pose, pred["pose_delta"])
            next_scan = self.field.scan(next_state, next_pose)
            gravity = F.normalize(next_state[:, 1:4], dim=-1, eps=1e-6)
            survival = torch.sigmoid((-gravity[:, 2:3] - math.cos(math.radians(55))) / .08)
            # 腿部擦碰在真实环境中仅受惩罚，不应当作死亡。
            # 学习的终止头监督真实terminated，几何代理显式覆盖倾角与低净空保护。
            height_survival = torch.sigmoid((next_state[:, :1] - 0.17) / 0.01)
            terminal = pred["terminated_logit"].sigmoid()
            continuation = survival * height_survival * (1.0 - terminal)
            reward = (0.02 * self.task_rewards.reward(
                next_state, action, previous, command, next_scan, collision, 1.0 - continuation
            ))[:, None]
            objective = objective + discount * alive * reward
            alive = alive * continuation
            rewards.append(reward)
            continuations.append(continuation)

            state, pose, scan = next_state, next_pose, next_scan
            contacts = pred["contact_logit"].sigmoid()
            force = pred["force"]
            previous = action
            frame = _imagined_frame(state, action, command, gyro_bias, joint_bias)
            history = _push_history(history, frame)
            discount *= args.gamma
            next_critic_inputs.append(_critic_features(
                state, command, previous, scan, context
            ))

        # Target Critic 参数被冻结，但不切断对末端想象状态的梯度；
        # 这与 Go2W rollout 的末端价值引导一致。
        final_value = self.target_critic(next_critic_inputs[-1])
        objective = objective + discount * alive * final_value
        actor_loss = -objective.mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(actor_loss):
            for parameter, requires_grad in zip(model_parameters, model_grad_state):
                parameter.requires_grad_(requires_grad)
            self.skipped_updates += 1
            return {"actor_loss": float("nan")}
        actor_loss.backward()
        actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
        if torch.isfinite(actor_grad):
            self.actor_optimizer.step()
        else:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.skipped_updates += 1
        for parameter, requires_grad in zip(model_parameters, model_grad_state):
            parameter.requires_grad_(requires_grad)
        if not torch.isfinite(actor_grad):
            return {"actor_loss": float(actor_loss.detach()), "actor_grad_norm": float("nan")}

        with torch.no_grad():
            reward_tensor = torch.stack(rewards)
            continuation_tensor = torch.stack(continuations)
            values = self.target_critic(torch.stack(next_critic_inputs)).squeeze(-1)
            carry = values[-1]
            targets = []
            for index in reversed(range(args.horizon)):
                carry = reward_tensor[index].squeeze(-1) + args.gamma * continuation_tensor[
                    index
                ].squeeze(-1) * ((1.0 - args.lam) * values[index] + args.lam * carry)
                targets.append(carry)
            target = torch.stack(targets[::-1]).reshape(-1)

        critic_x = torch.stack(critic_inputs).flatten(0, 1)
        critic_losses = []
        critic_grad_norms = []
        batch_size = critic_x.shape[0]
        minibatches = min(args.critic_minibatches, batch_size)
        for _ in range(args.critic_epochs):
            order = torch.randperm(batch_size, device=self.device)
            for indices in order.tensor_split(minibatches):
                prediction = self.critic(critic_x[indices]).squeeze(-1)
                critic_loss = F.mse_loss(prediction, target[indices])
                self.critic_optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(critic_loss):
                    self.skipped_updates += 1
                    continue
                critic_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                if torch.isfinite(grad_norm):
                    self.critic_optimizer.step()
                else:
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    self.skipped_updates += 1
                    continue
                critic_losses.append(critic_loss.detach())
                critic_grad_norms.append(grad_norm.detach())
        if not critic_losses:
            return {"actor_loss": float(actor_loss.detach()),
                    "actor_grad_norm": float(actor_grad.detach()),
                    "critic_loss": float("nan")}
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.lerp_(param, 0.02)
            self.target_critic.norm.load_state_dict(self.critic.norm.state_dict())
        self.policy_updates += 1
        return {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(torch.stack(critic_losses).mean()),
            "actor_grad_norm": float(actor_grad.detach()),
            "critic_grad_norm": float(torch.stack(critic_grad_norms).mean()),
            "imagined_reward": float(reward_tensor.mean().detach()),
        }

    def actor_learning_rate(self, iteration: int) -> float:
        args = self.args
        age = max(0, iteration - args.actor_schedule_start)
        progress = min(1.0, max(0, age - args.actor_hold_iterations)
                       / max(args.lr_decay_iterations, 1))
        cosine = args.lr_final_frac + (1.0 - args.lr_final_frac) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        ramp = min(1.0, (age + 1) / max(args.actor_ramp_iterations, 1))
        return args.lr * args.actor_lr_scale * cosine * ramp

    def state_dict(self, step: int) -> dict:
        return {
            "format": FORMAT,
            "step": step,
            "model": self.model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_hidden": self.actor.hidden,
            "config": vars(self.args).copy(),
            "policy_updates": self.policy_updates,
            "skipped_updates": self.skipped_updates,
        }

    def load_state_dict(self, checkpoint: dict) -> int:
        if checkpoint.get("format") != FORMAT:
            raise ValueError("checkpoint 不是点足版 SuperMimic")
        self.model.load_state_dict(checkpoint["model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.model_optimizer.load_state_dict(checkpoint["model_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.policy_updates = int(checkpoint.get("policy_updates", 0))
        self.skipped_updates = int(checkpoint.get("skipped_updates", 0))
        return int(checkpoint.get("step", 0))
