"""点足 SuperMimic：真实转移拟合、前视地形模型与可微想象 Actor。"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager

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
MODEL_TARGET_KEYS = ("next_state", "pose", "next_force", "next_contacts")
MODEL_SEED_KEYS = ("state", "action", "previous", "context", "world_pose")


def _finite_fields(batch, keys):
    return torch.stack([torch.isfinite(batch[key]).all(-1) for key in keys], -1).all(-1)


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
        valid = valid.reshape(-1).bool() & _finite_fields(batch, (
            "state", "action", "previous", "context", "scan", *MODEL_TARGET_KEYS,
        ))
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


class PointfootWindowArchive:
    """独立于在线回放的有界窗口环；只保留模型字段和原始有效前缀。"""

    def __init__(self, capacity: int, window: int):
        if capacity < 0 or window < 1:
            raise ValueError("archive容量不能为负，窗口必须为正数")
        self.capacity, self.window = capacity, window
        self.size = self.cursor = 0
        self.data = None

    @torch.no_grad()
    def add(self, windows: dict[str, torch.Tensor]) -> None:
        incoming = windows["state"].shape[0]
        count = min(incoming, self.capacity)
        if not count:
            return
        if any(value.shape[:2] != (incoming, self.window) for value in windows.values()):
            raise ValueError("archive窗口长度不一致")
        if self.data is None:
            self.data = {
                key: value.new_zeros((self.capacity, *value.shape[1:]))
                for key, value in windows.items()
            }
        # 超容量批次等价于逐条入环：丢掉前部，但游标仍推进完整批次。
        slots = (torch.arange(count, device=windows["state"].device)
                 + self.cursor + incoming - count).remainder(self.capacity)
        for key, value in self.data.items():
            value[slots] = windows[key][-count:].detach()
        self.cursor = (self.cursor + incoming) % self.capacity
        self.size = min(self.capacity, self.size + incoming)

    def sample(self, batch: int, window: int) -> dict[str, torch.Tensor]:
        if self.size == 0 or not 1 <= window <= self.window or batch < 1:
            raise ValueError("archive没有所需长度的窗口")
        ids = torch.randint(self.size, (batch,), device=self.data["state"].device)
        # 缩短窗口时仅取原前缀，不能从reset后的被屏蔽尾部重新开始。
        return {key: value[ids, :window] for key, value in self.data.items()}

    @property
    def storage_bytes(self) -> int:
        return sum(value.numel() * value.element_size()
                   for value in (self.data or {}).values())


class PointfootReplay:
    """在线均匀种子和模型专用事件/成功窗口；地形扫描按需重建。"""

    SHAPES = {
        "state": STATE_DIM, "action": ACTION_DIM, "previous": ACTION_DIM,
        "context": CONTEXT_DIM, "world_pose": POSE_DIM, "bias": 15,
        "command": COMMAND_DIM, "force": FORCE_DIM,
        "contacts": CONTACT_DIM, "history": POLICY_DIM, "pose": POSE_DIM,
        "next_state": STATE_DIM, "next_force": FORCE_DIM,
        "next_contacts": CONTACT_DIM, "reward": 1, "collision": 1,
    }

    MODEL_KEYS = (
        "state", "action", "previous", "context", "world_pose", "pose",
        "next_state", "next_force", "next_contacts", "collision",
    )

    def __init__(
        self, capacity: int, num_envs: int, device: torch.device,
        archive_windows: int = 0, success_archive_windows: int = 0, wm_window: int = 8,
    ):
        if (min(capacity, num_envs, wm_window) < 1
                or ((archive_windows or success_archive_windows) and wm_window > capacity)):
            raise ValueError("回放容量、环境数和窗口必须为正，窗口不能超过容量")
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
        self.archive = PointfootWindowArchive(archive_windows, wm_window)
        self.success_archive = PointfootWindowArchive(success_archive_windows, wm_window)
        self.last_sample_counts = {"wm_online_windows": 0, "wm_event_windows": 0,
                                   "wm_success_windows": 0}

    @property
    def size(self) -> int:
        return min(self.total_steps, self.capacity)

    @torch.no_grad()
    def add(
        self, transition: dict[str, torch.Tensor], terminated: torch.Tensor,
        valid: torch.Tensor, event: torch.Tensor, success_event: torch.Tensor | None = None,
    ) -> None:
        slot = self.total_steps % self.capacity
        # transition中的真实scan仍供update_stats临时使用，不进入持久存储。
        for name, storage in self.data.items():
            storage[slot].copy_(transition[name].detach().reshape(self.num_envs, -1))
        self.terminated[slot].copy_(terminated.detach().reshape(self.num_envs, 1).float())
        self.valid[slot].copy_((valid.detach().reshape(self.num_envs, 1).bool()
                               & ~self.terminated[slot].bool()).float())
        row = {key: value[slot] for key, value in self.data.items()}
        finite = _finite_fields(row, (*MODEL_SEED_KEYS, *MODEL_TARGET_KEYS))
        self.valid[slot].mul_(finite[:, None])
        self.events[slot].copy_(event.detach().reshape(self.num_envs).bool())
        self.stamps[slot] = self.total_steps
        self.total_steps += 1
        self._archive_events(self.archive, event)
        if success_event is not None:
            self._archive_events(self.success_archive, success_event)

    def _gather(self, starts, env_ids, window, model_only=False):
        stamps = starts[:, None] + torch.arange(window, device=self.device)[None, :]
        slots = stamps.remainder(self.capacity)
        keys = self.MODEL_KEYS if model_only else self.data.keys()
        sampled = {key: self.data[key][slots, env_ids[:, None]] for key in keys}
        sampled["terminated"] = self.terminated[slots, env_ids[:, None]]
        sampled["valid"] = self.valid[slots, env_ids[:, None]]
        sampled["valid"] = (sampled["valid"].bool()
                            & _finite_fields(sampled, MODEL_TARGET_KEYS).unsqueeze(-1)).float()
        matches = (self.stamps[slots].eq(stamps)
                   & _finite_fields(sampled, MODEL_SEED_KEYS)).unsqueeze(-1)
        continues = matches & sampled["valid"].bool() & ~sampled["terminated"].bool()
        prefix = torch.cat((torch.ones_like(continues[:, :1]), continues[:, :-1]), 1)
        # 首个终止转移仍监督事件头；其后所有reset状态均不可用。
        sampled["sequence_valid"] = (matches & prefix.cumprod(1).bool()).float()
        return sampled

    def _archive_events(self, archive, event):
        if not archive.capacity:
            return
        env_ids = event.detach().reshape(self.num_envs).bool().nonzero(as_tuple=True)[0]
        if not env_ids.numel():
            return
        latest = self.total_steps - 1
        oldest = max(0, self.total_steps - self.capacity)
        start = max(oldest, self.total_steps - archive.window)
        stamps = start + torch.arange(archive.window, device=self.device)
        slots = stamps.remainder(self.capacity)
        matches = self.stamps[slots].eq(stamps) & (stamps < latest)
        ended = ~self.valid[slots[:, None], env_ids[None, :], 0].bool()
        last_end = torch.where(ended & matches[:, None], stamps[:, None], -1).amax(0)
        # 事件前若有reset，窗口从该回合起点左对齐，未来位置以零和mask填充。
        starts = torch.maximum(last_end + 1, torch.full_like(last_end, start))
        windows = self._gather(starts, env_ids, archive.window, model_only=True)
        mask = windows["sequence_valid"].bool()
        windows = {key: torch.where(mask, value, torch.zeros_like(value))
                   for key, value in windows.items()}
        archive.add(windows)

    def _sample_indices(self, batch: int, window: int = 1, model_only=False):
        if batch < 1 or window < 1 or window > self.capacity:
            raise ValueError("采样批量和窗口必须为正，窗口不能超过回放容量")
        if self.size < window:
            raise RuntimeError("SuperMimic 回放数据不足")
        oldest = max(0, self.total_steps - self.capacity)
        latest = self.total_steps - window
        starts = torch.randint(oldest, latest + 1, (batch,), device=self.device)
        env_ids = torch.randint(self.num_envs, (batch,), device=self.device)
        return self._gather(starts, env_ids, window, model_only)

    def sample_windows(
        self, batch: int, window: int, *, prioritized: bool = True,
    ) -> dict[str, torch.Tensor]:
        counts = [min(batch // 4, archive.size)
                  if prioritized and window <= archive.window else 0
                  for archive in (self.archive, self.success_archive)]
        online_count = batch - sum(counts)
        parts = [self._sample_indices(online_count, window, model_only=True)]
        for archive, count in zip((self.archive, self.success_archive), counts):
            if count:
                parts.append(archive.sample(count, window))
        if prioritized:
            self.last_sample_counts = dict(zip(
                ("wm_online_windows", "wm_event_windows", "wm_success_windows"),
                (online_count, *counts),
            ))
        order = torch.randperm(batch, device=self.device)
        return {key: torch.cat([part[key] for part in parts])[order] for key in parts[0]}

    def sample_steps(self, batch: int) -> dict[str, torch.Tensor]:
        # Actor只从在线时间×环境的笛卡尔积均匀采样，绝不混入优先窗口。
        result = self._sample_indices(batch, 1)
        return {key: value[:, 0] for key, value in result.items()}

    def stats(self) -> dict[str, float]:
        online = (*self.data.values(), self.terminated, self.valid, self.events, self.stamps)
        return {
            "replay_online_steps": self.size,
            "replay_online_transitions": self.size * self.num_envs,
            "archive_windows": self.archive.size,
            "success_archive_windows": self.success_archive.size,
            "replay_storage_mib": sum(t.numel() * t.element_size() for t in online) / 2**20,
            "archive_storage_mib": (self.archive.storage_bytes
                                    + self.success_archive.storage_bytes) / 2**20,
            **self.last_sample_counts,
        }


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


@contextmanager
def _frozen(module):
    """冻结权重但保留输入梯度；包括异常退出在内都恢复原requires_grad。"""
    parameters = tuple(module.parameters())
    enabled = tuple(parameter.requires_grad for parameter in parameters)
    module.zero_grad(set_to_none=True)
    module.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in zip(parameters, enabled):
            parameter.requires_grad_(requires_grad)


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
        self.replay = PointfootReplay(
            args.buffer_len, num_envs, device,
            archive_windows=getattr(args, "archive_windows", 0),
            success_archive_windows=getattr(args, "success_archive_windows", 0),
            wm_window=getattr(args, "wm_window", 8),
        )
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr * args.actor_lr_scale)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.policy_updates = 0
        self.skipped_updates = 0

    def replay_stats(self) -> dict[str, float]:
        """供父训练脚本直接合并到日志；archive不进入checkpoint。"""
        return self.replay.stats()

    def fit_world_model(self) -> dict[str, float]:
        window = getattr(self.args, "wm_window", 8)
        batch = self.replay.sample_windows(self.args.wm_batch, window)
        seq = batch["sequence_valid"].squeeze(-1).bool()
        valid = batch["valid"].squeeze(-1).bool()
        state, pose = batch["state"][:, 0], batch["world_pose"][:, 0]
        active = seq[:, 0]
        accum = {key: [] for key in ("state", "pose", "force", "contact", "collision", "terminal")}
        scale = self.model.output_norm.scale
        for t in range(window):
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
            for name, logit, target in (("collision", "collision_logit", "collision"),
                                         ("terminal", "terminated_logit", "terminated")):
                label = batch[target][keep, t]
                finite = torch.isfinite(label).all(-1)
                if finite.any():
                    accum[name].append(F.binary_cross_entropy_with_logits(
                        pred[logit][finite], label[finite]))
            next_state = state.clone()
            next_pose = pose.clone()
            next_state[keep] = pred["state"]
            next_pose[keep] = advance_pose(pose[keep], pred["pose_delta"])
            state, pose = next_state, next_pose
            active = active & valid[:, t] & ~batch["terminated"][:, t, 0].bool()
        coefficients = {"state": 1., "pose": 1., "force": .1, "contact": .1,
                        "collision": 1., "terminal": .2}
        if not any(accum.values()):
            return {"wm_loss": 0.0}
        means = {key: torch.stack(value).mean() for key, value in accum.items() if value}
        loss = sum(coefficients[key] * value for key, value in means.items())
        self.model_optimizer.zero_grad(set_to_none=True)
        grad = loss.new_tensor(float("nan"))
        applied = False
        if torch.isfinite(loss):
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if torch.isfinite(grad):
                self.model_optimizer.step()
                applied = True
            else:
                self.model_optimizer.zero_grad(set_to_none=True)
                self.skipped_updates += 1
        else:
            self.skipped_updates += 1
        return {"wm_loss": float(loss.detach()), "wm_grad_norm": float(grad),
                "wm_update_applied": float(applied),
                "wm_valid_fraction": float((seq & valid).float().mean()),
                "wm_terminal_rate": float(batch["terminated"].squeeze(-1)[seq].mean()),
                **{f"wm_{key}": float(value.detach()) for key, value in means.items()}}

    @torch.no_grad()
    def prediction_metrics(self, batch_size: int = 128) -> dict[str, float]:
        """在线均匀回放的8步自回归诊断，并非独立验证集或泛化评测。

        rollout指标在有效动力学前缀上聚合，final指标只统计完整走到第8步的行。
        方向、速度、净空均与真实next_state对照，绝不用command代替真实目标。
        """
        if batch_size < 1:
            raise ValueError("预测诊断批量必须为正数")
        prefix = "replay_pred_8step_"
        metrics = {prefix + "transitions": 0.0, prefix + "final_windows": 0.0}
        if self.replay.size < 8:
            return metrics
        batch = self.replay.sample_windows(batch_size, 8, prioritized=False)
        state, pose = batch["state"][:, 0], batch["world_pose"][:, 0]
        active = batch["sequence_valid"][:, 0, 0].bool()
        totals, final, counts = {}, {}, 0
        event_totals, event_counts = {}, {}
        for t in range(8):
            active &= batch["sequence_valid"][:, t, 0].bool()
            keep = active.nonzero(as_tuple=True)[0]
            if not keep.numel():
                break
            pred = self.model(state[keep], batch["action"][keep, t],
                              batch["previous"][keep, t], batch["context"][keep, 0],
                              self.field.scan(state[keep], pose[keep]))
            for name, logit, target in (
                ("terminal_brier", "terminated_logit", "terminated"),
                ("collision_brier", "collision_logit", "collision"),
            ):
                label = batch[target][keep, t]
                finite = torch.isfinite(label).all(-1)
                if finite.any():
                    error = (pred[logit][finite].sigmoid() - label[finite]).square()
                    event_totals[name] = event_totals.get(name, 0) + error.sum()
                    event_counts[name] = event_counts.get(name, 0) + int(finite.sum())
            valid = (batch["valid"][keep, t, 0].bool()
                     & ~batch["terminated"][keep, t, 0].bool())
            next_pose = advance_pose(pose[keep], pred["pose_delta"])
            if valid.any():
                true = batch["next_state"][keep[valid], t]
                guess = pred["state"][valid]
                true_pose = advance_pose(batch["world_pose"][keep[valid], t],
                                         batch["pose"][keep[valid], t])
                yaw_error = next_pose[valid, 2] - true_pose[:, 2]
                errors = {
                    "gravity_mae": (F.normalize(guess[:, 1:4], dim=-1, eps=1e-6)
                                    - F.normalize(true[:, 1:4], dim=-1, eps=1e-6)).abs().mean(-1),
                    "lin_vel_mae_mps": (guess[:, 4:7] - true[:, 4:7]).abs().mean(-1),
                    "ang_vel_mae_radps": (guess[:, 7:10] - true[:, 7:10]).abs().mean(-1),
                    "base_clearance_mae_m": (guess[:, 0] - true[:, 0]).abs(),
                    "joint_pos_mae_rad": (guess[:, 10:22] - true[:, 10:22]).abs().mean(-1),
                    "pose_xy_mae_m": (next_pose[valid, :2] - true_pose[:, :2]).abs().mean(-1),
                    "yaw_mae_rad": torch.atan2(yaw_error.sin(), yaw_error.cos()).abs(),
                    "force_mae": (pred["force"][valid]
                                  - batch["next_force"][keep[valid], t]).abs().mean(-1),
                    "contact_brier": (pred["contact_logit"][valid].sigmoid()
                                      - batch["next_contacts"][keep[valid], t]).square().mean(-1),
                }
                count = int(valid.sum())
                counts += count
                for name, error in errors.items():
                    totals[name] = totals.get(name, 0) + error.sum()
                    if t == 7:
                        final[name] = error.mean()
                if t == 7:
                    metrics[prefix + "final_windows"] = count
            state = state.clone()
            pose = pose.clone()
            state[keep], pose[keep] = pred["state"], next_pose
            active[keep] &= valid
        metrics[prefix + "transitions"] = counts
        metrics.update({prefix + "rollout_" + name: float(value / counts)
                        for name, value in totals.items()} if counts else {})
        metrics.update({prefix + "final_" + name: float(value) for name, value in final.items()})
        metrics.update({prefix + name: float(value / event_counts[name])
                        for name, value in event_totals.items()})
        return metrics

    def _imagined_continuation(self, pred, scan=None, contacts=None, force=None):
        gravity = F.normalize(pred["state"][:, 1:4], dim=-1, eps=1e-6)
        survival = torch.sigmoid(
            (-gravity[:, 2:3] - math.cos(self.task_rewards.fall_angle)) / .08
        )
        clearance = pred["state"][:, 0]
        if self.task_rewards.profile == "full" and scan is not None:
            clearance = self.task_rewards.support_clearance(pred["state"], scan, contacts, force)
        height_survival = torch.sigmoid(
            (clearance[:, None] - self.task_rewards.hard_min_height) / .01
        )
        # 腿部擦碰仅受奖励惩罚；真实terminated头与profile几何保护共同决定生存。
        return survival * height_survival * (1.0 - pred["terminated_logit"].sigmoid())

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
        # 保留每回合传感器零偏，仿真想象帧沿用当时可观测到的估计。
        gyro_bias = seed["bias"][:, :3]
        joint_bias = seed["bias"][:, 3:]
        rewards, continuations, critic_inputs, next_critic_inputs = [], [], [], []
        quality = {key: state.new_zeros(()) for key in (
            "terminal_probability", "collision_probability", "contact_probability",
            "force", "action_saturation",
        )}
        alive = torch.ones(args.imag_batch, 1, device=self.device)
        discount = 1.0
        objective = torch.zeros_like(alive)
        # 与 Go2W 的 frozen(model, target) 一致：冻结模型权重，但保留
        # 想象状态对动作的导数，让梯度只更新 Actor。
        with _frozen(self.model):
            for _ in range(args.horizon):
                critic_inputs.append(_critic_features(
                    state, command, previous, scan, context
                ).detach())
                action = self.actor(history)
                pred = self.model(state, action, previous, context, scan)
                collision = pred["collision_logit"].sigmoid()
                contacts = pred["contact_logit"].sigmoid()
                force = pred["force"]
                next_state = pred["state"]
                next_pose = advance_pose(pose, pred["pose_delta"])
                next_scan = self.field.scan(next_state, next_pose)
                continuation = self._imagined_continuation(pred, next_scan, contacts, force)
                reward = (0.02 * self.task_rewards.reward(
                    next_state, action, previous, command, next_scan, collision,
                    1.0 - continuation, contacts=contacts, force=force,
                ))[:, None]
                objective = objective + discount * alive * reward
                alive = alive * continuation
                rewards.append(reward)
                continuations.append(continuation)
                for name, value in (
                    ("terminal_probability", pred["terminated_logit"].sigmoid()),
                    ("collision_probability", collision), ("contact_probability", contacts),
                    ("force", force), ("action_saturation", (action.abs() > .95).float()),
                ):
                    quality[name] += value.detach().mean()

                state, pose, scan = next_state, next_pose, next_scan
                previous = action
                frame = _imagined_frame(state, action, command, gyro_bias, joint_bias)
                history = _push_history(history, frame)
                discount *= args.gamma
                next_critic_inputs.append(_critic_features(
                    state, command, previous, scan, context
                ))

            # 保留完整horizon的回报及末端价值对Actor的梯度。
            final_value = self.target_critic(next_critic_inputs[-1])
            objective = objective + discount * alive * final_value
            actor_loss = -objective.mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(actor_loss):
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
            "imagined_continuation": float(continuation_tensor.mean()),
            "imagined_alive_final": float(alive.detach().mean()),
            **{f"imagined_{key}": float(value / args.horizon) for key, value in quality.items()},
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
