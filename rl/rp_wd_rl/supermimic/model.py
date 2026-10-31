"""RP_wd 世界模型、确定性教师、Critic 与 Odin 深度学生网络。"""

from __future__ import annotations

import copy

import torch
from torch import nn


STATE_DIM = 38
ACTION_DIM = 16
COMMAND_DIM = 6
SCAN_DIM = 16 * 12
TORQUE_DIM = 12
CONTACT_DIM = 4
WHEEL_FORCE_DIM = 4
ACTOR_DIM = STATE_DIM + COMMAND_DIM + ACTION_DIM + SCAN_DIM + TORQUE_DIM + CONTACT_DIM
CRITIC_DIM = ACTOR_DIM + WHEEL_FORCE_DIM
MODEL_INPUT_DIM = STATE_DIM + ACTION_DIM * 2 + COMMAND_DIM + SCAN_DIM
MODEL_CONT_DIM = STATE_DIM + 3 + SCAN_DIM + TORQUE_DIM + WHEEL_FORCE_DIM + 1
MODEL_AUX_DIM = CONTACT_DIM + 1
POLICY_DIM = 204
DEPTH_DIM = 8 * 16 * 12
DEPLOY_INPUT_DIM = POLICY_DIM + DEPTH_DIM
ACTION_LIMIT = 10.0
TEACHER_FORMAT = "rp-wd-supermimic-model-v1"
STUDENT_FORMAT = "rp-wd-supermimic-odin-student-v1"


class RunningNorm(nn.Module):
    """随 checkpoint 保存的在线均值方差归一化。"""

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
        batch_count = value.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.var.copy_(
            ((self.var * self.count + batch_var * batch_count
              + delta.square() * self.count * batch_count / total) / total).clamp_min(1.0e-8)
        )
        self.mean.add_(delta * batch_count / total)
        self.count.copy_(total)

    @property
    def scale(self) -> torch.Tensor:
        return self.var.clamp_min(1.0e-6).sqrt()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return ((value - self.mean) / self.scale).clamp(-10.0, 10.0)


def bounded_action(raw: torch.Tensor) -> torch.Tensor:
    """平滑逼近部署端 ±10 裁剪，既保持契约又给 Actor 提供边界梯度。"""
    return ACTION_LIMIT * torch.tanh(raw / ACTION_LIMIT)


def make_actor_features(
    state: torch.Tensor,
    command: torch.Tensor,
    previous_action: torch.Tensor,
    scan: torch.Tensor,
    torque: torch.Tensor,
    contacts: torch.Tensor,
) -> torch.Tensor:
    return torch.cat((state, command, previous_action, scan, torque, contacts), dim=-1)


def make_critic_features(
    actor_features: torch.Tensor, wheel_force: torch.Tensor
) -> torch.Tensor:
    return torch.cat((actor_features, wheel_force), dim=-1)


class DynamicsMember(nn.Module):
    """单个残差动力学成员：状态/位姿/高度图/载荷，以及奖励和终止辅助头。"""

    def __init__(self, hidden: tuple[int, ...] = (384, 384, 384)):
        super().__init__()
        layers: list[nn.Module] = []
        width = MODEL_INPUT_DIM
        for output_width in hidden:
            layers.extend((nn.Linear(width, output_width), nn.SiLU()))
            width = output_width
        self.trunk = nn.Sequential(*layers)
        self.continuous_head = nn.Linear(width, MODEL_CONT_DIM)
        self.auxiliary_head = nn.Linear(width, MODEL_AUX_DIM)
        self.input_norm = RunningNorm(MODEL_INPUT_DIM)
        self.output_norm = RunningNorm(MODEL_CONT_DIM)
        nn.init.zeros_(self.continuous_head.bias)
        nn.init.zeros_(self.auxiliary_head.weight)
        nn.init.constant_(self.auxiliary_head.bias[-1], -4.0)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        previous_action: torch.Tensor,
        command: torch.Tensor,
        scan: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        inputs = torch.cat((state, action, previous_action, command, scan), dim=-1)
        hidden = self.trunk(self.input_norm(inputs))
        continuous = self.continuous_head(hidden) * self.output_norm.scale + self.output_norm.mean
        auxiliary = self.auxiliary_head(hidden)
        offsets = (0, STATE_DIM, STATE_DIM + 3, STATE_DIM + 3 + SCAN_DIM,
                   STATE_DIM + 3 + SCAN_DIM + TORQUE_DIM,
                   STATE_DIM + 3 + SCAN_DIM + TORQUE_DIM + WHEEL_FORCE_DIM)
        i0, i1, i2, i3, i4, i5 = offsets
        return {
            "state": state + continuous[..., i0:i1],
            "pose_delta": continuous[..., i1:i2],
            "scan": (scan + continuous[..., i2:i3]).clamp(-1.0, 1.0),
            "torque": continuous[..., i3:i4].tanh(),
            "wheel_force": continuous[..., i4:i5].clamp(0.0, 1.0),
            "reward_scaled": continuous[..., i5:i5 + 1],
            "contacts_logit": auxiliary[..., :CONTACT_DIM],
            "terminated_logit": auxiliary[..., CONTACT_DIM:CONTACT_DIM + 1],
        }

    @staticmethod
    def continuous_target(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # 回放中的 reward 已是 [batch, 1]；不要再加一维，否则无法与其余目标拼接。
        reward_scaled = batch["reward"] / 5.0
        return torch.cat((
            batch["next_state"] - batch["state"],
            batch["pose_delta"],
            batch["next_scan"] - batch["scan"],
            batch["next_torque"],
            batch["next_wheel_force"],
            reward_scaled,
        ), dim=-1)


class DeterministicTeacher(nn.Module):
    """特权状态/扫描输入的确定性教师，动作范围与部署原始动作一致。"""

    def __init__(self, hidden: tuple[int, ...] = (512, 512, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        width = ACTOR_DIM
        for output_width in hidden:
            layers.extend((nn.Linear(width, output_width), nn.ELU()))
            width = output_width
        layers.append(nn.Linear(width, ACTION_DIM))
        self.net = nn.Sequential(*layers)
        self.hidden = tuple(hidden)
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return bounded_action(self.net(features))


class ValueCritic(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (512, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        width = CRITIC_DIM
        for output_width in hidden:
            layers.extend((nn.Linear(width, output_width), nn.ELU()))
            width = output_width
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        self.norm = RunningNorm(CRITIC_DIM)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(features))


class OdinDepthStudent(nn.Module):
    """1740→16 部署学生：204维本体历史与8×16×12深度记忆。"""

    def __init__(self, hidden: tuple[int, ...] = (512, 256)):
        super().__init__()
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ELU(),
            nn.Flatten(),
        )
        layers: list[nn.Module] = []
        width = POLICY_DIM + 32 * 8 * 6
        for output_width in hidden:
            layers.extend((nn.Linear(width, output_width), nn.ELU()))
            width = output_width
        layers.append(nn.Linear(width, ACTION_DIM))
        self.head = nn.Sequential(*layers)
        self.hidden = tuple(hidden)
        nn.init.uniform_(self.head[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != DEPLOY_INPUT_DIM:
            raise ValueError(
                f"Odin 学生输入必须为 {DEPLOY_INPUT_DIM} 维，实际 {observation.shape[-1]}"
            )
        policy = observation[..., :POLICY_DIM]
        depth = observation[..., POLICY_DIM:].reshape(-1, 8, 16, 12)
        features = torch.cat((policy, self.depth_encoder(depth)), dim=-1)
        return bounded_action(self.head(features))


def teacher_from_checkpoint(path: str, device: torch.device) -> DeterministicTeacher:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != TEACHER_FORMAT:
        raise ValueError("需要本分支 SuperMimic 世界模型教师 checkpoint")
    teacher = DeterministicTeacher(tuple(checkpoint["actor_hidden"])).to(device)
    teacher.load_state_dict(checkpoint["actor"])
    teacher.eval()
    return teacher


def student_from_checkpoint(path: str, device: torch.device) -> OdinDepthStudent:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format") != STUDENT_FORMAT:
        raise ValueError("需要本分支 1740→16 Odin 深度学生 checkpoint")
    student = OdinDepthStudent(tuple(checkpoint["hidden"])).to(device)
    student.load_state_dict(checkpoint["student"])
    student.eval()
    return student


def target_critic(critic: ValueCritic) -> ValueCritic:
    model = copy.deepcopy(critic)
    model.requires_grad_(False)
    return model
