"""按环境时间保留 RP_wd 转移序列，并提高接触/终止片段采样概率。"""

from __future__ import annotations

import torch


class TransitionReplay:
    """GPU环形回放；事件片段占比有限，避免稀有样本盖过平地分布。"""

    def __init__(self, capacity: int, num_envs: int, device: torch.device):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        self.shapes = {
            "state": 38, "action": 16, "previous_action": 16, "command": 6,
            "scan": 192, "pose": 3, "torque": 12, "contacts": 4, "wheel_force": 4,
            "next_state": 38, "pose_delta": 3, "next_scan": 192,
            "next_torque": 12, "next_contacts": 4, "next_wheel_force": 4,
            "reward": 1,
        }
        self.data = {
            name: torch.zeros(capacity, num_envs, width, device=device)
            for name, width in self.shapes.items()
        }
        self.terminated = torch.zeros(capacity, num_envs, 1, device=device)
        self.valid_dynamics = torch.zeros(capacity, num_envs, 1, device=device)
        self.event_mask = torch.zeros(capacity, num_envs, dtype=torch.bool, device=device)
        self.stamp = torch.full((capacity,), -1, dtype=torch.long, device=device)
        self.total_steps = 0

    def __len__(self) -> int:
        return min(self.total_steps, self.capacity)

    def add(self, transition: dict[str, torch.Tensor], terminated: torch.Tensor,
            valid_dynamics: torch.Tensor, event_mask: torch.Tensor) -> None:
        slot = self.total_steps % self.capacity
        for name, value in transition.items():
            self.data[name][slot].copy_(value.detach().reshape(self.num_envs, -1))
        self.terminated[slot].copy_(terminated.detach().reshape(self.num_envs, 1).float())
        self.valid_dynamics[slot].copy_(valid_dynamics.detach().reshape(self.num_envs, 1).float())
        self.event_mask[slot].copy_(event_mask.detach().reshape(self.num_envs).bool())
        self.stamp[slot] = self.total_steps
        self.total_steps += 1

    def _choose_starts(self, batch_size: int, window: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.total_steps < window:
            raise RuntimeError("回放尚未积累足够的连续模型窗口")
        oldest = max(0, self.total_steps - self.capacity)
        latest = self.total_steps - window
        count = batch_size
        active_events = int(self.event_mask.sum().item())
        event_count = min(int(batch_size * 0.25), active_events)
        uniform_count = count - event_count
        starts = torch.randint(oldest, latest + 1, (uniform_count,), device=self.device)
        env_ids = torch.randint(0, self.num_envs, (uniform_count,), device=self.device)
        if event_count:
            flat_events = self.event_mask.flatten().float()
            selected = torch.multinomial(flat_events, event_count, replacement=True)
            event_slots = torch.div(selected, self.num_envs, rounding_mode="floor")
            event_envs = selected.remainder(self.num_envs)
            event_steps = self.stamp[event_slots]
            offsets = torch.randint(0, window, (event_count,), device=self.device)
            event_starts = (event_steps - offsets).clamp(min=oldest, max=latest)
            starts = torch.cat((starts, event_starts))
            env_ids = torch.cat((env_ids, event_envs))
        order = torch.randperm(batch_size, device=self.device)
        return starts[order], env_ids[order]

    def sample_windows(self, batch_size: int, window: int) -> dict[str, torch.Tensor]:
        starts, env_ids = self._choose_starts(batch_size, window)
        stamps = starts[:, None] + torch.arange(window, device=self.device)[None, :]
        slots = stamps.remainder(self.capacity)
        result = {name: value[slots, env_ids[:, None]] for name, value in self.data.items()}
        result["terminated"] = self.terminated[slots, env_ids[:, None]]
        result["valid_dynamics"] = self.valid_dynamics[slots, env_ids[:, None]]
        result["sequence_valid"] = self.stamp[slots].eq(stamps).unsqueeze(-1).float()
        return result

    def sample_steps(self, batch_size: int) -> dict[str, torch.Tensor]:
        oldest = max(0, self.total_steps - self.capacity)
        stamps = torch.randint(oldest, self.total_steps, (batch_size,), device=self.device)
        env_ids = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        slots = stamps.remainder(self.capacity)
        result = {name: value[slots, env_ids] for name, value in self.data.items()}
        result["terminated"] = self.terminated[slots, env_ids]
        result["valid_dynamics"] = self.valid_dynamics[slots, env_ids]
        return result
