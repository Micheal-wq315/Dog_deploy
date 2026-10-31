"""RP_wd 的短时域世界模型训练、想象反传和确定性 Actor 更新。"""

from __future__ import annotations

from contextlib import contextmanager
import copy

import torch
import torch.nn.functional as F

from .model import (
    ACTION_LIMIT,
    CRITIC_DIM,
    DynamicsMember,
    DeterministicTeacher,
    TEACHER_FORMAT,
    ValueCritic,
    make_actor_features,
    make_critic_features,
)
from .replay import TransitionReplay


@contextmanager
def frozen_parameters(*modules: torch.nn.Module):
    """冻结网络权重，但仍让梯度经过模型输入和策略动作。"""
    parameters = [(parameter, parameter.requires_grad) for module in modules
                  for parameter in module.parameters()]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in parameters:
            parameter.requires_grad_(enabled)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    return (values * weight).sum() / (weight.sum() * values.shape[-1]).clamp_min(1.0)


class ModelBasedLearner:
    """Go2W式循环：真实交互→模型拟合→想象轨迹→Critic/Actor梯度更新。"""

    def __init__(self, args: object, device: torch.device, num_envs: int):
        self.args = args
        self.device = device
        self.models = torch.nn.ModuleList([
            DynamicsMember(tuple(args.wm_hidden)).to(device) for _ in range(args.ensemble)
        ])
        self.actor = DeterministicTeacher(tuple(args.actor_hidden)).to(device)
        self.critic = ValueCritic(tuple(args.critic_hidden)).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device).eval()
        self.target_critic.requires_grad_(False)
        self.replay = TransitionReplay(args.replay_steps, num_envs, device)
        self.model_optimizers = [torch.optim.Adam(model.parameters(), lr=args.wm_lr)
                                 for model in self.models]
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.policy_updates = 0

    def _update_norms(self, batch: dict[str, torch.Tensor]) -> None:
        valid = batch["sequence_valid"].squeeze(-1).bool() & batch["valid_dynamics"].squeeze(-1).bool()
        if not valid.any():
            return
        state = batch["state"][valid]
        action = batch["action"][valid]
        previous = batch["previous_action"][valid]
        command = batch["command"][valid]
        scan = batch["scan"][valid]
        target = DynamicsMember.continuous_target({key: value[valid] for key, value in batch.items()
                                                   if key in (
                                                       "reward", "next_state", "state", "pose_delta",
                                                       "next_scan", "scan", "next_torque", "next_wheel_force")})
        inputs = torch.cat((state, action, previous, command, scan), -1)
        for model in self.models:
            model.input_norm.update(inputs)
            model.output_norm.update(target)

    def _member_loss(self, model: DynamicsMember,
                     batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size, window = batch["action"].shape[:2]
        state = batch["state"][:, 0]
        scan = batch["scan"][:, 0]
        active = batch["sequence_valid"][:, 0, 0].bool()
        losses: dict[str, torch.Tensor] = {}
        accum: dict[str, list[torch.Tensor]] = {name: [] for name in
                                                ("state", "pose", "scan", "torque", "contact",
                                                 "force", "reward", "terminal")}
        count: dict[str, list[torch.Tensor]] = {name: [] for name in accum}

        for step in range(window):
            sequence_ok = batch["sequence_valid"][:, step, 0].bool()
            active = active & sequence_ok
            if not active.any():
                break
            current = model(
                state,
                batch["action"][:, step],
                batch["previous_action"][:, step],
                batch["command"][:, step],
                scan,
            )
            dyn_mask = active & batch["valid_dynamics"][:, step, 0].bool()
            term_mask = active

            if dyn_mask.any():
                norm = model.output_norm.scale
                state_pred = (current["state"] - state) / norm[:38]
                state_target = (batch["next_state"][:, step] - state) / norm[:38]
                pose_offset = 38
                pose_pred = current["pose_delta"] / norm[pose_offset:pose_offset + 3]
                pose_target = batch["pose_delta"][:, step] / norm[pose_offset:pose_offset + 3]
                scan_offset = 41
                scan_pred = (current["scan"] - scan) / norm[scan_offset:scan_offset + 192]
                scan_target = (batch["next_scan"][:, step] - scan) / norm[scan_offset:scan_offset + 192]
                torque_offset = 233
                torque_pred = current["torque"] / norm[torque_offset:torque_offset + 12]
                torque_target = batch["next_torque"][:, step] / norm[torque_offset:torque_offset + 12]
                force_offset = 245
                force_pred = current["wheel_force"] / norm[force_offset:force_offset + 4]
                force_target = batch["next_wheel_force"][:, step] / norm[force_offset:force_offset + 4]
                for name, prediction, target_value in (
                    ("state", state_pred, state_target), ("pose", pose_pred, pose_target),
                    ("scan", scan_pred, scan_target), ("torque", torque_pred, torque_target),
                    ("force", force_pred, force_target),
                ):
                    element = F.smooth_l1_loss(prediction, target_value, reduction="none")
                    accum[name].append(_masked_mean(element, dyn_mask))
                    count[name].append(dyn_mask.float().mean())

                contact_loss = F.binary_cross_entropy_with_logits(
                    current["contacts_logit"], batch["next_contacts"][:, step], reduction="none"
                )
                accum["contact"].append(_masked_mean(contact_loss, dyn_mask))
                count["contact"].append(dyn_mask.float().mean())

                state = torch.where(dyn_mask.unsqueeze(-1), current["state"], state)
                scan = torch.where(dyn_mask.unsqueeze(-1), current["scan"], scan)

            reward_target = batch["reward"][:, step] / 5.0
            reward_loss = F.smooth_l1_loss(current["reward_scaled"], reward_target, reduction="none")
            accum["reward"].append(_masked_mean(reward_loss, term_mask))
            count["reward"].append(term_mask.float().mean())
            terminal_loss = F.binary_cross_entropy_with_logits(
                current["terminated_logit"], batch["terminated"][:, step], reduction="none"
            )
            accum["terminal"].append(_masked_mean(terminal_loss, term_mask))
            count["terminal"].append(term_mask.float().mean())

            active = dyn_mask

        coefficients = {"state": 1.0, "pose": 0.5, "scan": 0.35, "torque": 0.1,
                        "contact": 0.2, "force": 0.1, "reward": 1.0, "terminal": 0.2}
        total = next(model.parameters()).sum() * 0.0
        metrics: dict[str, float] = {}
        for name, values in accum.items():
            if values:
                mean_loss = torch.stack(values).mean()
                total = total + coefficients[name] * mean_loss
                metrics[name] = float(mean_loss.detach())
        return total, metrics

    def fit_world_model(self) -> dict[str, float]:
        batch = self.replay.sample_windows(self.args.wm_batch, self.args.wm_window)
        self._update_norms(batch)
        combined: dict[str, list[float]] = {}
        for model, optimizer in zip(self.models, self.model_optimizers):
            bootstrap_size = max(1, int(batch["action"].shape[0] * 0.8))
            bootstrap = torch.randperm(batch["action"].shape[0], device=self.device)[:bootstrap_size]
            member_batch = {key: value[bootstrap] for key, value in batch.items()}
            loss, metrics = self._member_loss(model, member_batch)
            optimizer.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            for name, value in metrics.items():
                combined.setdefault(name, []).append(value)
        return {f"wm_{name}": sum(values) / len(values) for name, values in combined.items()}

    @staticmethod
    def _ensemble_predict(models: torch.nn.ModuleList, state: torch.Tensor,
                          action: torch.Tensor, previous: torch.Tensor,
                          command: torch.Tensor, scan: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        outputs = [model(state, action, previous, command, scan) for model in models]
        mean: dict[str, torch.Tensor] = {}
        for key in outputs[0]:
            values = torch.stack([item[key] for item in outputs])
            if key == "contacts_logit" or key == "terminated_logit":
                mean[key] = values.mean(0)
            else:
                mean[key] = values.mean(0)
        state_scales = models[0].output_norm.scale
        state_spread = torch.stack([item["state"] for item in outputs]).std(0, unbiased=False)
        scan_spread = torch.stack([item["scan"] for item in outputs]).std(0, unbiased=False)
        reward_spread = torch.stack([item["reward_scaled"] for item in outputs]).std(0, unbiased=False)
        uncertainty = (
            (state_spread / state_scales[:38]).mean(-1)
            + (scan_spread / state_scales[41:233]).mean(-1)
            + reward_spread.squeeze(-1)
        )
        return mean, uncertainty

    def _critic_input(self, state: torch.Tensor, command: torch.Tensor, previous: torch.Tensor,
                      scan: torch.Tensor, torque: torch.Tensor, contacts: torch.Tensor,
                      wheel_force: torch.Tensor) -> torch.Tensor:
        actor_features = make_actor_features(state, command, previous, scan, torque, contacts)
        return make_critic_features(actor_features, wheel_force)

    def imagine_and_improve(self) -> dict[str, float]:
        args = self.args
        seeds = self.replay.sample_steps(args.imag_batch)
        state = seeds["state"].detach()
        previous = seeds["previous_action"].detach()
        command = seeds["command"].detach()
        scan = seeds["scan"].detach()
        torque = seeds["torque"].detach()
        contacts = seeds["contacts"].detach()
        wheel_force = seeds["wheel_force"].detach()

        with torch.no_grad():
            initial_value_input = self._critic_input(
                state, command, previous, scan, torque, contacts, wheel_force
            )
            self.critic.norm.update(initial_value_input)
            self.target_critic.norm.load_state_dict(self.critic.norm.state_dict())

        rewards: list[torch.Tensor] = []
        continuations: list[torch.Tensor] = []
        critic_inputs: list[torch.Tensor] = []
        next_critic_inputs: list[torch.Tensor] = []
        terminal_critic_input: torch.Tensor | None = None
        with frozen_parameters(self.models, self.target_critic):
            discount = 1.0
            alive = torch.ones(args.imag_batch, 1, device=self.device)
            objective = torch.zeros_like(alive)
            for _ in range(args.horizon):
                current_critic_input = self._critic_input(
                    state, command, previous, scan, torque, contacts, wheel_force
                )
                critic_inputs.append(current_critic_input.detach())
                actor_features = make_actor_features(state, command, previous, scan, torque, contacts)
                action = self.actor(actor_features)
                prediction, uncertainty = self._ensemble_predict(
                    self.models, state, action, previous, command, scan
                )
                reward = (5.0 * prediction["reward_scaled"]).clamp(
                    -args.imagination_reward_clip, args.imagination_reward_clip
                )
                reward = reward - args.uncertainty_coef * uncertainty.unsqueeze(-1)
                continuation = 1.0 - prediction["terminated_logit"].sigmoid()
                objective = objective + discount * alive * reward
                alive = alive * continuation
                rewards.append(reward)
                continuations.append(continuation)

                state = prediction["state"]
                scan = prediction["scan"]
                torque = prediction["torque"]
                contacts = prediction["contacts_logit"].sigmoid()
                wheel_force = prediction["wheel_force"]
                previous = action
                discount *= args.gamma
                terminal_critic_input = self._critic_input(
                    state, command, previous, scan, torque, contacts, wheel_force
                )
                next_critic_inputs.append(terminal_critic_input.detach())

            if terminal_critic_input is None:
                raise RuntimeError("想象轨迹至少需要一个策略步")
            final_value = self.target_critic(terminal_critic_input)
            objective = objective + discount * alive * final_value
            actor_loss = -objective.mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(actor_loss):
                return {"actor_loss": float("nan"), "critic_loss": float("nan")}
            actor_loss.backward()
            actor_grad = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()

        with torch.no_grad():
            reward_tensor = torch.stack(rewards)
            continuation_tensor = torch.stack(continuations)
            next_values = self.target_critic(torch.stack(next_critic_inputs)).squeeze(-1)
            carry = next_values[-1]
            targets = []
            for index in reversed(range(args.horizon)):
                carry = reward_tensor[index].squeeze(-1) + args.gamma * continuation_tensor[index].squeeze(-1) * (
                    (1.0 - args.lam) * next_values[index] + args.lam * carry
                )
                targets.append(carry)
            targets = torch.stack(targets[::-1]).detach()

        value_input = torch.stack(critic_inputs).flatten(0, 1)
        target = targets.flatten()
        with torch.no_grad():
            self.critic.norm.update(value_input)
        value = self.critic(value_input).squeeze(-1)
        critic_loss = F.mse_loss(value, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, 0.02)
            self.target_critic.norm.load_state_dict(self.critic.norm.state_dict())
        self.policy_updates += 1
        return {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "actor_grad_norm": float(actor_grad.detach()),
            "critic_grad_norm": float(critic_grad.detach()),
            "imagined_reward": float(reward_tensor.mean().detach()),
        }

    def state_dict(self, step: int) -> dict[str, object]:
        return {
            "format": "rp-wd-supermimic-learner-v1",
            "step": step,
            "models": [model.state_dict() for model in self.models],
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "model_optimizers": [optimizer.state_dict() for optimizer in self.model_optimizers],
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "policy_updates": self.policy_updates,
            "actor_hidden": self.actor.hidden,
            "config": vars(self.args).copy(),
        }

    def load_state_dict(self, checkpoint: dict[str, object]) -> int:
        """恢复教师参数和优化器；环境与回放池由训练入口重新初始化。"""
        if checkpoint.get("format") != TEACHER_FORMAT:
            raise ValueError("resume checkpoint不是本分支的SuperMimic教师")
        model_states = checkpoint["models"]
        optimizer_states = checkpoint["model_optimizers"]
        if len(model_states) != len(self.models) or len(optimizer_states) != len(self.model_optimizers):
            raise ValueError("resume checkpoint的世界模型集成数量与当前配置不一致")
        for model, state in zip(self.models, model_states):
            model.load_state_dict(state)
        for optimizer, state in zip(self.model_optimizers, optimizer_states):
            optimizer.load_state_dict(state)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.policy_updates = int(checkpoint.get("policy_updates", 0))
        return int(checkpoint.get("step", 0))
