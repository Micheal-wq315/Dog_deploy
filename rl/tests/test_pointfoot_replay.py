"""点足回放和模型训练的纯CPU行为测试；不加载Isaac或创建CUDA张量。"""

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rp_wd_rl.supermimic.pointfoot import (  # noqa: E402
    ACTION_DIM, FORMAT, POLICY_DIM, STATE_DIM, PointfootDynamics, PointfootReplay,
    PointfootSuperMimicLearner, PointfootWindowArchive,
)
from rp_wd_rl.supermimic.pointfoot_terrain import (  # noqa: E402
    PointfootTaskRewards, PointfootTerrainField, advance_pose,
)


def transition(step, envs=1, encoded=True):
    data = {key: torch.zeros(envs, width) for key, width in PointfootReplay.SHAPES.items()}
    data["state"][:, 0] = .34
    data["state"][:, 3] = -1.
    data["state"][:, 4] = .3
    if encoded:
        data["state"][:, 10] = step * envs + torch.arange(envs)
    data["next_state"].copy_(data["state"])
    data["world_pose"][:, 0] = step * .006
    data["pose"][:, 0] = .006
    data["next_contacts"][:] = 1.
    data["next_force"][:] = .2
    data["scan"] = torch.full((envs, 273), .34)
    data["next_scan"] = data["scan"].clone()
    return data


def add(replay, step, event=None, success=None, terminated=None, valid=None, data=None):
    envs = replay.num_envs
    replay.add(
        transition(step, envs) if data is None else data,
        torch.zeros(envs, dtype=torch.bool) if terminated is None else terminated,
        torch.ones(envs, dtype=torch.bool) if valid is None else valid,
        torch.zeros(envs, dtype=torch.bool) if event is None else event,
        success_event=success,
    )


def learner(archive=0, success_archive=0, envs=1, capacity=8, profile="full"):
    args = SimpleNamespace(
        buffer_len=capacity, wm_window=8, wm_batch=4, imag_batch=2, horizon=32,
        wm_hidden=(16, 16), actor_hidden=(16,), critic_hidden=(16,), lr=2e-4,
        actor_lr_scale=.3, gamma=.99, lam=.95, critic_epochs=1, critic_minibatches=2,
    )
    # 故意保留完全没有新参数的旧args路径。
    if archive or success_archive:
        args.archive_windows, args.success_archive_windows = archive, success_archive
    names = [f"{leg}_{joint}_JOINT" for leg in ("LF", "RF", "LB", "RB")
             for joint in ("ABAD", "HIP", "KENN")]
    rewards = PointfootTaskRewards(
        names, torch.tensor([0., .8, -1.6] * 4), torch.tensor([[-3., 3.]] * 12), profile=profile,
    )
    field = PointfootTerrainField(torch.zeros(20, 20), (-2., -2.), .25)
    return PointfootSuperMimicLearner(args, torch.device("cpu"), envs, field, rewards)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_scans_are_temporary_and_stats_still_use_them(self):
        replay = PointfootReplay(8, 2, torch.device("cpu"))
        data = transition(0, 2)
        data["state"].requires_grad_(True)
        data["extra_temporary"] = torch.ones(2, 100)
        add(replay, 0, data=data)
        self.assertNotIn("scan", replay.data)
        self.assertNotIn("next_scan", replay.data)
        self.assertNotIn("extra_temporary", replay.data)
        self.assertFalse(replay.data["state"].requires_grad)
        expected_width = sum(PointfootReplay.SHAPES.values())
        self.assertEqual(sum(t.numel() for t in replay.data.values()), 8 * 2 * expected_width)
        model = PointfootDynamics((16,))
        model.update_stats(data, torch.ones(2, dtype=torch.bool))
        torch.testing.assert_close(model.input_norm.mean[-273:], torch.full((273,), .34),
                                   atol=2e-5, rtol=0)

    def test_actor_seeds_uniform_over_live_time_and_environments(self):
        replay = PointfootReplay(8, 2, torch.device("cpu"), 32, 32, 4)
        for step in range(24):
            tag = torch.tensor([step in (0, 23), False])
            add(replay, step, event=tag, success=tag.flip(0))
        sample = replay.sample_steps(24000)
        ids = sample["state"][:, 10].long()
        self.assertGreaterEqual(int(ids.min()), 32)
        self.assertLessEqual(int(ids.max()), 47)
        counts = torch.bincount(ids - 32, minlength=16)
        self.assertTrue(((counts - 1500).abs() < 180).all(), counts)
        # 有意把在线末步标成事件；原先25%事件偏置会使此检验明显失败。
        self.assertGreater(replay.archive.size, 0)

    def test_wm_half_online_quarter_event_quarter_success_and_persistence(self):
        replay = PointfootReplay(16, 2, torch.device("cpu"), 32, 32, 4)
        for step in range(12):
            add(replay, step, event=torch.tensor([True, False]),
                success=torch.tensor([False, True]))
        for step in range(12, 48):
            add(replay, step)
        sample = replay.sample_windows(40, 4)
        ids = sample["state"][:, 0, 10].long()
        old = ids < 24
        self.assertEqual(int(old.sum()), 20)
        self.assertEqual(int((old & (ids % 2 == 0)).sum()), 10)
        self.assertEqual(int((old & (ids % 2 == 1)).sum()), 10)
        self.assertEqual(replay.last_sample_counts, {
            "wm_online_windows": 20, "wm_event_windows": 10, "wm_success_windows": 10,
        })
        self.assertNotIn("history", sample)
        self.assertNotIn("scan", sample)
        self.assertNotIn("bias", replay.archive.data)
        uniform = replay.sample_windows(100, 4, prioritized=False)
        self.assertTrue((uniform["state"][:, 0, 10] >= 64).all())

    def test_sparse_archive_falls_back_to_online(self):
        replay = PointfootReplay(8, 1, torch.device("cpu"), 16, 8, 4)
        for step in range(8):
            add(replay, step, event=torch.tensor([step == 2]))
        replay.sample_windows(20, 4)
        self.assertEqual(replay.last_sample_counts["wm_online_windows"], 19)
        self.assertEqual(replay.last_sample_counts["wm_event_windows"], 1)
        self.assertEqual(replay.last_sample_counts["wm_success_windows"], 0)
        self.assertEqual(replay.success_archive.storage_bytes, 0)

    def test_archive_ring_matches_sequential_insertion_even_oversized(self):
        archive = PointfootWindowArchive(3, 2)
        expected = torch.zeros(3)
        cursor = total = 0
        pointer = None
        for incoming in (2, 7, 1, 5, 8):
            ids = torch.arange(total, total + incoming).float()
            for number in ids:
                expected[cursor] = number
                cursor = (cursor + 1) % 3
            archive.add({"state": ids[:, None, None].expand(-1, 2, STATE_DIM)})
            torch.testing.assert_close(archive.data["state"][:, 0, 0], expected)
            self.assertEqual(archive.cursor, cursor)
            self.assertEqual(archive.size, min(3, total + incoming))
            if pointer is not None:
                self.assertEqual(archive.data["state"].data_ptr(), pointer)
            pointer = archive.data["state"].data_ptr()
            total += incoming
        sampled = archive.sample(100, 2)["state"][:, 0, 0]
        self.assertTrue(torch.isin(sampled, expected).all())

    def test_reset_prefix_and_terminal_labels_survive_archive_and_wrap(self):
        replay = PointfootReplay(8, 1, torch.device("cpu"), 16, 8, 8)
        for step in range(8):
            add(replay, step, terminated=torch.tensor([step == 2]),
                valid=torch.tensor([step not in (2, 5)]),
                event=torch.tensor([step in (2, 7)]), success=torch.tensor([step == 5]))
        sample = replay.sample_windows(1, 8, prioritized=False)
        self.assertEqual(sample["sequence_valid"][0, :, 0].tolist(), [1, 1, 1, 0, 0, 0, 0, 0])
        event = replay.archive.data
        self.assertEqual(event["terminated"][0, :, 0].tolist(), [0, 0, 1, 0, 0, 0, 0, 0])
        self.assertEqual(event["valid"][0, :, 0].tolist(), [1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(event["state"][1, :2, 10].tolist(), [6, 7])
        self.assertEqual(event["sequence_valid"][1, :, 0].tolist(), [1, 1, 0, 0, 0, 0, 0, 0])
        success = replay.success_archive.data
        self.assertEqual(success["state"][0, :3, 10].tolist(), [3, 4, 5])
        self.assertEqual(success["terminated"][0].sum(), 0)  # 超时不变成摔倒。
        saved = {key: value.clone() for key, value in event.items()}
        for step in range(8, 32):
            add(replay, step)
        for key in saved:
            torch.testing.assert_close(event[key], saved[key])

    def test_nonfinite_targets_break_prefix_but_legal_terminal_input_is_kept(self):
        replay = PointfootReplay(8, 1, torch.device("cpu"), 8, 8, 8)
        for step in range(8):
            data = transition(step)
            if step == 2:
                data["next_state"][:] = float("nan")
            add(replay, step, data=data, event=torch.tensor([step == 2]),
                terminated=torch.tensor([step == 2]))
        sample = replay.sample_windows(1, 8, prioritized=False)
        self.assertEqual(sample["sequence_valid"][0, :, 0].tolist(), [1, 1, 1, 0, 0, 0, 0, 0])
        self.assertEqual(sample["valid"][0, 2, 0], 0)
        self.assertEqual(replay.archive.data["terminated"][0, 2, 0], 1)
        # 即使调用方误传valid=True，也不能把NaN写进归一化统计。
        model = PointfootDynamics((16,))
        invalid = transition(0)
        invalid["next_state"][:] = float("nan")
        model.update_stats(invalid, torch.ones(1, dtype=torch.bool))
        self.assertAlmostEqual(float(model.input_norm.count), 1e-4, places=7)

    def test_invalid_model_input_is_not_used_for_terminal_classification(self):
        replay = PointfootReplay(8, 1, torch.device("cpu"))
        for step in range(8):
            data = transition(step)
            if step == 2:
                data["context"][:] = float("inf")
            add(replay, step, data=data)
        sample = replay.sample_windows(1, 8)
        self.assertEqual(sample["sequence_valid"][0, :, 0].tolist(), [1, 1, 0, 0, 0, 0, 0, 0])

    def test_empty_and_over_capacity_window_fail_clearly(self):
        replay = PointfootReplay(3, 1, torch.device("cpu"))
        with self.assertRaises(RuntimeError):
            replay.sample_steps(1)
        for step in range(8):
            add(replay, step)
        with self.assertRaises(ValueError):
            replay.sample_windows(1, 4)
        with self.assertRaises(ValueError):
            replay.sample_steps(0)
        sample = replay.sample_windows(1, 3)
        self.assertEqual(sample["state"][0, :, 10].tolist(), [5, 6, 7])


class ExactDynamics(nn.Module):
    """可核算的测试动力学，用于分辨自回归误差与逐帧教师强制误差。"""

    def __init__(self, extra_velocity=0.):
        super().__init__()
        self.extra_velocity = extra_velocity
        self.grad_enabled = []

    def forward(self, state, action, previous, context, scan):
        self.grad_enabled.append(torch.is_grad_enabled())
        delta = torch.zeros_like(state)
        delta[:, 0], delta[:, 1], delta[:, 4], delta[:, 9] = .001, .01, .02, .01
        delta[:, 10:22] = .001
        delta[:, 4] += self.extra_velocity
        pose = state.new_tensor([.006, 0., .005]).expand(len(state), -1)
        force = state.new_full((len(state), 4), .2)
        return {
            "state": state + delta, "pose_delta": pose, "force": force,
            "continuous": torch.cat((delta, pose, force), -1),
            "contact_logit": state.new_full((len(state), 4), 100.),
            "terminated_logit": state.new_full((len(state), 1), -100.),
            "collision_logit": state.new_full((len(state), 1), -100.),
        }


def fill_exact(subject):
    exact = ExactDynamics()
    state = transition(0, encoded=False)["state"]
    pose = torch.tensor([[0., 0., 3.13]])
    for step in range(8):
        data = transition(step, encoded=False)
        data["state"], data["world_pose"] = state.clone(), pose.clone()
        data["command"][:] = torch.tensor([5., -5., 2.])
        predicted = exact(state, data["action"], data["previous"], data["context"], data["scan"])
        data["next_state"], data["pose"] = predicted["state"], predicted["pose_delta"]
        add(subject.replay, step, data=data)
        state = predicted["state"]
        pose = advance_pose(pose, predicted["pose_delta"])
        pose[:, 2] = torch.atan2(pose[:, 2].sin(), pose[:, 2].cos())


class LearnerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_actual_full_and_legacy_rewards_32step_gradients_and_checkpoint(self):
        for profile in ("legacy", "full"):
            with self.subTest(profile=profile):
                subject = learner(4, 2, envs=2, profile=profile)
                for step in range(8):
                    data = transition(step, 2, encoded=False)
                    valid = torch.tensor([step != 3, True])
                    if step == 3:
                        data["next_state"][0] = float("nan")
                    subject.model.update_stats(data, valid)
                    add(subject.replay, step, data=data, valid=valid, terminated=~valid,
                        event=torch.ones(2, dtype=torch.bool), success=valid)
                wm = subject.fit_world_model()
                before = [p.detach().clone() for p in subject.actor.parameters()]
                received = []
                original = subject.task_rewards.reward

                def reward(*args, **kwargs):
                    received.append(kwargs)
                    return original(*args, **kwargs)

                with patch.object(subject.task_rewards, "reward", side_effect=reward):
                    policy = subject.improve()
                self.assertEqual(len(received), 32)
                self.assertTrue(all(x["contacts"].shape == (2, 4)
                                    and x["force"].shape == (2, 4) for x in received))
                self.assertTrue(all(math.isfinite(v) for metrics in (wm, policy)
                                    for v in metrics.values()), (wm, policy))
                self.assertEqual(wm["wm_update_applied"], 1)
                self.assertGreater(policy["actor_grad_norm"], 0)
                self.assertGreater(policy["critic_grad_norm"], 0)
                self.assertTrue(any(not torch.equal(old, new) for old, new
                                    in zip(before, subject.actor.parameters())))
                self.assertTrue(all(p.grad is None and p.requires_grad
                                    for p in subject.model.parameters()))
                self.assertTrue(all(p.grad is None for p in subject.target_critic.parameters()))
                self.assertEqual(POLICY_DIM, 225)
                self.assertEqual(ACTION_DIM, 12)
                payload = subject.state_dict(16)
                self.assertEqual(payload["format"], FORMAT)
                self.assertEqual(FORMAT, "rp-wd-pointfoot-supermimic-v2")
                self.assertFalse(any("replay" in k or "archive" in k for k in payload))
                restored = learner(envs=2, profile=profile)
                self.assertEqual(restored.load_state_dict(payload), 16)
                self.assertEqual(restored.replay.size, 0)
                self.assertEqual(restored.replay.archive.size, 0)
                stats = subject.replay_stats()
                self.assertEqual(stats["archive_windows"], 4)
                self.assertEqual(stats["success_archive_windows"], 2)

    def test_fit_stops_at_first_terminal_and_ignores_nan_targets_after_reset(self):
        subject = learner()
        for step in range(8):
            data = transition(step, encoded=False)
            if step >= 2:
                data["next_state"][:] = float("nan")
            add(subject.replay, step, data=data, terminated=torch.tensor([step == 2]))
        with patch.object(subject.model, "forward", wraps=subject.model.forward) as calls:
            metrics = subject.fit_world_model()
        self.assertEqual(calls.call_count, 3)
        self.assertTrue(all(math.isfinite(v) for v in metrics.values()), metrics)
        self.assertEqual(metrics["wm_valid_fraction"], .25)
        self.assertAlmostEqual(metrics["wm_terminal_rate"], 1 / 3, places=6)
        self.assertGreater(abs(float(subject.model.event_head.bias.grad[-1])), 0)

    def test_survival_reads_profile_and_terminal_head_but_not_leg_collision(self):
        subject = learner()
        angle = math.radians(40)
        subject.task_rewards.fall_angle, subject.task_rewards.hard_min_height = angle, .23
        state = transition(0, encoded=False)["state"]
        state[:, 0], state[:, 1], state[:, 3] = .23, math.sin(angle), -math.cos(angle)
        pred = {"state": state, "terminated_logit": torch.tensor([[-100.]]),
                "collision_logit": torch.tensor([[100.]])}
        self.assertAlmostEqual(float(subject._imagined_continuation(pred)), .25, places=6)
        pred["collision_logit"] *= -1
        self.assertAlmostEqual(float(subject._imagined_continuation(pred)), .25, places=6)
        pred["terminated_logit"] *= -1
        self.assertLess(float(subject._imagined_continuation(pred)), 1e-8)

    def test_full_survival_uses_foot_support_when_body_crosses_twenty_cm_edge(self):
        scan = torch.full((1, 273), .30)
        contacts, force = torch.ones(1, 4), torch.full((1, 4), .2)
        for profile in ("full", "legacy"):
            with self.subTest(profile=profile):
                subject = learner(profile=profile)
                state = transition(0, encoded=False)["state"]
                state[:, 0] = .30
                pred = {"state": state, "terminated_logit": torch.tensor([[-100.]])}
                before = subject._imagined_continuation(pred, scan, contacts, force)
                # 机身正下方先越过20cm边缘，脚仍支撑在原踏面。
                state[:, 0] -= .20
                after = subject._imagined_continuation(pred, scan, contacts, force)
                if profile == "full":
                    torch.testing.assert_close(before, after)
                    self.assertGreater(float(after), .99)
                    self.assertLess(float(subject._imagined_continuation(pred)), .13)
                    # 零接触/载荷时support_clearance回退body净空，不能凭空生存。
                    no_support = subject._imagined_continuation(
                        pred, scan, torch.zeros_like(contacts), force,
                    )
                    self.assertLess(float(no_support), .13)
                else:
                    self.assertGreater(float(before), .99)
                    self.assertLess(float(after), .01)

    def test_prediction_metrics_are_autoregressive_no_grad_and_use_real_targets(self):
        subject = learner()
        fill_exact(subject)
        subject.model = ExactDynamics()
        metrics = subject.prediction_metrics(batch_size=3)
        self.assertEqual(metrics["replay_pred_8step_transitions"], 24)
        self.assertEqual(metrics["replay_pred_8step_final_windows"], 3)
        for key, value in metrics.items():
            if "mae" in key or "brier" in key:
                self.assertLess(value, 2e-6, (key, value))
        self.assertFalse(any(subject.model.grad_enabled))
        subject.model = ExactDynamics(extra_velocity=.05)
        metrics = subject.prediction_metrics(batch_size=2)
        self.assertAlmostEqual(metrics["replay_pred_8step_rollout_lin_vel_mae_mps"], .075, places=6)
        self.assertAlmostEqual(metrics["replay_pred_8step_final_lin_vel_mae_mps"], .4 / 3, places=6)

    def test_prediction_metrics_ignore_reset_and_nan_but_keep_terminal_score(self):
        subject = learner()
        fill_exact(subject)
        subject.model = ExactDynamics()
        subject.replay.data["next_state"][2] = float("nan")
        subject.replay.terminated[2] = 1.
        metrics = subject.prediction_metrics(batch_size=2)
        self.assertEqual(len(subject.model.grad_enabled), 3)
        self.assertEqual(metrics["replay_pred_8step_transitions"], 4)
        self.assertEqual(metrics["replay_pred_8step_final_windows"], 0)
        self.assertAlmostEqual(metrics["replay_pred_8step_terminal_brier"], 1 / 3, places=6)
        self.assertTrue(all(math.isfinite(v) for v in metrics.values()))

    def test_prediction_before_eight_steps_returns_counts_without_fake_error(self):
        subject = learner()
        self.assertEqual(subject.prediction_metrics(batch_size=2), {
            "replay_pred_8step_transitions": 0., "replay_pred_8step_final_windows": 0.,
        })

    def test_model_freeze_is_restored_when_reward_raises(self):
        subject = learner()
        for step in range(8):
            add(subject.replay, step, data=transition(step, encoded=False))
        with patch.object(subject.task_rewards, "reward", side_effect=RuntimeError("奖励异常")):
            with self.assertRaisesRegex(RuntimeError, "奖励异常"):
                subject.improve()
        self.assertTrue(all(p.requires_grad for p in subject.model.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
