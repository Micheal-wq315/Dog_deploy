"""台阶事件奖励、课程门槛与首帧评测采样的CPU回归。"""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rp_wd_rl.stair_logic import PerformanceGate, StairEvents  # noqa: E402
from transition_capture import capture_before_reset  # noqa: E402


class StairRewardTests(unittest.TestCase):
    def setUp(self):
        self.events = StairEvents(1, "cpu")
        self.ground = torch.zeros(1, 4)
        self.ahead = torch.zeros(1, 4)
        self.x = torch.zeros(1, 4)
        self.bottom = torch.zeros(1, 4)
        self.vz = torch.zeros(1, 4)
        self.contact = torch.ones(1, 4, dtype=torch.bool)
        self.kw = dict(forward_speed=torch.tensor([.3]), active=torch.tensor([True]),
                       safe=torch.tensor([True]), alive=torch.tensor([True]),
                       impact_ok=torch.ones(1, 4, dtype=torch.bool))
        self.events.update(ground=self.ground, ahead=self.ahead, wheel_x=self.x, bottom=self.bottom,
                           vertical_speed=self.vz, contact=self.contact, **self.kw)

    def step(self):
        return self.events.update(ground=self.ground, ahead=self.ahead, wheel_x=self.x, bottom=self.bottom,
                                  vertical_speed=self.vz, contact=self.contact, **self.kw)

    def test_lift_delta_crossing_and_landing_are_separate_single_events(self):
        self.ahead[0, 0] = .06
        self.step()  # 捕获台阶目标并进入抬升状态
        self.contact[0, 0] = False
        self.vz[0, 0] = .2
        self.bottom[0, 0] = .02
        first = self.step()["clearance"].item()
        self.vz.zero_()
        held = self.step()["clearance"].item()
        self.assertGreater(first, 0.)
        self.assertEqual(held, 0.)  # 悬轮保持不反复领奖

        self.bottom[0, 0] = .06
        self.ground[0, 0] = .06
        self.ahead[0, 0] = .06
        self.x[0, 0] = .10
        self.assertEqual(self.step()["crossing"].item(), .25)
        self.assertEqual(self.step()["crossing"].item(), 0.)
        self.contact[0, 0] = True
        self.assertEqual(self.step()["landing"].item(), 0.)
        self.assertEqual(self.step()["landing"].item(), 0.)
        self.assertEqual(self.step()["landing"].item(), .25)
        self.assertEqual(self.step()["landing"].item(), 0.)

    def test_stationary_lift_and_hard_landing_do_not_get_success_rewards(self):
        self.events.reset(torch.tensor([0]))
        self.events.update(ground=self.ground, ahead=self.ground, wheel_x=self.x, bottom=self.bottom,
                           vertical_speed=self.vz, contact=self.contact, **self.kw)
        self.kw["forward_speed"][:] = 0.
        self.ahead[0, 0] = .06
        self.step()
        self.contact[0, 0] = False
        self.bottom[0, 0] = .06
        self.assertEqual(self.step()["clearance"].item(), 0.)

        self.kw["forward_speed"][:] = .3
        self.kw["impact_ok"][0, 0] = False
        self.contact[0, 0] = True
        self.ground[0, 0] = .06
        self.x[0, 0] = .1
        result = self.step()
        self.assertEqual(result["crossing"].item(), 0.)
        for _ in range(4):
            result = self.step()
        self.assertEqual(result["landing"].item(), 0.)

    def test_partial_reset_preserves_other_environments(self):
        state = StairEvents(2, "cpu")
        state.valid[:] = True
        state.armed[1, 2] = True
        state.peak_fraction[1, 2] = .4
        state.reset(torch.tensor([0]))
        self.assertFalse(state.valid[0])
        self.assertTrue(state.armed[1, 2])
        self.assertAlmostEqual(state.peak_fraction[1, 2].item(), .4)


class CourseGateTests(unittest.TestCase):
    def test_performance_gate_requires_full_clean_window_and_restores(self):
        gate = PerformanceGate("cpu", groups=2, window=4)
        groups = torch.zeros(3, dtype=torch.long)
        levels = torch.zeros(3, dtype=torch.long)
        success = torch.ones(3, dtype=torch.bool)
        clean = torch.ones(3, dtype=torch.bool)
        error = torch.full((3,), .1)
        gate.record(groups, levels, success, clean, error, time_cap=1)
        self.assertEqual(gate.cap.tolist(), [0, 0])
        gate.record(groups[:1], levels[:1], success[:1], clean[:1], error[:1], time_cap=1)
        self.assertEqual(gate.cap.tolist(), [1, 0])
        clone = PerformanceGate("cpu", groups=2, window=4)
        clone.load_state_dict(gate.state_dict())
        self.assertEqual(clone.cap.tolist(), [1, 0])

    def test_failed_or_unclean_window_does_not_advance(self):
        gate = PerformanceGate("cpu", groups=1, window=4)
        groups = torch.zeros(4, dtype=torch.long)
        levels = torch.zeros(4, dtype=torch.long)
        gate.record(groups, levels, torch.tensor([1, 1, 1, 0], dtype=torch.bool),
                    torch.ones(4, dtype=torch.bool), torch.full((4,), .05), time_cap=1)
        self.assertEqual(gate.cap.tolist(), [0])


class TransitionCaptureTests(unittest.TestCase):
    def test_captures_first_physics_transition_before_reset(self):
        class Manager:
            terminated = False

            def compute(self):
                self.terminated = True
                return "done"

        manager = Manager()
        snapshots = []
        with capture_before_reset(manager, lambda: snapshots.append(manager.terminated) or manager.terminated) as seen:
            self.assertEqual(manager.compute(), "done")
            self.assertTrue(seen[0])
        self.assertEqual(snapshots, [True])
        self.assertTrue(manager.terminated)

    def test_restores_compute_after_exception(self):
        class Manager:
            def compute(self):
                raise RuntimeError("boom")

        manager = Manager()
        original = manager.compute
        with self.assertRaisesRegex(RuntimeError, "boom"):
            with capture_before_reset(manager, lambda: None):
                manager.compute()
        self.assertEqual(manager.compute.__func__, original.__func__)


if __name__ == "__main__":
    unittest.main()
