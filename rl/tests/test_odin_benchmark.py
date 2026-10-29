"""固定赛道统计和教师晋级门槛的快速回归，不启动仿真。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_metrics import benchmark_attempt_metrics
from qualify_odin_teacher import CORE_SCENARIOS, qualify


def sample(distance, episode=0, terminated=0, force=0):
    return dict(env_id=0, episode_id=episode, terrain_name="step_up_06cm",
                forward_progress_m=distance, lateral_offset_m=0, terminated=terminated,
                timeout=0, nonfoot_force_peak_n=force, wheel_force_peak_n=100,
                root_clearance_m=.419, tilt_deg=0)


class BenchmarkTests(unittest.TestCase):
    def test_approaching_is_not_completion(self):
        self.assertEqual(benchmark_attempt_metrics([sample(1.6)])["completed_attempts"], 0)

    def test_terminal_displacement_is_not_completion(self):
        self.assertEqual(benchmark_attempt_metrics([sample(3.4, terminated=1)])["completed_attempts"], 0)

    def test_crouched_endpoint_is_not_completion(self):
        row = sample(3.4)
        row["root_clearance_m"] = .25
        self.assertEqual(benchmark_attempt_metrics([row])["completed_attempts"], 0)

    def test_first_attempt_is_not_diluted_by_tail_reset(self):
        rows = [sample(0), sample(3.3), sample(4, terminated=1), sample(.1, episode=1)]
        result = benchmark_attempt_metrics(rows)
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["first_attempts"], 1)
        self.assertEqual(result["first_completed_attempts"], 1)
        self.assertEqual(result["unfinished_at_test_horizon"], 1)

    def test_contact_before_goal_is_retained(self):
        result = benchmark_attempt_metrics([sample(1, force=50), sample(3.3)])
        self.assertEqual(result["completed_attempts"], 1)
        self.assertEqual(result["low_contact_completed_attempts"], 0)

    def test_teacher_gate_requires_every_core_scenario(self):
        with tempfile.TemporaryDirectory() as folder:
            dirs = []
            for seed in (42, 43, 44):
                directory = Path(folder) / str(seed)
                directory.mkdir()
                dirs.append(directory)
                (directory / "config.json").write_text(json.dumps({"completed": True}))
                report = {name: dict(first_attempts=2, first_completed_attempts=2,
                                    first_low_contact_completed_attempts=2) for name in CORE_SCENARIOS}
                (directory / "terrain_summary.json").write_text(json.dumps(report))
            self.assertTrue(qualify(dirs)["passed"])
            report["step_down_08cm"]["first_completed_attempts"] = 0
            report["step_down_08cm"]["first_low_contact_completed_attempts"] = 0
            (dirs[-1] / "terrain_summary.json").write_text(json.dumps(report))
            self.assertFalse(qualify(dirs)["passed"])

    def test_partial_evaluation_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "config.json").write_text(json.dumps({"completed": False}))
            with self.assertRaises(ValueError):
                qualify([folder])

    def test_early_gate_does_not_require_untrained_high_obstacles(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps({"completed": True}))
            report = {name: dict(first_attempts=2, first_completed_attempts=0,
                                first_low_contact_completed_attempts=0) for name in CORE_SCENARIOS}
            report["flat_reference"].update(first_completed_attempts=2, first_low_contact_completed_attempts=2)
            (root / "terrain_summary.json").write_text(json.dumps(report))
            self.assertTrue(qualify([root], scenarios=("flat_reference",))["passed"])
            self.assertFalse(qualify([root])["passed"])
            (root / "summary.json").write_text(json.dumps([{"case": "flat_reference", "vx": .13}]))
            self.assertFalse(qualify([root], scenarios=("flat_reference",), min_flat_vx=.25)["passed"])
            (root / "summary.json").write_text(json.dumps([{"case": "flat_reference", "vx": .30}]))
            self.assertTrue(qualify([root], scenarios=("flat_reference",), min_flat_vx=.25)["passed"])
            with self.assertRaises(ValueError):
                qualify([root], scenarios=())


if __name__ == "__main__":
    unittest.main()
