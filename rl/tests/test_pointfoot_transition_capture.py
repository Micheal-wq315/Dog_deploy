"""无效物理帧必须在env.step自动reset之前改写done，且不影响旧评测协议。"""
from pathlib import Path
import importlib.util
import torch

spec = importlib.util.spec_from_file_location('transition_capture', Path(__file__).parents[1] / 'scripts/transition_capture.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class Manager:
    def __init__(self):
        self.terminated = torch.zeros(2, dtype=torch.bool)
    @property
    def dones(self):
        return self.terminated.clone()
    def compute(self):
        return torch.zeros(2, dtype=torch.bool)


def test_invalid_snapshot_overrides_done_before_course_reset():
    manager = Manager(); original = manager.compute
    def capture():
        manager.terminated[0] = True
        return 'physical_nan'
    with module.capture_before_reset(manager, capture, honor_captured_termination=True) as frames:
        result = manager.compute()
    assert result.tolist() == [True, False] and frames == ['physical_nan']
    assert manager.compute == original


def test_legacy_capture_default_preserves_original_compute_result():
    manager = Manager()
    def capture():
        manager.terminated[0] = True
        return 'observation'
    with module.capture_before_reset(manager, capture) as frames:
        result = manager.compute()
    assert not result.any() and frames == ['observation']
