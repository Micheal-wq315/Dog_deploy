"""确定性验收必须与训练的真实着陆条件一致，且不统计reset后的下一回合。"""
import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location("pointfoot_runtime", Path(__file__).parents[1] / "scripts/pointfoot_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def snapshot():
    state = torch.zeros(1, 34); state[:, 0] = .34; state[:, 3] = -1
    feet = torch.zeros(1, 4, 3); feet[..., 2] = .122
    return dict(root_pos=torch.tensor([[3., 0., .44]]), state=state, foot_positions=feet,
                contacts_current=torch.ones(1, 4), vertical_speed_world=torch.zeros(1))


def frame_pass(s):
    return runtime.course_frame_pass(s, torch.zeros(1, 3), torch.ones(1, dtype=torch.long), torch.zeros(1, dtype=torch.long))


def test_correct_landing_passes_but_historical_contact_is_insufficient():
    s = snapshot(); assert frame_pass(s).all()
    s["contacts_current"].zero_(); assert not frame_pass(s).any()


def test_wrong_tread_or_falling_is_not_landing():
    s = snapshot(); s["foot_positions"][..., 2] = .022
    assert not frame_pass(s).any()
    s = snapshot(); s["vertical_speed_world"][:] = -1
    assert not frame_pass(s).any()


def test_six_consecutive_frames_required():
    tracker = runtime.CoursePassTracker(1, "cpu", .02)
    yes, no = torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    for _ in range(5): tracker.update(torch.tensor([3.]), yes, yes, no)
    assert not tracker.reached.any()
    tracker.update(torch.tensor([3.]), yes, yes, no)
    assert tracker.reached.all()


def test_ended_trial_progress_cannot_use_next_episode():
    tracker = runtime.CoursePassTracker(1, "cpu", .02)
    yes, no = torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    tracker.update(torch.tensor([2.]), no, yes, no)
    tracker.update(torch.tensor([100.]), yes, no, no)
    assert float(tracker.progress[0]) == 2. and not tracker.reached.any()
