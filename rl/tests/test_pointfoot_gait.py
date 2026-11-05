"""新步态的几何、时钟、控制限幅、想象梯度与统计回归；全部运行于CPU。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rp_wd_rl.supermimic.pointfoot import (
    GaitPointfootActor, PointfootSuperMimicLearner, PointfootReplay,
)
from rp_wd_rl.supermimic.pointfoot_terrain import PointfootTerrainField
from rp_wd_rl.supermimic.pointfoot_gait import (
    GAIT_FREQUENCY, GAIT_DUTY, GAIT_CLEARANCE, PointfootGaitMetrics,
    env_phase, next_phase, phase_features, reset_gait_phase,
    gait_quality_ready,
    foot_support_mask,
)
from rp_wd_rl.supermimic.pointfoot_gait_rewards import GaitPointfootTaskRewards, GAIT_REWARD_WEIGHTS
from rp_wd_rl.supermimic.pointfoot_courses import FullCourseHistory

NAMES = [f"{leg}_{kind}_JOINT" for kind in ("ABAD", "HIP", "KENN")
         for leg in ("LB", "LF", "RB", "RF")]


def task():
    return GaitPointfootTaskRewards(NAMES, torch.tensor([0.] * 4 + [.72] * 4 + [-1.41] * 4),
                                    torch.tensor([[-2.8, 2.8]] * 12))


@pytest.mark.parametrize("command", [[.2, 0, 0], [-.2, 0, 0], [0, .15, 0], [0, -.15, 0],
                                      [0, 0, .4], [0, 0, -.4], [.25, .1, .3], [0, 0, 0]])
def test_ik_reaches_reference_with_actual_urdf(command):
    r = task()
    phase = torch.linspace(0, 1, 201)[:, None]
    target, _, _, _ = r.gait.trajectory(phase, torch.tensor([command], dtype=torch.float32).expand(len(phase), -1))
    state = torch.zeros(len(phase), 34)
    state[:, 10:22] = r.gait.inverse(target) - r.default_q
    torch.testing.assert_close(r.feet_body(state), target, atol=2e-6, rtol=0)


def test_full_stride_and_diagonal_swing():
    r = task()
    phase = torch.linspace(0, .9999, 4001)[:, None]
    target, stance, _, _ = r.gait.trajectory(phase, torch.tensor([[.2, 0., 0.]]).expand(len(phase), -1))
    assert torch.equal(stance[:, 0], stance[:, 3])
    assert torch.equal(stance[:, 1], stance[:, 2])
    assert torch.all(stance.sum(-1) >= 2)
    assert float((~stance[:, 0]).float().mean()) == pytest.approx(1 - GAIT_DUTY, abs=.001)
    assert float(target[:, 0, 0].max() - target[:, 0, 0].min()) >= .2 * GAIT_DUTY / GAIT_FREQUENCY
    assert float((target[:, 0, 2] - r.gait.neutral[0, 2]).max()) == pytest.approx(GAIT_CLEARANCE, abs=1e-5)
    # 首次落脚前的完整周期前进参考为v/f，而非每50Hz循环都抖动一次。
    assert .2 / GAIT_FREQUENCY == pytest.approx(.16)


@pytest.mark.parametrize("boundary", [0., GAIT_DUTY])
def test_reference_is_periodic_and_velocity_continuous(boundary):
    r = task()
    eps = 1e-4
    p = torch.tensor([[boundary - 2*eps], [boundary - eps], [boundary], [boundary + eps], [boundary + 2*eps]])
    xyz = r.gait.trajectory(p, torch.tensor([[.2, .1, .3]]).expand(5, -1))[0]
    left, right = (xyz[1] - xyz[0]) / eps, (xyz[4] - xyz[3]) / eps
    torch.testing.assert_close(left, right, atol=.003, rtol=.01)


def test_stand_does_not_force_gait_motion():
    r = task()
    p = torch.rand(20, 1)
    cmd = torch.zeros(20, 3)
    target = r.gait.trajectory(p, cmd)[0]
    torch.testing.assert_close(target, r.gait.neutral.expand(20, -1, -1))
    torch.testing.assert_close(r.gait.actions(p, cmd), torch.zeros(20, 12), atol=2e-6, rtol=0)


def test_riser_impact_is_not_mislabeled_as_ground_support():
    force = torch.tensor([[[80., 0., 0.], [0., 0., 12.], [0., 0., -20.], [10., 10., 7.]]])
    assert foot_support_mask(force).tolist() == [[False, True, False, False]]


def test_real_and_imagined_clock_agree_and_partial_reset_is_local():
    env = SimpleNamespace(num_envs=3, device="cpu", step_dt=.02, episode_length_buf=torch.tensor([10, 20, 30]))
    reset_gait_phase(env, None)
    p = env_phase(env)
    env.episode_length_buf += 1
    torch.testing.assert_close(env_phase(env), next_phase(p), atol=2e-7, rtol=0)
    before = env._pointfoot_gait_offset.clone()
    reset_gait_phase(env, [1])
    assert torch.equal(env._pointfoot_gait_offset[[0, 2]], before[[0, 2]])
    assert phase_features(p).shape == (3, 2)


def test_actor_smoothing_limits_exploration_and_keeps_residual_gradient():
    r = task(); actor = GaitPointfootActor((32, 16), r.gait)
    h = torch.randn(8, 225) * .05
    h[:, 42:45] = torch.tensor([.2, 0., 0.])
    h[:, -12:] = 0
    h[:, -24:-12] = 0
    phase = torch.rand(8, 1)
    action = actor(h, phase, noise=torch.randn(8, 12) * 20)
    assert action.abs().max() <= 1
    rates = action.abs() * r.gait.scales / .02
    assert (rates[:, r.ids] <= torch.tensor([6., 8., 10.]) + 1e-5).all()
    acceleration = action.abs() * r.gait.scales / .02**2
    assert (acceleration[:, r.ids] <= torch.tensor([60., 100., 140.]) + 1e-4).all()
    action = actor(h, phase)
    action.square().sum().backward()
    assert actor.net[-1].weight.grad.abs().sum() > 0
    assert torch.isfinite(actor.net[-1].weight.grad).all()


def test_new_reward_has_finite_gradient_and_requires_clock():
    r = task(); n = 8
    phase = torch.rand(n, 1); command = torch.tensor([[.2, 0, 0.]]).expand(n, -1)
    state = torch.zeros(n, 34); state[:, 0] = .339; state[:, 3] = -1
    state[:, 10:22] = r.gait.inverse(r.gait.trajectory(phase, command)[0]) - r.default_q
    state.requires_grad_()
    args = (state, torch.zeros(n, 12), torch.zeros(n, 12), command,
            torch.full((n, 273), .339), torch.zeros(n), torch.zeros(n))
    with pytest.raises(ValueError, match="相位"):
        r.reward(*args, contacts=torch.ones(n, 4), force=torch.ones(n, 4))
    terms = r.terms(*args, contacts=torch.ones(n, 4), force=torch.ones(n, 4), phase=phase)
    assert set(terms) == set(GAIT_REWARD_WEIGHTS)
    assert not any("torque" in k for k in terms)
    r.reward(*args, contacts=torch.ones(n, 4), force=torch.ones(n, 4), phase=phase).sum().backward()
    assert torch.isfinite(state.grad).all()


def test_support_feet_not_forced_to_lift_and_contact_cannot_erase_swing_deficit():
    r = task(); state = torch.zeros(1, 34); state[:, 0] = .339; state[:, 3] = -1
    args = (state, torch.zeros(1, 12), torch.zeros(1, 12), torch.tensor([[.2, 0, 0.]]),
            torch.full((1, 273), .339), torch.zeros(1), torch.zeros(1))
    terms = r.terms(*args, contacts=torch.ones(1, 4), force=torch.ones(1, 4), phase=torch.tensor([[0.]]))
    assert terms["stair_clearance"] == 0  # 此相位四脚均为支撑。
    terms = r.terms(*args, contacts=torch.ones(1, 4), force=torch.ones(1, 4), phase=torch.tensor([[(1+GAIT_DUTY)/2]]))
    assert terms["gait_swing_height"] > .2
    assert terms["stair_clearance"] > .1


def test_gait_metrics_measure_known_cycles_and_ignore_reset():
    m = PointfootGaitMetrics(1, "cpu", .02)
    for step in range(160):
        phase = step % 40
        snap = {"contacts": torch.full((1, 4), float(phase < 25)),
                "foot_positions": torch.full((1, 4, 3), step * .004)}
        # 只在X方向前进，每周期0.16m。
        snap["foot_positions"][..., 1:] = 0
        m.update(snap, torch.tensor([False]))
    value = m.summary()
    assert value["frequency_hz"] == pytest.approx(1.25, abs=1e-5)
    assert value["stride_m"] == pytest.approx(.16, abs=1e-5)
    assert value["air_time_s"] == pytest.approx(.30, abs=1e-5)
    m.update(snap, torch.tensor([True]))
    assert not m.seen.any()


def test_curriculum_does_not_promote_lucky_sixty_percent():
    h = FullCourseHistory([1], pass_rate=.8, minimum_trials=128)
    h.iteration = 5000
    cols = torch.zeros(64, dtype=torch.long); flags = torch.ones(64, dtype=torch.bool)
    success = torch.arange(64) < 39
    h.record(cols, cols, success, flags, ~flags, ~flags)
    h.record(cols, cols, success, flags, ~flags, ~flags)
    assert h.caps.item() == 0
    h.record(cols, cols, flags, flags, ~flags, ~flags)
    assert h.caps.item() == 1


def test_shuffling_cannot_unlock_terrain_and_quality_survives_resume():
    metrics = dict(cycles=500, frequency_hz=1.25, stride_m=.16, air_time_s=.25)
    assert gait_quality_ready(metrics)
    for key, value in (("cycles", 10), ("frequency_hz", 4.), ("stride_m", .015), ("air_time_s", .04)):
        assert not gait_quality_ready({**metrics, key: value})
    h = FullCourseHistory([1], require_gait_quality=True)
    h.iteration = 5000
    x = torch.zeros(64, dtype=torch.long); f = torch.ones(64, dtype=torch.bool)
    h.record(x, x, f, f, ~f, ~f)
    assert h.caps.item() == 0
    h.gait_ready = True
    restored = FullCourseHistory([1], require_gait_quality=True)
    restored.load_state_dict(h.state_dict())
    assert restored.gait_ready
    restored.record(x, x, f, f, ~f, ~f)
    assert restored.caps.item() == 1


def test_v4_six_cm_course_geometry_and_landing_use_same_height():
    from rp_wd_rl.supermimic.pointfoot_courses import pointfoot_full_course, stable_course_pass
    heights = (.02, .04, .06, .08, .10, .12)
    cfg = SimpleNamespace(size=(16., 16.), kind="up", stair_tread=.3, step_heights=heights)
    meshes, origin = pointfoot_full_course(2.1/6, cfg)
    assert max(mesh.bounds[1, 2] for mesh in meshes) == pytest.approx(.30)
    args = (torch.tensor([[3., 0., .64]]), torch.tensor([[0., 0., -1.]]),
            torch.tensor([2]), torch.tensor([1]), torch.full((1, 4), .30),
            torch.ones(1, 4, dtype=torch.bool), torch.zeros(1))
    assert stable_course_pass(*args, step_heights=heights).item()
    assert not stable_course_pass(*args).item()  # 旧版等级2为8cm，不能混用通过标准。


def test_imagination_updates_actor_through_model_with_phase_and_roundtrip():
    torch.manual_seed(8)
    r = task()
    args = SimpleNamespace(buffer_len=8, wm_window=4, wm_batch=4, imag_batch=2, horizon=8,
                           wm_hidden=(16,), actor_hidden=(16,), critic_hidden=(16,),
                           lr=2e-4, actor_lr_scale=.3, gamma=.99, lam=.95,
                           critic_epochs=1, critic_minibatches=1)
    field = PointfootTerrainField(torch.zeros(20, 20), (-2., -2.), .25)
    learner = PointfootSuperMimicLearner(args, torch.device("cpu"), 2, field, r)
    for i in range(8):
        row = {k: torch.zeros(2, v) for k, v in PointfootReplay.SHAPES.items()}
        row["phase"] = torch.full((2, 1), i * .025)
        row["state"][:, 0] = .339; row["state"][:, 3] = -1
        row["next_state"] = row["state"].clone()
        row["command"][:, 0] = .2; row["history"][:, 42] = .2
        row["scan"] = row["next_scan"] = torch.full((2, 273), .339)
        learner.replay.add(row, torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool),
                           torch.zeros(2, dtype=torch.bool))
    assert learner.replay.sample_steps(2)["phase"].shape == (2, 1)
    learner.fit_world_model()
    before = learner.actor.net[-1].weight.detach().clone()
    metrics = learner.improve()
    assert learner.policy_updates == 1 and learner.skipped_updates == 0
    assert torch.isfinite(torch.tensor(metrics["actor_loss"]))
    assert not torch.equal(before, learner.actor.net[-1].weight)
    assert all(p.grad is None for p in learner.model.parameters())
    payload = learner.state_dict(3)
    from rp_wd_rl.supermimic.pointfoot_gait import gait_parameters
    payload["training_contract"] = {"joint_names": NAMES, "gait": gait_parameters()}
    spec = importlib.util.spec_from_file_location("pf_runtime", Path(__file__).parents[1] / "scripts/pointfoot_runtime.py")
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    loaded = runtime.actor_from_checkpoint(payload, "cpu")
    h = torch.zeros(1, 225); p = torch.tensor([[.2]])
    torch.testing.assert_close(learner.actor(h, p), loaded(h, p))
    payload["training_contract"]["gait"]["frequency"] = 9.
    with pytest.raises(ValueError, match="不匹配"):
        runtime.actor_from_checkpoint(payload, "cpu")
