"""扩速门控应验证真实响应，不能仅凭全局平均误差很小就升级。"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("pointfoot_runtime", Path(__file__).parents[1] / "scripts/pointfoot_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def skills():
    targets = ((0, 0, 0), (.2, 0, 0), (-.2, 0, 0), (0, .1, 0), (0, -.1, 0), (0, 0, .2), (0, 0, -.2))
    return {name: dict(samples=100, velocity_mae=[.01] * 3, actual_mean=list(cmd), command_mean=list(cmd))
            for name, cmd in zip(runtime.COMMAND_MODE_NAMES, targets)}


def test_complete_skills_pass():
    assert runtime.command_skills_ready(skills())


def test_missing_direction_blocks_promotion():
    value = skills(); value["reverse"]["samples"] = 0
    assert not runtime.command_skills_ready(value)


def test_stationary_actor_small_mae_does_not_pass():
    value = skills()
    for name in ("left", "right", "yaw_left", "yaw_right"):
        value[name]["actual_mean"] = [0., 0., 0.]
        value[name]["velocity_mae"] = [abs(v) for v in value[name]["command_mean"]]
    assert not runtime.command_skills_ready(value)


def test_failed_braking_blocks_promotion():
    value = skills(); value["stand"]["velocity_mae"] = [.2, 0, 0]
    assert not runtime.command_skills_ready(value)


def test_command_frame_replaces_only_newest_frame():
    import torch
    from types import SimpleNamespace
    term = SimpleNamespace(vel_command_b=torch.zeros(2, 3), is_standing_env=torch.zeros(2, dtype=torch.bool),
                           time_left=torch.zeros(2))
    env = SimpleNamespace(device="cpu", command_manager=SimpleNamespace(get_term=lambda key: term))
    obs = torch.arange(225.).repeat(2, 1)
    previous = obs.clone()
    runtime.set_pointfoot_command(env, obs, [0, .05, 0])
    assert torch.equal(obs[:, :42], previous[:, :42]) and torch.equal(obs[:, 45:], previous[:, 45:])
    assert torch.allclose(obs[:, 42:45], torch.tensor([0., .05, 0.]).expand(2, 3))
    assert not term.is_standing_env.any()
