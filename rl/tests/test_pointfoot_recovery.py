"""无仿真的恢复回归：直接载入本机 Isaac Lab 2.3.2 的实际历史缓冲实现。

可用 ISAACLAB_PATH 指定安装根目录；不导入 isaaclab 包、不创建 SimulationApp。
ObservationManager 的实际 reset/compute/compute_group 由源码提取，仅替换环境和无噪声项。
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pointfoot_recovery import recover_invalid_rows  # noqa: E402

ISAAC_ROOT = Path(os.environ.get("ISAACLAB_PATH", "/media/rmer/data/IsaacLab"))
ISAAC_SOURCE = ISAAC_ROOT / "source/isaaclab/isaaclab"
BUFFER_SOURCE = ISAAC_SOURCE / "utils/buffers/circular_buffer.py"
if not BUFFER_SOURCE.is_file():
    raise unittest.SkipTest("需实际 Isaac Lab CircularBuffer 源码，请设置 ISAACLAB_PATH")

spec = importlib.util.spec_from_file_location("_pointfoot_actual_circular_buffer", BUFFER_SOURCE)
actual_buffer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(actual_buffer)
CircularBuffer = actual_buffer.CircularBuffer


def source_objects(path, names, namespace, class_name=None):
    """执行实际函数/类源码，避开 Isaac 包初始化和仿真依赖。"""
    tree = ast.parse(path.read_text())
    body = tree.body if class_name is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    nodes = [node for node in body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    if len(nodes) != len(names):
        raise RuntimeError(f"实际源码接口变化：{path}")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


observation_methods = source_objects(
    ISAAC_SOURCE / "managers/observation_manager.py", {"reset", "compute", "compute_group"},
    {"torch": torch, "CircularBuffer": CircularBuffer,
     "noise": NS(NoiseCfg=type("UnusedNoiseCfg", (), {}), NoiseModelCfg=type("UnusedNoiseModelCfg", (), {}))},
    class_name="ObservationManager",
)
course_namespace = source_objects(
    ROOT / "rp_wd_rl/supermimic/pointfoot_courses.py", {"FullCourseHistory"},
    {"torch": torch, "math": math, "COURSE_KINDS": ("flat", "up", "down", "platform", "slope", "rubble"),
     "STEP_HEIGHTS": (.02, .04, .08, .12, .16, .20), "HISTORY_WINDOW": 64,
     "STAT_FIELDS": ("trials", "passed", "failed", "falls", "stuck", "neutral", "episodes"),
     "MIN_ITERATIONS": (0, 1500, 4000, 8000, 12000, 16000)},
)
full_methods = source_objects(
    ROOT / "rp_wd_rl/tasks/locomotion/velocity/mdp/pointfoot_full.py", {"full_terrain_curriculum"},
    {"torch": torch, "_history": lambda env: env.course,
     "_episode_state": lambda env: env._pointfoot_full_episode,
     "_ids": lambda env, ids: ids,
     # 父脚本在真实 step 后恢复，此步课程计算已完成且按 common_step_counter 幂等。
     "update_full_course_step": lambda env: None},
)


class ObservationHarness:
    """实际 ObservationManager 方法和 CircularBuffer，数据由纯 CPU 环境提供。"""

    reset = observation_methods["reset"]
    compute_group = observation_methods["compute_group"]

    def __init__(self, env):
        self._env = env
        self.calls = []
        self._group_obs_term_names = {"policy": ["joint", "command", "one"], "extra": ["grid", "instant"]}
        self._group_obs_term_history_buffer = {"policy": {}, "extra": {}}
        self._group_obs_term_cfgs = {"policy": [], "extra": []}
        self._group_obs_class_term_cfgs = {"policy": [], "extra": []}
        self._group_obs_class_instances = []
        self._group_obs_concatenate = {"policy": True, "extra": False}
        self._group_obs_concatenate_dim = {"policy": -1, "extra": -1}
        self.shapes = {"joint": (3,), "command": (3,), "one": (1,), "grid": (2, 3), "instant": (2,)}
        lengths = {"joint": 5, "command": 2, "one": 1, "grid": 3, "instant": 0}
        for group, names in self._group_obs_term_names.items():
            for name in names:
                length = lengths[name]
                if length:
                    self._group_obs_term_history_buffer[group][name] = CircularBuffer(length, env.num_envs, "cpu")
                self._group_obs_term_cfgs[group].append(NS(
                    func=lambda subject, term=name: subject.frames[term], params={}, modifiers=None,
                    noise=None, clip=None, scale=None, history_length=length,
                    flatten_history_dim=(group == "policy"),
                ))

    def compute(self, update_history=False):
        self.calls.append(update_history)
        return observation_methods["compute"](self, update_history=update_history)

    def buffers(self):
        return [buffer for group in self._group_obs_term_history_buffer.values() for buffer in group.values()]


class EnvironmentHarness:
    def __init__(self, num_envs=4, full=True):
        self.num_envs, self.device = num_envs, "cpu"
        self.termination_manager = NS(terminated=torch.zeros(num_envs, dtype=torch.bool))
        self.observation_manager = ObservationHarness(self)
        self.frames = {}
        self.reset_calls = []
        self.reset_flags = []
        self.episode_length_buf = torch.full((num_envs,), 300, dtype=torch.long)
        self.step_dt = .02
        if full:
            self._pointfoot_full_episode = {"passed": torch.ones(num_envs, dtype=torch.bool),
                "traverse_time": torch.full((num_envs,), 6.), "stuck": torch.zeros(num_envs, dtype=torch.bool)}
            self._pointfoot_success_event = torch.ones(num_envs, dtype=torch.bool)
            self.course = course_namespace["FullCourseHistory"]((1,), rows=6, device="cpu")
            self.course.caps[:] = 5
            self.scene = NS(terrain=NS(
                terrain_types=torch.zeros(num_envs, dtype=torch.long),
                terrain_levels=torch.full((num_envs,), 2, dtype=torch.long),
                terrain_origins=torch.zeros(6, 1, 3), env_origins=torch.zeros(num_envs, 3),
            ))
        self.set_frames(0)

    def set_frames(self, time):
        for number, (name, shape) in enumerate(self.observation_manager.shapes.items()):
            values = torch.arange(math.prod(shape)).float().reshape(1, *shape)
            rows = torch.arange(self.num_envs).float().reshape(self.num_envs, *([1] * len(shape)))
            self.frames[name] = time * 100. + rows * 1000. + number * 10. + values

    def advance(self, time):
        self.set_frames(time)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf

    def reset(self, env_ids):
        self.reset_calls.append(env_ids.clone())
        flags = {"terminated": self.termination_manager.terminated.clone()}
        if hasattr(self, "_pointfoot_full_episode"):
            flags.update(passed=self._pointfoot_full_episode["passed"].clone(),
                         success=self._pointfoot_success_event.clone())
            full_methods["full_terrain_curriculum"](self, env_ids)
            self._pointfoot_full_episode["passed"][env_ids] = False
        self.reset_flags.append(flags)
        self.observation_manager.reset(env_ids)
        for frame in self.frames.values():
            frame[env_ids] = -10000. - env_ids.float().reshape(-1, *([1] * (frame.ndim - 1)))
        self.episode_length_buf[env_ids] = 0
        # 与 ManagerBasedEnv.reset 一致：所有行 append，而非只写 env_ids。
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, {}


class RecoveryTests(unittest.TestCase):
    def histories(self, env):
        return [(buffer.buffer.clone(), buffer._num_pushes.clone(), buffer._pointer)
                for buffer in env.observation_manager.buffers()]

    def assert_recovered(self, env, bad, before, obs):
        for buffer, (ordered, pushes, pointer) in zip(env.observation_manager.buffers(), before):
            self.assertIsInstance(buffer, actual_buffer.CircularBuffer)
            torch.testing.assert_close(buffer.buffer[~bad], ordered[~bad], atol=0, rtol=0)
            torch.testing.assert_close(buffer._num_pushes[~bad], pushes[~bad], atol=0, rtol=0)
            self.assertEqual(buffer._pointer, (pointer + 1) % buffer.max_length)
            self.assertTrue((buffer._num_pushes[bad] == 1).all())
        for group, terms in env.observation_manager._group_obs_term_history_buffer.items():
            for name, buffer in terms.items():
                expected = env.frames[name][bad, None].expand_as(buffer.buffer[bad])
                torch.testing.assert_close(buffer.buffer[bad], expected, atol=0, rtol=0)
                self.assertTrue(torch.isfinite(buffer.buffer[bad]).all())
        self.assertIs(obs, env.obs_buf)
        self.assertIs(obs, env.observation_manager._obs_buffer)
        self.assertEqual(env.observation_manager.calls[-2:], [True, False])

    def test_actual_buffers_keep_healthy_history_for_every_pointer_and_wrap(self):
        for steps in (1, 2, 3, 4, 5, 6, 8, 10, 14):
            with self.subTest(steps=steps):
                env = EnvironmentHarness()
                for step in range(steps):
                    env.advance(step)
                bad = torch.tensor([False, True, False, True])
                before = self.histories(env)
                prior_policy = env.obs_buf["policy"].clone()
                for frame in env.frames.values():
                    frame[bad] = float("nan")
                obs = recover_invalid_rows(env, bad)
                self.assert_recovered(env, bad, before, obs)
                torch.testing.assert_close(obs["policy"][~bad], prior_policy[~bad], atol=0, rtol=0)
                self.assertEqual(len(env.reset_calls), 1)
                torch.testing.assert_close(env.reset_calls[0], torch.tensor([1, 3]))

    def test_failure_and_success_flags_precede_actual_course_accounting(self):
        env = EnvironmentHarness()
        env.advance(1)
        bad = torch.tensor([False, True, False, True])
        recover_invalid_rows(env, bad)
        flags = env.reset_flags[0]
        self.assertTrue(flags["terminated"][bad].all())
        self.assertFalse(flags["terminated"][~bad].any())
        self.assertFalse(flags["passed"][bad].any())
        self.assertTrue(flags["passed"][~bad].all())
        self.assertFalse(flags["success"][bad].any())
        self.assertTrue(flags["success"][~bad].all())
        # 实际 FullCourseHistory.record + full_terrain_curriculum：两次失败、零成功。
        stats = dict(zip(course_namespace["STAT_FIELDS"], env.course.stats[1, 2].tolist()))
        self.assertEqual(stats["trials"], 2)
        self.assertEqual(stats["passed"], 0)
        self.assertEqual(stats["failed"], 2)
        self.assertEqual(stats["falls"], 2)
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [2, 1, 2, 1])
        self.assertFalse(env._pointfoot_success_event[bad].any())

    def test_repeated_recovery_preserves_different_push_counts_and_lag_reads(self):
        env = EnvironmentHarness(full=False)
        for step in range(7):
            env.advance(step)
        recover_invalid_rows(env, torch.tensor([True, False, False, False]))
        env.advance(8)
        bad = torch.tensor([False, False, True, False])
        before = self.histories(env)
        lagged = [[buffer[torch.full((env.num_envs,), lag, dtype=torch.long)].clone()
                   for lag in range(8)] for buffer in env.observation_manager.buffers()]
        obs = recover_invalid_rows(env, bad)
        self.assert_recovered(env, bad, before, obs)
        for buffer, previous in zip(env.observation_manager.buffers(), lagged):
            for lag, expected in enumerate(previous):
                actual = buffer[torch.full((env.num_envs,), lag, dtype=torch.long)]
                torch.testing.assert_close(actual[~bad], expected[~bad], atol=0, rtol=0)

    def test_all_bad_rows_keep_the_public_reset_new_frames(self):
        env = EnvironmentHarness()
        env.advance(3)
        bad = torch.ones(env.num_envs, dtype=torch.bool)
        before = self.histories(env)
        obs = recover_invalid_rows(env, bad)
        self.assert_recovered(env, bad, before, obs)
        self.assertTrue(torch.isfinite(obs["policy"]).all())

    def test_no_bad_rows_never_reset_or_append(self):
        env = EnvironmentHarness()
        env.advance(3)
        before = self.histories(env)
        flags = env._pointfoot_full_episode["passed"].clone()
        obs = recover_invalid_rows(env, torch.zeros(env.num_envs, dtype=torch.bool))
        self.assertFalse(env.reset_calls)
        for buffer, (ordered, pushes, pointer) in zip(env.observation_manager.buffers(), before):
            torch.testing.assert_close(buffer.buffer, ordered, atol=0, rtol=0)
            torch.testing.assert_close(buffer._num_pushes, pushes, atol=0, rtol=0)
            self.assertEqual(buffer._pointer, pointer)
        torch.testing.assert_close(env._pointfoot_full_episode["passed"], flags)
        self.assertEqual(env.observation_manager.calls[-1], False)
        self.assertIs(obs, env.obs_buf)

    def test_wrong_mask_shape_or_dtype_fails_before_mutating(self):
        env = EnvironmentHarness()
        env.advance(3)
        for bad in (torch.ones(4), torch.ones(4, 1, dtype=torch.bool), torch.ones(3, dtype=torch.bool)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                recover_invalid_rows(env, bad)
        self.assertFalse(env.reset_calls)
        self.assertFalse(env.termination_manager.terminated.any())

    def test_wrapped_environment_is_supported(self):
        env = EnvironmentHarness(full=False)
        env.advance(1)
        before = self.histories(env)
        bad = torch.tensor([True, False, False, False])
        obs = recover_invalid_rows(NS(unwrapped=env), bad)
        self.assert_recovered(env, bad, before, obs)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
