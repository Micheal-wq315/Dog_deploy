"""full-v3 几何、命令、门控及真实步/reset协议的独立 CPU 回归。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rp_wd_rl.supermimic.pointfoot_courses import (  # noqa: E402
    COMMAND_LIMITS, COURSE_KINDS, COURSE_PROPORTIONS, FullCourseHistory,
    MIN_ITERATIONS, STAT_FIELDS, STEP_HEIGHTS, column_kind_indices,
    difficulty_level, pointfoot_full_course, sample_mixed_commands, stable_course_pass,
)


def configs(weights=COURSE_PROPORTIONS):
    return {kind: NS(kind=kind, proportion=weight) for kind, weight in zip(COURSE_KINDS, weights)}


def ray_heights(mesh, xy):
    xy = np.asarray(xy)
    starts = np.column_stack((xy, np.full(len(xy), 4.)))
    hits, indices, _ = mesh.ray.intersects_location(
        starts, np.tile((0., 0., -1.), (len(xy), 1)), multiple_hits=False,
    )
    result = np.full(len(xy), np.nan)
    result[indices] = hits[:, 2]
    return result


class GeometryTests(unittest.TestCase):
    def test_generator_column_assignment_normalizes_and_uses_left_edge(self):
        cfg = configs()
        expected = (0, 1, 1, 1, 2, 2, 3, 4, 5, 5)
        self.assertEqual(column_kind_indices(cfg, 10), expected)
        self.assertEqual(column_kind_indices(configs(tuple(7 * p for p in COURSE_PROPORTIONS)), 10), expected)
        # 第一列的中心 .05 会落入 up，而 generator 的 .001 仍落入 flat。
        skew = configs((.02, .38, .2, .1, .1, .2))
        self.assertEqual(column_kind_indices(skew, 10)[0], 0)
        reordered = dict(reversed(list(cfg.items())))
        weights = np.array([c.proportion for c in reordered.values()])
        cumulative = np.cumsum(weights / weights.sum())
        expected = tuple(COURSE_KINDS.index(list(reordered)[np.flatnonzero(c / 10 + .001 < cumulative)[0]])
                         for c in range(10))
        self.assertEqual(column_kind_indices(reordered, 10), expected)

    def test_each_randomized_difficulty_stays_in_its_row(self):
        for row in range(6):
            for jitter in (0., .01, .49, .99, .999999):
                self.assertEqual(difficulty_level((row + jitter) / 6), row)
        self.assertEqual(difficulty_level(1.), 5)

    def test_all_kinds_and_levels_have_mirrored_solid_geometry_and_safe_spawn(self):
        points = np.array([[8., 8.], [8.4, 8.3], [7.6, 7.7], [13.6, 8.], [2.4, 8.]])
        for kind in COURSE_KINDS:
            for level, h in enumerate(STEP_HEIGHTS):
                with self.subTest(kind=kind, level=level):
                    meshes, origin = pointfoot_full_course((level + .5) / 6, NS(kind=kind, size=(16., 16.), seed=42))
                    mesh = trimesh.util.concatenate(meshes)
                    self.assertTrue(all(m.is_watertight for m in meshes))
                    self.assertTrue(np.isfinite(mesh.vertices).all())
                    np.testing.assert_allclose(mesh.bounds[0, :2], (0., 0.), atol=1e-6)
                    np.testing.assert_allclose(mesh.bounds[1, :2], (16., 16.), atol=1e-6)
                    heights = ray_heights(mesh, points)
                    expected_spawn = 5 * h if kind == "down" else 0.
                    np.testing.assert_allclose(origin, (8., 8., expected_spawn))
                    np.testing.assert_allclose(heights[:3], expected_spawn, atol=1e-6)
                    np.testing.assert_allclose(heights[3], heights[4], atol=1e-6)
                    if kind == "rubble":
                        self.assertLessEqual(mesh.vertices[:, 2].max(), h + 1e-6)

    def test_stairs_have_five_exact_risers_in_both_directions(self):
        distances = (.5, 1.05, 1.35, 1.65, 1.95, 2.25, 3.0, 5.5)
        for kind in ("up", "down"):
            for row, h in enumerate(STEP_HEIGHTS):
                meshes, origin = pointfoot_full_course((row + .5) / 6, NS(kind=kind, size=(16., 16.)))
                mesh = trimesh.util.concatenate(meshes)
                expected = np.array((0, 1, 2, 3, 4, 5, 5, 5)) * h
                if kind == "down":
                    expected = 5 * h - expected
                for direction in (-1, 1):
                    xy = [(8 + direction * d, 8.) for d in distances]
                    np.testing.assert_allclose(ray_heights(mesh, xy), expected, atol=1e-6)

    def test_default_urdf_feet_clear_the_central_ground_for_all_courses(self):
        from rp_wd_rl.assets.rp_wd_contract import DEFAULT_JOINT_POS, LEG_JOINT_NAMES
        from rp_wd_rl.supermimic.pointfoot_terrain import PointfootTaskRewards
        q = torch.tensor([DEFAULT_JOINT_POS[n.split("_")[1]] for n in LEG_JOINT_NAMES])
        task = PointfootTaskRewards(LEG_JOINT_NAMES, q, torch.tensor([[-3., 3.]] * 12))
        state = torch.zeros(1, 34)
        state[:, 0], state[:, 3] = .34, -1.
        feet = task.feet(state)[0]
        gap = .34 + feet[:, 2] - .022
        self.assertTrue(bool((gap > 0).all()))
        self.assertTrue(bool((gap < .05).all()))
        self.assertTrue(bool((feet[:, :2].abs() + .08 < .8).all()))


class SamplingTests(unittest.TestCase):
    def test_free_terrain_mode_ratios_and_stop_overlay(self):
        n = 60000
        values, traverse = sample_mixed_commands(n, torch.zeros(n, dtype=torch.bool), COMMAND_LIMITS[-1],
                                                generator=torch.Generator().manual_seed(42))
        nonzero = values.abs() > 0
        stop = ~nonzero.any(-1)
        self.assertAlmostEqual(float(stop.float().mean()), .1, delta=.008)
        active = nonzero[~stop]
        ratios = [((active.sum(-1) == 1) & active[:, i]).float().mean().item() for i in range(3)]
        ratios.append((active.sum(-1) == 3).float().mean().item())
        for ratio, expected in zip(ratios, (.4, .2, .2, .2)):
            self.assertAlmostEqual(ratio, expected, delta=.012)
        self.assertFalse(bool(traverse.any()))

    def test_stair_traversal_stop_ratios_symmetry_and_every_stage_bounds(self):
        n = 30000
        for stage, limits in enumerate(COMMAND_LIMITS):
            values, traverse = sample_mixed_commands(n, torch.ones(n, dtype=torch.bool), limits,
                                                    generator=torch.Generator().manual_seed(stage))
            stop = values.abs().amax(-1) == 0
            self.assertAlmostEqual(float(stop.float().mean()), .1, delta=.012)
            self.assertAlmostEqual(float(traverse[~stop].float().mean()), .8, delta=.015)
            self.assertTrue(bool((values.abs() <= torch.tensor(limits) + 1e-6).all()))
            self.assertTrue(bool((values[traverse, 1:] == 0).all()))
            self.assertTrue(bool((values[traverse, 0].abs() >= .2).all()))
            self.assertAlmostEqual(float((values[traverse, 0] > 0).float().mean()), .5, delta=.015)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.history = FullCourseHistory(column_kind_indices(configs(), 10))

    def record(self, col, row=0, count=64, passes=64, eligible=True, fallen=False, stuck=False):
        outcome = torch.arange(count) < passes
        self.history.record(torch.full((count,), col), torch.full((count,), row), outcome,
                            torch.full((count,), eligible), torch.full((count,), fallen),
                            torch.full((count,), stuck))

    def test_terrain_requires_64_trials_60_percent_and_minimum_iteration(self):
        self.record(1, count=63)
        self.assertEqual(int(self.history.caps[1]), 0)
        self.record(1, count=1)
        self.assertEqual(int(self.history.caps[1]), 0)
        self.history.iteration = MIN_ITERATIONS[1]
        self.record(1, count=1)
        self.assertEqual(int(self.history.caps[1]), 1)
        self.record(1, row=1, passes=38)
        self.assertEqual(int(self.history.caps[1]), 1)
        self.history.iteration = MIN_ITERATIONS[2]
        self.record(1, row=1, passes=38)
        self.assertEqual(int(self.history.caps[1]), 1)
        self.record(1, row=1, passes=39)
        self.assertEqual(int(self.history.caps[1]), 2)

    def test_large_parallel_batch_keeps_last_64_deterministically(self):
        self.record(1, count=130, passes=66)
        self.assertEqual(int(self.history.counts[1, 0]), 130)
        self.assertEqual(float(self.history.trials[1, 0].sum()), 0.)
        self.record(1, count=33, passes=33)
        self.assertEqual(float(self.history.trials[1, 0].sum()), 33.)

    def test_neutral_does_not_pollute_trials_but_keeps_fall_and_stuck_stats(self):
        self.record(4, count=10, passes=0, eligible=False, fallen=True, stuck=True)
        self.assertEqual(int(self.history.counts.sum()), 0)
        stats = self.history.stats[COURSE_KINDS.index("down"), 0]
        self.assertEqual(stats.tolist(), [0, 0, 0, 10, 10, 10, 10])

    def test_command_curriculum_requires_both_stair_directions_accuracy_and_time(self):
        for stage in range(5):
            self.record(1, row=stage)
            self.assertEqual(self.history.update_commands(MIN_ITERATIONS[stage + 1], (.05, .04, .1)), stage)
            self.record(4, row=stage)
            self.assertEqual(self.history.update_commands(MIN_ITERATIONS[stage + 1] - 1, (.05, .04, .1)), stage)
            self.assertEqual(self.history.update_commands(MIN_ITERATIONS[stage + 1], (.05, .04, .4)), stage)
            self.assertEqual(self.history.update_commands(MIN_ITERATIONS[stage + 1], (.05, float("nan"), .1)), stage)
            self.assertEqual(self.history.update_commands(MIN_ITERATIONS[stage + 1], (.05, .04, .1)), stage + 1)
        self.assertEqual(self.history.limits, (.6, .4, 1.))

    def test_save_load_roundtrip_preserves_aliases_and_summary_is_json(self):
        self.history.iteration = 1500
        self.record(1)
        self.record(4)
        self.history.update_commands(1500, (.04, .03, .1))
        saved = self.history.state_dict()
        caps = torch.zeros_like(self.history.caps)
        restored = FullCourseHistory(self.history.column_kinds, caps=caps)
        restored.load_state_dict(saved)
        self.assertIs(caps, restored.caps)
        self.assertEqual(restored.summary(), self.history.summary())
        json.dumps(restored.summary(), allow_nan=False)
        saved["caps"].zero_()
        self.assertEqual(int(restored.caps[1]), 1)


def load_adapter_without_isaac():
    """仅替换 Isaac 类型声明，执行真实适配函数；不导入 tasks/SimulationApp。"""
    names = ("isaaclab", "isaaclab.envs", "isaaclab.envs.mdp", "isaaclab.envs.mdp.commands",
             "isaaclab.terrains", "isaaclab.utils", "isaaclab.utils.math")
    modules = {name: ModuleType(name) for name in names}
    modules["isaaclab.envs.mdp.commands"].UniformVelocityCommand = object
    modules["isaaclab.envs.mdp.commands"].UniformVelocityCommandCfg = object
    modules["isaaclab.terrains"].SubTerrainBaseCfg = object
    modules["isaaclab.utils"].configclass = lambda cls: cls

    def quat_from_euler_xyz(roll, pitch, yaw):
        return torch.stack(((yaw / 2).cos(), torch.zeros_like(yaw), torch.zeros_like(yaw), (yaw / 2).sin()), -1)
    modules["isaaclab.utils.math"].quat_from_euler_xyz = quat_from_euler_xyz
    path = ROOT / "rp_wd_rl/tasks/locomotion/velocity/mdp/pointfoot_full.py"
    spec = importlib.util.spec_from_file_location("_pointfoot_full_cpu", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class FakeRobot:
    def __init__(self, n):
        self.data = NS(root_pos_w=torch.zeros(n, 3), root_lin_vel_w=torch.zeros(n, 3),
                       projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
                       body_pos_w=torch.zeros(n, 4, 3), default_root_state=torch.zeros(n, 13))
        self.data.default_root_state[:, 2], self.data.default_root_state[:, 3] = .34, 1.

    def find_bodies(self, names, preserve_order=True):
        return (list(range(4)), names)

    def write_root_pose_to_sim(self, pose, env_ids):
        self.data.root_pos_w[env_ids] = pose[:, :3]
        self.last_pose = pose.clone()

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.last_velocity = velocity.clone()


def fake_env(n=1):
    cfg = NS(num_cols=10, num_rows=6, sub_terrains=configs())
    terrain = NS(cfg=NS(terrain_generator=cfg), terrain_types=torch.ones(n, dtype=torch.long),
                 terrain_levels=torch.zeros(n, dtype=torch.long), env_origins=torch.zeros(n, 3),
                 terrain_origins=torch.zeros(6, 10, 3))
    robot = FakeRobot(n)
    force = torch.zeros(n, 5, 3)
    force[:, :4, 2] = 40.
    sensor = NS(data=NS(net_forces_w=force, net_forces_w_history=force[:, None].clone()))
    sensor.find_bodies = lambda names, preserve_order=True: ([4], names) if names == "base_link" else (list(range(4)), names)

    class Scene(dict):
        pass
    scene = Scene(robot=robot, contact_forces=sensor)
    scene.terrain, scene.env_origins = terrain, terrain.env_origins
    command = torch.tensor([[.22, 0., 0.]]).repeat(n, 1)
    term = NS(cfg=NS(ranges=NS()))
    return NS(num_envs=n, device="cpu", common_step_counter=0, step_dt=.02, scene=scene,
              cfg=NS(episode_length_s=35.), episode_length_buf=torch.ones(n, dtype=torch.long),
              termination_manager=NS(terminated=torch.zeros(n, dtype=torch.bool)),
              command_manager=NS(get_command=lambda _: command, get_term=lambda _: term))


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mdp = load_adapter_without_isaac()

    def settled(self, env, direction=1):
        env.scene["robot"].data.root_pos_w[:, 0] = 3.0 * direction
        env.scene["robot"].data.root_pos_w[:, 2] = .42
        env.scene["robot"].data.body_pos_w[:, :, 2] = .122

    def step(self, env):
        env.common_step_counter += 1
        env.episode_length_buf += 1
        return self.mdp.update_full_course_step(env)

    def test_success_is_symmetric_requires_settling_and_survives_reset_until_next_step(self):
        for direction in (-1, 1):
            env = fake_env()
            self.settled(env, direction)
            for _ in range(5):
                self.assertFalse(bool(self.step(env)[0]))
            self.assertTrue(bool(self.step(env)[0]))
            self.assertTrue(bool(self.mdp.update_full_course_step(env)[0]))
            self.mdp.reset_full_course_state(env, torch.tensor([0]))
            self.assertTrue(bool(env._pointfoot_success_event[0]))
            self.assertTrue(bool(self.mdp.update_full_course_step(env)[0]))
            self.assertFalse(bool(self.step(env)[0]))

    def test_no_landing_or_wrong_height_or_tilt_cannot_pass(self):
        for failure in ("air", "height", "tilt", "falling"):
            env = fake_env()
            self.settled(env)
            if failure == "air":
                env.scene["contact_forces"].data.net_forces_w.zero_()
            elif failure == "height":
                env.scene["robot"].data.body_pos_w[:, :, 2] += .2
            elif failure == "tilt":
                env.scene["robot"].data.projected_gravity_b[:, 2] = -.7
            else:
                env.scene["robot"].data.root_lin_vel_w[:, 2] = -1.
            for _ in range(10):
                self.assertFalse(bool(self.step(env)[0]))

    def test_stair_neutral_turning_and_stopping_do_not_count_as_failure(self):
        for command in ((0., 0., 0.), (0., 0., .4), (0., .15, 0.)):
            env = fake_env()
            env.scene.terrain.terrain_levels[:] = 2
            self.mdp._history(env).caps[:] = 2
            env.command_manager.get_command("base_velocity")[:] = torch.tensor(command)
            for _ in range(10):
                self.step(env)
            env.episode_length_buf[:] = 1000
            env.termination_manager.terminated[:] = True
            self.mdp.full_terrain_curriculum(env, [0])
            self.assertEqual(int(env.scene.terrain.terrain_levels[0]), 2)
            self.assertEqual(int(env._pointfoot_course_counts.sum()), 0)

    def test_traversal_failure_demotes_but_stuck_only_accumulates_under_motion_command(self):
        env = fake_env()
        env.scene.terrain.terrain_levels[:] = 2
        self.mdp._history(env).caps[:] = 2
        for _ in range(405):
            self.step(env)
        self.assertTrue(bool(self.mdp.full_course_stuck(env)[0]))
        self.mdp.full_terrain_curriculum(env, [0])
        self.assertEqual(int(env.scene.terrain.terrain_levels[0]), 1)
        self.assertEqual(int(env._pointfoot_course_counts[1, 2]), 1)
        self.assertEqual(self.mdp.full_summary(env)["terrain"]["up"]["by_level"][2][STAT_FIELDS.index("stuck")], 1)

    def test_pure_lateral_motion_is_neutral_on_obstacles_but_eligible_on_flat(self):
        env = fake_env(len(COURSE_KINDS))
        columns = column_kind_indices(configs(), 10)
        env.scene.terrain.terrain_types[:] = torch.tensor([columns.index(k) for k in range(len(COURSE_KINDS))])
        env.scene.terrain.terrain_levels[:] = 2
        self.mdp._history(env).caps[:] = 2
        env.command_manager.get_command("base_velocity")[:] = torch.tensor((0., .2, 0.))
        for _ in range(405):
            self.step(env)
        torch.testing.assert_close(self.mdp.full_course_stuck(env),
                                   torch.tensor((True, False, False, False, False, False)))
        self.mdp.full_terrain_curriculum(env, None)
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [1, 2, 2, 2, 2, 2])
        summary = self.mdp.full_summary(env)
        for kind in COURSE_KINDS:
            stats = summary["terrain"][kind]["by_level"][2]
            self.assertEqual(stats[STAT_FIELDS.index("trials")], int(kind == "flat"))
            self.assertEqual(stats[STAT_FIELDS.index("neutral")], int(kind != "flat"))

    def test_exit_has_no_legacy_reverse_boundary(self):
        env = fake_env()
        env.scene["robot"].data.root_pos_w[0, 0] = -.5
        self.assertFalse(bool(self.mdp.full_course_exit(env)[0]))
        for x, y in ((6.51, 0.), (-6.51, 0.), (0., 6.51), (0., -6.51)):
            env.scene["robot"].data.root_pos_w[0, :2] = torch.tensor((x, y))
            self.assertTrue(bool(self.mdp.full_course_exit(env)[0]))

    def test_stair_traversal_resampling_holds_episode_and_free_commands_use_normal_timer(self):
        env = fake_env(1000)
        env.scene.terrain.terrain_types[500:] = 0
        term = self.mdp.PointfootFullVelocityCommand()
        term._env = env
        term.cfg = NS(rel_standing_envs=.1, manual_control=False)
        term.vel_command_b = torch.zeros(1000, 3)
        term.is_standing_env = torch.zeros(1000, dtype=torch.bool)
        term.is_heading_env = torch.zeros(1000, dtype=torch.bool)
        term.time_left = torch.full((1000,), 4.)
        term._resample_command(torch.arange(1000))
        traverse = (term.vel_command_b[:500, 0].abs() >= .2) & (term.vel_command_b[:500, 1:].abs().amax(-1) == 0)
        self.assertTrue(bool((term.time_left[:500][traverse] == 36.).all()))
        self.assertTrue(bool((term.time_left[500:] == 4.).all()))

    def test_manual_control_reset_never_samples_random_history_commands(self):
        env = fake_env(4)
        term = self.mdp.PointfootFullVelocityCommand()
        term._env = env
        term.cfg = NS(manual_control=True, ranges=NS(lin_vel_x=(0., 0.), lin_vel_y=(0., 0.), ang_vel_z=(0., 0.)))
        term.vel_command_b = torch.ones(4, 3)
        term.is_standing_env = torch.ones(4, dtype=torch.bool)
        term.is_heading_env = torch.ones(4, dtype=torch.bool)
        term.time_left = torch.zeros(4)
        for _ in range(5):
            term._resample_command(torch.arange(4))
            self.assertTrue(bool((term.vel_command_b == 0).all()))
            self.assertTrue(bool((term.time_left == 1e9).all()))
            self.assertFalse(bool(term.is_heading_env.any()))
        self.assertFalse(hasattr(env, "_pointfoot_full_course"))

    def test_reset_randomizes_only_free_terrain_heading_and_keeps_spawn_height(self):
        env = fake_env(6000)
        columns = column_kind_indices(configs(), 10)
        env.scene.terrain.terrain_types[:] = torch.tensor([columns.index(k) for k in range(6)]).repeat(1000)
        env.scene.env_origins[:, 2] = .8
        self.mdp.reset_full_course_root(env, None)
        pose = env.scene["robot"].last_pose
        yaw = 2 * torch.atan2(pose[:, 6], pose[:, 3])
        flat = torch.arange(6000) % 6 == COURSE_KINDS.index("flat")
        self.assertGreater(float(yaw[flat].abs().max()), 3.)
        self.assertLessEqual(float(yaw[~flat].abs().max()), .08001)
        torch.testing.assert_close(pose[:, 2], torch.full((6000,), 1.14))

    def test_evaluation_can_narrow_free_yaw_range_for_every_course_kind(self):
        env = fake_env(600)
        columns = column_kind_indices(configs(), 10)
        env.scene.terrain.terrain_types[:] = torch.tensor([columns.index(k) for k in range(6)]).repeat(100)
        self.mdp.reset_full_course_root(env, None, free_yaw_range=(-.08, .08))
        pose = env.scene["robot"].last_pose
        yaw = 2 * torch.atan2(pose[:, 6], pose[:, 3])
        self.assertTrue(bool((yaw.abs() <= .08001).all()))
        # 非对称区间也应原样采样，仅影响 flat，不能改变其他五类赛道朝向。
        self.mdp.reset_full_course_root(env, None, free_yaw_range=(.2, .35))
        pose = env.scene["robot"].last_pose
        yaw = 2 * torch.atan2(pose[:, 6], pose[:, 3])
        free = torch.arange(600) % 6 == COURSE_KINDS.index("flat")
        self.assertTrue(bool(((yaw[free] >= .2) & (yaw[free] <= .35001)).all()))
        self.assertTrue(bool((yaw[~free].abs() <= .08001).all()))

    def test_parent_save_load_interfaces_preserve_existing_buffers(self):
        env = fake_env()
        history = self.mdp._history(env)
        history.iteration = 4000
        history.command_stage = 2
        state = self.mdp.full_course_state_dict(env)
        old_caps = env._pointfoot_course_caps
        history.command_stage = 0
        self.mdp.load_full_course_state_dict(env, state)
        self.assertIs(env._pointfoot_course_caps, old_caps)
        self.assertEqual(history.command_stage, 2)
        self.assertEqual(env.command_manager.get_term("base_velocity").cfg.ranges.lin_vel_x, (-.4, .4))
        json.dumps(self.mdp.full_summary(env), allow_nan=False)

    def test_hard_height_uses_shared_support_clearance_instead_of_center_ray(self):
        shared = ModuleType("_support_cpu.pointfoot_stairs")
        shared.pointfoot_support_clearance = lambda env: torch.tensor((.10, .12, .18, .32))
        with patch.object(self.mdp, "__package__", "_support_cpu"), patch.object(self.mdp, "__spec__", None), patch.dict(
            sys.modules, {"_support_cpu.pointfoot_stairs": shared},
        ):
            torch.testing.assert_close(self.mdp.full_low_support_height(None),
                                       torch.tensor((True, False, False, False)))


if __name__ == "__main__":
    unittest.main()
