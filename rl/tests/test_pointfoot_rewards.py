"""点足 full-v3 奖励的纯 CPU 回归：不导入 Isaac Sim，不创建 CUDA 张量。

运行：CUDA_VISIBLE_DEVICES='' python -m unittest discover -s rl/tests -p test_pointfoot_rewards.py -v
"""

import math
import sys
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rl"))

from rp_wd_rl.supermimic.pointfoot_terrain import (  # noqa: E402
    FULL_REWARD_WEIGHTS,
    REWARD_WEIGHTS,
    PointfootTaskRewards,
)

LEGS = ("LF", "RF", "LB", "RB")
# 使用真实训练的按关节种类分组顺序，避免测试误把 12 维状态当作按腿排列。
NAMES = [f"{leg}_{kind}_JOINT" for kind in ("ABAD", "HIP", "KENN")
         for leg in ("LB", "LF", "RB", "RF")]
DEFAULT = torch.tensor([0.] * 4 + [.72] * 4 + [-1.41] * 4, dtype=torch.float64)
LIMITS = torch.tensor([[-2., 2.]] * 12, dtype=torch.float64)
URDF = ET.parse(ROOT / "description/RP_wd_pointfoot/urdf/RP_wd_pointfoot.urdf").getroot()


def axis_rotation(axis, angle):
    """独立 Rodrigues 矩阵，供 URDF 通用链和刚体有限差分作为参照。"""
    x, y, z = axis.unbind()
    zero = x * 0
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    eye = torch.eye(3, dtype=axis.dtype)
    return eye + angle.sin() * skew + (1 - angle.cos()) * (skew @ skew)


def urdf_feet(absolute, names=NAMES, collision_center=True):
    """直接逐关节累乘 URDF 变换，不使用待测模块的 FK 或 Jacobian。"""
    points = []
    for leg in LEGS:
        rotation, translation = torch.eye(3, dtype=absolute.dtype), absolute.new_zeros(3)
        for kind in ("ABAD", "HIP", "KENN", "FOOT_FIXED"):
            name = f"{leg}_{kind}_JOINT"
            joint = URDF.find(f"joint[@name='{name}']")
            origin = absolute.new_tensor([float(v) for v in joint.find("origin").attrib["xyz"].split()])
            translation = translation + rotation @ origin
            if joint.attrib["type"] != "fixed":
                axis = absolute.new_tensor([float(v) for v in joint.find("axis").attrib["xyz"].split()])
                rotation = rotation @ axis_rotation(axis, absolute[names.index(name)])
        if collision_center:
            origin = URDF.find(f"link[@name='{leg}_FOOT_LINK']/collision/origin")
            center = absolute.new_tensor([float(v) for v in origin.attrib["xyz"].split()])
            translation = translation + rotation @ center
        points.append(translation)
    return torch.stack(points)


class PointfootRewardTests(unittest.TestCase):
    def setUp(self):
        self.full = PointfootTaskRewards(NAMES, DEFAULT, LIMITS, profile="full")
        self.legacy = PointfootTaskRewards(NAMES, DEFAULT, LIMITS)

    def state(self, batch=1):
        state = torch.zeros(batch, 34, dtype=torch.float64)
        state[:, 0], state[:, 3] = .339, -1.
        return state

    def scan(self, batch=1, rough=False):
        scan = torch.full((batch, 273), .339, dtype=torch.float64)
        if rough:
            scan += torch.linspace(-.1, .1, 21, dtype=scan.dtype).repeat_interleave(13)
        return scan

    def terms(self, state, *, scan=None, command=None, contacts=None, force=None,
              rewards=None, action=None, previous=None, collision=None, terminated=None):
        batch = state.shape[0]
        zeros = state.new_zeros(batch)
        return (rewards or self.full).terms(
            state, state.new_zeros(batch, 12) if action is None else action,
            state.new_zeros(batch, 12) if previous is None else previous,
            state.new_zeros(batch, 3) if command is None else command,
            self.scan(batch) if scan is None else scan,
            zeros if collision is None else collision,
            zeros if terminated is None else terminated, contacts=contacts, force=force,
        )

    def test_public_profiles_and_original_weights(self):
        self.assertEqual(self.legacy.profile, "legacy")
        self.assertEqual(self.legacy.weights, REWARD_WEIGHTS)
        self.assertEqual(self.full.weights, FULL_REWARD_WEIGHTS)
        self.assertEqual((self.legacy.hard_min_height, self.full.hard_min_height), (.17, .12))
        self.assertEqual((self.legacy.fall_angle, self.full.fall_angle),
                         (math.radians(55), math.radians(60)))
        self.assertEqual(REWARD_WEIGHTS["lin_vel_z_l2"], -.25)
        self.assertEqual(REWARD_WEIGHTS["leg_posture"], -.05)
        self.assertNotIn("base_height", REWARD_WEIGHTS)
        self.assertEqual(set(self.terms(self.state())), set(FULL_REWARD_WEIGHTS))
        for name in FULL_REWARD_WEIGHTS:
            self.assertFalse(any(word in name for word in ("torque", "rolling", "joint_velocity")))
        with self.assertRaisesRegex(ValueError, "profile"):
            PointfootTaskRewards(NAMES, DEFAULT, LIMITS, profile="invalid")

    def test_legacy_matches_prechange_reward_snapshot(self):
        generator = torch.Generator().manual_seed(21)
        state = torch.randn(3, 34, generator=generator, dtype=torch.float64) * .15
        state[:, 0] = torch.tensor([.339, .19, .48])
        state[:, 1:4] = torch.tensor([0., 0., -1.])
        action = torch.randn(3, 12, generator=generator, dtype=torch.float64) * .2
        previous = torch.randn(3, 12, generator=generator, dtype=torch.float64) * .2
        command = state.new_tensor([[.3, 0., 0.], [0., -.2, .1], [-.3, .1, -.1]])
        args = (state, action, previous, command,
                torch.linspace(.2, .4, 273, dtype=state.dtype).expand(3, -1),
                state.new_tensor([0., 1., 0.]), state.new_tensor([0., 0., 1.]))
        expected = state.new_tensor([2.6356480745188797, .5513669336314397, -98.33322283177705])
        torch.testing.assert_close(self.legacy.reward(*args), expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(self.legacy.reward(*args, contacts=state.new_ones(3, 4),
                                                     force=state.new_ones(3, 4)), expected)
        torch.testing.assert_close(self.legacy.feet_body(self.state()),
                                   urdf_feet(DEFAULT, collision_center=False)[None])

    def test_height_interval_has_exact_zero_value_and_gradient(self):
        state = self.state(7)
        state[:, 0] = state.new_tensor([.22, .23, .27, .339, .39, .41, .42])
        state.requires_grad_()
        cost = self.terms(state)["base_height"]
        self.assertTrue(torch.equal(cost, torch.zeros_like(cost)))
        cost.sum().backward()
        self.assertTrue(torch.equal(state.grad, torch.zeros_like(state)))

    def test_outside_height_interval_has_restoring_gradient(self):
        state = self.state(2)
        state[:, 0] = state.new_tensor([.12, .52])
        state.requires_grad_()
        cost = self.terms(state)["base_height"]
        torch.testing.assert_close(cost, state.new_ones(2))
        cost.sum().backward()
        torch.testing.assert_close(state.grad[:, 0], state.new_tensor([-20., 20.]))
        self.assertLess(abs(self.full.weights["base_height"]), .25)

    def test_rough_height_interval_relaxes_only_for_nearby_ground(self):
        state = self.state(4)
        state[:, 0] = state.new_tensor([.20, .21, .43, .44])
        self.assertTrue((self.terms(state)["base_height"] > 0).all())
        self.assertTrue((self.terms(state, scan=self.scan(4, rough=True))["base_height"] == 0).all())
        far = self.scan(4).reshape(4, 21, 13)
        far[:, 18:, :] -= .5
        torch.testing.assert_close(self.terms(state, scan=far.flatten(1))["base_height"],
                                   self.terms(state)["base_height"])

    def test_body_underpass_step_jump_does_not_change_support_clearance(self):
        before, after = self.state(), self.state()
        after[:, 0] -= .20
        scan_before, scan_after = self.scan(), self.scan()
        scan_after.reshape(1, 21, 13)[:, 10, 6] -= .20
        contacts = torch.ones(1, 4, dtype=torch.bool)
        for state, scan in ((before, scan_before), (after, scan_after)):
            result = self.full.support_clearance(state, scan, contacts)
            torch.testing.assert_close(result, state.new_tensor([.339]))
            self.assertEqual(self.terms(state, scan=scan, contacts=contacts)["base_height"].item(), 0.)
        self.assertGreater(self.terms(after, scan=scan_after)["base_height"].item(), 0.)

    def test_contact_and_load_weight_the_ground_under_each_foot(self):
        scan = self.scan().reshape(1, 21, 13)
        scan[:, 11:, :] -= .10  # 前脚站在高 10 cm 的踏面，后脚仍在平地。
        scan = scan.flatten(1)
        state = self.state()
        front = state.new_tensor([[1., 1., 0., 0.]])
        torch.testing.assert_close(self.full.support_clearance(state, scan, front), state.new_tensor([.239]))
        torch.testing.assert_close(self.full.support_clearance(state, scan, 1 - front), state.new_tensor([.339]))
        load = state.new_tensor([[3., 3., 1., 1.]])
        torch.testing.assert_close(self.full.support_clearance(state, scan, force=load), state.new_tensor([.264]))
        torch.testing.assert_close(self.full.support_clearance(state, scan, front, load), state.new_tensor([.239]))
        probability = state.new_tensor([[.25, .25, .75, .75]])
        torch.testing.assert_close(self.full.support_clearance(state, scan, probability), state.new_tensor([.314]))

    def test_missing_zero_or_nonfinite_support_falls_back_per_environment(self):
        state = self.state(3)
        state[:, 0] = state.new_tensor([.18, .24, .47])
        contacts = state.new_tensor([[0., 0., 0., 0.], [1., 1., 1., 1.], [float("nan")] * 4])
        force = state.new_tensor([[1.] * 4, [0.] * 4, [float("inf")] * 4])
        for kwargs in ({}, {"contacts": contacts, "force": force}):
            torch.testing.assert_close(self.full.support_clearance(state, self.scan(3), **kwargs), state[:, 0])
        with self.assertRaisesRegex(ValueError, "batch,4"):
            self.full.support_clearance(state, self.scan(3), contacts=state.new_ones(3, 3))

    def test_missing_scan_keeps_reward_and_fallback_gradients_finite(self):
        state = self.state()
        state[:, 0] = .18
        state.requires_grad_()
        scan = self.scan().fill_(float("nan")).requires_grad_()
        terms = self.terms(state, scan=scan, contacts=state.new_ones(1, 4))
        self.assertTrue(all(torch.isfinite(value).all() for value in terms.values()))
        sum(self.full.weights[name] * value.sum() for name, value in terms.items()).backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertTrue(torch.isfinite(scan.grad).all())
        self.assertGreater(state.grad[0, 0].item(), 0.)

    def test_support_clearance_is_differentiable_in_scan_contacts_and_load(self):
        state = self.state().requires_grad_()
        scan = self.scan(rough=True).requires_grad_()
        contacts = state.new_tensor([[.7, .8, .9, .6]], requires_grad=True)
        force = state.new_tensor([[.1, .2, .3, .4]], requires_grad=True)
        fn = lambda s, h, c, f: self.full.support_clearance(s, h, c, f)
        self.assertTrue(torch.autograd.gradcheck(fn, (state, scan, contacts, force)))
        fn(state, scan, contacts, force).sum().backward()
        self.assertEqual(state.grad[0, 0].item(), 0.)
        torch.testing.assert_close(scan.grad.sum(), state.new_tensor(1.))

    def test_real_and_imagined_terms_use_the_same_weighted_reward(self):
        state = self.state(2)
        state[:, 4], state[:, 10], state[:, 22] = .2, .15, .5
        action, previous = state.new_full((2, 12), .1), state.new_zeros(2, 12)
        command = state.new_tensor([[.3, 0., 0.], [0., 0., .2]])
        for contacts in (state.new_ones(2, 4), state.new_full((2, 4), .7)):
            force = state.new_full((2, 4), .1)
            args = (state, action, previous, command, self.scan(2, rough=True),
                    state.new_tensor([1., 0.]), state.new_tensor([0., .2]))
            terms = self.full.terms(*args, contacts=contacts, force=force)
            expected = sum(self.full.weights[name] * value for name, value in terms.items())
            torch.testing.assert_close(self.full.reward(*args, contacts=contacts, force=force), expected)
            torch.testing.assert_close(self.full.reward(*args, contacts, force), expected)

    def test_ten_degree_motion_is_free_and_terrain_relaxes_flexion(self):
        state = self.state(4)
        for i, degrees in enumerate((0, 9, 10, 25)):
            angle = math.radians(degrees)
            state[i, 1:4] = state.new_tensor([math.sin(angle), 0., -math.cos(angle)])
        upright = self.terms(state)["upright"]
        torch.testing.assert_close(upright[:3], state.new_ones(3), atol=1e-14, rtol=0)
        self.assertLess(upright[3].item(), 1.)
        state[:, 10:22] = .1
        flat = self.terms(state)["leg_posture"]
        rough = self.terms(state, scan=self.scan(4, rough=True))["leg_posture"]
        self.assertTrue((rough < flat).all())
        self.assertGreater(rough[0].item(), .08)  # 外展仍保留约束。
        for key in ("lin_vel_z_l2", "ang_vel_xy_l2", "leg_posture"):
            self.assertLess(abs(self.full.weights[key]), abs(self.legacy.weights[key]))

    def test_directional_progress_is_signed_capped_and_has_no_yaw_lift_credit(self):
        cases = [([.3, 0., 0.], [.3, 0.], 1.), ([.3, 0., 0.], [-.3, 0.], -1.),
                 ([.3, 0., 0.], [9., 0.], 1.), ([.3, 0., 0.], [0., .3], 0.),
                 ([-.3, 0., 0.], [-.3, 0.], 1.), ([0., .3, 0.], [0., .3], 1.),
                 ([0., 0., .5], [0., 0.], 0.), ([0., 0., 0.], [0., 0.], 0.)]
        state = self.state(len(cases))
        state[:, 4:6] = state.new_tensor([item[1] for item in cases])
        state[:, 6:10] = .8
        state[:, 18:22] = -.7
        command = state.new_tensor([item[0] for item in cases])
        terms = self.terms(state, scan=self.scan(len(cases), rough=True), command=command)
        torch.testing.assert_close(terms["obstacle_progress"], state.new_tensor([item[2] for item in cases]))

    def test_low_step_clearance_is_directional_and_only_removes_a_penalty(self):
        state, scan = self.state(), self.scan().reshape(1, 21, 13)
        scan[:, 14:, :] -= .04
        scan = scan.flatten(1)
        forward = state.new_tensor([[.3, 0., 0.]])
        low = self.terms(state, scan=scan, command=forward)["stair_clearance"]
        self.assertGreater(low.item(), 0.)
        raised = state.clone()
        raised[:, 10 + self.full.ids[:2, 2]] = -.7
        lifted = self.terms(raised, scan=scan, command=forward)
        self.assertLess(lifted["stair_clearance"].item(), low.item())
        self.assertLessEqual((self.full.weights["stair_clearance"] * lifted["stair_clearance"]).item(), 0.)
        for command in (forward * 0, -forward):
            self.assertEqual(self.terms(state, scan=scan, command=command)["stair_clearance"].item(), 0.)
        self.assertEqual(lifted["obstacle_progress"].item(), 0.)

    def test_stop_cost_freezes_body_and_feet_only_at_zero_command(self):
        quiet = self.terms(self.state())["standstill"]
        self.assertEqual(quiet.item(), 0.)
        state = self.state()
        state[:, 4:10], state[:, 22:34] = .2, .7
        stopping = self.terms(state)["standstill"]
        self.assertGreater(stopping.item(), 0.)
        for command in ([[.3, 0., 0.]], [[0., 0., .3]]):
            self.assertEqual(self.terms(state, command=state.new_tensor(command))["standstill"].item(), 0.)

    def test_fk_matches_independent_urdf_chain_and_collision_centers(self):
        generator = torch.Generator().manual_seed(42)
        state = self.state(4)
        state[:, 10:22] = torch.randn(4, 12, generator=generator, dtype=state.dtype) * .3
        expected = torch.stack([urdf_feet(q) for q in state[:, 10:22] + DEFAULT])
        torch.testing.assert_close(self.full.feet_body(state), expected, atol=1e-12, rtol=1e-12)
        stand_height = -self.full.feet_body(self.state())[0, :, 2] + self.full.radius
        torch.testing.assert_close(stand_height, torch.full_like(stand_height, .3403), atol=2e-5, rtol=0)

    def test_jacobian_matches_urdf_autograd_for_all_legs_and_joint_order(self):
        state = self.state()
        state[:, 10:22] = torch.linspace(-.25, .2, 12, dtype=state.dtype)
        actual = self.full.feet_jacobian(state)[0]
        reference = torch.autograd.functional.jacobian(urdf_feet, state[0, 10:22] + DEFAULT)
        for leg in range(4):
            torch.testing.assert_close(actual[leg], reference[leg][:, self.full.ids[leg]], atol=1e-12, rtol=1e-12)
            other = [i for i in range(12) if i not in self.full.ids[leg].tolist()]
            self.assertTrue((reference[leg][:, other] == 0).all())

    def test_foot_velocity_matches_rigid_body_and_joint_finite_difference(self):
        state = self.state()
        state[:, 4:10] = state.new_tensor([[.3, -.2, .1, .4, -.3, .5]])
        state[:, 10:22] = torch.linspace(-.2, .2, 12, dtype=state.dtype)
        state[:, 22:34] = torch.linspace(-1., 1., 12, dtype=state.dtype)
        rotation = (axis_rotation(state.new_tensor([0., 1., 0.]), state.new_tensor(-.2))
                    @ axis_rotation(state.new_tensor([1., 0., 0.]), state.new_tensor(.15)))
        state[:, 1:4] = rotation.T @ state.new_tensor([0., 0., -1.])
        epsilon, omega = 1e-6, state[0, 7:10]
        endpoints = []
        for dt in (-epsilon, epsilon):
            turned = rotation @ axis_rotation(omega / omega.norm(), omega.norm() * dt)
            point = urdf_feet(DEFAULT + state[0, 10:22] + dt * state[0, 22:34])
            endpoints.append(point @ turned.T + dt * (rotation @ state[0, 4:7]))
        finite_difference = (endpoints[1] - endpoints[0]) / (2 * epsilon)
        torch.testing.assert_close(self.full.feet_velocity(state)[0], finite_difference, atol=1e-9, rtol=1e-8)

    def test_slip_allows_joint_motion_that_cancels_body_velocity(self):
        state = self.state()
        state[:, 4], state[:, 9] = .3, .2
        point = self.full.feet_body(state)
        base_velocity = state[:, None, 4:7] + torch.linalg.cross(state[:, None, 7:10].expand_as(point), point)
        qd = torch.linalg.solve(self.full.feet_jacobian(state), -base_velocity.unsqueeze(-1)).squeeze(-1)
        contacts, command = state.new_ones(1, 4), state.new_tensor([[.3, 0., .2]])
        sliding = self.terms(state, contacts=contacts, command=command)["foot_slip"]
        self.assertGreater(sliding.item(), .05)
        state[:, 22 + self.full.ids] = qd
        matched = self.terms(state, contacts=contacts, command=command)
        self.assertGreater(state[:, 22:34].square().sum().item(), 1.)
        self.assertLess(matched["foot_slip"].item(), 1e-25)
        self.assertEqual(matched["standstill"].item(), 0.)

    def test_swing_foot_motion_is_not_slip_and_load_can_select_support(self):
        state = self.state()
        state[:, 22 + self.full.ids[0]] = state.new_tensor([1., 2., 3.])
        supporting = state.new_tensor([[0., 1., 1., 1.]])
        self.assertEqual(self.terms(state, contacts=supporting)["foot_slip"].item(), 0.)
        self.assertEqual(self.terms(state, force=supporting)["foot_slip"].item(), 0.)
        self.assertGreater(self.terms(state, contacts=1 - supporting)["foot_slip"].item(), 0.)
        self.assertEqual(self.terms(state)["foot_slip"].item(), 0.)

    def test_slip_has_correct_body_translation_gradient(self):
        state = self.state()
        state[:, 4] = .3
        state.requires_grad_()
        slip = self.terms(state, contacts=state.new_ones(1, 4))["foot_slip"]
        torch.testing.assert_close(slip, state.new_tensor([.09]))
        slip.sum().backward()
        self.assertAlmostEqual(state.grad[0, 4].item(), .6)
        self.assertTrue(torch.isfinite(state.grad).all())

    def test_float32_full_reward_has_finite_state_action_and_contact_gradients(self):
        rewards = PointfootTaskRewards(NAMES, DEFAULT.float(), LIMITS.float(), profile="full")
        state = self.state(2).float()
        state[:, 4], state[:, 10:22], state[:, 22:34] = .2, .1, .3
        state.requires_grad_()
        action = state.new_full((2, 12), .1, requires_grad=True)
        contacts = state.new_full((2, 4), .8, requires_grad=True)
        force = state.new_tensor([[.1, .2, .3, .4]] * 2, requires_grad=True)
        reward = rewards.reward(state, action, state.new_zeros(2, 12),
                                state.new_tensor([[.3, 0., 0.]] * 2), self.scan(2, rough=True).float(),
                                state.new_zeros(2), state.new_zeros(2), contacts=contacts, force=force)
        reward.sum().backward()
        for value in (state, action, contacts, force):
            self.assertEqual(value.device.type, "cpu")
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(state.grad.abs().sum().item(), 0.)
        self.assertGreater(action.grad.abs().sum().item(), 0.)

    def test_tracking_limits_collision_and_termination_remain_active(self):
        state = self.state()
        nominal = self.terms(state)
        state[:, 10], state[:, 4], state[:, 9] = 3., .5, .4
        bad = self.terms(state, collision=state.new_ones(1, 1), terminated=state.new_ones(1, 1))
        for name in ("track_linear_velocity", "track_angular_velocity"):
            self.assertLess(bad[name].item(), nominal[name].item())
        self.assertGreater(bad["dof_pos_limits"].item(), 0.)
        self.assertEqual(bad["nonfoot_collision"].item(), 1.)
        self.assertEqual(bad["terminated"].item(), 1.)
        self.assertEqual(self.full.weights["terminated"], -100.)


if __name__ == "__main__":
    unittest.main()
