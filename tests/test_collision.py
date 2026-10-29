"""Tests for robot-configuration and transition validity."""

import unittest

from robo_sim.collision import (
    configuration_collides,
    geometry_collides,
    is_configuration_valid,
    is_transition_valid,
    transition_samples,
    validate_path,
)
from robo_sim.kinematics import forward_kinematics
from robo_sim.models import Rectangle, RobotSpec


class ConfigurationCollisionTests(unittest.TestCase):
    def setUp(self):
        # Links of 170 and 130 with a 7.5 unit radius; at (0, 0) the arm lies
        # along the +X axis from (0,0) to (300,0).
        self.spec = RobotSpec(link_radius=7.5, end_effector_radius=9.0)

    def test_empty_workspace_is_collision_free(self):
        self.assertFalse(configuration_collides(self.spec, 0.0, 0.0, []))
        result = is_configuration_valid(self.spec, 35.0, -60.0, [])
        self.assertTrue(result.valid)

    def test_obstacle_far_away_does_not_collide(self):
        far = Rectangle(-300.0, -200.0, -300.0, -200.0)
        self.assertFalse(configuration_collides(self.spec, 0.0, 0.0, [far]))

    def test_link_one_crossing_an_obstacle(self):
        # Straddles the +X axis at x = 80, well inside link 1.
        blocking = Rectangle(60.0, 100.0, -20.0, 20.0)
        self.assertTrue(configuration_collides(self.spec, 0.0, 0.0, [blocking]))

    def test_link_two_crossing_an_obstacle(self):
        # x = 200 is past the elbow at x = 170, so only link 2 is there.
        blocking = Rectangle(190.0, 230.0, -20.0, 20.0)
        self.assertTrue(configuration_collides(self.spec, 0.0, 0.0, [blocking]))

    def test_link_thickness_causes_a_collision_the_centreline_misses(self):
        """The decisive test for modelling links as capsules rather than lines.

        The rectangle sits 5 units above the arm's centre line. A zero-width
        line would miss it; a link of radius 7.5 cannot.
        """
        near_miss = Rectangle(80.0, 120.0, 5.0, 40.0)
        geometry = forward_kinematics(self.spec, 0.0, 0.0)

        thin = RobotSpec(link_radius=1.0, end_effector_radius=1.0)
        self.assertIsNone(geometry_collides(geometry, thin, [near_miss]))

        thick = RobotSpec(link_radius=7.5, end_effector_radius=9.0)
        self.assertIsNotNone(geometry_collides(geometry, thick, [near_miss]))

    def test_end_effector_collision(self):
        """An obstacle reachable only by the gripper disc at the tip."""
        # The tip is at (300, 0). This box starts 8 units beyond it: outside
        # the 7.5 unit link radius, but inside the 9 unit gripper radius, so
        # only the end-effector disc can reach it.
        tip_box = Rectangle(308.0, 340.0, -4.0, 4.0)
        self.assertTrue(configuration_collides(self.spec, 0.0, 0.0, [tip_box]))

        # With a gripper no thicker than the links, the same pose is clear.
        small_gripper = RobotSpec(link_radius=7.5, end_effector_radius=7.5)
        self.assertFalse(configuration_collides(small_gripper, 0.0, 0.0, [tip_box]))

    def test_boundary_contact_counts_as_a_collision(self):
        """Touching exactly at the collision radius is rejected, not allowed."""
        # Box edge exactly 7.5 units above the centre line of the arm.
        touching = Rectangle(80.0, 120.0, 7.5, 40.0)
        self.assertTrue(configuration_collides(self.spec, 0.0, 0.0, [touching]))

        clear = Rectangle(80.0, 120.0, 7.6, 40.0)
        self.assertFalse(configuration_collides(self.spec, 0.0, 0.0, [clear]))

    def test_first_touched_obstacle_is_reported(self):
        geometry = forward_kinematics(self.spec, 0.0, 0.0)
        hit = Rectangle(60.0, 100.0, -20.0, 20.0)
        self.assertEqual(geometry_collides(geometry, self.spec, [hit]), hit)

    def test_joint_limits_are_part_of_validity(self):
        result = is_configuration_valid(self.spec, 0.0, 170.0, [])
        self.assertFalse(result.valid)
        self.assertIn("limits", result.reason)

    def test_collision_reason_names_the_obstacle(self):
        blocking = Rectangle(60.0, 100.0, -20.0, 20.0)
        result = is_configuration_valid(self.spec, 0.0, 0.0, [blocking])
        self.assertFalse(result.valid)
        self.assertIn("collides", result.reason)
        self.assertIn("60", result.reason)

    def test_non_finite_angles_are_rejected_not_raised(self):
        result = is_configuration_valid(self.spec, float("nan"), 0.0, [])
        self.assertFalse(result.valid)
        self.assertIn("finite", result.reason)

    def test_validation_result_is_falsy_when_invalid(self):
        self.assertTrue(bool(is_configuration_valid(self.spec, 0.0, 0.0, [])))
        self.assertFalse(bool(is_configuration_valid(self.spec, 0.0, 200.0, [])))


class TransitionSampleTests(unittest.TestCase):
    def test_samples_include_both_endpoints(self):
        samples = transition_samples((0.0, 0.0), (10.0, -4.0), max_step_deg=1.0)
        self.assertEqual(samples[0], (0.0, 0.0))
        self.assertAlmostEqual(samples[-1][0], 10.0)
        self.assertAlmostEqual(samples[-1][1], -4.0)

    def test_step_never_exceeds_the_limit(self):
        samples = transition_samples((0.0, 0.0), (10.0, -4.0), max_step_deg=1.0)
        for (a1, a2), (b1, b2) in zip(samples, samples[1:]):
            self.assertLessEqual(max(abs(b1 - a1), abs(b2 - a2)), 1.0 + 1e-9)

    def test_identical_poses_give_one_sample(self):
        self.assertEqual(transition_samples((5.0, 5.0), (5.0, 5.0)), [(5.0, 5.0)])

    def test_finer_step_gives_more_samples(self):
        coarse = transition_samples((0.0, 0.0), (10.0, 0.0), max_step_deg=5.0)
        fine = transition_samples((0.0, 0.0), (10.0, 0.0), max_step_deg=1.0)
        self.assertEqual(len(coarse), 3)
        self.assertEqual(len(fine), 11)


class TransitionValidityTests(unittest.TestCase):
    def setUp(self):
        self.spec = RobotSpec(link_radius=7.5, end_effector_radius=9.0)

    def test_valid_move_in_an_empty_workspace(self):
        result = is_transition_valid(self.spec, (0.0, 0.0), (45.0, -30.0), [])
        self.assertTrue(result.valid)

    def test_endpoints_valid_but_the_motion_sweeps_through_an_obstacle(self):
        """The case that makes endpoint-only checking unsafe.

        The arm is clear at -40 degrees and clear at +40 degrees, but rotating
        between them drags both links straight through a box sitting on the
        +X axis.
        """
        blocking = Rectangle(100.0, 160.0, -12.0, 12.0)

        start = (-40.0, 0.0)
        end = (40.0, 0.0)
        self.assertTrue(is_configuration_valid(self.spec, *start, [blocking]).valid)
        self.assertTrue(is_configuration_valid(self.spec, *end, [blocking]).valid)

        result = is_transition_valid(self.spec, start, end, [blocking])
        self.assertFalse(result.valid)
        self.assertIn("passes through", result.reason)

    def test_the_same_move_is_fine_without_the_obstacle(self):
        self.assertTrue(
            is_transition_valid(self.spec, (-40.0, 0.0), (40.0, 0.0), []).valid
        )

    def test_transition_from_an_invalid_start_is_rejected(self):
        blocking = Rectangle(60.0, 100.0, -20.0, 20.0)
        result = is_transition_valid(self.spec, (0.0, 0.0), (90.0, 0.0), [blocking])
        self.assertFalse(result.valid)

    def test_joint_limit_violation_at_an_endpoint_is_caught_without_obstacles(self):
        result = is_transition_valid(self.spec, (0.0, 0.0), (0.0, 200.0), [])
        self.assertFalse(result.valid)


class PathValidationTests(unittest.TestCase):
    def setUp(self):
        self.spec = RobotSpec(link_radius=7.5, end_effector_radius=9.0)

    def test_empty_path_is_invalid(self):
        self.assertFalse(validate_path(self.spec, [], []).valid)

    def test_single_valid_pose_is_a_valid_path(self):
        result = validate_path(self.spec, [(0.0, 0.0)], [])
        self.assertTrue(result.valid)

    def test_interpolated_path_is_checked_at_every_step(self):
        path = [(float(a), -60.0) for a in range(0, 51, 5)]
        self.assertTrue(validate_path(self.spec, path, []).valid)

    def test_a_path_whose_middle_step_sweeps_an_obstacle_is_rejected(self):
        blocking = Rectangle(100.0, 160.0, -12.0, 12.0)
        # Every listed pose is clear, but step 2 crosses the box.
        path = [(-60.0, 0.0), (-40.0, 0.0), (40.0, 0.0), (60.0, 0.0)]
        for pose in path:
            self.assertTrue(is_configuration_valid(self.spec, *pose, [blocking]).valid)

        result = validate_path(self.spec, path, [blocking])
        self.assertFalse(result.valid)
        self.assertIn("Step 2", result.reason)

    def test_invalid_start_pose_is_reported_as_such(self):
        blocking = Rectangle(60.0, 100.0, -20.0, 20.0)
        result = validate_path(self.spec, [(0.0, 0.0), (90.0, 0.0)], [blocking])
        self.assertFalse(result.valid)
        self.assertIn("Start pose", result.reason)

    def test_successful_validation_states_the_sampling_step(self):
        result = validate_path(self.spec, [(0.0, 0.0), (10.0, 0.0)], [])
        self.assertTrue(result.valid)
        self.assertIn("transitions validated", result.reason)


if __name__ == "__main__":
    unittest.main()
