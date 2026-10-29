"""Tests for forward and inverse kinematics."""

import math
import unittest

from robo_sim.kinematics import (
    end_effector_path_length,
    end_effector_position,
    forward_kinematics,
    inverse_kinematics,
    is_geometrically_reachable,
    is_within_joint_limits,
    target_error,
    total_joint_travel,
)
from robo_sim.models import InvalidInputError, Rectangle, RobotSpec


class ForwardKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.spec = RobotSpec(link1_length=170.0, link2_length=130.0)

    def test_both_angles_zero_extends_along_positive_x(self):
        geometry = forward_kinematics(self.spec, 0.0, 0.0)
        self.assertAlmostEqual(geometry.joint1[0], 170.0, places=9)
        self.assertAlmostEqual(geometry.joint1[1], 0.0, places=9)
        # x_tip = base_x + L1 + L2, y_tip = base_y
        self.assertAlmostEqual(geometry.end_effector[0], 300.0, places=9)
        self.assertAlmostEqual(geometry.end_effector[1], 0.0, places=9)

    def test_straight_up(self):
        geometry = forward_kinematics(self.spec, 90.0, 0.0)
        self.assertAlmostEqual(geometry.end_effector[0], 0.0, places=9)
        self.assertAlmostEqual(geometry.end_effector[1], 300.0, places=9)

    def test_right_angle_elbow(self):
        # Link 1 along +X, link 2 turned 90 degrees to point along +Y.
        geometry = forward_kinematics(self.spec, 0.0, 90.0)
        self.assertAlmostEqual(geometry.joint1, (170.0, 0.0))
        self.assertAlmostEqual(geometry.end_effector[0], 170.0, places=9)
        self.assertAlmostEqual(geometry.end_effector[1], 130.0, places=9)

    def test_folded_back_on_itself(self):
        geometry = forward_kinematics(self.spec, 0.0, 180.0)
        self.assertAlmostEqual(geometry.end_effector[0], 40.0, places=9)
        self.assertAlmostEqual(geometry.end_effector[1], 0.0, places=9)

    def test_second_angle_is_relative_to_link_one(self):
        # Absolute direction of link 2 is theta1 + theta2 = 60 degrees.
        geometry = forward_kinematics(self.spec, 20.0, 40.0)
        dx = geometry.end_effector[0] - geometry.joint1[0]
        dy = geometry.end_effector[1] - geometry.joint1[1]
        self.assertAlmostEqual(math.degrees(math.atan2(dy, dx)), 60.0, places=9)
        self.assertAlmostEqual(math.hypot(dx, dy), 130.0, places=9)

    def test_non_zero_base_offsets_everything(self):
        spec = RobotSpec(base_x=50.0, base_y=-25.0)
        geometry = forward_kinematics(spec, 0.0, 0.0)
        self.assertAlmostEqual(geometry.base, (50.0, -25.0))
        self.assertAlmostEqual(geometry.end_effector[0], 50.0 + 300.0, places=9)
        self.assertAlmostEqual(geometry.end_effector[1], -25.0, places=9)

    def test_link_segments_are_connected(self):
        geometry = forward_kinematics(self.spec, 35.0, -60.0)
        self.assertEqual(geometry.link1[1], geometry.link2[0])
        self.assertEqual(geometry.link1[0], geometry.base)
        self.assertEqual(geometry.link2[1], geometry.end_effector)

    def test_link_lengths_are_preserved_for_many_angles(self):
        for theta1 in range(-180, 181, 17):
            for theta2 in range(-150, 151, 23):
                geometry = forward_kinematics(self.spec, theta1, theta2)
                d1 = math.dist(geometry.base, geometry.joint1)
                d2 = math.dist(geometry.joint1, geometry.end_effector)
                self.assertAlmostEqual(d1, 170.0, places=9)
                self.assertAlmostEqual(d2, 130.0, places=9)

    def test_end_effector_position_matches_full_forward_kinematics(self):
        for theta1, theta2 in [(0, 0), (35, -60), (-120, 95), (179, -149)]:
            geometry = forward_kinematics(self.spec, theta1, theta2)
            quick = end_effector_position(self.spec, theta1, theta2)
            self.assertAlmostEqual(quick[0], geometry.end_effector[0], places=9)
            self.assertAlmostEqual(quick[1], geometry.end_effector[1], places=9)

    def test_non_finite_angles_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(InvalidInputError):
                forward_kinematics(self.spec, bad, 0.0)
            with self.assertRaises(InvalidInputError):
                forward_kinematics(self.spec, 0.0, bad)


class TargetErrorTests(unittest.TestCase):
    def test_zero_error_at_the_exact_tip(self):
        spec = RobotSpec()
        tip = end_effector_position(spec, 35.0, -60.0)
        self.assertAlmostEqual(target_error(spec, 35.0, -60.0, tip), 0.0, places=9)

    def test_error_is_euclidean_distance(self):
        spec = RobotSpec()
        tip = end_effector_position(spec, 0.0, 0.0)  # (300, 0)
        self.assertAlmostEqual(
            target_error(spec, 0.0, 0.0, (tip[0] + 3.0, tip[1] + 4.0)), 5.0, places=9
        )


class ReachabilityTests(unittest.TestCase):
    def setUp(self):
        self.spec = RobotSpec(link1_length=170.0, link2_length=130.0)

    def test_point_inside_annulus_is_reachable(self):
        self.assertTrue(is_geometrically_reachable(self.spec, (200.0, 0.0)))

    def test_point_beyond_full_extension_is_unreachable(self):
        self.assertFalse(is_geometrically_reachable(self.spec, (300.5, 0.0)))

    def test_point_inside_the_dead_zone_is_unreachable(self):
        # min reach is |170 - 130| = 40
        self.assertFalse(is_geometrically_reachable(self.spec, (39.0, 0.0)))

    def test_exact_boundaries_are_reachable(self):
        self.assertTrue(is_geometrically_reachable(self.spec, (300.0, 0.0)))
        self.assertTrue(is_geometrically_reachable(self.spec, (40.0, 0.0)))

    def test_equal_links_can_reach_their_own_base(self):
        spec = RobotSpec(link1_length=150.0, link2_length=150.0)
        self.assertEqual(spec.min_reach, 0.0)
        self.assertTrue(is_geometrically_reachable(spec, (0.0, 0.0)))

    def test_non_finite_target_is_not_reachable(self):
        self.assertFalse(is_geometrically_reachable(self.spec, (float("nan"), 0.0)))


class JointLimitTests(unittest.TestCase):
    def test_inside_limits(self):
        spec = RobotSpec()
        self.assertTrue(is_within_joint_limits(spec, 35.0, -60.0))

    def test_boundary_values_are_allowed(self):
        spec = RobotSpec()
        self.assertTrue(is_within_joint_limits(spec, -180.0, -150.0))
        self.assertTrue(is_within_joint_limits(spec, 180.0, 150.0))

    def test_outside_limits(self):
        spec = RobotSpec()
        self.assertFalse(is_within_joint_limits(spec, 0.0, 151.0))
        self.assertFalse(is_within_joint_limits(spec, -181.0, 0.0))


class InverseKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.spec = RobotSpec(link1_length=170.0, link2_length=130.0)

    def test_round_trip_through_forward_kinematics(self):
        """FK then IK must return to the same tip position."""
        poses = [
            (0.0, 45.0),
            (35.0, -60.0),
            (90.0, 30.0),
            (-45.0, -90.0),
            (120.0, 100.0),
            (-170.0, 20.0),
        ]
        for theta1, theta2 in poses:
            with self.subTest(pose=(theta1, theta2)):
                target = end_effector_position(self.spec, theta1, theta2)
                result = inverse_kinematics(self.spec, target)
                self.assertTrue(result.geometrically_reachable)
                self.assertTrue(result.success, result.message)
                error = target_error(
                    self.spec,
                    result.chosen.theta1_deg,
                    result.chosen.theta2_deg,
                    target,
                )
                self.assertLess(error, 1e-7)

    def test_original_pose_is_among_the_returned_solutions(self):
        theta1, theta2 = 35.0, -60.0
        target = end_effector_position(self.spec, theta1, theta2)
        result = inverse_kinematics(self.spec, target)
        matches = [
            s
            for s in result.solutions
            if abs(s.theta1_deg - theta1) < 1e-6 and abs(s.theta2_deg - theta2) < 1e-6
        ]
        self.assertEqual(len(matches), 1)

    def test_two_elbow_solutions_in_the_general_case(self):
        result = inverse_kinematics(self.spec, (200.0, 50.0))
        self.assertEqual(len(result.solutions), 2)
        elbows = {s.elbow for s in result.solutions}
        self.assertEqual(elbows, {"up", "down"})
        # Both put the tip in the same place.
        for solution in result.solutions:
            self.assertLess(solution.position_error, 1e-7)

    def test_single_solution_when_fully_extended(self):
        result = inverse_kinematics(self.spec, (300.0, 0.0))
        self.assertEqual(len(result.solutions), 1)
        self.assertAlmostEqual(result.solutions[0].theta2_deg, 0.0, places=6)

    def test_single_solution_at_the_inner_boundary(self):
        result = inverse_kinematics(self.spec, (40.0, 0.0))
        self.assertEqual(len(result.solutions), 1)
        self.assertAlmostEqual(abs(result.solutions[0].theta2_deg), 180.0, places=6)

    def test_unreachable_far_target_is_rejected_not_clamped(self):
        result = inverse_kinematics(self.spec, (500.0, 0.0))
        self.assertFalse(result.geometrically_reachable)
        self.assertFalse(result.success)
        self.assertEqual(result.solutions, ())
        self.assertIn("reach", result.message.lower())

    def test_unreachable_inner_target_is_rejected(self):
        result = inverse_kinematics(self.spec, (10.0, 0.0))
        self.assertFalse(result.geometrically_reachable)
        self.assertFalse(result.success)

    def test_near_boundary_target_still_solves(self):
        """A target a hair inside full extension must not trip the acos domain."""
        result = inverse_kinematics(self.spec, (299.9999999, 0.0))
        self.assertTrue(result.geometrically_reachable)
        self.assertTrue(result.success, result.message)

    def test_joint_limits_can_block_a_reachable_target(self):
        """Reachable and limit-blocked are reported as different failures."""
        narrow = RobotSpec(
            link1_length=170.0,
            link2_length=130.0,
            joint1_min=0.0,
            joint1_max=90.0,
            joint2_min=-10.0,
            joint2_max=10.0,
        )
        target = end_effector_position(narrow, 45.0, 120.0)
        result = inverse_kinematics(narrow, target)
        self.assertTrue(result.geometrically_reachable)
        self.assertFalse(result.success)
        self.assertTrue(result.limit_blocked)
        self.assertIn("joint limit", result.message)

    def test_obstacles_can_block_every_solution(self):
        """Reachable, legal, but every pose collides -- a third distinct case."""
        target = (250.0, 0.0)
        blocking = Rectangle(200.0, 300.0, -60.0, 60.0)
        result = inverse_kinematics(self.spec, target, obstacles=[blocking])
        self.assertTrue(result.geometrically_reachable)
        self.assertFalse(result.success)
        self.assertTrue(result.collision_blocked)
        self.assertIn("obstacle", result.message)

    def test_collision_free_is_none_when_no_obstacle_data_given(self):
        result = inverse_kinematics(self.spec, (200.0, 50.0))
        for solution in result.solutions:
            self.assertIsNone(solution.collision_free)

    def test_collision_free_is_boolean_when_obstacles_supplied(self):
        result = inverse_kinematics(self.spec, (200.0, 50.0), obstacles=[])
        for solution in result.solutions:
            self.assertIs(solution.collision_free, True)

    def test_prefers_the_solution_closest_to_the_current_pose(self):
        target = (200.0, 50.0)
        up = inverse_kinematics(self.spec, target, current=(0.0, 90.0))
        down = inverse_kinematics(self.spec, target, current=(60.0, -90.0))
        self.assertEqual(up.chosen.elbow, "up")
        self.assertEqual(down.chosen.elbow, "down")

    def test_non_finite_target_raises(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(InvalidInputError):
                inverse_kinematics(self.spec, (bad, 0.0))
            with self.assertRaises(InvalidInputError):
                inverse_kinematics(self.spec, (0.0, bad))

    def test_solutions_respect_a_shifted_base(self):
        spec = RobotSpec(base_x=-40.0, base_y=25.0)
        target = end_effector_position(spec, 20.0, -70.0)
        result = inverse_kinematics(spec, target)
        self.assertTrue(result.success)
        self.assertLess(result.chosen.position_error, 1e-7)


class PathMetricTests(unittest.TestCase):
    def setUp(self):
        self.spec = RobotSpec()

    def test_empty_and_single_pose_paths_have_zero_length(self):
        self.assertEqual(end_effector_path_length(self.spec, []), 0.0)
        self.assertEqual(end_effector_path_length(self.spec, [(0.0, 0.0)]), 0.0)
        self.assertEqual(total_joint_travel([]), 0.0)
        self.assertEqual(total_joint_travel([(0.0, 0.0)]), 0.0)

    def test_joint_travel_is_the_manhattan_sum(self):
        path = [(0.0, 0.0), (10.0, -5.0), (5.0, -5.0)]
        # |10| + |-5|  then  |-5| + |0|
        self.assertAlmostEqual(total_joint_travel(path), 20.0, places=9)

    def test_tip_path_length_of_a_pure_second_joint_rotation(self):
        """Rotating only joint 2 traces an arc of radius L2 about the elbow."""
        path = [(0.0, a) for a in range(0, 91, 1)]
        length = end_effector_path_length(self.spec, path)
        exact_arc = 130.0 * math.radians(90.0)
        # A 90-step polygonal approximation of the arc is slightly shorter.
        self.assertLess(length, exact_arc)
        self.assertAlmostEqual(length, exact_arc, delta=0.01)


if __name__ == "__main__":
    unittest.main()
