"""Tests for the joint-space graph, A* and Dijkstra."""

import math
import unittest

from robo_sim.collision import validate_path
from robo_sim.kinematics import end_effector_position, target_error
from robo_sim.models import (
    PLAN_NO_GOAL_STATE,
    PLAN_NO_PATH,
    PLAN_OK,
    PLAN_START_INVALID,
    PLAN_TARGET_UNREACHABLE,
    Rectangle,
    RobotSpec,
)
from robo_sim.path_planning import (
    GridTooLargeError,
    JointGrid,
    PlanningProblem,
    plan,
    plan_astar,
    plan_dijkstra,
    resolution_divides_range,
)

START = (35.0, -60.0)


def make_problem(obstacles=(), target=(150.0, -200.0), resolution=5.0, tolerance=12.0):
    return PlanningProblem(
        spec=RobotSpec(),
        obstacles=obstacles,
        target=target,
        resolution_deg=resolution,
        target_tolerance=tolerance,
    )


class JointGridTests(unittest.TestCase):
    def setUp(self):
        self.spec = RobotSpec()

    def test_grid_spans_the_joint_limits_inclusively(self):
        grid = JointGrid(self.spec, 5.0)
        self.assertEqual(grid.angles1[0], -180.0)
        self.assertEqual(grid.angles1[-1], 180.0)
        self.assertEqual(grid.angles2[0], -150.0)
        self.assertEqual(grid.angles2[-1], 150.0)

    def test_state_count_for_each_supported_resolution(self):
        # Joint 1 spans 360 degrees, joint 2 spans 300.
        for resolution, expected in ((3.0, 121 * 101), (5.0, 73 * 61), (10.0, 37 * 31)):
            grid = JointGrid(self.spec, resolution)
            self.assertEqual(grid.state_count, expected)
            self.assertTrue(grid.aligned)

    def test_supported_resolutions_divide_both_joint_ranges(self):
        for resolution in (3.0, 5.0, 10.0):
            self.assertTrue(resolution_divides_range(-180.0, 180.0, resolution))
            self.assertTrue(resolution_divides_range(-150.0, 150.0, resolution))

    def test_misaligned_resolution_is_flagged_but_keeps_the_exact_limits(self):
        grid = JointGrid(self.spec, 7.0)
        self.assertFalse(grid.aligned)
        self.assertEqual(grid.angles1[0], -180.0)
        self.assertEqual(grid.angles1[-1], 180.0)

    def test_snap_finds_the_nearest_grid_angles(self):
        grid = JointGrid(self.spec, 5.0)
        self.assertEqual(grid.angles(grid.snap(36.0, -61.0)), (35.0, -60.0))
        self.assertEqual(grid.angles(grid.snap(-2.4, 3.1)), (0.0, 5.0))

    def test_snapped_state_is_exact_when_already_on_the_grid(self):
        grid = JointGrid(self.spec, 5.0)
        self.assertEqual(grid.angles(grid.snap(35.0, -60.0)), (35.0, -60.0))

    def test_interior_state_has_eight_neighbours(self):
        grid = JointGrid(self.spec, 5.0)
        self.assertEqual(len(list(grid.neighbours((10, 10)))), 8)

    def test_corner_state_has_three_neighbours(self):
        grid = JointGrid(self.spec, 5.0)
        self.assertEqual(len(list(grid.neighbours((0, 0)))), 3)
        last = (grid.size1 - 1, grid.size2 - 1)
        self.assertEqual(len(list(grid.neighbours(last))), 3)

    def test_the_two_joint_one_extremes_are_not_connected(self):
        """-180 and +180 are the same direction but deliberately not joined."""
        grid = JointGrid(self.spec, 5.0)
        low = (0, 10)
        high = (grid.size1 - 1, 10)
        self.assertNotIn(high, set(grid.neighbours(low)))
        self.assertNotIn(low, set(grid.neighbours(high)))

    def test_too_fine_a_resolution_is_refused(self):
        with self.assertRaises(GridTooLargeError):
            JointGrid(self.spec, 0.5)

    def test_non_positive_resolution_is_refused(self):
        with self.assertRaises(ValueError):
            JointGrid(self.spec, 0.0)


class EdgeCostTests(unittest.TestCase):
    def test_straight_move_costs_one_resolution_step(self):
        problem = make_problem(resolution=5.0)
        self.assertAlmostEqual(problem.edge_cost((10, 10), (11, 10)), 5.0)
        self.assertAlmostEqual(problem.edge_cost((10, 10), (10, 11)), 5.0)

    def test_diagonal_move_costs_the_euclidean_amount(self):
        problem = make_problem(resolution=5.0)
        self.assertAlmostEqual(
            problem.edge_cost((10, 10), (11, 11)), 5.0 * math.sqrt(2.0)
        )

    def test_cost_is_symmetric(self):
        problem = make_problem()
        self.assertAlmostEqual(
            problem.edge_cost((4, 7), (5, 8)), problem.edge_cost((5, 8), (4, 7))
        )

    def test_cost_uses_real_angles_not_index_differences(self):
        problem = make_problem(resolution=10.0)
        self.assertAlmostEqual(problem.edge_cost((3, 3), (4, 3)), 10.0)


class GoalStateTests(unittest.TestCase):
    def test_multiple_goal_states_for_a_normal_target(self):
        problem = make_problem()
        goals = problem.goal_states()
        self.assertGreater(len(goals), 1)

    def test_every_goal_state_is_within_tolerance_and_valid(self):
        problem = make_problem(tolerance=12.0)
        spec = problem.spec
        for state in problem.goal_states():
            theta1, theta2 = problem.grid.angles(state)
            self.assertLessEqual(
                target_error(spec, theta1, theta2, problem.target), 12.0 + 1e-9
            )
            self.assertTrue(problem.is_state_valid(state))

    def test_goal_set_covers_both_elbow_configurations(self):
        """A target is normally reachable elbow-up and elbow-down."""
        problem = make_problem(target=(200.0, 60.0))
        signs = {
            math.copysign(1.0, problem.grid.angles(s)[1])
            for s in problem.goal_states()
            if problem.grid.angles(s)[1] != 0.0
        }
        self.assertEqual(signs, {1.0, -1.0})

    def test_a_tighter_tolerance_yields_fewer_goal_states(self):
        wide = make_problem(tolerance=20.0)
        tight = make_problem(tolerance=8.0)
        self.assertGreater(len(wide.goal_states()), len(tight.goal_states()))

    def test_obstacles_remove_colliding_goal_states(self):
        free = make_problem(target=(200.0, 60.0))
        blocked = make_problem(
            target=(200.0, 60.0), obstacles=[Rectangle(150.0, 260.0, 0.0, 120.0)]
        )
        self.assertGreater(len(free.goal_states()), 0)
        self.assertEqual(len(blocked.goal_states()), 0)

    def test_no_goal_state_when_the_tolerance_is_unattainably_small(self):
        problem = make_problem(tolerance=0.001)
        self.assertEqual(len(problem.goal_states()), 0)


class HeuristicTests(unittest.TestCase):
    def test_heuristic_is_zero_at_a_goal(self):
        problem = make_problem()
        for state in problem.goal_states():
            self.assertAlmostEqual(problem.heuristic(state), 0.0)

    def test_heuristic_never_exceeds_the_true_optimal_cost(self):
        """Admissibility, checked against the cost Dijkstra actually finds."""
        problem = make_problem(obstacles=[Rectangle(230.0, 290.0, -120.0, -40.0)])
        result = plan_dijkstra(problem, START)
        self.assertTrue(result.path_found)
        start_state = problem.grid.snap(*START)
        self.assertLessEqual(problem.heuristic(start_state), result.path_cost + 1e-9)

    def test_heuristic_is_consistent_along_every_edge(self):
        """h(n) <= cost(n, n') + h(n') for a sample of real edges."""
        problem = make_problem()
        problem.goal_states()
        for state in [(10, 10), (30, 25), (50, 40), (5, 55), (70, 2)]:
            h_state = problem.heuristic(state)
            for neighbour in problem.grid.neighbours(state):
                cost = problem.edge_cost(state, neighbour)
                self.assertLessEqual(
                    h_state, cost + problem.heuristic(neighbour) + 1e-9
                )

    def test_heuristic_is_zero_when_no_goal_exists(self):
        problem = make_problem(tolerance=0.001)
        self.assertEqual(problem.heuristic((10, 10)), 0.0)


class AStarTests(unittest.TestCase):
    def test_finds_a_path_in_an_empty_workspace(self):
        result = plan_astar(make_problem(), START)
        self.assertTrue(result.path_found, result.message)
        self.assertEqual(result.status, PLAN_OK)
        self.assertGreater(result.path_length, 1)

    def test_path_starts_at_the_actual_current_pose(self):
        result = plan_astar(make_problem(), START)
        self.assertEqual(result.path[0], START)

    def test_path_ends_within_the_target_tolerance(self):
        problem = make_problem(tolerance=12.0)
        result = plan_astar(problem, START)
        self.assertLessEqual(result.final_target_error, 12.0 + 1e-9)

    def test_returned_path_is_collision_validated(self):
        obstacles = [Rectangle(230.0, 290.0, -120.0, -40.0)]
        problem = make_problem(obstacles=obstacles)
        result = plan_astar(problem, START)
        self.assertTrue(result.path_found)
        self.assertIs(result.collision_validated, True)
        # Validate independently rather than trusting the planner's own flag.
        self.assertTrue(
            validate_path(problem.spec, result.path, obstacles).valid
        )

    @staticmethod
    def _grid_poses(result):
        """The part of the path that lies on the planning grid.

        When the robot's real pose is off-grid the planner prepends it, so
        that leading pose is not a graph node and is excluded here. START is
        itself a 5 degree grid angle, so in these tests nothing is prepended.
        """
        if result.path[0] == result.snapped_start:
            return result.path
        return result.path[1:]

    def test_consecutive_path_poses_are_adjacent_on_the_grid(self):
        """Path reconstruction must not skip states."""
        problem = make_problem()
        result = plan_astar(problem, START)
        grid_poses = self._grid_poses(result)
        for (a1, a2), (b1, b2) in zip(grid_poses, grid_poses[1:]):
            self.assertLessEqual(abs(b1 - a1), 5.0 + 1e-9)
            self.assertLessEqual(abs(b2 - a2), 5.0 + 1e-9)

    def test_reported_cost_matches_the_path_it_returned(self):
        problem = make_problem()
        result = plan_astar(problem, START)
        grid_poses = self._grid_poses(result)
        recomputed = sum(
            math.hypot(b1 - a1, b2 - a2)
            for (a1, a2), (b1, b2) in zip(grid_poses, grid_poses[1:])
        )
        self.assertAlmostEqual(result.path_cost, recomputed, places=9)

    def test_metrics_are_populated_and_sane(self):
        result = plan_astar(make_problem(), START)
        self.assertGreater(result.planning_time_ms, 0.0)
        self.assertGreater(result.expanded_states, 0)
        self.assertGreaterEqual(result.discovered_states, result.expanded_states)
        self.assertGreater(result.ee_path_length, 0.0)
        self.assertGreater(result.total_joint_travel, 0.0)
        self.assertEqual(result.resolution_deg, 5.0)

    def test_routes_around_an_obstacle_at_a_higher_cost(self):
        free = plan_astar(make_problem(), START)
        blocked = plan_astar(
            make_problem(obstacles=[Rectangle(230.0, 290.0, -120.0, -40.0)]), START
        )
        self.assertTrue(blocked.path_found)
        self.assertGreater(blocked.path_cost, free.path_cost)


class DijkstraTests(unittest.TestCase):
    def test_finds_a_path_in_an_empty_workspace(self):
        result = plan_dijkstra(make_problem(), START)
        self.assertTrue(result.path_found, result.message)
        self.assertEqual(result.algorithm, "Dijkstra")

    def test_returned_path_is_collision_validated(self):
        obstacles = [Rectangle(230.0, 290.0, -120.0, -40.0)]
        result = plan_dijkstra(make_problem(obstacles=obstacles), START)
        self.assertTrue(result.path_found)
        self.assertIs(result.collision_validated, True)

    def test_dijkstra_expands_more_states_than_a_star_here(self):
        """Not a universal law -- an observed property of these scenarios.

        Both are optimal, so the interesting difference is search effort.
        """
        astar = plan_astar(make_problem(), START)
        dijkstra = plan_dijkstra(make_problem(), START)
        self.assertGreater(dijkstra.expanded_states, astar.expanded_states)


class AlgorithmAgreementTests(unittest.TestCase):
    """Both planners are optimal, so on identical input their costs must match."""

    def _compare(self, obstacles=(), target=(150.0, -200.0)):
        astar = plan_astar(make_problem(obstacles, target), START)
        dijkstra = plan_dijkstra(make_problem(obstacles, target), START)
        return astar, dijkstra

    def test_equal_cost_in_free_space(self):
        astar, dijkstra = self._compare()
        self.assertTrue(astar.path_found and dijkstra.path_found)
        self.assertAlmostEqual(astar.path_cost, dijkstra.path_cost, places=6)

    def test_equal_cost_with_an_obstacle(self):
        astar, dijkstra = self._compare(
            obstacles=[Rectangle(230.0, 290.0, -120.0, -40.0)]
        )
        self.assertTrue(astar.path_found and dijkstra.path_found)
        self.assertAlmostEqual(astar.path_cost, dijkstra.path_cost, places=6)

    def test_both_agree_when_there_is_no_path(self):
        obstacles = [
            Rectangle(200.0, 250.0, -70.0, 20.0),
            Rectangle(175.0, 215.0, 25.0, 50.0),
        ]
        astar, dijkstra = self._compare(obstacles=obstacles, target=(170.0, -160.0))
        self.assertFalse(astar.path_found)
        self.assertFalse(dijkstra.path_found)
        self.assertEqual(astar.status, PLAN_NO_PATH)
        self.assertEqual(dijkstra.status, PLAN_NO_PATH)

    def test_both_use_the_same_goal_set(self):
        problem = make_problem()
        goals = problem.goal_states()
        astar = plan_astar(problem, START)
        dijkstra = plan_dijkstra(problem, START)
        for result in (astar, dijkstra):
            final = result.path[-1]
            self.assertIn(problem.grid.snap(*final), goals)


class PlanningFailureTests(unittest.TestCase):
    """Each distinct cause of failure must report its own status code."""

    def test_invalid_start_pose(self):
        # A box straddling link 1 in the start pose.
        obstacles = [Rectangle(60.0, 120.0, 40.0, 110.0)]
        result = plan_astar(make_problem(obstacles=obstacles), START)
        self.assertFalse(result.path_found)
        self.assertEqual(result.status, PLAN_START_INVALID)
        self.assertEqual(result.algorithm, "A*")

    def test_target_outside_the_workspace(self):
        result = plan_astar(make_problem(target=(900.0, 0.0)), START)
        self.assertFalse(result.path_found)
        self.assertEqual(result.status, PLAN_TARGET_UNREACHABLE)

    def test_target_inside_the_dead_zone(self):
        result = plan_astar(make_problem(target=(5.0, 5.0)), START)
        self.assertFalse(result.path_found)
        self.assertEqual(result.status, PLAN_TARGET_UNREACHABLE)

    def test_no_discrete_goal_state_at_this_resolution(self):
        result = plan_astar(make_problem(tolerance=0.001), START)
        self.assertFalse(result.path_found)
        self.assertEqual(result.status, PLAN_NO_GOAL_STATE)
        self.assertIn("finer resolution", result.message)

    def test_no_path_message_does_not_claim_impossibility(self):
        obstacles = [
            Rectangle(200.0, 250.0, -70.0, 20.0),
            Rectangle(175.0, 215.0, 25.0, 50.0),
        ]
        result = plan_astar(
            make_problem(obstacles=obstacles, target=(170.0, -160.0)), START
        )
        self.assertEqual(result.status, PLAN_NO_PATH)
        self.assertIn("does not prove", result.message)

    def test_failed_plans_report_zero_metrics_not_fabricated_ones(self):
        result = plan_astar(make_problem(target=(900.0, 0.0)), START)
        self.assertEqual(result.path, ())
        self.assertEqual(result.path_cost, 0.0)
        self.assertEqual(result.ee_path_length, 0.0)
        self.assertIsNone(result.collision_validated)

    def test_unknown_algorithm_name_raises(self):
        with self.assertRaises(KeyError):
            plan("Greedy", make_problem(), START)


class ResolutionTests(unittest.TestCase):
    def test_every_supported_resolution_can_plan(self):
        for resolution in (3.0, 5.0, 10.0):
            with self.subTest(resolution=resolution):
                result = plan_astar(make_problem(resolution=resolution), START)
                self.assertTrue(result.path_found, result.message)
                self.assertEqual(result.resolution_deg, resolution)

    def test_a_finer_grid_has_more_states(self):
        coarse = make_problem(resolution=10.0)
        fine = make_problem(resolution=3.0)
        self.assertGreater(fine.grid.state_count, coarse.grid.state_count)

    def test_snapping_is_reported_so_the_user_sees_the_approximation(self):
        problem = make_problem(resolution=10.0)
        result = plan_astar(problem, (37.0, -63.0))
        self.assertTrue(result.path_found)
        self.assertEqual(result.snapped_start, (40.0, -60.0))
        self.assertEqual(result.path[0], (37.0, -63.0))


class CacheTests(unittest.TestCase):
    def test_state_validity_is_cached_consistently(self):
        problem = make_problem(obstacles=[Rectangle(230.0, 290.0, -120.0, -40.0)])
        state = (40, 20)
        first = problem.is_state_valid(state)
        self.assertEqual(problem.is_state_valid(state), first)

    def test_edge_validity_is_symmetric(self):
        problem = make_problem(obstacles=[Rectangle(230.0, 290.0, -120.0, -40.0)])
        a, b = (40, 20), (41, 21)
        self.assertEqual(problem.is_edge_valid(a, b), problem.is_edge_valid(b, a))

    def test_successors_only_returns_traversable_edges(self):
        problem = make_problem(obstacles=[Rectangle(230.0, 290.0, -120.0, -40.0)])
        state = problem.grid.snap(*START)
        for neighbour, cost in problem.successors(state):
            self.assertTrue(problem.is_state_valid(neighbour))
            self.assertGreater(cost, 0.0)


if __name__ == "__main__":
    unittest.main()
