"""Motion planning in the robot's joint-configuration space.

**Why plan in joint space rather than in the workspace?**

The obvious approach -- route the end effector around the obstacles like a dot
on a map -- is wrong for a robot arm. The tip is not the only part that can
collide: rotating joint 1 sweeps the *whole* arm through a large region. A tip
trajectory that looks clear can only be executed by dragging link 1 straight
through an obstacle.

So ROBO-SIM searches the space of joint angles instead. A state is a pair of
joint-angle indices ``(i, j)`` on a regular grid, and a state is only inserted
into the graph if the complete arm -- both capsule-shaped links and the disc at
the tip -- is collision free in that pose. Obstacle avoidance then comes out of
an ordinary graph search, and whatever path is found is executable by
construction.

**The graph.**

* Nodes: grid states ``(i, j)``, with angles ``joint1_min + i*res`` and
  ``joint2_min + j*res``.
* Edges: 8-connected. Each index may change by -1, 0 or +1, excluding the
  no-move case. Edges exist only between valid states whose connecting motion
  is also collision free.
* Edge cost: the Euclidean change in joint angles,
  ``sqrt(dtheta1^2 + dtheta2^2)``, measured in degrees. A diagonal move
  therefore costs ``res * sqrt(2)`` and a straight move costs ``res``, which is
  what makes diagonal shortcuts priced correctly.

The -180° and +180° states of joint 1 are *not* joined, even though they are
the same physical direction. The graph models the configured joint range
literally; wrapping would silently permit motions that a real joint with those
limits could not perform.

A* and Dijkstra are implemented separately below and share this graph, these
costs, the same goal set and the same collision rules, so comparing them is
fair.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from typing import Iterator, Optional, Sequence

from . import config
from .collision import is_configuration_valid, is_transition_valid, validate_path
from .kinematics import (
    end_effector_path_length,
    end_effector_position,
    is_geometrically_reachable,
    target_error,
    total_joint_travel,
)
from .models import (
    PLAN_GRID_TOO_LARGE,
    PLAN_NO_GOAL_STATE,
    PLAN_NO_PATH,
    PLAN_OK,
    PLAN_START_INVALID,
    PLAN_START_SNAP_FAILED,
    PLAN_TARGET_UNREACHABLE,
    PlanResult,
    Rectangle,
    RobotSpec,
)

State = tuple[int, int]

# Relative index offsets for the 8-connected neighbourhood.
NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


class GridTooLargeError(ValueError):
    """Raised when the requested resolution would build an unusable graph."""


def _axis_angles(minimum: float, maximum: float, resolution: float) -> list[float]:
    """Grid angles from ``minimum`` to ``maximum`` inclusive.

    The caller is expected to have checked that the range divides evenly; the
    final value is snapped to ``maximum`` so accumulated floating-point error
    cannot make the last grid angle drift outside the joint limit.
    """
    span = maximum - minimum
    count = int(round(span / resolution))
    angles = [minimum + i * resolution for i in range(count + 1)]
    angles[-1] = maximum
    return angles


def resolution_divides_range(minimum: float, maximum: float, resolution: float) -> bool:
    """True when ``resolution`` fits a whole number of times into the range."""
    span = maximum - minimum
    return abs(span / resolution - round(span / resolution)) < 1e-9


class JointGrid:
    """The discretised set of joint angles the planner may use."""

    def __init__(self, spec: RobotSpec, resolution_deg: float) -> None:
        if resolution_deg <= 0:
            raise ValueError("Planning resolution must be positive.")

        self.spec = spec
        self.resolution = float(resolution_deg)

        # If the resolution does not divide a joint's range exactly, the joint
        # limit itself would not be a grid angle. Rather than quietly moving
        # the limit, the grid keeps the exact limit as its final sample; the
        # last step is then shorter than the others, and because edge costs are
        # computed from actual angles the search still prices it correctly.
        self.angles1 = _axis_angles(spec.joint1_min, spec.joint1_max, self.resolution)
        self.angles2 = _axis_angles(spec.joint2_min, spec.joint2_max, self.resolution)

        self.aligned = resolution_divides_range(
            spec.joint1_min, spec.joint1_max, self.resolution
        ) and resolution_divides_range(
            spec.joint2_min, spec.joint2_max, self.resolution
        )

        self.size1 = len(self.angles1)
        self.size2 = len(self.angles2)
        self.state_count = self.size1 * self.size2

        if self.state_count > config.MAX_GRID_STATES:
            raise GridTooLargeError(
                f"A {self.resolution:g}° grid over these joint limits needs "
                f"{self.state_count} states, above the {config.MAX_GRID_STATES} "
                f"limit for responsive planning. Use a coarser resolution or "
                f"narrower joint limits."
            )

    def angles(self, state: State) -> tuple[float, float]:
        """The joint angles of a grid state, in degrees."""
        return (self.angles1[state[0]], self.angles2[state[1]])

    def contains(self, state: State) -> bool:
        i, j = state
        return 0 <= i < self.size1 and 0 <= j < self.size2

    def snap(self, theta1_deg: float, theta2_deg: float) -> State:
        """The grid state whose angles are closest to the given pose."""
        i = min(
            range(self.size1), key=lambda k: abs(self.angles1[k] - theta1_deg)
        )
        j = min(
            range(self.size2), key=lambda k: abs(self.angles2[k] - theta2_deg)
        )
        return (i, j)

    def neighbours(self, state: State) -> Iterator[State]:
        """The in-bounds 8-connected neighbours of a state."""
        i, j = state
        for di, dj in NEIGHBOUR_OFFSETS:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size1 and 0 <= nj < self.size2:
                yield (ni, nj)


class PlanningProblem:
    """A fully specified search problem: robot, obstacles, grid, target.

    State validity and edge traversability are evaluated lazily and cached, so
    a search only pays for the part of the configuration space it actually
    visits. Both planners share one instance in a comparison run, which means
    the second planner benefits from the first one's cache -- so planning
    *times* are compared using a freshly built problem per algorithm, while the
    structural metrics (expanded states, path cost) are cache-independent.
    """

    def __init__(
        self,
        spec: RobotSpec,
        obstacles: Sequence[Rectangle],
        target: tuple[float, float],
        resolution_deg: float = config.DEFAULT_RESOLUTION,
        target_tolerance: float = config.TARGET_TOLERANCE,
        transition_step_deg: float = config.TRANSITION_STEP_DEG,
    ) -> None:
        self.spec = spec
        self.obstacles = tuple(obstacles)
        self.target = (float(target[0]), float(target[1]))
        self.tolerance = float(target_tolerance)
        self.transition_step = float(transition_step_deg)
        self.grid = JointGrid(spec, resolution_deg)

        self._valid_cache: dict[State, bool] = {}
        self._edge_cache: dict[tuple[State, State], bool] = {}
        self._goal_states: Optional[frozenset[State]] = None
        self._goal_angles: Optional[tuple[tuple[float, float], ...]] = None

    # -- state and edge tests ---------------------------------------------

    def is_state_valid(self, state: State) -> bool:
        """True when the arm is within limits and clear of obstacles there."""
        cached = self._valid_cache.get(state)
        if cached is not None:
            return cached
        theta1, theta2 = self.grid.angles(state)
        result = is_configuration_valid(self.spec, theta1, theta2, self.obstacles).valid
        self._valid_cache[state] = result
        return result

    def is_edge_valid(self, a: State, b: State) -> bool:
        """True when the arm can sweep from ``a`` to ``b`` without collision.

        Both endpoints must be valid *and* every interpolated pose between
        them must be valid. Checking only the endpoints would allow a link to
        pass straight through an obstacle mid-move.
        """
        key = (a, b)
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached

        if not (self.is_state_valid(a) and self.is_state_valid(b)):
            self._edge_cache[key] = False
            return False

        if not self.obstacles:
            # Without obstacles, linear interpolation between two in-limit
            # poses stays in limits, so the endpoints are sufficient.
            self._edge_cache[key] = True
            return True

        ok = is_transition_valid(
            self.spec,
            self.grid.angles(a),
            self.grid.angles(b),
            self.obstacles,
            self.transition_step,
        ).valid
        self._edge_cache[key] = ok
        # The relation is symmetric, so record the reverse direction too.
        self._edge_cache[(b, a)] = ok
        return ok

    def edge_cost(self, a: State, b: State) -> float:
        """Euclidean distance between two states in joint-angle space (degrees).

        Computed from the real grid angles rather than from index differences,
        so it stays correct even where the last grid step is shorter than the
        resolution.
        """
        a1, a2 = self.grid.angles(a)
        b1, b2 = self.grid.angles(b)
        return math.hypot(b1 - a1, b2 - a2)

    def successors(self, state: State) -> Iterator[tuple[State, float]]:
        """Traversable neighbours of a state with their edge costs."""
        for neighbour in self.grid.neighbours(state):
            if self.is_edge_valid(state, neighbour):
                yield neighbour, self.edge_cost(state, neighbour)

    # -- goal set ----------------------------------------------------------

    def goal_states(self) -> frozenset[State]:
        """Every valid grid state whose tip is within tolerance of the target.

        A two-link arm generally reaches a point in two ways (elbow up and
        elbow down), and the tolerance admits a neighbourhood around each, so
        there is rarely a single goal. The search accepts whichever it reaches
        first most cheaply.

        The cheap tip-position test runs first and the expensive full-body
        collision test only on the states that pass it.
        """
        if self._goal_states is not None:
            return self._goal_states

        spec = self.spec
        tx, ty = self.target
        tolerance_squared = self.tolerance * self.tolerance

        found: set[State] = set()
        for i, theta1 in enumerate(self.grid.angles1):
            for j, theta2 in enumerate(self.grid.angles2):
                x, y = end_effector_position(spec, theta1, theta2)
                dx = x - tx
                dy = y - ty
                if dx * dx + dy * dy > tolerance_squared:
                    continue
                if self.is_state_valid((i, j)):
                    found.add((i, j))

        self._goal_states = frozenset(found)
        self._goal_angles = tuple(self.grid.angles(s) for s in sorted(found))
        return self._goal_states

    def goal_angles(self) -> tuple[tuple[float, float], ...]:
        """The joint angles of every goal state."""
        self.goal_states()
        return self._goal_angles or ()

    def heuristic(self, state: State) -> float:
        """Minimum straight-line joint-space distance from ``state`` to a goal.

        Admissible: any path from this state to a goal is a chain of moves, and
        by the triangle inequality the sum of their Euclidean lengths is at
        least the straight-line distance to that goal. It can therefore never
        overestimate the true remaining cost.

        Consistent: for any edge ``(n, n')`` of cost ``c``, the minimum
        distance to the goal set changes by at most ``c``, so
        ``h(n) <= c + h(n')``. A consistent heuristic means A* finds an optimal
        path and never needs to reopen a closed state.

        It is measured in degrees, the same unit as the edge costs -- mixing
        units here is the classic way to make A* inadmissible and quietly
        non-optimal.
        """
        goals = self.goal_angles()
        if not goals:
            return 0.0
        theta1, theta2 = self.grid.angles(state)
        best = math.inf
        for g1, g2 in goals:
            d = math.hypot(g1 - theta1, g2 - theta2)
            if d < best:
                best = d
        return best

    @property
    def valid_states_seen(self) -> int:
        """Distinct states tested so far that turned out to be collision free.

        This counts what the searches actually touched, not the whole grid --
        validity is evaluated lazily and the grid is never fully enumerated
        except when building the goal set.
        """
        return sum(1 for ok in self._valid_cache.values() if ok)


# --------------------------------------------------------------------------
# Shared start/goal preparation
# --------------------------------------------------------------------------


def _reconstruct(came_from: dict[State, State], goal: State) -> list[State]:
    """Walk predecessor links back from the goal and reverse."""
    path = [goal]
    node = goal
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def _prepare(
    problem: PlanningProblem, start_angles: tuple[float, float]
) -> tuple[Optional[State], Optional[PlanResult], Optional[tuple[float, float]]]:
    """Validate the start pose and the goal set before searching.

    Returns ``(start_state, failure_result, snapped_angles)``. When
    ``failure_result`` is not None the search must not run, and that result is
    what the caller should report.

    Every failure mode gets its own status code, because "no path" has several
    very different causes and conflating them would mislead the user.
    """
    spec = problem.spec
    grid = problem.grid
    algorithm = ""  # filled in by the caller via dataclasses.replace-like copy

    def failure(status: str, message: str) -> PlanResult:
        return PlanResult(
            algorithm=algorithm,
            path_found=False,
            status=status,
            message=message,
            grid_states=grid.state_count,
            resolution_deg=grid.resolution,
        )

    # 1. Is the pose the robot is actually in legal?
    start_validity = is_configuration_valid(
        spec, start_angles[0], start_angles[1], problem.obstacles
    )
    if not start_validity.valid:
        return None, failure(
            PLAN_START_INVALID,
            f"The robot cannot start from its current pose. {start_validity.reason}",
        ), None

    # 2. Snap it onto the grid and make sure the snapped pose is legal too.
    start_state = grid.snap(start_angles[0], start_angles[1])
    snapped = grid.angles(start_state)

    if not problem.is_state_valid(start_state):
        return None, failure(
            PLAN_START_SNAP_FAILED,
            f"The current pose ({start_angles[0]:.1f}°, {start_angles[1]:.1f}°) is "
            f"valid, but the nearest {grid.resolution:g}° grid pose "
            f"({snapped[0]:.1f}°, {snapped[1]:.1f}°) is not. Try a finer resolution "
            f"or move the arm slightly.",
        ), None

    # 3. The move from the real pose onto the grid pose is itself a motion and
    #    has to be checked -- it is the first thing the robot would execute.
    if snapped != tuple(start_angles):
        bridge = is_transition_valid(
            spec, tuple(start_angles), snapped, problem.obstacles, problem.transition_step
        )
        if not bridge.valid:
            return None, failure(
                PLAN_START_SNAP_FAILED,
                f"Moving from the current pose onto the nearest grid pose is not "
                f"collision free. {bridge.reason}",
            ), None

    # 4. Is the target reachable at all, before blaming the search?
    if not is_geometrically_reachable(spec, problem.target):
        return None, failure(
            PLAN_TARGET_UNREACHABLE,
            f"The target is outside the arm's reachable workspace (radii "
            f"{spec.min_reach:.1f}–{spec.max_reach:.1f} u from the base).",
        ), None

    # 5. Does any grid pose actually put the tip close enough?
    goals = problem.goal_states()
    if not goals:
        return None, failure(
            PLAN_NO_GOAL_STATE,
            f"No collision-free pose on the {grid.resolution:g}° grid places the tip "
            f"within {problem.tolerance:g} u of the target. This does not mean the "
            f"target is impossible: try a finer resolution, a larger tolerance, or "
            f"check whether an obstacle or a joint limit blocks every pose that "
            f"reaches it.",
        ), None

    return start_state, None, snapped


def _build_result(
    algorithm: str,
    problem: PlanningProblem,
    states: list[State],
    elapsed_ms: float,
    expanded: int,
    discovered: int,
    snapped_start: tuple[float, float],
    start_angles: tuple[float, float],
) -> PlanResult:
    """Turn a successful search into a fully measured :class:`PlanResult`."""
    grid = problem.grid
    spec = problem.spec

    path: list[tuple[float, float]] = [grid.angles(s) for s in states]

    # The robot starts at its real pose, which may be a fraction of a degree
    # off the grid. Prepending it keeps the reported path identical to the
    # motion that will actually be executed and animated.
    if tuple(start_angles) != path[0]:
        path.insert(0, (float(start_angles[0]), float(start_angles[1])))

    cost = sum(
        problem.edge_cost(a, b) for a, b in zip(states, states[1:])
    )

    validation = validate_path(spec, path, problem.obstacles, problem.transition_step)
    final_error = target_error(spec, path[-1][0], path[-1][1], problem.target)

    return PlanResult(
        algorithm=algorithm,
        path_found=True,
        status=PLAN_OK,
        message=(
            f"{algorithm} found a path with {len(path)} configurations "
            f"(cost {cost:.2f}°, tip error {final_error:.2f} u)."
        ),
        path=tuple(path),
        planning_time_ms=elapsed_ms,
        expanded_states=expanded,
        discovered_states=discovered,
        path_cost=cost,
        total_joint_travel=total_joint_travel(path),
        ee_path_length=end_effector_path_length(spec, path),
        final_target_error=final_error,
        collision_validated=validation.valid,
        grid_states=grid.state_count,
        valid_grid_states=problem.valid_states_seen,
        goal_states=len(problem.goal_states()),
        resolution_deg=grid.resolution,
        snapped_start=snapped_start,
    )


def _no_path_result(
    algorithm: str,
    problem: PlanningProblem,
    elapsed_ms: float,
    expanded: int,
    discovered: int,
) -> PlanResult:
    """Report an exhausted search honestly."""
    return PlanResult(
        algorithm=algorithm,
        path_found=False,
        status=PLAN_NO_PATH,
        message=(
            f"{algorithm} explored {expanded} states and found no route to any of the "
            f"{len(problem.goal_states())} goal poses. The reachable part of the "
            f"{problem.grid.resolution:g}° grid is separated from the target by the "
            f"obstacles or the joint limits. A finer resolution may find a gap that "
            f"this grid is too coarse to represent; it does not prove that no "
            f"continuous collision-free motion exists."
        ),
        planning_time_ms=elapsed_ms,
        expanded_states=expanded,
        discovered_states=discovered,
        grid_states=problem.grid.state_count,
        valid_grid_states=problem.valid_states_seen,
        goal_states=len(problem.goal_states()),
        resolution_deg=problem.grid.resolution,
    )


def _with_algorithm(result: PlanResult, algorithm: str) -> PlanResult:
    """Attach the algorithm name to a failure produced before the search."""
    if result.algorithm:
        return result
    return PlanResult(
        algorithm=algorithm,
        path_found=result.path_found,
        status=result.status,
        message=result.message,
        grid_states=result.grid_states,
        resolution_deg=result.resolution_deg,
    )


# --------------------------------------------------------------------------
# A*
# --------------------------------------------------------------------------


def plan_astar(
    problem: PlanningProblem, start_angles: tuple[float, float]
) -> PlanResult:
    """A* search over the joint-space grid.

    A* orders its frontier by ``f = g + h``: the cost already paid plus the
    heuristic estimate of what remains. With the admissible, consistent
    heuristic in :meth:`PlanningProblem.heuristic` the first goal state popped
    is reached by an optimal path, so the search can stop immediately.

    ``expanded_states`` counts states popped from the queue and processed.
    States popped a second time after a cheaper route was already finalised are
    skipped and not counted -- the same rule is used in Dijkstra below, so the
    two counts mean the same thing.
    """
    algorithm = "A*"
    start_state, failure, snapped = _prepare(problem, start_angles)
    if failure is not None:
        return _with_algorithm(failure, algorithm)
    assert start_state is not None and snapped is not None

    goals = problem.goal_states()
    counter = itertools.count()  # tie-break so the heap order is deterministic

    g_score: dict[State, float] = {start_state: 0.0}
    came_from: dict[State, State] = {}
    closed: set[State] = set()

    frontier: list[tuple[float, float, int, State]] = []
    heapq.heappush(
        frontier, (problem.heuristic(start_state), 0.0, next(counter), start_state)
    )
    discovered = 1
    expanded = 0

    start_time = time.perf_counter()

    while frontier:
        _, cost_so_far, _, state = heapq.heappop(frontier)

        if state in closed:
            continue
        closed.add(state)
        expanded += 1

        if state in goals:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return _build_result(
                algorithm,
                problem,
                _reconstruct(came_from, state),
                elapsed_ms,
                expanded,
                discovered,
                snapped,
                start_angles,
            )

        for neighbour, step_cost in problem.successors(state):
            if neighbour in closed:
                continue
            tentative = cost_so_far + step_cost
            if tentative < g_score.get(neighbour, math.inf) - 1e-12:
                if neighbour not in g_score:
                    discovered += 1
                g_score[neighbour] = tentative
                came_from[neighbour] = state
                priority = tentative + problem.heuristic(neighbour)
                heapq.heappush(
                    frontier, (priority, tentative, next(counter), neighbour)
                )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return _no_path_result(algorithm, problem, elapsed_ms, expanded, discovered)


# --------------------------------------------------------------------------
# Dijkstra
# --------------------------------------------------------------------------


def plan_dijkstra(
    problem: PlanningProblem, start_angles: tuple[float, float]
) -> PlanResult:
    """Dijkstra's algorithm over the same joint-space grid.

    This is a genuinely separate implementation, not A* with the heuristic
    switched off, but it deliberately shares the graph, the edge costs, the
    goal set, the expansion-counting rule and the path-reconstruction
    convention so that a comparison against A* measures only the thing that
    differs: the frontier ordering.

    Dijkstra orders by accumulated cost ``g`` alone. It therefore grows a
    uniform-cost ball around the start and typically expands many more states
    than A* before reaching the goal, while finding a path of the same optimal
    cost.
    """
    algorithm = "Dijkstra"
    start_state, failure, snapped = _prepare(problem, start_angles)
    if failure is not None:
        return _with_algorithm(failure, algorithm)
    assert start_state is not None and snapped is not None

    goals = problem.goal_states()
    counter = itertools.count()

    distance: dict[State, float] = {start_state: 0.0}
    came_from: dict[State, State] = {}
    settled: set[State] = set()

    frontier: list[tuple[float, int, State]] = [(0.0, next(counter), start_state)]
    discovered = 1
    expanded = 0

    start_time = time.perf_counter()

    while frontier:
        cost_so_far, _, state = heapq.heappop(frontier)

        if state in settled:
            continue
        settled.add(state)
        expanded += 1

        if state in goals:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return _build_result(
                algorithm,
                problem,
                _reconstruct(came_from, state),
                elapsed_ms,
                expanded,
                discovered,
                snapped,
                start_angles,
            )

        for neighbour, step_cost in problem.successors(state):
            if neighbour in settled:
                continue
            tentative = cost_so_far + step_cost
            if tentative < distance.get(neighbour, math.inf) - 1e-12:
                if neighbour not in distance:
                    discovered += 1
                distance[neighbour] = tentative
                came_from[neighbour] = state
                heapq.heappush(frontier, (tentative, next(counter), neighbour))

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return _no_path_result(algorithm, problem, elapsed_ms, expanded, discovered)


PLANNERS = {
    "A*": plan_astar,
    "Dijkstra": plan_dijkstra,
}


def plan(
    algorithm: str, problem: PlanningProblem, start_angles: tuple[float, float]
) -> PlanResult:
    """Run the named planner. Raises ``KeyError`` for an unknown name."""
    try:
        planner = PLANNERS[algorithm]
    except KeyError:
        raise KeyError(
            f"Unknown algorithm {algorithm!r}. Available: {sorted(PLANNERS)}."
        ) from None
    return planner(problem, start_angles)
