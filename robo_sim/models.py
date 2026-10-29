"""Plain data structures shared by every part of ROBO-SIM.

These are deliberately immutable (``frozen=True``) dataclasses. An immutable
robot description cannot be modified by accident deep inside the planner, and
it can safely be used as part of a cache key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import config


class InvalidInputError(ValueError):
    """Raised when a caller supplies a value the model cannot represent.

    This covers non-finite numbers, non-positive link lengths and degenerate
    rectangles. It is a subclass of ``ValueError`` so ordinary ``except
    ValueError`` handling still works.
    """


def _require_finite(value: float, name: str) -> float:
    """Return ``value`` as a float, rejecting NaN and the infinities."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise InvalidInputError(f"{name} must be a finite number, got {number!r}")
    return number


@dataclass(frozen=True)
class Rectangle:
    """An axis-aligned rectangular obstacle."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float

    def __post_init__(self) -> None:
        for name in ("min_x", "max_x", "min_y", "max_y"):
            object.__setattr__(self, name, _require_finite(getattr(self, name), name))
        if self.max_x <= self.min_x:
            raise InvalidInputError(
                f"Obstacle needs positive width: max_x ({self.max_x}) must exceed "
                f"min_x ({self.min_x})."
            )
        if self.max_y <= self.min_y:
            raise InvalidInputError(
                f"Obstacle needs positive height: max_y ({self.max_y}) must exceed "
                f"min_y ({self.min_y})."
            )

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def center(self) -> tuple[float, float]:
        return ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)

    def corners(self) -> tuple[tuple[float, float], ...]:
        """The four corners, counter-clockwise from the bottom-left."""
        return (
            (self.min_x, self.min_y),
            (self.max_x, self.min_y),
            (self.max_x, self.max_y),
            (self.min_x, self.max_y),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.min_x, self.max_x, self.min_y, self.max_y)

    def describe(self) -> str:
        return (
            f"x [{self.min_x:g}, {self.max_x:g}] × "
            f"y [{self.min_y:g}, {self.max_y:g}]"
        )


@dataclass(frozen=True)
class RobotSpec:
    """Everything that defines the robot: dimensions, limits and thickness.

    The joint *angles* are deliberately not part of this object. The spec is
    the machine; the angles are its momentary pose.
    """

    base_x: float = config.BASE_X
    base_y: float = config.BASE_Y
    link1_length: float = config.LINK1_LENGTH
    link2_length: float = config.LINK2_LENGTH
    joint1_min: float = config.JOINT1_MIN
    joint1_max: float = config.JOINT1_MAX
    joint2_min: float = config.JOINT2_MIN
    joint2_max: float = config.JOINT2_MAX
    link_radius: float = config.LINK_RADIUS
    end_effector_radius: float = config.END_EFFECTOR_RADIUS

    def __post_init__(self) -> None:
        for name in (
            "base_x",
            "base_y",
            "link1_length",
            "link2_length",
            "joint1_min",
            "joint1_max",
            "joint2_min",
            "joint2_max",
            "link_radius",
            "end_effector_radius",
        ):
            object.__setattr__(self, name, _require_finite(getattr(self, name), name))
        if self.link1_length <= 0 or self.link2_length <= 0:
            raise InvalidInputError("Both link lengths must be greater than zero.")
        if self.link_radius < 0 or self.end_effector_radius < 0:
            raise InvalidInputError("Collision radii cannot be negative.")
        if self.joint1_max <= self.joint1_min:
            raise InvalidInputError("Joint 1 maximum must exceed its minimum.")
        if self.joint2_max <= self.joint2_min:
            raise InvalidInputError("Joint 2 maximum must exceed its minimum.")

    @property
    def base(self) -> tuple[float, float]:
        return (self.base_x, self.base_y)

    @property
    def max_reach(self) -> float:
        """Radius of the outer boundary of the reachable workspace."""
        return self.link1_length + self.link2_length

    @property
    def min_reach(self) -> float:
        """Radius of the inner "dead zone" the tip can never enter.

        When the links are of equal length the arm can fold onto itself and
        this is zero; otherwise the shorter link cannot cancel the longer one.
        """
        return abs(self.link1_length - self.link2_length)


@dataclass(frozen=True)
class ArmGeometry:
    """The result of forward kinematics: where every part of the arm is."""

    base: tuple[float, float]
    joint1: tuple[float, float]
    end_effector: tuple[float, float]
    theta1_deg: float
    theta2_deg: float

    @property
    def link1(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Link 1 as a (start, end) segment."""
        return (self.base, self.joint1)

    @property
    def link2(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Link 2 as a (start, end) segment."""
        return (self.joint1, self.end_effector)


@dataclass(frozen=True)
class IKSolution:
    """One analytical inverse-kinematics solution."""

    theta1_deg: float
    theta2_deg: float
    elbow: str  # "up" (theta2 > 0) or "down" (theta2 < 0)
    within_joint_limits: bool
    collision_free: Optional[bool]  # None when no obstacle data was supplied
    position_error: float
    joint_travel: float  # total absolute joint change from the current pose

    @property
    def usable(self) -> bool:
        """True when this solution can actually be commanded."""
        return self.within_joint_limits and self.collision_free is not False


@dataclass(frozen=True)
class IKResult:
    """The outcome of an inverse-kinematics request.

    The three questions below are kept strictly separate, because a target can
    be geometrically fine yet unusable for a completely different reason:

    * ``geometrically_reachable`` -- can the tip touch that point at all?
    * ``solutions``               -- which analytical poses put it there?
    * ``chosen``                  -- which of those is legal and collision free?
    """

    target: tuple[float, float]
    geometrically_reachable: bool
    solutions: tuple[IKSolution, ...]
    chosen: Optional[IKSolution]
    message: str

    @property
    def success(self) -> bool:
        return self.chosen is not None

    @property
    def limit_blocked(self) -> bool:
        """Reachable, but every solution breaks a joint limit."""
        return (
            self.geometrically_reachable
            and bool(self.solutions)
            and not any(s.within_joint_limits for s in self.solutions)
        )

    @property
    def collision_blocked(self) -> bool:
        """Reachable and legal, but every legal solution hits an obstacle."""
        legal = [s for s in self.solutions if s.within_joint_limits]
        return bool(legal) and all(s.collision_free is False for s in legal)


@dataclass(frozen=True)
class ValidationResult:
    """Why a configuration or a transition was accepted or rejected."""

    valid: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.valid


# Outcome codes used by the planner. They are deliberately fine-grained so the
# interface can explain *which* of several very different failures occurred.
PLAN_OK = "path_found"
PLAN_START_INVALID = "start_invalid"
PLAN_START_SNAP_FAILED = "start_snap_failed"
PLAN_TARGET_UNREACHABLE = "target_unreachable"
PLAN_NO_GOAL_STATE = "no_goal_state"
PLAN_NO_PATH = "no_path"
PLAN_GRID_TOO_LARGE = "grid_too_large"


@dataclass(frozen=True)
class PlanResult:
    """Metrics and the path produced by one run of one planner.

    Every numeric field is measured, never estimated. When ``path_found`` is
    False the path-related numbers are zero and ``status`` says why.
    """

    algorithm: str
    path_found: bool
    status: str
    message: str
    path: tuple[tuple[float, float], ...] = ()  # (theta1_deg, theta2_deg) poses
    planning_time_ms: float = 0.0
    expanded_states: int = 0
    discovered_states: int = 0
    path_cost: float = 0.0           # sum of planner edge costs, in degrees
    total_joint_travel: float = 0.0  # sum of |dtheta1| + |dtheta2|, in degrees
    ee_path_length: float = 0.0      # tip travel, in simulation units
    final_target_error: float = 0.0  # tip-to-target distance at the end
    collision_validated: Optional[bool] = None
    grid_states: int = 0
    valid_grid_states: int = 0
    goal_states: int = 0
    resolution_deg: float = 0.0
    snapped_start: Optional[tuple[float, float]] = None

    @property
    def path_length(self) -> int:
        """Number of configurations in the returned path."""
        return len(self.path)


@dataclass(frozen=True)
class ScenarioResult:
    """One scenario executed by one algorithm, flattened for CSV export."""

    scenario_id: str
    scenario_description: str
    link1_length: float
    link2_length: float
    start_theta1: float
    start_theta2: float
    target_x: float
    target_y: float
    joint1_min: float
    joint1_max: float
    joint2_min: float
    joint2_max: float
    resolution_deg: float
    target_tolerance: float
    obstacles: str
    algorithm: str
    path_found: bool
    status: str
    planning_time_ms: float
    expanded_states: int
    discovered_states: int
    path_cost: float
    path_configurations: int
    ee_path_length: float
    total_joint_travel: float
    final_target_error: float
    collision_validated: Optional[bool]


# Column order used for every CSV the project writes.
CSV_FIELDNAMES: tuple[str, ...] = (
    "scenario_id",
    "scenario_description",
    "link1_length",
    "link2_length",
    "start_theta1",
    "start_theta2",
    "target_x",
    "target_y",
    "joint1_min",
    "joint1_max",
    "joint2_min",
    "joint2_max",
    "resolution_deg",
    "target_tolerance",
    "obstacles",
    "algorithm",
    "path_found",
    "status",
    "planning_time_ms",
    "expanded_states",
    "discovered_states",
    "path_cost",
    "path_configurations",
    "ee_path_length",
    "total_joint_travel",
    "final_target_error",
    "collision_validated",
)


@dataclass(frozen=True)
class Scenario:
    """A deterministic, fully specified planning problem."""

    scenario_id: str
    description: str
    spec: RobotSpec
    start: tuple[float, float]
    target: tuple[float, float]
    obstacles: tuple[Rectangle, ...] = field(default_factory=tuple)
    resolution_deg: float = config.DEFAULT_RESOLUTION
    target_tolerance: float = config.TARGET_TOLERANCE
    expectation: str = ""


def describe_obstacles(obstacles: Sequence[Rectangle]) -> str:
    """Compact single-field description of an obstacle layout, for CSV output."""
    if not obstacles:
        return "none"
    return "; ".join(
        f"({r.min_x:g},{r.max_x:g},{r.min_y:g},{r.max_y:g})" for r in obstacles
    )
