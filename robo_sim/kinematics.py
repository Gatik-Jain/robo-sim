"""Forward and inverse kinematics for the two-link planar arm.

Coordinate convention used everywhere in ROBO-SIM:

* +X points right, +Y points up (ordinary mathematical axes).
* theta1 is measured from the +X axis.
* theta2 is measured *relative to link 1*, so the absolute direction of
  link 2 is theta1 + theta2.
* Angles are degrees in every public interface; radians appear only inside
  the trigonometric expressions.

This module is the single source of truth for the arm's geometry. The
visualization, the collision checker and the planner all call into it rather
than repeating the equations.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from .models import (
    ArmGeometry,
    IKResult,
    IKSolution,
    InvalidInputError,
    Rectangle,
    RobotSpec,
)

# Tolerance used when deciding whether an out-of-domain arccosine argument is
# floating-point noise or a genuinely unreachable target. cos(theta2) is built
# from a ratio of squared lengths, so rounding error is many orders of
# magnitude below this; anything larger is a real geometric violation.
ACOS_DOMAIN_EPS = 1e-9

# How far outside the annulus of reachable radii a target may sit and still be
# treated as "on the boundary" rather than unreachable, in simulation units.
REACH_EPS = 1e-7


def forward_kinematics(
    spec: RobotSpec, theta1_deg: float, theta2_deg: float
) -> ArmGeometry:
    """Return the position of every part of the arm for a given pose.

        x1 = bx + L1 * cos(theta1)
        y1 = by + L1 * sin(theta1)
        x2 = x1 + L2 * cos(theta1 + theta2)
        y2 = y1 + L2 * sin(theta1 + theta2)

    ``theta2`` is relative to link 1, which is why the second pair of
    equations uses the *sum* of the two angles.
    """
    if not (math.isfinite(theta1_deg) and math.isfinite(theta2_deg)):
        raise InvalidInputError("Joint angles must be finite numbers.")

    t1 = math.radians(theta1_deg)
    t2 = math.radians(theta2_deg)

    x1 = spec.base_x + spec.link1_length * math.cos(t1)
    y1 = spec.base_y + spec.link1_length * math.sin(t1)

    x2 = x1 + spec.link2_length * math.cos(t1 + t2)
    y2 = y1 + spec.link2_length * math.sin(t1 + t2)

    return ArmGeometry(
        base=(spec.base_x, spec.base_y),
        joint1=(x1, y1),
        end_effector=(x2, y2),
        theta1_deg=float(theta1_deg),
        theta2_deg=float(theta2_deg),
    )


def end_effector_position(
    spec: RobotSpec, theta1_deg: float, theta2_deg: float
) -> tuple[float, float]:
    """Just the tip position.

    Written out rather than calling :func:`forward_kinematics` because the
    planner evaluates this for every candidate goal state and skipping the
    dataclass construction is a measurable saving.
    """
    t1 = math.radians(theta1_deg)
    t12 = t1 + math.radians(theta2_deg)
    x = spec.base_x + spec.link1_length * math.cos(t1) + spec.link2_length * math.cos(t12)
    y = spec.base_y + spec.link1_length * math.sin(t1) + spec.link2_length * math.sin(t12)
    return (x, y)


def target_error(
    spec: RobotSpec, theta1_deg: float, theta2_deg: float, target: tuple[float, float]
) -> float:
    """Euclidean distance from the tip to ``target`` for the given pose."""
    x, y = end_effector_position(spec, theta1_deg, theta2_deg)
    return math.hypot(x - target[0], y - target[1])


def is_within_joint_limits(
    spec: RobotSpec, theta1_deg: float, theta2_deg: float
) -> bool:
    """True when both angles lie inside the configured limits."""
    return (
        spec.joint1_min <= theta1_deg <= spec.joint1_max
        and spec.joint2_min <= theta2_deg <= spec.joint2_max
    )


def is_geometrically_reachable(
    spec: RobotSpec, target: tuple[float, float]
) -> bool:
    """True when the tip could touch ``target`` ignoring limits and obstacles.

    The reachable set of a two-link arm is the annulus between |L1 - L2| and
    L1 + L2 centred on the base.
    """
    tx, ty = target
    if not (math.isfinite(tx) and math.isfinite(ty)):
        return False
    r = math.hypot(tx - spec.base_x, ty - spec.base_y)
    return (spec.min_reach - REACH_EPS) <= r <= (spec.max_reach + REACH_EPS)


def _normalise_angle(degrees: float) -> float:
    """Wrap an angle into (-180, 180]."""
    wrapped = math.fmod(degrees, 360.0)
    if wrapped <= -180.0:
        wrapped += 360.0
    elif wrapped > 180.0:
        wrapped -= 360.0
    return wrapped


def inverse_kinematics(
    spec: RobotSpec,
    target: tuple[float, float],
    obstacles: Optional[Sequence[Rectangle]] = None,
    current: Optional[tuple[float, float]] = None,
) -> IKResult:
    """Solve analytically for the joint angles that put the tip on ``target``.

    With ``r`` the distance from the base to the target::

        cos(theta2) = (r^2 - L1^2 - L2^2) / (2 * L1 * L2)
        theta2      = +/- acos(cos(theta2))
        theta1      = atan2(y, x) - atan2(L2*sin(theta2), L1 + L2*cos(theta2))

    The +/- gives the two classic "elbow up" and "elbow down" poses.

    The result separates three independent questions -- is the target
    geometrically reachable, do legal solutions exist, and is a legal solution
    collision free -- so the interface never reports the wrong reason for a
    failure. An unreachable target is reported as unreachable; it is never
    quietly clamped onto the workspace boundary.

    ``obstacles`` is optional. When omitted, ``collision_free`` on each
    solution is ``None`` ("not assessed") rather than ``True``.
    """
    # Imported here rather than at module scope: collision.py imports this
    # module for forward kinematics, so a top-level import would be circular.
    from .collision import configuration_collides

    tx, ty = target
    if not (math.isfinite(tx) and math.isfinite(ty)):
        raise InvalidInputError("Target coordinates must be finite numbers.")

    # Work relative to the base.
    x = tx - spec.base_x
    y = ty - spec.base_y
    r_squared = x * x + y * y
    r = math.sqrt(r_squared)

    l1 = spec.link1_length
    l2 = spec.link2_length

    if not is_geometrically_reachable(spec, target):
        if r > spec.max_reach:
            msg = (
                f"Target is {r:.1f} u from the base but the arm reaches only "
                f"{spec.max_reach:.1f} u. Move the target closer or lengthen a link."
            )
        else:
            msg = (
                f"Target is {r:.1f} u from the base, inside the unreachable inner "
                f"zone of radius {spec.min_reach:.1f} u. Move the target outwards "
                f"or make the link lengths more similar."
            )
        return IKResult(
            target=(tx, ty),
            geometrically_reachable=False,
            solutions=(),
            chosen=None,
            message=msg,
        )

    cos_theta2 = (r_squared - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)

    # Clamp only what is attributable to rounding. A larger excursion means the
    # reachability test above and this computation disagree, which would be a
    # bug rather than noise, so it is reported instead of silently fixed.
    if cos_theta2 > 1.0:
        if cos_theta2 - 1.0 > ACOS_DOMAIN_EPS:
            return IKResult(
                target=(tx, ty),
                geometrically_reachable=False,
                solutions=(),
                chosen=None,
                message="Target lies outside the reachable workspace.",
            )
        cos_theta2 = 1.0
    elif cos_theta2 < -1.0:
        if -1.0 - cos_theta2 > ACOS_DOMAIN_EPS:
            return IKResult(
                target=(tx, ty),
                geometrically_reachable=False,
                solutions=(),
                chosen=None,
                message="Target lies inside the unreachable inner zone.",
            )
        cos_theta2 = -1.0

    theta2_magnitude = math.acos(cos_theta2)

    # The two elbow configurations. When theta2 is 0 or pi the arm is fully
    # extended or fully folded and both branches coincide, so only one is kept.
    candidates: list[float] = [theta2_magnitude]
    if abs(math.sin(theta2_magnitude)) > 1e-12:
        candidates.append(-theta2_magnitude)

    base_angle = math.atan2(y, x)
    solutions: list[IKSolution] = []

    for t2 in candidates:
        t1 = base_angle - math.atan2(l2 * math.sin(t2), l1 + l2 * math.cos(t2))
        theta1_deg = _normalise_angle(math.degrees(t1))
        theta2_deg = _normalise_angle(math.degrees(t2))

        within_limits = is_within_joint_limits(spec, theta1_deg, theta2_deg)

        collision_free: Optional[bool] = None
        if obstacles is not None:
            collision_free = not configuration_collides(
                spec, theta1_deg, theta2_deg, obstacles
            )

        error = target_error(spec, theta1_deg, theta2_deg, (tx, ty))

        if current is None:
            travel = 0.0
        else:
            travel = abs(theta1_deg - current[0]) + abs(theta2_deg - current[1])

        solutions.append(
            IKSolution(
                theta1_deg=theta1_deg,
                theta2_deg=theta2_deg,
                elbow="up" if theta2_deg > 0 else ("down" if theta2_deg < 0 else "straight"),
                within_joint_limits=within_limits,
                collision_free=collision_free,
                position_error=error,
                joint_travel=travel,
            )
        )

    usable = [s for s in solutions if s.usable]
    if usable:
        # Prefer the pose that moves the joints least, which is both the
        # quickest motion and the least surprising one to watch.
        chosen = min(usable, key=lambda s: s.joint_travel)
        message = (
            f"Elbow-{chosen.elbow} solution: "
            f"J1 = {chosen.theta1_deg:.2f}°, J2 = {chosen.theta2_deg:.2f}°, "
            f"tip error {chosen.position_error:.3f} u."
        )
        return IKResult(
            target=(tx, ty),
            geometrically_reachable=True,
            solutions=tuple(solutions),
            chosen=chosen,
            message=message,
        )

    legal = [s for s in solutions if s.within_joint_limits]
    if not legal:
        detail = ", ".join(
            f"J1 = {s.theta1_deg:.1f}°, J2 = {s.theta2_deg:.1f}°" for s in solutions
        )
        message = (
            "The target is geometrically reachable, but every solution violates a "
            f"joint limit ({detail}). Widen the joint limits or choose another target."
        )
    else:
        message = (
            "The target is reachable and within the joint limits, but every solution "
            "puts the arm inside an obstacle. Move or remove an obstacle, or choose "
            "another target."
        )

    return IKResult(
        target=(tx, ty),
        geometrically_reachable=True,
        solutions=tuple(solutions),
        chosen=None,
        message=message,
    )


def end_effector_path_length(
    spec: RobotSpec, path: Sequence[tuple[float, float]]
) -> float:
    """Total distance travelled by the tip along a sequence of poses."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    previous = end_effector_position(spec, path[0][0], path[0][1])
    for theta1, theta2 in path[1:]:
        current = end_effector_position(spec, theta1, theta2)
        total += math.hypot(current[0] - previous[0], current[1] - previous[1])
        previous = current
    return total


def total_joint_travel(path: Sequence[tuple[float, float]]) -> float:
    """Sum of the absolute joint-angle changes along a path, in degrees.

    This differs from the planner's path cost: the cost uses the Euclidean
    norm of each step in joint space, whereas this uses the Manhattan sum.
    """
    if len(path) < 2:
        return 0.0
    total = 0.0
    for (a1, a2), (b1, b2) in zip(path, path[1:]):
        total += abs(b1 - a1) + abs(b2 - a2)
    return total
