"""Collision and validity checking for complete robot configurations.

This module answers two questions, and it is the *only* place in ROBO-SIM that
answers them. Manual joint control, inverse kinematics, the planner, animation
validation and the unit tests all call these same functions, so they can never
disagree about what counts as a collision.

1. Is a single pose legal?           -> :func:`is_configuration_valid`
2. Is the motion between two poses
   legal all the way through?        -> :func:`is_transition_valid`

The second question matters because two safe poses can be separated by unsafe
ones. An arm can be clear of an obstacle before a move and clear after it,
while sweeping straight through the obstacle in between. Validating only the
endpoints of a motion would miss exactly that case.

**Collision model.** Each link is a capsule: the set of points within
``link_radius`` of the link's centre line. The end effector is a disc of
``end_effector_radius`` at the tip. Both tests use exact distances from
:mod:`robo_sim.geometry`, so there is no conservative corner approximation to
account for. Contact exactly at the boundary is treated as a collision.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

from . import config
from .geometry import capsule_intersects_rect, disc_intersects_rect
from .kinematics import forward_kinematics, is_within_joint_limits
from .models import ArmGeometry, Rectangle, RobotSpec, ValidationResult


def geometry_collides(
    geometry: ArmGeometry,
    spec: RobotSpec,
    obstacles: Iterable[Rectangle],
) -> Optional[Rectangle]:
    """Return the first obstacle the arm touches, or None if it is clear.

    Returning the obstacle rather than a bare boolean lets the interface say
    *which* rectangle is in the way.
    """
    base = geometry.base
    joint1 = geometry.joint1
    tip = geometry.end_effector

    for rect in obstacles:
        # Link 1: base -> elbow.
        if capsule_intersects_rect(base, joint1, spec.link_radius, rect, config.CONTACT_EPS):
            return rect
        # Link 2: elbow -> tip.
        if capsule_intersects_rect(joint1, tip, spec.link_radius, rect, config.CONTACT_EPS):
            return rect
        # The gripper disc at the tip, which is thicker than the links.
        if disc_intersects_rect(tip, spec.end_effector_radius, rect, config.CONTACT_EPS):
            return rect
    return None


def configuration_collides(
    spec: RobotSpec,
    theta1_deg: float,
    theta2_deg: float,
    obstacles: Iterable[Rectangle],
) -> bool:
    """True when the pose puts any part of the arm inside an obstacle.

    Joint limits are *not* considered here -- see
    :func:`is_configuration_valid` for the combined test.
    """
    geometry = forward_kinematics(spec, theta1_deg, theta2_deg)
    return geometry_collides(geometry, spec, obstacles) is not None


def is_configuration_valid(
    spec: RobotSpec,
    theta1_deg: float,
    theta2_deg: float,
    obstacles: Sequence[Rectangle] = (),
) -> ValidationResult:
    """Full validity test for one pose: joint limits *and* collisions."""
    if not (math.isfinite(theta1_deg) and math.isfinite(theta2_deg)):
        return ValidationResult(False, "Joint angles must be finite numbers.")

    if not is_within_joint_limits(spec, theta1_deg, theta2_deg):
        return ValidationResult(
            False,
            f"Joint angles ({theta1_deg:.1f}°, {theta2_deg:.1f}°) are outside the "
            f"limits J1 [{spec.joint1_min:g}°, {spec.joint1_max:g}°], "
            f"J2 [{spec.joint2_min:g}°, {spec.joint2_max:g}°].",
        )

    geometry = forward_kinematics(spec, theta1_deg, theta2_deg)
    hit = geometry_collides(geometry, spec, obstacles)
    if hit is not None:
        return ValidationResult(
            False, f"The arm collides with the obstacle at {hit.describe()}."
        )

    return ValidationResult(True, "Configuration is within limits and collision free.")


def transition_samples(
    start: tuple[float, float],
    end: tuple[float, float],
    max_step_deg: float = config.TRANSITION_STEP_DEG,
) -> list[tuple[float, float]]:
    """Interpolate between two poses so no joint moves more than ``max_step_deg``.

    Both joints are interpolated linearly and simultaneously, which is what the
    animation does, so the samples describe the motion that is actually shown.
    The list always includes both endpoints.
    """
    d1 = end[0] - start[0]
    d2 = end[1] - start[1]
    largest = max(abs(d1), abs(d2))

    if largest == 0.0:
        return [start]

    steps = max(1, int(math.ceil(largest / max_step_deg)))
    return [
        (start[0] + d1 * i / steps, start[1] + d2 * i / steps)
        for i in range(steps + 1)
    ]


def is_transition_valid(
    spec: RobotSpec,
    start: tuple[float, float],
    end: tuple[float, float],
    obstacles: Sequence[Rectangle] = (),
    max_step_deg: float = config.TRANSITION_STEP_DEG,
) -> ValidationResult:
    """Check every intermediate pose along a straight joint-space move.

    The move is sampled at no coarser than ``max_step_deg`` of joint rotation
    (one degree by default, finer than the finest planning grid of three
    degrees). Sampling can in principle step over an obstacle thinner than the
    distance the arm covers between samples; with a one-degree step the tip of
    a 300 unit arm moves about 5 units per sample, so obstacles should be
    comfortably larger than that. This limitation is documented in the README.
    """
    if not obstacles:
        # With no obstacles only the joint limits can make a pose illegal, and
        # linear interpolation between two in-range angles stays in range.
        for pose in (start, end):
            result = is_configuration_valid(spec, pose[0], pose[1], ())
            if not result.valid:
                return result
        return ValidationResult(True, "Transition is collision free (no obstacles).")

    for theta1, theta2 in transition_samples(start, end, max_step_deg):
        result = is_configuration_valid(spec, theta1, theta2, obstacles)
        if not result.valid:
            return ValidationResult(
                False,
                f"The move passes through an invalid pose at "
                f"({theta1:.1f}°, {theta2:.1f}°): {result.reason}",
            )
    return ValidationResult(True, "Every sampled pose along the move is valid.")


def validate_path(
    spec: RobotSpec,
    path: Sequence[tuple[float, float]],
    obstacles: Sequence[Rectangle] = (),
    max_step_deg: float = config.TRANSITION_STEP_DEG,
) -> ValidationResult:
    """Validate a whole path: every pose and every transition between them.

    This runs before any trajectory is animated, so the motion shown to the
    user has been checked at the same density the planner used -- never just
    at its endpoints.
    """
    if not path:
        return ValidationResult(False, "The path is empty.")

    first = is_configuration_valid(spec, path[0][0], path[0][1], obstacles)
    if not first.valid:
        return ValidationResult(False, f"Start pose is invalid: {first.reason}")

    for index, (a, b) in enumerate(zip(path, path[1:])):
        result = is_transition_valid(spec, a, b, obstacles, max_step_deg)
        if not result.valid:
            return ValidationResult(
                False, f"Step {index + 1} of {len(path) - 1} is invalid: {result.reason}"
            )

    return ValidationResult(
        True,
        f"All {len(path)} configurations and {max(0, len(path) - 1)} transitions "
        f"validated at a {max_step_deg:g}° sampling step.",
    )
