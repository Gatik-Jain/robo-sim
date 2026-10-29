"""Reusable planar-geometry primitives.

Nothing in this module knows about robots. It deals only in points, segments
and axis-aligned rectangles, which makes every function here straightforward
to test in isolation.

The distances computed here are **exact**, not approximations. The key routine
is :func:`segment_to_rect_distance`, which combines with a radius to give exact
capsule-versus-rectangle collision tests, so there is no conservative margin to
reason about near rectangle corners.
"""

from __future__ import annotations

import math

from .models import Rectangle

Point = tuple[float, float]
Segment = tuple[Point, Point]


def point_distance(a: Point, b: Point) -> float:
    """Straight-line distance between two points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def point_in_rect(point: Point, rect: Rectangle) -> bool:
    """True when the point is inside the rectangle or on its boundary."""
    x, y = point
    return rect.min_x <= x <= rect.max_x and rect.min_y <= y <= rect.max_y


def point_to_rect_distance(point: Point, rect: Rectangle) -> float:
    """Shortest distance from a point to a rectangle; 0 if inside.

    Each axis contributes independently: how far the point is outside the
    rectangle's span on that axis. Combining them with ``hypot`` handles the
    corner regions correctly without any special cases.
    """
    x, y = point
    dx = max(rect.min_x - x, 0.0, x - rect.max_x)
    dy = max(rect.min_y - y, 0.0, y - rect.max_y)
    return math.hypot(dx, dy)


def point_to_segment_distance(point: Point, seg_a: Point, seg_b: Point) -> float:
    """Shortest distance from a point to a finite line segment.

    The point is projected onto the infinite line through the segment and the
    projection parameter is clamped to [0, 1], so the nearest point is either
    the perpendicular foot or one of the two endpoints.
    """
    ax, ay = seg_a
    bx, by = seg_b
    px, py = point

    dx = bx - ax
    dy = by - ay
    length_squared = dx * dx + dy * dy

    if length_squared == 0.0:
        # Degenerate segment: it is really just a point.
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * dx + (py - ay) * dy) / length_squared
    t = max(0.0, min(1.0, t))

    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def _orientation(a: Point, b: Point, c: Point) -> float:
    """Signed area of the triangle abc (positive when counter-clockwise)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    """True when collinear point ``c`` lies within the bounding box of ab."""
    return (
        min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    )


def segments_intersect(
    a1: Point, a2: Point, b1: Point, b2: Point, eps: float = 1e-12
) -> bool:
    """True when two closed segments share at least one point.

    Uses the standard orientation test, with the collinear cases handled
    explicitly so that touching endpoints count as an intersection.
    """
    d1 = _orientation(b1, b2, a1)
    d2 = _orientation(b1, b2, a2)
    d3 = _orientation(a1, a2, b1)
    d4 = _orientation(a1, a2, b2)

    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and (
        (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)
    ):
        return True

    # Collinear / touching cases.
    if abs(d1) <= eps and _on_segment(b1, b2, a1):
        return True
    if abs(d2) <= eps and _on_segment(b1, b2, a2):
        return True
    if abs(d3) <= eps and _on_segment(a1, a2, b1):
        return True
    if abs(d4) <= eps and _on_segment(a1, a2, b2):
        return True
    return False


def segment_to_segment_distance(
    a1: Point, a2: Point, b1: Point, b2: Point
) -> float:
    """Shortest distance between two finite segments.

    If they cross the distance is zero; otherwise the minimum is attained at
    an endpoint of one segment, so checking all four endpoint-to-segment
    distances is exact.
    """
    if segments_intersect(a1, a2, b1, b2):
        return 0.0
    return min(
        point_to_segment_distance(a1, b1, b2),
        point_to_segment_distance(a2, b1, b2),
        point_to_segment_distance(b1, a1, a2),
        point_to_segment_distance(b2, a1, a2),
    )


def segment_intersects_rect(seg_a: Point, seg_b: Point, rect: Rectangle) -> bool:
    """True when a segment touches or enters a rectangle."""
    # Cheap rejection: if the segment's bounding box misses the rectangle's,
    # they cannot touch.
    if (
        max(seg_a[0], seg_b[0]) < rect.min_x
        or min(seg_a[0], seg_b[0]) > rect.max_x
        or max(seg_a[1], seg_b[1]) < rect.min_y
        or min(seg_a[1], seg_b[1]) > rect.max_y
    ):
        return False

    # Either endpoint inside means the segment is at least partly inside.
    if point_in_rect(seg_a, rect) or point_in_rect(seg_b, rect):
        return True

    # Otherwise it must cross one of the four edges to reach the interior.
    corners = rect.corners()
    for i in range(4):
        if segments_intersect(seg_a, seg_b, corners[i], corners[(i + 1) % 4]):
            return True
    return False


def segment_to_rect_distance(seg_a: Point, seg_b: Point, rect: Rectangle) -> float:
    """Exact shortest distance between a segment and a rectangle.

    Zero when they touch or overlap. Otherwise the two shapes are disjoint
    convex sets, so the minimum distance is attained either at a segment
    endpoint (nearest to the rectangle) or at a rectangle corner (nearest to
    the segment). Both families are checked, which also handles the corner
    regions exactly -- no conservative approximation is used anywhere.
    """
    if segment_intersects_rect(seg_a, seg_b, rect):
        return 0.0

    best = min(
        point_to_rect_distance(seg_a, rect),
        point_to_rect_distance(seg_b, rect),
    )
    for corner in rect.corners():
        d = point_to_segment_distance(corner, seg_a, seg_b)
        if d < best:
            best = d
    return best


def capsule_intersects_rect(
    seg_a: Point, seg_b: Point, radius: float, rect: Rectangle, eps: float = 1e-9
) -> bool:
    """True when a thick segment (a capsule) overlaps a rectangle.

    A capsule is the set of points within ``radius`` of the segment, which is
    exactly the shape of a physical link of uniform thickness. Touching at the
    boundary counts as a collision: the test is ``distance <= radius + eps``.
    """
    return segment_to_rect_distance(seg_a, seg_b, rect) <= radius + eps


def disc_intersects_rect(
    center: Point, radius: float, rect: Rectangle, eps: float = 1e-9
) -> bool:
    """True when a disc overlaps a rectangle, counting boundary contact."""
    return point_to_rect_distance(center, rect) <= radius + eps


def rects_overlap(a: Rectangle, b: Rectangle) -> bool:
    """True when two axis-aligned rectangles share any area or boundary."""
    return not (
        a.max_x < b.min_x
        or b.max_x < a.min_x
        or a.max_y < b.min_y
        or b.max_y < a.min_y
    )
