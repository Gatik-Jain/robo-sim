"""Tests for the planar-geometry primitives."""

import math
import unittest

from robo_sim.geometry import (
    capsule_intersects_rect,
    disc_intersects_rect,
    point_distance,
    point_in_rect,
    point_to_rect_distance,
    point_to_segment_distance,
    rects_overlap,
    segment_intersects_rect,
    segment_to_rect_distance,
    segment_to_segment_distance,
    segments_intersect,
)
from robo_sim.models import InvalidInputError, Rectangle


class RectangleTests(unittest.TestCase):
    def test_dimensions(self):
        rect = Rectangle(10.0, 40.0, -5.0, 15.0)
        self.assertEqual(rect.width, 30.0)
        self.assertEqual(rect.height, 20.0)
        self.assertEqual(rect.center, (25.0, 5.0))

    def test_zero_width_rejected(self):
        with self.assertRaises(InvalidInputError):
            Rectangle(10.0, 10.0, 0.0, 5.0)

    def test_inverted_bounds_rejected(self):
        with self.assertRaises(InvalidInputError):
            Rectangle(40.0, 10.0, 0.0, 5.0)
        with self.assertRaises(InvalidInputError):
            Rectangle(0.0, 5.0, 40.0, 10.0)

    def test_non_finite_bounds_rejected(self):
        with self.assertRaises(InvalidInputError):
            Rectangle(float("nan"), 10.0, 0.0, 5.0)
        with self.assertRaises(InvalidInputError):
            Rectangle(0.0, float("inf"), 0.0, 5.0)

    def test_corners_are_counter_clockwise(self):
        rect = Rectangle(0.0, 10.0, 0.0, 4.0)
        self.assertEqual(
            rect.corners(), ((0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0))
        )


class PointTests(unittest.TestCase):
    def test_point_distance(self):
        self.assertAlmostEqual(point_distance((0.0, 0.0), (3.0, 4.0)), 5.0)

    def test_point_in_rect_including_boundary(self):
        rect = Rectangle(0.0, 10.0, 0.0, 10.0)
        self.assertTrue(point_in_rect((5.0, 5.0), rect))
        self.assertTrue(point_in_rect((0.0, 0.0), rect))
        self.assertTrue(point_in_rect((10.0, 10.0), rect))
        self.assertFalse(point_in_rect((10.001, 5.0), rect))

    def test_distance_from_inside_is_zero(self):
        rect = Rectangle(0.0, 10.0, 0.0, 10.0)
        self.assertEqual(point_to_rect_distance((5.0, 5.0), rect), 0.0)

    def test_distance_to_an_edge(self):
        rect = Rectangle(0.0, 10.0, 0.0, 10.0)
        self.assertAlmostEqual(point_to_rect_distance((15.0, 5.0), rect), 5.0)
        self.assertAlmostEqual(point_to_rect_distance((5.0, -3.0), rect), 3.0)

    def test_distance_to_a_corner(self):
        rect = Rectangle(0.0, 10.0, 0.0, 10.0)
        # Diagonally outside the top-right corner.
        self.assertAlmostEqual(point_to_rect_distance((13.0, 14.0), rect), 5.0)


class PointToSegmentTests(unittest.TestCase):
    def test_perpendicular_foot_inside_the_segment(self):
        self.assertAlmostEqual(
            point_to_segment_distance((5.0, 7.0), (0.0, 0.0), (10.0, 0.0)), 7.0
        )

    def test_projection_clamped_to_an_endpoint(self):
        self.assertAlmostEqual(
            point_to_segment_distance((15.0, 0.0), (0.0, 0.0), (10.0, 0.0)), 5.0
        )
        self.assertAlmostEqual(
            point_to_segment_distance((-3.0, 4.0), (0.0, 0.0), (10.0, 0.0)), 5.0
        )

    def test_point_on_the_segment(self):
        self.assertAlmostEqual(
            point_to_segment_distance((4.0, 0.0), (0.0, 0.0), (10.0, 0.0)), 0.0
        )

    def test_degenerate_segment_behaves_as_a_point(self):
        self.assertAlmostEqual(
            point_to_segment_distance((3.0, 4.0), (0.0, 0.0), (0.0, 0.0)), 5.0
        )


class SegmentIntersectionTests(unittest.TestCase):
    def test_crossing_segments(self):
        self.assertTrue(
            segments_intersect((0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0))
        )

    def test_parallel_segments_do_not_intersect(self):
        self.assertFalse(
            segments_intersect((0.0, 0.0), (10.0, 0.0), (0.0, 5.0), (10.0, 5.0))
        )

    def test_touching_endpoint_counts_as_intersection(self):
        self.assertTrue(
            segments_intersect((0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (10.0, 10.0))
        )

    def test_collinear_overlapping(self):
        self.assertTrue(
            segments_intersect((0.0, 0.0), (10.0, 0.0), (5.0, 0.0), (15.0, 0.0))
        )

    def test_collinear_disjoint(self):
        self.assertFalse(
            segments_intersect((0.0, 0.0), (10.0, 0.0), (12.0, 0.0), (20.0, 0.0))
        )

    def test_segments_that_would_cross_if_extended_do_not(self):
        self.assertFalse(
            segments_intersect((0.0, 0.0), (1.0, 1.0), (8.0, 10.0), (10.0, 8.0))
        )


class SegmentToSegmentDistanceTests(unittest.TestCase):
    def test_crossing_gives_zero(self):
        self.assertEqual(
            segment_to_segment_distance(
                (0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0)
            ),
            0.0,
        )

    def test_parallel_separation(self):
        self.assertAlmostEqual(
            segment_to_segment_distance(
                (0.0, 0.0), (10.0, 0.0), (0.0, 4.0), (10.0, 4.0)
            ),
            4.0,
        )

    def test_endpoint_to_endpoint(self):
        self.assertAlmostEqual(
            segment_to_segment_distance(
                (0.0, 0.0), (10.0, 0.0), (13.0, 4.0), (20.0, 20.0)
            ),
            5.0,
        )


class SegmentRectangleTests(unittest.TestCase):
    def setUp(self):
        self.rect = Rectangle(0.0, 10.0, 0.0, 10.0)

    def test_segment_passing_through(self):
        self.assertTrue(segment_intersects_rect((-5.0, 5.0), (15.0, 5.0), self.rect))
        self.assertEqual(
            segment_to_rect_distance((-5.0, 5.0), (15.0, 5.0), self.rect), 0.0
        )

    def test_segment_entirely_inside(self):
        self.assertTrue(segment_intersects_rect((2.0, 2.0), (8.0, 8.0), self.rect))

    def test_segment_with_one_endpoint_inside(self):
        self.assertTrue(segment_intersects_rect((5.0, 5.0), (50.0, 50.0), self.rect))

    def test_segment_clear_of_the_rectangle(self):
        self.assertFalse(segment_intersects_rect((-5.0, 20.0), (15.0, 20.0), self.rect))
        self.assertAlmostEqual(
            segment_to_rect_distance((-5.0, 20.0), (15.0, 20.0), self.rect), 10.0
        )

    def test_distance_is_measured_to_the_nearest_corner(self):
        """A segment angled past a corner: the corner is the closest feature."""
        # The line x + y = 25 passes 5/sqrt(2) from the corner (10, 10), and
        # that corner is the nearest point of the rectangle to it.
        distance = segment_to_rect_distance((0.0, 25.0), (25.0, 0.0), self.rect)
        self.assertAlmostEqual(distance, 5.0 / math.sqrt(2.0), places=9)

    def test_segment_grazing_a_corner_exactly_gives_zero(self):
        # The line x + y = 20 passes exactly through the corner (10, 10).
        distance = segment_to_rect_distance((0.0, 20.0), (20.0, 0.0), self.rect)
        self.assertEqual(distance, 0.0)

    def test_touching_the_boundary_gives_zero_distance(self):
        self.assertAlmostEqual(
            segment_to_rect_distance((10.0, -5.0), (10.0, 15.0), self.rect), 0.0
        )

    def test_segment_just_short_of_the_rectangle(self):
        self.assertAlmostEqual(
            segment_to_rect_distance((-1.0, -5.0), (-1.0, 15.0), self.rect), 1.0
        )


class CapsuleTests(unittest.TestCase):
    def setUp(self):
        self.rect = Rectangle(0.0, 10.0, 0.0, 10.0)

    def test_thin_capsule_clears_what_a_thick_one_hits(self):
        """The whole point of modelling thickness: same centre line, different verdict."""
        seg_a, seg_b = (-5.0, -4.0), (15.0, -4.0)  # 4 units below the rectangle
        self.assertFalse(capsule_intersects_rect(seg_a, seg_b, 3.0, self.rect))
        self.assertTrue(capsule_intersects_rect(seg_a, seg_b, 5.0, self.rect))

    def test_exact_contact_counts_as_a_collision(self):
        seg_a, seg_b = (-5.0, -4.0), (15.0, -4.0)
        self.assertTrue(capsule_intersects_rect(seg_a, seg_b, 4.0, self.rect))

    def test_capsule_hitting_a_corner_diagonally(self):
        """Near a corner the test must use the true radial distance."""
        # A point-like capsule at (13, 14): 5 units from the corner (10, 10).
        seg = ((13.0, 14.0), (13.0, 14.0))
        self.assertFalse(capsule_intersects_rect(seg[0], seg[1], 4.9, self.rect))
        self.assertTrue(capsule_intersects_rect(seg[0], seg[1], 5.1, self.rect))

    def test_zero_radius_capsule_is_just_the_segment(self):
        self.assertTrue(
            capsule_intersects_rect((-5.0, 5.0), (15.0, 5.0), 0.0, self.rect)
        )
        self.assertFalse(
            capsule_intersects_rect((-5.0, 11.0), (15.0, 11.0), 0.0, self.rect)
        )


class DiscTests(unittest.TestCase):
    def setUp(self):
        self.rect = Rectangle(0.0, 10.0, 0.0, 10.0)

    def test_disc_centre_inside(self):
        self.assertTrue(disc_intersects_rect((5.0, 5.0), 1.0, self.rect))

    def test_disc_overlapping_an_edge(self):
        self.assertTrue(disc_intersects_rect((12.0, 5.0), 3.0, self.rect))

    def test_disc_clear_of_the_rectangle(self):
        self.assertFalse(disc_intersects_rect((12.0, 5.0), 1.5, self.rect))

    def test_disc_touching_exactly(self):
        self.assertTrue(disc_intersects_rect((12.0, 5.0), 2.0, self.rect))

    def test_disc_near_a_corner_uses_radial_distance(self):
        # (14, 14) is 4*sqrt(2) ~ 5.657 from the corner (10, 10).
        self.assertFalse(disc_intersects_rect((14.0, 14.0), 5.0, self.rect))
        self.assertTrue(disc_intersects_rect((14.0, 14.0), 6.0, self.rect))


class RectOverlapTests(unittest.TestCase):
    def test_overlapping(self):
        self.assertTrue(
            rects_overlap(Rectangle(0, 10, 0, 10), Rectangle(5, 15, 5, 15))
        )

    def test_disjoint(self):
        self.assertFalse(
            rects_overlap(Rectangle(0, 10, 0, 10), Rectangle(11, 15, 0, 10))
        )

    def test_touching_counts_as_overlap(self):
        self.assertTrue(
            rects_overlap(Rectangle(0, 10, 0, 10), Rectangle(10, 15, 0, 10))
        )


if __name__ == "__main__":
    unittest.main()
