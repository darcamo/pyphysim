#!/usr/bin/env python

# pylint: disable=E1101,E0611
"""
Tests for the modules in the pointprocess package.
"""

import unittest

import numpy as np
from matplotlib import pyplot as plt

from pyphysim import pointprocess as pp


class PointProcessTestCase(unittest.TestCase):
    def setUp(self) -> None:
        """Called before each test."""
        pass

    def test_generate_random_points_in_circle(self) -> None:
        # This test will test the `generate_random_points_in_circle` function
        # when only a maximum radius is passed
        num_points = 500000
        radius = 15.0

        points = pp.generate_random_points_in_circle(num_points, radius)
        self.assertEqual(points.size, num_points)
        self.assertLessEqual(np.max(np.abs(points)), radius)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Compute how many points are inside the circle with radius 1 centered
        # at the origin
        small_radius1 = 1.0
        num_smaller_area_points1 = np.sum(np.abs(points) < small_radius1)
        expected_num_smaller_area_points1 = int(
            num_points * (small_radius1 * small_radius1) / (radius * radius))
        ratio1 = num_smaller_area_points1 / expected_num_smaller_area_points1
        self.assertAlmostEqual(ratio1, 1.0, 1)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Compute how many points are inside the circle with radius 2.5 centered
        # at the origin
        small_radius2 = 2.5
        num_smaller_area_points2 = np.sum(np.abs(points) < small_radius2)
        expected_num_smaller_area_points2 = int(
            num_points * (small_radius2 * small_radius2) / (radius * radius))
        ratio2 = num_smaller_area_points2 / expected_num_smaller_area_points2
        self.assertAlmostEqual(ratio2, 1.0, 1)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Compute how many points are inside the circle with radius 2.5 centered
        # at the origin, but now the big circle is centered at 2.5+0i
        small_radius3 = 2.5
        num_smaller_area_points3 = np.sum(np.abs(points + 2.5) < small_radius3)
        expected_num_smaller_area_points3 = int(
            num_points * (small_radius3 * small_radius3) / (radius * radius))
        ratio3 = num_smaller_area_points3 / expected_num_smaller_area_points3
        self.assertAlmostEqual(ratio3, 1.0, 1)

    def test_generate_random_points_in_circle2(self) -> None:
        # This test will test the `generate_random_points_in_circle` function
        # when both a maximum and a minimum radius are passed
        num_points = 500000
        radius = 5.0
        min_radius = 1.0

        points = pp.generate_random_points_in_circle(num_points, radius,
                                                     min_radius)
        self.assertEqual(points.size, num_points)
        self.assertLessEqual(np.max(np.abs(points)), radius)
        self.assertGreaterEqual(np.min(np.abs(points)), min_radius)

        small_radius1 = min_radius  # With this there should be no points inside this circle
        num_smaller_area_points1 = np.sum(np.abs(points) < small_radius1)
        self.assertEqual(num_smaller_area_points1, 0)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Compute how many points are inside the circle with radius 2.5 centered
        # at the origin
        small_radius2 = 2.5
        num_smaller_area_points2 = np.sum(np.abs(points) < small_radius2)
        expected_num_smaller_area_points2 = int(
            num_points * ((small_radius2 - min_radius)**2) /
            ((radius - min_radius)**2))
        ratio2 = num_smaller_area_points2 / expected_num_smaller_area_points2
        self.assertAlmostEqual(ratio2, 1.0, 1)

    def test_generate_random_points_in_rectangle(self) -> None:
        num_points = 50000
        width = 3.0
        height = 2.0

        points = pp.generate_random_points_in_rectangle(
            num_points, width, height)
        self.assertEqual(points.size, num_points)
        self.assertLessEqual(np.max(points.real), width / 2.0)
        self.assertLessEqual(np.max(points.imag), height / 2.0)
        self.assertGreaterEqual(np.min(points.real), -width / 2.0)
        self.assertGreaterEqual(np.min(points.imag), -height / 2.0)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Compute how many points are inside the circle with radius 1 centered
        # at the origin
        small_radius1 = 1.0
        num_smaller_area_points1 = np.sum(np.abs(points) < small_radius1)
        # The expected number of points is proportional to the ratio of the areas
        expected_num_smaller_area_points1 = int(
            num_points * np.pi * (small_radius1 * small_radius1) /
            (width * height))
        ratio1 = num_smaller_area_points1 / expected_num_smaller_area_points1
        self.assertAlmostEqual(ratio1, 1.0, 1)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
