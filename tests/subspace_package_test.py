#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101

"""
Tests for the modules in the subspace package.

Each module has doctests for its functions and all we need to do is run all
of them them.
"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
try:
    parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    sys.path.append(parent_dir)
except NameError:  # pragma: no cover
    sys.path.append('../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import unittest
import doctest
import numpy as np

from pyphysim.subspace import metrics, projections
from pyphysim.util.misc import randn_c


# noinspection PyMethodMayBeStatic
class SubspaceDoctestsTestCase(unittest.TestCase):
    """
    Test case that run all the doctests in the modules of the subspace package.
    """
    def test_metrics(self):
        """Run metrics doctests"""
        doctest.testmod(metrics)

    def test_projections(self):
        """Run projections doctests"""
        doctest.testmod(projections)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Projections Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ProjectionsTestCase(unittest.TestCase):
    """Unittests for the Projection class in the projections module."""

    def setUp(self):
        """Called before each test."""
        self.A = np.array([[1 + 1j, 2 - 2j], [3 - 2j, 0], [-1 - 1j, 2 - 3j]])
        self.P_obj = projections.Projection(self.A)
        self.v = np.array([1, 2, 3])

    def test_calcProjectionMatrix(self):
        Q = projections.calcProjectionMatrix(self.A)

        expected_Q = np.array(
            [[0.5239436 - 0.j, 0.0366197 + 0.3295774j,
              0.3661971 + 0.0732394j],
             [0.0366197 - 0.3295774j, 0.7690140 + 0.j,
              -0.0788732 + 0.2478873j],
             [0.3661971 - 0.0732394j, -0.0788732 - 0.2478873j,
              0.7070422 - 0j]])
        np.testing.assert_array_almost_equal(Q, expected_Q)

    def test_calcOrthogonalProjectionMatrix(self):
        Q = projections.calcProjectionMatrix(self.A)
        oQ = projections.calcOrthogonalProjectionMatrix(self.A)
        expected_oQ = np.eye(3) - Q
        np.testing.assert_array_almost_equal(oQ, expected_oQ)

    def test_project(self):
        v_proj = self.P_obj.project(self.v)
        expected_v_proj = np.array(
            [1.69577465 + 0.87887324j, 1.33802817 + 0.41408451j,
             2.32957746 - 0.56901408j])
        np.testing.assert_array_almost_equal(v_proj, expected_v_proj)

    def test_oProject(self):
        v_oproj = self.P_obj.oProject(self.v)
        expected_v_oproj = np.array(
            [-0.69577465 - 0.87887324j, 0.66197183 - 0.41408451j,
             0.67042254 + 0.56901408j])
        np.testing.assert_array_almost_equal(v_oproj, expected_v_oproj)

    def test_reflect(self):
        v_reflec = self.P_obj.reflect(self.v)
        expected_v_reflec = np.array(
            [-2.39154930 - 1.75774648j, -0.67605634 - 0.82816901j,
             -1.65915493 + 1.13802817j])
        np.testing.assert_array_almost_equal(v_reflec, expected_v_reflec)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Metrics Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MetricsTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.A = np.arange(1, 9.)
        self.A.shape = (4, 2)
        self.B = np.array([[1.2, 2.1], [2.9, 4.3], [5.2, 6.1], [6.8, 8.1]])

    def test_calc_principal_angles(self):
        np.testing.assert_array_almost_equal(
            metrics.calc_principal_angles(self.A, self.B),
            np.array([0.00796407, 0.49360193]))

        # Example from the Matrix computations book
        A = np.array([1, 2, 3, 4, 5, 6])
        A.shape = (3, 2)
        B = np.array([1, 5, 3, 7, 5, -1])
        B.shape = (3, 2)
        np.testing.assert_array_almost_equal(
            metrics.calc_principal_angles(A, B),
            np.array([0., 0.54312217]))

    def test_calculating_the_chordal_distance(self):
        expected_chord_dist = 0.473867859572

        # Test calcChordalDistance
        self.assertAlmostEqual(
            metrics.calc_chordal_distance(self.A, self.B),
            expected_chord_dist)

        # Test calcChordalDistance2
        self.assertAlmostEqual(
            metrics.calc_chordal_distance_2(self.A, self.B),
            expected_chord_dist)

        # Test
        principal_angles = metrics.calc_principal_angles(self.A, self.B)
        self.assertAlmostEqual(
            metrics.calc_chordal_distance_from_principal_angles(
                principal_angles),
            expected_chord_dist)

        # xxxxxxxxxx Now let's test with complex values xxxxxxxxxxxxxxxxxxx
        A = randn_c(3, 2)
        B = randn_c(3, 2)
        principal_angles2 = metrics.calc_principal_angles(A, B)
        dist1 = metrics.calc_chordal_distance(A, B)
        dist2 = metrics.calc_chordal_distance_2(A, B)
        dist3 = metrics.calc_chordal_distance_from_principal_angles(
            principal_angles2)

        self.assertAlmostEqual(dist1, dist2)
        self.assertAlmostEqual(dist3, dist2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
