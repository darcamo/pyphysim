#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the subspace package.

Each module has doctests for its functions and all we need to do is run all
of them them.
"""

import unittest
import doctest
import numpy as np

import sys
sys.path.append("../")

from subspace import metrics, projections


class SubspaceDoctestsTestCase(unittest.TestCase):
    """Teste case that run all the doctests in the modules of the subspace
    package.
    """
    def test_metrics(self):
        """Run metrics doctests"""
        doctest.testmod(metrics)

    def test_projections(self):
        """Run projections doctests"""
        doctest.testmod(projections)
        pass


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
            [[0.5239436 - 0.j, 0.0366197 + 0.3295774j, 0.3661971 + 0.0732394j],
             [0.0366197 - 0.3295774j, 0.7690140 + 0.j, -0.0788732 + 0.2478873j],
             [0.3661971 - 0.0732394j, -0.0788732 - 0.2478873j, 0.7070422 - 0j]])
        np.testing.assert_array_almost_equal(Q, expected_Q)

    def test_calcOrthogonalProjectionMatrix(self):
        Q = projections.calcProjectionMatrix(self.A)
        oQ = projections.calcOrthogonalProjectionMatrix(self.A)
        expected_oQ = np.eye(3) - Q
        np.testing.assert_array_almost_equal(oQ, expected_oQ)

    def test_project(self):
        v_proj = self.P_obj.project(self.v)
        expected_v_proj = np.array(
            [1.69577465 + 0.87887324j, 1.33802817 + 0.41408451j, 2.32957746 - 0.56901408j])
        np.testing.assert_array_almost_equal(v_proj, expected_v_proj)

    def test_oProject(self):
        v_oproj = self.P_obj.oProject(self.v)
        expected_v_oproj = np.array(
            [-0.69577465 - 0.87887324j, 0.66197183 - 0.41408451j, 0.67042254 + 0.56901408j])
        np.testing.assert_array_almost_equal(v_oproj, expected_v_oproj)

    def test_reflect(self):
        v_reflec = self.P_obj.reflect(self.v)
        expected_v_reflec = np.array([-2.39154930 - 1.75774648j, -0.67605634 - 0.82816901j, -1.65915493 + 1.13802817j])
        np.testing.assert_array_almost_equal(v_reflec, expected_v_reflec)

# xxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
