#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the MATLAB package.

Each module has doctests for its functions and all we need to do is run all
of them.
"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import unittest
import doctest
import numpy as np

from MATLAB import python2MATLAB


class MATLABDoctestsTestCase(unittest.TestCase):
    """Test case that run all the doctests in the modules of the MATLAB
    package.

    """
    def test_python2MATLAB(self):
        """Run python2MATLAB doctests"""
        doctest.testmod(python2MATLAB)


# TODO: finish implementation
class MATLABFunctionsTestCase(unittest.TestCase):
    def test_mmat(self):
        # Test 1D numpy array
        a = np.arange(10) + (10 - np.arange(10)) * 1j

        conv_a = python2MATLAB.to_mat_str(a)
        expected_conv_a = '[+0.000000000000e+00+1.000000000000e+01j, +1.000000000000e+00+9.000000000000e+00j, +2.000000000000e+00+8.000000000000e+00j, +3.000000000000e+00+7.000000000000e+00j, +4.000000000000e+00+6.000000000000e+00j, +5.000000000000e+00+5.000000000000e+00j, +6.000000000000e+00+4.000000000000e+00j, +7.000000000000e+00+3.000000000000e+00j, +8.000000000000e+00+2.000000000000e+00j, +9.000000000000e+00+1.000000000000e+00j]'
        self.assertEqual(conv_a, expected_conv_a)


        # Test a 2D numpy array with a single column (a column vector)
        b = np.arange(1, 5) - np.arange(1, 5) * 1j
        b.shape = (4, 1)
        conv_b = python2MATLAB.to_mat_str(b)
        expected_conv_b = '[+1.000000000000e+00-1.000000000000e+00j; +2.000000000000e+00-2.000000000000e+00j; +3.000000000000e+00-3.000000000000e+00j; +4.000000000000e+00-4.000000000000e+00j]'
        self.assertEqual(conv_b, expected_conv_b)

        # Test a 2D real numpy array
        c = np.arange(1, 10)
        c.shape = (3, 3)
        conv_c = python2MATLAB.to_mat_str(c)
        expected_conv_c = '[+1.000000000000e+00, +2.000000000000e+00, +3.000000000000e+00; +4.000000000000e+00, +5.000000000000e+00, +6.000000000000e+00; +7.000000000000e+00, +8.000000000000e+00, +9.000000000000e+00]'
        self.assertEqual(conv_c, expected_conv_c)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
