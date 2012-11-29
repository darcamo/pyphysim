#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the comp package.

Each module has several doctests that we run in addition to the unittests
defined here.
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

from comm import channels
from comp import comp
import util
from util.misc import least_right_singular_vectors


# UPDATE THIS CLASS if another module is added to the comm package
class CompDoctestsTestCase(unittest.TestCase):
    """Teste case that run all the doctests in the modules of the comp
    package.
    """

    def test_test_comp(self):
        """Run doctests in the comp module."""
        doctest.testmod(comp)


class CompModuleFunctionsTestCase(unittest.TestCase):
    def test_calc_stream_reduction_matrix(self):
        Re_k = util.misc.randn_c(3, 2)
        Re_k = np.dot(Re_k, Re_k.transpose().conjugate())

        P1 = comp._calc_stream_reduction_matrix(Re_k, 1)
        P2 = comp._calc_stream_reduction_matrix(Re_k, 2)
        P3 = comp._calc_stream_reduction_matrix(Re_k, 3)

        (min_Vs, remaining_Vs, S) = least_right_singular_vectors(Re_k, 3)
        expected_P1 = min_Vs[:, :1]
        expected_P2 = min_Vs[:, :2]
        expected_P3 = min_Vs[:, :3]

        np.testing.assert_array_almost_equal(P1, expected_P1)
        np.testing.assert_array_almost_equal(P2, expected_P2)
        np.testing.assert_array_almost_equal(P3, expected_P3)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx COMP Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TODO: finish implementation
class CompExtInt(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    # def test_calc_receive_filter(self):
    #     pass

    # def test_calc_SNRs(self):
    #     pass

    def test_perform_comp(self):
        Nr = np.array([2, 2])
        Nt = np.array([2, 2])
        K = Nt.size
        Nti = 1
        iPu = 0.8  # Power for each user
        noise_var = 1e-50

        multiUserChannel = channels.MultiUserChannelMatrixExtInt()
        multiUserChannel.randomize(Nr, Nt, K, Nti)

        # Create the comp object
        comp_obj = comp.CompExtInt(K, iPu, noise_var)
        comp_obj.perform_comp(multiUserChannel, noise_var)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    # plot_psd_OFDM_symbols()
    unittest.main()
