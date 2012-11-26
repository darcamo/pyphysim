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


# UPDATE THIS CLASS if another module is added to the comm package
class CompDoctestsTestCase(unittest.TestCase):
    """Teste case that run all the doctests in the modules of the comp
    package.
    """

    def test_test_comp(self):
        """Run doctests in the comp module."""
        doctest.testmod(comp)


class CompModuleFunctionsTestCase(unittest.TestCase):
    def test_calc_cov_matrix_extint_plus_noise(self):
        int_channel = util.misc.randn_c(6, 1)
        Nr = np.array([2, 2, 2])
        noise_var = 0.001

        H1 = int_channel[0:2, :]
        H2 = int_channel[2:4, :]
        H3 = int_channel[4:6, :]
        noise_cov = np.eye(2) * noise_var

        expected_cov_int = np.empty(3, dtype=np.ndarray)
        expected_cov_int_plus_noise = np.empty(3, dtype=np.ndarray)

        expected_cov_int[0] = np.dot(H1, H1.conjugate().transpose())
        expected_cov_int[1] = np.dot(H2, H2.conjugate().transpose())
        expected_cov_int[2] = np.dot(H3, H3.conjugate().transpose())

        expected_cov_int_plus_noise[0] = expected_cov_int[0] + noise_cov
        expected_cov_int_plus_noise[1] = expected_cov_int[1] + noise_cov
        expected_cov_int_plus_noise[2] = expected_cov_int[2] + noise_cov

        cov_int = comp.calc_cov_matrix_extint_plus_noise(int_channel, Nr)
        cov_int_plus_noise = comp.calc_cov_matrix_extint_plus_noise(
            int_channel,
            Nr,
            noise_var)

        self.assertEqual(cov_int.size, expected_cov_int.size)
        np.testing.assert_array_almost_equal(cov_int[0], expected_cov_int[0])
        np.testing.assert_array_almost_equal(cov_int[1], expected_cov_int[1])
        np.testing.assert_array_almost_equal(cov_int[2], expected_cov_int[2])

        self.assertEqual(cov_int_plus_noise.size,
                         expected_cov_int_plus_noise.size)

        np.testing.assert_array_almost_equal(cov_int_plus_noise[0],
                                             expected_cov_int_plus_noise[0])
        np.testing.assert_array_almost_equal(cov_int_plus_noise[1],
                                             expected_cov_int_plus_noise[1])
        np.testing.assert_array_almost_equal(cov_int_plus_noise[2],
                                             expected_cov_int_plus_noise[2])

    def test_calc_stream_reduction_matrix(self):
        pass
        # Re_k = util.misc.randn_c(3,2)
        # Re_k = np.dot(Re_k, Re_k.transpose().conjugate())

        # P1 = comp._calc_stream_reduction_matrix(Re_k, 1)
        # P2 = comp._calc_stream_reduction_matrix(Re_k, 2)
        # P3 = comp._calc_stream_reduction_matrix(Re_k, 3)
        # print
        # print P1.round(4)
        # print
        # print P2.round(4)
        # print
        # print P3.round(4)
        # U, S, V_h = np.linalg.svd(Re_k)
        # print " xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        # print U.round(4)
        # print
        # print S.round(4)
        # print
        # print V_h.round(4)
        # print " xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx COMP Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TODO: finish implementation
class CompExtInt(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_calc_receive_filter(self):
        pass

    def test_calc_SNRs(self):
        pass

    def test_perform_comp(self):
        Nr = np.array([2, 2])
        Nt = np.array([2, 2])
        K = Nt.size
        Nti = 1
        iPu = 0.8  # Power for each user
        noise_var = 1e-50

        full_channel = util.misc.randn_c(np.sum(Nr), np.sum(Nt) + Nti)
        Re = comp.calc_cov_matrix_extint_plus_noise(
            full_channel[:, np.sum(Nt):],
            Nr)

        # multiuser_channel = channels.MultiUserChannelMatrixExtInt()
        # multiuser_channel.randomize(Nr, Nt, K, Nti)

        users_channel = full_channel[:, 0:np.sum(Nt)]

        # Create the comp object
        comp_obj = comp.CompExtInt(K, iPu, noise_var)


        comp.perform_comp_with_ext_int(users_channel, K, iPu, noise_var, Re)

        #newH, Ms_good = comp.perform_comp_with_ext_int(users_channel, K, iPu, noise_var, Re)

        # print
        # print newH.shape
        # print Ms_good.shape

        # self.fail('finish the implementation')

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    # plot_psd_OFDM_symbols()
    unittest.main()
