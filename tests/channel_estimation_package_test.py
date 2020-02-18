#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the modules in the channel_estimation package.
"""

# # xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
# import sys
# import os

# try:
#     parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
#     sys.path.append(parent_dir)
# except NameError:  # pragma: no cover
#     sys.path.append('../')
# # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import math
import unittest

import numpy as np

from pyphysim.channel_estimation.estimators import (
    compute_ls_estimation, compute_mmse_estimation, compute_theoretical_ls_MSE,
    compute_theoretical_mmse_MSE)
from pyphysim.util.misc import randn_c


class ChannelEstimationFunctionsTest(unittest.TestCase):
    def test_compute_ls_estimation(self):
        num_pilots = 10
        Nr = 2
        Nt = 1
        pilot_power = 1.0
        noise_power = 0.5
        alpha = 1.0

        # Pilot symbols. Dimension: Nt x num_pilots
        s = np.exp(1j * 2 * np.pi * np.random.uniform(size=(Nt, num_pilots)))
        s_with_power = math.sqrt(pilot_power) * s

        # Noise vector. Dimention: Nr x num_pilots
        N = math.sqrt(noise_power) * randn_c(Nr, num_pilots)

        # Covariance matrix of the channel (correlation between receive
        # antennas). Dimension: Nr x Nr
        C = np.eye(Nr)

        # Channel matrix. Dimension: Nr x Nt
        h = ((np.random.multivariate_normal(np.zeros(Nr), C) +
              1j * np.random.multivariate_normal(np.zeros(Nr), C)) /
             math.sqrt(2.0))
        h = h[:, np.newaxis]
        # Channel with path loss
        h_with_pl = alpha * h

        Y = h_with_pl * s_with_power + N

        h_ls = compute_ls_estimation(Y, s_with_power)
        expected_h_ls = np.mean(Y / s_with_power, axis=1)[:, np.newaxis]

        np.testing.assert_array_almost_equal(h_ls, expected_h_ls)

    def test_compute_mmse_estimation(self):
        num_pilots = 10
        Nr = 2
        Nt = 1
        pilot_power = 1.5
        noise_power = 0.3
        alpha = 0.7

        # Pilot symbols. Dimension: Nt x num_pilots
        s = np.exp(1j * 2 * np.pi * np.random.uniform(size=(Nt, num_pilots)))
        s_with_power = math.sqrt(pilot_power) * s

        # Noise vector. Dimention: Nr x num_pilots
        N = math.sqrt(noise_power) * randn_c(Nr, num_pilots)

        # Covariance matrix of the channel (correlation between receive
        # antennas). Dimension: Nr x Nr
        C = np.eye(Nr)

        # Channel matrix. Dimension: Nr x Nt
        h = ((np.random.multivariate_normal(np.zeros(Nr), C) +
              1j * np.random.multivariate_normal(np.zeros(Nr), C)) /
             math.sqrt(2.0))
        h = h[:, np.newaxis]
        # Channel with path loss
        h_with_pl = alpha * h

        Y = h_with_pl * s_with_power + N

        # Y_vec = np.reshape(Y, (Nr * num_pilots, 1), order="F")
        h_mmse = compute_mmse_estimation(Y, s_with_power, noise_power, C)

        # Compute the expected MMSE estimation
        Y_vec = np.reshape(Y, (Nr * num_pilots, 1), order="F")
        S = np.kron(s_with_power.T, np.eye(Nr))
        I_Nr = np.eye(Nr)
        expected_h_mmse = (np.linalg.inv(noise_power * I_Nr + num_pilots * C)
                           @ C @ S.T.conj()) @ Y_vec

        np.testing.assert_array_almost_equal(h_mmse, expected_h_mmse)

    def test_compute_theoretical_ls_MSE(self):
        Nr = 4
        noise_power = 0.3
        alpha = 0.9
        pilot_power = 1.5
        num_pilots = 20

        mse = compute_theoretical_ls_MSE(Nr, noise_power, alpha, pilot_power,
                                         num_pilots)
        expected_mse = Nr * noise_power / (
            (alpha**2) * pilot_power * num_pilots)
        self.assertAlmostEqual(mse, expected_mse)

    def test_compute_theoretical_mmse_MSE(self):
        Nr = 4
        noise_power = 0.3
        alpha = 0.9
        pilot_power = 1.5
        num_pilots = 20
        C = np.eye(Nr)

        mse = compute_theoretical_mmse_MSE(Nr, noise_power, alpha, pilot_power,
                                           num_pilots, C)
        expected_mse = np.trace(C @ np.linalg.inv(
            np.eye(Nr) +
            alpha**2 * pilot_power * num_pilots / noise_power * C))
        self.assertAlmostEqual(mse, expected_mse)


if __name__ == '__main__':
    unittest.main()
