#!/usr/bin/env python
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


# shape is a tuple with the desired dimensiosn
def generate_pilots(pilot_power, shape):
    return math.sqrt(pilot_power) * np.exp(
        1j * 2 * np.pi * np.random.uniform(size=shape))


def generate_noise(noise_power, shape):
    return math.sqrt(noise_power) * randn_c(*shape)


def generate_channel_no_correlation(alpha, Nr, Nt, num_channels=None):
    if num_channels is None:
        return alpha * randn_c(Nr, Nt)

    return alpha * randn_c(num_channels, Nr, Nt)


def generate_channel_with_correlation(alpha, Nr, Nt, C, num_channels=None):
    """Generate channel with covariance matrix `C` on the receive antennas"""
    if num_channels is None:
        return (
            (np.random.multivariate_normal(np.zeros(Nr), C, size=Nt) +
             1j * np.random.multivariate_normal(np.zeros(Nr), C, size=Nt)) /
            math.sqrt(2.0)).T

    h = ((np.random.multivariate_normal(
        np.zeros(Nr), C, size=(Nt, num_channels)) +
          1j * np.random.multivariate_normal(
              np.zeros(Nr), C, size=(Nt, num_channels))) / math.sqrt(2.0))

    return np.transpose(h, [1, 2, 0])


class ChannelEstimationFunctionsTest(unittest.TestCase):
    def test_compute_ls_estimation(self):
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Case with a single transmit antenna and a single channel -> Y is 2D
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        num_pilots = 10
        Nr = 3
        Nt = 1
        pilot_power = 1.5
        noise_power = 0.5
        alpha = 0.7

        # Pilot symbols. Dimension: Nt x num_pilots
        s = generate_pilots(pilot_power=pilot_power, shape=(Nt, num_pilots))

        # Noise vector. Dimension: Nr x num_pilots
        N = generate_noise(noise_power, (Nr, num_pilots))

        # Channel with path loss
        h = generate_channel_no_correlation(alpha, Nr, Nt)

        Y = h @ s + N

        h_ls = compute_ls_estimation(Y, s)
        expected_h_ls = np.mean(Y / s, axis=1)[:, np.newaxis]

        np.testing.assert_array_almost_equal(h_ls, expected_h_ls)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Case with multiple transmit antennas and a single channel -> Y is 2D
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Nt = 2

        # Pilot symbols. Dimension: Nt x num_pilots
        s = generate_pilots(pilot_power=pilot_power, shape=(Nt, num_pilots))

        # Channel with path loss
        h = generate_channel_no_correlation(alpha, Nr, Nt)

        Y = h @ s + N
        h_ls = compute_ls_estimation(Y, s)
        expected_h_ls = (Y @ s.T.conj()) @ np.linalg.inv(s @ s.T.conj())

        np.testing.assert_array_almost_equal(h_ls, expected_h_ls)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Case with multiple transmit antennas and multiple channels -> Y is 3D
        # Same pilots in all different channel realizations -> s is 2D
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # In this case the number of received symbols must be a multiple of the
        # number of pilots. That means that different channel were used
        H = generate_channel_no_correlation(alpha, Nr, Nt, num_channels=2)
        N = generate_noise(noise_power, (2, Nr, num_pilots))
        Y = H @ s + N
        # Y_concat = np.concatenate([Y1[np.newaxis], Y2[np.newaxis]])
        expected_h_ls = np.concatenate([
            compute_ls_estimation(Y[0], s)[np.newaxis],
            compute_ls_estimation(Y[1], s)[np.newaxis]
        ])
        h_ls = compute_ls_estimation(Y, s)
        np.testing.assert_array_almost_equal(h_ls, expected_h_ls)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Case with multiple transmit antennas and multiple channels -> Y is 3D
        # Different pilots in all different channel realizations -> s is 3D
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # In this case the number of received symbols must be a multiple of the
        # number of pilots. That means that different channel were used
        H = generate_channel_no_correlation(alpha, Nr, Nt, num_channels=2)
        N = generate_noise(noise_power, (2, Nr, num_pilots))
        s = generate_pilots(pilot_power=pilot_power, shape=(2, Nt, num_pilots))
        Y = H @ s + N
        expected_h_ls = np.concatenate([
            compute_ls_estimation(Y[0], s[0])[np.newaxis],
            compute_ls_estimation(Y[1], s[1])[np.newaxis]
        ])
        h_ls = compute_ls_estimation(Y, s)
        np.testing.assert_array_almost_equal(h_ls, expected_h_ls)

    def test_compute_mmse_estimation(self):
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Case with a single channel -> Y is 2D
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        num_pilots = 10
        Nr = 3
        Nt = 1
        pilot_power = 1.5
        noise_power = 0.5
        alpha = 0.7

        # Pilot symbols. Dimension: Nt x num_pilots
        s = generate_pilots(pilot_power=pilot_power, shape=(Nt, num_pilots))

        # Noise vector. Dimention: Nr x num_pilots
        N = generate_noise(noise_power, (Nr, num_pilots))

        # Covariance matrix of the channel (correlation between receive
        # antennas). Dimension: Nr x Nr
        C = np.eye(Nr)

        # Channel matrix. Dimension: Nr x Nt
        h = generate_channel_with_correlation(alpha, Nr, Nt, C)
        assert (h.shape == (Nr, Nt))

        Y = h * s + N

        h_mmse = compute_mmse_estimation(Y, s, noise_power, C)

        # Compute the expected MMSE estimation
        Y_vec = np.reshape(Y, (Nr * num_pilots, 1), order="F")
        S = np.kron(s.T, np.eye(Nr))
        I_Nr = np.eye(Nr)
        expected_h_mmse = (np.linalg.inv(noise_power * I_Nr + C) @ C
                           @ S.T.conj()) @ Y_vec / np.linalg.norm(s)**2

        np.testing.assert_array_almost_equal(h_mmse, expected_h_mmse)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Case with a multiple channels -> Y is 3D
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # h1 = alpha * ((np.random.multivariate_normal(np.zeros(Nr), C) +
        #                1j * np.random.multivariate_normal(np.zeros(Nr), C)) /
        #               math.sqrt(2.0))[:, np.newaxis]
        # h2 = alpha * ((np.random.multivariate_normal(np.zeros(Nr), C) +
        #                1j * np.random.multivariate_normal(np.zeros(Nr), C)) /
        #               math.sqrt(2.0))[:, np.newaxis]
        H = generate_channel_with_correlation(alpha, Nr, Nt, C, num_channels=2)
        N = generate_noise(noise_power, (2, Nr, num_pilots))

        # Y1 = H[0] @ s + N[0]
        # Y2 = H[1] @ s + N[1]
        # Y_concat = np.concatenate([Y1[np.newaxis], Y2[np.newaxis]])
        Y = H @ s + N
        expected_h_mmse = np.concatenate([
            compute_mmse_estimation(Y[0], s, noise_power, C)[np.newaxis],
            compute_mmse_estimation(Y[1], s, noise_power, C)[np.newaxis]
        ])
        h_mmse = compute_mmse_estimation(Y, s, noise_power, C)
        np.testing.assert_array_almost_equal(h_mmse, expected_h_mmse)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Case with a multiple channels -> Y is 3D
        # Different pilots in all different channel realizations -> s is 3D
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # h1 = alpha * ((np.random.multivariate_normal(np.zeros(Nr), C) +
        #                1j * np.random.multivariate_normal(np.zeros(Nr), C)) /
        #               math.sqrt(2.0))[:, np.newaxis]
        # h2 = alpha * ((np.random.multivariate_normal(np.zeros(Nr), C) +
        #                1j * np.random.multivariate_normal(np.zeros(Nr), C)) /
        #               math.sqrt(2.0))[:, np.newaxis]
        H = generate_channel_with_correlation(alpha, Nr, Nt, C, num_channels=2)
        N = generate_noise(noise_power, (2, Nr, num_pilots))
        s = generate_pilots(pilot_power=pilot_power, shape=(2, Nt, num_pilots))

        # Y1 = H[0] @ s[0] + N[0]
        # Y2 = H[1] @ s[1] + N[1]

        # Y_concat = np.concatenate([Y1[np.newaxis], Y2[np.newaxis]])
        Y = H @ s + N
        expected_h_mmse = np.concatenate([
            compute_mmse_estimation(Y[0], s[0], noise_power, C)[np.newaxis],
            compute_mmse_estimation(Y[1], s[1], noise_power, C)[np.newaxis]
        ])

        h_mmse = compute_mmse_estimation(Y, s, noise_power, C)
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
