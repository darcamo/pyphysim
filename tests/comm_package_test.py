#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,E0611

"""Tests for the modules in the comm package.

Each module has several doctests that we run in addition to the unittests
defined here.
"""

__revision__ = "$Revision$"

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
try:
    parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    sys.path.append(parent_dir)
except NameError:
    sys.path.append('../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import unittest
import doctest
import numpy as np
from scipy.linalg import block_diag
from matplotlib import pylab
from matplotlib import pyplot as plt

from pyphysim.comm.ofdm import OFDM
from pyphysim.comm import modulators, blockdiagonalization, ofdm, mimo, pathloss, \
    waterfilling, channels
from pyphysim.ia.algorithms import ClosedFormIASolver
from pyphysim.util.misc import randn_c, least_right_singular_vectors, \
    calc_shannon_sum_capacity, calc_whitening_matrix
from pyphysim.util.conversion import dB2Linear, linear2dB, \
    single_matrix_to_matrix_of_matrices
from pyphysim.subspace.projections import calcProjectionMatrix
from pyphysim.comm.mimo import Blast, Alamouti

# UPDATE THIS CLASS if another module is added to the comm package
class CommDoctestsTestCase(unittest.TestCase):
    """Teste case that run all the doctests in the modules of the comm
    package."""

    def test_modulators(self):
        """Run doctests in the modulators module."""
        doctest.testmod(modulators)

    def test_blockdiagonalization(self, ):
        """Run doctests in the blockdiagonalization module."""
        doctest.testmod(blockdiagonalization)

    def test_ofdm(self, ):
        """Run doctests in the ofdm module."""
        doctest.testmod(ofdm)

    def test_mimo(self, ):
        """Run doctests in the mimo module."""
        doctest.testmod(mimo)

    def test_pathloss(self, ):
        """Run doctests in the pathloss module."""
        doctest.testmod(pathloss)

    def test_waterfilling(self, ):
        """Run doctests in the waterfilling module."""
        doctest.testmod(waterfilling)

    def test_channels(self, ):
        """Run doctests in the channels module."""
        doctest.testmod(channels)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx CHANNELS module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ModuleFunctionsTestCase(unittest.TestCase):
    def test_generate_jakes_samples(self):
        Fd = 5     # Doppler frequency (in Hz)
        Ts = 1e-3  # Sampling interval (in seconds)
        N = 1000   # Number of samples
        NRays = 8  # Number of rays for the Jakes model

        # Test generating channel samples for a SISO scenario
        h = channels.generate_jakes_samples(Fd, Ts, N, NRays)
        self.assertEqual(h.size, 1000)
        self.assertEqual(h.shape, (1000,))

        h2 = channels.generate_jakes_samples(Fd, Ts, N, NRays, shape=(4, 3))
        self.assertEqual(h2.shape, (4, 3, N))

        # Test with a given RandomState object.
        RS = np.random.RandomState()
        h3 = channels.generate_jakes_samples(Fd, Ts, N, NRays, shape=(3, 2),
                                             RS=RS)
        self.assertEqual(h3.shape, (3, 2, N))

    def test_calc_stream_reduction_matrix(self):
        Re_k = randn_c(3, 2)
        Re_k = np.dot(Re_k, Re_k.transpose().conjugate())

        P1 = blockdiagonalization._calc_stream_reduction_matrix(Re_k, 1)
        P2 = blockdiagonalization._calc_stream_reduction_matrix(Re_k, 2)
        P3 = blockdiagonalization._calc_stream_reduction_matrix(Re_k, 3)

        (min_Vs, _, _) = least_right_singular_vectors(Re_k, 3)
        expected_P1 = min_Vs[:, :1]
        expected_P2 = min_Vs[:, :2]
        expected_P3 = min_Vs[:, :3]

        np.testing.assert_array_almost_equal(P1, expected_P1)
        np.testing.assert_array_almost_equal(P2, expected_P2)
        np.testing.assert_array_almost_equal(P3, expected_P3)


class JakesSampleGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        Fd = 5     # Doppler frequency (in Hz)
        Ts = 1e-3  # Sampling interval (in seconds)
        NRays = 8  # Number of rays for the Jakes model

        self.obj = channels.JakesSampleGenerator(Fd, Ts, NRays)
        self.obj2 = channels.JakesSampleGenerator(Fd, Ts, NRays, shape=(3, 2))

    def test_generate_channel_samples(self):
        self.assertAlmostEqual(self.obj._current_time, 0.0)

        # xxxxxxxxxx First object -> shape is None xxxxxxxxxxxxxxxxxxxxxxxx
        # Generate 100 samples
        h1_part1 = self.obj.generate_channel_samples(100)
        self.assertEqual(h1_part1.shape, (100,))
        # For a sample interval of 1e-3 the last time sample generated was
        # 0.099. Therefore, the next time sample should be 0.099+1e-3 = 0.1
        self.assertAlmostEqual(self.obj._current_time, 0.1)

        # Generate 50 more samples
        h1_part2 = self.obj.generate_channel_samples(50)
        self.assertEqual(h1_part2.shape, (50,))
        self.assertAlmostEqual(self.obj._current_time, 0.15)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Second object -> shape is (3, 2) xxxxxxxxxxxxxxxxxxxxx
        # Generate 100 samples
        h2_part1 = self.obj2.generate_channel_samples(120)
        self.assertEqual(h2_part1.shape, (3, 2, 120,))
        # For a sample interval of 1e-3 the last time sample generated was
        # 0.099. Therefore, the next time sample should be 0.099+1e-3 = 0.1
        self.assertAlmostEqual(self.obj2._current_time, 0.12)

        # Generate 50 more samples
        h2_part2 = self.obj2.generate_channel_samples(60)
        self.assertEqual(h2_part2.shape, (3, 2, 60))
        self.assertAlmostEqual(self.obj2._current_time, 0.18)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class MultiUserChannelMatrixTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.multiH = channels.MultiUserChannelMatrix()
        self.H = np.array(
            [
                [0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
                [0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
                [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
                [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
                [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
                [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
                [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
            ]
        )
        self.K = 3
        self.Nr = np.array([2, 4, 6])
        self.Nt = np.array([2, 3, 5])

    def test_from_small_matrix_to_big_matrix(self):
        K = 3
        Nr = np.array([2, 4, 6])
        Nt = np.array([2, 3, 5])
        small_matrix = np.arange(1, 10)
        small_matrix.shape = (3, 3)
        big_matrix \
            = channels.MultiUserChannelMatrix._from_small_matrix_to_big_matrix(
                small_matrix, Nr, Nt, K)

        expected_big_matrix = np.array(
            [[1., 1., 2., 2., 2., 3., 3., 3., 3., 3.],
             [1., 1., 2., 2., 2., 3., 3., 3., 3., 3.],
             [4., 4., 5., 5., 5., 6., 6., 6., 6., 6.],
             [4., 4., 5., 5., 5., 6., 6., 6., 6., 6.],
             [4., 4., 5., 5., 5., 6., 6., 6., 6., 6.],
             [4., 4., 5., 5., 5., 6., 6., 6., 6., 6.],
             [7., 7., 8., 8., 8., 9., 9., 9., 9., 9.],
             [7., 7., 8., 8., 8., 9., 9., 9., 9., 9.],
             [7., 7., 8., 8., 8., 9., 9., 9., 9., 9.],
             [7., 7., 8., 8., 8., 9., 9., 9., 9., 9.],
             [7., 7., 8., 8., 8., 9., 9., 9., 9., 9.],
             [7., 7., 8., 8., 8., 9., 9., 9., 9., 9.]])

        np.testing.assert_array_equal(big_matrix, expected_big_matrix)

    def test_randomize(self):
        K = 3
        Nr = np.array([2, 4, 6])
        Nt = np.array([2, 3, 5])
        self.multiH.randomize(Nr, Nt, K)
        self.assertEqual(self.multiH.K, K)
        np.testing.assert_array_equal(self.multiH.Nr, Nr)
        np.testing.assert_array_equal(self.multiH.Nt, Nt)

        # Test the shape of the matrix of channels
        self.assertEqual(self.multiH.H.shape, (K, K))

        # Test the shape of each individual channel
        for rx in np.arange(K):
            for tx in np.arange(K):
                self.assertEqual(
                    self.multiH.H[rx, tx].shape,
                    (Nr[rx], Nt[tx]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now lets test when the number of transmit and receive antennas
        # are the same for all users
        Nr = 2
        Nt = 3
        self.multiH.randomize(Nr, Nt, K)
        # Test the shape of the matrix of channels
        self.assertEqual(self.multiH.H.shape, (K, K))
        # Test the shape of each individual channel
        for rx in np.arange(K):
            for tx in np.arange(K):
                self.assertEqual(self.multiH.H[rx, tx].shape, (Nr, Nt))

    def test_init_from_channel_matrix(self):
        H = self.H
        K = self.K
        Nr = self.Nr
        Nt = self.Nt

        # Test if the exception is raised when the number of transmit
        # antennas does not match the shape of the channel_matrix
        with self.assertRaises(ValueError):
            self.multiH.init_from_channel_matrix(H, Nr, np.array([2, 3, 3]), K)

        # Test if an exception is raised when the sizes of Nr and Nt do not
        # match
        with self.assertRaises(ValueError):
            self.multiH.init_from_channel_matrix(H, Nr, Nt, 2)

        # Test if an exception is raised of K does not match Nr and Nt
        with self.assertRaises(ValueError):
            self.multiH.init_from_channel_matrix(H, Nr, Nt, 2)

        # Test if everything is correctly assigned
        self.multiH.init_from_channel_matrix(H, Nr, Nt, K)
        self.assertEqual(self.multiH.K, K)
        np.testing.assert_array_equal(self.multiH.Nr, Nr)
        np.testing.assert_array_equal(self.multiH.Nt, Nt)

        self.assertEqual(self.multiH.H.shape, (K, K))

        # We don't really need to test multiH.H because the code was alread
        # tested in test_from_big_matrix

    def test_get_channel(self):
        H = self.H
        K = self.K
        Nr = self.Nr
        Nt = self.Nt
        self.multiH.init_from_channel_matrix(H, Nr, Nt, K)

        # xxxxxxxxxx Test get_channel without Pathloss xxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_equal(
            self.multiH.get_Hkl(0, 0),
            np.ones([2, 2]) * 0)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(0, 1),
            np.ones([2, 3]) * 1)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(0, 2),
            np.ones([2, 5]) * 2)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(1, 0),
            np.ones([4, 2]) * 3)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(1, 1),
            np.ones([4, 3]) * 4)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(1, 2),
            np.ones([4, 5]) * 5)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(2, 0),
            np.ones([6, 2]) * 6)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(2, 1),
            np.ones([6, 3]) * 7)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(2, 2),
            np.ones([6, 5]) * 8)

        # xxxxxxxxxx Test get_channel with Pathloss xxxxxxxxxxxxxxxxxxxxxxx
        # pathloss (in linear scale) must be a positive number
        pathloss = np.abs(np.random.randn(self.K, self.K))
        self.multiH.set_pathloss(pathloss)
        np.testing.assert_array_equal(
            self.multiH.get_Hkl(0, 0),
            np.sqrt(self.multiH.pathloss[0, 0]) * np.ones([2, 2]) * 0)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(0, 1),
            np.sqrt(self.multiH.pathloss[0, 1]) * np.ones([2, 3]) * 1)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(0, 2),
            np.sqrt(self.multiH.pathloss[0, 2]) * np.ones([2, 5]) * 2)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(1, 0),
            np.sqrt(self.multiH.pathloss[1, 0]) * np.ones([4, 2]) * 3)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(1, 1),
            np.sqrt(self.multiH.pathloss[1, 1]) * np.ones([4, 3]) * 4)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(1, 2),
            np.sqrt(self.multiH.pathloss[1, 2]) * np.ones([4, 5]) * 5)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(2, 0),
            np.sqrt(self.multiH.pathloss[2, 0]) * np.ones([6, 2]) * 6)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(2, 1),
            np.sqrt(self.multiH.pathloss[2, 1]) * np.ones([6, 3]) * 7)

        np.testing.assert_array_equal(
            self.multiH.get_Hkl(2, 2),
            np.sqrt(self.multiH.pathloss[2, 2]) * np.ones([6, 5]) * 8)

    def test_get_channel_all_transmitters_to_single_receiver(self):
        H = self.H
        K = self.K
        Nr = self.Nr
        Nt = self.Nt
        self.multiH.init_from_channel_matrix(H, Nr, Nt, K)

        expected_H1 = self.multiH.big_H[0:2, :]
        expected_H2 = self.multiH.big_H[2:6, :]
        expected_H3 = self.multiH.big_H[6:, :]

        # xxxxx Test without pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_equal(
            self.multiH.get_Hk(0),
            expected_H1)
        np.testing.assert_array_equal(
            self.multiH.get_Hk(1),
            expected_H2)
        np.testing.assert_array_equal(
            self.multiH.get_Hk(2),
            expected_H3)

        # xxxxx Test with pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        pathloss = np.abs(np.random.randn(self.K, self.K))
        self.multiH.set_pathloss(pathloss)
        expected_H1 = self.multiH.big_H[0:2, :]
        expected_H2 = self.multiH.big_H[2:6, :]
        expected_H3 = self.multiH.big_H[6:, :]
        np.testing.assert_array_equal(
            self.multiH.get_Hk(0),
            expected_H1
        )
        np.testing.assert_array_equal(
            self.multiH.get_Hk(1),
            expected_H2
        )
        np.testing.assert_array_equal(
            self.multiH.get_Hk(2),
            expected_H3
        )

    def test_H_and_big_H_properties(self):
        H = self.H
        K = self.K
        Nr = self.Nr
        Nt = self.Nt
        self.multiH.init_from_channel_matrix(H, Nr, Nt, K)
        # pathloss (in linear scale) must be a positive number
        pathloss = np.abs(np.random.randn(self.K, self.K))
        self.multiH.set_pathloss(pathloss)

        cumNr = np.hstack([0, np.cumsum(Nr)])
        cumNt = np.hstack([0, np.cumsum(Nt)])

        for row in range(K):
            for col in range(K):
                # Test the 'H' property
                np.testing.assert_array_equal(
                    self.multiH.get_Hkl(row, col), self.multiH.H[row, col])
                # Test the 'big_H' property
                np.testing.assert_array_equal(
                    self.multiH.get_Hkl(row, col),
                    self.multiH.big_H[
                        cumNr[row]:cumNr[row + 1], cumNt[col]:cumNt[col + 1]])

    def test_corrupt_data(self):
        NSymbs = 20
        # Create some input data for the 3 users
        input_data = np.zeros(self.K, dtype=np.ndarray)
        input_data[0] = randn_c(self.Nt[0], NSymbs)
        input_data[1] = randn_c(self.Nt[1], NSymbs)
        input_data[2] = randn_c(self.Nt[2], NSymbs)

        # Disable the path loss
        self.multiH.set_pathloss()
        self.multiH.randomize(self.Nr, self.Nt, self.K)

        # xxxxxxxxxx Test without pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Note that the corrupt_concatenated_data is implicitelly called by
        # corrupt_data and thus we will only test corrupt_data.
        output = self.multiH.corrupt_data(input_data)

        # Calculates the expected output (without pathloss)
        expected_output = np.zeros(self.K, dtype=np.ndarray)
        for rx in np.arange(self.K):
            for tx in np.arange(self.K):
                expected_output[rx] += np.dot(
                    self.multiH.get_Hkl(rx, tx), input_data[tx])

        # Test the received data for the 3 users
        np.testing.assert_array_almost_equal(output[0], expected_output[0])
        np.testing.assert_array_almost_equal(output[1], expected_output[1])
        np.testing.assert_array_almost_equal(output[2], expected_output[2])

        # xxxxxxxxxx Test with pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # pathloss (in linear scale) must be a positive number
        pathloss = np.abs(np.random.randn(self.K, self.K))
        self.multiH.set_pathloss(pathloss)

        # Note that the corrupt_concatenated_data is implicitelly called by
        # corrupt_data and thus we will only test corrupt_data. Also, they
        # are affected by the pathloss.
        output2 = self.multiH.corrupt_data(input_data)

        # Calculates the expected output (with pathloss)
        expected_output2 = np.zeros(self.K, dtype=np.ndarray)
        for rx in np.arange(self.K):
            for tx in np.arange(self.K):
                expected_output2[rx] += np.dot(
                    # Note that get_channel is affected by the pathloss
                    self.multiH.get_Hkl(rx, tx), input_data[tx])

        # Test the received data for the 3 users, but now with pathloss
        np.testing.assert_array_almost_equal(output2[0], expected_output2[0])
        np.testing.assert_array_almost_equal(output2[1], expected_output2[1])
        np.testing.assert_array_almost_equal(output2[2], expected_output2[2])

        # Now we also pass the noise_variance to corrupt_data to actually
        # call the code that does the noise addition, but with a variance
        # so low that the expected output should be the same (we do this in
        # order to be able to test it).
        output3 = self.multiH.corrupt_data(input_data, 1e-20)
        np.testing.assert_array_almost_equal(output3[0], expected_output2[0])
        np.testing.assert_array_almost_equal(output3[1], expected_output2[1])
        np.testing.assert_array_almost_equal(output3[2], expected_output2[2])

    def test_set_and_get_post_filter(self):
        self.multiH.randomize(self.Nr, self.Nt, self.K)
        self.assertIsNone(self.multiH._W)
        self.assertIsNone(self.multiH._big_W)

        self.assertIsNone(self.multiH.W)
        self.assertIsNone(self.multiH.big_W)

        W = [randn_c(2, 2),
             randn_c(2, 2),
             randn_c(2, 2)]

        self.multiH.set_post_filter(W)
        np.testing.assert_array_almost_equal(W, self.multiH._W)
        np.testing.assert_array_almost_equal(W, self.multiH.W)

        # _big_W is still None
        self.assertIsNone(self.multiH._big_W)

        expected_big_W = block_diag(*W)
        np.testing.assert_array_almost_equal(expected_big_W,
                                             self.multiH.big_W)
        self.assertIsNotNone(self.multiH._big_W)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        W2 = [randn_c(2, 2),
              randn_c(2, 2),
              randn_c(2, 2)]
        self.multiH.set_post_filter(W2)
        np.testing.assert_array_almost_equal(W2, self.multiH._W)
        np.testing.assert_array_almost_equal(W2, self.multiH.W)

        self.assertIsNone(self.multiH._big_W)
        expected_big_W2 = block_diag(*W2)
        np.testing.assert_array_almost_equal(expected_big_W2,
                                             self.multiH.big_W)
        self.assertIsNotNone(self.multiH._big_W)

    def test_corrupt_data_with_post_filter(self):
        NSymbs = 20
        # Create some input data for the 3 users
        input_data = np.zeros(self.K, dtype=np.ndarray)
        input_data[0] = randn_c(self.Nt[0], NSymbs)
        input_data[1] = randn_c(self.Nt[1], NSymbs)
        input_data[2] = randn_c(self.Nt[2], NSymbs)

        # Disable the path loss
        self.multiH.set_pathloss()
        self.multiH.randomize(self.Nr, self.Nt, self.K)

        # Set the post processing filter
        W = [randn_c(self.Nr[0], self.Nr[0]),
             randn_c(self.Nr[1], self.Nr[1]),
             randn_c(self.Nr[2], self.Nr[2])]
        self.multiH.set_post_filter(W)

        output = self.multiH.corrupt_data(input_data)

        # Calculates the expected output (without pathloss)
        expected_output = np.zeros(self.K, dtype=np.ndarray)
        for rx in np.arange(self.K):
            for tx in np.arange(self.K):
                expected_output[rx] += np.dot(
                    self.multiH.get_Hkl(rx, tx), input_data[tx])
            expected_output[rx] = np.dot(W[rx].conjugate().T,
                                         expected_output[rx])

        # Test the received data for the 3 users
        np.testing.assert_array_almost_equal(output[0], expected_output[0])
        np.testing.assert_array_almost_equal(output[1], expected_output[1])
        np.testing.assert_array_almost_equal(output[2], expected_output[2])

    def test_last_noise_property(self):
        noise_var = 1e-2
        H = np.eye(6)
        self.multiH.init_from_channel_matrix(H,
                                             np.array([2, 2, 2]),
                                             np.array([2, 2, 2]),
                                             3)

        data = randn_c(6, 10)

        corrupted_data = self.multiH.corrupt_concatenated_data(data, noise_var)
        last_noise = self.multiH.last_noise

        expected_corrupted_data = data + last_noise

        np.testing.assert_array_almost_equal(expected_corrupted_data,
                                             corrupted_data)

        last_noise_var = self.multiH.last_noise_var
        self.assertAlmostEqual(noise_var, last_noise_var)

        # Call corrupt_concatenated_data again, but without noise var. This
        # should set last_noise to None and last_noise_var to zero.
        corrupted_data = self.multiH.corrupt_concatenated_data(data)
        np.testing.assert_array_almost_equal(corrupted_data, data)
        self.assertIsNone(self.multiH.last_noise)
        self.assertAlmostEqual(self.multiH.last_noise_var, 0.0)

    def test_calc_Q(self):
        K = 3
        Nt = np.array([2, 2, 2])
        Nr = np.array([2, 2, 2])
        Ns = np.array([1, 1, 1])
        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        self.multiH.randomize(Nr, Nt, K)

        F_all_k = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F_all_k[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F_all_k[k] = (F_all_k[k]
                          / np.linalg.norm(F_all_k[k], 'fro')
                          * np.sqrt(P[k]))

        # xxxxx Calculate the expected Q[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H01_F1 = np.dot(
            self.multiH.get_Hkl(k, 1),
            F_all_k[1])
        H02_F2 = np.dot(
            self.multiH.get_Hkl(k, 2),
            F_all_k[2])
        expected_Q0 = np.dot(H01_F1,
                             H01_F1.transpose().conjugate()) + \
            np.dot(H02_F2,
                   H02_F2.transpose().conjugate())

        Qk = self.multiH.calc_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q0)

        # Now with noise variance different of 0
        noise_var = round(0.1 * np.random.rand(), 4)
        Qk = self.multiH.calc_Q(k, F_all_k, noise_var=noise_var)
        np.testing.assert_array_almost_equal(
            Qk,
            expected_Q0 + noise_var * np.eye(2))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0 = np.dot(
            self.multiH.get_Hkl(k, 0),
            F_all_k[0])
        H12_F2 = np.dot(
            self.multiH.get_Hkl(k, 2),
            F_all_k[2])
        expected_Q1 = np.dot(H10_F0,
                             H10_F0.transpose().conjugate()) + \
            np.dot(H12_F2,
                   H12_F2.transpose().conjugate())

        Qk = self.multiH.calc_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q1)

        # Now with noise variance different of 0
        noise_var = round(0.1 * np.random.rand(), 4)
        Qk = self.multiH.calc_Q(k, F_all_k, noise_var=noise_var)
        np.testing.assert_array_almost_equal(
            Qk,
            expected_Q1 + noise_var * np.eye(2))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H20_F0 = np.dot(
            self.multiH.get_Hkl(k, 0),
            F_all_k[0]
        )
        H21_F1 = np.dot(
            self.multiH.get_Hkl(k, 1),
            F_all_k[1]
        )
        expected_Q2 = np.dot(H20_F0,
                             H20_F0.transpose().conjugate()) + \
            np.dot(H21_F1,
                   H21_F1.transpose().conjugate())

        Qk = self.multiH.calc_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q2)

        # Now with noise variance different of 0
        noise_var = round(0.1 * np.random.rand(), 4)
        Qk = self.multiH.calc_Q(k, F_all_k, noise_var=noise_var)
        np.testing.assert_array_almost_equal(
            Qk,
            expected_Q2 + noise_var * np.eye(2))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_JP_Q(self):
        K = 3
        Nt = np.array([2, 2, 2])
        Nr = np.array([2, 2, 2])
        Ns = np.array([1, 1, 1])
        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        self.multiH.randomize(Nr, Nt, K)

        F_all_k = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F_all_k[k] = randn_c(np.sum(Nt), Ns[k]) * np.sqrt(P[k])
            F_all_k[k] = (F_all_k[k]
                          / np.linalg.norm(F_all_k[k], 'fro')
                          * np.sqrt(P[k]))

        # xxxxx Calculate the expected Q[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H0_F1 = np.dot(
            self.multiH.get_Hk(k),
            F_all_k[1]
        )
        H0_F2 = np.dot(
            self.multiH.get_Hk(k),
            F_all_k[2]
        )
        expected_Q0 = np.dot(H0_F1,
                             H0_F1.transpose().conjugate()) + \
            np.dot(H0_F2,
                   H0_F2.transpose().conjugate())

        Qk = self.multiH.calc_JP_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q0)

        # Now with noise variance different of 0
        noise_var = round(0.1 * np.random.rand(), 4)
        Qk = self.multiH.calc_JP_Q(k, F_all_k, noise_var=noise_var)
        np.testing.assert_array_almost_equal(
            Qk,
            expected_Q0 + noise_var * np.eye(2))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H1_F0 = np.dot(
            self.multiH.get_Hk(k),
            F_all_k[0]
        )
        H1_F2 = np.dot(
            self.multiH.get_Hk(k),
            F_all_k[2]
        )
        expected_Q1 = np.dot(H1_F0,
                             H1_F0.transpose().conjugate()) + \
            np.dot(H1_F2,
                   H1_F2.transpose().conjugate())

        Qk = self.multiH.calc_JP_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q1)

        # Now with noise variance different of 0
        noise_var = round(0.1 * np.random.rand(), 4)
        Qk = self.multiH.calc_JP_Q(k, F_all_k, noise_var=noise_var)
        np.testing.assert_array_almost_equal(
            Qk,
            expected_Q1 + noise_var * np.eye(2))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H2_F0 = np.dot(
            self.multiH.get_Hk(k),
            F_all_k[0]
        )
        H2_F1 = np.dot(
            self.multiH.get_Hk(k),
            F_all_k[1]
        )
        expected_Q2 = np.dot(H2_F0,
                             H2_F0.transpose().conjugate()) + \
            np.dot(H2_F1,
                   H2_F1.transpose().conjugate())

        Qk = self.multiH.calc_JP_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q2)

        # Now with noise variance different of 0
        noise_var = round(0.1 * np.random.rand(), 4)
        Qk = self.multiH.calc_JP_Q(k, F_all_k, noise_var=noise_var)
        np.testing.assert_array_almost_equal(
            Qk,
            expected_Q2 + noise_var * np.eye(2))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_Bkl_cov_matrix_first_part(self):
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = np.ones(K, dtype=int) * 1
        P = np.array([1.2, 1.5, 0.9])

        noise_power = 0.1

        self.multiH.randomize(Nr, Nt, K)
        F = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])

        # For ones stream the expected Bkl is equivalent to the Q matrix
        # plus the direct channel part.
        for k in range(K):
            Hkk = self.multiH.get_Hkl(k, k)
            Fk = F[k]
            HkkFk = np.dot(Hkk, Fk)
            expected_first_part = (self.multiH.calc_Q(k, F)
                                   + np.dot(HkkFk, HkkFk.conjugate().T))
            expected_first_part_with_noise = (
                self.multiH.calc_Q(k, F, noise_power)
                + np.dot(HkkFk, HkkFk.conjugate().T))

            # Test without noise
            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.multiH._calc_Bkl_cov_matrix_first_part(F, k))

            # Test with noise
            np.testing.assert_array_almost_equal(
                expected_first_part_with_noise,
                self.multiH._calc_Bkl_cov_matrix_first_part(F, k, noise_power))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test for more streams
        Nr = np.ones(K, dtype=int) * 4
        Nt = np.ones(K, dtype=int) * 4
        Ns = np.ones(3, dtype=int) * 2

        self.multiH.randomize(Nr, Nt, K)
        F = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])

        for k in range(K):
            expected_first_part = 0.0  # First part in the equation of Bkl
                                       # (the double summation)

            # The inner for loop will calculate
            # $\text{aux} = \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$
            for j in range(K):
                aux = 0.0
                Hkj = self.multiH.get_Hkl(k, j)
                Hkj_H = Hkj.conjugate().transpose()

                # Calculates individually for each stream
                for d in range(Ns[k]):
                    Vjd = F[j][:, d:d + 1]
                    Vjd_H = Vjd.conjugate().transpose()
                    aux = aux + np.dot(np.dot(Hkj, np.dot(Vjd, Vjd_H)), Hkj_H)

                expected_first_part = expected_first_part + aux

            expected_first_part_with_noise = (expected_first_part
                                              + np.eye(Nr[k]) * noise_power)

            # Test without noise
            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.multiH._calc_Bkl_cov_matrix_first_part(F, k))

            # Test with noise
            np.testing.assert_array_almost_equal(
                expected_first_part_with_noise,
                self.multiH._calc_Bkl_cov_matrix_first_part(F, k, noise_power))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_Bkl_cov_matrix_second_part(self):
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = np.ones(K, dtype=int) * 1
        P = np.array([1.2, 1.5, 0.9])

        self.multiH.randomize(Nr, Nt, K)
        F = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])

        for k in range(K):
            Hkk = self.multiH.get_Hkl(k, k)
            Hkk_H = Hkk.transpose().conjugate()
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(
                    Hkk,
                    np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.multiH._calc_Bkl_cov_matrix_second_part(F[k], k, l))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test for more streams
        K = 3
        Nr = np.ones(K, dtype=int) * 4
        Nt = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 2
        P = np.array([1.2, 1.5, 0.9])

        self.multiH.randomize(Nr, Nt, K)
        F = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])

        for k in range(K):
            Hkk = self.multiH.get_Hkl(k, k)
            Hkk_H = Hkk.transpose().conjugate()
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(
                    Hkk,
                    np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.multiH._calc_Bkl_cov_matrix_second_part(F[k], k, l))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test for all streams
        K = 3
        Nr = np.ones(K, dtype=int) * 4
        Nt = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 4
        P = np.array([1.2, 1.5, 0.9])

        self.multiH.randomize(Nr, Nt, K)
        F = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])

        for k in range(K):
            Hkk = self.multiH.get_Hkl(k, k)
            Hkk_H = Hkk.transpose().conjugate()
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(
                    Hkk,
                    np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.multiH._calc_Bkl_cov_matrix_second_part(F[k], k, l))

    def test_calc_Bkl(self):
        # For the case of a single stream oer user Bkl (which only has l=0)
        # is equal to Qk plus I (identity matrix)
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = np.ones(K, dtype=int) * 1
        P = np.array([1.2, 1.5, 0.9])
        noise_power = 0.568

        self.multiH.randomize(Nr, Nt, K)
        F = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])

        for k in range(K):
            # We only have the stream 0
            expected_Bk0 = (self.multiH.calc_Q(k, F)
                            + (noise_power * np.eye(Nr[k])))
            Bk0 = self.multiH._calc_Bkl_cov_matrix_all_l(
                F, k, N0_or_Rek=noise_power)[0]
            np.testing.assert_array_almost_equal(expected_Bk0, Bk0)

    def test_underline_calc_SINR_k(self):
        multiUserChannel = channels.MultiUserChannelMatrix()
        #iasolver = MaxSinrIASolver(multiUserChannel)
        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 2

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(Nr, Nt, K)
        F = np.empty(K, dtype=np.ndarray)
        U = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])
            U[k] = randn_c(Nr[k], Ns[k])

        for k in range(K):
            Hkk = multiUserChannel.get_Hkl(k, k)
            Bkl_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
                F, k, N0_or_Rek=0.0)
            Uk = U[k]
            Fk = F[k]
            # Uk_H = iasolver.full_W_H[k]

            SINR_k_all_l = multiUserChannel._calc_SINR_k(k, Fk, Uk, Bkl_all_l)

            for l in range(Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = F[k][:, l:l + 1]
                aux = np.dot(Ukl_H,
                             np.dot(Hkk, Vkl))

                expectedSINRkl = np.asscalar(
                    np.dot(aux, aux.transpose().conjugate()) / np.dot(
                        Ukl_H, np.dot(Bkl_all_l[l], Ukl)))

                np.testing.assert_array_almost_equal(expectedSINRkl,
                                                     SINR_k_all_l[l])

        # xxxxxxxxxx Repeat the tests, but now using an IA solution xxxxxxx
        iasolver = ClosedFormIASolver(multiUserChannel)
        iasolver.solve(Ns=2)
        F = iasolver.full_F
        U = iasolver.full_W

        for k in range(K):
            Hkk = multiUserChannel.get_Hkl(k, k)
            Uk = U[k]
            Fk = F[k]

            Bkl_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
                F, k, N0_or_Rek=0.0001)
            SINR_k_all_l = multiUserChannel._calc_SINR_k(
                k, F[k], U[k], Bkl_all_l)

            for l in range(Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = F[k][:, l:l + 1]
                aux = np.dot(Ukl_H,
                             np.dot(Hkk, Vkl))

                expectedSINRkl = abs(np.asscalar(
                    np.dot(aux, aux.transpose().conjugate()) / np.dot(
                        Ukl_H, np.dot(Bkl_all_l[l], Ukl))))

                np.testing.assert_array_almost_equal(expectedSINRkl,
                                                     SINR_k_all_l[l])

    def test_calc_SINR(self):
        multiUserChannel = channels.MultiUserChannelMatrix()
        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 2

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(Nr, Nt, K)
        iasolver = ClosedFormIASolver(multiUserChannel)
        iasolver.solve(Ns, P)
        F = iasolver.full_F
        U = iasolver.full_W

        SINR_all_users = multiUserChannel.calc_SINR(F, U, noise_power=0.0)

        # xxxxxxxxxx Noise Variance of 0.0 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # k = 0
        B0l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=0, N0_or_Rek=0.0)
        expected_SINR0 = multiUserChannel._calc_SINR_k(
            0, F[0], U[0], B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=1, N0_or_Rek=0.0)
        expected_SINR1 = multiUserChannel._calc_SINR_k(
            1, F[1], U[1], B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 1
        B2l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=2, N0_or_Rek=0.0)
        expected_SINR2 = multiUserChannel._calc_SINR_k(
            2, F[2], U[2], B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Noise Variance of 0.1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # k = 0
        iasolver.noise_var = 0.1
        SINR_all_users = multiUserChannel.calc_SINR(F, U, noise_power=0.1)
        B0l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=0, N0_or_Rek=0.1)
        expected_SINR0 = multiUserChannel._calc_SINR_k(
            0, F[0], U[0], B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=1, N0_or_Rek=0.1)
        expected_SINR1 = multiUserChannel._calc_SINR_k(
            1, F[1], U[1], B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 2
        B2l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=2, N0_or_Rek=0.1)
        expected_SINR2 = multiUserChannel._calc_SINR_k(
            2, F[2], U[2], B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_JP_Bkl_cov_matrix_first_part(self):
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = Nt
        iPu = 1.2
        noise_power = 0.1

        self.multiH.randomize(Nr, Nt, K)

        (_, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H, K, iPu, noise_power)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)

        # For ones stream the expected Bkl is equivalent to the Q matrix
        # plus the direct channel part.
        for k in range(K):
            Hk = self.multiH.get_Hk(k)
            Fk = F[k]
            HkFk = np.dot(Hk, Fk)
            expected_first_part = (self.multiH.calc_JP_Q(k, F)
                                   + np.dot(HkFk, HkFk.conjugate().T))
            expected_first_part_with_noise = (
                self.multiH.calc_JP_Q(k, F, noise_power)
                + np.dot(HkFk, HkFk.conjugate().T))

            # Test without noise
            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.multiH._calc_JP_Bkl_cov_matrix_first_part(F, k))

            # Test with noise
            np.testing.assert_array_almost_equal(
                expected_first_part_with_noise,
                self.multiH._calc_JP_Bkl_cov_matrix_first_part(
                    F, k, noise_power))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test for more streams
        Nr = np.ones(K, dtype=int) * 4
        Nt = np.ones(K, dtype=int) * 4
        Ns = Nt

        self.multiH.randomize(Nr, Nt, K)

        (_, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H, K, iPu, noise_power)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)

        for k in range(K):
            expected_first_part = 0.0  # First part in the equation of Bkl
                                       # (the double summation)

            # The inner for loop will calculate
            # $\text{aux} = \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$
            Hk = self.multiH.get_Hk(k)
            Hk_H = Hk.conjugate().transpose()
            for j in range(K):
                aux = 0.0
                # Calculates individually for each stream
                for d in range(Ns[k]):
                    Vjd = F[j][:, d:d + 1]
                    Vjd_H = Vjd.conjugate().transpose()
                    aux = aux + np.dot(np.dot(Hk, np.dot(Vjd, Vjd_H)), Hk_H)

                expected_first_part = expected_first_part + aux

            expected_first_part_with_noise = (expected_first_part
                                              + np.eye(Nr[k]) * noise_power)

            # Test without noise
            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.multiH._calc_JP_Bkl_cov_matrix_first_part(F, k))

            # Test with noise
            np.testing.assert_array_almost_equal(
                expected_first_part_with_noise,
                self.multiH._calc_JP_Bkl_cov_matrix_first_part(
                    F, k, noise_power))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_JP_Bkl_cov_matrix_second_part(self):
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = Nt
        iPu = 1.2
        noise_power = 0.1

        self.multiH.randomize(Nr, Nt, K)

        (_, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H, K, iPu, noise_power)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)

        for k in range(K):
            Hk = self.multiH.get_Hk(k)
            Hk_H = Hk.transpose().conjugate()
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(Hk,
                                              np.dot(np.dot(Vkl, Vkl_H), Hk_H))
                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.multiH._calc_JP_Bkl_cov_matrix_second_part(
                        F[k], k, l))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test for more streams
        K = 3
        Nr = np.ones(K, dtype=int) * 4
        Nt = np.ones(K, dtype=int) * 4
        Ns = Nt
        iPu = 1.2

        self.multiH.randomize(Nr, Nt, K)

        (_, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H, K, iPu, noise_power)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)

        for k in range(K):
            Hk = self.multiH.get_Hk(k)
            Hk_H = Hk.transpose().conjugate()
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(Hk,
                                              np.dot(np.dot(Vkl, Vkl_H), Hk_H))
                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.multiH._calc_JP_Bkl_cov_matrix_second_part(
                        F[k], k, l))

    def test_calc_JP_Bkl_cov_matrix_all_l(self):
        # For the case of a single stream oer user Bkl (which only has l=0)
        # is equal to Qk plus I (identity matrix)
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = Nt
        iPu = 1.2
        noise_power = 0.23

        self.multiH.randomize(Nr, Nt, K)

        (_, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H, K, iPu, noise_power)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)

        for k in range(K):
            Bkl_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
                F, k, noise_power)
            first_part = self.multiH._calc_JP_Bkl_cov_matrix_first_part(
                F, k, noise_power)
            for l in range(Ns[k]):
                second_part = self.multiH._calc_JP_Bkl_cov_matrix_second_part(
                    F[k], k, l)
                expected_Bkl = first_part - second_part

                np.testing.assert_array_almost_equal(expected_Bkl,
                                                     Bkl_all_l[l])

    def test_underline_calc_JP_SINR_k(self):
        # Test the _calc_JP_SINR_k method when joint processing is used.
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = Nt
        iPu = 1.2
        noise_power = 0.001

        self.multiH.randomize(Nr, Nt, K)

        # xxxxxxxxxx Test with random precoder and receive filter xxxxxxxxx
        F = np.empty(K, dtype=np.ndarray)
        U = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(np.sum(Nt), Ns[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(3 * iPu)
            U[k] = randn_c(Nr[k], Ns[k])

        for k in range(K):
            Hk = self.multiH.get_Hk(k)
            Bkl_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
                F, k, N0_or_Rek=noise_power)
            Uk = U[k]
            Fk = F[k]

            SINR_k_all_l = self.multiH._calc_JP_SINR_k(k, Fk, Uk, Bkl_all_l)

            for l in range(Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = F[k][:, l:l + 1]
                aux = np.dot(Ukl_H,
                             np.dot(Hk, Vkl))

                expectedSINRkl = np.abs(np.asscalar(
                    np.dot(aux, aux.transpose().conjugate()) / np.dot(
                        Ukl_H, np.dot(Bkl_all_l[l], Ukl))))

                np.testing.assert_array_almost_equal(expectedSINRkl,
                                                     SINR_k_all_l[l])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Repeat the test, but now using a BD solution xxxxxxxxx
        (newH, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H, K, iPu, 0.0)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)
        big_U = blockdiagonalization.calc_receive_filter(newH)
        aux = single_matrix_to_matrix_of_matrices(big_U, Ns, Nr)
        U = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            U[k] = aux[k, k].conjugate().T

        for k in range(K):
            Hk = self.multiH.get_Hk(k)
            Bkl_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
                F, k, N0_or_Rek=noise_power)
            Uk = U[k]
            Fk = F[k]

            SINR_k_all_l = self.multiH._calc_JP_SINR_k(k, Fk, Uk, Bkl_all_l)

            for l in range(Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = F[k][:, l:l + 1]
                aux = np.dot(Ukl_H,
                             np.dot(Hk, Vkl))

                expectedSINRkl = np.abs(np.asscalar(
                    np.dot(aux, aux.transpose().conjugate()) / np.dot(
                        Ukl_H, np.dot(Bkl_all_l[l], Ukl))))

                np.testing.assert_array_almost_equal(expectedSINRkl,
                                                     SINR_k_all_l[l])

    def test_calc_SINR_with_JP(self):
        # Test the _calc_SINR_k method when joint processing is used.
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = Nt
        iPu = 1.2
        #noise_power = 0.001

        self.multiH.randomize(Nr, Nt, K)

        (newH, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H, K, iPu, 0.0)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)
        big_U = blockdiagonalization.calc_receive_filter(newH)
        aux = single_matrix_to_matrix_of_matrices(big_U, Ns, Nr)
        U = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            U[k] = aux[k, k].conjugate().T


        SINR_all_users = self.multiH.calc_JP_SINR(F, U, noise_power=0.0)

        # xxxxxxxxxx Noise Variance of 0.0 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # k = 0
        B0l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
            F, k=0, N0_or_Rek=0.0)
        expected_SINR0 = self.multiH._calc_JP_SINR_k(
            0, F[0], U[0], B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
            F, k=1, N0_or_Rek=0.0)
        expected_SINR1 = self.multiH._calc_JP_SINR_k(
            1, F[1], U[1], B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 1
        B2l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
            F, k=2, N0_or_Rek=0.0)
        expected_SINR2 = self.multiH._calc_JP_SINR_k(
            2, F[2], U[2], B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Noise Variance of 0.1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # k = 0
        SINR_all_users = self.multiH.calc_JP_SINR(F, U, noise_power=0.1)
        B0l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
            F, k=0, N0_or_Rek=0.1)
        expected_SINR0 = self.multiH._calc_JP_SINR_k(
            0, F[0], U[0], B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
            F, k=1, N0_or_Rek=0.1)
        expected_SINR1 = self.multiH._calc_JP_SINR_k(
            1, F[1], U[1], B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 2
        B2l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
            F, k=2, N0_or_Rek=0.1)
        expected_SINR2 = self.multiH._calc_JP_SINR_k(
            2, F[2], U[2], B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class MultiUserChannelMatrixExtIntTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.multiH = channels.MultiUserChannelMatrixExtInt()
        self.H = np.array(
            [
                [0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
                [0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
                [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
                [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
                [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
                [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
                [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
            ]
        )

        self.K = 3
        self.Nr = np.array([2, 4, 6])
        self.Nt = np.array([2, 3, 5])
        # rank of the external interference. Here we are considering two
        # external interference sources with one and two antennas,
        # respectively. Note that we would get the same channel as if we
        # had considered a single external interference source with three
        # antennas
        self.NtE = np.array([1, 2])

        # Big channel matrix from the external interference to each receiver
        self.extH = 9 * np.ones([12, np.sum(self.NtE)], dtype=int)

    def test_init_from_channel_matrix_and_properties(self):
        # In order to call the init_from_channel_matrix method we need a
        # channel matrix that accounts not only the users' channel but also
        # the external interference sources.
        big_H = np.hstack([self.H, self.extH])

        self.multiH.init_from_channel_matrix(
            big_H, self.Nr, self.Nt, self.K, self.NtE)

        # Test the big_H property. It should be exactly equal to the big_H
        # variable passed to the init_from_channel_matrix method, since we
        # didn't set any path loss matrix yet.
        np.testing.assert_array_equal(
            self.multiH.big_H,
            big_H)

        # Test the properties
        np.testing.assert_array_equal(self.multiH.Nr, self.Nr)
        np.testing.assert_array_equal(self.multiH.Nt, self.Nt)
        np.testing.assert_array_equal(self.multiH.K, self.K)
        np.testing.assert_array_equal(self.multiH.extIntK, len(self.NtE))
        np.testing.assert_array_equal(self.multiH.extIntNt, self.NtE)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now we consider a single external interference source with three
        # antennas
        self.multiH.init_from_channel_matrix(
            big_H, self.Nr, self.Nt, self.K, np.sum(self.NtE))

        # Test the properties
        np.testing.assert_array_equal(self.multiH.Nr, self.Nr)
        np.testing.assert_array_equal(self.multiH.Nt, self.Nt)
        np.testing.assert_array_equal(self.multiH.K, self.K)
        np.testing.assert_array_equal(self.multiH.extIntK, 1)
        np.testing.assert_array_equal(self.multiH.extIntNt, np.sum(self.NtE))

        # We won't test the channels here because the code for setting
        # _big_H and _H was already well tested in the
        # MultiUserChannelMatrix class.

    def test_randomize(self):
        self.multiH.randomize(self.Nr, self.Nt, self.K, self.NtE)

        # Test the properties
        np.testing.assert_array_equal(self.multiH.Nr, self.Nr)
        np.testing.assert_array_equal(self.multiH.Nt, self.Nt)
        np.testing.assert_array_equal(self.multiH.K, self.K)
        np.testing.assert_array_equal(self.multiH.extIntK, len(self.NtE))
        np.testing.assert_array_equal(self.multiH.extIntNt, self.NtE)

        # Now test when the number of transmit/receive antennas is the same
        # in each node
        self.multiH.randomize(3, 2, 4, 1)
        np.testing.assert_array_equal(self.multiH.Nr, np.array([3, 3, 3, 3]))
        np.testing.assert_array_equal(self.multiH.Nt, np.array([2, 2, 2, 2]))
        self.assertEqual(self.multiH.K, 4)
        self.assertEqual(self.multiH.extIntK, 1)
        np.testing.assert_array_equal(self.multiH.extIntNt, np.array([1]))

    def test_big_H_no_ext_int_property(self):
        self.multiH.randomize(np.array([2, 2]), np.array([2, 2]), 2, 2)
        np.testing.assert_array_almost_equal(self.multiH.big_H_no_ext_int,
                                             self.multiH.big_H[:, :-2])

    def test_H_no_ext_int_property(self):
        self.multiH.randomize(np.array([2, 2]), np.array([2, 2]), 2, 2)
        big_H_no_ext_int = self.multiH.big_H_no_ext_int

        H_no_ext_int = self.multiH.H_no_ext_int

        self.assertEqual(H_no_ext_int.shape, (2, 2))
        np.testing.assert_array_almost_equal(H_no_ext_int[0, 0],
                                             big_H_no_ext_int[0:2, 0:2])
        np.testing.assert_array_almost_equal(H_no_ext_int[0, 1],
                                             big_H_no_ext_int[0:2, 2:])
        np.testing.assert_array_almost_equal(H_no_ext_int[1, 0],
                                             big_H_no_ext_int[2:, 0:2])
        np.testing.assert_array_almost_equal(H_no_ext_int[1, 1],
                                             big_H_no_ext_int[2:, 2:])

    def test_set_pathloss(self):
        self.multiH.randomize(self.Nr, self.Nt, self.K, self.NtE)
        K = self.multiH.K
        extIntK = self.multiH.extIntK

        pathloss = np.reshape(np.r_[1:K * K + 1], [K, K])
        pathloss_extint = np.reshape(np.r_[50:K * extIntK + 50], [K, extIntK])

        expected_pathloss = np.hstack([pathloss, pathloss_extint])
        expected_pathloss_big_matrix = np.array(
            [
                [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 50, 51, 51],
                [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 50, 51, 51],
                [4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 52, 53, 53],
                [4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 52, 53, 53],
                [4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 52, 53, 53],
                [4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 52, 53, 53],
                [7, 7, 8, 8, 8, 9, 9, 9, 9, 9, 54, 55, 55],
                [7, 7, 8, 8, 8, 9, 9, 9, 9, 9, 54, 55, 55],
                [7, 7, 8, 8, 8, 9, 9, 9, 9, 9, 54, 55, 55],
                [7, 7, 8, 8, 8, 9, 9, 9, 9, 9, 54, 55, 55],
                [7, 7, 8, 8, 8, 9, 9, 9, 9, 9, 54, 55, 55],
                [7, 7, 8, 8, 8, 9, 9, 9, 9, 9, 54, 55, 55],
            ])

        self.multiH.set_pathloss(pathloss, pathloss_extint)

        np.testing.assert_array_equal(expected_pathloss, self.multiH.pathloss)
        np.testing.assert_array_equal(expected_pathloss_big_matrix,
                                      self.multiH._pathloss_big_matrix)

        # Disable the pathloss
        self.multiH.set_pathloss()
        self.assertIsNone(self.multiH.pathloss)
        self.assertIsNone(self.multiH._pathloss_big_matrix)

    def test_get_H_property(self):
        # test the get_H property when there is pathloss
        self.multiH.randomize(self.Nr, self.Nt, self.K, self.NtE)
        K = self.multiH.K
        extIntK = self.multiH.extIntK

        pathloss = np.reshape(np.r_[1:K * K + 1], [K, K])
        pathloss_extint = np.reshape(np.r_[50:K * extIntK + 50], [K, extIntK])
        self.multiH.set_pathloss(pathloss, pathloss_extint)

        self.assertEqual(self.multiH.H.shape, (3, 5))

        # xxxxxxxxxx Sanity check xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # This will check if the H property matches the big_H property.
        expected_H = single_matrix_to_matrix_of_matrices(
            self.multiH.big_H,
            self.multiH.Nr,
            np.hstack([self.multiH.Nt, self.multiH.extIntNt]))

        nrows, ncols = expected_H.shape
        for r in range(nrows):
            for c in range(ncols):
                np.testing.assert_array_almost_equal(expected_H[r, c],
                                                     self.multiH.H[r, c])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_corrupt_data(self):
        Nt = np.array([2, 2])
        Nr = np.array([3, 2])
        K = len(Nt)
        NtE = np.array([1, 1])

        # Lets initialize our MultiUserChannelMatrixExtInt object with a
        # random channel
        self.multiH.randomize(Nr, Nt, K, NtE)

        # User's data (without the external interference source data)
        input_data = np.empty(2, dtype=np.ndarray)
        input_data[0] = randn_c(2, 5)
        input_data[1] = randn_c(2, 5)

        # External interference data: First lets try with only zeros
        input_data_extint = np.empty(2, dtype=np.ndarray)
        input_data_extint[0] = np.zeros([1, 5], dtype=complex)
        input_data_extint[1] = np.zeros([1, 5], dtype=complex)

        # Initialize a MultiUserChannelMatrix object with the same channel
        # as self.multiH (disregarding the external interference source
        # channel
        multiH_no_ext_int = channels.MultiUserChannelMatrix()
        multiH_no_ext_int.init_from_channel_matrix(
            self.multiH.big_H_no_ext_int, Nr, Nt, K)

        # Test if we receive the same data with and without the external
        # interference source. Note that the received data must be the same
        # since the interference zero for now
        received_data_expected = multiH_no_ext_int.corrupt_data(input_data)
        received_data = self.multiH.corrupt_data(input_data, input_data_extint)

        self.assertEqual(received_data_expected.shape, received_data.shape)
        for index in range(received_data.size):
            np.testing.assert_almost_equal(received_data[index],
                                           received_data_expected[index])

        # xxxxxxxxxx Now lets test with some external interference
        input_data_extint2 = np.empty(2, dtype=np.ndarray)
        input_data_extint2[0] = randn_c(1, 5)
        input_data_extint2[1] = randn_c(1, 5)

        received_data2_expected = received_data_expected

        received_data2 = self.multiH.corrupt_data(input_data,
                                                  input_data_extint2)

        # received_data2_expected for now has only the included the user's
        # signal. Lets add the external interference source's signal.
        received_data2_expected[0] = (
            received_data2_expected[0]
            + np.dot(self.multiH.get_Hkl(0, 2), input_data_extint2[0])
            + np.dot(self.multiH.get_Hkl(0, 3), input_data_extint2[1]))
        received_data2_expected[1] = (
            received_data2_expected[1]
            + np.dot(self.multiH.get_Hkl(1, 2), input_data_extint2[0])
            + np.dot(self.multiH.get_Hkl(1, 3), input_data_extint2[1]))

        # Now lets test if the received_data2 is correct
        self.assertEqual(received_data2_expected.shape, received_data2.shape)
        for index in range(received_data2.size):
            np.testing.assert_almost_equal(received_data2[index],
                                           received_data2_expected[index])

    def test_corrupt_data_with_post_filter(self):
        Nt = np.array([2, 2])
        Nr = np.array([3, 2])
        K = len(Nt)
        NtE = np.array([1, 1])

        # Lets initialize our MultiUserChannelMatrixExtInt object with a
        # random channel
        self.multiH.randomize(Nr, Nt, K, NtE)

        # Set the post processing filter
        W = [randn_c(Nr[0], Nr[0]),
             randn_c(Nr[1], Nr[1])]
        self.multiH.set_post_filter(W)

        # User's data (without the external interference source data)
        input_data = np.empty(2, dtype=np.ndarray)
        input_data[0] = randn_c(2, 5)
        input_data[1] = randn_c(2, 5)

        # External interference data: First lets try with only zeros
        input_data_extint = np.empty(2, dtype=np.ndarray)
        input_data_extint[0] = np.zeros([1, 5], dtype=complex)
        input_data_extint[1] = np.zeros([1, 5], dtype=complex)

        # Initialize a MultiUserChannelMatrix object with the same channel
        # as self.multiH (disregarding the external interference source
        # channel
        multiH_no_ext_int = channels.MultiUserChannelMatrix()
        multiH_no_ext_int.init_from_channel_matrix(
            self.multiH.big_H_no_ext_int,
            Nr,
            Nt,
            K)
        multiH_no_ext_int.set_post_filter(W)

        # Test if we receive the same data with and without the external
        # interference source. Note that the received data must be the same
        # since the interference zero for now
        received_data_expected = multiH_no_ext_int.corrupt_data(input_data)
        received_data = self.multiH.corrupt_data(input_data, input_data_extint)

        self.assertEqual(received_data_expected.shape, received_data.shape)
        for index in range(received_data.size):
            np.testing.assert_almost_equal(received_data[index],
                                           received_data_expected[index])

        # xxxxxxxxxx Now lets test with some external interference
        input_data_extint2 = np.empty(2, dtype=np.ndarray)
        input_data_extint2[0] = randn_c(1, 5)
        input_data_extint2[1] = randn_c(1, 5)

        received_data2_expected = received_data_expected

        received_data2 = self.multiH.corrupt_data(input_data,
                                                  input_data_extint2)

        # received_data2_expected for now has only the included the user's
        # signal. Lets add the external interference source's signal.
        received_data2_expected[0] = (
            # Original received data
            received_data2_expected[0]
            # Plus FILTERED interference from first interference source
            +
            np.dot(W[0].conjugate().T,
                   np.dot(self.multiH.get_Hkl(0, 2),
                          input_data_extint2[0]))
            # Plus FILTERED interference from second interference source
            +
            np.dot(W[0].conjugate().T,
                   np.dot(self.multiH.get_Hkl(0, 3),
                          input_data_extint2[1])))

        received_data2_expected[1] = (
            # Original received data
            received_data2_expected[1]
            # Plus FILTERED interference from first interference source
            +
            np.dot(W[1].conjugate().T,
                   np.dot(self.multiH.get_Hkl(1, 2),
                          input_data_extint2[0]))
            # Plus FILTERED interference from second interference source
            +
            np.dot(W[1].conjugate().T,
                   np.dot(self.multiH.get_Hkl(1, 3),
                          input_data_extint2[1])))

        # Now lets test if the received_data2 is correct
        self.assertEqual(received_data2_expected.shape, received_data2.shape)
        for index in range(received_data2.size):
            np.testing.assert_almost_equal(received_data2[index],
                                           received_data2_expected[index])

    def test_get_channel_all_transmitters_to_single_receiver(self):
        big_H = np.hstack([self.H, self.extH])
        K = self.K
        extIntK = self.NtE.size
        Nr = self.Nr
        Nt = self.Nt
        NtE = self.NtE

        self.multiH.init_from_channel_matrix(big_H, Nr, Nt, K, NtE)

        expected_H1 = self.multiH.big_H[0:2, :np.sum(Nt)]
        expected_H2 = self.multiH.big_H[2:6, :np.sum(Nt)]
        expected_H3 = self.multiH.big_H[6:, :np.sum(Nt)]

        # xxxxx Test without pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_equal(
            self.multiH.get_Hk_without_ext_int(0),
            expected_H1
        )
        np.testing.assert_array_equal(
            self.multiH.get_Hk_without_ext_int(1),
            expected_H2
        )
        np.testing.assert_array_equal(
            self.multiH.get_Hk_without_ext_int(2),
            expected_H3
        )

        # xxxxx Test with pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        pathloss = np.abs(np.random.randn(K, K))
        pathloss_extint = np.abs(np.random.randn(K, extIntK))
        self.multiH.set_pathloss(pathloss, pathloss_extint)
        expected_H1 = self.multiH.big_H[0:2, :np.sum(Nt)]
        expected_H2 = self.multiH.big_H[2:6, :np.sum(Nt)]
        expected_H3 = self.multiH.big_H[6:, :np.sum(Nt)]
        np.testing.assert_array_equal(
            self.multiH.get_Hk_without_ext_int(0),
            expected_H1
        )
        np.testing.assert_array_equal(
            self.multiH.get_Hk_without_ext_int(1),
            expected_H2
        )
        np.testing.assert_array_equal(
            self.multiH.get_Hk_without_ext_int(2),
            expected_H3
        )

    def test_calc_cov_matrix_extint_plus_noise(self):
        self.K = 3
        self.Nr = np.array([2, 4, 6])
        # self.Nt = np.array([2, 3, 5])
        # self.NtE = np.array([1, 2])
        noise_var = 0.01
        self.multiH.randomize(self.Nr, self.Nt, self.K, self.NtE)
        H1 = self.multiH.big_H[0:2, 10:]
        H2 = self.multiH.big_H[2:6, 10:]
        H3 = self.multiH.big_H[6:12, 10:]
        noise_cov1 = np.eye(self.Nr[0]) * noise_var
        noise_cov2 = np.eye(self.Nr[1]) * noise_var
        noise_cov3 = np.eye(self.Nr[2]) * noise_var

        expected_cov_int = np.empty(3, dtype=np.ndarray)
        expected_cov_int_plus_noise = np.empty(3, dtype=np.ndarray)

        expected_cov_int[0] = np.dot(H1, H1.conjugate().transpose())
        expected_cov_int[1] = np.dot(H2, H2.conjugate().transpose())
        expected_cov_int[2] = np.dot(H3, H3.conjugate().transpose())

        expected_cov_int_plus_noise[0] = expected_cov_int[0] + noise_cov1
        expected_cov_int_plus_noise[1] = expected_cov_int[1] + noise_cov2
        expected_cov_int_plus_noise[2] = expected_cov_int[2] + noise_cov3

        cov_int = self.multiH.calc_cov_matrix_extint_plus_noise()
        cov_int_plus_noise = self.multiH.calc_cov_matrix_extint_plus_noise(
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

    def test_calc_Q(self):
        K = 3
        Nt = np.array([2, 2, 2])
        Nr = np.array([2, 2, 2])
        Ns = np.array([1, 1, 1])
        NtE = np.array([1, 2])
        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        self.multiH.randomize(Nr, Nt, K, NtE)

        F_all_k = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F_all_k[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F_all_k[k] = (F_all_k[k]
                          / np.linalg.norm(F_all_k[k], 'fro')
                          * np.sqrt(P[k]))

        # xxxxx Calculate the expected Q[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H01_F1 = np.dot(
            self.multiH.get_Hkl(k, 1),
            F_all_k[1]
        )
        H02_F2 = np.dot(
            self.multiH.get_Hkl(k, 2),
            F_all_k[2]
        )
        R0_e0 = self.multiH.get_Hkl(0, 3)
        R0_e1 = self.multiH.get_Hkl(0, 4)

        expected_Q0_no_ext_int_or_noise = (
            # Internal interference part
            np.dot(H01_F1,
                   H01_F1.transpose().conjugate()) +
            np.dot(H02_F2,
                   H02_F2.transpose().conjugate())
            # # External interference part
            # np.dot(R0_e0, R0_e0.conjugate().T) +
            # np.dot(R0_e1, R0_e1.conjugate().T)
            )

        Qk = self.multiH.calc_Q(k, F_all_k, noise_var=0.0, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q0_no_ext_int_or_noise)

        # Now with external interference
        expected_Q0_no_noise = (expected_Q0_no_ext_int_or_noise +
                                np.dot(R0_e0, R0_e0.conjugate().T) +
                                np.dot(R0_e1, R0_e1.conjugate().T))
        Qk = self.multiH.calc_Q(k, F_all_k, noise_var=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q0_no_noise)

        # Now with external interference and noise
        noise_var = round(0.1 * np.random.rand(), 4)
        Qk = self.multiH.calc_Q(k, F_all_k, noise_var=noise_var)
        expected_Q0 = expected_Q0_no_noise + np.eye(2) * noise_var
        np.testing.assert_array_almost_equal(Qk, expected_Q0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0 = np.dot(
            self.multiH.get_Hkl(k, 0),
            F_all_k[0]
        )
        H12_F2 = np.dot(
            self.multiH.get_Hkl(k, 2),
            F_all_k[2]
        )
        R1_e0 = self.multiH.get_Hkl(1, 3)
        R1_e1 = self.multiH.get_Hkl(1, 4)

        expected_Q1_no_ext_int_or_noise = (
            np.dot(H10_F0,
                   H10_F0.transpose().conjugate()) +
            np.dot(H12_F2,
                   H12_F2.transpose().conjugate())
            )

        Qk = self.multiH.calc_Q(k, F_all_k, noise_var=0.0, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q1_no_ext_int_or_noise)

        # Now with external interference
        expected_Q1_no_noise = (expected_Q1_no_ext_int_or_noise +
                                np.dot(R1_e0, R1_e0.conjugate().T) +
                                np.dot(R1_e1, R1_e1.conjugate().T))
        Qk = self.multiH.calc_Q(k, F_all_k, noise_var=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q1_no_noise)

        # Now with external interference and noise
        noise_var = round(0.1 * np.random.rand(), 4)
        Qk = self.multiH.calc_Q(k, F_all_k, noise_var=noise_var)
        expected_Q1 = expected_Q1_no_noise + np.eye(2) * noise_var
        np.testing.assert_array_almost_equal(Qk, expected_Q1)
        # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H20_F0 = np.dot(
            self.multiH.get_Hkl(k, 0),
            F_all_k[0]
        )
        H21_F1 = np.dot(
            self.multiH.get_Hkl(k, 1),
            F_all_k[1]
        )
        R2_e0 = self.multiH.get_Hkl(2, 3)
        R2_e1 = self.multiH.get_Hkl(2, 4)

        expected_Q2_no_ext_int_or_noise = (
            np.dot(H20_F0,
                   H20_F0.transpose().conjugate()) +
            np.dot(H21_F1,
                   H21_F1.transpose().conjugate())
            )

        Qk = self.multiH.calc_Q(k, F_all_k, noise_var=0.0, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q2_no_ext_int_or_noise)

        # Now with external interference
        expected_Q2_no_noise = (expected_Q2_no_ext_int_or_noise +
                                np.dot(R2_e0, R2_e0.conjugate().T) +
                                np.dot(R2_e1, R2_e1.conjugate().T))
        Qk = self.multiH.calc_Q(k, F_all_k, noise_var=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q2_no_noise)

        # Now with external interference and noise
        noise_var = round(0.1 * np.random.rand(), 4)
        Qk = self.multiH.calc_Q(k, F_all_k, noise_var=noise_var)
        expected_Q2 = expected_Q2_no_noise + np.eye(2) * noise_var
        np.testing.assert_array_almost_equal(Qk, expected_Q2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_JP_Q(self):
        K = 3
        Nt = np.array([2, 2, 2])
        Nr = np.array([2, 2, 2])
        Ns = np.array([1, 1, 1])
        NtE = np.array([1, 2])
        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        self.multiH.randomize(Nr, Nt, K, NtE)

        F_all_k = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F_all_k[k] = randn_c(np.sum(Nt), Ns[k]) * np.sqrt(P[k])
            F_all_k[k] = (F_all_k[k]
                          / np.linalg.norm(F_all_k[k], 'fro')
                          * np.sqrt(P[k]))

        noise_var = round(0.1 * np.random.rand(), 4)
        Pe = round(np.random.rand(), 4)

        Re_no_noise = self.multiH.calc_cov_matrix_extint_plus_noise(
            noise_var=0.0,
            pe=Pe)
        # Re_with_noise = self.multiH.calc_cov_matrix_extint_plus_noise(
        #     noise_var=noise_var,
        #     pe=Pe)

        # xxxxx Calculate the expected Q[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H0_F1 = np.dot(
            self.multiH.get_Hk_without_ext_int(k),
            F_all_k[1]
        )
        H0_F2 = np.dot(
            self.multiH.get_Hk_without_ext_int(k),
            F_all_k[2]
        )

        expected_Q0_no_ext_int_or_noise = (
            # Internal interference part
            np.dot(H0_F1,
                   H0_F1.transpose().conjugate()) +
            np.dot(H0_F2,
                   H0_F2.transpose().conjugate()))

        Qk = self.multiH.calc_JP_Q(k, F_all_k, noise_var=0.0, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q0_no_ext_int_or_noise)

        # Now with external interference
        expected_Q0_no_noise = (expected_Q0_no_ext_int_or_noise +
                                Re_no_noise[0])
        Qk = self.multiH.calc_JP_Q(k, F_all_k, noise_var=0.0, pe=Pe)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q0_no_noise)

        # Now with external interference and noise
        Qk = self.multiH.calc_JP_Q(k, F_all_k, noise_var=noise_var, pe=Pe)
        expected_Q0 = expected_Q0_no_noise + np.eye(2) * noise_var
        np.testing.assert_array_almost_equal(Qk, expected_Q0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H1_F0 = np.dot(
            self.multiH.get_Hk_without_ext_int(k),
            F_all_k[0]
        )
        H1_F2 = np.dot(
            self.multiH.get_Hk_without_ext_int(k),
            F_all_k[2]
        )

        expected_Q1_no_ext_int_or_noise = (
            np.dot(H1_F0,
                   H1_F0.transpose().conjugate()) +
            np.dot(H1_F2,
                   H1_F2.transpose().conjugate())
            )

        Qk = self.multiH.calc_JP_Q(k, F_all_k, noise_var=0.0, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q1_no_ext_int_or_noise)

        # Now with external interference
        expected_Q1_no_noise = (expected_Q1_no_ext_int_or_noise +
                                Re_no_noise[1])
        Qk = self.multiH.calc_JP_Q(k, F_all_k, noise_var=0.0, pe=Pe)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q1_no_noise)

        # Now with external interference and noise
        Qk = self.multiH.calc_JP_Q(k, F_all_k, noise_var=noise_var, pe=Pe)
        expected_Q1 = expected_Q1_no_noise + np.eye(2) * noise_var
        np.testing.assert_array_almost_equal(Qk, expected_Q1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H2_F0 = np.dot(
            self.multiH.get_Hk_without_ext_int(k),
            F_all_k[0]
        )
        H2_F1 = np.dot(
            self.multiH.get_Hk_without_ext_int(k),
            F_all_k[1]
        )

        expected_Q2_no_ext_int_or_noise = (
            np.dot(H2_F0,
                   H2_F0.transpose().conjugate()) +
            np.dot(H2_F1,
                   H2_F1.transpose().conjugate())
            )

        Qk = self.multiH.calc_JP_Q(k, F_all_k, noise_var=0.0, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q2_no_ext_int_or_noise)

        # Now with external interference
        expected_Q2_no_noise = (expected_Q2_no_ext_int_or_noise +
                                Re_no_noise[2])
        Qk = self.multiH.calc_JP_Q(k, F_all_k, noise_var=0.0, pe=Pe)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q2_no_noise)

        # Now with external interference and noise
        Qk = self.multiH.calc_JP_Q(k, F_all_k, noise_var=noise_var, pe=Pe)
        expected_Q2 = expected_Q2_no_noise + np.eye(2) * noise_var
        np.testing.assert_array_almost_equal(Qk, expected_Q2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_Bkl_cov_matrix_first_part(self):
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = np.ones(K, dtype=int) * 1
        NtE = np.array([1])

        P = np.array([1.2, 1.5, 0.9])
        Pe = 0.7

        self.multiH.randomize(Nr, Nt, K, NtE)
        F = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])

        Re = self.multiH.calc_cov_matrix_extint_plus_noise(
            noise_var=0.0, pe=Pe)

        # For ones stream the expected Bkl is equivalent to the Q matrix
        # plus the direct channel part.
        for k in range(K):
            Hkk = self.multiH.get_Hkl(k, k)
            Fk = F[k]
            HkkFk = np.dot(Hkk, Fk)
            expected_first_part = (self.multiH.calc_Q(k, F, pe=Pe)
                                   + np.dot(HkkFk, HkkFk.conjugate().T))

            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.multiH._calc_Bkl_cov_matrix_first_part(F, k, Re[k]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test with noise variance different from zero
        noise_var = 0.13
        Re = self.multiH.calc_cov_matrix_extint_plus_noise(
            noise_var=noise_var, pe=Pe)

        # For ones stream the expected Bkl is equivalent to the Q matrix
        # plus the direct channel part.
        for k in range(K):
            Hkk = self.multiH.get_Hkl(k, k)
            Fk = F[k]
            HkkFk = np.dot(Hkk, Fk)
            expected_first_part = (
                self.multiH.calc_Q(k, F, noise_var=noise_var, pe=Pe)
                + np.dot(HkkFk, HkkFk.conjugate().T))

            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.multiH._calc_Bkl_cov_matrix_first_part(F, k, Re[k]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test for more streams
        Nr = np.ones(K, dtype=int) * 4
        Nt = np.ones(K, dtype=int) * 4
        Ns = np.ones(3, dtype=int) * 2
        NtE = np.array([1, 1])

        self.multiH.randomize(Nr, Nt, K, NtE)

        Re = self.multiH.calc_cov_matrix_extint_plus_noise(
            noise_var=noise_var, pe=Pe)

        F = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])

        for k in range(K):
            expected_first_part = 0.0  # First part in the equation of Bkl
                                       # (the double summation)

            # The inner for loop will calculate
            # $\text{aux} = \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$
            for j in range(K):
                aux = 0.0
                Hkj = self.multiH.get_Hkl(k, j)
                Hkj_H = Hkj.conjugate().transpose()

                # Calculates individually for each stream
                for d in range(Ns[k]):
                    Vjd = F[j][:, d:d + 1]
                    Vjd_H = Vjd.conjugate().transpose()
                    aux = aux + np.dot(np.dot(Hkj, np.dot(Vjd, Vjd_H)), Hkj_H)

                expected_first_part = expected_first_part + aux
            expected_first_part = expected_first_part + Re[k]

            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.multiH._calc_Bkl_cov_matrix_first_part(F, k, Re[k]))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_Bkl(self):
        # For the case of a single stream oer user Bkl (which only has l=0)
        # is equal to Qk
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = np.ones(K, dtype=int) * 1
        NtE = np.array([1])
        P = np.array([1.2, 1.5, 0.9])
        Pe = 0.7
        noise_power = 0.568

        self.multiH.randomize(Nr, Nt, K, NtE)

        Re = self.multiH.calc_cov_matrix_extint_plus_noise(
            noise_var=noise_power, pe=Pe)

        F = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])

        for k in range(K):
            # We only have the stream 0
            expected_Bk0 = self.multiH.calc_Q(
                k, F, noise_var=noise_power, pe=Pe)
            Bk0 = self.multiH._calc_Bkl_cov_matrix_all_l(F, k, Re[k])[0]
            np.testing.assert_array_almost_equal(expected_Bk0, Bk0)

    def test_underline_calc_SINR_k(self):
        multiUserChannel = channels.MultiUserChannelMatrixExtInt()
        #iasolver = MaxSinrIASolver(multiUserChannel)
        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 2
        NtE = np.array([1])
        Pe = 0.7
        noise_power = 0.568

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(Nr, Nt, K, NtE)

        Re = multiUserChannel.calc_cov_matrix_extint_plus_noise(
            noise_var=noise_power, pe=Pe)

        F = np.empty(K, dtype=np.ndarray)
        U = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])
            U[k] = randn_c(Nr[k], Ns[k])

        for k in range(K):
            Hkk = multiUserChannel.get_Hkl(k, k)
            Bkl_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
                F, k, Re[k])
            Uk = U[k]
            Fk = F[k]
            # Uk_H = iasolver.full_W_H[k]

            SINR_k_all_l = multiUserChannel._calc_SINR_k(k, Fk, Uk, Bkl_all_l)

            for l in range(Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = F[k][:, l:l + 1]
                aux = np.dot(Ukl_H,
                             np.dot(Hkk, Vkl))

                expectedSINRkl = np.asscalar(
                    np.dot(aux, aux.transpose().conjugate()) / np.dot(
                        Ukl_H, np.dot(Bkl_all_l[l], Ukl))
                )

                np.testing.assert_array_almost_equal(expectedSINRkl,
                                                     SINR_k_all_l[l])

        # xxxxxxxxxx Repeat the tests, but now using an IA solution xxxxxxx
        iasolver = ClosedFormIASolver(multiUserChannel)
        iasolver.solve(Ns=2)
        F = iasolver.full_F
        U = iasolver.full_W

        Pe = 0.01
        Re = multiUserChannel.calc_cov_matrix_extint_plus_noise(
            noise_var=0.0001, pe=Pe)

        for k in range(K):
            Hkk = multiUserChannel.get_Hkl(k, k)
            Uk = U[k]
            Fk = F[k]

            Bkl_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
                F, k, Re[k])
            SINR_k_all_l = multiUserChannel._calc_SINR_k(
                k, F[k], U[k], Bkl_all_l)

            for l in range(Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = F[k][:, l:l + 1]
                aux = np.dot(Ukl_H,
                             np.dot(Hkk, Vkl))

                expectedSINRkl = abs(np.asscalar(
                    np.dot(aux, aux.transpose().conjugate()) / np.dot(
                        Ukl_H, np.dot(Bkl_all_l[l], Ukl))))

                np.testing.assert_array_almost_equal(expectedSINRkl,
                                                     SINR_k_all_l[l])

    def test_calc_SINR(self):
        multiUserChannel = channels.MultiUserChannelMatrixExtInt()
        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 2
        NtE = np.array([1])

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(Nr, Nt, K, NtE)

        iasolver = ClosedFormIASolver(multiUserChannel)
        iasolver.solve(Ns, P)
        F = iasolver.full_F
        U = iasolver.full_W

        # xxxxxxxxxx Noise Variance = 0.0 and Pe = 0 xxxxxxxxxxxxxxxxxxxxxx
        Pe = 0.00
        noise_power = 0.00
        Re = multiUserChannel.calc_cov_matrix_extint_plus_noise(
            noise_var=noise_power, pe=Pe)

        SINR_all_users = multiUserChannel.calc_SINR(
            F, U, noise_power=noise_power, pe=Pe)

        # SINR of all users should be super high (inf)
        self.assertTrue(np.all(SINR_all_users[0] > 1e10))
        self.assertTrue(np.all(SINR_all_users[1] > 1e10))
        self.assertTrue(np.all(SINR_all_users[2] > 1e10))

        # k = 0
        B0l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=0, N0_or_Rek=Re[0])
        expected_SINR0 = multiUserChannel._calc_SINR_k(
            0, F[0], U[0], B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F,
            k=1,
            N0_or_Rek=Re[1])
        expected_SINR1 = multiUserChannel._calc_SINR_k(
            1, F[1], U[1], B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 2
        B2l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F,
            k=2,
            N0_or_Rek=Re[2])
        expected_SINR2 = multiUserChannel._calc_SINR_k(
            2, F[2], U[2], B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Noise Variance = 0.01 and Pe = 0.63 xxxxxxxxxxxxxxxxxx
        Pe = 0.01
        noise_power = 0.63
        Re = multiUserChannel.calc_cov_matrix_extint_plus_noise(
            noise_var=noise_power, pe=Pe)

        SINR_all_users = multiUserChannel.calc_SINR(
            F, U, noise_power=noise_power, pe=Pe)

        # SINR should lower than 10 for these values of noise variance and Pe
        self.assertTrue(np.all(SINR_all_users[0] < 10))
        self.assertTrue(np.all(SINR_all_users[1] < 10))
        self.assertTrue(np.all(SINR_all_users[2] < 10))

        # k = 0
        B0l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F,
            k=0,
            N0_or_Rek=Re[0])
        expected_SINR0 = multiUserChannel._calc_SINR_k(
            0, F[0], U[0], B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=1, N0_or_Rek=Re[1])
        expected_SINR1 = multiUserChannel._calc_SINR_k(
            1, F[1], U[1], B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 2
        B2l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=2, N0_or_Rek=Re[2])
        expected_SINR2 = multiUserChannel._calc_SINR_k(
            2, F[2], U[2], B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_JP_Bkl_cov_matrix_first_part(self):
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = Nt
        iPu = 1.2
        NtE = np.array([1])
        noise_power = 0.1
        Pe = 1.2

        self.multiH.randomize(Nr, Nt, K, NtE)

        (_, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H_no_ext_int,
            K, iPu, noise_power)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)

        # xxxxx Test with no external interference (zero energy) xxxxxxxxxx
        # For ones stream the expected Bkl is equivalent to the Q matrix
        # plus the direct channel part.
        for k in range(K):
            Hk = self.multiH.get_Hk_without_ext_int(k)
            Fk = F[k]
            HkFk = np.dot(Hk, Fk)
            expected_first_part = (self.multiH.calc_JP_Q(k, F, pe=0.0)
                                   + np.dot(HkFk, HkFk.conjugate().T))
            expected_first_part_with_noise = self.multiH.calc_JP_Q(
                k, F, noise_power, pe=0.0) + np.dot(HkFk, HkFk.conjugate().T)

            # Test without noise
            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.multiH._calc_JP_Bkl_cov_matrix_first_part(
                    F, k, np.zeros([Nr[0], Nr[1]])))

            # Test with noise
            np.testing.assert_array_almost_equal(
                expected_first_part_with_noise,
                self.multiH._calc_JP_Bkl_cov_matrix_first_part(
                    F, k, noise_power*np.eye(Nr[k])))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test with external interference xxxxxxxxxxxxxxxxxxxxxx
        Re_no_noise = self.multiH.calc_cov_matrix_extint_plus_noise(
            noise_var=0.0, pe=Pe)
        Re = self.multiH.calc_cov_matrix_extint_plus_noise(
            noise_var=noise_power, pe=Pe)

        # For ones stream the expected Bkl is equivalent to the Q matrix
        # plus the direct channel part.
        for k in range(K):
            Hk = self.multiH.get_Hk_without_ext_int(k)
            Fk = F[k]
            HkFk = np.dot(Hk, Fk)
            expected_first_part = (self.multiH.calc_JP_Q(k, F, pe=Pe)
                                   + np.dot(HkFk, HkFk.conjugate().T))
            expected_first_part_with_noise = (
                self.multiH.calc_JP_Q(k, F, noise_power, pe=Pe)
                + np.dot(HkFk, HkFk.conjugate().T))

            # Test without noise
            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.multiH._calc_JP_Bkl_cov_matrix_first_part(
                    F, k, Re_no_noise[k]))

            # Test with noise
            np.testing.assert_array_almost_equal(
                expected_first_part_with_noise,
                self.multiH._calc_JP_Bkl_cov_matrix_first_part(F, k, Re[k]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_JP_Bkl_cov_matrix_second_part(self):
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = Nt
        iPu = 1.2
        NtE = np.array([1])
        noise_power = 0.1
        # Pe = 1.2

        self.multiH.randomize(Nr, Nt, K, NtE)

        (_, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H_no_ext_int,
            K, iPu, noise_power)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)

        for k in range(K):
            Hk = self.multiH.get_Hk_without_ext_int(k)
            Hk_H = Hk.transpose().conjugate()
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(Hk,
                                              np.dot(np.dot(Vkl, Vkl_H), Hk_H))

                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.multiH._calc_JP_Bkl_cov_matrix_second_part(
                        F[k], k, l))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        K = 3
        Nr = np.ones(K, dtype=int) * 4
        Nt = np.ones(K, dtype=int) * 4
        Ns = Nt
        iPu = 1.2
        noise_power = 0.1
        Pe = 1.2

        self.multiH.randomize(Nr, Nt, K, NtE)

        (_, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H_no_ext_int,
            K, iPu, noise_power)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)

        for k in range(K):
            Hk = self.multiH.get_Hk_without_ext_int(k)
            Hk_H = Hk.transpose().conjugate()
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(Hk,
                                              np.dot(np.dot(Vkl, Vkl_H), Hk_H))

                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.multiH._calc_JP_Bkl_cov_matrix_second_part(
                        F[k], k, l))

    def test_calc_JP_Bkl_cov_matrix_all_l(self):
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = Nt
        iPu = 1.2
        NtE = np.array([1])
        noise_power = 0.1
        # Pe = 1.2

        self.multiH.randomize(Nr, Nt, K, NtE)

        (_, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H_no_ext_int,
            K, iPu, noise_power)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)

        for k in range(K):
            Bkl_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
                F, k, noise_power)
            first_part = self.multiH._calc_JP_Bkl_cov_matrix_first_part(
                F, k, noise_power)
            for l in range(Ns[k]):
                second_part = self.multiH._calc_JP_Bkl_cov_matrix_second_part(
                    F[k], k, l)
                expected_Bkl = first_part - second_part

                np.testing.assert_array_almost_equal(expected_Bkl,
                                                     Bkl_all_l[l])

    def test_underline_calc_JP_SINR_k(self):
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = Nt
        iPu = 1.2
        NtE = np.array([1])
        noise_power = 0.0001

        self.multiH.randomize(Nr, Nt, K, NtE)

        (newH, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H_no_ext_int,
            K, iPu, noise_var=noise_power)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)
        big_U = blockdiagonalization.calc_receive_filter(newH)
        aux = single_matrix_to_matrix_of_matrices(big_U, Ns, Nr)
        U = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            U[k] = aux[k, k].conjugate().T

        # xxxxx Test first with no external interference xxxxxxxxxxxxxxxxxx
        Pe = 0.00
        Re = self.multiH.calc_cov_matrix_extint_plus_noise(
            noise_var=noise_power, pe=Pe)

        for k in range(K):
            Hk = self.multiH.get_Hk_without_ext_int(k)
            Bkl_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F, k, Re[k])
            Uk = U[k]
            Fk = F[k]

            SINR_k_all_l = self.multiH._calc_JP_SINR_k(k, Fk, Uk, Bkl_all_l)

            for l in range(Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = F[k][:, l:l + 1]
                aux = np.dot(Ukl_H,
                             np.dot(Hk, Vkl))

                expectedSINRkl = np.abs(np.asscalar(
                    np.dot(aux, aux.transpose().conjugate()) / np.dot(
                        Ukl_H, np.dot(Bkl_all_l[l], Ukl))
                ))

                np.testing.assert_array_almost_equal(expectedSINRkl,
                                                     SINR_k_all_l[l])

        # xxxxx Repeat the test, but now with external interference xxxxxxx
        Pe = 0.1
        Re = self.multiH.calc_cov_matrix_extint_plus_noise(
            noise_var=noise_power, pe=Pe)

        for k in range(K):
            Hk = self.multiH.get_Hk_without_ext_int(k)
            Bkl_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F, k, Re[k])
            Uk = U[k]
            Fk = F[k]

            SINR_k_all_l = self.multiH._calc_JP_SINR_k(k, Fk, Uk, Bkl_all_l)

            for l in range(Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = F[k][:, l:l + 1]
                aux = np.dot(Ukl_H,
                             np.dot(Hk, Vkl))

                expectedSINRkl = np.abs(np.asscalar(
                    np.dot(aux, aux.transpose().conjugate()) / np.dot(
                        Ukl_H, np.dot(Bkl_all_l[l], Ukl))
                ))

                np.testing.assert_array_almost_equal(expectedSINRkl,
                                                     SINR_k_all_l[l])

    def test_calc_SINR_with_JP(self):
        K = 3
        Nr = np.ones(K, dtype=int) * 2
        Nt = np.ones(K, dtype=int) * 2
        Ns = Nt
        iPu = 1.2
        NtE = np.array([1])

        self.multiH.randomize(Nr, Nt, K, NtE)

        (newH, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H_no_ext_int,
            K, iPu, noise_var=0.0)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)
        big_U = blockdiagonalization.calc_receive_filter(newH)
        aux = single_matrix_to_matrix_of_matrices(big_U, Ns, Nr)
        U = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            U[k] = aux[k, k].conjugate().T

        SINR_all_users = self.multiH.calc_JP_SINR(
            F, U, noise_power=0.0, pe=0.0)

        # xxxxxxxxxx Noise Variance of 0.0, Pe of 0.0 xxxxxxxxxxxxxxxxxxxxx
        # k = 0
        B0l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
            F, k=0, N0_or_Rek=0.0)
        expected_SINR0 = self.multiH._calc_JP_SINR_k(
            0, F[0], U[0], B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
            F, k=1, N0_or_Rek=0.0)
        expected_SINR1 = self.multiH._calc_JP_SINR_k(
            1, F[1], U[1], B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 1
        B2l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
            F, k=2, N0_or_Rek=0.0)
        expected_SINR2 = self.multiH._calc_JP_SINR_k(
            2, F[2], U[2], B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Noise Variance of 0.01, Pe of 0.63 xxxxxxxxxxxxxxxxxx
        noise_var = 0.01
        Pe = 0.63
        Re = self.multiH.calc_cov_matrix_extint_plus_noise(noise_var, pe=Pe)

        SINR_all_users = self.multiH.calc_JP_SINR(
            F, U, noise_power=noise_var, pe=Pe)

        # k = 0
        B0l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
            F, k=0, N0_or_Rek=Re[0])
        expected_SINR0 = self.multiH._calc_JP_SINR_k(
            0, F[0], U[0], B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
            F, k=1, N0_or_Rek=Re[1])
        expected_SINR1 = self.multiH._calc_JP_SINR_k(
            1, F[1], U[1], B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 2
        B2l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(
            F, k=2, N0_or_Rek=Re[2])
        expected_SINR2 = self.multiH._calc_JP_SINR_k(
            2, F[2], U[2], B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx OFDM Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class OfdmTestCase(unittest.TestCase):
    """Unittests for the OFDM class in the ofdm module."""
    def setUp(self):
        """Called before each test."""
        self.ofdm_object = OFDM(64, 16, 52)

    def test_ofdm_set_parameters(self):
        # Test regular usage
        self.assertEqual(self.ofdm_object.fft_size, 64)
        self.assertEqual(self.ofdm_object.cp_size, 16)
        self.assertEqual(self.ofdm_object.num_used_subcarriers, 52)
        self.ofdm_object.set_parameters(128, 32, 100)
        self.assertEqual(self.ofdm_object.fft_size, 128)
        self.assertEqual(self.ofdm_object.cp_size, 32)
        self.assertEqual(self.ofdm_object.num_used_subcarriers, 100)

        # Test if an exception is raised when any invalid parameters are
        # passed to set_parameters

        # Raises an exception if number of used subcarriers is odd
        with self.assertRaises(ValueError):
            self.ofdm_object.set_parameters(64, 16, 51)

        # Raises an exception if number of used subcarriers is greater than
        # the fft_size
        with self.assertRaises(ValueError):
            self.ofdm_object.set_parameters(64, 16, 70)

        # Raises an exception if cp_size is negative
        with self.assertRaises(ValueError):
            self.ofdm_object.set_parameters(64, -2, 52)

        # Raises an exception if cp_size is greater than the fft_size
        with self.assertRaises(ValueError):
            self.ofdm_object.set_parameters(64, 65, 52)

        # Test if the number of subcarriers defaults to the fft_size when
        # not provided
        self.ofdm_object.set_parameters(64, 16)
        self.assertEqual(self.ofdm_object.num_used_subcarriers, 64)

    def test_prepare_input_signal(self, ):
        input_signal = np.r_[1:53]  # 52 elements -> exactly the number of
                                    # used subcarriers in the OFDM object

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Lets first test the case where we use exactly 52 subcarriers from
        # the 64 available subcarriers. That is, no zeropadding is needed.
        (zeropad, num_ofdm_symbols) = self.ofdm_object._calc_zeropad(
            input_signal.size)
        self.assertEqual(zeropad, 0)
        self.assertEqual(num_ofdm_symbols, 1)

        expected_data = np.array(
            [[0., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.,
              39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51.,
              52., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 4.,
              5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
              19., 20., 21., 22., 23., 24., 25., 26.]])
        np.testing.assert_array_equal(
            self.ofdm_object._prepare_input_signal(input_signal),
            expected_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now lets change the number of used subcarriers and repeat the
        # tests so that the case when zeropad is performed is also tested.
        self.ofdm_object.num_used_subcarriers = 60
        (zeropad, num_ofdm_symbols) = self.ofdm_object._calc_zeropad(input_signal.size)
        # We used 60 subcarriers, but we have 52 elements -> We need to add
        # 8 zeros at the end of the unput data
        self.assertEqual(zeropad, 8)
        # But we still have only one OFDM symbol
        self.assertEqual(num_ofdm_symbols, 1)

        expected_data2 = np.array(
            [[0., 31., 32., 33., 34., 35., 36., 37., 38., 39., 40., 41., 42.,
              43., 44., 45., 46., 47., 48., 49., 50., 51., 52., 0., 0., 0.,
              0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
              9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
              22., 23., 24., 25., 26., 27., 28., 29., 30., ]])
        np.testing.assert_array_equal(
            self.ofdm_object._prepare_input_signal(input_signal),
            expected_data2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now lets test the case when we use all subcarriers (but still
        # with zeropadding)
        self.ofdm_object.num_used_subcarriers = 64
        (zeropad, num_ofdm_symbols) = self.ofdm_object._calc_zeropad(
            input_signal.size)
        self.assertEqual(zeropad, 12)

        # But we still have only one OFDM symbol
        self.assertEqual(num_ofdm_symbols, 1)

        # Notice that in this case the DC subcarrier is used
        expected_data3 = np.array(
            [[33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45.,
              46., 47., 48., 49., 50., 51., 52., 0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.,
              12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
              25., 26., 27., 28., 29., 30., 31., 32.]])
        np.testing.assert_array_equal(self.ofdm_object._prepare_input_signal(input_signal), expected_data3)

    def test_prepare_decoded_signal(self):
        input1 = np.r_[1:105]
        input2 = self.ofdm_object._prepare_input_signal(input1)
        output = self.ofdm_object._prepare_decoded_signal(input2)
        np.testing.assert_array_equal(output, input1)

    def test_modulate(self):
        input_signal = np.r_[1:105]  # Exactly two OFDM symbols (with 52
                                     # used subcarriers)

        # xxxxx First lets try without cyclic prefix xxxxxxxxxxxxxxxxxxxxxx
        self.ofdm_object.set_parameters(64, 0, 52)
        (zeropad, num_ofdm_symbols) = self.ofdm_object._calc_zeropad(
            input_signal.size)

        self.assertEqual(zeropad, 0)

        # But we still have only one OFDM symbol
        self.assertEqual(num_ofdm_symbols, 2)

        input_ifft = self.ofdm_object._prepare_input_signal(input_signal)
        expected_data = np.hstack([
            np.fft.ifft(input_ifft[0, :]),
            np.fft.ifft(input_ifft[1, :]),
        ])

        np.testing.assert_array_almost_equal(
            self.ofdm_object.modulate(input_signal), expected_data)

        # Lets test each OFDM symbol individually
        np.testing.assert_array_almost_equal(
            self.ofdm_object.modulate(input_signal[0:52]),
            expected_data[0:64])

        np.testing.assert_array_almost_equal(
            self.ofdm_object.modulate(input_signal[52:]), expected_data[64:])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Now lets test with a cyclic prefix xxxxxxxxxxxxxxxxxxxxxxxx
        self.ofdm_object.set_parameters(64, 4, 52)
        expected_data2 = expected_data[0:64]
        expected_data2 = np.hstack(
            [expected_data2[-self.ofdm_object.cp_size:], expected_data[0:64]])
        np.testing.assert_array_almost_equal(
            self.ofdm_object.modulate(input_signal[0:52]), expected_data2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_demodulate(self):
        # xxxxx First lets try without cyclic prefix xxxxxxxxxxxxxxxxxxxxxx
        input_signal = np.r_[1:105]  # Exactly two OFDM symbols (with 52
                                     # used subcarriers)

        # xxxxx First lets try without cyclic prefix xxxxxxxxxxxxxxxxxxxxxx
        self.ofdm_object.set_parameters(64, 0, 52)
        modulated_ofdm_symbols = self.ofdm_object.modulate(input_signal)

        demodulated_symbols = self.ofdm_object.demodulate(
            modulated_ofdm_symbols)

        np.testing.assert_array_equal(
            np.real(demodulated_symbols.round()).astype(int),
            input_signal
        )
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Now lets test with a cyclic prefix xxxxxxxxxxxxxxxxxxxxxxxx
        input_signal2 = np.r_[1:105]  # Exactly two OFDM symbols (with 52
                                      # used subcarriers)
        self.ofdm_object.set_parameters(64, 16, 52)
        modulated_ofdm_symbols2 = self.ofdm_object.modulate(input_signal2)

        demodulated_symbols2 = self.ofdm_object.demodulate(
            modulated_ofdm_symbols2)

        np.testing.assert_array_equal(
            np.real(demodulated_symbols2.round()).astype(int),
            input_signal2
        )
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Now lets test the case with zeropadding xxxxxxxxxxxxxxxxxxx
        input_signal3 = np.r_[1:110]  # Exactly two OFDM symbols (with 52
                                      # used subcarriers)
        self.ofdm_object.set_parameters(64, 16, 52)
        modulated_ofdm_symbols3 = self.ofdm_object.modulate(input_signal3)

        demodulated_symbols3 = self.ofdm_object.demodulate(
            modulated_ofdm_symbols3)
        # OFDM will not remove the zeropadding therefore we need to do it
        # manually
        demodulated_symbols3 = demodulated_symbols3[0:109]
        np.testing.assert_array_equal(
            np.real(demodulated_symbols3.round()).astype(int),
            input_signal3
        )
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Power Specral Density
def plot_psd_OFDM_symbols():  # pragma: no cover
    """Plot the power spectral density of OFDM modulated symbols.

    This function is not used in any unittest, but it is interesting to
    visualize that the modulate method of the OFDM class is working as it
    should.
    """
    # xxxxxxxxxx OFDM Details xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    fft_size = 64
    cp_size = 12
    num_used_subcarriers = 52
    ofdm_object = ofdm.OFDM(fft_size, cp_size, num_used_subcarriers)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Input generation (not part of OFDM) xxxxxxxxxxxxxxxxxxxxxx
    num_bits = 2500
    # generating 1's and 0's
    ip_bits = np.random.random_integers(0, 1, num_bits)
    # Number of modulated symbols
    #num_mod_symbols = num_bits * 1
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # BPSK modulation
    # bit0 --> -1
    # bit1 --> +1
    ip_mod = 2 * ip_bits - 1

    # OFDM Modulation
    output = ofdm_object.modulate(ip_mod)

    # xxxxxxxxxx Plot the PSD xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # MATLAB code to plot the power spectral density
    # close all
    fsMHz = 20e6
    Pxx, W = pylab.psd(output, NFFT=fft_size, Fs=fsMHz)
    # [Pxx,W] = pwelch(output,[],[],4096,20);
    plt.plot(
        W,
        10 * np.log10(Pxx)
    )
    plt.xlabel('frequency, MHz')
    plt.ylabel('power spectral density')
    plt.title('Transmit spectrum OFDM (based on 802.11a)')
    plt.show()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MIMO Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MimoBaseTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.mimo_base = mimo.MimoBase()

    def test_dummy(self):
        pass


class BlastTestCase(unittest.TestCase):
    """Unittests for the Blast class in the mimo module.
    """
    def setUp(self):
        """Called before each test."""
        self.blast_object = Blast(3)

    def test_getNumberOfLayers(self):
        self.assertEqual(self.blast_object.getNumberOfLayers(), 3)
        blast2 = Blast(5)
        self.assertEqual(blast2.getNumberOfLayers(), 5)

    def test_encode(self):
        # Test if an exception is raised when the number of input symbols
        # is not multiple of the number of transmit antennas
        data = np.r_[0:14]
        with self.assertRaises(ValueError):
            self.blast_object.encode(data)

        # Test if the data is encoded correctly
        data = np.r_[0:15]
        expected_encoded_data = data.reshape(3, 5, order='F') / np.sqrt(3)
        np.testing.assert_array_almost_equal(
            self.blast_object.encode(data),
            expected_encoded_data)

    def test_decode(self):
        data = np.r_[0:15]
        encoded_data = self.blast_object.encode(data)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test with an identity channel
        channel = np.eye(3)
        decoded_data1 = self.blast_object.decode(encoded_data, channel)
        np.testing.assert_array_almost_equal(decoded_data1, data)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test with a random channel and a zero-force filter
        self.blast_object.set_noise_var(-1)  # This should use the ZF filter
        self.assertEqual(self.blast_object.calc_filter, mimo.MimoBase._calcZeroForceFilter)
        channel = randn_c(4, 3)  # 3 transmitt antennas and 4 receive antennas
        received_data2 = np.dot(channel, encoded_data)
        decoded_data2 = self.blast_object.decode(received_data2, channel)
        np.testing.assert_array_almost_equal(decoded_data2, data)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test with a random channel and a MMSE filter
        self.blast_object.set_noise_var(0.00000001)
        self.assertNotEqual(self.blast_object.calc_filter, mimo.MimoBase._calcMMSEFilter)
        channel = randn_c(4, 3)  # 3 transmitt antennas and 4 receive antennas
        received_data3 = np.dot(channel, encoded_data)
        decoded_data3 = self.blast_object.decode(received_data3, channel)
        np.testing.assert_array_almost_equal(decoded_data3.round(7), data)


# Implement test classes for other mimo schemes
class AlamoutiTestCase(unittest.TestCase):
    """Unittests for the Alamouti class in the mimo module.
    """

    def setUp(self):
        """Called before each test."""
        self.alamouti_object = Alamouti()

    def test_getNumberOfLayers(self):
        # The number of layers in the Alamouti scheme is always equal to
        # one.
        self.assertEqual(self.alamouti_object.getNumberOfLayers(), 1)

    def test_encode(self):
        data = np.r_[0:16] + np.r_[0:16] * 1j

        expected_encoded_data = np.array(
            [[0 + 0j, -1 + 1j, 2 + 2j, -3 + 3j, 4 + 4j, -5 + 5j, 6 + 6j,
              -7 + 7j, 8 + 8j, -9 + 9j, 10 + 10j, -11 + 11j, 12 + 12j,
              -13 + 13j, 14 + 14j, -15 + 15j],
             [1 + 1j, 0 - 0j, 3 + 3j, 2 - 2j, 5 + 5j, 4 - 4j, 7 + 7j,
              6 - 6j, 9 + 9j, 8 - 8j, 11 + 11j, 10 - 10j, 13 + 13j, 12 - 12j,
              15 + 15j, 14 - 14j]]
        ) / np.sqrt(2)

        np.testing.assert_array_almost_equal(
            self.alamouti_object.encode(data),
            expected_encoded_data)

    def test_decode(self):
        data = np.r_[0:16] + np.r_[0:16] * 1j
        encoded_data = self.alamouti_object.encode(data)
        # We will test the deconding with a random channel
        channel = randn_c(3, 2)
        received_data = np.dot(channel, encoded_data)
        decoded_data = self.alamouti_object.decode(received_data, channel)
        np.testing.assert_array_almost_equal(decoded_data, data)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Waterfilling Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class WaterfillingTestCase(unittest.TestCase):
    """Unittests for the waterfilling module.

    """
    def test_doWF(self):
        """
        - `vtChannels`: Numpy array with the channel POWER gains (power of the
        parallel AWGN channels).
        - `dPt`: Total available power.
        - `noiseVar`: Noise variance (power in linear scale)
        - `Es`: Symbol energy (in linear scale)
        """
        # See the link below this example
        #http://jungwon.comoj.com/ucsd_ece287b_spr12/lecture_slides/lecture4.pdf

        # Abs of the parallel channels
        vtChannels_abs = np.array([1.90, 1.76, 1.76, 1.35, 1.35, .733, .733,
                                   .100])
        # Power of the parallel channels
        channel_power_gains = vtChannels_abs ** 2

        # Total power available to be distributed
        total_power = 8.
        noise_var = 0.181

        # Calculates the optimum powers and water level
        (vtOptP, mu) = waterfilling.doWF(
            channel_power_gains, total_power, noise_var)

        # The sum of the powers in each channel must be equal to the
        # total_power
        self.assertAlmostEqual(np.sum(vtOptP), total_power)

        # Test the water level
        expected_mu = 1.29134061296
        self.assertAlmostEqual(mu, expected_mu)

        # test the powers in each channel
        expected_vtOptP = np.array([1.24120211, 1.23290828, 1.23290828,
                                    1.19202648, 1.19202648, 0.95446418,
                                    0.95446418, 0.])
        np.testing.assert_array_almost_equal(vtOptP, expected_vtOptP)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Block Diagonalization Module xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class BlockDiaginalizerTestCase(unittest.TestCase):
    """Unittests for the BlockDiaginalizer class in the blockdiagonalization
    module.

    """
    def setUp(self):
        """Called before each test."""
        self.Pu = 5.  # Power for each user
        self.noise_var = 1e-6
        self.num_users = 3
        self.num_antenas = 2

        self.iNrk = self.num_antenas  # Number of receive antennas per user
        self.iNtk = self.num_antenas  # Number of transmit antennas per user

        self.iNr = self.iNrk * self.num_users  # Total number of Rx antennas
        self.iNt = self.iNtk * self.num_users  # Total number of Tx antennas

        self.BD = blockdiagonalization.BlockDiaginalizer(
            self.num_users,
            self.Pu,
            self.noise_var)

    def test_calc_BD_matrix_no_power_scaling(self):
        channel = randn_c(self.iNr, self.iNt)
        (Ms_bad, _) = self.BD._calc_BD_matrix_no_power_scaling(channel)

        newH = np.dot(channel, Ms_bad)

        # Because Ms_bad does not have any power scaling and each column of
        # it comes is a singular vector calculated with the SVD, then the
        # square of its Frobenius norm is equal to its dimension.
        self.assertAlmostEqual(np.linalg.norm(Ms_bad, 'fro') ** 2,
                               Ms_bad.shape[0])

        # Now let's test if newH is really a block diagonal matrix.
        # First we create a 'mask'
        A = np.ones([self.iNrk, self.iNtk])
        mask = block_diag(A, A, A)

        # With the mask we can create a masked array of the block
        # diagonalized channel which effectively removes the elements in
        # the block diagonal
        masked_newH = np.ma.masked_array(newH, mask)

        # If we sum all the elements in the masked channel (the mask
        # removes the elements in the block diagonal) it should be equal to
        # zero
        self.assertAlmostEqual(0., np.abs(masked_newH).sum())

    def test_perform_global_waterfilling_power_scaling(self):
        channel = randn_c(self.iNr, self.iNt)
        (Ms_bad, Sigma) = self.BD._calc_BD_matrix_no_power_scaling(channel)

        Ms_good = self.BD._perform_global_waterfilling_power_scaling(
            Ms_bad, Sigma)

        # Ms_good must have the same shape as Ms_bad
        np.testing.assert_array_equal(Ms_bad.shape, Ms_good.shape)

        # The total available power is equal to the power per user times
        # the number of users
        total_power = self.Pu * self.num_users

        # The square of the Frobenius norm of Ms_good must be equal to the
        # total available power
        self.assertAlmostEqual(np.linalg.norm(Ms_good, 'fro') ** 2,
                               total_power)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Ms_good must still be able to block diagonalize the channel
        A = np.ones([self.iNrk, self.iNtk])
        mask = block_diag(A, A, A)
        newH = np.dot(channel, Ms_good)

        # With the mask we can create a masked array of the block
        # diagonalized channel
        masked_newH = np.ma.masked_array(newH, mask)
        # Now we can sum all elements of this masked array (which
        # effectively means all elements outside the block diagonal) and
        # see if it is close to zero.
        self.assertAlmostEqual(0., np.abs(masked_newH).sum())

    def test_perform_normalized_waterfilling_power_scaling(self):
        channel = randn_c(self.iNr, self.iNt)
        (Ms_bad, Sigma) = self.BD._calc_BD_matrix_no_power_scaling(channel)

        Ms_good = self.BD._perform_normalized_waterfilling_power_scaling(
            Ms_bad, Sigma)
        # Ms_good = self.BD._perform_global_waterfilling_power_scaling(
        #     Ms_bad, Sigma)

        # xxxxx Now lets test the power restriction xxxxxxxxxxxxxxxxxxxxxxx
        # Total power restriction
        total_power = self.Pu * self.num_users
        self.assertGreaterEqual(total_power,
                                np.linalg.norm(Ms_good, 'fro') ** 2)

        # xxxxx Test the Individual power restriction of each user xxxxxxxx
        # Cummulated number of transmit antennas
        cum_Nt = np.cumsum(
            np.hstack([0, np.ones(self.num_users, dtype=int) * self.num_antenas]))

        individual_powers = []
        for i in range(self.num_users):
            # Most likelly only one base station (the one with the worst
            # channel) will employ a precoder with total power of `Pu`,
            # while the other base stations will use less power.
            individual_powers.append(np.linalg.norm(Ms_good[:, cum_Nt[i]:cum_Nt[i] + self.num_antenas], 'fro') ** 2)
            # 1e-12 is included to avoid false test fails due to small
            # precision errors
            tol = 1e-12
            self.assertGreaterEqual(self.Pu + tol,
                                    individual_powers[-1])

    def test_block_diagonalize(self):
        Pu = self.Pu
        noise_var = self.noise_var
        num_users = self.num_users
        num_antenas = self.num_antenas

        channel = randn_c(self.iNr, self.iNt)
        (newH, Ms) = blockdiagonalization.block_diagonalize(
            channel, num_users, Pu, noise_var)

        # xxxxx Test if the channel is really block diagonal xxxxxxxxxxxxxx
        # First we build a 'mask' to filter out the elements in the block
        # diagonal.

        A = np.ones([self.iNrk, self.iNtk])
        mask = block_diag(A, A, A)

        # With the mask we can create a masked array of the block
        # diagonalized channel
        masked_newH = np.ma.masked_array(newH, mask)
        # Now we can sum all elements of this masked array (which
        # effectively means all elements outside the block diagonal) and
        # see if it is close to zero.
        self.assertAlmostEqual(0., np.abs(masked_newH).sum())

        # xxxxx Now lets test the power restriction xxxxxxxxxxxxxxxxxxxxxxx
        # Total power restriction
        total_power = num_users * Pu
        self.assertGreaterEqual(total_power,
                                np.linalg.norm(Ms, 'fro') ** 2)

        # Cummulated number of receive antennas
        cum_Nt = np.cumsum(
            np.hstack([0, np.ones(num_users, dtype=int) * num_antenas]))

        # Individual power restriction of each class
        individual_powers = []
        tol = 1e-12  # Tolerance for the GreaterEqual test
        for i in range(num_users):
            # Most likelly only one base station (the one with the worst
            # channel) will employ a precoder a precoder with total power
            # of `Pu`, while the other base stations will use less power.
            individual_powers.append(np.linalg.norm(Ms[:, cum_Nt[i]:cum_Nt[i] + num_antenas], 'fro') ** 2)
            self.assertGreaterEqual(Pu + tol,
                                    individual_powers[-1])

    def test_block_diagonalize_no_waterfilling(self):
        Pu = self.Pu
        num_users = self.num_users
        num_antenas = self.num_antenas

        channel = randn_c(self.iNr, self.iNt)
        (newH, Ms) = self.BD.block_diagonalize_no_waterfilling(channel)

        # xxxxx Test if the channel is really block diagonal xxxxxxxxxxxxxx
        # First we build a 'mask' to filter out the elements in the block
        # diagonal.

        A = np.ones([self.iNrk, self.iNtk])
        mask = block_diag(A, A, A)

        # With the mask we can create a masked array of the block
        # diagonalized channel
        masked_newH = np.ma.masked_array(newH, mask)
        # Now we can sum all elements of this masked array (which
        # effectively means all elements outside the block diagonal) and
        # see if it is close to zero.
        self.assertAlmostEqual(0., np.abs(masked_newH).sum())

        # xxxxx Now lets test the power restriction xxxxxxxxxxxxxxxxxxxxxxx
        tol = 1e-12  # Tolerance for the GreaterEqual test
        # Total power restriction
        total_power = num_users * Pu
        self.assertGreaterEqual(total_power + tol,
                                np.linalg.norm(Ms, 'fro') ** 2)

        # Cummulated number of receive antennas
        cum_Nt = np.cumsum(
            np.hstack([0, np.ones(num_users, dtype=int) * num_antenas]))

        # Individual power restriction of each class
        individual_powers = []
        for i in range(num_users):
            # Most likelly only one base station (the one with the worst
            # channel) will employ a precoder a precoder with total power
            # of `Pu`, while the other base stations will use less power.
            individual_powers.append(np.linalg.norm(Ms[:, cum_Nt[i]:cum_Nt[i] + num_antenas], 'fro') ** 2)
            self.assertGreaterEqual(Pu + tol,
                                    individual_powers[-1])

    def test_calc_receive_filter(self):
        Pu = self.Pu
        noise_var = self.noise_var
        num_users = self.num_users
        #num_antenas = self.num_antenas
        channel = randn_c(self.iNr, self.iNt)

        (newH, _) = blockdiagonalization.block_diagonalize(
            channel, num_users, Pu, noise_var)

        # W_bd is a block diagonal matrix, where each "small block" is the
        # receive filter of one user.
        W_bd = blockdiagonalization.calc_receive_filter(newH)

        np.testing.assert_array_almost_equal(np.dot(W_bd, newH),
                                             np.eye(np.sum(self.iNt)))

        # Retest for each individual user
        W0 = W_bd[0:2, 0:2]
        newH0 = newH[0:2, 0:2]
        np.testing.assert_array_almost_equal(np.dot(W0, newH0),
                                             np.eye(self.iNt/3))
        W1 = W_bd[2:4, 2:4]
        newH1 = newH[2:4, 2:4]
        np.testing.assert_array_almost_equal(np.dot(W1, newH1),
                                             np.eye(self.iNt/3))
        W2 = W_bd[4:, 4:]
        newH2 = newH[4:, 4:]
        np.testing.assert_array_almost_equal(np.dot(W2, newH2),
                                             np.eye(self.iNt/3))


# TODO: finish implementation
class BDWithExtIntBaseTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_calc_whitening_matrices(self):
        Nr = np.array([2, 2])
        Nt = np.array([2, 2])
        K = Nt.size
        Nti = 1
        iPu = 1e-1  # Power for each user (linear scale)
        pe = 1e-3  # External interference power (in linear scale)
        noise_var = 1e-4

        # Generate the multi-user channel
        mu_channel = channels.MultiUserChannelMatrixExtInt()
        mu_channel.randomize(Nr, Nt, K, Nti)

        bd_obj = blockdiagonalization.BDWithExtIntBase(K, iPu, noise_var, pe)
        W_all_k = bd_obj.calc_whitening_matrices(mu_channel, noise_var)

        R_all_k = mu_channel.calc_cov_matrix_extint_plus_noise(noise_var, pe)
        for W, R in zip(W_all_k, R_all_k):
            np.testing.assert_array_almost_equal(
                W,
                calc_whitening_matrix(R).conjugate().T)


# TODO: finish implementation
class WhiteningBDTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_block_diagonalize_no_waterfilling(self):
        Nr = np.array([2, 2])
        Nt = np.array([2, 2])
        K = Nt.size
        Nti = 1
        iPu = 1e-1  # Power for each user (linear scale)
        pe = 1e-3  # External interference power (in linear scale)
        noise_var = 1e-4

        # # The modulator and packet_length are required in the
        # # effective_throughput metric case
        # psk_obj = modulators.PSK(4)
        # packet_length = 120

        multiUserChannel = channels.MultiUserChannelMatrixExtInt()
        multiUserChannel.randomize(Nr, Nt, K, Nti)

        # Channel from all transmitters to the first receiver
        H1 = multiUserChannel.get_Hk_without_ext_int(0)
        # Channel from all transmitters to the second receiver
        H2 = multiUserChannel.get_Hk_without_ext_int(1)

        # Create the whiteningBD object and the regular BD object
        whiteningBD_obj = blockdiagonalization.WhiteningBD(K, iPu, noise_var, pe)

        #noise_plus_int_cov_matrix = multiUserChannel.calc_cov_matrix_extint_plus_noise(noise_var, pe)

        #xxxxx First we test without ext. int. handling xxxxxxxxxxxxxxxxxxx
        (Ms_all, Wk_all, Ns_all) = whiteningBD_obj.block_diagonalize_no_waterfilling(multiUserChannel)
        Ms1 = Ms_all[0]
        Ms2 = Ms_all[1]

        self.assertEqual(Ms1.shape[1], Ns_all[0])
        self.assertEqual(Ms2.shape[1], Ns_all[1])

        # Most likelly only one base station (the one with the worst
        # channel) will employ a precoder with total power of `Pu`,
        # while the other base stations will use less power.
        tol = 1e-10
        self.assertGreaterEqual(iPu + tol,
                                np.linalg.norm(Ms1, 'fro') ** 2)
        # 1e-12 is included to avoid false test fails due to small
        # precision errors
        self.assertGreaterEqual(iPu + tol,
                                np.linalg.norm(Ms2, 'fro') ** 2)

        # Test if the precoder block diagonalizes the channel
        self.assertNotAlmostEqual(np.linalg.norm(np.dot(H1, Ms1), 'fro'), 0)
        self.assertAlmostEqual(np.linalg.norm(np.dot(H1, Ms2), 'fro'), 0)
        self.assertNotAlmostEqual(np.linalg.norm(np.dot(H2, Ms2), 'fro'), 0)
        self.assertAlmostEqual(np.linalg.norm(np.dot(H2, Ms1), 'fro'), 0)


        # # xxxxxxxxxx Now lets test the receive filter xxxxxxxxxxxxxxxxxxxxx
        # print
        # #print Wk_all
        # np.set_printoptions(precision=4, linewidth=100)

        # print np.dot(H1, Ms1)
        # print
        # print np.dot(H2, Ms2)
        # print

        # print "big_H"
        # big_H = multiUserChannel.big_H_no_ext_int
        # print big_H

        # # print Ms_all[0]
        # # print
        # # print Ms_all[1]
        # # print

        # print "Ms"
        # Ms = np.hstack(Ms_all)
        # print Ms

        # print

        # print "Multiplication"
        # print np.dot(big_H, Ms).round(6)
        # print

        # Wk = block_diag(*Wk_all)
        # print "Wk"
        # print Wk
        # print

        # print "Final"
        # print np.dot(Wk, np.dot(big_H, Ms)).round(4)
        # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx



        # # Equivalent sinrs (in linear scale)
        # sinrs = np.empty(K, dtype=np.ndarray)
        # sinrs[0] = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
        #     np.dot(H1, Ms1),
        #     Wk_all[0],
        #     noise_plus_int_cov_matrix[0])
        # sinrs[1] = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
        #     np.dot(H2, Ms2),
        #     Wk_all[1],
        #     noise_plus_int_cov_matrix[1])

        # # Spectral efficiency
        # se = (np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
        #     linear2dB(sinrs[0]),
        #     packet_length))
        #     +
        #     np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
        #         linear2dB(sinrs[1]),
        #         packet_length)))
        # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # # xxxxxxxxxx For comparison, lets perform the regular BD xxxxxxxxxx
        # (newH, Ms_good_regular_bd) = regularBD_obj.block_diagonalize_no_waterfilling(multiUserChannel.big_H_no_ext_int)
        # Wk_all_regular_bd = regularBD_obj.calc_receive_filter(newH)
        # regularBD_obj._calc_linear_SINRs
        # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx





        # # Test if the effective_throughput obtains a better spectral
        # # efficiency then the capacity and not handling interference.
        # self.assertGreater(se3 + tol, se2)
        # self.assertGreater(se3 + tol, se)

        # # Note that almost always the capacity criterion will achieve a
        # # better spectral efficiency then not handling
        # # interference. However, sometimes it can get a worse spectral
        # # efficiency. We are not testing this here.


# TODO: finish implementation
class EnhancedBDTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_set_ext_int_handling_metric(self):
        K = 3
        iPu = 1e-3  # Power for each user (linear scale)
        noise_var = 1e-4
        pe = 0

        # Create the EnhancedBD object
        enhancedBD_obj = blockdiagonalization.EnhancedBD(K, iPu, noise_var, pe)

        # xxxxx Test if an assert is raised for invalid arguments xxxxxxxxx
        with self.assertRaises(AttributeError):
            enhancedBD_obj.set_ext_int_handling_metric('lala')

        with self.assertRaises(AttributeError):
            # If we set the metric to effective_throughput but not provide
            # the modulator and packet_length attributes.
            enhancedBD_obj.set_ext_int_handling_metric('effective_throughput')

        with self.assertRaises(AttributeError):
            enhancedBD_obj.set_ext_int_handling_metric('naive')
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test setting the metric to effective_throughput xxxxxxxxxxx
        psk_obj = modulators.PSK(4)
        enhancedBD_obj.set_ext_int_handling_metric('effective_throughput',
                                                   {'modulator': psk_obj,
                                                    'packet_length': 120})
        self.assertEqual(enhancedBD_obj._metric_func,
                         blockdiagonalization._calc_effective_throughput)
        self.assertEqual(enhancedBD_obj.metric_name, "effective_throughput")

        metric_func_extra_args = enhancedBD_obj._metric_func_extra_args
        self.assertEqual(metric_func_extra_args['modulator'], psk_obj)
        self.assertEqual(metric_func_extra_args['packet_length'], 120)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test setting the metric to capacity xxxxxxxxxxxxxxxxxxxxxxx
        enhancedBD_obj.set_ext_int_handling_metric('capacity')
        self.assertEqual(enhancedBD_obj._metric_func,
                         calc_shannon_sum_capacity)
        self.assertEqual(enhancedBD_obj.metric_name, "capacity")
        # metric_func_extra_args is an empty dictionary for the capacity
        # metric
        self.assertEqual(enhancedBD_obj._metric_func_extra_args, {})
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test setting the metric to None xxxxxxxxxxxxxxxxxxxxxxxxxxx
        enhancedBD_obj.set_ext_int_handling_metric(None)
        self.assertIsNone(enhancedBD_obj._metric_func)
        self.assertEqual(enhancedBD_obj.metric_name, "None")

        # metric_func_extra_args is an empty dictionary for the None metric
        self.assertEqual(enhancedBD_obj._metric_func_extra_args, {})

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test setting the metric to naive xxxxxxxxxxxxxxxxxxxxxxxxxx
        enhancedBD_obj.set_ext_int_handling_metric('naive',
                                                   {'num_streams': 2})
        self.assertIsNone(enhancedBD_obj._metric_func)
        self.assertEqual(enhancedBD_obj.metric_name, "naive")

        metric_func_extra_args = enhancedBD_obj._metric_func_extra_args
        self.assertEqual(metric_func_extra_args['num_streams'], 2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_receive_filter(self):
        # Equivalent channel without including stream reduction
        Heq_k = randn_c(3, 3)
        Re_k = randn_c(3, 2)
        Re_k = np.dot(Re_k, Re_k.transpose().conjugate())

        P1 = blockdiagonalization._calc_stream_reduction_matrix(Re_k, 1)
        P2 = blockdiagonalization._calc_stream_reduction_matrix(Re_k, 2)
        P3 = blockdiagonalization._calc_stream_reduction_matrix(Re_k, 3)

        # Equivalent channels with the stream reduction
        Heq_k_P1 = np.dot(Heq_k, P1)
        Heq_k_P2 = np.dot(Heq_k, P2)
        Heq_k_P3 = np.dot(Heq_k, P3)

        W1 = blockdiagonalization.EnhancedBD.calc_receive_filter_user_k(Heq_k_P1, P1)
        W2 = blockdiagonalization.EnhancedBD.calc_receive_filter_user_k(Heq_k_P2, P2)
        W3 = blockdiagonalization.EnhancedBD.calc_receive_filter_user_k(Heq_k_P3, P3)
        # Note that since P3 is actually including all streams, then the
        # performance is the same as if we don't reduce streams. However W3
        # and W_full are different matrices, since W3 has to compensate the
        # right multiplication of the equivalent channel by P3 and W_full
        # does not. The performance is the same because no energy is lost
        # due to stream reduction and the Frobenius norms of W3 and W_full
        # are equal.
        W_full = blockdiagonalization.EnhancedBD.calc_receive_filter_user_k(Heq_k)

        np.testing.assert_array_almost_equal(np.dot(W1, np.dot(Heq_k, P1)),
                                             np.eye(1))
        np.testing.assert_array_almost_equal(np.dot(W2, np.dot(Heq_k, P2)),
                                             np.eye(2))
        np.testing.assert_array_almost_equal(np.dot(W3, np.dot(Heq_k, P3)),
                                             np.eye(3))
        np.testing.assert_array_almost_equal(np.dot(W_full, Heq_k),
                                             np.eye(3))

        overbar_P2 = calcProjectionMatrix(P2)
        expected_W2 = np.dot(
            np.linalg.pinv(np.dot(overbar_P2, np.dot(Heq_k, P2))),
            overbar_P2)
        np.testing.assert_array_almost_equal(expected_W2, W2)

    # TODO: Implement-me
    def test_calc_linear_SINRs(self):
        # Heq_k_red = np.array([[2, 2], [1, 2]])
        # # Usually this will be the inverse of Heq_k_red, but for testing
        # # purposes we can specify a different Wk
        # Wk = np.array([[1, 1], [1.5, 0.5]])
        # Rk = np.array([[0.5, 0.2], [0.25, 0.1]])
        # SINRs = blockdiagonalization.EnhancedBD._calc_linear_SINRs(Heq_k_red, Wk, Rk)
        # print SINRs
        pass

    def test_calc_effective_throughput(self):
        psk_obj = modulators.PSK(8)
        packet_length = 60

        SINRs_dB = np.array([11.4, 20.3])
        sinrs_linear = dB2Linear(SINRs_dB)

        expected_spectral_efficiency = np.sum(
            psk_obj.calcTheoreticalSpectralEfficiency(SINRs_dB, packet_length))

        spectral_efficiency = blockdiagonalization._calc_effective_throughput(
            sinrs_linear, psk_obj, packet_length)

        np.testing.assert_array_almost_equal(spectral_efficiency,
                                             expected_spectral_efficiency)

    def test_block_diagonalize_no_waterfilling(self):
        Nr = np.array([2, 2])
        Nt = np.array([2, 2])
        K = Nt.size
        Nti = 1
        iPu = 1e-1  # Power for each user (linear scale)
        pe = 1e-3  # External interference power (in linear scale)
        noise_var = 1e-4

        # The modulator and packet_length are required in the
        # effective_throughput metric case
        psk_obj = modulators.PSK(4)
        packet_length = 120

        multiUserChannel = channels.MultiUserChannelMatrixExtInt()
        multiUserChannel.randomize(Nr, Nt, K, Nti)

        # Channel from all transmitters to the first receiver
        H1 = multiUserChannel.get_Hk_without_ext_int(0)
        # Channel from all transmitters to the second receiver
        H2 = multiUserChannel.get_Hk_without_ext_int(1)

        # Create the enhancedBD object
        enhancedBD_obj = blockdiagonalization.EnhancedBD(K, iPu, noise_var, pe)

        noise_plus_int_cov_matrix = multiUserChannel.calc_cov_matrix_extint_plus_noise(noise_var, pe)

        #xxxxx First we test without ext. int. handling xxxxxxxxxxxxxxxxxxx
        enhancedBD_obj.set_ext_int_handling_metric(None)
        (Ms_all, Wk_all, Ns_all) = enhancedBD_obj.block_diagonalize_no_waterfilling(multiUserChannel)
        Ms1 = Ms_all[0]
        Ms2 = Ms_all[1]

        self.assertEqual(Ms1.shape[1], Ns_all[0])
        self.assertEqual(Ms2.shape[1], Ns_all[1])

        # Most likelly only one base station (the one with the worst
        # channel) will employ a precoder with total power of `Pu`,
        # while the other base stations will use less power.
        tol = 1e-10
        self.assertGreaterEqual(iPu + tol,
                                np.linalg.norm(Ms1, 'fro') ** 2)
        # 1e-12 is included to avoid false test fails due to small
        # precision errors
        self.assertGreaterEqual(iPu + tol,
                                np.linalg.norm(Ms2, 'fro') ** 2)

        # Test if the precoder block diagonalizes the channel
        self.assertNotAlmostEqual(np.linalg.norm(np.dot(H1, Ms1), 'fro'), 0)
        self.assertAlmostEqual(np.linalg.norm(np.dot(H1, Ms2), 'fro'), 0)
        self.assertNotAlmostEqual(np.linalg.norm(np.dot(H2, Ms2), 'fro'), 0)
        self.assertAlmostEqual(np.linalg.norm(np.dot(H2, Ms1), 'fro'), 0)

        # Equivalent sinrs (in linear scale)
        sinrs = np.empty(K, dtype=np.ndarray)
        sinrs[0] = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
            np.dot(H1, Ms1),
            Wk_all[0],
            noise_plus_int_cov_matrix[0])
        sinrs[1] = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
            np.dot(H2, Ms2),
            Wk_all[1],
            noise_plus_int_cov_matrix[1])

        # Spectral efficiency
        se = (
            np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
                linear2dB(sinrs[0]),
                packet_length))
            +
            np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
                linear2dB(sinrs[1]),
                packet_length)))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Now with the Naive Stream Reduction xxxxxxxxxxxxxxxxxxxxxxx
        num_streams = 1
        enhancedBD_obj.set_ext_int_handling_metric(
            'naive',
            {'num_streams': num_streams})

        (MsPk_naive_all, Wk_naive_all, Ns_naive_all) = enhancedBD_obj.block_diagonalize_no_waterfilling(multiUserChannel)
        MsPk_naive_1 = MsPk_naive_all[0]
        MsPk_naive_2 = MsPk_naive_all[1]

        self.assertEqual(MsPk_naive_1.shape[1], Ns_naive_all[0])
        self.assertEqual(MsPk_naive_2.shape[1], Ns_naive_all[1])
        self.assertEqual(Ns_naive_all[0], num_streams)
        self.assertEqual(Ns_naive_all[1], num_streams)

        # Test if the square of the Frobenius norm of the precoder of each
        # user is equal to the power available to that user.
        self.assertAlmostEqual(iPu, np.linalg.norm(MsPk_naive_1, 'fro') ** 2)
        self.assertAlmostEqual(iPu, np.linalg.norm(MsPk_naive_2, 'fro') ** 2)

        # Test if MsPk really block diagonalizes the channel
        self.assertNotAlmostEqual(
            np.linalg.norm(np.dot(H1, MsPk_naive_1), 'fro'),
            0)
        self.assertAlmostEqual(
            np.linalg.norm(np.dot(H1, MsPk_naive_2), 'fro'),
            0)
        self.assertNotAlmostEqual(
            np.linalg.norm(np.dot(H2, MsPk_naive_2), 'fro'),
            0)
        self.assertAlmostEqual(
            np.linalg.norm(np.dot(H2, MsPk_naive_1), 'fro'),
            0)

        sinrs4 = np.empty(K, dtype=np.ndarray)
        sinrs4[0] = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
            np.dot(H1, MsPk_naive_1),
            Wk_naive_all[0],
            noise_plus_int_cov_matrix[0])
        sinrs4[1] = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
            np.dot(H2, MsPk_naive_2),
            Wk_naive_all[1],
            noise_plus_int_cov_matrix[1])

        # Spectral efficiency
        se4 = (
            np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
                linear2dB(sinrs4[0]),
                packet_length))
            +
            np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
                linear2dB(sinrs4[1]),
                packet_length)))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Now with the Fixed Stream Reduction xxxxxxxxxxxxxxxxxxxxxxx
        num_streams = 1
        enhancedBD_obj.set_ext_int_handling_metric(
            'fixed',
            {'num_streams': num_streams})

        (MsPk_fixed_all, Wk_fixed_all, Ns_fixed_all) = enhancedBD_obj.block_diagonalize_no_waterfilling(multiUserChannel)
        MsPk_fixed_1 = MsPk_fixed_all[0]
        MsPk_fixed_2 = MsPk_fixed_all[1]

        self.assertEqual(MsPk_fixed_1.shape[1], Ns_fixed_all[0])
        self.assertEqual(MsPk_fixed_2.shape[1], Ns_fixed_all[1])
        self.assertEqual(Ns_fixed_all[0], num_streams)
        self.assertEqual(Ns_fixed_all[1], num_streams)

        # Test if the square of the Frobenius norm of the precoder of each
        # user is equal to the power available to that user.
        self.assertAlmostEqual(iPu, np.linalg.norm(MsPk_fixed_1, 'fro') ** 2)
        self.assertAlmostEqual(iPu, np.linalg.norm(MsPk_fixed_2, 'fro') ** 2)

        # Test if MsPk really block diagonalizes the channel
        self.assertNotAlmostEqual(
            np.linalg.norm(np.dot(H1, MsPk_fixed_1), 'fro'),
            0)
        self.assertAlmostEqual(
            np.linalg.norm(np.dot(H1, MsPk_fixed_2), 'fro'),
            0)
        self.assertNotAlmostEqual(
            np.linalg.norm(np.dot(H2, MsPk_fixed_2), 'fro'),
            0)
        self.assertAlmostEqual(
            np.linalg.norm(np.dot(H2, MsPk_fixed_1), 'fro'),
            0)

        sinrs5 = np.empty(K, dtype=np.ndarray)
        sinrs5[0] = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
            np.dot(H1, MsPk_fixed_1),
            Wk_fixed_all[0],
            noise_plus_int_cov_matrix[0])
        sinrs5[1] = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
            np.dot(H2, MsPk_fixed_2),
            Wk_fixed_all[1],
            noise_plus_int_cov_matrix[1])

        # Spectral efficiency
        se5 = (
            np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
                linear2dB(sinrs5[0]),
                packet_length))
            +
            np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
                linear2dB(sinrs5[1]),
                packet_length)))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Handling external interference xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Handling external interference using the capacity metric
        enhancedBD_obj.set_ext_int_handling_metric('capacity')
        (MsPk_all, Wk_cap_all, Ns_cap_all) = enhancedBD_obj.block_diagonalize_no_waterfilling(multiUserChannel)
        MsPk_cap_1 = MsPk_all[0]
        MsPk_cap_2 = MsPk_all[1]

        self.assertEqual(MsPk_cap_1.shape[1], Ns_cap_all[0])
        self.assertEqual(MsPk_cap_2.shape[1], Ns_cap_all[1])

        # Test if the square of the Frobenius norm of the precoder of each
        # user is equal to the power available to that user.
        self.assertAlmostEqual(iPu, np.linalg.norm(MsPk_cap_1, 'fro') ** 2)
        self.assertAlmostEqual(iPu, np.linalg.norm(MsPk_cap_2, 'fro') ** 2)

        # Test if MsPk really block diagonalizes the channel
        self.assertNotAlmostEqual(
            np.linalg.norm(np.dot(H1, MsPk_cap_1), 'fro'), 0)
        self.assertAlmostEqual(
            np.linalg.norm(np.dot(H1, MsPk_cap_2), 'fro'), 0)
        self.assertNotAlmostEqual(
            np.linalg.norm(np.dot(H2, MsPk_cap_2), 'fro'), 0)
        self.assertAlmostEqual(
            np.linalg.norm(np.dot(H2, MsPk_cap_1), 'fro'), 0)

        sinrs2 = np.empty(K, dtype=np.ndarray)
        sinrs2[0] = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
            np.dot(H1, MsPk_cap_1),
            Wk_cap_all[0],
            noise_plus_int_cov_matrix[0])
        sinrs2[1] = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
            np.dot(H2, MsPk_cap_2),
            Wk_cap_all[1],
            noise_plus_int_cov_matrix[1])

        # Spectral efficiency
        se2 = (
            np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
                linear2dB(sinrs2[0]),
                packet_length))
            +
            np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
                linear2dB(sinrs2[1]),
                packet_length)))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Handling external interference xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Handling external interference using the effective_throughput metric
        enhancedBD_obj.set_ext_int_handling_metric(
            'effective_throughput',
            {'modulator': psk_obj,
             'packet_length': packet_length})

        (MsPk_effec_all, Wk_effec_all, Ns_effec_all) = enhancedBD_obj.block_diagonalize_no_waterfilling(multiUserChannel)
        MsPk_effec_1 = MsPk_effec_all[0]
        MsPk_effec_2 = MsPk_effec_all[1]

        self.assertEqual(MsPk_effec_1.shape[1], Ns_effec_all[0])
        self.assertEqual(MsPk_effec_2.shape[1], Ns_effec_all[1])

        # Test if the square of the Frobenius norm of the precoder of each
        # user is equal to the power available to that user.
        self.assertAlmostEqual(iPu, np.linalg.norm(MsPk_effec_1, 'fro') ** 2)
        self.assertAlmostEqual(iPu, np.linalg.norm(MsPk_effec_2, 'fro') ** 2)

        # Test if MsPk really block diagonalizes the channel
        self.assertNotAlmostEqual(
            np.linalg.norm(np.dot(H1, MsPk_effec_1), 'fro'),
            0)
        self.assertAlmostEqual(
            np.linalg.norm(np.dot(H1, MsPk_effec_2), 'fro'),
            0)
        self.assertNotAlmostEqual(
            np.linalg.norm(np.dot(H2, MsPk_effec_2), 'fro'),
            0)
        self.assertAlmostEqual(
            np.linalg.norm(np.dot(H2, MsPk_effec_1), 'fro'),
            0)

        sinrs3 = np.empty(K, dtype=np.ndarray)
        sinrs3[0] = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
            np.dot(H1, MsPk_effec_1),
            Wk_effec_all[0],
            noise_plus_int_cov_matrix[0])
        sinrs3[1] = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
            np.dot(H2, MsPk_effec_2),
            Wk_effec_all[1],
            noise_plus_int_cov_matrix[1])

        # Spectral efficiency
        se3 = (
            np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
                linear2dB(sinrs3[0]),
                packet_length))
            +
            np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
                linear2dB(sinrs3[1]),
                packet_length)))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Test if the effective_throughput obtains a better spectral
        # efficiency then the capacity and not handling interference.
        self.assertGreater(se3 + tol, se2)
        self.assertGreater(se3 + tol, se)

        # Note that almost always the capacity criterion will achieve a
        # better spectral efficiency then not handling
        # interference. However, sometimes it can get a worse spectral
        # efficiency. We are not testing this here.


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Modulators Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class PSKTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.psk_obj = modulators.PSK(4)
        self.psk_obj2 = modulators.PSK(8)

    def test_constellation(self):
        self.assertEqual(self.psk_obj.M, 4)
        self.assertAlmostEqual(self.psk_obj.K, 2)
        np.testing.assert_array_almost_equal(
            self.psk_obj.symbols,
            np.array([1. + 0.j, 0. + 1.j, 0. - 1.j, -1. + 0.j]))

        self.assertEqual(self.psk_obj2.M, 8)
        self.assertAlmostEqual(self.psk_obj2.K, 3)
        np.testing.assert_array_almost_equal(
            self.psk_obj2.symbols,
            np.array([1. + 0.j, 0.70710678 + 0.70710678j,
                      -0.70710678 + 0.70710678j, 0. + 1.j,
                      0.70710678 - 0.70710678j, 0. - 1.j,
                      -1. + 0.j, -0.70710678 - 0.70710678j]))

    def test_set_phase_offset(self):
        self.psk_obj.setPhaseOffset(np.pi / 4.)

        np.testing.assert_array_almost_equal(
            self.psk_obj.symbols,
            np.array([0.70710678 + 0.70710678j, -0.70710678 + 0.70710678j,
                      -0.70710678 - 0.70710678j, 0.70710678 - 0.70710678j]))

    def test_calc_theoretical_SER_and_BER(self):
        SNR_values = np.array([-5, 0, 5, 10])

        # xxxxxxxxxx Test for the 4-PSK xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        theoretical_ser = np.array([0.57388349, 0.31731051, 0.07535798, 0.0015654])
        np.testing.assert_array_almost_equal(
            self.psk_obj.calcTheoreticalSER(SNR_values),
            theoretical_ser)

        #self.psk_obj.calcTheoreticalBER
        np.testing.assert_array_almost_equal(
            self.psk_obj.calcTheoreticalBER(SNR_values),
            theoretical_ser / 2.)

        # xxxxxxxxxx Test for the 8 PSK xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        theoretical_ser2 = np.array([0.76087121, 0.58837243, 0.33584978, 0.08700502])
        np.testing.assert_array_almost_equal(
            self.psk_obj2.calcTheoreticalSER(SNR_values),
            theoretical_ser2)

        np.testing.assert_array_almost_equal(
            self.psk_obj2.calcTheoreticalBER(SNR_values),
            theoretical_ser2 / 3.)

    # The calcTheoreticalPER method is defined in the Modulatros class, but
    # can only be tested in a subclass, since it depends on the
    # calcTheoreticalBER method. Therefore, we chose to test it here.
    def test_calc_theoretical_PER(self):
        L1 = 50
        L2 = 120
        SNRs = np.array([10, 13])
        # The BER for SNR values of 10 and 13 are 7.82701129e-04 and
        # 3.96924840e-06, respectively
        BER = self.psk_obj.calcTheoreticalBER(SNRs)

        expected_PER1 = (1 - BER) ** L1
        expected_PER1 = 1 - expected_PER1

        expected_PER2 = (1 - BER) ** L2
        expected_PER2 = 1 - expected_PER2

        PER1 = self.psk_obj.calcTheoreticalPER(SNRs, L1)
        PER2 = self.psk_obj.calcTheoreticalPER(SNRs, L2)

        np.testing.assert_array_almost_equal(PER1, expected_PER1)
        np.testing.assert_array_almost_equal(PER2, expected_PER2)

        # Sanity check
        PER = self.psk_obj.calcTheoreticalPER(SNRs, 1)
        np.testing.assert_array_almost_equal(BER, PER)

    def test_calc_theoretical_spectral_efficiency(self):
        L1 = 50
        L2 = 120
        SNRs = np.array([10, 13])

        se = self.psk_obj.calcTheoreticalSpectralEfficiency(SNRs)
        se1 = self.psk_obj.calcTheoreticalSpectralEfficiency(SNRs, L1)
        se2 = self.psk_obj.calcTheoreticalSpectralEfficiency(SNRs, L2)

        K = self.psk_obj.K
        expected_se = K * (1 - self.psk_obj.calcTheoreticalPER(SNRs, 1))
        expected_se1 = K * (1 - self.psk_obj.calcTheoreticalPER(SNRs, L1))
        expected_se2 = K * (1 - self.psk_obj.calcTheoreticalPER(SNRs, L2))

        np.testing.assert_array_almost_equal(se, expected_se)
        np.testing.assert_array_almost_equal(se1, expected_se1)
        np.testing.assert_array_almost_equal(se2, expected_se2)

    def test_modulate_and_demodulate(self):
        noise = randn_c(20,) * 1e-2

        input_data = np.random.random_integers(0, 4 - 1, 20)
        modulated_data = self.psk_obj.modulate(input_data)
        demodulated_data = self.psk_obj.demodulate(modulated_data + noise)

        np.testing.assert_array_equal(input_data, demodulated_data)

        input_data2 = np.random.random_integers(0, 8 - 1, 20)
        modulated_data2 = self.psk_obj2.modulate(input_data2)
        demodulated_data2 = self.psk_obj2.demodulate(modulated_data2 + noise)
        np.testing.assert_array_equal(input_data2, demodulated_data2)

        # Test if an exception is raised for invalid arguments
        with self.assertRaises(ValueError):
            self.psk_obj.modulate(4)
        with self.assertRaises(ValueError):
            self.psk_obj2.modulate(10)


class BPSKTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.bpsk_obj = modulators.BPSK()

    def test_name(self):
        self.assertEqual(self.bpsk_obj.name, "BPSK")

    def test_constellation(self):
        self.assertEqual(self.bpsk_obj.M, 2)
        self.assertAlmostEqual(self.bpsk_obj.K, 1)
        np.testing.assert_array_almost_equal(self.bpsk_obj.symbols,
                                             np.array([1, -1]))

    def test_calc_theoretical_SER_and_BER(self):
        SNR_values = np.array([-5, 0, 5, 10])

        theoretical_ser = np.array([2.13228018e-01, 7.86496035e-02,
                                    5.95386715e-03, 3.87210822e-06])
        np.testing.assert_array_almost_equal(
            self.bpsk_obj.calcTheoreticalSER(SNR_values),
            theoretical_ser)

        # The SER and the BER are equal for BPSK modulation
        np.testing.assert_array_almost_equal(
            self.bpsk_obj.calcTheoreticalBER(SNR_values),
            theoretical_ser)

    def test_modulate_and_demodulate(self):
        input_data = np.random.random_integers(0, 1, 20)
        modulated_data = self.bpsk_obj.modulate(input_data)

        noise = randn_c(20,) * 1e-2

        demodulated_data = self.bpsk_obj.demodulate(modulated_data + noise)
        np.testing.assert_array_equal(input_data, demodulated_data)

        # Test if an exception is raised for invalid arguments
        with self.assertRaises(ValueError):
            self.bpsk_obj.modulate(2)


class QAMTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.qam_obj = modulators.QAM(4)
        self.qam_obj2 = modulators.QAM(16)
        self.qam_obj3 = modulators.QAM(64)

    def test_invalid_QAM_size(self):
        with self.assertRaises(ValueError):
            modulators.QAM(32)
        with self.assertRaises(ValueError):
            modulators.QAM(63)

    def test_constellation(self):
        self.assertEqual(self.qam_obj.M, 4)
        self.assertAlmostEqual(self.qam_obj.K, 2)
        np.testing.assert_array_almost_equal(
            self.qam_obj.symbols,
            np.array([-0.70710678 + 0.70710678j, 0.70710678 + 0.70710678j,
                      -0.70710678 - 0.70710678j, 0.70710678 - 0.70710678j]))

        self.assertEqual(self.qam_obj2.M, 16)
        self.assertAlmostEqual(self.qam_obj2.K, 4)
        np.testing.assert_array_almost_equal(
            self.qam_obj2.symbols,
            np.array([-0.94868330 + 0.9486833j, -0.31622777 + 0.9486833j,
                      0.94868330 + 0.9486833j, 0.31622777 + 0.9486833j,
                      -0.94868330 + 0.31622777j, -0.31622777 + 0.31622777j,
                      0.94868330 + 0.31622777j, 0.31622777 + 0.31622777j,
                      -0.94868330 - 0.9486833j, -0.31622777 - 0.9486833j,
                      0.94868330 - 0.9486833j, 0.31622777 - 0.9486833j,
                      -0.94868330 - 0.31622777j, -0.31622777 - 0.31622777j,
                      0.94868330 - 0.31622777j, 0.31622777 - 0.31622777j]))

        self.assertEqual(self.qam_obj3.M, 64)
        self.assertAlmostEqual(self.qam_obj3.K, 6)
        np.testing.assert_array_almost_equal(
            self.qam_obj3.symbols,
            np.array([-1.08012345 + 1.08012345j, -0.77151675 + 1.08012345j,
                      -0.15430335 + 1.08012345j, -0.46291005 + 1.08012345j,
                      0.77151675 + 1.08012345j, 1.08012345 + 1.08012345j,
                      0.46291005 + 1.08012345j, 0.15430335 + 1.08012345j,
                      -1.08012345 + 0.77151675j, -0.77151675 + 0.77151675j,
                      -0.15430335 + 0.77151675j, -0.46291005 + 0.77151675j,
                      0.77151675 + 0.77151675j, 1.08012345 + 0.77151675j,
                      0.46291005 + 0.77151675j, 0.15430335 + 0.77151675j,
                      -1.08012345 + 0.15430335j, -0.77151675 + 0.15430335j,
                      -0.15430335 + 0.15430335j, -0.46291005 + 0.15430335j,
                      0.77151675 + 0.15430335j, 1.08012345 + 0.15430335j,
                      0.46291005 + 0.15430335j, 0.15430335 + 0.15430335j,
                      -1.08012345 + 0.46291005j, -0.77151675 + 0.46291005j,
                      -0.15430335 + 0.46291005j, -0.46291005 + 0.46291005j,
                      0.77151675 + 0.46291005j, 1.08012345 + 0.46291005j,
                      0.46291005 + 0.46291005j, 0.15430335 + 0.46291005j,
                      -1.08012345 - 0.77151675j, -0.77151675 - 0.77151675j,
                      -0.15430335 - 0.77151675j, -0.46291005 - 0.77151675j,
                      0.77151675 - 0.77151675j, 1.08012345 - 0.77151675j,
                      0.46291005 - 0.77151675j, 0.15430335 - 0.77151675j,
                      -1.08012345 - 1.08012345j, -0.77151675 - 1.08012345j,
                      -0.15430335 - 1.08012345j, -0.46291005 - 1.08012345j,
                      0.77151675 - 1.08012345j, 1.08012345 - 1.08012345j,
                      0.46291005 - 1.08012345j, 0.15430335 - 1.08012345j,
                      -1.08012345 - 0.46291005j, -0.77151675 - 0.46291005j,
                      -0.15430335 - 0.46291005j, -0.46291005 - 0.46291005j,
                      0.77151675 - 0.46291005j, 1.08012345 - 0.46291005j,
                      0.46291005 - 0.46291005j, 0.15430335 - 0.46291005j,
                      -1.08012345 - 0.15430335j, -0.77151675 - 0.15430335j,
                      -0.15430335 - 0.15430335j, -0.46291005 - 0.15430335j,
                      0.77151675 - 0.15430335j, 1.08012345 - 0.15430335j,
                      0.46291005 - 0.15430335j, 0.15430335 - 0.15430335j]))

    def test_calc_theoretical_SER_and_BER(self):
        SNR_values = np.array([0, 5, 10, 15, 20])

        # xxxxxxxxxx Test for 4-QAM xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        theoretical_ser = np.array([2.92139018e-01, 7.39382701e-02,
                                    1.56478964e-03, 1.87220798e-08, 0])
        np.testing.assert_array_almost_equal(
            self.qam_obj.calcTheoreticalSER(SNR_values),
            theoretical_ser)

        theoretical_ber = np.array([1.58655254e-01, 3.76789881e-02,
                                    7.82701129e-04, 9.36103999e-09,
                                    7.61985302e-24])
        np.testing.assert_array_almost_equal(
            self.qam_obj.calcTheoreticalBER(SNR_values),
            theoretical_ber)

        # xxxxxxxxxx Test for 16-QAM xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        theoretical_ser2 = np.array([7.40960364e-01, 5.37385132e-01,
                                     2.22030850e-01, 1.77818422e-02,
                                     1.16162909e-05])
        np.testing.assert_array_almost_equal(
            self.qam_obj2.calcTheoreticalSER(SNR_values),
            theoretical_ser2)

        theoretical_ber2 = np.array([2.45520317e-01, 1.59921014e-01,
                                     5.89872026e-02, 4.46540036e-03,
                                     2.90408116e-06])
        np.testing.assert_array_almost_equal(
            self.qam_obj2.calcTheoreticalBER(SNR_values),
            theoretical_ber2)
        # xxxxxxxxxx Test for 64-QAM xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        theoretical_ser3 = np.array([0.92374224, 0.84846895, 0.67382633,
                                     0.3476243, 0.05027041])
        np.testing.assert_array_almost_equal(
            self.qam_obj3.calcTheoreticalSER(SNR_values),
            theoretical_ser3)

        theoretical_ber3 = np.array([0.24128398, 0.2035767, 0.14296128,
                                     0.06410074, 0.00848643])
        np.testing.assert_array_almost_equal(
            self.qam_obj3.calcTheoreticalBER(SNR_values),
            theoretical_ber3)

    def test_modulate_and_demodulate(self):
        noise = randn_c(20,) * 1e-2

        # xxxxxxxxxx Test for 4-QAM xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        input_data = np.random.random_integers(0, 4 - 1, 20)
        modulated_data = self.qam_obj.modulate(input_data)
        demodulated_data = self.qam_obj.demodulate(modulated_data + noise)
        np.testing.assert_array_equal(input_data, demodulated_data)

        # xxxxxxxxxx Test for 16-QAM xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        input_data2 = np.random.random_integers(0, 16 - 1, 20)
        modulated_data2 = self.qam_obj2.modulate(input_data2)
        demodulated_data2 = self.qam_obj2.demodulate(modulated_data2 + noise)
        np.testing.assert_array_equal(input_data2, demodulated_data2)

        # xxxxxxxxxx Test for 64-QAM xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        input_data3 = np.random.random_integers(0, 64 - 1, 20)
        modulated_data3 = self.qam_obj3.modulate(input_data3)
        demodulated_data3 = self.qam_obj3.demodulate(modulated_data3 + noise)
        np.testing.assert_array_equal(input_data3, demodulated_data3)

        # Test if an exception is raised for invalid arguments
        with self.assertRaises(ValueError):
            self.qam_obj.modulate(4)

        with self.assertRaises(ValueError):
            self.qam_obj2.modulate(16)

        with self.assertRaises(ValueError):
            self.qam_obj3.modulate(65)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Pathloss Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class PathLossBaseTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.pl = pathloss.PathLossBase()

    # This is here to make code nosetests coverage happy
    def test_not_implemented_methods(self):
        with self.assertRaises(NotImplementedError):
            self.pl.which_distance_dB(None)
        with self.assertRaises(NotImplementedError):
            self.pl._calc_deterministic_path_loss_dB(None)


class PathLossFreeSpaceTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.pl = pathloss.PathLossFreeSpace()

    def test_calc_path_loss(self):
        # Test for a single path loss value
        self.assertAlmostEqual(self.pl.calc_path_loss(1.2),
                               4.88624535312e-10)
        self.assertAlmostEqual(self.pl.calc_path_loss_dB(1.2),
                               93.1102472958)

        # Test for multiple path loss values
        np.testing.assert_array_almost_equal(
            self.pl.calc_path_loss([1.2, 1.4, 1.6]),
            np.array([4.88624535e-10, 3.58989455e-10, 2.74851301e-10]), 16)

        # Test test_calc_path_loss with shadow
        self.pl.use_shadow_bool = True
        # We don't know the value of the shadowing to test it, but we can
        # at least test that the shadowing modified the path loss
        self.assertNotAlmostEqual(self.pl.calc_path_loss_dB(1.2),
                                  93.1102472958)

        # TODO: Finish the implementation below
        # Test if calc_path_loss works with shadowing for multiple values.
        #self.pl.calc_path_loss([1.2, 1.4, 1.6]),

    def test_calc_which_distance(self):
        # Test which_distance and which_distance_dB for a single value.
        self.assertAlmostEqual(self.pl.which_distance(4.88624535312e-10),
                               1.2)
        self.assertAlmostEqual(self.pl.which_distance_dB(93.1102472958),
                               1.2)

        # Test which_distance and which_distance_dB for an array of values.
        np.testing.assert_array_almost_equal(
            self.pl.which_distance_dB(np.array([93.110247295, 91.526622374])),
            np.array([1.2, 1.0]))
        np.testing.assert_array_almost_equal(
            self.pl.which_distance(np.array([4.88624535e-10, 7.0361933e-10])),
            np.array([1.2, 1.0]))


# TODO: finish implementation
class PathLossOkomuraHataTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.pl = pathloss.PathLossOkomuraHata()

    def test_model_parameters(self):
        self.assertAlmostEqual(self.pl.hms, 1.0)
        self.assertAlmostEqual(self.pl.hbs, 30.0)
        self.assertAlmostEqual(self.pl.fc, 900.0)

        # Valid values -> no exceptions should be raised
        self.pl.hbs = 45.0
        self.pl.hms = 1.5
        self.pl.fc = 1100.0

        # Invalid values: an exception should be raised
        with self.assertRaises(RuntimeError):
            self.pl.hms = 0.8

        with self.assertRaises(RuntimeError):
            self.pl.hms = 11.4

        with self.assertRaises(RuntimeError):
            self.pl.hbs = 25.0

        with self.assertRaises(RuntimeError):
            self.pl.hbs = 205.3

        with self.assertRaises(RuntimeError):
            self.pl.fc = 130.0

        with self.assertRaises(RuntimeError):
            self.pl.fc = 1600.0

    def test_calc_deterministic_path_loss_dB(self):
        self.pl.fc = 900.0
        self.pl.hbs = 30.0
        self.pl.hms = 1.0

        # Distances for which the path loss will be calculated
        d = np.linspace(1, 20, 20)

        # xxxxxxxxxx Test for the 'open' area type xxxxxxxxxxxxxxxxxxxxxxxx
        self.pl.area_type = 'open'
        expected_open_pl = np.array(
            [99.1717017731874, 109.775439956383, 115.978229161017,
             120.379178139578, 123.792819371578, 126.581967344212,
             128.940158353991, 130.982916322773, 132.784756548846,
             134.396557554774, 135.854608919885, 137.185705527407,
             138.410195707052, 139.543896537186, 140.599346759408,
             141.586654505968, 142.514087575345, 143.388494732042,
             144.215612946935, 145.000295737969])

        np.testing.assert_array_almost_equal(
            expected_open_pl,
            self.pl._calc_deterministic_path_loss_dB(d))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test for the suburban area type xxxxxxxxxxxxxxxxxxxxxx
        self.pl.area_type = 'suburban'
        expected_suburban_pl = np.array(
            [117.735512612807, 128.339250796002, 134.542040000636,
             138.942988979197, 142.356630211198, 145.145778183831,
             147.50396919361, 149.546727162392, 151.348567388466,
             152.960368394393, 154.418419759504, 155.749516367027,
             156.974006546672, 158.107707376805, 159.163157599027,
             160.150465345588, 161.077898414965, 161.952305571661,
             162.779423786554, 163.564106577588])

        np.testing.assert_array_almost_equal(
            expected_suburban_pl,
            self.pl._calc_deterministic_path_loss_dB(d))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test for the medium and small city area types xxxxxxxxxxxxx
        self.pl.area_type = 'medium city'
        expected_urban_pl = np.array(
            [127.678119861049, 138.281858044244, 144.484647248879,
             148.88559622744, 152.29923745944, 155.088385432074,
             157.446576441852, 159.489334410635, 161.291174636708,
             162.902975642635, 164.361027007746, 165.692123615269,
             166.916613794914, 168.050314625048, 169.10576484727,
             170.09307259383, 171.020505663207, 171.894912819903,
             172.722031034797, 173.506713825831])

        np.testing.assert_array_almost_equal(
            expected_urban_pl,
            self.pl._calc_deterministic_path_loss_dB(d))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test for the 'large city' area type xxxxxxxxxxxxxxxxxx
        # TODO: The test below is only for frequency 900MHz. You need to
        # test for a lower frequency.
        self.pl.area_type = 'large city'
        expected_large_city_pl = np.array(
            [127.72522899, 138.32896717, 144.53175638, 148.93270536,
             152.34634659, 155.13549456, 157.49368557, 159.53644354,
             161.33828377, 162.95008477, 164.40813614, 165.73923275,
             166.96372293, 168.09742376, 169.15287398, 170.14018172,
             171.06761479, 171.94202195, 172.76914017, 173.55382296])
        np.testing.assert_array_almost_equal(
            expected_large_city_pl,
            self.pl._calc_deterministic_path_loss_dB(d))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    # plot_psd_OFDM_symbols()
    unittest.main()
