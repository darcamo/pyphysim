#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the comm package.

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
from scipy import linalg

from comm import modulators, blockdiagonalization, ofdm, mimo, pathloss, waterfilling, channels
from util.misc import randn_c


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
        big_matrix = channels.MultiUserChannelMatrix._from_small_matrix_to_big_matrix(small_matrix, Nr, Nt, K)

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
            self.multiH.get_channel(0, 0),
            np.ones([2, 2]) * 0)

        np.testing.assert_array_equal(
            self.multiH.get_channel(0, 1),
            np.ones([2, 3]) * 1)

        np.testing.assert_array_equal(
            self.multiH.get_channel(0, 2),
            np.ones([2, 5]) * 2)

        np.testing.assert_array_equal(
            self.multiH.get_channel(1, 0),
            np.ones([4, 2]) * 3)

        np.testing.assert_array_equal(
            self.multiH.get_channel(1, 1),
            np.ones([4, 3]) * 4)

        np.testing.assert_array_equal(
            self.multiH.get_channel(1, 2),
            np.ones([4, 5]) * 5)

        np.testing.assert_array_equal(
            self.multiH.get_channel(2, 0),
            np.ones([6, 2]) * 6)

        np.testing.assert_array_equal(
            self.multiH.get_channel(2, 1),
            np.ones([6, 3]) * 7)

        np.testing.assert_array_equal(
            self.multiH.get_channel(2, 2),
            np.ones([6, 5]) * 8)

        # xxxxxxxxxx Test get_channel with Pathloss xxxxxxxxxxxxxxxxxxxxxxx
        # pathloss (in linear scale) must be a positive number
        pathloss = np.abs(np.random.randn(self.K, self.K))
        self.multiH.set_pathloss(pathloss)
        np.testing.assert_array_equal(
            self.multiH.get_channel(0, 0),
            np.sqrt(self.multiH.pathloss[0, 0]) * np.ones([2, 2]) * 0)

        np.testing.assert_array_equal(
            self.multiH.get_channel(0, 1),
            np.sqrt(self.multiH.pathloss[0, 1]) * np.ones([2, 3]) * 1)

        np.testing.assert_array_equal(
            self.multiH.get_channel(0, 2),
            np.sqrt(self.multiH.pathloss[0, 2]) * np.ones([2, 5]) * 2)

        np.testing.assert_array_equal(
            self.multiH.get_channel(1, 0),
            np.sqrt(self.multiH.pathloss[1, 0]) * np.ones([4, 2]) * 3)

        np.testing.assert_array_equal(
            self.multiH.get_channel(1, 1),
            np.sqrt(self.multiH.pathloss[1, 1]) * np.ones([4, 3]) * 4)

        np.testing.assert_array_equal(
            self.multiH.get_channel(1, 2),
            np.sqrt(self.multiH.pathloss[1, 2]) * np.ones([4, 5]) * 5)

        np.testing.assert_array_equal(
            self.multiH.get_channel(2, 0),
            np.sqrt(self.multiH.pathloss[2, 0]) * np.ones([6, 2]) * 6)

        np.testing.assert_array_equal(
            self.multiH.get_channel(2, 1),
            np.sqrt(self.multiH.pathloss[2, 1]) * np.ones([6, 3]) * 7)

        np.testing.assert_array_equal(
            self.multiH.get_channel(2, 2),
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
            self.multiH.get_channel_all_tx_to_rx_k(0),
            expected_H1
        )
        np.testing.assert_array_equal(
            self.multiH.get_channel_all_tx_to_rx_k(1),
            expected_H2
        )
        np.testing.assert_array_equal(
            self.multiH.get_channel_all_tx_to_rx_k(2),
            expected_H3
        )

        # xxxxx Test with pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        pathloss = np.abs(np.random.randn(self.K, self.K))
        self.multiH.set_pathloss(pathloss)
        expected_H1 = self.multiH.big_H[0:2, :]
        expected_H2 = self.multiH.big_H[2:6, :]
        expected_H3 = self.multiH.big_H[6:, :]
        np.testing.assert_array_equal(
            self.multiH.get_channel_all_tx_to_rx_k(0),
            expected_H1
        )
        np.testing.assert_array_equal(
            self.multiH.get_channel_all_tx_to_rx_k(1),
            expected_H2
        )
        np.testing.assert_array_equal(
            self.multiH.get_channel_all_tx_to_rx_k(2),
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
                    self.multiH.get_channel(row, col), self.multiH.H[row, col])
                # Test the 'big_H' property
                np.testing.assert_array_equal(
                    self.multiH.get_channel(row, col),
                    self.multiH.big_H[cumNr[row]:cumNr[row + 1], cumNt[col]:cumNt[col + 1]])

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
                    self.multiH.get_channel(rx, tx), input_data[tx])

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
                    self.multiH.get_channel(rx, tx), input_data[tx])

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

    def test_big_H_no_ext_int_property(self):
        self.multiH.randomize(np.array([2, 2]), np.array([2, 2]), 2, 2)
        np.testing.assert_array_almost_equal(self.multiH.big_H_no_ext_int,
                                             self.multiH.big_H[:, :-2])

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
            self.multiH.big_H[:, :4],
            Nr,
            Nt,
            K)

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

        received_data2 = self.multiH.corrupt_data(input_data, input_data_extint2)

        # received_data2_expected for now has only the included the user's
        # signal. Lets add the external interference source's signal.
        received_data2_expected[0] = received_data2_expected[0] + np.dot(self.multiH.get_channel(0, 2), input_data_extint2[0]) + np.dot(self.multiH.get_channel(0, 3), input_data_extint2[1])
        received_data2_expected[1] = received_data2_expected[1] + np.dot(self.multiH.get_channel(1, 2), input_data_extint2[0]) + np.dot(self.multiH.get_channel(1, 3), input_data_extint2[1])

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

        # print
        # print self.multiH.big_H

        expected_H1 = self.multiH.big_H[0:2, :np.sum(Nt)]
        expected_H2 = self.multiH.big_H[2:6, :np.sum(Nt)]
        expected_H3 = self.multiH.big_H[6:, :np.sum(Nt)]

        # print
        # print expected_H1

        # xxxxx Test without pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_equal(
            self.multiH.get_channel_all_tx_to_rx_k(0),
            expected_H1
        )
        np.testing.assert_array_equal(
            self.multiH.get_channel_all_tx_to_rx_k(1),
            expected_H2
        )
        np.testing.assert_array_equal(
            self.multiH.get_channel_all_tx_to_rx_k(2),
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
            self.multiH.get_channel_all_tx_to_rx_k(0),
            expected_H1
        )
        np.testing.assert_array_equal(
            self.multiH.get_channel_all_tx_to_rx_k(1),
            expected_H2
        )
        np.testing.assert_array_equal(
            self.multiH.get_channel_all_tx_to_rx_k(2),
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
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx OFDM Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class OfdmTestCase(unittest.TestCase):
    """Unittests for the OFDM class in the ofdm module."""
    def setUp(self):
        """Called before each test."""
        from comm.ofdm import OFDM
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
        self.assertRaises(ValueError, self.ofdm_object.set_parameters, 64, 16, 51)
        # Raises an exception if number of used subcarriers is greater than
        # the fft_size
        self.assertRaises(ValueError, self.ofdm_object.set_parameters, 64, 16, 70)
        # Raises an exception if cp_size is negative
        self.assertRaises(ValueError, self.ofdm_object.set_parameters, 64, -2, 52)
        # Raises an exception if cp_size is greater than the fft_size
        self.assertRaises(ValueError, self.ofdm_object.set_parameters, 64, 65, 52)
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
        (zeropad, num_ofdm_symbols) = self.ofdm_object._calc_zeropad(input_signal.size)
        self.assertEqual(zeropad, 0)
        self.assertEqual(num_ofdm_symbols, 1)

        expected_data = np.array(
            [[0., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.,
              39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51.,
              52., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 4.,
              5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
              19., 20., 21., 22., 23., 24., 25., 26.]])
        np.testing.assert_array_equal(self.ofdm_object._prepare_input_signal(input_signal), expected_data)
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
        np.testing.assert_array_equal(self.ofdm_object._prepare_input_signal(input_signal), expected_data2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now lets test the case when we use all subcarriers (but still
        # with zeropadding)
        self.ofdm_object.num_used_subcarriers = 64
        (zeropad, num_ofdm_symbols) = self.ofdm_object._calc_zeropad(input_signal.size)
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
        input_signal = np.r_[1:105]  # Exactly two OFDM symbols (with 52 used
                                     # subcarriers)

        # xxxxx First lets try without cyclic prefix xxxxxxxxxxxxxxxxxxxxxx
        self.ofdm_object.set_parameters(64, 0, 52)
        (zeropad, num_ofdm_symbols) = self.ofdm_object._calc_zeropad(input_signal.size)
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
            self.ofdm_object.modulate(input_signal[0:52]), expected_data[0:64])
        np.testing.assert_array_almost_equal(
            self.ofdm_object.modulate(input_signal[52:]), expected_data[64:])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Now lets test with a cyclic prefix xxxxxxxxxxxxxxxxxxxxxxxx
        self.ofdm_object.set_parameters(64, 4, 52)
        expected_data2 = expected_data[0:64]
        expected_data2 = np.hstack([expected_data2[-self.ofdm_object.cp_size:], expected_data[0:64]])
        np.testing.assert_array_almost_equal(
            self.ofdm_object.modulate(input_signal[0:52]), expected_data2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_demodulate(self):
        # xxxxx First lets try without cyclic prefix xxxxxxxxxxxxxxxxxxxxxx
        input_signal = np.r_[1:105]  # Exactly two OFDM symbols (with 52 used
                                     # subcarriers)
        # print input_signal
        # print self.ofdm_object._prepare_input_signal(input_signal)

        # xxxxx First lets try without cyclic prefix xxxxxxxxxxxxxxxxxxxxxx
        self.ofdm_object.set_parameters(64, 0, 52)
        modulated_ofdm_symbols = self.ofdm_object.modulate(input_signal)

        demodulated_symbols = self.ofdm_object.demodulate(modulated_ofdm_symbols)
        np.testing.assert_array_equal(
            np.real(demodulated_symbols.round()).astype(int),
            input_signal
        )
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Now lets test with a cyclic prefix xxxxxxxxxxxxxxxxxxxxxxxx
        input_signal2 = np.r_[1:105]  # Exactly two OFDM symbols (with 52 used
                                     # subcarriers)
        self.ofdm_object.set_parameters(64, 16, 52)
        modulated_ofdm_symbols2 = self.ofdm_object.modulate(input_signal2)

        demodulated_symbols2 = self.ofdm_object.demodulate(modulated_ofdm_symbols2)
        np.testing.assert_array_equal(
            np.real(demodulated_symbols2.round()).astype(int),
            input_signal2
        )
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Now lets test the case with zeropadding xxxxxxxxxxxxxxxxxxx
        input_signal3 = np.r_[1:110]  # Exactly two OFDM symbols (with 52 used
                                     # subcarriers)
        self.ofdm_object.set_parameters(64, 16, 52)
        modulated_ofdm_symbols3 = self.ofdm_object.modulate(input_signal3)

        demodulated_symbols3 = self.ofdm_object.demodulate(modulated_ofdm_symbols3)
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
    from matplotlib import pylab
    from matplotlib import pyplot as plt

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
        from comm.mimo import Blast
        self.blast_object = Blast(3)

    def test_getNumberOfLayers(self):
        from comm.mimo import Blast
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
        self.blast_object.set_noise_var(0.00000001)  # This should use the ZF filter
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
        from comm.mimo import Alamouti
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
        (Ms_bad, Sigma) = self.BD._calc_BD_matrix_no_power_scaling(channel)

        newH = np.dot(channel, Ms_bad)

        # Because Ms_bad does not have any power scaling and each column of
        # it comes is a singular vector calculated with the SVD, then the
        # square of its Frobenius norm is equal to its dimension.
        self.assertAlmostEqual(np.linalg.norm(Ms_bad, 'fro') ** 2,
                               Ms_bad.shape[0])

        # Now let's test if newH is really a block diagonal matrix.
        # First we create a 'mask'
        A = np.ones([self.iNrk, self.iNtk])
        mask = linalg.block_diag(A, A, A)

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
        self.assertAlmostEqual(linalg.norm(Ms_good, 'fro') ** 2,
                               total_power)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Ms_good must still be able to block diagonalize the channel
        A = np.ones([self.iNrk, self.iNtk])
        mask = linalg.block_diag(A, A, A)
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
            # channel) will employ a precoder a precoder with total power
            # of `Pu`, while the other base stations will use less power.
            individual_powers.append(np.linalg.norm(Ms_good[:, cum_Nt[i]:cum_Nt[i] + self.num_antenas], 'fro') ** 2)
            self.assertGreaterEqual(self.Pu + 1e-12,
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
        mask = linalg.block_diag(A, A, A)

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
        for i in range(num_users):
            # Most likelly only one base station (the one with the worst
            # channel) will employ a precoder a precoder with total power
            # of `Pu`, while the other base stations will use less power.
            individual_powers.append(np.linalg.norm(Ms[:, cum_Nt[i]:cum_Nt[i] + num_antenas], 'fro') ** 2)
            self.assertGreaterEqual(Pu + 1e-8,
                                    individual_powers[-1])

    def test_calc_receive_filter(self):
        Pu = self.Pu
        noise_var = self.noise_var
        num_users = self.num_users
        #num_antenas = self.num_antenas
        channel = randn_c(self.iNr, self.iNt)

        (newH, Ms) = blockdiagonalization.block_diagonalize(
            channel, num_users, Pu, noise_var)

        W_bd = blockdiagonalization.calc_receive_filter(newH)

        np.testing.assert_array_almost_equal(np.dot(W_bd, newH),
                                             np.eye(np.sum(self.iNt)))


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

        # Test if calc_path_loss works with shadowing for multiple values.
        self.pl.calc_path_loss([1.2, 1.4, 1.6]),

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


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    # plot_psd_OFDM_symbols()
    unittest.main()
