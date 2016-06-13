#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,E0611

"""
Tests for the modules in the comm package.

Each module has several doctests that we run in addition to the unittests
defined here.
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
from scipy.linalg import block_diag

from pyphysim.comm import blockdiagonalization, waterfilling
from pyphysim.modulators import fundamental
from pyphysim.channels import multiuser
from pyphysim.util.misc import calc_whitening_matrix, randn_c, \
    calc_shannon_sum_capacity
from pyphysim.util.conversion import dB2Linear, linear2dB
from pyphysim.subspace.projections import calcProjectionMatrix


# UPDATE THIS CLASS if another module is added to the comm package
class CommDoctestsTestCase(unittest.TestCase):
    """Test case that run all the doctests in the modules of the comm
    package. """

    def test_blockdiagonalization(self, ):
        """Run doctests in the blockdiagonalization module."""
        doctest.testmod(blockdiagonalization)

    def test_waterfilling(self, ):
        """Run doctests in the waterfilling module."""
        doctest.testmod(waterfilling)


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
        # http://jungwon.comoj.com/ucsd_ece287b_spr12/lecture_slides/lecture4.pdf

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
    """Unittests for the BlockDiagonalizer class in the blockdiagonalization
    module.

    """

    def setUp(self):
        """Called before each test."""
        self.Pu = 5.  # Power for each user
        self.noise_var = 1e-6
        self.num_users = 3
        self.num_antennas = 2

        self.iNrk = self.num_antennas  # Number of receive antennas per user
        self.iNtk = self.num_antennas  # Number of transmit antennas per user

        self.iNr = self.iNrk * self.num_users  # Total number of Rx antennas
        self.iNt = self.iNtk * self.num_users  # Total number of Tx antennas

        self.BD = blockdiagonalization.BlockDiagonalizer(
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
        # accumulated number of transmit antennas
        cum_Nt = np.cumsum(
            np.hstack([0,
                       np.ones(self.num_users, dtype=int) * self.num_antennas]))

        individual_powers = []
        for i in range(self.num_users):
            # Most likely only one base station (the one with the worst
            # channel) will employ a precoder with total power of `Pu`,
            # while the other base stations will use less power.
            individual_powers.append(
                np.linalg.norm(
                    Ms_good[:, cum_Nt[i]:cum_Nt[i] + self.num_antennas], 'fro'
                ) ** 2)
            # 1e-12 is included to avoid false test fails due to small
            # precision errors
            tol = 1e-12
            self.assertGreaterEqual(self.Pu + tol,
                                    individual_powers[-1])

    def test_block_diagonalize(self):
        Pu = self.Pu
        noise_var = self.noise_var
        num_users = self.num_users
        num_antennas = self.num_antennas

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

        # accumulated number of receive antennas
        cum_Nt = np.cumsum(
            np.hstack([0, np.ones(num_users, dtype=int) * num_antennas]))

        # Individual power restriction of each class
        individual_powers = []
        tol = 1e-12  # Tolerance for the GreaterEqual test
        for i in range(num_users):
            # Most likely only one base station (the one with the worst
            # channel) will employ a precoder a precoder with total power
            # of `Pu`, while the other base stations will use less power.
            individual_powers.append(
                np.linalg.norm(
                    Ms[:, cum_Nt[i]:cum_Nt[i] + num_antennas], 'fro') ** 2)
            self.assertGreaterEqual(Pu + tol,
                                    individual_powers[-1])

    def test_block_diagonalize_no_waterfilling(self):
        Pu = self.Pu
        num_users = self.num_users
        num_antennas = self.num_antennas

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

        # accumulated number of receive antennas
        cum_Nt = np.cumsum(
            np.hstack([0, np.ones(num_users, dtype=int) * num_antennas]))

        # Individual power restriction of each class
        individual_powers = []
        for i in range(num_users):
            # Most likely only one base station (the one with the worst
            # channel) will employ a precoder a precoder with total power
            # of `Pu`, while the other base stations will use less power.
            individual_powers.append(
                np.linalg.norm(
                    Ms[:, cum_Nt[i]:cum_Nt[i] + num_antennas], 'fro'
                ) ** 2)
            self.assertGreaterEqual(Pu + tol,
                                    individual_powers[-1])

    def test_calc_receive_filter(self):
        Pu = self.Pu
        noise_var = self.noise_var
        num_users = self.num_users
        # num_antennas = self.num_antennas
        channel = randn_c(self.iNr, self.iNt)

        (newH, _) = blockdiagonalization.block_diagonalize(
            channel, num_users, Pu, noise_var)

        # W_bd is a block diagonal matrix, where each "small block" is the
        # receive filter of one user.
        W_bd = blockdiagonalization.calc_receive_filter(newH)

        np.testing.assert_array_almost_equal(np.dot(W_bd, newH),
                                             np.eye(self.iNt))

        # Retest for each individual user
        W0 = W_bd[0:2, 0:2]
        newH0 = newH[0:2, 0:2]
        np.testing.assert_array_almost_equal(np.dot(W0, newH0),
                                             np.eye(self.iNt / 3))
        W1 = W_bd[2:4, 2:4]
        newH1 = newH[2:4, 2:4]
        np.testing.assert_array_almost_equal(np.dot(W1, newH1),
                                             np.eye(self.iNt / 3))
        W2 = W_bd[4:, 4:]
        newH2 = newH[4:, 4:]
        np.testing.assert_array_almost_equal(np.dot(W2, newH2),
                                             np.eye(self.iNt / 3))


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
        mu_channel = multiuser.MultiUserChannelMatrixExtInt()
        mu_channel.randomize(Nr, Nt, K, Nti)
        mu_channel.noise_var = noise_var

        bd_obj = blockdiagonalization.BDWithExtIntBase(K, iPu, noise_var, pe)
        W_all_k = bd_obj.calc_whitening_matrices(mu_channel)

        R_all_k = mu_channel.calc_cov_matrix_extint_plus_noise(pe)
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

        multiUserChannel = multiuser.MultiUserChannelMatrixExtInt()
        multiUserChannel.randomize(Nr, Nt, K, Nti)

        # Channel from all transmitters to the first receiver
        H1 = multiUserChannel.get_Hk_without_ext_int(0)
        # Channel from all transmitters to the second receiver
        H2 = multiUserChannel.get_Hk_without_ext_int(1)

        # Create the whiteningBD object and the regular BD object
        whiteningBD_obj = blockdiagonalization.WhiteningBD(
            K, iPu, noise_var, pe)

        # noise_plus_int_cov_matrix \
        #     = multiUserChannel.calc_cov_matrix_extint_plus_noise(
        #         noise_var, pe)

        # xxxxx First we test without ext. int. handling xxxxxxxxxxxxxxxxxx
        (Ms_all, Wk_all, Ns_all) \
            = whiteningBD_obj.block_diagonalize_no_waterfilling(
            multiUserChannel)
        Ms1 = Ms_all[0]
        Ms2 = Ms_all[1]

        self.assertEqual(Ms1.shape[1], Ns_all[0])
        self.assertEqual(Ms2.shape[1], Ns_all[1])

        # Most likely only one base station (the one with the worst
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
        # (newH, Ms_good_regular_bd) \
        #     = regularBD_obj.block_diagonalize_no_waterfilling(
        #         multiUserChannel.big_H_no_ext_int)
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
        psk_obj = fundamental.PSK(4)
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

        W1 = blockdiagonalization.EnhancedBD.calc_receive_filter_user_k(
            Heq_k_P1, P1)
        W2 = blockdiagonalization.EnhancedBD.calc_receive_filter_user_k(
            Heq_k_P2, P2)
        W3 = blockdiagonalization.EnhancedBD.calc_receive_filter_user_k(
            Heq_k_P3, P3)
        # Note that since P3 is actually including all streams, then the
        # performance is the same as if we don't reduce streams. However W3
        # and W_full are different matrices, since W3 has to compensate the
        # right multiplication of the equivalent channel by P3 and W_full
        # does not. The performance is the same because no energy is lost
        # due to stream reduction and the Frobenius norms of W3 and W_full
        # are equal.
        W_full = blockdiagonalization.EnhancedBD.calc_receive_filter_user_k(
            Heq_k)

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
        # SINRs = blockdiagonalization.EnhancedBD._calc_linear_SINRs(
        #     Heq_k_red, Wk, Rk)
        # print SINRs
        pass

    def test_calc_effective_throughput(self):
        psk_obj = fundamental.PSK(8)
        packet_length = 60

        SINRs_dB = np.array([11.4, 20.3])
        sinrs_linear = dB2Linear(SINRs_dB)

        expected_spectral_efficiency = np.sum(
            psk_obj.calcTheoreticalSpectralEfficiency(SINRs_dB, packet_length))
        ":type: float"

        spectral_efficiency = blockdiagonalization._calc_effective_throughput(
            sinrs_linear, psk_obj, packet_length)

        self.assertAlmostEqual(spectral_efficiency,
                               expected_spectral_efficiency)

    def test_block_diagonalize_no_waterfilling(self):
        Nr = np.array([2, 2])
        Nt = np.array([2, 2])
        K = Nt.size
        Nti = 1
        iPu = 1e-1  # Power for each user (linear scale)
        pe = 1e-3  # External interference power (in linear scale)
        noise_var = 1e-1

        # The modulator and packet_length are required in the
        # effective_throughput metric case
        psk_obj = fundamental.PSK(4)
        packet_length = 120

        multiUserChannel = multiuser.MultiUserChannelMatrixExtInt()
        multiUserChannel.randomize(Nr, Nt, K, Nti)
        multiUserChannel.noise_var = noise_var

        # Channel from all transmitters to the first receiver
        H1 = multiUserChannel.get_Hk_without_ext_int(0)
        # Channel from all transmitters to the second receiver
        H2 = multiUserChannel.get_Hk_without_ext_int(1)

        # Create the enhancedBD object
        enhancedBD_obj = blockdiagonalization.EnhancedBD(K, iPu, noise_var, pe)

        noise_plus_int_cov_matrix \
            = multiUserChannel.calc_cov_matrix_extint_plus_noise(pe)

        # xxxxx First we test without ext. int. handling xxxxxxxxxxxxxxxxxx
        enhancedBD_obj.set_ext_int_handling_metric(None)
        (Ms_all, Wk_all, Ns_all) \
            = enhancedBD_obj.block_diagonalize_no_waterfilling(
            multiUserChannel)
        Ms1 = Ms_all[0]
        Ms2 = Ms_all[1]

        self.assertEqual(Ms1.shape[1], Ns_all[0])
        self.assertEqual(Ms2.shape[1], Ns_all[1])

        # Most likely only one base station (the one with the worst
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
        # noinspection PyPep8
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

        (MsPk_naive_all, Wk_naive_all, Ns_naive_all) \
            = enhancedBD_obj.block_diagonalize_no_waterfilling(
            multiUserChannel)
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
        # se4 = (
        #     np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
        #         linear2dB(sinrs4[0]),
        #         packet_length))
        #     +
        #     np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
        #         linear2dB(sinrs4[1]),
        #         packet_length)))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Now with the Fixed Stream Reduction xxxxxxxxxxxxxxxxxxxxxxx
        # The 'fixed' metric requires that metric_func_extra_args_dict is
        # provided and has the 'num_streams' key. If this is not the case
        # an exception is raised
        with self.assertRaises(AttributeError):
            enhancedBD_obj.set_ext_int_handling_metric('fixed')

        # Now let's test the fixed metric
        num_streams = 1
        enhancedBD_obj.set_ext_int_handling_metric(
            'fixed',
            {'num_streams': num_streams})

        (MsPk_fixed_all, Wk_fixed_all, Ns_fixed_all) \
            = enhancedBD_obj.block_diagonalize_no_waterfilling(
            multiUserChannel)
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
        # se5 = (
        #     np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
        #         linear2dB(sinrs5[0]),
        #         packet_length))
        #     +
        #     np.sum(psk_obj.calcTheoreticalSpectralEfficiency(
        #         linear2dB(sinrs5[1]),
        #         packet_length)))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Handling external interference xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Handling external interference using the capacity metric
        enhancedBD_obj.set_ext_int_handling_metric('capacity')
        (MsPk_all, Wk_cap_all, Ns_cap_all) \
            = enhancedBD_obj.block_diagonalize_no_waterfilling(
            multiUserChannel)
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
        # noinspection PyPep8
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

        (MsPk_effec_all, Wk_effec_all, Ns_effec_all) \
            = enhancedBD_obj.block_diagonalize_no_waterfilling(
            multiUserChannel)
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
        # noinspection PyPep8
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
if __name__ == "__main__":
    unittest.main()
