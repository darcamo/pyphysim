#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101

"""
Tests for the modules in the reference_signals package.

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

import pyphysim.reference_signals
from pyphysim.reference_signals.srs import SrsUeSequence
from pyphysim.reference_signals.channel_estimation import \
    CazacBasedChannelEstimator
from pyphysim.reference_signals.dmrs import DmrsUeSequence
from pyphysim.reference_signals.zadoffchu import calcBaseZC, \
    get_extended_ZF
from pyphysim.reference_signals.srs import get_srs_seq
from pyphysim.reference_signals.dmrs import get_dmrs_seq
from pyphysim.channels.fading import TdlChannel
from pyphysim.channels.fading import COST259_TUx
from pyphysim.channels.fading_generators import JakesSampleGenerator
from pyphysim.reference_signals.root_sequence import RootSequence
from pyphysim.reference_signals import zadoffchu


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SrsDoctestsTestCase(unittest.TestCase):
    """
    Test case that run all the doctests in the modules of the simulations
    package.
    """
    def test_srs_module(self):
        """Run reference_signals module doctests"""
        doctest.testmod(pyphysim.reference_signals.srs)

    def test_dmrs_module(self):
        """Run reference_signals module doctests"""
        doctest.testmod(pyphysim.reference_signals.dmrs)

    def test_root_sequence_module(self):
        """Run reference_signals module doctests"""
        doctest.testmod(pyphysim.reference_signals.root_sequence)

    def test_zadoffchu_module(self):
        """Run reference_signals module doctests"""
        doctest.testmod(pyphysim.reference_signals.zadoffchu)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Zadoff-Chu Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TODO: finish implementation
class ZadoffChuFunctionsTestCase(unittest.TestCase):
    """
    Module to test the Zadoff-Chu related functions in the zadoffchu
    module.
    """
    def setUp(self):
        """Called before each test."""
        pass

    def test_calcBaseZC(self):
        # TODO: Implement-me
        pass

    def test_getShiftedZF(self):
        # TODO: Implement-me
        pass

    def test_get_extended_ZF(self):
        a = zadoffchu.calcBaseZC(139, u=20)
        b = zadoffchu.calcBaseZC(31, u=14)
        c = zadoffchu.calcBaseZC(19, u=5)

        # Compute and test the extended root sequence a for size 150
        a_ext = zadoffchu.get_extended_ZF(a, 150)
        expected_a_ext = np.hstack([a, a[0:11]])
        self.assertEqual(a_ext.size, 150)
        np.testing.assert_almost_equal(expected_a_ext, a_ext)

        # Compute and test the extended root sequence b for size 32
        b_ext = zadoffchu.get_extended_ZF(b, 32)
        expected_b_ext = np.hstack([b, b[0]])
        self.assertEqual(b_ext.size, 32)
        np.testing.assert_almost_equal(expected_b_ext, b_ext)

        # Compute and test the extended root sequence c for size 32
        c_ext = zadoffchu.get_extended_ZF(c, 32)
        expected_c_ext = np.hstack([c, c[0:13]])
        self.assertEqual(c_ext.size, 32)
        np.testing.assert_almost_equal(expected_c_ext, c_ext)

        # Compute and test the extended root sequence c for size 64
        c_ext = zadoffchu.get_extended_ZF(c, 64)
        expected_c_ext = np.hstack([c, c, c, c[0:7]])
        self.assertEqual(c_ext.size, 64)
        np.testing.assert_almost_equal(expected_c_ext, c_ext)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Root Sequence Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class RootSequenceTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.root_seq_no_ext1 = RootSequence(root_index=25, Nzc=139)
        self.root_seq_no_ext2 = RootSequence(root_index=6, Nzc=31)
        self.root_seq1 = RootSequence(root_index=25, size=150)  # Nzc->149
        self.root_seq2 = RootSequence(root_index=12, size=150, Nzc=139)
        self.root_seq3 = RootSequence(root_index=25, size=64, Nzc=31)
        self.root_seq4 = RootSequence(root_index=6, size=64)  # Nzc->61
        self.root_seq5 = RootSequence(root_index=6, size=32)  # Nzc->31
        self.root_seq6 = RootSequence(root_index=6, size=256, Nzc=31)

        # Small available sizes are only 12 and 24.
        self.small_root_seq1 = RootSequence(root_index=15, size=12)
        self.small_root_seq2 = RootSequence(root_index=23, size=12)
        self.small_root_seq3 = RootSequence(root_index=15, size=24)
        self.small_root_seq4 = RootSequence(root_index=23, size=24)

    def test_init(self):
        with self.assertRaises(AttributeError):
            RootSequence(root_index=25)
        with self.assertRaises(AttributeError):
            RootSequence(root_index=25, size=64, Nzc=139)
        with self.assertRaises(AttributeError):
            RootSequence(root_index=25, size=3)
        with self.assertRaises(AttributeError):
            RootSequence(root_index=25, size=14)

    def test_Nzc(self):
        self.assertEqual(self.root_seq_no_ext1.Nzc, 139)
        self.assertEqual(self.root_seq_no_ext2.Nzc, 31)
        self.assertEqual(self.root_seq1.Nzc, 149)
        self.assertEqual(self.root_seq2.Nzc, 139)
        self.assertEqual(self.root_seq3.Nzc, 31)
        self.assertEqual(self.root_seq4.Nzc, 61)
        self.assertEqual(self.root_seq5.Nzc, 31)
        self.assertEqual(self.root_seq6.Nzc, 31)

    def test_size(self):
        self.assertEqual(self.root_seq1.size, 150)
        self.assertEqual(self.root_seq2.size, 150)
        self.assertEqual(self.root_seq3.size, 64)
        self.assertEqual(self.root_seq4.size, 64)
        self.assertEqual(self.root_seq5.size, 32)
        self.assertEqual(self.root_seq6.size, 256)

        self.assertEqual(self.small_root_seq1.size, 12)
        self.assertEqual(self.small_root_seq2.size, 12)
        self.assertEqual(self.small_root_seq3.size, 24)
        self.assertEqual(self.small_root_seq4.size, 24)

    def test_seq_array(self):
        # xxxxxxxxxx Small Root Sequences xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Line 15 of the first table
        expected_small_root_seq1 = np.exp(1j * (np.pi/4.0) * np.array(
            [3, -1, 1, -3, -1, -1, 1, 1, 3, 1, -1, -3]))
        # Line 23 of the first table
        expected_small_root_seq2 = np.exp(1j * (np.pi/4.0) * np.array(
            [1, 1, -1, -3, -1, -3, 1, -1, 1, 3, -1, 1]))
        # Line 15 of the second table
        expected_small_root_seq3 = np.exp(1j * (np.pi/4.0) * np.array(
            [-1, -1, 1, -3, 1, 3, -3, 1, -1, -3, -1, 3,
             1, 3, 1, -1, -3, -3, -1, -1, -3, -3, -3, -1]))
        # Line 23 of the second table
        expected_small_root_seq4 = np.exp(1j * (np.pi/4.0) * np.array(
            [-1, -1, -1, -1, 3, 3, 3, 1, 3, 3, -3, 1, 3,
             -1, 3, -1, 3, 3, -3, 3, 1, -1, 3, 3]))

        np.testing.assert_array_almost_equal(self.small_root_seq1.seq_array(),
                                             expected_small_root_seq1)
        np.testing.assert_array_almost_equal(self.small_root_seq2.seq_array(),
                                             expected_small_root_seq2)
        np.testing.assert_array_almost_equal(self.small_root_seq3.seq_array(),
                                             expected_small_root_seq3)
        np.testing.assert_array_almost_equal(self.small_root_seq4.seq_array(),
                                             expected_small_root_seq4)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Zadoff-Chu Sequences xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        expected_root__no_ext1 = calcBaseZC(139, 25)
        np.testing.assert_array_almost_equal(
            self.root_seq_no_ext1.seq_array(), expected_root__no_ext1)
        expected_root__no_ext2 = calcBaseZC(31, 6)
        np.testing.assert_array_almost_equal(
            self.root_seq_no_ext2.seq_array(), expected_root__no_ext2)

        expected_root_seq1 = calcBaseZC(149, 25)
        expected_root_seq1 = np.hstack([expected_root_seq1, expected_root_seq1[0:1]])
        np.testing.assert_array_almost_equal(
            self.root_seq1.seq_array(), expected_root_seq1)

        expected_root_seq2 = calcBaseZC(139, 12)
        expected_root_seq2 = np.hstack([expected_root_seq2, expected_root_seq2[0:11]])
        np.testing.assert_array_almost_equal(
            self.root_seq2.seq_array(), expected_root_seq2)

        expected_root_seq3 = calcBaseZC(31, 25)
        expected_root_seq3 = np.hstack(
            [expected_root_seq3, expected_root_seq3, expected_root_seq3[0:2]])
        np.testing.assert_array_almost_equal(
            self.root_seq3.seq_array(), expected_root_seq3)

        expected_root_seq4 = calcBaseZC(61, 6)
        expected_root_seq4 = np.hstack(
            [expected_root_seq4, expected_root_seq4[0:3]])
        np.testing.assert_array_almost_equal(
            self.root_seq4.seq_array(), expected_root_seq4)

        expected_root_seq5 = calcBaseZC(31, 6)
        expected_root_seq5 = np.hstack([expected_root_seq5, expected_root_seq5[0:1]])
        np.testing.assert_array_almost_equal(
            self.root_seq5.seq_array(), expected_root_seq5)

        expected_root_seq6 = calcBaseZC(31, 6)
        expected_root_seq6 = np.hstack(
            [expected_root_seq6, expected_root_seq6, expected_root_seq6,
             expected_root_seq6, expected_root_seq6, expected_root_seq6,
             expected_root_seq6, expected_root_seq6, expected_root_seq6[0:8]])
        np.testing.assert_array_almost_equal(
            self.root_seq6.seq_array(), expected_root_seq6)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SRS Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SrsUeSequenceTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        root_seq_no_ext1 = RootSequence(root_index=25, Nzc=139)
        self.user_seq_no_ext1 = SrsUeSequence(
            root_seq=root_seq_no_ext1, n_cs=3)

        root_seq_no_ext2 = RootSequence(root_index=6, Nzc=31)
        self.user_seq_no_ext2 = SrsUeSequence(
            root_seq=root_seq_no_ext2, n_cs=1)
        self.user_seq_no_ext2_other = SrsUeSequence(
            root_seq=root_seq_no_ext2, n_cs=3)

        root_seq1 = RootSequence(root_index=25, size=150, Nzc=139)
        self.user_seq1 = SrsUeSequence(root_seq=root_seq1, n_cs=7)

        root_seq2 = RootSequence(root_index=12, size=150, Nzc=139)
        self.user_seq2 = SrsUeSequence(root_seq=root_seq2, n_cs=4)

        root_seq3 = RootSequence(root_index=25, size=64, Nzc=31)
        self.user_seq3 = SrsUeSequence(root_seq=root_seq3, n_cs=1)

        root_seq4 = RootSequence(root_index=6, size=64, Nzc=31)
        self.user_seq4 = SrsUeSequence(root_seq=root_seq4, n_cs=2)

        root_seq5 = RootSequence(root_index=6, size=32, Nzc=31)
        self.user_seq5 = SrsUeSequence(root_seq=root_seq5, n_cs=3)

        root_seq6 = RootSequence(root_index=6, size=256, Nzc=31)
        self.user_seq6 = SrsUeSequence(root_seq=root_seq6, n_cs=5)

    def test_size(self):
        self.assertEqual(self.user_seq_no_ext1.size, 139)
        self.assertEqual(self.user_seq_no_ext2.size, 31)
        self.assertEqual(self.user_seq1.size, 150)
        self.assertEqual(self.user_seq2.size, 150)
        self.assertEqual(self.user_seq3.size, 64)
        self.assertEqual(self.user_seq4.size, 64)
        self.assertEqual(self.user_seq5.size, 32)
        self.assertEqual(self.user_seq6.size, 256)

    def test_seq_array(self):
        # calcBaseZC, get_srs_seq, get_extended_ZF

        expected_user_seq_no_ext1 = get_srs_seq(calcBaseZC(139, 25), 3)
        np.testing.assert_array_almost_equal(expected_user_seq_no_ext1,
                                             self.user_seq_no_ext1.seq_array())
        expected_user_seq_no_ext2 = get_srs_seq(calcBaseZC(31, 6), 1)
        np.testing.assert_array_almost_equal(expected_user_seq_no_ext2,
                                             self.user_seq_no_ext2.seq_array())
        expected_user_seq_no_ext2_other_shift = get_srs_seq(calcBaseZC(31, 6), 3)
        np.testing.assert_array_almost_equal(
            expected_user_seq_no_ext2_other_shift,
            self.user_seq_no_ext2_other.seq_array())

        expected_user_seq1 = get_srs_seq(get_extended_ZF(calcBaseZC(139, 25), 150), 7)
        np.testing.assert_array_almost_equal(self.user_seq1.seq_array(),
                                             expected_user_seq1)
        expected_user_seq2 = get_srs_seq(get_extended_ZF(calcBaseZC(139, 12), 150), 4)
        np.testing.assert_array_almost_equal(self.user_seq2.seq_array(),
                                             expected_user_seq2)
        expected_user_seq3 = get_srs_seq(get_extended_ZF(calcBaseZC(31, 25), 64), 1)
        np.testing.assert_array_almost_equal(self.user_seq3.seq_array(),
                                             expected_user_seq3)
        expected_user_seq4 = get_srs_seq(get_extended_ZF(calcBaseZC(31, 6), 64), 2)
        np.testing.assert_array_almost_equal(self.user_seq4.seq_array(),
                                             expected_user_seq4)
        expected_user_seq5 = get_srs_seq(get_extended_ZF(calcBaseZC(31, 6), 32), 3)
        np.testing.assert_array_almost_equal(self.user_seq5.seq_array(),
                                             expected_user_seq5)
        expected_user_seq6 = get_srs_seq(get_extended_ZF(calcBaseZC(31, 6), 256), 5)
        np.testing.assert_array_almost_equal(self.user_seq6.seq_array(),
                                             expected_user_seq6)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Channel Estimation Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class CazacBasedChannelEstimatorTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_estimate_channel(self):
        Nsc = 300                            # 300 subcarriers
        size = Nsc // 2
        Nzc = 139

        user1_seq = SrsUeSequence(
            RootSequence(root_index=25, size=size, Nzc=Nzc), 1)
        user2_seq = SrsUeSequence(
            RootSequence(root_index=25, size=size, Nzc=Nzc), 4)

        ue1_channel_estimator = CazacBasedChannelEstimator(user1_seq)

        speed_terminal = 3/3.6               # Speed in m/s
        fcDbl = 2.6e9                        # Central carrier frequency (in Hz)
        subcarrier_bandwidth = 15e3          # Subcarrier bandwidth (in Hz)
        wave_length = 3e8/fcDbl              # Carrier wave length
        Fd = speed_terminal / wave_length    # Doppler Frequency
        Ts = 1./(Nsc * subcarrier_bandwidth) # Sampling interval
        L = 16                               # Number of jakes taps

        jakes1 = JakesSampleGenerator(Fd, Ts, L)
        jakes2 = JakesSampleGenerator(Fd, Ts, L)

        # Create a TDL channel object for each user
        tdlchannel1 = TdlChannel(jakes1, channel_profile=COST259_TUx)
        tdlchannel2 = TdlChannel(jakes2, channel_profile=COST259_TUx)

        # Generate channel that would corrupt the transmit signal.
        tdlchannel1._generate_impulse_response(1)
        tdlchannel2._generate_impulse_response(1)

        # Get the generated impulse response
        impulse_response1 = tdlchannel1.get_last_impulse_response()
        impulse_response2 = tdlchannel2.get_last_impulse_response()

        # Get the corresponding frequency response
        freq_resp_1 = impulse_response1.get_freq_response(Nsc)
        H1 = freq_resp_1[:, 0]
        freq_resp_2 = impulse_response2.get_freq_response(Nsc)
        H2 = freq_resp_2[:, 0]

        # Sequence of the users
        r1 = user1_seq.seq_array()
        r2 = user2_seq.seq_array()

        # Received signal (in frequency domain) of user 1
        comb_indexes = np.arange(0, Nsc, 2)
        Y1 = H1[comb_indexes] * r1
        Y2 = H2[comb_indexes] * r2
        Y = Y1 + Y2

        # Calculate expected estimated channel for user 1
        y1 = np.fft.ifft(np.conj(r1) * Y, size)
        tilde_h1 = y1[0:16]
        tilde_H1 = np.fft.fft(tilde_h1, Nsc)

        # Test the CazacBasedChannelEstimator estimation
        np.testing.assert_array_almost_equal(
            ue1_channel_estimator.estimate_channel_freq_domain(Y, 16),
            tilde_H1)

        # Test if true channel and estimated channel are similar. Since the
        # channel estimation error is higher at the first and last
        # subcarriers we will test only the inner 200 subcarriers
        error = np.abs(H1[50:-50] - tilde_H1[50:-50])
        np.testing.assert_almost_equal(error/2., np.zeros(error.size), decimal=2)

    def test_estimate_channel_without_comb_pattern(self):
        Nsc = 300   # 300 subcarriers
        size = Nsc  # The size is also 300, since there is no comb pattern
        Nzc = 139

        user1_seq = SrsUeSequence(
            RootSequence(root_index=25, size=size, Nzc=Nzc), 1)
        user2_seq = SrsUeSequence(
            RootSequence(root_index=25, size=size, Nzc=Nzc), 4)

        # Set size_multiplier to 1, since we won't use the comb pattern
        ue1_channel_estimator = CazacBasedChannelEstimator(user1_seq,
                                                           size_multiplier=1)

        speed_terminal = 3/3.6               # Speed in m/s
        fcDbl = 2.6e9                        # Central carrier frequency (in Hz)
        subcarrier_bandwidth = 15e3          # Subcarrier bandwidth (in Hz)
        wave_length = 3e8/fcDbl              # Carrier wave length
        Fd = speed_terminal / wave_length    # Doppler Frequency
        Ts = 1./(Nsc * subcarrier_bandwidth) # Sampling interval
        L = 16                               # Number of jakes taps

        jakes1 = JakesSampleGenerator(Fd, Ts, L)
        jakes2 = JakesSampleGenerator(Fd, Ts, L)

        # Create a TDL channel object for each user
        tdlchannel1 = TdlChannel(jakes1, channel_profile=COST259_TUx)
        tdlchannel2 = TdlChannel(jakes2, channel_profile=COST259_TUx)

        # Generate channel that would corrupt the transmit signal.
        tdlchannel1._generate_impulse_response(1)
        tdlchannel2._generate_impulse_response(1)

        # Get the generated impulse response
        impulse_response1 = tdlchannel1.get_last_impulse_response()
        impulse_response2 = tdlchannel2.get_last_impulse_response()

        # Get the corresponding frequency response
        freq_resp_1 = impulse_response1.get_freq_response(Nsc)
        H1 = freq_resp_1[:, 0]
        freq_resp_2 = impulse_response2.get_freq_response(Nsc)
        H2 = freq_resp_2[:, 0]

        # Sequence of the users
        r1 = user1_seq.seq_array()
        r2 = user2_seq.seq_array()

        # Received signal (in frequency domain) of user 1
        Y1 = H1 * r1
        Y2 = H2 * r2
        Y = Y1 + Y2

        # Calculate expected estimated channel for user 1
        y1 = np.fft.ifft(np.conj(r1) * Y, size)
        tilde_h1 = y1[0:16]
        tilde_H1 = np.fft.fft(tilde_h1, Nsc)

        # Test the CazacBasedChannelEstimator estimation
        np.testing.assert_array_almost_equal(
            ue1_channel_estimator.estimate_channel_freq_domain(Y, 16),
            tilde_H1)

        # Test if true channel and estimated channel are similar. Since the
        # channel estimation error is higher at the first and last
        # subcarriers we will test only the inner 200 subcarriers
        error = np.abs(H1[50:-50] - tilde_H1[50:-50])
        np.testing.assert_almost_equal(error/2., np.zeros(error.size), decimal=2)

    def test_estimate_channel_multiple_rx(self):
        Nsc = 300                            # 300 subcarriers
        size = Nsc // 2
        Nzc = 139

        user1_seq = SrsUeSequence(
            RootSequence(root_index=25, size=size, Nzc=Nzc), 1)
        user2_seq = SrsUeSequence(
            RootSequence(root_index=25, size=size, Nzc=Nzc), 4)

        ue1_channel_estimator = CazacBasedChannelEstimator(user1_seq)

        speed_terminal = 3/3.6               # Speed in m/s
        fcDbl = 2.6e9                        # Central carrier frequency (in Hz)
        subcarrier_bandwidth = 15e3          # Subcarrier bandwidth (in Hz)
        wave_length = 3e8/fcDbl              # Carrier wave length
        Fd = speed_terminal / wave_length    # Doppler Frequency
        Ts = 1./(Nsc * subcarrier_bandwidth) # Sampling interval
        L = 16                               # Number of jakes taps

        # Create the fading generators and set multiple receive antennas
        jakes1 = JakesSampleGenerator(Fd, Ts, L, shape=(3, 1))
        jakes2 = JakesSampleGenerator(Fd, Ts, L, shape=(3, 1))

        # Create a TDL channel object for each user
        tdlchannel1 = TdlChannel(jakes1, channel_profile=COST259_TUx)
        tdlchannel2 = TdlChannel(jakes2, channel_profile=COST259_TUx)

        # Generate channel that would corrupt the transmit signal.
        tdlchannel1._generate_impulse_response(1)
        tdlchannel2._generate_impulse_response(1)

        # Get the generated impulse response
        impulse_response1 = tdlchannel1.get_last_impulse_response()
        impulse_response2 = tdlchannel2.get_last_impulse_response()

        # Get the corresponding frequency response
        freq_resp_1 = impulse_response1.get_freq_response(Nsc)
        H1 = freq_resp_1[:, :, 0, 0]
        freq_resp_2 = impulse_response2.get_freq_response(Nsc)
        H2 = freq_resp_2[:, :, 0, 0]

        # Sequence of the users
        r1 = user1_seq.seq_array()
        r2 = user2_seq.seq_array()

        # Received signal (in frequency domain) of user 1
        comb_indexes = np.arange(0, Nsc, 2)
        Y1 = H1[comb_indexes, :] * r1[:, np.newaxis]
        Y2 = H2[comb_indexes, :] * r2[:, np.newaxis]
        Y = Y1 + Y2

        # Calculate expected estimated channel for user 1
        y1 = np.fft.ifft(np.conj(r1[:, np.newaxis]) * Y, size, axis=0)
        tilde_h1_espected = y1[0:16]
        tilde_H1_espected = np.fft.fft(tilde_h1_espected, Nsc, axis=0)

        # Test the CazacBasedChannelEstimator estimation
        H1_estimated = ue1_channel_estimator.estimate_channel_freq_domain(Y.T, 16)
        np.testing.assert_array_almost_equal(
            H1_estimated, tilde_H1_espected.T)

        # Test if true channel and estimated channel are similar. Since the
        # channel estimation error is higher at the first and last
        # subcarriers we will test only the inner 200 subcarriers
        error = np.abs(H1[50:-50, :] - tilde_H1_espected[50:-50, :])
        np.testing.assert_almost_equal(error/2., np.zeros(error.shape), decimal=2)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx DMRS Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class DmrsUeSequenceTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        root_seq1 = RootSequence(root_index=15, size=12)
        self.dmrs_seq1 = DmrsUeSequence(
            root_seq=root_seq1, n_cs=3)
        root_seq2 = RootSequence(root_index=23, size=12)
        self.dmrs_seq2 = DmrsUeSequence(
            root_seq=root_seq2, n_cs=4)

        root_seq3 = RootSequence(root_index=15, size=24)
        self.dmrs_seq3 = DmrsUeSequence(
            root_seq=root_seq3, n_cs=3)
        root_seq4 = RootSequence(root_index=23, size=24)
        self.dmrs_seq4 = DmrsUeSequence(
            root_seq=root_seq4, n_cs=4)

        root_seq5 = RootSequence(root_index=15, size=48)
        self.dmrs_seq5 = DmrsUeSequence(
            root_seq=root_seq5, n_cs=3)
        root_seq6 = RootSequence(root_index=23, size=48)
        self.dmrs_seq6 = DmrsUeSequence(
            root_seq=root_seq6, n_cs=4)

    def test_size(self):
        self.assertEqual(self.dmrs_seq1.size, 12)
        self.assertEqual(self.dmrs_seq2.size, 12)
        self.assertEqual(self.dmrs_seq3.size, 24)
        self.assertEqual(self.dmrs_seq4.size, 24)
        self.assertEqual(self.dmrs_seq5.size, 48)
        self.assertEqual(self.dmrs_seq6.size, 48)

    def test_seq_array(self):
        expected_dmrs1 = get_dmrs_seq(RootSequence(15, 12).seq_array(), 3)
        expected_dmrs2 = get_dmrs_seq(RootSequence(23, 12).seq_array(), 4)
        expected_dmrs3 = get_dmrs_seq(RootSequence(15, 24).seq_array(), 3)
        expected_dmrs4 = get_dmrs_seq(RootSequence(23, 24).seq_array(), 4)
        expected_dmrs5 = get_dmrs_seq(RootSequence(15, 48).seq_array(), 3)
        expected_dmrs6 = get_dmrs_seq(RootSequence(23, 48).seq_array(), 4)

        np.testing.assert_array_almost_equal(
            expected_dmrs1, self.dmrs_seq1.seq_array())
        np.testing.assert_array_almost_equal(
            expected_dmrs2, self.dmrs_seq2.seq_array())
        np.testing.assert_array_almost_equal(
            expected_dmrs3, self.dmrs_seq3.seq_array())
        np.testing.assert_array_almost_equal(
            expected_dmrs4, self.dmrs_seq4.seq_array())
        np.testing.assert_array_almost_equal(
            expected_dmrs5, self.dmrs_seq5.seq_array())
        np.testing.assert_array_almost_equal(
            expected_dmrs6, self.dmrs_seq6.seq_array())


# xxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
