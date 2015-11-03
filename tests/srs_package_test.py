#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101

"""
Tests for the modules in the srs package.

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

from pyphysim.srs import srs
from pyphysim.srs.zadoffchu import calcBaseZC, getShiftedZF, get_extended_ZF
from pyphysim.channels.fading import TdlChannel
from pyphysim.channels.fading import COST259_TUx
from pyphysim.channels.fading_generators import JakesSampleGenerator


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SrsDoctestsTestCase(unittest.TestCase):
    """
    Test case that run all the doctests in the modules of the simulations
    package.
    """
    def test_srs_module(self):
        """Run srs module doctests"""
        doctest.testmod(srs)


class SrsRootSequenceTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.root_seq_no_ext1 = srs.SrsRootSequence(root_index=25, Nzc=139)
        self.root_seq_no_ext2 = srs.SrsRootSequence(root_index=6, Nzc=31)
        self.root_seq1 = srs.SrsRootSequence(root_index=25, Nzc=139, extend_to=150)
        self.root_seq2 = srs.SrsRootSequence(root_index=12, Nzc=139, extend_to=150)
        self.root_seq3 = srs.SrsRootSequence(root_index=25, Nzc=31, extend_to=64)
        self.root_seq4 = srs.SrsRootSequence(root_index=6, Nzc=31, extend_to=64)
        self.root_seq5 = srs.SrsRootSequence(root_index=6, Nzc=31, extend_to=32)
        self.root_seq6 = srs.SrsRootSequence(root_index=6, Nzc=31, extend_to=256)

    def test_init(self):
        with self.assertRaises(AttributeError):
            srs.SrsRootSequence(root_index=25, Nzc=139, extend_to=64)

    def test_Nzc(self):
        self.assertEqual(self.root_seq_no_ext1.Nzc, 139)
        self.assertEqual(self.root_seq_no_ext2.Nzc, 31)
        self.assertEqual(self.root_seq1.Nzc, 139)
        self.assertEqual(self.root_seq2.Nzc, 139)
        self.assertEqual(self.root_seq3.Nzc, 31)
        self.assertEqual(self.root_seq4.Nzc, 31)
        self.assertEqual(self.root_seq5.Nzc, 31)
        self.assertEqual(self.root_seq6.Nzc, 31)

    def test_size(self):
        self.assertEqual(self.root_seq1.size, 150)
        self.assertEqual(self.root_seq2.size, 150)
        self.assertEqual(self.root_seq3.size, 64)
        self.assertEqual(self.root_seq4.size, 64)
        self.assertEqual(self.root_seq5.size, 32)
        self.assertEqual(self.root_seq6.size, 256)

    def test_seq_array(self):
        expected_root__no_ext1 = calcBaseZC(139, 25)
        np.testing.assert_array_almost_equal(
            self.root_seq_no_ext1.seq_array(), expected_root__no_ext1)
        expected_root__no_ext2 = calcBaseZC(31, 6)
        np.testing.assert_array_almost_equal(
            self.root_seq_no_ext2.seq_array(), expected_root__no_ext2)

        expected_root_seq1 = calcBaseZC(139, 25)
        expected_root_seq1 = np.hstack([expected_root_seq1, expected_root_seq1[0:11]])
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

        expected_root_seq4 = calcBaseZC(31, 6)
        expected_root_seq4 = np.hstack(
            [expected_root_seq4, expected_root_seq4, expected_root_seq4[0:2]])
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


class SrsUeSequenceTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        root_seq_no_ext1 = srs.SrsRootSequence(root_index=25, Nzc=139)
        self.user_seq_no_ext1 = srs.SrsUeSequence(n_cs=3, root_seq=root_seq_no_ext1)

        root_seq_no_ext2 = srs.SrsRootSequence(root_index=6, Nzc=31)
        self.user_seq_no_ext2 = srs.SrsUeSequence(n_cs=1, root_seq=root_seq_no_ext2)
        self.user_seq_no_ext2_other = srs.SrsUeSequence(n_cs=3, root_seq=root_seq_no_ext2)

        root_seq1 = srs.SrsRootSequence(root_index=25, Nzc=139, extend_to=150)
        self.user_seq1 = srs.SrsUeSequence(n_cs=7, root_seq=root_seq1)

        root_seq2 = srs.SrsRootSequence(root_index=12, Nzc=139, extend_to=150)
        self.user_seq2 = srs.SrsUeSequence(n_cs=4, root_seq=root_seq2)

        root_seq3 = srs.SrsRootSequence(root_index=25, Nzc=31, extend_to=64)
        self.user_seq3 = srs.SrsUeSequence(n_cs=1, root_seq=root_seq3)

        root_seq4 = srs.SrsRootSequence(root_index=6, Nzc=31, extend_to=64)
        self.user_seq4 = srs.SrsUeSequence(n_cs=2, root_seq=root_seq4)

        root_seq5 = srs.SrsRootSequence(root_index=6, Nzc=31, extend_to=32)
        self.user_seq5 = srs.SrsUeSequence(n_cs=3, root_seq=root_seq5)

        root_seq6 = srs.SrsRootSequence(root_index=6, Nzc=31, extend_to=256)
        self.user_seq6 = srs.SrsUeSequence(n_cs=5, root_seq=root_seq6)

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
        # calcBaseZC, getShiftedZF, get_extended_ZF

        expected_user_seq_no_ext1 = getShiftedZF(calcBaseZC(139, 25), 3)
        np.testing.assert_array_almost_equal(expected_user_seq_no_ext1,
                                             self.user_seq_no_ext1.seq_array())
        expected_user_seq_no_ext2 = getShiftedZF(calcBaseZC(31, 6), 1)
        np.testing.assert_array_almost_equal(expected_user_seq_no_ext2,
                                             self.user_seq_no_ext2.seq_array())
        expected_user_seq_no_ext2_other_shift = getShiftedZF(calcBaseZC(31, 6), 3)
        np.testing.assert_array_almost_equal(
            expected_user_seq_no_ext2_other_shift,
            self.user_seq_no_ext2_other.seq_array())

        expected_user_seq1 = getShiftedZF(get_extended_ZF(calcBaseZC(139, 25), 150), 7)
        np.testing.assert_array_almost_equal(self.user_seq1.seq_array(),
                                             expected_user_seq1)
        expected_user_seq2 = getShiftedZF(get_extended_ZF(calcBaseZC(139, 12), 150), 4)
        np.testing.assert_array_almost_equal(self.user_seq2.seq_array(),
                                             expected_user_seq2)
        expected_user_seq3 = getShiftedZF(get_extended_ZF(calcBaseZC(31, 25), 64), 1)
        np.testing.assert_array_almost_equal(self.user_seq3.seq_array(),
                                             expected_user_seq3)
        expected_user_seq4 = getShiftedZF(get_extended_ZF(calcBaseZC(31, 6), 64), 2)
        np.testing.assert_array_almost_equal(self.user_seq4.seq_array(),
                                             expected_user_seq4)
        expected_user_seq5 = getShiftedZF(get_extended_ZF(calcBaseZC(31, 6), 32), 3)
        np.testing.assert_array_almost_equal(self.user_seq5.seq_array(),
                                             expected_user_seq5)
        expected_user_seq6 = getShiftedZF(get_extended_ZF(calcBaseZC(31, 6), 256), 5)
        np.testing.assert_array_almost_equal(self.user_seq6.seq_array(),
                                             expected_user_seq6)


# TODO: finish implementation
class SrsChannelEstimatorTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_estimate_channel(self):
        user1_seq = srs.SrsUeSequence(
            1,
            srs.SrsRootSequence(root_index=25, Nzc=139, extend_to=150))
        user2_seq = srs.SrsUeSequence(
            4,
            srs.SrsRootSequence(root_index=25, Nzc=139, extend_to=150))

        ue1_channel_estimator = srs.SrsChannelEstimator(user1_seq)

        Nsc = 300                         # 300 subcarriers
        speed_terminal = 3/3.6             # Speed in m/s
        fcDbl = 2.6e9                     # Central carrier frequency (in Hz)
        subcarrier_bandwidth = 15e3          # Subcarrier bandwidth (in Hz)
        wave_length = 3e8/fcDbl             # Carrier wave length
        Fd = speed_terminal / wave_length    # Doppler Frequency
        Ts = 1./(Nsc * subcarrier_bandwidth) # Sampling interval
        L = 16                            # Number of jakes taps

        jakes1 = JakesSampleGenerator(Fd, Ts, L)
        jakes2 = JakesSampleGenerator(Fd, Ts, L)

        # Create a TDL channel object for each user
        tdlchannel1 = TdlChannel(jakes1,
                                 tap_powers_dB=COST259_TUx.tap_powers_dB,
                                 tap_delays=COST259_TUx.tap_delays)
        tdlchannel2 = TdlChannel(jakes2,
                                 tap_powers_dB=COST259_TUx.tap_powers_dB,
                                 tap_delays=COST259_TUx.tap_delays)

        # Compute the fading map for each user
        fadingmap1 = tdlchannel1.get_fading_map(1)
        fadingmap2 = tdlchannel2.get_fading_map(1)

        freq_resp_1 = tdlchannel1.get_channel_freq_response(fadingmap1, Nsc)
        H1 = freq_resp_1[:, 0]

        freq_resp_2 = tdlchannel2.get_channel_freq_response(fadingmap2, Nsc)
        H2 = freq_resp_2[:, 0]

        # Sequence of user 1
        r1 = user1_seq.seq_array()
        # Sequence of user 2
        r2 = user2_seq.seq_array()

        # Received signal (in frequency domain) of user 1
        comb_indexes = np.arange(0, Nsc, 2)
        Y1 = H1[comb_indexes] * r1
        Y2 = H2[comb_indexes] * r2

        Y = Y1 + Y2

        # Calculate expected estimated channel for user 1
        y1 = np.fft.ifft(np.conj(r1) * Y, 150)
        tilde_h1 = y1[0:16]
        tilde_H1 = np.fft.fft(tilde_h1, Nsc)

        # Test the SrsChannelEstimator estimation
        np.testing.assert_array_almost_equal(ue1_channel_estimator.estimate_channel(Y),
                                             tilde_H1)

        # Test if true channel and estimated channel are similar. Since the
        # channel estimation error is higher at the first and last
        # subcarriers we will test only the inner 200 subcarriers
        error = np.abs(H1[50:-50] - tilde_H1[50:-50])
        np.testing.assert_almost_equal(error/2., np.zeros(error.size), decimal=2)


# xxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
