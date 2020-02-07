#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,E0611
"""
Tests for the modules in the mimo package.

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
import math
from pyphysim.modulators import fundamental, ofdm
from pyphysim.modulators import OFDM
from pyphysim.util.misc import randn_c
from pyphysim.channels import fading, fading_generators


# UPDATE THIS CLASS if another module is added to the comm package
# noinspection PyMethodMayBeStatic
class MimoDoctestsTestCase(unittest.TestCase):
    """Test case that run all the doctests in the modules of the comm
    package. """
    def test_modulators(self):
        """Run doctests in the modulators module."""
        doctest.testmod(fundamental)

    def test_ofdm(self, ):
        """Run doctests in the ofdm module."""
        doctest.testmod(ofdm)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Modulators Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class PSKTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.psk_obj = fundamental.PSK(4)
        self.psk_obj2 = fundamental.PSK(8)

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
            np.array([
                1. + 0.j, 0.70710678 + 0.70710678j, -0.70710678 + 0.70710678j,
                0. + 1.j, 0.70710678 - 0.70710678j, 0. - 1.j, -1. + 0.j,
                -0.70710678 - 0.70710678j
            ]))

    def test_set_phase_offset(self):
        self.psk_obj.setPhaseOffset(np.pi / 4.)

        np.testing.assert_array_almost_equal(
            self.psk_obj.symbols,
            np.array([
                0.70710678 + 0.70710678j, -0.70710678 + 0.70710678j,
                -0.70710678 - 0.70710678j, 0.70710678 - 0.70710678j
            ]))

    def test_calc_theoretical_SER_and_BER(self):
        SNR_values = np.array([-5, 0, 5, 10])

        # xxxxxxxxxx Test for the 4-PSK xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        theoretical_ser = np.array(
            [0.57388349, 0.31731051, 0.07535798, 0.0015654])
        np.testing.assert_array_almost_equal(
            self.psk_obj.calcTheoreticalSER(SNR_values), theoretical_ser)

        # self.psk_obj.calcTheoreticalBER
        np.testing.assert_array_almost_equal(
            self.psk_obj.calcTheoreticalBER(SNR_values), theoretical_ser / 2.)

        # xxxxxxxxxx Test for the 8 PSK xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        theoretical_ser2 = np.array(
            [0.76087121, 0.58837243, 0.33584978, 0.08700502])
        np.testing.assert_array_almost_equal(
            self.psk_obj2.calcTheoreticalSER(SNR_values), theoretical_ser2)

        np.testing.assert_array_almost_equal(
            self.psk_obj2.calcTheoreticalBER(SNR_values),
            theoretical_ser2 / 3.)

    # The calcTheoreticalPER method is defined in the Modulators class, but
    # can only be tested in a subclass, since it depends on the
    # calcTheoreticalBER method. Therefore, we chose to test it here.
    def test_calc_theoretical_PER(self):
        L1 = 50
        L2 = 120
        SNRs = np.array([10, 13])
        # The BER for SNR values of 10 and 13 are 7.82701129e-04 and
        # 3.96924840e-06, respectively
        BER = self.psk_obj.calcTheoreticalBER(SNRs)

        expected_PER1 = (1 - BER)**L1
        expected_PER1 = 1 - expected_PER1

        expected_PER2 = (1 - BER)**L2
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
        awgn_noise = randn_c(20, ) * 1e-2

        input_data = np.random.randint(0, 4, 20)
        modulated_data = self.psk_obj.modulate(input_data)
        demodulated_data = self.psk_obj.demodulate(modulated_data + awgn_noise)

        np.testing.assert_array_equal(input_data, demodulated_data)

        input_data2 = np.random.randint(0, 8, 20)
        modulated_data2 = self.psk_obj2.modulate(input_data2)
        demodulated_data2 = self.psk_obj2.demodulate(modulated_data2 +
                                                     awgn_noise)
        np.testing.assert_array_equal(input_data2, demodulated_data2)

        # Test if an exception is raised for invalid arguments
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            self.psk_obj.modulate(4)
        with self.assertRaises(ValueError):
            self.psk_obj2.modulate(10)


class BPSKTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.bpsk_obj = fundamental.BPSK()

    def test_name(self):
        self.assertEqual(self.bpsk_obj.name, "BPSK")

    def test_constellation(self):
        self.assertEqual(self.bpsk_obj.M, 2)
        self.assertAlmostEqual(self.bpsk_obj.K, 1)
        np.testing.assert_array_almost_equal(self.bpsk_obj.symbols,
                                             np.array([1, -1]))

    def test_calc_theoretical_SER_and_BER(self):
        SNR_values = np.array([-5, 0, 5, 10])

        theoretical_ser = np.array(
            [2.13228018e-01, 7.86496035e-02, 5.95386715e-03, 3.87210822e-06])
        np.testing.assert_array_almost_equal(
            self.bpsk_obj.calcTheoreticalSER(SNR_values), theoretical_ser)

        # The SER and the BER are equal for BPSK modulation
        np.testing.assert_array_almost_equal(
            self.bpsk_obj.calcTheoreticalBER(SNR_values), theoretical_ser)

    def test_modulate_and_demodulate(self):
        input_data = np.random.randint(0, 2, 20)
        modulated_data = self.bpsk_obj.modulate(input_data)

        awgn_noise = randn_c(20, ) * 1e-2

        demodulated_data = self.bpsk_obj.demodulate(modulated_data +
                                                    awgn_noise)
        np.testing.assert_array_equal(input_data, demodulated_data)

        # Test if an exception is raised for invalid arguments
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            self.bpsk_obj.modulate(2)


class QAMTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.qam_obj = fundamental.QAM(4)
        self.qam_obj2 = fundamental.QAM(16)
        self.qam_obj3 = fundamental.QAM(64)

    def test_invalid_QAM_size(self):
        with self.assertRaises(ValueError):
            fundamental.QAM(32)
        with self.assertRaises(ValueError):
            fundamental.QAM(63)

    def test_constellation(self):
        self.assertEqual(self.qam_obj.M, 4)
        self.assertAlmostEqual(self.qam_obj.K, 2)
        np.testing.assert_array_almost_equal(
            self.qam_obj.symbols,
            np.array([
                -0.70710678 + 0.70710678j, 0.70710678 + 0.70710678j,
                -0.70710678 - 0.70710678j, 0.70710678 - 0.70710678j
            ]))

        self.assertEqual(self.qam_obj2.M, 16)
        self.assertAlmostEqual(self.qam_obj2.K, 4)
        np.testing.assert_array_almost_equal(
            self.qam_obj2.symbols,
            np.array([
                -0.94868330 + 0.9486833j, -0.31622777 + 0.9486833j,
                0.94868330 + 0.9486833j, 0.31622777 + 0.9486833j,
                -0.94868330 + 0.31622777j, -0.31622777 + 0.31622777j,
                0.94868330 + 0.31622777j, 0.31622777 + 0.31622777j,
                -0.94868330 - 0.9486833j, -0.31622777 - 0.9486833j,
                0.94868330 - 0.9486833j, 0.31622777 - 0.9486833j,
                -0.94868330 - 0.31622777j, -0.31622777 - 0.31622777j,
                0.94868330 - 0.31622777j, 0.31622777 - 0.31622777j
            ]))

        self.assertEqual(self.qam_obj3.M, 64)
        self.assertAlmostEqual(self.qam_obj3.K, 6)
        np.testing.assert_array_almost_equal(
            self.qam_obj3.symbols,
            np.array([
                -1.08012345 + 1.08012345j, -0.77151675 + 1.08012345j,
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
                0.46291005 - 0.15430335j, 0.15430335 - 0.15430335j
            ]))

    def test_calc_theoretical_SER_and_BER(self):
        SNR_values = np.array([0, 5, 10, 15, 20])

        # xxxxxxxxxx Test for 4-QAM xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        theoretical_ser = np.array([
            2.92139018e-01, 7.39382701e-02, 1.56478964e-03, 1.87220798e-08, 0
        ])
        np.testing.assert_array_almost_equal(
            self.qam_obj.calcTheoreticalSER(SNR_values), theoretical_ser)

        theoretical_ber = np.array([
            1.58655254e-01, 3.76789881e-02, 7.82701129e-04, 9.36103999e-09,
            7.61985302e-24
        ])
        np.testing.assert_array_almost_equal(
            self.qam_obj.calcTheoreticalBER(SNR_values), theoretical_ber)

        # xxxxxxxxxx Test for 16-QAM xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        theoretical_ser2 = np.array([
            7.40960364e-01, 5.37385132e-01, 2.22030850e-01, 1.77818422e-02,
            1.16162909e-05
        ])
        np.testing.assert_array_almost_equal(
            self.qam_obj2.calcTheoreticalSER(SNR_values), theoretical_ser2)

        theoretical_ber2 = np.array([
            2.45520317e-01, 1.59921014e-01, 5.89872026e-02, 4.46540036e-03,
            2.90408116e-06
        ])
        np.testing.assert_array_almost_equal(
            self.qam_obj2.calcTheoreticalBER(SNR_values), theoretical_ber2)
        # xxxxxxxxxx Test for 64-QAM xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        theoretical_ser3 = np.array(
            [0.92374224, 0.84846895, 0.67382633, 0.3476243, 0.05027041])
        np.testing.assert_array_almost_equal(
            self.qam_obj3.calcTheoreticalSER(SNR_values), theoretical_ser3)

        theoretical_ber3 = np.array(
            [0.24128398, 0.2035767, 0.14296128, 0.06410074, 0.00848643])
        np.testing.assert_array_almost_equal(
            self.qam_obj3.calcTheoreticalBER(SNR_values), theoretical_ber3)

    def test_modulate_and_demodulate(self):
        awgn_noise = randn_c(20, ) * 1e-2

        # xxxxxxxxxx Test for 4-QAM xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        input_data = np.random.randint(0, 4, 20)
        modulated_data = self.qam_obj.modulate(input_data)
        demodulated_data = self.qam_obj.demodulate(modulated_data + awgn_noise)
        np.testing.assert_array_equal(input_data, demodulated_data)

        # xxxxxxxxxx Test for 16-QAM xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        input_data2 = np.random.randint(0, 16, 20)
        modulated_data2 = self.qam_obj2.modulate(input_data2)
        demodulated_data2 = self.qam_obj2.demodulate(modulated_data2 +
                                                     awgn_noise)
        np.testing.assert_array_equal(input_data2, demodulated_data2)

        # xxxxxxxxxx Test for 64-QAM xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        input_data3 = np.random.randint(0, 64, 20)
        modulated_data3 = self.qam_obj3.modulate(input_data3)
        demodulated_data3 = self.qam_obj3.demodulate(modulated_data3 +
                                                     awgn_noise)
        np.testing.assert_array_equal(input_data3, demodulated_data3)

        # Test if an exception is raised for invalid arguments
        with self.assertRaises(ValueError):
            self.qam_obj.modulate(4)

        with self.assertRaises(ValueError):
            self.qam_obj2.modulate(16)

        with self.assertRaises(ValueError):
            self.qam_obj3.modulate(65)


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
        # 52 elements -> exactly the number of used subcarriers in the OFDM
        # object
        input_signal = np.r_[1:53]

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Lets first test the case where we use exactly 52 subcarriers from
        # the 64 available subcarriers. That is, no zeropadding is needed.
        (zeropad,
         num_ofdm_symbols) = self.ofdm_object._calc_zeropad(input_signal.size)
        self.assertEqual(zeropad, 0)
        self.assertEqual(num_ofdm_symbols, 1)

        expected_data = np.array([[
            0., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36., 37., 38.,
            39., 40., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50., 51.,
            52., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 4.,
            5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
            19., 20., 21., 22., 23., 24., 25., 26.
        ]])
        np.testing.assert_array_equal(
            self.ofdm_object._prepare_input_signal(input_signal),
            expected_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now lets change the number of used subcarriers and repeat the
        # tests so that the case when zeropad is performed is also tested.
        self.ofdm_object.num_used_subcarriers = 60
        (zeropad,
         num_ofdm_symbols) = self.ofdm_object._calc_zeropad(input_signal.size)
        # We used 60 subcarriers, but we have 52 elements -> We need to add
        # 8 zeros at the end of the input data
        self.assertEqual(zeropad, 8)
        # But we still have only one OFDM symbol
        self.assertEqual(num_ofdm_symbols, 1)

        expected_data2 = np.array([[
            0.,
            31.,
            32.,
            33.,
            34.,
            35.,
            36.,
            37.,
            38.,
            39.,
            40.,
            41.,
            42.,
            43.,
            44.,
            45.,
            46.,
            47.,
            48.,
            49.,
            50.,
            51.,
            52.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            1.,
            2.,
            3.,
            4.,
            5.,
            6.,
            7.,
            8.,
            9.,
            10.,
            11.,
            12.,
            13.,
            14.,
            15.,
            16.,
            17.,
            18.,
            19.,
            20.,
            21.,
            22.,
            23.,
            24.,
            25.,
            26.,
            27.,
            28.,
            29.,
            30.,
        ]])
        np.testing.assert_array_equal(
            self.ofdm_object._prepare_input_signal(input_signal),
            expected_data2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now lets test the case when we use all subcarriers (but still
        # with zeropadding)
        self.ofdm_object.num_used_subcarriers = 64
        (zeropad,
         num_ofdm_symbols) = self.ofdm_object._calc_zeropad(input_signal.size)
        self.assertEqual(zeropad, 12)

        # But we still have only one OFDM symbol
        self.assertEqual(num_ofdm_symbols, 1)

        # Notice that in this case the DC subcarrier is used
        expected_data3 = np.array([[
            33., 34., 35., 36., 37., 38., 39., 40., 41., 42., 43., 44., 45.,
            46., 47., 48., 49., 50., 51., 52., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
            13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.,
            26., 27., 28., 29., 30., 31., 32.
        ]])
        np.testing.assert_array_equal(
            self.ofdm_object._prepare_input_signal(input_signal),
            expected_data3)

    def test_prepare_decoded_signal(self):
        input1 = np.r_[1:105]
        input2 = self.ofdm_object._prepare_input_signal(input1)
        output = self.ofdm_object._prepare_decoded_signal(input2)
        np.testing.assert_array_equal(output, input1)

    # noinspection PyPep8
    def test_calculate_power_scale(self):
        expected_power_scale = float(self.ofdm_object.fft_size) \
                               * (float(self.ofdm_object.fft_size) /
                                  (self.ofdm_object.num_used_subcarriers +
                                   self.ofdm_object.cp_size))

        self.assertAlmostEqual(expected_power_scale,
                               self.ofdm_object._calculate_power_scale())

        self.ofdm_object.fft_size = 1024.
        self.ofdm_object.cp_size = 100.
        self.ofdm_object.num_used_subcarriers = 900.
        self.assertAlmostEqual(1024. * (1024. / (900. + 100.)),
                               self.ofdm_object._calculate_power_scale())

    def test_modulate(self):
        # Exactly two OFDM symbols (with 52 used subcarriers)
        input_signal = np.r_[1:105]

        # xxxxx First lets try without cyclic prefix xxxxxxxxxxxxxxxxxxxxxx
        self.ofdm_object.set_parameters(64, 0, 52)
        (zeropad,
         num_ofdm_symbols) = self.ofdm_object._calc_zeropad(input_signal.size)

        self.assertEqual(zeropad, 0)

        # But we still have only one OFDM symbol
        self.assertEqual(num_ofdm_symbols, 2)

        input_ifft = self.ofdm_object._prepare_input_signal(input_signal)
        power_scale = self.ofdm_object._calculate_power_scale()
        expected_data_no_power_scale = np.hstack([
            np.fft.ifft(input_ifft[0, :]),
            np.fft.ifft(input_ifft[1, :]),
        ])
        expected_data = math.sqrt(power_scale) * expected_data_no_power_scale

        np.testing.assert_array_almost_equal(
            self.ofdm_object.modulate(input_signal), expected_data)

        # Let's test each OFDM symbol individually
        np.testing.assert_array_almost_equal(
            self.ofdm_object.modulate(input_signal[0:52]), expected_data[0:64])

        np.testing.assert_array_almost_equal(
            self.ofdm_object.modulate(input_signal[52:]), expected_data[64:])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Now lets test with a cyclic prefix xxxxxxxxxxxxxxxxxxxxxxxx
        self.ofdm_object.set_parameters(64, 4, 52)

        # input_ifft2 =
        #     self.ofdm_object._prepare_input_signal(input_signal[0:52])
        power_scale2 = self.ofdm_object._calculate_power_scale()
        expected_data_no_power_scale2 = np.hstack([
            np.fft.ifft(input_ifft[0, :]),
        ])
        expected_data2 = math.sqrt(
            power_scale2) * expected_data_no_power_scale2
        expected_data2 = np.hstack(
            [expected_data2[-self.ofdm_object.cp_size:], expected_data2[0:64]])

        np.testing.assert_array_almost_equal(
            self.ofdm_object.modulate(input_signal[0:52]), expected_data2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_demodulate(self):
        # xxxxx First lets try without cyclic prefix xxxxxxxxxxxxxxxxxxxxxx
        # Exactly two OFDM symbols (with 52 used subcarriers)
        input_signal = np.r_[1:105] + 1j * np.r_[1:105]
        ":type: np.ndarray"

        # xxxxx First lets try without cyclic prefix xxxxxxxxxxxxxxxxxxxxxx
        self.ofdm_object.set_parameters(64, 0, 52)
        modulated_ofdm_symbols = self.ofdm_object.modulate(input_signal)

        demodulated_symbols = self.ofdm_object.demodulate(
            modulated_ofdm_symbols)

        np.testing.assert_array_equal(demodulated_symbols.round(),
                                      input_signal)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Now lets test with a cyclic prefix xxxxxxxxxxxxxxxxxxxxxxxx
        # Exactly two OFDM symbols (with 52 used subcarriers)
        input_signal2 = np.r_[1:105] + 1j * np.r_[1:105]
        ":type: np.ndarray"

        self.ofdm_object.set_parameters(64, 16, 52)
        modulated_ofdm_symbols2 = self.ofdm_object.modulate(input_signal2)

        demodulated_symbols2 = self.ofdm_object.demodulate(
            modulated_ofdm_symbols2)

        np.testing.assert_array_equal(demodulated_symbols2.round(),
                                      input_signal2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Now lets test the case with zeropadding xxxxxxxxxxxxxxxxxxx
        # Exactly two OFDM symbols (with 52 used subcarriers)
        input_signal3 = np.r_[1:110] + 1j * np.r_[1:110]
        ":type: np.ndarray"

        self.ofdm_object.set_parameters(64, 16, 52)
        modulated_ofdm_symbols3 = self.ofdm_object.modulate(input_signal3)

        demodulated_symbols3 = self.ofdm_object.demodulate(
            modulated_ofdm_symbols3)
        # OFDM will not remove the zeropadding therefore we need to do it
        # manually
        demodulated_symbols3 = demodulated_symbols3[0:109]
        np.testing.assert_array_equal(demodulated_symbols3.round(),
                                      input_signal3)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# noinspection PyMethodMayBeStatic
class OfdmOneTapEqualizerTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_equalize_data(self):
        num_of_subcarriers = 24
        ofdm_obj = OFDM(num_of_subcarriers, cp_size=8)
        onetap_equalizer = ofdm.OfdmOneTapEqualizer(ofdm_obj)
        qam_obj = fundamental.QAM(4)

        # Input data
        data = np.random.randint(0, 4, size=2 * num_of_subcarriers)

        # Modulate with QAM
        symbols = qam_obj.modulate(data)

        # Modulate with OFDM
        transmit_data = ofdm_obj.modulate(symbols)

        # xxxxxxxxxx Channel Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Calculate the actual bandwidth that we will use
        bandwidth = 55e3 * num_of_subcarriers
        Fd = 50  # Doppler frequency (in Hz)
        Ts = 1. / bandwidth  # Sampling interval (in seconds)
        NRays = 16  # Number of rays for the Jakes model

        jakes = fading_generators.JakesSampleGenerator(Fd,
                                                       Ts,
                                                       NRays,
                                                       shape=None)

        tdlchannel = fading.TdlChannel(
            jakes,
            tap_powers_dB=fading.COST259_TUx.tap_powers_dB,
            tap_delays=fading.COST259_TUx.tap_delays)

        channel_memory = tdlchannel.num_taps_with_padding - 1
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Pass the signal through the channel xxxxxxxxxxxxxxxxxx
        received_signal = tdlchannel.corrupt_data(transmit_data)

        # Impulse response used to transmit the signal
        last_impulse_response = tdlchannel.get_last_impulse_response()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx OFDM Reception xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Demodulate with OFDM
        received_demodulated_data = ofdm_obj.demodulate(
            received_signal[:-channel_memory])

        received_equalized_data = onetap_equalizer.equalize_data(
            received_demodulated_data, last_impulse_response)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Demodulation and testing xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        demodulated_received_data = qam_obj.demodulate(received_equalized_data)
        np.testing.assert_array_equal(data, demodulated_received_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Power Spectral Density
def plot_psd_OFDM_symbols():  # pragma: no cover
    """Plot the power spectral density of OFDM modulated symbols.

    This function is not used in any unittest, but it is interesting to
    visualize that the modulate method of the OFDM class is working as it
    should.
    """
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
    ip_bits = np.random.randint(0, 2, num_bits)
    # Number of modulated symbols
    # num_mod_symbols = num_bits * 1
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
    Pxx, W = plt.psd(output, NFFT=fft_size, Fs=fsMHz)
    # [Pxx,W] = pwelch(output,[],[],4096,20);
    plt.plot(W, 10 * np.log10(Pxx))
    plt.xlabel('frequency, MHz')
    plt.ylabel('power spectral density')
    plt.title('Transmit spectrum OFDM (based on 802.11a)')
    plt.show()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    # plot_psd_OFDM_symbols()
    unittest.main()
