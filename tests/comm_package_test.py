#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the coom package.


Each module has several doctests that we run in addition to the unittests
defined here.
"""

import unittest
import doctest
import sys

import numpy as np
from comm import modulators, blockdiagonalization, ofdm, mimo, pathloss, waterfilling


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


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx OFDM Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class OfdmTestCase(unittest.TestCase):
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

        # Raises an exception if number of used subcarriers is negative
        self.assertRaises(ValueError, self.ofdm_object.set_parameters, 64, 16, 51)
        # Raises an exception if number of used subcarriers is greater than
        # the fft_size
        self.assertRaises(ValueError, self.ofdm_object.set_parameters, 64, 16, 70)
        # Raises an exception if cp_size is negative
        self.assertRaises(ValueError, self.ofdm_object.set_parameters, 64, -2, 52)
        # Raises an exception if cp_size is greater than the fft_size
        self.assertRaises(ValueError, self.ofdm_object.set_parameters, 64, 65, 52)

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
def plot_psd_OFDM_symbols():
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
    num_mod_symbols = num_bits * 1
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


# TODO: Create test cases for the other methods and classes in each module,
# even for methods that are already doctested.

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx MIMO Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class BlastTestCase(unittest.TestCase):
    """
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
        data = np.r_[0:15]
        expected_encoded_data = data.reshape(3, 5, order='F') / np.sqrt(3)
        np.testing.assert_array_almost_equal(
            self.blast_object.encode(data),
            expected_encoded_data)

    def test_decode(self):
        from util.misc import randn_c
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
        self.assertEqual(self.blast_object.calc_filter, mimo.Mimo._calcZeroForceFilter)
        channel = randn_c(4, 3)  # 3 transmitt antennas and 4 receive antennas
        received_data2 = np.dot(channel, encoded_data)
        decoded_data2 = self.blast_object.decode(received_data2, channel)
        np.testing.assert_array_almost_equal(decoded_data2, data)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test with a random channel and a MMSE filter
        self.blast_object.set_noise_var(0.00000001)  # This should use the ZF filter
        self.assertNotEqual(self.blast_object.calc_filter, mimo.Mimo._calcMMSEFilter)
        channel = randn_c(4, 3)  # 3 transmitt antennas and 4 receive antennas
        received_data3 = np.dot(channel, encoded_data)
        decoded_data3 = self.blast_object.decode(received_data3, channel)
        np.testing.assert_array_almost_equal(decoded_data3, data)


# Implement test classes for other mimo schemes

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# TODO: Test the other methods
if __name__ == "__main__":
    # plot_psd_OFDM_symbols()
    unittest.main()
