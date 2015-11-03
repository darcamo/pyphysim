#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101,E0611

"""
Tests for the modules in the channels package.

Each module has several doctests that we run in addition to the unittests
defined here.
"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
try:
    parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    sys.path.append(parent_dir)
except NameError:               # pragma: no cover
    sys.path.append('../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import unittest
import doctest
import warnings
from pyphysim import channels
import math
import numpy as np
from scipy.linalg import block_diag
from pyphysim.channels import noise, fading_generators, fading, multiuser, pathloss
from pyphysim.comm import blockdiagonalization
from pyphysim.ia.algorithms import ClosedFormIASolver
from pyphysim.util.conversion import single_matrix_to_matrix_of_matrices, dB2Linear
from pyphysim.util.misc import randn_c, least_right_singular_vectors


class ChannelsDoctestsTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_channels(self):
        """Run doctests in the channels module."""
        doctest.testmod(channels)

    def test_noise(self):
        """Run doctests in the noise module."""
        doctest.testmod(noise)

    def test_fading_generators(self):
        """Run doctests in the fading_generators module."""
        doctest.testmod(fading_generators)

    def test_pathloss(self):
        """Run doctests in the pathloss module."""
        doctest.testmod(pathloss)

    def test_multiuser(self):
        """Run doctests in the multiuser module."""
        doctest.testmod(multiuser)


class ModuleFunctionsTestCase(unittest.TestCase):
    def test_calc_thermal_noise_power(self):
        T = 23  # Temperature in degrees

        # Test for 1Hz
        delta_f = 1  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(
            T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -174, places=0)

        # Test for 10Hz
        delta_f = 10  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(
            T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -164, places=0)

        # Test for 100Hz
        delta_f = 100  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(
            T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -154, places=0)

        # Test for 200kHz
        delta_f = 200e3  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(
            T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -121, places=0)

        # Test for 1MHz
        delta_f = 1e6  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(
            T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -114, places=0)

        # Test for 5MHz
        delta_f = 5e6  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(
            T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -107, places=0)

        # Test for 20MHz
        delta_f = 20e6  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(
            T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -101, places=0)

    def test_generate_jakes_samples(self):
        Fd = 5     # Doppler frequency (in Hz)
        Ts = 1e-3  # Sampling interval (in seconds)
        N = 1000   # Number of samples
        NRays = 8  # Number of rays for the Jakes model

        # Test generating channel samples for a SISO scenario
        h = fading_generators.generate_jakes_samples(Fd, Ts, N, NRays)
        self.assertEqual(h.size, 1000)
        self.assertEqual(h.shape, (1000,))

        h2 = fading_generators.generate_jakes_samples(Fd, Ts, N, NRays, shape=(4, 3))
        self.assertEqual(h2.shape, (4, 3, N))

        # Test with a given RandomState object.
        RS = np.random.RandomState()
        h3 = fading_generators.generate_jakes_samples(Fd, Ts, N, NRays, shape=(3, 2),
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


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Fading_generators Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class RayleighSampleGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.obj1 = fading_generators.RayleighSampleGenerator()
        self.obj2 = fading_generators.RayleighSampleGenerator(shape=3)
        self.obj3 = fading_generators.RayleighSampleGenerator(shape=(4, 3))

    # Here we only test if the shape of the generated matrix is correct
    # TODO: check statistics of the generated matrix
    def test_generate_next_samples(self):
        self.assertEqual(self.obj1.get_samples().shape, (1,))
        self.assertEqual(self.obj2.get_samples().shape, (3,))
        self.assertEqual(self.obj3.get_samples().shape, (4, 3))


class JakesSampleGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        Fd = 5     # Doppler frequency (in Hz)
        Ts = 1e-3  # Sampling interval (in seconds)
        NRays = 8  # Number of rays for the Jakes model

        self.obj = fading_generators.JakesSampleGenerator(Fd, Ts, NRays)
        self.obj2 = fading_generators.JakesSampleGenerator(Fd, Ts, NRays, shape=(3, 2))

    def test_set_shape(self):
        NRays = 8
        self.assertIsNone(self.obj.shape)
        self.obj.shape = [3, 2]
        np.testing.assert_array_equal(self.obj.shape, [3, 2])

        # The first dimension is equal to the number of rays of the Jakes
        # generator. The last dimension is set to 1 to allow broadcast with
        # the time dimension later.
        np.testing.assert_array_equal(self.obj._phi_l.shape, [NRays, 3, 2, 1])
        np.testing.assert_array_equal(self.obj._psi_l.shape, [NRays, 3, 2, 1])

        # Now set the shape to None (SISO case)
        self.obj.shape = None
        self.assertIsNone(self.obj.shape)
        np.testing.assert_array_equal(self.obj._phi_l.shape, [NRays, 1])
        np.testing.assert_array_equal(self.obj._psi_l.shape, [NRays, 1])

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


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Fading Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class TdlChannelProfileTestCase(unittest.TestCase):
    def test_constructor(self):
        prof1 = fading.TdlChannelProfile(np.array([0, -3, -10]),
                                         np.array([0, 1e-3, 5e-4]))
        prof2 = fading.TdlChannelProfile(np.array([0, -3, -10, -30]),
                                         np.array([0, 1e-3, 5e-4, 1e-5]),
                                         name='some name')
        self.assertEqual(prof1.name, 'custom')
        self.assertEqual(prof2.name, 'some name')

    def test_properties(self):
        tu = fading.COST259_TUx
        ra = fading.COST259_RAx
        ht = fading.COST259_HTx

        # CHeck the number of taps in the profiles
        self.assertEqual(tu.num_taps, 20)
        self.assertEqual(ra.num_taps, 10)
        self.assertEqual(ht.num_taps, 20)

        # Check the names of the profiles
        self.assertAlmostEqual(tu.name, 'COST259_TU')
        self.assertAlmostEqual(ra.name, 'COST259_RA')
        self.assertAlmostEqual(ht.name, 'COST259_HT')

        # Check if the mean excess delays of the profiles are correct
        self.assertAlmostEqual(tu.mean_excess_delay, 5.00428208169e-07)
        self.assertAlmostEqual(ra.mean_excess_delay, 8.85375638731e-08)
        self.assertAlmostEqual(ht.mean_excess_delay, 8.93899719191e-07)

        # Check if the rms delay spreads of the profiles are correct
        self.assertAlmostEqual(tu.rms_delay_spread, 5.000561653134637e-07)
        self.assertAlmostEqual(ra.rms_delay_spread, 1.0000823342626581e-07)
        self.assertAlmostEqual(ht.rms_delay_spread, 3.039829880190327e-06)

        # Check if an exception is raised if we try to change the delays or
        # the powers of the taps.
        with self.assertRaises(ValueError):
            tu.tap_powers_dB[0] = 30
        with self.assertRaises(ValueError):
            tu.tap_delays[0] = 30
        with self.assertRaises(ValueError):
            tu.tap_powers_linear[0] = 30

        # Check the tap power and delay values
        np.testing.assert_array_almost_equal(
            tu.tap_powers_dB,
            np.array([-5.7, -7.6, -10.1, -10.2, -10.2, -11.5, -13.4, -16.3,
                      -16.9, -17.1, -17.4, -19, -19, -19.8, -21.5, -21.6,
                      -22.1, -22.6, -23.5, -24.3]))
        np.testing.assert_array_almost_equal(
            tu.tap_delays,
            np.array([0, 217, 512, 514, 517, 674, 882, 1230, 1287, 1311, 1349,
                      1533, 1535, 1622, 1818, 1836, 1884, 1943, 2048, 2140]) * 1e-9)


        np.testing.assert_array_almost_equal(
            ra.tap_powers_dB,
            np.array([-5.2, -6.4, -8.4, -9.3, -10.0, -13.1, -15.3, -18.5, -20.4, -22.4]))
        np.testing.assert_array_almost_equal(
            ra.tap_delays,
            np.array([0., 42., 101., 129., 149., 245., 312., 410., 469., 528]) * 1e-9)


        np.testing.assert_array_almost_equal(
            ht.tap_powers_dB,
            np.array([-3.6, -8.9, -10.2, -11.5, -11.8, -12.7, -13.0, -16.2,
                      -17.3, -17.7, -17.6, -22.7, -24.1, -25.8, -25.8, -26.2,
                      -29.0, -29.9, -30.0, -30.7]))
        np.testing.assert_array_almost_equal(
            ht.tap_delays,
            np.array([0., 356., 441., 528., 546., 609., 625., 842., 916., 941.,
                      15000., 16172., 16492., 16876., 16882., 16978., 17615.,
                      17827., 17849., 18016.]) * 1e-9)
    def test_discretize(self):
        maxSystemBand = 40e6  # 40 MHz bandwidth
        # Number of subcarriers in this bandwidth
        max_num_of_subcarriers = math.floor(maxSystemBand/15e3)
        # Find the maximum FFT size we can use which is below than or equal
        # to maxNumOfSubcarriersInt
        max_num_of_subcarriers = int(
            2 ** math.floor(math.log(max_num_of_subcarriers, 2)))
        # Calculate the actual bandwidth that we will use
        bandwidth = 15e3 * max_num_of_subcarriers

        Ts = 1. / bandwidth  # Sampling interval (in seconds)

        tu = fading.COST259_TUx
        tu_discretized = tu.get_discretize_profile(Ts)

        # xxxxx Calculate the expected discretized tap powers and delays xx
        # The COST259_TUx 20 taps. For the Ts calculated here the indexes
        # of the discretized taps (from the original ones) are
        # [ 0  7 16 16 16 21 27 38 40 40 41 47 47 50 56 56 58 60 63 66]
        # Note that some taps have the same indexes and this will be summed
        # toguether.
        tap_powers_linear = tu.tap_powers_linear
        # The TDL class will normalized the tap powers so that the channel
        # has unit power.
        tap_powers_linear = tap_powers_linear/np.sum(tap_powers_linear)

        expected_discretized_tap_powers_linear = np.zeros(15)

        expected_discretized_tap_powers_linear[0] += tap_powers_linear[0]
        expected_discretized_tap_powers_linear[1] += tap_powers_linear[1]
        expected_discretized_tap_powers_linear[2] += tap_powers_linear[2]
        expected_discretized_tap_powers_linear[2] += tap_powers_linear[3]
        expected_discretized_tap_powers_linear[2] += tap_powers_linear[4]
        expected_discretized_tap_powers_linear[3] += tap_powers_linear[5]
        expected_discretized_tap_powers_linear[4] += tap_powers_linear[6]
        expected_discretized_tap_powers_linear[5] += tap_powers_linear[7]
        expected_discretized_tap_powers_linear[6] += tap_powers_linear[8]
        expected_discretized_tap_powers_linear[6] += tap_powers_linear[9]
        expected_discretized_tap_powers_linear[7] += tap_powers_linear[10]
        expected_discretized_tap_powers_linear[8] += tap_powers_linear[11]
        expected_discretized_tap_powers_linear[8] += tap_powers_linear[12]
        expected_discretized_tap_powers_linear[9] += tap_powers_linear[13]
        expected_discretized_tap_powers_linear[10] += tap_powers_linear[14]
        expected_discretized_tap_powers_linear[10] += tap_powers_linear[15]
        expected_discretized_tap_powers_linear[11] += tap_powers_linear[16]
        expected_discretized_tap_powers_linear[12] += tap_powers_linear[17]
        expected_discretized_tap_powers_linear[13] += tap_powers_linear[18]
        expected_discretized_tap_powers_linear[14] += tap_powers_linear[19]

        # Check if the discretized tap powers are correct
        np.testing.assert_array_almost_equal(expected_discretized_tap_powers_linear,
                                             tu_discretized.tap_powers_linear)
        # Check if the discretized tap delays are correct. Note that they
        # are integers.
        np.testing.assert_array_equal(
            np.array([0, 7, 16, 21, 27, 38, 40, 41, 47, 50, 56, 58, 60, 63, 66]),
            tu_discretized.tap_delays)

        # Check if the Ts property is properly set
        self.assertAlmostEqual(tu_discretized.Ts, Ts)

        # Test the name of the discretized channel profile. It must be
        # equal to the original name appended with ' (discretized)'.
        self.assertEqual(tu_discretized.name, tu.name + ' (discretized)')

        # xxxxxxxxxx Test discretizing twice xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # If we try to discretize a TdlChannelProfile object that is
        # already discretized an exception should be raised
        with self.assertRaises(RuntimeError):
            tu_discretized.get_discretize_profile(Ts)

class TdlChannelTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_constructor(self):
        # TODO: implement-me
        pass

    def test_get_fading_map(self):
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        maxSystemBand = 40e6  # 40 MHz bandwidth
        # Number of subcarriers in this bandwidth
        max_num_of_subcarriers = math.floor(maxSystemBand/15e3)
        # Find the maximum FFT size we can use which is below than or equal
        # to maxNumOfSubcarriersInt
        max_num_of_subcarriers = int(
            2 ** math.floor(math.log(max_num_of_subcarriers, 2)))
        # Calculate the actual bandwidth that we will use
        bandwidth = 15e3 * max_num_of_subcarriers

        Fd = 5     # Doppler frequency (in Hz)
        Ts = 1./bandwidth  # Sampling interval (in seconds)
        NRays = 16  # Number of rays for the Jakes model

        # Create the jakes object that will be passed to TdlChannel
        jakes = fading_generators.JakesSampleGenerator(Fd, Ts, NRays, shape=None)
        tdlchannel = fading.TdlChannel.create_from_channel_profile(
            jakes, fading.COST259_TUx)

        # COST259_TUx profile has 20 taps. The TdlChannel class should have
        # changed the shape of the jakes object to [20]
        self.assertEqual(jakes.shape, (15,))

        # Let's generate 10 samples
        NSamples = 10
        fading_map = tdlchannel.get_fading_map(NSamples)

        # With the provided Ts the COST259 TU channel will have 67
        # discretized taps if we include the zeros. Only 15 of those taps
        # are different from zero and those are the ones stored in the
        # fading map.
        self.assertEqual(fading_map.shape, (15, 10))

        # test the non-zero taps
        for line in fading_map:
            self.assertTrue(np.all(np.abs(line) > 0))

        # xxxxxxxxxx Now let's test include_the_zeros_in_fading_map xxxxxxx
        full_fading_map = tdlchannel.include_the_zeros_in_fading_map(fading_map)
        self.assertEqual(full_fading_map.shape, (67, 10))


        # xxxxxxxxxx Now test with shape different from None xxxxxxxxxxxxxx
        # Create the jakes object that will be passed to TdlChannel
        jakes2 = fading_generators.JakesSampleGenerator(Fd, Ts, NRays, shape=(2,4))
        tdlchannel2 = fading.TdlChannel.create_from_channel_profile(
            jakes2, fading.COST259_TUx)
        # COST259_TUx profile has 20 taps. The TdlChannel class should have
        # changed the shape of the jakes object to [20]
        self.assertEqual(jakes2.shape, (15,2,4))

        # Let's generate 10 samples
        NSamples = 10
        fading_map2 = tdlchannel2.get_fading_map(NSamples)

        # With the provided Ts the COST259 TU channel will have 67
        # discretized taps if we include the zeros. Only 15 of those taps
        # are different from zero and those are the ones stored in the
        # fading map. Also, the Jakes object was created with a shape equal
        # to (2,4). Therefore, the shape of the fading map must be
        # (15, 2, 4, NSamples)
        self.assertEqual(fading_map2.shape, (15, 2, 4, 10))


    def test_get_channel_freq_response(self):
        maxSystemBand = 40e6  # 40 MHz bandwidth
        # Number of subcarriers in this bandwidth
        max_num_of_subcarriers = math.floor(maxSystemBand/15e3)
        # Find the maximum FFT size we can use which is below than or equal
        # to maxNumOfSubcarriersInt
        max_num_of_subcarriers = int(
            2 ** math.floor(math.log(max_num_of_subcarriers, 2)))
        # Calculate the actual bandwidth that we will use
        bandwidth = 15e3 * max_num_of_subcarriers

        Fd = 5     # Doppler frequency (in Hz)
        Ts = 1./bandwidth  # Sampling interval (in seconds)
        NRays = 16  # Number of rays for the Jakes model

        # Create the jakes object that will be passed to TdlChannel
        jakes = fading_generators.JakesSampleGenerator(Fd, Ts, NRays, shape=None)
        tdlchannel = fading.TdlChannel.create_from_channel_profile(
            jakes, fading.COST259_TUx)

        fading_map = tdlchannel.get_fading_map(10)
        full_fading_map = tdlchannel.include_the_zeros_in_fading_map(fading_map)
        freq_response = tdlchannel.get_channel_freq_response(full_fading_map, 2048)

        self.assertEqual(freq_response.shape, (2048, 10))

        # plt.plot(np.abs(freq_response[:,0]))
        # plt.show()

        # TODO: Implement-me
        pass


    def test_transmit_signal_with_known_fading_map(self):
        maxSystemBand = 40e6  # 40 MHz bandwidth
        # Number of subcarriers in this bandwidth
        max_num_of_subcarriers = math.floor(maxSystemBand/15e3)
        # Find the maximum FFT size we can use which is below than or equal
        # to maxNumOfSubcarriersInt
        max_num_of_subcarriers = int(
            2 ** math.floor(math.log(max_num_of_subcarriers, 2)))
        # Calculate the actual bandwidth that we will use
        bandwidth = 15e3 * max_num_of_subcarriers

        Fd = 5     # Doppler frequency (in Hz)
        Ts = 1./bandwidth  # Sampling interval (in seconds)
        NRays = 16  # Number of rays for the Jakes model

        # Create the jakes object that will be passed to TdlChannel
        jakes = fading_generators.JakesSampleGenerator(Fd, Ts, NRays, shape=None)
        tdlchannel = fading.TdlChannel.create_from_channel_profile(
            jakes, fading.COST259_TUx)

        # xxxxxxxxxx Test sending just a single impulse xxxxxxxxxxxxxxxxxxx
        signal = np.array([1.])

        num_samples = 1
        fading_map = tdlchannel.get_fading_map(num_samples)
        received_signal = tdlchannel.transmit_signal_with_known_fading_map(
            signal, fading_map)

        # Since only one sample was sent and it is equal to 1, then the
        # received signal will be equal to the full_fading_map
        full_fading_map = tdlchannel.include_the_zeros_in_fading_map(fading_map)

        np.testing.assert_almost_equal(full_fading_map.flatten(),
                                       received_signal)

        # xxxxxxxxxx Test sending a vector with 10 samples xxxxxxxxxxxxxxxx
        num_samples = 10
        fading_map = tdlchannel.get_fading_map(num_samples)

        signal = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        received_signal = tdlchannel.transmit_signal_with_known_fading_map(
            signal, fading_map)

        # Compute the expected received signal
        # For this Ts we have 15 discretized taps. The indexes of the 15
        # taps are:
        # [ 0,  7, 16, 21, 27, 38, 40, 41, 47, 50, 56, 58, 60, 63, 66]
        expected_received_signal = np.zeros(66 + num_samples, dtype=complex)
        expected_received_signal[0:0+num_samples] += signal * fading_map[0]
        expected_received_signal[7:7+num_samples] += signal * fading_map[1]
        expected_received_signal[16:16+num_samples] += signal * fading_map[2]
        expected_received_signal[21:21+num_samples] += signal * fading_map[3]
        expected_received_signal[27:27+num_samples] += signal * fading_map[4]
        expected_received_signal[38:38+num_samples] += signal * fading_map[5]
        expected_received_signal[40:40+num_samples] += signal * fading_map[6]
        expected_received_signal[41:41+num_samples] += signal * fading_map[7]
        expected_received_signal[47:47+num_samples] += signal * fading_map[8]
        expected_received_signal[50:50+num_samples] += signal * fading_map[9]
        expected_received_signal[56:56+num_samples] += signal * fading_map[10]
        expected_received_signal[58:58+num_samples] += signal * fading_map[11]
        expected_received_signal[60:60+num_samples] += signal * fading_map[12]
        expected_received_signal[63:63+num_samples] += signal * fading_map[13]
        expected_received_signal[66:66+num_samples] += signal * fading_map[14]

        np.testing.assert_array_almost_equal(expected_received_signal,
                                             received_signal)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Multiuser Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MuSisoChannelTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.musisochannel = multiuser.MuSisoChannel(N=2)
        self.musisochannel2 = multiuser.MuSisoChannel(N=3)

    def test_corrupt_data(self):
        # xxxxxxxxxx Test without pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        data1 = np.random.randint(0, 10, (2, 5))
        output1 = self.musisochannel.corrupt_data(data1)

        fading_samples_1 = self.musisochannel._get_channel_samples()
        expected_output1 = np.zeros((2, 5), dtype=complex)
        expected_output1[0] = (fading_samples_1[0, 0] * data1[0] +
                               fading_samples_1[0, 1] * data1[1])
        expected_output1[1] = (fading_samples_1[1, 0] * data1[0] +
                               fading_samples_1[1, 1] * data1[1])

        np.testing.assert_array_almost_equal(output1, expected_output1)

        data2 = np.random.randint(0, 10, (3, 5))
        output2 = self.musisochannel2.corrupt_data(data2)
        fading_samples_2 = self.musisochannel2._get_channel_samples()
        expected_output2 = np.zeros((3, 5), dtype=complex)
        expected_output2[0] = (fading_samples_2[0, 0] * data2[0] +
                               fading_samples_2[0, 1] * data2[1] +
                               fading_samples_2[0, 2] * data2[2])
        expected_output2[1] = (fading_samples_2[1, 0] * data2[0] +
                               fading_samples_2[1, 1] * data2[1] +
                               fading_samples_2[1, 2] * data2[2])
        expected_output2[2] = (fading_samples_2[2, 0] * data2[0] +
                               fading_samples_2[2, 1] * data2[1] +
                               fading_samples_2[2, 2] * data2[2])
        np.testing.assert_array_almost_equal(output2, expected_output2)

        # xxxxxxxxxx Now test with path loss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        pathloss_matrix1 = np.eye(2)
        self.musisochannel.set_pathloss(pathloss_matrix1)
        fading_samples_1_with_pl = self.musisochannel._get_channel_samples()

        output1_with_pl = self.musisochannel.corrupt_data(data1)
        expected_output1_with_pl = np.zeros((2, 5), dtype=complex)
        expected_output1_with_pl[0] = fading_samples_1_with_pl[0, 0] * data1[0]
        expected_output1_with_pl[1] = fading_samples_1_with_pl[1, 1] * data1[1]
        np.testing.assert_array_almost_equal(
            output1_with_pl, expected_output1_with_pl)

        pathloss_matrix1 = np.random.randn(3)
        self.musisochannel2.set_pathloss(pathloss_matrix1)
        fading_samples_2_with_pl = self.musisochannel2._get_channel_samples()

        output2_with_pl = self.musisochannel2.corrupt_data(data2)
        expected_output2_with_pl = np.zeros((3, 5), dtype=complex)
        expected_output2_with_pl[0] = (
            fading_samples_2_with_pl[0, 0] * data2[0] +
            fading_samples_2_with_pl[0, 1] * data2[1] +
            fading_samples_2_with_pl[0, 2] * data2[2])
        expected_output2_with_pl[1] = (
            fading_samples_2_with_pl[1, 0] * data2[0] +
            fading_samples_2_with_pl[1, 1] * data2[1] +
            fading_samples_2_with_pl[1, 2] * data2[2])
        expected_output2_with_pl[2] = (
            fading_samples_2_with_pl[2, 0] * data2[0] +
            fading_samples_2_with_pl[2, 1] * data2[1] +
            fading_samples_2_with_pl[2, 2] * data2[2])
        np.testing.assert_array_almost_equal(
            output2_with_pl, expected_output2_with_pl)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # TODO: Test if the fading samples are updated the next type we call the
        # corrupt_data

        # fading_map = self.musisochannel._get_channel_samples()
        # self.musisochannel.corrupt_data(data1)
        # fading_map_new = self.musisochannel._get_channel_samples()
        # np.testing.assert_array_not_almost_equal(fading_map, fading_map_new)


class MultiUserChannelMatrixTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.multiH = multiuser.MultiUserChannelMatrix()
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
            = multiuser.MultiUserChannelMatrix._from_small_matrix_to_big_matrix(
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
        self.multiH.noise_var = 1e-20
        output3 = self.multiH.corrupt_data(input_data)
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
        self.multiH.noise_var = noise_var

        H = np.eye(6)
        self.multiH.init_from_channel_matrix(H,
                                             np.array([2, 2, 2]),
                                             np.array([2, 2, 2]),
                                             3)

        data = randn_c(6, 10)

        corrupted_data = self.multiH.corrupt_concatenated_data(data)
        last_noise = self.multiH.last_noise

        expected_corrupted_data = data + last_noise

        np.testing.assert_array_almost_equal(expected_corrupted_data,
                                             corrupted_data)

        self.assertAlmostEqual(noise_var, self.multiH.noise_var)

        # Call corrupt_concatenated_data again, but without noise var. This
        # should set last_noise to None and noise_var to None.
        self.multiH.noise_var = None
        corrupted_data = self.multiH.corrupt_concatenated_data(data)
        np.testing.assert_array_almost_equal(corrupted_data, data)
        self.assertIsNone(self.multiH.last_noise)
        self.assertIsNone(self.multiH.noise_var)

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
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_Q(k, F_all_k)
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

        self.multiH.noise_var = 0.0
        Qk = self.multiH.calc_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q1)

        # Now with noise variance different of 0
        noise_var = round(0.1 * np.random.rand(), 4)
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_Q(k, F_all_k)
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

        # Calculate Qk without noise
        Qk = self.multiH._calc_Q_impl(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q2)

        # Now with noise variance different of 0
        noise_var = round(0.1 * np.random.rand(), 4)
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_Q(k, F_all_k)
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
        noise_var = round(0.1 * np.random.rand(), 4)
        self.multiH.noise_var = noise_var

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

        # Test if Qk (without noise) is equal to the expected output
        Qk = self.multiH._calc_JP_Q_impl(k, F_all_k)
        np.testing.assert_array_almost_equal(Qk, expected_Q0)

        # Now with noise variance different of 0
        Qk = self.multiH.calc_JP_Q(k, F_all_k)
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

        # Test if Qk (without noise) is equal to the expected output
        Qk = self.multiH._calc_JP_Q_impl(k, F_all_k)
        np.testing.assert_array_almost_equal(Qk, expected_Q1)

        # Now with noise variance different of 0
        Qk = self.multiH.calc_JP_Q(k, F_all_k)
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

        # Test if Qk (without noise) is equal to the expected output
        Qk = self.multiH._calc_JP_Q_impl(k, F_all_k)
        np.testing.assert_array_almost_equal(Qk, expected_Q2)

        # Now with noise variance different of 0
        Qk = self.multiH.calc_JP_Q(k, F_all_k)
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
        self.multiH.noise_var = noise_power

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
            # The _calc_Q_impl method does not include the noise, while the
            # calc_Q method does.
            expected_first_part = (self.multiH._calc_Q_impl(k, F)
                                   + np.dot(HkkFk, HkkFk.conjugate().T))
            expected_first_part_with_noise = (
                self.multiH.calc_Q(k, F)
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
            # First part in the equation of Bkl (the double summation)
            expected_first_part = 0.0

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

                expected_first_part += aux

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
        multiUserChannel = multiuser.MultiUserChannelMatrix()
        multiUserChannel.noise_var = 0.0

        # iasolver = MaxSinrIASolver(multiUserChannel)
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
        multiUserChannel = multiuser.MultiUserChannelMatrix()
        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 2

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(Nr, Nt, K)
        multiUserChannel.noise_var = 0.0
        iasolver = ClosedFormIASolver(multiUserChannel)
        iasolver.solve(Ns, P)
        F = iasolver.full_F
        U = iasolver.full_W

        multiUserChannel.noise_var = None
        SINR_all_users = multiUserChannel.calc_SINR(F, U)

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
        noise_var = 0.1
        multiUserChannel.noise_var = noise_var
        SINR_all_users = multiUserChannel.calc_SINR(F, U)
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
        self.multiH.noise_var = noise_power

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
            expected_first_part = (self.multiH._calc_JP_Q_impl(k, F)
                                   + np.dot(HkFk, HkFk.conjugate().T))
            expected_first_part_with_noise = (
                self.multiH.calc_JP_Q(k, F)
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
            # First part in the equation of Bkl (the double summation)
            expected_first_part = 0.0

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

                expected_first_part += aux

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
        # noise_power = 0.001

        self.multiH.randomize(Nr, Nt, K)

        (newH, Ms_good) = blockdiagonalization.block_diagonalize(
            self.multiH.big_H, K, iPu, 0.0)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)
        big_U = blockdiagonalization.calc_receive_filter(newH)
        aux = single_matrix_to_matrix_of_matrices(big_U, Ns, Nr)
        U = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            U[k] = aux[k, k].conjugate().T

        SINR_all_users = self.multiH.calc_JP_SINR(F, U)

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
        self.multiH.noise_var = 0.1
        SINR_all_users = self.multiH.calc_JP_SINR(F, U)
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
        self.multiH = multiuser.MultiUserChannelMatrixExtInt()
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
        multiH_no_ext_int = multiuser.MultiUserChannelMatrix()
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
        multiH_no_ext_int = multiuser.MultiUserChannelMatrix()
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

        cov_int = self.multiH.calc_cov_matrix_extint_plus_noise(pe=1.0)
        # Set the noise variance
        self.multiH.noise_var = noise_var
        cov_int_plus_noise = self.multiH.calc_cov_matrix_extint_plus_noise(
            pe=1.0)

        # Test without noise
        self.assertEqual(cov_int.size, expected_cov_int.size)
        np.testing.assert_array_almost_equal(cov_int[0], expected_cov_int[0])
        np.testing.assert_array_almost_equal(cov_int[1], expected_cov_int[1])
        np.testing.assert_array_almost_equal(cov_int[2], expected_cov_int[2])

        # Test with the noise
        self.assertEqual(
            cov_int_plus_noise.size, expected_cov_int_plus_noise.size)
        np.testing.assert_array_almost_equal(
            cov_int_plus_noise[0], expected_cov_int_plus_noise[0])
        np.testing.assert_array_almost_equal(
            cov_int_plus_noise[1], expected_cov_int_plus_noise[1])
        np.testing.assert_array_almost_equal(
            cov_int_plus_noise[2], expected_cov_int_plus_noise[2])

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

        # test without noise
        self.multiH.noise_var = None
        Qk = self.multiH.calc_Q(k, F_all_k, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q0_no_ext_int_or_noise)

        # Now with external interference
        expected_Q0_no_noise = (expected_Q0_no_ext_int_or_noise +
                                np.dot(R0_e0, R0_e0.conjugate().T) +
                                np.dot(R0_e1, R0_e1.conjugate().T))
        Qk = self.multiH.calc_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q0_no_noise)

        # Now with external interference and noise
        noise_var = round(0.1 * np.random.rand(), 4)
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_Q(k, F_all_k)
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

        # test without noise
        self.multiH.noise_var = None
        Qk = self.multiH.calc_Q(k, F_all_k, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q1_no_ext_int_or_noise)

        # Now with external interference
        expected_Q1_no_noise = (expected_Q1_no_ext_int_or_noise +
                                np.dot(R1_e0, R1_e0.conjugate().T) +
                                np.dot(R1_e1, R1_e1.conjugate().T))
        Qk = self.multiH.calc_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q1_no_noise)

        # Now with external interference and noise
        noise_var = round(0.1 * np.random.rand(), 4)
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_Q(k, F_all_k)
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

        # Test without noise
        self.multiH.noise_var = None
        Qk = self.multiH.calc_Q(k, F_all_k, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q2_no_ext_int_or_noise)

        # Now with external interference
        expected_Q2_no_noise = (expected_Q2_no_ext_int_or_noise +
                                np.dot(R2_e0, R2_e0.conjugate().T) +
                                np.dot(R2_e1, R2_e1.conjugate().T))
        Qk = self.multiH.calc_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q2_no_noise)

        # Now with external interference and noise
        noise_var = round(0.1 * np.random.rand(), 4)
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_Q(k, F_all_k)
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

        noise_var = round(0.1 * np.random.rand(), 4)
        Pe = round(np.random.rand(), 4)

        self.multiH.randomize(Nr, Nt, K, NtE)
        # We don't set the noise variance here because the first tests will
        # be without noise.

        F_all_k = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F_all_k[k] = randn_c(np.sum(Nt), Ns[k]) * np.sqrt(P[k])
            F_all_k[k] = (F_all_k[k]
                          / np.linalg.norm(F_all_k[k], 'fro')
                          * np.sqrt(P[k]))

        Re_no_noise = self.multiH.calc_cov_matrix_extint_without_noise(pe=Pe)
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

        Qk = self.multiH.calc_JP_Q(k, F_all_k, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q0_no_ext_int_or_noise)

        # Now with external interference
        expected_Q0_no_noise = (expected_Q0_no_ext_int_or_noise +
                                Re_no_noise[0])
        Qk = self.multiH.calc_JP_Q(k, F_all_k, pe=Pe)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q0_no_noise)

        # Now with external interference and noise
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_JP_Q(k, F_all_k, pe=Pe)
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

        # Test without noise
        self.multiH.noise_var = None
        Qk = self.multiH.calc_JP_Q(k, F_all_k, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q1_no_ext_int_or_noise)

        # Now with external interference
        expected_Q1_no_noise = (expected_Q1_no_ext_int_or_noise +
                                Re_no_noise[1])
        Qk = self.multiH.calc_JP_Q(k, F_all_k, pe=Pe)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q1_no_noise)

        # Now with external interference and noise
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_JP_Q(k, F_all_k, pe=Pe)
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

        # Test without noise
        self.multiH.noise_var = None
        Qk = self.multiH.calc_JP_Q(k, F_all_k, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q2_no_ext_int_or_noise)

        # Now with external interference
        expected_Q2_no_noise = (expected_Q2_no_ext_int_or_noise +
                                Re_no_noise[2])
        Qk = self.multiH.calc_JP_Q(k, F_all_k, pe=Pe)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q2_no_noise)

        # Now with external interference and noise
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_JP_Q(k, F_all_k, pe=Pe)
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
        self.multiH.noise_var = None

        F = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])

        Re = self.multiH.calc_cov_matrix_extint_plus_noise(pe=Pe)

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
        self.multiH.noise_var = noise_var
        Re = self.multiH.calc_cov_matrix_extint_plus_noise(pe=Pe)

        # For ones stream the expected Bkl is equivalent to the Q matrix
        # plus the direct channel part.
        for k in range(K):
            Hkk = self.multiH.get_Hkl(k, k)
            Fk = F[k]
            HkkFk = np.dot(Hkk, Fk)
            expected_first_part = (
                self.multiH.calc_Q(k, F, pe=Pe)
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

        Re = self.multiH.calc_cov_matrix_extint_plus_noise(pe=Pe)

        F = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])

        for k in range(K):
            # First part in the equation of Bkl (the double summation)
            expected_first_part = 0.0

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

                expected_first_part += aux
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
        self.multiH.noise_var = noise_power

        Re = self.multiH.calc_cov_matrix_extint_plus_noise(pe=Pe)

        F = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F[k] = randn_c(Nt[k], Ns[k]) * np.sqrt(P[k])
            F[k] = F[k] / np.linalg.norm(F[k], 'fro') * np.sqrt(P[k])

        for k in range(K):
            # We only have the stream 0
            expected_Bk0 = self.multiH.calc_Q(k, F, pe=Pe)
            Bk0 = self.multiH._calc_Bkl_cov_matrix_all_l(F, k, Re[k])[0]
            np.testing.assert_array_almost_equal(expected_Bk0, Bk0)

    def test_underline_calc_SINR_k(self):
        multiUserChannel = multiuser.MultiUserChannelMatrixExtInt()
        # iasolver = MaxSinrIASolver(multiUserChannel)
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
        multiUserChannel.noise_var = noise_power

        Re = multiUserChannel.calc_cov_matrix_extint_plus_noise(pe=Pe)

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
        multiUserChannel.noise_var = 0.0
        iasolver = ClosedFormIASolver(multiUserChannel)
        iasolver.solve(Ns=2)
        F = iasolver.full_F
        U = iasolver.full_W

        Pe = 0.01
        multiUserChannel.noise_var = 0.001
        Re = multiUserChannel.calc_cov_matrix_extint_plus_noise(pe=Pe)

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
        multiUserChannel = multiuser.MultiUserChannelMatrixExtInt()
        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 2
        NtE = np.array([1])

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(Nr, Nt, K, NtE)

        multiUserChannel.noise_var = 0.0
        iasolver = ClosedFormIASolver(multiUserChannel)
        iasolver.solve(Ns, P)
        F = iasolver.full_F
        U = iasolver.full_W

        # xxxxxxxxxx Noise Variance = 0.0 and Pe = 0 xxxxxxxxxxxxxxxxxxxxxx
        Pe = 0.00
        noise_power = 0.00
        multiUserChannel.noise_var = noise_power
        Re = multiUserChannel.calc_cov_matrix_extint_plus_noise(pe=Pe)

        SINR_all_users = multiUserChannel.calc_SINR(
            F, U, pe=Pe)

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
        multiUserChannel.noise_var = noise_power
        Re = multiUserChannel.calc_cov_matrix_extint_plus_noise(pe=Pe)

        SINR_all_users = multiUserChannel.calc_SINR(
            F, U, pe=Pe)

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
        self.multiH.noise_var = noise_power

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

            # Without noise
            self.multiH.noise_var = None
            expected_first_part = (self.multiH.calc_JP_Q(k, F, pe=0.0)
                                   + np.dot(HkFk, HkFk.conjugate().T))
            # With noise
            self.multiH.noise_var = noise_power
            expected_first_part_with_noise = self.multiH.calc_JP_Q(
                k, F, pe=0.0) + np.dot(HkFk, HkFk.conjugate().T)

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
        self.multiH.noise_var = None
        Re_no_noise = self.multiH.calc_cov_matrix_extint_plus_noise(pe=Pe)
        self.multiH.noise_var = noise_power
        Re = self.multiH.calc_cov_matrix_extint_plus_noise(pe=Pe)

        # For ones stream the expected Bkl is equivalent to the Q matrix
        # plus the direct channel part.
        for k in range(K):
            Hk = self.multiH.get_Hk_without_ext_int(k)
            Fk = F[k]
            HkFk = np.dot(Hk, Fk)

            # without noise
            self.multiH.noise_var = None
            expected_first_part = (self.multiH.calc_JP_Q(k, F, pe=Pe)
                                   + np.dot(HkFk, HkFk.conjugate().T))

            # with noise
            self.multiH.noise_var = noise_power
            expected_first_part_with_noise = (
                self.multiH.calc_JP_Q(k, F, pe=Pe)
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
        self.multiH.noise_var = noise_power
        Re = self.multiH.calc_cov_matrix_extint_plus_noise(pe=Pe)

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
        Re = self.multiH.calc_cov_matrix_extint_plus_noise(pe=Pe)

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

        SINR_all_users = self.multiH.calc_JP_SINR(F, U, pe=0.0)

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
        self.multiH.noise_var = noise_var
        Pe = 0.63
        Re = self.multiH.calc_cov_matrix_extint_plus_noise(pe=Pe)

        SINR_all_users = self.multiH.calc_JP_SINR(F, U, pe=Pe)

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
# xxxxxxxxxxxxxxx Pathloss Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class PathLossFreeSpaceTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.pl = pathloss.PathLossFreeSpace()

    def test_type(self):
        self.assertEqual(self.pl.type, 'outdoor')

    def test_calc_path_loss(self):
        # with a very small distance the path loss (in linear scale) would
        # be negative, which is not valid. For these cases we should throw
        # an exception.
        self.pl.handle_small_distances_bool = False
        with self.assertRaises(RuntimeError):
            self.pl.calc_path_loss(0.000011)

        self.pl.handle_small_distances_bool = True
        self.assertAlmostEqual(self.pl.calc_path_loss(0.000011), 1.0)
        np.testing.assert_array_almost_equal(
            self.pl.calc_path_loss_dB([0.000011, 0.00011, 0.0011]),
            np.array([0., 12.35447608, 32.35447608]))

        # xxxxxxxxxx Test for a single path loss value xxxxxxxxxxxxxxxxxxxx
        n = 2
        fc = 900
        d = 1.2
        expected_pl_in_dB = (
            10 * n * (np.log10(d) + np.log10(fc) + 6.0 - 4.377911390697565))

        self.assertAlmostEqual(self.pl.calc_path_loss_dB(1.2),
                               expected_pl_in_dB)
        self.assertAlmostEqual(self.pl.calc_path_loss(1.2),
                               dB2Linear(-expected_pl_in_dB))

        # When we change 'n', this will impact path loss calculation
        n = 2.7
        self.pl.n = n
        expected_pl_in_dB = (
            10 * n * (np.log10(d) + np.log10(fc) + 6.0 - 4.377911390697565))
        self.assertAlmostEqual(self.pl.calc_path_loss_dB(1.2),
                               expected_pl_in_dB)
        self.assertAlmostEqual(self.pl.calc_path_loss(1.2),
                               dB2Linear(-expected_pl_in_dB))

        # When we change 'fc', this will impact path loss calculation
        fc = 1100
        self.pl.fc = fc
        expected_pl_in_dB = (
            10 * n * (np.log10(d) + np.log10(fc) + 6.0 - 4.377911390697565))
        self.assertAlmostEqual(self.pl.calc_path_loss_dB(1.2),
                               expected_pl_in_dB)
        self.assertAlmostEqual(self.pl.calc_path_loss(1.2),
                               dB2Linear(-expected_pl_in_dB))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx est for multiple path loss values xxxxxxxxxxxxxxxxxxxx
        d = np.array([1.2, 1.4, 1.6])
        expected_pl_in_dB = (
            10 * n * (np.log10(d) + np.log10(fc) + 6.0 - 4.377911390697565))

        np.testing.assert_array_almost_equal(
            self.pl.calc_path_loss_dB([1.2, 1.4, 1.6]),
            expected_pl_in_dB, 16)

        np.testing.assert_array_almost_equal(
            self.pl.calc_path_loss([1.2, 1.4, 1.6]),
            dB2Linear(-expected_pl_in_dB), 16)

        # Change 'n' and 'fc'
        n = 2
        fc = 900
        self.pl.n = n
        self.pl.fc = fc
        expected_pl_in_dB = (
            10 * n * (np.log10(d) + np.log10(fc) + 6.0 - 4.377911390697565))

        np.testing.assert_array_almost_equal(
            self.pl.calc_path_loss_dB([1.2, 1.4, 1.6]),
            expected_pl_in_dB, 16)

        np.testing.assert_array_almost_equal(
            self.pl.calc_path_loss([1.2, 1.4, 1.6]),
            dB2Linear(-expected_pl_in_dB), 16)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test test_calc_path_loss with shadow xxxxxxxxxxxxxxxxx
        self.pl.use_shadow_bool = True
        # We don't know the value of the shadowing to test it, but we can
        # at least test that the shadowing modified the path loss
        self.assertNotAlmostEqual(self.pl.calc_path_loss_dB(1.2),
                                  93.1102472958)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

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

        # xxxxx Sanity check
        np.testing.assert_array_almost_equal(
            self.pl.which_distance_dB(self.pl.calc_path_loss_dB([1.4, 1.1])),
            np.array([1.4, 1.1]))
        np.testing.assert_array_almost_equal(
            self.pl.which_distance(self.pl.calc_path_loss([1.4, 1.1])),
            np.array([1.4, 1.1]))
        # xxxxx


class PathLoss3GPP1TestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.pl = pathloss.PathLoss3GPP1()

    def test_type(self):
        self.assertEqual(self.pl.type, 'outdoor')

    def test_calc_path_loss(self):
        # xxxxxxxxxx Test the case for very small distances xxxxxxxxxxxxxxx
        # with a very small distance the path loss (in linear scale) would
        # be negative, which is not valid. For these cases we should throw
        # an exception.
        self.pl.handle_small_distances_bool = False
        with self.assertRaises(RuntimeError):
            self.pl.calc_path_loss(1e-4)

        # With a very small distance the path loss (in linear scale) would
        # be negative, which is not valid. If "handle_small_distances_bool"
        # is set to True then a pathloss equal to 1.0 (in linear scale) is
        # returned for small distances.
        self.pl.handle_small_distances_bool = True
        self.assertAlmostEqual(self.pl.calc_path_loss(1e-4), 1.0)
        np.testing.assert_array_almost_equal(
            self.pl.calc_path_loss_dB(
                np.array([1e-4, 2e-4, 8e-4, 1e-3, 5e-3])),
            np.array([0., 0., 11.65618351, 15.3, 41.58127216]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test for a single path loss value xxxxxxxxxxxxxxxxxxxx
        expected_pl = dB2Linear(-(128.1 + 37.6 * np.log10(1.2)))
        self.assertAlmostEqual(self.pl.calc_path_loss(1.2),
                               expected_pl, places=14)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test for multiple path loss values xxxxxxxxxxxxxxxxxxx
        expected_pl = dB2Linear(
            -(128.1 + 37.6 * np.log10(np.array([1.2, 1.5, 1.8, 2.3]))))
        np.testing.assert_array_almost_equal(
            self.pl.calc_path_loss(np.array([1.2, 1.5, 1.8, 2.3])),
            expected_pl,
            decimal=16)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_which_distance(self):
        np.testing.assert_array_almost_equal(
            self.pl.which_distance(
                self.pl.calc_path_loss(np.array([1.2, 1.5, 1.8, 2.3]))),
            np.array([1.2, 1.5, 1.8, 2.3]),
            decimal=14)


# TODO: finish implementation
class PathLossMetisPS7TestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.pl = pathloss.PathLossMetisPS7()

    def test_type(self):
        self.assertEqual(self.pl.type, 'indoor')

    def test_calc_PS7_path_loss_dB_same_floor(self):
        A = 36.8
        B = 43.8
        C = 20

        num_walls = 1
        X = 5 * (num_walls - 1)

        fc_GHz = 0.9
        d = 10.0

        # xxxxxxxxxx Test NLOS case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Simple test
        expected_pl_dB_NLOS \
            = A * math.log10(d) + B + C * math.log10(fc_GHz / 5) + X
        self.assertAlmostEqual(
            self.pl._calc_PS7_path_loss_dB_same_floor(d, num_walls=num_walls),
            expected_pl_dB_NLOS)

        # Test with a different frequency value
        fc_GHz = 6.0
        self.pl.fc = 6e3

        expected_pl_dB_NLOS \
            = A * math.log10(d) + B + C * math.log10(fc_GHz / 5) + X
        self.assertAlmostEqual(
            self.pl._calc_PS7_path_loss_dB_same_floor(d, num_walls=num_walls),
            expected_pl_dB_NLOS)

        # Test with different number of walls
        num_walls = 3
        X = 5 * (num_walls - 1)
        expected_pl_dB_NLOS \
            = A * math.log10(d) + B + C * math.log10(fc_GHz / 5) + X
        self.assertAlmostEqual(
            self.pl._calc_PS7_path_loss_dB_same_floor(d, num_walls=num_walls),
            expected_pl_dB_NLOS)

        # Test with d as an array
        d = np.array([10., 50., 100., 1000.])
        expected_pl_dB_NLOS \
            = A * np.log10(d) + B + C * math.log10(fc_GHz / 5) + X
        np.testing.assert_array_almost_equal(
            self.pl._calc_PS7_path_loss_dB_same_floor(d, num_walls=num_walls),
            expected_pl_dB_NLOS)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test the LOS Case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        A = 18.7
        B = 46.8
        C = 20
        X = 0

        # Simple test
        d = 10
        expected_pl_dB_LOS \
            = A * np.log10(d) + B + C * math.log10(fc_GHz / 5)
        self.assertAlmostEqual(
            self.pl._calc_PS7_path_loss_dB_same_floor(d, num_walls=0),
            expected_pl_dB_LOS)

        # Test with a different frequency
        fc_GHz = 1.1
        self.pl.fc = 1.1e3
        expected_pl_dB_LOS \
            = A * np.log10(d) + B + C * math.log10(fc_GHz / 5)
        self.assertAlmostEqual(
            self.pl._calc_PS7_path_loss_dB_same_floor(d, num_walls=0),
            expected_pl_dB_LOS)

        # Test with d as an array
        d = np.array([10., 50., 100., 1000.])
        expected_pl_dB_LOS \
            = A * np.log10(d) + B + C * math.log10(fc_GHz / 5)
        np.testing.assert_array_almost_equal(
            self.pl._calc_PS7_path_loss_dB_same_floor(d, num_walls=0),
            expected_pl_dB_LOS)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test with LOS and NLOS at the same time xxxxxxxxxxxxxx
        # Now we test when `d` and `num_walls` are numpy arrays. Some of
        # the values in num_walls array are equal to zero, while others are
        # above zero. LOS path loss model must be used for the values in
        # num_walls which are equal to zero while NLOS must be used for the
        # other values.
        num_walls = np.array([1, 2, 0, 2, 0], dtype=int)
        d = np.array([30., 30., 30., 200., 10.])

        LOS_index = (num_walls == 0)
        NLOS_index = ~LOS_index

        X = 5 * (num_walls[NLOS_index] - 1)

        expected_pl_dB = np.empty(5, dtype=float)

        # First calculated the expected path loss for the LOS values
        A = 18.7
        B = 46.8
        C = 20
        expected_pl_dB[LOS_index] \
            = A * np.log10(d[LOS_index]) + B + C * math.log10(fc_GHz / 5)

        # Now calculate the expected path loss for the NLOS values
        A = 36.8
        B = 43.8
        C = 20

        X = 5 * (num_walls[NLOS_index] - 1)
        expected_pl_dB[NLOS_index] \
            = A * np.log10(d[NLOS_index]) + B + C * math.log10(fc_GHz / 5) + X

        np.testing.assert_array_almost_equal(
            self.pl._calc_PS7_path_loss_dB_same_floor(d, num_walls),
            expected_pl_dB)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_path_loss(self):
        pass


# TODO: finish implementation
class PathLossOkomuraHataTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.pl = pathloss.PathLossOkomuraHata()

    def test_type(self):
        self.assertEqual(self.pl.type, 'outdoor')

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

        # Area type can only be one of the following values:
        # 'open', 'suburban', 'medium city' or 'large city'
        with self.assertRaises(RuntimeError):
            self.pl.area_type = 'some_invalid_string'

        # Test if a warning is raised when the distance is smaller then
        # 1Km. For that we capture the warnings ...
        with warnings.catch_warnings(record=True) as w:
            # then we call the method with the distance smaller than 1Km
            # ...
            self.pl._calc_deterministic_path_loss_dB(0.9)
            # and we test if captured 1 warning.
            self.assertEqual(len(w), 1, msg='Warning was not raised')

        # Test with a single distance value
        self.pl.area_type = 'open'
        self.assertTrue(
            isinstance(self.pl._calc_deterministic_path_loss_dB(20), float))
        self.assertAlmostEqual(self.pl._calc_deterministic_path_loss_dB(20),
                               145.000295737969)

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
