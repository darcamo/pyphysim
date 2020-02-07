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
except NameError:  # pragma: no cover
    sys.path.append('../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import unittest
import doctest
import warnings
from pyphysim import channels
import math
from copy import copy
import numpy as np
from scipy.linalg import block_diag
from pyphysim.channels import noise, fading_generators, fading, singleuser, \
    multiuser, pathloss, antennagain
from pyphysim.comm import blockdiagonalization
from pyphysim.ia.algorithms import ClosedFormIASolver
from pyphysim.util.conversion import single_matrix_to_matrix_of_matrices, \
    dB2Linear
from pyphysim.util.misc import randn_c, least_right_singular_vectors


# noinspection PyMethodMayBeStatic
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


# noinspection PyMethodMayBeStatic
class ModuleFunctionsTestCase(unittest.TestCase):
    def test_calc_thermal_noise_power(self):
        T = 23  # Temperature in degrees

        # Test for 1Hz
        delta_f = 1  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -174, places=0)

        # Test for 10Hz
        delta_f = 10  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -164, places=0)

        # Test for 100Hz
        delta_f = 100  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -154, places=0)

        # Test for 200kHz
        delta_f = 200e3  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -121, places=0)

        # Test for 1MHz
        delta_f = 1e6  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -114, places=0)

        # Test for 5MHz
        delta_f = 5e6  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -107, places=0)

        # Test for 20MHz
        delta_f = 20e6  # Bandwidth in Hz
        noise_power_dBm = noise.calc_thermal_noise_power_dBm(T, delta_f)
        self.assertAlmostEqual(noise_power_dBm, -101, places=0)

    def test_generate_jakes_samples(self):
        Fd = 5  # Doppler frequency (in Hz)
        Ts = 1e-3  # Sampling interval (in seconds)
        N = 1000  # Number of samples
        NRays = 8  # Number of rays for the Jakes model

        # Test generating channel samples for a SISO scenario
        new_current_time, h = fading_generators.generate_jakes_samples(
            Fd, Ts, N, NRays)
        self.assertEqual(h.size, 1000)
        self.assertEqual(h.shape, (1000, ))
        self.assertAlmostEqual(new_current_time, 1000 * Ts)

        new_current_time, h2 = fading_generators.generate_jakes_samples(
            Fd, Ts, N, NRays, shape=(4, 3), current_time=3752 * Ts)
        self.assertAlmostEqual(new_current_time, 4752 * Ts)
        self.assertEqual(h2.shape, (4, 3, N))

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
class FadingSampleGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_shape_property(self):
        obj1 = fading_generators.FadingSampleGenerator(shape=None)
        obj2 = fading_generators.FadingSampleGenerator(shape=4)
        obj3 = fading_generators.FadingSampleGenerator(shape=(2, 3))
        self.assertEqual(obj1.shape, None)
        self.assertEqual(obj2.shape, (4, ))
        self.assertEqual(obj3.shape, (2, 3))

        obj1.shape = (2, 5)
        self.assertEqual(obj1.shape, (2, 5))


class RayleighSampleGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.obj0 = fading_generators.RayleighSampleGenerator(shape=None)
        self.obj1 = fading_generators.RayleighSampleGenerator(shape=1)
        self.obj2 = fading_generators.RayleighSampleGenerator(shape=3)
        self.obj3 = fading_generators.RayleighSampleGenerator(shape=(4, 3))

    def test_generate_more_samples(self):
        # num_samples is None
        self.assertTrue(isinstance(self.obj0.get_samples(), complex))
        self.assertEqual(self.obj1.get_samples().shape, (1, ))
        self.assertEqual(self.obj2.get_samples().shape, (3, ))
        self.assertEqual(self.obj3.get_samples().shape, (4, 3))

        # num_samples is not None
        self.obj0.generate_more_samples(num_samples=5)
        self.obj1.generate_more_samples(num_samples=5)
        self.obj2.generate_more_samples(num_samples=5)
        self.obj3.generate_more_samples(num_samples=5)
        self.assertEqual(self.obj0.get_samples().shape, (5, ))
        self.assertEqual(self.obj1.get_samples().shape, (1, 5))
        self.assertEqual(self.obj2.get_samples().shape, (3, 5))
        self.assertEqual(self.obj3.get_samples().shape, (4, 3, 5))

    def test_get_similar_fading_generator(self):
        # RayleighSampleGenerator only has the _shape attribute
        obj0 = self.obj0.get_similar_fading_generator()
        obj1 = self.obj1.get_similar_fading_generator()
        obj2 = self.obj2.get_similar_fading_generator()
        obj3 = self.obj3.get_similar_fading_generator()

        self.assertEqual(obj0.shape, self.obj0.shape)
        self.assertEqual(obj1.shape, self.obj1.shape)
        self.assertEqual(obj2.shape, self.obj2.shape)
        self.assertEqual(obj3.shape, self.obj3.shape)


class JakesSampleGeneratorTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        Fd = 5  # Doppler frequency (in Hz)
        Ts = 1e-3  # Sampling interval (in seconds)
        NRays = 8  # Number of rays for the Jakes model

        self.obj1 = fading_generators.JakesSampleGenerator(Fd, Ts, NRays)
        self.obj2 = fading_generators.JakesSampleGenerator(Fd,
                                                           Ts,
                                                           NRays,
                                                           shape=(3, 2))
        self.Ts = Ts
        self.NRays = NRays
        self.Fd = Fd

    def test_phi_and_psi(self):
        # Note that phi and psi are computed during the object
        # creation. They would only need to change if the number of rays of
        # the Jakes model or the shape were changed.

        # The first dimension is equal to the number of rays of the Jakes
        # generator. The last dimension is set to 1 to allow broadcast with
        # the time dimension later.
        np.testing.assert_array_equal(self.obj1._phi_l.shape, [self.NRays, 1])
        np.testing.assert_array_equal(self.obj2._psi_l.shape,
                                      [self.NRays, 3, 2, 1])

    def test_properties(self):
        # Try to change Jakes parameters that phi and psi depend on
        self.assertIsNone(self.obj1.shape)
        self.assertEqual(self.obj2.shape, (3, 2))

        # Set the shape
        self.obj1.shape = 3
        self.obj2.shape = (4, 3)

        np.testing.assert_array_equal(self.obj1._phi_l.shape,
                                      [self.NRays, 3, 1])
        np.testing.assert_array_equal(self.obj2._psi_l.shape,
                                      [self.NRays, 4, 3, 1])

        # Try to set some attributes that are not allowed
        with self.assertRaises(AttributeError):
            # noinspection PyPropertyAccess
            self.obj1.L = 16
        with self.assertRaises(AttributeError):
            # noinspection PyPropertyAccess
            self.obj1.Ts = 5e-4
        with self.assertRaises(AttributeError):
            # noinspection PyPropertyAccess
            self.obj1.Fd = 50

    def test_generate_more_samples(self):
        sample1_obj1 = self.obj1.get_samples()
        sample1_obj2 = self.obj2.get_samples()
        self.assertEqual(sample1_obj1.shape, (1, ))
        self.assertEqual(sample1_obj2.shape, (3, 2, 1))

        # When we create the object it will generate one sample. Therefore,
        # the current_value time corresponds to the sampling interval
        self.assertAlmostEqual(self.obj1._current_time, self.Ts)
        self.assertAlmostEqual(self.obj2._current_time, self.Ts)

        # Generate 100 samples
        self.obj1.generate_more_samples(100)
        self.obj2.generate_more_samples(100)
        self.assertAlmostEqual(self.obj1._current_time, 101 * self.Ts)
        self.assertAlmostEqual(self.obj2._current_time, 101 * self.Ts)

    def test_skip_samples_for_next_generation(self):
        # Obj2 is a copy of self.obj1 and will generate the same samples
        obj2 = copy(self.obj1)

        # Generate one sample
        self.obj1.generate_more_samples()
        obj2.generate_more_samples()

        # Notice how they are equal
        np.testing.assert_array_almost_equal(self.obj1.get_samples(),
                                             obj2.get_samples())

        old_sample = obj2.get_samples()

        # Now lets generate 5 samples for self.obj1
        self.obj1.generate_more_samples(5)
        # and let's skip 5 samples for obj2
        obj2.skip_samples_for_next_generation(5)

        # obj2 still has the old sample
        np.testing.assert_array_almost_equal(obj2.get_samples(), old_sample)

        # Now let's generate 3 samples for both objects
        self.obj1.generate_more_samples(3)
        obj2.generate_more_samples(3)

        # Now let's test that they have the same samples thus confirming
        # the effect of skip_samples_for_next_generation
        np.testing.assert_array_almost_equal(self.obj1.get_samples(),
                                             obj2.get_samples())

    def test_get_similar_fading_generator(self):
        obj1 = self.obj1.get_similar_fading_generator()
        obj2 = self.obj2.get_similar_fading_generator()
        # Modify the new objects. Since we will only compare the parameters
        # (the same ones used in the constructor) then generating more
        # samples here should not be a problem.
        obj1.generate_more_samples()
        obj2.generate_more_samples()

        self.assertEqual(obj1.shape, self.obj1.shape)
        self.assertEqual(obj2.shape, self.obj2.shape)
        self.assertEqual(obj1._Fd, self.obj1._Fd)
        self.assertEqual(obj2._Fd, self.obj2._Fd)
        self.assertEqual(obj1._Ts, self.obj1._Ts)
        self.assertEqual(obj2._Ts, self.obj2._Ts)
        self.assertEqual(obj1._L, self.obj1._L)
        self.assertEqual(obj2._L, self.obj2._L)


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

        # Check the number of taps in the profiles
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
            np.array([
                -5.7, -7.6, -10.1, -10.2, -10.2, -11.5, -13.4, -16.3, -16.9,
                -17.1, -17.4, -19, -19, -19.8, -21.5, -21.6, -22.1, -22.6,
                -23.5, -24.3
            ]))
        np.testing.assert_array_almost_equal(
            tu.tap_delays,
            np.array([
                0, 217, 512, 514, 517, 674, 882, 1230, 1287, 1311, 1349, 1533,
                1535, 1622, 1818, 1836, 1884, 1943, 2048, 2140
            ]) * 1e-9)

        np.testing.assert_array_almost_equal(
            ra.tap_powers_dB,
            np.array([
                -5.2, -6.4, -8.4, -9.3, -10.0, -13.1, -15.3, -18.5, -20.4,
                -22.4
            ]))
        np.testing.assert_array_almost_equal(
            ra.tap_delays,
            np.array([0., 42., 101., 129., 149., 245., 312., 410., 469., 528])
            * 1e-9)

        np.testing.assert_array_almost_equal(
            ht.tap_powers_dB,
            np.array([
                -3.6, -8.9, -10.2, -11.5, -11.8, -12.7, -13.0, -16.2, -17.3,
                -17.7, -17.6, -22.7, -24.1, -25.8, -25.8, -26.2, -29.0, -29.9,
                -30.0, -30.7
            ]))
        np.testing.assert_array_almost_equal(
            ht.tap_delays,
            np.array([
                0., 356., 441., 528., 546., 609., 625., 842., 916., 941.,
                15000., 16172., 16492., 16876., 16882., 16978., 17615., 17827.,
                17849., 18016.
            ]) * 1e-9)

    def test_discretize(self):
        maxSystemBand = 40e6  # 40 MHz bandwidth
        # Number of subcarriers in this bandwidth
        max_num_of_subcarriers = math.floor(maxSystemBand / 15e3)
        # Find the maximum FFT size we can use which is below than or equal
        # to maxNumOfSubcarriersInt
        max_num_of_subcarriers = int(2**math.floor(
            math.log(max_num_of_subcarriers, 2)))
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
        # together.
        tap_powers_linear = tu.tap_powers_linear
        # The TDL class will normalized the tap powers so that the channel
        # has unit power.
        tap_powers_linear = tap_powers_linear / np.sum(tap_powers_linear)

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
        np.testing.assert_array_almost_equal(
            expected_discretized_tap_powers_linear,
            tu_discretized.tap_powers_linear)
        # Check if the discretized tap delays are correct. Note that they
        # are integers.
        np.testing.assert_array_equal(
            np.array(
                [0, 7, 16, 21, 27, 38, 40, 41, 47, 50, 56, 58, 60, 63, 66]),
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

    def test_num_taps_with_padding(self):
        tu = fading.COST259_TUx
        ra = fading.COST259_RAx
        ht = fading.COST259_HTx

        Ts = 3.255e-08

        # For non-discretized profiles an exception should be raised if we
        # try to get num_taps_with_padding
        with self.assertRaises(RuntimeError):
            _ = tu.num_taps_with_padding
        with self.assertRaises(RuntimeError):
            _ = ra.num_taps_with_padding
        with self.assertRaises(RuntimeError):
            _ = ht.num_taps_with_padding

        tu_d = tu.get_discretize_profile(Ts)
        ra_d = ra.get_discretize_profile(Ts)
        ht_d = ht.get_discretize_profile(Ts)

        self.assertEqual(tu_d.num_taps, 15)
        self.assertEqual(tu_d.num_taps_with_padding, 67)
        self.assertEqual(ra_d.num_taps, 10)
        self.assertEqual(ra_d.num_taps_with_padding, 17)
        self.assertEqual(ht_d.num_taps, 18)
        self.assertEqual(ht_d.num_taps_with_padding, 554)


class TdlImpulseResponseTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.Ts = 3.255e-08
        tu = fading.COST259_TUx
        tu_discretized = tu.get_discretize_profile(self.Ts)

        num_samples = 5
        self.tap_values = (np.random.randn(15, num_samples) +
                           1j * np.random.randn(15, num_samples))

        self.impulse_response = fading.TdlImpulseResponse(
            self.tap_values, tu_discretized)

    def test_constructor(self):
        num_samples = 5
        tap_values = (np.random.randn(15, num_samples) +
                      1j * np.random.randn(15, num_samples))
        tu = fading.COST259_TUx

        # If we try to create an impulse response object using a
        # non-discretized channel profile an exception is raised
        with self.assertRaises(RuntimeError):
            fading.TdlImpulseResponse(tap_values, tu)

    def test_properties(self):
        # With Ts = 3.255e-8, the discretized TU channel profile has 15 non
        # zero taps. Including the zero taps we have 67 taps. Here we will
        # test these dimensions

        num_taps = 15
        num_taps_with_padding = 67
        num_samples = 5
        Ts = self.Ts

        self.assertAlmostEqual(self.impulse_response.Ts, self.Ts)
        self.assertEqual(self.impulse_response.num_samples, num_samples)

        self.assertEqual(self.impulse_response.tap_values.shape,
                         (num_taps_with_padding, num_samples))
        self.assertEqual(self.impulse_response.tap_values_sparse.shape,
                         (num_taps, num_samples))

        np.testing.assert_array_equal(
            self.impulse_response.tap_indexes_sparse,
            np.array(
                [0, 7, 16, 21, 27, 38, 40, 41, 47, 50, 56, 58, 60, 63, 66]))

        np.testing.assert_array_almost_equal(
            self.impulse_response.tap_delays_sparse,
            Ts * np.array(
                [0, 7, 16, 21, 27, 38, 40, 41, 47, 50, 56, 58, 60, 63, 66]))

    def test_multiply(self):
        impulse_response_scaled = self.impulse_response * 0.42
        # Test that a new object is returned
        self.assertTrue(impulse_response_scaled is not self.impulse_response)
        # Test that it shares the same channel_profile object, the same Ts
        self.assertTrue(impulse_response_scaled.channel_profile is
                        self.impulse_response.channel_profile)
        self.assertTrue(impulse_response_scaled.Ts is self.impulse_response.Ts)

        self.assertTrue(self.impulse_response.tap_values_sparse is
                        not impulse_response_scaled.tap_values_sparse)
        np.testing.assert_array_almost_equal(
            impulse_response_scaled.tap_values_sparse,
            self.impulse_response.tap_values_sparse * 0.42)
        np.testing.assert_array_almost_equal(
            impulse_response_scaled.tap_values,
            self.impulse_response.tap_values * 0.42)

    def test_get_freq_response(self):
        fft_size = 1024

        freq_response = self.impulse_response.get_freq_response(fft_size)
        self.assertEqual(freq_response.shape, (fft_size, 5))

        tap_values = self.impulse_response.tap_values

        expected_frequency_response = np.zeros(shape=(fft_size, 5),
                                               dtype=complex)
        expected_frequency_response[:, 0] = np.fft.fft(tap_values[:, 0],
                                                       fft_size)
        expected_frequency_response[:, 1] = np.fft.fft(tap_values[:, 1],
                                                       fft_size)
        expected_frequency_response[:, 2] = np.fft.fft(tap_values[:, 2],
                                                       fft_size)
        expected_frequency_response[:, 3] = np.fft.fft(tap_values[:, 3],
                                                       fft_size)
        expected_frequency_response[:, 4] = np.fft.fft(tap_values[:, 4],
                                                       fft_size)
        np.testing.assert_array_almost_equal(freq_response,
                                             expected_frequency_response)

    def test_concatenate_samples(self):
        num_samples2 = 13
        tap_values2 = (np.random.randn(15, num_samples2) +
                       1j * np.random.randn(15, num_samples2))

        impulse_response2 = fading.TdlImpulseResponse(
            tap_values2, self.impulse_response.channel_profile)

        num_samples3 = 9
        tap_values3 = (np.random.randn(15, num_samples3) +
                       1j * np.random.randn(15, num_samples3))

        impulse_response3 = fading.TdlImpulseResponse(
            tap_values3, self.impulse_response.channel_profile)

        # Get the 3 TdlImpulseResponse objects in a list
        impulse_responses = [
            self.impulse_response, impulse_response2, impulse_response3
        ]

        # Concatenate the objects in the list and return a new
        # TdlImpulseResponse object.
        concatenated_impulse_response = \
            fading.TdlImpulseResponse.concatenate_samples(impulse_responses)

        self.assertEqual(concatenated_impulse_response.num_samples,
                         5 + num_samples2 + num_samples3)

        # xxxxxxxxxx test if the values of the concatenated taps xxxxxxxxxx
        np.testing.assert_array_almost_equal(
            concatenated_impulse_response.tap_values_sparse[:, 0:5],
            self.impulse_response.tap_values_sparse)

        np.testing.assert_array_almost_equal(
            concatenated_impulse_response.tap_values_sparse[:, 5:18],
            impulse_response2.tap_values_sparse)

        np.testing.assert_array_almost_equal(
            concatenated_impulse_response.tap_values_sparse[:, 18:],
            impulse_response3.tap_values_sparse)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test the delays xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_almost_equal(
            concatenated_impulse_response.tap_delays_sparse,
            self.impulse_response.tap_delays_sparse)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        self.assertAlmostEqual(concatenated_impulse_response.Ts,
                               self.impulse_response.Ts)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test if we call with a single TdlImpulseResponse object. The same
        # object should be returned.
        out = fading.TdlImpulseResponse.concatenate_samples(
            [impulse_response2])
        self.assertTrue(out is impulse_response2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxx Test if an exception is raised xxxxxxxxxxxxxxxxxx
        # If we try to concatenate impulse responses with different channel
        # profiles objects (identity, not just parameters) an exception
        # should be raised
        Ts = 3.255e-08
        tu = fading.COST259_TUx
        tu_discretized = tu.get_discretize_profile(Ts)
        impulse_response3 = fading.TdlImpulseResponse(tap_values2,
                                                      tu_discretized)
        with self.assertRaises(ValueError):
            fading.TdlImpulseResponse.concatenate_samples(
                [self.impulse_response, impulse_response3])
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_plot_impulse_response(self):
        # self.impulse_response.plot_impulse_response()
        pass

    def test_plot_frequency_response(self):
        # self.impulse_response.plot_frequency_response(300)
        pass


class TdlChannelTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        maxSystemBand = 40e6  # 40 MHz bandwidth
        # Number of subcarriers in this bandwidth
        max_num_of_subcarriers = math.floor(maxSystemBand / 15e3)
        # Find the maximum FFT size we can use which is below than or equal
        # to maxNumOfSubcarriersInt
        max_num_of_subcarriers = int(2**math.floor(
            math.log(max_num_of_subcarriers, 2)))
        # Calculate the actual bandwidth that we will use
        bandwidth = 15e3 * max_num_of_subcarriers

        self.Fd = 5  # Doppler frequency (in Hz)
        self.Ts = 1. / bandwidth  # Sampling interval (in seconds)
        self.NRays = 16  # Number of rays for the Jakes model

        # xxxxxxxxxx Create the TDL SISO channel for testing xxxxxxxxxxxxxx
        # Create the jakes object that will be passed to TdlChannel
        self.jakes = fading_generators.JakesSampleGenerator(self.Fd,
                                                            self.Ts,
                                                            self.NRays,
                                                            shape=None)

        self.tdlchannel = fading.TdlChannel(
            self.jakes,
            tap_powers_dB=fading.COST259_TUx.tap_powers_dB,
            tap_delays=fading.COST259_TUx.tap_delays)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # noinspection PyTypeChecker
    def test_constructor_and_num_taps(self):
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # test constructor if we only provide the fading generator

        # For the RayleighSampleGenerator generator Ts will be 1
        tdlchannel_ray = fading.TdlChannel(
            fading_generators.RayleighSampleGenerator())
        self.assertEqual(tdlchannel_ray.channel_profile.Ts, 1)

        # For the JakesSampleGenerator Ts will be the same value from Jakes
        # generator
        tdlchannel_jakes = fading.TdlChannel(
            fading_generators.JakesSampleGenerator())
        self.assertEqual(tdlchannel_jakes.channel_profile.Ts, 0.001)

        # In both cases the channel profile has only one tap with unitary
        # power end delay 0
        np.testing.assert_array_almost_equal(
            tdlchannel_ray.channel_profile.tap_powers_dB, 0.0)
        np.testing.assert_array_almost_equal(
            tdlchannel_jakes.channel_profile.tap_powers_dB, 0.0)
        np.testing.assert_array_almost_equal(
            tdlchannel_ray.channel_profile.tap_delays, 0.0)
        np.testing.assert_array_almost_equal(
            tdlchannel_jakes.channel_profile.tap_delays, 0.0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # The constructor provided the tap powers and delays. The
        # TdlChannel constructor used that, as well as the sampling time Ts
        # from the jakes object and created a custom channel profile
        self.assertAlmostEqual(self.tdlchannel.channel_profile.Ts, self.Ts)

        tdlchannel2 = fading.TdlChannel(self.jakes,
                                        channel_profile=fading.COST259_TUx)
        self.assertAlmostEqual(tdlchannel2.channel_profile.Ts, self.Ts)

        self.assertEqual(self.tdlchannel.num_taps, 15)
        self.assertEqual(tdlchannel2.num_taps, 15)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # This one has delays 10 times greater then the delays in the TU
        # channel profile. This means that it will have more discretized
        # taps
        tdlchannel3 = fading.TdlChannel(
            self.jakes,
            tap_powers_dB=fading.COST259_TUx.tap_powers_dB,
            tap_delays=10 * fading.COST259_TUx.tap_delays)

        self.assertEqual(tdlchannel3.num_taps, 20)
        self.assertEqual(tdlchannel3.num_taps_with_padding, 658)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test if an exception is raised if we provide Ts and it is
        # different from the one in the jakes object
        with self.assertRaises(RuntimeError):
            fading.TdlChannel(self.jakes,
                              tap_powers_dB=fading.COST259_TUx.tap_powers_dB,
                              tap_delays=10 * fading.COST259_TUx.tap_delays,
                              Ts=0.002)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test if an exception is raised if we provide an already
        # discretized channel profile, but its sample time is different
        # from the one in the jakes sample generator.
        tu_discretized = fading.COST259_TUx.get_discretize_profile(0.002)
        with self.assertRaises(RuntimeError):
            fading.TdlChannel(self.jakes, channel_profile=tu_discretized)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_num_taps_with_and_without_padding(self):
        self.assertEqual(self.tdlchannel.num_taps_with_padding, 67)
        self.assertEqual(self.tdlchannel.num_taps, 15)

    def test_generate_and_get_last_impulse_response(self):
        with self.assertRaises(RuntimeError):
            self.tdlchannel.get_last_impulse_response()
        self.tdlchannel.generate_impulse_response(num_samples=20)
        last_impulse_response = self.tdlchannel.get_last_impulse_response()

        # This is a SISO channel and therefore number of transmit and
        # receive antennas is returned as -1
        self.assertEqual(self.tdlchannel.num_tx_antennas, -1)
        self.assertEqual(self.tdlchannel.num_rx_antennas, -1)

        self.assertEqual(last_impulse_response.num_samples, 20)
        self.assertEqual(last_impulse_response.tap_values_sparse.shape,
                         (15, 20))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now test with a different shape
        #
        # For this Ts there are 15 non zero taps and 67 taps in total
        # including zero padding
        Ts = 3.255e-08
        jakes = fading_generators.JakesSampleGenerator(shape=(4, 3), Ts=Ts)
        tdlchannel = fading.TdlChannel(jakes, fading.COST259_TUx)
        tdlchannel.generate_impulse_response(10)
        last_impulse_response = tdlchannel.get_last_impulse_response()

        # This is a MIMO channel. Let's check the number of transmit and
        # receive antennas
        self.assertEqual(tdlchannel.num_tx_antennas, 3)
        self.assertEqual(tdlchannel.num_rx_antennas, 4)
        self.assertEqual(last_impulse_response.num_samples, 10)
        self.assertEqual(last_impulse_response.tap_values_sparse.shape,
                         (15, 4, 3, 10))
        self.assertEqual(last_impulse_response.tap_values.shape,
                         (67, 4, 3, 10))

    def test_corrupt_data(self):
        # xxxxxxxxxx Test sending just a single impulse xxxxxxxxxxxxxxxxxxx
        signal = np.array([1.])

        received_signal = self.tdlchannel.corrupt_data(signal)

        # Impulse response used to transmit the signal
        last_impulse_response = self.tdlchannel.get_last_impulse_response()

        # Since only one sample was sent and it is equal to 1, then the
        # received signal will be equal to the full_fading_map
        np.testing.assert_almost_equal(
            last_impulse_response.tap_values.flatten(), received_signal)

        # xxxxxxxxxx Test sending a vector with 10 samples xxxxxxxxxxxxxxxx
        num_samples = 10
        signal = (np.random.standard_normal(num_samples) +
                  1j * np.random.standard_normal(num_samples))
        received_signal = self.tdlchannel.corrupt_data(signal)
        last_impulse_response = self.tdlchannel.get_last_impulse_response()

        # Compute the expected received signal
        # For this Ts we have 15 discretized taps. The indexes of the 15
        # taps are:
        # [ 0,  7, 16, 21, 27, 38, 40, 41, 47, 50, 56, 58, 60, 63, 66]
        np.testing.assert_array_equal(
            last_impulse_response.tap_indexes_sparse,
            np.array(
                [0, 7, 16, 21, 27, 38, 40, 41, 47, 50, 56, 58, 60, 63, 66]))

        # Including zero padding, the impulse response has 67 taps. That
        # means the channel memory is equal to 66
        channel_memory = 66
        expected_received_signal = np.zeros(channel_memory + num_samples,
                                            dtype=complex)

        # Let's compute the expected received signal
        taps_sparse = last_impulse_response.tap_values_sparse
        expected_received_signal[0:0 + num_samples] += signal * taps_sparse[0]
        expected_received_signal[7:7 + num_samples] += signal * taps_sparse[1]
        expected_received_signal[16:16 +
                                 num_samples] += signal * taps_sparse[2]
        expected_received_signal[21:21 +
                                 num_samples] += signal * taps_sparse[3]
        expected_received_signal[27:27 +
                                 num_samples] += signal * taps_sparse[4]
        expected_received_signal[38:38 +
                                 num_samples] += signal * taps_sparse[5]
        expected_received_signal[40:40 +
                                 num_samples] += signal * taps_sparse[6]
        expected_received_signal[41:41 +
                                 num_samples] += signal * taps_sparse[7]
        expected_received_signal[47:47 +
                                 num_samples] += signal * taps_sparse[8]
        expected_received_signal[50:50 +
                                 num_samples] += signal * taps_sparse[9]
        expected_received_signal[56:56 +
                                 num_samples] += signal * taps_sparse[10]
        expected_received_signal[58:58 +
                                 num_samples] += signal * taps_sparse[11]
        expected_received_signal[60:60 +
                                 num_samples] += signal * taps_sparse[12]
        expected_received_signal[63:63 +
                                 num_samples] += signal * taps_sparse[13]
        expected_received_signal[66:66 +
                                 num_samples] += signal * taps_sparse[14]

        # Check if the received signal is correct
        np.testing.assert_array_almost_equal(expected_received_signal,
                                             received_signal)

    def test_corrupt_data_in_freq_domain(self):
        fft_size = 16
        num_samples = 5 * fft_size
        signal = np.ones(num_samples)
        # num_full_blocks = num_samples // fft_size

        jakes1 = fading_generators.JakesSampleGenerator(self.Fd,
                                                        self.Ts,
                                                        self.NRays,
                                                        shape=None)

        # Note that tdlchannel will modify the jakes1 object
        tdlchannel1 = fading.TdlChannel(fading_generator=jakes1,
                                        channel_profile=fading.COST259_TUx)

        # we want tdlchannel2 to be a copy of tdlchannel1 and generate the
        # same samples
        tdlchannel2 = copy(tdlchannel1)
        # After the copy it will use the same fading_generator
        # object. Let's copy the fading_generator and replace the one in
        # tdlchannel2 with the copy
        jakes2 = copy(jakes1)
        tdlchannel2._fading_generator = jakes2

        # xxxxxxxxxx Perform the actual transmission xxxxxxxxxxxxxxxxxxxxxx
        received_signal = tdlchannel1.corrupt_data_in_freq_domain(
            signal, fft_size)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Compute frequency response for all samples xxxxxxxxxxx
        tdlchannel2.generate_impulse_response(num_samples)
        impulse_response_all = tdlchannel2.get_last_impulse_response()
        # Note that here we have the frequency response for `num_samples`
        # samples. But the `corrupt_data_in_freq_domain` method only use
        # multiples of `fft_size` (0*fft_size, 1*fft_size, ...)
        freq_response_all = impulse_response_all.get_freq_response(fft_size)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test if the received signal is correct xxxxxxxxxxxxxxx
        # First OFDM symbol
        # Since we transmitted just 1's, then the received signal should be
        # equal to the frequency response af the start of the OFDM symbol
        np.testing.assert_array_almost_equal(received_signal[0:fft_size],
                                             freq_response_all[:, 0],
                                             decimal=8)
        # Second OFDM symbol
        np.testing.assert_array_almost_equal(received_signal[fft_size:2 *
                                                             fft_size],
                                             freq_response_all[:, fft_size],
                                             decimal=8)

        # Third OFDM symbol
        np.testing.assert_array_almost_equal(
            received_signal[2 * fft_size:3 * fft_size],
            freq_response_all[:, 2 * fft_size],
            decimal=8)

        # Fourth OFDM symbol
        np.testing.assert_array_almost_equal(
            received_signal[3 * fft_size:4 * fft_size],
            freq_response_all[:, 3 * fft_size],
            decimal=8)

        # Fifth OFDM symbol
        np.testing.assert_array_almost_equal(
            received_signal[4 * fft_size:5 * fft_size],
            freq_response_all[:, 4 * fft_size],
            decimal=8)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test impulse response after transmission xxxxxxxxxxxxx
        # Since signal corresponds to 5 OFDM symbols, then we should have 5
        # "samples" in the returned impulse response.
        impulse_response = tdlchannel1.get_last_impulse_response()
        self.assertEqual(impulse_response.num_samples, num_samples // fft_size)

        freq_response = impulse_response.get_freq_response(fft_size)
        np.testing.assert_array_almost_equal(
            freq_response[:, 0], freq_response_all[:, 0 * fft_size])
        np.testing.assert_array_almost_equal(
            freq_response[:, 1], freq_response_all[:, 1 * fft_size])
        np.testing.assert_array_almost_equal(
            freq_response[:, 2], freq_response_all[:, 2 * fft_size])
        np.testing.assert_array_almost_equal(
            freq_response[:, 3], freq_response_all[:, 3 * fft_size])
        np.testing.assert_array_almost_equal(
            freq_response[:, 4], freq_response_all[:, 4 * fft_size])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_corrupt_data_in_freq_domain2(self):
        # This method tests corrupt_data_in_freq_domain, but now specifying
        # the indexes of the used subcarriers

        fft_size = 16
        num_samples = 5 * fft_size
        signal = np.ones(num_samples)
        # For these particular indexes we will use half of the subcarriers
        subcarrier_indexes = np.r_[0:fft_size:2]

        jakes1 = fading_generators.JakesSampleGenerator(self.Fd,
                                                        self.Ts,
                                                        self.NRays,
                                                        shape=None)

        # Note that tdlchannel will modify the jakes1 object
        tdlchannel1 = fading.TdlChannel(fading_generator=jakes1,
                                        channel_profile=fading.COST259_TUx)

        # we want tdlchannel2 to be a copy of tdlchannel1 and generate the
        # same samples
        tdlchannel2 = copy(tdlchannel1)
        # After the copy it will use the same fading_generator
        # object. Let's copy the fading_generator and replace the one in
        # tdlchannel2 with the copy
        jakes2 = copy(jakes1)
        tdlchannel2._fading_generator = jakes2

        # xxxxxxxxxx Perform the actual transmission xxxxxxxxxxxxxxxxxxxxxx
        received_signal = tdlchannel1.corrupt_data_in_freq_domain(
            signal, fft_size, subcarrier_indexes)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Compute frequency response for all samples xxxxxxxxxxx
        tdlchannel2.generate_impulse_response(2 * num_samples)
        impulse_response_all = tdlchannel2.get_last_impulse_response()
        # Note that here we have the frequency response for `num_samples`
        # samples. But the `corrupt_data_in_freq_domain` method only use
        # multiples of `fft_size` (0*fft_size, 1*fft_size, ...)
        freq_response_all = impulse_response_all.get_freq_response(fft_size)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test if the received signal is correct xxxxxxxxxxxxxxx
        block_size = fft_size // 2
        # First OFDM symbol
        # Since we transmitted just 1's, then the received signal should be
        # equal to the frequency response af the start of the OFDM symbol
        np.testing.assert_array_almost_equal(
            received_signal[0:block_size],
            freq_response_all[subcarrier_indexes, 0],
            decimal=8)

        # Second OFDM symbol
        np.testing.assert_array_almost_equal(
            received_signal[block_size:2 * block_size],
            freq_response_all[subcarrier_indexes, fft_size],
            decimal=8)

        # Third OFDM symbol
        np.testing.assert_array_almost_equal(
            received_signal[2 * block_size:3 * block_size],
            freq_response_all[subcarrier_indexes, 2 * fft_size],
            decimal=8)

        # Remaining OFDM symbols (from 4 to 10)
        for i in range(3, 10):
            np.testing.assert_array_almost_equal(
                received_signal[i * block_size:(i + 1) * block_size],
                freq_response_all[subcarrier_indexes, i * fft_size],
                decimal=8)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test impulse response after transmission xxxxxxxxxxxxx
        # Since signal corresponds to 10 OFDM symbols (using only half of
        # the subcarriers in each OFDM symbol), then we should have 10
        # "samples" in the returned impulse response.
        impulse_response = tdlchannel1.get_last_impulse_response()
        self.assertEqual(impulse_response.num_samples,
                         num_samples // block_size)

        freq_response = impulse_response.get_freq_response(fft_size)
        for i in range(10):
            np.testing.assert_array_almost_equal(
                freq_response[:, i], freq_response_all[:, i * fft_size])
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# noinspection PyMethodMayBeStatic
class TdlMIMOChannelTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        maxSystemBand = 40e6  # 40 MHz bandwidth
        # Number of subcarriers in this bandwidth
        max_num_of_subcarriers = math.floor(maxSystemBand / 15e3)
        # Find the maximum FFT size we can use which is below than or equal
        # to maxNumOfSubcarriersInt
        max_num_of_subcarriers = int(2**math.floor(
            math.log(max_num_of_subcarriers, 2)))
        # Calculate the actual bandwidth that we will use
        bandwidth = 15e3 * max_num_of_subcarriers

        self.Fd = 5  # Doppler frequency (in Hz)
        self.Ts = 1. / bandwidth  # Sampling interval (in seconds)
        self.NRays = 16  # Number of rays for the Jakes model

        # xxxxxxxxxx Create the TDL MIMO channel for testing xxxxxxxxxxxxxx
        # Create the jakes object that will be passed to TdlMimoChannel
        self.jakes = fading_generators.JakesSampleGenerator(self.Fd,
                                                            self.Ts,
                                                            self.NRays,
                                                            shape=(3, 2))

        self.tdlmimochannel = fading.TdlMimoChannel(
            self.jakes,
            tap_powers_dB=fading.COST259_TUx.tap_powers_dB,
            tap_delays=fading.COST259_TUx.tap_delays)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_constructor_and_num_taps(self):
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # test constructor if we only provide the fading generator

        # For the RayleighSampleGenerator generator Ts will be 1
        tdlmimochannel_ray = fading.TdlMimoChannel(
            fading_generators.RayleighSampleGenerator(shape=(3, 2)))
        self.assertEqual(tdlmimochannel_ray.channel_profile.Ts, 1)
        self.assertEqual(tdlmimochannel_ray.num_tx_antennas, 2)
        self.assertEqual(tdlmimochannel_ray.num_rx_antennas, 3)

        # For the JakesSampleGenerator Ts will be the same value from Jakes
        # generator
        tdlmimochannel_jakes = fading.TdlMimoChannel(
            fading_generators.JakesSampleGenerator(shape=(3, 2)))
        self.assertEqual(tdlmimochannel_jakes.channel_profile.Ts, 0.001)
        self.assertEqual(tdlmimochannel_jakes.num_tx_antennas, 2)
        self.assertEqual(tdlmimochannel_jakes.num_rx_antennas, 3)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test changing the shape after creation xxxxxxxxxxxxxxx
        # Note that we are using a TdlChannel object instead of a
        # TdlMimoChannel object.
        tdlchannel_jakes = fading.TdlChannel(
            fading_generators.JakesSampleGenerator())
        tdlchannel_jakes.generate_impulse_response()
        # Set the number of Tx and Rx antennas
        tdlchannel_jakes.set_num_antennas(4, 3)
        self.assertEqual(tdlchannel_jakes.num_rx_antennas, 4)
        self.assertEqual(tdlchannel_jakes.num_tx_antennas, 3)
        self.assertEqual(tdlchannel_jakes.num_taps, 1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_num_taps_with_and_without_padding(self):
        self.assertEqual(self.tdlmimochannel.num_taps_with_padding, 67)
        self.assertEqual(self.tdlmimochannel.num_taps, 15)

    def test_generate_and_get_last_impulse_response(self):
        with self.assertRaises(RuntimeError):
            self.tdlmimochannel.get_last_impulse_response()
        self.tdlmimochannel.generate_impulse_response(num_samples=20)
        last_impulse_response = self.tdlmimochannel.get_last_impulse_response()

        self.assertEqual(last_impulse_response.num_samples, 20)
        self.assertEqual(last_impulse_response.tap_values_sparse.shape,
                         (15, 3, 2, 20))
        self.assertEqual(last_impulse_response.tap_values.shape,
                         (67, 3, 2, 20))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # For this Ts there are 15 non zero taps and 67 taps in total
        # including zero padding
        Ts = 3.255e-08
        jakes = fading_generators.JakesSampleGenerator(shape=(4, 3), Ts=Ts)
        tdlmimochannel = fading.TdlMimoChannel(jakes, fading.COST259_TUx)
        tdlmimochannel.generate_impulse_response(10)
        last_impulse_response = tdlmimochannel.get_last_impulse_response()
        self.assertEqual(last_impulse_response.num_samples, 10)
        self.assertEqual(last_impulse_response.tap_values_sparse.shape,
                         (15, 4, 3, 10))
        self.assertEqual(last_impulse_response.tap_values.shape,
                         (67, 4, 3, 10))

    def test_corrupt_data(self):
        # xxxxx Test sending single impulse in flat fading channel xxxxxxxx
        jakes = fading_generators.JakesSampleGenerator(shape=(3, 2))

        tdlmimochannel_flat = fading.TdlMimoChannel(
            jakes, channel_profile=fading.COST259_TUx)
        num_samples = 3
        signal = (np.random.randn(2, num_samples) +
                  1j * np.random.randn(2, num_samples))

        received_signal_flat = tdlmimochannel_flat.corrupt_data(signal)

        # Impulse response used to transmit the signal
        last_impulse_response = tdlmimochannel_flat.get_last_impulse_response()

        # Impulse response at each sample is just a matrix, since we only
        # have one tap
        h0 = last_impulse_response.tap_values[0, :, :, 0]
        h1 = last_impulse_response.tap_values[0, :, :, 1]
        h2 = last_impulse_response.tap_values[0, :, :, 2]
        expected_received_signal_flat = np.zeros((3, num_samples),
                                                 dtype=complex)
        expected_received_signal_flat[:, 0] = h0.dot(signal[:, 0])
        expected_received_signal_flat[:, 1] = h1.dot(signal[:, 1])
        expected_received_signal_flat[:, 2] = h2.dot(signal[:, 2])

        # Since only one sample was sent and it is equal to 1, then the
        # received signal will be equal to the full_fading_map
        np.testing.assert_almost_equal(expected_received_signal_flat,
                                       received_signal_flat)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test sending a vector with 10 samples xxxxxxxxxxxxxxxx
        num_samples = 10
        signal = (np.random.randn(2, num_samples) +
                  1j * np.random.randn(2, num_samples))
        received_signal = self.tdlmimochannel.corrupt_data(signal)
        last_impulse_response = self.tdlmimochannel.get_last_impulse_response()

        # Compute the expected received signal
        # For this Ts we have 15 discretized taps. The indexes of the 15
        # taps are:
        # [ 0,  7, 16, 21, 27, 38, 40, 41, 47, 50, 56, 58, 60, 63, 66]
        np.testing.assert_array_equal(
            last_impulse_response.tap_indexes_sparse,
            np.array(
                [0, 7, 16, 21, 27, 38, 40, 41, 47, 50, 56, 58, 60, 63, 66]))

        # Including zero padding, the impulse response has 67 taps. That
        # means the channel memory is equal to 66
        channel_memory = 66
        self.assertEqual(received_signal.shape,
                         (self.tdlmimochannel.num_rx_antennas,
                          num_samples + channel_memory))

        expected_received_signal = np.zeros(
            (self.tdlmimochannel.num_rx_antennas,
             channel_memory + num_samples),
            dtype=complex)

        # Let's compute the expected received signal
        tap_values_sparse = last_impulse_response.tap_values_sparse

        expected_received_signal[:, 0:0 + num_samples] += (
            signal[0] * tap_values_sparse[0, :, 0, :] +
            signal[1] * tap_values_sparse[0, :, 1, :])
        expected_received_signal[:, 7:7 + num_samples] += (
            signal[0] * tap_values_sparse[1, :, 0, :] +
            signal[1] * tap_values_sparse[1, :, 1, :])
        expected_received_signal[:, 16:16 + num_samples] += (
            signal[0] * tap_values_sparse[2, :, 0, :] +
            signal[1] * tap_values_sparse[2, :, 1, :])
        expected_received_signal[:, 21:21 + num_samples] += (
            signal[0] * tap_values_sparse[3, :, 0, :] +
            signal[1] * tap_values_sparse[3, :, 1, :])
        expected_received_signal[:, 27:27 + num_samples] += (
            signal[0] * tap_values_sparse[4, :, 0, :] +
            signal[1] * tap_values_sparse[4, :, 1, :])
        expected_received_signal[:, 38:38 + num_samples] += (
            signal[0] * tap_values_sparse[5, :, 0, :] +
            signal[1] * tap_values_sparse[5, :, 1, :])
        expected_received_signal[:, 40:40 + num_samples] += (
            signal[0] * tap_values_sparse[6, :, 0, :] +
            signal[1] * tap_values_sparse[6, :, 1, :])
        expected_received_signal[:, 41:41 + num_samples] += (
            signal[0] * tap_values_sparse[7, :, 0, :] +
            signal[1] * tap_values_sparse[7, :, 1, :])
        expected_received_signal[:, 47:47 + num_samples] += (
            signal[0] * tap_values_sparse[8, :, 0, :] +
            signal[1] * tap_values_sparse[8, :, 1, :])
        expected_received_signal[:, 50:50 + num_samples] += (
            signal[0] * tap_values_sparse[9, :, 0, :] +
            signal[1] * tap_values_sparse[9, :, 1, :])
        expected_received_signal[:, 56:56 + num_samples] += (
            signal[0] * tap_values_sparse[10, :, 0, :] +
            signal[1] * tap_values_sparse[10, :, 1, :])
        expected_received_signal[:, 58:58 + num_samples] += (
            signal[0] * tap_values_sparse[11, :, 0, :] +
            signal[1] * tap_values_sparse[11, :, 1, :])
        expected_received_signal[:, 60:60 + num_samples] += (
            signal[0] * tap_values_sparse[12, :, 0, :] +
            signal[1] * tap_values_sparse[12, :, 1, :])
        expected_received_signal[:, 63:63 + num_samples] += (
            signal[0] * tap_values_sparse[13, :, 0, :] +
            signal[1] * tap_values_sparse[13, :, 1, :])
        expected_received_signal[:, 66:66 + num_samples] += (
            signal[0] * tap_values_sparse[14, :, 0, :] +
            signal[1] * tap_values_sparse[14, :, 1, :])

        # Check if the received signal is correct
        np.testing.assert_array_almost_equal(expected_received_signal,
                                             received_signal)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Test with switched directions xxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # First let's create a channel without time variation so that we
        # can perform the transmission twice (once in each direction) and
        # we have the same fading.
        maxSystemBand = 40e6  # 40 MHz bandwidth
        # Number of subcarriers in this bandwidth
        max_num_of_subcarriers = math.floor(maxSystemBand / 15e3)
        # Find the maximum FFT size we can use which is below than or equal
        # to maxNumOfSubcarriersInt
        max_num_of_subcarriers = int(2**math.floor(
            math.log(max_num_of_subcarriers, 2)))
        # Calculate the actual bandwidth that we will use
        bandwidth = 15e3 * max_num_of_subcarriers

        Fd = 0  # Doppler frequency (in Hz)
        Ts = 1. / bandwidth  # Sampling interval (in seconds)
        NRays = 16  # Number of rays for the Jakes model

        # xxxxxxxxxx Create the TDL MIMO channel for testing xxxxxxxxxxxxxx

        # Create the jakes object that will be passed to TdlChannel
        num_rx_ant = 3
        num_tx_ant = 2
        jakes = fading_generators.JakesSampleGenerator(Fd,
                                                       Ts,
                                                       NRays,
                                                       shape=(num_rx_ant,
                                                              num_tx_ant))

        tdlmimochannel = fading.TdlMimoChannel(
            jakes, channel_profile=fading.COST259_TUx)

        num_samples = 10

        signal_uplink = np.random.randint(0, 10, (num_rx_ant, num_samples))

        # If we try to set switched_direction to anything that is not True
        # or False an exception is raised
        with self.assertRaises(TypeError):
            tdlmimochannel.switched_direction = 1

        # Now correctly set switched_direction to True
        tdlmimochannel.switched_direction = True  # switch to uplink
        # received_signal_uplink = tdlmimochannel.corrupt_data(signal_uplink)
        last_impulse_response = self.tdlmimochannel.get_last_impulse_response()

        # Let's compute the expected received signal
        # tap_values_sparse = last_impulse_response.tap_values_sparse

        expected_received_signal_uplink = np.zeros(
            (num_tx_ant, channel_memory + num_samples), dtype=complex)

        # Let's compute the expected received signal
        tap_values_sparse = last_impulse_response.tap_values_sparse

        expected_received_signal_uplink[:, 0:0 + num_samples] += (
            signal_uplink[0] * tap_values_sparse[0, 0, :, :] +
            signal_uplink[1] * tap_values_sparse[0, 1, :, :] +
            signal_uplink[2] * tap_values_sparse[0, 2, :, :])
        expected_received_signal_uplink[:, 7:7 + num_samples] += (
            signal[0] * tap_values_sparse[1, 0, :, :] +
            signal[1] * tap_values_sparse[1, 1, :, :])
        expected_received_signal_uplink[:, 16:16 + num_samples] += (
            signal[0] * tap_values_sparse[2, 0, :, :] +
            signal[1] * tap_values_sparse[2, 1, :, :])
        expected_received_signal_uplink[:, 21:21 + num_samples] += (
            signal[0] * tap_values_sparse[3, 0, :, :] +
            signal[1] * tap_values_sparse[3, 1, :, :])
        expected_received_signal_uplink[:, 27:27 + num_samples] += (
            signal[0] * tap_values_sparse[4, 0, :, :] +
            signal[1] * tap_values_sparse[4, 1, :, :])
        expected_received_signal_uplink[:, 38:38 + num_samples] += (
            signal[0] * tap_values_sparse[5, 0, :, :] +
            signal[1] * tap_values_sparse[5, 1, :, :])
        expected_received_signal_uplink[:, 40:40 + num_samples] += (
            signal[0] * tap_values_sparse[6, 0, :, :] +
            signal[1] * tap_values_sparse[6, 1, :, :])
        expected_received_signal_uplink[:, 41:41 + num_samples] += (
            signal[0] * tap_values_sparse[7, 0, :, :] +
            signal[1] * tap_values_sparse[7, 1, :, :])
        expected_received_signal_uplink[:, 47:47 + num_samples] += (
            signal[0] * tap_values_sparse[8, 0, :, :] +
            signal[1] * tap_values_sparse[8, 1, :, :])
        expected_received_signal_uplink[:, 50:50 + num_samples] += (
            signal[0] * tap_values_sparse[9, 0, :, :] +
            signal[1] * tap_values_sparse[9, 1, :, :])
        expected_received_signal_uplink[:, 56:56 + num_samples] += (
            signal[0] * tap_values_sparse[10, 0, :, :] +
            signal[1] * tap_values_sparse[10, 1, :, :])
        expected_received_signal_uplink[:, 58:58 + num_samples] += (
            signal[0] * tap_values_sparse[11, 0, :] +
            signal[1] * tap_values_sparse[11, 1, :])
        expected_received_signal_uplink[:, 60:60 + num_samples] += (
            signal[0] * tap_values_sparse[12, 0, :, :] +
            signal[1] * tap_values_sparse[12, 1, :, :])
        expected_received_signal_uplink[:, 63:63 + num_samples] += (
            signal[0] * tap_values_sparse[13, 0, :, :] +
            signal[1] * tap_values_sparse[13, 1, :, :])
        expected_received_signal_uplink[:, 66:66 + num_samples] += (
            signal[0] * tap_values_sparse[14, 0, :, :] +
            signal[1] * tap_values_sparse[14, 1, :, :])

        # Check if the received signal is correct
        np.testing.assert_array_almost_equal(expected_received_signal,
                                             received_signal)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_corrupt_data2(self):
        # This method tests test_corrupt_data, but now for a SIMO
        # system. The only difference is that the transmit signal can be a
        # 1D array (since we only have one transmit stream) or a 2D array
        # with only 1 row. Both cases should yield the same solution.

        # xxxxx Test sending single impulse in flat fading channel xxxxxxxx
        jakes = fading_generators.JakesSampleGenerator(shape=(3, 1),
                                                       Fd=0,
                                                       Ts=2e-7)

        tdlmimochannel_flat = fading.TdlMimoChannel(
            jakes, channel_profile=fading.COST259_TUx)
        num_samples = 20
        signal1 = (np.random.standard_normal(num_samples) +
                   1j * np.random.standard_normal(num_samples))
        signal2 = signal1[np.newaxis, :]

        received_signal_flat1 = tdlmimochannel_flat.corrupt_data(signal1)
        received_signal_flat2 = tdlmimochannel_flat.corrupt_data(signal2)

        np.testing.assert_array_almost_equal(received_signal_flat1,
                                             received_signal_flat2)

    def test_corrupt_data_in_freq_domain(self):
        fft_size = 16
        num_samples = 3 * fft_size
        signal = np.ones((2, num_samples))
        # num_full_blocks = num_samples // fft_size

        jakes1 = fading_generators.JakesSampleGenerator(self.Fd,
                                                        self.Ts,
                                                        self.NRays,
                                                        shape=(3, 2))

        # Note that tdlmimochannel will modify the jakes1 object
        tdlmimochannel1 = fading.TdlMimoChannel(
            fading_generator=jakes1, channel_profile=fading.COST259_TUx)

        # we want tdlmimochannel2 to be a copy of tdlmimochannel1 and
        # generate the same samples
        tdlmimochannel2 = copy(tdlmimochannel1)
        # After the copy it will use the same fading_generator
        # object. Let's copy the fading_generator and replace the one in
        # tdlmimochannel2 with the copy
        jakes2 = copy(jakes1)
        tdlmimochannel2._fading_generator = jakes2

        # xxxxxxxxxx Perform the actual transmission xxxxxxxxxxxxxxxxxxxxxx
        received_signal = tdlmimochannel1.corrupt_data_in_freq_domain(
            signal, fft_size)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        self.assertEqual(received_signal.shape, (3, num_samples))

        # xxxxxxxxxx Compute frequency response for all samples xxxxxxxxxxx
        tdlmimochannel2.generate_impulse_response(num_samples)
        impulse_response_all = tdlmimochannel2.get_last_impulse_response()
        # Note that here we have the frequency response for `num_samples`
        # samples. But the `corrupt_data_in_freq_domain` method only use
        # multiples of `fft_size` (0*fft_size, 1*fft_size, ...)
        freq_response_all = impulse_response_all.get_freq_response(fft_size)
        # Frequency response from each transmitter to each receiver
        freq_response00 = freq_response_all[:, 0, 0, :]
        freq_response01 = freq_response_all[:, 0, 1, :]
        freq_response10 = freq_response_all[:, 1, 0, :]
        freq_response11 = freq_response_all[:, 1, 1, :]
        freq_response20 = freq_response_all[:, 2, 0, :]
        freq_response21 = freq_response_all[:, 2, 1, :]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test if the received signal is correct xxxxxxxxxxxxxxx
        # First OFDM symbol
        # Since we transmitted just 1's, then the received signal should be
        # equal to the frequency response af the start of the OFDM symbol

        # Receiver 0
        np.testing.assert_array_almost_equal(
            received_signal[0, 0:fft_size],
            freq_response00[:, 0] + freq_response01[:, 0])
        # Receiver 1
        np.testing.assert_array_almost_equal(
            received_signal[1, 0:fft_size],
            freq_response10[:, 0] + freq_response11[:, 0])
        # Receiver 2
        np.testing.assert_array_almost_equal(
            received_signal[2, 0:fft_size],
            freq_response20[:, 0] + freq_response21[:, 0])

        # Second OFDM symbol
        np.testing.assert_array_almost_equal(
            received_signal[0, fft_size:2 * fft_size],
            freq_response00[:, fft_size] + freq_response01[:, fft_size])
        np.testing.assert_array_almost_equal(
            received_signal[1, fft_size:2 * fft_size],
            freq_response10[:, fft_size] + freq_response11[:, fft_size])
        np.testing.assert_array_almost_equal(
            received_signal[2, fft_size:2 * fft_size],
            freq_response20[:, fft_size] + freq_response21[:, fft_size])

        # Third OFDM symbol
        np.testing.assert_array_almost_equal(
            received_signal[0, 2 * fft_size:3 * fft_size],
            freq_response00[:, 2 * fft_size] +
            freq_response01[:, 2 * fft_size])
        np.testing.assert_array_almost_equal(
            received_signal[1, 2 * fft_size:3 * fft_size],
            freq_response10[:, 2 * fft_size] +
            freq_response11[:, 2 * fft_size])
        np.testing.assert_array_almost_equal(
            received_signal[2, 2 * fft_size:3 * fft_size],
            freq_response20[:, 2 * fft_size] +
            freq_response21[:, 2 * fft_size])
        # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test impulse response after transmission xxxxxxxxxxxxx
        # Since signal corresponds to 3 OFDM symbols, then we should have 3
        # "samples" in the returned impulse response.
        impulse_response = tdlmimochannel1.get_last_impulse_response()
        self.assertEqual(impulse_response.num_samples, num_samples // fft_size)

        freq_response = impulse_response.get_freq_response(fft_size)

        np.testing.assert_array_almost_equal(
            freq_response[:, :, :, 0], freq_response_all[:, :, :,
                                                         0 * fft_size])
        np.testing.assert_array_almost_equal(
            freq_response[:, :, :, 1], freq_response_all[:, :, :,
                                                         1 * fft_size])
        np.testing.assert_array_almost_equal(
            freq_response[:, :, :, 2], freq_response_all[:, :, :,
                                                         2 * fft_size])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test corrupt signal with wrong number of elements xxxxxxxxx
        signal2 = np.ones((2, num_samples - 1))
        with self.assertRaises(ValueError):
            tdlmimochannel1.corrupt_data_in_freq_domain(signal2, fft_size)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Test with switched directions xxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # First let's create a channel without time variation so that we
        # can perform the transmission twice (once in each direction) and
        # we have the same fading.
        maxSystemBand = 40e6  # 40 MHz bandwidth
        # Number of subcarriers in this bandwidth
        max_num_of_subcarriers = math.floor(maxSystemBand / 15e3)
        # Find the maximum FFT size we can use which is below than or equal
        # to maxNumOfSubcarriersInt
        max_num_of_subcarriers = int(2**math.floor(
            math.log(max_num_of_subcarriers, 2)))
        # Calculate the actual bandwidth that we will use
        bandwidth = 15e3 * max_num_of_subcarriers

        Fd = 0  # Doppler frequency (in Hz)
        Ts = 1. / bandwidth  # Sampling interval (in seconds)
        NRays = 16  # Number of rays for the Jakes model

        # xxxxxxxxxx Create the TDL MIMO channel for testing xxxxxxxxxxxxxx

        # Create the jakes object that will be passed to TdlChannel
        num_rx_ant = 3
        num_tx_ant = 2
        jakes = fading_generators.JakesSampleGenerator(Fd,
                                                       Ts,
                                                       NRays,
                                                       shape=(num_rx_ant,
                                                              num_tx_ant))

        tdlmimochannel = fading.TdlMimoChannel(
            jakes, channel_profile=fading.COST259_TUx)

        signal_uplink = np.random.randint(0, 10, (num_rx_ant, num_samples))
        tdlmimochannel.switched_direction = True  # switch to uplink
        received_signal_uplink = tdlmimochannel.corrupt_data_in_freq_domain(
            signal_uplink, fft_size)

        self.assertEqual(received_signal_uplink.shape,
                         (num_tx_ant, num_samples))

        last_impulse_response = tdlmimochannel.get_last_impulse_response()
        freq_response_all = last_impulse_response.get_freq_response(fft_size)
        # Frequency response from each transmitter to each receiver
        freq_response00 = freq_response_all[:, 0, 0, :]
        freq_response01 = freq_response_all[:, 0, 1, :]
        freq_response10 = freq_response_all[:, 1, 0, :]
        freq_response11 = freq_response_all[:, 1, 1, :]
        freq_response20 = freq_response_all[:, 2, 0, :]
        freq_response21 = freq_response_all[:, 2, 1, :]

        # xxxxxxxxxx First OFDM symbol
        expected_received_signal_uplink = np.zeros((num_tx_ant, num_samples),
                                                   dtype=complex)
        # First antenna
        expected_received_signal_uplink[0, 0:fft_size] = (
            signal_uplink[0, 0:fft_size] * freq_response00[:, 0] +
            signal_uplink[1, 0:fft_size] * freq_response10[:, 0] +
            signal_uplink[2, 0:fft_size] * freq_response20[:, 0])
        # Second antenna
        expected_received_signal_uplink[1, 0:fft_size] = (
            signal_uplink[0, 0:fft_size] * freq_response01[:, 0] +
            signal_uplink[1, 0:fft_size] * freq_response11[:, 0] +
            signal_uplink[2, 0:fft_size] * freq_response21[:, 0])
        # xxxxxxxxxx Second OFDM symbol
        # First antenna
        expected_received_signal_uplink[0, fft_size:2 * fft_size] = (
            signal_uplink[0, fft_size:2 * fft_size] * freq_response00[:, 1] +
            signal_uplink[1, fft_size:2 * fft_size] * freq_response10[:, 1] +
            signal_uplink[2, fft_size:2 * fft_size] * freq_response20[:, 1])
        # Second antenna
        expected_received_signal_uplink[1, fft_size:2 * fft_size] = (
            signal_uplink[0, fft_size:2 * fft_size] * freq_response01[:, 1] +
            signal_uplink[1, fft_size:2 * fft_size] * freq_response11[:, 1] +
            signal_uplink[2, fft_size:2 * fft_size] * freq_response21[:, 1])
        # xxxxxxxxxx ThirdOFDM symbol
        # First antenna
        expected_received_signal_uplink[0, 2 * fft_size:3 * fft_size] = (
            signal_uplink[0, 2 * fft_size:3 * fft_size] * freq_response00[:, 2]
            +
            signal_uplink[1, 2 * fft_size:3 * fft_size] * freq_response10[:, 2]
            + signal_uplink[2, 2 * fft_size:3 * fft_size] *
            freq_response20[:, 2])
        # Second antenna
        expected_received_signal_uplink[1, 2 * fft_size:3 * fft_size] = (
            signal_uplink[0, 2 * fft_size:3 * fft_size] * freq_response01[:, 2]
            +
            signal_uplink[1, 2 * fft_size:3 * fft_size] * freq_response11[:, 2]
            + signal_uplink[2, 2 * fft_size:3 * fft_size] *
            freq_response21[:, 2])

        # Test if expected signal and received signal are equal
        np.testing.assert_array_almost_equal(expected_received_signal_uplink,
                                             received_signal_uplink)

    def test_corrupt_data_in_freq_domain2(self):
        # This method tests corrupt_data_in_freq_domain, but now for a SIMO
        # system. The only difference is that the transmit signal can be a
        # 1D array (since we only have one transmit stream) or a 2D array
        # with only 1 row. Both cases should yield the same solution.
        num_rx_ant = 3
        num_tx_ant = 1

        fft_size = 16
        num_samples = 5 * fft_size
        signal1 = np.ones(num_samples)
        signal2 = signal1[np.newaxis, :]

        # For these particular indexes we will use half of the subcarriers
        subcarrier_indexes = np.r_[0:fft_size:2]

        jakes1 = fading_generators.JakesSampleGenerator(Fd=0.0,
                                                        Ts=self.Ts,
                                                        L=self.NRays,
                                                        shape=(num_rx_ant,
                                                               num_tx_ant))

        # Note that tdlmimochannel will modify the jakes1 object
        tdlmimochannel1 = fading.TdlMimoChannel(
            fading_generator=jakes1, channel_profile=fading.COST259_TUx)

        # xxxxxxxxxx Perform the actual transmission xxxxxxxxxxxxxxxxxxxxxx
        received_signal1 = tdlmimochannel1.corrupt_data_in_freq_domain(
            signal1, fft_size, subcarrier_indexes)

        received_signal2 = tdlmimochannel1.corrupt_data_in_freq_domain(
            signal2, fft_size, subcarrier_indexes)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Check that both received signals are equal (note that this is
        # only valid because the channel does not vary in time domain
        # (Doppler Frequency was set to zero.)
        np.testing.assert_array_almost_equal(received_signal1,
                                             received_signal2)

        # We don't need to test if received_signal1 is correct, since it
        # was already tested for the MIMO case.

    def test_corrupt_data_in_freq_domain3(self):
        # This method tests corrupt_data_in_freq_domain, but now specifying
        # the indexes of the used subcarriers
        num_rx_ant = 3
        num_tx_ant = 2

        fft_size = 16
        num_samples = 5 * fft_size
        signal = np.ones((2, num_samples))
        # For these particular indexes we will use half of the subcarriers
        subcarrier_indexes = np.r_[0:fft_size:2]

        jakes1 = fading_generators.JakesSampleGenerator(self.Fd,
                                                        self.Ts,
                                                        self.NRays,
                                                        shape=(num_rx_ant,
                                                               num_tx_ant))

        # Note that tdlmimochannel will modify the jakes1 object
        tdlmimochannel1 = fading.TdlMimoChannel(
            fading_generator=jakes1, channel_profile=fading.COST259_TUx)

        # we want tdlmimochannel2 to be a copy of tdlmimochannel1 and
        # generate the same samples
        tdlmimochannel2 = copy(tdlmimochannel1)
        # After the copy it will use the same fading_generator
        # object. Let's copy the fading_generator and replace the one in
        # tdlmimochannel2 with the copy
        jakes2 = copy(jakes1)
        tdlmimochannel2._fading_generator = jakes2

        # xxxxxxxxxx Perform the actual transmission xxxxxxxxxxxxxxxxxxxxxx
        received_signal = tdlmimochannel1.corrupt_data_in_freq_domain(
            signal, fft_size, subcarrier_indexes)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Compute frequency response for all samples xxxxxxxxxxx
        tdlmimochannel2.generate_impulse_response(2 * num_samples)
        impulse_response_all = tdlmimochannel2.get_last_impulse_response()
        # Note that here we have the frequency response for `num_samples`
        # samples. But the `corrupt_data_in_freq_domain` method only use
        # multiples of `fft_size` (0*fft_size, 1*fft_size, ...)
        freq_response_all = impulse_response_all.get_freq_response(fft_size)
        # Frequency response from each transmitter to each receiver
        freq_response00 = freq_response_all[:, 0, 0, :]
        freq_response01 = freq_response_all[:, 0, 1, :]
        freq_response10 = freq_response_all[:, 1, 0, :]
        freq_response11 = freq_response_all[:, 1, 1, :]
        freq_response20 = freq_response_all[:, 2, 0, :]
        freq_response21 = freq_response_all[:, 2, 1, :]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test if the received signal is correct xxxxxxxxxxxxxxx
        block_size = fft_size // 2
        # First OFDM symbol
        # Since we transmitted just 1's, then the received signal should be
        # equal to the frequency response af the start of the OFDM symbol

        # Receiver 0
        np.testing.assert_array_almost_equal(
            received_signal[0, 0:block_size],
            (freq_response00[subcarrier_indexes, 0] +
             freq_response01[subcarrier_indexes, 0]))
        # Receiver 1
        np.testing.assert_array_almost_equal(
            received_signal[1, 0:block_size],
            (freq_response10[subcarrier_indexes, 0] +
             freq_response11[subcarrier_indexes, 0]))
        # Receiver 2
        np.testing.assert_array_almost_equal(
            received_signal[2, 0:block_size],
            (freq_response20[subcarrier_indexes, 0] +
             freq_response21[subcarrier_indexes, 0]))

        # Second OFDM symbol
        np.testing.assert_array_almost_equal(
            received_signal[0, 1 * block_size:2 * block_size],
            (freq_response00[subcarrier_indexes, 1 * fft_size] +
             freq_response01[subcarrier_indexes, 1 * fft_size]))
        np.testing.assert_array_almost_equal(
            received_signal[1, 1 * block_size:2 * block_size],
            (freq_response10[subcarrier_indexes, 1 * fft_size] +
             freq_response11[subcarrier_indexes, 1 * fft_size]))
        np.testing.assert_array_almost_equal(
            received_signal[2, 1 * block_size:2 * block_size],
            (freq_response20[subcarrier_indexes, 1 * fft_size] +
             freq_response21[subcarrier_indexes, 1 * fft_size]))

        # Remaining OFDM symbols (from 2 to 10)
        for i in range(2, 10):
            np.testing.assert_array_almost_equal(
                received_signal[0, i * block_size:(i + 1) * block_size],
                (freq_response00[subcarrier_indexes, i * fft_size] +
                 freq_response01[subcarrier_indexes, i * fft_size]))
            np.testing.assert_array_almost_equal(
                received_signal[1, i * block_size:(i + 1) * block_size],
                (freq_response10[subcarrier_indexes, i * fft_size] +
                 freq_response11[subcarrier_indexes, i * fft_size]))
            np.testing.assert_array_almost_equal(
                received_signal[2, i * block_size:(i + 1) * block_size],
                (freq_response20[subcarrier_indexes, i * fft_size] +
                 freq_response21[subcarrier_indexes, i * fft_size]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test impulse response after transmission xxxxxxxxxxxxx
        # Since signal corresponds to 10 OFDM symbols (using only half of
        # the subcarriers in each OFDM symbol), then we should have 10
        # "samples" in the returned impulse response.
        impulse_response = tdlmimochannel1.get_last_impulse_response()
        self.assertEqual(impulse_response.num_samples,
                         num_samples // block_size)

        freq_response = impulse_response.get_freq_response(fft_size)
        for i in range(10):
            for tx_idx in range(num_tx_ant):
                for rx_idx in range(num_rx_ant):
                    np.testing.assert_array_almost_equal(
                        freq_response[:, rx_idx, tx_idx, i],
                        freq_response_all[:, rx_idx, tx_idx, i * fft_size])
                    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Single user Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SuSisoChannelTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        maxSystemBand = 40e6  # 40 MHz bandwidth
        # Number of subcarriers in this bandwidth
        max_num_of_subcarriers = math.floor(maxSystemBand / 15e3)
        # Find the maximum FFT size we can use which is below than or equal
        # to maxNumOfSubcarriersInt
        max_num_of_subcarriers = int(2**math.floor(
            math.log(max_num_of_subcarriers, 2)))
        # Calculate the actual bandwidth that we will use
        bandwidth = 15e3 * max_num_of_subcarriers

        self.Fd = 5  # Doppler frequency (in Hz)
        self.Ts = 1. / bandwidth  # Sampling interval (in seconds)
        self.NRays = 16  # Number of rays for the Jakes model

        # Create the jakes object that will be passed to TdlChannel
        self.jakes = fading_generators.JakesSampleGenerator(self.Fd,
                                                            self.Ts,
                                                            self.NRays,
                                                            shape=None)

        self.susisochannel = singleuser.SuChannel(
            self.jakes, channel_profile=fading.COST259_TUx)

    # noinspection PyTypeChecker
    def test_constructor(self):
        # xxxxxxxxxx IID Flat fading channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Create a SuChannel without specifying any parameter. In this
        # case a Rayleigh generator will be assumed and channel will be
        # also flat.
        flat_rayleight_suchannel = singleuser.SuChannel()
        self.assertEqual(flat_rayleight_suchannel.num_taps, 1)
        np.testing.assert_array_almost_equal(
            flat_rayleight_suchannel.channel_profile.tap_powers_linear, 1.0)
        np.testing.assert_array_almost_equal(
            flat_rayleight_suchannel.channel_profile.tap_delays, 0.0)
        self.assertAlmostEqual(flat_rayleight_suchannel.channel_profile.Ts, 1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Flat fading channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Create a SuChannel channel only passing the fading
        # generator. Note that in this case the channel will be flat,
        # containing only one tap with power 0dB and delay 0.
        jakes = fading_generators.JakesSampleGenerator(self.Fd,
                                                       self.Ts,
                                                       self.NRays,
                                                       shape=None)
        suchannel = singleuser.SuChannel(jakes)
        self.assertEqual(suchannel.num_taps, 1)
        np.testing.assert_array_almost_equal(
            suchannel.channel_profile.tap_powers_linear, 1.0)
        np.testing.assert_array_almost_equal(
            suchannel.channel_profile.tap_delays, 0.0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Frequency Selective channel xxxxxxxxxxxxxxxxxxxxxxxxxx
        jakes = fading_generators.JakesSampleGenerator(self.Fd,
                                                       self.Ts,
                                                       self.NRays,
                                                       shape=None)
        suchannel = singleuser.SuChannel(jakes,
                                         channel_profile=fading.COST259_TUx)
        self.assertEqual(suchannel.num_taps, 15)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_set_pathloss(self):
        # Test that default value of set_pathloss is None
        self.susisochannel.set_pathloss(0.01)
        self.susisochannel.set_pathloss()  # Use default value
        self.assertIsNone(self.susisochannel._pathloss_value)

        # Test if an exception is raised if we set a negative pathloss
        with self.assertRaises(ValueError):
            self.susisochannel.set_pathloss(-0.003)

        # Test if an exception is raised if we set a value greater than 1
        with self.assertRaises(ValueError):
            self.susisochannel.set_pathloss(1.03)

        # Now set to some positive value between 0 and 1
        self.susisochannel.set_pathloss(0.003)
        self.assertAlmostEqual(self.susisochannel._pathloss_value,
                               0.003,
                               delta=1e-20)

        self.susisochannel.set_pathloss(1e-12)
        self.assertAlmostEqual(self.susisochannel._pathloss_value,
                               1e-12,
                               delta=1e-20)

    def test_corrupt_data(self):
        # xxxxxxxxxx Test sending just a single impulse xxxxxxxxxxxxxxxxxxx
        signal = np.array([1.])

        received_signal = self.susisochannel.corrupt_data(signal)

        # Impulse response used to transmit the signal
        last_impulse_response = self.susisochannel.get_last_impulse_response()

        # Since only one sample was sent and it is equal to 1, then the
        # received signal will be equal to the full_fading_map
        np.testing.assert_almost_equal(
            last_impulse_response.tap_values.flatten(), received_signal)

        # xxxxxxxxxx Test with pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        pathloss = 0.0025
        self.susisochannel.set_pathloss(pathloss)

        received_signal = self.susisochannel.corrupt_data(signal)
        # get_last_impulse_response does not include pathloss effect
        last_impulse_response = self.susisochannel.get_last_impulse_response()
        np.testing.assert_almost_equal(
            last_impulse_response.tap_values.flatten(), received_signal)

        # Disable pathloss for the next tests
        self.susisochannel.set_pathloss(None)

        # xxxxxxxxxx Test sending a vector with 10 samples xxxxxxxxxxxxxxxx
        num_samples = 10
        signal = (np.random.standard_normal(num_samples) +
                  1j * np.random.standard_normal(num_samples))
        received_signal = self.susisochannel.corrupt_data(signal)
        last_impulse_response = self.susisochannel.get_last_impulse_response()

        # Compute the expected received signal
        # For this Ts we have 15 discretized taps. The indexes of the 15
        # taps are:
        # [ 0,  7, 16, 21, 27, 38, 40, 41, 47, 50, 56, 58, 60, 63, 66]
        np.testing.assert_array_equal(
            last_impulse_response.tap_indexes_sparse,
            np.array(
                [0, 7, 16, 21, 27, 38, 40, 41, 47, 50, 56, 58, 60, 63, 66]))

        # Including zero padding, the impulse response has 67 taps. That
        # means the channel memory is equal to 66
        channel_memory = 66
        expected_received_signal = np.zeros(channel_memory + num_samples,
                                            dtype=complex)

        # Let's compute the expected received signal
        taps_sparse = last_impulse_response.tap_values_sparse
        expected_received_signal[0:0 + num_samples] += signal * taps_sparse[0]
        expected_received_signal[7:7 + num_samples] += signal * taps_sparse[1]
        expected_received_signal[16:16 +
                                 num_samples] += signal * taps_sparse[2]
        expected_received_signal[21:21 +
                                 num_samples] += signal * taps_sparse[3]
        expected_received_signal[27:27 +
                                 num_samples] += signal * taps_sparse[4]
        expected_received_signal[38:38 +
                                 num_samples] += signal * taps_sparse[5]
        expected_received_signal[40:40 +
                                 num_samples] += signal * taps_sparse[6]
        expected_received_signal[41:41 +
                                 num_samples] += signal * taps_sparse[7]
        expected_received_signal[47:47 +
                                 num_samples] += signal * taps_sparse[8]
        expected_received_signal[50:50 +
                                 num_samples] += signal * taps_sparse[9]
        expected_received_signal[56:56 +
                                 num_samples] += signal * taps_sparse[10]
        expected_received_signal[58:58 +
                                 num_samples] += signal * taps_sparse[11]
        expected_received_signal[60:60 +
                                 num_samples] += signal * taps_sparse[12]
        expected_received_signal[63:63 +
                                 num_samples] += signal * taps_sparse[13]
        expected_received_signal[66:66 +
                                 num_samples] += signal * taps_sparse[14]

        # Check if the received signal is correct
        np.testing.assert_array_almost_equal(expected_received_signal,
                                             received_signal)

    def test_corrupt_data_in_freq_domain(self):
        fft_size = 16
        num_samples = 5 * fft_size
        signal = np.ones(num_samples)
        # num_full_blocks = num_samples // fft_size

        jakes1 = fading_generators.JakesSampleGenerator(self.Fd,
                                                        self.Ts,
                                                        self.NRays,
                                                        shape=None)

        # Note that tdlchannel will modify the jakes1 object
        susisochannel1 = singleuser.SuChannel(
            fading_generator=jakes1, channel_profile=fading.COST259_TUx)

        # Set the path loss. The received signal will be multiplied by
        # sqrt(pathloss)
        pathloss = 0.0025
        susisochannel1.set_pathloss(pathloss)

        # we want tdlchannel2 to be a copy of tdlchannel1 and generate the
        # same samples
        susisochannel2 = copy(susisochannel1)

        # After the copy it will use the same tdlchannel and
        # fading_generator objects. Let's copy them and replace the ones in
        # susisochannel2 with the copy
        tdlchannel2 = copy(susisochannel1._tdlchannel)
        jakes2 = copy(jakes1)
        tdlchannel2._fading_generator = jakes2
        susisochannel2._tdlchannel = tdlchannel2

        # xxxxxxxxxx Perform the actual transmission xxxxxxxxxxxxxxxxxxxxxx
        received_signal = susisochannel1.corrupt_data_in_freq_domain(
            signal, fft_size)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Compute frequency response for all samples xxxxxxxxxxx
        susisochannel2._tdlchannel.generate_impulse_response(num_samples)
        impulse_response_all = susisochannel2.get_last_impulse_response()
        # Note that here we have the frequency response for `num_samples`
        # samples. But the `corrupt_data_in_freq_domain` method only use
        # multiples of `fft_size` (0*fft_size, 1*fft_size, ...)
        freq_response_all = impulse_response_all.get_freq_response(fft_size)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test if the received signal is correct xxxxxxxxxxxxxxx
        # First OFDM symbol
        # Since we transmitted just 1's, then the received signal should be
        # equal to the frequency response af the start of the OFDM symbol
        np.testing.assert_array_almost_equal(received_signal[0:fft_size],
                                             freq_response_all[:, 0],
                                             decimal=7)
        # Second OFDM symbol
        np.testing.assert_array_almost_equal(received_signal[fft_size:2 *
                                                             fft_size],
                                             freq_response_all[:, fft_size],
                                             decimal=8)

        # Third OFDM symbol
        np.testing.assert_array_almost_equal(
            received_signal[2 * fft_size:3 * fft_size],
            freq_response_all[:, 2 * fft_size],
            decimal=8)

        # Fourth OFDM symbol
        np.testing.assert_array_almost_equal(
            received_signal[3 * fft_size:4 * fft_size],
            freq_response_all[:, 3 * fft_size],
            decimal=8)

        # Fifth OFDM symbol
        np.testing.assert_array_almost_equal(
            received_signal[4 * fft_size:5 * fft_size],
            freq_response_all[:, 4 * fft_size],
            decimal=8)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test impulse response after transmission xxxxxxxxxxxxx
        # Since signal corresponds to 5 OFDM symbols, then we should have 5
        # "samples" in the returned impulse response.
        impulse_response = susisochannel1.get_last_impulse_response()
        self.assertEqual(impulse_response.num_samples, num_samples // fft_size)

        freq_response = impulse_response.get_freq_response(fft_size)
        np.testing.assert_array_almost_equal(
            freq_response[:, 0], freq_response_all[:, 0 * fft_size])
        np.testing.assert_array_almost_equal(
            freq_response[:, 1], freq_response_all[:, 1 * fft_size])
        np.testing.assert_array_almost_equal(
            freq_response[:, 2], freq_response_all[:, 2 * fft_size])
        np.testing.assert_array_almost_equal(
            freq_response[:, 3], freq_response_all[:, 3 * fft_size])
        np.testing.assert_array_almost_equal(
            freq_response[:, 4], freq_response_all[:, 4 * fft_size])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class SuMimoChannelTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        maxSystemBand = 40e6  # 40 MHz bandwidth
        # Number of subcarriers in this bandwidth
        max_num_of_subcarriers = math.floor(maxSystemBand / 15e3)
        # Find the maximum FFT size we can use which is below than or equal
        # to maxNumOfSubcarriersInt
        max_num_of_subcarriers = int(2**math.floor(
            math.log(max_num_of_subcarriers, 2)))
        # Calculate the actual bandwidth that we will use
        bandwidth = 15e3 * max_num_of_subcarriers

        self.Fd = 5  # Doppler frequency (in Hz)
        self.Ts = 1. / bandwidth  # Sampling interval (in seconds)
        self.NRays = 16  # Number of rays for the Jakes model

        # Create the jakes object that will be passed to TdlChannel
        self.jakes = fading_generators.JakesSampleGenerator(self.Fd,
                                                            self.Ts,
                                                            self.NRays,
                                                            shape=None)

        self.sumimochannel = singleuser.SuMimoChannel(
            num_antennas=3,
            fading_generator=self.jakes,
            channel_profile=fading.COST259_TUx)

    def test_constructor(self):
        # xxxxxxxxxx IID Flat fading channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Create a SuMimoChannel without specifying any parameter. In this
        # case a Rayleigh generator will be assumed and channel will be
        # also flat.
        flat_rayleight_suchannel = singleuser.SuMimoChannel(num_antennas=3)
        self.assertEqual(flat_rayleight_suchannel.num_taps, 1)
        np.testing.assert_array_almost_equal(
            flat_rayleight_suchannel.channel_profile.tap_powers_linear,
            np.array([1.0]))
        np.testing.assert_array_almost_equal(
            flat_rayleight_suchannel.channel_profile.tap_delays,
            np.array([0.0]))
        self.assertAlmostEqual(flat_rayleight_suchannel.channel_profile.Ts, 1)

        self.assertEqual(flat_rayleight_suchannel.num_tx_antennas, 3)
        self.assertEqual(flat_rayleight_suchannel.num_rx_antennas, 3)
        self.assertEqual(flat_rayleight_suchannel.num_taps, 1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Flat fading channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Create a SuMimoChannel channel only passing the fading
        # generator. Note that in this case the channel will be flat,
        # containing only one tap with power 0dB and delay 0.
        jakes = fading_generators.JakesSampleGenerator(self.Fd,
                                                       self.Ts,
                                                       self.NRays,
                                                       shape=None)
        suchannel = singleuser.SuMimoChannel(num_antennas=3,
                                             fading_generator=jakes)
        self.assertEqual(suchannel.num_taps, 1)
        np.testing.assert_array_almost_equal(
            suchannel.channel_profile.tap_powers_linear, np.array([1.0]))
        np.testing.assert_array_almost_equal(
            suchannel.channel_profile.tap_delays, np.array([0.0]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Frequency Selective channel xxxxxxxxxxxxxxxxxxxxxxxxxx
        jakes = fading_generators.JakesSampleGenerator(self.Fd,
                                                       self.Ts,
                                                       self.NRays,
                                                       shape=None)
        suchannel = singleuser.SuMimoChannel(
            num_antennas=3,
            fading_generator=jakes,
            channel_profile=fading.COST259_TUx)
        self.assertEqual(suchannel.num_taps, 15)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # TODO: implement-me
    def test_corrupt_data(self):
        pass

    # TODO: implement-me
    def test_corrupt_data_in_freq_domain(self):
        pass


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Multiuser Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MuSisoChannelTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.musisochannel = multiuser.MuChannel(N=2)
        self.musisochannel2 = multiuser.MuChannel(N=3)

        # self.musisochannel = multiuser.MuSisoFlatFadingChannel(N=2)
        # self.musisochannel2 = multiuser.MuSisoFlatFadingChannel(N=3)

    def test_repr(self):
        self.assertEqual(repr(self.musisochannel),
                         "MuChannel(shape=2x2, switched=False)")
        self.musisochannel.switched_direction = True
        self.assertEqual(repr(self.musisochannel),
                         "MuChannel(shape=2x2, switched=True)")

        self.assertEqual(repr(self.musisochannel2),
                         "MuChannel(shape=3x3, switched=False)")
        self.musisochannel2.switched_direction = True
        self.assertEqual(repr(self.musisochannel2),
                         "MuChannel(shape=3x3, switched=True)")

    def test_constructor(self):
        N = 4
        # We are only providing the N parameters. That means each link will
        # take only one path with power 0dB and delay 0. Ts will be 1 for
        # the RayleighSampleGenerator. For the JakesSampleGenerator Ts will
        # assume the value from the JakesSampleGenerator.
        musisochannel = multiuser.MuChannel(N=N)
        musisochannel2 = multiuser.MuChannel(
            N=N, fading_generator=fading_generators.JakesSampleGenerator())

        self.assertEqual(musisochannel._su_siso_channels.shape, (N, N))
        self.assertEqual(musisochannel2._su_siso_channels.shape, (N, N))

        for tx in range(N):
            for rx in range(N):
                suchannel = musisochannel._su_siso_channels[rx, tx]
                suchannel2 = musisochannel2._su_siso_channels[rx, tx]
                np.testing.assert_array_almost_equal(
                    suchannel.channel_profile.tap_delays, np.array([0]))
                np.testing.assert_array_almost_equal(
                    suchannel.channel_profile.tap_powers_dB, np.array([0]))
                self.assertEqual(suchannel.channel_profile.Ts, 1)

                np.testing.assert_array_almost_equal(
                    suchannel2.channel_profile.tap_delays, np.array([0]))
                np.testing.assert_array_almost_equal(
                    suchannel2.channel_profile.tap_powers_dB, np.array([0]))
                self.assertEqual(suchannel2.channel_profile.Ts, 0.001)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now lets test specifying more parameters
        N = 2
        Ts = 3.25e-8
        channel_profile = fading.COST259_TUx
        jakes = fading_generators.JakesSampleGenerator(Ts=Ts)

        musisochannel = multiuser.MuChannel(N=N,
                                            channel_profile=channel_profile,
                                            Ts=Ts)
        musisochannel2 = multiuser.MuChannel(N=N,
                                             channel_profile=channel_profile,
                                             fading_generator=jakes)

        self.assertEqual(musisochannel._su_siso_channels.shape, (N, N))
        self.assertEqual(musisochannel2._su_siso_channels.shape, (N, N))

        for tx in range(N):
            for rx in range(N):
                suchannel = musisochannel._su_siso_channels[rx, tx]
                suchannel2 = musisochannel2._su_siso_channels[rx, tx]
                np.testing.assert_array_almost_equal(
                    suchannel.channel_profile.tap_delays,
                    np.array([
                        0, 7, 16, 21, 27, 38, 40, 42, 47, 50, 56, 58, 60, 63,
                        66
                    ]))
                np.testing.assert_array_almost_equal(
                    suchannel.channel_profile.tap_powers_dB,
                    np.array([
                        -05.696548, -07.596548, -05.391745, -11.496548,
                        -13.396548, -16.296548, -13.985097, -17.396548,
                        -15.986248, -19.796548, -18.535960, -22.096548,
                        -22.596548, -23.496548, -24.296548
                    ]))
                self.assertEqual(suchannel.channel_profile.Ts, Ts)

                np.testing.assert_array_almost_equal(
                    suchannel2.channel_profile.tap_delays,
                    np.array([
                        0, 7, 16, 21, 27, 38, 40, 42, 47, 50, 56, 58, 60, 63,
                        66
                    ]))
                np.testing.assert_array_almost_equal(
                    suchannel2.channel_profile.tap_powers_dB,
                    np.array([
                        -05.696548, -07.596548, -05.391745, -11.496548,
                        -13.396548, -16.296548, -13.985097, -17.396548,
                        -15.986248, -19.796548, -18.535960, -22.096548,
                        -22.596548, -23.496548, -24.296548
                    ]))
                self.assertEqual(suchannel2.channel_profile.Ts, Ts)

        # Test unequal number of transmitters and receivers
        musisochannel = multiuser.MuChannel(N=(1, 3))
        self.assertEqual(musisochannel._su_siso_channels.shape, (1, 3))

    def test_corrupt_data(self):
        num_samples = 5
        # xxxxxxxxxx Test without pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Generate data for 2 transmitters
        data1 = np.random.randint(0, 10, (2, num_samples))

        # Pass data through he channel
        output1 = self.musisochannel.corrupt_data(data1)

        # Get the impulse response from two transmitters to the first
        # receiver. The channel is flat, thus we only have one tap.
        impulse_response00 = self.musisochannel.get_last_impulse_response(0, 0)
        impulse_response01 = self.musisochannel.get_last_impulse_response(0, 1)
        h00 = impulse_response00.tap_values[0]  # We only have the first tap
        h01 = impulse_response01.tap_values[0]

        expected_received_data0 = data1[0] * h00 + data1[1] * h01
        # Test if data received at the first receiver is correct
        np.testing.assert_array_almost_equal(expected_received_data0,
                                             output1[0])

        # Get the impulse response from two transmitters to the second
        # receiver. The channel is flat, thus we only have one tap.
        impulse_response10 = self.musisochannel.get_last_impulse_response(1, 0)
        impulse_response11 = self.musisochannel.get_last_impulse_response(1, 1)
        h10 = impulse_response10.tap_values[0]  # We only have the first tap
        h11 = impulse_response11.tap_values[0]

        expected_received_data1 = data1[0] * h10 + data1[1] * h11
        # Test if data received at the first receiver is correct
        np.testing.assert_array_almost_equal(expected_received_data1,
                                             output1[1])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test with pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # First we set the pathloss. We will use the same channel object as
        # before
        pathloss_matrix = np.array([[0.8, 0.4], [0.3, 0.95]])
        self.musisochannel.set_pathloss(pathloss_matrix)

        # Pass data through the channel
        output1 = self.musisochannel.corrupt_data(data1)

        # Get the impulse response from two transmitters to the first
        # receiver. The channel is flat, thus we only have one tap.
        impulse_response00 = self.musisochannel.get_last_impulse_response(0, 0)
        impulse_response01 = self.musisochannel.get_last_impulse_response(0, 1)
        # Note that path loss effect is already included in impulse_response
        h00 = impulse_response00.tap_values[0]  # We only have the first tap
        h01 = impulse_response01.tap_values[0]

        expected_received_data0 = (data1[0] * h00 + data1[1] * h01)
        # Test if data received at the first receiver is correct
        np.testing.assert_array_almost_equal(expected_received_data0,
                                             output1[0])

        # Get the impulse response from two transmitters to the first
        # receiver. The channel is flat, thus we only have one tap.
        impulse_response10 = self.musisochannel.get_last_impulse_response(1, 0)
        impulse_response11 = self.musisochannel.get_last_impulse_response(1, 1)
        h10 = impulse_response10.tap_values[0]  # We only have the first tap
        h11 = impulse_response11.tap_values[0]

        expected_received_data1 = (data1[0] * h10 + data1[1] * h11)
        # Test if data received at the first receiver is correct
        np.testing.assert_array_almost_equal(expected_received_data1,
                                             output1[1])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test with a channel with memory xxxxxxxxxxxxxxxxxxxxxx
        Ts = 3.25e-8
        # Create a jakes fading generator
        jakes = fading_generators.JakesSampleGenerator(Fd=30, Ts=Ts, L=16)
        # Create a channel profile with 2 (sparse) taps. Including zero
        # padding, this channel as 4 taps with the non zeros taps at delays 0
        # and 3.
        channel_profile = fading.TdlChannelProfile(
            tap_powers_dB=np.array([0, -5]), tap_delays=np.array([0, 3 * Ts]))
        musisochannel = multiuser.MuChannel(N=2,
                                            fading_generator=jakes,
                                            channel_profile=channel_profile)
        self.assertEqual(musisochannel.num_taps, 2)
        self.assertEqual(musisochannel.num_taps_with_padding, 4)
        channel_memory = musisochannel.num_taps_with_padding - 1

        # Pass data through he channel
        output1 = musisochannel.corrupt_data(data1)

        # Due to channel memory the received signal has more samples
        self.assertEqual(output1[0].shape, num_samples + channel_memory)
        self.assertEqual(output1[1].shape, num_samples + channel_memory)

        # Get the impulse response from two transmitters to the first
        # receiver.
        impulse_response00 = musisochannel.get_last_impulse_response(0, 0)
        impulse_response01 = musisochannel.get_last_impulse_response(0, 1)
        h00 = impulse_response00.tap_values  # Dim: `num_taps x num samples`
        h01 = impulse_response01.tap_values  # Dim: `num_taps x num samples`

        # Get the impulse response from two transmitters to the second
        # receiver.
        impulse_response10 = musisochannel.get_last_impulse_response(1, 0)
        impulse_response11 = musisochannel.get_last_impulse_response(1, 1)
        h10 = impulse_response10.tap_values  # Dim: `num_taps x num samples`
        h11 = impulse_response11.tap_values  # Dim: `num_taps x num samples`

        expected_received_data0 = np.zeros(num_samples + channel_memory,
                                           dtype=complex)
        expected_received_data1 = np.zeros(num_samples + channel_memory,
                                           dtype=complex)
        for i in range(musisochannel.num_taps_with_padding):
            expected_received_data0[i:i + num_samples] += (data1[0] * h00[i] +
                                                           data1[1] * h01[i])
            expected_received_data1[i:i + num_samples] += (data1[0] * h10[i] +
                                                           data1[1] * h11[i])

        np.testing.assert_array_almost_equal(expected_received_data0,
                                             output1[0])
        np.testing.assert_array_almost_equal(expected_received_data1,
                                             output1[1])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Now test with path loss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        pathloss_matrix = np.array([[0.8, 0.4], [0.3, 0.95]])
        musisochannel.set_pathloss(pathloss_matrix)

        # Pass data through he channel
        output1 = musisochannel.corrupt_data(data1)

        # Due to channel memory the received signal has more samples
        self.assertEqual(output1[0].shape, num_samples + channel_memory)
        self.assertEqual(output1[1].shape, num_samples + channel_memory)

        # Get the impulse response from two transmitters to the first
        # receiver.
        impulse_response00 = musisochannel.get_last_impulse_response(0, 0)
        impulse_response01 = musisochannel.get_last_impulse_response(0, 1)
        h00 = impulse_response00.tap_values  # Dim: `num_taps x num samples`
        h01 = impulse_response01.tap_values  # Dim: `num_taps x num samples`

        # Get the impulse response from two transmitters to the second
        # receiver.
        impulse_response10 = musisochannel.get_last_impulse_response(1, 0)
        impulse_response11 = musisochannel.get_last_impulse_response(1, 1)
        h10 = impulse_response10.tap_values  # Dim: `num_taps x num samples`
        h11 = impulse_response11.tap_values  # Dim: `num_taps x num samples`

        expected_received_data0 = np.zeros(num_samples + channel_memory,
                                           dtype=complex)
        expected_received_data1 = np.zeros(num_samples + channel_memory,
                                           dtype=complex)
        for i in range(musisochannel.num_taps_with_padding):
            expected_received_data0[i:i + num_samples] += (data1[0] * h00[i] +
                                                           data1[1] * h01[i])
            expected_received_data1[i:i + num_samples] += (data1[0] * h10[i] +
                                                           data1[1] * h11[i])

        np.testing.assert_array_almost_equal(expected_received_data0,
                                             output1[0])
        np.testing.assert_array_almost_equal(expected_received_data1,
                                             output1[1])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Now test with switched directions xxxxxxxxxxxxxxxxxxxx
        musisochannel.switched_direction = True
        output_swiched = musisochannel.corrupt_data(data1)

        # Get the impulse response from two transmitters to the first
        # receiver.
        impulse_response00 = musisochannel.get_last_impulse_response(0, 0)
        impulse_response01 = musisochannel.get_last_impulse_response(0, 1)
        h00 = impulse_response00.tap_values  # Dim: `num_taps x num samples`
        h01 = impulse_response01.tap_values  # Dim: `num_taps x num samples`

        # Get the impulse response from two transmitters to the second
        # receiver.
        impulse_response10 = musisochannel.get_last_impulse_response(1, 0)
        impulse_response11 = musisochannel.get_last_impulse_response(1, 1)
        h10 = impulse_response10.tap_values  # Dim: `num_taps x num samples`
        h11 = impulse_response11.tap_values  # Dim: `num_taps x num samples`

        expected_received_data0 = np.zeros(num_samples + channel_memory,
                                           dtype=complex)
        expected_received_data1 = np.zeros(num_samples + channel_memory,
                                           dtype=complex)
        for i in range(musisochannel.num_taps_with_padding):
            expected_received_data0[i:i + num_samples] += (data1[0] * h00[i] +
                                                           data1[1] * h10[i])
            expected_received_data1[i:i + num_samples] += (data1[0] * h01[i] +
                                                           data1[1] * h11[i])

        np.testing.assert_array_almost_equal(expected_received_data0,
                                             output_swiched[0])
        np.testing.assert_array_almost_equal(expected_received_data1,
                                             output_swiched[1])

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_corrupt_data2(self):
        # Test for an unequal number of transmitters and receivers

        # xxxxxxxxxx Test for 1 receiver and 2 transmitters xxxxxxxxxxxxxxx
        musisochannel = multiuser.MuChannel(N=(1, 2))

        num_samples = 5

        # Generate data for 2 transmitters
        data1 = np.random.randint(0, 10, (2, num_samples))

        # Pass data through he channel
        output1 = musisochannel.corrupt_data(data1)

        # Get the impulse response from two transmitters to the first
        # receiver. The channel is flat, thus we only have one tap.
        impulse_response0 = musisochannel.get_last_impulse_response(0, 0)
        impulse_response1 = musisochannel.get_last_impulse_response(0, 1)
        h0 = impulse_response0.tap_values[0]  # We only have the first tap
        h1 = impulse_response1.tap_values[0]

        # For a flat channel the number of elements in the output should be
        # equal to the number of elements in the input.
        self.assertEqual(output1[0].size, num_samples)
        expected_received_data0 = data1[0] * h0 + data1[1] * h1
        # Test if data received at the first receiver is correct
        np.testing.assert_array_almost_equal(expected_received_data0,
                                             output1[0])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test for 1 transmitter and 2 receivers xxxxxxxxxxxxxxx
        musisochannel = multiuser.MuChannel(N=(2, 1))

        num_samples = 5

        # Generate data for 1 transmitter
        data1 = np.random.randint(0, 10, (1, num_samples))

        # Pass data through he channel
        output1 = musisochannel.corrupt_data(data1)

        # Get the impulse response from the single transmitter to each
        # receiver. The channel is flat, thus we only have one tap.
        impulse_response0 = musisochannel.get_last_impulse_response(0, 0)
        impulse_response1 = musisochannel.get_last_impulse_response(1, 0)
        h0 = impulse_response0.tap_values[0]  # We only have the first tap
        h1 = impulse_response1.tap_values[0]

        expected_received_data0 = data1[0] * h0
        expected_received_data1 = data1[0] * h1

        # For a flat channel the number of elements in the output should be
        # equal to the number of elements in the input.
        self.assertEqual(output1[0].size, num_samples)
        self.assertEqual(output1[1].size, num_samples)

        # Test if data received at the first receiver is correct
        np.testing.assert_array_almost_equal(expected_received_data0,
                                             output1[0])
        # Test if data received at the second receiver is correct
        np.testing.assert_array_almost_equal(expected_received_data1,
                                             output1[1])

        # Repeat the test, but now data has a single dimension (since we
        # only have one transmitter)
        # Generate data for 1 transmitter
        data2 = np.random.randint(0, 10, num_samples)

        # Pass data through he channel
        output2 = musisochannel.corrupt_data(data2)

        # Get the impulse response from the single transmitter to each
        # receiver. The channel is flat, thus we only have one tap.
        impulse_response0 = musisochannel.get_last_impulse_response(0, 0)
        impulse_response1 = musisochannel.get_last_impulse_response(1, 0)
        h0 = impulse_response0.tap_values[0]  # We only have the first tap
        h1 = impulse_response1.tap_values[0]

        expected_received_data0 = data2 * h0
        expected_received_data1 = data2 * h1

        # For a flat channel the number of elements in the output should be
        # equal to the number of elements in the input.
        self.assertEqual(output2[0].size, num_samples)
        self.assertEqual(output2[1].size, num_samples)

        # Test if data received at the first receiver is correct
        np.testing.assert_array_almost_equal(expected_received_data0,
                                             output2[0])
        # Test if data received at the second receiver is correct
        np.testing.assert_array_almost_equal(expected_received_data1,
                                             output2[1])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Now test with switched directions xxxxxxxxxxxxxxxxxxxx
        # We have 1 transmitter and 2 receivers in the normal direction
        musisochannel = multiuser.MuChannel(N=(2, 1))
        # But we will switch directions and thus now we have 2 transmitters
        # and 1 receiver
        musisochannel.switched_direction = True

        num_samples = 5

        # Generate data for 1 transmitter
        data = np.random.randint(0, 10, (2, num_samples))

        # Pass data through he channel
        output = musisochannel.corrupt_data(data)

        # Get the impulse response from the single transmitter to each
        # receiver. The channel is flat, thus we only have one tap.
        impulse_response0 = musisochannel.get_last_impulse_response(0, 0)
        impulse_response1 = musisochannel.get_last_impulse_response(1, 0)
        h0 = impulse_response0.tap_values[0]  # We only have the first tap
        h1 = impulse_response1.tap_values[0]

        expected_received_data = data[0] * h0 + data[1] * h1

        # For a flat channel the number of elements in the output should be
        # equal to the number of elements in the input.
        self.assertEqual(output[0].size, num_samples)

        # Test if received data is correct
        np.testing.assert_array_almost_equal(expected_received_data, output[0])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_corrupt_data_in_freq_domain(self):
        Ts = 3.25e-8
        # Create a jakes fading generator
        jakes = fading_generators.JakesSampleGenerator(Fd=30, Ts=Ts, L=16)
        # Create a channel profile with 2 (sparse) taps. Including zero
        # padding, this channel as 4 taps with the non zeros taps at delays 0
        # and 3.
        channel_profile = fading.COST259_TUx
        musisochannel = multiuser.MuChannel(N=2,
                                            fading_generator=jakes,
                                            channel_profile=channel_profile)

        fft_size = 64
        num_samples = 4 * fft_size
        # xxxxxxxxxx Test without pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Generate data for 2 transmitters
        # data = np.ones((2, num_samples))
        data = np.random.randint(0, 10, (2, num_samples))

        # Pass data through he channel
        output = musisochannel.corrupt_data_in_freq_domain(data, fft_size)

        self.assertEqual(data[0].shape, output[0].shape)
        self.assertEqual(data[1].shape, output[1].shape)

        impulse_response00 = musisochannel.get_last_impulse_response(0, 0)
        impulse_response01 = musisochannel.get_last_impulse_response(0, 1)

        freq_response00 = impulse_response00.get_freq_response(fft_size)
        freq_response01 = impulse_response01.get_freq_response(fft_size)

        # xxxxxxxxxx Expected received signal at first receiver xxxxxxxxxxx
        expected_ofdm_symb1 = (data[0, 0:fft_size] * freq_response00[:, 0] +
                               data[1, 0:fft_size] * freq_response01[:, 0])
        expected_ofdm_symb2 = (
            data[0, fft_size:2 * fft_size] * freq_response00[:, 1] +
            data[1, fft_size:2 * fft_size] * freq_response01[:, 1])
        expected_ofdm_symb3 = (
            data[0, 2 * fft_size:3 * fft_size] * freq_response00[:, 2] +
            data[1, 2 * fft_size:3 * fft_size] * freq_response01[:, 2])
        expected_ofdm_symb4 = (
            data[0, 3 * fft_size:4 * fft_size] * freq_response00[:, 3] +
            data[1, 3 * fft_size:4 * fft_size] * freq_response01[:, 3])

        np.testing.assert_array_almost_equal(
            output[0],
            np.hstack([
                expected_ofdm_symb1, expected_ofdm_symb2, expected_ofdm_symb3,
                expected_ofdm_symb4
            ]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Now test with switched directions xxxxxxxxxxxxxxxxxxxx
        # Switch directions
        musisochannel.switched_direction = True

        # Transmit in the other direction
        output = musisochannel.corrupt_data_in_freq_domain(data, fft_size)

        self.assertEqual(data[0].shape, output[0].shape)
        self.assertEqual(data[1].shape, output[1].shape)

        impulse_response00 = musisochannel.get_last_impulse_response(0, 0)
        impulse_response10 = musisochannel.get_last_impulse_response(1, 0)

        freq_response00 = impulse_response00.get_freq_response(fft_size)
        freq_response10 = impulse_response10.get_freq_response(fft_size)

        # xxxxxxxxxx Expected received signal at first receiver xxxxxxxxxxx

        expected_ofdm_symb1 = (data[0, 0:fft_size] * freq_response00[:, 0] +
                               data[1, 0:fft_size] * freq_response10[:, 0])
        expected_ofdm_symb2 = (
            data[0, fft_size:2 * fft_size] * freq_response00[:, 1] +
            data[1, fft_size:2 * fft_size] * freq_response10[:, 1])
        expected_ofdm_symb3 = (
            data[0, 2 * fft_size:3 * fft_size] * freq_response00[:, 2] +
            data[1, 2 * fft_size:3 * fft_size] * freq_response10[:, 2])
        expected_ofdm_symb4 = (
            data[0, 3 * fft_size:4 * fft_size] * freq_response00[:, 3] +
            data[1, 3 * fft_size:4 * fft_size] * freq_response10[:, 3])

        np.testing.assert_array_almost_equal(
            output[0],
            np.hstack([
                expected_ofdm_symb1, expected_ofdm_symb2, expected_ofdm_symb3,
                expected_ofdm_symb4
            ]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_corrupt_data_in_freq_domain2(self):
        # This method tests corrupt_data_in_freq_domain, but now specifying
        # the indexes of the used subcarriers

        Ts = 3.25e-8
        # Create a jakes fading generator
        jakes = fading_generators.JakesSampleGenerator(Fd=30, Ts=Ts, L=16)
        # Create a channel profile with 2 (sparse) taps. Including zero
        # padding, this channel as 4 taps with the non zeros taps at delays 0
        # and 3.
        channel_profile = fading.COST259_TUx
        musisochannel = multiuser.MuChannel(N=2,
                                            fading_generator=jakes,
                                            channel_profile=channel_profile)

        fft_size = 64
        # For these particular indexes we will use half of the subcarriers
        subcarrier_indexes = np.r_[0:fft_size:2]
        block_size = len(subcarrier_indexes)

        num_samples = 4 * block_size

        # xxxxxxxxxx Test without pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Generate data for 2 transmitters
        # data = np.ones((2, num_samples))
        data = np.random.randint(0, 10, (2, num_samples))

        # Pass data through he channel
        output = musisochannel.corrupt_data_in_freq_domain(
            data, fft_size, subcarrier_indexes)

        self.assertEqual(data[0].shape, output[0].shape)
        self.assertEqual(data[1].shape, output[1].shape)

        impulse_response00 = musisochannel.get_last_impulse_response(0, 0)
        impulse_response01 = musisochannel.get_last_impulse_response(0, 1)
        freq_response00 = impulse_response00.get_freq_response(fft_size)
        freq_response01 = impulse_response01.get_freq_response(fft_size)

        impulse_response10 = musisochannel.get_last_impulse_response(1, 0)
        impulse_response11 = musisochannel.get_last_impulse_response(1, 1)
        freq_response10 = impulse_response10.get_freq_response(fft_size)
        freq_response11 = impulse_response11.get_freq_response(fft_size)

        # xxxxxxxxxx Expected received signal at first receiver xxxxxxxxxxx
        expected_ofdm_symb1 = (
            data[0, 0:block_size] * freq_response00[subcarrier_indexes, 0] +
            data[1, 0:block_size] * freq_response01[subcarrier_indexes, 0])
        expected_ofdm_symb2 = (data[0, block_size:2 * block_size] *
                               freq_response00[subcarrier_indexes, 1] +
                               data[1, block_size:2 * block_size] *
                               freq_response01[subcarrier_indexes, 1])
        expected_ofdm_symb3 = (data[0, 2 * block_size:3 * block_size] *
                               freq_response00[subcarrier_indexes, 2] +
                               data[1, 2 * block_size:3 * block_size] *
                               freq_response01[subcarrier_indexes, 2])
        expected_ofdm_symb4 = (data[0, 3 * block_size:4 * block_size] *
                               freq_response00[subcarrier_indexes, 3] +
                               data[1, 3 * block_size:4 * block_size] *
                               freq_response01[subcarrier_indexes, 3])

        np.testing.assert_array_almost_equal(
            output[0],
            np.hstack([
                expected_ofdm_symb1, expected_ofdm_symb2, expected_ofdm_symb3,
                expected_ofdm_symb4
            ]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Expected received signal at second receiver xxxxxxxxxx
        expected_ofdm_symb1 = (
            data[0, 0:block_size] * freq_response10[subcarrier_indexes, 0] +
            data[1, 0:block_size] * freq_response11[subcarrier_indexes, 0])
        expected_ofdm_symb2 = (data[0, block_size:2 * block_size] *
                               freq_response10[subcarrier_indexes, 1] +
                               data[1, block_size:2 * block_size] *
                               freq_response11[subcarrier_indexes, 1])
        expected_ofdm_symb3 = (data[0, 2 * block_size:3 * block_size] *
                               freq_response10[subcarrier_indexes, 2] +
                               data[1, 2 * block_size:3 * block_size] *
                               freq_response11[subcarrier_indexes, 2])
        expected_ofdm_symb4 = (data[0, 3 * block_size:4 * block_size] *
                               freq_response10[subcarrier_indexes, 3] +
                               data[1, 3 * block_size:4 * block_size] *
                               freq_response11[subcarrier_indexes, 3])

        np.testing.assert_array_almost_equal(
            output[1],
            np.hstack([
                expected_ofdm_symb1, expected_ofdm_symb2, expected_ofdm_symb3,
                expected_ofdm_symb4
            ]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Now test with switched directions xxxxxxxxxxxxxxxxxxxx
        # Switch directions
        musisochannel.switched_direction = True

        # Pass data through he channel
        output = musisochannel.corrupt_data_in_freq_domain(
            data, fft_size, subcarrier_indexes)

        self.assertEqual(data[0].shape, output[0].shape)
        self.assertEqual(data[1].shape, output[1].shape)

        impulse_response00 = musisochannel.get_last_impulse_response(0, 0)
        impulse_response10 = musisochannel.get_last_impulse_response(1, 0)
        freq_response00 = impulse_response00.get_freq_response(fft_size)
        freq_response10 = impulse_response10.get_freq_response(fft_size)

        impulse_response01 = musisochannel.get_last_impulse_response(0, 1)
        impulse_response11 = musisochannel.get_last_impulse_response(1, 1)
        freq_response01 = impulse_response01.get_freq_response(fft_size)
        freq_response11 = impulse_response11.get_freq_response(fft_size)

        # xxxxxxxxxx Expected received signal at first receiver xxxxxxxxxxx
        expected_ofdm_symb1 = (
            data[0, 0:block_size] * freq_response00[subcarrier_indexes, 0] +
            data[1, 0:block_size] * freq_response10[subcarrier_indexes, 0])
        expected_ofdm_symb2 = (data[0, block_size:2 * block_size] *
                               freq_response00[subcarrier_indexes, 1] +
                               data[1, block_size:2 * block_size] *
                               freq_response10[subcarrier_indexes, 1])
        expected_ofdm_symb3 = (data[0, 2 * block_size:3 * block_size] *
                               freq_response00[subcarrier_indexes, 2] +
                               data[1, 2 * block_size:3 * block_size] *
                               freq_response10[subcarrier_indexes, 2])
        expected_ofdm_symb4 = (data[0, 3 * block_size:4 * block_size] *
                               freq_response00[subcarrier_indexes, 3] +
                               data[1, 3 * block_size:4 * block_size] *
                               freq_response10[subcarrier_indexes, 3])

        np.testing.assert_array_almost_equal(
            output[0],
            np.hstack([
                expected_ofdm_symb1, expected_ofdm_symb2, expected_ofdm_symb3,
                expected_ofdm_symb4
            ]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Expected received signal at second receiver xxxxxxxxxx
        expected_ofdm_symb1 = (
            data[0, 0:block_size] * freq_response01[subcarrier_indexes, 0] +
            data[1, 0:block_size] * freq_response11[subcarrier_indexes, 0])
        expected_ofdm_symb2 = (data[0, block_size:2 * block_size] *
                               freq_response01[subcarrier_indexes, 1] +
                               data[1, block_size:2 * block_size] *
                               freq_response11[subcarrier_indexes, 1])
        expected_ofdm_symb3 = (data[0, 2 * block_size:3 * block_size] *
                               freq_response01[subcarrier_indexes, 2] +
                               data[1, 2 * block_size:3 * block_size] *
                               freq_response11[subcarrier_indexes, 2])
        expected_ofdm_symb4 = (data[0, 3 * block_size:4 * block_size] *
                               freq_response01[subcarrier_indexes, 3] +
                               data[1, 3 * block_size:4 * block_size] *
                               freq_response11[subcarrier_indexes, 3])

        np.testing.assert_array_almost_equal(
            output[1],
            np.hstack([
                expected_ofdm_symb1, expected_ofdm_symb2, expected_ofdm_symb3,
                expected_ofdm_symb4
            ]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_corrupt_data_in_freq_domain3(self):
        # Test for an unequal number of transmitters and receivers
        Ts = 3.25e-8
        num_tx = 1
        num_rx = 2
        # Create a jakes fading generator
        jakes = fading_generators.JakesSampleGenerator(Fd=30, Ts=Ts, L=16)
        # Create a channel profile with 2 (sparse) taps. Including zero
        # padding, this channel as 4 taps with the non zeros taps at delays 0
        # and 3.
        channel_profile = fading.COST259_TUx
        musisochannel = multiuser.MuChannel(N=(num_rx, num_tx),
                                            fading_generator=jakes,
                                            channel_profile=channel_profile)

        fft_size = 64
        num_samples = 4 * fft_size
        # xxxxxxxxxx Test without pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Generate data for 2 transmitters
        # data = np.ones((2, num_samples))
        data = np.random.randint(0, 10, (1, num_samples))

        # Pass data through he channel
        output = musisochannel.corrupt_data_in_freq_domain(data, fft_size)

        self.assertEqual(data[0].shape, output[0].shape)
        self.assertEqual(data[0].shape, output[1].shape)

        # Impulse response tx 1 to rx 1
        impulse_response0 = musisochannel.get_last_impulse_response(0, 0)
        # Impulse response tx 1 to rx 2
        impulse_response1 = musisochannel.get_last_impulse_response(1, 0)
        # Frequency response tx 1 to rx 1
        freq_response0 = impulse_response0.get_freq_response(fft_size)
        # Frequency response tx 1 to rx 2
        freq_response1 = impulse_response1.get_freq_response(fft_size)

        # xxxxxxxxxx Expected received signal at first receiver xxxxxxxxxxx
        expected_ofdm_symb1_rx1 = (data[0, 0:fft_size] * freq_response0[:, 0])
        expected_ofdm_symb1_rx2 = (data[0, 0:fft_size] * freq_response1[:, 0])
        expected_ofdm_symb2_rx1 = (data[0, fft_size:2 * fft_size] *
                                   freq_response0[:, 1])
        expected_ofdm_symb2_rx2 = (data[0, fft_size:2 * fft_size] *
                                   freq_response1[:, 1])
        expected_ofdm_symb3_rx1 = (data[0, 2 * fft_size:3 * fft_size] *
                                   freq_response0[:, 2])
        expected_ofdm_symb3_rx2 = (data[0, 2 * fft_size:3 * fft_size] *
                                   freq_response1[:, 2])
        expected_ofdm_symb4_rx1 = (data[0, 3 * fft_size:4 * fft_size] *
                                   freq_response0[:, 3])
        expected_ofdm_symb4_rx2 = (data[0, 3 * fft_size:4 * fft_size] *
                                   freq_response1[:, 3])

        # Test received signal at first receiver
        np.testing.assert_array_almost_equal(
            output[0],
            np.hstack([
                expected_ofdm_symb1_rx1, expected_ofdm_symb2_rx1,
                expected_ofdm_symb3_rx1, expected_ofdm_symb4_rx1
            ]))

        # Test received signal at second receiver
        np.testing.assert_array_almost_equal(
            output[1],
            np.hstack([
                expected_ofdm_symb1_rx2, expected_ofdm_symb2_rx2,
                expected_ofdm_symb3_rx2, expected_ofdm_symb4_rx2
            ]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Repeat the test, but now data has a single dimension (since we
        # only have one transmitter)
        # Generate data for 1 transmitter
        data2 = np.random.randint(0, 10, num_samples)

        # Pass data through he channel
        output2 = musisochannel.corrupt_data_in_freq_domain(data2, fft_size)

        self.assertEqual(data2.shape, output2[0].shape)
        self.assertEqual(data2.shape, output2[1].shape)

        # Impulse response tx 1 to rx 1
        impulse_response0 = musisochannel.get_last_impulse_response(0, 0)
        # Impulse response tx 1 to rx 2
        impulse_response1 = musisochannel.get_last_impulse_response(1, 0)
        # Frequency response tx 1 to rx 1
        freq_response0 = impulse_response0.get_freq_response(fft_size)
        # Frequency response tx 1 to rx 2
        freq_response1 = impulse_response1.get_freq_response(fft_size)

        # xxxxxxxxxx Expected received signal at first receiver xxxxxxxxxxx
        expected_ofdm_symb1_rx1 = (data2[0:fft_size] * freq_response0[:, 0])
        expected_ofdm_symb1_rx2 = (data2[0:fft_size] * freq_response1[:, 0])
        expected_ofdm_symb2_rx1 = (data2[fft_size:2 * fft_size] *
                                   freq_response0[:, 1])
        expected_ofdm_symb2_rx2 = (data2[fft_size:2 * fft_size] *
                                   freq_response1[:, 1])
        expected_ofdm_symb3_rx1 = (data2[2 * fft_size:3 * fft_size] *
                                   freq_response0[:, 2])
        expected_ofdm_symb3_rx2 = (data2[2 * fft_size:3 * fft_size] *
                                   freq_response1[:, 2])
        expected_ofdm_symb4_rx1 = (data2[3 * fft_size:4 * fft_size] *
                                   freq_response0[:, 3])
        expected_ofdm_symb4_rx2 = (data2[3 * fft_size:4 * fft_size] *
                                   freq_response1[:, 3])

        # Test received signal at first receiver
        np.testing.assert_array_almost_equal(
            output2[0],
            np.hstack([
                expected_ofdm_symb1_rx1, expected_ofdm_symb2_rx1,
                expected_ofdm_symb3_rx1, expected_ofdm_symb4_rx1
            ]))

        # Test received signal at second receiver
        np.testing.assert_array_almost_equal(
            output2[1],
            np.hstack([
                expected_ofdm_symb1_rx2, expected_ofdm_symb2_rx2,
                expected_ofdm_symb3_rx2, expected_ofdm_symb4_rx2
            ]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Now test with switched directions xxxxxxxxxxxxxxxxxxxx
        # Switch directions. Now we have 2 transmitter and 1 receiver
        musisochannel.switched_direction = True

        # Generate data for 2 transmitters
        # data = np.ones((2, num_samples))
        data = np.random.randint(0, 10, (2, num_samples))

        # Pass data through he channel
        output = musisochannel.corrupt_data_in_freq_domain(data, fft_size)

        self.assertEqual(data[0].shape, output[0].shape)

        # Impulse response tx 1 to rx 1
        impulse_response0 = musisochannel.get_last_impulse_response(0, 0)
        # Impulse response tx 1 to rx 2
        impulse_response1 = musisochannel.get_last_impulse_response(1, 0)
        # Frequency response tx 1 to rx 1
        freq_response0 = impulse_response0.get_freq_response(fft_size)
        # Frequency response tx 1 to rx 2
        freq_response1 = impulse_response1.get_freq_response(fft_size)

        # xxxxxxxxxx Expected received signal at first receiver xxxxxxxxxxx
        expected_ofdm_symb1_rx = (data[0, 0:fft_size] * freq_response0[:, 0] +
                                  data[1, 0:fft_size] * freq_response1[:, 0])
        expected_ofdm_symb2_rx = (
            data[0, fft_size:2 * fft_size] * freq_response0[:, 1] +
            data[1, fft_size:2 * fft_size] * freq_response1[:, 1])
        expected_ofdm_symb3_rx = (
            data[0, 2 * fft_size:3 * fft_size] * freq_response0[:, 2] +
            data[1, 2 * fft_size:3 * fft_size] * freq_response1[:, 2])
        expected_ofdm_symb4_rx = (
            data[0, 3 * fft_size:4 * fft_size] * freq_response0[:, 3] +
            data[1, 3 * fft_size:4 * fft_size] * freq_response1[:, 3])

        # Test received signal at the receiver
        np.testing.assert_array_almost_equal(
            output[0],
            np.hstack([
                expected_ofdm_symb1_rx, expected_ofdm_symb2_rx,
                expected_ofdm_symb3_rx, expected_ofdm_symb4_rx
            ]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class MuMimoChannelTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.num_rx_antennas = 3
        self.num_tx_antennas = 2
        self.N = 2

        self.mumimochannel = multiuser.MuMimoChannel(
            N=self.N,
            num_rx_antennas=self.num_rx_antennas,
            num_tx_antennas=self.num_tx_antennas)

    def test_repr(self):
        self.assertEqual(repr(self.mumimochannel),
                         "MuMimoChannel(shape=2x2, switched=False)")
        self.mumimochannel.switched_direction = True
        self.assertEqual(repr(self.mumimochannel),
                         "MuMimoChannel(shape=2x2, switched=True)")

        mumimochannel2 = multiuser.MuMimoChannel(N=4,
                                                 num_rx_antennas=3,
                                                 num_tx_antennas=5)
        self.assertEqual(repr(mumimochannel2),
                         "MuMimoChannel(shape=4x4, switched=False)")
        mumimochannel2.switched_direction = True
        self.assertEqual(repr(mumimochannel2),
                         "MuMimoChannel(shape=4x4, switched=True)")

    def test_constructor(self):
        N = 4
        num_rx_antennas = 3
        num_tx_antennas = 2

        mumimochannel = multiuser.MuMimoChannel(N, num_rx_antennas,
                                                num_tx_antennas)

        self.assertEqual(mumimochannel._su_siso_channels.shape, (N, N))

        for tx in range(N):
            for rx in range(N):
                self.assertEqual(
                    mumimochannel._su_siso_channels[rx, tx].num_tx_antennas,
                    num_tx_antennas)
                self.assertEqual(
                    mumimochannel._su_siso_channels[rx, tx].num_rx_antennas,
                    num_rx_antennas)

    def test_corrupt_data(self):
        num_samples = 10
        data1 = np.random.randint(0, 10,
                                  (self.N, self.num_tx_antennas, num_samples))

        output1 = self.mumimochannel.corrupt_data(data1)
        impulse_response00 = self.mumimochannel.get_last_impulse_response(0, 0)
        impulse_response01 = self.mumimochannel.get_last_impulse_response(0, 1)
        impulse_response10 = self.mumimochannel.get_last_impulse_response(1, 0)
        impulse_response11 = self.mumimochannel.get_last_impulse_response(1, 1)

        # We only have the first tap. hXX has dimension
        # `num_rx_ant x num_tx_ant x num_samples`
        h00 = impulse_response00.tap_values[0]
        h01 = impulse_response01.tap_values[0]
        h10 = impulse_response10.tap_values[0]
        h11 = impulse_response11.tap_values[0]

        expected_output1 = np.empty(self.N, dtype=object)
        expected_output1[0] = np.zeros((self.num_rx_antennas, num_samples),
                                       dtype=complex)
        expected_output1[1] = np.zeros((self.num_rx_antennas, num_samples),
                                       dtype=complex)
        for i in range(num_samples):
            expected_output1[0][:, i] = (h00[:, :, i].dot(data1[0, :, i]) +
                                         h01[:, :, i].dot(data1[1, :, i]))
            expected_output1[1][:, i] = (h10[:, :, i].dot(data1[0, :, i]) +
                                         h11[:, :, i].dot(data1[1, :, i]))

        np.testing.assert_array_almost_equal(expected_output1[0], output1[0])
        np.testing.assert_array_almost_equal(expected_output1[1], output1[1])

        # xxxxxxxxxx Now test with switched directions xxxxxxxxxxxxxxxxxxxx
        # First let's switch directions
        self.mumimochannel.switched_direction = True

        # In the reverse direction we have self.num_rx_antennas TRANSMIT
        # antennas
        data2 = np.random.randint(0, 10,
                                  (self.N, self.num_rx_antennas, num_samples))

        # Now let's transmit in the other direction
        output2 = self.mumimochannel.corrupt_data(data2)

        impulse_response00 = self.mumimochannel.get_last_impulse_response(0, 0)
        impulse_response10 = self.mumimochannel.get_last_impulse_response(1, 0)
        impulse_response01 = self.mumimochannel.get_last_impulse_response(0, 1)
        impulse_response11 = self.mumimochannel.get_last_impulse_response(1, 1)

        # We only have the first tap.
        h00 = impulse_response00.tap_values[0].T
        h10 = impulse_response10.tap_values[0].T
        h01 = impulse_response01.tap_values[0].T
        h11 = impulse_response11.tap_values[0].T

        expected_output2 = np.empty(self.N, dtype=object)
        expected_output2[0] = np.zeros((self.num_tx_antennas, num_samples),
                                       dtype=complex)
        expected_output2[1] = np.zeros((self.num_tx_antennas, num_samples),
                                       dtype=complex)
        for i in range(num_samples):
            expected_output2[0][:, i] = (h00[i, :, :].dot(data2[0, :, i]) +
                                         h10[i, :, :].dot(data2[1, :, i]))
            expected_output2[1][:, i] = (h01[i, :, :].dot(data2[0, :, i]) +
                                         h11[i, :, :].dot(data2[1, :, i]))

        np.testing.assert_array_almost_equal(expected_output2[0], output2[0])
        np.testing.assert_array_almost_equal(expected_output2[1], output2[1])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_corrupt_data_in_freq_domain(self):
        Ts = 3.25e-8
        # Create a jakes fading generator
        jakes = fading_generators.JakesSampleGenerator(Fd=30, Ts=Ts, L=16)
        # Create a channel profile with 2 (sparse) taps. Including zero
        # padding, this channel as 4 taps with the non zeros taps at delays 0
        # and 3.
        channel_profile = fading.COST259_TUx
        mumimochannel = multiuser.MuMimoChannel(
            N=self.N,
            num_rx_antennas=self.num_rx_antennas,
            num_tx_antennas=self.num_tx_antennas,
            fading_generator=jakes,
            channel_profile=channel_profile)

        fft_size = 64
        num_blocks = 4
        num_samples = num_blocks * fft_size
        # xxxxxxxxxx Test without pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Generate data for 2 transmitters
        # data = np.ones((2, num_samples))
        data = np.random.randint(0, 10,
                                 (self.N, self.num_tx_antennas, num_samples))

        # Pass data through he channel
        output = mumimochannel.corrupt_data_in_freq_domain(data, fft_size)
        self.assertEqual((self.num_rx_antennas, num_samples), output[0].shape)
        self.assertEqual((self.num_rx_antennas, num_samples), output[1].shape)

        impulse_response00 = mumimochannel.get_last_impulse_response(0, 0)
        impulse_response01 = mumimochannel.get_last_impulse_response(0, 1)
        impulse_response10 = mumimochannel.get_last_impulse_response(1, 0)
        impulse_response11 = mumimochannel.get_last_impulse_response(1, 1)

        freq_response00 = impulse_response00.get_freq_response(fft_size)
        freq_response01 = impulse_response01.get_freq_response(fft_size)
        freq_response10 = impulse_response10.get_freq_response(fft_size)
        freq_response11 = impulse_response11.get_freq_response(fft_size)

        # xxxxxxxxxx Expected received signal xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        expected_output = np.empty(self.N, dtype=object)
        expected_output[0] = np.zeros((self.num_rx_antennas, num_samples),
                                      dtype=complex)
        expected_output[1] = np.zeros((self.num_rx_antennas, num_samples),
                                      dtype=complex)

        for b in range(num_blocks):
            start_idx = b * fft_size
            # end_idx = (b+1)*fft_size
            for k in range(fft_size):
                expected_output[0][:, start_idx + k] = (
                    freq_response00[k, :, :, b].dot(data[0, :,
                                                         start_idx + k]) +
                    freq_response01[k, :, :, b].dot(data[1, :, start_idx + k]))
                expected_output[1][:, start_idx + k] = (
                    freq_response10[k, :, :, b].dot(data[0, :,
                                                         start_idx + k]) +
                    freq_response11[k, :, :, b].dot(data[1, :, start_idx + k]))

        # Test if the output and the expected output are equal
        np.testing.assert_array_almost_equal(expected_output[0], output[0])
        np.testing.assert_array_almost_equal(expected_output[1], output[1])

        # xxxxxxxxxx Now test with switched directions xxxxxxxxxxxxxxxxxxxx
        # First we switch directions
        mumimochannel.switched_direction = True

        # Pass data through he channel
        data2 = np.random.randint(0, 10,
                                  (self.N, self.num_rx_antennas, num_samples))
        output2 = mumimochannel.corrupt_data_in_freq_domain(data2, fft_size)
        # Since we switched directions, the number of "receive" antennas is
        # equal to the number of transmit antennas in the original
        # direction
        self.assertEqual((self.num_tx_antennas, num_samples), output2[0].shape)
        self.assertEqual((self.num_tx_antennas, num_samples), output2[1].shape)

        impulse_response00 = mumimochannel.get_last_impulse_response(0, 0)
        impulse_response10 = mumimochannel.get_last_impulse_response(1, 0)
        impulse_response01 = mumimochannel.get_last_impulse_response(0, 1)
        impulse_response11 = mumimochannel.get_last_impulse_response(1, 1)

        freq_response00 = impulse_response00.get_freq_response(fft_size)
        freq_response10 = impulse_response10.get_freq_response(fft_size)
        freq_response01 = impulse_response01.get_freq_response(fft_size)
        freq_response11 = impulse_response11.get_freq_response(fft_size)

        # xxxxxxxxxx Expected received signal xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        expected_output2 = np.empty(self.N, dtype=object)
        expected_output2[0] = np.zeros((self.num_tx_antennas, num_samples),
                                       dtype=complex)
        expected_output2[1] = np.zeros((self.num_tx_antennas, num_samples),
                                       dtype=complex)
        for b in range(num_blocks):
            start_idx = b * fft_size
            # end_idx = (b+1)*fft_size
            for k in range(fft_size):
                expected_output2[0][:, start_idx +
                                    k] = (freq_response00[k, :, :, b].T.dot(
                                        data2[0, :, start_idx + k]) +
                                          freq_response10[k, :, :, b].T.dot(
                                              data2[1, :, start_idx + k]))
                expected_output2[1][:, start_idx +
                                    k] = (freq_response01[k, :, :, b].T.dot(
                                        data2[0, :, start_idx + k]) +
                                          freq_response11[k, :, :, b].T.dot(
                                              data2[1, :, start_idx + k]))

        # Test if the output2 and the expected output2 are equal
        np.testing.assert_array_almost_equal(expected_output2[0], output2[0])
        np.testing.assert_array_almost_equal(expected_output2[1], output2[1])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# noinspection PyMethodMayBeStatic
class MultiUserChannelMatrixTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.multiH = multiuser.MultiUserChannelMatrix()
        self.H = np.array([
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
        ])
        self.K = 3
        self.Nr = np.array([2, 4, 6])
        self.Nt = np.array([2, 3, 5])

    def test_from_small_matrix_to_big_matrix(self):
        K = 3
        Nr = np.array([2, 4, 6])
        Nt = np.array([2, 3, 5])
        small_matrix = np.arange(1, 10)
        small_matrix.shape = (3, 3)
        big_matrix = \
            multiuser.MultiUserChannelMatrix._from_small_matrix_to_big_matrix(
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
                self.assertEqual(self.multiH.H[rx, tx].shape, (Nr[rx], Nt[tx]))
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

        # We don't really need to test multiH.H because the code was already
        # tested in test_from_big_matrix

    def test_get_channel(self):
        H = self.H
        K = self.K
        Nr = self.Nr
        Nt = self.Nt
        self.multiH.init_from_channel_matrix(H, Nr, Nt, K)

        # xxxxxxxxxx Test get_channel without Pathloss xxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_equal(self.multiH.get_Hkl(0, 0),
                                      np.ones([2, 2]) * 0)

        np.testing.assert_array_equal(self.multiH.get_Hkl(0, 1),
                                      np.ones([2, 3]) * 1)

        np.testing.assert_array_equal(self.multiH.get_Hkl(0, 2),
                                      np.ones([2, 5]) * 2)

        np.testing.assert_array_equal(self.multiH.get_Hkl(1, 0),
                                      np.ones([4, 2]) * 3)

        np.testing.assert_array_equal(self.multiH.get_Hkl(1, 1),
                                      np.ones([4, 3]) * 4)

        np.testing.assert_array_equal(self.multiH.get_Hkl(1, 2),
                                      np.ones([4, 5]) * 5)

        np.testing.assert_array_equal(self.multiH.get_Hkl(2, 0),
                                      np.ones([6, 2]) * 6)

        np.testing.assert_array_equal(self.multiH.get_Hkl(2, 1),
                                      np.ones([6, 3]) * 7)

        np.testing.assert_array_equal(self.multiH.get_Hkl(2, 2),
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
        np.testing.assert_array_equal(self.multiH.get_Hk(0), expected_H1)
        np.testing.assert_array_equal(self.multiH.get_Hk(1), expected_H2)
        np.testing.assert_array_equal(self.multiH.get_Hk(2), expected_H3)

        # xxxxx Test with pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        pathloss = np.abs(np.random.randn(self.K, self.K))
        self.multiH.set_pathloss(pathloss)
        expected_H1 = self.multiH.big_H[0:2, :]
        expected_H2 = self.multiH.big_H[2:6, :]
        expected_H3 = self.multiH.big_H[6:, :]
        np.testing.assert_array_equal(self.multiH.get_Hk(0), expected_H1)
        np.testing.assert_array_equal(self.multiH.get_Hk(1), expected_H2)
        np.testing.assert_array_equal(self.multiH.get_Hk(2), expected_H3)

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
                np.testing.assert_array_equal(self.multiH.get_Hkl(row, col),
                                              self.multiH.H[row, col])
                # Test the 'big_H' property
                np.testing.assert_array_equal(
                    self.multiH.get_Hkl(row, col),
                    self.multiH.big_H[cumNr[row]:cumNr[row + 1],
                                      cumNt[col]:cumNt[col + 1]])

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
        # Note that the corrupt_concatenated_data is implicitly called by
        # corrupt_data and thus we will only test corrupt_data.
        output = self.multiH.corrupt_data(input_data)

        # Calculates the expected output (without pathloss)
        expected_output = np.zeros(self.K, dtype=np.ndarray)
        for rx in np.arange(self.K):
            for tx in np.arange(self.K):
                expected_output[rx] += np.dot(self.multiH.get_Hkl(rx, tx),
                                              input_data[tx])

        # Test the received data for the 3 users
        np.testing.assert_array_almost_equal(output[0], expected_output[0])
        np.testing.assert_array_almost_equal(output[1], expected_output[1])
        np.testing.assert_array_almost_equal(output[2], expected_output[2])

        # xxxxxxxxxx Test with pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # pathloss (in linear scale) must be a positive number
        pathloss = np.abs(np.random.randn(self.K, self.K))
        self.multiH.set_pathloss(pathloss)

        # Note that the corrupt_concatenated_data is implicitly called by
        # corrupt_data and thus we will only test corrupt_data. Also, they
        # are affected by the pathloss.
        output2 = self.multiH.corrupt_data(input_data)

        # Calculates the expected output (with pathloss)
        expected_output2 = np.zeros(self.K, dtype=np.ndarray)
        for rx in np.arange(self.K):
            for tx in np.arange(self.K):
                expected_output2[rx] += np.dot(
                    # Note that get_channel is affected by the pathloss
                    self.multiH.get_Hkl(rx, tx),
                    input_data[tx])

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

        W = [randn_c(2, 2), randn_c(2, 2), randn_c(2, 2)]

        self.multiH.set_post_filter(W)
        np.testing.assert_array_almost_equal(W, self.multiH._W)
        np.testing.assert_array_almost_equal(W, self.multiH.W)

        # _big_W is still None
        self.assertIsNone(self.multiH._big_W)

        expected_big_W = block_diag(*W)
        np.testing.assert_array_almost_equal(expected_big_W, self.multiH.big_W)
        self.assertIsNotNone(self.multiH._big_W)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        W2 = [randn_c(2, 2), randn_c(2, 2), randn_c(2, 2)]
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
        W = [
            randn_c(self.Nr[0], self.Nr[0]),
            randn_c(self.Nr[1], self.Nr[1]),
            randn_c(self.Nr[2], self.Nr[2])
        ]
        self.multiH.set_post_filter(W)

        output = self.multiH.corrupt_data(input_data)

        # Calculates the expected output (without pathloss)
        expected_output = np.zeros(self.K, dtype=np.ndarray)
        for rx in np.arange(self.K):
            for tx in np.arange(self.K):
                expected_output[rx] += np.dot(self.multiH.get_Hkl(rx, tx),
                                              input_data[tx])
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
        self.multiH.init_from_channel_matrix(H, np.array([2, 2, 2]),
                                             np.array([2, 2, 2]), 3)

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
            F_all_k[k] = (F_all_k[k] / np.linalg.norm(F_all_k[k], 'fro') *
                          np.sqrt(P[k]))

        # xxxxx Calculate the expected Q[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H01_F1 = np.dot(self.multiH.get_Hkl(k, 1), F_all_k[1])
        H02_F2 = np.dot(self.multiH.get_Hkl(k, 2), F_all_k[2])
        expected_Q0 = (np.dot(H01_F1,
                              H01_F1.transpose().conjugate()) +
                       np.dot(H02_F2,
                              H02_F2.transpose().conjugate()))

        Qk = self.multiH.calc_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q0)

        # Now with noise variance different of 0
        noise_var = round(0.1 * np.random.random_sample(), 4)
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_Q(k, F_all_k)
        np.testing.assert_array_almost_equal(
            Qk, expected_Q0 + noise_var * np.eye(2))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0 = np.dot(self.multiH.get_Hkl(k, 0), F_all_k[0])
        H12_F2 = np.dot(self.multiH.get_Hkl(k, 2), F_all_k[2])
        expected_Q1 = (np.dot(H10_F0,
                              H10_F0.transpose().conjugate()) +
                       np.dot(H12_F2,
                              H12_F2.transpose().conjugate()))

        self.multiH.noise_var = 0.0
        Qk = self.multiH.calc_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q1)

        # Now with noise variance different of 0
        noise_var = round(0.1 * np.random.random_sample(), 4)
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_Q(k, F_all_k)
        np.testing.assert_array_almost_equal(
            Qk, expected_Q1 + noise_var * np.eye(2))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H20_F0 = np.dot(self.multiH.get_Hkl(k, 0), F_all_k[0])
        H21_F1 = np.dot(self.multiH.get_Hkl(k, 1), F_all_k[1])
        expected_Q2 = (np.dot(H20_F0,
                              H20_F0.transpose().conjugate()) +
                       np.dot(H21_F1,
                              H21_F1.transpose().conjugate()))

        # Calculate Qk without noise
        Qk = self.multiH._calc_Q_impl(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q2)

        # Now with noise variance different of 0
        noise_var = round(0.1 * np.random.random_sample(), 4)
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_Q(k, F_all_k)
        np.testing.assert_array_almost_equal(
            Qk, expected_Q2 + noise_var * np.eye(2))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_JP_Q(self):
        K = 3
        Nt = np.array([2, 2, 2])
        Nr = np.array([2, 2, 2])
        Ns = np.array([1, 1, 1])
        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        self.multiH.randomize(Nr, Nt, K)
        noise_var = round(0.1 * np.random.random_sample(), 4)
        self.multiH.noise_var = noise_var

        F_all_k = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F_all_k[k] = randn_c(np.sum(Nt), Ns[k]) * np.sqrt(P[k])
            F_all_k[k] = (F_all_k[k] / np.linalg.norm(F_all_k[k], 'fro') *
                          np.sqrt(P[k]))

        # xxxxx Calculate the expected Q[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H0_F1 = np.dot(self.multiH.get_Hk(k), F_all_k[1])
        H0_F2 = np.dot(self.multiH.get_Hk(k), F_all_k[2])
        expected_Q0 = (np.dot(H0_F1,
                              H0_F1.transpose().conjugate()) +
                       np.dot(H0_F2,
                              H0_F2.transpose().conjugate()))

        # Test if Qk (without noise) is equal to the expected output
        Qk = self.multiH._calc_JP_Q_impl(k, F_all_k)
        np.testing.assert_array_almost_equal(Qk, expected_Q0)

        # Now with noise variance different of 0
        Qk = self.multiH.calc_JP_Q(k, F_all_k)
        np.testing.assert_array_almost_equal(
            Qk, expected_Q0 + noise_var * np.eye(2))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H1_F0 = np.dot(self.multiH.get_Hk(k), F_all_k[0])
        H1_F2 = np.dot(self.multiH.get_Hk(k), F_all_k[2])
        expected_Q1 = (np.dot(H1_F0,
                              H1_F0.transpose().conjugate()) +
                       np.dot(H1_F2,
                              H1_F2.transpose().conjugate()))

        # Test if Qk (without noise) is equal to the expected output
        Qk = self.multiH._calc_JP_Q_impl(k, F_all_k)
        np.testing.assert_array_almost_equal(Qk, expected_Q1)

        # Now with noise variance different of 0
        Qk = self.multiH.calc_JP_Q(k, F_all_k)
        np.testing.assert_array_almost_equal(
            Qk, expected_Q1 + noise_var * np.eye(2))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H2_F0 = np.dot(self.multiH.get_Hk(k), F_all_k[0])
        H2_F1 = np.dot(self.multiH.get_Hk(k), F_all_k[1])
        expected_Q2 = (np.dot(H2_F0,
                              H2_F0.transpose().conjugate()) +
                       np.dot(H2_F1,
                              H2_F1.transpose().conjugate()))

        # Test if Qk (without noise) is equal to the expected output
        Qk = self.multiH._calc_JP_Q_impl(k, F_all_k)
        np.testing.assert_array_almost_equal(Qk, expected_Q2)

        # Now with noise variance different of 0
        Qk = self.multiH.calc_JP_Q(k, F_all_k)
        np.testing.assert_array_almost_equal(
            Qk, expected_Q2 + noise_var * np.eye(2))
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
            expected_first_part = (self.multiH._calc_Q_impl(k, F) +
                                   np.dot(HkkFk,
                                          HkkFk.conjugate().T))
            expected_first_part_with_noise = (self.multiH.calc_Q(k, F) +
                                              np.dot(HkkFk,
                                                     HkkFk.conjugate().T))

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

        # noinspection PyPep8
        for k in range(K):
            # First part in the equation of Bkl (the double summation)
            expected_first_part = 0.0
            ":type: np.ndarray"

            # The inner for loop will calculate
            # $\text{aux} = \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$
            for j in range(K):
                aux = 0.0
                ":type: np.ndarray"
                Hkj = self.multiH.get_Hkl(k, j)
                Hkj_H = Hkj.conjugate().transpose()

                # Calculates individually for each stream
                for d in range(Ns[k]):
                    Vjd = F[j][:, d:d + 1]
                    Vjd_H = Vjd.conjugate().transpose()
                    aux = aux + np.dot(np.dot(Hkj, np.dot(Vjd, Vjd_H)), Hkj_H)

                expected_first_part += aux

            expected_first_part_with_noise = (expected_first_part +
                                              np.eye(Nr[k]) * noise_power)
            ":type: np.ndarray"

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
            # noinspection PyPep8
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(
                    Hkk, np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
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
            # noinspection PyPep8
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(
                    Hkk, np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
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
            # noinspection PyPep8
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(
                    Hkk, np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
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
            expected_Bk0 = (self.multiH.calc_Q(k, F) +
                            (noise_power * np.eye(Nr[k])))
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
                aux = np.dot(Ukl_H, np.dot(Hkk, Vkl))

                expectedSINRkl = (
                    np.dot(aux,
                           aux.transpose().conjugate()) /
                    np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))).item()

                self.assertAlmostEqual(expectedSINRkl, SINR_k_all_l[l])

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
            SINR_k_all_l = multiUserChannel._calc_SINR_k(k, Fk, Uk, Bkl_all_l)

            for l in range(Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = F[k][:, l:l + 1]
                aux = np.dot(Ukl_H, np.dot(Hkk, Vkl))

                expectedSINRkl = abs(
                    (np.dot(aux,
                            aux.transpose().conjugate()) /
                     np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))).item())

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
        B0l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(F,
                                                                k=0,
                                                                N0_or_Rek=0.0)
        expected_SINR0 = multiUserChannel._calc_SINR_k(0, F[0], U[0],
                                                       B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(F,
                                                                k=1,
                                                                N0_or_Rek=0.0)
        expected_SINR1 = multiUserChannel._calc_SINR_k(1, F[1], U[1],
                                                       B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 1
        B2l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(F,
                                                                k=2,
                                                                N0_or_Rek=0.0)
        expected_SINR2 = multiUserChannel._calc_SINR_k(2, F[2], U[2],
                                                       B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Noise Variance of 0.1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # k = 0
        noise_var = 0.1
        multiUserChannel.noise_var = noise_var
        SINR_all_users = multiUserChannel.calc_SINR(F, U)
        B0l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(F,
                                                                k=0,
                                                                N0_or_Rek=0.1)
        expected_SINR0 = multiUserChannel._calc_SINR_k(0, F[0], U[0],
                                                       B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(F,
                                                                k=1,
                                                                N0_or_Rek=0.1)
        expected_SINR1 = multiUserChannel._calc_SINR_k(1, F[1], U[1],
                                                       B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 2
        B2l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(F,
                                                                k=2,
                                                                N0_or_Rek=0.1)
        expected_SINR2 = multiUserChannel._calc_SINR_k(2, F[2], U[2],
                                                       B2l_all_l)
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
            expected_first_part = (self.multiH._calc_JP_Q_impl(k, F) +
                                   np.dot(HkFk,
                                          HkFk.conjugate().T))
            expected_first_part_with_noise = (self.multiH.calc_JP_Q(k, F) +
                                              np.dot(HkFk,
                                                     HkFk.conjugate().T))

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

        # noinspection PyPep8
        for k in range(K):
            # First part in the equation of Bkl (the double summation)
            expected_first_part = 0.0
            ":type: np.ndarray"

            # The inner for loop will calculate
            # $\text{aux} = \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$
            Hk = self.multiH.get_Hk(k)
            Hk_H = Hk.conjugate().transpose()
            for j in range(K):
                aux = 0.0
                ":type: np.ndarray"

                # Calculates individually for each stream
                for d in range(Ns[k]):
                    Vjd = F[j][:, d:d + 1]
                    Vjd_H = Vjd.conjugate().transpose()
                    aux = aux + np.dot(np.dot(Hk, np.dot(Vjd, Vjd_H)), Hk_H)

                expected_first_part += aux

            expected_first_part_with_noise = (expected_first_part +
                                              np.eye(Nr[k]) * noise_power)
            ":type: np.ndarray"

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
            # noinspection PyPep8
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
            # noinspection PyPep8
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
                aux = np.dot(Ukl_H, np.dot(Hk, Vkl))

                expectedSINRkl = np.abs(
                    (np.dot(aux,
                            aux.transpose().conjugate()) /
                     np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))).item())

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
                aux = np.dot(Ukl_H, np.dot(Hk, Vkl))

                expectedSINRkl = np.abs(
                    (np.dot(aux,
                            aux.transpose().conjugate()) /
                     np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))).item())

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
        B0l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F,
                                                              k=0,
                                                              N0_or_Rek=0.0)
        expected_SINR0 = self.multiH._calc_JP_SINR_k(0, F[0], U[0], B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F,
                                                              k=1,
                                                              N0_or_Rek=0.0)
        expected_SINR1 = self.multiH._calc_JP_SINR_k(1, F[1], U[1], B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 1
        B2l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F,
                                                              k=2,
                                                              N0_or_Rek=0.0)
        expected_SINR2 = self.multiH._calc_JP_SINR_k(2, F[2], U[2], B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Noise Variance of 0.1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # k = 0
        self.multiH.noise_var = 0.1
        SINR_all_users = self.multiH.calc_JP_SINR(F, U)
        B0l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F,
                                                              k=0,
                                                              N0_or_Rek=0.1)
        expected_SINR0 = self.multiH._calc_JP_SINR_k(0, F[0], U[0], B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F,
                                                              k=1,
                                                              N0_or_Rek=0.1)
        expected_SINR1 = self.multiH._calc_JP_SINR_k(1, F[1], U[1], B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 2
        B2l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F,
                                                              k=2,
                                                              N0_or_Rek=0.1)
        expected_SINR2 = self.multiH._calc_JP_SINR_k(2, F[2], U[2], B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# noinspection PyPep8
class MultiUserChannelMatrixExtIntTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.multiH = multiuser.MultiUserChannelMatrixExtInt()
        self.H = np.array([
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
        ])

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

        self.multiH.init_from_channel_matrix(big_H, self.Nr, self.Nt, self.K,
                                             self.NtE)

        # Test the big_H property. It should be exactly equal to the big_H
        # variable passed to the init_from_channel_matrix method, since we
        # didn't set any path loss matrix yet.
        np.testing.assert_array_equal(self.multiH.big_H, big_H)

        # Test the properties
        np.testing.assert_array_equal(self.multiH.Nr, self.Nr)
        np.testing.assert_array_equal(self.multiH.Nt, self.Nt)
        self.assertEqual(self.multiH.K, self.K)
        self.assertEqual(self.multiH.extIntK, len(self.NtE))
        np.testing.assert_array_equal(self.multiH.extIntNt, self.NtE)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now we consider a single external interference source with three
        # antennas
        self.multiH.init_from_channel_matrix(big_H, self.Nr, self.Nt, self.K,
                                             np.sum(self.NtE))

        # Test the properties
        np.testing.assert_array_equal(self.multiH.Nr, self.Nr)
        np.testing.assert_array_equal(self.multiH.Nt, self.Nt)
        self.assertEqual(self.multiH.K, self.K)
        self.assertEqual(self.multiH.extIntK, 1)
        np.testing.assert_array_equal(self.multiH.extIntNt, np.sum(self.NtE))

        # We won't test the channels here because the code for setting
        # _big_H and _H was already well tested in the
        # MultiUserChannelMatrix class.

    def test_randomize(self):
        self.multiH.randomize(self.Nr, self.Nt, self.K, self.NtE)

        # Test the properties
        np.testing.assert_array_equal(self.multiH.Nr, self.Nr)
        np.testing.assert_array_equal(self.multiH.Nt, self.Nt)
        self.assertEqual(self.multiH.K, self.K)
        self.assertEqual(self.multiH.extIntK, len(self.NtE))
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
        expected_pathloss_big_matrix = np.array([
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
            self.multiH.big_H, self.multiH.Nr,
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
            received_data2_expected[0] +
            np.dot(self.multiH.get_Hkl(0, 2), input_data_extint2[0]) +
            np.dot(self.multiH.get_Hkl(0, 3), input_data_extint2[1]))
        received_data2_expected[1] = (
            received_data2_expected[1] +
            np.dot(self.multiH.get_Hkl(1, 2), input_data_extint2[0]) +
            np.dot(self.multiH.get_Hkl(1, 3), input_data_extint2[1]))

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
        W = [randn_c(Nr[0], Nr[0]), randn_c(Nr[1], Nr[1])]
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
            self.multiH.big_H_no_ext_int, Nr, Nt, K)
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
            + np.dot(W[0].conjugate().T,
                     np.dot(self.multiH.get_Hkl(0, 2), input_data_extint2[0]))
            # Plus FILTERED interference from second interference source
            + np.dot(W[0].conjugate().T,
                     np.dot(self.multiH.get_Hkl(0, 3), input_data_extint2[1])))

        received_data2_expected[1] = (
            # Original received data
            received_data2_expected[1]
            # Plus FILTERED interference from first interference source
            + np.dot(W[1].conjugate().T,
                     np.dot(self.multiH.get_Hkl(1, 2), input_data_extint2[0]))
            # Plus FILTERED interference from second interference source
            + np.dot(W[1].conjugate().T,
                     np.dot(self.multiH.get_Hkl(1, 3), input_data_extint2[1])))

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
        np.testing.assert_array_equal(self.multiH.get_Hk_without_ext_int(0),
                                      expected_H1)
        np.testing.assert_array_equal(self.multiH.get_Hk_without_ext_int(1),
                                      expected_H2)
        np.testing.assert_array_equal(self.multiH.get_Hk_without_ext_int(2),
                                      expected_H3)

        # xxxxx Test with pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        pathloss = np.abs(np.random.randn(K, K))
        pathloss_extint = np.abs(np.random.randn(K, extIntK))
        self.multiH.set_pathloss(pathloss, pathloss_extint)
        expected_H1 = self.multiH.big_H[0:2, :np.sum(Nt)]
        expected_H2 = self.multiH.big_H[2:6, :np.sum(Nt)]
        expected_H3 = self.multiH.big_H[6:, :np.sum(Nt)]
        np.testing.assert_array_equal(self.multiH.get_Hk_without_ext_int(0),
                                      expected_H1)
        np.testing.assert_array_equal(self.multiH.get_Hk_without_ext_int(1),
                                      expected_H2)
        np.testing.assert_array_equal(self.multiH.get_Hk_without_ext_int(2),
                                      expected_H3)

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
            F_all_k[k] = (F_all_k[k] / np.linalg.norm(F_all_k[k], 'fro') *
                          np.sqrt(P[k]))

        # xxxxx Calculate the expected Q[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H01_F1 = np.dot(self.multiH.get_Hkl(k, 1), F_all_k[1])
        H02_F2 = np.dot(self.multiH.get_Hkl(k, 2), F_all_k[2])
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
                                np.dot(R0_e0,
                                       R0_e0.conjugate().T) +
                                np.dot(R0_e1,
                                       R0_e1.conjugate().T))
        Qk = self.multiH.calc_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q0_no_noise)

        # Now with external interference and noise
        noise_var = round(0.1 * np.random.random_sample(), 4)
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_Q(k, F_all_k)
        expected_Q0 = expected_Q0_no_noise + np.eye(2) * noise_var
        np.testing.assert_array_almost_equal(Qk, expected_Q0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0 = np.dot(self.multiH.get_Hkl(k, 0), F_all_k[0])
        H12_F2 = np.dot(self.multiH.get_Hkl(k, 2), F_all_k[2])
        R1_e0 = self.multiH.get_Hkl(1, 3)
        R1_e1 = self.multiH.get_Hkl(1, 4)

        expected_Q1_no_ext_int_or_noise = (
            np.dot(H10_F0,
                   H10_F0.transpose().conjugate()) +
            np.dot(H12_F2,
                   H12_F2.transpose().conjugate()))

        # test without noise
        self.multiH.noise_var = None
        Qk = self.multiH.calc_Q(k, F_all_k, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q1_no_ext_int_or_noise)

        # Now with external interference
        expected_Q1_no_noise = (expected_Q1_no_ext_int_or_noise +
                                np.dot(R1_e0,
                                       R1_e0.conjugate().T) +
                                np.dot(R1_e1,
                                       R1_e1.conjugate().T))
        Qk = self.multiH.calc_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q1_no_noise)

        # Now with external interference and noise
        noise_var = round(0.1 * np.random.random_sample(), 4)
        self.multiH.noise_var = noise_var
        Qk = self.multiH.calc_Q(k, F_all_k)
        expected_Q1 = expected_Q1_no_noise + np.eye(2) * noise_var
        np.testing.assert_array_almost_equal(Qk, expected_Q1)
        # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H20_F0 = np.dot(self.multiH.get_Hkl(k, 0), F_all_k[0])
        H21_F1 = np.dot(self.multiH.get_Hkl(k, 1), F_all_k[1])
        R2_e0 = self.multiH.get_Hkl(2, 3)
        R2_e1 = self.multiH.get_Hkl(2, 4)

        expected_Q2_no_ext_int_or_noise = (
            np.dot(H20_F0,
                   H20_F0.transpose().conjugate()) +
            np.dot(H21_F1,
                   H21_F1.transpose().conjugate()))

        # Test without noise
        self.multiH.noise_var = None
        Qk = self.multiH.calc_Q(k, F_all_k, pe=0.0)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk,
                                             expected_Q2_no_ext_int_or_noise)

        # Now with external interference
        expected_Q2_no_noise = (expected_Q2_no_ext_int_or_noise +
                                np.dot(R2_e0,
                                       R2_e0.conjugate().T) +
                                np.dot(R2_e1,
                                       R2_e1.conjugate().T))
        Qk = self.multiH.calc_Q(k, F_all_k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q2_no_noise)

        # Now with external interference and noise
        noise_var = round(0.1 * np.random.random_sample(), 4)
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

        noise_var = round(0.1 * np.random.random_sample(), 4)
        Pe = round(np.random.random_sample(), 4)

        self.multiH.randomize(Nr, Nt, K, NtE)
        # We don't set the noise variance here because the first tests will
        # be without noise.

        F_all_k = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            F_all_k[k] = randn_c(np.sum(Nt), Ns[k]) * np.sqrt(P[k])
            F_all_k[k] = (F_all_k[k] / np.linalg.norm(F_all_k[k], 'fro') *
                          np.sqrt(P[k]))

        Re_no_noise = self.multiH.calc_cov_matrix_extint_without_noise(pe=Pe)
        # Re_with_noise = self.multiH.calc_cov_matrix_extint_plus_noise(
        #     noise_var=noise_var,
        #     pe=Pe)

        # xxxxx Calculate the expected Q[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H0_F1 = np.dot(self.multiH.get_Hk_without_ext_int(k), F_all_k[1])
        H0_F2 = np.dot(self.multiH.get_Hk_without_ext_int(k), F_all_k[2])

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
        H1_F0 = np.dot(self.multiH.get_Hk_without_ext_int(k), F_all_k[0])
        H1_F2 = np.dot(self.multiH.get_Hk_without_ext_int(k), F_all_k[2])

        expected_Q1_no_ext_int_or_noise = (
            np.dot(H1_F0,
                   H1_F0.transpose().conjugate()) +
            np.dot(H1_F2,
                   H1_F2.transpose().conjugate()))

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
        H2_F0 = np.dot(self.multiH.get_Hk_without_ext_int(k), F_all_k[0])
        H2_F1 = np.dot(self.multiH.get_Hk_without_ext_int(k), F_all_k[1])

        expected_Q2_no_ext_int_or_noise = (
            np.dot(H2_F0,
                   H2_F0.transpose().conjugate()) +
            np.dot(H2_F1,
                   H2_F1.transpose().conjugate()))

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
            expected_first_part = (self.multiH.calc_Q(k, F, pe=Pe) +
                                   np.dot(HkkFk,
                                          HkkFk.conjugate().T))

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
            expected_first_part = (self.multiH.calc_Q(k, F, pe=Pe) +
                                   np.dot(HkkFk,
                                          HkkFk.conjugate().T))

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
            ":type: np.ndarray"

            # The inner for loop will calculate
            # $\text{aux} = \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$
            for j in range(K):
                aux = 0.0
                ":type: np.ndarray"
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
                aux = np.dot(Ukl_H, np.dot(Hkk, Vkl))

                expectedSINRkl = (
                    np.dot(aux,
                           aux.transpose().conjugate()) /
                    np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))).item()

                self.assertAlmostEqual(expectedSINRkl, SINR_k_all_l[l])

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
            SINR_k_all_l = multiUserChannel._calc_SINR_k(k, Fk, Uk, Bkl_all_l)

            for l in range(Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = F[k][:, l:l + 1]
                aux = np.dot(Ukl_H, np.dot(Hkk, Vkl))

                expectedSINRkl = abs(
                    (np.dot(aux,
                            aux.transpose().conjugate()) /
                     np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))).item())

                np.testing.assert_array_almost_equal(expectedSINRkl,
                                                     SINR_k_all_l[l])

    # noinspection PyTypeChecker
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

        SINR_all_users = multiUserChannel.calc_SINR(F, U, pe=Pe)

        # SINR of all users should be super high (inf)
        self.assertTrue(np.all(SINR_all_users[0] > 1e10))
        self.assertTrue(np.all(SINR_all_users[1] > 1e10))
        self.assertTrue(np.all(SINR_all_users[2] > 1e10))

        # k = 0
        B0l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=0, N0_or_Rek=Re[0])
        expected_SINR0 = multiUserChannel._calc_SINR_k(0, F[0], U[0],
                                                       B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=1, N0_or_Rek=Re[1])
        expected_SINR1 = multiUserChannel._calc_SINR_k(1, F[1], U[1],
                                                       B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 2
        B2l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=2, N0_or_Rek=Re[2])
        expected_SINR2 = multiUserChannel._calc_SINR_k(2, F[2], U[2],
                                                       B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Noise Variance = 0.01 and Pe = 0.63 xxxxxxxxxxxxxxxxxx
        Pe = 0.01
        noise_power = 0.63
        multiUserChannel.noise_var = noise_power
        Re = multiUserChannel.calc_cov_matrix_extint_plus_noise(pe=Pe)

        SINR_all_users = multiUserChannel.calc_SINR(F, U, pe=Pe)

        # SINR should lower than 10 for these values of noise variance and Pe
        self.assertTrue(np.all(SINR_all_users[0] < 10))
        self.assertTrue(np.all(SINR_all_users[1] < 10))
        self.assertTrue(np.all(SINR_all_users[2] < 10))

        # k = 0
        B0l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=0, N0_or_Rek=Re[0])
        expected_SINR0 = multiUserChannel._calc_SINR_k(0, F[0], U[0],
                                                       B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=1, N0_or_Rek=Re[1])
        expected_SINR1 = multiUserChannel._calc_SINR_k(1, F[1], U[1],
                                                       B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 2
        B2l_all_l = multiUserChannel._calc_Bkl_cov_matrix_all_l(
            F, k=2, N0_or_Rek=Re[2])
        expected_SINR2 = multiUserChannel._calc_SINR_k(2, F[2], U[2],
                                                       B2l_all_l)
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
            self.multiH.big_H_no_ext_int, K, iPu, noise_power)

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
            expected_first_part = (self.multiH.calc_JP_Q(k, F, pe=0.0) +
                                   np.dot(HkFk,
                                          HkFk.conjugate().T))
            # With noise
            self.multiH.noise_var = noise_power
            expected_first_part_with_noise = self.multiH.calc_JP_Q(
                k, F, pe=0.0) + np.dot(HkFk,
                                       HkFk.conjugate().T)

            # Test without noise
            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.multiH._calc_JP_Bkl_cov_matrix_first_part(
                    F, k, np.zeros([Nr[0], Nr[1]])))

            # Test with noise
            # noinspection PyTypeChecker
            np.testing.assert_array_almost_equal(
                expected_first_part_with_noise,
                self.multiH._calc_JP_Bkl_cov_matrix_first_part(
                    F, k, noise_power * np.eye(Nr[k])))
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
            expected_first_part = (self.multiH.calc_JP_Q(k, F, pe=Pe) +
                                   np.dot(HkFk,
                                          HkFk.conjugate().T))

            # with noise
            self.multiH.noise_var = noise_power
            expected_first_part_with_noise = (
                self.multiH.calc_JP_Q(k, F, pe=Pe) +
                np.dot(HkFk,
                       HkFk.conjugate().T))

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
            self.multiH.big_H_no_ext_int, K, iPu, noise_power)

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
            self.multiH.big_H_no_ext_int, K, iPu, noise_power)

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
            self.multiH.big_H_no_ext_int, K, iPu, noise_power)

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
            self.multiH.big_H_no_ext_int, K, iPu, noise_var=noise_power)

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
                aux = np.dot(Ukl_H, np.dot(Hk, Vkl))

                expectedSINRkl = np.abs(
                    (np.dot(aux,
                            aux.transpose().conjugate()) /
                     np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))).item())

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
                aux = np.dot(Ukl_H, np.dot(Hk, Vkl))

                expectedSINRkl = np.abs(
                    (np.dot(aux,
                            aux.transpose().conjugate()) /
                     np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))).item())

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
            self.multiH.big_H_no_ext_int, K, iPu, noise_var=0.0)

        F = single_matrix_to_matrix_of_matrices(Ms_good, None, Ns)
        big_U = blockdiagonalization.calc_receive_filter(newH)
        aux = single_matrix_to_matrix_of_matrices(big_U, Ns, Nr)
        U = np.empty(K, dtype=np.ndarray)
        for k in range(K):
            U[k] = aux[k, k].conjugate().T

        SINR_all_users = self.multiH.calc_JP_SINR(F, U, pe=0.0)

        # xxxxxxxxxx Noise Variance of 0.0, Pe of 0.0 xxxxxxxxxxxxxxxxxxxxx
        # k = 0
        B0l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F,
                                                              k=0,
                                                              N0_or_Rek=0.0)
        expected_SINR0 = self.multiH._calc_JP_SINR_k(0, F[0], U[0], B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F,
                                                              k=1,
                                                              N0_or_Rek=0.0)
        expected_SINR1 = self.multiH._calc_JP_SINR_k(1, F[1], U[1], B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 1
        B2l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F,
                                                              k=2,
                                                              N0_or_Rek=0.0)
        expected_SINR2 = self.multiH._calc_JP_SINR_k(2, F[2], U[2], B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Noise Variance of 0.01, Pe of 0.63 xxxxxxxxxxxxxxxxxx
        noise_var = 0.01
        self.multiH.noise_var = noise_var
        Pe = 0.63
        Re = self.multiH.calc_cov_matrix_extint_plus_noise(pe=Pe)

        SINR_all_users = self.multiH.calc_JP_SINR(F, U, pe=Pe)

        # k = 0
        B0l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F,
                                                              k=0,
                                                              N0_or_Rek=Re[0])
        expected_SINR0 = self.multiH._calc_JP_SINR_k(0, F[0], U[0], B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F,
                                                              k=1,
                                                              N0_or_Rek=Re[1])
        expected_SINR1 = self.multiH._calc_JP_SINR_k(1, F[1], U[1], B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 2
        B2l_all_l = self.multiH._calc_JP_Bkl_cov_matrix_all_l(F,
                                                              k=2,
                                                              N0_or_Rek=Re[2])
        expected_SINR2 = self.multiH._calc_JP_SINR_k(2, F[2], U[2], B2l_all_l)
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
        ":type: np.ndarray"

        self.assertAlmostEqual(self.pl.calc_path_loss_dB(1.2),
                               expected_pl_in_dB)
        self.assertAlmostEqual(self.pl.calc_path_loss(1.2),
                               dB2Linear(-expected_pl_in_dB))

        # When we change 'n', this will impact path loss calculation
        n = 2.7
        self.pl.n = n
        expected_pl_in_dB = (
            10 * n * (np.log10(d) + np.log10(fc) + 6.0 - 4.377911390697565))
        ":type: np.ndarray"
        self.assertAlmostEqual(self.pl.calc_path_loss_dB(1.2),
                               expected_pl_in_dB)
        self.assertAlmostEqual(self.pl.calc_path_loss(1.2),
                               dB2Linear(-expected_pl_in_dB))

        # When we change 'fc', this will impact path loss calculation
        fc = 1100
        self.pl.fc = fc
        expected_pl_in_dB = (
            10 * n * (np.log10(d) + np.log10(fc) + 6.0 - 4.377911390697565))
        ":type: np.ndarray"
        self.assertAlmostEqual(self.pl.calc_path_loss_dB(1.2),
                               expected_pl_in_dB)
        self.assertAlmostEqual(self.pl.calc_path_loss(1.2),
                               dB2Linear(-expected_pl_in_dB))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx est for multiple path loss values xxxxxxxxxxxxxxxxxxxx
        d = np.array([1.2, 1.4, 1.6])
        expected_pl_in_dB = (
            10 * n * (np.log10(d) + np.log10(fc) + 6.0 - 4.377911390697565))
        ":type: np.ndarray"

        np.testing.assert_array_almost_equal(
            self.pl.calc_path_loss_dB([1.2, 1.4, 1.6]), expected_pl_in_dB, 16)

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
        ":type: np.ndarray"

        np.testing.assert_array_almost_equal(
            self.pl.calc_path_loss_dB([1.2, 1.4, 1.6]), expected_pl_in_dB, 16)

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

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # test calc_path_loss (linear scale)
        expected_pl_linear = dB2Linear(-expected_pl_in_dB)
        ":type: np.ndarray"

        np.testing.assert_array_almost_equal(
            self.pl.calc_path_loss([1.2, 1.4, 1.6]), expected_pl_linear)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_which_distance(self):
        # Test which_distance and which_distance_dB for a single value.
        self.assertAlmostEqual(self.pl.which_distance(4.88624535312e-10), 1.2)
        self.assertAlmostEqual(self.pl.which_distance_dB(93.1102472958), 1.2)

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
            self.pl.calc_path_loss_dB(np.array([1e-4, 2e-4, 8e-4, 1e-3,
                                                5e-3])),
            np.array([0., 0., 11.65618351, 15.3, 41.58127216]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test for a single path loss value xxxxxxxxxxxxxxxxxxxx
        expected_pl = dB2Linear(-(128.1 + 37.6 * np.log10(1.2)))
        self.assertAlmostEqual(self.pl.calc_path_loss(1.2),
                               expected_pl,
                               places=14)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test for multiple path loss values xxxxxxxxxxxxxxxxxxx
        expected_pl = dB2Linear(
            -(128.1 + 37.6 * np.log10(np.array([1.2, 1.5, 1.8, 2.3]))))
        np.testing.assert_array_almost_equal(self.pl.calc_path_loss(
            np.array([1.2, 1.5, 1.8, 2.3])),
                                             expected_pl,
                                             decimal=16)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_which_distance(self):
        np.testing.assert_array_almost_equal(self.pl.which_distance(
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

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test if an exception is raised if num_walls is negative
        with self.assertRaises(ValueError):
            self.pl._calc_PS7_path_loss_dB_same_floor(d, num_walls=-5)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

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
        ":type: np.ndarray"
        np.testing.assert_array_almost_equal(
            self.pl._calc_PS7_path_loss_dB_same_floor(d, num_walls=num_walls),
            expected_pl_dB_NLOS)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test the LOS Case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        A = 18.7
        B = 46.8
        C = 20
        # X = 0

        # Simple test
        d = 10
        expected_pl_dB_LOS \
            = A * math.log10(d) + B + C * math.log10(fc_GHz / 5)
        self.assertAlmostEqual(
            self.pl._calc_PS7_path_loss_dB_same_floor(d, num_walls=0),
            expected_pl_dB_LOS)

        # Test with a different frequency
        fc_GHz = 1.1
        self.pl.fc = 1.1e3
        expected_pl_dB_LOS \
            = A * math.log10(d) + B + C * math.log10(fc_GHz / 5)
        self.assertAlmostEqual(
            self.pl._calc_PS7_path_loss_dB_same_floor(d, num_walls=0),
            expected_pl_dB_LOS)

        # Test with d as an array
        d = np.array([10., 50., 100., 1000.])
        expected_pl_dB_LOS \
            = A * np.log10(d) + B + C * math.log10(fc_GHz / 5)
        ":type: np.ndarray"
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

        # X = 5 * (num_walls[NLOS_index] - 1)

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
        ":type: np.ndarray"

        # xxxxxxxxxx Test for the 'open' area type xxxxxxxxxxxxxxxxxxxxxxxx
        self.pl.area_type = 'open'
        expected_open_pl = np.array([
            99.1717017731874, 109.775439956383, 115.978229161017,
            120.379178139578, 123.792819371578, 126.581967344212,
            128.940158353991, 130.982916322773, 132.784756548846,
            134.396557554774, 135.854608919885, 137.185705527407,
            138.410195707052, 139.543896537186, 140.599346759408,
            141.586654505968, 142.514087575345, 143.388494732042,
            144.215612946935, 145.000295737969
        ])

        np.testing.assert_array_almost_equal(
            expected_open_pl, self.pl._calc_deterministic_path_loss_dB(d))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test for the suburban area type xxxxxxxxxxxxxxxxxxxxxx
        self.pl.area_type = 'suburban'
        expected_suburban_pl = np.array([
            117.735512612807, 128.339250796002, 134.542040000636,
            138.942988979197, 142.356630211198, 145.145778183831,
            147.50396919361, 149.546727162392, 151.348567388466,
            152.960368394393, 154.418419759504, 155.749516367027,
            156.974006546672, 158.107707376805, 159.163157599027,
            160.150465345588, 161.077898414965, 161.952305571661,
            162.779423786554, 163.564106577588
        ])

        np.testing.assert_array_almost_equal(
            expected_suburban_pl, self.pl._calc_deterministic_path_loss_dB(d))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test for the medium and small city area types xxxxxxxxxxxxx
        self.pl.area_type = 'medium city'
        expected_urban_pl = np.array([
            127.678119861049, 138.281858044244, 144.484647248879,
            148.88559622744, 152.29923745944, 155.088385432074,
            157.446576441852, 159.489334410635, 161.291174636708,
            162.902975642635, 164.361027007746, 165.692123615269,
            166.916613794914, 168.050314625048, 169.10576484727,
            170.09307259383, 171.020505663207, 171.894912819903,
            172.722031034797, 173.506713825831
        ])

        np.testing.assert_array_almost_equal(
            expected_urban_pl, self.pl._calc_deterministic_path_loss_dB(d))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test for the 'large city' area type xxxxxxxxxxxxxxxxxx
        # TODO: The test below is only for frequency 900MHz. You need to
        # test for a lower frequency.
        self.pl.area_type = 'large city'
        expected_large_city_pl = np.array([
            127.72522899, 138.32896717, 144.53175638, 148.93270536,
            152.34634659, 155.13549456, 157.49368557, 159.53644354,
            161.33828377, 162.95008477, 164.40813614, 165.73923275,
            166.96372293, 168.09742376, 169.15287398, 170.14018172,
            171.06761479, 171.94202195, 172.76914017, 173.55382296
        ])
        np.testing.assert_array_almost_equal(
            expected_large_city_pl,
            self.pl._calc_deterministic_path_loss_dB(d))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Antenna Gain Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx1xx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class AntGainOmniTestCase(unittest.TestCase):
    def test_get_antenna_gain(self):
        A = antennagain.AntGainOmni()
        angle1 = 10
        angle2 = -35
        angle3 = 163
        angle4 = -68
        self.assertAlmostEqual(A.get_antenna_gain(angle1), 1.0)
        self.assertAlmostEqual(A.get_antenna_gain(angle2), 1.0)
        self.assertAlmostEqual(A.get_antenna_gain(angle3), 1.0)
        self.assertAlmostEqual(A.get_antenna_gain(angle4), 1.0)

        # Now with numpy arrays
        gains = A.get_antenna_gain(np.array([angle1, angle2, angle3, angle4]))
        self.assertEqual(gains.shape, (4, ))
        np.testing.assert_array_almost_equal(gains, np.ones(4))

        # Repeat the test for an antenna with a gain
        B = antennagain.AntGainOmni(ant_gain=-1)  # gain in dBi
        C = antennagain.AntGainOmni(ant_gain=3)  # gain in dBi

        self.assertAlmostEqual(B.get_antenna_gain(angle1), dB2Linear(-1.0))
        self.assertAlmostEqual(B.get_antenna_gain(angle2), dB2Linear(-1.0))
        self.assertAlmostEqual(B.get_antenna_gain(angle3), dB2Linear(-1.0))
        self.assertAlmostEqual(B.get_antenna_gain(angle4), dB2Linear(-1.0))

        self.assertAlmostEqual(C.get_antenna_gain(angle1), dB2Linear(3.0))
        self.assertAlmostEqual(C.get_antenna_gain(angle2), dB2Linear(3.0))
        self.assertAlmostEqual(C.get_antenna_gain(angle3), dB2Linear(3.0))
        self.assertAlmostEqual(C.get_antenna_gain(angle4), dB2Linear(3.0))


class AntGain3GPP25996TestCase(unittest.TestCase):
    # noinspection PyPep8
    def test_get_antenna_gain(self):
        # \(-\min\left[ 12\left( \frac{\theta}{\theta_{3dB}} \right)^2, A_m \right]\), where \(-180 \geq \theta \geq 180\)
        # xxxxxxxxxx Test for 3-Sector cells xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        A = antennagain.AntGainBS3GPP25996()
        antenna_gain = A.ant_gain
        angle1 = 10
        angle2 = -35
        angle3 = 163
        angle4 = -68
        angle5 = -90
        angle6 = -95
        expected_gain1 = antenna_gain * dB2Linear(-12 * (angle1 / 70.)**2)
        expected_gain2 = antenna_gain * dB2Linear(-12 * (angle2 / 70.)**2)
        expected_gain3 = antenna_gain * dB2Linear(-20)
        expected_gain4 = antenna_gain * dB2Linear(-12 * (angle4 / 70.)**2)
        expected_gain5 = antenna_gain * dB2Linear(-12 * (angle5 / 70.)**2)
        expected_gain6 = antenna_gain * dB2Linear(-20)
        self.assertAlmostEqual(A.get_antenna_gain(angle1), expected_gain1)
        self.assertAlmostEqual(A.get_antenna_gain(angle2), expected_gain2)
        self.assertAlmostEqual(A.get_antenna_gain(angle3), expected_gain3)
        self.assertAlmostEqual(A.get_antenna_gain(angle4), expected_gain4)
        self.assertAlmostEqual(A.get_antenna_gain(angle5), expected_gain5)
        self.assertAlmostEqual(A.get_antenna_gain(angle6), expected_gain6)

        # Now with numpy arrays
        gains = A.get_antenna_gain(
            np.array([angle1, angle2, angle3, angle4, angle5, angle6]))
        self.assertEqual(gains.shape, (6, ))
        np.testing.assert_array_almost_equal(
            gains,
            np.array([
                expected_gain1, expected_gain2, expected_gain3, expected_gain4,
                expected_gain5, expected_gain6
            ]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test for 6-sector cells xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        B = antennagain.AntGainBS3GPP25996(number_of_sectors=6)
        antenna_gain = B.ant_gain
        expected_gain1 = antenna_gain * dB2Linear(-12 * (angle1 / 35.)**2)
        expected_gain2 = antenna_gain * dB2Linear(-12 * (angle2 / 35.)**2)
        expected_gain3 = antenna_gain * dB2Linear(-23)
        expected_gain4 = antenna_gain * dB2Linear(-23)
        expected_gain5 = antenna_gain * dB2Linear(-23)
        expected_gain6 = antenna_gain * dB2Linear(-23)
        self.assertAlmostEqual(B.get_antenna_gain(angle1), expected_gain1)
        self.assertAlmostEqual(B.get_antenna_gain(angle2), expected_gain2)
        self.assertAlmostEqual(B.get_antenna_gain(angle3), expected_gain3)
        self.assertAlmostEqual(B.get_antenna_gain(angle4), expected_gain4)
        self.assertAlmostEqual(B.get_antenna_gain(angle5), expected_gain5)
        self.assertAlmostEqual(B.get_antenna_gain(angle6), expected_gain6)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Different number of sectors raise an exception xxxxxxxxxxxx
        with self.assertRaises(ValueError):
            antennagain.AntGainBS3GPP25996(number_of_sectors=9)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    # plot_psd_OFDM_symbols()
    unittest.main()
