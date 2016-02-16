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
except NameError:               # pragma: no cover
    sys.path.append('../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import warnings
import unittest
import doctest
import numpy as np
import math
from pyphysim import mimo
from pyphysim.mimo.mimo import Blast, Alamouti, MRC, MRT, SVDMimo, GMDMimo, \
    calc_post_processing_SINRs
from pyphysim.util.misc import randn_c, gmd
from pyphysim.util.conversion import linear2dB


# UPDATE THIS CLASS if another module is added to the comm package
class MimoDoctestsTestCase(unittest.TestCase):
    """
    Test case that run all the doctests in the modules of the mimo module.
    """
    def test_mimo(self, ):
        """Run doctests in the mimo module."""
        doctest.testmod(mimo)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MIMO Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Function defined here for test purposes
def calc_Bl(channel, W, l, noise_var=0.0):
    """
    Calculate the Bl matrix corresponding to the interference plus noise
    covariance matrix for the l-th stream.

    \\[B_l = \\sum_{d=1, d\\neq l}^{d_i} (\\mtH \\mtV^{[\\star d]} \\mtV^{[\\star d] \\mtH} \\mtH^H - \\mtH \\mtV^{[\\star l]} \\mtV^{[\\star l] \\mtH} \\mtH^H + \\sigma_n^2 \\mtI_{N_r} )\\]

    Parameters
    ----------
    channel : np.ndarray
        The single user MIMO channel matrix (2D numpy array).
    W : np.ndarray
        The user precoder (2D numpy array).
    l : int
        The index of the desired stream.
    noise_var : float
        The noise variance.

    Returns
    -------
    Bl : np.ndarray
        The interference plus noise covariance matrix (2D numpy array) for
        stream 'l'.
    """
    _, Ns = W.shape
    Nr, _ = channel.shape
    N = np.eye(Nr) * noise_var
    # Idex of all streams, except 'l'
    idx = np.array([i for i in range(Ns) if i != l])

    if Ns > 1:
        Bl = channel.dot(
            W[:, idx].dot(W[:, idx].conj().T)).dot(channel.conj().T) + N
    else:
        # If there is only one stream then we don't have interference from
        # other streams and the interference plus noise covariance matrix
        # Bl will be only the covariance matrix of the noise
        Bl = N
    return Bl


# Function defined here for test purposes
def calc_SINRs(channel, W, G_H, noise_var):
    """
    Calculate the post processing SINRs of all streams.

    Parameters
    ----------
    channel : np.ndarray
        The MIMO channel (2D numpy array).
    W : np.ndarray
        The precoder for the MIMO scheme (2D numpy array).
    G_H : np.ndarray
        The receive filter for the MIMO scheme (2D numpy array).
    noise_var : float
        The noise variance

    Returns
    -------
    sinrs : np.ndarray
        The SINR (in linear scale). of all streams (1D numpy array).
    """
    _, Ns = W.shape
    sinrs = np.empty(Ns, dtype=float)
    for l in range(Ns):
        Bl = calc_Bl(channel, W, l, noise_var)
        num = np.linalg.norm((G_H[l].dot(channel).dot(W[:, l])))**2
        den = np.abs(G_H[l].dot(Bl).dot(G_H[l].conj()))
        sinrs[l] = num/den

    return sinrs


class BlastTestCase(unittest.TestCase):
    """Unittests for the Blast class in the mimo module.
    """
    def setUp(self):
        """Called before each test."""
        self.blast_object = Blast()

    def test_set_channel_matrix(self):
        # Test if a warning is raised when the number of transmit antennas
        # is greater then the number of receive antennas
        #
        # For that we capture the warnings ...
        with warnings.catch_warnings(record=True) as w:
            # then we call the set_channel_matrix method
            self.blast_object.set_channel_matrix(np.random.randn(3, 4))
            # and we test if captured 1 warning.
            self.assertEqual(len(w), 1, msg='Warning was not raised')

    def test_getNumberOfLayers(self):
        channel = np.eye(3)
        self.blast_object.set_channel_matrix(channel)

        self.assertEqual(self.blast_object.getNumberOfLayers(), 3)

        channel2 = np.eye(5)
        blast2 = Blast(channel2)
        self.assertEqual(blast2.getNumberOfLayers(), 5)

    def test_encode(self):
        channel = np.eye(3)
        # Set the channel so that the getNumberOfLayers method works
        self.blast_object.set_channel_matrix(channel)

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

    def test_set_noise_var(self):
        self.blast_object.set_noise_var(0.001)
        self.assertAlmostEqual(self.blast_object._noise_var, 0.001)

        self.blast_object.set_noise_var(None)
        self.assertAlmostEqual(self.blast_object._noise_var, 0.0)

        with self.assertRaises(ValueError):
            self.blast_object.set_noise_var(-0.001)

    def test_decode(self):
        data = np.r_[0:15]

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test with an identity channel
        channel = np.eye(3)
        self.blast_object.set_channel_matrix(channel)
        encoded_data = self.blast_object.encode(data)
        decoded_data1 = self.blast_object.decode(encoded_data)
        np.testing.assert_array_almost_equal(decoded_data1, data)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test with a random channel and a zero-force filter
        self.blast_object.set_noise_var(None)  # This should use the ZF filter
        channel = randn_c(4, 3)  # 3 transmitt antennas and 4 receive antennas
        self.blast_object.set_channel_matrix(channel)
        received_data2 = np.dot(channel, encoded_data)
        decoded_data2 = self.blast_object.decode(received_data2)
        np.testing.assert_array_almost_equal(decoded_data2, data)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test with a random channel and a MMSE filter
        self.blast_object.set_noise_var(0.00000001)
        channel = randn_c(4, 3)  # 3 transmitt antennas and 4 receive antennas
        self.blast_object.set_channel_matrix(channel)
        received_data3 = np.dot(channel, encoded_data)
        decoded_data3 = self.blast_object.decode(received_data3)
        np.testing.assert_array_almost_equal(decoded_data3.round(7), data)

    def test_calc_post_processing_SINRs(self):
        Nr = 4
        Nt = 3
        noise_var = 0.001
        channel = randn_c(Nr, Nt)
        self.blast_object.set_noise_var(noise_var)
        self.blast_object.set_channel_matrix(channel)

        W = self.blast_object._calc_precoder(channel)
        G_H = self.blast_object._calc_receive_filter(channel, noise_var)
        expected_sinrs = linear2dB(calc_SINRs(channel, W, G_H, noise_var))

        # Calculate the SINR using the function in the mimo module. Note
        # that we need to pass the channel, the precoder, the receive
        # filter and the noise variance.
        sinrs = calc_post_processing_SINRs(channel, W, G_H, noise_var)
        np.testing.assert_array_almost_equal(sinrs, expected_sinrs, 2)

        # Calculate the SINR using method in the MIMO class. Note that we
        # only need to pass the noise variance, since the mimo object knows
        # the channel and it can calculate the precoder and receive filter.
        sinrs_other = self.blast_object.calc_linear_SINRs(noise_var)
        np.testing.assert_array_almost_equal(sinrs_other, expected_sinrs, 2)


class MRTTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.mrt_object = MRT()

    def test_init(self):
        channel1 = randn_c(3)
        mrt_object1 = MRT(channel1)
        self.assertEqual(3, mrt_object1.Nt)
        self.assertEqual(1, mrt_object1.Nr)

        channel2 = randn_c(1, 3)
        mrt_object2 = MRT(channel2)
        self.assertEqual(3, mrt_object2.Nt)
        self.assertEqual(1, mrt_object2.Nr)

        channel3 = randn_c(2, 3)
        # Number of receive antennas must be exact 1. Since channel3 has 2
        # receive antennas, an exception should be raised
        with self.assertRaises(ValueError):
            MRT(channel3)

    def test_getNumberOfLayers(self):
        self.assertEqual(self.mrt_object.getNumberOfLayers(), 1)

    def test_encode(self):
        data = np.r_[0:15]

        # xxxxxxxxxx test the case with Ntx=2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Nt = 2
        channel = randn_c(Nt)
        self.mrt_object.set_channel_matrix(channel)

        data_aux = data.reshape(1, data.size)  # Useful for broadcasting
        ":type: np.ndarray"
        W = np.exp(-1j * np.angle(channel)).reshape(Nt, 1) / math.sqrt(Nt)
        ":type: np.ndarray"

        encoded_data = self.mrt_object.encode(data)

        expected_encoded_data = W * data_aux
        np.testing.assert_array_almost_equal(expected_encoded_data,
                                             encoded_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx test the case with Ntx=4 xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Nt = 4
        channel = randn_c(Nt)
        self.mrt_object.set_channel_matrix(channel)

        data_aux = data.reshape(1, data.size)  # Useful for broadcasting
        ":type: np.ndarray"

        encoded_data = self.mrt_object.encode(data)
        W = np.exp(-1j * np.angle(channel)).reshape(Nt, 1)
        ":type: np.ndarray"

        expected_encoded_data = (1. / math.sqrt(Nt)) * W * data_aux
        np.testing.assert_array_almost_equal(expected_encoded_data,
                                             encoded_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test the case where the channel is 2D xxxxxxxxxxxxxxxx
        # Note tha in this case even though the channel is a 2D numpy array
        # the size of the first dimension (receive antennas) must be equal
        # to 1.
        Nt = 4
        channel2 = randn_c(1, Nt)
        self.mrt_object.set_channel_matrix(channel2)

        data_aux = data.reshape(1, data.size)  # Useful for broadcasting
        ":type: np.ndarray"

        encoded_data = self.mrt_object.encode(data)
        W = np.exp(-1j * np.angle(channel2)).reshape(Nt, 1)
        ":type: np.ndarray"

        expected_encoded_data = (1. / math.sqrt(Nt)) * W * data_aux
        np.testing.assert_array_almost_equal(expected_encoded_data,
                                             encoded_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_decode(self):
        Nt = 4

        # xxxxxxxxxx test the case with a single receive antenna xxxxxxxxxx
        channel = randn_c(Nt)
        self.mrt_object.set_channel_matrix(channel)

        data = np.r_[0:15]
        encoded_data = self.mrt_object.encode(data)

        # Add '0.1' as a noise term
        received_data = channel.dot(encoded_data) + 0.0001
        ":type: np.ndarray"

        decoded_data = self.mrt_object.decode(received_data)

        self.assertEqual(len(decoded_data.shape), 1)
        np.testing.assert_array_almost_equal(decoded_data, data, decimal=4)

        # Now we are explicitting changing the shape of the channel
        # variable to include the first dimension corresponding to a single
        # receive antenna
        channel.shape = (1, Nt)
        self.mrt_object.set_channel_matrix(channel)
        # The encoded data should be the same
        encoded_data = self.mrt_object.encode(data)

        # Add '0.1' as a noise term
        received_data = channel.dot(encoded_data) + 0.0001
        ":type: np.ndarray"

        decoded_data = self.mrt_object.decode(received_data)

        self.assertEqual(len(decoded_data.shape), 1)
        np.testing.assert_array_almost_equal(decoded_data, data, decimal=4)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_post_processing_SINRs(self):
        Nr = 1
        Nt = 3
        noise_var = 0.001
        channel = randn_c(Nr, Nt)
        self.mrt_object.set_channel_matrix(channel)

        W = self.mrt_object._calc_precoder(channel)
        G_H = self.mrt_object._calc_receive_filter(channel, noise_var)
        G_H = np.array([[G_H]])
        expected_sinrs = linear2dB(calc_SINRs(channel, W, G_H, noise_var))

        # Calculate the SINR using the function in the mimo module. Note
        # that we need to pass the channel, the precoder, the receive
        # filter and the noise variance.
        sinrs = calc_post_processing_SINRs(channel, W, G_H, noise_var)
        np.testing.assert_array_almost_equal(sinrs, expected_sinrs, 2)

        # Calculate the SINR using method in the MIMO class. Note that we
        # only need to pass the noise variance, since the mimo object knows
        # the channel and it can calculate the precoder and receive filter.
        sinrs_other = self.mrt_object.calc_linear_SINRs(noise_var)
        np.testing.assert_array_almost_equal(sinrs_other, expected_sinrs, 2)


class MRCTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.mrc_object = MRC()

    def test_decode(self):
        data = np.r_[0:15]
        num_streams = 3

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test with an identity channel
        channel = np.eye(num_streams)
        self.mrc_object.set_channel_matrix(channel)
        encoded_data = self.mrc_object.encode(data)
        decoded_data1 = self.mrc_object.decode(encoded_data)
        np.testing.assert_array_almost_equal(decoded_data1, data)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test with a random channel and a zero-force filter
        self.mrc_object.set_noise_var(None)  # This should use the ZF filter
        channel = randn_c(4, num_streams)  # 4 receive antennas
        self.mrc_object.set_channel_matrix(channel)
        received_data2 = np.dot(channel, encoded_data)
        decoded_data2 = self.mrc_object.decode(received_data2)
        np.testing.assert_array_almost_equal(decoded_data2, data)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test with a random channel and a MMSE filter
        self.mrc_object.set_noise_var(0.00000001)
        channel = randn_c(4, num_streams)  # 4 receive antennas
        self.mrc_object.set_channel_matrix(channel)
        received_data3 = np.dot(channel, encoded_data)
        decoded_data3 = self.mrc_object.decode(received_data3)
        np.testing.assert_array_almost_equal(decoded_data3.round(7), data)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # test with a single stream
        self.mrc_object.set_noise_var(None)  # This should use the ZF filter
        channel = randn_c(4)  # 4 receive antennas
        self.mrc_object.set_channel_matrix(channel)
        encoded_data2 = self.mrc_object.encode(data)
        received_data4 = np.dot(channel[:,np.newaxis], encoded_data2)
        decoded_data4 = self.mrc_object.decode(received_data4)
        np.testing.assert_array_almost_equal(decoded_data4, data)

    def test_calc_post_processing_SINRs(self):
        Nr = 3
        Nt = 1
        noise_var = 0.001
        channel = randn_c(Nr, Nt)
        self.mrc_object.set_channel_matrix(channel)

        W = self.mrc_object._calc_precoder(channel)
        G_H = self.mrc_object._calc_receive_filter(channel, noise_var)
        expected_sinrs = linear2dB(calc_SINRs(channel, W, G_H, noise_var))

        # Calculate the SINR using the function in the mimo module. Note
        # that we need to pass the channel, the precoder, the receive
        # filter and the noise variance.
        sinrs = calc_post_processing_SINRs(channel, W, G_H, noise_var)
        np.testing.assert_array_almost_equal(sinrs, expected_sinrs, 2)

        # Calculate the SINR using method in the MIMO class. Note that we
        # only need to pass the noise variance, since the mimo object knows
        # the channel and it can calculate the precoder and receive filter.
        sinrs_other = self.mrc_object.calc_linear_SINRs(noise_var)
        np.testing.assert_array_almost_equal(sinrs_other, expected_sinrs, 2)


class SVDMimoTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.svdmimo_object = SVDMimo()

    def test_encode(self):
        # xxxxxxxxxx test the case with Ntx=2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Nt = 2
        Nr = 2
        data = np.r_[0:15*Nt]
        data_aux = data.reshape(Nt, -1)
        channel = randn_c(Nr, Nt)
        self.svdmimo_object.set_channel_matrix(channel)

        encoded_data = self.svdmimo_object.encode(data)

        _, _, V_H = np.linalg.svd(channel)
        W = V_H.conj().T / math.sqrt(Nt)
        ":type: np.ndarray"

        expected_encoded_data = W.dot(data_aux)
        np.testing.assert_array_almost_equal(
            expected_encoded_data, encoded_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx test the case with Ntx=4 xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Nt = 4
        Nr = 4
        data = np.r_[0:15*Nt]
        data_aux = data.reshape(Nt, -1)
        channel = randn_c(Nr, Nt)
        self.svdmimo_object.set_channel_matrix(channel)

        encoded_data = self.svdmimo_object.encode(data)

        _, _, V_H = np.linalg.svd(channel)
        W = V_H.conj().T / math.sqrt(Nt)
        ":type: np.ndarray"

        expected_encoded_data = W.dot(data_aux)
        np.testing.assert_array_almost_equal(
            expected_encoded_data, encoded_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test if an exception is raised for wrong size xxxxxxxxxxxxx
        # The exception is raised if the input array size is not a multiple
        # of the number of transmit antennas
        data2 = np.r_[0:15*Nt+1]
        with self.assertRaises(ValueError):
            self.svdmimo_object.encode(data2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_decode(self):
        # xxxxxxxxxx test the case with Ntx=2, NRx=2 xxxxxxxxxxxxxxxxxxxxxx
        Nt = 2
        Nr = 2
        data = np.r_[0:15*Nt]
        channel = randn_c(Nr, Nt)
        self.svdmimo_object.set_channel_matrix(channel)

        encoded_data = self.svdmimo_object.encode(data)
        received_data = channel.dot(encoded_data)
        decoded_data = self.svdmimo_object.decode(received_data)
        np.testing.assert_array_almost_equal(data, decoded_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_post_processing_SINRs(self):
        Nr = 3
        Nt = 3
        noise_var = 0.01
        channel = randn_c(Nr, Nt)
        self.svdmimo_object.set_channel_matrix(channel)

        W = self.svdmimo_object._calc_precoder(channel)
        G_H = self.svdmimo_object._calc_receive_filter(channel, noise_var)
        expected_sinrs = linear2dB(calc_SINRs(channel, W, G_H, noise_var))

        # Calculate the SINR using the function in the mimo module. Note
        # that we need to pass the channel, the precoder, the receive
        # filter and the noise variance.
        sinrs = calc_post_processing_SINRs(channel, W, G_H, noise_var)
        np.testing.assert_array_almost_equal(sinrs, expected_sinrs, 2)

        # Calculate the SINR using method in the MIMO class. Note that we
        # only need to pass the noise variance, since the mimo object knows
        # the channel and it can calculate the precoder and receive filter.
        sinrs_other = self.svdmimo_object.calc_linear_SINRs(noise_var)
        np.testing.assert_array_almost_equal(sinrs_other, expected_sinrs, 2)


class GMDMimoTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.gmdmimo_object = GMDMimo()

    def test_encode(self):
        # xxxxxxxxxx test the case with Ntx=2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Nt = 2
        Nr = 2
        data = np.r_[0:15*Nt]

        channel = randn_c(Nr, Nt)
        self.gmdmimo_object.set_channel_matrix(channel)

        encoded_data = self.gmdmimo_object.encode(data)

        # data_aux = data.reshape(Nt, -1)
        U, S, V_H = np.linalg.svd(channel)
        _, _, P = gmd(U, S, V_H)
        W = P / math.sqrt(Nt)
        ":type: np.ndarray"

        expected_encoded_data = W.dot(data.reshape(Nr, -1))
        np.testing.assert_array_almost_equal(
            expected_encoded_data, encoded_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test if an exception is raised for wrong size xxxxxxxxxxxxx
        # The exception is raised if the input array size is not a multiple
        # of the number of transmit antennas
        data2 = np.r_[0:15*Nt+1]
        with self.assertRaises(ValueError):
            self.gmdmimo_object.encode(data2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_decode(self):
        # xxxxxxxxxx test the case with Ntx=2, NRx=2 xxxxxxxxxxxxxxxxxxxxxx
        Nt = 2
        Nr = 2
        data = np.r_[0:15*Nt]
        channel = randn_c(Nr, Nt)
        self.gmdmimo_object.set_channel_matrix(channel)

        encoded_data = self.gmdmimo_object.encode(data)
        received_data = channel.dot(encoded_data)

        decoded_data = self.gmdmimo_object.decode(received_data)
        np.testing.assert_array_almost_equal(data, decoded_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_post_processing_SINRs(self):
        Nr = 3
        Nt = 3
        noise_var = 0.01
        channel = randn_c(Nr, Nt)
        self.gmdmimo_object.set_noise_var(noise_var)
        self.gmdmimo_object.set_channel_matrix(channel)

        W = self.gmdmimo_object._calc_precoder(channel)
        G_H = self.gmdmimo_object._calc_receive_filter(channel, noise_var)
        expected_sinrs = linear2dB(calc_SINRs(channel, W, G_H, noise_var))

        # Calculate the SINR using the function in the mimo module. Note
        # that we need to pass the channel, the precoder, the receive
        # filter and the noise variance.
        sinrs = calc_post_processing_SINRs(channel, W, G_H, noise_var)
        np.testing.assert_array_almost_equal(sinrs, expected_sinrs, 1)

        # Calculate the SINR using method in the MIMO class. Note that we
        # only need to pass the noise variance, since the mimo object knows
        # the channel and it can calculate the precoder and receive filter.
        sinrs_other = self.gmdmimo_object.calc_linear_SINRs(noise_var)
        np.testing.assert_array_almost_equal(sinrs_other, expected_sinrs, 1)


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

    def test_set_channel_matrix(self):
        self.alamouti_object.set_channel_matrix(randn_c(2))
        self.assertEqual(self.alamouti_object.Nt, 2)
        self.assertEqual(self.alamouti_object.Nr, 1)

        self.alamouti_object.set_channel_matrix(randn_c(4, 2))
        self.assertEqual(self.alamouti_object.Nt, 2)
        self.assertEqual(self.alamouti_object.Nr, 4)

        with self.assertRaises(ValueError):
            self.alamouti_object.set_channel_matrix(randn_c(4, 3))

    def test_encode(self):
        data = np.r_[0:16] + np.r_[0:16] * 1j
        ":type: np.ndarray"

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
        ":type: np.ndarray"

        encoded_data = self.alamouti_object.encode(data)
        # We will test the deconding with a random channel
        channel = randn_c(3, 2)
        self.alamouti_object.set_channel_matrix(channel)
        received_data = np.dot(channel, encoded_data)
        decoded_data = self.alamouti_object.decode(received_data)
        np.testing.assert_array_almost_equal(decoded_data, data)

    def test_calc_post_processing_SINRs(self):
        Nr = 1
        Nt = 2
        noise_var = 0.01
        channel = randn_c(Nr, Nt)
        self.alamouti_object.set_channel_matrix(channel)

        # W = self.alamouti_object._calc_precoder(channel)
        # G_H = self.alamouti_object._calc_receive_filter(channel, noise_var)

        expected_sinrs = linear2dB(
            (np.linalg.norm(channel, 'fro')**2)/noise_var)

        # Calculate the SINR using method in the Alamouti class. Note that
        # we only need to pass the noise variance, since the mimo object
        # knows the channel.
        sinrs = self.alamouti_object.calc_SINRs(noise_var)
        np.testing.assert_array_almost_equal(sinrs, expected_sinrs, 2)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx



# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    # plot_psd_OFDM_symbols()
    unittest.main()
