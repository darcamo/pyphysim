#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1101
"""
Tests for the modules in the ia package.

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

try:
    import cPickle as pickle
except ImportError as e:  # pragma: no cover
    import pickle  # type: ignore

import unittest
import doctest
import numpy as np
from numpy.linalg import norm

from pyphysim import channels
import pyphysim.ia  # Import the package ia
from pyphysim.ia.algorithms import AlternatingMinIASolver, IASolverBaseClass, \
    MaxSinrIASolver, MinLeakageIASolver, ClosedFormIASolver, MMSEIASolver, \
    IterativeIASolverBaseClass, GreedStreamIASolver, BruteForceStreamIASolver
from pyphysim.util.misc import peig, leig, randn_c
from pyphysim.util.conversion import linear2dB


class CustomTestCase(unittest.TestCase):
    """
    This class inherits from unittest.TestCase and implements the
    `_save_state` and `_maybe_load_state_and_randomize_channel` methods
    that can be used in the IA Solver test case classes.

    Notes
    -----

    In the implementation of the setUp method of a derived class you must
    set the values of the variables `_new_test`, `iasolver_state`,
    `channel_state`, `noise_state` and `iasolver`.
    """
    def __init__(self, methodName='runTest'):
        """Init method.
        """
        unittest.TestCase.__init__(self, methodName)

        self._new_test = None  # Don't change this

        # Variable used to save the RandomState object for the IA
        # solver. Don't change this.
        self.iasolver_state = None
        # Variable used to save the RandomState object for the channel
        # samples generation. Don't change this.
        self.channel_state = None
        # Variable used to save the RandomState object for the noise
        # generation. Don't change this.
        self.noise_state = None

    def _save_state(self, filename):  # pragma: no cover
        """
        When a test fails, call this method to save the state of the channel
        and IA solver random generators so that you can reproduce this fail
        again.

        Note that for this method to work, you need to have called the
        `_maybe_load_state_and_randomize_channel` at the beginning of the
        test method where `_save_state` will be called.

        Parameters
        ----------
        filename : str
            Name of the file where the state will be saved.
        """
        if self._new_test is True:
            MMSE_test_solve_state = {
                'iasolver_state': self.iasolver_state,
                'channel_state': self.channel_state,
                'noise_state': self.noise_state
            }
            with open(filename, 'wb') as fid:
                pickle.dump(MMSE_test_solve_state, fid,
                            pickle.HIGHEST_PROTOCOL)

    def _maybe_load_state_and_randomize_channel(  # pragma: no cover
        self,
        filename,
        iasolver=None,
        Nr=None,
        Nt=None,
        K=None):
        """
        Load the state of a previous test fail, if the saved file exists.

        The state includes the state of the random generators for the
        iasolver precoders, for the channel samples, and for the channel
        noise.

        Parameters
        ----------
        filename : str
            The name of the file to load the state.
        iasolver : IA Solver object
            The IA solver whose state will be saved (the state of the
            MultiUserChannelMatrix associated with that IA solver object
            will also be saved).
        Nr : int of numpy array
            Number of receive antennas of all/each user.
        Nt : int of numpy array
            Number of receive antennas of all/each user.
        K : int
            Number of users
        """
        multiUserChannel = iasolver._multiUserChannel
        self._new_test = None

        try:  # pragma: nocover
            # If the file pointed by `filename` exists, that means that a
            # previous run of the test method failed and the random states
            # were saved. In that case we will load those random states and
            # 'randomize' the channel and precoders with them to reproduce
            # the previous failed test.
            with open(filename, 'r') as fid:
                MMSE_test_solve_state = pickle.load(fid)
            iasolver_state = MMSE_test_solve_state['iasolver_state']
            channel_state = MMSE_test_solve_state['channel_state']
            noise_state = MMSE_test_solve_state['noise_state']

            iasolver._rs.set_state(iasolver_state)
            multiUserChannel._RS_channel.set_state(channel_state)
            multiUserChannel._RS_noise.set_state(noise_state)

            self._new_test = False

        except IOError:
            # State of the RandomState objects used to get the random
            # precoder, random channel and random noise samples. Since we
            # could not load the states from the file, that means that the
            # file does not exist. In that case, let's store the state
            # objects into these three member variables so that the
            # `_save_state` method can be latter called to save these
            # states IF the test fails.
            self.iasolver_state = iasolver._rs.get_state()
            self.channel_state = multiUserChannel._RS_channel.get_state()
            self.noise_state = multiUserChannel._RS_noise.get_state()

            self._new_test = True
        finally:
            multiUserChannel.randomize(Nr, Nt, K)


# UPDATE THIS CLASS if another module is added to the ia package
# noinspection PyMethodMayBeStatic
class IaDoctestsTestCase(unittest.TestCase):
    """Test case that run all the doctests in the modules of the ia package. """
    def test_ia(self):
        """Run doctests in the ia module."""
        doctest.testmod(pyphysim.ia)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx IA Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Since IASolverBaseClass is an abstract class, lets define a concrete
# class so that we can test it.
class IASolverBaseClassConcret(IASolverBaseClass):
    """
    Concrete class derived from IASolverBaseClass for testing purposes.
    """
    def solve(self, Ns, P=None):
        pass  # pragma: nocover


# Since IterativeIASolverBaseClass is an abstract class, lets define a
# concrete class so that we can test it.
class IterativeIASolverBaseClassConcrete(IterativeIASolverBaseClass):
    """
    Concrete class derived from IterativeIASolverBaseClass for testing
    purposes.
    """
    def _updateW(self):
        pass

    def _updateF(self):
        pass  # pragma: nocover


class IASolverBaseClassTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        self.iasolver = IASolverBaseClassConcret(multiUserChannel)

    def test_init(self):
        # Try to initialize the IASolverBaseClass object with some
        # parameter which is not a MultiUserChannelMatrix object
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            IASolverBaseClassConcret(3)

    def test_get_cost(self):
        self.assertEqual(self.iasolver.get_cost(), -1)

    def test_properties(self):
        K = 3
        Nr = np.array([2, 4, 6])
        Nt = np.array([2, 3, 5])
        Ns = np.array([1, 2, 3])
        multiUserChannel = self.iasolver._multiUserChannel
        multiUserChannel.randomize(Nr, Nt, K)
        # Setting P here will be tested in test_randomizeF
        self.iasolver.randomizeF(Ns, P=None)

        # xxxxx Test the properties Nr, Nt and Ns xxxxxxxxxxxxxxxxxxxxxxxxx
        self.assertEqual(self.iasolver.K, K)
        np.testing.assert_array_equal(self.iasolver.Nr, Nr)
        np.testing.assert_array_equal(self.iasolver.Nt, Nt)
        np.testing.assert_array_equal(self.iasolver.Ns, Ns)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test getting and setting the P (power) property xxxxxxxxxxx
        np.testing.assert_array_almost_equal(self.iasolver.P,
                                             np.ones(K, dtype=float))
        self.iasolver.P = 1.5
        np.testing.assert_array_almost_equal(self.iasolver.P, [1.5, 1.5, 1.5])
        self.iasolver.P = [1.3, 1.2, 1.8]
        np.testing.assert_array_almost_equal(self.iasolver.P,
                                             np.array([1.3, 1.2, 1.8]))

        # If we try to set P with a sequence of wrong length (different
        # from the number of users) an exception should be raised.
        with self.assertRaises(ValueError):
            self.iasolver.P = [1.2, 2.1]

        with self.assertRaises(ValueError):
            self.iasolver.P = -1.5

        with self.assertRaises(ValueError):
            self.iasolver.P = 0.0

        with self.assertRaises(ValueError):
            self.iasolver.P = [1.2, -1.3, 1.1]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test the noise_var property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # The noise_var property in the IA solver object basically returns
        # the noise_var of the channel object (or zero if it is not zet in
        # the channel object).
        self.assertIsNone(multiUserChannel.noise_var)

        # If we try to get the value of the noise_var property it will
        # return the value of the noise_var property of the
        # multiUserChannel object
        self.assertAlmostEqual(self.iasolver.noise_var, 0.0)

        # If we change the noise_var property int the channel object it
        # will change in the IA solver object.
        multiUserChannel.noise_var = 1.3
        self.assertEqual(self.iasolver.noise_var, 1.3)

        multiUserChannel.noise_var = 1.5
        self.assertEqual(self.iasolver.noise_var, 1.5)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_randomizeF(self):
        K = 3
        Nt = np.array([2, 3, 5])
        Ns = np.array([1, 2, 3])
        P = np.array([1.2, 0.9, 1.4])  # Power of each user
        multiUserChannel = self.iasolver._multiUserChannel

        multiUserChannel.randomize(5 * np.ones(K, dtype=int), Nt, K)

        self.iasolver.randomizeF(Ns, P)

        # The Frobenius norm of the _F variable must be equal to one
        self.assertAlmostEqual(norm(self.iasolver._F[0], 'fro'), 1.0)
        self.assertAlmostEqual(norm(self.iasolver._F[1], 'fro'), 1.0)
        self.assertAlmostEqual(norm(self.iasolver._F[2], 'fro'), 1.0)

        # The square of the Frobenius norm of the property bust be equal to
        # the power
        self.assertAlmostEqual(norm(self.iasolver.full_F[0], 'fro')**2, P[0])
        self.assertAlmostEqual(norm(self.iasolver.full_F[1], 'fro')**2, P[1])
        self.assertAlmostEqual(norm(self.iasolver.full_F[2], 'fro')**2, P[2])

        # The shape of the precoder is the number of users
        self.assertEqual(self.iasolver._F.shape, (K, ))

        # The power of each user
        np.testing.assert_array_almost_equal(self.iasolver.P, P)

        # The shape of the precoder of each user is Nt[user] x Ns[user]
        self.assertEqual(self.iasolver._F[0].shape, (Nt[0], Ns[0]))
        self.assertEqual(self.iasolver._F[1].shape, (Nt[1], Ns[1]))
        self.assertEqual(self.iasolver._F[2].shape, (Nt[2], Ns[2]))

        # Test when the number of streams is an scalar (the same value will
        # be used for all users)
        Ns = 2
        self.iasolver.randomizeF(Ns)

        # The shape of the precoder of each user is Nt[user] x Ns[user]
        self.assertEqual(self.iasolver._F[0].shape, (Nt[0], Ns))
        self.assertEqual(self.iasolver._F[1].shape, (Nt[1], Ns))
        self.assertEqual(self.iasolver._F[2].shape, (Nt[2], Ns))

        # Test if the power is None (which means "use 1" whenever needed),
        # since it was not set. Note that self.iasolver.P (the property) is
        # an array of ones with the length equal to the number of
        # users. However, the number of users is taken from the multiuser
        # channel but since it has not been initialized in this updateF
        # test method self.iasolver.P is a zero length array.
        self.assertIsNone(self.iasolver._P)

    def test_F_and_full_F(self):
        self.iasolver._multiUserChannel.randomize(4, 4, 3)
        self.iasolver.randomizeF(1)
        self.assertIsNotNone(self.iasolver._F)
        self.assertIsNone(self.iasolver._full_F)

        # Since the power was not set yet then full_F should have the same
        # value as F
        self.assertIsNone(self.iasolver._P)
        np.testing.assert_almost_equal(self.iasolver.F[0],
                                       self.iasolver.full_F[0])
        np.testing.assert_almost_equal(self.iasolver.F[1],
                                       self.iasolver.full_F[1])
        np.testing.assert_almost_equal(self.iasolver.F[2],
                                       self.iasolver.full_F[2])

        # Let's change F and see if full_F matches
        self.iasolver.randomizeF(1)
        np.testing.assert_almost_equal(self.iasolver.F[0],
                                       self.iasolver.full_F[0])
        np.testing.assert_almost_equal(self.iasolver.F[1],
                                       self.iasolver.full_F[1])
        np.testing.assert_almost_equal(self.iasolver.F[2],
                                       self.iasolver.full_F[2])

    def test_set_precoders(self):
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        multiUserChannel.randomize(4, 4, 3)

        iasolver1 = IASolverBaseClassConcret(multiUserChannel)
        iasolver2 = IASolverBaseClassConcret(multiUserChannel)
        iasolver3 = IASolverBaseClassConcret(multiUserChannel)
        P = np.array([1.2, 0.8, 1.1])

        with self.assertRaises(RuntimeError):
            iasolver1.set_precoders()

        F = np.empty(3, dtype=np.ndarray)
        full_F = np.empty(3, dtype=np.ndarray)
        full_F_other = np.empty(3, dtype=np.ndarray)

        F[0] = np.random.randn(3, 2)
        F[1] = np.random.randn(3, 1)
        F[2] = np.random.randn(3, 2)

        for k in range(3):
            F[k] /= np.linalg.norm(F[k], 'fro')
            full_F[k] = np.sqrt(P[k]) * F[k]

            # factor can be any value from 0.75 to 1.0
            factor = np.random.random_sample() / 4 + 0.75
            full_F_other[k] = np.sqrt(factor) * full_F[k]

        iasolver1.set_precoders(F, P=P)
        np.testing.assert_array_almost_equal(F[0], iasolver1.F[0])
        np.testing.assert_array_almost_equal(F[1], iasolver1.F[1])
        np.testing.assert_array_almost_equal(F[2], iasolver1.F[2])
        np.testing.assert_array_almost_equal(iasolver1.P, P)
        np.testing.assert_array_almost_equal(iasolver1.full_F[0], full_F[0])
        np.testing.assert_array_almost_equal(iasolver1.full_F[1], full_F[1])
        np.testing.assert_array_almost_equal(iasolver1.full_F[2], full_F[2])
        np.testing.assert_array_equal(iasolver1.Ns, np.array([2, 1, 2]))

        iasolver2.set_precoders(full_F=full_F)
        np.testing.assert_array_almost_equal(iasolver2.full_F[0], full_F[0])
        np.testing.assert_array_almost_equal(iasolver2.full_F[1], full_F[1])
        np.testing.assert_array_almost_equal(iasolver2.full_F[2], full_F[2])
        np.testing.assert_array_almost_equal(iasolver2.F[0], F[0])
        np.testing.assert_array_almost_equal(iasolver2.F[1], F[1])
        np.testing.assert_array_almost_equal(iasolver2.F[2], F[2])
        np.testing.assert_array_equal(iasolver2.Ns, np.array([2, 1, 2]))

        # The full_F precoders don't use all of the available power
        iasolver3.set_precoders(F=F, full_F=full_F_other, P=P)
        np.testing.assert_array_almost_equal(iasolver3.P, P)
        np.testing.assert_array_almost_equal(iasolver3.F[0], F[0])
        np.testing.assert_array_almost_equal(iasolver3.F[1], F[1])
        np.testing.assert_array_almost_equal(iasolver3.F[2], F[2])
        np.testing.assert_array_almost_equal(iasolver3.full_F[0],
                                             full_F_other[0])
        np.testing.assert_array_almost_equal(iasolver3.full_F[1],
                                             full_F_other[1])
        np.testing.assert_array_almost_equal(iasolver3.full_F[2],
                                             full_F_other[2])
        np.testing.assert_array_equal(iasolver3.Ns, np.array([2, 1, 2]))

    def test_set_receive_filters(self):
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        multiUserChannel.randomize(4, 4, 3)

        iasolver1 = IASolverBaseClassConcret(multiUserChannel)

        P = np.array([1.2, 0.8, 1.1])
        iasolver1.randomizeF([2, 3, 2], P)

        W = np.empty(3, dtype=np.ndarray)
        W_H = np.empty(3, dtype=np.ndarray)

        W[0] = np.random.randn(4, 2)
        W[1] = np.random.randn(4, 3)
        W[2] = np.random.randn(4, 2)
        for k in range(3):
            W_H[k] = W[k].conj().T

        self.assertIsNone(iasolver1._W)
        self.assertIsNone(iasolver1._W_H)
        self.assertIsNone(iasolver1._full_W)
        self.assertIsNone(iasolver1._full_W_H)

        # Can't call set_receive_filters without arguments
        with self.assertRaises(RuntimeError):
            iasolver1.set_receive_filters()

        # Can't call set_receive_filters with two arguments
        with self.assertRaises(RuntimeError):
            iasolver1.set_receive_filters(W, W_H)

        iasolver1.set_receive_filters(W=W)
        np.testing.assert_array_almost_equal(iasolver1.W[0], W[0])
        np.testing.assert_array_almost_equal(iasolver1.W[1], W[1])
        np.testing.assert_array_almost_equal(iasolver1.W[2], W[2])

        np.testing.assert_array_almost_equal(iasolver1.W_H[0], W_H[0])
        np.testing.assert_array_almost_equal(iasolver1.W_H[1], W_H[1])
        np.testing.assert_array_almost_equal(iasolver1.W_H[2], W_H[2])

        iasolver1._clear_receive_filter()
        iasolver1.set_receive_filters(W_H=W_H)
        np.testing.assert_array_almost_equal(iasolver1.W[0], W[0])
        np.testing.assert_array_almost_equal(iasolver1.W[1], W[1])
        np.testing.assert_array_almost_equal(iasolver1.W[2], W[2])

        np.testing.assert_array_almost_equal(iasolver1.W_H[0], W_H[0])
        np.testing.assert_array_almost_equal(iasolver1.W_H[1], W_H[1])
        np.testing.assert_array_almost_equal(iasolver1.W_H[2], W_H[2])

    def test_calc_Q(self):
        K = 3
        Nt = np.array([2, 2, 2])
        Nr = np.array([2, 2, 2])
        Ns = np.array([1, 1, 1])
        multiUserChannel = self.iasolver._multiUserChannel

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver.randomizeF(Ns, P)

        # xxxxx Calculate the expected Q[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H01_F1 = np.dot(self.iasolver._get_channel(k, 1),
                        self.iasolver.full_F[1])
        H02_F2 = np.dot(self.iasolver._get_channel(k, 2),
                        self.iasolver.full_F[2])
        expected_Q0 = (np.dot(H01_F1,
                              H01_F1.transpose().conjugate()) +
                       np.dot(H02_F2,
                              H02_F2.transpose().conjugate()))

        Qk = self.iasolver.calc_Q(k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0 = np.dot(self.iasolver._get_channel(k, 0), self.iasolver._F[0])
        H12_F2 = np.dot(self.iasolver._get_channel(k, 2), self.iasolver._F[2])
        expected_Q1 = (np.dot(P[0] * H10_F0,
                              H10_F0.transpose().conjugate()) +
                       np.dot(P[2] * H12_F2,
                              H12_F2.transpose().conjugate()))

        Qk = self.iasolver.calc_Q(k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H20_F0 = np.dot(self.iasolver._get_channel(k, 0), self.iasolver._F[0])
        H21_F1 = np.dot(self.iasolver._get_channel(k, 1), self.iasolver._F[1])
        expected_Q2 = (np.dot(P[0] * H20_F0,
                              H20_F0.transpose().conjugate()) +
                       np.dot(P[1] * H21_F1,
                              H21_F1.transpose().conjugate()))

        Qk = self.iasolver.calc_Q(k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_remaining_interference_percentage(self):
        K = 3
        Nt = np.array([2, 2, 2])
        Nr = np.array([2, 2, 2])
        Ns = np.array([1, 1, 1])
        multiUserChannel = self.iasolver._multiUserChannel

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver.randomizeF(Ns, P)

        # xxxxxxxxxx k = 0 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 0
        Qk = self.iasolver.calc_Q(k)
        pk = self.iasolver.calc_remaining_interference_percentage(k, Qk)

        [_, D] = leig(Qk, Ns[k])
        expected_pk = np.sum(np.abs(D)) / np.abs(np.trace(Qk))
        self.assertAlmostEqual(pk, expected_pk)

        # xxxxxxxxxx k = 1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 1
        Qk = self.iasolver.calc_Q(k)
        pk = self.iasolver.calc_remaining_interference_percentage(k)

        [_, D] = leig(Qk, Ns[k])
        expected_pk = np.sum(np.abs(D)) / np.abs(np.trace(Qk))
        self.assertAlmostEqual(pk, expected_pk)

        # xxxxxxxxxx k = 2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 2
        Qk = self.iasolver.calc_Q(k)
        pk = self.iasolver.calc_remaining_interference_percentage(k)

        [_, D] = leig(Qk, Ns[k])
        expected_pk = np.sum(np.abs(D)) / np.abs(np.trace(Qk))
        self.assertAlmostEqual(pk, expected_pk)

    def test_calc_Bkl_cov_matrix_first_part(self):
        Nr = 2
        Nt = 2
        Ns = 1 * np.ones(3, dtype=int)
        K = 3
        P = np.array([1.2, 1.5, 0.9])

        self.iasolver._multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver.randomizeF(Ns, P)

        # For one stream the expected Bkl is equivalent to the Q matrix
        # plus the direct channel part.
        for k in range(self.iasolver.K):
            Hkk = self.iasolver._get_channel(k, k)
            Fk = self.iasolver.full_F[k]
            HkkFk = np.dot(Hkk, Fk)
            expected_first_part = (self.iasolver.calc_Q(k) +
                                   np.dot(HkkFk,
                                          HkkFk.transpose().conjugate()))

            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.iasolver._calc_Bkl_cov_matrix_first_part(k))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test for more streams
        Nr = 4
        Nt = 4
        Ns = 2 * np.ones(3, dtype=int)

        self.iasolver._multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver.randomizeF(Ns, P)

        for k in range(self.iasolver.K):
            # First part in the equation of Bkl (the double summation)
            expected_first_part = 0.0

            # The outer for loop will calculate
            # first_part = $\sum_{j=1}^{K} \frac{P[k]}{Ns[k]} \text{aux}$
            # noinspection PyPep8
            for j in range(self.iasolver.K):
                # The inner for loop will calculate
                # $\text{aux} = \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$
                aux = 0.0
                Hkj = self.iasolver._get_channel(k, j)
                Hkj_H = Hkj.conjugate().transpose()

                for d in range(self.iasolver.Ns[k]):
                    Vjd = self.iasolver.full_F[j][:, d:d + 1]
                    Vjd_H = Vjd.conjugate().transpose()
                    aux = aux + np.dot(np.dot(Hkj, np.dot(Vjd, Vjd_H)), Hkj_H)

                expected_first_part += aux

            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.iasolver._calc_Bkl_cov_matrix_first_part(k))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_Bkl_cov_matrix_second_part(self):
        Nr = 2
        Nt = 2
        Ns = 1 * np.ones(3, dtype=int)
        K = 3
        P = np.array([1.2, 1.5, 0.9])

        self.iasolver._multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver.randomizeF(Ns, P)

        for k in range(K):
            Hkk = self.iasolver._get_channel(k, k)
            Hkk_H = Hkk.transpose().conjugate()
            # noinspection PyPep8
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = self.iasolver.full_F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(
                    Hkk, np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.iasolver._calc_Bkl_cov_matrix_second_part(k, l))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test for more streams
        Nr = 4
        Nt = 4
        Ns = 2 * np.ones(3, dtype=int)
        K = 3
        P = np.array([1.2, 1.5, 0.9])

        self.iasolver._multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver.randomizeF(Ns, P)

        for k in range(K):
            Hkk = self.iasolver._get_channel(k, k)
            Hkk_H = Hkk.transpose().conjugate()
            # noinspection PyPep8
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = self.iasolver.full_F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(
                    Hkk, np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.iasolver._calc_Bkl_cov_matrix_second_part(k, l))

    # noinspection PyTypeChecker
    def test_calc_Bkl(self):
        # For the case of a single stream per user Bkl (which only has l=0)
        # is equal to Qk plus I (identity matrix)
        Nr = 2
        Nt = 2
        Ns = 1 * np.ones(3, dtype=int)
        K = 3
        P = np.array([1.2, 1.5, 0.9])
        noise_power = 0.568

        self.iasolver._multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver.randomizeF(Ns, P)

        for k in range(K):
            # We only have the stream 0
            expected_Bk0 = self.iasolver.calc_Q(k) + (noise_power * np.eye(Nr))
            Bk0 = self.iasolver._calc_Bkl_cov_matrix_all_l(
                k, noise_power=noise_power)[0]
            np.testing.assert_array_almost_equal(expected_Bk0, Bk0)

        # xxxxx Test the case with more than one stream per user xxxxxxxxxx
        Nr = 4
        Nt = 4
        Ns = 2
        self.iasolver.clear()
        self.iasolver._multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver.randomizeF(Ns, P)

        V0 = self.iasolver.full_F[0]
        V1 = self.iasolver.full_F[1]
        V2 = self.iasolver.full_F[2]
        V00 = V0[:, 0, np.newaxis]  # Guaranteed to be column vectors
        V01 = V0[:, 1, np.newaxis]  # Guaranteed to be column vectors
        V10 = V1[:, 0, np.newaxis]  # Guaranteed to be column vectors
        V11 = V1[:, 1, np.newaxis]  # Guaranteed to be column vectors
        V20 = V2[:, 0, np.newaxis]  # Guaranteed to be column vectors
        V21 = V2[:, 1, np.newaxis]  # Guaranteed to be column vectors

        H00 = self.iasolver._get_channel(0, 0)
        H01 = self.iasolver._get_channel(0, 1)
        H02 = self.iasolver._get_channel(0, 2)
        H10 = self.iasolver._get_channel(1, 0)
        H11 = self.iasolver._get_channel(1, 1)
        H12 = self.iasolver._get_channel(1, 2)
        H20 = self.iasolver._get_channel(2, 0)
        H21 = self.iasolver._get_channel(2, 1)
        H22 = self.iasolver._get_channel(2, 2)

        # Noise matrix
        Z = np.eye(4) * noise_power

        expected_first_part_user0 = (H00 @ V00 @ V00.T.conj() @ H00.T.conj() +
                                     H00 @ V01 @ V01.T.conj() @ H00.T.conj() +
                                     H01 @ V10 @ V10.T.conj() @ H01.T.conj() +
                                     H01 @ V11 @ V11.T.conj() @ H01.T.conj() +
                                     H02 @ V20 @ V20.T.conj() @ H02.T.conj() +
                                     H02 @ V21 @ V21.T.conj() @ H02.T.conj())
        expected_second_part_user0_l0 = H00 @ V00 @ V00.T.conj() @ H00.T.conj()
        expected_second_part_user0_l1 = H00 @ V01 @ V01.T.conj() @ H00.T.conj()
        np.testing.assert_array_almost_equal(
            expected_first_part_user0,
            self.iasolver._calc_Bkl_cov_matrix_first_part(0))
        np.testing.assert_array_almost_equal(
            expected_second_part_user0_l0,
            self.iasolver._calc_Bkl_cov_matrix_second_part(0, 0))
        np.testing.assert_array_almost_equal(
            expected_second_part_user0_l1,
            self.iasolver._calc_Bkl_cov_matrix_second_part(0, 1))
        expected_B00 \
            = expected_first_part_user0 - expected_second_part_user0_l0 + Z
        expected_B01 \
            = expected_first_part_user0 - expected_second_part_user0_l1 + Z
        B0 = self.iasolver._calc_Bkl_cov_matrix_all_l(0,
                                                      noise_power=noise_power)
        np.testing.assert_array_almost_equal(expected_B00, B0[0])
        np.testing.assert_array_almost_equal(expected_B01, B0[1])

        expected_first_part_user1 = (H10 @ V00 @ V00.T.conj() @ H10.T.conj() +
                                     H10 @ V01 @ V01.T.conj() @ H10.T.conj() +
                                     H11 @ V10 @ V10.T.conj() @ H11.T.conj() +
                                     H11 @ V11 @ V11.T.conj() @ H11.T.conj() +
                                     H12 @ V20 @ V20.T.conj() @ H12.T.conj() +
                                     H12 @ V21 @ V21.T.conj() @ H12.T.conj())
        expected_second_part_user1_l0 = H11 @ V10 @ V10.T.conj() @ H11.T.conj()
        expected_second_part_user1_l1 = H11 @ V11 @ V11.T.conj() @ H11.T.conj()
        np.testing.assert_array_almost_equal(
            expected_first_part_user1,
            self.iasolver._calc_Bkl_cov_matrix_first_part(1))
        np.testing.assert_array_almost_equal(
            expected_second_part_user1_l0,
            self.iasolver._calc_Bkl_cov_matrix_second_part(1, 0))
        np.testing.assert_array_almost_equal(
            expected_second_part_user1_l1,
            self.iasolver._calc_Bkl_cov_matrix_second_part(1, 1))
        expected_B10 \
            = expected_first_part_user1 - expected_second_part_user1_l0 + Z
        expected_B11 \
            = expected_first_part_user1 - expected_second_part_user1_l1 + Z

        B1 = self.iasolver._calc_Bkl_cov_matrix_all_l(1,
                                                      noise_power=noise_power)
        np.testing.assert_array_almost_equal(expected_B10, B1[0])
        np.testing.assert_array_almost_equal(expected_B11, B1[1])

        expected_first_part_user2 = (H20 @ V00 @ V00.T.conj() @ H20.T.conj() +
                                     H20 @ V01 @ V01.T.conj() @ H20.T.conj() +
                                     H21 @ V10 @ V10.T.conj() @ H21.T.conj() +
                                     H21 @ V11 @ V11.T.conj() @ H21.T.conj() +
                                     H22 @ V20 @ V20.T.conj() @ H22.T.conj() +
                                     H22 @ V21 @ V21.T.conj() @ H22.T.conj())
        expected_second_part_user2_l0 = H22 @ V20 @ V20.T.conj() @ H22.T.conj()
        expected_second_part_user2_l1 = H22 @ V21 @ V21.T.conj() @ H22.T.conj()
        np.testing.assert_array_almost_equal(
            expected_first_part_user2,
            self.iasolver._calc_Bkl_cov_matrix_first_part(2))
        np.testing.assert_array_almost_equal(
            expected_second_part_user2_l0,
            self.iasolver._calc_Bkl_cov_matrix_second_part(2, 0))
        np.testing.assert_array_almost_equal(
            expected_second_part_user2_l1,
            self.iasolver._calc_Bkl_cov_matrix_second_part(2, 1))
        expected_B20 \
            = expected_first_part_user2 - expected_second_part_user2_l0 + Z
        expected_B21 \
            = expected_first_part_user2 - expected_second_part_user2_l1 + Z

        B2 = self.iasolver._calc_Bkl_cov_matrix_all_l(2,
                                                      noise_power=noise_power)
        np.testing.assert_array_almost_equal(expected_B20, B2[0])
        np.testing.assert_array_almost_equal(expected_B21, B2[1])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_SINR_k(self):
        # This test is implemented in the MaxSinrIASolverTestCase class.
        pass


class ClosedFormIASolverTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        self.iasolver = ClosedFormIASolver(multiUserChannel)
        self.K = 3
        self.Nr = np.array([2, 2, 2])
        self.Nt = np.array([2, 2, 2])
        multiUserChannel.randomize(self.Nr, self.Nt, self.K)

    def test_sanity(self):
        # The number of users is always equal to 3
        self.assertEqual(self.iasolver.K, 3)
        # np.testing.assert_array_equal(np.ones(3), self.iasolver.Ns)

    def test_invalid_solve(self):
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        Ns = 1
        # ClosedFormIASolver only works with 3 users ...
        iasolver2 = ClosedFormIASolver(multiUserChannel)
        K = 4
        multiUserChannel.randomize(3, 3, K)
        # ... Therefore an AssertionError will be raised if we try to call
        # the solve method.
        with self.assertRaises(AssertionError):
            iasolver2.solve(Ns)

    def test_calc_E(self):
        H31 = self.iasolver._get_channel(2, 0)
        H32 = self.iasolver._get_channel(2, 1)
        H12 = self.iasolver._get_channel(0, 1)
        H13 = self.iasolver._get_channel(0, 2)
        H23 = self.iasolver._get_channel(1, 2)
        H21 = self.iasolver._get_channel(1, 0)

        inv = np.linalg.inv
        expected_E = inv(H31) @ H32 @ inv(H12) @ H13 @ inv(H23) @ H21
        np.testing.assert_array_almost_equal(expected_E,
                                             self.iasolver._calc_E())

    def test_calc_all_F_initializations(self):
        # xxxxx Test the case with Nt = Ns = 2 and Ns = 1 xxxxxxxxxxxxxxxxx
        Ns = 1
        E = self.iasolver._calc_E()
        all_eigenvectors = np.linalg.eig(E)[1]
        expected_all_subsets = [
            all_eigenvectors[:, (0, )], all_eigenvectors[:, (1, )]
        ]

        all_subsets = self.iasolver._calc_all_F_initializations(Ns)

        for a, b in zip(expected_all_subsets, all_subsets):
            np.testing.assert_array_almost_equal(a, b)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test the case with Nt = Ns = 4 and Ns = 2 xxxxxxxxxxxxxxxxx
        Ns2 = 2
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        iasolver = ClosedFormIASolver(multiUserChannel)
        K = 3
        Nr = np.array([4, 4, 4])
        Nt = np.array([4, 4, 4])
        multiUserChannel.randomize(Nr, Nt, K)
        E2 = iasolver._calc_E()
        all_eigenvectors2 = np.linalg.eig(E2)[1]
        expected_all_subsets2 = [
            all_eigenvectors2[:, (0, 1)], all_eigenvectors2[:, (0, 2)],
            all_eigenvectors2[:, (0, 3)], all_eigenvectors2[:, (1, 2)],
            all_eigenvectors2[:, (1, 3)], all_eigenvectors2[:, (2, 3)]
        ]

        all_subsets2 = iasolver._calc_all_F_initializations(Ns2)

        for a, b in zip(expected_all_subsets2, all_subsets2):
            np.testing.assert_array_almost_equal(a, b)

    def test_updateF(self):
        P = np.array([1.1, 0.86, 1.328])
        self.iasolver.P = P
        Ns = 1
        E = self.iasolver._calc_E()
        [_, eigenvectors] = np.linalg.eig(E)

        inv = np.linalg.inv

        # V1 is the expected precoder for the first user
        V1 = eigenvectors[:, 0:Ns]

        H32 = self.iasolver._get_channel(2, 1)
        H31 = self.iasolver._get_channel(2, 0)
        H23 = self.iasolver._get_channel(1, 2)
        H21 = self.iasolver._get_channel(1, 0)

        # Expected precoder for the second user
        V2 = inv(H32) @ H31 @ V1
        # Expected precoder for the third user
        V3 = inv(H23) @ H21 @ V1

        # Normalize the precoders
        V1 /= norm(V1, 'fro')
        V2 /= norm(V2, 'fro')
        V3 /= norm(V3, 'fro')

        # The number of streams _Ns is set in the solve method, before
        # _updateF is called. However, since we are testing the _updateF
        # method alone here we need to set _Ns manually.
        self.iasolver._Ns = np.ones(self.iasolver.K, dtype=int) * Ns

        # Find the precoders using the iasolver
        self.iasolver._updateF()

        np.testing.assert_array_almost_equal(V1, self.iasolver.F[0])
        np.testing.assert_array_almost_equal(V2, self.iasolver.F[1])
        np.testing.assert_array_almost_equal(V3, self.iasolver.F[2])

        self.assertAlmostEqual(norm(self.iasolver.F[0], 'fro'), 1.0)
        self.assertAlmostEqual(norm(self.iasolver.F[1], 'fro'), 1.0)
        self.assertAlmostEqual(norm(self.iasolver.F[2], 'fro'), 1.0)

        self.assertAlmostEqual(norm(self.iasolver.full_F[0], 'fro')**2, P[0])
        self.assertAlmostEqual(norm(self.iasolver.full_F[1], 'fro')**2, P[1])
        self.assertAlmostEqual(norm(self.iasolver.full_F[2], 'fro')**2, P[2])

        np.testing.assert_array_almost_equal(
            self.iasolver.full_F[0], self.iasolver.F[0] * np.sqrt(P[0]))
        np.testing.assert_array_almost_equal(
            self.iasolver.full_F[1], self.iasolver.F[1] * np.sqrt(P[1]))
        np.testing.assert_array_almost_equal(
            self.iasolver.full_F[2], self.iasolver.F[2] * np.sqrt(P[2]))

    # noinspection PyTypeChecker
    def test_updateW(self):
        P = np.array([1.1, 0.86, 1.328])
        self.iasolver.P = P
        Ns = 1
        # The number of streams _Ns is set in the solve method, before
        # _updateF and the _updateW methods are called. However, since we
        # are testing the _updateW method alone here we need to set _Ns
        # manually.
        self.iasolver._Ns = np.ones(self.iasolver.K, dtype=int) * Ns

        self.iasolver._updateF()
        self.iasolver._updateW()
        V1 = self.iasolver.F[0]
        V2 = self.iasolver.F[1]
        # V3 = self.iasolver.F[2]

        full_V1 = self.iasolver.full_F[0]
        full_V2 = self.iasolver.full_F[1]
        full_V3 = self.iasolver.full_F[2]

        H11 = self.iasolver._get_channel(0, 0)
        H12 = self.iasolver._get_channel(0, 1)
        H13 = self.iasolver._get_channel(0, 2)
        H21 = self.iasolver._get_channel(1, 0)
        H22 = self.iasolver._get_channel(1, 1)
        H23 = self.iasolver._get_channel(1, 2)
        H31 = self.iasolver._get_channel(2, 0)
        H32 = self.iasolver._get_channel(2, 1)
        H33 = self.iasolver._get_channel(2, 2)

        U1 = H12 @ V2
        U1 = leig(U1 @ U1.T.conj(), 1)[0]
        U2 = H21 @ V1
        U2 = leig(U2 @ U2.T.conj(), 1)[0]
        U3 = H31 @ V1
        U3 = leig(U3 @ U3.T.conj(), 1)[0]

        np.testing.assert_array_almost_equal(self.iasolver._W[0], U1)
        np.testing.assert_array_almost_equal(self.iasolver._W[1], U2)
        np.testing.assert_array_almost_equal(self.iasolver._W[2], U3)

        # xxxxx Test the direct channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.assertAlmostEqual(
            (self.iasolver.full_W_H[0] @ H11 @ full_V1)[0, 0], 1.0)
        self.assertAlmostEqual(
            (self.iasolver.full_W_H[1] @ H22 @ full_V2)[0, 0], 1.0)
        self.assertAlmostEqual(
            (self.iasolver.full_W_H[2] @ H33 @ full_V3)[0, 0], 1.0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test if the interference is cancelled xxxxxxxxxxxxxxxxxxxxx
        I1 = ((self.iasolver.W_H[0] @ H12 @ self.iasolver.F[1]) +
              (self.iasolver.W_H[0] @ H13 @ self.iasolver.F[2]))
        np.testing.assert_array_almost_equal(I1, 0.0)

        I2 = (self.iasolver.W_H[1] @ H21 @ self.iasolver.F[0] +
              self.iasolver.W_H[1] @ H23 @ self.iasolver.F[2])
        np.testing.assert_array_almost_equal(I2, 0.0)

        I3 = (self.iasolver.W_H[2] @ H31 @ self.iasolver.F[0] +
              self.iasolver.W_H[2] @ H32 @ self.iasolver.F[1])
        np.testing.assert_array_almost_equal(I3, 0.0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Sanity tests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test if the hermitian of W[k] is equal to W[k]
        for k in range(self.iasolver.K):
            self.assertIsNotNone(self.iasolver.W[k])
            np.testing.assert_array_almost_equal(self.iasolver.W[k].conj().T,
                                                 self.iasolver.W_H[k])
            # Note: the full_W_H property will be tested in the
            # test_solve_and_calc_equivalent_channel method.
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_solve_and_calc_equivalent_channel(self):
        Ns = 1
        P = [0.8, 1.1, 0.956]
        self.iasolver._multiUserChannel.noise_var = 1e-4

        self.iasolver.solve(Ns, P)

        for l in range(3):
            for k in range(3):
                Hlk = self.iasolver._get_channel(l, k)
                Wl_H = self.iasolver.W_H[l]
                full_Wl_H = self.iasolver.full_W_H[l]
                full_Fk = self.iasolver.full_F[k]
                s = np.dot(Wl_H, np.dot(Hlk, full_Fk))[0][0]
                s2 = np.dot(full_Wl_H, np.dot(Hlk, full_Fk))[0][0]
                if l == k:
                    Hk_eq = self.iasolver._calc_equivalent_channel(k)
                    # We only have one stream -> the equivalent channel is
                    # an scalar.
                    s3 = s / Hk_eq[0, 0]
                    self.assertAlmostEqual(1.0, s2)
                    self.assertAlmostEqual(1.0, s3)
                else:
                    # Test if the interference is equal to 0.0
                    self.assertAlmostEqual(0.0, s)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test if an exception is raised if we try to use the
        # ClosedFormIASolver class with a number of users different from 3.
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        Nr = 2
        Nt = 2
        multiUserChannel.randomize(Nr, Nt, 2)
        iasolver = ClosedFormIASolver(multiUserChannel)
        with self.assertRaises(AssertionError):
            iasolver.solve(Ns=1)

        multiUserChannel.randomize(Nr, Nt, 3)
        iasolver = ClosedFormIASolver(multiUserChannel)
        with self.assertRaises(AssertionError):
            iasolver.solve(Ns=np.array([1, 1]))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_solve_best_solution(self):
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        iasolver = ClosedFormIASolver(multiUserChannel, use_best_init=True)
        iasolver2 = ClosedFormIASolver(multiUserChannel, use_best_init=False)
        K = 3
        Nr = 4
        Nt = 4
        Ns = 2

        multiUserChannel.randomize(Nr, Nt, K)

        iasolver.solve(Ns)
        iasolver2.solve(Ns)

        # Test of the equivalent direct channel is equal to one while the
        # equivalent cross channels are equivalent to zero
        for l in range(3):
            for k in range(3):
                Hlk = iasolver._get_channel(l, k)
                Wl_H = iasolver.W_H[l]
                Fk = iasolver.F[k]
                s = np.dot(Wl_H, np.dot(Hlk, Fk))[0][0]
                if l == k:
                    Hk_eq = iasolver._calc_equivalent_channel(k)
                    # We only have one stream -> the equivalent channel is
                    # an scalar.
                    s2 = s / Hk_eq[0, 0]
                    self.assertAlmostEqual(1.0, s2)
                else:
                    # Test if the interference is equal to 0.0
                    self.assertAlmostEqual(0.0, s)

        SINRs = iasolver.calc_SINR()
        SINRs2 = iasolver2.calc_SINR()

        # Sum Capacity using the best initialization
        sum_capacity1 = np.sum(np.log2(1 + np.hstack(SINRs)))
        # Sum Capacity using the first initialization
        sum_capacity2 = np.sum(np.log2(1 + np.hstack(SINRs2)))
        self.assertTrue(sum_capacity1 >= sum_capacity2)

    def test_full_W_H_property(self):
        Ns = 1
        P = np.array([0.7, 1.34, 0.94])
        self.iasolver.solve(Ns, P)
        for k in range(self.iasolver.K):
            Hkk = self.iasolver._get_channel(k, k)
            full_Wk_H = self.iasolver.full_W_H[k]
            full_Fk = self.iasolver.full_F[k]
            s = np.dot(full_Wk_H, np.dot(Hkk, full_Fk))[0][0]
            self.assertAlmostEqual(1.0, s)


class IterativeIASolverBaseClassTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_is_diff_significant(self):
        F0 = np.array([[1, 2, 3], [3, 2, 2], [1, 4, 2]], dtype=float)
        F1 = np.array([[2, 4, 5], [2, -3, -2], [5, -4, 3]], dtype=float)
        F2 = np.array([[4, 1, 1], [-3, -7, 4], [2, -2, 5]], dtype=float)
        F_old = np.empty(3, dtype=np.ndarray)
        F_old[0] = F0
        F_old[1] = F1
        F_old[2] = F2

        F_new = np.empty(3, dtype=np.ndarray)
        F_new[0] = F0.copy()
        F_new[1] = F1.copy()
        F_new[2] = F2.copy()

        self.assertFalse(
            IterativeIASolverBaseClass._is_diff_significant(
                F_old, F_new, 1e-3))
        F_new[1][1, 2] += 9e-4
        F_new[2][0, 0] += 6e-4
        self.assertFalse(
            IterativeIASolverBaseClass._is_diff_significant(
                F_old, F_new, 1e-3))
        F_new[2][2, 2] += 2e-3
        self.assertTrue(
            IterativeIASolverBaseClass._is_diff_significant(
                F_old, F_new, 1e-3))

    def test_initialize_with_property(self):
        channel = channels.multiuser.MultiUserChannelMatrix()
        solver = IterativeIASolverBaseClassConcrete(channel)

        self.assertEqual(solver.initialize_with, 'random')
        solver.initialize_with = 'fix'
        self.assertEqual(solver.initialize_with, 'fix')

        solver.initialize_with = 'closed_form'
        self.assertEqual(solver.initialize_with, 'closed_form')

        solver.initialize_with = 'alt_min'
        self.assertEqual(solver.initialize_with, 'alt_min')

        solver.initialize_with = 'svd'
        self.assertEqual(solver.initialize_with, 'svd')

        with self.assertRaises(RuntimeError):
            solver.initialize_with = 'invalid_option'

        with self.assertRaises(RuntimeError):
            solver.initialize_with = 'fix'
            # When the 'fix' initialization type is set the randomizeF
            # method must be called before the
            # '_dont_initialize_F_and_only_and_find_W' method (called
            # inside 'solve_init' method) is called. Since that is not the
            # case here, an exception should be raised.
            solver._dont_initialize_F_and_only_and_find_W()

    def test_solve_init(self):
        channel = channels.multiuser.MultiUserChannelMatrix()
        channel.randomize(4, 4, 3)
        solver = IterativeIASolverBaseClassConcrete(channel)
        # Not really necessary, since we just created the solver object
        solver.clear()

        # xxxxxxxxxx Test the random initialization xxxxxxxxxxxxxxxxxxxxxxx
        solver.initialize_with = 'random'
        # Initialize the normalized precoder of each user with a random one
        # (with 2 streams)

        Ns = 2
        solver._solve_init(Ns=Ns, P=1.0)

        # solver._F was randomly initialized
        self.assertIsNotNone(solver._F)
        self.assertEqual(solver._F.shape, (3, ))
        self.assertEqual(solver._F[0].shape, (4, Ns))
        self.assertEqual(solver._F[1].shape, (4, Ns))
        self.assertEqual(solver._F[2].shape, (4, Ns))
        self.assertAlmostEqual(np.linalg.norm(solver._F[0]), 1.0)
        self.assertAlmostEqual(np.linalg.norm(solver._F[1]), 1.0)
        self.assertAlmostEqual(np.linalg.norm(solver._F[2]), 1.0)

        # The other variables are still None
        self.assertIsNone(solver._full_F)
        self.assertIsNone(solver._W)
        self.assertIsNone(solver._W_H)
        self.assertIsNone(solver._full_W)
        self.assertIsNone(solver._full_W_H)

        solver.clear()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test the closed form initialization xxxxxxxxxxxxxxxxxx
        solver.initialize_with = 'closed_form'
        solver._solve_init(Ns=Ns, P=1.0)

        # solver._F was initialized from the closed form solution
        self.assertIsNotNone(solver._F)
        self.assertEqual(solver._F.shape, (3, ))
        self.assertEqual(solver._F[0].shape, (4, Ns))
        self.assertEqual(solver._F[1].shape, (4, Ns))
        self.assertEqual(solver._F[2].shape, (4, Ns))
        self.assertAlmostEqual(np.linalg.norm(solver._F[0]), 1.0)
        self.assertAlmostEqual(np.linalg.norm(solver._F[1]), 1.0)
        self.assertAlmostEqual(np.linalg.norm(solver._F[2]), 1.0)

        # solver._W was initialized from the closed form solution
        self.assertIsNotNone(solver._W)
        self.assertEqual(solver._W.shape, (3, ))
        self.assertEqual(solver._W[0].shape, (4, Ns))
        self.assertEqual(solver._W[1].shape, (4, Ns))
        self.assertEqual(solver._W[2].shape, (4, Ns))

        # The other variables are still None
        self.assertIsNone(solver._full_F)
        self.assertIsNone(solver._W_H)
        self.assertIsNone(solver._full_W)
        self.assertIsNone(solver._full_W_H)

        solver.clear()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test the alt_min initialization xxxxxxxxxxxxxxxxxxxxxx
        solver.initialize_with = 'alt_min'
        solver._solve_init(Ns=Ns, P=1.0)

        # solver._F was initialized from the closed form solution
        self.assertIsNotNone(solver._F)
        self.assertEqual(solver._F.shape, (3, ))
        self.assertEqual(solver._F[0].shape, (4, Ns))
        self.assertEqual(solver._F[1].shape, (4, Ns))
        self.assertEqual(solver._F[2].shape, (4, Ns))
        self.assertAlmostEqual(np.linalg.norm(solver._F[0]), 1.0)
        self.assertAlmostEqual(np.linalg.norm(solver._F[1]), 1.0)
        self.assertAlmostEqual(np.linalg.norm(solver._F[2]), 1.0)

        # solver._W was initialized from the closed form solution
        self.assertIsNotNone(solver._W)
        self.assertEqual(solver._W.shape, (3, ))
        self.assertEqual(solver._W[0].shape, (4, Ns))
        self.assertEqual(solver._W[1].shape, (4, Ns))
        self.assertEqual(solver._W[2].shape, (4, Ns))

        # The other variables are still None
        self.assertIsNone(solver._full_F)
        self.assertIsNone(solver._W_H)
        self.assertIsNone(solver._full_W)
        self.assertIsNone(solver._full_W_H)

        solver.clear()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test the SVD initialization xxxxxxxxxxxxxxxxxxxxxxxxxx
        solver.initialize_with = 'svd'
        # Initialize the normalized precoder of each user with a random one
        # (with 2 streams)
        solver._solve_init(Ns=Ns, P=1.0)

        # solver._F was randomly initialized
        self.assertIsNotNone(solver._F)
        self.assertEqual(solver._F.shape, (3, ))
        self.assertEqual(solver._F[0].shape, (4, Ns))
        self.assertEqual(solver._F[1].shape, (4, Ns))
        self.assertEqual(solver._F[2].shape, (4, Ns))
        self.assertAlmostEqual(np.linalg.norm(solver._F[0]), 1.0)
        self.assertAlmostEqual(np.linalg.norm(solver._F[1]), 1.0)
        self.assertAlmostEqual(np.linalg.norm(solver._F[2]), 1.0)

        # The other variables are still None
        self.assertIsNone(solver._full_F)
        self.assertIsNone(solver._W)
        self.assertIsNone(solver._W_H)
        self.assertIsNone(solver._full_W)
        self.assertIsNone(solver._full_W_H)

        solver.clear()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test the Fix initialization xxxxxxxxxxxxxxxxxxxxxxxxxx
        # It is not really necessary to test this case
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_clear(self):
        # This method is tested in the AlternatingMinIASolverTestCase class
        pass


# noinspection PyMethodMayBeStatic
class AlternatingMinIASolverTestCase(CustomTestCase):
    """Unittests for the AlternatingMinIASolver class in the ia module."""
    def setUp(self):
        """Called before each test."""
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        self.iasolver = AlternatingMinIASolver(multiUserChannel)

        self.K = 3
        self.Nr = np.array([2, 4, 6])
        self.Nt = np.array([2, 3, 5])
        self.Ns = np.array([1, 2, 3])

        # Transmit power of all users
        self.P = np.array([1.1, 0.876, 1.23])

        # Randomize the channel
        multiUserChannel.randomize(self.Nr, self.Nt, self.K)

    def test_updateC(self):
        self.iasolver.randomizeF(self.Ns, self.P)

        # Dimensions of the interference subspace
        Ni = self.Nr - self.Ns

        self.iasolver._updateC()

        # xxxxx Calculate the expected C[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H01_F1 = np.dot(self.iasolver._get_channel(k, 1),
                        self.iasolver.full_F[1])
        H02_F2 = np.dot(self.iasolver._get_channel(k, 2),
                        self.iasolver.full_F[2])
        expected_C0 = (np.dot(H01_F1,
                              H01_F1.transpose().conjugate()) +
                       np.dot(H02_F2,
                              H02_F2.transpose().conjugate()))
        expected_C0 = peig(expected_C0, Ni[k])[0]

        # Test if C[0] is equal to the expected output
        np.testing.assert_array_almost_equal(self.iasolver._C[0], expected_C0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected C[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0 = np.dot(self.iasolver._get_channel(k, 0),
                        self.iasolver.full_F[0])
        H12_F2 = np.dot(self.iasolver._get_channel(k, 2),
                        self.iasolver.full_F[2])
        expected_C1 = (np.dot(H10_F0,
                              H10_F0.transpose().conjugate()) +
                       np.dot(H12_F2,
                              H12_F2.transpose().conjugate()))
        expected_C1 = peig(expected_C1, Ni[k])[0]

        # Test if C[1] is equal to the expected output
        np.testing.assert_array_almost_equal(self.iasolver._C[1], expected_C1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected C[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H20_F0 = np.dot(self.iasolver._get_channel(k, 0),
                        self.iasolver.full_F[0])
        H21_F1 = np.dot(self.iasolver._get_channel(k, 1),
                        self.iasolver.full_F[1])
        expected_C2 = (np.dot(H20_F0,
                              H20_F0.transpose().conjugate()) +
                       np.dot(H21_F1,
                              H21_F1.transpose().conjugate()))
        expected_C2 = peig(expected_C2, Ni[k])[0]

        # Test if C[2] is equal to the expected output
        np.testing.assert_array_almost_equal(self.iasolver._C[2], expected_C2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_updateF(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._updateC()
        self.iasolver._updateF()

        # xxxxxxxxxx Aliases for each channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        H01 = self.iasolver._get_channel(0, 1)
        H02 = self.iasolver._get_channel(0, 2)

        H10 = self.iasolver._get_channel(1, 0)
        H12 = self.iasolver._get_channel(1, 2)

        H20 = self.iasolver._get_channel(2, 0)
        H21 = self.iasolver._get_channel(2, 1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Aliases for (I-Ck Ck^H)) for each k xxxxxxxxxxxxxxxxxx
        Y0 = (np.eye(self.Nr[0], dtype=complex) - np.dot(
            self.iasolver._C[0], self.iasolver._C[0].conjugate().transpose()))

        Y1 = (np.eye(self.Nr[1], dtype=complex) - np.dot(
            self.iasolver._C[1], self.iasolver._C[1].conjugate().transpose()))

        Y2 = (np.eye(self.Nr[2], dtype=complex) - np.dot(
            self.iasolver._C[2], self.iasolver._C[2].conjugate().transpose()))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected F[0] after one step xxxxxxxxxxxxxxxx
        # l = 0 -> k = 1 and k = 2
        expected_F0 = (np.dot(np.dot(H10.conjugate().transpose(), Y1), H10) +
                       np.dot(np.dot(H20.conjugate().transpose(), Y2), H20))
        expected_F0 = leig(expected_F0, self.Ns[0])[0]
        expected_F0 /= np.linalg.norm(expected_F0, 'fro')
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected F[1] after one step xxxxxxxxxxxxxxxx
        # l = 1 -> k = 0 and k = 2
        expected_F1 = (np.dot(np.dot(H01.conjugate().transpose(), Y0), H01) +
                       np.dot(np.dot(H21.conjugate().transpose(), Y2), H21))
        expected_F1 = leig(expected_F1, self.Ns[1])[0]
        expected_F1 /= np.linalg.norm(expected_F1, 'fro')
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected F[1] after one step xxxxxxxxxxxxxxxx
        # l = 2 -> k = 0 and k = 1
        expected_F2 = (np.dot(np.dot(H02.conjugate().transpose(), Y0), H02) +
                       np.dot(np.dot(H12.conjugate().transpose(), Y1), H12))
        expected_F2 = leig(expected_F2, self.Ns[2])[0]
        expected_F2 /= np.linalg.norm(expected_F2, 'fro')
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Get the precoders xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        F0 = self.iasolver.F[0]
        full_F0 = self.iasolver.full_F[0]
        F1 = self.iasolver.F[1]
        full_F1 = self.iasolver.full_F[1]
        F2 = self.iasolver.F[2]
        full_F2 = self.iasolver.full_F[2]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Finally perform the tests xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_almost_equal(F0, expected_F0)
        np.testing.assert_array_almost_equal(F1, expected_F1)
        np.testing.assert_array_almost_equal(F2, expected_F2)
        np.testing.assert_array_almost_equal(full_F0,
                                             expected_F0 * np.sqrt(self.P[0]))
        np.testing.assert_array_almost_equal(full_F1,
                                             expected_F1 * np.sqrt(self.P[1]))
        np.testing.assert_array_almost_equal(full_F2,
                                             expected_F2 * np.sqrt(self.P[2]))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test the norm of the precoders xxxxxxxxxxxxxxxxxxxxxxx
        self.assertAlmostEqual(np.linalg.norm(F0, 'fro'), 1.0)
        self.assertAlmostEqual(np.linalg.norm(F1, 'fro'), 1.0)
        self.assertAlmostEqual(np.linalg.norm(F2, 'fro'), 1.0)
        self.assertAlmostEqual(np.linalg.norm(full_F0, 'fro')**2, self.P[0])
        self.assertAlmostEqual(np.linalg.norm(full_F1, 'fro')**2, self.P[1])
        self.assertAlmostEqual(np.linalg.norm(full_F2, 'fro')**2, self.P[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_updateW(self):
        # Initialize with some random precoders and then call _updateC()
        # and _updateW()
        self.iasolver._solve_init(self.Ns, self.P)

        # Call updateC, updateF and updateW
        self.iasolver._updateF()  # Depend on the value of C
        self.iasolver._updateC()
        self.iasolver._updateW()  # Depend on the value of C

        # xxxxx Calculates the expected receive filter for user 0 xxxxxxxxx
        tildeH0 = np.dot(self.iasolver._get_channel(0, 0), self.iasolver.F[0])
        tildeH0 = np.hstack([tildeH0, self.iasolver._C[0]])
        expected_W0_H = np.linalg.inv(tildeH0)[0:self.iasolver.Ns[0]]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for user 1 xxxxxxxxx
        tildeH1 = np.dot(self.iasolver._get_channel(1, 1), self.iasolver.F[1])
        tildeH1 = np.hstack([tildeH1, self.iasolver._C[1]])
        expected_W1_H = np.linalg.inv(tildeH1)[0:self.iasolver.Ns[1]]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for user 2 xxxxxxxxx
        tildeH2 = np.dot(self.iasolver._get_channel(2, 2), self.iasolver.F[2])
        tildeH2 = np.hstack([tildeH2, self.iasolver._C[2]])
        expected_W2_H = np.linalg.inv(tildeH2)[0:self.iasolver.Ns[2]]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Finally perform the tests xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_almost_equal(self.iasolver.W_H[0],
                                             expected_W0_H)
        np.testing.assert_array_almost_equal(self.iasolver.W_H[1],
                                             expected_W1_H)
        np.testing.assert_array_almost_equal(self.iasolver.W_H[2],
                                             expected_W2_H)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Equivalent channels xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Heq0 = self.iasolver._calc_equivalent_channel(0)
        Heq1 = self.iasolver._calc_equivalent_channel(1)
        Heq2 = self.iasolver._calc_equivalent_channel(2)
        expected_full_W0_H = np.dot(np.linalg.inv(Heq0), expected_W0_H)
        expected_full_W1_H = np.dot(np.linalg.inv(Heq1), expected_W1_H)
        expected_full_W2_H = np.dot(np.linalg.inv(Heq2), expected_W2_H)
        np.testing.assert_array_almost_equal(self.iasolver.full_W_H[0],
                                             expected_full_W0_H)
        np.testing.assert_array_almost_equal(self.iasolver.full_W_H[1],
                                             expected_full_W1_H)
        np.testing.assert_array_almost_equal(self.iasolver.full_W_H[2],
                                             expected_full_W2_H)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Sanity tests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test if the hermitian of W[k] is equal to W[k]
        for k in range(self.iasolver.K):
            self.assertIsNotNone(self.iasolver.W[k])
            np.testing.assert_array_almost_equal(self.iasolver.W[k].conj().T,
                                                 self.iasolver.W_H[k])
        # Perform one step and test again
        self.iasolver._step()
        for k in range(self.iasolver.K):
            self.assertIsNotNone(self.iasolver.W[k])
            np.testing.assert_array_almost_equal(self.iasolver.W[k].conj().T,
                                                 self.iasolver.W_H[k])
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_getCost(self):
        P = np.array([1.23, 0.965])
        K = 2
        Nr = np.array([3, 3])
        Nt = np.array([3, 3])
        Ns = np.array([2, 2])
        multiUserChannel = self.iasolver._multiUserChannel
        multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver._solve_init(Ns, P)

        # Call updateC, updateF and updateW
        self.iasolver._step()

        Cost = 0
        k, l = (0, 1)
        H01_F1 = np.dot(self.iasolver._get_channel(k, l),
                        self.iasolver.full_F[l])
        Cost += norm(
            H01_F1 - np.dot(
                np.dot(self.iasolver._C[k],
                       self.iasolver._C[k].transpose().conjugate()), H01_F1),
            'fro')**2

        k, l = (1, 0)
        H10_F0 = np.dot(self.iasolver._get_channel(k, l),
                        self.iasolver.full_F[l])
        Cost += norm(
            H10_F0 - np.dot(
                np.dot(self.iasolver._C[k],
                       self.iasolver._C[k].transpose().conjugate()), H10_F0),
            'fro')**2

        self.assertAlmostEqual(self.iasolver.get_cost(), Cost)

    def test_solve(self):
        Nr = 2
        Nt = 2
        Ns = 1
        K = 3
        P = np.array([0.97, 1.125, 1.342])

        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        iasolver = AlternatingMinIASolver(multiUserChannel)

        # If a previous run of this test failed, this will load the state
        # of the failed test so that it is reproduced.
        self._maybe_load_state_and_randomize_channel(
            filename='Alt_Min_test_solve_state.pickle',
            iasolver=iasolver,
            Nr=Nr,
            Nt=Nt,
            K=K)

        iasolver.max_iterations = 200
        multiUserChannel.noise_var = 1e-10
        iasolver.solve(Ns, P)

        full_F0 = iasolver.full_F[0]
        full_F1 = iasolver.full_F[1]
        full_F2 = iasolver.full_F[2]

        full_W_H0 = iasolver.full_W_H[0]
        full_W_H1 = iasolver.full_W_H[1]
        full_W_H2 = iasolver.full_W_H[2]

        H00 = iasolver._get_channel(0, 0)
        H01 = iasolver._get_channel(0, 1)
        H02 = iasolver._get_channel(0, 2)
        H10 = iasolver._get_channel(1, 0)
        H11 = iasolver._get_channel(1, 1)
        H12 = iasolver._get_channel(1, 2)
        H20 = iasolver._get_channel(2, 0)
        H21 = iasolver._get_channel(2, 1)
        H22 = iasolver._get_channel(2, 2)

        # Perform the actual tests
        try:
            # xxxxx Test if the transmit power limit is respected xxxxxxxxx
            self.assertTrue(np.linalg.norm(full_F0, 'fro')**2 <= P[0] + 1e-12)
            self.assertTrue(np.linalg.norm(full_F1, 'fro')**2 <= P[1] + 1e-12)
            self.assertTrue(np.linalg.norm(full_F2, 'fro')**2 <= P[2] + 1e-12)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Test the equivalent channel xxxxxxxxxxxxxxxxxxxxxxxxxxx
            np.testing.assert_array_almost_equal(full_W_H0 @ H00 @ full_F0,
                                                 np.eye(Ns))
            np.testing.assert_array_almost_equal(full_W_H1 @ H11 @ full_F1,
                                                 np.eye(Ns))
            np.testing.assert_array_almost_equal(full_W_H2 @ H22 @ full_F2,
                                                 np.eye(Ns))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxxxxxxx test the remaining interference xxxxxxxxxxxxxxxxxx
            norm_value = np.linalg.norm(full_W_H0 @ H01 @ full_F1, 'fro')**2

            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H0 @ H02 @ full_F2, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H1 @ H10 @ full_F0, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H1 @ H12 @ full_F2, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H2 @ H20 @ full_F0, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H2 @ H21 @ full_F1, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))
        except AssertionError:  # pragma: nocover
            # Since this test failed, let's save its state so that we can
            # reproduce it
            self._save_state('Alt_Min_test_solve_state.pickle')
            raise  # re-raises the last exception
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # # print "Darlan"
            # print P
            # print np.linalg.norm(full_F0, 'fro')**2
            # print np.linalg.norm(full_F1, 'fro')**2
            # print np.linalg.norm(full_F2, 'fro')**2
            # # print
            # print np.linalg.norm(full_W_H0 * H01 * full_F1)**2
            # print np.linalg.norm(full_W_H0 * H02 * full_F2)**2
            # print np.linalg.norm(full_W_H1 * H10 * full_F0)**2
            # print np.linalg.norm(full_W_H1 * H12 * full_F2)**2
            # print np.linalg.norm(full_W_H2 * H20 * full_F0)**2
            # print np.linalg.norm(full_W_H2 * H21 * full_F1)**2

    # noinspection PyTypeChecker
    def test_calc_SINR_old(self):
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        iasolver = AlternatingMinIASolver(multiUserChannel)
        K = 3
        Nr = 4
        Nt = 4
        Ns = 2

        multiUserChannel.randomize(Nr, Nt, K)
        iasolver.max_iterations = 1
        iasolver.solve(Ns)

        SINRs = iasolver.calc_SINR_old()

        # Calculates the expected SINRs
        F0 = iasolver.F[0]
        F1 = iasolver.F[1]
        F2 = iasolver.F[2]

        W0 = iasolver.W[0]
        W1 = iasolver.W[1]
        W2 = iasolver.W[2]
        W0_H = iasolver.W_H[0]
        W1_H = iasolver.W_H[1]
        W2_H = iasolver.W_H[2]

        H00 = iasolver._get_channel(0, 0)
        H11 = iasolver._get_channel(1, 1)
        H22 = iasolver._get_channel(2, 2)

        H01 = iasolver._get_channel(0, 1)
        H02 = iasolver._get_channel(0, 2)
        H10 = iasolver._get_channel(1, 0)
        H12 = iasolver._get_channel(1, 2)
        H20 = iasolver._get_channel(2, 0)
        H21 = iasolver._get_channel(2, 1)

        expected_SINRs = np.empty(K, dtype=np.ndarray)

        # xxxxx k = 0 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        numerator0 = W0_H @ H00 @ F0
        numerator0 = numerator0 @ numerator0.T.conj()
        numerator0 = np.abs(np.diag(numerator0))

        denominator0 = W0_H @ H01 @ F1 + W0_H @ H02 @ F2
        denominator0 = denominator0 @ denominator0.T.conj()
        denominator0 = np.abs(np.diag(denominator0))

        expected_SINRs[0] = numerator0 / denominator0

        # xxxxx k = 1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        numerator1 = W1_H @ H11 @ F1
        numerator1 = numerator1 @ numerator1.T.conj()
        numerator1 = np.abs(np.diag(numerator1))

        denominator1 = W1_H @ H10 @ F0 + W1_H @ H12 @ F2
        denominator1 = denominator1 @ denominator1.T.conj()
        denominator1 = np.abs(np.diag(denominator1))
        expected_SINRs[1] = numerator1 / denominator1

        # xxxxx k = 2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        numerator2 = W2_H @ H22 @ F2
        numerator2 = numerator2 @ numerator2.T.conj()
        numerator2 = np.abs(np.diag(numerator2))

        denominator2 = W2_H @ H20 @ F0 + W2_H @ H21 @ F1
        denominator2 = denominator2 @ denominator2.T.conj()
        denominator2 = np.abs(np.diag(denominator2))
        expected_SINRs[2] = numerator2 / denominator2

        for k in range(K):
            np.testing.assert_array_almost_equal(SINRs[k], expected_SINRs[k])

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Repeat the calculation, but now including the noise
        noise_var = 1e-2
        multiUserChannel.noise_var = noise_var
        SINRs = iasolver.calc_SINR_old()

        expected_SINRs2 = np.empty(K, dtype=np.ndarray)

        # xxxxx k = 0 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        noise_term0 = W0_H @ W0 * noise_var
        denominator0_with_noise = denominator0 + np.abs(np.diag(noise_term0))
        expected_SINRs2[0] = numerator0 / denominator0_with_noise

        # xxxxx k = 1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        noise_term1 = W1_H @ W1 * noise_var
        denominator1_with_noise = denominator1 + np.abs(np.diag(noise_term1))
        expected_SINRs2[1] = numerator1 / denominator1_with_noise

        # xxxxx k = 2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        noise_term2 = W2_H @ W2 * noise_var
        denominator2_with_noise = denominator2 + np.abs(np.diag(noise_term2))
        expected_SINRs2[2] = numerator2 / denominator2_with_noise

        for k in range(K):
            np.testing.assert_array_almost_equal(SINRs[k], expected_SINRs2[k])

    def test_clear(self):
        Ns = 1
        self.iasolver.solve(Ns)
        self.iasolver.clear()
        self.assertIsNone(self.iasolver.F)
        self.assertIsNone(self.iasolver.W)
        self.assertIsNone(self.iasolver.W_H)
        self.assertIsNone(self.iasolver.full_W_H)
        self.assertAlmostEqual(self.iasolver.runned_iterations, 0.0)


# noinspection PyMethodMayBeStatic
class MaxSinrIASolverTestCase(CustomTestCase):
    def setUp(self):
        """Called before each test."""
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        self.iasolver = MaxSinrIASolver(multiUserChannel)

        self.K = 3
        self.Nt = np.ones(self.K, dtype=int) * 2
        self.Nr = np.ones(self.K, dtype=int) * 2
        self.Ns = np.ones(self.K, dtype=int) * 1

        # Transmit power of all users
        self.P = np.array([1.2, 1.5, 0.9])

        # Randomize the channel
        multiUserChannel.randomize(self.Nr, self.Nt, self.K)

    def test_calc_Bkl_cov_matrix_first_part_rev(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._updateW()

        for k in range(self.K):
            expected_first_part_rev = 0.0
            for j in range(self.K):
                aux = 0.0
                Hkj = self.iasolver._get_channel_rev(k, j)
                Hkj_H = Hkj.conjugate().transpose()

                for d in range(self.Ns[k]):
                    Vjd = self.iasolver._W[j][:, d:d + 1]
                    Vjd_H = Vjd.conjugate().transpose()
                    aux = aux + np.dot(np.dot(Hkj, np.dot(Vjd, Vjd_H)), Hkj_H)

                expected_first_part_rev += (self.P[j] / self.Ns[j]) * aux

            np.testing.assert_array_almost_equal(
                expected_first_part_rev,
                self.iasolver._calc_Bkl_cov_matrix_first_part_rev(k))

    # noinspection PyPep8
    def test_calc_Bkl_cov_matrix_second_part_rev(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._updateW()

        for k in range(self.K):
            Hkk = self.iasolver._get_channel_rev(k, k)
            Hkk_H = Hkk.transpose().conjugate()
            for l in range(self.Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = self.iasolver._W[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(
                    Hkk, np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
                expected_second_part = (self.P[k] / self.Ns[k]) * \
                                       expected_second_part
                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.iasolver._calc_Bkl_cov_matrix_second_part_rev(k, l))

    def test_calc_Bkl_cov_matrix_all_l(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._updateW()

        # Calculates Bkl for all streams (l index) of all users (k index)
        for k in range(self.K):
            # The first_part does not depend on the stream index. Only on
            # the user index.
            first_part = self.iasolver._calc_Bkl_cov_matrix_first_part(k)

            # xxxxx Calculates the Second Part xxxxxxxxxxxxxxxxxxxxxxxxxxxx
            expected_Bkl = np.empty(self.Ns[k], dtype=np.ndarray)
            for l in range(self.Ns[k]):
                # The second part depend on the user index and the stream
                # index.
                second_part \
                    = self.iasolver._calc_Bkl_cov_matrix_second_part(k, l)
                expected_Bkl[l] = first_part - second_part + np.eye(self.Nr[k])

            Bkl_all_l \
                = self.iasolver._calc_Bkl_cov_matrix_all_l(k, noise_power=1.0)

            # Test if the Bkl for all l of user k were calculated correctly
            for l in range(self.Ns[k]):
                np.testing.assert_array_almost_equal(expected_Bkl[l],
                                                     Bkl_all_l[l])

            # Repeat the test, but now without setting the noise variance
            # explicitly. It should use the self.noise_var property from
            # the IA solver class, which in turn will use the
            # noise_var property of the Mutiuser channel class.
            self.iasolver._multiUserChannel.noise_var = 0.14

            # xxxxx Calculates the Second Part xxxxxxxxxxxxxxxxxxxxxxxxxxxx
            expected_Bkl = np.empty(self.Ns[k], dtype=np.ndarray)
            for l in range(self.Ns[k]):
                # The second part depend on the user index and the stream
                # index.
                second_part \
                    = self.iasolver._calc_Bkl_cov_matrix_second_part(k, l)
                expected_Bkl[l] \
                    = first_part - second_part + 0.14 * np.eye(self.Nr[k])

            Bkl_all_l = self.iasolver._calc_Bkl_cov_matrix_all_l(k)

            # Test if the Bkl for all l of user k were calculated correctly
            for l in range(self.Ns[k]):
                np.testing.assert_array_almost_equal(expected_Bkl[l],
                                                     Bkl_all_l[l])

    def test_calc_Bkl_cov_matrix_all_l_rev(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._updateW()

        self.iasolver._multiUserChannel.noise_var = 1.0

        # Calculates Bkl for all streams (l index) of all users (k index)
        for k in range(self.K):
            # The first_part does not depend on the stream index. Only on
            # the user index.
            first_part = self.iasolver._calc_Bkl_cov_matrix_first_part_rev(k)

            # xxxxx Calculates the Second Part xxxxxxxxxxxxxxxxxxxxxxxxxxxx
            expected_Bkl = np.empty(self.Ns[k], dtype=np.ndarray)
            for l in range(self.Ns[k]):
                # The second part depend on the user index and the stream
                # index.
                second_part \
                    = self.iasolver._calc_Bkl_cov_matrix_second_part_rev(k, l)
                expected_Bkl[l] = first_part - second_part + np.eye(self.Nr[k])

            Bkl_all_l = self.iasolver._calc_Bkl_cov_matrix_all_l_rev(k)

            # Test if the Bkl for all l of user k were calculated correctly
            for l in range(self.Ns[k]):
                np.testing.assert_array_almost_equal(expected_Bkl[l],
                                                     Bkl_all_l[l])

    def test_calc_Ukl(self):
        self.iasolver.randomizeF(self.Ns, self.P)

        for k in range(self.K):
            Hkk = self.iasolver._get_channel(k, k)
            Bkl_all_l = self.iasolver._calc_Bkl_cov_matrix_all_l(k)
            F = self.iasolver.full_F[k]
            for l in range(self.Ns[k]):
                expected_Ukl = np.dot(np.linalg.inv(Bkl_all_l[l]),
                                      np.dot(Hkk, F[:, l:l + 1]))
                expected_Ukl /= norm(expected_Ukl, 'fro')
                Ukl = self.iasolver._calc_Ukl(Hkk, F, Bkl_all_l[l], l)
                np.testing.assert_array_almost_equal(expected_Ukl, Ukl)

    def teste_calc_Uk(self):
        self.iasolver.randomizeF(self.Ns, self.P)

        for k in range(self.K):
            Bkl_all_l = self.iasolver._calc_Bkl_cov_matrix_all_l(k)
            Hkk = self.iasolver._get_channel(k, k)
            Vk = self.iasolver.full_F[k]
            Uk = self.iasolver._calc_Uk(Hkk, Vk, Bkl_all_l)

            expected_Uk = np.empty([self.Nr[k], self.Ns[k]], dtype=complex)
            for l in range(self.Ns[k]):
                expected_Uk[:, l] = self.iasolver._calc_Ukl(
                    Hkk, Vk, Bkl_all_l[l], l)[:, 0]
            np.testing.assert_array_almost_equal(expected_Uk, Uk)

    def test_underline_calc_SINR_k(self):
        self.iasolver.randomizeF(self.Ns, self.P)

        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        iasolver = MaxSinrIASolver(multiUserChannel)
        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 2

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(Nr, Nt, K)
        iasolver.randomizeF(Ns, P)
        iasolver._updateW()

        for k in range(K):
            Hkk = iasolver._get_channel(k, k)
            Bkl_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k)
            Uk = iasolver.full_W[k]
            SINR_k_all_l = iasolver._calc_SINR_k(k, Bkl_all_l)

            for l in range(Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = iasolver.full_F[k][:, l:l + 1]
                aux = np.dot(Ukl_H, np.dot(Hkk, Vkl))

                expectedSINRkl = (
                    np.dot(aux,
                           aux.transpose().conjugate()) /
                    np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))).item()

                self.assertAlmostEqual(expectedSINRkl, SINR_k_all_l[l])

    def test_calc_SINR(self):
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        iasolver = MaxSinrIASolver(multiUserChannel)
        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 2

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(Nr, Nt, K)
        iasolver.randomizeF(Ns, P)
        iasolver._updateW()

        SINR_all_users = iasolver.calc_SINR()
        SINR_all_users_in_dB = iasolver.calc_SINR_in_dB()

        # xxxxxxxxxx Noise Variance of 0.0 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # k = 0
        B0l_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k=0, noise_power=0.0)
        expected_SINR0 = iasolver._calc_SINR_k(0, B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])
        np.testing.assert_almost_equal(linear2dB(expected_SINR0),
                                       SINR_all_users_in_dB[0])

        # k = 1
        B1l_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k=1, noise_power=0.0)
        expected_SINR1 = iasolver._calc_SINR_k(1, B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])
        np.testing.assert_almost_equal(linear2dB(expected_SINR1),
                                       SINR_all_users_in_dB[1])

        # k = 1
        B2l_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k=2, noise_power=0.0)
        expected_SINR2 = iasolver._calc_SINR_k(2, B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        np.testing.assert_almost_equal(linear2dB(expected_SINR2),
                                       SINR_all_users_in_dB[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Noise Variance of 0.1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # k = 0
        multiUserChannel.noise_var = 0.1
        SINR_all_users = iasolver.calc_SINR()
        SINR_all_users_in_dB = iasolver.calc_SINR_in_dB()

        B0l_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k=0, noise_power=0.1)
        expected_SINR0 = iasolver._calc_SINR_k(0, B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])
        np.testing.assert_almost_equal(linear2dB(expected_SINR0),
                                       SINR_all_users_in_dB[0])

        # k = 1
        B1l_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k=1, noise_power=0.1)
        expected_SINR1 = iasolver._calc_SINR_k(1, B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])
        np.testing.assert_almost_equal(linear2dB(expected_SINR1),
                                       SINR_all_users_in_dB[1])

        # k = 1
        B2l_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k=2, noise_power=0.1)
        expected_SINR2 = iasolver._calc_SINR_k(2, B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        np.testing.assert_almost_equal(linear2dB(expected_SINR2),
                                       SINR_all_users_in_dB[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_get_channel_rev(self):

        for k in range(self.K):
            for l in range(self.K):
                Hlk = self.iasolver._get_channel(l, k)
                expected_Hkl_rev = Hlk.transpose().conjugate()
                Hkl_rev = self.iasolver._get_channel_rev(k, l)
                np.testing.assert_array_almost_equal(expected_Hkl_rev, Hkl_rev)

    def test_calc_Uk_all_k(self):
        self.iasolver.randomizeF(self.Ns, self.P)

        Uk = self.iasolver._calc_Uk_all_k()

        for k in range(self.K):
            Hkk = self.iasolver._get_channel(k, k)
            Vk = self.iasolver.full_F[k]
            Bkl_all_l = self.iasolver._calc_Bkl_cov_matrix_all_l(k)
            expectedUk = self.iasolver._calc_Uk(Hkk, Vk, Bkl_all_l)
            np.testing.assert_array_almost_equal(Uk[k], expectedUk)

    def test_calc_Uk_all_k_rev(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._updateW()

        Uk = self.iasolver._calc_Uk_all_k_rev()

        for k in range(self.K):
            Hkk = self.iasolver._get_channel_rev(k, k)
            Vk = self.iasolver._W[k]
            Bkl_all_l = self.iasolver._calc_Bkl_cov_matrix_all_l_rev(k)
            expectedUk = self.iasolver._calc_Uk(Hkk, Vk, Bkl_all_l)
            np.testing.assert_array_almost_equal(Uk[k], expectedUk)

    # Test the calc_Q_rev method from IASolverBaseClass
    def test_calc_Q_rev(self):
        K = 3
        Nt = np.array([2, 2, 2])
        Nr = np.array([3, 3, 3])
        Ns = np.array([1, 1, 1])
        multiUserChannel = self.iasolver._multiUserChannel
        multiUserChannel.noise_var = 1e-10

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        self._maybe_load_state_and_randomize_channel(
            filename='MaxSINR_test_calc_Q_state.pickle',
            iasolver=self.iasolver,
            Nr=Nr,
            Nt=Nt,
            K=K)

        multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver.randomizeF(Ns, P)

        try:
            self.iasolver._W = self.iasolver._calc_Uk_all_k()
        except Exception:  # pragma: no cover
            self._save_state(filename='MaxSINR_test_calc_Q_state.pickle')
            raise  # re-raises the last exception

        # xxxxx Calculate the expected Q[0]_rev after one step xxxxxxxxxxxx
        k = 0
        H01_F1_rev = np.dot(self.iasolver._get_channel_rev(k, 1),
                            self.iasolver._W[1])
        H02_F2_rev = np.dot(self.iasolver._get_channel_rev(k, 2),
                            self.iasolver._W[2])
        expected_Q0_rev = (np.dot(P[1] * H01_F1_rev,
                                  H01_F1_rev.transpose().conjugate()) +
                           np.dot(P[2] * H02_F2_rev,
                                  H02_F2_rev.transpose().conjugate()))

        Q0_rev = self.iasolver.calc_Q_rev(k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Q0_rev, expected_Q0_rev)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0_rev = np.dot(self.iasolver._get_channel_rev(k, 0),
                            self.iasolver._W[0])
        H12_F2_rev = np.dot(self.iasolver._get_channel_rev(k, 2),
                            self.iasolver._W[2])
        expected_Q1_rev = (np.dot(P[0] * H10_F0_rev,
                                  H10_F0_rev.transpose().conjugate()) +
                           np.dot(P[2] * H12_F2_rev,
                                  H12_F2_rev.transpose().conjugate()))

        Q1_rev = self.iasolver.calc_Q_rev(k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Q1_rev, expected_Q1_rev)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H20_F0_rev = np.dot(self.iasolver._get_channel_rev(k, 0),
                            self.iasolver._W[0])
        H21_F1_rev = np.dot(self.iasolver._get_channel_rev(k, 1),
                            self.iasolver._W[1])
        expected_Q2_rev = (np.dot(P[0] * H20_F0_rev,
                                  H20_F0_rev.transpose().conjugate()) +
                           np.dot(P[1] * H21_F1_rev,
                                  H21_F1_rev.transpose().conjugate()))

        Q2_rev = self.iasolver.calc_Q_rev(k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Q2_rev, expected_Q2_rev)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_updateW(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._updateW()

        for k in range(self.iasolver.K):
            np.testing.assert_array_almost_equal(
                self.iasolver.W[k],
                self.iasolver._calc_Uk_all_k()[k])

        # xxxxx Sanity tests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test if the hermitian of W[k] is equal to W[k]
        for k in range(self.iasolver.K):
            self.assertIsNotNone(self.iasolver.W[k])
            np.testing.assert_array_almost_equal(self.iasolver.W[k].conj().T,
                                                 self.iasolver.W_H[k])
        # Perform one step and test again
        self.iasolver._step()
        for k in range(self.iasolver.K):
            self.assertIsNotNone(self.iasolver.W[k])
            np.testing.assert_array_almost_equal(self.iasolver.W[k].conj().T,
                                                 self.iasolver.W_H[k])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        self.iasolver._clear_receive_filter()
        self.assertIsNone(self.iasolver.W)
        self.assertIsNone(self.iasolver.W_H)
        self.assertIsNone(self.iasolver.full_W_H)

    def test_full_W_H_property(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._updateW()

        self.iasolver._step()
        full_F = self.iasolver.full_F
        full_W_H = self.iasolver.full_W_H
        H00 = self.iasolver._get_channel(0, 0)
        H11 = self.iasolver._get_channel(1, 1)
        H22 = self.iasolver._get_channel(2, 2)
        self.assertAlmostEqual(
            np.dot(full_W_H[0], np.dot(H00, full_F[0]))[0][0], 1.0)
        self.assertAlmostEqual(
            np.dot(full_W_H[1], np.dot(H11, full_F[1]))[0][0], 1.0)
        self.assertAlmostEqual(
            np.dot(full_W_H[2], np.dot(H22, full_F[2]))[0][0], 1.0)

    def test_solve(self):
        K = 3
        Nt = np.ones(K, dtype=int) * 2
        Nr = np.ones(K, dtype=int) * 2
        Ns = np.ones(K, dtype=int) * 1

        # Transmit power of all users
        P = np.array([2.0, 1.5, 0.9])

        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        iasolver = MaxSinrIASolver(multiUserChannel)

        # If a previous run of this test failed, this will load the state
        # of the failed test so that it is reproduced.
        self._maybe_load_state_and_randomize_channel(
            filename='MaxSINR_test_solve_state.pickle',
            iasolver=iasolver,
            Nr=Nr,
            Nt=Nt,
            K=K)

        multiUserChannel.noise_var = 1e-4
        iasolver.max_iterations = 200
        try:
            iasolver.randomizeF(Ns, P)
            iasolver.initialize_with = 'fix'
            iasolver.solve(Ns, P)
        except Exception:  # pragma: no cover
            self._save_state(filename='MaxSINR_test_solve_state.pickle')
            raise  # re-raises the last exception

        H00 = iasolver._get_channel(0, 0)
        H11 = iasolver._get_channel(1, 1)
        H22 = iasolver._get_channel(2, 2)
        H01 = iasolver._get_channel(0, 1)
        H02 = iasolver._get_channel(0, 2)
        H10 = iasolver._get_channel(1, 0)
        H12 = iasolver._get_channel(1, 2)
        H20 = iasolver._get_channel(2, 0)
        H21 = iasolver._get_channel(2, 1)

        full_F0 = iasolver.full_F[0]
        full_F1 = iasolver.full_F[1]
        full_F2 = iasolver.full_F[2]
        full_W_H0 = iasolver.full_W_H[0]
        full_W_H1 = iasolver.full_W_H[1]
        full_W_H2 = iasolver.full_W_H[2]

        # Perform the actual tests
        try:
            # xxxxx Test the equivalent channel xxxxxxxxxxxxxxxxxxxxxxxxxxx
            np.testing.assert_array_almost_equal(full_W_H0 @ H00 @ full_F0,
                                                 np.eye(Ns[0]))
            np.testing.assert_array_almost_equal(full_W_H1 @ H11 @ full_F1,
                                                 np.eye(Ns[1]))
            np.testing.assert_array_almost_equal(full_W_H2 @ H22 @ full_F2,
                                                 np.eye(Ns[2]))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxxxxxxx test the remaining interference xxxxxxxxxxxxxxxxxxxxxx
            norm_value = np.linalg.norm(full_W_H0 @ H01 @ full_F1, 'fro')**2
            self.assertTrue(norm_value < 0.1,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H0 @ H02 @ full_F2, 'fro')**2
            self.assertTrue(norm_value < 0.1,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H1 @ H10 @ full_F0, 'fro')**2
            self.assertTrue(norm_value < 0.1,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H1 @ H12 @ full_F2, 'fro')**2
            self.assertTrue(norm_value < 0.1,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H2 @ H20 @ full_F0, 'fro')**2
            self.assertTrue(norm_value < 0.1,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H2 @ H21 @ full_F1, 'fro')**2
            self.assertTrue(norm_value < 0.1,
                            msg="Norm Value: {0}".format(norm_value))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        except AssertionError:  # pragma: nocover
            # Since this test failed, let's save its state so that we can
            # reproduce it
            self._save_state(filename='MaxSINR_test_solve_state.pickle')
            raise  # re-raises the last exception

            # print "Precoder Powers"
            # print np.linalg.norm(full_F0, 'fro')**2
            # print np.linalg.norm(full_F1, 'fro')**2
            # print np.linalg.norm(full_F2, 'fro')**2

            # print "Direct Links"
            # print np.linalg.norm(full_W_H0 * H00 * full_F0)**2
            # print np.linalg.norm(full_W_H1 * H11 * full_F1)**2
            # print np.linalg.norm(full_W_H2 * H22 * full_F2)**2

            # print "Interfering Links power"
            # print np.linalg.norm(full_W_H0 * H01 * full_F1)**2
            # print np.linalg.norm(full_W_H0 * H02 * full_F2)**2
            # print np.linalg.norm(full_W_H1 * H10 * full_F0)**2
            # print np.linalg.norm(full_W_H1 * H12 * full_F2)**2
            # print np.linalg.norm(full_W_H2 * H20 * full_F0)**2
            # print np.linalg.norm(full_W_H2 * H21 * full_F1)**2

            # print
            # print map(linear2dB, iasolver.calc_SINR())
            # print iasolver.initialize_with

    def test_solve_finalize(self):
        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 2

        # Transmit power of all users. We set the power of the first user
        # to a very low value so that the ia solver sets 0 energy to one of
        # the streams (due to the waterfilling algorithm deciding is is
        # better to focus all the energy into one stream). This will make
        # the code in the _solve_finalize method to reduce the number of
        # streams of the first user, which we will test here.
        P = np.array([0.0001, 100.8, 230.0])

        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        multiUserChannel.randomize(Nr, Nt, K)
        multiUserChannel.noise_var = 1e-8

        iasolver = MaxSinrIASolver(multiUserChannel)

        iasolver.solve(Ns, P)

        self.assertEqual(iasolver.F[0].shape, (4, 1))
        self.assertEqual(iasolver.F[1].shape, (4, 2))
        self.assertEqual(iasolver.F[2].shape, (4, 2))

        self.assertEqual(iasolver.W[0].shape, (4, 1))
        self.assertEqual(iasolver.W[1].shape, (4, 2))
        self.assertEqual(iasolver.W[2].shape, (4, 2))

        self.assertEqual(iasolver.full_F[0].shape, (4, 1))
        self.assertEqual(iasolver.full_F[1].shape, (4, 2))
        self.assertEqual(iasolver.full_F[2].shape, (4, 2))

        self.assertEqual(iasolver.full_W[0].shape, (4, 1))
        self.assertEqual(iasolver.full_W[1].shape, (4, 2))
        self.assertEqual(iasolver.full_W[2].shape, (4, 2))

        np.testing.assert_array_equal(iasolver.Ns, np.array([1, 2, 2]))
        # The Ns array passed to the IA solver object should not be
        # changed.
        np.testing.assert_array_equal(Ns, np.array([2, 2, 2]))


# TODO: Finish the implementation
class MinLeakageIASolverTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        self.iasolver = MinLeakageIASolver(multiUserChannel)
        self.K = 3
        self.Nt = np.ones(self.K, dtype=int) * 2
        self.Nr = np.ones(self.K, dtype=int) * 2
        self.Ns = np.ones(self.K, dtype=int) * 1

        # Transmit power of all users
        self.P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(self.Nr, self.Nt, self.K)

    def test_getCost(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._W = self.iasolver._calc_Uk_all_k()

        Q0 = self.iasolver.calc_Q(0)
        W0 = self.iasolver._W[0]
        Q1 = self.iasolver.calc_Q(1)
        W1 = self.iasolver._W[1]
        Q2 = self.iasolver.calc_Q(2)
        W2 = self.iasolver._W[2]
        expected_cost = np.trace(
            np.abs(W0.T.conj() @ Q0 @ W0 + W1.T.conj() @ Q1 @ W1 +
                   W2.T.conj() @ Q2 @ W2))
        self.assertAlmostEqual(expected_cost, self.iasolver.get_cost())

        self.iasolver._step()
        Q0 = self.iasolver.calc_Q(0)
        W0 = self.iasolver._W[0]
        Q1 = self.iasolver.calc_Q(1)
        W1 = self.iasolver._W[1]
        Q2 = self.iasolver.calc_Q(2)
        W2 = self.iasolver._W[2]
        expected_cost2 = np.trace(
            np.abs(W0.T.conj() @ Q0 @ W0 + W1.T.conj() @ Q1 @ W1 +
                   W2.T.conj() @ Q2 @ W2))
        self.assertAlmostEqual(expected_cost2, self.iasolver.get_cost())

        self.assertTrue(expected_cost2 < expected_cost)

    def test_calc_Uk_all_k(self):
        self.iasolver.randomizeF(self.Ns, self.P)

        Uk_all = self.iasolver._calc_Uk_all_k()

        # xxxxxxxxxx First User xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 0
        Qk = self.iasolver.calc_Q(k)
        [expected_Uk0, _] = leig(Qk, self.Ns[k])
        np.testing.assert_array_almost_equal(expected_Uk0, Uk_all[k])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Second User xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 1
        Qk = self.iasolver.calc_Q(k)
        [expected_Uk1, _] = leig(Qk, self.Ns[k])
        np.testing.assert_array_almost_equal(expected_Uk1, Uk_all[k])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Third user xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 2
        Qk = self.iasolver.calc_Q(k)
        [expected_Uk2, _] = leig(Qk, self.Ns[k])
        np.testing.assert_array_almost_equal(expected_Uk2, Uk_all[k])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_calc_Uk_all_k_rev(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._W = self.iasolver._calc_Uk_all_k()

        Uk_all = self.iasolver._calc_Uk_all_k_rev()

        # xxxxxxxxxx First User xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 0
        Qk_rev = self.iasolver.calc_Q_rev(k)
        [expected_Uk0_rev, _] = leig(Qk_rev, self.Ns[k])
        np.testing.assert_array_almost_equal(expected_Uk0_rev, Uk_all[k])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Second User xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 1
        Qk_rev = self.iasolver.calc_Q_rev(k)
        [expected_Uk1_rev, _] = leig(Qk_rev, self.Ns[k])
        np.testing.assert_array_almost_equal(expected_Uk1_rev, Uk_all[k])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Third user xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 2
        Qk_rev = self.iasolver.calc_Q_rev(k)
        [expected_Uk2_rev, _] = leig(Qk_rev, self.Ns[k])
        np.testing.assert_array_almost_equal(expected_Uk2_rev, Uk_all[k])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_step(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._W = self.iasolver._calc_Uk_all_k()

        last_cost = self.iasolver.get_cost()
        for _ in range(5):
            self.iasolver._step()
            new_cost = self.iasolver.get_cost()
            self.assertTrue(new_cost < last_cost)
            last_cost = new_cost

    def test_updateW(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._updateW()

        for k in range(self.iasolver.K):
            np.testing.assert_array_almost_equal(
                self.iasolver.W[k],
                self.iasolver._calc_Uk_all_k()[k])

        # xxxxx Sanity tests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test if the hermitian of W[k] is equal to W[k]
        for k in range(self.iasolver.K):
            self.assertIsNotNone(self.iasolver.W[k])
            np.testing.assert_array_almost_equal(self.iasolver.W[k].conj().T,
                                                 self.iasolver.W_H[k])
        # Perform one step and test again
        self.iasolver._step()
        for k in range(self.iasolver.K):
            self.assertIsNotNone(self.iasolver.W[k])
            np.testing.assert_array_almost_equal(self.iasolver.W[k].conj().T,
                                                 self.iasolver.W_H[k])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        self.iasolver._clear_receive_filter()
        self.assertIsNone(self.iasolver.W)
        self.assertIsNone(self.iasolver.W_H)
        self.assertIsNone(self.iasolver.full_W_H)

    def test_full_W_H_property(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._W = self.iasolver._calc_Uk_all_k()

        self.iasolver._step()
        full_F = self.iasolver.full_F
        full_W_H = self.iasolver.full_W_H
        H00 = self.iasolver._get_channel(0, 0)
        H11 = self.iasolver._get_channel(1, 1)
        H22 = self.iasolver._get_channel(2, 2)
        self.assertAlmostEqual(
            np.dot(full_W_H[0], np.dot(H00, full_F[0]))[0][0], 1.0)
        self.assertAlmostEqual(
            np.dot(full_W_H[1], np.dot(H11, full_F[1]))[0][0], 1.0)
        self.assertAlmostEqual(
            np.dot(full_W_H[2], np.dot(H22, full_F[2]))[0][0], 1.0)

    def test_solve(self):
        self.iasolver.max_iterations = 1
        # We are only testing if this does not thrown an exception. That's
        # why there is no assert clause here
        self.iasolver.solve(Ns=1)


# noinspection PyMethodMayBeStatic
class MMSEIASolverTestCase(CustomTestCase):
    def setUp(self):
        """Called before each test."""
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        self.iasolver = MMSEIASolver(multiUserChannel)

        self.K = 3
        self.Nt = np.ones(self.K, dtype=int) * 4
        self.Nr = np.ones(self.K, dtype=int) * 4
        self.Ns = np.ones(self.K, dtype=int) * 2

        # Transmit power of all users
        self.P = np.array([1.2, 1.5, 0.9])

        # Randomize the channel
        multiUserChannel.randomize(self.Nr, self.Nt, self.K)

    def test_updateW(self):
        self.iasolver._multiUserChannel.noise_var = 1e-2
        P = self.P
        # self.iasolver._initialize_F_and_W_from_closed_form(self.Ns, self.P)
        self.iasolver._initialize_F_randomly_and_find_W(self.Ns, P)

        np.testing.assert_array_almost_equal(self.iasolver.P, P)

        F0 = self.iasolver.F[0]
        F1 = self.iasolver.F[1]
        F2 = self.iasolver.F[2]

        H00 = self.iasolver._get_channel(0, 0)
        H11 = self.iasolver._get_channel(1, 1)
        H22 = self.iasolver._get_channel(2, 2)
        H01 = self.iasolver._get_channel(0, 1)
        H02 = self.iasolver._get_channel(0, 2)
        H10 = self.iasolver._get_channel(1, 0)
        H12 = self.iasolver._get_channel(1, 2)
        H20 = self.iasolver._get_channel(2, 0)
        H21 = self.iasolver._get_channel(2, 1)

        # xxxxx Calculates the expected receive filter for the user 0 xxxxx
        H00_F0 = np.sqrt(P[0]) * np.dot(H00, F0)
        H01_F1 = np.sqrt(P[1]) * np.dot(H01, F1)
        H02_F2 = np.sqrt(P[2]) * np.dot(H02, F2)
        sum0 = (np.dot(H00_F0,
                       H00_F0.conj().T) + np.dot(H01_F1,
                                                 H01_F1.conj().T) +
                np.dot(H02_F2,
                       H02_F2.conj().T))
        expected_W0 = np.dot(
            np.linalg.inv(sum0 + self.iasolver.noise_var * np.eye(self.Nr[0])),
            H00_F0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for the user 1 xxxxx
        H11_F1 = np.sqrt(P[1]) * np.dot(H11, F1)
        H10_F0 = np.sqrt(P[0]) * np.dot(H10, F0)
        # H11_F2 = np.sqrt(P[1]) * np.dot(H11, F1)
        H12_F2 = np.sqrt(P[2]) * np.dot(H12, F2)
        sum1 = (np.dot(H10_F0,
                       H10_F0.conj().T) + np.dot(H11_F1,
                                                 H11_F1.conj().T) +
                np.dot(H12_F2,
                       H12_F2.conj().T))
        expected_W1 = np.dot(
            np.linalg.inv(sum1 + self.iasolver.noise_var * np.eye(self.Nr[1])),
            H11_F1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for the user 2 xxxxx
        # H22_F2 = np.sqrt(P[2]) * np.dot(H22, F2)
        H20_F0 = np.sqrt(P[0]) * np.dot(H20, F0)
        H21_F1 = np.sqrt(P[1]) * np.dot(H21, F1)
        H22_F2 = np.sqrt(P[2]) * np.dot(H22, F2)
        sum2 = (np.dot(H20_F0,
                       H20_F0.conj().T) + np.dot(H21_F1,
                                                 H21_F1.conj().T) +
                np.dot(H22_F2,
                       H22_F2.conj().T))
        expected_W2 = np.dot(
            np.linalg.inv(sum2 + self.iasolver.noise_var * np.eye(self.Nr[1])),
            H22_F2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Update the receive filters
        self.iasolver._updateW()

        # Test if the update was performed correctly
        np.testing.assert_array_almost_equal(self.iasolver.W[0], expected_W0)
        np.testing.assert_array_almost_equal(self.iasolver.W[1], expected_W1)
        np.testing.assert_array_almost_equal(self.iasolver.W[2], expected_W2)

    # noinspection PyTypeChecker
    def test_updateW_with_very_small_power(self):
        self.iasolver._multiUserChannel.noise_var = 1e-22
        P = self.P * 1e-20
        # self.iasolver._initialize_F_and_W_from_closed_form(self.Ns, self.P)
        self.iasolver._initialize_F_randomly_and_find_W(self.Ns, P)

        np.testing.assert_array_almost_equal(self.iasolver.P, P)

        F0 = self.iasolver.F[0]
        F1 = self.iasolver.F[1]
        F2 = self.iasolver.F[2]

        H00 = self.iasolver._get_channel(0, 0)
        H11 = self.iasolver._get_channel(1, 1)
        H22 = self.iasolver._get_channel(2, 2)
        H01 = self.iasolver._get_channel(0, 1)
        H02 = self.iasolver._get_channel(0, 2)
        H10 = self.iasolver._get_channel(1, 0)
        H12 = self.iasolver._get_channel(1, 2)
        H20 = self.iasolver._get_channel(2, 0)
        H21 = self.iasolver._get_channel(2, 1)

        # xxxxx Calculates the expected receive filter for the user 0 xxxxx
        H00_F0 = np.sqrt(P[0]) * np.dot(H00, F0)
        H01_F1 = np.sqrt(P[1]) * np.dot(H01, F1)
        H02_F2 = np.sqrt(P[2]) * np.dot(H02, F2)
        sum0 = (np.dot(H00_F0,
                       H00_F0.conj().T) + np.dot(H01_F1,
                                                 H01_F1.conj().T) +
                np.dot(H02_F2,
                       H02_F2.conj().T))
        expected_W0 = np.dot(
            np.linalg.inv(sum0 + self.iasolver.noise_var * np.eye(self.Nr[0])),
            H00_F0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for the user 1 xxxxx
        # H11_F1 = np.sqrt(P[1]) * np.dot(H11, F1)
        H10_F0 = np.sqrt(P[0]) * np.dot(H10, F0)
        H11_F1 = np.sqrt(P[1]) * np.dot(H11, F1)
        H12_F2 = np.sqrt(P[2]) * np.dot(H12, F2)
        sum1 = (np.dot(H10_F0,
                       H10_F0.conj().T) + np.dot(H11_F1,
                                                 H11_F1.conj().T) +
                np.dot(H12_F2,
                       H12_F2.conj().T))
        expected_W1 = np.dot(
            np.linalg.inv(sum1 + self.iasolver.noise_var * np.eye(self.Nr[1])),
            H11_F1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for the user 2 xxxxx
        # H22_F2 = np.sqrt(P[2]) * np.dot(H22, F2)
        H20_F0 = np.sqrt(P[0]) * np.dot(H20, F0)
        H21_F1 = np.sqrt(P[1]) * np.dot(H21, F1)
        H22_F2 = np.sqrt(P[2]) * np.dot(H22, F2)
        sum2 = (np.dot(H20_F0,
                       H20_F0.conj().T) + np.dot(H21_F1,
                                                 H21_F1.conj().T) +
                np.dot(H22_F2,
                       H22_F2.conj().T))
        expected_W2 = np.dot(
            np.linalg.inv(sum2 + self.iasolver.noise_var * np.eye(self.Nr[1])),
            H22_F2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Update the receive filters
        self.iasolver._updateW()

        # Test if the update was performed correctly
        np.testing.assert_array_almost_equal(1e-10 * self.iasolver.W[0],
                                             1e-10 * expected_W0)
        np.testing.assert_array_almost_equal(1e-10 * self.iasolver.W[1],
                                             1e-10 * expected_W1)
        np.testing.assert_array_almost_equal(1e-10 * self.iasolver.W[2],
                                             1e-10 * expected_W2)

    def test_calc_Vi_for_a_given_mu(self):
        sum_term = randn_c(3, 3)
        sum_term = sum_term.dot(sum_term.conj().T)
        ":type: np.ndarray"

        mu = 0.135
        H_herm_U = randn_c(3, 2)

        expected_vi = np.dot(np.linalg.inv(sum_term + mu * np.eye(3)),
                             H_herm_U)

        vi = MMSEIASolver._calc_Vi_for_a_given_mu(sum_term, mu, H_herm_U)
        vi2 = MMSEIASolver._calc_Vi_for_a_given_mu2(np.linalg.inv(sum_term),
                                                    mu, H_herm_U)

        np.testing.assert_array_almost_equal(expected_vi, vi)
        np.testing.assert_array_almost_equal(expected_vi, vi2)

    def test_calc_Vi(self):
        # If a previous run of this test failed, this will load the state
        # of the failed test so that it is reproduced.
        self._maybe_load_state_and_randomize_channel(
            filename='MMSE_test_calc_Vi_state.pickle',
            iasolver=self.iasolver,
            Nr=self.Nr,
            Nt=self.Nt,
            K=self.K)

        # For now we use an arbitrarily chosen value
        mu = np.array([0.9, 1.1, 0.8])
        self.iasolver.initialize_with_closed_form = True
        self.iasolver._solve_init(self.Ns, self.P)

        U0 = self.iasolver.W[0]
        U1 = self.iasolver.W[1]
        U2 = self.iasolver.W[2]

        H00 = self.iasolver._get_channel(0, 0)
        H11 = self.iasolver._get_channel(1, 1)
        H22 = self.iasolver._get_channel(2, 2)
        H01 = self.iasolver._get_channel(0, 1)
        H02 = self.iasolver._get_channel(0, 2)
        H10 = self.iasolver._get_channel(1, 0)
        H12 = self.iasolver._get_channel(1, 2)
        H20 = self.iasolver._get_channel(2, 0)
        H21 = self.iasolver._get_channel(2, 1)

        # xxxxx Calculates the expected precoder for the user 0 xxxxxxxxxxx
        aux0 = 0.0
        H00_herm_U0 = np.dot(H00.conj().T, U0)

        aux = np.dot(H00.conj().T, U0)
        aux0 = aux0 + np.dot(aux, aux.conj().T)

        aux = np.dot(H10.conj().T, U1)
        aux0 = aux0 + np.dot(aux, aux.conj().T)

        aux = np.dot(H20.conj().T, U2)
        aux0 = aux0 + np.dot(aux, aux.conj().T)

        expected_V0 = np.dot(np.linalg.inv(aux0 + mu[0] * np.eye(self.Nt[0])),
                             H00_herm_U0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected precoder for the user 1 xxxxxxxxxxx
        aux1 = 0.0
        H11_herm_U1 = np.dot(H11.conj().T, U1)

        aux = np.dot(H01.conj().T, U0)
        aux1 = aux1 + np.dot(aux, aux.conj().T)

        aux = np.dot(H11.conj().T, U1)
        aux1 = aux1 + np.dot(aux, aux.conj().T)

        aux = np.dot(H21.conj().T, U2)
        aux1 = aux1 + np.dot(aux, aux.conj().T)

        expected_V1 = np.dot(np.linalg.inv(aux1 + mu[1] * np.eye(self.Nt[1])),
                             H11_herm_U1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected precoder for the user 2 xxxxxxxxxxx
        aux2 = 0.0
        H22_herm_U2 = np.dot(H22.conj().T, U2)

        aux = np.dot(H02.conj().T, U0)
        aux2 = aux2 + np.dot(aux, aux.conj().T)

        aux = np.dot(H12.conj().T, U1)
        aux2 = aux2 + np.dot(aux, aux.conj().T)

        aux = np.dot(H22.conj().T, U2)
        aux2 = aux2 + np.dot(aux, aux.conj().T)

        expected_V2 = np.dot(np.linalg.inv(aux2 + mu[2] * np.eye(self.Nt[2])),
                             H22_herm_U2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Calculates the precoders using the _calc_Vi method
        V0 = self.iasolver._calc_Vi(0, mu[0])
        V1 = self.iasolver._calc_Vi(1, mu[1])
        V2 = self.iasolver._calc_Vi(2, mu[2])

        try:
            # Test if the calculated values are equal to the expected values
            np.testing.assert_array_almost_equal(expected_V0, V0)
            np.testing.assert_array_almost_equal(expected_V1, V1)
            np.testing.assert_array_almost_equal(expected_V2, V2)

            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            # Now lets repeat the tests, but without specifying the value of
            # mu. In that case, the optimum value of mu is also calculated in
            # the _calc_Vi method. Since it is hard to test this here we will
            # only test if the power of the calculated precoders respect the
            # power constraint.
            V0_best = self.iasolver._calc_Vi(0)
            V1_best = self.iasolver._calc_Vi(1)
            V2_best = self.iasolver._calc_Vi(2)

            self.assertTrue(
                np.linalg.norm(V0_best, 'fro')**2 <= 1.0000000001 * self.P[0])
            self.assertTrue(
                np.linalg.norm(V1_best, 'fro')**2 <= 1.0000000001 * self.P[1])
            self.assertTrue(
                np.linalg.norm(V2_best, 'fro')**2 <= 1.0000000001 * self.P[2])

            # TODO: Find a way to test the case when the best value of mu
            # is found
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        except AssertionError:  # pragma: nocover
            # Since this test failed, let's save its state so that we can
            # reproduce it
            self._save_state(filename='MMSE_test_calc_Vi_state.pickle')
            raise  # re-raises the last exception

    def test_updateF(self):
        # We are only testing the transmit powers here. If the precoders
        # are not calculated correctly then the test for the solve method
        # should fail.
        Ns = 2
        P = np.array([0.67, 0.89, 1.1])
        self.iasolver._multiUserChannel.noise_var = 0.5
        self.iasolver._initialize_F_randomly_and_find_W(Ns, P)
        self.iasolver._updateF()

        self.assertAlmostEqual(norm(self.iasolver.F[0], 'fro')**2, 1.0)
        self.assertAlmostEqual(norm(self.iasolver.F[1], 'fro')**2, 1.0)
        self.assertAlmostEqual(norm(self.iasolver.F[2], 'fro')**2, 1.0)

        self.assertTrue((norm(self.iasolver.full_F[0], 'fro')**2 <=
                         1.0000000001 * self.iasolver.P[0]))
        self.assertTrue((norm(self.iasolver.full_F[1], 'fro')**2 <=
                         1.0000000001 * self.iasolver.P[1]))
        self.assertTrue((norm(self.iasolver.full_F[2], 'fro')**2 <=
                         1.0000000001 * self.iasolver.P[2]))

    def test_updateF_with_very_small_power(self):
        # If a previous run of this test failed, this will load the state
        # of the failed test so that it is reproduced.
        self._maybe_load_state_and_randomize_channel(
            filename='MMSE_test_updateF_with_very_small_power.pickle',
            iasolver=self.iasolver,
            Nr=self.Nr,
            Nt=self.Nt,
            K=self.K)

        # We are only testing the transmit powers here. If the precoders
        # are not calculated correctly then the test for the solve method
        # should fail.
        Ns = 1
        P = np.array([0.67, 0.89, 1.1]) * 1e-14
        self.iasolver._multiUserChannel.noise_var = 0.5e-14
        self.iasolver._initialize_F_randomly_and_find_W(Ns, P)
        self.iasolver._updateF()

        self.assertAlmostEqual(norm(self.iasolver.F[0], 'fro')**2, 1.0)
        self.assertAlmostEqual(norm(self.iasolver.F[1], 'fro')**2, 1.0)
        self.assertAlmostEqual(norm(self.iasolver.F[2], 'fro')**2, 1.0)

        try:
            self.assertTrue((norm(self.iasolver.full_F[0], 'fro')**2 <=
                             1.000001 * self.iasolver.P[0]))
            self.assertTrue((norm(self.iasolver.full_F[1], 'fro')**2 <=
                             1.000001 * self.iasolver.P[1]))
            self.assertTrue((norm(self.iasolver.full_F[2], 'fro')**2 <=
                             1.000001 * self.iasolver.P[2]))

        except AssertionError:  # pragma: nocover
            # Since this test failed, let's save its state so that we can
            # reproduce it
            self._save_state(
                filename='MMSE_test_updateF_with_very_small_power.pickle')
            raise  # re-raises the last exception

    def test_solve(self):
        # If a previous run of this test failed, this will load the state
        # of the failed test so that it is reproduced.
        self._maybe_load_state_and_randomize_channel(
            filename='MMSE_test_solve_state.pickle',
            iasolver=self.iasolver,
            Nr=self.Nr,
            Nt=self.Nt,
            K=self.K)

        self.iasolver._multiUserChannel.noise_var = 1e-3
        P = self.P

        self.iasolver.max_iterations = 200

        # xxxxxxxxxx Test with the random initialization type xxxxxxxxxxxxx
        self.iasolver.initialize_with = 'random'

        niter = self.iasolver.solve(self.Ns, P)

        self.assertTrue(niter <= self.iasolver.max_iterations)

        full_F0 = self.iasolver.full_F[0]
        full_F1 = self.iasolver.full_F[1]
        full_F2 = self.iasolver.full_F[2]

        full_W_H0 = self.iasolver.full_W_H[0]
        full_W_H1 = self.iasolver.full_W_H[1]
        full_W_H2 = self.iasolver.full_W_H[2]

        H00 = self.iasolver._get_channel(0, 0)
        H01 = self.iasolver._get_channel(0, 1)
        H02 = self.iasolver._get_channel(0, 2)
        H10 = self.iasolver._get_channel(1, 0)
        H11 = self.iasolver._get_channel(1, 1)
        H12 = self.iasolver._get_channel(1, 2)
        H20 = self.iasolver._get_channel(2, 0)
        H21 = self.iasolver._get_channel(2, 1)
        H22 = self.iasolver._get_channel(2, 2)

        # Perform the actual tests
        try:
            # xxxxx Test if the transmit power limit is respected xxxxxxxxx
            self.assertTrue(
                np.linalg.norm(full_F0, 'fro')**2 <= 1.000000001 * P[0])
            self.assertTrue(
                np.linalg.norm(full_F1, 'fro')**2 <= 1.000000001 * P[1])
            self.assertTrue(
                np.linalg.norm(full_F2, 'fro')**2 <= 1.000000001 * P[2])
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Test the equivalent channel xxxxxxxxxxxxxxxxxxxxxxxxxxx
            np.testing.assert_array_almost_equal(full_W_H0 @ H00 @ full_F0,
                                                 np.eye(self.Ns[0]))
            np.testing.assert_array_almost_equal(full_W_H1 @ H11 @ full_F1,
                                                 np.eye(self.Ns[0]))
            np.testing.assert_array_almost_equal(full_W_H2 @ H22 @ full_F2,
                                                 np.eye(self.Ns[0]))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxxxxxxx test the remaining interference xxxxxxxxxxxxxxxxxx
            self.assertTrue(
                np.linalg.norm(full_W_H0 @ H01 @ full_F1, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H0 @ H02 @ full_F2, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H1 @ H10 @ full_F0, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H1 @ H12 @ full_F2, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H2 @ H20 @ full_F0, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H2 @ H21 @ full_F1, 'fro')**2 < 0.1)
        except AssertionError:  # pragma: nocover
            # Since this test failed, let's save its state so that we can
            # reproduce it
            self._save_state(filename='MMSE_test_solve_state.pickle')
            raise  # re-raises the last exception
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test with the closed_form initialization type xxxxxxxxxxxx
        self.iasolver.clear()
        self.iasolver.initialize_with = 'closed_form'

        niter = self.iasolver.solve(self.Ns, P)

        self.assertTrue(niter <= self.iasolver.max_iterations)

        full_F0 = self.iasolver.full_F[0]
        full_F1 = self.iasolver.full_F[1]
        full_F2 = self.iasolver.full_F[2]

        full_W_H0 = self.iasolver.full_W_H[0]
        full_W_H1 = self.iasolver.full_W_H[1]
        full_W_H2 = self.iasolver.full_W_H[2]

        # Perform the actual tests
        try:
            # xxxxx Test if the transmit power limit is respected xxxxxxxxx
            self.assertTrue(
                np.linalg.norm(full_F0, 'fro')**2 <= 1.000000001 * P[0])
            self.assertTrue(
                np.linalg.norm(full_F1, 'fro')**2 <= 1.000000001 * P[1])
            self.assertTrue(
                np.linalg.norm(full_F2, 'fro')**2 <= 1.000000001 * P[2])
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Test the equivalent channel xxxxxxxxxxxxxxxxxxxxxxxxxxx
            np.testing.assert_array_almost_equal(full_W_H0 @ H00 @ full_F0,
                                                 np.eye(self.Ns[0]))
            np.testing.assert_array_almost_equal(full_W_H1 @ H11 @ full_F1,
                                                 np.eye(self.Ns[0]))
            np.testing.assert_array_almost_equal(full_W_H2 @ H22 @ full_F2,
                                                 np.eye(self.Ns[0]))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxxxxxxx test the remaining interference xxxxxxxxxxxxxxxxxx
            self.assertTrue(
                np.linalg.norm(full_W_H0 @ H01 @ full_F1, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H0 @ H02 @ full_F2, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H1 @ H10 @ full_F0, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H1 @ H12 @ full_F2, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H2 @ H20 @ full_F0, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H2 @ H21 @ full_F1, 'fro')**2 < 0.1)
        except AssertionError:  # pragma: nocover
            # Since this test failed, let's save its state so that we can
            # reproduce it
            self._save_state(filename='MMSE_test_solve_state.pickle')
            raise  # re-raises the last exception
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test with the alt_min initialization type xxxxxxxxxxxx
        self.iasolver.clear()
        self.iasolver.initialize_with = 'alt_min'

        niter = self.iasolver.solve(self.Ns, P)

        self.assertTrue(niter <= self.iasolver.max_iterations)

        full_F0 = self.iasolver.full_F[0]
        full_F1 = self.iasolver.full_F[1]
        full_F2 = self.iasolver.full_F[2]

        full_W_H0 = self.iasolver.full_W_H[0]
        full_W_H1 = self.iasolver.full_W_H[1]
        full_W_H2 = self.iasolver.full_W_H[2]

        # Perform the actual tests
        try:
            # xxxxx Test if the transmit power limit is respected xxxxxxxxx
            self.assertTrue(
                np.linalg.norm(full_F0, 'fro')**2 <= 1.000000001 * P[0])
            self.assertTrue(
                np.linalg.norm(full_F1, 'fro')**2 <= 1.000000001 * P[1])
            self.assertTrue(
                np.linalg.norm(full_F2, 'fro')**2 <= 1.000000001 * P[2])
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Test the equivalent channel xxxxxxxxxxxxxxxxxxxxxxxxxxx
            np.testing.assert_array_almost_equal(full_W_H0 @ H00 @ full_F0,
                                                 np.eye(self.Ns[0]))
            np.testing.assert_array_almost_equal(full_W_H1 @ H11 @ full_F1,
                                                 np.eye(self.Ns[0]))
            np.testing.assert_array_almost_equal(full_W_H2 @ H22 @ full_F2,
                                                 np.eye(self.Ns[0]))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxxxxxxx test the remaining interference xxxxxxxxxxxxxxxxxx
            self.assertTrue(
                np.linalg.norm(full_W_H0 @ H01 @ full_F1, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H0 @ H02 @ full_F2, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H1 @ H10 @ full_F0, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H1 @ H12 @ full_F2, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H2 @ H20 @ full_F0, 'fro')**2 < 0.1)
            self.assertTrue(
                np.linalg.norm(full_W_H2 @ H21 @ full_F1, 'fro')**2 < 0.1)
        except AssertionError:  # pragma: nocover
            # Since this test failed, let's save its state so that we can
            # reproduce it
            self._save_state(filename='MMSE_test_solve_state.pickle')
            raise  # re-raises the last exception
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_solve_finalize(self):
        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 2

        # Transmit power of all users. We set the power of the first user
        # to a very low value so that the ia solver sets 0 energy to one of
        # the streams (due to the waterfilling algorithm deciding is is
        # better to focus all the energy into one stream). This will make
        # the code in the _solve_finalize method to reduce the number of
        # streams of the first user, which we will test here.
        P = np.array([0.0001, 100.8, 230.0])

        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        multiUserChannel.randomize(Nr, Nt, K)

        iasolver = MMSEIASolver(multiUserChannel)
        multiUserChannel.noise_var = 0.1

        iasolver.solve(Ns, P)

        self.assertEqual(iasolver.F[0].shape, (4, 1))
        self.assertEqual(iasolver.F[1].shape, (4, 2))
        self.assertEqual(iasolver.F[2].shape, (4, 2))

        self.assertEqual(iasolver.W[0].shape, (4, 1))
        self.assertEqual(iasolver.W[1].shape, (4, 2))
        self.assertEqual(iasolver.W[2].shape, (4, 2))

        self.assertEqual(iasolver.full_F[0].shape, (4, 1))
        self.assertEqual(iasolver.full_F[1].shape, (4, 2))
        self.assertEqual(iasolver.full_F[2].shape, (4, 2))

        self.assertEqual(iasolver.full_W[0].shape, (4, 1))
        self.assertEqual(iasolver.full_W[1].shape, (4, 2))
        self.assertEqual(iasolver.full_W[2].shape, (4, 2))

        np.testing.assert_array_equal(iasolver.Ns, np.array([1, 2, 2]))
        # The Ns array passed to the IA solver object should not be
        # changed.
        np.testing.assert_array_equal(Ns, np.array([2, 2, 2]))


class GreedStreamIASolverTestCase(CustomTestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_solve(self):
        # xxxxxxxxxx Initializations xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        alt_min_iasolver = AlternatingMinIASolver(multiUserChannel)

        iasolver = GreedStreamIASolver(alt_min_iasolver)

        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        # Note that for this configuration IA is not feasible
        Ns = np.array([2, 2, 1])  # np.ones(K, dtype=int) * 3

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        # Randomize the channel
        multiUserChannel.randomize(Nr, Nt, K)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # If a previous run of this test failed, this will load the state
        # of the failed test so that it is reproduced.
        self._maybe_load_state_and_randomize_channel(
            filename='GreedStream_test_solve_state.pickle',
            iasolver=iasolver._iasolver,
            Nr=Nr,
            Nt=Nt,
            K=K)

        # xxxxxxxxxx Find the solution xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        iasolver._iasolver.max_iterations = 200
        iasolver.solve(Ns, P)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Solution found by the algorithm xxxxxxxxxxxxxxxxxxxxxx
        full_F0 = iasolver._iasolver.full_F[0]
        full_F1 = iasolver._iasolver.full_F[1]
        full_F2 = iasolver._iasolver.full_F[2]

        full_W_H0 = iasolver._iasolver.full_W_H[0]
        full_W_H1 = iasolver._iasolver.full_W_H[1]
        full_W_H2 = iasolver._iasolver.full_W_H[2]

        H00 = iasolver._iasolver._get_channel(0, 0)
        H01 = iasolver._iasolver._get_channel(0, 1)
        H02 = iasolver._iasolver._get_channel(0, 2)
        H10 = iasolver._iasolver._get_channel(1, 0)
        H11 = iasolver._iasolver._get_channel(1, 1)
        H12 = iasolver._iasolver._get_channel(1, 2)
        H20 = iasolver._iasolver._get_channel(2, 0)
        H21 = iasolver._iasolver._get_channel(2, 1)
        H22 = iasolver._iasolver._get_channel(2, 2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        try:
            # xxxxx Number of streams that are actually used. xxxxxxxxxxxxx
            # Note that the values in final_Ns can be lower than or equal to
            # the values in Ns.
            final_Ns = iasolver._iasolver.Ns
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Test if the transmit power limit is respected xxxxxxxxx
            self.assertTrue(np.linalg.norm(full_F0, 'fro')**2 <= P[0] + 1e-12)
            self.assertTrue(np.linalg.norm(full_F1, 'fro')**2 <= P[1] + 1e-12)
            self.assertTrue(np.linalg.norm(full_F2, 'fro')**2 <= P[2] + 1e-12)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Test the equivalent channel xxxxxxxxxxxxxxxxxxxxxxxxxxx
            np.testing.assert_array_almost_equal(full_W_H0 @ H00 @ full_F0,
                                                 np.eye(final_Ns[0]))
            np.testing.assert_array_almost_equal(full_W_H1 @ H11 @ full_F1,
                                                 np.eye(final_Ns[1]))
            np.testing.assert_array_almost_equal(full_W_H2 @ H22 @ full_F2,
                                                 np.eye(final_Ns[2]))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxxxxxxx test the remaining interference xxxxxxxxxxxxxxxxxx
            norm_value = np.linalg.norm(full_W_H0 @ H01 @ full_F1, 'fro')**2

            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H0 @ H02 @ full_F2, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H1 @ H10 @ full_F0, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H1 @ H12 @ full_F2, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H2 @ H20 @ full_F0, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H2 @ H21 @ full_F1, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

        except AssertionError:  # pragma: no cover
            # Since this test failed, let's save its state so that we can
            # reproduce it
            self._save_state('GreedStream_test_solve_state.pickle')
            raise  # re-raises the last exception


class BruteForceStreamIASolverTestCase(CustomTestCase):
    def setUp(self):
        """Called before each test."""
        pass

    def test_solve_and_clear(self):
        # Test the solve method and the clear method

        # xxxxxxxxxx Initializations xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        multiUserChannel = channels.multiuser.MultiUserChannelMatrix()
        alt_min_iasolver = AlternatingMinIASolver(multiUserChannel)

        iasolver = BruteForceStreamIASolver(alt_min_iasolver)

        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        # Note that for this configuration IA is not feasible
        Ns = np.array([3, 4, 2])  # np.ones(K, dtype=int) * 3

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        # Randomize the channel
        multiUserChannel.randomize(Nr, Nt, K)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # If a previous run of this test failed, this will load the state
        # of the failed test so that it is reproduced.
        self._maybe_load_state_and_randomize_channel(
            filename='BruteForce_test_solve_state.pickle',
            iasolver=iasolver._iasolver,
            Nr=Nr,
            Nt=Nt,
            K=K)

        # xxxxxxxxxx Find the solution xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        iasolver._iasolver.max_iterations = 200
        self.assertEqual(iasolver.runned_iterations, 0)

        iasolver.solve(Ns, P)
        self.assertGreater(iasolver.runned_iterations, 0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Solution found by the algorithm xxxxxxxxxxxxxxxxxxxxxx
        full_F0 = iasolver._iasolver.full_F[0]
        full_F1 = iasolver._iasolver.full_F[1]
        full_F2 = iasolver._iasolver.full_F[2]

        full_W_H0 = iasolver._iasolver.full_W_H[0]
        full_W_H1 = iasolver._iasolver.full_W_H[1]
        full_W_H2 = iasolver._iasolver.full_W_H[2]

        H00 = iasolver._iasolver._get_channel(0, 0)
        H01 = iasolver._iasolver._get_channel(0, 1)
        H02 = iasolver._iasolver._get_channel(0, 2)
        H10 = iasolver._iasolver._get_channel(1, 0)
        H11 = iasolver._iasolver._get_channel(1, 1)
        H12 = iasolver._iasolver._get_channel(1, 2)
        H20 = iasolver._iasolver._get_channel(2, 0)
        H21 = iasolver._iasolver._get_channel(2, 1)
        H22 = iasolver._iasolver._get_channel(2, 2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        try:
            # xxxxx Number of streams that are actually used. xxxxxxxxxxxxx
            # Note that the values in final_Ns can be lower than or equal to
            # the values in Ns.
            final_Ns = iasolver._iasolver.Ns
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Test if the transmit power limit is respected xxxxxxxxx
            self.assertTrue(np.linalg.norm(full_F0, 'fro')**2 <= P[0] + 1e-12)
            self.assertTrue(np.linalg.norm(full_F1, 'fro')**2 <= P[1] + 1e-12)
            self.assertTrue(np.linalg.norm(full_F2, 'fro')**2 <= P[2] + 1e-12)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Test the equivalent channel xxxxxxxxxxxxxxxxxxxxxxxxxxx
            np.testing.assert_array_almost_equal(full_W_H0 @ H00 @ full_F0,
                                                 np.eye(final_Ns[0]))
            np.testing.assert_array_almost_equal(full_W_H1 @ H11 @ full_F1,
                                                 np.eye(final_Ns[1]))
            np.testing.assert_array_almost_equal(full_W_H2 @ H22 @ full_F2,
                                                 np.eye(final_Ns[2]))
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxxxxxxx test the remaining interference xxxxxxxxxxxxxxxxxx
            norm_value = np.linalg.norm(full_W_H0 @ H01 @ full_F1, 'fro')**2

            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H0 @ H02 @ full_F2, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H1 @ H10 @ full_F0, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H1 @ H12 @ full_F2, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H2 @ H20 @ full_F0, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            norm_value = np.linalg.norm(full_W_H2 @ H21 @ full_F1, 'fro')**2
            self.assertTrue(norm_value < 0.05,
                            msg="Norm Value: {0}".format(norm_value))

            # xxxxxxxxxx Now test the clear method xxxxxxxxxxxxxxxxxxxxxxxx
            self.assertNotEqual(iasolver.stream_combinations, ())
            self.assertNotEqual(iasolver.every_sum_capacity, [])
            self.assertIsNotNone(iasolver._best_F)
            self.assertIsNotNone(iasolver._best_full_F)
            self.assertIsNotNone(iasolver._best_W_H)
            self.assertIsNotNone(iasolver._best_Ns)

            iasolver.clear()

            self.assertEqual(iasolver.stream_combinations, [])
            self.assertEqual(iasolver.every_sum_capacity, [])
            self.assertIsNone(iasolver._best_F)
            self.assertIsNone(iasolver._best_full_F)
            self.assertIsNone(iasolver._best_W_H)
            self.assertIsNone(iasolver._best_Ns)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        except AssertionError:  # pragma: no cover
            # Since this test failed, let's save its state so that we can
            # reproduce it
            self._save_state('BruteForce_test_solve_state.pickle')
            raise  # re-raises the last exception


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":  # pragma: nocover
    unittest.main()
