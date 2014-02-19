#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the ia package.

Each module has several doctests that we run in addition to the unittests
defined here.
"""

__revision__ = "$Revision$"

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import unittest
import doctest
import numpy as np
from numpy.linalg import norm

from pyphysim.comm import channels
import pyphysim.ia  # Import the package ia
from pyphysim.ia.ia import AlternatingMinIASolver, IASolverBaseClass, MaxSinrIASolver, \
    MinLeakageIASolver, ClosedFormIASolver, MMSEIASolver, \
    IterativeIASolverBaseClass
from pyphysim.util.misc import peig, leig


# UPDATE THIS CLASS if another module is added to the ia package
class IaDoctestsTestCase(unittest.TestCase):
    """Teste case that run all the doctests in the modules of the ia
    package."""

    def test_ia(self):
        """Run doctests in the ia module."""
        doctest.testmod(pyphysim.ia)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx IA Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class IASolverBaseClassTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        multiUserChannel = channels.MultiUserChannelMatrix()
        self.iasolver = IASolverBaseClass(multiUserChannel)

    def test_init(self):
        # Try to initialize the IASolverBaseClass object with some
        # parameter which is not a MultiUserChannelMatrix object
        with self.assertRaises(ValueError):
            IASolverBaseClass(3)

    def test_get_cost(self):
        self.assertEqual(self.iasolver.get_cost(), -1)

    def test_properties(self):
        K = 3
        Nr = np.array([2, 4, 6])
        Nt = np.array([2, 3, 5])
        Ns = np.array([1, 2, 3])
        multiUserChannel = self.iasolver._multiUserChannel
        multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver.randomizeF(Ns, P=None)  # Setting P here will be
                                              # tested in test_randomizeF
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

        # If we try to set P with a sequency of wrong length (different
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
        # It starts as "None"
        self.assertIsNone(self.iasolver._noise_var)
        # If we try to get the value of the last_noise_var property it will
        # return the value of the last_noise_var property of the
        # multiUserChannel object
        self.assertEqual(self.iasolver.noise_var, 0.0)
        self.iasolver._multiUserChannel._last_noise_var = 1.3
        self.assertEqual(self.iasolver.noise_var, 1.3)
        # But if we set the noise_var property to some (non negative) value
        # it will be respected.
        self.iasolver.noise_var = 1.5
        self.assertEqual(self.iasolver.noise_var, 1.5)
        with self.assertRaises(AssertionError):
            self.iasolver.noise_var = -0.6
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
        self.assertAlmostEqual(norm(self.iasolver.full_F[0], 'fro') ** 2, P[0])
        self.assertAlmostEqual(norm(self.iasolver.full_F[1], 'fro') ** 2, P[1])
        self.assertAlmostEqual(norm(self.iasolver.full_F[2], 'fro') ** 2, P[2])

        # The shape of the precoder is the number of users
        self.assertEqual(self.iasolver._F.shape, (K,))

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
        self.iasolver.randomizeF(1)
        self.assertIsNotNone(self.iasolver._F)
        self.assertIsNone(self.iasolver._full_F)

        # Since the power was not set yet then full_F should have the same
        # value as F
        self.assertIsNone(self.iasolver._P)
        np.testing.assert_almost_equal(self.iasolver.F, self.iasolver.full_F)

        # Let's change F and see if full_F matches
        self.iasolver.randomizeF(1)
        np.testing.assert_almost_equal(self.iasolver.F, self.iasolver.full_F)

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
        H01_F1 = np.dot(
            self.iasolver._get_channel(k, 1),
            self.iasolver.full_F[1]
        )
        H02_F2 = np.dot(
            self.iasolver._get_channel(k, 2),
            self.iasolver.full_F[2]
        )
        expected_Q0 = np.dot(H01_F1,
                             H01_F1.transpose().conjugate()) + \
            np.dot(H02_F2,
                   H02_F2.transpose().conjugate())

        Qk = self.iasolver.calc_Q(k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0 = np.dot(
            self.iasolver._get_channel(k, 0),
            self.iasolver._F[0]
        )
        H12_F2 = np.dot(
            self.iasolver._get_channel(k, 2),
            self.iasolver._F[2]
        )
        expected_Q1 = np.dot(P[0] * H10_F0,
                             H10_F0.transpose().conjugate()) + \
            np.dot(P[2] * H12_F2,
                   H12_F2.transpose().conjugate())

        Qk = self.iasolver.calc_Q(k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H20_F0 = np.dot(
            self.iasolver._get_channel(k, 0),
            self.iasolver._F[0]
        )
        H21_F1 = np.dot(
            self.iasolver._get_channel(k, 1),
            self.iasolver._F[1]
        )
        expected_Q2 = np.dot(P[0] * H20_F0,
                             H20_F0.transpose().conjugate()) + \
            np.dot(P[1] * H21_F1,
                   H21_F1.transpose().conjugate())

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

        #xxxxxxxxxx k = 0 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 0
        Qk = self.iasolver.calc_Q(k)
        pk = self.iasolver.calc_remaining_interference_percentage(k, Qk)

        [_, D] = leig(Qk, Ns[k])
        expected_pk = np.sum(np.abs(D)) / np.abs(np.trace(Qk))
        self.assertAlmostEqual(pk, expected_pk)

        #xxxxxxxxxx k = 1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 1
        Qk = self.iasolver.calc_Q(k)
        pk = self.iasolver.calc_remaining_interference_percentage(k)

        [_, D] = leig(Qk, Ns[k])
        expected_pk = np.sum(np.abs(D)) / np.abs(np.trace(Qk))
        self.assertAlmostEqual(pk, expected_pk)

        #xxxxxxxxxx k = 2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 2
        Qk = self.iasolver.calc_Q(k)
        pk = self.iasolver.calc_remaining_interference_percentage(k)

        [V, D] = leig(Qk, Ns[k])
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

        # For ones stream the expected Bkl is equivalent to the Q matrix
        # plus the direct channel part.
        for k in range(self.iasolver.K):
            Hkk = self.iasolver._get_channel(k, k)
            Fk = self.iasolver.F[k]
            HkkFk = np.dot(Hkk, Fk)
            expected_first_part = self.iasolver.calc_Q(k) + P[k] / \
                Ns[k].astype(float) * \
                np.dot(HkkFk, HkkFk.transpose().conjugate())

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
            expected_first_part = 0.0  # First part in the equation of Bkl
                                       # (the double summation)

            # The outer for loop will calculate
            # first_part = $\sum_{j=1}^{K} \frac{P[k]}{Ns[k]} \text{aux}$
            for j in range(self.iasolver.K):
                aux = 0.0  # The inner for loop will calculate
                            # $\text{aux} = \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$
                Hkj = self.iasolver._get_channel(k, j)
                Hkj_H = Hkj.conjugate().transpose()

                for d in range(self.iasolver.Ns[k]):
                    Vjd = self.iasolver.full_F[j][:, d:d + 1]
                    Vjd_H = Vjd.conjugate().transpose()
                    aux = aux + np.dot(np.dot(Hkj, np.dot(Vjd, Vjd_H)), Hkj_H)

                expected_first_part = expected_first_part + aux

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
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = self.iasolver.full_F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(Hkk,
                                              np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
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
            for l in range(Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = self.iasolver.full_F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(Hkk,
                                              np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.iasolver._calc_Bkl_cov_matrix_second_part(k, l))

    def test_calc_Bkl(self):
        # For the case of a single stream oer user Bkl (which only has l=0)
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
            Bk0 = self.iasolver._calc_Bkl_cov_matrix_all_l(k, noise_power=noise_power)[0]
            np.testing.assert_array_almost_equal(expected_Bk0, Bk0)

    def test_solve(self):
        with self.assertRaises(NotImplementedError):
            self.iasolver.solve(Ns=1)


class ClosedFormIASolverTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        multiUserChannel = channels.MultiUserChannelMatrix()
        self.iasolver = ClosedFormIASolver(multiUserChannel)
        self.K = 3
        self.Nr = np.array([2, 2, 2])
        self.Nt = np.array([2, 2, 2])
        multiUserChannel.randomize(self.Nr, self.Nt, self.K)

    def test_sanity(self):
        # The number of users is always equal to 3
        self.assertEqual(self.iasolver.K, 3)
        #np.testing.assert_array_equal(np.ones(3), self.iasolver.Ns)

    def test_invalid_solve(self):
        multiUserChannel = channels.MultiUserChannelMatrix()
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
        H31 = np.matrix(self.iasolver._get_channel(2, 0))
        H32 = np.matrix(self.iasolver._get_channel(2, 1))
        H12 = np.matrix(self.iasolver._get_channel(0, 1))
        H13 = np.matrix(self.iasolver._get_channel(0, 2))
        H23 = np.matrix(self.iasolver._get_channel(1, 2))
        H21 = np.matrix(self.iasolver._get_channel(1, 0))

        expected_E = H31.I * H32 * H12.I * H13 * H23.I * H21
        np.testing.assert_array_almost_equal(expected_E, self.iasolver._calc_E())

    def test_calc_all_F_initializations(self):
        # xxxxx Test the case with Nt = Ns = 2 and Ns = 1 xxxxxxxxxxxxxxxxx
        Ns = 1
        E = self.iasolver._calc_E()
        all_eigenvectors = np.linalg.eig(E)[1]
        expected_all_subsets = [all_eigenvectors[:, (0,)], all_eigenvectors[:, (1,)]]

        all_subsets = self.iasolver._calc_all_F_initializations(Ns)

        for a, b in zip(expected_all_subsets, all_subsets):
            np.testing.assert_array_almost_equal(a, b)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test the case with Nt = Ns = 4 and Ns = 2 xxxxxxxxxxxxxxxxx
        Ns2 = 2
        multiUserChannel = channels.MultiUserChannelMatrix()
        iasolver = ClosedFormIASolver(multiUserChannel)
        K = 3
        Nr = np.array([4, 4, 4])
        Nt = np.array([4, 4, 4])
        multiUserChannel.randomize(Nr, Nt, K)
        E2 = iasolver._calc_E()
        all_eigenvectors2 = np.linalg.eig(E2)[1]
        expected_all_subsets2 = [all_eigenvectors2[:, (0, 1)],
                                 all_eigenvectors2[:, (0, 2)],
                                 all_eigenvectors2[:, (0, 3)],
                                 all_eigenvectors2[:, (1, 2)],
                                 all_eigenvectors2[:, (1, 3)],
                                 all_eigenvectors2[:, (2, 3)]]

        all_subsets2 = iasolver._calc_all_F_initializations(Ns2)

        for a, b in zip(expected_all_subsets2, all_subsets2):
            np.testing.assert_array_almost_equal(a, b)

    def test_updateF(self):
        Ns = 1
        E = self.iasolver._calc_E()
        [_, eigenvectors] = np.linalg.eig(E)
        # V1 is the expected precoder for the first user
        V1 = np.matrix(eigenvectors[:, 0:Ns])

        H32 = np.matrix(self.iasolver._get_channel(2, 1))
        H31 = np.matrix(self.iasolver._get_channel(2, 0))
        H23 = np.matrix(self.iasolver._get_channel(1, 2))
        H21 = np.matrix(self.iasolver._get_channel(1, 0))

        # Expected precoder for the second user
        V2 = H32.I * H31 * V1
        # Expected precoder for the third user
        V3 = H23.I * H21 * V1

        # Normalize the precoders
        V1 = V1 / norm(V1, 'fro')
        V2 = V2 / norm(V2, 'fro')
        V3 = V3 / norm(V3, 'fro')

        # The number of streams _Ns is set in the solve method, before
        # _updateF is called. However, since we are testing the _updateF
        # method alone here we need to set _Ns manually.
        self.iasolver._Ns = np.ones(self.iasolver.K, dtype=int) * Ns

        # Find the precoders using the iasolver
        self.iasolver._updateF()

        np.testing.assert_array_almost_equal(V1, self.iasolver.F[0])
        np.testing.assert_array_almost_equal(V2, self.iasolver.F[1])
        np.testing.assert_array_almost_equal(V3, self.iasolver.F[2])

        self.assertAlmostEqual(norm(V1, 'fro'), 1.0)
        self.assertAlmostEqual(norm(V2, 'fro'), 1.0)
        self.assertAlmostEqual(norm(V3, 'fro'), 1.0)

    def test_updateW(self):
        Ns = 1
        # The number of streams _Ns is set in the solve method, before
        # _updateF and the _updateW methods are called. However, since we
        # are testing the _updateW method alone here we need to set _Ns
        # manually.
        self.iasolver._Ns = np.ones(self.iasolver.K, dtype=int) * Ns

        self.iasolver._updateF()
        self.iasolver._updateW()
        V1 = np.matrix(self.iasolver.F[0])
        V2 = np.matrix(self.iasolver.F[1])
        #V3 = np.matrix(self.iasolver.F[2])

        H12 = np.matrix(self.iasolver._get_channel(0, 1))
        H13 = np.matrix(self.iasolver._get_channel(0, 2))
        H21 = np.matrix(self.iasolver._get_channel(1, 0))
        H23 = np.matrix(self.iasolver._get_channel(1, 2))
        H31 = np.matrix(self.iasolver._get_channel(2, 0))
        H32 = np.matrix(self.iasolver._get_channel(2, 1))

        U1 = H12 * V2
        U1 = leig(U1 * U1.H, 1)[0]
        U2 = H21 * V1
        U2 = leig(U2 * U2.H, 1)[0]
        U3 = H31 * V1
        U3 = leig(U3 * U3.H, 1)[0]

        np.testing.assert_array_almost_equal(self.iasolver._W[0], U1)
        np.testing.assert_array_almost_equal(self.iasolver._W[1], U2)
        np.testing.assert_array_almost_equal(self.iasolver._W[2], U3)

        # xxxxx Test if the interference is cancelled xxxxxxxxxxxxxxxxxxxxx
        I1 = np.dot(self.iasolver.W_H[0], np.dot(H12, self.iasolver.F[1])) + \
            np.dot(self.iasolver.W_H[0], np.dot(H13, self.iasolver.F[2]))
        self.assertAlmostEqual(I1, 0.0)

        I2 = np.dot(self.iasolver.W_H[1], np.dot(H21, self.iasolver.F[0])) + \
            np.dot(self.iasolver.W_H[1], np.dot(H23, self.iasolver.F[2]))
        self.assertAlmostEqual(I2, 0.0)

        I3 = np.dot(self.iasolver.W_H[2], np.dot(H31, self.iasolver.F[0])) + \
            np.dot(self.iasolver.W_H[2], np.dot(H32, self.iasolver.F[1]))
        self.assertAlmostEqual(I3, 0.0)
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
        self.iasolver.solve(Ns)
        for l in range(3):
            for k in range(3):
                Hlk = self.iasolver._get_channel(l, k)
                Wl_H = self.iasolver.W_H[l]
                Fk = self.iasolver.F[k]
                s = np.dot(Wl_H, np.dot(Hlk, Fk))[0][0]
                if l == k:
                    Hk_eq = self.iasolver._calc_equivalent_channel(k)
                    s2 = s / Hk_eq[0, 0]  # We only have one stream -> the
                                          # equivalent channel is an scalar.
                    self.assertAlmostEqual(1.0, s2)
                else:
                    # Test if the interference is equal to 0.0
                    self.assertAlmostEqual(0.0, s)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Test if an exception is raised if we try to use the
        # ClosedFormIASolver class with a number of users different from 3.
        multiUserChannel = channels.MultiUserChannelMatrix()
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
        multiUserChannel = channels.MultiUserChannelMatrix()
        iasolver = ClosedFormIASolver(multiUserChannel, use_best_init=True)
        iasolver2 = ClosedFormIASolver(multiUserChannel, use_best_init=False)
        K = 3
        Nr = 4
        Nt = 4
        Ns = 2

        # xxxxx Debug xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # multiUserChannel.set_channel_seed(43)
        # multiUserChannel.set_noise_seed(456)
        # np.random.seed(25)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        multiUserChannel.randomize(Nr, Nt, K)

        iasolver.solve(Ns)
        iasolver2.solve(Ns)

        # Test of the equivalent direct channel is equal to one while the
        # equivalent clross channels are equivalent to zero
        for l in range(3):
            for k in range(3):
                Hlk = iasolver._get_channel(l, k)
                Wl_H = iasolver.W_H[l]
                Fk = iasolver.F[k]
                s = np.dot(Wl_H, np.dot(Hlk, Fk))[0][0]
                if l == k:
                    Hk_eq = iasolver._calc_equivalent_channel(k)
                    s2 = s / Hk_eq[0, 0]  # We only have one stream -> the
                                          # equivalent channel is an scalar.
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
        self.iasolver.solve(Ns)
        for k in range(self.iasolver.K):
            Hkk = self.iasolver._get_channel(k, k)
            full_Wk_H = self.iasolver.full_W_H[k]
            Fk = self.iasolver.F[k]
            s = np.dot(full_Wk_H, np.dot(Hkk, Fk))[0][0]
            self.assertAlmostEqual(1.0, s)


# TODO: finish implementation
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

        self.assertFalse(IterativeIASolverBaseClass._is_diff_significant(F_old, F_new))
        F_new[1][1, 2] += 9e-4
        F_new[2][0, 0] += 6e-4
        self.assertFalse(IterativeIASolverBaseClass._is_diff_significant(F_old, F_new))
        F_new[2][2, 2] += 2e-3
        self.assertTrue(IterativeIASolverBaseClass._is_diff_significant(F_old, F_new))


class AlternatingMinIASolverTestCase(unittest.TestCase):
    """Unittests for the AlternatingMinIASolver class in the ia module."""
    def setUp(self):
        """Called before each test."""
        multiUserChannel = channels.MultiUserChannelMatrix()
        self.iasolver = AlternatingMinIASolver(multiUserChannel)
        self.K = 3
        self.Nr = np.array([2, 4, 6])
        self.Nt = np.array([2, 3, 5])
        self.Ns = np.array([1, 2, 3])
        multiUserChannel.randomize(self.Nr, self.Nt, self.K)
        #self.iasolver.randomizeF(self.Ns)

    def test_updateC(self):
        self.iasolver.randomizeF(self.Ns)

        # Dimensions of the interference subspace
        Ni = self.Nr - self.Ns

        self.iasolver._updateC()

        # xxxxx Calculate the expected C[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H01_F1 = np.dot(
            self.iasolver._get_channel(k, 1),
            self.iasolver.F[1]
        )
        H02_F2 = np.dot(
            self.iasolver._get_channel(k, 2),
            self.iasolver.F[2]
        )
        expected_C0 = np.dot(H01_F1, H01_F1.transpose().conjugate()) + \
            np.dot(H02_F2, H02_F2.transpose().conjugate())
        expected_C0 = peig(expected_C0, Ni[k])[0]

        # Test if C[0] is equal to the expected output
        np.testing.assert_array_almost_equal(self.iasolver._C[0], expected_C0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected C[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0 = np.dot(
            self.iasolver._get_channel(k, 0),
            self.iasolver.F[0]
        )
        H12_F2 = np.dot(
            self.iasolver._get_channel(k, 2),
            self.iasolver.F[2]
        )
        expected_C1 = np.dot(H10_F0, H10_F0.transpose().conjugate()) + \
            np.dot(H12_F2, H12_F2.transpose().conjugate())
        expected_C1 = peig(expected_C1, Ni[k])[0]

        # Test if C[1] is equal to the expected output
        np.testing.assert_array_almost_equal(self.iasolver._C[1], expected_C1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected C[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H20_F0 = np.dot(
            self.iasolver._get_channel(k, 0),
            self.iasolver.F[0]
        )
        H21_F1 = np.dot(
            self.iasolver._get_channel(k, 1),
            self.iasolver.F[1]
        )
        expected_C2 = np.dot(H20_F0, H20_F0.transpose().conjugate()) + \
            np.dot(H21_F1, H21_F1.transpose().conjugate())
        expected_C2 = peig(expected_C2, Ni[k])[0]

        # Test if C[2] is equal to the expected output
        np.testing.assert_array_almost_equal(self.iasolver._C[2], expected_C2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_updateF(self):
        self.iasolver.randomizeF(self.Ns)
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
        Y0 = np.eye(self.Nr[0], dtype=complex) - \
            np.dot(
                self.iasolver._C[0],
                self.iasolver._C[0].conjugate().transpose())

        Y1 = np.eye(self.Nr[1], dtype=complex) - \
            np.dot(
                self.iasolver._C[1],
                self.iasolver._C[1].conjugate().transpose())

        Y2 = np.eye(self.Nr[2], dtype=complex) - \
            np.dot(
                self.iasolver._C[2],
                self.iasolver._C[2].conjugate().transpose())
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected F[0] after one step xxxxxxxxxxxxxxxx
        # l = 0 -> k = 1 and k = 2
        expected_F0 = np.dot(np.dot(H10.conjugate().transpose(), Y1), H10) + \
            np.dot(np.dot(H20.conjugate().transpose(), Y2), H20)
        expected_F0 = leig(expected_F0, self.Ns[0])[0]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected F[1] after one step xxxxxxxxxxxxxxxx
        # l = 1 -> k = 0 and k = 2
        expected_F1 = np.dot(np.dot(H01.conjugate().transpose(), Y0), H01) + \
            np.dot(np.dot(H21.conjugate().transpose(), Y2), H21)
        expected_F1 = leig(expected_F1, self.Ns[1])[0]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected F[1] after one step xxxxxxxxxxxxxxxx
        # l = 2 -> k = 0 and k = 1
        expected_F2 = np.dot(np.dot(H02.conjugate().transpose(), Y0), H02) + \
            np.dot(np.dot(H12.conjugate().transpose(), Y1), H12)
        expected_F2 = leig(expected_F2, self.Ns[2])[0]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Finally perform the tests xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_almost_equal(self.iasolver.F[0], expected_F0)
        np.testing.assert_array_almost_equal(self.iasolver.F[1], expected_F1)
        np.testing.assert_array_almost_equal(self.iasolver.F[2], expected_F2)

    def test_updateW(self):
        self.iasolver.randomizeF(self.Ns)

        # Call updateC, updateF and updateW
        self.iasolver._step()

        # xxxxx Calculates the expected receive filter for user 0 xxxxxxxxx
        tildeH0 = np.dot(
            self.iasolver._get_channel(0, 0),
            self.iasolver.F[0])
        tildeH0 = np.hstack([tildeH0, self.iasolver._C[0]])
        expected_W0_H = np.linalg.inv(tildeH0)[0:self.iasolver.Ns[0]]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for user 1 xxxxxxxxx
        tildeH1 = np.dot(
            self.iasolver._get_channel(1, 1),
            self.iasolver.F[1])
        tildeH1 = np.hstack([tildeH1, self.iasolver._C[1]])
        expected_W1_H = np.linalg.inv(tildeH1)[0:self.iasolver.Ns[1]]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for user 2 xxxxxxxxx
        tildeH2 = np.dot(
            self.iasolver._get_channel(2, 2),
            self.iasolver.F[2])
        tildeH2 = np.hstack([tildeH2, self.iasolver._C[2]])
        expected_W2_H = np.linalg.inv(tildeH2)[0:self.iasolver.Ns[2]]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Finally perform the tests xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_almost_equal(self.iasolver.W_H[0], expected_W0_H)
        np.testing.assert_array_almost_equal(self.iasolver.W_H[1], expected_W1_H)
        np.testing.assert_array_almost_equal(self.iasolver.W_H[2], expected_W2_H)
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
        K = 2
        Nr = np.array([3, 3])
        Nt = np.array([3, 3])
        Ns = np.array([2, 2])
        multiUserChannel = self.iasolver._multiUserChannel
        multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver.randomizeF(Ns)

        # Call updateC, updateF and updateW
        self.iasolver._step()

        Cost = 0
        k, l = (0, 1)
        H01_F1 = np.dot(
            self.iasolver._get_channel(k, l),
            self.iasolver.F[l])
        Cost = Cost + norm(
            H01_F1 -
            np.dot(
                np.dot(self.iasolver._C[k], self.iasolver._C[k].transpose().conjugate()),
                H01_F1
            ), 'fro') ** 2

        k, l = (1, 0)
        H10_F0 = np.dot(
            self.iasolver._get_channel(k, l),
            self.iasolver.F[l])
        Cost = Cost + norm(
            H10_F0 -
            np.dot(
                np.dot(self.iasolver._C[k], self.iasolver._C[k].transpose().conjugate()),
                H10_F0
            ), 'fro') ** 2

        self.assertAlmostEqual(self.iasolver.get_cost(), Cost)

    # def test_solve(self):
    #     self.iasolver.max_iterations = 1
    #     # We are only testing if this does not thrown an exception. That's
    #     # why there is no assert clause here
    #     self.iasolver.solve(self.Ns)

    def test_calc_SINR_old(self):
        multiUserChannel = channels.MultiUserChannelMatrix()

        # xxxxxxxxxx Debug xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        multiUserChannel.set_channel_seed(42)
        multiUserChannel.set_noise_seed(456)
        np.random.seed(25)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
        F0 = np.matrix(iasolver.F[0])
        F1 = np.matrix(iasolver.F[1])
        F2 = np.matrix(iasolver.F[2])

        W0 = np.matrix(iasolver.W[0])
        W1 = np.matrix(iasolver.W[1])
        W2 = np.matrix(iasolver.W[2])
        W0_H = np.matrix(iasolver.W_H[0])
        W1_H = np.matrix(iasolver.W_H[1])
        W2_H = np.matrix(iasolver.W_H[2])

        H00 = np.matrix(iasolver._get_channel(0, 0))
        H11 = np.matrix(iasolver._get_channel(1, 1))
        H22 = np.matrix(iasolver._get_channel(2, 2))

        H01 = np.matrix(iasolver._get_channel(0, 1))
        H02 = np.matrix(iasolver._get_channel(0, 2))
        H10 = np.matrix(iasolver._get_channel(1, 0))
        H12 = np.matrix(iasolver._get_channel(1, 2))
        H20 = np.matrix(iasolver._get_channel(2, 0))
        H21 = np.matrix(iasolver._get_channel(2, 1))

        expected_SINRs = np.empty(K, dtype=np.ndarray)

        # xxxxx k = 0 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        numerator0 = W0_H * H00 * F0
        numerator0 = numerator0 * numerator0.H
        numerator0 = np.abs(np.diag(numerator0))

        denominator0 = W0_H * H01 * F1 + W0_H * H02 * F2
        denominator0 = denominator0 * denominator0.H
        denominator0 = np.abs(np.diag(denominator0))

        expected_SINRs[0] = numerator0 / denominator0

        # xxxxx k = 1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        numerator1 = W1_H * H11 * F1
        numerator1 = numerator1 * numerator1.H
        numerator1 = np.abs(np.diag(numerator1))

        denominator1 = W1_H * H10 * F0 + W1_H * H12 * F2
        denominator1 = denominator1 * denominator1.H
        denominator1 = np.abs(np.diag(denominator1))
        expected_SINRs[1] = numerator1 / denominator1

        # xxxxx k = 2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        numerator2 = W2_H * H22 * F2
        numerator2 = numerator2 * numerator2.H
        numerator2 = np.abs(np.diag(numerator2))

        denominator2 = W2_H * H20 * F0 + W2_H * H21 * F1
        denominator2 = denominator2 * denominator2.H
        denominator2 = np.abs(np.diag(denominator2))
        expected_SINRs[2] = numerator2 / denominator2

        for k in range(K):
            np.testing.assert_array_almost_equal(SINRs[k], expected_SINRs[k])

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Repeat the calculation, but now including the noise
        noise_var = 1e-2
        iasolver.noise_var = noise_var
        SINRs = iasolver.calc_SINR_old()

        expected_SINRs2 = np.empty(K, dtype=np.ndarray)

        # xxxxx k = 0 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        noise_term0 = W0_H * W0 * noise_var
        denominator0_with_noise = denominator0 + np.abs(np.diag(noise_term0))
        expected_SINRs2[0] = numerator0 / denominator0_with_noise

        # xxxxx k = 1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        noise_term1 = W1_H * W1 * noise_var
        denominator1_with_noise = denominator1 + np.abs(np.diag(noise_term1))
        expected_SINRs2[1] = numerator1 / denominator1_with_noise

        # xxxxx k = 2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        noise_term2 = W2_H * W2 * noise_var
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
        self.assertAlmostEqual(self.iasolver._runned_iterations, 0.0)


class MaxSinrIASolerTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        multiUserChannel = channels.MultiUserChannelMatrix()
        self.iasolver = MaxSinrIASolver(multiUserChannel)
        self.K = 3
        self.Nt = np.ones(self.K, dtype=int) * 2
        self.Nr = np.ones(self.K, dtype=int) * 2
        self.Ns = np.ones(self.K, dtype=int) * 1

        # Transmit power of all users
        self.P = np.array([1.2, 1.5, 0.9])

        # xxxxx Debug xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.random.seed(42)  # Used in the generation of the random precoder
        self.iasolver._multiUserChannel.set_channel_seed(324)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        multiUserChannel.randomize(self.Nr, self.Nt, self.K)
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._updateW()

    def test_calc_Bkl_cov_matrix_first_part_rev(self):
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

                expected_first_part_rev = expected_first_part_rev + (self.P[j] / self.Ns[j]) * aux

            np.testing.assert_array_almost_equal(
                expected_first_part_rev,
                self.iasolver._calc_Bkl_cov_matrix_first_part_rev(k)
            )

    def test_calc_Bkl_cov_matrix_second_part_rev(self):
        for k in range(self.K):
            Hkk = self.iasolver._get_channel_rev(k, k)
            Hkk_H = Hkk.transpose().conjugate()
            for l in range(self.Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = self.iasolver._W[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(Hkk,
                                              np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
                expected_second_part = (self.P[k] / self.Ns[k]) * \
                    expected_second_part
                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.iasolver._calc_Bkl_cov_matrix_second_part_rev(k, l))

    def test_calc_Bkl_cov_matrix_all_l(self):
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
                second_part = self.iasolver._calc_Bkl_cov_matrix_second_part(k, l)
                expected_Bkl[l] = first_part - second_part + np.eye(self.Nr[k])

            Bkl_all_l = self.iasolver._calc_Bkl_cov_matrix_all_l(k, noise_power=1.0)

            # Test if the Bkl for all l of user k were calculated correctly
            for l in range(self.Ns[k]):
                np.testing.assert_array_almost_equal(expected_Bkl[l], Bkl_all_l[l])

    def test_calc_Bkl_cov_matrix_all_l_rev(self):
        self.iasolver.noise_var = 1.0

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
                second_part = self.iasolver._calc_Bkl_cov_matrix_second_part_rev(
                    k, l)
                expected_Bkl[l] = first_part - second_part + np.eye(self.Nr[k])

            Bkl_all_l = self.iasolver._calc_Bkl_cov_matrix_all_l_rev(k)

            # Test if the Bkl for all l of user k were calculated correctly
            for l in range(self.Ns[k]):
                np.testing.assert_array_almost_equal(expected_Bkl[l], Bkl_all_l[l])

    def test_calc_Ukl(self):
        for k in range(self.K):
            Hkk = self.iasolver._get_channel(k, k)
            Bkl_all_l = self.iasolver._calc_Bkl_cov_matrix_all_l(k)
            F = self.iasolver.F[k]
            for l in range(self.Ns[k]):
                expected_Ukl = np.dot(
                    np.linalg.inv(Bkl_all_l[l]),
                    np.dot(Hkk, F[:, l:l + 1]))
                expected_Ukl = expected_Ukl / norm(expected_Ukl, 'fro')
                Ukl = self.iasolver._calc_Ukl(Hkk, F, Bkl_all_l[l], k, l)
                np.testing.assert_array_almost_equal(expected_Ukl, Ukl)

    def teste_calc_Uk(self):
        for k in range(self.K):
            Bkl_all_l = self.iasolver._calc_Bkl_cov_matrix_all_l(k)
            expected_Uk = np.empty(self.Ns[k], dtype=np.ndarray)
            Hkk = self.iasolver._get_channel(k, k)
            Vk = self.iasolver.F[k]
            Uk = self.iasolver._calc_Uk(Hkk, Vk, Bkl_all_l, k)

            expected_Uk = np.empty([self.Nr[k], self.Ns[k]], dtype=complex)
            for l in range(self.Ns[k]):
                expected_Uk[:, l] = self.iasolver._calc_Ukl(Hkk, Vk, Bkl_all_l[l], k, l)[:, 0]
            np.testing.assert_array_almost_equal(expected_Uk, Uk)

    def test_underline_calc_SINR_k(self):
        multiUserChannel = channels.MultiUserChannelMatrix()
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
            # Uk_H = iasolver.full_W_H[k]

            SINR_k_all_l = iasolver._calc_SINR_k(k, Bkl_all_l)

            for l in range(Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = iasolver.full_F[k][:, l:l + 1]
                aux = np.dot(Ukl_H,
                             np.dot(Hkk, Vkl))

                expectedSINRkl = np.asscalar(
                    np.dot(aux, aux.transpose().conjugate()) / np.dot(
                        Ukl_H, np.dot(Bkl_all_l[l], Ukl))
                )

                np.testing.assert_array_almost_equal(expectedSINRkl,
                                                     SINR_k_all_l[l])

    def test_calc_SINR(self):
        multiUserChannel = channels.MultiUserChannelMatrix()
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

        # xxxxxxxxxx Noise Variance of 0.0 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # k = 0
        B0l_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k=0, noise_power=0.0)
        expected_SINR0 = iasolver._calc_SINR_k(0, B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k=1, noise_power=0.0)
        expected_SINR1 = iasolver._calc_SINR_k(1, B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 1
        B2l_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k=2, noise_power=0.0)
        expected_SINR2 = iasolver._calc_SINR_k(2, B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Noise Variance of 0.1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # k = 0
        iasolver.noise_var = 0.1
        SINR_all_users = iasolver.calc_SINR()
        B0l_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k=0, noise_power=0.1)
        expected_SINR0 = iasolver._calc_SINR_k(0, B0l_all_l)
        np.testing.assert_almost_equal(expected_SINR0, SINR_all_users[0])

        # k = 1
        B1l_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k=1, noise_power=0.1)
        expected_SINR1 = iasolver._calc_SINR_k(1, B1l_all_l)
        np.testing.assert_almost_equal(expected_SINR1, SINR_all_users[1])

        # k = 1
        B2l_all_l = iasolver._calc_Bkl_cov_matrix_all_l(k=2, noise_power=0.1)
        expected_SINR2 = iasolver._calc_SINR_k(2, B2l_all_l)
        np.testing.assert_almost_equal(expected_SINR2, SINR_all_users[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_get_channel_rev(self):
        for k in range(self.K):
            for l in range(self.K):
                Hlk = self.iasolver._get_channel(l, k)
                expected_Hkl_rev = Hlk.transpose().conjugate()
                Hkl_rev = self.iasolver._get_channel_rev(k, l)
                np.testing.assert_array_almost_equal(expected_Hkl_rev, Hkl_rev)

    def test_calc_Uk_all_k(self):
        Uk = self.iasolver._calc_Uk_all_k()

        for k in range(self.K):
            Hkk = self.iasolver._get_channel(k, k)
            Vk = self.iasolver.F[k]
            Bkl_all_l = self.iasolver._calc_Bkl_cov_matrix_all_l(k)
            expectedUk = self.iasolver._calc_Uk(Hkk, Vk, Bkl_all_l, k)
            np.testing.assert_array_almost_equal(Uk[k], expectedUk)

    def test_calc_Uk_all_k_rev(self):
        Uk = self.iasolver._calc_Uk_all_k_rev()

        for k in range(self.K):
            Hkk = self.iasolver._get_channel_rev(k, k)
            Vk = self.iasolver._W[k]
            Bkl_all_l = self.iasolver._calc_Bkl_cov_matrix_all_l_rev(k)
            expectedUk = self.iasolver._calc_Uk(Hkk, Vk, Bkl_all_l, k)
            np.testing.assert_array_almost_equal(Uk[k], expectedUk)

    # Test the calc_Q_rev method from IASolverBaseClass
    def test_calc_Q_rev(self):
        K = 3
        Nt = np.array([2, 2, 2])
        Nr = np.array([3, 3, 3])
        Ns = np.array([1, 1, 1])
        multiUserChannel = self.iasolver._multiUserChannel

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(Nr, Nt, K)
        self.iasolver.randomizeF(Ns, P)
        self.iasolver._W = self.iasolver._calc_Uk_all_k()

        # xxxxx Calculate the expected Q[0]_rev after one step xxxxxxxxxxxx
        k = 0
        H01_F1_rev = np.dot(
            self.iasolver._get_channel_rev(k, 1),
            self.iasolver._W[1]
        )
        H02_F2_rev = np.dot(
            self.iasolver._get_channel_rev(k, 2),
            self.iasolver._W[2]
        )
        expected_Q0_rev = np.dot(P[1] * H01_F1_rev,
                                 H01_F1_rev.transpose().conjugate()) + \
            np.dot(P[2] * H02_F2_rev,
                   H02_F2_rev.transpose().conjugate())

        Q0_rev = self.iasolver.calc_Q_rev(k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Q0_rev, expected_Q0_rev)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0_rev = np.dot(
            self.iasolver._get_channel_rev(k, 0),
            self.iasolver._W[0]
        )
        H12_F2_rev = np.dot(
            self.iasolver._get_channel_rev(k, 2),
            self.iasolver._W[2]
        )
        expected_Q1_rev = np.dot(P[0] * H10_F0_rev,
                                 H10_F0_rev.transpose().conjugate()) + \
            np.dot(P[2] * H12_F2_rev,
                   H12_F2_rev.transpose().conjugate())

        Q1_rev = self.iasolver.calc_Q_rev(k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Q1_rev, expected_Q1_rev)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H20_F0_rev = np.dot(
            self.iasolver._get_channel_rev(k, 0),
            self.iasolver._W[0]
        )
        H21_F1_rev = np.dot(
            self.iasolver._get_channel_rev(k, 1),
            self.iasolver._W[1]
        )
        expected_Q2_rev = np.dot(P[0] * H20_F0_rev,
                                 H20_F0_rev.transpose().conjugate()) + \
            np.dot(P[1] * H21_F1_rev,
                   H21_F1_rev.transpose().conjugate())

        Q2_rev = self.iasolver.calc_Q_rev(k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Q2_rev, expected_Q2_rev)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_updateW(self):
        self.iasolver._updateW()

        for k in range(self.iasolver.K):
            np.testing.assert_array_almost_equal(self.iasolver.W[k], self.iasolver._calc_Uk_all_k()[k])

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
        self.iasolver._step()
        F = self.iasolver.F
        full_W_H = self.iasolver.full_W_H
        H00 = self.iasolver._get_channel(0, 0)
        H11 = self.iasolver._get_channel(1, 1)
        H22 = self.iasolver._get_channel(2, 2)
        self.assertAlmostEqual(np.dot(full_W_H[0], np.dot(H00, F[0]))[0][0], 1.0)
        self.assertAlmostEqual(np.dot(full_W_H[1], np.dot(H11, F[1]))[0][0], 1.0)
        self.assertAlmostEqual(np.dot(full_W_H[2], np.dot(H22, F[2]))[0][0], 1.0)

    def test_solve(self):
        K = 3
        Nt = np.ones(K, dtype=int) * 4
        Nr = np.ones(K, dtype=int) * 4
        Ns = np.ones(K, dtype=int) * 2

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        multiUserChannel = channels.MultiUserChannelMatrix()
        multiUserChannel.randomize(Nr, Nt, K)
        iasolver = MaxSinrIASolver(multiUserChannel)
        iasolver.P = P
        iasolver.noise_var = 1e-20
        # iasolver.max_iterations = 200

        iasolver.solve(Ns)

        full_W_H = iasolver.full_W_H
        F = iasolver.F
        H00 = iasolver._get_channel(0, 0)
        H11 = iasolver._get_channel(1, 1)
        H22 = iasolver._get_channel(2, 2)

        H01 = iasolver._get_channel(0, 1)
        # xxxxx Test the remaining interference xxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.assertAlmostEqual(np.dot(full_W_H[0], np.dot(H00, F[0]))[0][0], 1.0)
        self.assertAlmostEqual(np.dot(full_W_H[1], np.dot(H11, F[1]))[0][0], 1.0)
        self.assertAlmostEqual(np.dot(full_W_H[2], np.dot(H22, F[2]))[0][0], 1.0)

        # self.assertAlmostEqual(np.dot(full_W_H[0], np.dot(H01, F[1]))[0][0], 0.0)
        # self.assertAlmostEqual(np.dot(full_W_H[0], np.dot(H02, F[2]))[0][0], 0.0)

        # F0 = np.matrix(self.iasolver._F[0])
        # F1 = np.matrix(self.iasolver._F[1])
        # F2 = np.matrix(self.iasolver._F[2])

        # W_H0 = np.matrix(self.iasolver.W_H[0])
        # W_H1 = np.matrix(self.iasolver.W_H[1])
        # W_H2 = np.matrix(self.iasolver.W_H[2])

        # H00 = np.matrix(self.iasolver._get_channel(0,0))
        # H01 = np.matrix(self.iasolver._get_channel(0,1))
        # H02 = np.matrix(self.iasolver._get_channel(0,2))
        # H10 = np.matrix(self.iasolver._get_channel(1,0))
        # H11 = np.matrix(self.iasolver._get_channel(1,1))
        # H12 = np.matrix(self.iasolver._get_channel(1,2))
        # H20 = np.matrix(self.iasolver._get_channel(2,0))
        # H21 = np.matrix(self.iasolver._get_channel(2,1))
        # H22 = np.matrix(self.iasolver._get_channel(2,2))

        # import pudb; pudb.set_trace()  ## DEBUG ##y

        # TODO: DARLAN -> Implement-me


# TODO: Finish the implementation
class MinLeakageIASolverTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        multiUserChannel = channels.MultiUserChannelMatrix()
        self.iasolver = MinLeakageIASolver(multiUserChannel)
        self.K = 3
        self.Nt = np.ones(self.K, dtype=int) * 2
        self.Nr = np.ones(self.K, dtype=int) * 2
        self.Ns = np.ones(self.K, dtype=int) * 1

        # Transmit power of all users
        self.P = np.array([1.2, 1.5, 0.9])

        # xxxxx Debug xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.random.seed(42)  # Used in the generation of the random precoder
        self.iasolver._multiUserChannel.set_channel_seed(324)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        multiUserChannel.randomize(self.Nr, self.Nt, self.K)
        #self.iasolver.randomizeF(self.Ns, self.P)
        #self.iasolver._W = self.iasolver._calc_Uk_all_k()

    def test_getCost(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._W = self.iasolver._calc_Uk_all_k()

        Q0 = np.matrix(self.iasolver.calc_Q(0))
        W0 = np.matrix(self.iasolver._W[0])
        Q1 = np.matrix(self.iasolver.calc_Q(1))
        W1 = np.matrix(self.iasolver._W[1])
        Q2 = np.matrix(self.iasolver.calc_Q(2))
        W2 = np.matrix(self.iasolver._W[2])
        expected_cost = np.trace(np.abs(
            W0.H * Q0 * W0 + W1.H * Q1 * W1 + W2.H * Q2 * W2))
        self.assertAlmostEqual(expected_cost, self.iasolver.get_cost())

        self.iasolver._step()
        Q0 = np.matrix(self.iasolver.calc_Q(0))
        W0 = np.matrix(self.iasolver._W[0])
        Q1 = np.matrix(self.iasolver.calc_Q(1))
        W1 = np.matrix(self.iasolver._W[1])
        Q2 = np.matrix(self.iasolver.calc_Q(2))
        W2 = np.matrix(self.iasolver._W[2])
        expected_cost2 = np.trace(np.abs(
            W0.H * Q0 * W0 + W1.H * Q1 * W1 + W2.H * Q2 * W2))
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
        for i in range(5):
            self.iasolver._step()
            new_cost = self.iasolver.get_cost()
            self.assertTrue(new_cost < last_cost)
            last_cost = new_cost

    def test_updateW(self):
        self.iasolver.randomizeF(self.Ns, self.P)
        self.iasolver._updateW()

        for k in range(self.iasolver.K):
            np.testing.assert_array_almost_equal(self.iasolver.W[k], self.iasolver._calc_Uk_all_k()[k])

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
        F = self.iasolver.F
        full_W_H = self.iasolver.full_W_H
        H00 = self.iasolver._get_channel(0, 0)
        H11 = self.iasolver._get_channel(1, 1)
        H22 = self.iasolver._get_channel(2, 2)
        self.assertAlmostEqual(np.dot(full_W_H[0], np.dot(H00, F[0]))[0][0], 1.0)
        self.assertAlmostEqual(np.dot(full_W_H[1], np.dot(H11, F[1]))[0][0], 1.0)
        self.assertAlmostEqual(np.dot(full_W_H[2], np.dot(H22, F[2]))[0][0], 1.0)

    def test_solve(self):
        self.iasolver.max_iterations = 1
        # We are only testing if this does not thrown an exception. That's
        # why there is no assert clause here
        self.iasolver.solve(Ns=1)


# TODO: finish implementation
class MMSEIASolverTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        multiUserChannel = channels.MultiUserChannelMatrix()
        self.iasolver = MMSEIASolver(multiUserChannel)

        self.K = 3
        self.Nt = np.ones(self.K, dtype=int) * 2
        self.Nr = np.ones(self.K, dtype=int) * 2
        self.Ns = np.ones(self.K, dtype=int) * 1

        # # xxxxx Debug xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # multiUserChannel.set_channel_seed(43)
        # multiUserChannel.set_noise_seed(456)
        # np.random.seed(25)
        # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Transmit power of all users
        self.P = np.array([1.2, 1.5, 0.9])

        multiUserChannel.randomize(self.Nr, self.Nt, self.K)

        self.iasolver._initialize_F_and_W(1, 1)
        self.iasolver.P = self.P

        self.iasolver.noise_var = 1e-3

    def test_updateW(self):
        P = self.iasolver.P

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
        aux0 = 0.0
        H00_F0 = np.sqrt(P[0]) * np.dot(H00, F0)
        aux = np.dot(H00, F0)
        aux0 = aux0 + (np.dot(aux, aux.conj().T) * P[0])
        aux = np.dot(H01, F1)
        aux0 = aux0 + (np.dot(aux, aux.conj().T) * P[1])
        aux = np.dot(H02, F2)
        aux0 = aux0 + (np.dot(aux, aux.conj().T) * P[2])
        expected_W0 = np.dot(
            np.linalg.inv(aux0 + self.iasolver.noise_var * np.eye(self.Nr[0])),
            H00_F0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for the user 1 xxxxx
        aux1 = 0.0
        H11_F1 = np.sqrt(P[1]) * np.dot(H11, F1)
        aux = np.dot(H10, F0)
        aux1 = aux1 + np.dot(aux, aux.conj().T) * P[0]
        aux = np.dot(H11, F1)
        aux1 = aux1 + np.dot(aux, aux.conj().T) * P[1]
        aux = np.dot(H12, F2)
        aux1 = aux1 + np.dot(aux, aux.conj().T) * P[2]
        expected_W1 = np.dot(
            np.linalg.inv(aux1 + self.iasolver.noise_var * np.eye(self.Nr[1])),
            H11_F1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for the user 2 xxxxx
        aux2 = 0.0
        H22_F2 = np.sqrt(P[2]) * np.dot(H22, F2)
        aux = np.dot(H20, F0)
        aux2 = aux2 + np.dot(aux, aux.conj().T) * P[0]
        aux = np.dot(H21, F1)
        aux2 = aux2 + np.dot(aux, aux.conj().T) * P[1]
        aux = np.dot(H22, F2)
        aux2 = aux2 + np.dot(aux, aux.conj().T) * P[2]
        expected_W2 = np.dot(
            np.linalg.inv(aux2 + self.iasolver.noise_var * np.eye(self.Nr[1])),
            H22_F2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Update the receive filters
        self.iasolver._updateW()

        # Test if the update was performed correctly
        np.testing.assert_array_almost_equal(self.iasolver.W[0],
                                             expected_W0)
        np.testing.assert_array_almost_equal(self.iasolver.W[1],
                                             expected_W1)
        np.testing.assert_array_almost_equal(self.iasolver.W[2],
                                             expected_W2)

    def test_calc_Vi(self):
        # For now we use an arbitrarily chosen value
        mu = np.array([0.9, 1.1, 0.8])

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

        expected_V0 = np.dot(
            np.linalg.inv(aux0 + mu[0] * np.eye(self.Nt[0])),
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

        expected_V1 = np.dot(
            np.linalg.inv(aux1 + mu[1] * np.eye(self.Nt[1])),
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

        expected_V2 = np.dot(
            np.linalg.inv(aux2 + mu[2] * np.eye(self.Nt[2])),
            H22_herm_U2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Calculates the precoders using the _calc_Vi method
        V0 = self.iasolver._calc_Vi(0, mu[0])
        V1 = self.iasolver._calc_Vi(1, mu[1])
        V2 = self.iasolver._calc_Vi(2, mu[2])

        # Test if the calculated values are equal to the expected values
        np.testing.assert_array_almost_equal(expected_V0, V0)
        np.testing.assert_array_almost_equal(expected_V1, V1)
        np.testing.assert_array_almost_equal(expected_V2, V2)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now lets repeat the tests, but without specifying the value of
        # mu. Therefore, the optimum value of mu will be also calculated.

        V0_best = self.iasolver._calc_Vi(0)
        V1_best = self.iasolver._calc_Vi(1)
        V2_best = self.iasolver._calc_Vi(2)

        # TODO: Find a way to test the case when the best value of mu is found

    # TODO: Finish this implementation or erase this method
    def test_calc_Vi2(self):
        # This method test the case when IA is not feasible
        K = 3
        Nt = 4 * np.ones(K)
        Nr = 4 * np.ones(K)
        Ns = 2
        P = 1.0

        # This specific channel will yield a degenerated solution solution
        big_H = np.load("{0}/tests/big_H.npy".format(parent_dir))
        F = np.load("{0}/tests/F.npy".format(parent_dir))
        W = np.load("{0}/tests/W.npy".format(parent_dir))

        multi_user_channel = channels.MultiUserChannelMatrix()
        multi_user_channel.init_from_channel_matrix(big_H, Nr, Nt, K)

        iasolver = MMSEIASolver(multi_user_channel)
        iasolver.noise_var=10  # Set a very high noise variance (that is, a
                               # very low SINR)
        #iasolver._initialize_F_and_W(Ns, P)

        # iasolver._F = F
        # iasolver._W = W

        #import pudb; pudb.set_trace()  ## DEBUG ##
        iasolver.solve(Ns, P)

        iasolver._calc_Vi(0)

        #print np.linalg.svd(iasolver.F[0])[1]

        #import pudb; pudb.set_trace()  ## DEBUG ##

        # iasolver.noise_var = 1e-6
        # iasolver.solve(Ns)

    # TODO: Finish this implementation or erase this method
    def test_calc_Vi3(self):
        # This method test the case when IA is not feasible
        K = 3
        Nt = 3 * np.ones(K)
        Nr = 3 * np.ones(K)
        Ns = 2
        P = 1.0

        # This specific channel will yield a degenerated solution solution
        big_H = np.load("{0}/tests/big_H2.npy".format(parent_dir))
        multi_user_channel = channels.MultiUserChannelMatrix()
        multi_user_channel.init_from_channel_matrix(big_H, Nr, Nt, K)

        iasolver = MMSEIASolver(multi_user_channel)
        iasolver.noise_var = 1000.0

        iasolver.solve(Ns)

    def test_updateF(self):
        self.iasolver._updateF()

        self.assertAlmostEqual(norm(self.iasolver.F[0], 'fro') ** 2, 1.0)
        self.assertAlmostEqual(norm(self.iasolver.F[1], 'fro') ** 2, 1.0)
        self.assertAlmostEqual(norm(self.iasolver.F[2], 'fro') ** 2, 1.0)

        self.assertAlmostEqual(norm(self.iasolver.full_F[0], 'fro') ** 2,
                               self.iasolver.P[0])
        self.assertAlmostEqual(norm(self.iasolver.full_F[1], 'fro') ** 2,
                               self.iasolver.P[1])
        self.assertAlmostEqual(norm(self.iasolver.full_F[2], 'fro') ** 2,
                               self.iasolver.P[2])

        # TODO: implement-me
        pass

    def test_solve(self):
        # Test if the solution is better then the closed form solution
        self.iasolver.P = 1.0
        Ns = self.Ns
        #self.iasolver.max_iterations = 200
        self.iasolver.noise_var = 1e-20

        self.iasolver.solve(Ns)
        # print self.iasolver.calc_SINR_old()

        F0 = np.matrix(self.iasolver._F[0])
        F1 = np.matrix(self.iasolver._F[1])
        F2 = np.matrix(self.iasolver._F[2])

        W_H0 = np.matrix(self.iasolver.W_H[0])
        W_H1 = np.matrix(self.iasolver.W_H[1])
        W_H2 = np.matrix(self.iasolver.W_H[2])

        H00 = np.matrix(self.iasolver._get_channel(0,0))
        H01 = np.matrix(self.iasolver._get_channel(0,1))
        H02 = np.matrix(self.iasolver._get_channel(0,2))
        H10 = np.matrix(self.iasolver._get_channel(1,0))
        H11 = np.matrix(self.iasolver._get_channel(1,1))
        H12 = np.matrix(self.iasolver._get_channel(1,2))
        H20 = np.matrix(self.iasolver._get_channel(2,0))
        H21 = np.matrix(self.iasolver._get_channel(2,1))
        H22 = np.matrix(self.iasolver._get_channel(2,2))

        # import pudb; pudb.set_trace()  ## DEBUG ##


        # # Tesf if the solution aligns the interference
        # for l in range(3):
        #     for k in range(3):
        #         Hlk = self.iasolver._get_channel(l, k)
        #         Wl_H = self.iasolver.W_H[l]
        #         Fk = self.iasolver.F[k]
        #         s = np.dot(Wl_H, np.dot(Hlk, Fk))[0][0]
        #         if l == k:
        #             Hk_eq = self.iasolver._calc_equivalent_channel(k)
        #             s2 = s / Hk_eq[0, 0]  # We only have one stream -> the
        #                                   # equivalent channel is an scalar.
        #             self.assertAlmostEqual(1.0, s2)
        #         # else:
        #         #     # Test if the interference is equal to 0.0
        #         #     self.assertAlmostEqual(0.0, s)

        # TODO: Implement-me
        pass


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    # plot_psd_OFDM_symbols()
    unittest.main()
