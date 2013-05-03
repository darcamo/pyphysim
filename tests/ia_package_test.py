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

import ia  # Import the package ia
from ia.ia import AlternatingMinIASolver, IASolverBaseClass, MaxSinrIASolver
from util.misc import peig, leig, randn_c


# UPDATE THIS CLASS if another module is added to the ia package
class IaDoctestsTestCase(unittest.TestCase):
    """Teste case that run all the doctests in the modules of the ia
    package."""

    def test_ia(self):
        """Run doctests in the ia module."""
        doctest.testmod(ia)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx IA Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class IASolverBaseClassTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.iasolver = IASolverBaseClass()

    def test_properties(self):
        K = 3
        Nr = np.array([2, 4, 6])
        Nt = np.array([2, 3, 5])
        Ns = np.array([1, 2, 3])
        self.iasolver.randomizeH(Nr, Nt, K)
        self.iasolver.randomizeF(Nt, Ns, K)

        # Test the properties
        self.assertEqual(self.iasolver.K, K)
        np.testing.assert_array_equal(self.iasolver.Nr, Nr)
        np.testing.assert_array_equal(self.iasolver.Nt, Nt)
        np.testing.assert_array_equal(self.iasolver.Ns, Ns)

    def test_randomizeF(self):
        K = 3
        Nt = np.array([2, 3, 5])
        Ns = np.array([1, 2, 3])
        P = np.array([1.2, 0.9, 1.4])  # Power of each user

        self.iasolver.randomizeF(Nt, Ns, K, P)

        # The shape of the precoder is the number of users
        self.assertEqual(self.iasolver.F.shape, (K,))

        # The power of each user
        np.testing.assert_array_almost_equal(self.iasolver.P, P)

        # The shape of the precoder of each user is Nt[user] x Ns[user]
        self.assertEqual(self.iasolver.F[0].shape, (Nt[0], Ns[0]))
        self.assertEqual(self.iasolver.F[1].shape, (Nt[1], Ns[1]))
        self.assertEqual(self.iasolver.F[2].shape, (Nt[2], Ns[2]))

        # Test if the generated precoder of each user has a Frobenius norm
        # equal to one.
        self.assertAlmostEqual(np.linalg.norm(self.iasolver.F[0], 'fro'), 1.)
        self.assertAlmostEqual(np.linalg.norm(self.iasolver.F[1], 'fro'), 1.)
        self.assertAlmostEqual(np.linalg.norm(self.iasolver.F[2], 'fro'), 1.)

        # Test when the number of streams and transmit antennas is an
        # scalar (the same value will be used for all users)
        Nt = 3
        Ns = 2
        self.iasolver.randomizeF(Nt, Ns, K)
        # The shape of the precoder of each user is Nt[user] x Ns[user]
        self.assertEqual(self.iasolver.F[0].shape, (Nt, Ns))
        self.assertEqual(self.iasolver.F[1].shape, (Nt, Ns))
        self.assertEqual(self.iasolver.F[2].shape, (Nt, Ns))

        # Test if the power is None (which means "use 1" whenever needed),
        # since it was not set.
        self.assertIsNone(self.iasolver.P)

    def test_calc_Q(self):
        K = 3
        Nt = np.array([2, 2, 2])
        Nr = np.array([2, 2, 2])
        Ns = np.array([1, 1, 1])

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        self.iasolver.randomizeF(Nt, Ns, K, P)
        self.iasolver.randomizeH(Nr, Nt, K)

        # xxxxx Calculate the expected Q[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H01_F1 = np.dot(
            self.iasolver.get_channel(k, 1),
            self.iasolver.F[1]
        )
        H02_F2 = np.dot(
            self.iasolver.get_channel(k, 2),
            self.iasolver.F[2]
        )
        expected_Q0 = np.dot(P[1] * H01_F1,
                             H01_F1.transpose().conjugate()) + \
                      np.dot(P[2] * H02_F2,
                             H02_F2.transpose().conjugate())

        Qk = self.iasolver.calc_Q(k)
        # Test if Qk is equal to the expected output
        np.testing.assert_array_almost_equal(Qk, expected_Q0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected Q[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0 = np.dot(
            self.iasolver.get_channel(k, 0),
            self.iasolver.F[0]
        )
        H12_F2 = np.dot(
            self.iasolver.get_channel(k, 2),
            self.iasolver.F[2]
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
            self.iasolver.get_channel(k, 0),
            self.iasolver.F[0]
        )
        H21_F1 = np.dot(
            self.iasolver.get_channel(k, 1),
            self.iasolver.F[1]
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

        # Transmit power of all users
        P = np.array([1.2, 1.5, 0.9])

        self.iasolver.randomizeF(Nt, Ns, K, P)
        self.iasolver.randomizeH(Nr, Nt, K)

        #xxxxxxxxxx k = 0 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 0
        Qk = self.iasolver.calc_Q(k)
        pk = self.iasolver.calc_remaining_interference_percentage(k, Qk)

        [V, D] = leig(Qk, Ns[k])
        expected_pk = np.sum(np.abs(D)) / np.abs(np.trace(Qk))
        self.assertAlmostEqual(pk, expected_pk)

        #xxxxxxxxxx k = 1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 1
        Qk = self.iasolver.calc_Q(k)
        pk = self.iasolver.calc_remaining_interference_percentage(k)

        [V, D] = leig(Qk, Ns[k])
        expected_pk = np.sum(np.abs(D)) / np.abs(np.trace(Qk))
        self.assertAlmostEqual(pk, expected_pk)

        #xxxxxxxxxx k = 2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        k = 2
        Qk = self.iasolver.calc_Q(k)
        pk = self.iasolver.calc_remaining_interference_percentage(k)

        [V, D] = leig(Qk, Ns[k])
        expected_pk = np.sum(np.abs(D)) / np.abs(np.trace(Qk))
        self.assertAlmostEqual(pk, expected_pk)

    def test_solve(self):
        with self.assertRaises(NotImplementedError):
            self.iasolver.solve()


class AlternatingMinIASolverTestCase(unittest.TestCase):
    """Unittests for the AlternatingMinIASolver class in the ia module."""
    def setUp(self):
        """Called before each test."""
        self.iasolver = AlternatingMinIASolver()
        self.K = 3
        self.Nr = np.array([2, 4, 6])
        self.Nt = np.array([2, 3, 5])
        self.Ns = np.array([1, 2, 3])
        self.iasolver.randomizeH(self.Nr, self.Nt, self.K)
        self.iasolver.randomizeF(self.Nt, self.Ns, self.K)

    def test_updateC(self):
        # We only need to initialize a random channel here for this test
        # and "self.iasolver.randomizeH(self.Nr, self.Nt, self.K)" would be simpler. However,
        # in order to call the init_from_channel_matrix at least once in
        # these tests we are using it here.
        self.iasolver.init_from_channel_matrix(
            randn_c(np.sum(self.Nr), np.sum(self.Nt)),
            self.Nr,
            self.Nt,
            self.K)

        # Dimensions of the interference subspace
        Ni = self.Nr - self.Ns

        self.iasolver.updateC()

        # xxxxx Calculate the expected C[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H01_F1 = np.dot(
            self.iasolver.get_channel(k, 1),
            self.iasolver.F[1]
        )
        H02_F2 = np.dot(
            self.iasolver.get_channel(k, 2),
            self.iasolver.F[2]
        )
        expected_C0 = np.dot(H01_F1, H01_F1.transpose().conjugate()) + \
                      np.dot(H02_F2, H02_F2.transpose().conjugate())
        expected_C0 = peig(expected_C0, Ni[k])[0]

        # Test if C[0] is equal to the expected output
        np.testing.assert_array_almost_equal(self.iasolver.C[0], expected_C0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected C[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0 = np.dot(
            self.iasolver.get_channel(k, 0),
            self.iasolver.F[0]
        )
        H12_F2 = np.dot(
            self.iasolver.get_channel(k, 2),
            self.iasolver.F[2]
        )
        expected_C1 = np.dot(H10_F0, H10_F0.transpose().conjugate()) + \
                      np.dot(H12_F2, H12_F2.transpose().conjugate())
        expected_C1 = peig(expected_C1, Ni[k])[0]

        # Test if C[1] is equal to the expected output
        np.testing.assert_array_almost_equal(self.iasolver.C[1], expected_C1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected C[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H20_F0 = np.dot(
            self.iasolver.get_channel(k, 0),
            self.iasolver.F[0]
        )
        H21_F1 = np.dot(
            self.iasolver.get_channel(k, 1),
            self.iasolver.F[1]
        )
        expected_C2 = np.dot(H20_F0, H20_F0.transpose().conjugate()) + \
                      np.dot(H21_F1, H21_F1.transpose().conjugate())
        expected_C2 = peig(expected_C2, Ni[k])[0]

        # Test if C[2] is equal to the expected output
        np.testing.assert_array_almost_equal(self.iasolver.C[2], expected_C2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_updateF(self):
        self.iasolver.updateC()
        self.iasolver.updateF()

        # xxxxxxxxxx Aliases for each channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        H01 = self.iasolver.get_channel(0, 1)
        H02 = self.iasolver.get_channel(0, 2)

        H10 = self.iasolver.get_channel(1, 0)
        H12 = self.iasolver.get_channel(1, 2)

        H20 = self.iasolver.get_channel(2, 0)
        H21 = self.iasolver.get_channel(2, 1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Aliases for (I-Ck Ck^H)) for each k xxxxxxxxxxxxxxxxxx
        Y0 = np.eye(self.Nr[0], dtype=complex) - \
             np.dot(
                 self.iasolver.C[0],
                 self.iasolver.C[0].conjugate().transpose())

        Y1 = np.eye(self.Nr[1], dtype=complex) - \
             np.dot(
                 self.iasolver.C[1],
                 self.iasolver.C[1].conjugate().transpose())

        Y2 = np.eye(self.Nr[2], dtype=complex) - \
             np.dot(
                 self.iasolver.C[2],
                 self.iasolver.C[2].conjugate().transpose())
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
        # Call updateC, updateF and updateW
        self.iasolver.step()

        # xxxxx Calculates the expected receive filter for user 0 xxxxxxxxx
        tildeH0 = np.dot(
            self.iasolver.get_channel(0, 0),
            self.iasolver.F[0])
        tildeH0 = np.hstack([tildeH0, self.iasolver.C[0]])
        expected_W0 = np.linalg.inv(tildeH0)[0:self.iasolver.Ns[0]]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for user 1 xxxxxxxxx
        tildeH1 = np.dot(
            self.iasolver.get_channel(1, 1),
            self.iasolver.F[1])
        tildeH1 = np.hstack([tildeH1, self.iasolver.C[1]])
        expected_W1 = np.linalg.inv(tildeH1)[0:self.iasolver.Ns[1]]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for user 2 xxxxxxxxx
        tildeH2 = np.dot(
            self.iasolver.get_channel(2, 2),
            self.iasolver.F[2])
        tildeH2 = np.hstack([tildeH2, self.iasolver.C[2]])
        expected_W2 = np.linalg.inv(tildeH2)[0:self.iasolver.Ns[2]]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Finally perform the tests xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_almost_equal(self.iasolver.W[0], expected_W0)
        np.testing.assert_array_almost_equal(self.iasolver.W[1], expected_W1)
        np.testing.assert_array_almost_equal(self.iasolver.W[2], expected_W2)

    def test_getCost(self):
        K = 2
        Nr = np.array([3, 3])
        Nt = np.array([3, 3])
        Ns = np.array([2, 2])
        self.iasolver.randomizeH(Nr, Nt, K)
        self.iasolver.randomizeF(Nt, Ns, K)

        # Call updateC, updateF and updateW
        self.iasolver.step()

        Cost = 0
        k, l = (0, 1)
        H01_F1 = np.dot(
            self.iasolver.get_channel(k, l),
            self.iasolver.F[l])
        Cost = Cost + np.linalg.norm(
            H01_F1 -
            np.dot(
                np.dot(self.iasolver.C[k], self.iasolver.C[k].transpose().conjugate()),
                H01_F1
            ), 'fro') ** 2

        k, l = (1, 0)
        H10_F0 = np.dot(
            self.iasolver.get_channel(k, l),
            self.iasolver.F[l])
        Cost = Cost + np.linalg.norm(
            H10_F0 -
            np.dot(
                np.dot(self.iasolver.C[k], self.iasolver.C[k].transpose().conjugate()),
                H10_F0
            ), 'fro') ** 2

        self.assertAlmostEqual(self.iasolver.getCost(), Cost)

    def test_solve(self):
        self.iasolver.max_iterations = 1
        # We are only testing if this does not thrown an exception. That's
        # why there is no assert clause here
        self.iasolver.solve()


# TODO: finish implementation
class MaxSinrIASolverTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.iasolver = MaxSinrIASolver()
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

        self.iasolver.randomizeF(self.Nt, self.Ns, self.K, self.P)
        self.iasolver.randomizeH(self.Nr, self.Nt, self.K)
        self.iasolver._W = self.iasolver.calc_Uk_all_k()

    def test_calc_Bkl_cov_matrix_first_part(self):
        for k in range(self.K):
            expected_first_part = 0.0  # First part in the equation of Bkl
                                       # (the double summation)

            # The outer for loop will calculate
            # first_part = $\sum_{j=1}^{K} \frac{P[k]}{Ns[k]} \text{aux}$
            for j in range(self.K):
                aux = 0.0  # The inner for loop will calculate
                            # $\text{aux} = \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$
                Hkj = self.iasolver.get_channel(k, j)
                Hkj_H = Hkj.conjugate().transpose()

                for d in range(self.Ns[k]):
                    Vjd = self.iasolver.F[j][:, d:d + 1]
                    Vjd_H = Vjd.conjugate().transpose()
                    aux = aux + np.dot(np.dot(Hkj, np.dot(Vjd, Vjd_H)), Hkj_H)

                expected_first_part = expected_first_part + \
                    (self.P[j] / self.Ns[j]) * aux

            np.testing.assert_array_almost_equal(
                expected_first_part,
                self.iasolver._calc_Bkl_cov_matrix_first_part(k))

    def test_calc_Bkl_cov_matrix_first_part_rev(self):
        for k in range(self.K):
            expected_first_part_rev = 0.0
            for j in range(self.K):
                aux = 0.0
                Hkj = self.iasolver.get_channel_rev(k, j)
                Hkj_H = Hkj.conjugate().transpose()

                for d in range(self.Ns[k]):
                    Vjd = self.iasolver.W[j][:, d:d + 1]
                    Vjd_H = Vjd.conjugate().transpose()
                    aux = aux + np.dot(np.dot(Hkj, np.dot(Vjd, Vjd_H)), Hkj_H)

                expected_first_part_rev = expected_first_part_rev + (self.P[j] / self.Ns[j]) * aux

            np.testing.assert_array_almost_equal(
                expected_first_part_rev,
                self.iasolver._calc_Bkl_cov_matrix_first_part_rev(k)
            )

    def test_calc_Bkl_cov_matrix_second_part(self):
        for k in range(self.K):
            Hkk = self.iasolver.get_channel(k, k)
            Hkk_H = Hkk.transpose().conjugate()
            for l in range(self.Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = self.iasolver.F[k][:, l:l + 1]
                Vkl_H = Vkl.transpose().conjugate()
                expected_second_part = np.dot(Hkk,
                                              np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
                expected_second_part = (self.P[k] / self.Ns[k]) * \
                    expected_second_part
                np.testing.assert_array_almost_equal(
                    expected_second_part,
                    self.iasolver._calc_Bkl_cov_matrix_second_part(k, l))

    def test_calc_Bkl_cov_matrix_second_part_rev(self):
        for k in range(self.K):
            Hkk = self.iasolver.get_channel_rev(k, k)
            Hkk_H = Hkk.transpose().conjugate()
            for l in range(self.Ns[k]):
                # Calculate the second part in Equation (28). The second part
                # is different for each value of l and is given by
                # second_part = $\frac{P[k]}{Ns} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk] \dagger}$
                Vkl = self.iasolver.W[k][:, l:l + 1]
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

            Bkl_all_l = self.iasolver.calc_Bkl_cov_matrix_all_l(k)

            # Test if the Bkl for all l of user k were calculated correctly
            for l in range(self.Ns[k]):
                np.testing.assert_array_almost_equal(expected_Bkl[l], Bkl_all_l[l])

    def test_calc_Bkl_cov_matrix_all_l_rev(self):
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

            Bkl_all_l = self.iasolver.calc_Bkl_cov_matrix_all_l_rev(k)

            # Test if the Bkl for all l of user k were calculated correctly
            for l in range(self.Ns[k]):
                np.testing.assert_array_almost_equal(expected_Bkl[l], Bkl_all_l[l])

    def test_calc_Ukl(self):
        for k in range(self.K):
            Hkk = self.iasolver.get_channel(k, k)
            Bkl_all_l = self.iasolver.calc_Bkl_cov_matrix_all_l(k)
            F = self.iasolver.F[k]
            for l in range(self.Ns[k]):
                expected_Ukl = np.dot(
                    np.linalg.inv(Bkl_all_l[l]),
                    np.dot(Hkk, F[:, l:l + 1]))
                expected_Ukl = expected_Ukl / np.linalg.norm(expected_Ukl, 'fro')
                Ukl = self.iasolver._calc_Ukl(Hkk, F, Bkl_all_l[l], k, l)
                np.testing.assert_array_almost_equal(expected_Ukl, Ukl)

    def teste_calc_Uk(self):
        for k in range(self.K):
            Bkl_all_l = self.iasolver.calc_Bkl_cov_matrix_all_l(k)
            expected_Uk = np.empty(self.Ns[k], dtype=np.ndarray)
            Hkk = self.iasolver.get_channel(k, k)
            Vk = self.iasolver.F[k]
            Uk = self.iasolver._calc_Uk(Hkk, Vk, Bkl_all_l, k)

            expected_Uk = np.empty([self.Nr[k], self.Ns[k]], dtype=complex)
            for l in range(self.Ns[k]):
                expected_Uk[:, l] = self.iasolver._calc_Ukl(Hkk, Vk, Bkl_all_l[l], k, l)[:, 0]
            np.testing.assert_array_almost_equal(expected_Uk, Uk)

    def test_calc_SINR_k(self):
        for k in range(self.K):
            Hkk = self.iasolver.get_channel(k, k)
            Vk = self.iasolver.F[k]
            Bkl_all_l = self.iasolver.calc_Bkl_cov_matrix_all_l(k)
            Uk = self.iasolver._calc_Uk(Hkk, Vk, Bkl_all_l, k)

            SINR_k_all_l = self.iasolver.calc_SINR_k(Bkl_all_l, Uk, k)

            for l in range(self.Ns[k]):
                Ukl = Uk[:, l:l + 1]
                Ukl_H = Ukl.transpose().conjugate()
                Vkl = self.iasolver.F[k][:, l:l + 1]
                aux = np.dot(Ukl_H,
                             np.dot(Hkk, Vkl))

                expectedSINRkl = np.asscalar(
                    np.dot(aux, aux.transpose().conjugate()) * (self.P[k] / self.Ns[k]) / np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))
                )

                np.testing.assert_array_almost_equal(expectedSINRkl,
                                                     SINR_k_all_l[l])

    def test_get_channel_rev(self):
        for k in range(self.K):
            for l in range(self.K):
                Hlk = self.iasolver.get_channel(l, k)
                expected_Hkl_rev = Hlk.transpose().conjugate()
                Hkl_rev = self.iasolver.get_channel_rev(k, l)
                np.testing.assert_array_almost_equal(expected_Hkl_rev, Hkl_rev)

    def test_calc_Uk_all_k(self):
        Uk = self.iasolver.calc_Uk_all_k()

        for k in range(self.K):
            Hkk = self.iasolver.get_channel(k, k)
            Vk = self.iasolver.F[k]
            Bkl_all_l = self.iasolver.calc_Bkl_cov_matrix_all_l(k)
            expectedUk = self.iasolver._calc_Uk(Hkk, Vk, Bkl_all_l, k)
            np.testing.assert_array_almost_equal(Uk[k], expectedUk)

    def test_calc_Uk_all_k_rev(self):
        Uk = self.iasolver.calc_Uk_all_k_rev()

        for k in range(self.K):
            Hkk = self.iasolver.get_channel_rev(k, k)
            Vk = self.iasolver.W[k]
            Bkl_all_l = self.iasolver.calc_Bkl_cov_matrix_all_l_rev(k)
            expectedUk = self.iasolver._calc_Uk(Hkk, Vk, Bkl_all_l, k)
            np.testing.assert_array_almost_equal(Uk[k], expectedUk)

    # def test_step(self):
    #     repmax = 20
    #     pk = np.zeros([repmax, self.K])
    #     #Qk = np.empty([self.K, repmax], dtype=np.ndarray)
    #     SINR_k = np.zeros([repmax, self.K])

    #     print

    #     # print "F00:\n{0}".format(self.iasolver.F[0])
    #     # print "F10:\n{0}".format(self.iasolver.F[1])
    #     # print "F20:\n{0}".format(self.iasolver.F[2])

    #     #self.iasolver.step()
    #     #print "um passo feito"
    #     # print "F00:\n{0}".format(self.iasolver.F[0])
    #     # print "F10:\n{0}".format(self.iasolver.F[1])
    #     # print "F20:\n{0}".format(self.iasolver.F[2])

    #     for step in range(repmax):
    #         self.iasolver.step()
    #         for k in range(self.K):
    #             Qk = self.iasolver.calc_Q(k)
    #             pk[step, k] = self.iasolver.calc_remaining_interference_percentage(k, Qk)

    #             Bkl = self.iasolver.calc_Bkl_cov_matrix_all_l(k)
    #             SINR_k[step, k] = self.iasolver.calc_SINR_k(Bkl, self.iasolver.W[k], k)
    #     print "SINR_k:\n{0}".format(SINR_k)
    #         # Calc SINR

    #     print "pk:\n{0}".format(pk)
    #         # xxxxxxxxxxxxxxxxxxxx

    # def test_solve(self):
    #     self.iasolver.solve()
    #     # TODO: Implement-me
    #     pass


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    # plot_psd_OFDM_symbols()
    unittest.main()


## Use this if you want to optimize the code on the ia.ia module
# if __name__ == '__main__':
#     import time
#     from misc import pretty_time

#     tic = time.time()

#     K = 4
#     Nr = np.array([5, 5, 5, 5])
#     Nt = np.array([5, 5, 5, 5])
#     Ns = np.array([2, 2, 2, 2])
#     alt = AlternatingMinIASolver()
#     alt.randomizeH(Nr, Nt, K)
#     alt.randomizeF(Nt, Ns, K)

#     maxIter = 5000
#     Cost = np.zeros(maxIter)
#     for i in np.arange(maxIter):
#         alt.step()
#         Cost[i] = alt.getCost()

#     toc = time.time()
#     print pretty_time(toc - tic)
