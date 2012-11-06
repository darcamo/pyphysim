#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the ia package.

Each module has several doctests that we run in addition to the unittests
defined here.
"""

import unittest
import doctest

import numpy as np

import sys
sys.path.append("../")

import ia  # Import the package ia
from ia.ia import AlternatingMinIASolver
from util.misc import peig, leig, randn_c


# UPDATE THIS CLASS if another module is added to the comm package
class IaDoctestsTestCase(unittest.TestCase):
    """Teste case that run all the doctests in the modules of the ia
    package."""

    def test_ia(self):
        """Run doctests in the ia module."""
        doctest.testmod(ia)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx IA Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class AlternatingMinIASolverTestCase(unittest.TestCase):
    """Unittests for the AlternatingMinIASolver class in the ia module."""
    def setUp(self):
        """Called before each test."""
        self.alt = AlternatingMinIASolver()

    def test_properties(self):
        K = 3
        Nr = np.array([2, 4, 6])
        Nt = np.array([2, 3, 5])
        Ns = np.array([1, 2, 3])
        self.alt.randomizeH(Nr, Nt, K)
        self.alt.randomizeF(Nt, Ns, K)

        # Test the properties
        self.assertEqual(self.alt.K, K)
        np.testing.assert_array_equal(self.alt.Nr, Nr)
        np.testing.assert_array_equal(self.alt.Nt, Nt)
        np.testing.assert_array_equal(self.alt.Ns, Ns)

    def test_randomizeF(self):
        K = 3
        Nt = np.array([2, 3, 5])
        Ns = np.array([1, 2, 3])
        self.alt.randomizeF(Nt, Ns, K)

        # The shape of the precoder is the number of users
        self.assertEqual(self.alt.F.shape, (K,))

        # The shape of the precoder of each user is Nt[user] x Ns[user]
        self.assertEqual(self.alt.F[0].shape, (Nt[0], Ns[0]))
        self.assertEqual(self.alt.F[1].shape, (Nt[1], Ns[1]))
        self.assertEqual(self.alt.F[2].shape, (Nt[2], Ns[2]))

        # Test if the generated precoder of each user has a Frobenius norm
        # equal to one.
        self.assertAlmostEqual(np.linalg.norm(self.alt.F[0], 'fro'), 1.)
        self.assertAlmostEqual(np.linalg.norm(self.alt.F[1], 'fro'), 1.)
        self.assertAlmostEqual(np.linalg.norm(self.alt.F[2], 'fro'), 1.)

        # Test when the number of streams and transmit antennas is an
        # scalar (the same value will be used for all users)
        Nt = 3
        Ns = 2
        self.alt.randomizeF(Nt, Ns, K)
        # The shape of the precoder of each user is Nt[user] x Ns[user]
        self.assertEqual(self.alt.F[0].shape, (Nt, Ns))
        self.assertEqual(self.alt.F[1].shape, (Nt, Ns))
        self.assertEqual(self.alt.F[2].shape, (Nt, Ns))

    def test_updateC(self):
        K = 3
        Nr = np.array([2, 4, 6])
        Nt = np.array([2, 3, 5])
        Ns = np.array([1, 2, 3])
        # We only need to initialize a random channel here for this test
        # and "self.alt.randomizeH(Nr, Nt, K)" would be simpler. However,
        # in order to call the init_from_channel_matrix at least once in
        # these tests we are using it here.
        self.alt.init_from_channel_matrix(
            randn_c(np.sum(Nr), np.sum(Nt)),
            Nr,
            Nt,
            K)
        self.alt.randomizeF(Nt, Ns, K)

        # Dimensions of the interference subspace
        Ni = Nr - Ns

        self.alt.updateC()

        # xxxxx Calculate the expected C[0] after one step xxxxxxxxxxxxxxxx
        k = 0
        H01_F1 = np.dot(
            self.alt.get_channel(k, 1),
            self.alt.F[1]
        )
        H02_F2 = np.dot(
            self.alt.get_channel(k, 2),
            self.alt.F[2]
        )
        expected_C0 = np.dot(H01_F1, H01_F1.transpose().conjugate()) + \
                      np.dot(H02_F2, H02_F2.transpose().conjugate())
        expected_C0 = peig(expected_C0, Ni[k])[0]

        # Test if C[0] is equal to the expected output
        np.testing.assert_array_almost_equal(self.alt.C[0], expected_C0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected C[1] after one step xxxxxxxxxxxxxxxx
        k = 1
        H10_F0 = np.dot(
            self.alt.get_channel(k, 0),
            self.alt.F[0]
        )
        H12_F2 = np.dot(
            self.alt.get_channel(k, 2),
            self.alt.F[2]
        )
        expected_C1 = np.dot(H10_F0, H10_F0.transpose().conjugate()) + \
                      np.dot(H12_F2, H12_F2.transpose().conjugate())
        expected_C1 = peig(expected_C1, Ni[k])[0]

        # Test if C[1] is equal to the expected output
        np.testing.assert_array_almost_equal(self.alt.C[1], expected_C1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected C[2] after one step xxxxxxxxxxxxxxxx
        k = 2
        H20_F0 = np.dot(
            self.alt.get_channel(k, 0),
            self.alt.F[0]
        )
        H21_F1 = np.dot(
            self.alt.get_channel(k, 1),
            self.alt.F[1]
        )
        expected_C2 = np.dot(H20_F0, H20_F0.transpose().conjugate()) + \
                      np.dot(H21_F1, H21_F1.transpose().conjugate())
        expected_C2 = peig(expected_C2, Ni[k])[0]

        # Test if C[2] is equal to the expected output
        np.testing.assert_array_almost_equal(self.alt.C[2], expected_C2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_updateF(self):
        K = 3
        Nr = np.array([2, 4, 6])
        Nt = np.array([2, 3, 5])
        Ns = np.array([1, 2, 3])
        self.alt.randomizeH(Nr, Nt, K)
        self.alt.randomizeF(Nt, Ns, K)

        self.alt.updateC()
        self.alt.updateF()

        # xxxxxxxxxx Aliases for each channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        H01 = self.alt.get_channel(0, 1)
        H02 = self.alt.get_channel(0, 2)

        H10 = self.alt.get_channel(1, 0)
        H12 = self.alt.get_channel(1, 2)

        H20 = self.alt.get_channel(2, 0)
        H21 = self.alt.get_channel(2, 1)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Aliases for (I-Ck Ck^H)) for each k xxxxxxxxxxxxxxxxxx
        Y0 = np.eye(Nr[0], dtype=complex) - \
             np.dot(
                 self.alt.C[0],
                 self.alt.C[0].conjugate().transpose())

        Y1 = np.eye(Nr[1], dtype=complex) - \
             np.dot(
                 self.alt.C[1],
                 self.alt.C[1].conjugate().transpose())

        Y2 = np.eye(Nr[2], dtype=complex) - \
             np.dot(
                 self.alt.C[2],
                 self.alt.C[2].conjugate().transpose())
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected F[0] after one step xxxxxxxxxxxxxxxx
        # l = 0 -> k = 1 and k = 2
        expected_F0 = np.dot(np.dot(H10.conjugate().transpose(), Y1), H10) + \
                      np.dot(np.dot(H20.conjugate().transpose(), Y2), H20)
        expected_F0 = leig(expected_F0, Ns[0])[0]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected F[1] after one step xxxxxxxxxxxxxxxx
        # l = 1 -> k = 0 and k = 2
        expected_F1 = np.dot(np.dot(H01.conjugate().transpose(), Y0), H01) + \
                      np.dot(np.dot(H21.conjugate().transpose(), Y2), H21)
        expected_F1 = leig(expected_F1, Ns[1])[0]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculate the expected F[1] after one step xxxxxxxxxxxxxxxx
        # l = 2 -> k = 0 and k = 1
        expected_F2 = np.dot(np.dot(H02.conjugate().transpose(), Y0), H02) + \
                      np.dot(np.dot(H12.conjugate().transpose(), Y1), H12)
        expected_F2 = leig(expected_F2, Ns[2])[0]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Finally perform the tests xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_almost_equal(self.alt.F[0], expected_F0)
        np.testing.assert_array_almost_equal(self.alt.F[1], expected_F1)
        np.testing.assert_array_almost_equal(self.alt.F[2], expected_F2)

    def test_updateW(self):
        K = 3
        Nr = np.array([2, 4, 6])
        Nt = np.array([2, 3, 5])
        Ns = np.array([1, 2, 3])
        self.alt.randomizeH(Nr, Nt, K)
        self.alt.randomizeF(Nt, Ns, K)

        # Call updateC, updateF and updateW
        self.alt.step()

        # xxxxx Calculates the expected receive filter for user 0 xxxxxxxxx
        tildeH0 = np.dot(
            self.alt.get_channel(0, 0),
            self.alt.F[0])
        tildeH0 = np.hstack([tildeH0, self.alt.C[0]])
        expected_W0 = np.linalg.inv(tildeH0)[0:self.alt.Ns[0]]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for user 1 xxxxxxxxx
        tildeH1 = np.dot(
            self.alt.get_channel(1, 1),
            self.alt.F[1])
        tildeH1 = np.hstack([tildeH1, self.alt.C[1]])
        expected_W1 = np.linalg.inv(tildeH1)[0:self.alt.Ns[1]]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the expected receive filter for user 2 xxxxxxxxx
        tildeH2 = np.dot(
            self.alt.get_channel(2, 2),
            self.alt.F[2])
        tildeH2 = np.hstack([tildeH2, self.alt.C[2]])
        expected_W2 = np.linalg.inv(tildeH2)[0:self.alt.Ns[2]]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Finally perform the tests xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        np.testing.assert_array_almost_equal(self.alt.W[0], expected_W0)
        np.testing.assert_array_almost_equal(self.alt.W[1], expected_W1)
        np.testing.assert_array_almost_equal(self.alt.W[2], expected_W2)

    def test_getCost(self):
        K = 2
        Nr = np.array([3, 3])
        Nt = np.array([3, 3])
        Ns = np.array([2, 2])
        self.alt.randomizeH(Nr, Nt, K)
        self.alt.randomizeF(Nt, Ns, K)

        # Call updateC, updateF and updateW
        self.alt.step()

        Cost = 0
        k, l = (0, 1)
        H01_F1 = np.dot(
            self.alt.get_channel(k, l),
            self.alt.F[l])
        Cost = Cost + np.linalg.norm(
            H01_F1 -
            np.dot(
                np.dot(self.alt.C[k], self.alt.C[k].transpose().conjugate()),
                H01_F1
            ), 'fro') ** 2

        k, l = (1, 0)
        H10_F0 = np.dot(
            self.alt.get_channel(k, l),
            self.alt.F[l])
        Cost = Cost + np.linalg.norm(
            H10_F0 -
            np.dot(
                np.dot(self.alt.C[k], self.alt.C[k].transpose().conjugate()),
                H10_F0
            ), 'fro') ** 2

        self.assertAlmostEqual(self.alt.getCost(), Cost)


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
