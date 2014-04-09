#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1103

"""Tests for the modules in the util package.

Each module has doctests for its functions and all we need to do is run all
of them.
"""

__revision__ = "$Revision$"

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
current_dir = os.path.abspath(os.path.dirname(__file__))
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import unittest
import doctest
import numpy as np

from pyphysim.util import misc, conversion


class UtilDoctestsTestCase(unittest.TestCase):
    """Test case that run all the doctests in the modules of the util
    package.

    """
    def test_misc(self):
        """Run misc doctests"""
        doctest.testmod(misc)

    def test_conversion(self):
        """Run conversion doctests"""
        doctest.testmod(conversion)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Conversion Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ConversionTestCase(unittest.TestCase):
    def test_single_matrix_to_matrix_of_matrices(self):
        nrows = np.array([2, 4, 6])
        ncols = np.array([2, 3, 5])
        single_matrix = np.array(
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

        # xxxxx Convert 'single 2D array' to 2D array of 2D arrays xxxxxxxx
        matrix_of_matrices = conversion.single_matrix_to_matrix_of_matrices(
            single_matrix, nrows, ncols)
        self.assertEqual(matrix_of_matrices.shape, (3, 3))

        np.testing.assert_array_equal(
            matrix_of_matrices[0, 0],
            np.ones([2, 2]) * 0)

        np.testing.assert_array_equal(
            matrix_of_matrices[0, 1],
            np.ones([2, 3]) * 1)

        np.testing.assert_array_equal(
            matrix_of_matrices[0, 2],
            np.ones([2, 5]) * 2)

        np.testing.assert_array_equal(
            matrix_of_matrices[1, 0],
            np.ones([4, 2]) * 3)

        np.testing.assert_array_equal(
            matrix_of_matrices[1, 1],
            np.ones([4, 3]) * 4)

        np.testing.assert_array_equal(
            matrix_of_matrices[1, 2],
            np.ones([4, 5]) * 5)

        np.testing.assert_array_equal(
            matrix_of_matrices[2, 0],
            np.ones([6, 2]) * 6)

        np.testing.assert_array_equal(
            matrix_of_matrices[2, 1],
            np.ones([6, 3]) * 7)

        np.testing.assert_array_equal(
            matrix_of_matrices[2, 2],
            np.ones([6, 5]) * 8)

        # xxxxx Convert 'single 2D array' to 2D array of 2D arrays xxxxxxxx
        # In this case we break the channel into packs of lines
        matrix_of_matrices2 = conversion.single_matrix_to_matrix_of_matrices(
            single_matrix, nrows)
        self.assertEqual(matrix_of_matrices2.shape, (3,))

        expected1 = np.array([[0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
                              [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]])
        expected2 = np.array([[3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
                              [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
                              [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
                              [3, 3, 4, 4, 4, 5, 5, 5, 5, 5]])
        expected3 = np.array([[6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                              [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                              [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                              [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                              [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
                              [6, 6, 7, 7, 7, 8, 8, 8, 8, 8]])

        np.testing.assert_array_equal(expected1, matrix_of_matrices2[0])
        np.testing.assert_array_equal(expected2, matrix_of_matrices2[1])
        np.testing.assert_array_equal(expected3, matrix_of_matrices2[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Convert 'single 2D array' to 2D array of 2D arrays xxxxxxxx
        # In this case we break the channel into packs of columns
        matrix_of_matrices3 = conversion.single_matrix_to_matrix_of_matrices(
            single_matrix, None, ncols)
        self.assertEqual(matrix_of_matrices3.shape, (3,))

        expected1 = np.array([[0, 0], [0, 0], [3, 3], [3, 3], [3, 3], [3, 3],
                              [6, 6], [6, 6], [6, 6], [6, 6], [6, 6], [6, 6]])
        expected2 = np.array([[1, 1, 1], [1, 1, 1], [4, 4, 4], [4, 4, 4],
                              [4, 4, 4], [4, 4, 4], [7, 7, 7], [7, 7, 7],
                              [7, 7, 7], [7, 7, 7], [7, 7, 7], [7, 7, 7]])
        expected3 = np.array([[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [5, 5, 5, 5, 5],
                              [5, 5, 5, 5, 5], [5, 5, 5, 5, 5], [5, 5, 5, 5, 5],
                              [8, 8, 8, 8, 8], [8, 8, 8, 8, 8], [8, 8, 8, 8, 8],
                              [8, 8, 8, 8, 8], [8, 8, 8, 8, 8], [8, 8, 8, 8, 8]])
        np.testing.assert_array_equal(expected1, matrix_of_matrices3[0])
        np.testing.assert_array_equal(expected2, matrix_of_matrices3[1])
        np.testing.assert_array_equal(expected3, matrix_of_matrices3[2])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Convert 'single 1D array' to 1D array of 1D arrays xxxxxxxx
        #nrows = np.array([2, 4, 6])
        expected1 = np.ones(2) * 2
        expected2 = np.ones(4) * 4
        expected3 = np.ones(6) * 6
        single_array = np.hstack([expected1, expected2, expected3])
        array_of_arrays = conversion.single_matrix_to_matrix_of_matrices(
            single_array, nrows)

        self.assertEqual(array_of_arrays.shape, (3,))
        np.testing.assert_array_equal(array_of_arrays[0],
                                      expected1)

        np.testing.assert_array_equal(array_of_arrays[1],
                                      expected2)

        np.testing.assert_array_equal(array_of_arrays[2],
                                      expected3)

    def test_dB2Linear(self):
        self.assertAlmostEqual(conversion.dB2Linear(30),
                               1000.0)

    def test_linear2dB(self):
        self.assertAlmostEqual(conversion.linear2dB(1000),
                               30.0)

    def test_dBm2Linear(self):
        self.assertAlmostEqual(conversion.dBm2Linear(60),
                               1000.0)

    def test_linear2dBm(self):
        self.assertAlmostEqual(conversion.linear2dBm(1000),
                               60.0)

    def test_binary2gray(self):
        np.testing.assert_array_equal(conversion.binary2gray(np.arange(0, 8)),
                                      np.array([0, 1, 3, 2, 6, 7, 5, 4]))

    def test_gray2binary(self):
        vec = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        np.testing.assert_array_equal(
            conversion.gray2binary(conversion.binary2gray(vec)),
            vec)

    def test_SNR_dB_to_EbN0_dB(self):
        bps_1 = 2
        bps_2 = 4
        SNR1 = 10
        SNR2 = 15

        self.assertAlmostEqual(conversion.SNR_dB_to_EbN0_dB(SNR1, bps_1),
                               6.98970004336)
        self.assertAlmostEqual(conversion.SNR_dB_to_EbN0_dB(SNR1, bps_2),
                               3.97940008672)

        self.assertAlmostEqual(conversion.SNR_dB_to_EbN0_dB(SNR2, bps_1),
                               11.9897000434)
        self.assertAlmostEqual(conversion.SNR_dB_to_EbN0_dB(SNR2, bps_2),
                               8.97940008672)

    def test_EbN0_dB_to_SNR_dB(self):
        bps_1 = 2
        bps_2 = 4
        SNR1 = 10
        SNR2 = 15

        EbN0_1_1 = conversion.SNR_dB_to_EbN0_dB(SNR1, bps_1)
        EbN0_1_2 = conversion.SNR_dB_to_EbN0_dB(SNR1, bps_2)

        EbN0_2_1 = conversion.SNR_dB_to_EbN0_dB(SNR2, bps_1)
        EbN0_2_2 = conversion.SNR_dB_to_EbN0_dB(SNR2, bps_2)

        self.assertAlmostEqual(conversion.EbN0_dB_to_SNR_dB(EbN0_1_1, bps_1),
                               SNR1)
        self.assertAlmostEqual(conversion.EbN0_dB_to_SNR_dB(EbN0_1_2, bps_2),
                               SNR1)

        self.assertAlmostEqual(conversion.EbN0_dB_to_SNR_dB(EbN0_2_1, bps_1),
                               SNR2)
        self.assertAlmostEqual(conversion.EbN0_dB_to_SNR_dB(EbN0_2_2, bps_2),
                               SNR2)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx misc Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MiscFunctionsTestCase(unittest.TestCase):
    """Test the functions in the module."""
    def test_peig(self):
        A = np.array(
            [[2 - 0j, 3 + 12j, 7 + 1j],
             [3 - 12j, 6 + 0j, 5 + 3j],
             [7 - 1j, 5 - 3j, 4 + 0j]])

        # Test if an exception is raised if 'n' is greater then the number
        # of columns of A
        with self.assertRaises(ValueError):
            misc.peig(A, 4)

        # xxxxx Test for n==3 (all columns) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n3, _] = misc.peig(A, 3)

        expected_V_n3 = np.array(
            [[0.27354856 + 0.54286421j, 0.15266747 - 0.35048035j, 0.69593520],
             [0.68522942, -0.24255902 + 0.37567057j, -0.02693857 + 0.57425752j],
             [0.38918583 + 0.09728652j, 0.80863645, -0.40625488 - 0.14189355j]])
        np.testing.assert_array_almost_equal(V_n3, expected_V_n3)

        # xxxxx Test for n==2 (two columns) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n2, _] = misc.peig(A, 2)

        expected_V_n2 = np.array(
            [[0.27354856 + 0.54286421j, 0.15266747 - 0.35048035j],
             [0.68522942, -0.24255902 + 0.37567057j],
             [0.38918583 + 0.09728652j, 0.80863645]])
        np.testing.assert_array_almost_equal(V_n2, expected_V_n2)

        # xxxxx Test for n==1 (one column) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n1, _] = misc.peig(A, 1)

        expected_V_n1 = np.array(
            [[0.27354856 + 0.54286421j],
             [0.68522942],
             [0.38918583 + 0.09728652j]])
        np.testing.assert_array_almost_equal(V_n1, expected_V_n1)

    def test_leig(self):
        A = np.array(
            [[2 - 0j, 3 + 12j, 7 + 1j],
             [3 - 12j, 6 + 0j, 5 + 3j],
             [7 - 1j, 5 - 3j, 4 + 0j]])

        # Test if an exception is raised if 'n' is greater then the number
        # of columns of A
        with self.assertRaises(ValueError):
            misc.leig(A, 4)

        # xxxxx Test for n==3 (all columns) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n3, _] = misc.leig(A, 3)

        expected_V_n3 = np.array(
            [[0.69593520, 0.15266747 - 0.35048035j, 0.27354856 + 0.54286421j],
             [-0.02693857 + 0.57425752j, -0.24255902 + 0.37567057j, 0.68522942],
             [-0.40625488 - 0.14189355j, 0.80863645, 0.38918583 + 0.09728652j]])
        np.testing.assert_array_almost_equal(V_n3, expected_V_n3)

        # xxxxx Test for n==2 (two columns) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n2, _] = misc.leig(A, 2)

        expected_V_n2 = np.array(
            [[0.69593520, 0.15266747 - 0.35048035j],
             [-0.02693857 + 0.57425752j, -0.24255902 + 0.37567057j],
             [-0.40625488 - 0.14189355j, 0.80863645]])
        np.testing.assert_array_almost_equal(V_n2, expected_V_n2)

        # xxxxx Test for n==1 (one column) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n1, _] = misc.leig(A, 1)

        expected_V_n1 = np.array(
            [[0.69593520],
             [-0.02693857 + 0.57425752j],
             [-0.40625488 - 0.14189355j]])
        np.testing.assert_array_almost_equal(V_n1, expected_V_n1)

    def test_level2bits(self):
        self.assertEqual(
            list(map(misc.level2bits, range(1, 20))),
            [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5])

        # Test if an exception is raised for a value of n lower then 1
        with self.assertRaises(ValueError):
            misc.level2bits(0)
        with self.assertRaises(ValueError):
            misc.level2bits(-2)

    def test_int2bits(self):
        self.assertEqual(
            list(map(misc.int2bits, range(0, 19))),
            [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5])

        # Test if an exception is raised for a negative value of n
        with self.assertRaises(ValueError):
            misc.int2bits(-1)

    def test_count_bits(self):
        n = np.r_[0:9, 15]

        # First we test for the scalar input values
        self.assertEqual(misc.count_bits(0), 0)
        self.assertEqual(misc.count_bits(1), 1)
        self.assertEqual(misc.count_bits(2), 1)
        self.assertEqual(misc.count_bits(3), 2)
        self.assertEqual(misc.count_bits(4), 1)
        self.assertEqual(misc.count_bits(5), 2)
        self.assertEqual(misc.count_bits(6), 2)
        self.assertEqual(misc.count_bits(7), 3)
        self.assertEqual(misc.count_bits(8), 1)
        self.assertEqual(misc.count_bits(15), 4)

        # Now we test for a numpy array
        expected_num_bits = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 4])
        np.testing.assert_array_equal(misc.count_bits(n), expected_num_bits)

    def test_count_bit_errors(self):
        a = np.random.randint(0, 16, 20)
        b = np.random.randint(0, 16, 20)
        expected_bit_count = np.sum(misc.count_bits(misc.xor(a, b)))
        self.assertEqual(expected_bit_count, misc.count_bit_errors(a, b))

    def test_qfunc(self):
        self.assertAlmostEqual(misc.qfunc(0.0), 0.5)
        self.assertAlmostEqual(misc.qfunc(1.0), 0.158655254, 9)
        self.assertAlmostEqual(misc.qfunc(3.0), 0.001349898, 9)

    def test_least_right_singular_vectors(self):
        A = np.array([1, 2, 3, 6, 5, 4, 2, 2, 1])
        A.shape = (3, 3)
        (min_Vs, remaining_Vs, S) = misc.least_right_singular_vectors(A, 1)
        np.testing.assert_array_almost_equal(min_Vs,
                                             np.array([[-0.4474985],
                                                       [0.81116484],
                                                       [-0.3765059]]),
                                             8)
        np.testing.assert_array_almost_equal(
            remaining_Vs,
            np.array([[-0.62341491, -0.64116998],
                      [0.01889071, -0.5845124],
                      [0.78166296, -0.49723869]]),
            8)

        np.testing.assert_array_almost_equal(
            S,
            np.array([1.88354706, 9.81370681]),
            8)

    def test_calc_unorm_autocorr(self):
        x = np.array([4., 2, 1, 3, 7, 3, 8])
        unorm_autocor = misc.calc_unorm_autocorr(x)
        expected_unorm_autocor = np.array([152, 79, 82, 53, 42, 28, 32])
        np.testing.assert_array_equal(unorm_autocor, expected_unorm_autocor)

    def test_calc_autocorr(self):
        x = np.array([4., 2, 1, 3, 7, 3, 8])
        autocor = misc.calc_autocorr(x)
        expected_autocor = np.array([1., -0.025, 0.15, -0.175, -0.25, -0.2, 0.])
        np.testing.assert_array_almost_equal(autocor, expected_autocor)

    def test_calc_confidence_interval(self):
        # xxxxx Test for a 95% confidence interval xxxxxxxxxxxxxxxxxxxxxxxx
        mean = 250.2
        std = 2.5
        n = 25
        interval = misc.calc_confidence_interval(mean, std, n, P=95)
        expected_interval = [249.22, 251.18]
        np.testing.assert_array_almost_equal(interval, expected_interval)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Test for a 90% confidence interval xxxxxxxxxxxxxxxxxxxxxxxx
        mean = 101.82
        std = 1.2
        n = 6
        interval = misc.calc_confidence_interval(mean, std, n, P=90)
        expected_interval = [101.01411787, 102.62588213]
        np.testing.assert_array_almost_equal(interval, expected_interval)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_update_inv_sum_diag(self):
        A = np.random.randn(3, 3)
        D = np.diag([1.2, 1.5, 0.9])
        invA = np.linalg.inv(A)
        expected_invB = np.linalg.inv(A + D)
        invB = misc.update_inv_sum_diag(invA, D.diagonal())
        # There was a bug before that misc.update_inv_sum_diag was
        # modifying invA. We call misc.update_inv_sum_diag again to capture
        # that bug if it ever comes back.
        invB = misc.update_inv_sum_diag(invA, D.diagonal())
        np.testing.assert_array_almost_equal(expected_invB, invB)

    def test_get_principal_component_matrix(self):
        A = np.array([[0.03300776 - 0.77428109j, 0.13839634 + 0.24361978j],
                      [-0.07248757 + 0.35349072j, -0.04558698 - 0.12223548j],
                      [-0.19711349 + 0.33872698j, -0.00456194 - 0.14161498j]])
        n = 1
        new_A = misc.get_principal_component_matrix(A, n)

        # Calculates the expected new_A matrix
        [U, S, V_H] = np.linalg.svd(A)
        newS = np.zeros(3, dtype=complex)
        newS[:n] = S[:n]
        newS = np.diag(newS)[:, :2]
        expected_new_A = np.dot(U, np.dot(newS, V_H[:, :n]))

        np.testing.assert_array_almost_equal(expected_new_A, new_A)

    def test_get_range_representation(self):
        a = np.array([5, 10, 15])
        expr_a = misc.get_range_representation(a)
        expected_expr_a = "5,10,15"
        self.assertEqual(expr_a, expected_expr_a)

        a = np.array([5, 10, 15, 20])
        expr_a = misc.get_range_representation(a)
        expected_expr_a = "5:5:20"
        self.assertEqual(expr_a, expected_expr_a)

        a = np.array([5, 10, 15, 20])
        expr_a = misc.get_range_representation(a, True)
        expected_expr_a = "5_(5)_20"
        self.assertEqual(expr_a, expected_expr_a)

        a = np.array([5, 10, 15, 20, 25, 30, 35, 40])
        expr_a = misc.get_range_representation(a)
        expected_expr_a = "5:5:40"
        self.assertEqual(expr_a, expected_expr_a)

        b = np.array([2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7])
        expr_b = misc.get_range_representation(b)
        misc.get_range_representation(b)
        expected_expr_b = "2.3:0.3:4.7"
        self.assertEqual(expr_b, expected_expr_b)

        b = np.array([2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7])
        expr_b = misc.get_range_representation(b, True)
        misc.get_range_representation(b)
        expected_expr_b = "2.3_(0.3)_4.7"
        self.assertEqual(expr_b, expected_expr_b)

        c = np.array([10.2, 9., 7.8, 6.6, 5.4, 4.2])
        expr_c = misc.get_range_representation(c)
        expected_expr_c = "10.2:-1.2:4.2"
        self.assertEqual(expr_c, expected_expr_c)

        c = np.array([10.2, 9., 7.8, 6.6, 5.4, 4.2])
        expr_c = misc.get_range_representation(c, True)
        expected_expr_c = "10.2_(-1.2)_4.2"
        self.assertEqual(expr_c, expected_expr_c)

        # This array is not an arithmetic progression and
        # get_range_representation should return None
        d = np.array([1, 3, 9, 4])
        self.assertIsNone(misc.get_range_representation(d))

        # Return None when a valid range representation does not exist,
        # such as when the array has a single element or when the array is
        # not an arithmetic progression.
        self.assertIsNone(misc.get_range_representation(np.array([6, 10, 20, 50])))

    def test_get_mixed_range_representation(self):
        a = np.array([2])
        expr_a = misc.get_mixed_range_representation(a)
        expected_expr_a = "2"
        self.assertEqual(expr_a, expected_expr_a)

        a = np.array([1, 2, 4])
        expr_a = misc.get_mixed_range_representation(a)
        expected_expr_a = "1,2,4"
        self.assertEqual(expr_a, expected_expr_a)

        a = np.array([1, 2, 4, 6])
        expr_a = misc.get_mixed_range_representation(a)
        expected_expr_a = "1,2,4,6"
        self.assertEqual(expr_a, expected_expr_a)

        a = np.array([0, 2, 4, 6])
        expr_a = misc.get_mixed_range_representation(a)
        expected_expr_a = "0:2:6"
        self.assertEqual(expr_a, expected_expr_a)

        a = np.array([0, 2, 4, 6])
        expr_a = misc.get_mixed_range_representation(a, True)
        expected_expr_a = "0_(2)_6"
        self.assertEqual(expr_a, expected_expr_a)

        a = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40])
        expr_a = misc.get_mixed_range_representation(a)
        expected_expr_a = "1,5:5:40"
        self.assertEqual(expr_a, expected_expr_a)

        b = np.array([1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 100])
        expr_b = misc.get_mixed_range_representation(b)
        expected_expr_b = "1,2,3,5:5:40,100"
        self.assertEqual(expr_b, expected_expr_b)

        b = np.array([1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 100])
        expr_b = misc.get_mixed_range_representation(b, True)
        expected_expr_b = "1,2,3,5_(5)_40,100"
        self.assertEqual(expr_b, expected_expr_b)

        c = np.array([1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30, 35, 40, 50, 100])
        expr_c = misc.get_mixed_range_representation(c)
        expected_expr_c = "1:1:6,10:5:40,50,100"
        self.assertEqual(expr_c, expected_expr_c)

        c = np.array([1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30, 35, 40, 50, 100])
        expr_c = misc.get_mixed_range_representation(c, True)
        expected_expr_c = "1_(1)_6,10_(5)_40,50,100"
        self.assertEqual(expr_c, expected_expr_c)

        c = np.array([1, 2, 3, 4, 5, 6, 10, 15, 20, 25, 30, 35, 40, 50, 100, 150, 200, 250, 300, 1000])
        expr_c = misc.get_mixed_range_representation(c, True)
        expected_expr_c = "1_(1)_6,10_(5)_40,50_(50)_300,1000"
        self.assertEqual(expr_c, expected_expr_c)

        d = np.array([2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7])
        expr_d = misc.get_mixed_range_representation(d)
        expected_expr_d = "2.3:0.3:4.7"
        self.assertEqual(expr_d, expected_expr_d)

        c = np.array([10.2, 9., 7.8, 6.6, 5.4, 4.2])
        expr_c = misc.get_mixed_range_representation(c)
        expected_expr_c = "10.2:-1.2:4.2"
        self.assertEqual(expr_c, expected_expr_c)

        c = np.array([11.0, 10.2, 9., 7.8, 6.6, 5.4, 4.2])
        expr_c = misc.get_mixed_range_representation(c)
        expected_expr_c = "11.0,10.2:-1.2:4.2"
        self.assertEqual(expr_c, expected_expr_c)

        c = np.array([11.0, 10.2, 8.4, 8.2, 8.0, 7.8, 7.6, 5, 3, 1, -1, -3, -10])
        expr_c = misc.get_mixed_range_representation(c)
        expected_expr_c = "11.0,10.2,8.4:-0.2:7.6,5.0:-2.0:-3.0,-10.0"
        self.assertEqual(expr_c, expected_expr_c)

    def test_replace_dict_values(self):
        name = "something {value1} - {value2} something else {value3}"
        dictionary = {'value1': 'bla bla',
                      'value2': np.array([15]),
                      'value3': 76}
        new_name = misc.replace_dict_values(name, dictionary)
        expected_new_name = 'something bla bla - [15] something else 76'
        self.assertEqual(new_name, expected_new_name)

        name = "something {value1} - {value2} something else {value3}"
        dictionary = {'value1': 'bla bla',
                      'value2': np.array([5, 10, 15, 20, 25, 30]),
                      'value3': 76}
        new_name = misc.replace_dict_values(name, dictionary)
        expected_new_name = 'something bla bla - [5_(5)_30] something else 76'
        self.assertEqual(new_name, expected_new_name)

        # Value2 is not an arithmetic progression
        dictionary2 = {'value1': 'bla bla',
                       'value2': np.array([5, 10, 18, 20, 25, 30]),
                       'value3': 76}
        new_name2 = misc.replace_dict_values(name, dictionary2)
        expected_new_name2 = 'something bla bla - [5,10,18,20,25,30] something else 76'
        self.assertEqual(new_name2, expected_new_name2)

        # Value3 has parts that are arithmetic progressions and others that
        # are not an arithmetic progression
        dictionary3 = {'value1': 'bla bla',
                       'value2': np.array([2, 5, 10, 15, 20, 25, 30, 31, 32, 50]),
                       'value3': 76}
        new_name3 = misc.replace_dict_values(name, dictionary3)
        expected_new_name3 = 'something bla bla - [2,5_(5)_30,31,32,50] something else 76'
        self.assertEqual(new_name3, expected_new_name3)

    def test_pretty_time(self):
        self.assertEqual(misc.pretty_time(0), '0.00s')
        self.assertEqual(misc.pretty_time(2.3), '2.30s')
        self.assertEqual(misc.pretty_time(5.15), '5.15s')
        self.assertEqual(misc.pretty_time(23.44), '23.44s')

        # Note that once we passed one minute, we always use two digits for
        # the seconds
        self.assertEqual(misc.pretty_time(60), '1m:00s')
        self.assertEqual(misc.pretty_time(63), '1m:03s')

        # Note that the seconds are now rounded to the closest integer
        # value.
        self.assertEqual(misc.pretty_time(65.7), '1m:06s')

        # Note that once we passed one hour, we always use two digits for
        # the minutes
        self.assertEqual(misc.pretty_time(3745), '1h:02m:25s')

        self.assertEqual(misc.pretty_time(6000), '1h:40m:00s')
        self.assertEqual(misc.pretty_time(6015), '1h:40m:15s')
        # Lets add two minuts (120 seconds)
        self.assertEqual(misc.pretty_time(6135), '1h:42m:15s')

        self.assertEqual(misc.pretty_time(6137), '1h:42m:17s')


    def test_calc_decorrelation_matrix(self):
        A = misc.randn_c(3, 3)

        # B is symmetric and positive semi-definite
        B = np.dot(A, A.conjugate().T)

        Wd = misc.calc_decorrelation_matrix(B)

        D = np.dot(np.dot(Wd.conjugate().T, B), Wd)

        # D must be a diagonal matrix
        np.testing.assert_array_almost_equal(D, np.diag(D.diagonal()))

    def test_calc_whitening_matrix(self):
        A = misc.randn_c(3, 3)

        # B is symmetric and positive semi-definite
        B = np.dot(A, A.conjugate().T)

        Wd = misc.calc_whitening_matrix(B)

        D = np.dot(np.dot(Wd.conjugate().T, B), Wd)

        # D must be an identity matrix
        np.testing.assert_array_almost_equal(D, np.eye(3))

    def test_calc_shannon_sum_capacity(self):
        sinrs_linear = np.array([11.4, 20.3])
        expected_sum_capacity = np.sum(np.log2(1 + sinrs_linear))
        self.assertAlmostEqual(
            expected_sum_capacity,
            misc.calc_shannon_sum_capacity(sinrs_linear))


# xxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
