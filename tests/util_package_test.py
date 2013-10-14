#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from util import misc, progressbar, simulations, conversion
from util.simulations import *
from util.simulations import _parse_float_range_expr, _real_numpy_array_check, _integer_numpy_array_check


# Define a _DummyRunner class for the testing the simulate and
# simulate_in_parallel methods in the SimulationRunner class.
class _DummyRunner(simulations.SimulationRunner):
    def __init__(self):
        simulations.SimulationRunner.__init__(self)
        # Set the progress bar message to None to avoid print the
        # progressbar in these testes.
        self.rep_max = 2
        self.update_progress_function_style = None
        # Now we add a dummy parameter to our runner object
        self.params.add('SNR', np.array([0., 5., 10., 15., 20.]))
        self.params.add('bias', 1.3)
        self.params.set_unpack_parameter('SNR')

    @staticmethod
    def _run_simulation(current_params):
        SNR = current_params['SNR']
        bias = current_params['bias']
        sim_results = SimulationResults()

        value = 1.2 * SNR + bias
        # The correct result will be SNR * 1.2
        sim_results.add_new_result('lala', Result.RATIOTYPE, value, 1)
        return sim_results


class UtilDoctestsTestCase(unittest.TestCase):
    """Test case that run all the doctests in the modules of the util
    package.

    """
    def test_progressbar(self):
        """Run progressbar doctests"""
        doctest.testmod(progressbar)

    def test_misc(self):
        """Run misc doctests"""
        doctest.testmod(misc)

    def test_simulations(self):
        """Run simulations doctests"""
        doctest.testmod(simulations)

    def test_conversion(self):
        """Run conversion doctests"""
        doctest.testmod(conversion)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Conversion Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TODO: finish implementation
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
# xxxxxxxxxx simulations Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SimulationsModuleFunctionsTestCase(unittest.TestCase):
    def test_parse_range_expr(self):
        import validate

        expr = "10:15"
        expected_parsed_expr = np.r_[10:15]
        parsed_expr = _parse_float_range_expr(expr)

        np.testing.assert_array_almost_equal(expected_parsed_expr,
                                             parsed_expr)

        expr = "10:2:15"
        expected_parsed_expr = np.r_[10:15:2]
        parsed_expr = _parse_float_range_expr(expr)

        np.testing.assert_array_almost_equal(expected_parsed_expr,
                                             parsed_expr)

        expr = "-3.4:0.5:5"
        expected_parsed_expr = np.r_[-3.4:5.0:0.5]
        parsed_expr = _parse_float_range_expr(expr)

        np.testing.assert_array_almost_equal(expected_parsed_expr,
                                             parsed_expr)

        # xxxxx Test invalid values xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        expr = "a string"
        with self.assertRaises(validate.VdtTypeError):
            _parse_float_range_expr(expr)

        expr = "10,5"
        with self.assertRaises(validate.VdtTypeError):
            _parse_float_range_expr(expr)

        expr = "10.5."
        with self.assertRaises(validate.VdtTypeError):
            _parse_float_range_expr(expr)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_real_numpy_array_check(self):
        import validate

        array_string = "[0 5 10:15]"
        parsed_array = _real_numpy_array_check(array_string, min=0, max=30)
        expected_parsed_array = np.array([0., 5., 10., 11., 12., 13., 14.])
        self.assertTrue(parsed_array.dtype is np.dtype('float'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        array_string = "10:15"
        parsed_array = _real_numpy_array_check(array_string, min=0, max=30)
        expected_parsed_array = np.array([10., 11., 12., 13., 14.])
        self.assertTrue(parsed_array.dtype is np.dtype('float'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        array_string = "[10:15]"
        parsed_array = _real_numpy_array_check(array_string, min=0, max=30)
        expected_parsed_array = np.array([10., 11., 12., 13., 14.])
        self.assertTrue(parsed_array.dtype is np.dtype('float'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        array_string = "[0,5,10:15,20]"
        parsed_array = _real_numpy_array_check(array_string, min=0, max=30)
        expected_parsed_array = np.array([0., 5., 10., 11., 12., 13., 14., 20.])
        self.assertTrue(parsed_array.dtype is np.dtype('float'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        # xxxxx Test validation against the minimum allowed value xxxxxxxxx
        array_string = "[0,5,10:15,20]"
        with self.assertRaises(validate.VdtValueTooSmallError):
            parsed_array = _real_numpy_array_check(array_string,
                                                   min=4,
                                                   max=30)

        with self.assertRaises(validate.VdtValueTooBigError):
            parsed_array = _real_numpy_array_check(array_string,
                                                   min=0,
                                                   max=15)

    def test_integer_numpy_array_check(self):
        import validate

        array_string = "[0 5 10:15]"
        parsed_array = _integer_numpy_array_check(array_string, min=0, max=30)
        expected_parsed_array = np.array([0., 5., 10., 11., 12., 13., 14.])
        self.assertTrue(parsed_array.dtype is np.dtype('int'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        array_string = "10:15"
        parsed_array = _integer_numpy_array_check(array_string, min=0, max=30)
        expected_parsed_array = np.array([10., 11., 12., 13., 14.])
        self.assertTrue(parsed_array.dtype is np.dtype('int'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        array_string = "[10:15]"
        parsed_array = _integer_numpy_array_check(array_string, min=0, max=30)
        expected_parsed_array = np.array([10., 11., 12., 13., 14.])
        self.assertTrue(parsed_array.dtype is np.dtype('int'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        array_string = "[0,5,10:15,20]"
        parsed_array = _integer_numpy_array_check(array_string, min=0, max=30)
        expected_parsed_array = np.array([0., 5., 10., 11., 12., 13., 14., 20.])
        self.assertTrue(parsed_array.dtype is np.dtype('int'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        # xxxxx Test validation against the minimum allowed value xxxxxxxxx
        array_string = "[0,5,10:15,20]"
        with self.assertRaises(validate.VdtValueTooSmallError):
            parsed_array = _integer_numpy_array_check(array_string,
                                                      min=4,
                                                      max=30)

        with self.assertRaises(validate.VdtValueTooBigError):
            parsed_array = _integer_numpy_array_check(array_string,
                                                      min=0,
                                                      max=15)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class SimulationParametersTestCase(unittest.TestCase):
    """Unit-tests for the SimulationParameters class in the simulations
    module.
    """
    def setUp(self):
        params_dict = {'first': 10, 'second': 20}
        self.sim_params = SimulationParameters.create(params_dict)

    def test_create(self):
        # The create method was already called in the setUp.
        self.assertEqual(len(self.sim_params), 2)
        self.assertEqual(self.sim_params['first'], 10)
        self.assertEqual(self.sim_params['second'], 20)

    def test_add(self):
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.assertEqual(len(self.sim_params), 3)
        np.testing.assert_array_equal(
            self.sim_params['third'], np.array([1, 3, 2, 5]))

    def test_unpacking_parameters(self):
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.add('fourth', ['A', 'B'])
        self.assertEqual(self.sim_params.get_num_unpacked_variations(), 1)
        self.sim_params.set_unpack_parameter('third')
        self.sim_params.set_unpack_parameter('fourth')

        # One unpacked param with four values and other with two will give
        # us 4x2=8 unpacked variations.
        self.assertEqual(self.sim_params.get_num_unpacked_variations(), 8)
        # We make the unpacked_parameters and the expected value sets
        # because the order does not matter
        self.assertEqual(
            set(self.sim_params.unpacked_parameters),
            set(['third', 'fourth']))

        # We may have 8 variations, but there are still only 4 parameters
        self.assertEqual(len(self.sim_params), 4)

        # Test if an exception is raised if we try to set a non iterable
        # parameter to be unpacked.
        self.sim_params.add('fifth', 10)
        with self.assertRaises(ValueError):
            self.sim_params.set_unpack_parameter('fifth')

        # Test if an exception is thrown if we try to set a non existing
        # parameter to be unset.
        with self.assertRaises(ValueError):
            self.sim_params.set_unpack_parameter('sixth')

        if sys.version_info[0] < 3:
            # Now that a few parameters were added and set to be unpacked,
            # lets test the representation of the SimulationParameters
            # object. Note that the parameters that are marked for
            # unpacking have '*' appended to their name.
            # THIS TEST WILL NOT BE PERFORMED IN PYTHON 3
            self.assertEqual(self.sim_params.__repr__(), """{'second': 20, 'fifth': 10, 'fourth*': ['A', 'B'], 'third*': [1 3 2 5], 'first': 10}""")

        # Test if we can unset a parameter that was previously set to be
        # unpacked.
        self.sim_params.set_unpack_parameter('fourth', False)
        self.assertEqual(
            set(self.sim_params.unpacked_parameters),
            set(['third']))

    def test_equality(self):
        other = SimulationParameters()
        self.assertFalse(self.sim_params == other)
        other.add('first', 10)
        other.add('second', 20)
        self.assertTrue(self.sim_params == other)

        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.assertFalse(self.sim_params == other)
        other.add('third', np.array([1, 3, 2, 5]))
        self.assertTrue(self.sim_params == other)

        self.sim_params.set_unpack_parameter('third')
        self.assertFalse(self.sim_params == other)
        other.set_unpack_parameter('third')
        self.assertTrue(self.sim_params == other)

        other.parameters['third'][2] = 10
        self.assertFalse(self.sim_params == other)
        self.sim_params.parameters['third'][2] = 10
        self.assertTrue(self.sim_params == other)

    def test_get_unpacked_params_list(self):
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.add('fourth', ['A', 'B'])
        self.sim_params.set_unpack_parameter('third')
        self.sim_params.set_unpack_parameter('fourth')

        params_dict = {'first': [], 'second': [], 'third': [], 'fourth': []}
        unpacked_param_list = self.sim_params.get_unpacked_params_list()
        for i in unpacked_param_list:
            # This will add value multiple times when it shouldn't
            params_dict['first'].append(i['first'])
            params_dict['second'].append(i['second'])
            params_dict['third'].append(i['third'])
            params_dict['fourth'].append(i['fourth'])

        # We change all values to sets to remove repeated values for
        # testing purposes.
        self.assertEqual(set(params_dict['first']),
                         set([self.sim_params['first']]))
        self.assertEqual(set(params_dict['second']),
                         set([self.sim_params['second']]))
        self.assertEqual(set(params_dict['third']),
                         set(self.sim_params['third']))
        self.assertEqual(set(params_dict['fourth']),
                         set(self.sim_params['fourth']))

        # Test if the _unpack_index and the _original_sim_params member
        # variables are correct for each unpacked variation
        for i in range(self.sim_params.get_num_unpacked_variations()):
            self.assertEqual(unpacked_param_list[i]._unpack_index, i)
            self.assertTrue(unpacked_param_list[i]._original_sim_params is self.sim_params)

    def test_get_pack_indexes(self):
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.add('fourth', ['A', 'B'])
        self.sim_params.set_unpack_parameter('third')
        self.sim_params.set_unpack_parameter('fourth')

        # The parameters 'third' and 'fourth' are marked to be unpacked,
        # while the parameters 'first' and 'second' will always be the
        # same. The combinations after unpacking are shown below
        #   {'second': 20, 'fourth': A, 'third': 1, 'first': 10}
        #   {'second': 20, 'fourth': A, 'third': 3, 'first': 10}
        #   {'second': 20, 'fourth': A, 'third': 2, 'first': 10}
        #   {'second': 20, 'fourth': A, 'third': 5, 'first': 10}
        #   {'second': 20, 'fourth': B, 'third': 1, 'first': 10}
        #   {'second': 20, 'fourth': B, 'third': 3, 'first': 10}
        #   {'second': 20, 'fourth': B, 'third': 2, 'first': 10}
        #   {'second': 20, 'fourth': B, 'third': 5, 'first': 10}
        #
        # Lets focus on the 'third' and 'fourth' parameters, since they are
        # the only ones changing. Suppose we want to get the indexes
        # corresponding to varying the 'fourth' parameters with the 'third'
        # parameter equal to 2. We create a dictionary
        fixed_third_2 = {'third': 2}

        # The desired indexes are [2, 6]
        np.testing.assert_array_equal(
            self.sim_params.get_pack_indexes(fixed_third_2),
            [2, 6])

        fixed_third_5 = {'third': 5}
        # The desired indexes are [3, 7]
        np.testing.assert_array_equal(
            self.sim_params.get_pack_indexes(fixed_third_5),
            [3, 7])

        # Now lets fix the 'fourth' parameter and let the 'third' vary.
        fixed_fourth_A = {'fourth': 'A'}
        # The desired indexes are [0, 1, 2, 3]
        np.testing.assert_array_equal(
            self.sim_params.get_pack_indexes(fixed_fourth_A),
            [0, 1, 2, 3])

        fixed_fourth_B = {'fourth': 'B'}
        # The desired indexes are [4, 5, 6, 7]
        np.testing.assert_array_equal(
            self.sim_params.get_pack_indexes(fixed_fourth_B),
            [4, 5, 6, 7])

        # Lets try to fix some invalid value to see if an exception is
        # raised
        fixed_fourth_invalid = {'fourth': 'C'}
        with self.assertRaises(ValueError):
            # This should raise a ValueError, since the parameter 'fourth'
            # has no value 'C'
            self.sim_params.get_pack_indexes(fixed_fourth_invalid)

        # Now lets fix the third and the fourth parameter. This should get
        # me a single index.
        self.assertEqual(
            self.sim_params.get_pack_indexes({'third': 5, 'fourth': 'B'}),
            7)
        self.assertEqual(
            self.sim_params.get_pack_indexes({'third': 5, 'fourth': 'B'}),
            7)

    def test_save_to_and_load_from_file(self):
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.add('fourth', ['A', 'B'])
        self.sim_params.set_unpack_parameter('third')
        self.sim_params.set_unpack_parameter('fourth')

        filename = 'params.pickle'
        # Let's make sure the file does not exist
        try:
            os.remove(filename)
        except OSError:  # pragma: no cover
            pass

        # Save to the file
        self.sim_params.save_to_pickled_file(filename)

        # Load from the file
        sim_params2 = simulations.SimulationParameters.load_from_pickled_file(filename)

        self.assertEqual(self.sim_params['first'], sim_params2['first'])
        self.assertEqual(self.sim_params['second'], sim_params2['second'])
        self.assertEqual(len(self.sim_params), len(sim_params2))
        self.assertEqual(self.sim_params.get_num_unpacked_variations(),
                         sim_params2.get_num_unpacked_variations())

    def test_load_from_config_file(self):
        filename = 'test_config_file.txt'

        # xxxxxxxxxx Write the config file xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        try:
            os.remove(filename)
        except OSError:  # pragma: no cover
            pass

        fid = open(filename, 'w')
        fid.write("modo=test\n[Scenario]\nSNR=0,5,10\nM=4\nmodulator=PSK\n[IA Algorithm]\nmax_iterations=60")
        fid.close()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Read the parameters from the file xxxxxxxxxxxxxxxxxxxx
        # Since we are not specifying a "validation spec" all parameters
        # will be read as strings or list of strings.
        params = SimulationParameters.load_from_config_file(filename)
        self.assertEqual(len(params), 5)
        self.assertEqual(params['modo'], 'test')
        self.assertEqual(params['SNR'], ['0', '5', '10'])
        self.assertEqual(params['M'], '4')
        self.assertEqual(params['modulator'], 'PSK')
        self.assertEqual(params['max_iterations'], '60')
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Read the parameters from file with a validation spec xxxxxx
        spec = """modo=string
        [Scenario]
        SNR=real_numpy_array(default=15)
        M=integer(min=4, max=512, default=4)
        modulator=option('PSK', 'QAM', 'BPSK', default="PSK")
        [IA Algorithm]
        max_iterations=integer(min=1)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")
        params2 = SimulationParameters.load_from_config_file(
            filename, spec)
        self.assertEqual(len(params2), 6)
        self.assertEqual(params2['modo'], 'test')
        np.testing.assert_array_almost_equal(params2['SNR'],
                                             np.array([0., 5., 10.]))
        self.assertEqual(params2['M'], 4)
        self.assertEqual(params2['modulator'], 'PSK')
        self.assertEqual(params2['max_iterations'], 60)
        self.assertEqual(params2['unpacked_parameters'], ['SNR'])
        self.assertEqual(params2.unpacked_parameters, ['SNR'])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Lets create an invalid config file and try to load the parameters
        # First we provide an invalid value for M
        fid = open(filename, 'w')
        fid.write("modo=test\n[Scenario]\nSNR=0,5,10\nM=-4\nmodulator=PSK\n[IA Algorithm]\nmax_iterations=60")
        fid.close()

        with self.assertRaises(Exception):
            params2 = SimulationParameters.load_from_config_file(
                filename, spec)

        # Now we do not provide the required parameter max_iterations
        fid = open(filename, 'w')
        fid.write("modo=test\n[Scenario]\nSNR=0,5,10\nM=4\nmodulator=PSK\n[IA Algorithm]")
        fid.close()

        with self.assertRaises(Exception):
            params2 = SimulationParameters.load_from_config_file(
                filename, spec)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Remove the config file used in this test xxxxxxxxxxxxx
        try:
            os.remove(filename)
        except OSError:  # pragma: no cover
            pass
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class ResultTestCase(unittest.TestCase):
    """Unit-tests for the Result class in the simulations module."""

    def setUp(self):
        """Called before each test."""
        self.result1 = Result("name", Result.SUMTYPE)
        self.result2 = Result("name2", Result.RATIOTYPE)
        self.result3 = Result("name3", Result.MISCTYPE)

    def test_get_update_type(self):
        """Test the two properties, one to get the update type code and
        other to get the update type name. Note that both properties
        reflect the value of the same variable, the self._update_type
        variable.
        """
        self.assertEqual(self.result1.type_code, Result.SUMTYPE)
        self.assertEqual(self.result1.type_name, "SUMTYPE")

        self.assertEqual(self.result2.type_code, Result.RATIOTYPE)
        self.assertEqual(self.result2.type_name, "RATIOTYPE")

        self.assertEqual(self.result3.type_code, Result.MISCTYPE)
        self.assertEqual(self.result3.type_name, "MISCTYPE")

    def test_update(self):
        # Test the update function of the SUMTYPE
        self.result1.update(13)
        self.result1.update(4)
        self.assertEqual(self.result1._value, 17)
        self.assertEqual(self.result1.get_result(), 17)

        # Test the update function of the RATIOTYPE
        self.result2.update(3, 4)
        self.result2.update(9, 36)
        self.assertEqual(self.result2._value, 12)
        self.assertEqual(self.result2._total, 40)
        self.assertEqual(self.result2.get_result(), 0.3)
        self.result2.update(12, 8)
        self.assertEqual(self.result2.get_result(), 0.5)

        # Test the update function of the MISCTYPE. Note how we can store
        # anything.
        self.result3.update("First")
        self.assertEqual(self.result3.get_result(), "First")
        self.result3.update("Second")
        self.assertEqual(self.result3.get_result(), "Second")
        self.result3.update(0.4)
        self.assertEqual(self.result3.get_result(), 0.4)
        self.result3.update(0.4)
        self.assertEqual(self.result3.get_result(), 0.4)

        # Test if an exception is raised when updating a Result of the
        # RATIOTYPE without specifying both the value and the total.
        with self.assertRaises(ValueError):
            self.result2.update(3)

        # Test if an exception is thrown when updating a result of some
        # unknown type
        result_invalid = Result('invalid', 'invalid_type')
        with self.assertRaises(ValueError):
            result_invalid.update(10)

    def test_update_with_accumulate(self):
        result1 = Result('name', Result.SUMTYPE, accumulate_values=True)
        self.assertEqual(result1.accumulate_values_bool, result1._accumulate_values_bool)
        result1.update(13)
        result1.update(30)
        self.assertEqual(result1._value, 43)
        self.assertEqual(result1.get_result(), 43)
        self.assertEqual(result1._total, 0)
        self.assertEqual(result1._value_list, [13, 30])
        self.assertEqual(result1._total_list, [])

        result2 = Result('name', Result.RATIOTYPE, accumulate_values=True)
        result2.update(3, 10)
        result2.update(6, 7)
        result2.update(1, 15)
        self.assertEqual(result2._value, 10)
        self.assertEqual(result2._total, 32)
        self.assertEqual(result2.get_result(), 0.3125)
        self.assertEqual(result2._value_list, [3, 6, 1])
        self.assertEqual(result2._total_list, [10, 7, 15])

        result3 = Result('name', Result.MISCTYPE, accumulate_values=True)
        result3.update(3)
        result3.update("some string")
        result3.update(2)
        self.assertEqual(result3._value, 2)
        self.assertEqual(result3._total, 0)
        self.assertEqual(result3.get_result(), 2)
        self.assertEqual(result3._value_list, [3, "some string", 2])
        self.assertEqual(result3._total_list, [])

    def test_merge(self):
        # Test merge of Results of SUMTYPE
        self.result1.update(13)
        self.result1.update(30)
        result1_other = Result.create("name", Result.SUMTYPE, 11)
        self.result1.merge(result1_other)
        self.assertEqual(self.result1.name, "name")
        self.assertEqual(self.result1.get_result(), 54)
        self.assertEqual(self.result1.num_updates, 3)

        # Test merge of Results of RATIOTYPE
        self.result2.update(3, 10)
        self.result2.update(6, 7)
        self.result2.update(1, 15)
        result2_other = Result.create("name2", Result.RATIOTYPE, 34, 50)
        result2_other.update(12, 18)
        self.result2.merge(result2_other)
        self.assertEqual(self.result2.name, "name2")
        self.assertEqual(self.result2._value, 56)
        self.assertEqual(self.result2._total, 100)
        self.assertEqual(self.result2.get_result(), 0.56)
        self.assertEqual(self.result2.num_updates, 5)

        # Test merge of Results of MISCTYPE
        # There is no merge for misc type and an exception should be raised
        self.result3.update(0.4)
        result3_other = Result.create("name3", Result.MISCTYPE, 0.3)
        with self.assertRaises(AssertionError):
            self.result3.merge(result3_other)

        # Test merging results with different name or type
        result4 = Result.create("name4", Result.SUMTYPE, 3)
        with self.assertRaises(AssertionError):
            self.result1.merge(result4)

        result5 = Result.create("name", Result.RATIOTYPE, 3, 4)
        with self.assertRaises(AssertionError):
            self.result1.merge(result5)

    def test_merge_with_accumulate(self):
        result1 = Result('name', Result.SUMTYPE, accumulate_values=True)
        result1.update(13)
        result1.update(30)
        result1_other = Result.create("name", Result.SUMTYPE, 11, accumulate_values=True)
        result1_other.update(22)
        result1_other.update(4)
        result1.merge(result1_other)
        self.assertEqual(result1.get_result(), 80)
        self.assertEqual(result1._value_list, [13, 30, 11, 22, 4])
        self.assertEqual(result1._total_list, [])

        result2 = Result('name2', Result.RATIOTYPE, accumulate_values=True)
        result2.update(3, 10)
        result2.update(6, 7)
        result2.update(1, 15)

        result2_other = Result.create("name2", Result.RATIOTYPE, 34, 50, accumulate_values=True)
        result2_other.update(12, 18)
        result2.merge(result2_other)
        self.assertEqual(result2._value, 56)
        self.assertEqual(result2._value_list, [3, 6, 1, 34, 12])
        self.assertEqual(result2._total, 100)
        self.assertEqual(result2._total_list, [10, 7, 15, 50, 18])
        self.assertEqual(result2.get_result(), 0.56)
        self.assertEqual(result2.num_updates, 5)

    def test_representation(self):
        self.assertEqual(self.result1.__repr__(), "Result -> name: Nothing yet")
        self.assertEqual(self.result2.__repr__(), "Result -> name2: 0/0 -> NaN")
        self.assertEqual(self.result3.__repr__(),
                         "Result -> name3: Nothing yet")

        self.result1.update(10)
        self.result2.update(2, 4)
        self.result3.update(0.4)

        self.assertEqual(self.result1.__repr__(), "Result -> name: 10")
        self.assertEqual(self.result2.__repr__(), "Result -> name2: 2/4 -> 0.5")
        self.assertEqual(self.result3.__repr__(), "Result -> name3: 0.4")

    def test_calc_confidence_interval(self):
        # Test if an exceptions is raised for a Result object of the
        # MISCTYPE update type.
        with self.assertRaises(RuntimeError):
            # A result object of the MISCTYPE type cannot use the
            # calc_confidence_interval method, since this update ignores
            # the accumulate_values option and never accumulates any value.
            self.result3.get_confidence_interval()

        # Test if an exception is raised if the accumulate_values option
        # was not set to True
        with self.assertRaises(RuntimeError):
            self.result1.get_confidence_interval()

        result = Result('name', Result.RATIOTYPE, accumulate_values=True)
        # Test if an exception is raised if there are not stored values yet
        with self.assertRaises(RuntimeError):
            result.get_confidence_interval()

        # Now lets finally store some values.
        result.update(10, 30)
        result.update(3, 24)
        result.update(15, 42)
        result.update(5, 7)

        # Calculate the expected confidence interval
        A = np.array(result._value_list, dtype=float) / np.array(result._total_list, dtype=float)
        expected_confidence_interval = misc.calc_confidence_interval(A.mean(), A.std(), A.size, P=95)
        confidence_interval = result.get_confidence_interval(P=95)
        np.testing.assert_array_almost_equal(expected_confidence_interval,
                                             confidence_interval)


class SimulationResultsTestCase(unittest.TestCase):
    """Unit-tests for the SimulationResults class in the simulations
    module.

    """
    def setUp(self):
        # First SimulationResults object
        self.simresults = SimulationResults()
        self.simresults.add_new_result("lala", Result.SUMTYPE, 13)
        result2 = Result("lele", Result.RATIOTYPE)
        result2.update(3, 10)
        result2.update(8, 10)
        self.simresults.add_result(result2)

        # Second SimulationResults object
        self.other_simresults = SimulationResults()
        result1_other = Result.create('lala', Result.SUMTYPE, 30)
        result2_other = Result.create('lele', Result.RATIOTYPE, 4, 10)
        result3 = Result.create('lili', Result.MISCTYPE, "a string")
        self.other_simresults.add_result(result1_other)
        self.other_simresults.add_result(result2_other)
        self.other_simresults.add_result(result3)

    def test_params_property(self):
        params = SimulationParameters()
        params.add('number', 10)
        params.add('name', 'lala')

        # Try to set the parameters to an invalid object
        with self.assertRaises(ValueError):
            self.simresults.set_parameters(10)

        # Set the simulation parameters
        self.simresults.set_parameters(params)

        # test the get property
        params2 = self.simresults.params
        self.assertEqual(len(params), len(params2))
        self.assertEqual(params['number'], params2['number'])
        self.assertEqual(params['name'], params2['name'])

    def test_get_result_names(self):
        # The output of the get_result_names is a list of names. We
        # transform it into a set in this test only to make the order of
        # the names uninportant.
        expected_output = set(['lala', 'lele'])
        self.assertEqual(set(self.simresults.get_result_names()), expected_output)
        # Test also the representation of the SimulationResults object
        self.assertEqual(self.simresults.__repr__(), """SimulationResults: ['lala', 'lele']""")

    def test_add_result(self):
        # Add a result with the same name of an existing result -> Should
        # replace it
        result1_other = Result.create("lala", Result.SUMTYPE, 25)
        self.simresults.add_result(result1_other)
        self.assertEqual(len(self.simresults['lala']), 1)
        self.assertEqual(self.simresults['lala'][0].get_result(), 25)

        # Add a new result
        result3 = Result.create('lili', Result.MISCTYPE, "a string")
        self.simresults.add_result(result3)
        self.assertEqual(set(self.simresults.get_result_names()),
                         set(["lala", "lele", "lili"]))
        self.assertEqual(self.simresults['lili'][0].get_result(), "a string")

    def test_append_result(self):
        result1_other = Result.create("lala", Result.SUMTYPE, 25)
        self.simresults.append_result(result1_other)
        # Since we append a new Result with the name 'lala', then now we
        # should have two Results for 'lala' (in a simulation these two
        # results would probably corresponds to 'lala' results with
        # different simulation parameters)
        self.assertEqual(len(self.simresults['lala']), 2)
        self.assertEqual(self.simresults['lala'][0].get_result(), 13)
        self.assertEqual(self.simresults['lala'][1].get_result(), 25)

        # Test if an exception is thrown if we try to append result with a
        # different type
        result1_wrong = Result.create("lala", Result.RATIOTYPE, 25, 30)
        with self.assertRaises(ValueError):
            self.simresults.append_result(result1_wrong)

    def test_append_all_results(self):
        self.simresults.append_all_results(self.other_simresults)
        # Note that self.simresults only has the 'lala' and 'lele' results.
        # After we append the results in self.other_simresults
        # self.simresults should have also the 'lili' result, but with only
        # a single result for 'lili' and two results for both 'lala' and
        # 'lele'..
        self.assertEqual(set(self.simresults.get_result_names()),
                         set(["lala", "lele", "lili"]))
        self.assertEqual(len(self.simresults['lala']), 2)
        self.assertEqual(len(self.simresults['lele']), 2)
        self.assertEqual(len(self.simresults['lili']), 1)

    def test_merge_all_results(self):
        # Note that even though there is a 'lili' result in
        # self.other_simresults, only 'lala' and 'lele' will be
        # merged. Also, self.other_simresults must have all the results in
        # self.simresults otherwise there will be a KeyError.
        self.simresults.merge_all_results(self.other_simresults)
        self.assertEqual(self.simresults['lala'][-1].get_result(), 43)
        self.assertEqual(
            self.simresults['lele'][-1].get_result(),
            (11. + 4.) / (20. + 10.))

        # One update from the 'lala' result in self.simresults and other
        # from the 'lala' result in self.other_simresults
        self.assertEqual(self.simresults['lala'][0].num_updates, 2)

        # Two updates from the 'lele' result in self.simresults and other
        # from the 'lele' result in self.other_simresults
        self.assertEqual(self.simresults['lele'][0].num_updates, 3)

        # Test if an empty SimulationResults object can merge with another
        # SimulationResults objec.
        emptyresults = SimulationResults()
        emptyresults.merge_all_results(self.simresults)
        self.assertEqual(
            set(emptyresults.get_result_names()),
            set(['lala', 'lele']))

    def test_get_result_values_list(self):
        self.simresults.append_all_results(self.other_simresults)

        self.assertEqual(
            self.simresults.get_result_values_list('lala'),
            [13, 30])
        self.assertEqual(
            self.simresults.get_result_values_list('lele'),
            [0.55, 0.4])

        # There is only one result for 'lili', which comes from
        # self.other_simresults.
        self.assertEqual(
            self.simresults.get_result_values_list('lili'),
            ['a string'])

    def test_get_result_values_confidence_intervals(self):
        simresults = SimulationResults()
        result = Result('name', Result.RATIOTYPE, accumulate_values=True)
        result_other = Result('name', Result.RATIOTYPE, accumulate_values=True)
        result.update(3, 10)
        result.update(7, 9)
        result.update(2, 5)
        result.update(3, 3)
        result.update(7, 15)

        result_other.update(13, 15)
        result_other.update(15, 20)
        result_other.update(4, 9)

        simresults.add_result(result)
        simresults.append_result(result_other)

        P = 95  # Confidence level of 95%
        list_of_confidence_intervals = simresults.get_result_values_confidence_intervals(
            'name', P)

        # Calculates the expected list of confidence intervals
        expected_list_of_confidence_intervals = [i.get_confidence_interval(P) for i in simresults['name']]

        # Test of they are equal
        for a, b in zip(list_of_confidence_intervals,
                        expected_list_of_confidence_intervals):
            np.testing.assert_array_almost_equal(a, b)

    def test_save_to_and_load_from_file(self):
        filename = 'results.pickle'
        # Let's make sure the file does not exist
        try:
            os.remove(filename)
        except OSError:  # pragma: no cover
            pass

        # Set sime simulation parameters
        self.simresults.params.add('factor', 0.5)
        self.simresults.params.add('temperature', 50.5)
        self.simresults.params.add('age', 3)

        # Save to the file
        self.simresults.save_to_file(filename)

        # Load from the file
        simresults2 = simulations.SimulationResults.load_from_file(filename)

        self.assertEqual(len(self.simresults), len(simresults2))
        self.assertEqual(set(self.simresults.get_result_names()),
                         set(simresults2.get_result_names()))

        self.assertEqual(self.simresults['lala'][0].type_code,
                         simresults2['lala'][0].type_code)
        self.assertEqual(self.simresults['lele'][0].type_code,
                         simresults2['lele'][0].type_code)

        self.assertAlmostEqual(self.simresults['lala'][0].get_result(),
                               simresults2['lala'][0].get_result(),)
        self.assertAlmostEqual(self.simresults['lele'][0].get_result(),
                               simresults2['lele'][0].get_result(),)

        # test if the parameters were also saved
        self.assertEqual(self.simresults.params['age'],
                         simresults2.params['age'])
        self.assertAlmostEqual(self.simresults.params['temperature'],
                               simresults2.params['temperature'])
        self.assertAlmostEqual(self.simresults.params['factor'],
                               simresults2.params['factor'])

    def test_save_to_and_load_from_hdf5_file(self):
        filename = 'test_results_hdf5.h5'
        # Let's make sure the file does not exist
        try:
            os.remove(filename)
        except OSError:  # pragma: no cover
            pass

        # Set sime simulation parameters
        self.simresults.params.add('factor', [0.5, 0.6])
        self.simresults.params.add('temperature', [50.5, 60.0, 70.8])
        self.simresults.params.add('age', 3)
        self.simresults.params.set_unpack_parameter('temperature')
        self.simresults.params.set_unpack_parameter('factor')

        # Save to the file
        self.simresults.save_to_hdf5_file(filename)

        # Load from the file
        simresults2 = simulations.SimulationResults.load_from_hdf5_file(filename)
        self.assertEqual(len(self.simresults), len(simresults2))
        self.assertEqual(set(self.simresults.get_result_names()),
                         set(simresults2.get_result_names()))

        self.assertEqual(self.simresults['lala'][0].type_code,
                         simresults2['lala'][0].type_code)
        self.assertEqual(self.simresults['lele'][0].type_code,
                         simresults2['lele'][0].type_code)

        self.assertAlmostEqual(self.simresults['lala'][0].get_result(),
                               simresults2['lala'][0].get_result(),)
        self.assertAlmostEqual(self.simresults['lele'][0].get_result(),
                               simresults2['lele'][0].get_result(),)

        # test if the parameters were also saved
        self.assertEqual(self.simresults.params['age'],
                         simresults2.params['age'])
        np.testing.assert_almost_equal(self.simresults.params['factor'],
                                       simresults2.params['factor'])
        np.testing.assert_almost_equal(self.simresults.params['temperature'],
                                       simresults2.params['temperature'])

        # Test if the unpacked parameters where also saved
        self.assertEqual(self.simresults.params.unpacked_parameters[0],
                         simresults2.params.unpacked_parameters[0])

        # Remove the file created during the test
        try:
            os.remove(filename)
        except OSError:  # pragma: no cover
            pass


    # def test_save_to_and_load_from_pytables_file(self):
    #     filename = 'results_pytables.h5'
    #     # Let's make sure the file does not exist
    #     try:
    #         os.remove(filename)
    #     except OSError:  # pragma: no cover
    #         pass

    #     # Set sime simulation parameters
    #     self.simresults.params.add('factor', [0.5, 0.6])
    #     self.simresults.params.add('temperature', [50.5, 60.0, 70.8])
    #     self.simresults.params.add('age', 3)
    #     self.simresults.params.set_unpack_parameter('temperature')
    #     self.simresults.params.set_unpack_parameter('factor')

    #     # Save to the file
    #     self.simresults.save_to_pytables_file(filename)

    #     # Load from the file
    #     simresults2 = simulations.SimulationResults.load_from_pytables_file(filename)
    #     self.assertEqual(len(self.simresults), len(simresults2))
    #     self.assertEqual(set(self.simresults.get_result_names()),
    #                      set(simresults2.get_result_names()))

    #     self.assertEqual(self.simresults['lala'][0].type_code,
    #                      simresults2['lala'][0].type_code)
    #     self.assertEqual(self.simresults['lele'][0].type_code,
    #                      simresults2['lele'][0].type_code)

    #     self.assertAlmostEqual(self.simresults['lala'][0].get_result(),
    #                            simresults2['lala'][0].get_result(),)
    #     self.assertAlmostEqual(self.simresults['lele'][0].get_result(),
    #                            simresults2['lele'][0].get_result(),)

    #     # test if the parameters were also saved
    #     self.assertEqual(self.simresults.params['age'],
    #                      simresults2.params['age'])
    #     np.testing.assert_almost_equal(self.simresults.params['factor'],
    #                                    simresults2.params['factor'])
    #     np.testing.assert_almost_equal(self.simresults.params['temperature'],
    #                                    simresults2.params['temperature'])

    #     # Test if the unpacked parameters where also saved
    #     self.assertEqual(self.simresults.params.unpacked_parameters[0],
    #                      simresults2.params.unpacked_parameters[0])

class SimulationRunnerTestCase(unittest.TestCase):
    """Unit-tests for the SimulationRunner class in the simulations
    module.
    """

    def setUp(self):
        self.runner = SimulationRunner()

    # Test if the SimulationRunner sets a few default attributs in its init
    # method.
    def test_default_values(self):
        # Note that we are also testing the elapsed_time and runned_reps
        # properties, which should just return these attributes.
        self.assertEqual(self.runner.rep_max, 1)
        self.assertEqual(self.runner._elapsed_time, 0.0)
        self.assertEqual(self.runner.elapsed_time, "0.00s")
        self.assertEqual(self.runner.runned_reps, [])
        self.assertTrue(isinstance(self.runner.params, SimulationParameters))
        self.assertTrue(isinstance(self.runner.results, SimulationResults))
        self.assertEqual(self.runner.progressbar_message, "Progress")

    def test_not_implemented_methods(self):
        #self.assertRaises(NotImplementedError, self.S1._get_vertex_positions)
        with self.assertRaises(NotImplementedError):
            self.runner._run_simulation(None)

    def test_keep_going(self):
        # the _keep_going method in the SimulationRunner class should
        # return True
        self.assertTrue(self.runner._keep_going(None, None, None))

    def test_simulate(self):
        from tests.util_package_test import _DummyRunner
        dummyrunner = _DummyRunner()
        # then we call its simulate method
        dummyrunner.simulate()  # The results will be the SNR values
                                # multiplied by 1.2. plus the bias
                                # parameter
        lala_results = [r.get_result() for r in dummyrunner.results['lala']]
        expected_lala_results = [1.3, 7.3, 13.3, 19.3, 25.3]

        self.assertAlmostEqual(lala_results, expected_lala_results)

    # This test method is normally skipped, unless you have started an
    # IPython cluster with a "tests" profile so that you have at least one
    # engine running.
    def test_simulate_in_parallel(self):
        try:
            from IPython.parallel import Client
            cl = Client(profile="tests")

            dview = cl.direct_view()
            dview.execute('%reset')  # Reset the engines so that we don't have
                                     # variables there from last computations
            dview.execute('import sys')
            # We use block=True to ensure that all engines have modified
            # their path to include the folder with the simulator before we
            # create the load lanced view in the following.
            dview.execute('sys.path.append("{0}")'.format(current_dir), block=True)

            lview = cl.load_balanced_view()
            if len(lview) == 0:
                self.skipTest("At least one IPython engine must be running.")
        except Exception:
            self.skipTest("The IPython engines were not found.")

        dview.execute('import util_package_test', block=True)

        from util_package_test import _DummyRunner
        runner = _DummyRunner()
        runner.progressbar_message = 'bla'
        runner.update_progress_function_style = 'text1'

        runner.simulate_in_parallel(lview)

        lala_results = [r.get_result() for r in runner.results['lala']]
        expected_lala_results = [1.3, 7.3, 13.3, 19.3, 25.3]

        self.assertAlmostEqual(lala_results, expected_lala_results)


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
        [V_n3, D_n3] = misc.peig(A, 3)

        expected_V_n3 = np.array(
            [[0.27354856 + 0.54286421j, 0.15266747 - 0.35048035j, 0.69593520],
             [0.68522942, -0.24255902 + 0.37567057j, -0.02693857 + 0.57425752j],
             [0.38918583 + 0.09728652j, 0.80863645, -0.40625488 - 0.14189355j]])
        np.testing.assert_array_almost_equal(V_n3, expected_V_n3)

        # xxxxx Test for n==2 (two columns) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n2, D_n2] = misc.peig(A, 2)

        expected_V_n2 = np.array(
            [[0.27354856 + 0.54286421j, 0.15266747 - 0.35048035j],
             [0.68522942, -0.24255902 + 0.37567057j],
             [0.38918583 + 0.09728652j, 0.80863645]])
        np.testing.assert_array_almost_equal(V_n2, expected_V_n2)

        # xxxxx Test for n==1 (one column) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n1, D_n1] = misc.peig(A, 1)

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
        [V_n3, D_n3] = misc.leig(A, 3)

        expected_V_n3 = np.array(
            [[0.69593520, 0.15266747 - 0.35048035j, 0.27354856 + 0.54286421j],
             [-0.02693857 + 0.57425752j, -0.24255902 + 0.37567057j, 0.68522942],
             [-0.40625488 - 0.14189355j, 0.80863645, 0.38918583 + 0.09728652j]])
        np.testing.assert_array_almost_equal(V_n3, expected_V_n3)

        # xxxxx Test for n==2 (two columns) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n2, D_n2] = misc.leig(A, 2)

        expected_V_n2 = np.array(
            [[0.69593520, 0.15266747 - 0.35048035j],
             [-0.02693857 + 0.57425752j, -0.24255902 + 0.37567057j],
             [-0.40625488 - 0.14189355j, 0.80863645]])
        np.testing.assert_array_almost_equal(V_n2, expected_V_n2)

        # xxxxx Test for n==1 (one column) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        [V_n1, D_n1] = misc.leig(A, 1)

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
        x = np.array([4, 2, 1, 3, 7, 3, 8])
        unorm_autocor = misc.calc_unorm_autocorr(x)
        expected_unorm_autocor = np.array([152, 79, 82, 53, 42, 28, 32])
        np.testing.assert_array_equal(unorm_autocor, expected_unorm_autocor)

    def test_calc_autocorr(self):
        x = np.array([4, 2, 1, 3, 7, 3, 8])
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
        a = np.array([5, 10, 15, 20, 25, 30, 35, 40])
        expr_a = misc.get_range_representation(a)
        expected_expr_a = "5:5:40"
        self.assertEqual(expr_a, expected_expr_a)

        b = np.array([2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7])
        expr_b = misc.get_range_representation(b)
        expected_expr_b = "2.3:0.3:4.7"
        self.assertEqual(expr_b, expected_expr_b)

        c = np.array([10.2, 9., 7.8, 6.6, 5.4, 4.2])
        expr_c = misc.get_range_representation(c)
        expected_expr_c = "10.2:-1.2:4.2"
        self.assertEqual(expr_c, expected_expr_c)

        # This array is not an arithmetic progression and
        # get_range_representation should return None
        d = np.array([1, 3, 9, 4])
        self.assertIsNone(misc.get_range_representation(d))

    def test_replace_dict_values(self):
        name = "something {value1} - {value2} something else {value3}"
        dictionary = {'value1': 'bla bla', 'value2': np.array([5, 10, 15, 20, 25, 30]), 'value3': 76}
        new_name = misc.replace_dict_values(name, dictionary)
        expected_new_name = 'something bla bla - [5_(5)_30] something else 76'
        self.assertEqual(new_name, expected_new_name)

        # Value2 is not an arithmetic progression
        dictionary2 = {'value1': 'bla bla', 'value2': np.array([5, 10, 18, 20, 25, 30]), 'value3': 76}
        new_name2 = misc.replace_dict_values(name, dictionary2)
        expected_new_name2 = 'something bla bla - [ 5,10,18,20,25,30] something else 76'
        self.assertEqual(new_name2, expected_new_name2)



# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# def producer(queue, sleep_time=0):
#     process_pid = multiprocessing.current_process().pid
#     for i in range(1, 11):
#         sleep(sleep_time)
#         # TODO: put a tuple in the queue instead of just the value. The
#         # tuple should have the process identifier and the value.
#         queue.put((process_pid, i))
#     queue.put((process_pid, -1))
#
#
# def consumer(queue):
#     print "consumer started"
#     while True:
#         sleep(0.8)
#         if queue.empty() == False:
#             value = queue.get()
#             if value < 0:
#                 # Negative value means stop
#                 print "received poison pill"
#                 break
#             print "Consumer read the value: {0}".format(value)
#     print "consumer ended"
#
#
# if __name__ == '__main__1':
#     # Extract the code below somewhere later
#     import multiprocessing
#     from time import sleep
#
#     queue = multiprocessing.Queue()
#
#     p = multiprocessing.Process(target=producer, args=[queue])
#     c = multiprocessing.Process(target=consumer, args=[queue])
#
#     p.start()
#     c.start()
#     p.join()
#     #queue.put(-1)
#     c.join()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#
#
# from progressbar import ProgressbarText
#
# def progress_bar_background(queue):
#     bar = ProgressbarText(10, "o", "Teste")
#     simulating = True
#     # Processes that put something in the queue
#     #runnin_processes = set()
#     while simulating:
#         if queue.empty() == False:
#             (process_pid, value) = queue.get()
#             if value < 0:
#                 # Negative value means stop
#                 simulating = False
#             else:
#                 #sleep(1)
#                 bar.progress(value)
#
#
# if __name__ == '__main__2':
#     import multiprocessing
#     from time import sleep
#     from progressbar import ProgressbarMultiProcessText
#
#     # Runs in a different process and owns the queue
#     manager = multiprocessing.Manager()
#
#     queue = manager.Queue()
#     # args: queue, job_id, sleep_time
#     p2 = multiprocessing.Process(target=producer, args=[queue, 0, 0.8])
#     p1 = multiprocessing.Process(target=producer, args=[queue, 1, 0.5])
#     #p2 = multiprocessing.Process(target=producer, args=[queue])
#
#     # bar = ProgressbarMultiProcessText(queue, total_final_count=40, progresschar='o', message="Teste")
#     #c = progressbar.start_updater()
#     c = multiprocessing.Process(target=progress_bar_background, args=[queue])
#
#     p1.start()
#     p2.start()
#     c.start()
#     #c = progressbar.start_updater()
#
#     p1.join()
#     p2.join()
#     queue.put(-1)
#     c.join()
#
#     print "FIM"
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class ProgressbarTextTestCase(unittest.TestCase):
    def setUp(self):
        from StringIO import StringIO
        message = "ProgressbarText Unittest"
        # The progress will be printed to the StringIO object instead of
        # sys.stdout
        self.out = StringIO()
        self.pbar = progressbar.ProgressbarText(50, '*', message, output=self.out)

        self.out2 = StringIO()
        self.pbar2 = progressbar.ProgressbarText(25, 'x', output=self.out2)

    def test_progress(self):
        # Progress 20% (10 is equivalent to 20% of 50)
        self.pbar.progress(10)
        self.assertEqual(self.out.getvalue(), """------------ ProgressbarText Unittest -----------1
    1    2    3    4    5    6    7    8    9    0
----0----0----0----0----0----0----0----0----0----0
**********""")

        # Progress to 70%
        self.pbar.progress(35)
        self.assertEqual(self.out.getvalue(), """------------ ProgressbarText Unittest -----------1
    1    2    3    4    5    6    7    8    9    0
----0----0----0----0----0----0----0----0----0----0
***********************************""")

        # Progress to 100% -> Note that in the case of 100% a new line is
        # added at the end.
        self.pbar.progress(50)
        self.assertEqual(self.out.getvalue(), """------------ ProgressbarText Unittest -----------1
    1    2    3    4    5    6    7    8    9    0
----0----0----0----0----0----0----0----0----0----0
**************************************************\n""")

        # Test with pbar2, which uses the default progress message and the
        # character 'x' to indicate progress.
        self.pbar2.progress(20)
        self.assertEqual(self.out2.getvalue(), """------------------ % Progress -------------------1
    1    2    3    4    5    6    7    8    9    0
----0----0----0----0----0----0----0----0----0----0
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx""")

    def test_small_progress_and_zero_finalcount(self):
        # Test the case when the progress is lower then 1%.
        pbar3 = progressbar.ProgressbarText(finalcount=200, output=self.out)
        pbar3.progress(1)
        self.assertEqual(self.out.getvalue(), """------------------ % Progress -------------------1
    1    2    3    4    5    6    7    8    9    0
----0----0----0----0----0----0----0----0----0----0
""")

        # Test the case when finalcount is zero.
        pbar4 = progressbar.ProgressbarText(0, output=self.out2)
        # Any progress will get the bar to 100%
        pbar4.progress(1)
        self.assertEqual(self.out2.getvalue(), """------------------ % Progress -------------------1\n    1    2    3    4    5    6    7    8    9    0\n----0----0----0----0----0----0----0----0----0----0\n**************************************************\n""")


class ProgressbarText2TestCase(unittest.TestCase):
    def setUp(self):
        from StringIO import StringIO
        message = "ProgressbarText Unittest"
        # The progress will be printed to the StringIO object instead of
        # sys.stdout
        self.out = StringIO()
        self.pbar = progressbar.ProgressbarText2(50, '*', message, output=self.out)

        self.out2 = StringIO()
        self.pbar2 = progressbar.ProgressbarText2(50, '*', output=self.out2)

    def test_some_method(self):
        self.pbar.progress(15)
        self.assertEqual(self.out.getvalue(), "\r[**************        30%                       ]  ProgressbarText Unittest")

        self.pbar.progress(50)
        self.assertEqual(self.out.getvalue(), "\r[**************        30%                       ]  ProgressbarText Unittest\r[*********************100%***********************]  ProgressbarText Unittest\n")

        # Progressbar with no message -> Use a default message
        self.pbar2.progress(15)
        self.assertEqual(self.out2.getvalue(), "\r[**************        30%                       ]  15 of 50 complete")


class ProgressbarText3TestCase(unittest.TestCase):
    def setUp(self):
        from StringIO import StringIO
        message = "ProgressbarText Unittest"
        # The progress will be printed to the StringIO object instead of
        # sys.stdout
        self.out = StringIO()
        self.pbar = progressbar.ProgressbarText3(50, '*', message, output=self.out)

        self.out2 = StringIO()
        self.pbar2 = progressbar.ProgressbarText3(50, '*', output=self.out2)

    def test_some_method(self):
        self.pbar.progress(15)

        # print
        #print self.out.getvalue()

        self.assertEqual(self.out.getvalue(), "\r********* ProgressbarText Unittest 15/50 *********\n")

        self.pbar.progress(50)
        self.assertEqual(self.out.getvalue(), "\r********* ProgressbarText Unittest 15/50 *********\n\r********* ProgressbarText Unittest 50/50 *********\n")

        # Test with no message (use default message)
        self.pbar2.progress(40)
        self.assertEqual(self.out2.getvalue(), "\r********************** 40/50 *********************\n")


class ProgressbarMultiProcessTextTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.output_filename = "ProgressbarMultiProcessTextTestCase.out"

        self.mpbar = progressbar.ProgressbarMultiProcessText(message="Some message", sleep_time=0.1, filename=self.output_filename)
        self.proxybar1 = self.mpbar.register_client_and_get_proxy_progressbar(10)
        self.proxybar2 = self.mpbar.register_client_and_get_proxy_progressbar(15)

    def test_register(self):
        # Test last_id and total_final_count of the main progress bar
        self.assertEqual(self.mpbar._last_id, 1)
        self.assertEqual(self.mpbar._total_final_count, 25)

        # Register a new proxy progressbar and test the last_id and
        # total_final_count again.
        proxybar3 = self.mpbar.register_client_and_get_proxy_progressbar(13)
        self.assertEqual(self.mpbar._last_id, 2)
        self.assertEqual(self.mpbar._total_final_count, 38)

    def test_proxy_progressbars(self):
        # Test the information in the proxybar1
        self.assertEqual(self.proxybar1.client_id, 0)
        self.assertTrue(self.proxybar1._client_data_list is
                        self.mpbar._client_data_list)

        # Test the information in the proxybar2
        self.assertEqual(self.proxybar2.client_id, 1)
        self.assertTrue(self.proxybar2._client_data_list is
                        self.mpbar._client_data_list)

    # Note: This method will sleep for 0.3 seconds thus adding to the total
    # amount of time required to run all tests. Unfortunatelly, this is a
    # necessary cost.
    def test_updater(self):
        import os
        import time

        # Remove old file from previous test run
        try:
            os.remove(self.output_filename)
        except Exception:  # Pragma: no cover
            pass

        # Suppose that the first process already started and called the
        # proxybar1 to update its progress.
        self.proxybar1.progress(6)

        # Then we start the "updater" of the main progressbar.
        self.mpbar.start_updater()

        # Then the second process updates its progress
        self.proxybar2.progress(6)
        #self.mpbar.stop_updater()

        # Sleep for a very short time so that the
        # ProgressbarMultiProcessText object has time to create the file
        # with the current progress
        time.sleep(0.3)

        self.mpbar.stop_updater(0)

        # Open and read the progress from the file
        progress_output_file = open(self.output_filename)
        progress_string = progress_output_file.read()

        # Expected string with the progress output
        expected_progress_string = """------------------ Some message -----------------1
    1    2    3    4    5    6    7    8    9    0
----0----0----0----0----0----0----0----0----0----0
************************"""

        self.assertEqual(progress_string, expected_progress_string)


# TODO: finish implementation
class ProgressbarZMQTextTestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.output_filename = "ProgressbarZMQTextTestCase.out"

        self.zmqbar = progressbar.ProgressbarZMQText(message="Some message", sleep_time=0.1, filename=self.output_filename)
        self.proxybar1 = self.zmqbar.register_client_and_get_proxy_progressbar(10)
        self.proxybar2 = self.zmqbar.register_client_and_get_proxy_progressbar(15)

    def tearDown(self):
        self.zmqbar._zmq_pull_socket.close()

    def test_register(self):
        # Test last_id and total_final_count of the main progress bar
        self.assertEqual(self.zmqbar._last_id, 1)
        self.assertEqual(self.zmqbar._total_final_count, 25)

        # Register a new proxy progressbar and test the last_id and
        # total_final_count again.
        proxybar3 = self.zmqbar.register_client_and_get_proxy_progressbar(13)
        self.assertEqual(self.zmqbar._last_id, 2)
        self.assertEqual(self.zmqbar._total_final_count, 38)

        # Test IP and port of the proxy progress bars
        self.assertEqual(self.proxybar1.ip, self.zmqbar._ip)
        self.assertEqual(self.proxybar1.port, self.zmqbar._port)
        self.assertEqual(self.proxybar2.ip, self.zmqbar._ip)
        self.assertEqual(self.proxybar2.port, self.zmqbar._port)
        self.assertEqual(proxybar3.ip, self.zmqbar._ip)
        self.assertEqual(proxybar3.port, self.zmqbar._port)

    def test_proxy_progressbars(self):
        # Test the information in the proxybar1
        self.assertEqual(self.proxybar1.client_id, 0)
        self.assertEqual(self.proxybar1.ip, self.zmqbar._ip)
        self.assertEqual(self.proxybar1.port, self.zmqbar._port)

        # Test the information in the proxybar2
        self.assertEqual(self.proxybar2.client_id, 1)
        self.assertEqual(self.proxybar2.ip, self.zmqbar._ip)
        self.assertEqual(self.proxybar2.port, self.zmqbar._port)

        # Since we did not call the progress method of the proxy
        # progressbars not even once yet, they have not created their
        # sockets yet.
        self.assertIsNone(self.proxybar1._zmq_push_socket)
        self.assertIsNone(self.proxybar2._zmq_push_socket)
        self.assertIsNone(self.proxybar1._zmq_context)
        self.assertIsNone(self.proxybar2._zmq_context)

        # Before the first time the progress method in self.proxybar1 and
        # self.proxybar2 is called their "_progress_func" variable points
        # to the "_connect_and_update_progress" method
        self.assertTrue(self.proxybar1._progress_func == progressbar.ProgressbarZMQProxy._connect_and_update_progress)
        self.assertTrue(self.proxybar2._progress_func == progressbar.ProgressbarZMQProxy._connect_and_update_progress)


# TODO: finish implementation
class ProgressbarZMQText2TestCase(unittest.TestCase):
    def setUp(self):
        """Called before each test."""
        self.output_filename = "ProgressbarZMQText2TestCase.out"

        self.zmqbar = progressbar.ProgressbarZMQText2(message="Some message", sleep_time=0.1, filename=self.output_filename)
        self.proxybar1 = self.zmqbar.register_client_and_get_proxy_progressbar(10)
        self.proxybar2 = self.zmqbar.register_client_and_get_proxy_progressbar(15)

    def tearDown(self):
        #self.zmqbar._zmq_pull_socket.close()
        pass

    def test_register(self):
        # Test last_id and total_final_count of the main progress bar
        self.assertEqual(self.zmqbar._last_id, 1)
        self.assertEqual(self.zmqbar._total_final_count, 25)

        # Register a new proxy progressbar and test the last_id and
        # total_final_count again.
        proxybar3 = self.zmqbar.register_client_and_get_proxy_progressbar(13)
        self.assertEqual(self.zmqbar._last_id, 2)
        self.assertEqual(self.zmqbar._total_final_count, 38)

        # Test IP and port of the proxy progress bars
        self.assertEqual(self.proxybar1.ip, self.zmqbar.ip)
        self.assertEqual(self.proxybar1.port, self.zmqbar.port)
        self.assertEqual(self.proxybar2.ip, self.zmqbar.ip)
        self.assertEqual(self.proxybar2.port, self.zmqbar.port)
        self.assertEqual(proxybar3.ip, self.zmqbar.ip)
        self.assertEqual(proxybar3.port, self.zmqbar.port)


    def test_proxy_progressbars(self):
        # Test the information in the proxybar1
        self.assertEqual(self.proxybar1.client_id, 0)
        self.assertEqual(self.proxybar1.ip, self.zmqbar.ip)
        self.assertEqual(self.proxybar1.port, self.zmqbar.port)

        # Test the information in the proxybar2
        self.assertEqual(self.proxybar2.client_id, 1)
        self.assertEqual(self.proxybar2.ip, self.zmqbar.ip)
        self.assertEqual(self.proxybar2.port, self.zmqbar.port)

        # Since we did not call the progress method of the proxy
        # progressbars not even once yet, they have not created their
        # sockets yet.
        self.assertIsNone(self.proxybar1._zmq_push_socket)
        self.assertIsNone(self.proxybar2._zmq_push_socket)
        self.assertIsNone(self.proxybar1._zmq_context)
        self.assertIsNone(self.proxybar2._zmq_context)

        # Before the first time the progress method in self.proxybar1 and
        # self.proxybar2 is called their "_progress_func" variable points
        # to the "_connect_and_update_progress" method
        self.assertTrue(self.proxybar1._progress_func == progressbar.ProgressbarZMQProxy._connect_and_update_progress)
        self.assertTrue(self.proxybar2._progress_func == progressbar.ProgressbarZMQProxy._connect_and_update_progress)

    def test_update_progress(self):
        from time import sleep
        #self.zmqbar._sleep_time = 5
        self.zmqbar.start_updater()
        self.proxybar1.progress(5)
        self.proxybar2.progress(10)
        sleep(0.3)
        self.zmqbar.stop_updater()

        # Open and read the progress from the file
        progress_output_file = open(self.output_filename)
        progress_string = progress_output_file.read()

        # Expected string with the progress output
        expected_progress_string = """------------------ Some message -----------------1
    1    2    3    4    5    6    7    8    9    0
----0----0----0----0----0----0----0----0----0----0
******************************"""

        self.assertEqual(progress_string, expected_progress_string)

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def progress_producer2(bar, sleep_time=0.5):  # pragma: no cover
    total_count = 20
    for i in range(1, total_count + 1):
        sleep(sleep_time)
        bar.progress(i)


def progress_producer(process_id, process_data_list, sleep_time=0.5):  # pragma: no cover
    total_count = 20
    for i in range(1, total_count + 1):
        sleep(sleep_time)
        process_data_list[process_id] = i


if __name__ == '__main__1':  # pragma: no cover
    from time import sleep
    import multiprocessing
    from util.progressbar import ProgressbarMultiProcessText
    import sys

    bar = ProgressbarMultiProcessText(sleep_time=1)

    # # xxxxx Option 1: register_client xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # # Register two functions with count 20, each, in the progressbar
    # func_1_data = bar._register_client(20)
    # func_2_data = bar._register_client(20)

    # # Create the processes to run the functions
    # p1 = multiprocessing.Process(target=progress_producer, args=(func_1_data[0], func_1_data[1], 0.2))
    # p2 = multiprocessing.Process(target=progress_producer, args=(func_2_data[0], func_2_data[1], 0.3))
    # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Option 2: register_client_and_get_proxy_progressbar xxxxxxxxx
    # Register two functions with count 20, each, in the progressbar
    proxybar1 = bar.register_client_and_get_proxy_progressbar(20)
    proxybar2 = bar.register_client_and_get_proxy_progressbar(20)

    # Create the processes to run the functions
    p1 = multiprocessing.Process(target=progress_producer2, args=(proxybar1, 0.2))
    p2 = multiprocessing.Process(target=progress_producer2, args=(proxybar2, 0.3))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Start the processes and the progressbar updating
    bar.start_updater()
    p1.start()
    p2.start()

    p1.join()
    p2.join()

    # Stop the process that updates the progressbar.
    bar.stop_updater()

    print("The End")


# xxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
