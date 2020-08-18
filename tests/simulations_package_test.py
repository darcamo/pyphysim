#!/usr/bin/env python

# pylint: disable=E1101,E1103,W0403
"""
Tests for the modules in the simulations package.

Each module has doctests for its functions and all we need to do is run all
of them.
"""

import doctest
import glob
import json
import os
import sys
import unittest
from copy import copy
from itertools import repeat
from pathlib import Path

import numpy as np

from pyphysim import progressbar
from pyphysim.simulations import (configobjvalidation, parameters, results,
                                  runner, simulationhelpers)
# noinspection PyProtectedMember
from pyphysim.simulations.configobjvalidation import (
    _parse_float_range_expr, integer_scalar_or_integer_numpy_array_check,
    real_scalar_or_real_numpy_array_check)
from pyphysim.simulations.parameters import (SimulationParameters,
                                             combine_simulation_parameters)
from pyphysim.simulations.results import (Result, SimulationResults,
                                          combine_simulation_results)
from pyphysim.simulations.runner import (SimulationRunner, SkipThisOne,
                                         get_common_parser)
from pyphysim.util.misc import calc_confidence_interval

try:
    # noinspection PyUnresolvedReferences
    from ipyparallel import CompositeError

    _IPYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover
    _IPYTHON_AVAILABLE = False

try:  # pragma: nocover
    # noinspection PyUnresolvedReferences
    from pandas import DataFrame
    assert DataFrame  # Avoid unused import warning for DataFrame

    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: nocover
    _PANDAS_AVAILABLE = False


def delete_file_if_possible(filename):
    """
    Try to delete the file with name `filename`.

    Parameters
    ----------
    filename : str
        The name of the file to be removed.
    """
    try:
        os.remove(filename)
    except OSError:  # pragma: no cover
        pass


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# noinspection PyMethodMayBeStatic
class SimulationsDoctestsTestCase(unittest.TestCase):
    """
    Test case that run all the doctests in the modules of the simulations
    package.
    """
    def test_configobjvalidation(self) -> None:
        """Run configobjvalidation doctests"""
        doctest.testmod(configobjvalidation)

    def test_parameters(self) -> None:
        """Run parameters doctests"""
        doctest.testmod(parameters)

    def test_progressbar(self) -> None:
        """Run progressbar doctests"""
        doctest.testmod(progressbar)

    def test_results(self) -> None:
        """Run results doctests"""
        doctest.testmod(results)

    def test_simulationhelpers(self) -> None:
        """Run simulationhelpers doctests"""
        doctest.testmod(simulationhelpers)

    def test_runner(self) -> None:
        """Run runner doctests"""
        doctest.testmod(runner)


class SimulationHelpersTestCase(unittest.TestCase):
    def test_get_common_parser(self) -> None:
        p = get_common_parser()
        p2 = get_common_parser()
        self.assertTrue(p is p2)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx configobjvalidation Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# noinspection PyUnboundLocalVariable
class ConfigobjvalidationModuleFunctionsTestCase(unittest.TestCase):
    """
    Unit-tests for the module functions in the in the configobjvalidation
    module.
    """
    def setUp(self) -> None:
        """Called before each test."""
        pass

    def test_parse_range_expr(self) -> None:
        try:
            # noinspection PyUnresolvedReferences
            import validate
        except ImportError:  # pragma: no cover
            self.skipTest("test_parse_range_expr - "
                          "The validate module is not installed")

        expr = "10:15"
        expected_parsed_expr = np.r_[10:15]
        parsed_expr = _parse_float_range_expr(expr)

        np.testing.assert_array_almost_equal(expected_parsed_expr, parsed_expr)

        expr = "10:2:15"
        expected_parsed_expr = np.r_[10:15:2]
        parsed_expr = _parse_float_range_expr(expr)

        np.testing.assert_array_almost_equal(expected_parsed_expr, parsed_expr)

        expr = "-3.4:0.5:5"
        expected_parsed_expr = np.r_[-3.4:5.0:0.5]
        parsed_expr = _parse_float_range_expr(expr)

        np.testing.assert_array_almost_equal(expected_parsed_expr, parsed_expr)

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

    # Note: Since the "real_scalar_or_real_numpy_array_check" function will
    # call the "real_numpy_array_check" function we only need a test case
    # for the "real_scalar_or_real_numpy_array_check" function.
    def test_real_scalar_or_real_numpy_array_check(self) -> None:
        try:
            # noinspection PyUnresolvedReferences
            import validate
        except ImportError:  # pragma: no cover
            self.skipTest("test_real_scalar_or_real_numpy_array_check - "
                          "The validate module is not installed")

        # xxxxxxxxxx Try to parse float scalar values xxxxxxxxxxxxxxxxxxxxx
        value = "4.6"
        expected_parsed_value = 4.6
        self.assertAlmostEqual(real_scalar_or_real_numpy_array_check(value),
                               expected_parsed_value)
        self.assertTrue(
            isinstance(real_scalar_or_real_numpy_array_check(value), float))

        value = "76.21"
        expected_parsed_value = 76.21
        self.assertAlmostEqual(real_scalar_or_real_numpy_array_check(value),
                               expected_parsed_value)
        self.assertTrue(
            isinstance(real_scalar_or_real_numpy_array_check(value), float))

        # Test validation against the minimum and maximum allowed value
        value = "5.7"
        with self.assertRaises(validate.VdtValueTooSmallError):
            real_scalar_or_real_numpy_array_check(value, min=10.0)

        with self.assertRaises(validate.VdtValueTooBigError):
            real_scalar_or_real_numpy_array_check(value, max=5.0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Now we will parse range expressions xxxxxxxxxxxxxxxxxx
        # Test when the input is a list of strings (with the numbers)
        list_of_strings = ['0', '6', '17']
        parsed_array = real_scalar_or_real_numpy_array_check(list_of_strings,
                                                             min=0,
                                                             max=30)
        expected_parsed_array = np.array([0., 6., 17.])
        self.assertTrue(np.array(parsed_array).dtype is np.dtype('float'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        # Test when the input is a string representation of a list with
        # numbers and range expressions.
        array_string = "[0 5 10:15]"
        parsed_array = real_scalar_or_real_numpy_array_check(array_string,
                                                             min=0,
                                                             max=30)
        expected_parsed_array = np.array([0., 5., 10., 11., 12., 13., 14.])
        self.assertTrue(np.array(parsed_array).dtype is np.dtype('float'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        array_string = "10:15"
        parsed_array = real_scalar_or_real_numpy_array_check(array_string,
                                                             min=0,
                                                             max=30)
        expected_parsed_array = np.array([10., 11., 12., 13., 14.])
        self.assertTrue(np.array(parsed_array).dtype is np.dtype('float'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        array_string = "[10:15]"
        parsed_array = real_scalar_or_real_numpy_array_check(array_string,
                                                             min=0,
                                                             max=30)
        expected_parsed_array = np.array([10., 11., 12., 13., 14.])
        self.assertTrue(np.array(parsed_array).dtype is np.dtype('float'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        array_string = "[0,5,10:15,20]"
        parsed_array = real_scalar_or_real_numpy_array_check(array_string,
                                                             min=0,
                                                             max=30)
        expected_parsed_array = np.array(
            [0., 5., 10., 11., 12., 13., 14., 20.])
        self.assertTrue(np.array(parsed_array).dtype is np.dtype('float'))
        np.testing.assert_array_almost_equal(parsed_array,
                                             expected_parsed_array)

        # xxxxx Test validation against the minimum allowed value xxxxxxxxx
        array_string = "[0,5,10:15,20]"
        with self.assertRaises(validate.VdtValueTooSmallError):
            real_scalar_or_real_numpy_array_check(array_string, min=4, max=30)

        # xxxxx Test validation against the minimum allowed value xxxxxxxxx
        with self.assertRaises(validate.VdtValueTooBigError):
            real_scalar_or_real_numpy_array_check(array_string, min=0, max=15)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Note: Since the "integer_scalar_or_integer_numpy_array_check"
    # function will call the "integer_numpy_array_check" function we only
    # need a test case for the
    # "integer_scalar_or_integer_numpy_array_check" function.
    def test_integer_scalar_or_integer_numpy_array_check(self) -> None:
        try:
            # noinspection PyUnresolvedReferences
            import validate
        except ImportError:  # pragma: no cover
            self.skipTest("test_integer_scalar_or_integer_numpy_array_check - "
                          "The validate module is not installed")

        # xxxxxxxxxx Try to parse float scalar values xxxxxxxxxxxxxxxxxxxxx
        value = "4"
        expected_parsed_value = 4
        self.assertAlmostEqual(
            integer_scalar_or_integer_numpy_array_check(value),
            expected_parsed_value)
        self.assertTrue(
            isinstance(integer_scalar_or_integer_numpy_array_check(value),
                       int))

        value = "76"
        expected_parsed_value = 76
        self.assertAlmostEqual(
            integer_scalar_or_integer_numpy_array_check(value),
            expected_parsed_value)
        self.assertTrue(
            isinstance(integer_scalar_or_integer_numpy_array_check(value),
                       int))

        # Test validation against the minimum and maximum allowed value
        value = "6"
        with self.assertRaises(validate.VdtValueTooSmallError):
            integer_scalar_or_integer_numpy_array_check(value, min=10)

        with self.assertRaises(validate.VdtValueTooBigError):
            integer_scalar_or_integer_numpy_array_check(value, max=5)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Now we will parse range expressions xxxxxxxxxxxxxxxxxx
        # test when the input is a list of strings
        list_of_strings = ['0', '6', '17']
        parsed_array = integer_scalar_or_integer_numpy_array_check(
            list_of_strings, min=0, max=30)
        expected_parsed_array = np.array([0, 6, 17])
        self.assertTrue(np.array(parsed_array).dtype is np.dtype('int'))
        np.testing.assert_array_equal(parsed_array, expected_parsed_array)

        array_string = "[0 5 10:15]"
        parsed_array = integer_scalar_or_integer_numpy_array_check(
            array_string, min=0, max=30)
        expected_parsed_array = np.array([0, 5, 10, 11, 12, 13, 14])
        self.assertTrue(np.array(parsed_array).dtype is np.dtype('int'))
        np.testing.assert_array_equal(parsed_array, expected_parsed_array)

        array_string = "10:15"
        parsed_array = integer_scalar_or_integer_numpy_array_check(
            array_string, min=0, max=30)
        expected_parsed_array = np.array([10, 11, 12, 13, 14])
        self.assertTrue(np.array(parsed_array).dtype is np.dtype('int'))
        np.testing.assert_array_equal(parsed_array, expected_parsed_array)

        array_string = "[10:15]"
        parsed_array = integer_scalar_or_integer_numpy_array_check(
            array_string, min=0, max=30)
        expected_parsed_array = np.array([10, 11, 12, 13, 14])
        self.assertTrue(np.array(parsed_array).dtype is np.dtype('int'))
        np.testing.assert_array_equal(parsed_array, expected_parsed_array)

        array_string = "[0,5,10:15,20]"
        parsed_array = integer_scalar_or_integer_numpy_array_check(
            array_string, min=0, max=30)
        expected_parsed_array = np.array([0, 5, 10, 11, 12, 13, 14, 20])
        self.assertTrue(np.array(parsed_array).dtype is np.dtype('int'))
        np.testing.assert_array_equal(parsed_array, expected_parsed_array)

        # xxxxx Test validation against the minimum allowed value xxxxxxxxx
        array_string = "[0,5,10:15,20]"
        with self.assertRaises(validate.VdtValueTooSmallError):
            integer_scalar_or_integer_numpy_array_check(array_string,
                                                        min=4,
                                                        max=30)

        with self.assertRaises(validate.VdtValueTooBigError):
            integer_scalar_or_integer_numpy_array_check(array_string,
                                                        min=0,
                                                        max=15)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Parameters Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ParametersModuleFunctionsTestCase(unittest.TestCase):
    """
    Unit-tests for the functions in the parameters module.
    """
    def setUp(self) -> None:
        """Called before each test."""
        pass

    def test_combine_simulation_parameters(self) -> None:
        sim_params1 = SimulationParameters.create({
            'first':
            10,
            'second':
            20,
            'third':
            np.array([1, 3, 2, 5]),
            'fourth': ['A', 'B']
        })
        sim_params1.set_unpack_parameter('third')
        sim_params1.set_unpack_parameter('fourth')

        sim_params2 = SimulationParameters.create({
            'first':
            10,
            'second':
            20,
            'third':
            np.array([-1, 1, 3, 8])
        })
        sim_params2.set_unpack_parameter('third')

        sim_params3 = SimulationParameters.create({
            'first':
            10,
            'third':
            np.array([-1, 1, 3, 8]),
            'fourth': ['B', 'C']
        })
        sim_params3.set_unpack_parameter('third')
        sim_params3.set_unpack_parameter('fourth')

        sim_params4 = SimulationParameters.create({
            'first':
            10,
            'second':
            30,
            'third':
            np.array([-1, 1, 3, 8]),
            'fourth': ['B', 'C']
        })
        sim_params4.set_unpack_parameter('third')
        sim_params4.set_unpack_parameter('fourth')

        sim_params5 = SimulationParameters.create({
            'first':
            10,
            'second':
            20,
            'third':
            np.array([-1, 1, 3, 8]),
            'fourth': ['B', 'C']
        })
        sim_params5.set_unpack_parameter('fourth')

        sim_params6 = SimulationParameters.create({
            'first':
            10,
            'second':
            20,
            'third':
            np.array([-1, 1, 3, 8]),
            'fourth': ['B', 'C']
        })
        sim_params6.set_unpack_parameter('third')
        sim_params6.set_unpack_parameter('fourth')

        # xxxxxxxxxx Test invalid cases xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        with self.assertRaises(RuntimeError):
            # sim_params2 does not have the same parameters of sim_params1
            combine_simulation_parameters(sim_params1, sim_params2)

        with self.assertRaises(RuntimeError):
            # sim_params3 does not have the same parameters of sim_params1
            combine_simulation_parameters(sim_params1, sim_params3)

        with self.assertRaises(RuntimeError):
            # sim_params4 has the same parameters of sim_params1, but the
            # value of one of the fixed parameters is different
            combine_simulation_parameters(sim_params1, sim_params4)

        with self.assertRaises(RuntimeError):
            # sim_params4 has the same parameters of sim_params1, but the
            # parameters set to be unpacked are different
            combine_simulation_parameters(sim_params1, sim_params5)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test the valid case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        union = combine_simulation_parameters(sim_params1, sim_params6)
        self.assertEqual(union['first'], sim_params1['first'])
        self.assertEqual(union['second'], sim_params1['second'])
        np.testing.assert_array_almost_equal(union['third'],
                                             np.array([-1, 1, 2, 3, 5, 8]))
        np.testing.assert_array_equal(union['fourth'],
                                      np.array(['A', 'B', 'C']))

        self.assertEqual(set(union.unpacked_parameters), {'third', 'fourth'})


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class SimulationParametersTestCase(unittest.TestCase):
    """
    Unit-tests for the SimulationParameters class in the parameters module.
    """
    def setUp(self) -> None:
        params_dict = {'first': 10, 'second': 20}
        self.sim_params = SimulationParameters.create(params_dict)

    def test_create(self) -> None:
        # The create method was already called in the setUp.
        self.assertEqual(len(self.sim_params), 2)
        self.assertEqual(self.sim_params['first'], 10)
        self.assertEqual(self.sim_params['second'], 20)

    def test_add(self) -> None:
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.assertEqual(len(self.sim_params), 3)
        np.testing.assert_array_equal(self.sim_params['third'],
                                      np.array([1, 3, 2, 5]))

        # We can also add a parameter with __setitem__
        self.sim_params["fourth"] = 17.5
        self.assertEqual(len(self.sim_params), 4)
        np.testing.assert_array_equal(self.sim_params['fourth'], 17.5)

    def test_unpacking_parameters(self) -> None:
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.add('fourth', ['A', 'B'])
        self.assertEqual(self.sim_params.get_num_unpacked_variations(), 1)

        # Test the correct static parameters (should be all parameters for
        # now)
        self.assertEqual(set(self.sim_params.fixed_parameters),
                         {'first', 'second', 'third', 'fourth'})

        # Let's unpack the parameters 'third' and 'fourth'
        self.sim_params.set_unpack_parameter('third')
        self.sim_params.set_unpack_parameter('fourth')

        # One unpacked param with four values and other with two will give
        # us 4x2=8 unpacked variations. Let's test the
        # get_num_unpacked_variations() method.
        self.assertEqual(self.sim_params.get_num_unpacked_variations(), 8)
        # We make the unpacked_parameters and the expected value sets
        # because the order does not matter
        self.assertEqual(set(self.sim_params.unpacked_parameters),
                         {'third', 'fourth'})

        # We may have 8 variations, but there are still only 4 parameters
        self.assertEqual(len(self.sim_params), 4)

        # Test the correct static parameters
        self.assertEqual(set(self.sim_params.fixed_parameters),
                         {'first', 'second'})

        # Test if an exception is raised if we try to set a non iterable
        # parameter to be unpacked.
        self.sim_params.add('fifth', 10)
        with self.assertRaises(ValueError):
            self.sim_params.set_unpack_parameter('fifth')

        # Test if an exception is thrown if we try to set a non existing
        # parameter to be unset.
        with self.assertRaises(ValueError):
            self.sim_params.set_unpack_parameter('sixth')

        # xxxxx THIS TEST WILL NOT BE PERFORMED IN PYTHON 3 xxxxxxxxxxxxxxx
        if sys.version_info[0] < 3:
            # Now that a few parameters were added and set to be unpacked,
            # lets test the representation of the SimulationParameters
            # object. Note that the parameters that are marked for
            # unpacking have '*' appended to their name.
            self.assertEqual(
                self.sim_params.__repr__(),
                "{'second': 20, 'fifth': 10, 'fourth*': ['A', 'B'], "
                "'third*': [1 3 2 5], 'first': 10}")
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Test if we can unset a parameter that was previously set to be
        # unpacked.
        self.sim_params.set_unpack_parameter('fourth', False)
        self.assertEqual(set(self.sim_params.unpacked_parameters), {'third'})

    def test_remove(self) -> None:
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.add('fourth', ['A', 'B'])
        self.sim_params.add('fifth', ['Z', 'W', 'Y'])

        self.sim_params.set_unpack_parameter('third')
        self.sim_params.set_unpack_parameter('fourth')

        # Remove parameters second and third
        self.sim_params.remove('second')
        # Note that this parameter was marked to be unpacked
        self.sim_params.remove('third')

        expected_parameters = {
            'first': 10,
            'fourth': ['A', 'B'],
            'fifth': ['Z', 'W', 'Y']
        }
        expected_unpacked_parameters = {'fourth'}
        self.assertEqual(self.sim_params.parameters, expected_parameters)
        self.assertEqual(set(self.sim_params.unpacked_parameters),
                         expected_unpacked_parameters)

    def test_equal_and_not_equal_operators(self) -> None:
        other = SimulationParameters()
        self.assertFalse(self.sim_params == other)
        self.assertTrue(self.sim_params != other)
        other.add('first', 10)
        other.add('second', 20)
        self.assertTrue(self.sim_params == other)
        self.assertFalse(self.sim_params != other)

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

        # The rep_max parameter is not considering when testing if two
        # SimulationParameters objects are equal or not.
        other.add('rep_max', 30)
        self.sim_params.add('rep_max', 40)
        self.assertTrue(self.sim_params == other)

        # Test comparison with something of a different class
        # noinspection PyTypeChecker
        self.assertFalse(self.sim_params == 10.4)

        # Test if two objects are different if the _unpack_index value is
        # different
        a = SimulationParameters()
        a.add('first', [10, 20, 30])
        a.add('second', [60, 20, 0])
        a.set_unpack_parameter('first')
        b = SimulationParameters()
        b.add('first', [10, 20, 30])
        b.add('second', [60, 20, 0])
        b.set_unpack_parameter('first')
        self.assertTrue(a == b)
        a._unpack_index = 1
        b._unpack_index = 3
        self.assertFalse(a == b)

    def test_repr(self) -> None:
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.set_unpack_parameter("third")
        self.assertEqual(repr(self.sim_params),
                         "{'first': 10, 'second': 20, 'third*': [1 3 2 5]}")

    def test_get_unpacked_params_list(self) -> None:
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.add('fourth', ['A', 'B'])

        unpacked_param_list = self.sim_params.get_unpacked_params_list()
        self.assertEqual(unpacked_param_list, [self.sim_params])

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
        self.assertEqual(set(params_dict['first']), {self.sim_params['first']})
        self.assertEqual(set(params_dict['second']),
                         {self.sim_params['second']})
        self.assertEqual(set(params_dict['third']),
                         set(self.sim_params['third']))
        self.assertEqual(set(params_dict['fourth']),
                         set(self.sim_params['fourth']))

        # Test if the _unpack_index and the _original_sim_params member
        # variables are correct for each unpacked variation
        for i in range(self.sim_params.get_num_unpacked_variations()):
            self.assertEqual(unpacked_param_list[i]._unpack_index, i)
            self.assertTrue(
                unpacked_param_list[i]._original_sim_params is self.sim_params)

    def test_get_num_unpacked_variations(self) -> None:
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.add('fourth', ['A', 'B'])
        self.assertEqual(self.sim_params.get_num_unpacked_variations(), 1)
        self.sim_params.set_unpack_parameter('third')
        self.sim_params.set_unpack_parameter('fourth')

        # One unpacked param with four values and other with two will give
        # us 4x2=8 unpacked variations. Let's test the
        # get_num_unpacked_variations() method.
        self.assertEqual(self.sim_params.get_num_unpacked_variations(), 8)

        # Get the unpacked params list.
        unpacked_params_list = self.sim_params.get_unpacked_params_list()

        # If we call the get_num_unpacked_variations method of a unpacked
        # SimulationParameters object, it should return the number of
        # unpacked variations of the PARENT object, instead just returning
        # 1.
        for u in unpacked_params_list:
            self.assertEqual(self.sim_params.get_num_unpacked_variations(),
                             u.get_num_unpacked_variations())

    def test_get_pack_indexes(self) -> None:
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.add('fourth', ['A', 'B'])
        self.sim_params.add('fifth', ['Z', 'X', 'W'])
        self.sim_params.set_unpack_parameter('third')
        self.sim_params.set_unpack_parameter('fourth')
        self.sim_params.set_unpack_parameter('fifth')

        unpacked_list = self.sim_params.get_unpacked_params_list()

        # The parameters 'third' and 'fourth' are marked to be unpacked,
        # while the parameters 'first' and 'second' will always be the
        # same. The combinations after unpacking are shown below (the order
        # might be different depending on which version of the python
        # interpreter was used)
        #
        #   {'second': 20, 'fifth': Z, 'fourth': A, 'third': 1, 'first': 10}
        #   {'second': 20, 'fifth': Z, 'fourth': A, 'third': 3, 'first': 10}
        #   {'second': 20, 'fifth': Z, 'fourth': A, 'third': 2, 'first': 10}
        #   {'second': 20, 'fifth': Z, 'fourth': A, 'third': 5, 'first': 10}
        #   {'second': 20, 'fifth': Z, 'fourth': B, 'third': 1, 'first': 10}
        #   {'second': 20, 'fifth': Z, 'fourth': B, 'third': 3, 'first': 10}
        #   {'second': 20, 'fifth': Z, 'fourth': B, 'third': 2, 'first': 10}
        #   {'second': 20, 'fifth': Z, 'fourth': B, 'third': 5, 'first': 10}
        #   {'second': 20, 'fifth': X, 'fourth': A, 'third': 1, 'first': 10}
        #   {'second': 20, 'fifth': X, 'fourth': A, 'third': 3, 'first': 10}
        #   {'second': 20, 'fifth': X, 'fourth': A, 'third': 2, 'first': 10}
        #   {'second': 20, 'fifth': X, 'fourth': A, 'third': 5, 'first': 10}
        #   {'second': 20, 'fifth': X, 'fourth': B, 'third': 1, 'first': 10}
        #   {'second': 20, 'fifth': X, 'fourth': B, 'third': 3, 'first': 10}
        #   {'second': 20, 'fifth': X, 'fourth': B, 'third': 2, 'first': 10}
        #   {'second': 20, 'fifth': X, 'fourth': B, 'third': 5, 'first': 10}
        #   {'second': 20, 'fifth': W, 'fourth': A, 'third': 1, 'first': 10}
        #   {'second': 20, 'fifth': W, 'fourth': A, 'third': 3, 'first': 10}
        #   {'second': 20, 'fifth': W, 'fourth': A, 'third': 2, 'first': 10}
        #   {'second': 20, 'fifth': W, 'fourth': A, 'third': 5, 'first': 10}
        #   {'second': 20, 'fifth': W, 'fourth': B, 'third': 1, 'first': 10}
        #   {'second': 20, 'fifth': W, 'fourth': B, 'third': 3, 'first': 10}
        #   {'second': 20, 'fifth': W, 'fourth': B, 'third': 2, 'first': 10}
        #   {'second': 20, 'fifth': W, 'fourth': B, 'third': 5, 'first': 10}
        #
        # Lets focus on the 'third' and 'fourth' parameters, since they are
        # the only ones changing. Suppose we want to get the indexes
        # corresponding to varying the 'fourth' parameters with the 'third'
        # parameter equal to 2. We create a dictionary
        fixed_third_2 = {'third': 2}

        # Get the indexes where the third parameter has a value of 2.
        fixed_third_2_indexes = self.sim_params.get_pack_indexes(fixed_third_2)
        self.assertEqual(len(fixed_third_2_indexes), 6)

        # Now we test if for these indexes the value of the third parameter
        # is really 2
        for i in fixed_third_2_indexes:
            self.assertEqual(unpacked_list[i]['third'], 2)

        fixed_third_5 = {'third': 5}
        fixed_third_5_indexes = self.sim_params.get_pack_indexes(fixed_third_5)
        self.assertEqual(len(fixed_third_5_indexes), 6)
        for i in fixed_third_5_indexes:
            self.assertEqual(unpacked_list[i]['third'], 5)

        # now lets fix the 'fourth' parameter and let the 'third' vary.
        fixed_fourth_A = {'fourth': 'A'}
        fixed_fourth_A_indexes = self.sim_params.get_pack_indexes(
            fixed_fourth_A)
        self.assertEqual(len(fixed_fourth_A_indexes), 12)
        for i in fixed_fourth_A_indexes:
            self.assertEqual(unpacked_list[i]['fourth'], 'A')

        fixed_fourth_B = {'fourth': 'B'}
        fixed_fourth_B_indexes = self.sim_params.get_pack_indexes(
            fixed_fourth_B)
        self.assertEqual(len(fixed_fourth_B_indexes), 12)
        for i in fixed_fourth_B_indexes:
            self.assertEqual(unpacked_list[i]['fourth'], 'B')

        # Lets try to fix some invalid value to see if an exception is
        # raised
        fixed_fourth_invalid = {'fourth': 'C'}
        with self.assertRaises(ValueError):
            # This should raise a ValueError, since the parameter 'fourth'
            # has no value 'C'
            self.sim_params.get_pack_indexes(fixed_fourth_invalid)

        # Now lets fix the third, fourth and fifth parameters. This should
        # get me a single index.
        index1 = self.sim_params.get_pack_indexes({
            'third': 5,
            'fourth': 'B',
            'fifth': 'Z'
        })[0]
        # Now we use the index to get an element in the unpacked_list and
        # check if the values are the ones that we have fixed.
        self.assertEqual(unpacked_list[index1]['third'], 5)
        self.assertEqual(unpacked_list[index1]['fourth'], 'B')
        self.assertEqual(unpacked_list[index1]['fifth'], 'Z')

        index2 = self.sim_params.get_pack_indexes({
            'third': 5,
            'fourth': 'B',
            'fifth': 'X'
        })[0]
        # Now we use the index to get an element in the unpacked_list and
        # check if the values are the ones that we have fixed.
        self.assertEqual(unpacked_list[index2]['third'], 5)
        self.assertEqual(unpacked_list[index2]['fourth'], 'B')
        self.assertEqual(unpacked_list[index2]['fifth'], 'X')

        index3 = self.sim_params.get_pack_indexes({
            'third': 2,
            'fourth': 'A',
            'fifth': 'Z'
        })[0]
        # Now we use the index to get an element in the unpacked_list and
        # check if the values are the ones that we have fixed.
        self.assertEqual(unpacked_list[index3]['third'], 2)
        self.assertEqual(unpacked_list[index3]['fourth'], 'A')
        self.assertEqual(unpacked_list[index3]['fifth'], 'Z')

    def test_to_dict_and_from_dict(self) -> None:
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.add('fourth', ['A', 'B'])
        self.sim_params.set_unpack_parameter('third')
        self.sim_params.set_unpack_parameter('fourth')

        # xxxxxxxxxx Test converting to and from a dictionary xxxxxxxxxxxxx
        # First test converting to a dictionary
        out = self.sim_params._to_dict()
        self.assertIsInstance(out, dict)
        self.assertEqual(out['parameters'], self.sim_params.parameters)
        self.assertEqual(out['unpacked_parameters_set'],
                         self.sim_params._unpacked_parameters_set)
        self.assertEqual(out['unpack_index'], self.sim_params._unpack_index)
        self.assertEqual(out['original_sim_params'],
                         self.sim_params._original_sim_params)

        # Now test converting from the dictionary
        sim_params = SimulationParameters._from_dict(out)
        self.assertEqual(sim_params, self.sim_params)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Repeat tests with unpacked parameters xxxxxxxxxxxxxxxx
        # First test converting to a dictionary
        sim_params_unpacked_list = self.sim_params.get_unpacked_params_list()

        for sim_params_elem in sim_params_unpacked_list:
            # sim_params_elem = sim_params_unpacked_list[0]
            out_elem = sim_params_elem._to_dict()
            self.assertIsInstance(out_elem, dict)
            self.assertEqual(out_elem['parameters'],
                             sim_params_elem.parameters)
            self.assertEqual(out_elem['unpacked_parameters_set'],
                             sim_params_elem._unpacked_parameters_set)
            self.assertEqual(out_elem['unpack_index'],
                             sim_params_elem._unpack_index)
            self.assertEqual(out_elem['original_sim_params'],
                             sim_params_elem._original_sim_params._to_dict())

            # Now test converting from the dictionary
            sim_params_elem_d = SimulationParameters._from_dict(out_elem)
            self.assertEqual(sim_params_elem, sim_params_elem_d)
            self.assertEqual(sim_params_elem_d._original_sim_params,
                             self.sim_params)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test with an empty SimulationParameters xxxxxxxxxxxxxx
        empty_simparams = SimulationParameters()
        empty_simparams_dict = empty_simparams._to_dict()
        empty_simparams_d = SimulationParameters._from_dict(
            empty_simparams_dict)
        self.assertEqual(empty_simparams, empty_simparams_d)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_to_json_and_from_json(self) -> None:
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.add('fourth', ['A', 'B'])
        self.sim_params.set_unpack_parameter('third')
        self.sim_params.set_unpack_parameter('fourth')

        # xxxxxxxxxx Test converting to and from a json xxxxxxxxxxxxxxxxxxx
        # First test converting to json
        encoded_params = self.sim_params.to_json()
        self.assertIsInstance(encoded_params, str)

        # This will raise an exception if encoded_params is not valid json
        _ = json.loads(encoded_params)

        # Now test converting from json
        decoded_params = SimulationParameters.from_json(encoded_params)
        self.assertTrue(self.sim_params == decoded_params)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Repeat tests with unpacked parameters xxxxxxxxxxxxxxxx
        sim_params_unpacked_list = self.sim_params.get_unpacked_params_list()
        for sim_params_elem in sim_params_unpacked_list:
            # First test converting to json
            encoded_sim_params_elem = sim_params_elem.to_json()
            self.assertIsInstance(encoded_sim_params_elem, str)

            # This will raise an exception if encoded_params is not valid json
            _ = json.loads(encoded_params)

            # Now test converting from json
            decoded_sim_params_elem = SimulationParameters.from_json(
                encoded_sim_params_elem)

            self.assertEqual(decoded_sim_params_elem, sim_params_elem)
            self.assertEqual(decoded_sim_params_elem._original_sim_params,
                             self.sim_params)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_save_to_and_load_from_pickle_file(self) -> None:
        self.sim_params.add('third', np.array([1, 3, 2, 5]))
        self.sim_params.add('fourth', ['A', 'B'])
        self.sim_params.set_unpack_parameter('third')
        self.sim_params.set_unpack_parameter('fourth')

        filename = 'params.pickle'
        # Let's make sure the file does not exist
        delete_file_if_possible(filename)

        # Save to the file
        self.sim_params.save_to_pickled_file(filename)

        # Load from the file
        sim_params2 = SimulationParameters.load_from_pickled_file(filename)

        self.assertEqual(self.sim_params['first'], sim_params2['first'])
        self.assertEqual(self.sim_params['second'], sim_params2['second'])
        self.assertEqual(len(self.sim_params), len(sim_params2))
        self.assertEqual(self.sim_params.get_num_unpacked_variations(),
                         sim_params2.get_num_unpacked_variations())

        # Delete the where the parameters were saved
        delete_file_if_possible(filename)

        # xxxxx Test saving and loading one of the unpacked variations xxxx
        filename2 = 'params_3.pickle'
        fourth_unpacked_param = self.sim_params.get_unpacked_params_list()[3]
        fourth_unpacked_param.save_to_pickled_file(filename2)

        fourth_unpacked_param2 = SimulationParameters.load_from_pickled_file(
            filename2)
        self.assertEqual(fourth_unpacked_param, fourth_unpacked_param2)

        # Delete the where the parameters were saved
        delete_file_if_possible(filename2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_save_and_load_sanity(self) -> None:
        # Test if when saving and loading the simulation parameters the
        # list of unpacked parameters is in the same order.
        #
        # Previously the order of the list of unpacked parameters could
        # change after saving and loading the SimulationParameters object
        # from a file. This would cause problems for the SimulationResults
        # class. This bug was fixed by making sure the order of the
        # unpacked parameters list is fixed (sorted by alphabetic order)
        # and here we test for the bug in case it ever comes back after
        # some code change.
        params = SimulationParameters.create({
            'first':
            np.array([0, 5, 10, 26]),
            'second': ['aha', 'ahe'],
            'third': [10],
            'fourth': [30],
            'fifth': ['hum', 'blabla'],
            'sixth': ['a', 'b', 'c'],
            'seventh': ['ok', 'not ok', 'maybe']
        })

        filename = 'paramsfile.pickle'

        # Test without unpacking any parameter
        unpacked_parameters = copy(params.unpacked_parameters)
        delete_file_if_possible(filename)
        params.save_to_pickled_file(filename)
        params2 = SimulationParameters.load_from_pickled_file(filename)
        self.assertEqual(unpacked_parameters, params2.unpacked_parameters)
        delete_file_if_possible(filename)

        # Unpack the first one and test
        params.set_unpack_parameter('first')
        unpacked_parameters = copy(params.unpacked_parameters)
        delete_file_if_possible(filename)
        params.save_to_pickled_file(filename)
        params2 = SimulationParameters.load_from_pickled_file(filename)
        self.assertEqual(unpacked_parameters, params2.unpacked_parameters)
        delete_file_if_possible(filename)

        # Unpack the second one and test again
        params.set_unpack_parameter('second')
        unpacked_parameters = copy(params.unpacked_parameters)
        delete_file_if_possible(filename)
        params.save_to_pickled_file(filename)
        params2 = SimulationParameters.load_from_pickled_file(filename)
        self.assertEqual(unpacked_parameters, params2.unpacked_parameters)
        delete_file_if_possible(filename)

        # Unpack the third one and test again
        params.set_unpack_parameter('third')
        unpacked_parameters = copy(params.unpacked_parameters)
        delete_file_if_possible(filename)
        params.save_to_pickled_file(filename)
        params2 = SimulationParameters.load_from_pickled_file(filename)
        self.assertEqual(unpacked_parameters, params2.unpacked_parameters)
        delete_file_if_possible(filename)

        # Unpack the fourth one and test again
        params.set_unpack_parameter('fourth')
        unpacked_parameters = copy(params.unpacked_parameters)
        delete_file_if_possible(filename)
        params.save_to_pickled_file(filename)
        params2 = SimulationParameters.load_from_pickled_file(filename)
        self.assertEqual(unpacked_parameters, params2.unpacked_parameters)
        delete_file_if_possible(filename)

        # Unpack the fifth one and test again
        params.set_unpack_parameter('fifth')
        unpacked_parameters = copy(params.unpacked_parameters)
        delete_file_if_possible(filename)
        params.save_to_pickled_file(filename)
        params2 = SimulationParameters.load_from_pickled_file(filename)
        self.assertEqual(unpacked_parameters, params2.unpacked_parameters)
        delete_file_if_possible(filename)

        # Unpack the sixth one and test again
        params.set_unpack_parameter('sixth')
        unpacked_parameters = copy(params.unpacked_parameters)
        delete_file_if_possible(filename)
        params.save_to_pickled_file(filename)
        params2 = SimulationParameters.load_from_pickled_file(filename)
        self.assertEqual(unpacked_parameters, params2.unpacked_parameters)
        delete_file_if_possible(filename)

        # Unpack the seventh one and test again
        params.set_unpack_parameter('seventh')
        unpacked_parameters = copy(params.unpacked_parameters)
        delete_file_if_possible(filename)
        params.save_to_pickled_file(filename)
        params2 = SimulationParameters.load_from_pickled_file(filename)
        self.assertEqual(unpacked_parameters, params2.unpacked_parameters)
        delete_file_if_possible(filename)

    # noinspection PyUnresolvedReferences
    def test_load_from_config_file(self) -> None:
        try:
            import configobj
            import validate
            del configobj
            del validate
        except ImportError:  # pragma: no cover
            self.skipTest(
                "test_load_from_config_file - "
                "This configobj and validate modules must be installed.")

        filename = 'test_config_file.txt'

        # xxxxxxxxxx Write the config file xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # First we delete the file if it exists
        delete_file_if_possible(filename)

        fid = open(filename, 'w')
        fid.write("modo=test\n[Scenario]\nSNR=0,5,10\nM=4\nmodulator=PSK\n"
                  "[IA Algorithm]\nmax_iterations=60")
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
        params2 = SimulationParameters.load_from_config_file(filename, spec)
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
        fid.write("modo=test\n[Scenario]\nSNR=0,5,10\nM=-4\nmodulator=PSK\n"
                  "[IA Algorithm]\nmax_iterations=60")
        fid.close()

        with self.assertRaises(Exception):
            SimulationParameters.load_from_config_file(filename, spec)

        # Now we do not provide the required parameter max_iterations
        fid = open(filename, 'w')
        fid.write("modo=test\n[Scenario]\nSNR=0,5,10\nM=4\nmodulator=PSK\n"
                  "[IA Algorithm]")
        fid.close()

        with self.assertRaises(Exception):
            SimulationParameters.load_from_config_file(filename, spec)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Remove the config file used in this test xxxxxxxxxxxxx
        delete_file_if_possible(filename)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_to_dataframe(self) -> None:
        # If the pandas package is not installed, we will skip testing this
        # method
        if not _PANDAS_AVAILABLE:  # pragma: nocover
            self.skipTest("test_to_dataframe - Pandas is not installed")

        # Create some dummy parameters (including two parameters set to be
        # unpacked)
        params = SimulationParameters()

        # Try converting an empty SimulationParameters
        df = params.to_dataframe()
        self.assertTrue(isinstance(df, DataFrame))
        self.assertTrue(df.empty)

        params.add("extra", 2.3)
        params.add("SNR", np.array([0, 3, 6, 9]))
        params.add("bias", [1.2, 1.6])
        params.set_unpack_parameter('SNR')
        params.set_unpack_parameter('bias')
        params.add("Name", "Some string")

        df = params.to_dataframe()
        self.assertEqual(len(df.columns), 4)
        self.assertEqual(len(df), params.get_num_unpacked_variations())

        col_names = ["SNR", "bias", "Name", "extra"]
        for name in col_names:
            self.assertIn(name, df.columns)

        for i, l in enumerate(params.get_unpacked_params_list()):
            for col_names in ["SNR", "bias", "Name", "extra"]:
                self.assertAlmostEqual(l[col_names], df.loc[i, col_names])


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Results Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ResultsModuleFunctionsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        """Called before each test."""
        pass

    def test_combine_simulation_results(self) -> None:
        dummyrunner = _DummyRunner()
        dummyrunner.simulate()

        dummyrunner2 = _DummyRunner()
        dummyrunner2.simulate()

        dummyrunner3 = _DummyRunner()
        dummyrunner3.params.add('extra', np.array([4.1, 5.7, 6.2]))
        dummyrunner3.simulate()

        results1 = dummyrunner.results
        results2 = dummyrunner2.results
        results3 = dummyrunner3.results
        results2.add_new_result('bla', Result.SUMTYPE, 10)

        # Lets modify the result1 just to make the tests later not trivial
        for r in results1['lala']:
            r._value += 0.8

        # If both SimulationResults objects do not have exactly the same
        # results (names, not values), an exception will be raised.
        with self.assertRaises(RuntimeError):
            combine_simulation_results(results1, results2)

        # results1 and results3 can be combined
        union = combine_simulation_results(results1, results3)

        # xxxxxxxxxx Test the parameters of the combined object xxxxxxxxxxx
        # Note that combine_simulation_parameters is already tested
        comb_params = combine_simulation_parameters(results1.params,
                                                    results3.params)
        # Test if the parameters of the combined result object are correct
        self.assertEqual(comb_params, union.params)

        # xxxxxxxxxx Test the results of the combined object xxxxxxxxxxxxxx
        # Note that the 'elapsed_time' and 'num_skipped_reps' results are
        # always added by the SimulationRunner class.
        self.assertEqual(set(union.get_result_names()),
                         {'elapsed_time', 'lala', 'num_skipped_reps'})

        # The unpacked param list of union is (the order might be different)
        # [{'bias': 1.3, 'SNR': 0.0, 'extra': 2.2},
        #  {'bias': 1.3, 'SNR': 0.0, 'extra': 4.1},
        #  {'bias': 1.3, 'SNR': 0.0, 'extra': 5.7},
        #  {'bias': 1.3, 'SNR': 0.0, 'extra': 6.2},
        #  {'bias': 1.3, 'SNR': 5.0, 'extra': 2.2},
        #  {'bias': 1.3, 'SNR': 5.0, 'extra': 4.1},
        #  {'bias': 1.3, 'SNR': 5.0, 'extra': 5.7},
        #  {'bias': 1.3, 'SNR': 5.0, 'extra': 6.2},
        #  {'bias': 1.3, 'SNR': 10.0, 'extra': 2.2},
        #  {'bias': 1.3, 'SNR': 10.0, 'extra': 4.1},
        #  {'bias': 1.3, 'SNR': 10.0, 'extra': 5.7},
        #  {'bias': 1.3, 'SNR': 10.0, 'extra': 6.2},
        #  {'bias': 1.3, 'SNR': 15.0, 'extra': 2.2},
        #  {'bias': 1.3, 'SNR': 15.0, 'extra': 4.1},
        #  {'bias': 1.3, 'SNR': 15.0, 'extra': 5.7},
        #  {'bias': 1.3, 'SNR': 15.0, 'extra': 6.2},
        #  {'bias': 1.3, 'SNR': 20.0, 'extra': 2.2},
        #  {'bias': 1.3, 'SNR': 20.0, 'extra': 4.1},
        #  {'bias': 1.3, 'SNR': 20.0, 'extra': 5.7},
        #  {'bias': 1.3, 'SNR': 20.0, 'extra': 6.2}]
        #
        # The number of updates of each result in union should be equal to
        # 2, except for the results corresponding to the 'extra' parameter
        # value of 4.1, which is repeated in results1 and results3.

        all_indexes = set(range(union.params.get_num_unpacked_variations()))
        index_extra_4dot1 = set(union.params.get_pack_indexes({'extra': 4.1}))
        other_indexes = all_indexes - index_extra_4dot1

        for index in index_extra_4dot1:
            self.assertEqual(union['lala'][index].num_updates, 4)

        for index in other_indexes:
            self.assertEqual(union['lala'][index].num_updates, 2)

        # Calculate the expected lala results
        expected_lala_results = list(
            repeat(0, union.params.get_num_unpacked_variations()))
        for index, variation in enumerate(
                union.params.get_unpacked_params_list()):
            count = 0.
            snr = variation['SNR']
            extra = variation['extra']

            try:
                r1index = results1.params.get_pack_indexes({
                    'SNR': snr,
                    'extra': extra
                })[0]
                count += 1.
                expected_lala_results[index] \
                    += results1['lala'][r1index].get_result()
            except ValueError:
                pass

            try:
                r3index = results3.params.get_pack_indexes({
                    'SNR': snr,
                    'extra': extra
                })[0]
                count += 1.
                expected_lala_results[index] \
                    += results3['lala'][r3index].get_result()
            except ValueError:
                pass
            expected_lala_results[index] /= count

        union_lala_results = [v.get_result() for v in union['lala']]
        self.assertEqual(expected_lala_results, union_lala_results)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class ResultTestCase(unittest.TestCase):
    """
    Unit-tests for the Result class in the results module.
    """
    def setUp(self) -> None:
        """Called before each test."""
        self.result1 = Result("name", Result.SUMTYPE)
        self.result2 = Result("name2", Result.RATIOTYPE)
        self.result3 = Result("name3", Result.MISCTYPE)
        self.result4 = Result("name4", Result.CHOICETYPE, choice_num=6)

    def test_init(self) -> None:
        # Test if an exception is raised if we try to create a result type
        # with choice_num not being an integer
        with self.assertRaises(RuntimeError):
            # noinspection PyTypeChecker
            Result("name4", Result.CHOICETYPE, choice_num=6.6)

    def test_get_update_type(self) -> None:
        """
        Test the two properties, one to get the update type code and other to
        get the update type name. Note that both properties reflect the
        value of the same variable, the self._update_type variable.
        """
        self.assertEqual(self.result1.type_code, Result.SUMTYPE)
        self.assertEqual(self.result1.type_name, "SUMTYPE")

        self.assertEqual(self.result2.type_code, Result.RATIOTYPE)
        self.assertEqual(self.result2.type_name, "RATIOTYPE")

        self.assertEqual(self.result3.type_code, Result.MISCTYPE)
        self.assertEqual(self.result3.type_name, "MISCTYPE")

        self.assertEqual(self.result4.type_code, Result.CHOICETYPE)
        self.assertEqual(self.result4.type_name, "CHOICETYPE")

        # Test that the possible type codes are different
        self.assertEqual(
            4,
            len({
                Result.SUMTYPE, Result.RATIOTYPE, Result.MISCTYPE,
                Result.CHOICETYPE
            }))

    def test_update(self) -> None:
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

        # Test the update function of the CHOICETYPE.
        self.result4.update(0)
        self.result4.update(1)
        self.result4.update(0)
        self.result4.update(4)
        np.testing.assert_array_almost_equal(self.result4.get_result(),
                                             [.5, .25, 0, 0, .25, 0])
        self.result4.update(5)
        np.testing.assert_array_almost_equal(self.result4.get_result(),
                                             [.4, .2, 0, 0, .2, .2])

        # Test if an exception is raised when updating a Result of the
        # RATIOTYPE without specifying both the value and the total.
        with self.assertRaises(ValueError):
            self.result2.update(3)

        # Test if an exception is thrown when updating a result of some
        # unknown type
        # noinspection PyTypeChecker
        result_invalid = Result('invalid', 'invalid_type')
        with self.assertRaises(ValueError):
            result_invalid.update(10)

        # Test if an exception is raised for the CHOICETYPE if the updated
        # value is not a correct index.
        with self.assertRaises(IndexError):
            self.result4.update(8)
        with self.assertRaises(AssertionError):
            self.result4.update(3.4)

    def test_update_with_accumulate(self) -> None:
        result1 = Result('name', Result.SUMTYPE, accumulate_values=True)
        self.assertEqual(result1.accumulate_values_bool,
                         result1._accumulate_values_bool)
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

        result4 = Result('name',
                         Result.CHOICETYPE,
                         accumulate_values=True,
                         choice_num=5)
        result4.update(3)
        result4.update(1)
        result4.update(0)
        result4.update(3)
        result4.update(4)
        np.testing.assert_array_almost_equal(result4._value, [1, 1, 0, 2, 1])
        np.testing.assert_array_almost_equal(result4.get_result(),
                                             [.2, .2, 0, .4, .2])
        self.assertEqual(result4._total, 5)
        self.assertEqual(result4._value_list, [3, 1, 0, 3, 4])
        self.assertEqual(result4._total_list, [])

    def test_create(self) -> None:
        r1 = Result.create(name='nome1',
                           update_type=Result.SUMTYPE,
                           value=10.4,
                           accumulate_values=False)
        r1.update(3.5)
        self.assertEqual(r1.num_updates, 2)
        self.assertEqual(r1.get_result(), 13.9)
        self.assertEqual(r1._value_list, [])
        self.assertEqual(r1._total_list, [])

        r2 = Result.create(name='nome2',
                           update_type=Result.RATIOTYPE,
                           value=10.4,
                           total=20.2,
                           accumulate_values=True)
        r2.update(2, 3)
        r2.update(4.2, 5.3)
        self.assertEqual(r2.num_updates, 3)
        self.assertAlmostEqual(r2.get_result(), 0.58245614)
        self.assertEqual(r2._value_list, [10.4, 2, 4.2])
        self.assertEqual(r2._total_list, [20.2, 3, 5.3])

        r3 = Result.create(name='nome3',
                           update_type=Result.MISCTYPE,
                           value='valor',
                           accumulate_values=True)
        self.assertEqual(r3.num_updates, 1)
        self.assertEqual(r3.get_result(), 'valor')

        with self.assertRaises(RuntimeError):
            # Try to create a result of CHOICETYPE without specifying the total
            # raises an exception
            Result.create(name='nome4',
                          update_type=Result.CHOICETYPE,
                          value=3,
                          accumulate_values=True)

        r4 = Result.create(name='nome4',
                           update_type=Result.CHOICETYPE,
                           value=3,
                           total=6,
                           accumulate_values=True)
        r4.update(5)
        self.assertEqual(r4.num_updates, 2)
        np.testing.assert_equal(r4._value, [0, 0, 0, 1, 0, 1])

    def test_merge(self) -> None:
        # Test merge of Results of SUMTYPE
        self.result1.update(13)
        self.result1.update(30)
        result_sum_before1 = self.result1._result_sum
        result_sum_sqr_before1 = self.result1._result_squared_sum

        result1_other = Result.create("name", Result.SUMTYPE, 11)
        expected_result_sum1 = result_sum_before1 + result1_other._result_sum
        expected_result_sqr_sum1 = (result_sum_sqr_before1 +
                                    result1_other._result_squared_sum)

        self.result1.merge(result1_other)
        self.assertEqual(self.result1.name, "name")
        self.assertEqual(self.result1.get_result(), 54)
        self.assertEqual(self.result1.num_updates, 3)
        self.assertEqual(self.result1._result_sum, expected_result_sum1)
        self.assertEqual(self.result1._result_squared_sum,
                         expected_result_sqr_sum1)

        # Test merge of Results of RATIOTYPE
        self.result2.update(3, 10)
        self.result2.update(6, 7)
        self.result2.update(1, 15)
        result_sum_before2 = self.result2._result_sum
        result_sum_sqr_before2 = self.result2._result_squared_sum

        result2_other = Result.create("name2", Result.RATIOTYPE, 34, 50)
        result2_other.update(12, 18)
        expected_result_sum2 = result_sum_before2 + result2_other._result_sum
        expected_result_sqr_sum2 = (result_sum_sqr_before2 +
                                    result2_other._result_squared_sum)

        self.result2.merge(result2_other)
        self.assertEqual(self.result2.name, "name2")
        self.assertEqual(self.result2._value, 56)
        self.assertEqual(self.result2._total, 100)
        self.assertEqual(self.result2.get_result(), 0.56)
        self.assertEqual(self.result2.num_updates, 5)
        self.assertEqual(self.result2._result_sum, expected_result_sum2)
        self.assertEqual(self.result2._result_squared_sum,
                         expected_result_sqr_sum2)

        # Test merge of Results of MISCTYPE
        # There is no merge for misc type and an exception should be raised
        self.result3.update(0.4)
        result3_other = Result.create("name3", Result.MISCTYPE, 3)
        # with self.assertRaises(AssertionError):
        self.result3.merge(result3_other)
        self.assertEqual(self.result3.get_result(), 3)

        # Test merge of Results of CHOICETYPE
        result4_other = Result.create("name4", Result.CHOICETYPE, 3, 6)
        result4_other.update(2)
        self.result4.update(1)
        self.result4.update(0)
        self.result4.update(3)
        self.result4.merge(result4_other)
        self.assertEqual(self.result4.name, 'name4')
        np.testing.assert_array_almost_equal(self.result4._value,
                                             [1, 1, 1, 2, 0, 0])
        self.assertEqual(self.result4._total, 5)
        np.testing.assert_array_almost_equal(self.result4.get_result(),
                                             [.2, .2, .2, .4, 0, 0])
        self.assertEqual(self.result4.num_updates, 5)

        # Test merging results with different name or type
        result5 = Result.create("name5", Result.SUMTYPE, 3)
        with self.assertRaises(AssertionError):
            self.result1.merge(result5)

        result6 = Result.create("name", Result.RATIOTYPE, 3, 4)
        with self.assertRaises(AssertionError):
            self.result1.merge(result6)

    def test_merge_with_accumulate(self) -> None:
        result1 = Result('name', Result.SUMTYPE, accumulate_values=True)
        result1.update(13)
        result1.update(30)
        result1_other = Result.create("name",
                                      Result.SUMTYPE,
                                      11,
                                      accumulate_values=True)
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

        result2_other = Result.create("name2",
                                      Result.RATIOTYPE,
                                      34,
                                      50,
                                      accumulate_values=True)
        result2_other.update(12, 18)
        result2.merge(result2_other)
        self.assertEqual(result2._value, 56)
        self.assertEqual(result2._value_list, [3, 6, 1, 34, 12])
        self.assertEqual(result2._total, 100)
        self.assertEqual(result2._total_list, [10, 7, 15, 50, 18])
        self.assertEqual(result2.get_result(), 0.56)
        self.assertEqual(result2.num_updates, 5)

        # Test if an exception is raised if we try to merge with a Result
        # object which does not have the accumulate_values property set to
        # True
        result3 = Result('name2', Result.RATIOTYPE, accumulate_values=False)
        result3.update(2, 6)
        result3.update(7, 15)
        result3.update(2, 5)
        with self.assertRaises(AssertionError):
            result2.merge(result3)

        # Not that the opposite is possible. That is, a Result object
        # without accumulated values can be merged with a Result object
        # with accumulated values. In that case, the accumulated values of
        # the second object will be ignored.
        result3.merge(result2)
        self.assertEqual(result3._value, 67)
        self.assertEqual(result3._total, 126)
        self.assertEqual(result3._value_list, [])
        self.assertEqual(result3._total_list, [])

        # Test for the CHOICETYPE type
        result4 = Result.create('name4',
                                Result.CHOICETYPE,
                                2,
                                4,
                                accumulate_values=True)
        result4.update(0)
        result4.update(3)
        result4_other = Result.create('name4',
                                      Result.CHOICETYPE,
                                      0,
                                      4,
                                      accumulate_values=True)
        result4_other.update(3)
        result4.merge(result4_other)
        self.assertEqual(result4._value_list, [2, 0, 3, 0, 3])
        self.assertEqual(result4._total_list, [])
        np.testing.assert_array_almost_equal(result4.get_result(),
                                             np.array([2, 0, 1, 2]) / 5.)

    def test_get_result_mean_and_var(self) -> None:
        # Test for Result.SUMTYPE
        result1 = Result('name', Result.SUMTYPE, accumulate_values=True)
        result1.update(13)
        result1.update(30)
        result1_other = Result.create("name",
                                      Result.SUMTYPE,
                                      11,
                                      accumulate_values=True)
        result1_other.update(22)
        result1_other.update(4)
        result1.merge(result1_other)
        self.assertEqual(result1.get_result(), 80)

        expected_mean1 = np.array(result1._value_list, dtype=float).mean()
        self.assertAlmostEqual(result1.get_result_mean(), expected_mean1)

        expected_var1 = np.array(result1._value_list).var()
        self.assertAlmostEqual(result1.get_result_var(), expected_var1)

        # Test for Result.RATIOTYPE
        result2 = Result('name2', Result.RATIOTYPE, accumulate_values=True)
        result2.update(3, 10)
        result2.update(6, 7)
        result2.update(1, 15)

        result2_other = Result.create("name2",
                                      Result.RATIOTYPE,
                                      34,
                                      50,
                                      accumulate_values=True)
        result2_other.update(12, 18)
        result2.merge(result2_other)

        aux2 = (np.array(result2._value_list, dtype=float) /
                np.array(result2._total_list, dtype=float))
        expected_mean2 = aux2.mean()
        self.assertAlmostEqual(result2.get_result_mean(), expected_mean2)

        expected_var2 = aux2.var()
        self.assertAlmostEqual(result2.get_result_var(), expected_var2)

    def test_representation(self) -> None:
        self.assertEqual(self.result1.__repr__(),
                         "Result -> name: Nothing yet")
        self.assertEqual(self.result2.__repr__(),
                         "Result -> name2: 0/0 -> NaN")
        self.assertEqual(self.result3.__repr__(),
                         "Result -> name3: Nothing yet")
        self.assertEqual(self.result4.__repr__(),
                         "Result -> name4: Nothing yet")

        self.result1.update(10)
        self.result2.update(2, 4)
        self.result3.update(0.4)
        self.result4.update(3)
        self.result4.update(2)

        self.assertEqual(self.result1.__repr__(), "Result -> name: 10")
        self.assertEqual(self.result2.__repr__(),
                         "Result -> name2: 2/4 -> 0.5")
        self.assertEqual(self.result3.__repr__(), "Result -> name3: 0.4")
        self.assertEqual(self.result4.__repr__(),
                         "Result -> name4: [0.  0.  0.5 0.5 0.  0. ]")

    def test_equal_and_not_equal_operators(self) -> None:
        self.result1.update(10)
        self.result1.update(7)

        result1 = Result.create("name", Result.SUMTYPE, 7)
        result1.update(10)
        self.assertTrue(self.result1 == result1)
        self.assertFalse(self.result1 != result1)

        result1_other = Result.create("name", Result.SUMTYPE, 7)
        result1_other.update(9)
        self.assertFalse(self.result1 == result1_other)
        self.assertTrue(self.result1 != result1_other)

        result1._update_type_code = 1
        self.assertFalse(self.result1 == result1)
        self.assertTrue(self.result1 != result1)

        # Also test for the CHOICETYPE type, since it store the _value
        # member variable as a numpy array
        self.result4.update(3)
        self.result4.update(1)
        result4 = Result.create('name4', Result.CHOICETYPE, 1, 4)
        result4.update(3)
        # result4 only has 4 elements, while self.result6 has 6.
        self.assertTrue(self.result4 != result4)

        result4_other = Result.create('name4', Result.CHOICETYPE, 1, 6)
        result4_other.update(3)
        self.assertTrue(self.result4 == result4_other)
        result4_other.update(2)
        self.result4.update(1)
        self.assertFalse(self.result4 == result4_other)

        # Test comparison with something of a different class
        self.assertFalse(self.result4 == 10.4)

    def test_calc_confidence_interval(self) -> None:
        # Test if an exceptions is raised for a Result object of the
        # MISCTYPE update type.
        with self.assertRaises(RuntimeError):
            # A result object of the MISCTYPE type cannot use the
            # calc_confidence_interval method, since this update ignores
            # the accumulate_values option and never accumulates any value.
            self.result3.get_confidence_interval()

        # # Test if an exception is raised if the accumulate_values option
        # # was not set to True
        # with self.assertRaises(RuntimeError):
        #     self.result1.get_confidence_interval()

        result = Result('name', Result.RATIOTYPE, accumulate_values=True)
        # # Test if an exception is raised if there are not stored values yet
        # with self.assertRaises(RuntimeError):
        #     result.get_confidence_interval()

        # Now lets finally store some values.
        result.update(10, 30)
        result.update(3, 24)
        result.update(15, 42)
        result.update(5, 7)

        # Calculate the expected confidence interval
        A = (np.array(result._value_list, dtype=float) /
             np.array(result._total_list, dtype=float))
        expected_confidence_interval = calc_confidence_interval(A.mean(),
                                                                A.std(),
                                                                A.size,
                                                                P=95)
        confidence_interval = result.get_confidence_interval(P=95)
        np.testing.assert_array_almost_equal(expected_confidence_interval,
                                             confidence_interval)

    def test_to_dict_from_dict(self) -> None:
        self.result1.update(13)
        self.result1.update(30)

        self.result2.update(3, 10)
        self.result2.update(6, 7)
        self.result2.update(1, 15)

        self.result3.update(3)
        self.result3.update("some string")
        self.result3.update(2)

        self.result4.update(3)
        self.result4.update(1)
        self.result4.update(0)
        self.result4.update(3)
        self.result4.update(4)

        # xxxxxxxxxx Test converting to a dictionary xxxxxxxxxxxxxxxxxxxxxx
        result1_dict = self.result1.to_dict()
        result2_dict = self.result2.to_dict()
        result3_dict = self.result3.to_dict()
        result4_dict = self.result4.to_dict()

        self.assertIsInstance(result1_dict, dict)
        self.assertIsInstance(result2_dict, dict)
        self.assertIsInstance(result3_dict, dict)
        self.assertIsInstance(result4_dict, dict)

        # We will not test individual dictionary keys here, since we will
        # test later if we can recover the actual Result object from the
        # dictionary and that should be enough.
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test converting from a dictionary xxxxxxxxxxxxxxxxxxxx
        result1 = Result.from_dict(result1_dict)
        result2 = Result.from_dict(result2_dict)
        result3 = Result.from_dict(result3_dict)
        result4 = Result.from_dict(result4_dict)

        self.assertEqual(self.result1, result1)
        self.assertEqual(self.result2, result2)
        self.assertEqual(self.result3, result3)
        self.assertEqual(self.result4, result4)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_to_json_and_from_json(self) -> None:
        self.result1.update(13)
        self.result1.update(30)

        self.result2.update(3, 10)
        self.result2.update(6, 7)
        self.result2.update(1, 15)

        self.result3.update(3)
        self.result3.update("some string")
        self.result3.update(2)

        self.result4.update(3)
        self.result4.update(1)
        self.result4.update(0)
        self.result4.update(3)
        self.result4.update(4)

        # TODO: finish implementation
        # xxxxxxxxxx Test converting to and from a json xxxxxxxxxxxxxxxxxxx
        # First test converting to json
        result1_json = self.result1.to_json()
        result2_json = self.result2.to_json()
        result3_json = self.result3.to_json()
        result4_json = self.result4.to_json()

        # This will raise an exception if encoded_params is not valid json
        _ = json.loads(result1_json)
        _ = json.loads(result2_json)
        _ = json.loads(result3_json)
        _ = json.loads(result4_json)

        # Now test converting from json
        result1 = Result.from_json(result1_json)
        result2 = Result.from_json(result2_json)
        result3 = Result.from_json(result3_json)
        result4 = Result.from_json(result4_json)

        self.assertEqual(self.result1, result1)
        self.assertEqual(self.result2, result2)
        self.assertEqual(self.result3, result3)
        self.assertEqual(self.result4, result4)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# noinspection PyMethodMayBeStatic
class SimulationResultsTestCase(unittest.TestCase):
    """
    Unit-tests for the SimulationResults class in the results module.
    """
    def setUp(self) -> None:
        # First SimulationResults object
        self.simresults = SimulationResults()
        self.simresults.add_new_result("lala", Result.SUMTYPE, 13)
        result2 = Result("lele", Result.RATIOTYPE)
        result2.update(3, 10)
        result2.update(8, 10)
        self.simresults.add_result(result2)
        result4 = Result.create('lulu', Result.CHOICETYPE, 3, 6)
        result4.update(1)
        self.simresults.add_result(result4)

        # Second SimulationResults object
        self.other_simresults = SimulationResults()
        result1_other = Result.create('lala', Result.SUMTYPE, 30)
        result2_other = Result.create('lele', Result.RATIOTYPE, 4, 10)
        result3 = Result.create('lili', Result.MISCTYPE, "a string")
        self.other_simresults.add_result(result1_other)
        self.other_simresults.add_result(result2_other)
        self.other_simresults.add_result(result3)
        result4_other = Result.create('lulu', Result.CHOICETYPE, 0, 6)
        result4_other.update(5)
        result4_other.update(5)
        self.other_simresults.add_result(result4_other)

    def test_params_property(self) -> None:
        params = SimulationParameters()
        params.add('number', 10)
        params.add('name', 'lala')

        # Try to set the parameters to an invalid object
        with self.assertRaises(ValueError):
            # noinspection PyTypeChecker
            self.simresults.set_parameters(10)

        # Set the simulation parameters
        self.simresults.set_parameters(params)

        # test the get property
        params2 = self.simresults.params
        self.assertEqual(len(params), len(params2))
        self.assertEqual(params['number'], params2['number'])
        self.assertEqual(params['name'], params2['name'])

    def test_get_result_names(self) -> None:
        # The output of the get_result_names is a list of names. We
        # transform it into a set in this test only to make the order of
        # the names unimportant.
        expected_output = {'lala', 'lele', 'lulu'}
        self.assertEqual(set(self.simresults.get_result_names()),
                         expected_output)
        # Test also the representation of the SimulationResults object
        self.assertEqual(self.simresults.__repr__(),
                         """SimulationResults: ['lala', 'lele', 'lulu']""")

    def test_add_result(self) -> None:
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
                         {"lala", "lele", "lili", "lulu"})
        self.assertEqual(self.simresults['lili'][0].get_result(), "a string")

    def test_append_result(self) -> None:
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

    def test_append_all_results(self) -> None:
        self.simresults.append_all_results(self.other_simresults)
        # Note that self.simresults only has the 'lala' and 'lele' results.
        # After we append the results in self.other_simresults
        # self.simresults should have also the 'lili' result, but with only
        # a single result for 'lili' and two results for both 'lala' and
        # 'lele'..
        self.assertEqual(set(self.simresults.get_result_names()),
                         {"lulu", "lala", "lele", "lili"})
        self.assertEqual(len(self.simresults['lala']), 2)
        self.assertEqual(len(self.simresults['lele']), 2)
        self.assertEqual(len(self.simresults['lili']), 1)
        self.assertEqual(len(self.simresults['lulu']), 2)

    def test_merge_all_results(self) -> None:
        # Note that even though there is a 'lili' result in
        # self.other_simresults, only 'lala' and 'lele' will be
        # merged. Also, self.other_simresults must have all the results in
        # self.simresults otherwise there will be a KeyError.
        self.simresults.merge_all_results(self.other_simresults)
        self.assertEqual(self.simresults['lala'][-1].get_result(), 43)
        self.assertEqual(self.simresults['lele'][-1].get_result(),
                         (11. + 4.) / (20. + 10.))

        # One update from the 'lala' result in self.simresults and other
        # from the 'lala' result in self.other_simresults
        self.assertEqual(self.simresults['lala'][0].num_updates, 2)

        # Two updates from the 'lele' result in self.simresults and other
        # from the 'lele' result in self.other_simresults
        self.assertEqual(self.simresults['lele'][0].num_updates, 3)

        # Test if an empty SimulationResults object can merge with another
        # SimulationResults object.
        emptyresults = SimulationResults()
        emptyresults.merge_all_results(self.simresults)
        self.assertEqual(set(emptyresults.get_result_names()),
                         {'lala', 'lele', 'lulu'})

        # xxxxx Test the merge with the num_skipped_reps result xxxxxxxxxxx
        simresults1 = SimulationResults()
        simresults2 = SimulationResults()
        simresults1.add_new_result('name1', Result.SUMTYPE, 3)
        simresults1.add_new_result('num_skipped_reps', Result.SUMTYPE, 3)
        simresults2.add_new_result('name1', Result.SUMTYPE, 2)
        simresults1.merge_all_results(simresults2)
        self.assertEqual(set(simresults1.get_result_names()),
                         {'name1', 'num_skipped_reps'})
        self.assertEqual(set(simresults2.get_result_names()), {'name1'})

        simresults3 = SimulationResults()
        simresults3.add_new_result('name1', Result.SUMTYPE, 4)
        simresults3.merge_all_results(simresults1)
        self.assertEqual(set(simresults3.get_result_names()),
                         {'name1', 'num_skipped_reps'})
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_equal_and_not_equal_operators(self) -> None:
        elapsed_time_result = Result.create('elapsed_time', Result.SUMTYPE, 30)
        self.simresults.add_result(elapsed_time_result)

        simresults = SimulationResults()
        lala_result = Result('lala', Result.SUMTYPE)
        lele_result = Result('lele', Result.RATIOTYPE)
        lulu_result = Result('lulu', Result.CHOICETYPE, choice_num=6)
        lala_result.update(13)
        lele_result.update(8, 10)
        lele_result.update(3, 10)
        lulu_result.update(3)
        lulu_result.update(1)
        elapsed_time_result2 = Result.create('elapsed_time', Result.SUMTYPE,
                                             20)
        simresults.add_result(lala_result)
        simresults.add_result(lele_result)
        simresults.add_result(lulu_result)
        simresults.add_result(elapsed_time_result2)

        # Note that the elapsed_time result is different, but it is not
        # accounted
        self.assertTrue(self.simresults == simresults)
        self.assertFalse(self.simresults != simresults)

        # Let's change the parameters in the SimulationResults objects to
        # see if it impacts equality
        simresults.params.add('value', 10)
        self.assertFalse(self.simresults == simresults)
        self.assertTrue(self.simresults != simresults)
        self.simresults.params.add('value', 10)
        self.assertTrue(self.simresults == simresults)
        self.assertFalse(self.simresults != simresults)

        # Let's change one Result in one of them to see if it impacts
        # equality
        simresults['lala'][0].update(5)
        self.assertFalse(self.simresults == simresults)
        self.assertTrue(self.simresults != simresults)
        self.simresults['lala'][0].update(5)
        self.assertTrue(self.simresults == simresults)
        self.assertFalse(self.simresults != simresults)

        # Lets add a new result to one of them which is not in the other
        lili_result = Result('lili', Result.SUMTYPE)
        simresults.add_result(lili_result)
        self.assertFalse(self.simresults == simresults)
        self.assertTrue(self.simresults != simresults)

        # Let's test with something of a different class
        # noinspection PyTypeChecker
        self.assertFalse(self.simresults == 20)

    def test_get_result_values_list(self) -> None:
        self.simresults.append_all_results(self.other_simresults)

        self.assertEqual(self.simresults.get_result_values_list('lala'),
                         [13, 30])
        self.assertEqual(self.simresults.get_result_values_list('lele'),
                         [0.55, 0.4])

        lulu_list = self.simresults.get_result_values_list('lulu')

        np.testing.assert_array_almost_equal(
            lulu_list[0], np.array([0., 0.5, 0., 0.5, 0., 0.]))
        np.testing.assert_array_almost_equal(
            lulu_list[1], np.array([0.33333333, 0., 0., 0., 0., 0.66666667]))

        # There is only one result for 'lili', which comes from
        # self.other_simresults.
        self.assertEqual(self.simresults.get_result_values_list('lili'),
                         ['a string'])

    def test_get_result_values_confidence_intervals(self) -> None:
        simresults = SimulationResults()
        simresults.params.add('P', [1, 2])
        simresults.params.set_unpack_parameter('P')
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
        list_of_confidence_intervals \
            = simresults.get_result_values_confidence_intervals('name', P)

        # Calculates the expected list of confidence intervals
        expected_list_of_confidence_intervals \
            = [i.get_confidence_interval(P) for i in simresults['name']]

        # Test of they are equal
        for a, b in zip(list_of_confidence_intervals,
                        expected_list_of_confidence_intervals):
            np.testing.assert_array_almost_equal(a, b)

        # xxxxxxxxxx Test for a subset of the parameters xxxxxxxxxxxxxxxxxx
        c1 = simresults.get_result_values_confidence_intervals(
            'name', P, fixed_params={'P': 1.0})
        c2 = simresults.get_result_values_confidence_intervals(
            'name', P, fixed_params={'P': 2.0})
        np.testing.assert_array_almost_equal(
            c1[0], expected_list_of_confidence_intervals[0])
        np.testing.assert_array_almost_equal(
            c2[0], expected_list_of_confidence_intervals[1])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_to_dict_from_dict(self) -> None:
        # xxxxxxxxxx Test converting to a dictionary xxxxxxxxxxxxxxxxxxxxxx
        simresults_dict = self.simresults._to_dict()
        self.assertIsInstance(simresults_dict, dict)

        # We will not test all individual dictionary keys here, since we
        # will test later if we can recover the actual SimulationResults
        # object from the dictionary and that should be enough.

        # For now we only test in the individual Result objects in
        # SimulationResults were converted to their own dictionary
        # representations

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test converting from a dictionary xxxxxxxxxxxxxxxxxxxx
        simresults = SimulationResults._from_dict(simresults_dict)
        self.assertEqual(self.simresults, simresults)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_to_json_and_from_json(self) -> None:
        # xxxxxxxxxx Test converting to and from a json xxxxxxxxxxxxxxxxxxx
        # First test converting to json
        simresults_json = self.simresults.to_json()
        # This will raise an exception if encoded_params is not valid json
        _ = json.loads(simresults_json)

        # Now test converting from json
        simresults = SimulationResults.from_json(simresults_json)

        self.assertEqual(self.simresults, simresults)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_save_to_and_load_from_file(self) -> None:
        base_filename = 'results_({age})_({temperature})_({factor})'
        base_pickle_filename = "{0}.pickle".format(base_filename)
        base_json_filename = "{0}.json".format(base_filename)

        # Set simulation parameters
        self.simresults.params.add('factor', 0.5)
        self.simresults.params.add('temperature', 50.5)
        self.simresults.params.add('age', 3)

        # xxxxx Add a result with accumulate set to True xxxxxxxxxxxxxxxxxx
        result_acu = Result('name', Result.SUMTYPE, accumulate_values=True)
        result_acu.update(13)
        result_acu.update(30)
        self.simresults.add_result(result_acu)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Save to the file and get the actual filename used to save the file
        # Note that if we pass the base filename without extension a 'pickle'
        # extension will be added
        filename_pickle = self.simresults.save_to_file(base_filename)
        filename_json = self.simresults.save_to_file(base_json_filename)

        # Load from the file
        simresults2 = SimulationResults.load_from_file(filename_pickle)
        simresults3 = SimulationResults.load_from_file(filename_json)

        # xxxxxxxxxx Test file saved and loaded from pickle file xxxxxxxxxx
        self.assertEqual(simresults2.original_filename, base_pickle_filename)
        self.assertEqual(len(self.simresults), len(simresults2))
        self.assertEqual(set(self.simresults.get_result_names()),
                         set(simresults2.get_result_names()))

        self.assertEqual(self.simresults['lala'][0].type_code,
                         simresults2['lala'][0].type_code)
        self.assertEqual(self.simresults['lele'][0].type_code,
                         simresults2['lele'][0].type_code)
        self.assertEqual(self.simresults['lulu'][0].type_code,
                         simresults2['lulu'][0].type_code)

        self.assertAlmostEqual(
            self.simresults['lala'][0].get_result(),
            simresults2['lala'][0].get_result(),
        )
        self.assertAlmostEqual(
            self.simresults['lele'][0].get_result(),
            simresults2['lele'][0].get_result(),
        )
        np.testing.assert_array_almost_equal(
            self.simresults['lulu'][0].get_result(),
            simresults2['lulu'][0].get_result(),
        )
        self.assertAlmostEqual(
            self.simresults['name'][0].get_result(),
            simresults2['name'][0].get_result(),
        )

        # test if the parameters were also saved
        self.assertEqual(self.simresults.params['age'],
                         simresults2.params['age'])
        self.assertAlmostEqual(self.simresults.params['temperature'],
                               simresults2.params['temperature'])
        self.assertAlmostEqual(self.simresults.params['factor'],
                               simresults2.params['factor'])

        # Delete the where the results were saved
        delete_file_if_possible(filename_pickle)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test file saved and loaded from json file xxxxxxxxxxxx
        # This will raise an exception if encoded_params is not valid json
        with open(filename_json, 'r') as fid:
            _ = json.load(fid)

        self.assertEqual(simresults3.original_filename, base_json_filename)
        self.assertEqual(len(self.simresults), len(simresults3))
        self.assertEqual(set(self.simresults.get_result_names()),
                         set(simresults3.get_result_names()))

        self.assertEqual(self.simresults['lala'][0].type_code,
                         simresults3['lala'][0].type_code)
        self.assertEqual(self.simresults['lele'][0].type_code,
                         simresults3['lele'][0].type_code)
        self.assertEqual(self.simresults['lulu'][0].type_code,
                         simresults3['lulu'][0].type_code)

        self.assertAlmostEqual(
            self.simresults['lala'][0].get_result(),
            simresults3['lala'][0].get_result(),
        )
        self.assertAlmostEqual(
            self.simresults['lele'][0].get_result(),
            simresults3['lele'][0].get_result(),
        )
        np.testing.assert_array_almost_equal(
            self.simresults['lulu'][0].get_result(),
            simresults3['lulu'][0].get_result(),
        )
        self.assertAlmostEqual(
            self.simresults['name'][0].get_result(),
            simresults3['name'][0].get_result(),
        )

        # test if the parameters were also saved
        self.assertEqual(self.simresults.params['age'],
                         simresults3.params['age'])
        self.assertAlmostEqual(self.simresults.params['temperature'],
                               simresults3.params['temperature'])
        self.assertAlmostEqual(self.simresults.params['factor'],
                               simresults3.params['factor'])

        # Delete the where the results were saved
        delete_file_if_possible(filename_json)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_to_dataframe(self) -> None:
        # If the pandas package is not installed, we will skip testing this
        # method
        if not _PANDAS_AVAILABLE:  # pragma: nocover
            self.skipTest("test_to_dataframe - Pandas is not installed")

        # Create some dummy parameters (including two parameters set to be
        # unpacked)
        params = SimulationParameters()
        params.add("extra", 2.3)
        params.add("SNR", np.array([0, 3, 6, 9]))
        params.add("bias", [1.2, 1.6])
        params.set_unpack_parameter('SNR')
        params.set_unpack_parameter('bias')
        params.add("Name", "Some string")

        sim_results = SimulationResults()

        # Try converting an empty SimulationResults
        df = sim_results.to_dataframe()
        self.assertTrue(isinstance(df, DataFrame))
        self.assertTrue(df.empty)

        for p in params.get_unpacked_params_list():
            extra = p['extra']
            SNR = p['SNR']
            bias = p['bias']
            sim_results.append_result(
                Result.create('res1', Result.SUMTYPE, extra * SNR + bias))
            sim_results.append_result(
                Result.create('res2', Result.SUMTYPE, bias * SNR + extra))
        sim_results.set_parameters(params)

        # When a simulation results is obtained from the simulation runner it
        # will also have a "runner_reps" attribute that should be converted to the dtaframe
        runned_reps = [2] * params.get_num_unpacked_variations()
        sim_results.runned_reps = runned_reps

        # Now lets convert this SimulationResults object to a pandas
        # DataFrame
        df = sim_results.to_dataframe()

        # The DataFrame should have the same number of rows as the number
        # of parameters variations
        self.assertEqual(len(df), params.get_num_unpacked_variations())

        # Test the parameters
        expected_bias = [a['bias'] for a in params.get_unpacked_params_list()]
        expected_SNR = [a['SNR'] for a in params.get_unpacked_params_list()]
        expected_name = [a['Name'] for a in params.get_unpacked_params_list()]
        expected_extra \
            = [a['extra'] for a in params.get_unpacked_params_list()]
        np.testing.assert_array_equal(df.bias, expected_bias)
        np.testing.assert_array_equal(df.SNR, expected_SNR)
        np.testing.assert_array_equal(df.Name, expected_name)
        np.testing.assert_array_equal(df.extra, expected_extra)
        np.testing.assert_array_equal(df.runned_reps, runned_reps)

        # Test the results
        for index, p in enumerate(params.get_unpacked_params_list()):
            extra = p['extra']
            SNR = p['SNR']
            bias = p['bias']
            expected_res1 = extra * SNR + bias
            expected_res2 = bias * SNR + extra
            self.assertAlmostEqual(expected_res1, df.res1[index])
            self.assertAlmostEqual(expected_res2, df.res2[index])


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Runner Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# This function is used in test methods for the SimulationRunner class
def _delete_pickle_files():
    """
    Delete all files with a '.pickle' extension in the current folder.
    """
    files = glob.glob('*.pickle')
    for f in files:
        os.remove(f)

    try:
        # Remove files in the partial_results folder
        files = glob.glob('./partial_results/*.pickle')
        for f in files:
            os.remove(f)
        os.rmdir('./partial_results')
    except OSError:  # pragma: nocover
        pass


def _delete_progressbar_output_files():  # pragma: no cover
    """
    Delete all the files with *results*.txt names in the current folder.
    """
    progressbar_files = glob.glob('*results*.txt')
    for f in progressbar_files:  # pragma: no cover
        try:
            os.remove(f)
        except OSError:  # pragma: nocover
            pass


# Define a _DummyRunner class for the testing the simulate and
# simulate_in_parallel methods in the SimulationRunner class.
class _DummyRunner(SimulationRunner):
    def __init__(self):
        super().__init__(read_command_line_args=False)
        # Set the progress bar message to None to avoid print the
        # progressbar in these testes.
        self.rep_max = 2
        self.update_progress_function_style = None
        # Now we add a dummy parameter to our runner object
        self.params.add('SNR', np.array([0., 5., 10., 15., 20.]))
        self.params.add('bias', 1.3)
        self.params.add('extra', np.array([2.2, 4.1]))
        self.params.set_unpack_parameter('SNR')
        self.params.set_unpack_parameter('extra')
        self.delete_partial_results_bool = True

    @staticmethod
    def calc_result(SNR, bias, extra):
        value = 1.2 * SNR + bias + extra
        return value

    def _run_simulation(self, current_params):
        SNR = current_params['SNR']
        bias = current_params['bias']
        extra = current_params['extra']
        sim_results = SimulationResults()

        value = self.calc_result(SNR, bias, extra)

        # The correct result will be SNR * 1.2 + 1.3 + extra
        sim_results.add_new_result('lala', Result.RATIOTYPE, value, 1)
        return sim_results


class _DummyRunnerRandom(SimulationRunner):  # pragma: no cover
    def __init__(self):
        super().__init__(read_command_line_args=False)
        # Set the progress bar message to None to avoid print the
        # progressbar in these testes.
        self.rep_max = 2
        self.update_progress_function_style = None
        # Now we add a dummy parameter to our runner object
        self.params.add('P', np.array([2., 2., 2., 2., 2.]))
        self.params.set_unpack_parameter('P')
        self.delete_partial_results_bool = True

        self.rs = np.random.RandomState()
        self.rs2 = np.random.RandomState()

    def _on_simulate_current_params_start(self, current_params):
        # Ideally we should re-seed any random number sources stored in a
        # SimulationRunner object. However, for testing purposes we will
        # only re-seed self.rs2 here.
        self.rs2.seed()

    def _run_simulation(self, current_params):
        P = current_params['P']
        sim_results = SimulationResults()

        # This will have a different value for each simulation parameters
        random_value = np.random.random_sample()
        value = 1.2 * P + random_value
        sim_results.add_new_result('result1', Result.RATIOTYPE, value, 1)

        random_value2 = self.rs.rand()
        value2 = 1.2 * P + random_value2
        sim_results.add_new_result('result2', Result.RATIOTYPE, value2, 1)

        # self.rs2.seed()
        random_value3 = self.rs2.rand()
        value3 = 1.2 * P + random_value3
        sim_results.add_new_result('result3', Result.RATIOTYPE, value3, 1)

        return sim_results


# Define a _DummyRunnerWithSkip class for the testing the simulate when a
# SkipThisOne exception is raised in the implemented _run_simulation
# method.
class _DummyRunnerWithSkip(SimulationRunner):
    def __init__(self):
        super().__init__(read_command_line_args=False)
        # This is used only for testing purposes. You would not have this
        # attribute in derives classes from SimulationRunner
        self._num_skipped = -1

        self.rep_max = 5
        self.update_progress_function_style = None
        # Now we add a dummy parameter to our runner object
        self.params.add('SNR', np.array([0., 5., 10., 15., 20.]))
        self.params.add('bias', 1.3)
        self.params.add('extra', np.array([2.2, 4.1]))
        self.params.set_unpack_parameter('SNR')
        self.params.set_unpack_parameter('extra')
        self.delete_partial_results_bool = True

    @staticmethod
    def calc_result(SNR, bias, extra):
        value = 1.2 * SNR + bias + extra
        return value

    def _run_simulation(self, current_params):
        # In practice, you would raise a SkipThisOne exception when some
        # problem occurs, such as a bad channel realization, singular
        # matrix in some algorithm, etc. Here will will raise this
        # exception in the first two times the _run_simulation method is
        # called for testing purposes.
        if self._num_skipped < 2:
            self._num_skipped += 1
            if self._num_skipped > 0:
                raise SkipThisOne('Skipping this one')
            else:
                pass  # pragma: nocover

        SNR = current_params['SNR']
        bias = current_params['bias']
        extra = current_params['extra']
        sim_results = SimulationResults()

        value = self.calc_result(SNR, bias, extra)

        # The correct result will be SNR * 1.2 + 1.3 + extra
        sim_results.add_new_result('lala', Result.RATIOTYPE, value, 1)
        return sim_results


# noinspection PyUnboundLocalVariable
class SimulationRunnerTestCase(unittest.TestCase):
    """
    Unit-tests for the SimulationRunner class in the runner module.
    """
    def setUp(self) -> None:
        self.runner = SimulationRunner(read_command_line_args=False)

    # Test if the SimulationRunner sets a few default attributes in its init
    # method.
    def test_default_values(self) -> None:
        # Note that we are also testing the elapsed_time and runned_reps
        # properties, which should just return these attributes.
        self.assertEqual(self.runner.rep_max, 1)
        self.assertEqual(self.runner.elapsed_time, "0.00s")
        self.assertEqual(self.runner.runned_reps, [])
        self.assertTrue(isinstance(self.runner.params, SimulationParameters))
        self.assertTrue(isinstance(self.runner.results, SimulationResults))
        self.assertEqual(self.runner.progressbar_message, "Progress")
        self.assertEqual(self.runner.update_progress_function_style, 'text2')

    def test_not_implemented_methods(self) -> None:
        # self.assertRaises(NotImplementedError,
        #                   self.S1._get_vertex_positions)
        with self.assertRaises(NotImplementedError):
            # noinspection PyTypeChecker
            self.runner._run_simulation(None)

    def test_keep_going(self) -> None:
        # the _keep_going method in the SimulationRunner class should
        # return True
        # noinspection PyTypeChecker
        self.assertTrue(self.runner._keep_going(None, None, None))

    def test_set_results_filename(self) -> None:
        dummyrunner = _DummyRunner()
        dummyrunner.set_results_filename()
        self.assertIsNone(dummyrunner.results_filename)

        dummyrunner.set_results_filename("some_name_{bias}_{extra}")

        # Note that this line would be unnecessary if we had call the
        # 'simulate' method of dummyrunner.
        dummyrunner.results.set_parameters(dummyrunner.params)

        self.assertEqual("some_name_1.3_[2.2,4.1].pickle",
                         dummyrunner.results_filename)

        # xxxxxxxxxx Test setting file name with extension xxxxxxxxxxxxxxxx
        dummyrunner2 = _DummyRunner()
        dummyrunner2.set_results_filename()
        self.assertIsNone(dummyrunner2.results_filename)

        dummyrunner2.set_results_filename("some_name_{bias}_{extra}.pickle")

        # Note that this line would be unnecessary if we had call the
        # 'simulate' method of dummyrunner.
        dummyrunner2.results.set_parameters(dummyrunner.params)

        self.assertEqual("some_name_1.3_[2.2,4.1].pickle",
                         dummyrunner2.results_filename)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def test_simulate(self) -> None:
        self.runner.progressbar_message = "Simulating"
        self.assertEqual(self.runner.progressbar_message, "Simulating")

        self.assertEqual(self.runner.partial_results_folder, 'partial_results')
        self.runner.partial_results_folder = "partial_test_results"

        # from tests.simulations_package_test import _DummyRunner
        dummyrunner = _DummyRunner()

        # xxxxxxxxxx Set the name of the results file xxxxxxxxxxxxxxxxxxxxx
        filename = 'dummyrunner_results_bias_{bias}'
        dummyrunner.set_results_filename(filename)

        # We will set the progress_output_type to 'file'
        # This will make the progressbar print to a file, instead of stdout
        dummyrunner.progress_output_type = 'file'  # Default is 'screen'
        self.assertEqual(dummyrunner.progress_output_type, 'file')

        # Note that progress_output_type can only be 'screen' of 'file' and any
        # other value raises an exception without changing the value
        with self.assertRaises(RuntimeError):
            dummyrunner.progress_output_type = 'invalid_value'  # type: ignore
        self.assertEqual(dummyrunner.progress_output_type, 'file')
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Perform a simulation of a single params variation xxxx
        delete_file_if_possible(
            "partial_results/dummyrunner_results_bias_1.3_unpack_03.pickle")
        partial_file = Path(
            "partial_results/dummyrunner_results_bias_1.3_unpack_03.pickle")
        self.assertFalse(partial_file.exists())
        dummyrunner.simulate(param_variation_index=3)
        self.assertTrue(partial_file.exists())
        delete_file_if_possible(
            "partial_results/dummyrunner_results_bias_1.3_unpack_03.pickle")

        # The results in the SImulationRunner is still empty, since only partial
        # results were saved to a file
        self.assertEqual(len(dummyrunner.results), 0)
        dummyrunner.clear()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # The results will be the SNR values multiplied by 1.2. plus the
        # bias parameter
        self.assertEqual(dummyrunner.runned_reps, [])
        dummyrunner.simulate()
        self.assertGreater(dummyrunner._simulation_tracking._elapsed_time, 0.0)
        self.assertEqual(dummyrunner.runned_reps, [dummyrunner.rep_max] *
                         dummyrunner.params.get_num_unpacked_variations())
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Perform the tests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        results_extra_1 = dummyrunner.results.get_result_values_list(
            'lala', {'extra': 2.2})
        expected_results_extra_1 = [3.5, 9.5, 15.5, 21.5, 27.5]
        np.testing.assert_array_almost_equal(results_extra_1,
                                             expected_results_extra_1)

        results_extra_2 = dummyrunner.results.get_result_values_list(
            'lala', {'extra': 4.1})
        expected_results_extra_2 = [5.4, 11.4, 17.4, 23.4, 29.4]
        np.testing.assert_array_almost_equal(results_extra_2,
                                             expected_results_extra_2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test if the results were saved correctly xxxxxxxxxxxxx
        sim_results = SimulationResults.load_from_file(
            dummyrunner.results_filename)
        self.assertEqual(sim_results, dummyrunner.results)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Repeat the test xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now we do not set the results filename
        dummyrunner2 = _DummyRunner()
        dummyrunner2.simulate()
        self.assertGreater(dummyrunner2._simulation_tracking._elapsed_time,
                           0.0)

        self.assertEqual(dummyrunner.results, dummyrunner2.results)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Repeat the test with wrong partial results xxxxxxxxxxx
        # First we run a usual simulation and keep the partial results
        dummyrunner3 = _DummyRunner()
        dummyrunner3.set_results_filename('dummyrunner3_results')
        self.assertEqual(dummyrunner3.delete_partial_results_bool, True)
        dummyrunner3.delete_partial_results_bool = False
        self.assertEqual(dummyrunner3.delete_partial_results_bool, False)
        dummyrunner3.simulate()
        self.assertGreater(dummyrunner3._simulation_tracking._elapsed_time,
                           0.0)

        # Now we change the bias parameter
        dummyrunner3.params.add('bias', 1.5)

        # If we run a simulation with different parameters it will try to
        # load the partial results with wrong parameters and an exception
        # should be raised
        with self.assertRaises(ValueError):
            dummyrunner3.simulate()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Repeat the test loading the partial results xxxxxxxxxx
        dummyrunner4 = _DummyRunner()
        dummyrunner4.set_results_filename('dummyrunner3_results')
        dummyrunner4.delete_partial_results_bool = True
        dummyrunner4.simulate()
        self.assertGreater(dummyrunner4._simulation_tracking._elapsed_time,
                           0.0)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Delete the pickle files in the same folder
        _delete_pickle_files()

    def test_simulate_with_param_variation_index(self) -> None:
        # Test the "simulate" method when the param_variation_index
        # argument is specified.
        # from tests.simulations_package_test import _DummyRunner
        dummyrunner = _DummyRunner()

        # Try to simulate for a given param_variation_index before setting
        # the filename. This should raise an exception.
        with self.assertRaises(RuntimeError):
            dummyrunner.simulate(3)

        # xxxxxxxxxx Set the name of the results file xxxxxxxxxxxxxxxxxxxxx
        filename = 'dummyrunner_results_bias_{bias}'
        dummyrunner.set_results_filename(filename)
        # This will make the progressbar print to a file, instead of stdout
        dummyrunner.progress_output_type = 'file'  # Default is 'screen'

        # Now we perform the simulation
        dummyrunner.simulate(param_variation_index=4)
        # Note that `dummyrunner.runned_reps` is not a list, but an integer
        self.assertEqual(dummyrunner.runned_reps, 2)
        pr = SimulationResults.load_from_file(
            'partial_results/dummyrunner_results_bias_1.3_unpack_04.pickle')

        #  Get the parameters from the loded result
        bias = pr.params['bias']
        snr = pr.params['SNR']
        extra = pr.params['extra']

        # Calculate the expected value
        expected_value = _DummyRunner.calc_result(snr, bias, extra)

        self.assertEqual(len(pr['lala']), 1)
        self.assertAlmostEqual(pr['lala'][0].get_result(), expected_value)

        _delete_pickle_files()

    # This test method is normally skipped, unless you have started an
    # IPython cluster with a "tests" profile so that you have at least one
    # engine running.
    def test_simulate_in_parallel(self) -> None:  # pragma: no cover
        if not _IPYTHON_AVAILABLE:
            self.skipTest(
                "test_simulate_in_parallel - IPython is not installed")

        try:
            from ipyparallel import Client
            cl = Client(profile=b"tests", timeout=1.0)

            dview = cl.direct_view()
            # Reset the engines so that we don't have variables there from
            # last computations
            dview.execute('%reset')
            dview.execute('import sys')
            # We use block=True to ensure that all engines have modified
            # their path to include the folder with the simulator before we
            # create the load lanced view in the following.
            dview.execute('sys.path.append("{0}")'.format(os.getcwd()),
                          block=True)

            lview = cl.load_balanced_view()
            if len(lview) == 0:
                self.skipTest("test_simulate_in_parallel - "
                              "At least one IPython engine must be running.")
        except IOError:
            self.skipTest(
                "test_simulate_in_parallel - "
                "The IPython engines were not found. ('tests' profile)")

        # noinspection PyUnresolvedReferences
        from tests.simulations_package_test import _DummyRunner
        dview.execute('import simulations_package_test', block=True)

        sim_runner = _DummyRunner()
        sim_runner.progressbar_message = 'bla'
        # runner.update_progress_function_style = 'text1'

        # xxxxxxxxxx Set the name of the results file xxxxxxxxxxxxxxxxxxxxx
        filename = 'runner_results_bias_{bias}.pickle'
        sim_runner.set_results_filename(filename)

        # This will make the progressbar print to a file, instead of stdout
        sim_runner.progress_output_type = 'file'  # Default is 'screen'

        # Remove old file from previous test run
        _delete_pickle_files()

        # Delete all the *results*.txt files (created by the progressbar)
        _delete_progressbar_output_files()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        self.assertEqual(sim_runner.runned_reps, [])
        sim_runner.simulate_in_parallel(lview)
        self.assertGreater(sim_runner._simulation_tracking._elapsed_time, 0.0)
        self.assertEqual(sim_runner.runned_reps, [sim_runner.rep_max] *
                         sim_runner.params.get_num_unpacked_variations())

        results_extra_1 = sim_runner.results.get_result_values_list(
            'lala', {'extra': 2.2})
        expected_results_extra_1 = [3.5, 9.5, 15.5, 21.5, 27.5]
        np.testing.assert_array_almost_equal(results_extra_1,
                                             expected_results_extra_1)

        results_extra_2 = sim_runner.results.get_result_values_list(
            'lala', {'extra': 4.1})
        expected_results_extra_2 = [5.4, 11.4, 17.4, 23.4, 29.4]
        np.testing.assert_array_almost_equal(results_extra_2,
                                             expected_results_extra_2)

        # xxxxxxxxxx Test if the results were saved correctly xxxxxxxxxxxxx
        sim_results = SimulationResults.load_from_file(
            sim_runner.results_filename)
        self.assertEqual(sim_results, sim_runner.results)
        _delete_pickle_files()
        _delete_progressbar_output_files()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Repeat the test xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now we do not set the results filename
        runner2 = _DummyRunner()
        runner2.simulate_in_parallel(lview)
        self.assertEqual(sim_results, runner2.results)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Repeat the test with wrong partial results xxxxxxxxxxx
        runner3 = _DummyRunner()
        runner3.set_results_filename('runner3_results')
        runner3.delete_partial_results_bool = False
        runner3.simulate_in_parallel(lview)

        # Now we change the bias parameter
        runner3.params.add('bias', 1.5)

        # If we run a simulation with different parameters it will try to
        # load the partial results with wrong parameters and an exception
        # should be raised. The raised Exception is an ValueError, but
        # since they are raised in the IPython engines, IPython itself will
        # raise a CompositeError exception.
        with self.assertRaises(CompositeError):
            runner3.simulate_in_parallel(lview)

        # Delete all *.pickle files in the same folder
        _delete_pickle_files()
        # Delete all the *results*.txt files (created by the progressbar)
        _delete_progressbar_output_files()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # This test method is normally skipped, unless you have started an
    # IPython cluster with a "tests" profile so that you have at least one
    # engine running.
    def test_simulate_in_parallel_with_random_values(  # pragma: no cover
            self) -> None:
        if not _IPYTHON_AVAILABLE:
            self.skipTest("test_simulate_in_parallel_with_random_values - "
                          "IPython is not installed")

        try:
            from ipyparallel import Client
            cl = Client(profile=b"tests", timeout=1.0)

            dview = cl.direct_view()
            # Reset the engines so that we don't have variables there from
            # last computations
            dview.execute('%reset')
            dview.execute('import sys')
            # We use block=True to ensure that all engines have modified
            # their path to include the folder with the simulator before we
            # create the load lanced view in the following.
            dview.execute('sys.path.append("{0}")'.format(os.getcwd()),
                          block=True)

            lview = cl.load_balanced_view()
            if len(lview) == 0:  # pragma: no cover
                self.skipTest("At least one IPython engine must be running.")
        except IOError:  # pragma: no cover
            self.skipTest("test_simulate_in_parallel_with_random_values - "
                          "The IPython engines were not found. ('tests' "
                          "profile)")
        #
        #
        # This test is intended to clarify some special care that must be
        # taken regarding random sources when using the
        # simulate_in_parallel method.
        #
        # The _DummyRunnerRandom class will generate three results which
        # are generated as a sum of one element of the 'P' parameter and a
        # random value. The 'P' parameter is an array with 5 elements, all
        # of them equal to 2.0. That means that if we didn't have a random
        # part all elements in the returned results would be equal.
        # noinspection PyUnresolvedReferences
        from tests.simulations_package_test import _DummyRunnerRandom
        dummyrunnerrandom = _DummyRunnerRandom()
        dummyrunnerrandom.simulate_in_parallel(lview)

        # For the result1 the random part is generated by calling the
        # numpy.random.rand function inside _run_simulation. Because we are
        # using the module level function 'rand' we are using the global
        # RandomState object in numpy. This global RandomState object will
        # be naturally different in each ipython engine and thus each
        # element in result1 will be different.
        result1 = dummyrunnerrandom.results.get_result_values_list('result1')
        self.assertNotAlmostEqual(result1[0], result1[1])
        self.assertNotAlmostEqual(result1[1], result1[2])
        self.assertNotAlmostEqual(result1[2], result1[3])
        self.assertNotAlmostEqual(result1[3], result1[4])
        self.assertEqual(len(set(result1)), 5)  # 5 different elements
        # print; print result1

        # For result2 the random part is generated by calling the rand
        # method of a RandomState object created in the __init__ method of
        # the _DummyRunnerRandom object. The problem is that each ipython
        # engine will receive a copy of the _DummyRunnerRandom object and
        # thus of the RandomState object. That means that for each value of
        # the parameter 'P' the same random value will be generated and
        # thus 'result2' will have 5 equal elements.
        result2 = dummyrunnerrandom.results.get_result_values_list('result2')
        self.assertAlmostEqual(result2[0], result2[1])
        self.assertAlmostEqual(result2[1], result2[2])
        self.assertAlmostEqual(result2[2], result2[3])
        self.assertAlmostEqual(result2[3], result2[4])
        self.assertEqual(len(set(result2)), 1)  # 5 equal elements
        # print; print result2

        # For result3 the random part is generated by calling the rand
        # method of a RandomState object created in the __init__ method of
        # the _DummyRunnerRandom object. However, in the
        # _on_simulate_current_params_start method we re-seed this
        # RandomState object. Since _on_simulate_current_params_start is
        # called once for each different value of the 'P' parameter, then
        # the random value will be different for each value in 'P' and thus
        # result3 will have 5 different values.
        result3 = dummyrunnerrandom.results.get_result_values_list('result3')
        self.assertNotAlmostEqual(result3[0], result3[1])
        self.assertNotAlmostEqual(result3[1], result3[2])
        self.assertNotAlmostEqual(result3[2], result3[3])
        self.assertNotAlmostEqual(result3[3], result3[4])
        self.assertEqual(len(set(result3)), 5)  # 5 different elements

    # Test the simulate method when the SkipThisOne exception is raised in
    # the _run_simulation method.
    def test_simulate_with_skipthisone(self) -> None:
        # from tests.simulations_package_test import _DummyRunnerWithSkip
        dummyrunner = _DummyRunnerWithSkip()

        # xxxxxxxxxx Set the name of the results file xxxxxxxxxxxxxxxxxxxxx
        filename = 'dummyrunnerwithskip_results_bias_{bias}'
        dummyrunner.set_results_filename(filename)
        # This will make the progressbar print to a file, instead of stdout
        dummyrunner.progress_output_type = 'file'  # Default is 'screen'

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # The results will be the SNR values multiplied by 1.2. plus the
        # bias parameter
        dummyrunner.simulate()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Perform the tests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        results_extra_1 = dummyrunner.results.get_result_values_list(
            'lala', {'extra': 2.2})
        expected_results_extra_1 = [3.5, 9.5, 15.5, 21.5, 27.5]
        np.testing.assert_array_almost_equal(results_extra_1,
                                             expected_results_extra_1)

        results_extra_2 = dummyrunner.results.get_result_values_list(
            'lala', {'extra': 4.1})
        expected_results_extra_2 = [5.4, 11.4, 17.4, 23.4, 29.4]
        np.testing.assert_array_almost_equal(results_extra_2,
                                             expected_results_extra_2)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Test if the results were saved correctly xxxxxxxxxxxxx
        sim_results = SimulationResults.load_from_file(
            dummyrunner.results_filename)
        self.assertEqual(sim_results, dummyrunner.results)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # # xxxxxxxxxx Repeat the test xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # # Now we do not set the results filename
        # dummyrunner2 = _DummyRunner()
        # dummyrunner2.simulate()

        # self.assertEqual(dummyrunner.results, dummyrunner2.results)
        # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # # xxxxxxxxxx Repeat the test with wrong partial results xxxxxxxxxxx
        # # First we run a usual simulation and keep the partial results
        # dummyrunner3 = _DummyRunner()
        # dummyrunner3.set_results_filename('dummyrunner3_results')
        # dummyrunner3.delete_partial_results_bool = False
        # dummyrunner3.simulate()

        # # Now we change the bias parameter
        # dummyrunner3.params.add('bias', 1.5)

        # # If we run a simulation with different parameters it will try to
        # # load the partial results with wrong parameters and an exception
        # # should be raised
        # with self.assertRaises(ValueError):
        #     dummyrunner3.simulate()
        # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # # xxxxxxxxxx Repeat the test loading the partial results xxxxxxxxxx
        # dummyrunner4 = _DummyRunner()
        # dummyrunner4.set_results_filename('dummyrunner3_results')
        # dummyrunner4.delete_partial_results_bool = True
        # dummyrunner4.simulate()
        # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Delete the pickle files in the same folder
        _delete_pickle_files()


# xxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
