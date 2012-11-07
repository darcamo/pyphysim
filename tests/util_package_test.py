#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the modules in the util package.

Each module has doctests for its functions and all we need to do is run all
of them.
"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import unittest
import doctest
import numpy as np

from util import misc, progressbar, simulations, conversion
from util.simulations import *


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


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx simulations Module xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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

    def test_test_representation(self):
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


class SimulationResultsTestCase(unittest.TestCase):
    """Unit-tests for the SimulationResults class in the simulations
    module.

    """
    def setUp(self):
        # First SimulationResults object
        result1 = Result.create("lala", Result.SUMTYPE, 13)
        result2 = Result("lele", Result.RATIOTYPE)
        result2.update(3, 10)
        result2.update(8, 10)
        self.simresults = SimulationResults()
        self.simresults.add_result(result1)
        self.simresults.add_result(result2)

        # Second SimulationResults object
        self.other_simresults = SimulationResults()
        result1_other = Result.create('lala', Result.SUMTYPE, 30)
        result2_other = Result.create('lele', Result.RATIOTYPE, 4, 10)
        result3 = Result.create('lili', Result.MISCTYPE, "a string")
        self.other_simresults.add_result(result1_other)
        self.other_simresults.add_result(result2_other)
        self.other_simresults.add_result(result3)

    def test_get_result_names(self):
        # The output of the get_result_names is a list of names. We
        # transform it into a set in this test only to make the order of
        # the names uninportant.
        expected_output = set(['lala', 'lele'])
        self.assertEqual(set(self.simresults.get_result_names()), expected_output)
        # Test also the representation of the SimulationResults object
        self.assertEqual(self.simresults.__repr__(), """SimulationResults: ['lele', 'lala']""")

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

        # Now that a few parameters were added and set to be unpacked, lets
        # test the representation of the SimulationParameters object. Note
        # that the parameters that are marked for unpacking have '*'
        # appended to their name.
        self.assertEqual(self.sim_params.__repr__(), """{'second': 20, 'fifth': 10, 'fourth*': ['A', 'B'], 'third*': [1 3 2 5], 'first': 10}""")

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

        # We make change all values to sets to remove repeated values for
        # testing purposes.
        self.assertEqual(set(params_dict['first']),
                         set([self.sim_params['first']]))
        self.assertEqual(set(params_dict['second']),
                         set([self.sim_params['second']]))
        self.assertEqual(set(params_dict['third']),
                         set(self.sim_params['third']))
        self.assertEqual(set(params_dict['fourth']),
                         set(self.sim_params['fourth']))

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

    def test_save_to_and_load_from_file(self):
        # TODO: Implement-me
        pass


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
        pass
        #self.assertRaises(NotImplementedError, self.S1._get_vertex_positions)
        with self.assertRaises(NotImplementedError):
            self.runner._run_simulation(None)

    def test_keep_going(self):
        # the _keep_going method in the SimulationRunner class should
        # return True
        self.assertTrue(self.runner._keep_going(None))

    def test_simulate(self):
        # First we need to create a dummy subclass of SimulationRunner that
        # implements the _run_simulation method.
        class DummyRunner(simulations.SimulationRunner):
            def __init__(self):
                simulations.SimulationRunner.__init__(self)
                # Set the progress bar message to None to avoid print the
                # progressbar in these testes.
                self.rep_max = 2
                self.progressbar_message = None
                # Now we add a dummy parameter to our runner object
                self.params.add('dummy_increment', 1)

            def _run_simulation(self, current_params):
                return SimulationResults()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now we create an instance of DummyRunner
        dummyrunner = DummyRunner()
        # then we call its simulate method
        dummyrunner.simulate()

        dummyrunner.progressbar_message = 'Progress'
        dummyrunner.simulate()


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
            map(misc.level2bits, range(1, 20)),
            [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5])

        # Test if an exception is raised for a value of n lower then 1
        with self.assertRaises(ValueError):
            misc.level2bits(0)
        with self.assertRaises(ValueError):
            misc.level2bits(-2)

    def test_int2bits(self):
        self.assertEqual(
            map(misc.int2bits, range(0, 19)),
            [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5])

        # Test if an exception is raised for a negative value of n
        with self.assertRaises(ValueError):
            misc.int2bits(-1)

    def test_bitCount(self):
        self.assertEqual(misc.bitCount(0), 0)
        self.assertEqual(misc.bitCount(1), 1)
        self.assertEqual(misc.bitCount(2), 1)
        self.assertEqual(misc.bitCount(3), 2)
        self.assertEqual(misc.bitCount(4), 1)
        self.assertEqual(misc.bitCount(5), 2)
        self.assertEqual(misc.bitCount(6), 2)
        self.assertEqual(misc.bitCount(7), 3)
        self.assertEqual(misc.bitCount(8), 1)
        self.assertEqual(misc.bitCount(15), 4)

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
        expected_autocor = np.array(
            [1. , -0.025, 0.15 , -0.175, -0.25 , -0.2 , 0.])
        np.testing.assert_array_almost_equal(autocor, expected_autocor)


# xxxxxxxxxx Doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == "__main__":
    unittest.main()
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


def progress_producer2(bar, sleep_time=0.5):
    total_count = 20
    for i in range(1, total_count + 1):
        sleep(sleep_time)
        bar.progress(i)


def progress_producer(process_id, process_data_list, sleep_time=0.5):
    total_count = 20
    for i in range(1, total_count + 1):
        sleep(sleep_time)
        process_data_list[process_id] = i


if __name__ == '__main__1':
    from time import sleep
    import multiprocessing
    from progressbar import ProgressbarMultiProcessText
    bar = ProgressbarMultiProcessText(sleep_time=1)

    # # xxxxx Option 1: register_function xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # # Register two functions with count 20, each, in the progressbar
    # func_1_data = bar.register_function(20)
    # func_2_data = bar.register_function(20)

    # # Create the processes to run the functions
    # p1 = multiprocessing.Process(target=progress_producer, args=(func_1_data[0], func_1_data[1], 0.2))
    # p2 = multiprocessing.Process(target=progress_producer, args=(func_2_data[0], func_2_data[1], 0.3))
    # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Option 2: register_function_and_get_proxy_progressbar xxxxxxxxx
    # Register two functions with count 20, each, in the progressbar
    proxybar1 = bar.register_function_and_get_proxy_progressbar(20)
    proxybar2 = bar.register_function_and_get_proxy_progressbar(20)

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

    print "FIM"
