#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module docstring"""

__version__ = "$Revision: 5 $"
# $Source$

import math
import sys
import numpy as np
# import copy
sys.path.append("/home/darlan/cvs_files/Python_Funcs/")
#sys.path.append("../")

import modulators as mod
from util.darlan import dB2Linear
from util.progressbar import ProgressbarText


class SimulationRunner:
    """
    Helper class to run a (simulation) function several times and calculate the
    average result.

    It is expected that the provided (simulation) function returns a tuple with
    the number of errors (bits or symbols) and the input data size (number of
    simulated bits or symbols).

    There are two stop criteria: maximum number of obtained errors and maximum
    number of repetitions. The simulation will be stopped when any of these
    criteria is reached.
    """
    # Will store the simulation function that will be run several times
    func = None
    """The function that will be called for each simulation iteration."""

    def __init__(self, param_name, limit_value, rep_max, func, func_args):
        """Init function.
        @param param_name: Name of the parameter that will be used as an
                           early stop criteria
        @type param_name: string

        @param limit_value: Limit value of the parameter used as an stop
        criterion. If the type of the stop criterion parameter is SUMTYPE,
        then limit_value is the maximum allowed value for that
        parameter. If the type is RATIOTYPE, then limit_value corresponds
        to a minimum relative precision (a limit_value of 0.01 means that
        the simulation will stop when the parameter does not change more
        then 1% of the current stored value).

        @type limit_value: scalar (integer for a type equal to SUMTYPE or a
        float for a type equal to RATIOTYPE).

        @param rep_max: Maximum number of repetitions.
        @type rep_max: scalar

        @param func: The function that performs the actual simulation. It
        must return a SimulationResults object, where one of its element (a
        Result object) has the same name as param_name.

        @param func_args: Tuple containing the arguments to be passed to func.
        @type func_args: tuple

        """
        self.param_name = param_name
        self.limit_value = limit_value
        self.rep_max = rep_max
        self.func = func
        self.func_args = func_args

        # Internal counter for the elapsed time since the simulate function
        # started. __elapsed_time is reseted to zero if simulate is called
        # again.
        self.__elapsed_time = 0.0

    def simulate(self, unpack, use_progress_bar=False):
        """
        Run the simulation until one of the stop criteria (max number of
        errors or maximum repetitions) is reached.

        @param unpack: Which parameter in func_args should be
        unpacked. This usually corresponds to the SNR. If an array of SNRs
        is the first element in func_args (provided in the __init__
        function), then the value of unpack is 0.)

        @type unpack: scalar
        @param use_progress_bar: If a progress bar should be printed during the
        @type use_progress_bar: bool
        simulation.

        """
        # xxxxxxxxxxxxxxx Defines the update_progress function xxxxxxxxxxxx
        def get_update_progress_function(message):
            """Return a function that should be called to update the
            progress.

            `message`: The message to be written in the progressbar, if it
            is used.

            The returned function accepts a value between 0 and 1.
            """
            if(use_progress_bar):
                # If use_progress_bar is true, we create a progressbar and the
                # function update_progress will update the bar
                self.bar = ProgressbarText(self.limit_value, '*', message)
                update_progress = lambda value: self.bar.progress(value)
            else:
                # If use_progress_bar is false, the function update_progress
                # won't do anything
                update_progress = lambda value: None
            return update_progress
        # Reset the elapsed time and the number of repetitions
        self.__elapsed_time = 0.0

        # xxxxx Defines the keep_going function xxxxxxxxxxxxxxxxxxxxxxxxxxx
        def keep_going(simulation_results):
            """Return True if simulation should continue or False
            otherwise. That is, return False when the the Result object in
            simulation_results with name given by self.param_name has a
            value greater then self.limit_value.
            """
            param = simulation_results[self.param_name][-1]
            # current_value =
            # simulation_results.get_last_result_value(self.param_name)
            current_value = param.get_result()
            return current_value < self.limit_value

        # xxxxxxxxxxxxxxx Some initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        from time import time
        tic = time()
        func_args = list(self.func_args)
        pack_size = len(func_args[unpack])
        #errorrate   = np.zeros(pack_size, dtype=np.double)
        self.rep = np.zeros(pack_size, dtype=np.int)
        self.errors = np.zeros(pack_size, dtype=np.int)
        index = 0
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        sim_results = SimulationResults()
        for i in self.func_args[unpack]:  # SNR
            # Get the function called to update the progress
            update_progress = get_update_progress_function(
                'Simulating for SNR: %s' % i)

            # xxxxx Perform the simulation for one unpacked value xxxxxxxxx
            func_args[unpack] = i

            # Perform the first iteration and append the results in the
            # sim_results object
            sim_results.append_all_results(self.func(*func_args))
            # Perform the remaining iterations and updates current SNR results
            while (keep_going(sim_results) and
                   (self.rep[index] < self.rep_max)):
                #             for j in xrange(0,Repmax):
                sim_results.merge_all_results(self.func(*func_args))
                # err = results[0]
                # self.nsymbits = results[1]
                # self.errors[index] += err
                # update_progress(
                #     sim_results.get_last_result_value(self.param_name))

                update_progress(sim_results[self.param_name][-1].get_result())
                self.rep[index] += 1
            # errorrate[index] += (np.double(self.errors[index]) /
            #                       (self.rep[index] * self.nsymbits))
            index += 1
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        toc = time()
        self.__elapsed_time = toc - tic
        print "\nxxxxxxxxxxxxxxx End of Simulation xxxxxxxxxxxxxxxx\n"
        print "Elapsed Time: %s" % pretty_time(self.__elapsed_time)
        # Return the errorrate
        #return errorrate
        return sim_results

    def elapsed_time(self):
        """property: Get the simulation elapsed time. Do not set this value."""
        return pretty_time(self.__elapsed_time)
    # xxxxxxxxxx End of SimulationRunner class xxxxxxxxxxxxxxxxxxxxxxxxxxxx


# TODO: Add doctests
class SimulationResults():
    """Store results from simulations.

    This class is used in the SimulationRunner class in order to store
    results from each simulation. It is able to combine the results from
    multiple simulations.

    >>> result1 = Result("lala", Result.SUMTYPE)
    >>> result1.update(13)
    >>> result2 = Result("lele", Result.RATIOTYPE)
    >>> result2.update(3, 10)
    >>> result2.update(8, 10)
    >>> simresults = SimulationResults()
    >>> simresults.add_result(result1)
    >>> simresults.add_result(result2)
    >>> simresults.get_result_names()
    ['lele', 'lala']
    >>> simresults
    SimulationResults: ['lele', 'lala']

    >>> result1_other = Result('lala', Result.SUMTYPE)
    >>> result1_other.update(7)
    >>> simresults.append_result(result1_other)
    >>> simresults.get_result_values_list('lala')
    [13, 7]
    >>> simresults['lala']
    [Result -> lala: 13, Result -> lala: 7]
    >>> len(simresults)
    2

    >>> result3 = Result("lala", Result.SUMTYPE)
    >>> result3.update(2)
    >>> result4 = Result("lele", Result.RATIOTYPE)
    >>> result4.update(1, 2)
    >>> result4.update(3, 3)
    >>> simresults2 = SimulationResults()
    >>> simresults2.add_result(result3)
    >>> simresults2.add_result(result4)
    >>> simresults2.merge_all_results(simresults)
    >>> simresults2['lala']
    [Result -> lala: 9]
    >>> simresults2['lele']
    [Result -> lele: 15/25 -> 0.6]
    >>> simresults3 = SimulationResults()
    >>> simresults3.append_all_results(simresults)
    >>> simresults3['lala']
    [Result -> lala: 13, Result -> lala: 7]
    >>> simresults3['lele']
    [Result -> lele: 11/20 -> 0.55]
    """
    def __init__(self):
        """
        """
        self.__results = dict()

    def __repr__(self):
        lista = [i for i in self.__results.keys()]
        repr = "SimulationResults: %s" % lista
        return repr

    def add_result(self, result):
        """Add a new result to the SimulationResults object. If there is
        already a result stored with the same name, this will replace it.

        Arguments:
        - `result`: Must be an object of the Result class.
        """
        # Added as a list with a single element
        self.__results[result.name] = [result]

    def append_result(self, result):
        """Append a result to the SimulationResults object. This
        efectivelly means that the SimulationResults object will now store
        a list for the given result name. This allow you, for instance, to
        store multiple bit error rates with the 'BER' name such that
        simulation_results_object['BER'] will return a list with the Result
        objects for each value.

        Note that if multiple values for some Result are stored, then only
        the last value can be updated with merge_all_results.

        Arguments:
        - `result`: A Result object

        """
        if result.name in self.__results.keys():
            self.__results[result.name].append(result)
        else:
            self.add_result(result)

    def append_all_results(self, other):
        """Append all the results of the other SimulationResults object
        with self.

        Arguments:
        - `other`: Another SimulationResults object

        """
        for results in other:
            # There can be more then one value for the same result name
            for result in results:
                self.append_result(result)

    def merge_all_results(self, other):
        """Merge all the results of the other SimulationResults object with
        the results in self.

        When there is more then one result with the same name stored in
        self (for instance two bit error rates) then only the last one will
        be merged with the one in "other". That also means that only one
        result for that name should be stored in "other".

        Arguments:
        - `other`: Another SimulationResults object

        """
        # If the current SimulationResults object is empty
        if len(self) == 0:
            for name in other.get_result_names():
                self.__results[name] = other[name]
        else:
            for item in self.__results.keys():
                self.__results[item][-1].merge(other[item][-1])

    # def get_last_result_value(self, result_name):
    #     """Get the value of the last result with name given by
    #     "result_name".

    #     Since an SimulationResults object can store multiple Result objects
    #     with the same name, which are stored in a list. This functions
    #     provides an easy way to get the value of the last Result stored for
    #     "result_name".

    #     Arguments:
    #     - `result_name`: A string with the name of the desired result.

    #     """
    #     return self.__results[result_name][-1].get_result()

    def get_result_names(self):
        return self.__results.keys()

    def get_result_values_list(self, result_name):
        """Get the values for the results with name "result_name"

        Returns a list with the values.

        Arguments:
        - `result_name`: A string
        """
        return [i.value for i in self[result_name]]

    def __getitem__(self, key):
        # if key in self.__results.keys():
        return self.__results[key]
        # else:
        #     raise KeyError("Invalid key: %s" % key)

    def __len__(self):
        """Get the number of results stored in self.
        """
        return len(self.__results)

    def __iter__(self):
        # """Get an iterator to the internal dictionary. Therefore iterating
        # through this will iterate through the dictionary keys, that is, the
        # name of the results stored in the SimulationResults object.
        # """
        """Get an iterator to the results stored in the SimulationResults
        object.
        """
        return self.__results.itervalues()


class Result():
    """Class to store a single simulation result.

    The simulation result can be anything, such as the number of errors, a
    string, an error rate, etc. When creating a Result object one needs to
    specify only the name of the stored result and the result type.

    The diferent types indicate how the Result can be updated (combined
    with other samples). The possible values are SUMTYPE, RATIOTYPE and
    STRINGTYPE.

    In the SUMTYPE the new value should be added to current one in update
    function.

    In the RATIOTYPE the new value should be added to current one and total
    should be also updated in the update function. One caveat is that rates
    are stored as a number (numerator) and a total (denominator) instead of
    as a float.

    In the STRINGTYPE the update should replace current the value, since it
    is a string.

    >>> result1 = Result("name", Result.SUMTYPE)
    >>> result1.update(13)
    >>> result1.update(4)
    >>> result1.value
    17
    >>> result1.get_result()
    17
    >>> result1.num_updates
    2
    >>> result1
    Result -> name: 17
    >>> result1.get_type_name()
    'SUMTYPE'
    >>> result1.get_type()
    0
    >>> print result1
    Result -> name: 17

    >>> result2 = Result("name2", Result.RATIOTYPE)
    >>> result2.update(4,10)
    >>> result2.update(3,4)
    >>> result2.get_result()
    0.5
    >>> result2.get_type_name()
    'RATIOTYPE'
    >>> result2.get_type()
    1
    >>> result2_other = Result("name2", Result.RATIOTYPE)
    >>> result2_other.update(3,11)
    >>> result2_other.merge(result2)
    >>> result2_other.get_result()
    0.4
    >>> result2_other.num_updates
    3
    >>> result2_other.value
    10
    >>> result2_other.total
    25
    >>> print result2_other
    Result -> name2: 10/25 -> 0.4
    """
    # Like an Enumeration for the type of results.
    (SUMTYPE, RATIOTYPE, STRINGTYPE, FLOATTYPE) = range(4)
    all_types = {
        SUMTYPE: "SUMTYPE",
        RATIOTYPE: "RATIOTYPE",
        STRINGTYPE: "STRINGTYPE",
        FLOATTYPE: "FLOATTYPE",
    }

    def __init__(self, name, update_type):
        """
        """
        self.name = name
        self.__update_type = update_type
        self.value = 0
        self.total = 0
        self.num_updates = 0  # Number of times the Result object was
                              # updated

    def __repr__(self):
        if self.__update_type == Result.RATIOTYPE:
            v = self.value
            t = self.total
            return "Result -> {0}: {1}/{2} -> {3}".format(
                self.name, v, t, float(v) / t)
        else:
            return "Result -> {0}: {1}".format(self.name, self.get_result())

    def update(self, value, total=0):
        """Update the current value.

        Arguments:
        - `value`: Value to be added to (or replaced) the current value
        - `total`: Value to be added to (if applied) the current total
          (only useful for the RATIOTYPE update type)

        How the update is performed for each Result type
        - RATIOTYPE: Add "value" to current value and "total" to current total
        - SUMTYPE: Add "value" to current value. "total" is ignored.
        - STRINGTYPE: Replace the current value with "value".
        - FLOATTYPE: Replace the current value with "value".

        """
        self.num_updates += 1

        # Python does not have a switch statement. It usually uses
        # dictionaries for this
        possible_updates = {
            Result.RATIOTYPE: self.__update_RATIOTYPE_value,
            Result.STRINGTYPE: self.__update_by_replacing_current_value,
            Result.SUMTYPE: self.__update_SUMTYPE_value,
            Result.FLOATTYPE: self.__update_by_replacing_current_value
            }

        # Call the apropriated update method
        possible_updates.get(self.__update_type,
                             self.__default_update)(value, total)

    def __default_update(self, ignored1, ignored2):
        print("Warning: update not performed for unknown type %s" %
              self.__update_type)
        pass

    def __update_SUMTYPE_value(self, value, ignored):
        """Only called inside the update function"""
        self.value += value

    def __update_RATIOTYPE_value(self, value, total):
        """Only called inside the update function"""
        assert value <= total, ("__update_RATIOTYPE_value: "
                                "'value cannot be greater then total'")
        if total == 0:
            print("Update Ignored: total should be provided and be greater "
                  "then 0 when the update type is RATIOTYPE")
        else:
            self.value += value
            self.total += total

    def __update_by_replacing_current_value(self, value, ignored):
        """Only called inside the update function"""
        self.value = value

    def get_type_name(self):
        return Result.all_types[self.__update_type]

    def get_type(self):
        """Get the Result type.

        The returned value is one of the keys in Result.all_types.
        """
        return self.__update_type

    def merge(self, other):
        """Merge the result from other with self.

        Arguments:
        - `other`: Another Result object.
        """
        assert self.__update_type == other.__update_type, (
            "Can only merge to objects with the same name and type")
        assert self.__update_type != Result.STRINGTYPE, (
            "Cannot merge results of the STRINGTYPE type")
        assert self.name == other.name, (
            "Can only merge to objects with the same name and update_type")
        self.num_updates += other.num_updates
        self.value += other.value
        self.total += other.total

    def get_result(self):
        if self.num_updates == 0:
            return "Nothing yet".format(self.name)
        else:
            if self.__update_type == Result.RATIOTYPE:
                #assert self.total != 0, 'Total should not be zero'
                return float(self.value) / self.total
            else:
                return self.value


def pretty_time(time_in_seconds):
    """Return the time in a more friendly way.

    >>> pretty_time(30)
    '30.00s'
    >>> pretty_time(76)
    '1m:16s'
    >>> pretty_time(4343)
    '1h:12m:23s'
    """
    seconds = time_in_seconds
    minutes = int(seconds) / 60
    seconds = int(round(seconds % 60))

    hours = minutes / 60
    minutes = minutes % 60

    if(hours > 0):
        return "%sh:%sm:%ss" % (hours, minutes, seconds)
    elif(minutes > 0):
        return "%sm:%ss" % (minutes, seconds)
    else:
        return "%.2fs" % time_in_seconds


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxx Simulate functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def simulate_one_psk(SNR, NSymbs=100, M=4):
    """ Simulates one iteration of a PSK scheme"""
    # xxxxx Input Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    inputData = np.random.randint(0, M, NSymbs)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Modulate input data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    psk = mod.PSK(M)
    modulatedData = psk.modulate(inputData)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    noiseVar = 1 / dB2Linear(SNR)
    noise = ((np.random.randn(NSymbs) + 1j * np.random.randn(NSymbs)) *
             math.sqrt(noiseVar / 2))
    receivedData = modulatedData + noise
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Demodulate received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    demodulatedData = psk.demodulate(receivedData)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Calculates the symbol and bit error rates xxxxxxxxxxxxxxxxxxxxx
    symbolErrors = sum(inputData != demodulatedData)
    aux = mod.xor(inputData, demodulatedData)
    # Count the number of bits in aux
    bitErrors = sum(mod.bitCount(aux))
    numSymbols = inputData.size
    numBits = inputData.size * mod.level2bits(M)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Essa parte abaixo por enquanto é ignorada
    # xxxxx Return the simulation results xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    symbolErrorsResult = Result("symbol_errors", Result.SUMTYPE)
    symbolErrorsResult.update(symbolErrors)
    numSymbolsResult = Result("num_symbols", Result.SUMTYPE)
    numSymbolsResult.update(numSymbols)
    bitErrorsResult = Result("bit_errors", Result.SUMTYPE)
    bitErrorsResult.update(bitErrors)
    numBitsResult = Result("num_bits", Result.SUMTYPE)
    numBitsResult.update(numBits)

    berResult = Result("ber", Result.RATIOTYPE)
    berResult.update(bitErrors, numBits)
    serResult = Result("ser", Result.RATIOTYPE)
    berResult.update(symbolErrors, numSymbols)

    simResults = SimulationResults()
    simResults.add_result(symbolErrorsResult)
    simResults.add_result(numSymbolsResult)
    simResults.add_result(bitErrorsResult)
    simResults.add_result(numBitsResult)

    simResults.add_result(berResult)
    simResults.add_result(serResult)

    #return (symbolErrors, numSymbols, bitErrors, numBits)
    return simResults


def simulate_psk_with_runner():
    NSymbs = int(100)
    M = 4
    SNR = np.array([0, 3, 5])
    max_errors = 10000
    rep_max = 10000

    runner = SimulationRunner('bit_errors',
                              max_errors,
                              rep_max,
                              simulate_one_psk,
                              (SNR, NSymbs, M))
    errorrate = runner.simulate(0, True)

    # Calculates the theoretical BER
    psk = mod.PSK(M)
    theoretical_ber = psk.calcTheoreticalBER(dB2Linear(SNR))
    print theoretical_ber

    return (errorrate, runner)


def calcTheorecticalPSKBitErrorRate():
    SNRs = np.arange(-2, 11, 2)
    psk = mod.PSK(4)
    SER = psk.calcTheoreticalSER(dB2Linear(SNRs))
    print "BER"
    print SER
    BER = psk.calcTheoreticalBER(dB2Linear(SNRs))
    print "BER"
    print BER


def test_results():
    result1 = Result("lala", Result.SUMTYPE)
    result1.update(13)

    result2 = Result("lele", Result.RATIOTYPE)
    result2.update(3, 10)
    result2.update(8, 10)

    simresults = SimulationResults()
    simresults.add_result(result1)
    #simresults.add_result(result1)
    simresults.add_result(result2)

    result1_other = Result('lala', Result.SUMTYPE)
    result1_other.update(7)
    simresults.append_result(result1_other)
    #print simresults['lala']

    result3 = Result("lala", Result.SUMTYPE)
    result3.update(2)

    result4 = Result("lele", Result.RATIOTYPE)
    result4.update(1, 2)
    result4.update(3, 3)

    simresults2 = SimulationResults()
    simresults2.add_result(result3)
    simresults2.add_result(result4)

    simresults2.merge_all_results(simresults)

    simresults3 = SimulationResults()
    simresults3.append_all_results(simresults)

    return (simresults, simresults2, simresults3)


if __name__ == '__main__1':
    (simresults1, simresults2, simresults3) = test_results()


if __name__ == '__main__2':
    (sim_results, simrunner) = simulate_psk_with_runner()

    # Calculates the error rates
    num_bits = np.array(sim_results.get_result_values_list('num_bits'))
    bit_errors = np.array(sim_results.get_result_values_list('bit_errors'))
    error_rate = bit_errors.astype(float) / num_bits
    print error_rate


if __name__ == '__main__':
    # When this module is run as a script the doctests are executed
    import doctest
    doctest.testmod()
