#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perform the simulation of the transmission of PSK symbols through an
awgn channel."""

import numpy as np
import math
import itertools
import copy
from collections import OrderedDict

from util.conversion import dB2Linear
from util import misc
import comm.modulators as mod
from simulations import SimulationResults, Result, SimulationRunner
from util.progressbar import ProgressbarText
from misc import pretty_time

# So that we can test "isinstance(object, collections.Iterable)"
import collections


class SimulationParameters():
    """Class to store the simulation parameters.

    A SimulationParameters object acts as a container for all simulation
    parameters. To add a new parameter to the object just call the `add`
    method passing the name and the value of the parameter. The value can
    be anything as long as the _run_simulation function can understand it.
    """
    def __init__(self):
        """
        """
        # Dictionary that will store the parameters. The key is the
        # parameter name and the value is the parameter value.
        self.parameters = {}

        # A set to store the names of the parameters that will be unpacked.
        self.unpacked_parameters = set()

        # # Number of possible combinations we will get after unpacking the marked parameters
        # self._unpack_combinations = 0

    @staticmethod
    def create(params_dict):
        """Create a new SimulationParameters object.

        This static method provides a different way to create a
        SimulationParameters object, already containing the parameters in
        the `params_dict` dictionary.
        Arguments:
        - `params_dict`: Dictionary containing the parameters. Each
                         dictionary key corresponds to a parameter.
        """
        sim_params = SimulationParameters()
        sim_params.parameters = copy.deepcopy(params_dict)
        return sim_params

    def add(self, name, value):
        """Add a new parameter.

        If there is already a parameter with the same name it will be
        replaced.

        Arguments:
        - `name`: Name of the parameter
        - `value`: Value of the parameter
        """
        self.parameters[name] = value

    def set_unpack_parameter(self, name, unpack_bool=True):
        """Set the unpack property of the parameter with name `name`

        This is used in the SimulationRunner.
        Arguments:
        - `name`: Name of the parameter to be unpacked
        - `unpack_bool`:

        Raises:
        - ValueError: if `name` is not in parameters or is not iterable.
        """
        if name in self.parameters.keys():
            if isinstance(self.parameters[name], collections.Iterable):
                self.unpacked_parameters.add(name)
            else:
                raise ValueError("Parameter {0} is not iterable".format(name))
        else:
            raise ValueError("Unknown parameter: `{0}`".format(name))

    def __getitem__(self, name):
        """Return the parameter with name `name`

        Arguments:
        - `name`: Name of the desired parameter
        """
        return self.parameters[name]

    def __repr__(self):
        """
        """
        # TODO: Add an * in the name of the parameters set to be unpacked
        return str(self.parameters)

    # TODO: termine de implementar
    def get_unpacked_params_list(self):
        """Get a list of SimulationParameters objects, each one
        corresponding to a possible combination of parameters.

        Supose you have a SimulationParameters object with the parameters
        a=1, b=2, c=[3,4] and d=[5,6]
        and the parameters `c` and `d` were set to be unpacked.  Then
        get_unpacked_params_list would return a list of four
        SimulationParameters objects with parameters (may have a different
        order)
        {'a': 1, 'c': 3, 'b': 2, 'd': 5}
        {'a': 1, 'c': 3, 'b': 2, 'd': 6}
        {'a': 1, 'c': 4, 'b': 2, 'd': 5}
        {'a': 1, 'c': 4, 'b': 2, 'd': 6}
        """
        # If unpacked_parameters is empty, return self
        if not self.unpacked_parameters:
            return [self]

        # Lambda function to get an iterator to a (iterable) parameter
        # given its name
        f = lambda name: iter(self.parameters[name])

        # Dictionary that stores the name and an iterator of a parameter
        # marked to be unpacked
        unpacked_params_iter_dict = OrderedDict()
        for i in self.unpacked_parameters:
            unpacked_params_iter_dict[i] = f(i)
        keys = unpacked_params_iter_dict.keys()

        # Using itertools.product we can convert the multiple iterators
        # (for the different parameters marked to be unpacked) to a single
        # iterator that returns all the possible combinations (cartesian
        # product) of the individual iterators.
        all_combinations = itertools.product(*(unpacked_params_iter_dict.values()))

        # Names of the parameters that don't need to be unpacked
        regular_params = set(self.parameters.keys()) - self.unpacked_parameters

        # Constructs a list with dictionaries, where each dictionary
        # corresponds to a possible parameters combination
        unpack_params_length = len(self.unpacked_parameters)
        all_possible_dicts_list = []
        for comb in all_combinations:
            new_dict = {}
            # Add current combination of the unpacked parameters
            for index in range(unpack_params_length):
                new_dict[keys[index]] = comb[index]
            # Add the regular parameters
            for param in regular_params:
                new_dict[param] = self.parameters[param]
            all_possible_dicts_list.append(new_dict)

        # Map the list of dictionaries to a list of SimulationParameters
        # objects and return it
        return map(SimulationParameters.create, all_possible_dicts_list)

    def save_to_file(self, file_name):
        """

        Arguments:
        - `file_name`: Name of the file to save the parameters.
        """
        NotImplemented("SimulationParameters.save_to_file: Implement-me")


class SimulationRunner2():
    """Base class to run simulations.

    You need to derive from this class and implement at least the
    _run_simulation function (see `_run_simulation` help). If a stop
    criterion besides the maximum number of iterations is desired, then you
    need to also reimplement the _keep_going function.

    Note that since _run_simulation receives no argument, then whatever is
    needed must be added to the `params` atribute. That is, in the
    construtor of the derived class call the `add` method of `params` for
    each parameter you need in the _run_simulation function.

    Likewise, the _run_simulation method should return the results as a
    SimulationResults object.
    """

    def __init__(self, rep_max, use_progress_bar=True):
        """
        """
        self.rep_max = rep_max
        self._elapsed_time = 0.0
        self._runned_reps = []  # Number of iterations performed by
                                # simulation when it finished
        self.use_progress_bar = use_progress_bar
        self.params = SimulationParameters()

        self.results = []

    def _run_simulation(self, current_parameters):
        """Performs the one simulation.

        This function must be implemented in a subclass. It should take the
        needed parameters from the params class attribute (which was filled
        in the constructor of the derived class) and return the results as
        a SimulationResults object.

        Note that _run_simulation will be called self.rep_max times (or
        less if an early stop criteria is reached, which requires
        reimplementing the _keep_going function in the derived class) and
        the results from multiple repetitions will be merged.

        Arguments:
        - `current_parameters`: SimulationParameters object woth the
                                parameters for the simulation. The
                                self.params variable is not used because it
                                may need to be unpacked.
        """
        NotImplemented("This function must be implemented in a subclass")

    def _keep_going(self, simulation_results):
        """Check if the simulation should continue or stop.

        This function must be reimplemented in the derived class if a stop
        condition besides the number of iterations is desired.  The idea is
        that _run_simulation returns a SimulationResults object, which is
        then passed to _keep_going, which is then in charge of deciding if
        the simulation should stop or not.

        Arguments:
        - `simulation_results`: SimulationResults object from the last
                                iteration (merged with the previous
                                results)
        """
        # If this function is not reimplemented in a subclass it always
        # returns True and therefore the simulation will only stop when the
        # maximum number of allowed iterations is reached.
        return True

    def simulate(self):
        """
        """
        # xxxxxxxxxxxxxxx Defines the update_progress function xxxxxxxxxxxx
        def get_update_progress_function(message):
            """Return a function that should be called to update the
            progress.

            `message`: The message to be written in the progressbar, if it
            is used.

            The returned function accepts a value between 0 and 1.
            """
            if(self.use_progress_bar):
                # If use_progress_bar is true, we create a progressbar and the
                # function update_progress will update the bar
                self.bar = ProgressbarText(self.rep_max, '*', message)
                update_progress = lambda value: self.bar.progress(value)
            else:
                # If use_progress_bar is false, the function update_progress
                # won't do anything
                update_progress = lambda value: None
            return update_progress
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxx Some initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        print "\nxxxxxxxxxxxxxxx Start of the Simulation xxxxxxxxxx\n"
        from time import time
        tic = time()
        #func_args = list(self.func_args)
        #pack_size = len(func_args[unpack])
        #errorrate   = np.zeros(pack_size, dtype=np.double)
        #self.rep = np.zeros(pack_size, dtype=np.int)
        #self.errors = np.zeros(pack_size, dtype=np.int)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx FOR UNPACKED PARAMETERS xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        for current_params in self.params.get_unpacked_params_list():
            print current_params

            current_rep = 0  # np.zeros(pack_size, dtype=np.int) -> current_rep[index] < rep_max
            update_progress = get_update_progress_function(
                "PSK Simulation - SNR: {0}".format(self.params['SNR']))

            # First iteration
            current_sim_results = self._run_simulation(current_params)
            current_cumulated_errors = 0
            while (self._keep_going(current_sim_results, current_cumulated_errors) and
                   current_rep < self.rep_max):
                current_sim_results.merge_all_results(self._run_simulation(current_params))
                current_cumulated_errors = current_sim_results['bit_errors'][-1].get_result()
                update_progress(current_rep + 1)
                current_rep += 1

            # If the while loop ended before rep_max repetitions (because
            # _keep_going returned false) the set the progressbar as full.
            update_progress(self.rep_max)
            self._runned_reps.append(current_rep)  # Store the number of repetitions actually runned
            self.results.append(current_sim_results)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # index += 1  # Parameters unpack index
        toc = time()
        self._elapsed_time = toc - tic

        print "\nxxxxxxxxxxxxxxx End of Simulation xxxxxxxxxxxxxxxx\n"

        print "Performed {0} iterations".format(self._runned_reps)
        print "Elapsed Time: {0}".format(self.get_elapsed_time())

        print "Resultados:"
        print self.results

    def get_elapsed_time(self):
        """property: Get the simulation elapsed time. Do not set this
        value."""
        return pretty_time(self._elapsed_time)


class PskSimulationRunner(SimulationRunner2):
    """
    """

    def __init__(self, rep_Max):
        """
        """
        SimulationRunner2.__init__(self, rep_Max)
        # We can add anything to the simulation parameters. Note that most
        # of these parameters will be used in the _run_simulation function
        # and we could put them there, but putting the parameters here
        # makes thinks more modular.
        self.params.add("description", "Parameters for the simulation of a PSK transmission through an AWGN channel ")
        self.params.add("SNR", np.array([0, 5]))
        self.params.add("M", 4)         # Modulation cardinality
        self.params.add("NSymbs", 100)  # Number of symbols that will be
                                        # transmitted in the _run_simulation
                                        # function

        # Unpack the SNR parameter
        self.params.set_unpack_parameter("SNR")

        # We will stop when the number of bit errors is greater than
        # max_errors
        self.max_errors = 10000

    def _run_simulation(self, current_parameters):
        # Input parameters (set in the constructor)
        # NSymbs = self.params["NSymbs"]
        # M = self.params["M"]
        # SNR = self.params["SNR"]

        NSymbs = current_parameters["NSymbs"]
        M = current_parameters["M"]
        SNR = current_parameters["SNR"]

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
        aux = misc.xor(inputData, demodulatedData)
        # Count the number of bits in aux
        bitErrors = sum(misc.bitCount(aux))
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
        serResult.update(symbolErrors, numSymbols)

        simResults = SimulationResults()
        simResults.add_result(symbolErrorsResult)
        simResults.add_result(numSymbolsResult)
        simResults.add_result(bitErrorsResult)
        simResults.add_result(numBitsResult)

        simResults.add_result(berResult)
        simResults.add_result(serResult)

        #return (symbolErrors, numSymbols, bitErrors, numBits)
        return simResults

    def _keep_going(self, simulation_results, cumulated_errors):
        # Return true as long as cumulated_errors is lower then max_errors
        return cumulated_errors < self.max_errors


if __name__ == '__main__':
    # params = SimulationParameters()

    # params.add("SNR", np.arange(1, 5))
    # params.add("SNR2", np.arange(10, 15))
    # params.add("alpha", 0.1)
    # params.add("beta", 0.4)

    # params.set_unpack_parameter("SNR")
    # params.set_unpack_parameter("SNR2")

    # for i in params.get_unpacked_params_list():
    #     print i

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # dicionario = params.parameters
    # params2 = SimulationParameters.create(dicionario)
    # dicionario['alpha']=3
    # print params2
    # print params

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    psk_runner = PskSimulationRunner(1000)
    psk_runner.simulate()
    print psk_runner.results[0]['ber']
    print psk_runner.results[1]['ber']
    for i in psk_runner._runned_reps:
        print i
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


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
    aux = misc.xor(inputData, demodulatedData)
    # Count the number of bits in aux
    bitErrors = sum(misc.bitCount(aux))
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
    import sys
    sys.path.append("../")

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


if __name__ == '__main__1':
    (sim_results, simrunner) = simulate_psk_with_runner()

    # Calculates the error rates
    num_bits = np.array(sim_results.get_result_values_list('num_bits'))
    bit_errors = np.array(sim_results.get_result_values_list('bit_errors'))
    error_rate = bit_errors.astype(float) / num_bits
    print error_rate
