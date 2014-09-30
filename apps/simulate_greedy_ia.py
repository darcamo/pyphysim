#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module containing simulation runners for the several Interference
Alignment algorithms in the algorithms.ia module.
"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
try:
    parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    sys.path.append(parent_dir)
except NameError:
    sys.path.append('../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx Import Statements xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
from time import time
import numpy as np
# from pprint import pprint

from pyphysim.simulations.runner import SimulationRunner
from pyphysim.simulations.parameters import SimulationParameters
from pyphysim.simulations.results import SimulationResults, Result
from pyphysim.simulations.simulationhelpers import simulate_do_what_i_mean
from pyphysim.comm import modulators, channels
from pyphysim.util.conversion import dB2Linear
from pyphysim.util import misc
from pyphysim.ia import algorithms
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class IASimulationRunner(SimulationRunner):
    """
    Base class for the simulation runners here in this simulator.

    Most of the code in the simulation runners for the different algorithms
    is the same and thus we put it here in the IASimulationRunner class.

    Parameters
    ----------
    default_config_file : string
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values in the `spec`.
    alg : str
        The top algorithm to run. This can be either 'greedy' or 'brute'.
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """

    def __init__(self, default_config_file, alg='greedy',
                 read_command_line_args=True):
        """
        Constructor of the IASimulationRunner class.
        """

        spec = """[Scenario]
        SNR=real_numpy_array(min=-50, max=100, default=0:5:31)
        M=integer(min=4, max=512, default=4)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="PSK")
        NSymbs=integer(min=10, max=1000000, default=200)
        K=integer(min=2,default=3)
        Nr=integer_scalar_or_integer_numpy_array_check(min=2,default=2)
        Nt=integer_scalar_or_integer_numpy_array_check(min=2,default=2)
        Ns=integer_scalar_or_integer_numpy_array_check(min=1,default=1)
        [IA Algorithm]
        max_iterations=integer_numpy_array(min=1, default=60)
        initialize_with=string_list(default=list('random'))
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        SimulationRunner.__init__(
            self, default_config_file, spec, read_command_line_args)

        # Set the max_bit_errors and rep_max attributes
        self.max_bit_errors = self.params['max_bit_errors']
        self.rep_max = self.params['rep_max']

        # Create the modulator object
        M = self.params['M']
        modulator_options = {'PSK': modulators.PSK,
                             'QPSK': modulators.QPSK,
                             'QAM': modulators.QAM,
                             'BPSK': modulators.BPSK}
        self.modulator = modulator_options[self.params['modulator']](M)

        # Create the channel object
        self.multiUserChannel = channels.MultiUserChannelMatrix()

        # Create the basic IA Solver object
        self.ia_solver = algorithms.MMSEIASolver(self.multiUserChannel)

        # Create the 'top' IA solver object: either "greedy" or the "brute
        # force". This object will use the basic IA solver object.
        if alg == 'greedy':
            self.ia_top_object = algorithms.GreedStreamIASolver(self.ia_solver)
        elif alg == 'brute':
            self.ia_top_object = algorithms.BruteForceStreamIASolver(
                self.ia_solver)
        else:
            raise ValueError("Invalid choice: '{0}'".format(alg))

        # This can be either 'screen' or 'file'. If it is 'file' then the
        # progressbar will write the progress to a file with appropriated
        # filename
        self.progress_output_type = 'screen'

        # xxxxxxxxxx Set the progressbar message xxxxxxxxxxxxxxxxxxxxxxxxxx
        self.progressbar_message = "SNR: {{SNR}}".format(
            self.modulator.name)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _run_simulation(self,   # pylint: disable=R0914,R0915
                        current_parameters):
        # xxxxx Input parameters (set in the constructor) xxxxxxxxxxxxxxxxx
        M = self.modulator.M
        NSymbs = current_parameters["NSymbs"]
        K = current_parameters["K"]
        Nr = current_parameters["Nr"]
        Nt = current_parameters["Nt"]
        Ns = current_parameters["Ns"]
        SNR = current_parameters["SNR"]

        # Store the original (maximum) number of streams for each user for
        # later usage
        if isinstance(Ns, int):
            orig_Ns = np.ones(K, dtype=int) * Ns
        else:
            orig_Ns = Ns.copy()

        # Dependent parameters
        noise_var = 1 / dB2Linear(SNR)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calc. precoders and receive filters for IA xxxxxxxxxxxxxxxx
        # We need to perform IA before generating any data so that we know
        # how many streams we need to send (and thus generate data. Note
        # that it is not always equal to Ns. It can be lower for some user
        # if the IA algorithm chooses a precoder that sends zero energy in
        # some stream.
        self.multiUserChannel.randomize(Nr, Nt, K)
        self.multiUserChannel.noise_var = noise_var

        self.ia_solver.clear()
        self.ia_solver.initialize_with = current_parameters['initialize_with']
        self.ia_top_object.solve(Ns=Ns, P=1)

        # If any of the Nr, Nt or Ns variables were integers (meaning all
        # users have the same value) we will convert them by numpy arrays
        # with correct size (K).
        Nr = self.ia_solver.Nr
        Nt = self.ia_solver.Nt
        Ns = self.ia_solver.Ns

        cumNs = np.cumsum(self.ia_solver.Ns)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Input Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # inputData has the data of all users (vertically stacked)
        inputData = np.random.randint(0, M, [np.sum(Ns), NSymbs])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Modulate input data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # modulatedData has the data of all users (vertically stacked)
        modulatedData = self.modulator.modulate(inputData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Perform the Interference Alignment xxxxxxxxxxxxxxxxxxx
        # Split the data. transmit_signal will be a list and each element
        # is a numpy array with the data of a user
        transmit_signal = np.split(modulatedData, cumNs[:-1])
        transmit_signal_precoded = map(
            np.dot, self.ia_solver.full_F, transmit_signal)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        multi_user_channel = self.ia_solver._multiUserChannel
        # received_data is an array of matrices, one matrix for each receiver.
        received_data = multi_user_channel.corrupt_data(
            transmit_signal_precoded)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Perform the Interference Cancelation xxxxxxxxxxxxxxxxxxxxxx
        received_data_no_interference = map(
            np.dot, self.ia_solver.full_W_H, received_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Demodulate Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        received_data_no_interference = np.vstack(
            received_data_no_interference)
        demodulated_data = self.modulator.demodulate(
            received_data_no_interference)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the symbol and bit error rates xxxxxxxxxxxxxxxxx
        symbolErrors = np.sum(inputData != demodulated_data)
        bitErrors = misc.count_bit_errors(inputData, demodulated_data)
        numSymbols = inputData.size
        numBits = inputData.size * modulators.level2bits(M)
        ia_cost = self.ia_solver.get_cost()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Calculates the Sum Capacity xxxxxxxxxxxxxxxxxxxxxxxxxx
        sirn_all_k = self.ia_solver.calc_SINR()
        calc_capacity = lambda sirn: np.sum(np.log2(1 + sirn))
        # Array with the sum capacity of each user
        sum_capacity = map(calc_capacity, sirn_all_k)
        # Total sum capacity
        total_sum_capacity = np.sum(sum_capacity)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Number of iterations of the IA algorithm xxxxxxxxxxxxx
        ia_runned_iterations = self.ia_solver.runned_iterations
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Return the simulation results xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        symbolErrorsResult = Result.create(
            "symbol_errors", Result.SUMTYPE, symbolErrors)

        numSymbolsResult = Result.create(
            "num_symbols", Result.SUMTYPE, numSymbols)

        bitErrorsResult = Result.create(
            "bit_errors", Result.SUMTYPE, bitErrors)

        numBitsResult = Result.create("num_bits", Result.SUMTYPE, numBits)

        berResult = Result.create("ber", Result.RATIOTYPE, bitErrors, numBits,
                                  accumulate_values=False)

        serResult = Result.create("ser", Result.RATIOTYPE, symbolErrors,
                                  numSymbols, accumulate_values=False)

        ia_costResult = Result.create(
            "ia_cost", Result.RATIOTYPE, ia_cost, 1, accumulate_values=False)

        sum_capacityResult = Result.create(
            "sum_capacity", Result.RATIOTYPE, total_sum_capacity, 1,
            accumulate_values=False)

        ia_runned_iterationsResult = Result.create(
            "ia_runned_iterations", Result.RATIOTYPE, ia_runned_iterations, 1,
            accumulate_values=False)

        # xxxxxxxxxx chosen stream configuration index xxxxxxxxxxxxxxxxxxxx
        # Interpret Ns as a multidimensional index
        stream_index_multi = Ns - 1
        # Convert to a 1D index suitable for storing
        stream_index = int(np.ravel_multi_index(stream_index_multi, orig_Ns))
        num_choices = np.prod(orig_Ns)
        stream_statistics = Result.create(
            "stream_statistics", Result.CHOICETYPE, stream_index, num_choices)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        simResults = SimulationResults()
        simResults.add_result(symbolErrorsResult)
        simResults.add_result(numSymbolsResult)
        simResults.add_result(bitErrorsResult)
        simResults.add_result(numBitsResult)
        simResults.add_result(berResult)
        simResults.add_result(serResult)
        simResults.add_result(ia_costResult)
        simResults.add_result(sum_capacityResult)
        simResults.add_result(ia_runned_iterationsResult)
        simResults.add_result(stream_statistics)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return simResults

    def _keep_going(self, current_params, current_sim_results, current_rep):
        """
        Check if the simulation should continue or stop.

        Parameters
        ----------
        current_params : SimulationParameters object
            SimulationParameters object with the parameters of the
            simulation.
        current_sim_results : SimulationResults object
            SimulationResults object from the last iteration (merged with
            all the previous results)
        current_rep : int
            Number of iterations already run.

        Returns
        -------
        result : bool
            True if the simulation should continue or False otherwise.
        """
        # For each multiple of 300 iterations we test if the length of the
        # confidence interval is greater then one tenth of the actual
        # value. If it is that means that we still need to run more
        # iterations and thus re return True. If it is not, than we can
        # stop the iterations for the current parameters and thus we return
        # false. This choice was arbitrarily, but seems reasonable.
        if current_rep % 300 == 0:
            ber_result = current_sim_results['ber'][-1]
            ber_value = ber_result.get_result()
            if ber_value == 0.0:
                return True
            else:
                conf_interval = ber_result.get_confidence_interval(P=95)
                error = np.abs(conf_interval[1] - conf_interval[0])

                # If error is lower then one fifth of the current result
                # and we have runned at least 5000 iterations, then we have
                # enough and we return False to indicate the simulation of
                # the current parameters can stop.
                if error < ber_value / 10.0 and current_rep > 5000:
                    return False

        return True

    # Except for the closed form algorithm, all the other algorithms
    # algorithms are iterative and we need to set the maximum number of
    # iterations of the iterative algorithm. We do this by implementing the
    # _on_simulate_current_params_start method.
    #
    # Here we will both set the max_iterations and the initialize_with
    # parameter. Re-implement this method any subclass that does not need
    # them.
    def _on_simulate_current_params_start(self, current_params):
        self.multiUserChannel.re_seed()
        self.ia_solver.max_iterations = current_params['max_iterations']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Main xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def main_simulate():
    """Function called to perform the simulation.
    """
    tic = time()

    base_name = ("results_{SNR}_{M}-{modulator}_{Nr}x{Nt}_({Ns})_MaxIter"
                 "_{max_iterations}_{initialize_with}")

    greedy_runner = IASimulationRunner('greedy_config_file.txt', 'greedy')
    greedy_runner.set_results_filename("greedy_{0}".format(base_name))

    brute_runner = IASimulationRunner('greedy_config_file.txt', 'brute')
    brute_runner.set_results_filename("brute_force_{0}".format(base_name))

    simulate_do_what_i_mean([greedy_runner, brute_runner], parent_dir)

    toc = time()
    print "Total Elapsed Time: {0}".format(misc.pretty_time(toc - tic))


# This function is only used in the implementation of the main_plot
# function.
def _plot_ber(simulationresults_obj, fixed_params, ax, label, fmt):
    """
    Function with the common code to plot the BER.
    """
    # Get the SNR
    SNR = np.array(simulationresults_obj.params['SNR'])

    # Get the BER and BER eror interval limits
    ber = simulationresults_obj.get_result_values_list(
        'ber',
        fixed_params=fixed_params)
    ber_CF = simulationresults_obj.get_result_values_confidence_intervals(
        'ber',
        P=95,
        fixed_params=fixed_params)
    ber_errors = np.abs([i[1] - i[0] for i in ber_CF])

    ax.errorbar(SNR, ber, ber_errors, fmt=fmt, elinewidth=2.0, label=label)


# This function is only used in the implementation of the main_plot
# function.
def _plot_sum_capacity(simulationresults_obj, fixed_params, ax, label, fmt):
    """
    Function with the common code to plot the Sum Capacity.
    """
    # Get the SNR
    SNR = np.array(simulationresults_obj.params['SNR'])

    sum_capacity = simulationresults_obj.get_result_values_list(
        'sum_capacity',
        fixed_params=fixed_params)
    sum_capacity_CF \
        = simulationresults_obj.get_result_values_confidence_intervals(
            'sum_capacity', P=95, fixed_params=fixed_params)
    sum_capacity_errors = np.abs([i[1] - i[0] for i in sum_capacity_CF])

    ax.errorbar(SNR, sum_capacity, sum_capacity_errors,
                fmt=fmt, elinewidth=2.0, label=label)


def main_plot(index=0):  # pylint: disable=R0914,R0915
    """
    Function called to plot the results from a previous simulation.
    """
    from matplotlib import pyplot as plt

    config_file = 'greedy_config_file.txt'

    # xxxxxxxxxx Config spec for the config file xxxxxxxxxxxxxxxxxxxxxxxxxx
    spec = """[Scenario]
        SNR=real_numpy_array(min=-50, max=100, default=0:5:31)
        M=integer(min=4, max=512, default=4)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="PSK")
        NSymbs=integer(min=10, max=1000000, default=200)
        K=integer(min=2,default=3)
        Nr=integer_scalar_or_integer_numpy_array_check(min=2,default=2)
        Nt=integer_scalar_or_integer_numpy_array_check(min=2,default=2)
        Ns=integer_scalar_or_integer_numpy_array_check(min=1,default=1)
        [IA Algorithm]
        max_iterations=integer_numpy_array(min=1, default=60)
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        [Plot]
        max_iterations_plot=integer(default=5)
        initialize_with_plot=option('random', 'alt_min', default='random')
        """.split("\n")
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    params = SimulationParameters.load_from_config_file(config_file, spec)
    max_iterations = params['max_iterations_plot']
    # initialize_with = params['initialize_with_plot']
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Results base name xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    base_name = ("results_{SNR}_{M}-{modulator}_{Nr}x{Nt}_({Ns})_MaxIter"
                 "_{max_iterations}_{initialize_with}")
    base_name = misc.replace_dict_values(base_name, params.parameters, True)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig2, ax2 = plt.subplots(nrows=1, ncols=1)

    greedy_results = SimulationResults.load_from_file(
        'greedy_{0}.pickle'.format(base_name))
    brute_force_results = SimulationResults.load_from_file(
        'brute_force_{0}.pickle'.format(base_name))

    # We only get the parameters from the greedy object, since we use the
    # same parameters for greedy and brute force
    parameters_dict = greedy_results.params.parameters
    fixed_params = {'max_iterations': max_iterations}

    _plot_ber(greedy_results, fixed_params, ax, 'Greedy', '-r*')
    _plot_sum_capacity(greedy_results, fixed_params, ax2, 'Greedy', '-r*')
    _plot_ber(brute_force_results, fixed_params, ax, 'Brute Force', '-b*')
    _plot_sum_capacity(
        brute_force_results, fixed_params, ax2, 'Brute Force', '-b*')

    # xxxxxxxxxx BER Plot Options xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ax.set_xlabel('SNR')
    ax.set_ylabel('BER')
    title = ("BER for Different Algorithms ({max_iterations} Max Iterations)\n"
             "K={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}")
    title = title.replace("{max_iterations}", str(max_iterations))
    ax.set_title(title.format(**parameters_dict))

    ax.set_yscale('log')
    ax.legend(fancybox=True, shadow=True, loc='best')
    ax.grid(True, which='both', axis='both')

    # plt.show(block=False)
    fig.savefig('ber_all_ia_algorithms.pgf')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Sum Capacity Plot Options xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ax2.set_xlabel('SNR')
    ax2.set_ylabel('Sum Capacity')
    title = ("Sum Capacity for Different Algorithms ({max_iterations} Max "
             "Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}")
    title = title.replace("{max_iterations}", str(max_iterations))
    ax2.set_title(title.format(**parameters_dict))

    ax2.legend(fancybox=True, shadow=True, loc=2)
    ax2.grid(True, which='both', axis='both')
    # plt.show()
    fig2.savefig('sum_capacity_all_ia_algorithms.pgf')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    plt.show()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Main xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    from apps.simulate_greedy_ia import IASimulationRunner
    main_simulate()
    # main_plot()
