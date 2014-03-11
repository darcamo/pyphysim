#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing simulation runners for the several Interference
Alignment algorithms in the ia.ia module."""

__revision__ = "$Revision$"

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxx Import Statements xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import numpy as np
from pprint import pprint

from pyphysim.simulations.core import SimulationRunner, SimulationParameters, SimulationResults, Result, get_common_parser
from pyphysim.simulations.simulationhelpers import simulate_do_what_i_mean
from pyphysim.comm import modulators, channels
from pyphysim.util.conversion import dB2Linear
from pyphysim.util import misc
from pyphysim.ia import ia
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class IASimulationRunner(SimulationRunner):
    """
    Base class for the IA simulation runners.

    Most of the code in the simulation runners for the different IA
    algorithms is the same and is here in the IASimulationRunner class.

    Parameters
    ----------
    IaSolverClass : The class of the IA solver object
        The IA solver class, which should be a derived class from
        ia.IASolverBaseClass
    default_config_file : string
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values in the `spec`.
    spec : List of strings
        The specification used to read the simulation configuration
        from the file `default_config_file`. See the validation part in the
        docummentation of the configobj module for details.
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """

    def __init__(self, IaSolverClass, default_config_file, spec, read_command_line_args=True):
        SimulationRunner.__init__(self,
                                  default_config_file,
                                  spec,
                                  read_command_line_args)

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

        # Create the IA Solver object
        self.ia_solver = IaSolverClass(self.multiUserChannel)

        # For the ClosedFormIASolver class we manually add a
        # _runned_iterations member variable with value of 0. This member
        # variable is not used and does not exist in the ClosedFormIASolver
        # solver. However, since we will store this variable as a result
        # for each algorithm we manually added to the ClosedFormIASolver
        # object just to make the code in _run_simulation equal for all
        # solvers.
        if isinstance(self.ia_solver, ia.ClosedFormIASolver):
            self.ia_solver.runned_iterations = 0.0

        # This can be either 'screen' or 'file'. If it is 'file' then the
        # progressbar will write the progress to a file with appropriated
        # filename
        self.progress_output_type = 'screen'

        # xxxxxxxxxx Set the progressbar message xxxxxxxxxxxxxxxxxxxxxxxxxx
        self.progressbar_message = "SNR: {{SNR}}".format(
            self.modulator.name)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _run_simulation(self, current_parameters):  # pylint: disable=R0914,R0915
        # xxxxx Input parameters (set in the constructor) xxxxxxxxxxxxxxxxx
        M = self.modulator.M
        NSymbs = current_parameters["NSymbs"]
        K = current_parameters["K"]
        Nr = np.ones(K, dtype=int) * current_parameters["Nr"]
        Nt = np.ones(K, dtype=int) * current_parameters["Nt"]
        Ns = np.ones(K, dtype=int) * current_parameters["Ns"]
        SNR = current_parameters["SNR"]

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
        # We wouldn't need to explicitly set self.ia_solver.noise_var
        # variable if the multiUserChannel object had the correct value at
        # this point.
        self.ia_solver.noise_var = noise_var
        self.ia_solver.clear()
        self.ia_solver.solve(Ns)

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
        transmit_signal_precoded = map(np.dot, self.ia_solver.F, transmit_signal)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        multi_user_channel = self.ia_solver._multiUserChannel
        # received_data is an array of matrices, one matrix for each receiver.
        received_data = multi_user_channel.corrupt_data(
            transmit_signal_precoded, noise_var)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Perform the Interference Cancelation xxxxxxxxxxxxxxxxxxxxxx
        received_data_no_interference = map(np.dot,
                                            self.ia_solver.full_W_H, received_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Demodulate Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        received_data_no_interference = np.vstack(received_data_no_interference)
        demodulated_data = self.modulator.demodulate(received_data_no_interference)
        # demodulated_data = map(self.modulator.demodulate, received_data_no_interference)
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

        #xxxxxxxxxx Number of iterations of the IA algorithm xxxxxxxxxxxxxx
        ia_runned_iterations = self.ia_solver.runned_iterations
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Return the simulation results xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        symbolErrorsResult = Result.create(
            "symbol_errors", Result.SUMTYPE, symbolErrors)

        numSymbolsResult = Result.create(
            "num_symbols", Result.SUMTYPE, numSymbols)

        bitErrorsResult = Result.create("bit_errors", Result.SUMTYPE, bitErrors)

        numBitsResult = Result.create("num_bits", Result.SUMTYPE, numBits)

        berResult = Result.create("ber", Result.RATIOTYPE, bitErrors, numBits,
                                  accumulate_values=False)

        serResult = Result.create(
            "ser", Result.RATIOTYPE, symbolErrors, numSymbols, accumulate_values=False)

        ia_costResult = Result.create(
            "ia_cost", Result.RATIOTYPE, ia_cost, 1, accumulate_values=False)

        sum_capacityResult = Result.create(
            "sum_capacity", Result.RATIOTYPE, total_sum_capacity, 1,
            accumulate_values=False)

        ia_runned_iterationsResult = Result.create(
            "ia_runned_iterations", Result.RATIOTYPE, ia_runned_iterations, 1, accumulate_values=False)

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
                conf_interval = ber_result.get_confidence_interval()
                error = np.abs(conf_interval[1] - conf_interval[0])

                # If error is lower then one fifth of the current result
                # and we have runned at least 5000 iterations, then we have
                # enough and we return False to indicate the simulation of
                # the current parameters can stop.
                if error < ber_value / 10.0 and current_rep > 5000:
                    return False

        return True

    # Except for the closed form algorithm, all the other ia algorithms are
    # iterative and we need to set the maximum number of iterations of the
    # iterative algorithm. We do this by implementing the
    # _on_simulate_current_params_start method.
    #
    # Furthermore, since we create the channel object in the __init__
    # method, we need to re-seed the channel for each set of parameters.
    def _on_simulate_current_params_start(self, current_params):
        self.multiUserChannel.re_seed()
        self.ia_solver.max_iterations = current_params['max_iterations']


class AlternatingSimulationRunner(IASimulationRunner):
    """
    Implements a simulation runner for a transmission with the Alternating
    Minimizations Interference Alignment Algorithm.

    Parameters
    ----------
    default_config_file : string
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """

    def __init__(self, default_config_file, read_command_line_args=True):
        spec = """[Scenario]
        SNR=real_numpy_array(min=-50, max=100, default=0:5:31)
        M=integer(min=4, max=512, default=4)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="PSK")
        NSymbs=integer(min=10, max=1000000, default=200)
        K=integer(min=2,default=3)
        Nr=integer(min=2,default=2)
        Nt=integer(min=2,default=2)
        Ns=integer(min=1,default=1)
        [IA Algorithm]
        max_iterations=integer_numpy_array(min=1, default=60)
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        IASimulationRunner.__init__(self,
                                    ia.AlternatingMinIASolver,
                                    default_config_file,
                                    spec,
                                    read_command_line_args)

        #self.update_progress_function_style = None


class ClosedFormSimulationRunner(IASimulationRunner):
    """
    Implements a simulation runner for a transmission with the Closed-Form
    Interference Alignment Algorithm.

    Parameters
    ----------
    default_config_file : string
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """

    def __init__(self, default_config_file, read_command_line_args=True):
        spec = """[Scenario]
        SNR=real_numpy_array(min=-50, max=100, default=0:5:31)
        M=integer(min=4, max=512, default=4)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="PSK")
        NSymbs=integer(min=10, max=1000000, default=200)
        K=integer(min=2,default=3)
        Nr=integer(min=2,default=2)
        Nt=integer(min=2,default=2)
        Ns=integer(min=1,default=1)
        [IA Algorithm]
        max_iterations=integer_numpy_array(min=1, default=60)
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        IASimulationRunner.__init__(self,
                                    ia.ClosedFormIASolver,
                                    default_config_file,
                                    spec,
                                    read_command_line_args)

    # Since we create the channel object in the __init__ method of
    # IASimulationRunner, we need to re-seed the channel for each set of
    # parameters.
    def _on_simulate_current_params_start(self, current_params):
        self.multiUserChannel.re_seed()


class MinLeakageSimulationRunner(IASimulationRunner):
    """
    Implements a simulation runner for a transmission with the Minimum
    Leakage Interference Alignment Algorithm.

    Parameters:
    -----------
    default_config_file : str
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """

    def __init__(self, default_config_file, read_command_line_args=True):
        spec = """[Scenario]
        SNR=real_numpy_array(min=-50, max=100, default=0:5:31)
        M=integer(min=4, max=512, default=4)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="PSK")
        NSymbs=integer(min=10, max=1000000, default=200)
        K=integer(min=2,default=3)
        Nr=integer(min=2,default=2)
        Nt=integer(min=2,default=2)
        Ns=integer(min=1,default=1)
        [IA Algorithm]
        max_iterations=integer_numpy_array(min=1, default=60)
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        IASimulationRunner.__init__(self,
                                    ia.MinLeakageIASolver,
                                    default_config_file,
                                    spec,
                                    read_command_line_args)


class MaxSINRSimulationRunner(IASimulationRunner):
    """
    Implements a simulation runner for a transmission with the MaxSINR
    Minimizations Interference Alignment Algorithm.

    Parameters:
    -----------
    default_config_file : str
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """
    def __init__(self, default_config_file, read_command_line_args=True):
        spec = """[Scenario]
        SNR=real_numpy_array(min=-50, max=100, default=0:5:31)
        M=integer(min=4, max=512, default=4)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="PSK")
        NSymbs=integer(min=10, max=1000000, default=200)
        K=integer(min=2,default=3)
        Nr=integer(min=2,default=2)
        Nt=integer(min=2,default=2)
        Ns=integer(min=1,default=1)
        [IA Algorithm]
        max_iterations=integer_numpy_array(min=1, default=60)
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        IASimulationRunner.__init__(self,
                                    ia.MaxSinrIASolver,
                                    default_config_file,
                                    spec,
                                    read_command_line_args)


class MMSESimulationRunner(IASimulationRunner):
    """
    Implements a simulation runner for a transmission with the MMSE based
    Interference Alignment Algorithm.

    Parameters:
    -----------
    default_config_file : str
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """
    def __init__(self, default_config_file, read_command_line_args=True):
        spec = """[Scenario]
        SNR=real_numpy_array(min=-50, max=100, default=0:5:31)
        M=integer(min=4, max=512, default=4)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="PSK")
        NSymbs=integer(min=10, max=1000000, default=200)
        K=integer(min=2,default=3)
        Nr=integer(min=2,default=2)
        Nt=integer(min=2,default=2)
        Ns=integer(min=1,default=1)
        [IA Algorithm]
        max_iterations=integer_numpy_array(min=1, default=60)
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        IASimulationRunner.__init__(self,
                                    ia.MMSEIASolver,
                                    default_config_file,
                                    spec,
                                    read_command_line_args)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Main xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
## xxxxxxxxxx Main - Perform the simulations xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# This will perform the simulation
if __name__ == '__main__':
    from time import time
    from pyphysim.util.misc import pretty_time
    from apps.simulate_ia import ClosedFormSimulationRunner, AlternatingSimulationRunner, MMSESimulationRunner, MaxSINRSimulationRunner, MinLeakageSimulationRunner

    tic = time()

    # xxxxxxxxxx Command Line options xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # The returned parser already has two possiblr arguments: "config" and
    # "index".
    parser = get_common_parser()
    parser.description = "Simulate several Interference Alignment Algorithms"
    group = parser.add_argument_group('IA Algorithms to Simulate. Default is all of them.')
    group.add_argument('--closed_form', action="store_true", default=False, help="Simulate the Closed Form algorithm.")
    group.add_argument('--alt_min', action="store_true", default=False, help="Simulate the Alternating Minimizations algorithm.")
    group.add_argument('--max_sinr', action="store_true", default=False, help="Simulate the Max SINR algorithm.")
    group.add_argument('--mmse', action="store_true", default=False, help="Simulate the MMSE algorithm.")

    args = parser.parse_args()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Determine which algorithms should be simulated xxxxxxxxxxx
    algorithms_to_simulate = []

    simulate_all_algorithms = True
    if args.closed_form is True:
        algorithms_to_simulate.append("Closed Form")
        simulate_all_algorithms = False

    if args.alt_min is True:
        algorithms_to_simulate.append("Alt Min")
        simulate_all_algorithms = False

    if args.max_sinr is True:
        algorithms_to_simulate.append("Max SINR")
        simulate_all_algorithms = False

    if args.mmse is True:
        algorithms_to_simulate.append("MMSE")
        simulate_all_algorithms = False

    if simulate_all_algorithms is True:
        algorithms_to_simulate = [
            "Closed Form",
            "Alt Min",
            "Max SINR",
            "MMSE"]
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    all_simulation_runner_objs = []

    # xxxxxxxxxx Closed Form Runner xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if "Closed Form" in algorithms_to_simulate:
        print "Simulating Closed Form algorithm"
        closed_form_runner = ClosedFormSimulationRunner('ia_config_file.txt')
        #del closed_form_runner.params.parameters['max_iterations']
        try:
            closed_form_runner.params.set_unpack_parameter('max_iterations', False)
        except KeyError as e:
            pass

        pprint(closed_form_runner.params.parameters)
        print("IA Solver: {0}\n".format(closed_form_runner.ia_solver.__class__))
        closed_form_runner.set_results_filename('ia_closed_form_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_MaxIter_{max_iterations}')
        all_simulation_runner_objs.append(closed_form_runner)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Alt. Min. Runner xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if "Alt Min" in algorithms_to_simulate:
        print "Simulating Alternating Minimizations algorithm"
        alt_min_runner = AlternatingSimulationRunner('ia_config_file.txt')
        pprint(alt_min_runner.params.parameters)
        print("IA Solver: {0}\n".format(alt_min_runner.ia_solver.__class__))
        alt_min_runner.set_results_filename(
        'ia_alt_min_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_MaxIter_{max_iterations}')
        all_simulation_runner_objs.append(alt_min_runner)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Max SINR Runner xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if "Max SINR" in algorithms_to_simulate:
        print "Simulating Max SINR algorithm"
        max_sinrn_runner = MaxSINRSimulationRunner('ia_config_file.txt')
        pprint(max_sinrn_runner.params.parameters)
        print("IA Solver: {0}\n".format(max_sinrn_runner.ia_solver.__class__))
        max_sinrn_runner.set_results_filename(
        'ia_max_sinr_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_MaxIter_{max_iterations}')
        all_simulation_runner_objs.append(max_sinrn_runner)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx MMSE Runner xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if "MMSE" in algorithms_to_simulate:
        print "Simulating MMSE algorithm"
        mmse_runner = MMSESimulationRunner('ia_config_file.txt')
        pprint(mmse_runner.params.parameters)
        print("IA Solver: {0}\n".format(mmse_runner.ia_solver.__class__))
        mmse_runner.set_results_filename('ia_mmse_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_MaxIter_{max_iterations}')
        all_simulation_runner_objs.append(mmse_runner)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    simulate_do_what_i_mean(all_simulation_runner_objs, parent_dir)

    # xxxxxxxxxx Some finalization message xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    toc = time()
    print "Total Elapsed Time: {0}".format(pretty_time(toc - tic))
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


## xxxxxxxxxx Plot xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# This will read results from a previous simulation and draw some plots
if __name__ == '__main__1':
    from matplotlib import pyplot as plt

    # xxxxx Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    params = SimulationParameters.load_from_config_file('ia_config_file.txt')
    K = params['K']
    Nr = params['Nr']
    Nt = params['Nt']
    Ns = params['Ns']
    max_iterations = params['max_iterations']
    M = params['M']
    modulator = params['modulator']
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Results base name xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    base_name = 'results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_MaxIter_[120]'.format(**params.parameters)
    base_name_no_iter = 'results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_MaxIter_[120]'.format(**params.parameters)  # Used only for the closed form algorithm, which is not iterative
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    alt_min_results = SimulationResults.load_from_file(
        'ia_alt_min_{0}.pickle'.format(base_name))
    closed_form_results = SimulationResults.load_from_file(
        'ia_closed_form_{0}.pickle'.format(base_name_no_iter))
    # closed_form_first_results = SimulationResults.load_from_file(
    #     'ia_closed_form_first_init_{0}.pickle'.format(base_name))
    max_sinrn_results = SimulationResults.load_from_file(
        'ia_max_sinr_{0}.pickle'.format(base_name))
    # min_leakage_results = SimulationResults.load_from_file(
    #     'ia_min_leakage_{0}.pickle'.format(base_name))
    mmse_results = SimulationResults.load_from_file(
        'ia_mmse_{0}.pickle'.format(base_name))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Plot BER (all) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    SNR_alt_min = np.array(alt_min_results.params['SNR'])
    SNR_closed_form = np.array(closed_form_results.params['SNR'])
    SNR_max_SINR = np.array(max_sinrn_results.params['SNR'])
    # SNR_min_leakage = np.array(min_leakage_results.params['SNR'])
    SNR_mmse = np.array(mmse_results.params['SNR'])

    max_iterations = 120

    ber_alt_min = alt_min_results.get_result_values_list(
        'ber',
        fixed_params={'max_iterations': max_iterations})
    ber_CF_alt_min = alt_min_results.get_result_values_confidence_intervals(
        'ber',
        P=95,
        fixed_params={'max_iterations': max_iterations})
    ber_errors_alt_min = np.abs([i[1] - i[0] for i in ber_CF_alt_min])

    ber_closed_form = closed_form_results.get_result_values_list(
        'ber',
        fixed_params={'max_iterations': max_iterations})
    ber_CF_closed_form = closed_form_results.get_result_values_confidence_intervals(
        'ber',
        P=95,
        fixed_params={'max_iterations': max_iterations})
    ber_errors_closed_form = np.abs([i[1] - i[0] for i in ber_CF_closed_form])

    # ber_closed_form_first = closed_form_first_results.get_result_values_list('ber')
    # ber_CF_closed_form_first = closed_form_first_results.get_result_values_confidence_intervals('ber', P=95)
    # ber_errors_closed_form_first = np.abs([i[1] - i[0] for i in ber_CF_closed_form_first])

    ber_max_sinr = max_sinrn_results.get_result_values_list(
        'ber',
        fixed_params={'max_iterations': max_iterations})
    ber_CF_max_sinr = max_sinrn_results.get_result_values_confidence_intervals(
        'ber',
        P=95,
        fixed_params={'max_iterations': max_iterations})
    ber_errors_max_sinr = np.abs([i[1] - i[0] for i in ber_CF_max_sinr])

    # ber_min_leakage = min_leakage_results.get_result_values_list('ber')
    # ber_CF_min_leakage = min_leakage_results.get_result_values_confidence_intervals('ber', P=95)
    # ber_errors_min_leakage = np.abs([i[1] - i[0] for i in ber_CF_min_leakage])

    ber_mmse = mmse_results.get_result_values_list(
        'ber',
        fixed_params={'max_iterations': max_iterations})
    ber_CF_mmse = mmse_results.get_result_values_confidence_intervals(
        'ber',
        P=95,
        fixed_params={'max_iterations': max_iterations})
    ber_errors_mmse = np.abs([i[1] - i[0] for i in ber_CF_mmse])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.errorbar(SNR_alt_min, ber_alt_min, ber_errors_alt_min, fmt='-r*', elinewidth=2.0, label='Alt. Min.')
    ax.errorbar(SNR_closed_form, ber_closed_form, ber_errors_closed_form, fmt='-b*', elinewidth=2.0, label='Closed Form')
    ax.errorbar(SNR_max_SINR, ber_max_sinr, ber_errors_max_sinr, fmt='-g*', elinewidth=2.0, label='Max SINR')
    # ax.errorbar(SNR, ber_min_leakage, ber_errors_min_leakage, fmt='-k*', elinewidth=2.0, label='Min Leakage.')
    ax.errorbar(SNR_mmse, ber_mmse, ber_errors_mmse, fmt='-m*', elinewidth=2.0, label='MMSE.')

    plt.xlabel('SNR')
    plt.ylabel('BER')
    title = 'BER for Different Algorithms ({max_iterations} Max Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}'.replace("{max_iterations}", str(max_iterations))
    plt.title(title.format(**alt_min_results.params.parameters))

    ax.set_yscale('log')
    leg = ax.legend(fancybox=True, shadow=True, loc='best')
    ax.grid(True, which='both', axis='both')
    plt.show(block=False)
    fig.savefig('ber_all_ia_algorithms.pgf')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Plot Sum Capacity (all) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    sum_capacity_alt_min = alt_min_results.get_result_values_list(
        'sum_capacity',
        fixed_params={'max_iterations': max_iterations})
    sum_capacity_CF_alt_min = alt_min_results.get_result_values_confidence_intervals(
        'sum_capacity',
        P=95,
        fixed_params={'max_iterations': max_iterations})
    sum_capacity_errors_alt_min = np.abs([i[1] - i[0] for i in sum_capacity_CF_alt_min])

    sum_capacity_closed_form = closed_form_results.get_result_values_list(
        'sum_capacity',
        fixed_params={'max_iterations': max_iterations})
    sum_capacity_CF_closed_form = closed_form_results.get_result_values_confidence_intervals(
        'sum_capacity',
        P=95,
        fixed_params={'max_iterations': max_iterations})
    sum_capacity_errors_closed_form = np.abs([i[1] - i[0] for i in sum_capacity_CF_closed_form])

    sum_capacity_max_sinr = max_sinrn_results.get_result_values_list(
        'sum_capacity',
        fixed_params={'max_iterations': max_iterations})
    sum_capacity_CF_max_sinr = max_sinrn_results.get_result_values_confidence_intervals(
        'sum_capacity',
        P=95,
        fixed_params={'max_iterations': max_iterations})
    sum_capacity_errors_max_sinr = np.abs([i[1] - i[0] for i in sum_capacity_CF_max_sinr])

    # sum_capacity_min_leakage = min_leakage_results.get_result_values_list('sum_capacity')
    # sum_capacity_CF_min_leakage = min_leakage_results.get_result_values_confidence_intervals('sum_capacity', P=95)
    # sum_capacity_errors_min_leakage = np.abs([i[1] - i[0] for i in sum_capacity_CF_min_leakage])

    sum_capacity_mmse = mmse_results.get_result_values_list(
        'sum_capacity',
        fixed_params={'max_iterations': max_iterations})
    sum_capacity_CF_mmse = mmse_results.get_result_values_confidence_intervals(
        'sum_capacity',
        P=95,
        fixed_params={'max_iterations': max_iterations})
    sum_capacity_errors_mmse = np.abs([i[1] - i[0] for i in sum_capacity_CF_mmse])

    fig2, ax2 = plt.subplots(nrows=1, ncols=1)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ax2.errorbar(SNR_alt_min, sum_capacity_alt_min, sum_capacity_errors_alt_min, fmt='-r*', elinewidth=2.0, label='Alt. Min.')
    ax2.errorbar(SNR_closed_form, sum_capacity_closed_form, sum_capacity_errors_closed_form, fmt='-b*', elinewidth=2.0, label='Closed Form')
    ax2.errorbar(SNR_max_SINR, sum_capacity_max_sinr, sum_capacity_errors_max_sinr, fmt='-g*', elinewidth=2.0, label='Max SINR')
    # ax2.errorbar(SNR, sum_capacity_min_leakage, sum_capacity_errors_min_leakage, fmt='-k*', elinewidth=2.0, label='Min Leakage.')
    ax2.errorbar(SNR_mmse, sum_capacity_mmse, sum_capacity_errors_mmse, fmt='-m*', elinewidth=2.0, label='MMSE.')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    plt.xlabel('SNR')
    plt.ylabel('Sum Capacity')
    title = 'Sum Capacity for Different Algorithms ({max_iterations} Max Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}'.replace("{max_iterations}", str(max_iterations))
    plt.title(title.format(**alt_min_results.params.parameters))

    leg2 = ax2.legend(fancybox=True, shadow=True, loc=2)
    ax2.grid(True, which='both', axis='both')
    plt.show()
    fig2.savefig('sum_capacity_all_ia_algorithms.pgf')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
