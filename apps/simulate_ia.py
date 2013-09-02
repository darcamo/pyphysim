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
from util.simulations import SimulationRunner, SimulationParameters, SimulationResults, Result
from comm import modulators, channels
from util.conversion import dB2Linear
from util import misc
from ia import ia
import numpy as np
from pprint import pprint
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
    config_filename : string
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values in the `spec`.
    spec : List of strings
        The specification used to read the simulation configuration
        from the file `config_filename`. See the validation part in the
        docummentation of the configobj module for details.
    """

    def __init__(self, IaSolverClass, config_filename, spec):
        SimulationRunner.__init__(self)

        # Read the simulation configuration from the file. What is read and
        self.params = SimulationParameters.load_from_config_file(
            config_filename,
            spec,
            save_parsed_file=True)

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
            self.ia_solver._runned_iterations = 0.0

    def _run_simulation(self, current_parameters):
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
        sirn_all_k = self.ia_solver.calc_SINR_old()
        calc_capacity = lambda sirn: np.sum(np.log2(1 + sirn))
        # Array with the sum capacity of each user
        sum_capacity = map(calc_capacity, sirn_all_k)
        # Total sum capacity
        total_sum_capacity = np.sum(sum_capacity)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        #xxxxxxxxxx Number of iterations of the IA algorithm xxxxxxxxxxxxxx
        ia_runned_iterations = self.ia_solver._runned_iterations
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Return the simulation results xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        symbolErrorsResult = Result.create(
            "symbol_errors", Result.SUMTYPE, symbolErrors)

        numSymbolsResult = Result.create(
            "num_symbols", Result.SUMTYPE, numSymbols)

        bitErrorsResult = Result.create("bit_errors", Result.SUMTYPE, bitErrors)

        numBitsResult = Result.create("num_bits", Result.SUMTYPE, numBits)

        berResult = Result.create("ber", Result.RATIOTYPE, bitErrors, numBits,
                                  accumulate_values=True)

        serResult = Result.create(
            "ser", Result.RATIOTYPE, symbolErrors, numSymbols, accumulate_values=True)

        ia_costResult = Result.create(
            "ia_cost", Result.RATIOTYPE, ia_cost, 1, accumulate_values=True)

        sum_capacityResult = Result.create(
            "sum_capacity", Result.RATIOTYPE, total_sum_capacity, 1,
            accumulate_values=True)

        ia_runned_iterationsResult = Result.create(
            "ia_runned_iterations", Result.RATIOTYPE, ia_runned_iterations, 1, accumulate_values=True)

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


class AlternatingSimulationRunner(IASimulationRunner):
    """
    Implements a simulation runner for a transmission with the Alternating
    Minimizations Interference Alignment Algorithm.

    Parameters
    ----------
    config_filename : string
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    """

    def __init__(self, config_filename):
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
        max_iterations=integer(min=1, default=120)
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        IASimulationRunner.__init__(self,
                                    ia.AlternatingMinIASolver,
                                    config_filename,
                                    spec)

        # xxxxxxxxxx Set the progressbar message xxxxxxxxxxxxxxxxxxxxxxxxxx
        self.progressbar_message = "Alternating Min. ({0} mod.) - SNR: {{SNR}}".format(
            self.modulator.name)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Almost everything is already set in the __init__ from th
        # IASimulationRunner class. Now we set the remaining parameters
        # which are not common to all IASolvers

        # Iterations of the AlternatingMinIASolver algorithm.
        self.ia_solver.max_iterations = self.params['max_iterations']


class ClosedFormSimulationRunner(IASimulationRunner):
    """
    Implements a simulation runner for a transmission with the Closed-Form
    Interference Alignment Algorithm.

    Parameters
    ----------
    config_filename : string
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    """

    def __init__(self, config_filename):
        spec = """[Scenario]
        SNR=real_numpy_array(min=-50, max=100, default=0:5:31)
        M=integer(min=4, max=512, default=4)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="PSK")
        NSymbs=integer(min=10, max=1000000, default=200)
        K=integer(min=2,default=3)
        Nr=integer(min=2,default=2)
        Nt=integer(min=2,default=2)
        Ns=integer(min=1,default=1)
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        IASimulationRunner.__init__(self,
                                    ia.ClosedFormIASolver,
                                    config_filename,
                                    spec)

        # xxxxxxxxxx Set the progressbar message xxxxxxxxxxxxxxxxxxxxxxxxxx
        self.progressbar_message = "Closed-Form ({0} mod.) - SNR: {{SNR}}".format(self.modulator.name)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class MinLeakageSimulationRunner(IASimulationRunner):
    """
    Implements a simulation runner for a transmission with the Minimum
    Leakage Interference Alignment Algorithm.

    Parameters:
    -----------
    config_filename : str
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    """

    def __init__(self, config_filename):
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
        max_iterations=integer(min=1, default=120)
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        IASimulationRunner.__init__(self,
                                    ia.MinLeakageIASolver,
                                    config_filename,
                                    spec)

        # xxxxxxxxxx Set the progressbar message xxxxxxxxxxxxxxxxxxxxxxxxxx
        self.progressbar_message = "Min Leakage ({0} mod.) - SNR: {{SNR}}".format(self.modulator.name)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Almost everything is already set in the __init__ from th
        # IASimulationRunner class. Now we set the remaining parameters
        # which are not common to all IASolvers

        # Iterations of the AlternatingMinIASolver algorithm.
        self.ia_solver.max_iterations = self.params['max_iterations']


class MaxSINRSimulationRunner(IASimulationRunner):
    """
    Implements a simulation runner for a transmission with the MaxSINR
    Minimizations Interference Alignment Algorithm.

    Parameters:
    -----------
    config_filename : str
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    """
    def __init__(self, config_filename):
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
        max_iterations=integer(min=1, default=120)
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        IASimulationRunner.__init__(self,
                                    ia.MaxSinrIASolver,
                                    config_filename,
                                    spec)

        # xxxxxxxxxx Set the progressbar message xxxxxxxxxxxxxxxxxxxxxxxxxx
        self.progressbar_message = "Max SINR ({0} mod.) - SNR: {{SNR}}".format(self.modulator.name)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Almost everything is already set in the __init__ from th
        # IASimulationRunner class. Now we set the remaining parameters
        # which are not common to all IASolvers

        # Iterations of the AlternatingMinIASolver algorithm.
        self.ia_solver.max_iterations = self.params['max_iterations']


class MMSESimulationRunner(IASimulationRunner):
    """
    Implements a simulation runner for a transmission with the MMSE based
    Interference Alignment Algorithm.

    Parameters:
    -----------
    config_filename : str
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    """
    def __init__(self, config_filename):
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
        max_iterations=integer(min=1, default=120)
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        IASimulationRunner.__init__(self,
                                    ia.MMSEIASolver,
                                    config_filename,
                                    spec)

        # xxxxxxxxxx Set the progressbar message xxxxxxxxxxxxxxxxxxxxxxxxxx
        self.progressbar_message = "MMSE ({0} mod.) - SNR: {{SNR}}".format(self.modulator.name)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Almost everything is already set in the __init__ from th
        # IASimulationRunner class. Now we set the remaining parameters
        # which are not common to all IASolvers

        # Iterations of the AlternatingMinIASolver algorithm.
        self.ia_solver.max_iterations = self.params['max_iterations']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Functions Simulating each IA Algorithm xxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def simulate_general(runner, results_filename):
    """
    Run a simulation with the provided SimulationResults object `runner`.

    The results will be stored in the file with filename
    `results_filename`.

    Parameters
    ----------
    runner : An object of a subclass of SimulationRunner.
        The SimulationRunner object that will run the simulation.
    results_filename : str
        The name of the file where the simulation results will be stored.
        This name is formatted with the simulation parameters. Therefore,
        if there are parameter Nr=2 and Nt=1, for instance, then if
        `results_filename` is equal to "results for {Nr}x{Nt}" then
        "results for 2x1.pickle" will be used.
    """
    # xxxxxxxxxx Print the simulation parameters xxxxxxxxxxxxxxxxxxxxxxxxxx
    pprint(runner.params.parameters)
    print("IA Solver: {0}".format(runner.ia_solver.__class__))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Replace any parameter mention in results_filename xxxxxxxxxxxxx
    results_filename = results_filename.format(**runner.params.parameters)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # The simulation will be run either in parallel or serially depending
    # if the IPython engines are running or not.
    run_in_parallel = True
    try:
        # If we can get an IPython view that means that the IPython engines
        # are running. In that case we will perform the simulation in
        # parallel
        from IPython.parallel import Client
        # cl = Client(profile="ssh")
        cl = Client(profile="default")
        # We create a direct view to run coe in all engines
        dview = cl.direct_view()
        dview.execute('%reset')  # Reset the engines so that we don't have
                                 # variables there from last computations
        dview.execute('import sys')
        # We use block=True to ensure that all engines have modified their
        # path to include the folder with the simulator before we create
        # the load lanced view in the following.
        dview.execute('sys.path.append("{0}")'.format(parent_dir), block=True)

        # But for the actual simulation we are better using a load balanced view
        lview = cl.load_balanced_view()
    except Exception:
        # If we can't get an IPython view then we will perform the
        # simulation serially
        run_in_parallel = False

    if run_in_parallel is True:
        print("Simulation will be run in Parallel")
        runner.simulate_in_parallel(lview)
    else:
        print("Simulation will be run serially")
        runner.simulate()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Save the simulation results to a file xxxxxxxxxxxxxxxxxxxx
    results_filename = '{0}.pickle'.format(results_filename)
    runner.results.save_to_file(results_filename)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    print "Runned iterations: {0}".format(runner.runned_reps)
    print "Elapsed Time: {0}".format(runner.elapsed_time)
    print "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"

    return runner.results, results_filename


def save_results(runner, results_filename):
    """
    Save the results in the runner object to a file.

    The runner object must already have completed its simulation.

    Parameters
    ----------
    runner : An object of a subclass of SimulationRunner.
        The SimulationRunner object that will run the simulation.
    results_filename : str
        The name of the file where the simulation results will be stored.
        This name is formatted with the simulation parameters. Therefore,
        if there are parameter Nr=2 and Nt=1, for instance, then if
        `results_filename` is equal to "results for {Nr}x{Nt}" then
        "results for 2x1.pickle" will be used.

    Returns
    -------
    filename : str
        Name of the file where the results were saved.
    """
    # xxxxx Replace any parameter mention in results_filename xxxxxxxxxxxxx
    results_filename = results_filename.format(**runner.params.parameters)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Save the simulation results to a file xxxxxxxxxxxxxxxxxxxx
    results_filename = '{0}.pickle'.format(results_filename)
    runner.results.save_to_file(results_filename)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results_filename


def simulate_alternating():
    from apps.simulate_ia import AlternatingSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = AlternatingSimulationRunner('ia_config_file.txt')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner,
        'ia_alt_min_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_{max_iterations}_IA_Iter')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


def simulate_closed_form():
    from apps.simulate_ia import ClosedFormSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = ClosedFormSimulationRunner('ia_config_file.txt')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner,
        'ia_closed_form_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_{max_iterations}_IA_Iter')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


def simulate_min_leakage():
    from apps.simulate_ia import MinLeakageSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = MinLeakageSimulationRunner('ia_config_file.txt')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner,
        'ia_min_leakage_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_{max_iterations}_IA_Iter')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


def simulate_max_sinr():
    from apps.simulate_ia import MaxSINRSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = MaxSINRSimulationRunner('ia_config_file.txt')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner,
        'ia_max_sinr_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_{max_iterations}_IA_Iter')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


def simulate_mmse():
    from apps.simulate_ia import MMSESimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = MMSESimulationRunner('ia_config_file.txt')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner,
        'ia_mmse_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_{max_iterations}_IA_Iter')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Main xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx Main - Perform the simulations xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Performs the simulation in parallel
if __name__ == '__main__':
    from time import time
    from util.misc import pretty_time
    from apps.simulate_ia import ClosedFormSimulationRunner, AlternatingSimulationRunner, MMSESimulationRunner, MaxSINRSimulationRunner, MinLeakageSimulationRunner
    tic = time()

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxx Get the IPython view for the parallel simulation xxxxxxxxxxxxxx
    from IPython.parallel import Client
    # cl = Client(profile="ssh")
    cl = Client(profile="default")
    # We create a direct view to run coe in all engines
    dview = cl.direct_view()
    dview.execute('%reset')  # Reset the engines so that we don't have
                             # variables there from last computations
    dview.execute('import sys')
    # We use block=True to ensure that all engines have modified their
    # path to include the folder with the simulator before we create
    # the load lanced view in the following.
    dview.execute('sys.path.append("{0}")'.format(parent_dir), block=True)

    # But for the actual simulation we are better using a load balanced view
    lview = cl.load_balanced_view()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Creates the Closed Form Runner xxxxxxxxxxxxxxxxxxxxxxxxxxx
    print "Simulating Closed Form algorithm"
    closed_form_runner = ClosedFormSimulationRunner('ia_config_file.txt')
    closed_form_runner.simulate_in_parallel(lview, wait=False)
    pprint(closed_form_runner.params.parameters)
    print("IA Solver: {0}".format(closed_form_runner.ia_solver.__class__))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Creates the Alt. Min. Runner xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    print "Simulating Alternating Minimizations algorithm"
    alt_min_runner = AlternatingSimulationRunner('ia_config_file.txt')
    alt_min_runner.simulate_in_parallel(lview, wait=False)
    pprint(alt_min_runner.params.parameters)
    print("IA Solver: {0}".format(alt_min_runner.ia_solver.__class__))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Creates the Max SINR Runner xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    print "Simulating Max SINR algorithm"
    max_sinrn_runner = MaxSINRSimulationRunner('ia_config_file.txt')
    max_sinrn_runner.simulate_in_parallel(lview, wait=False)
    pprint(max_sinrn_runner.params.parameters)
    print("IA Solver: {0}".format(max_sinrn_runner.ia_solver.__class__))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Creates the MMSE Runner xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    print "Simulating MMSE algorithm"
    mmse_runner = MMSESimulationRunner('ia_config_file.txt')
    mmse_runner.simulate_in_parallel(lview, wait=False)
    pprint(mmse_runner.params.parameters)
    print("IA Solver: {0}".format(mmse_runner.ia_solver.__class__))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Wait for all simulations to stop xxxxxxxxxxxxxxxxxxxxxxxxx
    closed_form_runner.wait_parallel_simulation()
    alt_min_runner.wait_parallel_simulation()
    max_sinrn_runner.wait_parallel_simulation()
    mmse_runner.wait_parallel_simulation()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Save all results to respective files xxxxxxxxxxxxxxxxxxxxx
    save_results(closed_form_runner, 'ia_closed_form_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_{max_iterations}_IA_Iter')
    print "Closed Form Runned iterations: {0}".format(closed_form_runner.runned_reps)
    print "Closed Form Elapsed Time: {0}".format(closed_form_runner.elapsed_time)

    save_results(alt_min_runner, 'ia_alt_min_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_{max_iterations}_IA_Iter')
    print "Alt. Min. Runned iterations: {0}".format(alt_min_runner.runned_reps)
    print "Alt. Min. Elapsed Time: {0}".format(alt_min_runner.elapsed_time)

    save_results(max_sinrn_runner, 'ia_max_sinr_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_{max_iterations}_IA_Iter')
    print "Max SINR Runned iterations: {0}".format(max_sinrn_runner.runned_reps)
    print "Max SINR Elapsed Time: {0}".format(max_sinrn_runner.elapsed_time)

    save_results(mmse_runner, 'ia_mmse_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_{max_iterations}_IA_Iter')
    print "MMSE Runned iterations: {0}".format(mmse_runner.runned_reps)
    print "MMSE Elapsed Time: {0}".format(mmse_runner.elapsed_time)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    toc = time()
    print "Total Elapsed Time: {0}".format(pretty_time(toc - tic))


if __name__ == '__main__1':
    from time import time
    from util.misc import pretty_time
    tic = time()

    print "Simulating Closed Form algorithm"
    closed_form_results, closed_form_filename = simulate_closed_form()

    print "Simulating Alternating Min. algorithm"
    alt_min_results, alt_min_filename = simulate_alternating()

    print "Simulating Max SINR algorithm"
    max_sinrn_results, max_sinrn_filename = simulate_max_sinr()

    print "Simulating MMSE algorithm"
    mmse_results, mmse_filename = simulate_mmse()

    # print "Simulating Min. Leakage algorithm"
    # min_leakage_results, min_leakage_filename = simulate_min_leakage()

    toc = time()
    print "Elapsed Time: {0}".format(pretty_time(toc - tic))

if __name__ == '__main__':
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
    base_name = 'results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_{max_iterations}_IA_Iter'.format(**params.parameters)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    alt_min_results = SimulationResults.load_from_file(
        'ia_alt_min_{0}.pickle'.format(base_name))
    closed_form_results = SimulationResults.load_from_file(
        'ia_closed_form_{0}.pickle'.format(base_name))
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
    SNR = np.array(alt_min_results.params['SNR'])

    ber_alt_min = alt_min_results.get_result_values_list('ber')
    ber_CF_alt_min = alt_min_results.get_result_values_confidence_intervals('ber', P=95)
    ber_errors_alt_min = np.abs([i[1] - i[0] for i in ber_CF_alt_min])

    ber_closed_form = closed_form_results.get_result_values_list('ber')
    ber_CF_closed_form = closed_form_results.get_result_values_confidence_intervals('ber', P=95)
    ber_errors_closed_form = np.abs([i[1] - i[0] for i in ber_CF_closed_form])

    # ber_closed_form_first = closed_form_first_results.get_result_values_list('ber')
    # ber_CF_closed_form_first = closed_form_first_results.get_result_values_confidence_intervals('ber', P=95)
    # ber_errors_closed_form_first = np.abs([i[1] - i[0] for i in ber_CF_closed_form_first])

    ber_max_sinr = max_sinrn_results.get_result_values_list('ber')
    ber_CF_max_sinr = max_sinrn_results.get_result_values_confidence_intervals('ber', P=95)
    ber_errors_max_sinr = np.abs([i[1] - i[0] for i in ber_CF_max_sinr])

    # ber_min_leakage = min_leakage_results.get_result_values_list('ber')
    # ber_CF_min_leakage = min_leakage_results.get_result_values_confidence_intervals('ber', P=95)
    # ber_errors_min_leakage = np.abs([i[1] - i[0] for i in ber_CF_min_leakage])

    ber_mmse = mmse_results.get_result_values_list('ber')
    ber_CF_mmse = mmse_results.get_result_values_confidence_intervals('ber', P=95)
    ber_errors_mmse = np.abs([i[1] - i[0] for i in ber_CF_mmse])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.errorbar(SNR, ber_alt_min, ber_errors_alt_min, fmt='-r*', elinewidth=2.0, label='Alt. Min.')
    ax.errorbar(SNR, ber_closed_form, ber_errors_closed_form, fmt='-b*', elinewidth=2.0, label='Closed Form')
    ax.errorbar(SNR, ber_max_sinr, ber_errors_max_sinr, fmt='-g*', elinewidth=2.0, label='Max SINR')
    # ax.errorbar(SNR, ber_min_leakage, ber_errors_min_leakage, fmt='-k*', elinewidth=2.0, label='Min Leakage.')
    ax.errorbar(SNR, ber_mmse, ber_errors_mmse, fmt='-m*', elinewidth=2.0, label='MMSE.')

    plt.xlabel('SNR')
    plt.ylabel('BER')
    title = 'BER for Different Algorithms ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}'
    plt.title(title.format(**alt_min_results.params.parameters))

    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, which='both', axis='both')
    plt.show(block=False)
    fig.savefig('ber_all_ia_algorithms.pgf')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Plot Sum Capacity (all) xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    sum_capacity_alt_min = alt_min_results.get_result_values_list('sum_capacity')
    sum_capacity_CF_alt_min = alt_min_results.get_result_values_confidence_intervals('sum_capacity', P=95)
    sum_capacity_errors_alt_min = np.abs([i[1] - i[0] for i in sum_capacity_CF_alt_min])

    sum_capacity_closed_form = closed_form_results.get_result_values_list('sum_capacity')
    sum_capacity_CF_closed_form = closed_form_results.get_result_values_confidence_intervals('sum_capacity', P=95)
    sum_capacity_errors_closed_form = np.abs([i[1] - i[0] for i in sum_capacity_CF_closed_form])

    # sum_capacity_closed_form_first = closed_form_first_results.get_result_values_list('sum_capacity')
    # sum_capacity_CF_closed_form_first = closed_form_first_results.get_result_values_confidence_intervals('sum_capacity', P=95)
    # sum_capacity_errors_closed_form_first = np.abs([i[1] - i[0] for i in sum_capacity_CF_closed_form_first])

    sum_capacity_max_sinr = max_sinrn_results.get_result_values_list('sum_capacity')
    sum_capacity_CF_max_sinr = max_sinrn_results.get_result_values_confidence_intervals('sum_capacity', P=95)
    sum_capacity_errors_max_sinr = np.abs([i[1] - i[0] for i in sum_capacity_CF_max_sinr])

    # sum_capacity_min_leakage = min_leakage_results.get_result_values_list('sum_capacity')
    # sum_capacity_CF_min_leakage = min_leakage_results.get_result_values_confidence_intervals('sum_capacity', P=95)
    # sum_capacity_errors_min_leakage = np.abs([i[1] - i[0] for i in sum_capacity_CF_min_leakage])

    sum_capacity_mmse = mmse_results.get_result_values_list('sum_capacity')
    sum_capacity_CF_mmse = mmse_results.get_result_values_confidence_intervals('sum_capacity', P=95)
    sum_capacity_errors_mmse = np.abs([i[1] - i[0] for i in sum_capacity_CF_mmse])

    fig2, ax2 = plt.subplots(nrows=1, ncols=1)
    ax2.errorbar(SNR, sum_capacity_alt_min, sum_capacity_errors_alt_min, fmt='-r*', elinewidth=2.0, label='Alt. Min.')
    ax2.errorbar(SNR, sum_capacity_closed_form, sum_capacity_errors_closed_form, fmt='-b*', elinewidth=2.0, label='Closed Form')
    ax2.errorbar(SNR, sum_capacity_max_sinr, sum_capacity_errors_max_sinr, fmt='-g*', elinewidth=2.0, label='Max SINR')
    # ax2.errorbar(SNR, sum_capacity_min_leakage, sum_capacity_errors_min_leakage, fmt='-k*', elinewidth=2.0, label='Min Leakage.')
    ax2.errorbar(SNR, sum_capacity_mmse, sum_capacity_errors_mmse, fmt='-m*', elinewidth=2.0, label='MMSE.')

    plt.xlabel('SNR')
    plt.ylabel('Sum Capacity')
    title = 'Sum Capacity for Different Algorithms ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}'
    plt.title(title.format(**alt_min_results.params.parameters))

    ax2.legend(loc=2)
    ax2.grid(True, which='both', axis='both')
    plt.show()
    fig2.savefig('sum_capacity_all_ia_algorithms.pgf')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
