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

    def _run_simulation(self, current_parameters):
        # xxxxx Input parameters (set in the constructor) xxxxxxxxxxxxxxxxx
        M = self.modulator.M
        NSymbs = current_parameters["NSymbs"]
        K = current_parameters["K"]
        Nr = np.ones(K, dtype=int) * current_parameters["Nr"]
        Nt = np.ones(K, dtype=int) * current_parameters["Nt"]
        Ns = np.ones(K, dtype=int) * current_parameters["Ns"]
        SNR = current_parameters["SNR"]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Input Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # inputData has the data of all users (vertically stacked)
        inputData = np.random.randint(0, M, [np.sum(Ns), NSymbs])
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Modulate input data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # modulatedData has the data of all users (vertically stacked)
        modulatedData = self.modulator.modulate(inputData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Perform the Interference Alignment xxxxxxxxxxxxxxxxxxxxxxxx
        cumNs = np.cumsum(Ns)
        # Split the data. transmit_signal will be a list and each element
        # is a numpy array with the data of a user
        transmit_signal = np.split(modulatedData, cumNs[:-1])

        self.multiUserChannel.randomize(Nr, Nt, K)
        self.ia_solver.clear()
        self.ia_solver.solve(Ns)

        transmit_signal_precoded = map(np.dot, self.ia_solver.F, transmit_signal)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        noise_var = 1 / dB2Linear(SNR)
        import pudb; pudb.set_trace()  ## DEBUG ##
        #self.ia_solver.set_noise_power(noise_var)
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
        sirn_all_k = self.ia_solver.calc_SINR_old(noise_var=noise_var)
        calc_capacity = lambda sirn:np.sum(np.log2(1 + sirn))
        # Array with the sum capacity of each user
        sum_capacity = map(calc_capacity, sirn_all_k)
        # Total sum capacity
        total_sum_capacity = np.sum(sum_capacity)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Return the simulation results xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        symbolErrorsResult = Result.create(
            "symbol_errors", Result.SUMTYPE, symbolErrors)

        numSymbolsResult = Result.create(
            "num_symbols", Result.SUMTYPE, numSymbols)

        bitErrorsResult = Result.create("bit_errors", Result.SUMTYPE, bitErrors)

        numBitsResult = Result.create("num_bits", Result.SUMTYPE, numBits)

        berResult = Result.create("ber", Result.RATIOTYPE, bitErrors, numBits)

        serResult = Result.create(
            "ser", Result.RATIOTYPE, symbolErrors, numSymbols)

        ia_costResult = Result.create(
            "ia_cost", Result.RATIOTYPE, ia_cost, 1)

        sum_capacityResult = Result.create(
            "sum_capacity", Result.RATIOTYPE, total_sum_capacity, 1)

        simResults = SimulationResults()
        simResults.add_result(symbolErrorsResult)
        simResults.add_result(numSymbolsResult)
        simResults.add_result(bitErrorsResult)
        simResults.add_result(numBitsResult)
        simResults.add_result(berResult)
        simResults.add_result(serResult)
        simResults.add_result(ia_costResult)
        simResults.add_result(sum_capacityResult)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return simResults

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
        SNR=real_numpy_array(min=0, max=100, default=0:5:31)
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
        SNR=real_numpy_array(min=0, max=100, default=0:5:31)
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
        SNR=real_numpy_array(min=0, max=100, default=0:5:31)
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
        SNR=real_numpy_array(min=0, max=100, default=0:5:31)
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


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Functions Simulating each IA Algorithm xxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def plot_ber(results, plot_title = None, block=True):
    """
    Parameters
    ----------
    results : A SimulationResults object
        The results from a simulation.
    plot_title : str
        The tittle of the plot. Any mention to "{parameter name}" will be
        replaced by the parameter value.
    block : bool
        If True the plot will block and code will only continue after the
        plot window is closed. Set it to False if you want iterative mode.
    """
    from matplotlib import pyplot as plt

    # Get the BER and SER from the results object
    ber = results.get_result_values_list('ber')
    ser = results.get_result_values_list('ser')

    # Get the SNR from the simulation parameters
    SNR = np.array(results.params['SNR'])

    # Can only plot if we simulated for more then one value of SNR
    if SNR.size > 1:
        fig = plt.figure()
        ax = plt.axes()
        ax.semilogy(SNR, ber, '--g*', label='BER')
        ax.semilogy(SNR, ser, '--b*', label='SER')
        plt.xlabel('SNR')
        plt.ylabel('Error')
        if plot_title is not None:
            #plt.title('Min Leakage IA Algorithm ({5} Iterations)\nK={0}, Nr={1}, Nt={2}, Ns={3}, {4}'.format(K, Nr, Nt, Ns, modulator_name, ia_iterations))
            plt.title(plot_title.format(**results.params.parameters))
        ax.legend()

        ax.grid(True, which='both', axis='both')
        plt.show(block=block)


def plot_sum_capacity(results, plot_title = None, block=True):
    """
    Parameters
    ----------
    results : A SimulationResults object
        The results from a simulation.
    plot_title : str
        The tittle of the plot. Any mention to "{parameter name}" will be
        replaced by the parameter value.
    block : bool
        If True the plot will block and code will only continue after the
        plot window is closed. Set it to False if you want iterative mode.
    """
    from matplotlib import pyplot as plt

    # Get the BER and SER from the results object
    sum_capacity = results.get_result_values_list('sum_capacity')

    # Get the SNR from the simulation parameters
    SNR = np.array(results.params['SNR'])

    # Can only plot if we simulated for more then one value of SNR
    if SNR.size > 1:
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(SNR, sum_capacity, '--g*', label='Sum Capacity')
        plt.xlabel('SNR')
        plt.ylabel('Sum Capacity (bits/channel user')
        if plot_title is not None:
            #plt.title('Min Leakage IA Algorithm ({5} Iterations)\nK={0}, Nr={1}, Nt={2}, Ns={3}, {4}'.format(K, Nr, Nt, Ns, modulator_name, ia_iterations))
            plt.title(plot_title.format(**results.params.parameters))
        # ax.legend()

        ax.grid(True, which='both', axis='both')
        plt.show(block=block)



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
    run_in_parallel=True
    try:
        # If we can get an IPython view that means that the IPython engines
        # are running. In that case we will perform the simulation in
        # parallel
        from IPython.parallel import Client
        cl = Client()
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
    except Exception, e:
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


def simulate_alternating():
    from apps.simulate_ia import AlternatingSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = AlternatingSimulationRunner('ia_config_file.txt')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner,
        'ia_alt_min_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})')
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
        'ia_closed_form_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})')
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
        'ia_min_leakage_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})')
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
        'ia_max_sinr_results_{M}-{modulator}_{Nr}x{Nt}_({Ns})')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Main xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__1':
    # xxxxxxxxxx Simulate the Alternating Min. algorithm xxxxxxxxxxxxxxxxxx
    # Run the simulation
    alt_min_results, alt_min_filename = simulate_alternating()
    # Plot the results
    plot_ber(alt_min_results, plot_title='Alternating Min IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Simulate the Cloded-Form algorithm xxxxxxxxxxxxxxxxxxxxxxx
    # Run the simulation
    closed_form_results, closed_form_filename = simulate_closed_form()
    # Plot the results
    plot_ber(closed_form_results, plot_title='Closed-Form IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Simulate the Max SINRN algorithm xxxxxxxxxxxxxxxxxxxxxxxxx
    # Run the simulation
    max_sinrn_results, max_sinrn_filename = simulate_max_sinr()
    # Plot the results
    plot_ber(max_sinrn_results, plot_title='Max SINR IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Simulate the min leakage algorithm xxxxxxxxxxxxxxxxxxxxxxx
    # Run the simulation
    min_leakage_results, min_leakage_filename = simulate_min_leakage()
    # Plot the results
    plot_ber(min_leakage_results, plot_title='Min Leakage IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=True)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# 120 Iterations
if __name__ == '__main__1':
    alt_min_results = SimulationResults.load_from_file('ia_alt_min_results_4-PSK_2x2_(1)_120_IA_Iter.pickle')
    closed_form_results = SimulationResults.load_from_file('ia_closed_form_results_4-PSK_2x2_(1)_120_IA_Iter.pickle')
    max_sinrn_results = SimulationResults.load_from_file('ia_max_sinr_results_4-PSK_2x2_(1)_120_IA_Iter.pickle')
    min_leakage_results = SimulationResults.load_from_file('ia_min_leakage_results_4-PSK_2x2_(1)_120_IA_Iter.pickle')

    plot_ber(alt_min_results, plot_title='Alternating Min IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    plot_ber(closed_form_results, plot_title='Closed-Form IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    plot_ber(max_sinrn_results, plot_title='Max SINR IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    plot_ber(min_leakage_results, plot_title='Min Leakage IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=True)


# 60 Iterations
if __name__ == '__main__1':
    alt_min_results = SimulationResults.load_from_file('ia_alt_min_results_4-PSK_2x2_(1)_60_IA_Iter.pickle')
    closed_form_results = SimulationResults.load_from_file('ia_closed_form_results_4-PSK_2x2_(1)_60_IA_Iter.pickle')
    max_sinrn_results = SimulationResults.load_from_file('ia_max_sinr_results_4-PSK_2x2_(1)_60_IA_Iter.pickle')
    min_leakage_results = SimulationResults.load_from_file('ia_min_leakage_results_4-PSK_2x2_(1)_60_IA_Iter.pickle')

    plot_ber(alt_min_results, plot_title='Alternating Min IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    plot_ber(closed_form_results, plot_title='Closed-Form IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    plot_ber(max_sinrn_results, plot_title='Max SINR IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    plot_ber(min_leakage_results, plot_title='Min Leakage IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=True)


# 120 Iterations with capacity
if __name__ == '__main__1':
    alt_min_results = SimulationResults.load_from_file('ia_alt_min_results_4-PSK_2x2_(1)_120_IA_Iter_C.pickle')
    closed_form_results = SimulationResults.load_from_file('ia_closed_form_results_4-PSK_2x2_(1)_120_IA_Iter_C.pickle')
    max_sinrn_results = SimulationResults.load_from_file('ia_max_sinr_results_4-PSK_2x2_(1)_120_IA_Iter_C.pickle')
    min_leakage_results = SimulationResults.load_from_file('ia_min_leakage_results_4-PSK_2x2_(1)_120_IA_Iter_C.pickle')

    plot_ber(alt_min_results, plot_title='Alternating Min IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    plot_sum_capacity(alt_min_results, plot_title='Alternating Min IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)

    plot_ber(closed_form_results, plot_title='Closed-Form IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    plot_sum_capacity(closed_form_results, plot_title='Closed-Form IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)

    plot_ber(max_sinrn_results, plot_title='Max SINR IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    plot_sum_capacity(max_sinrn_results, plot_title='Max SINR IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)

    plot_ber(min_leakage_results, plot_title='Min Leakage IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    plot_sum_capacity(min_leakage_results, plot_title='Min Leakage IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=True)


# 60 Iterations with capacity
if __name__ == '__main__1':
    alt_min_results = SimulationResults.load_from_file('ia_alt_min_results_4-PSK_2x2_(1)_60_IA_Iter_C.pickle')
    # closed_form_results = SimulationResults.load_from_file('ia_closed_form_results_4-PSK_2x2_(1)_60_IA_Iter_C.pickle')
    # max_sinrn_results = SimulationResults.load_from_file('ia_max_sinr_results_4-PSK_2x2_(1)_60_IA_Iter_C.pickle')
    # min_leakage_results = SimulationResults.load_from_file('ia_min_leakage_results_4-PSK_2x2_(1)_60_IA_Iter_C.pickle')

    plot_ber(alt_min_results, plot_title='Alternating Min IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    plot_sum_capacity(alt_min_results, plot_title='Alternating Min IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)

    # plot_ber(closed_form_results, plot_title='Closed-Form IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    # plot_sum_capacity(closed_form_results, plot_title='Closed-Form IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)

    # plot_ber(max_sinrn_results, plot_title='Max SINR IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    # plot_sum_capacity(max_sinrn_results, plot_title='Max SINR IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)

    # plot_ber(min_leakage_results, plot_title='Min Leakage IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=False)
    # plot_sum_capacity(min_leakage_results, plot_title='Min Leakage IA Algorithm ({max_iterations} Iterations)\nK={K}, Nr={Nr}, Nt={Nt}, Ns={Ns}, {M}-{modulator}', block=True)
