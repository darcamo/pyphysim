#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module containing simulation runners for the several MIMO schemes
algorithms in the comm.mimo module.
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
import numpy as np
from pprint import pprint

from pyphysim.simulations.core import SimulationRunner, SimulationParameters, SimulationResults, Result
from pyphysim.comm import modulators, mimo
from pyphysim.util.conversion import dB2Linear
from pyphysim.util import misc
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# TODO: Implement-me
class MIMOSimulationRunner(SimulationRunner):
    """
    Base class for the MIMO simulation runners.

    Most of the code in the simulation runners for the different MIMO
    schemes is common to all schemes and thus is here in the
    MIMOSimulationRunner class.

    Parameters
    ----------
    MimoSchemeClass : The class of the MIMO scheme object
        The MIMO scheme class, which should be a subclass of comm.mimo
    """

    def __init__(self, MimoSchemeClass, config_filename, spec):
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

        if self.params['modulator'] == 'BPSK' or self.params['modulator'] == 'QPSK':
            self.modulator = modulator_options[self.params['modulator']]()
        else:
            self.modulator = modulator_options[self.params['modulator']](M)

        # Create the MIMO object
        if MimoSchemeClass is mimo.Blast:
            self.mimo_object = MimoSchemeClass(self.params['Nt'])
        else:
            self.mimo_object = MimoSchemeClass()

    def _run_simulation(self, current_parameters):
        # xxxxx Input parameters (set in the constructor) xxxxxxxxxxxxxxxxx
        NSymbs = current_parameters["NSymbs"]
        M = self.modulator.M
        Nr = current_parameters["Nr"]
        SNR = current_parameters["SNR"]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Input Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        num_layers = self.mimo_object.getNumberOfLayers()
        inputData = np.random.randint(0, M, NSymbs * num_layers)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Modulate input data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        modulatedData = self.modulator.modulate(inputData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Encode with the MIMO scheme xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        transmit_signal = self.mimo_object.encode(
            modulatedData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        channel = misc.randn_c(Nr, 2)
        noiseVar = 1 / dB2Linear(SNR)
        awgn_noise = (misc.randn_c(Nr, NSymbs) * np.sqrt(noiseVar))
        received_signal = np.dot(channel, transmit_signal) + awgn_noise
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Decode with the MIMO Scheme xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        mimo_decoded_data = self.mimo_object.decode(
            received_signal, channel)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Demodulate received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        demodulatedData = self.modulator.demodulate(mimo_decoded_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the symbol and bit error rates xxxxxxxxxxxxxxxxx
        symbolErrors = sum(inputData != demodulatedData)
        bitErrors = misc.count_bit_errors(inputData, demodulatedData)
        numSymbols = inputData.size
        numBits = inputData.size * modulators.level2bits(M)
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

        simResults = SimulationResults()
        simResults.add_result(symbolErrorsResult)
        simResults.add_result(numSymbolsResult)
        simResults.add_result(bitErrorsResult)
        simResults.add_result(numBitsResult)
        simResults.add_result(berResult)
        simResults.add_result(serResult)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return simResults

    # def _keep_going(self, current_params, simulation_results, current_rep):
    #     #return True
    #     cumulated_bit_errors = simulation_results['bit_errors'][-1].get_result()
    #     max_bit_errors = current_params['max_bit_errors']
    #     return cumulated_bit_errors < max_bit_errors


class AlamoutiSimulationRunner(MIMOSimulationRunner):
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
        SNR=real_numpy_array(min=0, max=100, default=0:5:21)
        M=integer(min=4, max=512, default=16)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="QAM")
        NSymbs=integer(min=10, max=1000000, default=200)
        Nr=integer(min=1,default=1)
        [General]
        rep_max=integer(min=1, default=5000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        MIMOSimulationRunner.__init__(
            self,
            mimo.Alamouti,
            config_filename,
            spec)


class BlastSimulationRunner(MIMOSimulationRunner):
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
        SNR=real_numpy_array(min=0, max=100, default=0:5:21)
        M=integer(min=4, max=512, default=16)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="QAM")
        NSymbs=integer(min=10, max=1000000, default=200)
        Nt=integer(min=2,default=2)
        Nr=integer(min=2,default=2)
        [General]
        rep_max=integer(min=1, default=5000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        MIMOSimulationRunner.__init__(
            self,
            mimo.Blast,
            config_filename,
            spec)


def simulate_general(runner, results_filename):
    # xxxxxxxxxx Print the simulation parameters xxxxxxxxxxxxxxxxxxxxxxxxxx
    pprint(runner.params.parameters)
    print("MIMO Scheme: {0}".format(runner.mimo_object.__class__))
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
    except Exception:  # pylint: disable=W0703
        # If we can't get an IPython view then we will perform the
        # simulation serially
        run_in_parallel = False

    if run_in_parallel is True:
        print("-----> Simulation will be run in Parallel")
        runner.simulate_in_parallel(lview)
    else:
        print("-----> Simulation will be run serially")
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


def simulate_alamouti(config_file_name='mimo_alamouti_config_file.txt'):
    from apps.simulate_mimo import AlamoutiSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = AlamoutiSimulationRunner(config_file_name)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner,
        'alamouti_results_{M}-{modulator}_Nr_{Nr}_receive_antennas')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


def simulate_blast(config_file_name='mimo_blast_config_file.txt'):
    from apps.simulate_mimo import BlastSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = BlastSimulationRunner(config_file_name)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner,
        'blast_results_{M}-{modulator}_Nr_{Nr}_Nt_{Nt}_receive_antennas')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


def plot_ber_and_ser(results, plot_title = None, block=True):
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

    ber = results.get_result_values_list('ber')
    ser = results.get_result_values_list('ser')

    # Get the SNR from the simulation parameters
    SNR = np.array(results.params['SNR'])

    # modulator_name = '{0}-{1}'.format(results.params['M'],
    #                                   results.params['modulator'])

    # Can only plot if we simulated for more then one value of SNR
    if SNR.size > 1:
        fig = plt.figure()
        ax = plt.axes()
        ax.semilogy(SNR, ber, '--g*', label='BER')
        ax.semilogy(SNR, ser, '--b*', label='SER')
        plt.xlabel('SNR')
        plt.ylabel('Error')
        if plot_title is not None:
            plt.title(plot_title.format(**results.params.parameters))

        ax.legend()

        ax.grid(True, which='both', axis='both')
        plt.show(block=block)


if __name__ == '__main__':
    results, filename = simulate_alamouti()
    plot_ber_and_ser(results, plot_title='BER and SER for {M}-{modulator} with Alamouti (Nr={Nr})', block=True)

    # results, filename = simulate_blast()
    # plot_ber_and_ser(results, plot_title='BER and SER for {M}-{modulator} with BLAST (Nr={Nr}, Nt={Nt})', block=True)
