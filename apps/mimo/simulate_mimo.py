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
    grandparent_dir = os.path.split(parent_dir)[0]
    sys.path.append(grandparent_dir)
except NameError:
    sys.path.append('../../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx Import Statements xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint
from copy import copy

from pyphysim.simulations import SimulationRunner, SimulationParameters, \
    SimulationResults, Result
from pyphysim.mimo import mimo
from pyphysim.modulators import fundamental
from pyphysim.util.conversion import dB2Linear, linear2dB
from pyphysim.util import misc
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class MIMOSimulationRunner(SimulationRunner):
    """
    Base class for the MIMO simulation runners.

    Most of the code in the simulation runners for the different MIMO
    schemes is common to all schemes and thus is here in the
    MIMOSimulationRunner class.

    Parameters
    ----------
    MimoSchemeClass : T < mimo.MimoBase
        The class of the MIMO scheme object.
        The MIMO scheme class, which should be a subclass of comm.mimo
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """

    def __init__(self,
                 MimoSchemeClass,
                 config_filename,
                 spec,
                 read_command_line_args=True):
        SimulationRunner.__init__(self,
                                  read_command_line_args=read_command_line_args)

        # Read the simulation configuration from the file. What is read and
        self.params = SimulationParameters.load_from_config_file(
            config_filename, spec, save_parsed_file=True)

        # Set the max_bit_errors and rep_max attributes
        self.max_bit_errors = self.params['max_bit_errors']
        self.rep_max = self.params['rep_max']

        # Create the modulator object
        M = self.params['M']
        modulator_options = {
            'PSK': fundamental.PSK,
            'QPSK': fundamental.QPSK,
            'QAM': fundamental.QAM,
            'BPSK': fundamental.BPSK
        }

        modulator_string = self.params['modulator']
        if modulator_string == 'BPSK' or modulator_string == 'QPSK':
            self.modulator = modulator_options[modulator_string]()
        else:
            self.modulator = modulator_options[modulator_string](M)

        # Create the MIMO object
        self.mimo_object = MimoSchemeClass()

    def _run_simulation(self, current_parameters):
        # xxxxx Input parameters (set in the constructor) xxxxxxxxxxxxxxxxx
        NSymbs = current_parameters["NSymbs"]
        M = self.modulator.M
        Nr = current_parameters["Nr"]
        Nt = current_parameters["Nt"]
        SNR = current_parameters["SNR"]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Create the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        channel = misc.randn_c(Nr, Nt)
        self.mimo_object.set_channel_matrix(channel)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Input Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        num_layers = self.mimo_object.getNumberOfLayers()
        inputData = np.random.randint(0, M, NSymbs * num_layers)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Modulate input data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        modulatedData = self.modulator.modulate(inputData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Encode with the MIMO scheme xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        transmit_signal = self.mimo_object.encode(modulatedData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        noiseVar = 1 / dB2Linear(SNR)
        awgn_noise = (misc.randn_c(Nr, NSymbs) * np.sqrt(noiseVar))
        received_signal = np.dot(channel, transmit_signal) + awgn_noise
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Decode with the MIMO Scheme xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        mimo_decoded_data = self.mimo_object.decode(received_signal)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Demodulate received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        demodulatedData = self.modulator.demodulate(mimo_decoded_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the symbol and bit error rates xxxxxxxxxxxxxxxxx
        symbolErrors = sum(inputData != demodulatedData)
        bitErrors = misc.count_bit_errors(inputData, demodulatedData)
        numSymbols = inputData.size
        numBits = inputData.size * fundamental.level2bits(M)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Return the simulation results xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        symbolErrorsResult = Result.create("symbol_errors", Result.SUMTYPE,
                                           symbolErrors)

        numSymbolsResult = Result.create("num_symbols", Result.SUMTYPE,
                                         numSymbols)

        bitErrorsResult = Result.create("bit_errors", Result.SUMTYPE, bitErrors)

        numBitsResult = Result.create("num_bits", Result.SUMTYPE, numBits)

        berResult = Result.create("ber", Result.RATIOTYPE, bitErrors, numBits)

        serResult = Result.create("ser", Result.RATIOTYPE, symbolErrors,
                                  numSymbols)

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
    #     accumulated_bit_errors \
    #         = simulation_results['bit_errors'][-1].get_result()

    #     max_bit_errors = current_params['max_bit_errors']
    #     return accumulated_bit_errors < max_bit_errors


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
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """

    def __init__(self, config_filename, read_command_line_args=True):
        spec = """[Scenario]
        SNR=real_numpy_array(min=0, max=100, default=0:5:21)
        M=integer(min=4, max=512, default=16)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="QAM")
        NSymbs=integer(min=10, max=1000000, default=200)
        Nr=integer(min=1,default=1)
        Nt=integer(min=2,max=2,default=2)
        [General]
        rep_max=integer(min=1, default=5000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        MIMOSimulationRunner.__init__(
            self,
            mimo.Alamouti,
            config_filename,
            spec,
            read_command_line_args=read_command_line_args)


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
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """

    def __init__(self, config_filename, read_command_line_args=True):
        spec = """[Scenario]
        SNR=real_numpy_array(min=0, max=100, default=0:5:21)
        M=integer(min=4, max=512, default=16)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="QAM")
        NSymbs=integer(min=10, max=1000000, default=200)
        Nt=integer(min=1,default=2)
        Nr=integer(min=1,default=2)
        [General]
        rep_max=integer(min=1, default=5000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        MIMOSimulationRunner.__init__(
            self,
            mimo.Blast,
            config_filename,
            spec,
            read_command_line_args=read_command_line_args)


class MRCSimulationRunner(MIMOSimulationRunner):
    """
    Implements a simulation runner for a transmission with the Alternating
    Minimizations Interference Alignment Algorithm.

    Parameters
    ----------
    config_filename : string
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """

    def __init__(self, config_filename, read_command_line_args=True):
        spec = """[Scenario]
        SNR=real_numpy_array(min=0, max=100, default=0:5:21)
        M=integer(min=4, max=512, default=16)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="QAM")
        NSymbs=integer(min=10, max=1000000, default=200)
        Nt=integer(min=1,default=2)
        Nr=integer(min=1,default=2)
        [General]
        rep_max=integer(min=1, default=5000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        MIMOSimulationRunner.__init__(
            self,
            mimo.MRC,
            config_filename,
            spec,
            read_command_line_args=read_command_line_args)


class MRTSimulationRunner(MIMOSimulationRunner):
    """
    Implements a simulation runner for a transmission with the Alternating
    Minimizations Interference Alignment Algorithm.

    Parameters
    ----------
    config_filename : string
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """

    def __init__(self, config_filename, read_command_line_args=True):
        spec = """[Scenario]
        SNR=real_numpy_array(min=0, max=100, default=0:5:21)
        M=integer(min=4, max=512, default=16)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="QAM")
        NSymbs=integer(min=10, max=1000000, default=200)
        Nt=integer(min=2,default=2)
        Nr=integer(min=1,default=2)
        [General]
        rep_max=integer(min=1, default=5000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        MIMOSimulationRunner.__init__(
            self,
            mimo.MRT,
            config_filename,
            spec,
            read_command_line_args=read_command_line_args)


class SVDMimoSimulationRunner(MIMOSimulationRunner):
    """
    Implements a simulation runner for a transmission with the SVD MIMO
    scheme.

    Parameters
    ----------
    config_filename : string
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """

    def __init__(self, config_filename, read_command_line_args=True):
        spec = """[Scenario]
        SNR=real_numpy_array(min=0, max=100, default=0:5:21)
        M=integer(min=4, max=512, default=16)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="QAM")
        NSymbs=integer(min=10, max=1000000, default=200)
        Nt=integer(min=1,default=2)
        Nr=integer(min=1,default=2)
        [General]
        rep_max=integer(min=1, default=5000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        MIMOSimulationRunner.__init__(
            self,
            mimo.SVDMimo,
            config_filename,
            spec,
            read_command_line_args=read_command_line_args)


class GMDMimoSimulationRunner(MIMOSimulationRunner):
    """
    Implements a simulation runner for a transmission with the GMD MIMO
    scheme.

    Parameters
    ----------
    config_filename : string
        Name of the file containing the simulation parameters. If the file
        does not exist, a new file will be created with the provided name
        containing the default parameter values.
    read_command_line_args : bool
        If True (default), read and parse command line arguments.
    """

    def __init__(self, config_filename, read_command_line_args=True):
        spec = """[Scenario]
        SNR=real_numpy_array(min=0, max=100, default=0:5:21)
        M=integer(min=4, max=512, default=16)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="QAM")
        NSymbs=integer(min=10, max=1000000, default=200)
        Nt=integer(min=1,default=2)
        Nr=integer(min=1,default=2)
        [General]
        rep_max=integer(min=1, default=5000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR'))
        """.split("\n")

        MIMOSimulationRunner.__init__(
            self,
            mimo.GMDMimo,
            config_filename,
            spec,
            read_command_line_args=read_command_line_args)


def simulate_general(runner, results_filename):
    """
    Function with the general code to simulate the MIMO schemes.

    Parameters
    ----------
    runner : MIMOSimulationRunner
        The simulation runner object.
    results_filename : str
        Name of the file where results should be saved.

    Returns
    -------
    (SimulationResults, str)
        Simulation results and name of the file where it was saved.
    """

    # xxxxxxxxxx Print the simulation parameters xxxxxxxxxxxxxxxxxxxxxxxxxx
    pprint(runner.params.parameters)
    print("MIMO Scheme: {0}".format(runner.mimo_object.__class__.__name__))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Replace any parameter mention in results_filename xxxxxxxxxxxxx
    runner.set_results_filename(results_filename)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # The simulation will be run either in parallel or serially depending
    # if the IPython engines are running or not.
    run_in_parallel = True
    # noinspection PyBroadException,PyBroadException
    try:
        # If we can get an IPython view that means that the IPython engines
        # are running. In that case we will perform the simulation in
        # parallel
        from ipyparallel import Client
        cl = Client()
        # We create a direct view to run coe in all engines
        dview = cl.direct_view()

        # Reset the engines so that we don't have variables there from last
        # computations
        dview.execute('%reset')

        dview.execute('import sys')
        # We use block=True to ensure that all engines have modified their
        # path to include the folder with the simulator before we create
        # the load lanced view in the following.
        dview.execute('sys.path.append("{0}")'.format(parent_dir), block=True)

        # But for the actual simulation we are better using a load balanced
        # view
        lview = cl.load_balanced_view()
    except Exception:  # pylint: disable=W0703
        # If we can't get an IPython view then we will perform the
        # simulation serially
        run_in_parallel = False

    if run_in_parallel is True:
        print("-----> Simulation will be run in Parallel")
        # noinspection PyUnboundLocalVariable
        runner.simulate_in_parallel(lview)
    else:
        print("-----> Simulation will be run serially")
        runner.simulate()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    print("Runned iterations: {0}".format(runner.runned_reps))
    print("Elapsed Time: {0}".format(runner.elapsed_time))
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")

    return runner.results, runner.results_filename


def simulate_alamouti(config_file_name='mimo_alamouti_config_file.txt'):
    # noinspection PyUnresolvedReferences
    from apps.mimo.simulate_mimo import AlamoutiSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = AlamoutiSimulationRunner(config_file_name)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner, 'alamouti_results_{M}-{modulator}_Nr_{Nr}_receive_antennas')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


def simulate_blast(config_file_name='mimo_blast_config_file.txt'):
    # noinspection PyUnresolvedReferences
    from apps.mimo.simulate_mimo import BlastSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = BlastSimulationRunner(config_file_name)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner,
        'blast_results_{M}-{modulator}_Nr_{Nr}_Nt_{Nt}_receive_antennas')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


def simulate_mrc(config_file_name='mimo_mrc_config_file.txt'):
    # noinspection PyUnresolvedReferences
    from apps.mimo.simulate_mimo import MRCSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = MRCSimulationRunner(config_file_name)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner, 'mrc_results_{M}-{modulator}_Nr_{Nr}_Nt_{Nt}_receive_antennas')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


def simulate_mrt(config_file_name='mimo_mrt_config_file.txt'):
    # noinspection PyUnresolvedReferences
    from apps.mimo.simulate_mimo import MRTSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = MRTSimulationRunner(config_file_name)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner, 'mrt_results_{M}-{modulator}_Nr_{Nr}_Nt_{Nt}_receive_antennas')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


def simulate_svdmimo(config_file_name='mimo_svdmimo_config_file.txt'):
    # noinspection PyUnresolvedReferences
    from apps.mimo.simulate_mimo import SVDMimoSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = SVDMimoSimulationRunner(config_file_name)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner,
        'svdmimo_results_{M}-{modulator}_Nr_{Nr}_Nt_{Nt}_receive_antennas')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


def simulate_gmdmimo(config_file_name='mimo_gmdmimo_config_file.txt'):
    # noinspection PyUnresolvedReferences
    from apps.mimo.simulate_mimo import GMDMimoSimulationRunner

    # xxxxxxxxxx Creates the simulation runner object xxxxxxxxxxxxxxxxxxxxx
    runner = GMDMimoSimulationRunner(config_file_name)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results, filename = simulate_general(
        runner,
        'gmdmimo_results_{M}-{modulator}_Nr_{Nr}_Nt_{Nt}_receive_antennas')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results, filename


def get_ebn0_vec(results):
    """
    Get the Eb/N0 vector suitable for the plot.

    Parameters
    ----------
    results : A SimulationResults object
        The results from a simulation.
    """
    SNR = np.array(results.params['SNR'])

    modulator = results.params['modulator']
    if modulator == 'BPSK':
        K = 1
    elif modulator == 'QPSK':
        K = 2
    else:
        M = results.params['M']
        K = np.round(np.log2(M))

    ebn0 = SNR + linear2dB(1. / K)

    return ebn0


def plot_ber(results,
             ax=None,
             name=None,
             block=True,
             X_axis='SNR',
             plot_args=None):
    """
    Plot the BER in `results`.

    Parameters
    ----------
    results : A SimulationResults object
        The results from a simulation.
    ax : A matplotlib axis.
        The axis to plot the curve. If not specified a new one will be
        created.
    name : str
        The name of the curve to be plotted.
    block : bool
        If True the plot will block and code will only continue after the
        plot window is closed. Set it to False if you want iterative mode.
    X_axis : str
        The values to use for the X axis. This can be either 'SNR' or
        'EbN0'.
    plot_args : dict
        A dictionary with extra options to pass to matplotlib.
        Ex: plot_args={'color':'green', 'linestyle':'dashed'}

    Returns
    -------
    ax : A matplotlib axis.
        The axis where the curve was plotted.
    """
    if plot_args is None:
        plot_args = {}

    ber = results.get_result_values_list('ber')

    # Get the SNR from the simulation parameters
    SNR = np.array(results.params['SNR'])
    EBN0 = get_ebn0_vec(results)

    if X_axis == 'SNR':
        X = SNR
    else:
        X = EBN0

    # Can only plot if we simulated for more then one value of X
    if X.size > 1:
        if name is None:
            label = 'BER'
        else:
            label = 'BER for {0}'.format(name)

        if ax is None:
            _, cur_ax = plt.subplots()
            plt.xlabel(X_axis)
            plt.ylabel('Error')
            cur_ax.legend()
            cur_ax.grid(True, which='major', axis='both')
        else:
            cur_ax = ax

        cur_ax.semilogy(X, ber, marker='*', label=label, **plot_args)
        cur_ax.legend(loc=3)
        try:
            # There is no 'block' keyword argument when running inside
            # IPython notebook
            plt.show(block=block)
        except TypeError:
            plt.show()

        return ax


def plot_ser(results,
             ax=None,
             name=None,
             block=True,
             X_axis='SNR',
             plot_args=None):
    """
    Plot the SER in `results`.

    Parameters
    ----------
    results : A SimulationResults object
        The results from a simulation.
    ax : A matplotlib axis.
        The axis to plot the curve. If not specified a new one will be
        created.
    name : str
        The name of the curve to be plotted.
    block : bool
        If True the plot will block and code will only continue after the
        plot window is closed. Set it to False if you want iterative mode.
    X_axis : str
        The values to use for the X axis. This can be either 'SNR' or
        'EbN0'.
    plot_args : dict
        A dictionary with extra options to pass to matplotlib.
        Ex: plot_args={'color':'green', 'linestyle':'dashed'}

    Returns
    -------
    ax : A matplotlib axis.
        The axis where the curve was plotted.
    """
    if plot_args is None:
        plot_args = {}

    ser = results.get_result_values_list('ser')

    # Get the SNR from the simulation parameters
    SNR = np.array(results.params['SNR'])
    EBN0 = get_ebn0_vec(results)

    if X_axis == 'SNR':
        X = SNR
    else:
        X = EBN0

    # Can only plot if we simulated for more then one value of X
    if X.size > 1:
        if name is None:
            label = 'SER'
        else:
            label = 'SER for {0}'.format(name)

        if ax is None:
            _, cur_ax = plt.subplots()
            plt.xlabel(X_axis)
            plt.ylabel('Error')
            cur_ax.legend()
            cur_ax.grid(True, which='major', axis='both')
        else:
            cur_ax = ax

        cur_ax.semilogy(X, ser, marker='*', label=label, **plot_args)
        cur_ax.legend(loc=3)
        plt.show(block=block)

        return ax


def plot_ber_and_ser(results,
                     ax=None,
                     name=None,
                     block=True,
                     X_axis='SNR',
                     plot_args=None):
    """
    Plot the BER and the SER.

    Parameters
    ----------
    results : A SimulationResults object
        The results from a simulation.
    ax : A matplotlib axis.
        The axis to plot the curve. If not specified a new one will be
        created.
    name : str
        The name of the curve to be plotted.
    block : bool
        If True the plot will block and code will only continue after the
        plot window is closed. Set it to False if you want iterative mode.
    X_axis : str
        The values to use for the X axis. This can be either 'SNR' or
        'EbN0'.
    plot_args : dict
        A dictionary with extra options to pass to matplotlib.
        Ex: plot_args={'color':'green', 'linestyle':'dashed'}

    Returns
    -------
    ax : A matplotlib axis.
        The axis where the curve was plotted.
    """
    if plot_args is None:
        plot_args = {}

    new_ax = plot_ber(results, ax, name, block, X_axis, plot_args)
    plot_args2 = copy(plot_args)
    plot_args2['linestyle'] = 'dashed'
    plot_ser(results, new_ax, name, block, X_axis, plot_args2)


if __name__ == '__main__':
    fig, ax = plt.subplots()
    X_axis = 'EBN0'  # SNR or EBN0

    # xxxxxxxxxx Plot Labels and Tittle xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    plt.xlabel(X_axis)
    plt.ylabel('Error')
    plt.title('Comparison of multiple MIMO schemes')
    ax.grid(True, which='major', axis='both')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    results1, filename1 = simulate_alamouti()
    plot_ber(results1,
             ax=ax,
             name='Alamouti',
             block=False,
             X_axis=X_axis,
             plot_args={'color': 'green'})

    results2, filename2 = simulate_blast()
    plot_ber(results2,
             ax=ax,
             name='BLAST',
             block=False,
             X_axis=X_axis,
             plot_args={'color': 'blue'})

    results3, filename3 = simulate_mrc()
    plot_ber(results3,
             ax=ax,
             name='MRC',
             block=False,
             X_axis=X_axis,
             plot_args={'color': 'red'})

    results4, filename4 = simulate_mrt()
    plot_ber(results4,
             ax=ax,
             name='MRT',
             block=False,
             X_axis=X_axis,
             plot_args={'color': 'magenta'})

    results5, filename5 = simulate_svdmimo()
    plot_ber(results5,
             ax=ax,
             name='SVD MIMO',
             block=False,
             X_axis=X_axis,
             plot_args={'color': 'cyan'})

    results7, filename7 = simulate_gmdmimo()
    plot_ber(results7,
             ax=ax,
             name='GMD MIMO',
             block=True,
             X_axis=X_axis,
             plot_args={'color': 'pink'})
