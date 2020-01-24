#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Perform the simulation of the transmission of PSK symbols through an
awgn channel.

This performs the same simulation "in serial" and "in parallel" as an
example of the differences between both methods in a SimulationRunner
subclass.
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
    parent_dir = './'
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import numpy as np

from pyphysim.simulations import *
from pyphysim.modulators import fundamental
from pyphysim.util.conversion import dB2Linear
from pyphysim.util import misc


class VerySimplePskSimulationRunner(SimulationRunner):
    """This is a complete example with the minimum code to actually perform
    a simulation.

    Basically, we implement the _run_simulation method, which here performs
    the simulation of a 4-PSK transmission in an AWGN channel, as well as
    the optional _keep_going method to allow an earlier termination of the
    simulation when a maximum number of bit errors is achieved.

    The simulation parameters must be added to the self.params object (this
    can be done, for instance, in the __init__ method)

    can be directly set as regular attributes in
    the __init__ method, since they can be accessed in the _run_simulation
    method this way. The only exception is the SNR parameter, which is
    instead added to a SimulationParameters object (which is a regular
    attribute). The reason for this is because we want pass a vector with
    SNR values and employ the "unpack" functionality of the
    SimulationParameters class.

    """

    def __init__(self,):
        SimulationRunner.__init__(self)

        SNR = np.array([0, 3, 6, 9, 12])
        M = 4
        modulator = fundamental.PSK(M)
        NSymbs = 500

        self.params.add('modulator', modulator)
        self.params.add('NSymbs', NSymbs)

        self.rep_max = 1000
        # self.max_bit_errors = 1. / 100. * NSymbs * self.rep_max
        max_bit_errors = 1. / 100. * NSymbs * self.rep_max
        self.params.add('max_bit_errors', max_bit_errors)

        # self.progressbar_message = None
        self.progressbar_message = "{0}-PSK".format(M) + \
                                   " Simulation - SNR: {SNR}"

        # Add the parameters to the self.params variable
        self.params.add('SNR', SNR)
        self.params.set_unpack_parameter('SNR')

    def _run_simulation(self, current_parameters):
        """The _run_simulation method is where the actual code to simulate
        the system is.

        The implementation of this method is required by every subclass of
        SimulationRunner.
        """
        # xxxxx Input parameters (set in the constructor) xxxxxxxxxxxxxxxxx
        NSymbs = current_parameters['NSymbs']
        modulator = current_parameters['modulator']
        M = modulator.M
        SNR = current_parameters["SNR"]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Input Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        inputData = np.random.randint(0, M, NSymbs)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Modulate input data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        modulatedData = modulator.modulate(inputData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        noiseVar = 1. / dB2Linear(SNR)
        noise = misc.randn_c(NSymbs) * np.sqrt(noiseVar)
        receivedData = modulatedData + noise
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Demodulate received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        demodulatedData = modulator.demodulate(receivedData)
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

    def _keep_going(self, current_params, simulation_results, current_rep):
        """
        Check if the simulation should continue or stop.

        Parameters
        ----------
        current_params : SimulationParameters object
            SimulationParameters object with the parameters of the
            simulation.
        simulation_results : SimulationResults object
            SimulationResults object from the last iteration (merged with
            all the previous results)
        current_rep : int
            Number of iterations already run.

        Returns
        -------
        result : bool
            True if the simulation should continue or False otherwise.
        """
        # Return true as long as accumulated_bit_errors is lower then
        # max_bit_errors
        accumulated_bit_errors \
            = simulation_results['bit_errors'][-1].get_result()
        max_bit_errors = current_params['max_bit_errors']
        return accumulated_bit_errors < max_bit_errors

    def get_data_to_be_plotted(self):
        """The get_data_to_be_plotted is not part of the simulation, but it
        is useful after the simulation is finished to get the results
        easily for plot.
        """
        modulator = self.params['modulator']

        ber = self.results.get_result_values_list('ber')
        ser = self.results.get_result_values_list('ser')

        # Get the SNR from the simulation parameters
        SNR = np.array(self.params['SNR'])

        # Calculates the Theoretical SER and BER
        theoretical_ser = modulator.calcTheoreticalSER(SNR)
        theoretical_ber = modulator.calcTheoreticalBER(SNR)
        return SNR, ber, ser, theoretical_ber, theoretical_ser


if __name__ == '__main__':
    # Since we are using the parallel capabilities provided by IPython, we
    # need to create a client and then a view of the IPython engines that
    # will be used.
    from ipyparallel import Client
    cl = Client()
    dview = cl.direct_view()

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # NOTE: Before running the code above, initialize the ipython
    # engines. One easy way to do that is to call the "ipcluster start"
    # command in a terminal.
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Add the folder containing PyPhysim to the python path in all the
    # engines
    dview.execute('import sys')
    dview.execute('sys.path.append("{0}")'.format(parent_dir))

    from matplotlib import pyplot as plt
    # noinspection PyUnresolvedReferences
    from apps.awgn_modulators.simulate_parallel_psk import \
        VerySimplePskSimulationRunner

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxx Parallel Simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    sim_p = VerySimplePskSimulationRunner()
    sim_p.simulate_in_parallel(dview)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxx Serial Simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    sim_s = VerySimplePskSimulationRunner()
    sim_s.simulate()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Print the results xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    SNR_p, ber_p, ser_p, theoretical_ber_p, theoretical_ser_p \
        = sim_p.get_data_to_be_plotted()
    print("SER_P: {0}".format(ser_p))
    print("BER_P: {0}".format(ber_p))
    print(sim_p.elapsed_time)

    SNR_s, ber_s, ser_s, theoretical_ber_s, theoretical_ser_s \
        = sim_s.get_data_to_be_plotted()
    print("SER_s: {0}".format(ser_s))
    print("BER_s: {0}".format(ber_s))
    print(sim_s.elapsed_time)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Plot the results xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Can only plot if we simulated for more then one value of SNR_p
    if SNR_p.size > 1:
        f = plt.figure(figsize=(12, 4.5))
        ax1 = plt.subplot(121)

        ax1.semilogy(SNR_p, ber_p, '--g*', label='BER_P')
        ax1.semilogy(SNR_p, ser_p, '--b*', label='SER_P')
        ax1.semilogy(SNR_p, theoretical_ber_p, '-g+', label='Theoretical BER_P')
        ax1.semilogy(SNR_p, theoretical_ser_p, '-b+', label='theoretical SER_P')

        modulator_obj = sim_p.params['modulator']

        ax1.set_xlabel('SNR')
        ax1.set_ylabel('Error')
        ax1.set_title('{0} modulation (Parallel Simulation)'.format(
            modulator_obj.name))
        ax1.legend()

        ax1.grid(True, which='both', axis='both')
        # plt.show()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Can onl plot if we simulated for more then one value of SNR_s
    if SNR_s.size > 1:
        ax2 = plt.subplot(122)
        ax2.semilogy(SNR_s, ber_s, '--g*', label='BER_s')
        ax2.semilogy(SNR_s, ser_s, '--b*', label='SER_s')
        ax2.semilogy(SNR_s, theoretical_ber_s, '-g+', label='Theoretical BER_s')
        ax2.semilogy(SNR_s, theoretical_ser_s, '-b+', label='theoretical SER_s')

        modulator_obj = sim_s.params['modulator']

        ax2.set_xlabel('SNR')
        ax2.set_ylabel('Error')
        ax2.set_title('{0} modulation (Serial Simulation)'.format(
            modulator_obj.name))
        ax2.legend()

        ax2.grid(True, which='both', axis='both')
        plt.show()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
