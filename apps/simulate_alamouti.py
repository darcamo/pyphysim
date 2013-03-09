#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perform the simulation of the transmission using the Alamouti MIMO
scheme.

"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import numpy as np

from util.simulations import *
from comm import modulators, mimo
from util import misc
from util.conversion import dB2Linear


class AlamoutiSimulationRunner(SimulationRunner):
    """Implements a simulation runner for a transmission with the Alamouti
    MIMO scheme.

    """
    # Alamouti object to encode and decode data with the Alamouti MIMO scheme
    alamouti = mimo.Alamouti()

    def __init__(self):
        SimulationRunner.__init__(self)

        # The _keep_going method will stop the simulation earlier when
        # max_bit_errors are achieved.
        max_bit_errors = 2000
        self.params.add('max_bit_errors', max_bit_errors)
        SNR = np.array([0., 5, 10, 15, 20, 25])
        M = 16
        #self.alamouti = mimo.Alamouti()
        NSymbs = 400
        modulator = modulators.QAM(M)
        Nr = 2
        # Note that Nt is equal to 2 for the Alamouti scheme

        # xxxxx Declared in the SimulationRunner class xxxxxxxxxxxxxxxxxxxx
        # We need to set these two in all simulations
        self.rep_max = 3000
        self.progressbar_message = "Alamouti with {0} - SNR: {{SNR}}".format(
            modulator.name)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # We need to add the parameters to the self.param variable.
        self.params.add('SNR', SNR)
        self.params.set_unpack_parameter('SNR')
        self.params.add('modulator', modulator)
        self.params.add('NSymbs', NSymbs)
        self.params.add('Nr', Nr)

    @staticmethod
    def _run_simulation(current_parameters):
        # xxxxx Input parameters (set in the constructor) xxxxxxxxxxxxxxxxx
        NSymbs = current_parameters['NSymbs']
        modulator = current_parameters['modulator']
        M = modulator.M
        SNR = current_parameters["SNR"]
        Nr = current_parameters["Nr"]

        #K = self.K
        # M = modulator.M
        # NSymbs = self.NSymbs
        # Nr = self.Nr
        # SNR = current_parameters["SNR"]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Input Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        inputData = np.random.randint(0, M, NSymbs)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Modulate input data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        modulatedData = modulator.modulate(inputData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Encode with Alamouti xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        transmit_signal = AlamoutiSimulationRunner.alamouti.encode(
            modulatedData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        channel = misc.randn_c(Nr, 2)
        noiseVar = 1 / dB2Linear(SNR)
        awgn_noise = (misc.randn_c(Nr, NSymbs) * np.sqrt(noiseVar))
        received_signal = np.dot(channel, transmit_signal) + awgn_noise
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Decode Alamouti xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        alamouti_decoded_data = AlamoutiSimulationRunner.alamouti.decode(
            received_signal, channel)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Demodulate received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        demodulatedData = modulator.demodulate(alamouti_decoded_data)
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

    @staticmethod
    def _keep_going(current_params, simulation_results):
        #return True
        cumulated_bit_errors = simulation_results['bit_errors'][-1].get_result()
        max_bit_errors = current_params['max_bit_errors']
        return cumulated_bit_errors < max_bit_errors

    def get_data_to_be_plotted(self):
        """The get_data_to_be_plotted is not part of the simulation, but it
        is useful after the simulation is finished to get the results
        easily for plot.
        """
        ber = self.results.get_result_values_list('ber')
        ser = self.results.get_result_values_list('ser')

        # Get the SNR from the simulation parameters
        SNR = np.array(self.params['SNR'])

        return (SNR, ber, ser)


# Serial Version
if __name__ == '__main__1':
    from pylab import *
    from apps.simulate_alamouti import AlamoutiSimulationRunner

    #write_config_file_template()

    sim = AlamoutiSimulationRunner()
    sim.simulate()

    SNR, ber, ser = sim.get_data_to_be_plotted()

    modulator_obj = sim.params['modulator']
    # Can only plot if we simulated for more then one value of SNR
    if SNR.size > 1:
        semilogy(SNR, ber, '--g*', label='BER')
        semilogy(SNR, ser, '--b*', label='SER')
        xlabel('SNR')
        ylabel('Error')
        title('BER and SER for {0} modulation with Alamouti'.format(
            modulator_obj.name))
        legend()

        grid(True, which='both', axis='both')
        show()

    print sim.elapsed_time


# Parallel version
if __name__ == '__main__':
    # Since we are using the parallel capabilities provided by IPython, we
    # need to create a client and then a view of the IPython engines that
    # will be used.
    from IPython.parallel import Client
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

    from pylab import *
    from apps.simulate_alamouti import AlamoutiSimulationRunner

    #write_config_file_template()

    sim = AlamoutiSimulationRunner()
    sim.simulate_in_parallel(dview)

    SNR, ber, ser = sim.get_data_to_be_plotted()

    modulator_obj = sim.params['modulator']
    # Can only plot if we simulated for more then one value of SNR
    if SNR.size > 1:
        semilogy(SNR, ber, '--g*', label='BER')
        semilogy(SNR, ser, '--b*', label='SER')
        xlabel('SNR')
        ylabel('Error')
        title('BER and SER for {0} modulation with Alamouti'.format(modulator_obj.name))
        legend()

        grid(True, which='both', axis='both')
        show()

    print sim.elapsed_time
