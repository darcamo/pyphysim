#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perform the simulation of the transmission using the Alamouti MIMO
scheme.

"""

from simulations import *
from comm import modulators, mimo
import misc
from util.conversion import dB2Linear


class AlamoutiSimulationRunner(SimulationRunner):
    """Implements a simulation runner for a transmission with the Alamouti
    MIMO scheme.

    """
    def __init__(self):
        SimulationRunner.__init__(self)

        # The _keep_going method will stop the simulation earlier when
        # max_bit_errors are achieved.
        self.max_bit_errors = 2000
        SNR = np.array([0., 5, 10, 15, 20])
        M = 16
        self.alamouti = mimo.Alamouti()
        self.NSymbs = 1000
        self.modulator = modulators.QAM(M)
        self.Nr = 2
        # Note that Nt is equal to 2 for the Alamouti scheme

        # xxxxx Declared in the SimulationRunner class xxxxxxxxxxxxxxxxxxxx
        # We need to set these two in all simulations
        self.rep_max = 5000
        self.progressbar_message = "Alamouti with {0}-QAM - SNR: {{SNR}}".format(M)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # We need to add the parameters to the self.param variable.
        self.params.add('SNR', SNR)
        self.params.set_unpack_parameter('SNR')

    def _run_simulation(self, current_parameters):
        # xxxxx Input parameters (set in the constructor) xxxxxxxxxxxxxxxxx
        #K = self.K
        M = self.modulator.M
        NSymbs = self.NSymbs
        Nr = self.Nr
        SNR = current_parameters["SNR"]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Input Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        inputData = np.random.randint(0, M, NSymbs)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Modulate input data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        modulatedData = self.modulator.modulate(inputData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Encode with Alamouti xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        transmit_signal = self.alamouti.encode(modulatedData)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        channel = misc.randn_c(Nr, 2)
        noiseVar = 1 / dB2Linear(SNR)
        awgn_noise = (misc.randn_c(Nr, NSymbs) * np.sqrt(noiseVar))
        received_signal = np.dot(channel, transmit_signal) + awgn_noise
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Decode Alamouti xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        alamouti_decoded_data = self.alamouti.decode(received_signal, channel)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Demodulate received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        demodulatedData = self.modulator.demodulate(alamouti_decoded_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the symbol and bit error rates xxxxxxxxxxxxxxxxx
        symbolErrors = sum(inputData != demodulatedData)
        aux = misc.xor(inputData, demodulatedData)
        # Count the number of bits in aux
        bitErrors = sum(misc.bitCount(aux))
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

    def _keep_going(self, simulation_results):
        #return True
        cumulated_bit_errors = simulation_results['bit_errors'][-1].get_result()
        return cumulated_bit_errors < self.max_bit_errors

    def get_data_to_be_plotted(self):
        """The get_data_to_be_plotted is not part of the simulation, but it
        is useful after the simulation is finished to get the results
        easily for plot.
        """
        ber = self.results.get_result_values_list('ber')
        ser = self.results.get_result_values_list('ber')

        # Get the SNR from the simulation parameters
        SNR = np.array(self.params['SNR'])

        return (SNR, ber, ser)


if __name__ == '__main__':
    from pylab import *

    #write_config_file_template()

    sim = AlamoutiSimulationRunner()
    sim.simulate()

    SNR, ber, ser = sim.get_data_to_be_plotted()

    # Can only plot if we simulated for more then one value of SNR
    if SNR.size > 1:
        semilogy(SNR, ber, '--g*', label='BER')
        semilogy(SNR, ser, '--b*', label='SER')
        xlabel('SNR')
        ylabel('Error')
        title('BER and SER for {0} modulation with Alamouti'.format(sim.modulator.name))
        legend()

        grid(True, which='both', axis='both')
        show()

    print sim.elapsed_time
