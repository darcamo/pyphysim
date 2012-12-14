#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simulate the Interference Alignment algorithm described in the paper
"Interference Alignment via Alternating Minimization"

"""
# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

from util.simulations import *
from comm import modulators
from util.conversion import dB2Linear
from util import misc
from ia import ia


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class AlternatingSimulationRunner(SimulationRunner):
    """Implements a simulation runner for a transmission with the
    Alternating Minimizations Interference Alignment Algorithm.
    """
    def __init__(self):
        SimulationRunner.__init__(self)

        # The _keep_going method will stop the simulation earlier when
        # max_bit_errors are achieved.
        self.max_bit_errors = 3000

        #SNR = np.array([0., 3., 6, 9])
        # SNR = np.array([0., 3, 6, 9, 12])
        SNR = np.array([0., 5, 10, 15, 20, 25, 30])
        #SNR = np.array([50])
        M = 16
        self.NSymbs = 50
        self.modulator = modulators.PSK(M)
        self.K = 3
        self.Nr = np.ones(self.K) * 2
        self.Nt = np.ones(self.K) * 2
        self.Ns = np.ones(self.K) * 1
        # Iterations of the AlternatingMinIASolver algorithm.
        self.alternating_iterations = 50

        self.ia_solver = ia.AlternatingMinIASolver()

        # xxxxx Declared in the SimulationRunner class xxxxxxxxxxxxxxxxxxxx
        # We need to set these two in all simulations
        self.rep_max = 1000
        #self.rep_max = 200
        self.progressbar_message = "Alternating Min. ({0}-QAM mod.) - SNR: {{SNR}}".format(M)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # We need to add the parameters to the self.param variable.
        self.params.add('SNR', SNR)
        self.params.set_unpack_parameter('SNR')

    def _run_simulation(self, current_parameters):
        # xxxxx Input parameters (set in the constructor) xxxxxxxxxxxxxxxxx
        K = self.K
        M = self.modulator.M
        NSymbs = self.NSymbs
        Nr = self.Nr
        Nt = self.Nt
        Ns = self.Ns
        SNR = current_parameters["SNR"]

        # print "Simulation Parameters"
        # print "K: {K}\nNr: {Nr}\nNt: {Nt}\nNs: {Ns}\nNSymbs: {NSymbs}".format(**locals())
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

        self.ia_solver.randomizeH(Nr, Nt, K)
        self.ia_solver.randomizeF(Nt, Ns, K)
        for i in range(self.alternating_iterations):
            self.ia_solver.step()

        transmit_signal_precoded = map(np.dot, self.ia_solver.F, transmit_signal)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        noise_var = 1 / dB2Linear(SNR)
        multi_user_channel = self.ia_solver._multiUserChannel
        # received_data is an array of matrices, one matrix for each receiver.
        received_data = multi_user_channel.corrupt_data(
            transmit_signal_precoded, noise_var)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Perform the Interference Cancelation xxxxxxxxxxxxxxxxxxxxxx
        received_data_no_interference = map(np.dot,
                                            self.ia_solver.W, received_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Demodulate Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        received_data_no_interference = np.vstack(received_data_no_interference)
        demodulated_data = self.modulator.demodulate(received_data_no_interference)
        # demodulated_data = map(self.modulator.demodulate, received_data_no_interference)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Debug xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # print "IA Cost: {0:f}".format(self.ia_solver.getCost())
        # print inputData - demodulated_data
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the symbol and bit error rates xxxxxxxxxxxxxxxxx
        symbolErrors = np.sum(inputData != demodulated_data)
        bitErrors = misc.count_bit_errors(inputData, demodulated_data)
        numSymbols = inputData.size
        numBits = inputData.size * modulators.level2bits(M)
        ia_cost = self.ia_solver.getCost()
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

        simResults = SimulationResults()
        simResults.add_result(symbolErrorsResult)
        simResults.add_result(numSymbolsResult)
        simResults.add_result(bitErrorsResult)
        simResults.add_result(numBitsResult)
        simResults.add_result(berResult)
        simResults.add_result(serResult)
        simResults.add_result(ia_costResult)
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


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx Main xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Run the AlternatingSimulationRunner and plot the results
if __name__ == '__main__':
    from pylab import *

    sim = AlternatingSimulationRunner()
    sim.simulate()

    SNR, ber, ser = sim.get_data_to_be_plotted()

    # Can only plot if we simulated for more then one value of SNR
    if SNR.size > 1:
        semilogy(SNR, ber, '--g*', label='BER')
        semilogy(SNR, ser, '--b*', label='SER')
        xlabel('SNR')
        ylabel('Error')
        title('Interference Alignment\nK={0}, Nr={1}, Nt={2}, Ns={3} System'.format(sim.K, sim.Nr, sim.Nt, sim.Ns))
        legend()

        grid(True, which='both', axis='both')
        show()

    print "Runned iterations: {0}".format(sim.runned_reps)
    print sim.elapsed_time
