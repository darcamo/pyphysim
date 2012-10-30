#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perform the simulation of COmP transmission (using the BD algorithm)"""


import sys
sys.path.append('../../')

import numpy as np

from util import simulations, conversion
from comm import modulators, pathloss


class CompSimulationRunner(simulations.SimulationRunner):
    """Implements a simulation runner for a COmP transmission."""

    def __init__(self, ):
        simulations.SimulationRunner.__init__(self)

        # xxxxxxxxxx Channel Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.Nr = 2  # Number of receive antennas
        self.Nt = 2  # Number of transmit antennas
        self.Ns_BD = self.Nr  # Number of streams (per user) in the BD algorithm
        # self.AlphaValues = 0.2;
        # self.BetaValues = 0;
        self._path_loss_obj = pathloss.PathLoss3GPP1()

        # xxxxxxxxxx Cell and Grid Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.cell_radius = 1.000  # Cell radius (in Km)
        self.min_dist = 0.250  # Minimum allowed distance from a bse
                               # station and its user (same unit as
                               # cell_radius)
        #self.users_per_cell = 1  # Number of users in each cell
        self.num_cells = 3

        # xxxxxxxxxx Modulation Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.M = 4
        self.modulator = modulators.PSK(self.M)

        # xxxxxxxxxx Transmission Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.SNR = 20.
        #SNR = np.array([5., 10., 15.])
        self.NSymbs = 500
        # self.params.add('SNR', SNR)
        # self.params.set_unpack_parameter('SNR')
        self.N0 = -116.4  # Noise power (in dBm)

        # xxxxxxxxxx General Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.rep_max = 100  # Maximum number of repetitions for each
                            # unpacked parameters set self.params
                            # self.results
        self.max_bit_errors = 200  # Used in the _keep_going method to stop
                                   # the simulation earlier if possible.
        # self.progressbar_message = "COmP Simulation with {0}-PSK - SNR: {{SNR}}".format(self.M)

        # xxxxxxxxxx Dependent parameters (don't change these) xxxxxxxxxxxx
        #self.path_loss_border =

    def _run_simulation(self, current_parameters):
        """The _run_simulation method is where the actual code to simulate
        the system is.

        The implementation of this method is required by every subclass of
        SimulationRunner.

        Arguments:

        - `current_parameters`: SimulationParameters object with the
                                parameters for the simulation. The
                                self.params variable is not used
                                directly. It is first unpacked (in the
                                SimulationRunner.simulate method which then
                                calls _run_simulation) for each
                                combination.

        """
        # xxxxx Simulation Code here xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Transmit power at each base station
        # pt = CompSimulationRunner._calc_transmit_power(
        #     current_parameters['SNR'],
        #     self.N0,
        #     self.cell_radius,
        #     self._path_loss_obj)

        # xxxxx Return the Simulation results for this iteration xxxxxxxxxx
        simResults = simulations.SimulationResults()
        # simResults.add_result(symbolErrorsResult)
        # simResults.add_result(numSymbolsResult)
        # simResults.add_result(bitErrorsResult)
        # simResults.add_result(numBitsResult)
        # simResults.add_result(berResult)
        # simResults.add_result(serResult)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return simResults

    def _keep_going(self, current_sim_results):
        return True

    @staticmethod
    def _calc_transmit_power(SNR_dB, N0_dBm, cell_radius, path_loss_obj):
        """Calculates the required transmit power (in linear scale) to
        achieve the desired mean SNR value at the cell border.

        This method calculates the path loss at the cell border and
        then finds the transmit power that gives the desired mean SNR
        at the cell border.

        Arguments:
        - `SNR_dB`: SNR value (in dB)
        - `N0_dBm`: Noise power (in dBm)
        - `cell_radius`: Cell radius (in Km)
        - `path_loss_obj`: Object of a pathloss class used to calculate
                           the path loss.

        """
        path_loss_border = path_loss_obj.calc_path_loss(cell_radius)
        snr = conversion.dB2Linear(SNR_dB)
        pt = snr * conversion.dBm2Linear(N0_dBm) / path_loss_border
        return pt

if __name__ == '__main__':
    runner = CompSimulationRunner()

    runner.simulate()

    #print runner.results
