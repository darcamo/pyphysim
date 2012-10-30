#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perform the simulation of COmP transmission (using the BD algorithm).

Different scenarios can be simulated such as:
- 'RandomUsers': One user at each cell and users are placed at a random
                 position in each cell.
# - 'SymmetricFar': User placed at symmetric locations at each cell as far as
#                   possible. This is shown in the figure below.

"""
# TODO: Create the several simulation scenarios

import sys
sys.path.append('../../')

import numpy as np

from util import simulations, conversion
from comm import modulators, pathloss
from cell import cell


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
        self.cell_radius = 0.5  # Cell radius (in Km)
        self.min_dist = 0.250   # Minimum allowed distance from a bse
                                # station and its user (same unit as
                                # cell_radius)
        #self.users_per_cell = 1  # Number of users in each cell
        self.num_cells = 3
        self.num_clusters = 1

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

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Dependent parameters (don't change these) xxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Path loss (in linear scale) from the cell center to
        self.path_loss_border = self._path_loss_obj.calc_path_loss(self.cell_radius)
        # Cell Grid
        self.cell_grid = cell.Grid()
        self.cell_grid.create_clusters(self.num_clusters, self.num_cells, self.cell_radius)

        # xxxxxxxxxx Scenario specific variables xxxxxxxxxxxxxxxxxxxxxxxxxx
        # the scenario specific variables are created by running the
        # _prepare_scenario method.
        #
        # Here we set the _prepare_scenario method to be the method to
        # prepare the RandomUsers scenario.
        self._prepare_scenario = self._prepare_random_users_scenario
        #self._prepare_scenario = self._prepare_symmetric_far_away_scenario

    def _prepare_random_users_scenario(self):
        """Run this method to set variables specific to the 'RandomUsers' scenario.

        The 'RandomUsers'
        """
        import pudb
        cluster0 = self.cell_grid._clusters[0]
        cell_ids = np.arange(1, self.num_cells + 1)
        cluster0.remove_all_users()
        cluster0.add_random_users(cell_ids)
        pudb.set_trace()

        # Distances between each transmitter and each receiver
        dists = np.zeros([self.num_cells, self.num_cells], dtype=float)
        for base_station in range(self.num_cells):
            for mobile_station in range(self.num_cells):
                # Remember that the Base Stations are the transmitters and
                # the mobile stations (one in each cell) are the receivers
                #dists[mobile_station, base_station] = cluster0
                pass



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
        # recalculate the scenario (and user positions) at each iteration
        self._prepare_scenario()

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
    print "Elapsed Time: {0}".format(runner.elapsed_time)

    #print runner.results
