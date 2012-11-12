#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perform the simulation of COmP transmission (using the BD algorithm).

Different scenarios can be simulated such as:
- 'RandomUsers': One user at each cell and users are placed at a random
                 position in each cell.
# - 'SymmetricFar': User placed at symmetric locations at each cell as far as
#                   possible. This is shown in the figure below.

The external interference is generated in the
_prepare_external_interference method.

"""
# TODO: Create the several simulation scenarios

import sys
sys.path.append('../')

import numpy as np

from util import simulations, conversion, misc
from cell import cell
from comp import comp
from comm import pathloss, channels, modulators


class CompSimulationRunner(simulations.SimulationRunner):
    """Implements a simulation runner for a COmP transmission."""

    def __init__(self, ):
        simulations.SimulationRunner.__init__(self)

        # xxxxxxxxxx Cell and Grid Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.cell_radius = 1.0  # Cell radius (in Km)
        #self.min_dist = 0.250   # Minimum allowed distance from a bse
                                # station and its user (same unit as
                                # cell_radius)
        #self.users_per_cell = 1  # Number of users in each cell
        self.num_cells = 3
        self.num_clusters = 1

        # xxxxxxxxxx Channel Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.Nr = np.ones(self.num_cells) * 2  # Number of receive antennas
        self.Nt = np.ones(self.num_cells) * 2  # Number of transmit antennas
        self.Ns_BD = self.Nt  # Number of streams (per user) in the BD algorithm
        # self.AlphaValues = 0.2;
        # self.BetaValues = 0;
        self.path_loss_obj = pathloss.PathLoss3GPP1()
        self.multiuser_channel = channels.MultiUserChannelMatrixExtInt()

        # xxxxxxxxxx Modulation Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.M = 4
        self.modulator = modulators.PSK(self.M)

        # xxxxxxxxxx Transmission Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Number of symbols (per stream per user simulated at each
        # iteration of _run_simulation
        self.NSymbs = 500
        SNR = np.linspace(0, 30, 7)
        self.params.add('SNR', SNR)
        self.params.set_unpack_parameter('SNR')
        self.N0 = -116.4  # Noise power (in dBm)

        # xxxxxxxxxx External Interference Parameters xxxxxxxxxxxxxxxxxxxxx
        self.Pe_dBm = -10000  # transmit power (in dBm) of the ext. interference
        self.ext_int_rank = 1  # Rank of the external interference

        # xxxxxxxxxx General Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.rep_max = 10000  # Maximum number of repetitions for each
                              # unpacked parameters set self.params
                              # self.results

        # max_bit_errors is used in the _keep_going method to stop the
        # simulation earlier if possible. We stop the simulation if the
        # accumulated number of bit errors becomes greater then 5% of the
        # total number of simulated bits
        self.max_bit_errors = self.rep_max * self.NSymbs * 5. / 100.
        self.progressbar_message = "COmP Simulation with {0}-PSK - SNR: {{SNR}}".format(self.M)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Dependent parameters (don't change these) xxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Path loss (in linear scale) from the cell center to
        self.path_loss_border = self.path_loss_obj.calc_path_loss(self.cell_radius)
        # Cell Grid
        self.cell_grid = cell.Grid()
        self.cell_grid.create_clusters(self.num_clusters, self.num_cells, self.cell_radius)
        self.noise_var = conversion.dBm2Linear(self.N0)

        # External interference power
        self.pe = conversion.dBm2Linear(self.Pe_dBm)
        # xxxxxxxxxx Scenario specific variables xxxxxxxxxxxxxxxxxxxxxxxxxx
        # the scenario specific variables are created by running the
        # _create_users_according_to_scenario method.
        #
        # Here we set the _create_users_according_to_scenario method to be
        # one of the methods responsible for the users creation. Each
        # possible method corresponds to a different scenario.
        #self._create_users_according_to_scenario = self._create_random_users_scenario
        self._create_users_according_to_scenario = self._create_symmetric_far_away_users_scenario

    def _create_random_users_scenario(self):
        """Run this method to set variables specific to the 'RandomUsers'
        scenario.

        The 'RandomUsers' scenario places a user at a random location in
        each cell.

        """
        cluster0 = self.cell_grid._clusters[0]
        cell_ids = np.arange(1, self.num_cells + 1)
        cluster0.remove_all_users()
        cluster0.add_random_users(cell_ids)

    def _create_symmetric_far_away_users_scenario(self):
        """Run this method to set variables specific to the 'FarAwayUsers70%'
        scenario.

        The 'FarAwayUsers70%' scenario place a user in each cell at a the
        angle further away from the cell center in a distance from the cell
        center to the cell border equivalent to 70% of the cell radius.

        """
        cluster0 = self.cell_grid._clusters[0]
        cell_ids = np.arange(1, self.num_cells + 1)
        angles = np.array([210, -30, 90])
        cluster0.remove_all_users()
        cluster0.add_border_users(cell_ids, angles, 0.7)

    def _create_users_channels(self):
        """Create the channels of all the users.

        The users must have already been created.
        """
        cluster0 = self.cell_grid._clusters[0]

        # xxxxx Distances between each transmitter and each receiver xxxxxx
        # This `dists` matrix may be indexed as dists[user, cell].
        dists = cluster0.calc_dist_all_cells_to_all_users()
        # Path loss from each base station to each user
        pathloss = self.path_loss_obj.calc_path_loss(dists)

        # xxx Distances between each receiver and the ext. int. source xxxx
        # Calculates the distance of each user to the cluster center
        #
        # Note: Because we are using the cluster0.get_all_users() method
        # THIS CODE ONLY WORKS when there is a single user at each cell.
        distance_users_to_cluster_center = np.array(
            [cluster0.calc_dist(i) for i in cluster0.get_all_users()])

        pathlossInt = self.path_loss_obj.calc_path_loss(
            cluster0.external_radius - distance_users_to_cluster_center)
        # The number of rows is equal to the number of receivers, while the
        # cumber of columns is equal to the number of external interference
        # sources.
        pathlossInt.shape = (self.num_cells, 1)

        # Generate a random channel and set the path loss
        self.multiuser_channel.randomize(self.Nr, self.Nt, self.num_cells, self.ext_int_rank)
        self.multiuser_channel.set_pathloss(pathloss, pathlossInt)

    def _on_simulate_current_params_start(self, current_params):
        """This method is called once for each combination of transmit
        parameters.

        """
        # xxxxx Calculates the transmit power at each base station. xxxxxxx
        # Because this value does not change in the different iterations of
        # _run_simulation, but only when the parameters change the
        # calculation is performed here in the
        # _on_simulate_current_params_start.
        self.transmit_power = CompSimulationRunner._calc_transmit_power(
            current_params['SNR'],
            self.N0,
            self.cell_radius,
            self.path_loss_obj)

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
        # xxxxxxxxxx Prepare the scenario for this iteration. xxxxxxxxxxxxx
        # This will place the users at the locations specified by the
        # scenario (random locations or not), calculate the path loss and
        # generate a new random channel (in the self.multiuser_channel
        # variable).
        self._create_users_according_to_scenario()

        # This will calculate pathloss and generate random channels from
        # all transmitters to all receivers as well as from the external
        # interference sources to all receivers. This method must be called
        # after the _create_users_according_to_scenario method so that the
        # users are already created (we need their positions for the
        # pathloss)
        self._create_users_channels()

        # xxxxxxxxxx Generate the transmit symbols xxxxxxxxxxxxxxxxxxxxxxxx
        input_data = np.random.randint(0,
                                       self.M,
                                       [np.sum(self.Ns_BD), self.NSymbs])
        symbols = self.modulator.modulate(input_data)

        # xxxxxxxxxx Perform the block diagonalization xxxxxxxxxxxxxxxxxxxx
        (newH, Ms) = comp.perform_comp(
            # We only add the first np.sum(self.Nt) columns of big_H
            # because the remaining columns come from the external
            # interference sources, which don't participate in the Block
            # Diagonalization Process.
            self.multiuser_channel.big_H[:, 0:np.sum(self.Nt)],
            self.num_cells,
            self.transmit_power,
            #self.noise_var
            1e-50
        )

        # Prepare the transmit data. That is, the precoded_data as well as
        # the external interferece sources' data.
        precoded_data = np.dot(Ms, symbols)
        external_int_data = np.sqrt(self.pe) * misc.randn_c(self.ext_int_rank, self.NSymbs)
        all_data = np.vstack([precoded_data, external_int_data])

        #xxxxxxxxxx Pass the precoded data through the channel xxxxxxxxxxxx
        received_signal = self.multiuser_channel.corrupt_concatenated_data(
            all_data,
            self.noise_var
        )

        # xxxxxxxxxx Filter the received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        receive_filter = np.linalg.pinv(newH)
        received_symbols = np.dot(receive_filter, received_signal)

        # xxxxxxxxxx Demodulate the filtered symbols xxxxxxxxxxxxxxxxxxxxxx
        decoded_symbols = self.modulator.demodulate(received_symbols)

        # xxxxxxxxxx Calculates the Symbol Error Rate xxxxxxxxxxxxxxxxxxxxx
        num_symbol_errors = np.sum(decoded_symbols != input_data)
        num_symbols = input_data.size
        # SER = float(num_symbol_errors) / float(num_symbols)

        # xxxxxxxxxx Calculates the Bit Error Rate xxxxxxxxxxxxxxxxxxxxxxxx
        num_bit_errors = np.sum(misc.xor(decoded_symbols, input_data))
        num_bits = num_symbols * np.log2(self.M)
        # BER = float(num_bit_errors) / num_bits

        # xxxxx Return the Simulation results for this iteration xxxxxxxxxx
        ber_result = simulations.Result.create(
            'ber',
            simulations.Result.RATIOTYPE,
            num_bit_errors,
            num_bits)
        ser_result = simulations.Result.create(
            'ser',
            simulations.Result.RATIOTYPE,
            num_symbol_errors,
            num_symbols)

        simResults = simulations.SimulationResults()
        simResults.add_result(ber_result)
        simResults.add_result(ser_result)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return simResults

    def _keep_going(self, current_sim_results):
        ber_result = current_sim_results['ber'][-1]
        num_bit_errors = ber_result._value
        return num_bit_errors < self.max_bit_errors

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

    def get_data_to_be_plotted(self):
        """The get_data_to_be_plotted is not part of the simulation, but it is
        useful after the simulation is finished to get the results easily
        for plot.

        """
        ber = self.results.get_result_values_list('ber')
        ser = self.results.get_result_values_list('ser')

        # Get the SNR from the simulation parameters
        SNR = np.array(self.params['SNR'])

        return (SNR, ber, ser)

if __name__ == '__main__':
    # Lets import matplotlib if it is available
    try:
        from matplotlib import pyplot as plt
        _MATPLOTLIB_AVAILABLE = True
    except ImportError:
        _MATPLOTLIB_AVAILABLE = False

    runner = CompSimulationRunner()
    runner.simulate()

    # File name (without extension) for the figure and result files.
    results_filename = 'comp_results_1Km_radius_(Pure_BD_symetric_users_with_ext_int)_Pe_minus_10000_dBm'

    # xxxxxxxxxx Save the simulation results to a file xxxxxxxxxxxxxxxxxxxx
    # First we add the simulation parameters to the results
    runner.results.params = runner.params
    runner.results.save_to_file('{0}.pickle'.format(results_filename))

    # xxxxxxxxxx Plot the results xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # We can only plot the results if matplotlib is available
    if _MATPLOTLIB_AVAILABLE is True:

        SNR, ber, ser = runner.get_data_to_be_plotted()

        # Can only plot if we simulated for more then one value of SNR
        if SNR.size > 1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            p1 = ax.semilogy(SNR, ber, '--g*', label='BER')
            ax.hold(True)
            p2 = ax.semilogy(SNR, ser, '--b*', label='SER')

            plt.xlabel('SNR')
            plt.ylabel('Error')
            ax.set_title('BER and SER for a COmP simulation ({0} modulation)'.format(runner.modulator.name))
            plt.legend()

            plt.grid(True, which='both', axis='both')
            plt.show()

            fig.savefig('{0}.pdf'.format(results_filename))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    print "Elapsed Time: {0}".format(runner.elapsed_time)
