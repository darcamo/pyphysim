#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perform the simulation of CoMP transmission (using the BD algorithm).

Different scenarios can be simulated such as:
- 'RandomUsers': One user at each cell and users are placed at a random
                 position in each cell.
# - 'SymmetricFar': User placed at symmetric locations at each cell as far as
#                   possible. This is shown in the figure below.

The external interference is generated in the
_prepare_external_interference method.

"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import numpy as np
from scipy import linalg as sp_linalg

from util import simulations, conversion, misc
from cell import cell
from comp import comp
from comm import pathloss, channels, modulators


# def sum_user_data(data_all_users, Ns_all_users):
#     """Sum the data from the same user.

#     The multiple rows of `data_all_users` correspond to streams of the
#     different users. The `sum_user_data` function sum the data in the
#     different streams of each user (but not from different users) and
#     return a new matrix where each row corresponds to the summed data of a
#     user.

#     Parameters
#     ----------
#     data_all_users : 2D numpy array
#         Data from all users, with dimension sum(Ns) x num_data
#     Ns_all_users : 1D numpy array
#         An array containing the number of streams of each user.

#     Returns
#     -------
#     data_all_users2 : 2D numpy array
#         Data from all users, but with the data of each user summed.

#     Examples
#     --------
#     >>> data_all_users = np.array([285, 3, 1, 20, 7])
#     >>> Ns_all_users = np.array([2, 1, 2])
#     >>> print data_all_users
#     [285   3  13  20   7]
#     >>> data_all_users2 = sum_user_data(data_all_users, Ns_all_users)
#     >>> print data_all_users2
#     [288  13  27]
#     """
#     num_users = Ns_all_users.size
#     cum_Ns = np.hstack([0, np.cumsum(Ns_all_users)])

#     shape = Ns_all_users.shape
#     data_all_users2 = np.empty(shape, dtype=data_all_users.dtype)

#     for userindex in range(num_users):
#         data_all_users2[userindex] = np.sum(data_all_users[cum_Ns[userindex]:cum_Ns[userindex + 1]])

#     return data_all_users2


class CompSimulationRunner(simulations.SimulationRunner):
    """Implements a simulation runner for a CoMP transmission."""

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
        self.Nr = np.ones(self.num_cells, dtype=int) * 2  # N. of Rx antennas
        self.Nt = np.ones(self.num_cells, dtype=int) * 2  # N. of Tx antennas

        self.Ns_BD = self.Nt  # Number of streams (per user) in the BD alg.
        # self.AlphaValues = 0.2;
        # self.BetaValues = 0;
        self.path_loss_obj = pathloss.PathLoss3GPP1()
        self.multiuser_channel = channels.MultiUserChannelMatrixExtInt()

        # xxxxxxxxxx RandomState objects seeds xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # This is only useful to reproduce a simulation for debugging purposed
        channel_seed = None  # 22522
        self.noise_seed = None  # 4445
        self.data_gen_seed = np.random.randint(10000)  # 2105
        ext_data_gen_seed = None  # 6114
        #
        self.multiuser_channel.set_channel_seed(channel_seed)
        self.multiuser_channel.set_noise_seed(self.noise_seed)
        self.data_RS = np.random.RandomState(self.data_gen_seed)
        self.ext_data_RS = np.random.RandomState(ext_data_gen_seed)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Modulation Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.M = 4
        self.modulator = modulators.PSK(self.M)
        self.packet_length = 60

        # xxxxxxxxxx Transmission Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Number of symbols (per stream per user simulated at each
        # iteration of _run_simulation
        self.NSymbs = 500
        #SNR = np.linspace(0, 30, 7)
        SNR = np.linspace(0, 30, 16)
        #SNR = np.array([5])
        self.params.add('SNR', SNR)
        self.params.set_unpack_parameter('SNR')
        self.N0 = -116.4  # Noise power (in dBm)

        # xxxxxxxxxx External Interference Parameters xxxxxxxxxxxxxxxxxxxxx
        # transmit power (in dBm) of the ext. interference
        #Pe_dBm = np.array([-10000, -10, 0, 10, 20])
        Pe_dBm = np.array([-10, 0, 10])
        #Pe_dBm = np.array([-10])
        self.params.add('Pe_dBm', Pe_dBm)
        self.params.set_unpack_parameter('Pe_dBm')
        self.ext_int_rank = 1  # Rank of the external interference

        # xxxxx Metric used for the stream reduction xxxxxxxxxxxxxxxxxxxxxx
        # Metric used for the stream reduction decision used by the
        # CompExtInt class to mitigate external interference.

        #ext_int_comp_metric = [None, 'capacity', 'effective_throughput']
        #ext_int_comp_metric = [None, 'naive', 'fixed', 'capacity', 'effective_throughput']
        #ext_int_comp_metric = ['effective_throughput']
        #self.params.add('metric', ext_int_comp_metric)
        #self.params.set_unpack_parameter('metric')

        # xxxxxxxxxx General Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.rep_max = 5000  # Maximum number of repetitions for each
                              # unpacked parameters set self.params
                              # self.results

        # max_bit_errors is used in the _keep_going method to stop the
        # simulation earlier if possible. We stop the simulation if the
        # accumulated number of bit errors becomes greater then 5% of the
        # total number of simulated bits
        self.max_bit_errors = self.rep_max * self.NSymbs * 5. / 100.
        self.progressbar_message = "SNR: {{SNR}}, Pe_dBm: {{Pe_dBm}}".format(self.M)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Dependent parameters (don't change these) xxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # These two will be set in the _on_simulate_current_params_start
        # method
        self.pe = 0

        # Path loss (in linear scale) from the cell center to
        self.path_loss_border = self.path_loss_obj.calc_path_loss(self.cell_radius)
        # Cell Grid
        self.cell_grid = cell.Grid()
        self.cell_grid.create_clusters(self.num_clusters, self.num_cells, self.cell_radius)
        self.noise_var = conversion.dBm2Linear(self.N0)

        # xxxxxxxxxx Scenario specific variables xxxxxxxxxxxxxxxxxxxxxxxxxx
        # This must be either 'Symmetric Far Away' of 'Random'
        self._scenario = 'Symmetric Far Away'

        # the scenario specific variables are created by running the
        # _create_users_according_to_scenario method. Depending on the
        # value of self._scenario _create_users_according_to_scenario will
        # call the appropriated method.

    def _create_users_according_to_scenario(self):
        if self._scenario == 'Symmetric Far Away':
            self._create_symmetric_far_away_users_scenario()
        elif self._scenario == 'Random':
            self._create_random_users_scenario()

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
        transmit_power = CompSimulationRunner._calc_transmit_power(
            current_params['SNR'],
            self.N0,
            self.cell_radius,
            self.path_loss_obj)

        # External interference power
        self.pe = conversion.dBm2Linear(current_params['Pe_dBm'])

        # xxxxx Create the CoMP object with the None metric xxxxxxxxxxxxxxx
        self.comp_obj_None = comp.EnhancedBD(self.num_cells,
                                             transmit_power,
                                             self.noise_var,
                                             self.pe)
        self.comp_obj_None.set_ext_int_handling_metric(
            "None", {'modulator': self.modulator,
                     'packet_length': self.packet_length,
                     'num_streams': 1})
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Create the CoMP object with the Naive metric xxxxxxxxxxxxxx
        self.comp_obj_naive = comp.EnhancedBD(self.num_cells,
                                              transmit_power,
                                              self.noise_var,
                                              self.pe)
        self.comp_obj_naive.set_ext_int_handling_metric(
            "naive", {'modulator': self.modulator,
                                       'packet_length': self.packet_length,
                                       'num_streams': 1})
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Create the CoMP object with the fixed metric xxxxxxxxxxxxxxx
        self.comp_obj_fixed = comp.EnhancedBD(self.num_cells,
                                              transmit_power,
                                              self.noise_var,
                                              self.pe)
        self.comp_obj_fixed.set_ext_int_handling_metric(
            "fixed", {'modulator': self.modulator,
                                       'packet_length': self.packet_length,
                                       'num_streams': 1})
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Create the CoMP object with the capacity metric xxxxxxxxxxx
        self.comp_obj_capacity = comp.EnhancedBD(self.num_cells,
                                                 transmit_power,
                                                 self.noise_var,
                                                 self.pe)
        self.comp_obj_capacity.set_ext_int_handling_metric(
            "capacity", {'modulator': self.modulator,
                         'packet_length': self.packet_length,
                         'num_streams': 1})
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xx Create the CoMP object with the effective_throughput metric xx
        self.comp_obj_effec_throughput = comp.EnhancedBD(self.num_cells,
                                                         transmit_power,
                                                         self.noise_var,
                                                         self.pe)
        self.comp_obj_effec_throughput.set_ext_int_handling_metric(
            "effective_throughput", {'modulator': self.modulator,
                                     'packet_length': self.packet_length,
                                     'num_streams': 1})
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _run_simulation(self, current_parameters):
        """The _run_simulation method is where the actual code to simulate
        the system is.

        The implementation of this method is required by every subclass of
        SimulationRunner.

        Parameters
        ----------
        current_parameters : SimulationParameters object
            SimulationParameters object with the parameters for the
            simulation. The self.params variable is not used directly. It
            is first unpacked (in the SimulationRunner.simulate method
            which then calls _run_simulation) for each combination.

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
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Perform the block diagonalization for each metric xxxxxxxxx
        # None Metric
        (MsPk_all_users_None,
         Wk_all_users_None,
         Ns_all_users_None) = self.comp_obj_None.perform_BD_no_waterfilling(
             self.multiuser_channel)

        # Naive Metric
        (MsPk_all_users_naive,
         Wk_all_users_naive,
         Ns_all_users_naive) = self.comp_obj_naive.perform_BD_no_waterfilling(
             self.multiuser_channel)

        # Fixed Metric
        (MsPk_all_users_fixed,
         Wk_all_users_fixed,
         Ns_all_users_fixed) = self.comp_obj_fixed.perform_BD_no_waterfilling(
             self.multiuser_channel)

        # Capacity Metric
        (MsPk_all_users_capacity,
         Wk_all_users_capacity,
         Ns_all_users_capacity) = self.comp_obj_capacity.perform_BD_no_waterfilling(
             self.multiuser_channel)

        # effective_throughput Metric
        (MsPk_all_users_effec_throughput,
         Wk_all_users_effec_throughput,
         Ns_all_users_effec_throughput) = self.comp_obj_effec_throughput.perform_BD_no_waterfilling(
             self.multiuser_channel)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Below we will perform the transmission with the CoMP object for
        # each different metric

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxx None Case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Generate the transmit symbols xxxxxxxxxxxxxxxxxxxxxxxx
        Ns_total_None = np.sum(Ns_all_users_None)
        self.data_RS = np.random.RandomState(self.data_gen_seed)
        input_data_None = self.data_RS.randint(
            0,
            self.M,
            [Ns_total_None, self.NSymbs])
        symbols_None = self.modulator.modulate(input_data_None)

        # Prepare the transmit data. That is, the precoded_data as well as
        # the external interferece sources' data.
        precoded_data_None = np.dot(np.hstack(MsPk_all_users_None),
                                        symbols_None)
        external_int_data_all_metrics = np.sqrt(self.pe) * misc.randn_c_RS(self.ext_data_RS, self.ext_int_rank, self.NSymbs)
        all_data_None = np.vstack([precoded_data_None,
                                       external_int_data_all_metrics])

        #xxxxxxxxxx Pass the precoded data through the channel xxxxxxxxxxxx
        self.multiuser_channel.set_noise_seed(self.noise_seed)
        received_signal_None = self.multiuser_channel.corrupt_concatenated_data(
            all_data_None,
            self.noise_var
        )

        # xxxxxxxxxx Filter the received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Wk_None = sp_linalg.block_diag(*Wk_all_users_None)
        received_symbols_None = np.dot(Wk_None, received_signal_None)

        # xxxxxxxxxx Demodulate the filtered symbols xxxxxxxxxxxxxxxxxxxxxx
        decoded_symbols_None = self.modulator.demodulate(received_symbols_None)

        # xxxxxxxxxx Calculates the Symbol Error Rate xxxxxxxxxxxxxxxxxxxxx
        num_symbol_errors_None = np.sum(decoded_symbols_None != input_data_None, 1)
        # num_symbol_errors_None = sum_user_data(num_symbol_errors_None,
        #                                            Ns_all_users_None)
        num_symbols_None = np.ones(Ns_total_None) * input_data_None.shape[1]

        # xxxxxxxxxx Calculates the Bit Error Rate xxxxxxxxxxxxxxxxxxxxxxxx
        num_bit_errors_None = misc.count_bit_errors(decoded_symbols_None, input_data_None, 1)
        # num_bit_errors_None = sum_user_data(num_bit_errors_None,
        #                                         Ns_all_users_None)

        num_bits_None = num_symbols_None * np.log2(self.M)

        # xxxxxxxxxx Calculates the Package Error Rate xxxxxxxxxxxxxxxxxxxx
        ber_None = num_bit_errors_None / num_bits_None
        per_None = 1. - ((1. - ber_None) ** self.packet_length)
        num_packages_None = num_bits_None / self.packet_length
        num_package_errors_None = per_None * num_packages_None

        # xxxxxxxxxx Calculates the Spectral Efficiency xxxxxxxxxxxxxxxxxxx
        # nominal spectral Efficiency per stream
        nominal_spec_effic_None = self.modulator.K
        effective_spec_effic_None = (1 - per_None) * nominal_spec_effic_None

        # xxxxx Map the per stream metric to a global metric xxxxxxxxxxxxxx
        num_bit_errors_None = np.sum(num_bit_errors_None)
        num_bits_None = np.sum(num_bits_None)
        num_symbol_errors_None = np.sum(num_symbol_errors_None)
        num_symbols_None = np.sum(num_symbols_None)
        num_package_errors_None = np.sum(num_package_errors_None)
        num_packages_None = np.sum(num_packages_None)
        effective_spec_effic_None = np.sum(effective_spec_effic_None)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxx naive Case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Generate the transmit symbols xxxxxxxxxxxxxxxxxxxxxxxx
        Ns_total_naive = np.sum(Ns_all_users_naive)
        self.data_RS = np.random.RandomState(self.data_gen_seed)
        input_data_naive = self.data_RS.randint(
            0,
            self.M,
            [Ns_total_naive, self.NSymbs])
        symbols_naive = self.modulator.modulate(input_data_naive)

        # Prepare the transmit data. That is, the precoded_data as well as
        # the external interferece sources' data.
        precoded_data_naive = np.dot(np.hstack(MsPk_all_users_naive),
                                        symbols_naive)
        # external_int_data_all_metrics = np.sqrt(self.pe) * misc.randn_c(
        #     self.ext_int_rank, self.NSymbs)
        all_data_naive = np.vstack([precoded_data_naive,
                                       external_int_data_all_metrics])

        #xxxxxxxxxx Pass the precoded data through the channel xxxxxxxxxxxx
        self.multiuser_channel.set_noise_seed(self.noise_seed)
        received_signal_naive = self.multiuser_channel.corrupt_concatenated_data(
            all_data_naive,
            self.noise_var
        )

        # xxxxxxxxxx Filter the received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Wk_naive = sp_linalg.block_diag(*Wk_all_users_naive)
        received_symbols_naive = np.dot(Wk_naive, received_signal_naive)

        # xxxxxxxxxx Demodulate the filtered symbols xxxxxxxxxxxxxxxxxxxxxx
        decoded_symbols_naive = self.modulator.demodulate(received_symbols_naive)

        # xxxxxxxxxx Calculates the Symbol Error Rate xxxxxxxxxxxxxxxxxxxxx
        num_symbol_errors_naive = np.sum(decoded_symbols_naive != input_data_naive, 1)
        # num_symbol_errors_naive = sum_user_data(num_symbol_errors_naive,
        #                                            Ns_all_users_naive)
        num_symbols_naive = np.ones(Ns_total_naive) * input_data_naive.shape[1]

        # xxxxxxxxxx Calculates the Bit Error Rate xxxxxxxxxxxxxxxxxxxxxxxx
        num_bit_errors_naive = misc.count_bit_errors(decoded_symbols_naive, input_data_naive, 1)
        # num_bit_errors_naive = sum_user_data(num_bit_errors_naive,
        #                                         Ns_all_users_naive)

        num_bits_naive = num_symbols_naive * np.log2(self.M)

        # xxxxxxxxxx Calculates the Package Error Rate xxxxxxxxxxxxxxxxxxxx
        ber_naive = num_bit_errors_naive / num_bits_naive
        per_naive = 1. - ((1. - ber_naive) ** self.packet_length)
        num_packages_naive = num_bits_naive / self.packet_length
        num_package_errors_naive = per_naive * num_packages_naive

        # xxxxxxxxxx Calculates the Spectral Efficiency xxxxxxxxxxxxxxxxxxx
        # nominal spectral Efficiency per stream
        nominal_spec_effic_naive = self.modulator.K
        effective_spec_effic_naive = (1 - per_naive) * nominal_spec_effic_naive

        # xxxxx Map the per stream metric to a global metric xxxxxxxxxxxxxx
        num_bit_errors_naive = np.sum(num_bit_errors_naive)
        num_bits_naive = np.sum(num_bits_naive)
        num_symbol_errors_naive = np.sum(num_symbol_errors_naive)
        num_symbols_naive = np.sum(num_symbols_naive)
        num_package_errors_naive = np.sum(num_package_errors_naive)
        num_packages_naive = np.sum(num_packages_naive)
        effective_spec_effic_naive = np.sum(effective_spec_effic_naive)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxx fixed Case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Generate the transmit symbols xxxxxxxxxxxxxxxxxxxxxxxx
        Ns_total_fixed = np.sum(Ns_all_users_fixed)
        self.data_RS = np.random.RandomState(self.data_gen_seed)
        input_data_fixed = self.data_RS.randint(
            0,
            self.M,
            [Ns_total_fixed, self.NSymbs])
        symbols_fixed = self.modulator.modulate(input_data_fixed)

        # Prepare the transmit data. That is, the precoded_data as well as
        # the external interferece sources' data.
        precoded_data_fixed = np.dot(np.hstack(MsPk_all_users_fixed),
                                        symbols_fixed)
        # external_int_data_all_metrics = np.sqrt(self.pe) * misc.randn_c(
        #     self.ext_int_rank, self.NSymbs)
        all_data_fixed = np.vstack([precoded_data_fixed,
                                       external_int_data_all_metrics])

        #xxxxxxxxxx Pass the precoded data through the channel xxxxxxxxxxxx
        self.multiuser_channel.set_noise_seed(self.noise_seed)
        received_signal_fixed = self.multiuser_channel.corrupt_concatenated_data(
            all_data_fixed,
            self.noise_var
        )

        # xxxxxxxxxx Filter the received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Wk_fixed = sp_linalg.block_diag(*Wk_all_users_fixed)
        received_symbols_fixed = np.dot(Wk_fixed, received_signal_fixed)

        # xxxxxxxxxx Demodulate the filtered symbols xxxxxxxxxxxxxxxxxxxxxx
        decoded_symbols_fixed = self.modulator.demodulate(received_symbols_fixed)

        # xxxxxxxxxx Calculates the Symbol Error Rate xxxxxxxxxxxxxxxxxxxxx
        num_symbol_errors_fixed = np.sum(decoded_symbols_fixed != input_data_fixed, 1)
        # num_symbol_errors_fixed = sum_user_data(num_symbol_errors_fixed,
        #                                            Ns_all_users_fixed)
        num_symbols_fixed = np.ones(Ns_total_fixed) * input_data_fixed.shape[1]

        # xxxxxxxxxx Calculates the Bit Error Rate xxxxxxxxxxxxxxxxxxxxxxxx
        num_bit_errors_fixed = misc.count_bit_errors(decoded_symbols_fixed, input_data_fixed, 1)
        # num_bit_errors_fixed = sum_user_data(num_bit_errors_fixed,
        #                                         Ns_all_users_fixed)

        num_bits_fixed = num_symbols_fixed * np.log2(self.M)

        # xxxxxxxxxx Calculates the Package Error Rate xxxxxxxxxxxxxxxxxxxx
        ber_fixed = num_bit_errors_fixed / num_bits_fixed
        per_fixed = 1. - ((1. - ber_fixed) ** self.packet_length)
        num_packages_fixed = num_bits_fixed / self.packet_length
        num_package_errors_fixed = per_fixed * num_packages_fixed

        # xxxxxxxxxx Calculates the Spectral Efficiency xxxxxxxxxxxxxxxxxxx
        # nominal spectral Efficiency per stream
        nominal_spec_effic_fixed = self.modulator.K
        effective_spec_effic_fixed = (1 - per_fixed) * nominal_spec_effic_fixed

        # xxxxx Map the per stream metric to a global metric xxxxxxxxxxxxxx
        num_bit_errors_fixed = np.sum(num_bit_errors_fixed)
        num_bits_fixed = np.sum(num_bits_fixed)
        num_symbol_errors_fixed = np.sum(num_symbol_errors_fixed)
        num_symbols_fixed = np.sum(num_symbols_fixed)
        num_package_errors_fixed = np.sum(num_package_errors_fixed)
        num_packages_fixed = np.sum(num_packages_fixed)
        effective_spec_effic_fixed = np.sum(effective_spec_effic_fixed)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxx capacity Case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Generate the transmit symbols xxxxxxxxxxxxxxxxxxxxxxxx
        Ns_total_capacity = np.sum(Ns_all_users_capacity)
        self.data_RS = np.random.RandomState(self.data_gen_seed)
        input_data_capacity = self.data_RS.randint(
            0,
            self.M,
            [Ns_total_capacity, self.NSymbs])
        symbols_capacity = self.modulator.modulate(input_data_capacity)

        # Prepare the transmit data. That is, the precoded_data as well as
        # the external interferece sources' data.
        precoded_data_capacity = np.dot(np.hstack(MsPk_all_users_capacity),
                                        symbols_capacity)
        # external_int_data_all_metrics = np.sqrt(self.pe) * misc.randn_c(
        #     self.ext_int_rank, self.NSymbs)
        all_data_capacity = np.vstack([precoded_data_capacity,
                                       external_int_data_all_metrics])

        #xxxxxxxxxx Pass the precoded data through the channel xxxxxxxxxxxx
        self.multiuser_channel.set_noise_seed(self.noise_seed)
        received_signal_capacity = self.multiuser_channel.corrupt_concatenated_data(
            all_data_capacity,
            self.noise_var
        )

        # xxxxxxxxxx Filter the received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Wk_capacity = sp_linalg.block_diag(*Wk_all_users_capacity)
        received_symbols_capacity = np.dot(Wk_capacity, received_signal_capacity)

        # xxxxxxxxxx Demodulate the filtered symbols xxxxxxxxxxxxxxxxxxxxxx
        decoded_symbols_capacity = self.modulator.demodulate(received_symbols_capacity)

        # xxxxxxxxxx Calculates the Symbol Error Rate xxxxxxxxxxxxxxxxxxxxx
        num_symbol_errors_capacity = np.sum(decoded_symbols_capacity != input_data_capacity, 1)
        # num_symbol_errors_capacity = sum_user_data(num_symbol_errors_capacity,
        #                                            Ns_all_users_capacity)
        num_symbols_capacity = np.ones(Ns_total_capacity) * input_data_capacity.shape[1]

        # xxxxxxxxxx Calculates the Bit Error Rate xxxxxxxxxxxxxxxxxxxxxxxx
        num_bit_errors_capacity = misc.count_bit_errors(decoded_symbols_capacity, input_data_capacity, 1)
        # num_bit_errors_capacity = sum_user_data(num_bit_errors_capacity,
        #                                         Ns_all_users_capacity)

        num_bits_capacity = num_symbols_capacity * np.log2(self.M)

        # xxxxxxxxxx Calculates the Package Error Rate xxxxxxxxxxxxxxxxxxxx
        ber_capacity = num_bit_errors_capacity / num_bits_capacity
        per_capacity = 1. - ((1. - ber_capacity) ** self.packet_length)
        num_packages_capacity = num_bits_capacity / self.packet_length
        num_package_errors_capacity = per_capacity * num_packages_capacity

        # xxxxxxxxxx Calculates the Spectral Efficiency xxxxxxxxxxxxxxxxxxx
        # nominal spectral Efficiency per stream
        nominal_spec_effic_capacity = self.modulator.K
        effective_spec_effic_capacity = (1 - per_capacity) * nominal_spec_effic_capacity

        # xxxxx Map the per stream metric to a global metric xxxxxxxxxxxxxx
        num_bit_errors_capacity = np.sum(num_bit_errors_capacity)
        num_bits_capacity = np.sum(num_bits_capacity)
        num_symbol_errors_capacity = np.sum(num_symbol_errors_capacity)
        num_symbols_capacity = np.sum(num_symbols_capacity)
        num_package_errors_capacity = np.sum(num_package_errors_capacity)
        num_packages_capacity = np.sum(num_packages_capacity)
        effective_spec_effic_capacity = np.sum(effective_spec_effic_capacity)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxx effec_throughput Case xxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Generate the transmit symbols xxxxxxxxxxxxxxxxxxxxxxxx
        Ns_total_effec_throughput = np.sum(Ns_all_users_effec_throughput)
        self.data_RS = np.random.RandomState(self.data_gen_seed)
        input_data_effec_throughput = self.data_RS.randint(
            0,
            self.M,
            [Ns_total_effec_throughput, self.NSymbs])
        symbols_effec_throughput = self.modulator.modulate(input_data_effec_throughput)

        # Prepare the transmit data. That is, the precoded_data as well as
        # the external interferece sources' data.
        precoded_data_effec_throughput = np.dot(np.hstack(MsPk_all_users_effec_throughput),
                                        symbols_effec_throughput)
        # external_int_data_all_metrics = np.sqrt(self.pe) * misc.randn_c(
        #     self.ext_int_rank, self.NSymbs)
        all_data_effec_throughput = np.vstack([precoded_data_effec_throughput,
                                       external_int_data_all_metrics])

        #xxxxxxxxxx Pass the precoded data through the channel xxxxxxxxxxxx
        self.multiuser_channel.set_noise_seed(self.noise_seed)
        received_signal_effec_throughput = self.multiuser_channel.corrupt_concatenated_data(
            all_data_effec_throughput,
            self.noise_var
        )

        # xxxxxxxxxx Filter the received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Wk_effec_throughput = sp_linalg.block_diag(*Wk_all_users_effec_throughput)
        received_symbols_effec_throughput = np.dot(Wk_effec_throughput, received_signal_effec_throughput)

        # xxxxxxxxxx Demodulate the filtered symbols xxxxxxxxxxxxxxxxxxxxxx
        decoded_symbols_effec_throughput = self.modulator.demodulate(received_symbols_effec_throughput)

        # xxxxxxxxxx Calculates the Symbol Error Rate xxxxxxxxxxxxxxxxxxxxx
        num_symbol_errors_effec_throughput = np.sum(decoded_symbols_effec_throughput != input_data_effec_throughput, 1)
        # num_symbol_errors_effec_throughput = sum_user_data(num_symbol_errors_effec_throughput,
        #                                            Ns_all_users_effec_throughput)
        num_symbols_effec_throughput = np.ones(Ns_total_effec_throughput) * input_data_effec_throughput.shape[1]

        # xxxxxxxxxx Calculates the Bit Error Rate xxxxxxxxxxxxxxxxxxxxxxxx
        num_bit_errors_effec_throughput = misc.count_bit_errors(decoded_symbols_effec_throughput, input_data_effec_throughput, 1)
        # num_bit_errors_effec_throughput = sum_user_data(num_bit_errors_effec_throughput,
        #                                         Ns_all_users_effec_throughput)

        num_bits_effec_throughput = num_symbols_effec_throughput * np.log2(self.M)

        # xxxxxxxxxx Calculates the Package Error Rate xxxxxxxxxxxxxxxxxxxx
        ber_effec_throughput = num_bit_errors_effec_throughput / num_bits_effec_throughput
        per_effec_throughput = 1. - ((1. - ber_effec_throughput) ** self.packet_length)
        num_packages_effec_throughput = num_bits_effec_throughput / self.packet_length
        num_package_errors_effec_throughput = per_effec_throughput * num_packages_effec_throughput

        # xxxxxxxxxx Calculates the Spectral Efficiency xxxxxxxxxxxxxxxxxxx
        # nominal spectral Efficiency per stream
        nominal_spec_effic_effec_throughput = self.modulator.K
        effective_spec_effic_effec_throughput = (1 - per_effec_throughput) * nominal_spec_effic_effec_throughput

        # xxxxx Map the per stream metric to a global metric xxxxxxxxxxxxxx
        num_bit_errors_effec_throughput = np.sum(num_bit_errors_effec_throughput)
        num_bits_effec_throughput = np.sum(num_bits_effec_throughput)
        num_symbol_errors_effec_throughput = np.sum(num_symbol_errors_effec_throughput)
        num_symbols_effec_throughput = np.sum(num_symbols_effec_throughput)
        num_package_errors_effec_throughput = np.sum(num_package_errors_effec_throughput)

        num_packages_effec_throughput = np.sum(num_packages_effec_throughput)
        effective_spec_effic_effec_throughput = np.sum(effective_spec_effic_effec_throughput)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxx Return the Simulation results for this iteration xxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # None metric
        ber_result_None = simulations.Result.create(
            'ber_None',
            simulations.Result.RATIOTYPE,
            num_bit_errors_None,
            num_bits_None)
        ser_result_None = simulations.Result.create(
            'ser_None',
            simulations.Result.RATIOTYPE,
            num_symbol_errors_None,
            num_symbols_None)

        per_result_None = simulations.Result.create(
            'per_None',
            simulations.Result.RATIOTYPE,
            num_package_errors_None,
            num_packages_None)

        spec_effic_result_None = simulations.Result.create(
            'spec_effic_None',
            simulations.Result.RATIOTYPE,
            effective_spec_effic_None,
            1)

        # Naive metric
        ber_result_naive = simulations.Result.create(
            'ber_naive',
            simulations.Result.RATIOTYPE,
            num_bit_errors_naive,
            num_bits_naive)
        ser_result_naive = simulations.Result.create(
            'ser_naive',
            simulations.Result.RATIOTYPE,
            num_symbol_errors_naive,
            num_symbols_naive)

        per_result_naive = simulations.Result.create(
            'per_naive',
            simulations.Result.RATIOTYPE,
            num_package_errors_naive,
            num_packages_naive)

        spec_effic_result_naive = simulations.Result.create(
            'spec_effic_naive',
            simulations.Result.RATIOTYPE,
            effective_spec_effic_naive,
            1)

        # Fixed metric
        ber_result_fixed = simulations.Result.create(
            'ber_fixed',
            simulations.Result.RATIOTYPE,
            num_bit_errors_fixed,
            num_bits_fixed)
        ser_result_fixed = simulations.Result.create(
            'ser_fixed',
            simulations.Result.RATIOTYPE,
            num_symbol_errors_fixed,
            num_symbols_fixed)

        per_result_fixed = simulations.Result.create(
            'per_fixed',
            simulations.Result.RATIOTYPE,
            num_package_errors_fixed,
            num_packages_fixed)

        spec_effic_result_fixed = simulations.Result.create(
            'spec_effic_fixed',
            simulations.Result.RATIOTYPE,
            effective_spec_effic_fixed,
            1)

        # Capacity metric
        ber_result_capacity = simulations.Result.create(
            'ber_capacity',
            simulations.Result.RATIOTYPE,
            num_bit_errors_capacity,
            num_bits_capacity)
        ser_result_capacity = simulations.Result.create(
            'ser_capacity',
            simulations.Result.RATIOTYPE,
            num_symbol_errors_capacity,
            num_symbols_capacity)

        per_result_capacity = simulations.Result.create(
            'per_capacity',
            simulations.Result.RATIOTYPE,
            num_package_errors_capacity,
            num_packages_capacity)

        spec_effic_result_capacity = simulations.Result.create(
            'spec_effic_capacity',
            simulations.Result.RATIOTYPE,
            effective_spec_effic_capacity,
            1)

        # Effective Throughput metric
        ber_result_effec_throughput = simulations.Result.create(
            'ber_effec_throughput',
            simulations.Result.RATIOTYPE,
            num_bit_errors_effec_throughput,
            num_bits_effec_throughput)
        ser_result_effec_throughput = simulations.Result.create(
            'ser_effec_throughput',
            simulations.Result.RATIOTYPE,
            num_symbol_errors_effec_throughput,
            num_symbols_effec_throughput)

        per_result_effec_throughput = simulations.Result.create(
            'per_effec_throughput',
            simulations.Result.RATIOTYPE,
            num_package_errors_effec_throughput,
            num_packages_effec_throughput)

        spec_effic_result_effec_throughput = simulations.Result.create(
            'spec_effic_effec_throughput',
            simulations.Result.RATIOTYPE,
            effective_spec_effic_effec_throughput,
            1)

        simResults = simulations.SimulationResults()
        # Add the 'None' results
        simResults.add_result(ber_result_None)
        simResults.add_result(ser_result_None)
        simResults.add_result(per_result_None)
        simResults.add_result(spec_effic_result_None)

        # Add the naive results
        simResults.add_result(ber_result_naive)
        simResults.add_result(ser_result_naive)
        simResults.add_result(per_result_naive)
        simResults.add_result(spec_effic_result_naive)

        # Add the fixed results
        simResults.add_result(ber_result_fixed)
        simResults.add_result(ser_result_fixed)
        simResults.add_result(per_result_fixed)
        simResults.add_result(spec_effic_result_fixed)

        # Add the capacity results
        simResults.add_result(ber_result_capacity)
        simResults.add_result(ser_result_capacity)
        simResults.add_result(per_result_capacity)
        simResults.add_result(spec_effic_result_capacity)

        # Add the effective thoughput results
        simResults.add_result(ber_result_effec_throughput)
        simResults.add_result(ser_result_effec_throughput)
        simResults.add_result(per_result_effec_throughput)
        simResults.add_result(spec_effic_result_effec_throughput)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # print
        # print "SNR: {0}".format(current_parameters['SNR'])
        # print "None: {0}".format(effective_spec_effic_None)
        # print "Naive: {0}".format(effective_spec_effic_naive)
        # print "Fixed: {0}".format(np.sum(effective_spec_effic_fixed))
        # print "Capacity: {0}".format(effective_spec_effic_capacity)
        # print "Effec_Throu: {0}".format(effective_spec_effic_effec_throughput)
        # print
        # print "ber_None: {0}".format(ber_result_None.get_result())
        # print "ber_naive: {0}".format(ber_result_naive.get_result())
        # print "ber_fixed: {0}".format(ber_result_fixed.get_result())
        # print "ber_capacity: {0}".format(ber_result_capacity.get_result())
        # print "ber_effec_throughput: {0}".format(ber_result_effec_throughput.get_result())
        # print
        # print "Ns_all_users_None: {0}".format(Ns_all_users_None)
        # print "Ns_all_users_naive: {0}".format(Ns_all_users_naive)
        # print "Ns_all_users_effec_throughput: {0}".format(Ns_all_users_effec_throughput)
        # print "Ns_all_users_fixed: {0}".format(Ns_all_users_fixed)
        # print "Ns_all_users_capacity: {0}".format(Ns_all_users_capacity)

        # print
        # print ber_capacity
        # print ber_effec_throughput

        return simResults

    # def _keep_going(self, current_sim_results, current_rep):
    #     ber_result = current_sim_results['ber'][-1]
    #     num_bit_errors = ber_result._value
    #     return num_bit_errors < self.max_bit_errors

    @staticmethod
    def _calc_transmit_power(SNR_dB, N0_dBm, cell_radius, path_loss_obj):
        """Calculates the required transmit power (in linear scale) to
        achieve the desired mean SNR value at the cell border.

        This method calculates the path loss at the cell border and
        then finds the transmit power that gives the desired mean SNR
        at the cell border.

        Parameters
        ----------
        SNR_dB : SNR value (in dB)
        N0_dBm : Noise power (in dBm)
        cell_radius : Cell radius (in Km)
        path_loss_obj : Object of a pathloss class used to calculate the path loss.

        Returns
        -------
        transmit_power : float
            Desired transmit power (in linear scale).
        """
        path_loss_border = path_loss_obj.calc_path_loss(cell_radius)
        snr = conversion.dB2Linear(SNR_dB)
        pt = snr * conversion.dBm2Linear(N0_dBm) / path_loss_border
        return pt
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


def get_result_data(results, Pe_dBm, metric):
    """Get the result data (SNR, BER, SER, PER, Spec_Effic) for a specific
    Pe_dBm and metric

    Parameters
    ----------
    results : SimulationResults object
        The simulation results.
    Pe_dBm : float
        The desired external interference power in dB. Must be one of the
        values specified in `results.params`
    metric : str
        The metric used for external interference handling.

    Returns
    -------
    SNR : 1D numpy Array
        The SNR values
    BER : 1D numpy Array
        Bit Error Rate for each SNR value.
    SER : 1D numpy Array
        Symbol Error Rate for each SNR value.
    PER : 1D numpy Array
        Package Error Rate for each SNR value.
    Spec_Effic : 1D numpy Array
        Spec_Effic for each SNR value.

    """
    params = results.params
    SNR = params['SNR']

    ber_string = "ber_{0}".format(metric)
    ser_string = "ser_{0}".format(metric)
    per_string = "per_{0}".format(metric)
    spec_effic_string = "spec_effic_{0}".format(metric)

    ber_all = np.array(results.get_result_values_list(ber_string))
    ser_all = np.array(results.get_result_values_list(ser_string))
    per_all = np.array(results.get_result_values_list(per_string))
    spec_effic = np.array(results.get_result_values_list(spec_effic_string))

    result_indexes = params.get_pack_indexes(
        {'Pe_dBm': Pe_dBm, 'metric': metric})

    BER = ber_all[result_indexes]
    SER = ser_all[result_indexes]
    PER = per_all[result_indexes]
    Spec_Effic = spec_effic[result_indexes]
    reps = results.runned_reps

    # SNR, BER, SER, PER, Spectral Efficiency, number of simulated repetitions
    return (SNR, BER, SER, PER, Spec_Effic, reps)


def plot_error_results_fixed_Pe_metric(results, Pe_dBm, metric):
    """Plot the error results for a specified Pe_dBm and metric.

    Parameters
    ----------
    results : SimulationResults object
        The simulation results.
    Pe_dBm : float
        The desired external interference power in dB. Must be one of the
        values specified in `results.params`
    metric : str
        The metric used for external interference handling. Must be one of
        the values specified in `results.params`

    Returns
    -------
    fig : A Matplotlib figure

        A Matplotlib figure with the plot of the error rates. You can call
        the 'save' method of `fig` to save the figure to a file.

    fig_data : A tuple
        A tuple with the data used to plot the figure.

    """
    from matplotlib import pyplot as plt

    (SNR, BER, SER, PER, Spec_Effic, reps) = get_result_data(results,
                                                             Pe_dBm,
                                                             metric)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Plot the Error rates (SER, BER and PER)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot the Symbol Error Rate
    ax.semilogy(SNR, SER, 'g-s', label='SER')
    ax.hold(True)

    # Plot the Bit Error Rate
    ax.semilogy(SNR, BER, 'b-^', label='BER')

    # Plot the Package Error Rate
    ax.semilogy(SNR, PER, 'k-o', label='PER')

    plt.xlabel('SNR')
    plt.ylabel('Error Rate')
    ax.set_title('Enhanced BD simulation: Error Rates x SNR')
    plt.legend()

    plt.grid(True, which='both', axis='both')

    return fig  # , (SNR, BER, SER, PER, Spec_Effic, reps)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


def plot_spec_effic(results, Pe_dBm, metric):
    from matplotlib import pyplot as plt

    (SNR, BER, SER, PER, Spec_Effic, reps) = get_result_data(results,
                                                             Pe_dBm,
                                                             metric)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Plot The Spectral Efficiency
    fig = plt.figure()
    ax2 = fig.add_subplot(111)

    # Plot the Symbol Error Rate
    ax2.plot(SNR, Spec_Effic, 'g-*', label='Spectral Efficiency')

    plt.xlabel('SNR')
    plt.ylabel('Spectral Efficiency')
    ax2.set_title('Enhanced BD simulation (Pe: {0} dBm, metric: {1})'.format(
        Pe_dBm,
        metric))
    #plt.legend()

    plt.grid(True, which='both', axis='both')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return fig


def plot_spectral_efficience_all_metrics(results, Pe_dBm):
    from matplotlib import pyplot as plt

    params = results.params
    SNR = params['SNR']
    spec_effic_None = np.array(results.get_result_values_list('spec_effic_None'))
    spec_effic_naive = np.array(results.get_result_values_list('spec_effic_naive'))
    spec_effic_fixed = np.array(results.get_result_values_list('spec_effic_fixed'))
    spec_effic_capacity = np.array(results.get_result_values_list('spec_effic_capacity'))
    spec_effic_effective_throughput = np.array(results.get_result_values_list('spec_effic_effec_throughput'))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    result_indexes = params.get_pack_indexes(
        {'Pe_dBm': Pe_dBm})

    # xxxxx Plot The Spectral Efficiency with no stream reduction xxxxxxxxx
    ax.plot(SNR, spec_effic_None[result_indexes],
            'g-o', label='No Stream Reduction')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Plot The Spectral Efficiency with capacity metric xxxxxxxxxxxxx
    ax.plot(SNR, spec_effic_capacity[result_indexes],
            'b-s', label='Capacity Metric')

    # xxxxx Plot the Spec. Effic. with effective_throughput metric xxxxxxxx
    ax.plot(SNR, spec_effic_effective_throughput[result_indexes],
            'k-*', label='Effective Throughput Metric')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Plot the Spec. Effic. with naive metric xxxxxxxxxxxxxxxxxxxxxxx
    ax.plot(SNR, spec_effic_naive[result_indexes],
            'm--', label='Naive Case')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Plot the Spec. Effic. with fixed metric xxxxxxxxxxxxxxxxxxxxxxx
    ax.plot(SNR, spec_effic_fixed[result_indexes],
            'r-^', label='Fixed Case')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    plt.xlabel('SNR (dB)')
    plt.ylabel('Spectral Efficiency (bits/channel use)')
    ax.set_title('Enhanced BD simulation: Spectral Efficiency')
    plt.legend(loc=0)

    plt.grid(True, which='both', axis='both')

    return fig


def plot_per_all_metrics(results, Pe_dBm):
    from matplotlib import pyplot as plt

    params = results.params
    SNR = params['SNR']
    per_None = np.array(results.get_result_values_list('per_None'))
    per_naive = np.array(results.get_result_values_list('per_naive'))
    per_fixed = np.array(results.get_result_values_list('per_fixed'))
    per_capacity = np.array(results.get_result_values_list('per_capacity'))
    per_effective_throughput = np.array(results.get_result_values_list('per_effec_throughput'))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    result_indexes = params.get_pack_indexes(
        {'Pe_dBm': Pe_dBm})

    # xxxxx Plot The Spectral Efficiency with no stream reduction xxxxxxxxx
    ax.plot(SNR, per_None[result_indexes],
            'g-o', label='No Stream Reduction')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Plot The Spectral Efficiency with capacity metric xxxxxxxxxxxxx
    ax.plot(SNR, per_capacity[result_indexes],
            'b-s', label='Capacity Metric')

    # xxxxx Plot the Spec. Effic. with effective_throughput metric xxxxxxxx
    ax.plot(SNR, per_effective_throughput[result_indexes],
            'k-*', label='Effective Throughput Metric')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Plot the Spec. Effic. with naive metric xxxxxxxxxxxxxxxxxxxxxxx
    ax.plot(SNR, per_naive[result_indexes],
            'm--', label='Naive Case')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Plot the Spec. Effic. with fixed metric xxxxxxxxxxxxxxxxxxxxxxx
    ax.plot(SNR, per_fixed[result_indexes],
            'r-^', label='Fixed Case')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    plt.xlabel('SNR (dB)')
    plt.ylabel('Packet Error Rate')
    ax.set_title('Enhanced BD simulation: Packet Error Rate')
    plt.legend(loc=0)

    plt.grid(True, which='both', axis='both')

    return fig


## xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    # Lets import matplotlib if it is available
    try:
        from matplotlib import pyplot as plt
        _MATPLOTLIB_AVAILABLE = True
    except ImportError:
        _MATPLOTLIB_AVAILABLE = False

    from apps.simulate_comp import CompSimulationRunner

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # File name (without extension) for the figure and result files.
    results_filename = 'comp_results'
    runner = CompSimulationRunner()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    ## xxxxxxxxxx Perform the simulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # The simulation will be run either in parallel or serially depending
    # if the IPython engines are running or not.
    run_in_parallel = True
    try:
        # If we can get an IPython view that means that the IPython engines
        # are running. In that case we will perform the simulation in
        # parallel
        from IPython.parallel import Client
        # cl = Client(profile="ssh")
        cl = Client(profile="default")
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
    except Exception:
        # If we can't get an IPython view then we will perform the
        # simulation serially
        run_in_parallel = False

    if run_in_parallel is True:
        print("Simulation will be run in Parallel")
        # Remove the " - SNR: {SNR}" string in the progressbar message,
        # since when the simulation is performed in parallel we get a
        # single progressbar for all the simulation.
        runner.progressbar_message = 'Elapsed Time: {{elapsed_time}}'
        runner.simulate_in_parallel(lview, results_filename=results_filename)
    else:
        print("Simulation will be run serially")
        runner.simulate(results_filename)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #
    #
    print "Runned iterations: {0}".format(runner.runned_reps)
    print "Elapsed Time: {0}".format(runner.elapsed_time)
    #
    #
    #
    ## xxxxxxxx Load the results from the file xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results_filename = 'comp_results'
    results = simulations.SimulationResults.load_from_file(
        '{0}{1}'.format(results_filename, '.pickle'))

    SNR = results.params['SNR']
    if _MATPLOTLIB_AVAILABLE is True and SNR.size > 1:
        Pe_dBm = 10

        # error_None_fig = plot_error_results_fixed_Pe_metric(
        #     results, Pe_dBm, 'None')
        # error_capacity_fig = plot_error_results_fixed_Pe_metric(
        #     results, Pe_dBm, 'capacity')
        # error_fixed_fig = plot_error_results_fixed_Pe_metric(
        #     results, Pe_dBm, 'fixed')
        # error_naive_fig = plot_error_results_fixed_Pe_metric(
        #     results, Pe_dBm, 'naive')
        # error_effec_throughput_fig = plot_error_results_fixed_Pe_metric(
        #     results, Pe_dBm, 'effec_throughput')

        # plot_spec_effic(results, Pe_dBm, None)
        # plot_spec_effic(results, Pe_dBm, 'naive')
        # plot_spec_effic(results, Pe_dBm, 'fixed')
        # plot_spec_effic(results, Pe_dBm, 'capacity')
        # plot_spec_effic(results, Pe_dBm, 'effec_throughput')

        # Save the Spectral Efficiency curve for the given Pe_dBm
        spec_fig = plot_spectral_efficience_all_metrics(results, Pe_dBm)
        # spec_fig.tight_layout()
        spec_fig.subplots_adjust(bottom=0.08,right=0.98,top=0.95,left=0.07)
        spec_fig.savefig('{0}_Pe_{1}_spec_effic.pgf'.format(results_filename, Pe_dBm))

        per_all_fig = plot_per_all_metrics(results, Pe_dBm)
        # per_all_fig.tight_layout()
        per_all_fig.subplots_adjust(bottom=0.08,right=0.98,top=0.95,left=0.07)
        per_all_fig.savefig('{0}_Pe_{1}_per_all.pgf'.format(results_filename, Pe_dBm))


        # # Save the Error rates curves for each metric for the given Pe_dBm
        # error_None_fig.savefig('{0}_Pe_{1}_error_rates_None.pgf'.format(results_filename, Pe_dBm))
        # error_naive_fig.savefig('{0}_Pe_{1}_error_rates_naive.pgf'.format(results_filename, Pe_dBm))
        # error_fixed_fig.savefig('{0}_Pe_{1}_error_rates_fixed.pgf'.format(results_filename, Pe_dBm))
        # error_capacity_fig.savefig('{0}_Pe_{1}_error_rates_capacity.pgf'.format(results_filename, Pe_dBm))
        # error_effec_throughput_fig.savefig('{0}_Pe_{1}_error_rates_spec_eff.pgf'.format(results_filename, Pe_dBm))

        plt.show()
