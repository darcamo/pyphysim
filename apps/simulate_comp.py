#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Perform the simulation of the Block Diagonalization (BD) algorithm
(several variations of the BD algorithm).

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
from comm import pathloss, channels, modulators
from comm.blockdiagonalization import EnhancedBD, WhiteningBD


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


class BDSimulationRunner(simulations.SimulationRunner):
    """
    Implements a simulation runner for a Block Diagonalization
    transmission.
    """

    def __init__(self, ):
        simulations.SimulationRunner.__init__(self)

        # xxxxxxxxxx DARLAN xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        spec = """[Grid]
        cell_radius=float(min=0.01, default=1.0)
        num_cells=integer(min=3,default=3)
        num_clusters=integer(min=1,default=1)

        [Scenario]
        NSymbs=integer(min=10, max=1000000, default=500)
        SNR=real_numpy_array(min=-50, max=100, default=0:3:31)
        Pe_dBm=real_numpy_array(min=-50, max=100, default=[-10. 0. 10.])
        Nr=integer(default=2)
        Nt=integer(default=2)
        N0=float(default=-116.4)
        ext_int_rank=integer(min=1,default=1)
        user_positioning_method=option("Random", 'Symmetric Far Away', default="Symmetric Far Away")

        [Modulation]
        M=integer(min=4, max=512, default=4)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="PSK")
        packet_length=integer(min=1,default=60)

        [General]
        rep_max=integer(min=1, default=5000)
        unpacked_parameters=string_list(default=list('SNR','Pe_dBm'))

        """.split("\n")
        self.params = simulations.SimulationParameters.load_from_config_file(
            'bd_config_file.txt',
            spec,
            save_parsed_file=True)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Channel Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # self.AlphaValues = 0.2;
        # self.BetaValues = 0;
        self.path_loss_obj = pathloss.PathLoss3GPP1()
        self.multiuser_channel = channels.MultiUserChannelMatrixExtInt()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

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

        # xxxxxxxxxx Creates the modulator object xxxxxxxxxxxxxxxxxxxxxxxxx
        M = self.params['M']
        modulator_options = {'PSK': modulators.PSK,
                             'QPSK': modulators.QPSK,
                             'QAM': modulators.QAM,
                             'BPSK': modulators.BPSK}
        self.modulator = modulator_options[self.params['modulator']](M)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Transmission Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Number of symbols (per stream per user simulated at each
        # iteration of _run_simulation
        # self.NSymbs = 500
        # #SNR = np.linspace(0, 30, 7)
        # SNR = np.linspace(0, 30, 16)
        # self.params.add('SNR', SNR)
        # self.params.set_unpack_parameter('SNR')
        # self.N0 = -116.4  # Noise power (in dBm)

        # xxxxxxxxxx External Interference Parameters xxxxxxxxxxxxxxxxxxxxx
        # transmit power (in dBm) of the ext. interference
        #Pe_dBm = np.array([-10000, -10, 0, 10, 20])
        # Pe_dBm = np.array([-10, 0, 10])
        # self.params.add('Pe_dBm', Pe_dBm)
        # self.params.set_unpack_parameter('Pe_dBm')
        # self.ext_int_rank = 1  # Rank of the external interference

        # xxxxxxxxxx General Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.rep_max = self.params['rep_max']
        # self.rep_max = 5000  # Maximum number of repetitions for each
        #                      # unpacked parameters set self.params
        #                      # self.results

        # max_bit_errors is used in the _keep_going method to stop the
        # simulation earlier if possible. We stop the simulation if the
        # accumulated number of bit errors becomes greater then 5% of the
        # total number of simulated bits
        NSymbs = self.params['NSymbs']
        M = self.params['M']
        self.max_bit_errors = self.rep_max * NSymbs * 5. / 100.
        self.progressbar_message = "SNR: {{SNR}}, Pe_dBm: {{Pe_dBm}}".format(M)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Dependent parameters (don't change these) xxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # These two will be set in the _on_simulate_current_params_start
        # method
        self.pe = 0

        # Path loss (in linear scale) from the cell center to
        # self.path_loss_border = self.path_loss_obj.calc_path_loss(self.cell_radius)

        # Cell Grid
        self.cell_grid = cell.Grid()
        self.cell_grid.create_clusters(self.params['num_clusters'],
                                       self.params['num_cells'],
                                       self.params['cell_radius'])
        self.noise_var = conversion.dBm2Linear(self.params['N0'])

        # xxxxxxxxxx Scenario specific variables xxxxxxxxxxxxxxxxxxxxxxxxxx
        # This must be either 'Symmetric Far Away' of 'Random'
        #self._scenario = self.params['user_positioning_method']

        # the scenario specific variables are created by running the
        # _create_users_according_to_scenario method. Depending on the
        # value of self._scenario _create_users_according_to_scenario will
        # call the appropriated method.

    def _create_users_according_to_scenario(self, current_params):
        scenario = current_params['user_positioning_method']
        if scenario == 'Symmetric Far Away':
            self._create_symmetric_far_away_users_scenario(current_params)
        elif scenario == 'Random':
            self._create_random_users_scenario(current_params)
        else:
            raise RuntimeError(
                "Invalid scenario: {0}".format(self._scenario))

    def _create_random_users_scenario(self, current_params):
        """Run this method to set variables specific to the 'RandomUsers'
        scenario.

        The 'RandomUsers' scenario places a user at a random location in
        each cell.

        """
        cluster0 = self.cell_grid._clusters[0]
        cell_ids = np.arange(1, current_params['num_cells'] + 1)
        cluster0.remove_all_users()
        cluster0.add_random_users(cell_ids)

    def _create_symmetric_far_away_users_scenario(self, current_params):
        """Run this method to set variables specific to the 'FarAwayUsers70%'
        scenario.

        The 'FarAwayUsers70%' scenario place a user in each cell at a the
        angle further away from the cell center in a distance from the cell
        center to the cell border equivalent to 70% of the cell radius.

        """
        cluster0 = self.cell_grid._clusters[0]
        cell_ids = np.arange(1, current_params['num_cells'] + 1)
        angles = np.array([210, -30, 90])
        cluster0.remove_all_users()
        cluster0.add_border_users(cell_ids, angles, 0.7)

    def _create_users_channels(self, current_params):
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
        pathlossInt.shape = (current_params['num_cells'], 1)

        # Generate a random channel and set the path loss
        self.multiuser_channel.randomize(current_params['Nr'],
                                         current_params['Nt'],
                                         current_params['num_cells'],
                                         current_params['ext_int_rank'])
        self.multiuser_channel.set_pathloss(pathloss, pathlossInt)

    def _on_simulate_current_params_start(self, current_params):
        """This method is called once for each combination of transmit
        parameters.

        """
        # IMPORTANT: Re-seed the channel and the noise RandomState
        # objects. Without this, when you perform the simulation in
        # parallel (call the simulate_in_parallel method(from the
        # SimulationRunner class) you will get the same channel samples and
        # noise for all parallel process.
        self.multiuser_channel.re_seed()

        # xxxxx Calculates the transmit power at each base station. xxxxxxx
        # Because this value does not change in the different iterations of
        # _run_simulation, but only when the parameters change the
        # calculation is performed here in the
        # _on_simulate_current_params_start.
        transmit_power = BDSimulationRunner._calc_transmit_power(
            current_params['SNR'],
            current_params['N0'],
            current_params['cell_radius'],
            self.path_loss_obj)

        # External interference power
        self.pe = conversion.dBm2Linear(current_params['Pe_dBm'])

        # xxxxx Create the BD object with the None metric xxxxxxxxxxxxxxxxx
        self.bd_obj_None = EnhancedBD(current_params['num_cells'],
                                      transmit_power,
                                      self.noise_var,
                                      self.pe)
        self.bd_obj_None.set_ext_int_handling_metric(
            "None",
            {'modulator': self.modulator,
             'packet_length': current_params['packet_length'],
             'num_streams': 1})
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Create the BD object with the Naive metric xxxxxxxxxxxxxxxx
        self.bd_obj_naive = EnhancedBD(current_params['num_cells'],
                                       transmit_power,
                                       self.noise_var,
                                       self.pe)
        self.bd_obj_naive.set_ext_int_handling_metric(
            "naive",
            {'modulator': self.modulator,
             'packet_length': current_params['packet_length'],
             'num_streams': 1})
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Create the BD object with the fixed metric xxxxxxxxxxxxxxxx
        self.bd_obj_fixed = EnhancedBD(current_params['num_cells'],
                                       transmit_power,
                                       self.noise_var,
                                       self.pe)
        self.bd_obj_fixed.set_ext_int_handling_metric(
            "fixed",
            {'modulator': self.modulator,
             'packet_length': current_params['packet_length'],
             'num_streams': 1})
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Create the BD object with the capacity metric xxxxxxxxxxxxx
        self.bd_obj_capacity = EnhancedBD(current_params['num_cells'],
                                          transmit_power,
                                          self.noise_var,
                                          self.pe)
        self.bd_obj_capacity.set_ext_int_handling_metric(
            "capacity",
            {'modulator': self.modulator,
             'packet_length': current_params['packet_length'],
             'num_streams': 1})
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xx Create the BD object with the effective_throughput metric xxxx
        self.bd_obj_effec_throughput = EnhancedBD(
            current_params['num_cells'],
            transmit_power,
            self.noise_var,
            self.pe)
        self.bd_obj_effec_throughput.set_ext_int_handling_metric(
            "effective_throughput",
            {'modulator': self.modulator,
             'packet_length': current_params['packet_length'],
             'num_streams': 1})
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Create the BD object with whitening xxxxxxxxxxxxxxxxxxxxxxx
        self.bd_obj_whitening = WhiteningBD(current_params['num_cells'],
                                      transmit_power,
                                      self.noise_var,
                                      self.pe)
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
        self._create_users_according_to_scenario(current_parameters)

        # This will calculate pathloss and generate random channels from
        # all transmitters to all receivers as well as from the external
        # interference sources to all receivers. This method must be called
        # after the _create_users_according_to_scenario method so that the
        # users are already created (we need their positions for the
        # pathloss)
        self._create_users_channels(current_parameters)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Perform the block diagonalization for each metric xxxxxxxxx
        # None Metric
        (MsPk_all_users_None,
         Wk_all_users_None,
         Ns_all_users_None) = self.bd_obj_None.block_diagonalize_no_waterfilling(
             self.multiuser_channel)

        # Naive Metric
        (MsPk_all_users_naive,
         Wk_all_users_naive,
         Ns_all_users_naive) = self.bd_obj_naive.block_diagonalize_no_waterfilling(
             self.multiuser_channel)

        # Fixed Metric
        (MsPk_all_users_fixed,
         Wk_all_users_fixed,
         Ns_all_users_fixed) = self.bd_obj_fixed.block_diagonalize_no_waterfilling(
             self.multiuser_channel)

        # Capacity Metric
        (MsPk_all_users_capacity,
         Wk_all_users_capacity,
         Ns_all_users_capacity) = self.bd_obj_capacity.block_diagonalize_no_waterfilling(
             self.multiuser_channel)

        # effective_throughput Metric
        (MsPk_all_users_effec_throughput,
         Wk_all_users_effec_throughput,
         Ns_all_users_effec_throughput) = self.bd_obj_effec_throughput.block_diagonalize_no_waterfilling(
             self.multiuser_channel)

        (Ms_all_users_Whitening,
         Wk_all_users_Whitening,
         Ns_all_users_Whitening) = self.bd_obj_whitening.block_diagonalize_no_waterfilling(self.multiuser_channel)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Since we will use the same data for the external interference no
        # matter which metric is used, lets create that data here.
        external_int_data_all_metrics = (
            np.sqrt(self.pe) *
            misc.randn_c_RS(
                self.ext_data_RS,
                current_parameters['ext_int_rank'],
                current_parameters['NSymbs']))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxx Run the Simulation and get the results for each metric xxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # None metric
        (ber_result_None,
         ser_result_None,
         per_result_None,
         spec_effic_result_None) = self.__simulate_for_one_metric(
             Ns_all_users_None,
             external_int_data_all_metrics,
             MsPk_all_users_None,
             Wk_all_users_None,
             'None',
             current_parameters)

        # naive metric
        (ber_result_naive,
         ser_result_naive,
         per_result_naive,
         spec_effic_result_naive) = self.__simulate_for_one_metric(
             Ns_all_users_naive,
             external_int_data_all_metrics,
             MsPk_all_users_naive,
             Wk_all_users_naive,
             'naive',
             current_parameters)

        # fixed metric
        (ber_result_fixed,
         ser_result_fixed,
         per_result_fixed,
         spec_effic_result_fixed) = self.__simulate_for_one_metric(
             Ns_all_users_fixed,
             external_int_data_all_metrics,
             MsPk_all_users_fixed,
             Wk_all_users_fixed,
             'fixed',
             current_parameters)

        # capacity metric
        (ber_result_capacity,
         ser_result_capacity,
         per_result_capacity,
         spec_effic_result_capacity) = self.__simulate_for_one_metric(
             Ns_all_users_capacity,
             external_int_data_all_metrics,
             MsPk_all_users_capacity,
             Wk_all_users_capacity,
             'capacity',
             current_parameters)

        # effective throughput metric
        (ber_result_effec_throughput,
         ser_result_effec_throughput,
         per_result_effec_throughput,
         spec_effic_result_effec_throughput) = self.__simulate_for_one_metric(
             Ns_all_users_effec_throughput,
             external_int_data_all_metrics,
             MsPk_all_users_effec_throughput,
             Wk_all_users_effec_throughput,
             'effec_throughput',
             current_parameters)

        # Whitening BD
        (ber_result_Whitening,
         ser_result_Whitening,
         per_result_Whitening,
         spec_effic_result_Whitening) = self.__simulate_for_one_metric(
             Ns_all_users_Whitening,
             external_int_data_all_metrics,
             Ms_all_users_Whitening,
             Wk_all_users_Whitening,
             'Whitening',
             current_parameters)

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

        # Add the 'Whitening' results
        simResults.add_result(ber_result_Whitening)
        simResults.add_result(ser_result_Whitening)
        simResults.add_result(per_result_Whitening)
        simResults.add_result(spec_effic_result_Whitening)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return simResults

    def __simulate_for_one_metric(self,
                                  Ns_all_users,
                                  external_int_data_all_metrics,
                                  MsPk_all_users,
                                  Wk_all_users,
                                  metric_name,
                                  current_parameters):
        """
        This method is only called inside the _run_simulation method.

        This method has the common code that is execute for each metric
        inside the _run_simulation method.

        Parameters
        ----------
        Ns_all_users : 1D numpy array of size K.
            Number of streams for each user. This variable controls how
            many data streams will be generated for each user of the K
            users.
        external_int_data_all_metrics : 2D numpy array
            The data of the external interference sources.
        MsPk_all_users : 1D numpy array of 2D numpy arrays
            The precoders of all users returned by the block diagonalize
            method for the given metric.
        Wk_all_users : 1D numpy array of 2D numpy arrays
            The receive filter for all users.
        metric_name : string
            Metric name. This string will be appended to each result name.
        """
        Ns_total = np.sum(Ns_all_users)
        self.data_RS = np.random.RandomState(self.data_gen_seed)
        input_data = self.data_RS.randint(
            0,
            current_parameters['M'],
            [Ns_total, current_parameters['NSymbs']])
        symbols = self.modulator.modulate(input_data)

        # Prepare the transmit data. That is, the precoded_data as well as
        # the external interferece sources' data.
        precoded_data = np.dot(np.hstack(MsPk_all_users),
                                        symbols)
        # external_int_data_all_metrics = np.sqrt(self.pe) * misc.randn_c_RS(self.ext_data_RS, self.ext_int_rank, self.NSymbs)
        all_data = np.vstack([precoded_data,
                              external_int_data_all_metrics])

        #xxxxxxxxxx Pass the precoded data through the channel xxxxxxxxxxxx
        self.multiuser_channel.set_noise_seed(self.noise_seed)
        received_signal = self.multiuser_channel.corrupt_concatenated_data(
            all_data,
            self.noise_var
        )

        # xxxxxxxxxx Filter the received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Wk = sp_linalg.block_diag(*Wk_all_users)
        received_symbols = np.dot(Wk, received_signal)

        # xxxxxxxxxx Demodulate the filtered symbols xxxxxxxxxxxxxxxxxxxxxx
        decoded_symbols = self.modulator.demodulate(received_symbols)

        # xxxxxxxxxx Calculates the Symbol Error Rate xxxxxxxxxxxxxxxxxxxxx
        num_symbol_errors = np.sum(decoded_symbols != input_data, 1)
        # num_symbol_errors = sum_user_data(num_symbol_errors,
        #                                            Ns_all_users)
        num_symbols = np.ones(Ns_total) * input_data.shape[1]

        # xxxxxxxxxx Calculates the Bit Error Rate xxxxxxxxxxxxxxxxxxxxxxxx
        num_bit_errors = misc.count_bit_errors(decoded_symbols, input_data, 1)
        # num_bit_errors = sum_user_data(num_bit_errors,
        #                                         Ns_all_users)

        num_bits = num_symbols * np.log2(current_parameters['M'])

        # xxxxxxxxxx Calculates the Package Error Rate xxxxxxxxxxxxxxxxxxxx
        ber = num_bit_errors / num_bits
        per = 1. - ((1. - ber) ** current_parameters['packet_length'])
        num_packages = num_bits / current_parameters['packet_length']
        num_package_errors = per * num_packages

        # xxxxxxxxxx Calculates the Spectral Efficiency xxxxxxxxxxxxxxxxxxx
        # nominal spectral Efficiency per stream
        nominal_spec_effic = self.modulator.K
        effective_spec_effic = (1 - per) * nominal_spec_effic

        # xxxxx Map the per stream metric to a global metric xxxxxxxxxxxxxx
        num_bit_errors = np.sum(num_bit_errors)
        num_bits = np.sum(num_bits)
        num_symbol_errors = np.sum(num_symbol_errors)
        num_symbols = np.sum(num_symbols)
        num_package_errors = np.sum(num_package_errors)
        num_packages = np.sum(num_packages)
        effective_spec_effic = np.sum(effective_spec_effic)

        # None metric
        ber_result = simulations.Result.create(
            'ber_{0}'.format(metric_name),
            simulations.Result.RATIOTYPE,
            num_bit_errors,
            num_bits)
        ser_result = simulations.Result.create(
            'ser_{0}'.format(metric_name),
            simulations.Result.RATIOTYPE,
            num_symbol_errors,
            num_symbols)

        per_result = simulations.Result.create(
            'per_{0}'.format(metric_name),
            simulations.Result.RATIOTYPE,
            num_package_errors,
            num_packages)

        spec_effic_result = simulations.Result.create(
            'spec_effic_{0}'.format(metric_name),
            simulations.Result.RATIOTYPE,
            effective_spec_effic,
            1)

        return (ber_result,
                ser_result,
                per_result,
                spec_effic_result)
        # return (num_packages,
        #         effective_spec_effic,
        #         num_symbols,
        #         num_symbol_errors,
        #         num_bit_errors,
        #         num_bits,
        #         num_package_errors)

    # def _keep_going(self, current_sim_results, current_rep):
    #     ber_result = current_sim_results['ber'][-1]
    #     num_bit_errors = ber_result._value
    #     return num_bit_errors < self.max_bit_errors

    @staticmethod
    def _calc_transmit_power(SNR_dB, N0_dBm, cell_radius, path_loss_obj):
        """
        Calculates the required transmit power (in linear scale) to
        achieve the desired mean SNR value at the cell border.

        This method calculates the path loss at the cell border and
        then finds the transmit power that gives the desired mean SNR
        at the cell border.

        Parameters
        ----------
        SNR_dB : float
            SNR value (in dB)
        N0_dBm : float
            Noise power (in dBm)
        cell_radius : float
            Cell radius (in Km)
        path_loss_obj : Path Loss object
            Object of a pathloss class used to calculate the path loss (see
            `comm.pathloss`.

        Returns
        -------
        transmit_power : float
            Desired transmit power (in linear scale).
        """
        # Path loss (in linear scale) from the cell center to
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


def plot_spectral_efficience_all_metrics(results, Pe_dBm, ax=None):
    from matplotlib import pyplot as plt

    params = results.params
    SNR = params['SNR']
    spec_effic_None = np.array(results.get_result_values_list('spec_effic_None'))
    spec_effic_naive = np.array(results.get_result_values_list('spec_effic_naive'))
    spec_effic_fixed = np.array(results.get_result_values_list('spec_effic_fixed'))
    spec_effic_capacity = np.array(results.get_result_values_list('spec_effic_capacity'))
    spec_effic_effective_throughput = np.array(results.get_result_values_list('spec_effic_effec_throughput'))
    spec_effic_Whitening = np.array(results.get_result_values_list('spec_effic_Whitening'))

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        fig = ax.get_figure()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

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

    # xxxxx Plot the Spec. Effic. with whitening BD xxxxxxxxxxxxxxxxxxxxxxx
    ax.plot(SNR, spec_effic_Whitening[result_indexes],
            'c-D', label='Whitening BD')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Spectral Efficiency (bits/channel use)')
    ax.set_title('Spectral Efficiency for Pe: {0}dBm'.format(Pe_dBm))
    ax.legend(loc=0)

    ax.grid(True, which='both', axis='both')

    return fig


def plot_per_all_metrics(results, Pe_dBm, ax=None):
    from matplotlib import pyplot as plt

    params = results.params
    SNR = params['SNR']
    per_None = np.array(results.get_result_values_list('per_None'))
    per_naive = np.array(results.get_result_values_list('per_naive'))
    per_fixed = np.array(results.get_result_values_list('per_fixed'))
    per_capacity = np.array(results.get_result_values_list('per_capacity'))
    per_effective_throughput = np.array(results.get_result_values_list('per_effec_throughput'))
    per_effective_whitening = np.array(results.get_result_values_list('per_Whitening'))

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        fig = ax.get_figure()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

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

    # xxxxx Plot the Spec. Effic. with whitening BD xxxxxxxxxxxxxxxxxxxxxxx
    ax.plot(SNR, per_effective_whitening[result_indexes],
            'c-D', label='Whitening BD')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Packet Error Rate')
    ax.set_title('Packet Error Rate for Pe: {0}dBm'.format(Pe_dBm))
    ax.legend(loc=0)

    ax.grid(True, which='both', axis='both')

    return fig


## xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    from apps.simulate_comp import BDSimulationRunner

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # File name (without extension) for the figure and result files.
    results_filename = 'bd_results_APAGAR'
    runner = BDSimulationRunner()
    runner.set_results_filename(results_filename)
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
        runner.simulate_in_parallel(lview)
    else:
        print("Simulation will be run serially")
        runner.simulate()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #
    #
    print "Runned iterations: {0}".format(runner.runned_reps)
    print "Elapsed Time: {0}".format(runner.elapsed_time)
    #
    #
    #


if __name__ == '__main__':
    try:
        from matplotlib import pyplot as plt
        _MATPLOTLIB_AVAILABLE = True
    except ImportError:
        _MATPLOTLIB_AVAILABLE = False

    ## xxxxxxxx Load the results from the file xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results_filename = 'bd_results_APAGAR'
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
        spec_fig.savefig('{0}_Pe_{1}_spec_effic.pdf'.format(results_filename, Pe_dBm))

        per_all_fig = plot_per_all_metrics(results, Pe_dBm)
        # per_all_fig.tight_layout()
        per_all_fig.subplots_adjust(bottom=0.08,right=0.98,top=0.95,left=0.07)
        per_all_fig.savefig('{0}_Pe_{1}_per_all.pdf'.format(results_filename, Pe_dBm))


        # # Save the Error rates curves for each metric for the given Pe_dBm
        # error_None_fig.savefig('{0}_Pe_{1}_error_rates_None.pgf'.format(results_filename, Pe_dBm))
        # error_naive_fig.savefig('{0}_Pe_{1}_error_rates_naive.pgf'.format(results_filename, Pe_dBm))
        # error_fixed_fig.savefig('{0}_Pe_{1}_error_rates_fixed.pgf'.format(results_filename, Pe_dBm))
        # error_capacity_fig.savefig('{0}_Pe_{1}_error_rates_capacity.pgf'.format(results_filename, Pe_dBm))
        # error_effec_throughput_fig.savefig('{0}_Pe_{1}_error_rates_spec_eff.pgf'.format(results_filename, Pe_dBm))

        plt.show()
