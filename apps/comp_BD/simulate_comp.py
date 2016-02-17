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

try:
    pyphysim_dir = os.path.split(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))[0]
except NameError:
    pyphysim_dir = '../'
sys.path.append(pyphysim_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import numpy as np
from scipy.linalg import block_diag

from pyphysim.simulations import SimulationRunner, \
    SimulationResults, Result, SimulationParameters
from pyphysim.simulations.simulationhelpers import simulate_do_what_i_mean
from pyphysim.util.conversion import dB2Linear, dBm2Linear
from pyphysim.util import misc
from pyphysim.cell import cell
from pyphysim.modulators import fundamental
from pyphysim.channels import pathloss, multiuser
from pyphysim.comm.blockdiagonalization import EnhancedBD, WhiteningBD


class BDSimulationRunner(SimulationRunner):  # pylint: disable=R0902
    """
    Simulation runner for a Block Diagonalization transmission.
    """

    def __init__(self, read_command_line_args=True, save_parsed_file=False):
        default_config_file = 'bd_config_file.txt'

        # xxxxxxxxxx Simulation Parameters Specification xxxxxxxxxxxxxxxxxx
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
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Initialize parameters configuration xxxxxxxxxxxxxxxxxx
        # Among other things, this will create the self.params object with
        # the simulation parameters read from the config file.
        SimulationRunner.__init__(
            self,
            default_config_file=default_config_file,
            config_spec=spec,
            read_command_line_args=read_command_line_args,
            save_parsed_file=save_parsed_file)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Channel Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self.path_loss_obj = pathloss.PathLoss3GPP1()
        self.multiuser_channel = multiuser.MultiUserChannelMatrixExtInt()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx RandomState objects seeds xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # This is only useful to reproduce a simulation for debugging
        # purposed
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
        modulator_options = {'PSK': fundamental.PSK,
                             'QPSK': fundamental.QPSK,
                             'QAM': fundamental.QAM,
                             'BPSK': fundamental.BPSK}
        self.modulator = modulator_options[self.params['modulator']](M)
        ":type: fundamental.PSK | fundamental.QPSK | fundamental.QAM | fundamental.BPSK"

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx General Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Maximum number of repetitions for each unpacked parameters set
        # self.params self.results
        self.rep_max = self.params['rep_max']

        # # max_bit_errors is used in the _keep_going method to stop the
        # # simulation earlier if possible. We stop the simulation if the
        # # accumulated number of bit errors becomes greater then 5% of the
        # # total number of simulated bits
        # self.max_bit_errors = self.rep_max * NSymbs * 5. / 100.

        self.progressbar_message = "SNR: {SNR}, Pe_dBm: {Pe_dBm}"

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Dependent parameters (don't change these) xxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # These two will be set in the _on_simulate_current_params_start
        # method
        self.pe = 0

        # Path loss (in linear scale) from the cell center to
        # self.path_loss_border = self.path_loss_obj.calc_path_loss(
        #     self.cell_radius)

        # Cell Grid
        self.cell_grid = cell.Grid()
        self.cell_grid.create_clusters(self.params['num_clusters'],
                                       self.params['num_cells'],
                                       self.params['cell_radius'])
        self.noise_var = dBm2Linear(self.params['N0'])
        self.multiuser_channel.noise_var = self.noise_var

        # This can be either 'screen' or 'file'. If it is 'file' then the
        # progressbar will write the progress to a file with appropriated
        # filename
        self.progress_output_type = 'screen'

    def _create_users_according_to_scenario(self, current_params):
        """
        Create the users according to the
        current_params['user_positioning_method'] parameter.

        This method will call one of the "_create*_scenario" methods.
        """
        scenario = current_params['user_positioning_method']
        if scenario == 'Symmetric Far Away':
            self._create_symmetric_far_away_users_scenario(current_params)
        elif scenario == 'Random':
            self._create_random_users_scenario(current_params)
        else:
            raise RuntimeError(
                "Invalid scenario: {0}".format(scenario))

    def _create_random_users_scenario(self, current_params):
        """Run this method to set variables specific to the 'RandomUsers'
        scenario.

        The 'RandomUsers' scenario places a user at a random location in
        each cell.

        """
        cluster0 = self.cell_grid.get_cluster_from_index(0)
        cell_ids = np.arange(1, current_params['num_cells'] + 1)
        cluster0.delete_all_users()
        cluster0.add_random_users(cell_ids)

    def _create_symmetric_far_away_users_scenario(self, current_params):
        """Run this method to set variables specific to the 'FarAwayUsers'
        scenario.

        The 'FarAwayUsers70%' scenario place a user in each cell at a the
        angle further away from the cell center in a distance from the cell
        center to the cell border equivalent to 70% of the cell radius.

        """
        cluster0 = self.cell_grid.get_cluster_from_index(0)
        cell_ids = np.arange(1, current_params['num_cells'] + 1)
        angles = np.array([210, -30, 90])
        cluster0.delete_all_users()
        cluster0.add_border_users(cell_ids, angles, 0.7)

    def _create_users_channels(self, current_params):
        """Create the channels of all the users.

        The users must have already been created.
        """
        cluster0 = self.cell_grid.get_cluster_from_index(0)

        # xxxxx Distances between each transmitter and each receiver xxxxxx
        # This `dists` matrix may be indexed as dists[user, cell].
        dists = cluster0.calc_dist_all_users_to_each_cell()
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
        # pylint: disable=W0201

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
        self.pe = dBm2Linear(current_params['Pe_dBm'])

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

    def _run_simulation(self, current_parameters):  # pylint: disable=R0914
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
         Ns_all_users_None) \
            = self.bd_obj_None.block_diagonalize_no_waterfilling(
            self.multiuser_channel)

        # Naive Metric
        (MsPk_all_users_naive,
         Wk_all_users_naive,
         Ns_all_users_naive) \
            = self.bd_obj_naive.block_diagonalize_no_waterfilling(
            self.multiuser_channel)

        # Fixed Metric
        (MsPk_all_users_fixed,
         Wk_all_users_fixed,
         Ns_all_users_fixed) \
            = self.bd_obj_fixed.block_diagonalize_no_waterfilling(
            self.multiuser_channel)

        # Capacity Metric
        (MsPk_all_users_capacity,
         Wk_all_users_capacity,
         Ns_all_users_capacity) \
            = self.bd_obj_capacity.block_diagonalize_no_waterfilling(
            self.multiuser_channel)

        # effective_throughput Metric
        (MsPk_all_users_effec_throughput,
         Wk_all_users_effec_throughput,
         Ns_all_users_effec_throughput) \
            = self.bd_obj_effec_throughput.block_diagonalize_no_waterfilling(
            self.multiuser_channel)

        (Ms_all_users_Whitening,
         Wk_all_users_Whitening,
         Ns_all_users_Whitening) \
            = self.bd_obj_whitening.block_diagonalize_no_waterfilling(
                self.multiuser_channel)

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
         spec_effic_result_None,
         sinr_result_None) = self.__simulate_for_one_metric(
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
         spec_effic_result_naive,
         sinr_result_naive) = self.__simulate_for_one_metric(
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
         spec_effic_result_fixed,
         sinr_result_fixed) = self.__simulate_for_one_metric(
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
         spec_effic_result_capacity,
         sinr_result_capacity) = self.__simulate_for_one_metric(
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
         spec_effic_result_effec_throughput,
         sinr_result_effec_throughput) = self.__simulate_for_one_metric(
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
         spec_effic_result_Whitening,
         sinr_result_Whitening) = self.__simulate_for_one_metric(
            Ns_all_users_Whitening,
            external_int_data_all_metrics,
            Ms_all_users_Whitening,
            Wk_all_users_Whitening,
            'Whitening',
            current_parameters)

        simResults = SimulationResults()
        # Add the 'None' results
        simResults.add_result(ber_result_None)
        simResults.add_result(ser_result_None)
        simResults.add_result(per_result_None)
        simResults.add_result(spec_effic_result_None)
        simResults.add_result(sinr_result_None)

        # Add the naive results
        simResults.add_result(ber_result_naive)
        simResults.add_result(ser_result_naive)
        simResults.add_result(per_result_naive)
        simResults.add_result(spec_effic_result_naive)
        simResults.add_result(sinr_result_naive)

        # Add the fixed results
        simResults.add_result(ber_result_fixed)
        simResults.add_result(ser_result_fixed)
        simResults.add_result(per_result_fixed)
        simResults.add_result(spec_effic_result_fixed)
        simResults.add_result(sinr_result_fixed)

        # Add the capacity results
        simResults.add_result(ber_result_capacity)
        simResults.add_result(ser_result_capacity)
        simResults.add_result(per_result_capacity)
        simResults.add_result(spec_effic_result_capacity)
        simResults.add_result(sinr_result_capacity)

        # Add the effective throughput results
        simResults.add_result(ber_result_effec_throughput)
        simResults.add_result(ser_result_effec_throughput)
        simResults.add_result(per_result_effec_throughput)
        simResults.add_result(spec_effic_result_effec_throughput)
        simResults.add_result(sinr_result_effec_throughput)

        # Add the 'Whitening' results
        simResults.add_result(ber_result_Whitening)
        simResults.add_result(ser_result_Whitening)
        simResults.add_result(per_result_Whitening)
        simResults.add_result(spec_effic_result_Whitening)
        simResults.add_result(sinr_result_Whitening)
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
        Ns_all_users : np.ndarray
            Number of streams for each user. This variable controls how
            many data streams will be generated for each user of the K
            users. This is a 1D numpy array of size K.
        external_int_data_all_metrics : np.ndarray
            The data of the external interference sources (2D numpy array).
        MsPk_all_users : np.ndarray
            The precoders of all users returned by the block diagonalize
            method for the given metric. This is a 1D numpy array of 2D numpy
            arrays.
        Wk_all_users : np.ndarray
            The receive filter for all users (1D numpy array of 2D numpy
            arrays).
        metric_name : string
            Metric name. This string will be appended to each result name.

        Returns
        -------
        TODO: Write the rest of the docstring
        """
        # pylint: disable=R0914
        Ns_total = np.sum(Ns_all_users)
        self.data_RS = np.random.RandomState(self.data_gen_seed)
        input_data = self.data_RS.randint(
            0,
            current_parameters['M'],
            [Ns_total, current_parameters['NSymbs']])
        symbols = self.modulator.modulate(input_data)

        # Prepare the transmit data. That is, the precoded_data as well as
        # the external interference sources' data.
        precoded_data = np.dot(np.hstack(MsPk_all_users),
                               symbols)

        # external_int_data_all_metrics = (
        #     np.sqrt(self.pe)
        #     * misc.randn_c_RS(
        #         self.ext_data_RS, self.ext_int_rank, self.NSymbs))

        all_data = np.vstack([precoded_data,
                              external_int_data_all_metrics])

        # xxxxxxxxxx Pass the precoded data through the channel xxxxxxxxxxx
        self.multiuser_channel.set_noise_seed(self.noise_seed)
        received_signal = self.multiuser_channel.corrupt_concatenated_data(
            all_data
        )

        # xxxxxxxxxx Filter the received data xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # noinspection PyArgumentList
        Wk = block_diag(*Wk_all_users)
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

        # xxxxx Calculate teh SINR xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Uk_all_users = np.empty(Wk_all_users.size, dtype=np.ndarray)
        for ii in range(Wk_all_users.size):
            Uk_all_users[ii] = Wk_all_users[ii].conjugate().T
        SINR_all_k = self.multiuser_channel.calc_JP_SINR(MsPk_all_users,
                                                         Uk_all_users)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # None metric
        ber_result = Result.create(
            'ber_{0}'.format(metric_name),
            Result.RATIOTYPE,
            num_bit_errors,
            num_bits)
        ser_result = Result.create(
            'ser_{0}'.format(metric_name),
            Result.RATIOTYPE,
            num_symbol_errors,
            num_symbols)

        per_result = Result.create(
            'per_{0}'.format(metric_name),
            Result.RATIOTYPE,
            num_package_errors,
            num_packages)

        spec_effic_result = Result.create(
            'spec_effic_{0}'.format(metric_name),
            Result.RATIOTYPE,
            effective_spec_effic,
            1)

        sinr_result = Result(
            'sinr_{0}'.format(metric_name),
            Result.RATIOTYPE,
            accumulate_values=True)

        for k in range(Wk_all_users.size):
            sinr_k = SINR_all_k[k]
            for value in sinr_k:
                sinr_result.update(value, 1)

        return (ber_result,
                ser_result,
                per_result,
                spec_effic_result,
                sinr_result)

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
        snr = dB2Linear(SNR_dB)
        pt = snr * dBm2Linear(N0_dBm) / path_loss_border
        return pt
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


def plot_spectral_efficience_all_metrics(results, Pe_dBm, ax=None):
    """
    Plot the spectral efficiency from the `results` object for all metrics
    given a desirable external interference power `Pe_dBm`

    Parameters
    ----------
    results : SimulationResults
        The simulation results.
    Pe_dBm : float | int
        The external interference power.
    ax : Matplotlib axes
        The where to plot.
    """
    from matplotlib import pyplot as plt

    params = results.params
    SNR = params['SNR']
    spec_effic_None = np.array(
        results.get_result_values_list('spec_effic_None'))
    spec_effic_naive = np.array(
        results.get_result_values_list('spec_effic_naive'))
    spec_effic_fixed = np.array(
        results.get_result_values_list('spec_effic_fixed'))
    spec_effic_capacity = np.array(
        results.get_result_values_list('spec_effic_capacity'))
    spec_effic_effective_throughput = np.array(
        results.get_result_values_list('spec_effic_effec_throughput'))
    spec_effic_Whitening = np.array(
        results.get_result_values_list('spec_effic_Whitening'))

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        fig = ax.get_figure()

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
    """
    Plot the packet error rate from the `results` object for all metrics
    given a desirable external interference power `Pe_dBm`

    Parameters
    ----------
    results : SimulationResults
        The simulation results.
    Pe_dBm : float | int
        The external interference power.
    ax : Matplotlib axes
        The where to plot.
    """
    # noinspection PyShadowingNames
    from matplotlib import pyplot as plt

    params = results.params
    SNR = params['SNR']
    per_None = np.array(results.get_result_values_list('per_None'))
    per_naive = np.array(results.get_result_values_list('per_naive'))
    per_fixed = np.array(results.get_result_values_list('per_fixed'))
    per_capacity = np.array(results.get_result_values_list('per_capacity'))
    per_effective_throughput = np.array(
        results.get_result_values_list('per_effec_throughput'))
    per_effective_whitening = np.array(
        results.get_result_values_list('per_Whitening'))

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    else:
        fig = ax.get_figure()

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


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Main xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    from apps.comp_BD.simulate_comp import BDSimulationRunner

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # File name (without extension) for the figure and result files.
    results_filename = 'bd_results_{Nr}x{Nt}_ext_int_rank_{ext_int_rank}'
    runner = BDSimulationRunner()
    runner.set_results_filename(results_filename)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    simulate_do_what_i_mean(runner, pyphysim_dir)

    if runner.command_line_args.index is None:
        print ("Runned iterations: {0}".format(runner.runned_reps))
        print ("Elapsed Time: {0}".format(runner.elapsed_time))


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__1':
    try:
        # noinspection PyUnresolvedReferences
        from matplotlib import pyplot as plt
        _MATPLOTLIB_AVAILABLE = True
    except ImportError:
        _MATPLOTLIB_AVAILABLE = False

    # xxxxxxxxxx Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    params = SimulationParameters.load_from_config_file('bd_config_file.txt')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxx Load the results from the file xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    results_filename = 'bd_results_{Nr}x{Nt}_ext_int_rank_{ext_int_rank}'
    results_filename.format(**params.parameters)
    results = SimulationResults.load_from_file(
        '{0}{1}'.format(results_filename, '.pickle'))

    SNR = results.params['SNR']
    if _MATPLOTLIB_AVAILABLE is True and SNR.size > 1:
        Pe_dBm = 10.

        # Save the Spectral Efficiency curve for the given Pe_dBm
        spec_fig = plot_spectral_efficience_all_metrics(results, Pe_dBm)
        # spec_fig.tight_layout()
        spec_fig.subplots_adjust(bottom=0.08, right=0.98, top=0.95, left=0.07)
        spec_fig.savefig('{0}_Pe_{1}_spec_effic.pdf'.format(
            results_filename, Pe_dBm))

        per_all_fig = plot_per_all_metrics(results, Pe_dBm)
        # per_all_fig.tight_layout()
        per_all_fig.subplots_adjust(
            bottom=0.08, right=0.98, top=0.95, left=0.07)
        per_all_fig.savefig('{0}_Pe_{1}_per_all.pdf'.format(
            results_filename, Pe_dBm))

        plt.show()
