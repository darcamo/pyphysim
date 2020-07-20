import math

import numpy as np

from pyphysim.modulators.fundamental import BPSK, QAM, Modulator
from pyphysim.simulations import Result, SimulationResults, SimulationRunner
from pyphysim.util.conversion import dB2Linear
from pyphysim.util.misc import count_bit_errors, randn_c


class RayleighOrAwgnSimulator(SimulationRunner):
    def __init__(self, SINR_dB_values, simulate_with_rayleigh=False):
        # If simulate_with_rayleigh is false only AWGN is used
        super().__init__()

        self._simulate_with_rayleigh = simulate_with_rayleigh

        # Add the simulation parameters to the `params` attribute.
        self.params.add('EbN0_db', SINR_dB_values)
        self.params.set_unpack_parameter('EbN0_db')

        # Note that when M=2 BPSK modulation will be used, while other values will use QAM
        self.params.add("M", [2, 4, 16, 64, 256])
        self.params.set_unpack_parameter('M')

        # Number of times the `_run_simulation` method will run when `simulate` method is called.
        # We are using a value 100 times larger than before, but the simulation will not take
        # 100 times the previous elapsed time to finish thanks to the implementation of the
        # `_keep_going` method that will allow us to skip many of these iterations for low EbN0 values
        self.rep_max = 50000

        # Number of symbols generated for this realization
        self.num_symbols = 1000

        # Used in the implementation of `_keep_going` method. This is the maximum numbers of symbol
        # errors we allow before `_run_simulation` is stoped for a given configuration
        self.max_symbol_errors = 1. / 300. * self.num_symbols * self.rep_max

        # Set a nice message for the progressbar
        self.progressbar_message = f"Simulating for {self.params.get_num_unpacked_variations()} configurations"

        self.update_progress_function_style = "text1"  # "ipython"

    def _keep_going(self, current_params, current_sim_results, current_rep):
        # Note that we have added a "symbol_errors" result in `_run_simulation` to use here

        # Get the last value in the "symbol_errors" results list, which corresponds to the current configuration
        cumulated_symbol_errors \
            = current_sim_results['symbol_errors'][-1].get_result()
        return cumulated_symbol_errors < self.max_symbol_errors

    def _run_simulation(self, current_parameters):
        # Since EbN0_db is an "unpacked parameter" a single value is passed to `_run_simulation`.
        # We can get the current value as below
        sinr_dB = current_parameters['EbN0_db']
        M = current_parameters['M']

        modulator = BPSK() if M == 2 else QAM(M)

        # Find the noise power from the EbN0 value (in dB)
        EbN0_linear = dB2Linear(sinr_dB)
        snr_linear = EbN0_linear * math.log2(M)
        noise_power = 1 / snr_linear

        # Generate random transmit data and modulate it
        data = np.random.randint(0, modulator.M, size=self.num_symbols)
        modulated_data = modulator.modulate(data)

        # Noise vector
        n = math.sqrt(noise_power) * randn_c(self.num_symbols)

        if self._simulate_with_rayleigh:
            # Rayleigh channel
            h = randn_c(modulated_data.size)

            # Receive the corrupted data
            received_data = h * modulated_data + n

            # Equalization
            received_data /= h

        else:
            # Receive the corrupted data
            received_data = modulated_data + n

        # Demodulate the received data and compute the number of symbol errors
        demodulated_data = modulator.demodulate(received_data)
        symbol_errors = sum(demodulated_data != data)

        num_bit_errors = count_bit_errors(data, demodulated_data)

        # Create a SimulationResults object and save the symbol error rate.
        # Note that the symbol error rate is given by the number of symbol errors divided by the number of
        # transmited symbols. We want to combine the symbol error rate for the many calls of `_run_simulation`.
        # Thus, we choose `Result.RATIOTYPE` as the "update_type". See the documentation of the `Result` class
        # for more about it.
        simResults = SimulationResults()
        simResults.add_new_result("symbol_error_rate",
                                  Result.RATIOTYPE,
                                  value=symbol_errors,
                                  total=self.num_symbols)
        simResults.add_new_result("symbol_errors",
                                  Result.SUMTYPE,
                                  value=symbol_errors)
        simResults.add_new_result("num_bit_errors",
                                  Result.SUMTYPE,
                                  value=num_bit_errors)
        simResults.add_new_result("bit_error_rate",
                                  Result.RATIOTYPE,
                                  value=num_bit_errors,
                                  total=int(np.log2(modulator.M)) * data.size)
        return simResults
