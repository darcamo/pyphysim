#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from pyphysim.modulators import QPSK, OFDM
from pyphysim.modulators.ofdm import OfdmOneTapEqualizer
from pyphysim.channels.singleuser import SuChannel
from pyphysim.channels.fading_generators import JakesSampleGenerator
from pyphysim.channels.fading import COST259_TUx
"""Simulate an OFDM transmission through a Tapped Delay Line channel. """

if __name__ == '__main__':

    # xxxxxxxxxx Simulation parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    num_symbols = 10000
    # OFDM parametersConfiguration
    fft_size = 512
    num_used_subcarriers = 300
    cp_size = 52
    # Jakes sample generator parameters
    Fd = 5
    Ts = 2.0e-7
    L = 20
    # Channel Profile
    channel_profile = COST259_TUx.get_discretize_profile(Ts)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Input generation (not part of OFDM) xxxxxxxxxxxxxxxxxxxxxx
    # generate input data
    input_data = np.random.random_integers(0, 4 - 1, num_symbols)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx QPSK and OFDM Modulators xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    qpsk_obj = QPSK()
    ofdm_obj = OFDM(fft_size=fft_size,
                    cp_size=cp_size,
                    num_used_subcarriers=num_used_subcarriers)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Modulate data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    modulated_data = qpsk_obj.modulate(input_data)
    transmit_data = ofdm_obj.modulate(modulated_data)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Transmit data through the channel xxxxxxxxxxxxxxxxxxxxxxxx
    jakes = JakesSampleGenerator(Fd, Ts, L)
    channel = SuChannel(jakes, channel_profile)
    channel_memory = channel.num_taps_with_padding - 1
    received_data = channel.corrupt_data(transmit_data)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx OFDM Demodulate data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ofdm_received_data = ofdm_obj.demodulate(received_data[:-channel_memory])
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx One-Tap Equalization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # First we need to get the frequency response
    impulse_response = channel.get_last_impulse_response()
    freq_response = impulse_response.get_freq_response(fft_size)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Equalization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    equalizer = OfdmOneTapEqualizer(ofdm_obj)
    equalized_ofdm_received_data = equalizer.equalize_data(
        ofdm_received_data, impulse_response)
    equalized_ofdm_received_data = equalized_ofdm_received_data[:num_symbols]
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Plot demodulated symbols and original symbols xxxxxxxxxxxx
    plt.scatter(equalized_ofdm_received_data.real,
                equalized_ofdm_received_data.imag,
                color='r')
    plt.scatter(modulated_data.real, modulated_data.imag)
    plt.show()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx QPSK Demodulate data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    demodulated_data = qpsk_obj.demodulate(equalized_ofdm_received_data)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # print(sum(input_data == demodulated_data[:10000]))
