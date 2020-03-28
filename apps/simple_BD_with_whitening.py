#!/usr/bin/env python
"""module docstring"""

from time import time

import numpy as np

import pyphysim.channels.multiuser
from pyphysim.comm import blockdiagonalization
from pyphysim.modulators import fundamental
from pyphysim.util import conversion

tic = time()

K = 3

# Channel parameters
Nt = 2 * np.ones(K)
Nr = 2 * np.ones(K)
Ns_BD = Nt

# Modulator Parameters
M = 4
modulator = fundamental.PSK(M)

# Transmission Parameters
NSymbs = 500  # Number of symbols (per stream per user simulated at each
# iteration
SNR_dB = 15.
N0_dBm = -116.4  # Noise power (in dBm)

# External Interference Parameters
Pe_dBm = -100  # transmit power (in dBm) of the ext. interference
ext_int_rank = 1  # Rank of the external interference

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Dependent Variables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
noise_var = conversion.dBm2Linear(N0_dBm)
snr = conversion.dB2Linear(SNR_dB)
transmit_power = 1.0  # snr * noise_var
# External interference power
pe = conversion.dBm2Linear(Pe_dBm)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Generate the channel
multiuser_channel = pyphysim.channels.multiuser.MultiUserChannelMatrixExtInt()
multiuser_channel.randomize(Nr, Nt, K, ext_int_rank)

# Generate input data and modulate it
input_data = np.random.randint(0, M, [np.sum(Ns_BD), NSymbs])
symbols = modulator.modulate(input_data)

BD = blockdiagonalization.BlockDiagonalizer(K, transmit_power, noise_var)
enhancedBD = blockdiagonalization.EnhancedBD(K, transmit_power, noise_var, pe)

(newH,
 Ms) = BD.block_diagonalize_no_waterfilling(multiuser_channel.big_H_no_ext_int)
(newH_ext, Ms_ext, Ns_all_users
 ) = enhancedBD.block_diagonalize_no_waterfilling(multiuser_channel)
