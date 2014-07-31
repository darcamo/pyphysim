#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to simpulate a CoMP transmission with the possible stream
reduction
"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
try:
    parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    sys.path.append(parent_dir)
except NameError:
    sys.path.append('../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import numpy as np
from scipy import linalg as sp_linalg
from time import time

from pyphysim.util import conversion, misc
from pyphysim.simulations import progressbar
from pyphysim.cell import cell
from pyphysim.comm import blockdiagonalization
from pyphysim.comm import pathloss, channels, modulators


tic = time()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Simulation Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx Cell and Grid Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
cell_radius = 1.0  # Cell radius (in Km)
num_cells = 3
num_clusters = 1
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx Channel Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Nr = np.ones(num_cells, dtype=int) * 2  # Number of receive antennas
Nt = np.ones(num_cells, dtype=int) * 2  # Number of transmit antennas
#Ns_BD = Nt  # Number of streams (per user) in the BD algorithm
path_loss_obj = pathloss.PathLoss3GPP1()
multiuser_channel = channels.MultiUserChannelMatrixExtInt()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx Modulation Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
M = 4
modulator = modulators.PSK(M)
packet_length = 60
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx Transmission Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
NSymbs = 500  # Number of symbols (per stream per user simulated at each
              # iteration
SNR_dB = 15.
N0_dBm = -116.4  # Noise power (in dBm)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx External Interference Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Pe_dBm = 10  # transmit power (in dBm) of the ext. interference
ext_int_rank = 1  # Rank of the external interference
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx General Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
rep_max = 10000   # Maximum number of repetitions for each

pbar = progressbar.ProgressbarText(rep_max, message="Simulating for SNR: {0}, Pe_dBm: {1}".format(SNR_dB, Pe_dBm))
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx Dependent parameters (don't change these) xxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx Path loss (in linear scale) from the cell center to xxxxxxxxxx
path_loss_border = path_loss_obj.calc_path_loss(cell_radius)
noise_var = conversion.dBm2Linear(N0_dBm)
snr = conversion.dB2Linear(SNR_dB)
transmit_power = snr * noise_var / path_loss_border
# External interference power
pe = conversion.dBm2Linear(Pe_dBm)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx Cell Grid xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
cell_grid = cell.Grid()
cell_grid.create_clusters(num_clusters, num_cells, cell_radius)
cluster0 = cell_grid._clusters[0]
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxx Create the scenario xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
cell_ids = np.arange(1, num_cells + 1)
angles = np.array([210, -30, 90])
cluster0.delete_all_users()
cluster0.add_border_users(cell_ids, angles, 0.7)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Simulation loop xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
dists = cluster0.calc_dist_all_users_to_each_cell()
pathloss = path_loss_obj.calc_path_loss(dists)
distance_users_to_cluster_center = np.array(
    [cluster0.calc_dist(i) for i in cluster0.get_all_users()])
pathlossInt = path_loss_obj.calc_path_loss(
    cluster0.external_radius - distance_users_to_cluster_center)
pathlossInt.shape = (num_cells, 1)

num_symbol_errors = 0.
num_symbols = 0.
num_bit_errors = 0.
num_bits = 0.
for rep in range(rep_max):
    # Randomize users channels
    multiuser_channel.randomize(Nr, Nt, num_cells, ext_int_rank)
    multiuser_channel.set_pathloss(pathloss, pathlossInt)

    # Create the comp_obj
    comp_obj = blockdiagonalization.EnhancedBD(num_cells, transmit_power, noise_var, pe)
    #comp_obj.set_ext_int_handling_metric('capacity')
    comp_obj.set_ext_int_handling_metric('effective_throughput',
                                         {'modulator': modulator,
                                          'packet_length': packet_length})

    (MsPk_all_users,
     Wk_all_users,
     Ns_all_users) = comp_obj.block_diagonalize_no_waterfilling(multiuser_channel)

    # xxxxx Performs the actual transmission for each user xxxxxxxxxxxxxxxx
    # Generate input data and modulate it
    input_data = np.random.randint(0, M, [np.sum(Ns_all_users), NSymbs])
    symbols = modulator.modulate(input_data)

    # Prepare the transmit data.
    precoded_data = np.dot(np.hstack(MsPk_all_users), symbols)
    external_int_data = np.sqrt(pe) * misc.randn_c(ext_int_rank, NSymbs)
    all_data = np.vstack([precoded_data, external_int_data])

    # Pass the precoded data through the channel
    received_signal = multiuser_channel.corrupt_concatenated_data(
        all_data,
        noise_var
    )

    # Filter the received data
    Wk = sp_linalg.block_diag(*Wk_all_users)
    received_symbols = np.dot(Wk, received_signal)

    # Demodulate the filtered symbols
    decoded_symbols = modulator.demodulate(received_symbols)

    # Calculates the number of symbol errors
    num_symbol_errors += np.sum(decoded_symbols != input_data)
    num_symbols += input_data.size

    # Calculates the number of bit errors
    num_bit_errors += misc.count_bit_errors(input_data, decoded_symbols)
    num_bits += input_data.size * modulators.level2bits(M)

    pbar.progress(rep + 1)

# Calculate the Symbol Error Rate
print
print "num_symbol_errors: {0}".format(num_symbol_errors)
print "num_symbols: {0}".format(num_symbols)
SER = float(num_symbol_errors) / float(num_symbols)
BER = float(num_bit_errors) / float(num_bits)
PER = 1 - ((1 - BER) ** packet_length)
# Spectral efficiency
SE = modulator.K * (1 - PER)

print "SER: {0}".format(SER)
print "BER: {0}".format(BER)
print "PER: {0}".format(PER)
print "SpectralEfficiency: {0}".format(SE)

# xxxxxxxxxx Finished xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
toc = time()
print misc.pretty_time(toc - tic)

# xxxxx Resultados xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxx Pe_dBm = 35, packet_length = 60 -> Sem stream reduction xxxxxxxxxxx
# num_symbol_errors: 42221279.0
# num_symbols: 60000000.0
# SER: 0.703687983333
# BER: 0.4580317
# PER: 1.0
# SpectralEfficiency: 2.22044604925e-16
# 1m:39s

# num_symbol_errors: 42229299.0
# num_symbols: 60000000.0
# SER: 0.70382165
# BER: 0.458099175
# PER: 1.0
# SpectralEfficiency: 2.22044604925e-16
# 1m:41s
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxx Pe_dBm de 35, packet_length = 60 -> Metrica da capacidade xxxxxxxxx
# num_symbol_errors: 314956.0
# num_symbols: 30101000.0
# SER: 0.0104633068669
# BER: 0.00592712866682
# PER: 0.300007619319
# SpectralEfficiency: 1.39998476136
# 1m:54s

# num_symbol_errors: 321488.0
# num_symbols: 30107500.0
# SER: 0.0106780038196
# BER: 0.00607136095657
# PER: 0.3060754185
# SpectralEfficiency: 1.387849163
# 1m:55s
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxx Pe_dBm de 35, packet_length = 60 -> Metrica de effec. xxxxxxxxxxxxx
# num_symbol_errors: 319667.0
# num_symbols: 30105500.0
# SER: 0.0106182259056
# BER: 0.00602494560795
# PER: 0.304128407729
# SpectralEfficiency: 1.39174318454
# 2m:1s

# num_symbol_errors: 311942.0
# num_symbols: 30102500.0
# SER: 0.0103626609086
# BER: 0.00586898098165
# PER: 0.297546637618
# SpectralEfficiency: 1.40490672476
# 2m:2s
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
