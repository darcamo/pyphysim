#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""module docstring"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import numpy as np
from time import time

from util import conversion, misc, progressbar
from cell import cell
from comp import comp
from comm import pathloss, channels, modulators

tic = time()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Simulation Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Cell and Grid Parameters
cell_radius = 1.0  # Cell radius (in Km)
num_cells = 3
num_clusters = 1

# Channel Parameters
Nr = np.ones(num_cells) * 2  # Number of receive antennas
Nt = np.ones(num_cells) * 2  # Number of transmit antennas
Ns_BD = Nt  # Number of streams (per user) in the BD algorithm
path_loss_obj = pathloss.PathLoss3GPP1()
multiuser_channel = channels.MultiUserChannelMatrixExtInt()

# Modulation Parameters
M = 4
modulator = modulators.PSK(M)

# Transmission Parameters
NSymbs = 500  # Number of symbols (per stream per user simulated at each
              # iteration
SNR_dB = 15.
N0_dBm = -116.4  # Noise power (in dBm)

# External Interference Parameters
Pe_dBm = -10000  # transmit power (in dBm) of the ext. interference
ext_int_rank = 1  # Rank of the external interference

# xxxxxxxxxx General Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
rep_max = 10000   # Maximum number of repetitions for each

pbar = progressbar.ProgressbarText(rep_max, message="Simulating for SNR: {0}".format(SNR_dB))

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx Dependent parameters (don't change these) xxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Path loss (in linear scale) from the cell center to
path_loss_border = path_loss_obj.calc_path_loss(cell_radius)
noise_var = conversion.dBm2Linear(N0_dBm)
snr = conversion.dB2Linear(SNR_dB)
transmit_power = snr * noise_var / path_loss_border
# External interference power
pe = conversion.dBm2Linear(Pe_dBm)

# Cell Grid
cell_grid = cell.Grid()
cell_grid.create_clusters(num_clusters, num_cells, cell_radius)
cluster0 = cell_grid._clusters[0]
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx Create the scenario xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
cell_ids = np.arange(1, num_cells + 1)
angles = np.array([210, -30, 90])
cluster0.remove_all_users()
cluster0.add_border_users(cell_ids, angles, 0.7)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Simulation loop xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
dists = cluster0.calc_dist_all_cells_to_all_users()
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

    # Generate input data and modulate it
    input_data = np.random.randint(0, M, [np.sum(Ns_BD), NSymbs])
    symbols = modulator.modulate(input_data)

    # Perform the Block Diagonalization of the channel
    H = multiuser_channel.big_H[:, 0:np.sum(Nt)]
    H_extInt = multiuser_channel.big_H[:, np.sum(Nt):]
    # Calculate the covariance matrix of the external interference plus noise
    R_extint_plus_noise = pe * np.dot(H_extInt, H_extInt.transpose().conjugate()) + np.eye(np.sum(Nr)) * noise_var

    (newH, Ms) = comp.perform_comp_with_ext_int_no_waterfilling(
        # We only add the first np.sum(Nt) columns of big_H
        # because the remaining columns come from the external
        # interference sources, which don't participate in the Block
        # Diagonalization Process.
        multiuser_channel.big_H[:, 0:np.sum(Nt)],
        num_cells,
        transmit_power,
        #noise_var
        1e-50,
        R_extint_plus_noise
    )

    # Prepare the transmit data.
    precoded_data = np.dot(Ms, symbols)
    external_int_data = np.sqrt(pe) * misc.randn_c(ext_int_rank, NSymbs)
    all_data = np.vstack([precoded_data, external_int_data])

    # Pass the precoded data through the channel
    received_signal = multiuser_channel.corrupt_concatenated_data(
        all_data,
        noise_var
    )

    # Filter the received data
    receive_filter = np.linalg.pinv(newH)
    received_symbols = np.dot(receive_filter, received_signal)

    # Demodulate the filtered symbols
    decoded_symbols = modulator.demodulate(received_symbols)

    # Calculates the number of symbol errors
    num_symbol_errors += np.sum(decoded_symbols != input_data)
    num_symbols += input_data.size

    # Calculates the number of bit errors
    aux = misc.xor(input_data, decoded_symbols)
    # Count the number of bits in aux
    num_bit_errors += np.sum(misc.bitCount(aux))
    num_bits += input_data.size * modulators.level2bits(M)

    # symbolErrors = sum(inputData != demodulatedData)
    # aux = misc.xor(inputData, demodulatedData)
    # # Count the number of bits in aux
    # bitErrors = sum(misc.bitCount(aux))
    # numSymbols = inputData.size
    # numBits = inputData.size * modulators.level2bits(M)

    pbar.progress(rep + 1)

# Calculate the Symbol Error Rate
print
print num_symbol_errors
print num_symbols
print "SER: {0}".format(float(num_symbol_errors) / float(num_symbols))
print "BER: {0}".format(float(num_bit_errors) / float(num_bits))

# xxxxxxxxxx Finished xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
toc = time()
print misc.pretty_time(toc - tic)

# Resultados
