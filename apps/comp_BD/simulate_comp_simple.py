#!/usr/bin/env python
"""
Simple script to simulate a CoMP transmission which consists of a
simple block diagonalization of the channel.
"""

from time import time

import numpy as np

import pyphysim.channels.multiuser
from pyphysim.cell import cell
from pyphysim.channels import pathloss
from pyphysim.comm import blockdiagonalization
from pyphysim.modulators import fundamental
from pyphysim.simulations import progressbar
from pyphysim.util import conversion, misc

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
multiuser_channel = pyphysim.channels.multiuser.MultiUserChannelMatrixExtInt()

# Modulation Parameters
M = 4
modulator = fundamental.PSK(M)

# Transmission Parameters
NSymbs = 500  # Number of symbols (/stream /user simulated at each iteration
SNR_dB = 15.
N0_dBm = -116.4  # Noise power (in dBm)

# External Interference Parameters
Pe_dBm = -10000  # transmit power (in dBm) of the ext. interference
ext_int_rank = 1  # Rank of the external interference

# xxxxxxxxxx General Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
rep_max = 20000  # Maximum number of repetitions for each

pbar = progressbar.ProgressbarText(
    rep_max, message="Simulating for SNR: {0}".format(SNR_dB))

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx Dependent parameters (don't change these) xxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Path loss (in linear scale) from the cell center to
path_loss_border = path_loss_obj.calc_path_loss(cell_radius)
noise_var = conversion.dBm2Linear(N0_dBm)
snr = conversion.dB2Linear(SNR_dB)
transmit_power = snr * noise_var / path_loss_border
":type: float"
# External interference power
pe = conversion.dBm2Linear(Pe_dBm)

# Cell Grid
cell_grid = cell.Grid()
cell_grid.create_clusters(num_clusters, num_cells, cell_radius)
# noinspection PyProtectedMember
cluster0 = cell_grid._clusters[0]
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
pathlossInt = path_loss_obj.calc_path_loss(cluster0.external_radius -
                                           distance_users_to_cluster_center)
pathlossInt.shape = (num_cells, 1)

num_symbol_errors = 0.
num_symbols = 0.
num_bit_errors = 0.
num_bits = 0.
for rep in range(rep_max):
    # Randomize users channels
    multiuser_channel.randomize(Nr, Nt, num_cells, ext_int_rank)
    multiuser_channel.set_pathloss(pathloss, pathlossInt)
    multiuser_channel.noise_var = noise_var

    # Generate input data and modulate it
    input_data = np.random.randint(0, M, [np.sum(Ns_BD), NSymbs])
    symbols = modulator.modulate(input_data)

    # Perform the Block Diagonalization of the channel
    (newH, Ms) = blockdiagonalization.block_diagonalize(
        # We only add the first np.sum(Nt) columns of big_H
        # because the remaining columns come from the external
        # interference sources, which don't participate in the Block
        # Diagonalization Process.
        multiuser_channel.big_H[:, 0:np.sum(Nt)],
        num_cells,
        transmit_power,
        # noise_var
        1e-50)

    # Prepare the transmit data.
    precoded_data = np.dot(Ms, symbols)
    external_int_data = np.sqrt(pe) * misc.randn_c(ext_int_rank, NSymbs)
    all_data = np.vstack([precoded_data, external_int_data])

    # Pass the precoded data through the channel
    received_signal = multiuser_channel.corrupt_concatenated_data(all_data)

    # Filter the received data
    receive_filter = np.linalg.pinv(newH)
    received_symbols = np.dot(receive_filter, received_signal)

    # Demodulate the filtered symbols
    decoded_symbols = modulator.demodulate(received_symbols)

    # Calculates the number of symbol errors
    num_symbol_errors += np.sum(decoded_symbols != input_data)
    num_symbols += input_data.size

    # Calculates the number of bit errors
    num_bit_errors += misc.count_bit_errors(input_data, decoded_symbols)
    num_bits += input_data.size * fundamental.level2bits(M)

    pbar.progress(rep + 1)

# Calculate the Symbol Error Rate
print()
print(num_symbol_errors)
print(num_symbols)
print("SER: {0}".format(float(num_symbol_errors) / float(num_symbols)))
print("BER: {0}".format(float(num_bit_errors) / float(num_bits)))

# xxxxxxxxxx Finished xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
toc = time()
print(misc.pretty_time(toc - tic))

# Resultados
#   272356.0
# 15000000.0
# 0.0181570666667 -> Symbol Error (SNR=15)
# 2m:25s

#  264737.0
# 15000000.0
# 0.0176491333333 -> Symbol Error (SNR=15)
# 2m:31s

# 256828.0
# 15000000.0
# 0.0171218666667
# 2m:26s

# 264940.0
# 15000000.0
# 0.0176626666667
# 2m:26s

# 276546.0
# 15000000.0
# 0.0184364
# 2m:14s

# 267012.0
# 15000000.0
# 0.0178008
# 2m:17s

# 261969.0
# 15000000.0
# 0.0174646
# 19.19s
