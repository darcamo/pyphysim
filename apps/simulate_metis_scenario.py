#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module containing simulation runners for the several Interference
Alignment algorithms in the algorithms.ia module.
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

# xxxxxxxxxx Import Statements xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import numpy as np
from matplotlib import pyplot as plt
# import matplotlib as mpl

from pyphysim.util.conversion import dB2Linear, dBm2Linear, linear2dB
from pyphysim.cell import cell
from pyphysim.comm import pathloss
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


def calc_wall_losses(side_length, room_positions, single_wall_loss_dB=5):
    """
    Calculate the wall losses from each room to each other room.

    Parameters
    ----------
    side_length : float
        The side length of the square cell.
    cell_positions : 1D complex numpy array
        The positions of all cells in grid.
    single_wall_loss_dB : float
        The signal loss (in dB) when the signal passes a single wall.

    Returns
    -------
    wall_losses_dB : 2D numpy array of floats
        The wall losses (in dB) from each room to each room.
    """
    num_rooms = room_positions.size

    all_room_positions_diffs = (room_positions.reshape(num_rooms, 1)
                                - 1.0001*room_positions.reshape(1, num_rooms))

    num_rooms_steps = np.floor(
        np.abs(np.real(all_room_positions_diffs / side_length)) +
        np.abs(np.imag(all_room_positions_diffs / side_length)))

    wall_losses_dB = single_wall_loss_dB * num_rooms_steps

    return wall_losses_dB


def get_cell_users_indexes(cell_index, num_users_per_cell):
    """

    Parameters
    ----------
    cell_index : int
        Index of the desired cell.
    num_users_per_cell : int
        Number of users in each cell.
    """
    return np.arange(0, num_users_per_cell) + cell_index * num_users_per_cell


def prepare_sinr_array_for_color_plot(sinr_array,
                                      num_cells_per_side,
                                      num_discrete_positions_per_cell):
    """

    Parameters
    ----------
    sinr_array : TYPE
    num_cells_per_side : TYPE
    num_discrete_positions_per_cell : TYPE
    """
    # dummy = np.zeros(
    #     [num_cells,
    #      num_discrete_positions_per_cell * num_discrete_positions_per_cell])
    dummy = sinr_array
    # for cell_idx in range(num_cells):
    #     dummy[cell_idx] = cell_idx

    dummy2 = dummy.reshape(
        [num_cells_per_side,
         num_cells_per_side,
         num_discrete_positions_per_cell,
         num_discrete_positions_per_cell])

    dummy3 = np.swapaxes(dummy2, 1, 2).reshape(
        [num_cells_per_side * num_discrete_positions_per_cell,
         num_cells_per_side * num_discrete_positions_per_cell],
        order='C')
    return dummy3


if __name__ == '__main__':
    # xxxxxxxxxx Simulation Configuration xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    side_length = 10

    # Square of 12 x 12 square cells
    num_cells_per_side = 12
    num_cells = num_cells_per_side ** 2

    # xxxxxxxxxx Discretization of ther possible positions xxxxxxxxxxxxxxxx
    num_discrete_positions_per_cell = 15  # Number of discrete positions
    step = 1. / (num_discrete_positions_per_cell + 1)
    aux = np.linspace(
        -(1. - step), (1. - step), num_discrete_positions_per_cell)
    aux = np.meshgrid(aux, aux, indexing='ij')
    user_relative_positions = aux[0] + 1j * aux[1]
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Transmit Power and noise
    Pt = dBm2Linear(20)  # 20 dBm transmit power
    noise_var = 0.0  # dBm2Linear(-97)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Create the cluster xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    cluster = cell.Cluster(
        cell_radius=side_length, num_cells=num_cells, cell_type='square')
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate cell positions and wall losses xxxxxxxxxxxxxxxxx
    cell_positions = np.array([c.pos for c in cluster])
    wall_losses_dB = calc_wall_losses(side_length, cell_positions)
    wall_losses = dB2Linear(-wall_losses_dB)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Create the path loss object xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    pl_3gpp_obj = pathloss.PathLoss3GPP1()
    pl_free_space_obj = pathloss.PathLossFreeSpace()
    pl_3gpp_obj.handle_small_distances_bool = True
    pl_free_space_obj.handle_small_distances_bool = True
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Add one user in each position of each room xxxxxxxxxxxxxxx
    for c in cluster:
        for rel_pos in user_relative_positions.flat:
            user = cell.Node(rel_pos)
            c.add_user(user, relative_pos_bool=True)
    num_users_per_cell = user_relative_positions.size
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Output SINR vector xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # One output for each case: no pathloss, 3GPP path loss and free space
    # path loss
    sinr_array_pl_nothing = np.zeros(
        [num_cells, num_users_per_cell], dtype=float)
    sinr_array_pl_3gpp = np.zeros(
        [num_cells, num_users_per_cell], dtype=float)
    sinr_array_pl_free_space = np.zeros(
        [num_cells, num_users_per_cell], dtype=float)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Let's do the simulations xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # pbar = ProgressbarText(rep_max,
    #   message="Simulating {0} iterations".format(rep_max))

    # xxxxxxxxxx Calculate the distance and path losses xxxxxxxxxxxxxxx
    dists = cluster.calc_dist_all_users_to_each_cell()
    pl_3gpp = pl_3gpp_obj.calc_path_loss(dists)
    pl_free_space = pl_free_space_obj.calc_path_loss(dists)
    pl_nothing = np.ones([num_cells * num_users_per_cell, num_cells],
                         dtype=float)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    for cell_idx in range(num_cells):
        # Index of the users in the current cell
        users_idx = get_cell_users_indexes(cell_idx, num_users_per_cell)

        # Mask to get path loss of all transmitters ...
        mask = np.ones(num_cells, dtype=bool)
        # ... except the desired transmitter
        mask[cell_idx] = 0

        # xxxxxxxxxx Case without pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        pl = pl_nothing

        # Get the desired power of the users in the cell
        desired_power = Pt * pl[users_idx, cell_idx]

        # Calculate the sum of all interference powers
        undesired_power = np.sum(
            Pt * pl[users_idx][:, mask] * wall_losses[cell_idx, mask],
            axis=-1)

        # Calculate the SINR of the user
        sinr_users_in_current_cell = (desired_power /
                                      (undesired_power + noise_var))
        sinr_array_pl_nothing[cell_idx, :] = sinr_users_in_current_cell
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Case with 3GPP pathloss xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        pl = pl_3gpp

        # Get the desired power of the users in the cell
        desired_power = Pt * pl[users_idx, cell_idx]

        # Calculate the sum of all interference powers
        undesired_power = np.sum(
            Pt * pl[users_idx][:, mask] * wall_losses[cell_idx, mask],
            axis=-1)

        # Calculate the SINR of the user
        sinr_users_in_current_cell = (desired_power /
                                      (undesired_power + noise_var))
        sinr_array_pl_3gpp[cell_idx, :] = sinr_users_in_current_cell
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Case with Free Space path loss xxxxxxxxxxxxxxxxxxxxxxx
        pl = pl_free_space

        # Get the desired power of the users in the cell
        desired_power = Pt * pl[users_idx, cell_idx]

        # Calculate the sum of all interference powers
        undesired_power = np.sum(
            Pt * pl[users_idx][:, mask] * wall_losses[cell_idx, mask],
            axis=-1)

        # Calculate the SINR of the user
        sinr_users_in_current_cell = (desired_power /
                                      (undesired_power + noise_var))
        sinr_array_pl_free_space[cell_idx, :] = sinr_users_in_current_cell
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Convert values to dB xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    sinr_array_pl_nothing_dB = linear2dB(sinr_array_pl_nothing)
    sinr_array_pl_3gpp_dB = linear2dB(sinr_array_pl_3gpp)
    sinr_array_pl_free_space_dB = linear2dB(sinr_array_pl_free_space)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    print "Mean SINR value (no PL): {0}".format(
        sinr_array_pl_nothing_dB.mean())
    print "Mean SINR value (3GPP): {0}".format(sinr_array_pl_3gpp_dB.mean())
    print "Mean SINR value (Free Space): {0}".format(
        sinr_array_pl_free_space_dB.mean())

    # xxxxxxxxxx Prepare data to be plotted xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    sinr_array_pl_nothing_dB2 = prepare_sinr_array_for_color_plot(
        sinr_array_pl_nothing_dB,
        num_cells_per_side,
        num_discrete_positions_per_cell)
    sinr_array_pl_3gpp_dB2 = prepare_sinr_array_for_color_plot(
        sinr_array_pl_3gpp_dB,
        num_cells_per_side,
        num_discrete_positions_per_cell)
    sinr_array_pl_free_space_dB2 = prepare_sinr_array_for_color_plot(
        sinr_array_pl_free_space_dB,
        num_cells_per_side,
        num_discrete_positions_per_cell)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Plot each case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # No path loss
    ax1 = plt.matshow(sinr_array_pl_nothing_dB2)
    plt.colorbar()

    # 3GPP path loss
    ax2 = plt.matshow(sinr_array_pl_3gpp_dB2)
    plt.colorbar()

    ax3 = plt.matshow(sinr_array_pl_free_space_dB2)
    plt.colorbar()

    plt.show()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
