#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulator for the SINRs and capacity of a dense indoor scenario.

The scenario is a very simplified version of the Test Case 2 from the METIS
project. Only one floor of one building is simulated and only the indoor
access points are considered.
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

from apps.simulate_metis_scenario import *
from pyphysim.util.conversion import dB2Linear, dBm2Linear, linear2dB
from pyphysim.cell import shapes, cell
from pyphysim.comm import pathloss
from pyphysim.comm.channels import calc_thermal_noise_power_dBm
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


def simulate_for_a_given_ap_assoc(
        pl, ap_assoc, wall_losses_dB, transmitting_aps, Pt, noise_var):
    sinr_array = np.empty(ap_assoc.shape, dtype=float)
    capacity = np.empty(ap_assoc.shape, dtype=float)

    wall_losses = dB2Linear(-wall_losses_dB)

    num_users, = ap_assoc.shape

    # For each transmitting AP
    for index, ap in enumerate(transmitting_aps):
        # 'ap' is the index of the current AP in the list of all APs
        # (including the non transmitting APs), while 'index' is the index
        # or the current AP in transmitting_aps

        # Index of the users associated with the current AP
        current_ap_users_idx = np.arange(num_users)[ap_assoc == ap]

        # Mask to get the interfering APs
        mask_i_aps = np.ones(len(transmitting_aps), dtype=bool)
        mask_i_aps[index] = False

        # Desired power of these users
        desired_power = (Pt
                         * wall_losses[current_ap_users_idx, index]
                         * pl[current_ap_users_idx, index])

        undesired_power = np.sum(
            Pt
            * wall_losses[current_ap_users_idx][:, mask_i_aps]
            * pl[current_ap_users_idx][:, mask_i_aps],
            axis=-1)

        sinr_array[current_ap_users_idx] = (desired_power
                                            / (undesired_power + noise_var))

        # The capacity (actually, the spectral efficiency since we didn't
        # multiply by the bandwidth) is calculated from the SINR. However,
        # if there is more then one user associated with the current AP we
        # assume bandwidth will be equally divided among all of them.
        capacity[current_ap_users_idx] = (
            np.log2(1 + sinr_array[current_ap_users_idx])
            / len(current_ap_users_idx))

    return (linear2dB(sinr_array), capacity)


def perform_simulation_SINR_heatmap(scenario_params, power_params):
    # xxxxxxxxxx Simulation Scenario Configuration xxxxxxxxxxxxxxxxxxxxxxxx
    # The size of the side of each square room
    side_length = scenario_params['side_length']
    # How much (in dB) is lost for each wall teh signal has to pass
    single_wall_loss_dB = scenario_params['single_wall_loss_dB']

    # Square of 12 x 12 square rooms
    num_rooms_per_side = scenario_params['num_rooms_per_side']
    # Total number of rooms in the grid
    num_rooms = num_rooms_per_side ** 2

    # 1 means 1 ap every room. 2 means 1 ap every 2 rooms and so on. Valid
    # values are: 1, 2, 4 and 9.
    ap_decimation = scenario_params['ap_decimation']
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Simulation Power Configuration xxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Transmit power of each access point
    Pt_dBm = power_params['Pt_dBm']
    noise_power_dBm = power_params['noise_power_dBm']

    Pt = dBm2Linear(Pt_dBm)  # 20 dBm transmit power
    noise_var = dBm2Linear(noise_power_dBm)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate the positions of all rooms xxxxxxxxxxxxxxxxxxxxx
    room_positions = calc_room_positions_square(side_length, num_rooms)
    room_positions.shape = (num_rooms_per_side, num_rooms_per_side)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Create the path loss object xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # pl_3gpp_obj = pathloss.PathLoss3GPP1()
    # pl_free_space_obj = pathloss.PathLossFreeSpace()
    # pl_3gpp_obj.handle_small_distances_bool = True
    # pl_free_space_obj.handle_small_distances_bool = True
    pl_metis_ps7_obj = pathloss.PathLossMetisPS7()
    pl_metis_ps7_obj.handle_small_distances_bool = True
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Add users in random positions in the 2D grid xxxxxxxxxxxxx
    num_users = 30  # We will create this many users in the 2D grid
    users_positions = (
        num_rooms_per_side * side_length * (
            np.random.rand(num_users) + 1j * np.random.rand(num_users)
            - 0.5 - 0.5j)
    )
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx AP Allocation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # 1 AP in each room
    ap_positions = get_ap_positions(room_positions, ap_decimation)
    num_aps = ap_positions.size
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate distances: each user to each AP xxxxxxxxxxxxxxxx
    # Dimension: (num_users, num_APs)
    dists_m = np.abs(
        users_positions[:, np.newaxis]
        - ap_positions[np.newaxis, :])
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate AP association xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Determine with which AP each user is associated with.
    # Each user will associate with the CLOSEST access point.
    ap_assoc = np.argmin(dists_m, axis=-1)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Find which Access Points should stay on xxxxxxxxxxxxxxxxxx
    # Indexes of the active APs
    transmitting_aps, users_count = np.unique(ap_assoc, return_counts=True)

    # Create a mask for the active APs
    transmitting_aps_mask = np.zeros(num_aps, dtype=bool)
    transmitting_aps_mask[transmitting_aps] = True

    # Save how many users are associated with each AP
    users_per_ap = np.zeros(num_aps, dtype=int)
    users_per_ap[transmitting_aps_mask] = users_count
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate wall losses xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Number of walls from each room to each other room
    num_walls_all = calc_num_walls(side_length, room_positions,
                                   ap_positions[transmitting_aps])
    # Find in which room each user is
    users_rooms = np.argmin(
        np.abs(room_positions.reshape([-1, 1])
               - users_positions[np.newaxis, :]),
        axis=0
        )

    # Number of walls from each user and each room
    num_walls = num_walls_all.take(users_rooms, axis=0)

    wall_losses_dB = num_walls * single_wall_loss_dB
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Distance from each user to each transmitting AP xxxxxxxxxx
    dists_m2 = dists_m.take(transmitting_aps, axis=1)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate the SINRs for each path loss model xxxxxxxxxxxxx
    pl_metis_ps7 = pl_metis_ps7_obj.calc_path_loss(dists_m2,
                                                   num_walls=num_walls)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate the SINRs and capacity xxxxxxxxxxxxxxxxxxxxxxxxx
    sinr_array_pl_metis_ps7_dB, capacity_metis_ps7 \
        = simulate_for_a_given_ap_assoc(
            pl_metis_ps7, ap_assoc, wall_losses_dB,
            transmitting_aps, Pt, noise_var)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    print(sinr_array_pl_metis_ps7_dB)
    print(capacity_metis_ps7)

    print(("\nMin/Mean/Max SINR value (METIS PS7):"
           "\n    {0}\n    {1}\n    {2}").format(
               sinr_array_pl_metis_ps7_dB.min(),
               sinr_array_pl_metis_ps7_dB.mean(),
               sinr_array_pl_metis_ps7_dB.max()))

    print(("\nMin/Mean/Max Capacity value (METIS PS7):"
           "\n    {0}\n    {1}\n    {2}").format(
               capacity_metis_ps7.min(),
               capacity_metis_ps7.mean(),
               capacity_metis_ps7.max()))

    # xxxxxxxxxx Plot all rooms and users xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    all_rooms = [shapes.Rectangle(pos - side_length/2. - side_length*1j/2.,
                                  pos + side_length/2. + side_length*1j/2.)
                 for pos in room_positions.flatten()]
    all_aps = np.array([cell.AccessPoint(pos)
                        for pos in ap_positions])
    all_users = np.array([cell.Node(r)
                          for r in users_positions])

    ax = plot_all_rooms(all_rooms)
    ax.hold(True)

    # Show the AccessPoints
    for ap in all_aps[transmitting_aps_mask]:
        ap.plot(ax)

    for ap in all_aps[np.logical_not(transmitting_aps_mask)]:
        ap.marker_color = 'gray'
        ap.plot(ax)

    # Show the users
    for u in all_users:
        u.plot_node(ax)

    plt.draw()
    plt.show()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    return sinr_array_pl_metis_ps7_dB, capacity_metis_ps7


if __name__ == '__main__':
    scenario_params = {
        'side_length': 10.,  # 10 meters side length
        'single_wall_loss_dB': 5.,
        'num_rooms_per_side': 12,
        'ap_decimation': 1}

    power_params = {
        'Pt_dBm': 20.,  # 20 dBm transmit power
        # Noise power for 25°C for a bandwidth of 5 MHz ->  -106.87 dBm
        'noise_power_dBm': calc_thermal_noise_power_dBm(25, 5e6)
    }

    out = perform_simulation_SINR_heatmap(scenario_params, power_params)
