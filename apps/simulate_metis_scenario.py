#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulator for the SINRs of a dense indoor scenario.

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
import math
import numpy as np
from matplotlib import pyplot as plt
# import matplotlib as mpl

from pyphysim.util.conversion import dB2Linear, dBm2Linear, linear2dB
# from pyphysim.cell import shapes
from pyphysim.comm import pathloss
from pyphysim.comm.channels import calc_thermal_noise_power_dBm
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


def calc_room_positions_square(side_length, num_rooms):
    """
    Calculate the central positions of the square rooms.
    """
    sqrt_num_rooms = int(math.sqrt(num_rooms))

    if sqrt_num_rooms ** 2 != num_rooms:
        raise ValueError("num_rooms must be a perfect square number")

    int_positions = np.unravel_index(np.arange(num_rooms), (sqrt_num_rooms,
                                                            sqrt_num_rooms))

    room_positions = (side_length * (int_positions[1] + 1j *
                                     int_positions[0][::-1] - 0.5-0.5j))

    # Shift the room positions so that the origin becomes the center of all
    # rooms
    shift = side_length * (sqrt_num_rooms - 1) // 2
    room_positions = (room_positions
                      - shift - 1j * shift
                      + side_length / 2. + 1j * side_length / 2.)

    return room_positions


def plot_all_rooms(all_rooms, ax=None):
    """
    Plot all Rectangle shapes in `all_rooms` using the `ax` axis.

    Parameters
    ----------
    ax  : matplotlib axis.
        The axis where the rooms will be plotted.
    all_rooms : iterable of shape.Rectangle objects
        The rooms to be plotted.

    Returns
    -------
    ax : Matplotlib axes
        The axes used to plot.
    """
    standalone = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
        standalone = True

    for room in all_rooms:
        room.plot(ax)

    if standalone is True:
        # Do some extra stuff like setting the plot axis limits.
        # First we get all vertices of all rectangles.
        all_vertices = np.vstack([r.vertices for r in all_rooms])
        xmin = all_vertices.real.min()
        xmax = all_vertices.real.max()
        ymin = all_vertices.real.min()
        ymax = all_vertices.real.max()
        ax.set_ylim([ymin, ymax])
        ax.set_xlim([xmin, xmax])
        ax.set_xlabel("Position X coordinate")
        ax.set_ylabel("Position Y coordinate")
        ax.set_title("Plot of all Rooms")

    return ax


def calc_num_walls(side_length, room_positions, ap_positions):
    """
    Calculate the number of walls between each room to each AP.

    This is used to calculated the wall losses as well as the indoor
    pathloss.

    Parameters
    ----------
    side_length : float
        The side length of the square room.
    room_positions : 2D complex numpy array
        The positions of all rooms in the grid.
    ap_positions : 1D complex numpy array
        The positions of access points in the grid.

    Returns
    -------
    num_walls : 2D numpy array of ints
        The number of walls from each room to access point.
    """
    all_positions_diffs = (room_positions.reshape(-1, 1)
                           - 1.0001*ap_positions.reshape(1, -1))

    num_walls \
        = np.round(
            np.absolute(np.real(all_positions_diffs / side_length)) +
            np.absolute(np.imag(all_positions_diffs / side_length))
        ).astype(int)

    return num_walls


def prepare_sinr_array_for_color_plot(
        sinr_array, num_rooms_per_side, num_discrete_positions_per_room):
    """

    Parameters
    ----------
    sinr_array : TYPE
    num_rooms_per_side : TYPE
    num_discrete_positions_per_room : TYPE
    """
    out = np.swapaxes(sinr_array, 1, 2).reshape(
        [num_rooms_per_side * num_discrete_positions_per_room,
         num_rooms_per_side * num_discrete_positions_per_room],
        order='C')

    return out


def get_ap_positions(room_positions, decimation=1):
    """
    Get the array of AccessPoint positions for the desired decimation and
    room_positions.

    Each access point is placed in the center of the room where it is
    located. The value of `decimation` controls the frequency of APs. A
    value of 1 means one AP in each room. A value of 2 means one AP each 2
    rooms and so on.

    The valid decimation values are 1, 2, 4 and 9. Any other value will
    raise an exception.

    Parameters
    ----------
    room_positions : 2D numpy array with shape (n, n)
        The positions of each room.
    decimation : int
        The decimation (in number of room) of the APs.

    Returns
    -------
    ap_positions : 1D numpy array with shape (n**2)
        The position of the arrays.
    """
    mask = np.zeros(room_positions.shape, dtype=bool)

    if decimation == 1:
        mask[:, :] = True
    elif decimation == 2:
        mask[1::2, ::2] = True
        mask[::2, 1::2] = True
    elif decimation == 4:
        mask[1::2, 1::2] = True
    elif decimation == 9:
        mask[1::3, 1::3] = True
    else:
        raise ValueError('Invalid decimation value: {0}'.format(decimation))

    ap_positions = room_positions[mask]
    return ap_positions.flatten()


def simulate_for_a_given_ap_assoc(
        pl, ap_assoc, wall_losses_dB, Pt, noise_var):
    """
    Simulate and return the SINR for a given path loss and AP associations.

    This is an internal function called inside
    `perform_simulation_SINR_heatmap`

    Parameters
    ----------
    pl : 5D numpy float array
        The path loss (in LINEAR SCALE) from each discrete position in each
        room to each access point. Dimension: (n, n, d, d, a) where 'n' is
        the number of rooms per dimension, 'd' is the number of discrete
        positons in one room (per dimension) and 'a' is the number of
        access points.
    ap_assoc : 4D numpy int array
        The index of the access point that each discrete point in each room
        is associated with. Dimension: (n, n, d, d)
    wall_losses_dB : 5D numpy int array
        The wall losses (in dB) from each discrete user in each room to
        each access point. Dimension: (n, n, d, d, a)
    Pt : float
        Transmit power.
    noise_var : float
        Noise variance (power)

    Returns
    -------
    sinr_array_dB : 4D numpy array
        The SINR (in dB) of each discrete point of each room.
    """
    wall_losses = dB2Linear(-wall_losses_dB)

    # Number of APs is the last dimension in the path loss array
    num_aps = pl.shape[-1]

    # Output variable
    sinr_array = np.empty(ap_assoc.shape, dtype=float)

    for ap_idx in range(num_aps):
        # Mask of the users associated with the current access point
        mask = (ap_assoc == ap_idx)

        # # Mask of the users NOT associated with the current access point
        # mask_n = np.logical_not(mask)

        # Mask with all APs except the current one (that is, the
        # interfering APs)
        mask_i_aps = np.arange(num_aps) != ap_idx

        # Each element in desired_power is the desired power of one user
        # associated with the current access point
        desired_power = Pt * wall_losses[mask, ap_idx] * pl[mask, ap_idx]
        undesired_power = np.sum(
            Pt * wall_losses[mask][:, mask_i_aps] * pl[mask][:, mask_i_aps],
            axis=-1)

        sinr_array[mask] = (desired_power / (undesired_power + noise_var))

    return linear2dB(sinr_array)


def perform_simulation_SINR_heatmap(scenario_params,  # pylint: disable=R0914
                                    power_params):
    """
    Perform the simulation.
    """
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

    # xxxxxxxxxx Discretization of the possible positions xxxxxxxxxxxxxxxxx
    num_discrete_positions_per_room = 15  # Number of discrete positions
    step = 1. / (num_discrete_positions_per_room)
    aux = np.linspace(
        -(1. - step), (1. - step), num_discrete_positions_per_room)
    aux = np.meshgrid(aux, aux, indexing='ij')
    user_relative_positions = aux[1] + 1j * aux[0][::-1]
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate the positions of all rooms xxxxxxxxxxxxxxxxxxxxx
    room_positions = calc_room_positions_square(side_length, num_rooms)
    room_positions.shape = (num_rooms_per_side, num_rooms_per_side)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Create the path loss object xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    pl_3gpp_obj = pathloss.PathLoss3GPP1()
    pl_free_space_obj = pathloss.PathLossFreeSpace()
    pl_3gpp_obj.handle_small_distances_bool = True
    pl_free_space_obj.handle_small_distances_bool = True
    pl_metis_ps7_obj = pathloss.PathLossMetisPS7()
    pl_metis_ps7_obj.handle_small_distances_bool = True
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxx Add one user in each discrete position of each room xxxxxxxxxx
    user_relative_positions2 = user_relative_positions * side_length / 2.
    user_positions = (room_positions[:, :, np.newaxis, np.newaxis] +
                      user_relative_positions2[np.newaxis, np.newaxis, :, :])
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx AP Allocation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # 1 AP in each room
    ap_positions = get_ap_positions(room_positions, ap_decimation)
    num_aps = ap_positions.size
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate distances: each user to each AP xxxxxxxxxxxxxxxx
    # Dimension: (romm_row, room_c, user_row, user_col, num_APs)
    dists_m = np.abs(
        user_positions[:, :, :, :, np.newaxis]
        - ap_positions.reshape([1, 1, 1, 1, -1]))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate AP association xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Determine with which AP each user is associated with.
    # Each user will associate with the CLOSEST access point.
    ap_assoc = np.argmin(dists_m, axis=-1)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate wall losses xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # We want to calculate the number walls from each room to the rooms
    # which have an access point.
    # Dimension is (num_rooms, num_aps)
    num_walls = calc_num_walls(side_length, room_positions, ap_positions)
    # Reshape it to (num_rooms_per_side, num_rooms_per_side, 1, 1, num_aps)
    num_walls_extended = num_walls.reshape(
        [num_rooms_per_side, num_rooms_per_side, 1, 1, num_aps])
    # And finally broadcast the (1, 1) dimensions to the number of users
    # per room. This will make num_walls_extended have the same dimension
    # as dists_m.
    num_walls_extended, _ = np.broadcast_arrays(num_walls_extended, dists_m)

    wall_losses_dB = num_walls_extended * single_wall_loss_dB
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate the path losses xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # The METIS PS7 path loss model require distance values in meters,
    # while the others are in Kms. All distances were calculates in meters
    # and, therefore, we divide the distance in by 1000 for 3GPP and free
    # space.
    pl_3gpp = pl_3gpp_obj.calc_path_loss(dists_m/1000.)
    pl_free_space = pl_free_space_obj.calc_path_loss(dists_m/1000.)
    pl_nothing = np.ones(
        [num_rooms_per_side,
         num_rooms_per_side,
         num_discrete_positions_per_room,
         num_discrete_positions_per_room, num_aps],
        dtype=float)

    # We need to know the number of walls the signal must pass to reach the
    # receiver to calculate the path loss for the METIS PS7 model.
    pl_metis_ps7 = pl_metis_ps7_obj.calc_path_loss(
        dists_m,
        num_walls=num_walls_extended)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate the SINRs for each path loss model xxxxxxxxxxxxx
    sinr_array_pl_nothing_dB = simulate_for_a_given_ap_assoc(
        pl_nothing, ap_assoc, wall_losses_dB, Pt, noise_var)

    sinr_array_pl_3gpp_dB = simulate_for_a_given_ap_assoc(
        pl_3gpp, ap_assoc, wall_losses_dB, Pt, noise_var)

    sinr_array_pl_free_space_dB = simulate_for_a_given_ap_assoc(
        pl_free_space, ap_assoc, wall_losses_dB, Pt, noise_var)

    sinr_array_pl_metis_ps7_dB = simulate_for_a_given_ap_assoc(
        pl_metis_ps7, ap_assoc, wall_losses_dB, Pt, noise_var)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    out = (sinr_array_pl_nothing_dB,
           sinr_array_pl_3gpp_dB,
           sinr_array_pl_free_space_dB,
           sinr_array_pl_metis_ps7_dB)
    return out


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

    (sinr_array_pl_nothing_dB,
     sinr_array_pl_3gpp_dB,
     sinr_array_pl_free_space_dB,
     sinr_array_pl_metis_ps7_dB) = out

    (num_rooms_per_side,
     _,  # num_rooms_per_side
     _,  # num_discrete_positions_per_room again
     num_discrete_positions_per_room) = sinr_array_pl_nothing_dB.shape

    print("\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("SINR statistics for the 64 central rooms")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n")

    print(("Min/Mean/Max SINR value (no PL):"
           "\n    {0}\n    {1}\n    {2}").format(
               sinr_array_pl_nothing_dB[2:-2, 2:-2].min(),
               sinr_array_pl_nothing_dB[2:-2, 2:-2].mean(),
               sinr_array_pl_nothing_dB[2:-2, 2:-2].max()))
    print(("Min/Mean/Max SINR value (3GPP):"
           "\n    {0}\n    {1}\n    {2}").format(
               sinr_array_pl_3gpp_dB[2:-2, 2:-2].min(),
               sinr_array_pl_3gpp_dB[2:-2, 2:-2].mean(),
               sinr_array_pl_3gpp_dB[2:-2, 2:-2].max()))
    print(("Min/Mean/Max SINR value (Free Space):"
           "\n    {0}\n    {1}\n    {2}").format(
               sinr_array_pl_free_space_dB[2:-2, 2:-2].min(),
               sinr_array_pl_free_space_dB[2:-2, 2:-2].mean(),
               sinr_array_pl_free_space_dB[2:-2, 2:-2].max()))
    print(("Min/Mean/Max SINR value (METIS PS7):"
           "\n    {0}\n    {1}\n    {2}").format(
               sinr_array_pl_metis_ps7_dB[2:-2, 2:-2].min(),
               sinr_array_pl_metis_ps7_dB[2:-2, 2:-2].mean(),
               sinr_array_pl_metis_ps7_dB[2:-2, 2:-2].max()))

    # xxxxxxxxxx Prepare data to be plotted xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    sinr_array_pl_nothing_dB2 = prepare_sinr_array_for_color_plot(
        sinr_array_pl_nothing_dB,
        num_rooms_per_side,
        num_discrete_positions_per_room)
    sinr_array_pl_3gpp_dB2 = prepare_sinr_array_for_color_plot(
        sinr_array_pl_3gpp_dB,
        num_rooms_per_side,
        num_discrete_positions_per_room)
    sinr_array_pl_free_space_dB2 = prepare_sinr_array_for_color_plot(
        sinr_array_pl_free_space_dB,
        num_rooms_per_side,
        num_discrete_positions_per_room)
    sinr_array_pl_metis_ps7_dB2 = prepare_sinr_array_for_color_plot(
        sinr_array_pl_metis_ps7_dB,
        num_rooms_per_side,
        num_discrete_positions_per_room)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Plot each case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # No path loss
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im1 = ax1.imshow(sinr_array_pl_nothing_dB2,
                     interpolation='nearest', vmax=-1.5, vmin=-5)
    ax1.set_title('No Path Loss')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    fig1.colorbar(im1)
    # plot_all_rooms(all_rooms, ax1)

    # 3GPP path loss
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.imshow(sinr_array_pl_3gpp_dB2,
                     interpolation='nearest', vmax=30, vmin=-2.5)
    ax2.set_title('3GPP Path Loss')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    fig2.colorbar(im2)

    # Free Space path loss
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    im3 = ax3.imshow(sinr_array_pl_free_space_dB2,
                     interpolation='nearest', vmax=30, vmin=-2.5)
    ax3.set_title('Free Space Path Loss')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    fig3.colorbar(im3)

    # METIS PS7 path loss
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    im4 = ax4.imshow(sinr_array_pl_metis_ps7_dB2,
                     interpolation='nearest', vmax=30, vmin=-2.5)
    ax4.set_title('METIS PS7 Path Loss')
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    fig4.colorbar(im4)

    plt.show()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
