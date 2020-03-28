#!/usr/bin/env python
"""
Simulator for the SINRs and capacity of a dense indoor scenario.

The scenario is a very simplified version of the Test Case 2 from the METIS
project. Only one floor of one building is simulated and only the indoor
access points are considered.
"""



# xxxxxxxxxx Import Statements xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
# import matplotlib as mpl

from apps.metis_scenarios.simulate_metis_scenario import \
    calc_room_positions_square, get_ap_positions, calc_num_walls, plot_all_rooms
from pyphysim.util.conversion import dB2Linear, dBm2Linear, linear2dB
from pyphysim.cell import shapes
from pyphysim.channels import pathloss
from pyphysim.channels.noise import calc_thermal_noise_power_dBm

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


def find_ap_assoc_best_channel(pl_all_plus_wl):
    """
    Find with which AP each user should associate with using the best
    channel criterion.

    Basically the  user will associate with the AP with the lowest path loss

    Parameters
    ----------
    pl_all_plus_wl : np.ndarray
        The path loss in linear scale including also any wall losses. This is
        a 2D numpy array (Dim: num users x num APs).

    Return
    ------
    ap_assoc : np.ndarray
        The int vector indicating with which AP each user is associated. This
        is a 1D int numpy array and the number of elements in this vector is
        equal to the number of users and each element is the index of the AP
        that the user will associate with.
    """
    ap_assoc = np.argmax(pl_all_plus_wl, axis=-1)
    return ap_assoc


def simulate_for_a_given_ap_assoc(pl_plus_wl_tx_aps, ap_assoc,
                                  transmitting_aps, Pt, noise_var):
    """
    Perform the simulation for a given AP association.

    Parameters
    ----------
    pl_plus_wl_tx_aps : np.ndarray
    ap_assoc : np.ndarray
    transmitting_aps : np.ndarray
    Pt : float | np.ndarray
    noise_var : float
        The noise variance.

    Returns
    -------
    (np.ndarray, np.ndarray)
        A tuple with the SINRs and the capacity.
    """
    # Output variables
    sinr_array = np.empty(ap_assoc.shape, dtype=float)
    capacity = np.empty(ap_assoc.shape, dtype=float)

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
        desired_power = (Pt * pl_plus_wl_tx_aps[current_ap_users_idx, index])

        undesired_power = np.sum(
            Pt * pl_plus_wl_tx_aps[current_ap_users_idx][:, mask_i_aps],
            axis=-1)

        sinr_array[current_ap_users_idx] = (desired_power /
                                            (undesired_power + noise_var))

        # The capacity (actually, the spectral efficiency since we didn't
        # multiply by the bandwidth) is calculated from the SINR. However,
        # if there is more then one user associated with the current AP we
        # assume bandwidth will be equally divided among all of them.
        capacity[current_ap_users_idx] = (
            np.log2(1 + sinr_array[current_ap_users_idx]) /
            len(current_ap_users_idx))

    return linear2dB(sinr_array), capacity


def perform_simulation(
    scenario_params,  # pylint: disable=R0914
    power_params,
    plot_results_bool=True):
    """
    Run the simulation.

    Parameters
    ----------
    scenario_params : dict
        Dictionary with simulation parameters.
        The keys are: 'side_length', 'single_wall_loss_dB',
        'num_rooms_per_side' and 'ap_decimation'.
    power_params : dict
        Dictionary with the power related parameters.
        The keys are 'Pt_dBm' and 'noise_power_dBm'.
    plot_results_bool : bool
        True if results should be plotted after the simulation finishes.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Tuple with (sinr_array_pl_metis_ps7_dB, capacity_metis_ps7)
    """
    # xxxxxxxxxx Simulation Scenario Configuration xxxxxxxxxxxxxxxxxxxxxxxx
    # The size of the side of each square room
    side_length = scenario_params['side_length']
    # How much (in dB) is lost for each wall teh signal has to pass
    single_wall_loss_dB = scenario_params['single_wall_loss_dB']

    # Square of 12 x 12 square rooms
    num_rooms_per_side = scenario_params['num_rooms_per_side']
    # Total number of rooms in the grid
    num_rooms = num_rooms_per_side**2

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
    ":type: float"
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate the positions of all rooms xxxxxxxxxxxxxxxxxxxxx
    room_positions = calc_room_positions_square(side_length, num_rooms)
    room_positions.shape = (num_rooms_per_side, num_rooms_per_side)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Create the path loss object xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    pl_metis_ps7_obj = pathloss.PathLossMetisPS7()
    pl_metis_ps7_obj.handle_small_distances_bool = True
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Add users in random positions in the 2D grid xxxxxxxxxxxxx
    num_users = 100  # We will create this many users in the 2D grid
    users_positions = (num_rooms_per_side * side_length *
                       (np.random.random_sample(num_users) +
                        1j * np.random.random_sample(num_users) - 0.5 - 0.5j))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx AP Allocation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # 1 AP in each room
    ap_positions = get_ap_positions(room_positions, ap_decimation)
    num_aps = ap_positions.size
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate distances: each user to each AP xxxxxxxxxxxxxxxx
    # Dimension: (num_users, num_APs)
    dists_m = np.abs(users_positions[:, np.newaxis] -
                     ap_positions[np.newaxis, :])
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxx Calculate AP association xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # INPUTS
    # Find in which room each user is
    users_rooms = np.argmin(np.abs(
        room_positions.reshape([-1, 1]) - users_positions[np.newaxis, :]),
                            axis=0)

    # Number of walls from each room to each other room
    num_walls_all_rooms = calc_num_walls(side_length, room_positions,
                                         ap_positions)
    # Number of walls from each room that has at least one user to each
    # room with an AP
    num_walls_rooms_with_users = num_walls_all_rooms[users_rooms]

    # Path loss from each user to each AP (no matter if it will be a
    # transmitting AP or not, since we still have to perform the AP
    # association)
    pl_all = pl_metis_ps7_obj.calc_path_loss(
        dists_m, num_walls=num_walls_rooms_with_users)

    # Calculate wall losses from each user to each AP (no matter if it will
    # be a transmitting AP or not, since we still have to perform the AP
    # association)
    wall_losses_dB_all = num_walls_rooms_with_users * single_wall_loss_dB

    # Calculate path loss plus wall losses (we multiply the linear values)
    # from each user to each AP (no matter if it will be a transmitting AP
    # or not, since we still have to perform the AP association)
    pl_all_plus_wl = pl_all * dB2Linear(-wall_losses_dB_all)

    # OUTPUTS
    # Determine with which AP each user is associated with.
    # Each user will associate with the CLOSEST access point.
    ap_assoc = find_ap_assoc_best_channel(pl_all_plus_wl)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Find which Access Points should stay on xxxxxxxxxxxxxxxxxx
    # Indexes of the active APs
    transmitting_aps, users_count = np.unique(ap_assoc, return_counts=True)
    # Asserts to tell pycharm that these are numpy arrays
    assert isinstance(transmitting_aps, np.ndarray)
    assert isinstance(users_count, np.ndarray)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate the SINRs for each path loss model xxxxxxxxxxxxx
    # Take the path loss plus wall losses only for the transmitting aps
    pl_all_plus_wall_losses_tx_aps = pl_all_plus_wl.take(transmitting_aps,
                                                         axis=1)
    ":type: np.ndarray"
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Calculate the SINRs and capacity xxxxxxxxxxxxxxxxxxxxxxxxx
    sinr_array_pl_metis_ps7_dB, capacity_metis_ps7 \
        = simulate_for_a_given_ap_assoc(
            pl_all_plus_wall_losses_tx_aps, ap_assoc,
            transmitting_aps, Pt, noise_var)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxx Plot the results xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if plot_results_bool is True:
        print(("\nMin/Mean/Max SINR value (METIS PS7):"
               "\n    {0}\n    {1}\n    {2}").format(
                   sinr_array_pl_metis_ps7_dB.min(),
                   sinr_array_pl_metis_ps7_dB.mean(),
                   sinr_array_pl_metis_ps7_dB.max()))

        print(("\nMin/Mean/Max Capacity value (METIS PS7):"
               "\n    {0}\n    {1}\n    {2}").format(capacity_metis_ps7.min(),
                                                     capacity_metis_ps7.mean(),
                                                     capacity_metis_ps7.max()))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxx Plot the results xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Create a mask for the active APs
        transmitting_aps_mask = np.zeros(num_aps, dtype=bool)
        transmitting_aps_mask[transmitting_aps] = True

        # Save how many users are associated with each AP
        users_per_ap = np.zeros(num_aps, dtype=int)
        users_per_ap[transmitting_aps_mask] = users_count

        # xxxxxxxxxx Plot all rooms and users xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        all_rooms = [
            shapes.Rectangle(pos - side_length / 2. - side_length * 1j / 2.,
                             pos + side_length / 2. + side_length * 1j / 2.)
            for pos in room_positions.flatten()
        ]

        # Plot all Rooms and save the axis where they were plotted
        plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        # ax1 is where we will plot everything
        ax1 = plt.subplot(gs[0])
        ax1.set_xlabel("Position X coordinate")
        ax1.set_ylabel("Position Y coordinate")
        ax1.set_title("Plot of all Rooms")
        ax1.set_ylim([-60, 60])
        ax1.set_xlim([-60, 60])
        ax1 = plot_all_rooms(all_rooms, ax1)
        #ax1.hold(True)

        # ax2 will be used for annotations
        ax2 = plt.subplot(gs[1])
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax2.get_yticklabels(), visible=False)
        ax2.set_ylim([0, 10])
        ax2.set_xlim([0, 10])
        details = ax2.text(5,
                           5,
                           'Details',
                           verticalalignment='center',
                           horizontalalignment='center',
                           family='monospace')

        # Set the an array with colors for the access points. Transmitting APs
        # will be blue, while inactive APs will be gray
        ap_colors = np.empty(ap_positions.shape, dtype='U4')
        ap_colors[transmitting_aps_mask] = 'b'
        ap_colors[np.logical_not(transmitting_aps_mask)] = 'gray'

        # Plot the access points. We set linewidth to 0.0 so that there is no
        # border. We set the size ('s' keyword) to 50 to make it larger. The
        # colors are set according to the ap_colors array.
        # Note that we set a 5 points tolerance for the pick event.
        aps_plt = ax1.scatter(ap_positions.real,
                              ap_positions.imag,
                              marker='^',
                              c=ap_colors,
                              linewidths=0.1,
                              s=50,
                              picker=3)

        # Plot the users
        # Note that we set a 5 points tolerance for the pick event.
        users_plt = ax1.scatter(users_positions.real,
                                users_positions.imag,
                                marker='*',
                                c='r',
                                linewidth=0.1,
                                s=50,
                                picker=3)

        # xxxxxxxxxx Define a function to call for the pick_event Circle used
        # to select an AP. We will set its visibility to False here. When an AP
        # is selected, we move this circle to its position and set its
        # visibility to True.
        selected_ap_circle = ax1.plot([0], [0],
                                      'o',
                                      ms=12,
                                      alpha=0.4,
                                      color='yellow',
                                      visible=False)[0]

        # Define the callback function for the pick event
        def on_pick(event):
            """Callback for the pick event in the matplotlib plot.

            Parameters
            ----------
            event : Matplotlib event
            """
            # We will reset users colors on each pick
            users_colors = np.empty(ap_assoc.size, dtype='U1')
            users_colors[:] = 'r'

            # Index of the point clicked
            ind = event.ind[0]

            if event.artist == aps_plt:
                # Disable the circle in the AP
                selected_ap_circle.set_visible(False)

                if ind not in ap_assoc:
                    # Text information for the disabled AP
                    text = "AP {0} (Disabled)".format(ind)
                else:
                    # Text information for the selected AP
                    text = "AP {0} with {1} user(s)\nTotal throughput: {2:7.4f}"
                    text = text.format(
                        ind, users_per_ap[ind],
                        np.sum(capacity_metis_ps7[ap_assoc == ind]))

                    # Change the colors of the users associated with the
                    # current AP to green
                    users_colors[ap_assoc == ind] = 'g'

            elif event.artist == users_plt:
                # Text information for the selected user
                text = "User {0}\n    SINR: {1:7.4f}\nCapacity: {2:7.4f}".format(
                    ind, sinr_array_pl_metis_ps7_dB[ind],
                    capacity_metis_ps7[ind])

                # If there other users are associated with the same AP of the
                # current user
                if users_per_ap[ap_assoc[ind]] > 1:
                    text = "{0}\nShares AP with {1} other user(s)".format(
                        text, users_per_ap[ap_assoc[ind]] - 1)

                users_AP = ap_assoc[ind]
                # Plot a yellow circle in the user's AP
                ap_pos = ap_positions[users_AP]
                # Change the color of other users in the same AP to green and
                # the current user to cyan
                users_colors[ap_assoc == users_AP] = 'g'
                users_colors[ind] = 'c'

                selected_ap_circle.set_visible(True)
                selected_ap_circle.set_data([ap_pos.real], [ap_pos.imag])

            # Set users colors
            users_plt.set_color(users_colors)

            # Set the details text
            # noinspection PyUnboundLocalVariable
            details.set_text(text)
            ax1.figure.canvas.draw()

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Connect the on_pick function with the pick event
        ax1.figure.canvas.mpl_connect('pick_event', on_pick)

        plt.show()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxx Return the results xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    return sinr_array_pl_metis_ps7_dB, capacity_metis_ps7


if __name__ == '__main__':
    scenario_params = {
        'side_length': 10.,  # 10 meters side length
        'single_wall_loss_dB': 5.,
        'num_rooms_per_side': 12,
        'ap_decimation': 2
    }

    power_params = {
        'Pt_dBm': 20.,  # 20 dBm transmit power
        # Noise power for 25°C for a bandwidth of 5 MHz ->  -106.87 dBm
        'noise_power_dBm': calc_thermal_noise_power_dBm(25, 5e6)
    }

    out = perform_simulation(scenario_params,
                             power_params,
                             plot_results_bool=True)
    sinr_array_pl_metis_ps7_dB, capacity_metis_ps7 = out
