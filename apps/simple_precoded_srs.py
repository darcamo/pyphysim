#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('../')
# from matplotlib import pyplot as plt

import math
import numpy as np
from matplotlib import pyplot as plt
from pyphysim.reference_signals.zadoffchu import calcBaseZC, \
    get_extended_ZF
from pyphysim.reference_signals.srs import get_srs_seq
from pyphysim.channels.fading import COST259_TUx, TdlChannel
from pyphysim.channels.fading_generators import JakesSampleGenerator
from pyphysim.util.conversion import linear2dB

# noinspection PyPackageRequirements
import bokeh.plotting as bp
# from bokeh.plotting import figure, output_server, show, ColumnDataSource, \
# gridplot
# noinspection PyPackageRequirements
from bokeh.models import HoverTool
# noinspection PyPackageRequirements
import bokeh.models.widgets as bw
# noinspection PyPackageRequirements
from bokeh.io import gridplot


# matplotlib.interactive(True)


def plot_true_and_estimated_channel(
        true_channel, estimated_channel, title='', antenna=0):
    true_channel_time = np.fft.ifft(true_channel, axis=0)

    f = plt.figure(figsize=(12, 14))
    ax = f.add_subplot(2, 1, 1)
    ax.set_title(title)
    ax.stem(np.abs(true_channel_time[:, antenna]))

    ax = f.add_subplot(2, 1, 2)
    ax.set_title(title)
    ax.plot(np.abs(estimated_channel[:, antenna]), 'r')
    ax.hold(True)
    ax.plot(np.abs(true_channel[:, antenna]))
    ax.legend(['Estimated', 'True'])

    return f


def plot_true_and_estimated_channel_with_bokeh(
        true_channel, estimated_channel, title='', antenna=0):
    true_channel_time = np.fft.ifft(true_channel[:, antenna], axis=0)

    num_subcarriers = true_channel.shape[0]

    data = {
        'index': np.r_[0:num_subcarriers],
        'channel_time': np.abs(true_channel_time),
        'channel': np.abs(true_channel[:, antenna]),
        'estimated_channel': np.abs(estimated_channel[:, antenna]),
        'error': linear2dB(
            np.abs(true_channel[:, antenna] - estimated_channel[:, antenna]) / \
            np.abs(true_channel[:, antenna]))}
    source00 = bp.ColumnDataSource(data=data)

    # Specify the tools by name. This does not allow us to set parameters
    # for the tools
    TOOLS = "pan,wheel_zoom,box_zoom,reset,resize,crosshair"
    # Create a hover tool and tell it to only show for a curve with name
    # 'estimated'
    hover_tool = HoverTool(
        names=['estimated'],
        tooltips=[('True Channel', ''), ('Error (in dB)', '@error')])
    # p1 = bp.figure(tools=TOOLS, width=600, height=200)
    # p1.circle('index', 'channel_time', source=source00)
    # p1.title = title

    p2 = bp.figure(tools=TOOLS, plot_width=400, plot_height=300)
    p2.add_tools(hover_tool)

    p2.line('index', 'channel', source=source00, legend='True Channel')
    # We specify the 'name' attribute so that we can specify the 'names'
    # attribute for the hover tool and tell it to only show for the
    # estimated channel curve.
    p2.line('index', 'estimated_channel', source=source00, color='red',
            name='estimated', legend='Estimated Channel')

    p2.title = title

    # p = bp.vplot(p1, p2)
    # return p
    return p2


def plot_true_and_estimated_channel_with_bokeh_all_antennas(
        true_channel, estimated_channel, title=''):
    p0 = plot_true_and_estimated_channel_with_bokeh(
        true_channel, estimated_channel, 'Antenna 1', 0)
    p1 = plot_true_and_estimated_channel_with_bokeh(
        true_channel, estimated_channel, 'Antenna 2', 1)
    p2 = plot_true_and_estimated_channel_with_bokeh(
        true_channel, estimated_channel, 'Antenna 3', 2)
    p3 = plot_true_and_estimated_channel_with_bokeh(
        true_channel, estimated_channel, 'Antenna 4', 3)
    p1.x_range = p0.x_range
    p2.x_range = p0.x_range
    p3.x_range = p0.x_range
    p1.y_range = p0.y_range
    p2.y_range = p0.y_range
    p3.y_range = p0.y_range

    p = gridplot([[p0, p1], [p2, p3]])

    par = bw.Paragraph(text=title)
    vbox = bw.VBox(children=[par, p])
    return vbox


def estimate_channels_remove_only_direct(Y1, Y2, Y3, r1, r2, r3, Nsc,
                                         comb_indexes):
    # xxxxxxxxxx AN 1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    tilde_y11 = np.fft.ifft(Y1 * r1[:, np.newaxis].conj(), n=Nsc // 2, axis=0)
    tilde_y11[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH11_eq_est = np.fft.fft(tilde_y11, n=Nsc, axis=0)

    Y1_no_dir = Y1 - (uH11_eq_est[comb_indexes] * r1[:, np.newaxis])

    # UE 2 to AN 1
    tilde_y12 = np.fft.ifft(Y1_no_dir * r2[:, np.newaxis].conj(), n=Nsc // 2,
                            axis=0)
    tilde_y12[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH12_eq_est = np.fft.fft(tilde_y12, n=Nsc, axis=0)

    # UE 3 to AN 1
    tilde_y13 = np.fft.ifft(Y1_no_dir * r3[:, np.newaxis].conj(), n=Nsc // 2,
                            axis=0)
    tilde_y13[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH13_eq_est = np.fft.fft(tilde_y13, n=Nsc, axis=0)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx AN 2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # UE 2 to AN 2
    tilde_y22 = np.fft.ifft(Y2 * r2[:, np.newaxis].conj(), n=Nsc // 2, axis=0)
    tilde_y22[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH22_eq_est = np.fft.fft(tilde_y22, n=Nsc, axis=0)
    Y2_no_dir = Y2 - (uH22_eq_est[comb_indexes] * r2[:, np.newaxis])

    # UE 1 to AN 2
    tilde_y21 = np.fft.ifft(Y2_no_dir * r1[:, np.newaxis].conj(), n=Nsc // 2,
                            axis=0)
    tilde_y21[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH21_eq_est = np.fft.fft(tilde_y21, n=Nsc, axis=0)

    # UE 3 to AN 2
    tilde_y23 = np.fft.ifft(Y2_no_dir * r3[:, np.newaxis].conj(), n=Nsc // 2,
                            axis=0)
    tilde_y23[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH23_eq_est = np.fft.fft(tilde_y23, n=Nsc, axis=0)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx AN 3 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # UE 3 to AN 3
    tilde_y33 = np.fft.ifft(Y3 * r3[:, np.newaxis].conj(), n=Nsc // 2, axis=0)
    tilde_y33[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH33_eq_est = np.fft.fft(tilde_y33, n=Nsc, axis=0)
    Y3_no_dir = Y3 - (uH33_eq_est[comb_indexes] * r3[:, np.newaxis])

    # UE 1 to AN 3
    tilde_y31 = np.fft.ifft(Y3_no_dir * r1[:, np.newaxis].conj(), n=Nsc // 2,
                            axis=0)
    tilde_y31[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH31_eq_est = np.fft.fft(tilde_y31, n=Nsc, axis=0)

    # UE 2 to AN 3
    tilde_y32 = np.fft.ifft(Y3_no_dir * r2[:, np.newaxis].conj(), n=Nsc // 2,
                            axis=0)
    tilde_y32[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH32_eq_est = np.fft.fft(tilde_y32, n=Nsc, axis=0)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return (uH11_eq_est, uH12_eq_est, uH13_eq_est, uH21_eq_est, uH22_eq_est,
            uH23_eq_est, uH31_eq_est, uH32_eq_est, uH33_eq_est)


def estimate_channels_remove_direct_and_perform_SIC(
        Y1, Y2, Y3, r1, r2, r3, Nsc, comb_indexes):
    # xxxxxxxxxx AN 1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    tilde_y11 = np.fft.ifft(Y1 * r1[:, np.newaxis].conj(), n=Nsc // 2, axis=0)
    tilde_y11[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH11_eq_est = np.fft.fft(tilde_y11, n=Nsc, axis=0)

    Y1_no_dir = Y1 - (uH11_eq_est[comb_indexes] * r1[:, np.newaxis])

    # UE 2 to AN 1
    tilde_y12 = np.fft.ifft(Y1_no_dir * r2[:, np.newaxis].conj(), n=Nsc // 2,
                            axis=0)
    tilde_y12[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH12_eq_est = np.fft.fft(tilde_y12, n=Nsc, axis=0)

    # UE 3 to AN 1
    tilde_y13 = np.fft.ifft(Y1_no_dir * r3[:, np.newaxis].conj(), n=Nsc // 2,
                            axis=0)
    tilde_y13[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH13_eq_est = np.fft.fft(tilde_y13, n=Nsc, axis=0)

    # Perform SIC for the weakest interfering link
    if np.linalg.norm(uH12_eq_est) > np.linalg.norm(uH13_eq_est):
        # H12 is stronger than H13. Let' remove interference from UE 2 and
        # estimate again for UE 3
        Y1_SIC = Y1_no_dir - (uH12_eq_est[comb_indexes] * r2[:, np.newaxis])
        tilde_y13 = np.fft.ifft(Y1_SIC * r3[:, np.newaxis].conj(), n=Nsc // 2,
                                axis=0)
        tilde_y13[11:,
        :] = 0  # Only keep the first 11 time samples for each antenna
        uH13_eq_est = np.fft.fft(tilde_y13, n=Nsc, axis=0)
        pass
    else:
        # H13 is stronger than H12. Let' remove interference from UE 3 and
        # estimate again for UE 2
        Y1_SIC = Y1_no_dir - (uH13_eq_est[comb_indexes] * r3[:, np.newaxis])
        tilde_y12 = np.fft.ifft(Y1_SIC * r2[:, np.newaxis].conj(),
                                n=Nsc // 2,
                                axis=0)
        # Only keep the first 11 time samples for each antenna
        tilde_y12[11:, :] = 0
        uH12_eq_est = np.fft.fft(tilde_y12, n=Nsc, axis=0)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx AN 2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # UE 2 to AN 2
    tilde_y22 = np.fft.ifft(Y2 * r2[:, np.newaxis].conj(), n=Nsc // 2, axis=0)
    tilde_y22[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH22_eq_est = np.fft.fft(tilde_y22, n=Nsc, axis=0)
    Y2_no_dir = Y2 - (uH22_eq_est[comb_indexes] * r2[:, np.newaxis])

    # UE 1 to AN 2
    tilde_y21 = np.fft.ifft(Y2_no_dir * r1[:, np.newaxis].conj(), n=Nsc // 2,
                            axis=0)
    tilde_y21[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH21_eq_est = np.fft.fft(tilde_y21, n=Nsc, axis=0)

    # UE 3 to AN 2
    tilde_y23 = np.fft.ifft(Y2_no_dir * r3[:, np.newaxis].conj(), n=Nsc // 2,
                            axis=0)
    tilde_y23[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH23_eq_est = np.fft.fft(tilde_y23, n=Nsc, axis=0)

    # Perform SIC for the weakest interfering link
    if np.linalg.norm(uH21_eq_est) > np.linalg.norm(uH23_eq_est):
        Y2_SIC = Y2_no_dir - (uH21_eq_est[comb_indexes] * r1[:, np.newaxis])
        tilde_y23 = np.fft.ifft(Y2_SIC * r3[:, np.newaxis].conj(), n=Nsc // 2,
                                axis=0)
        tilde_y23[11:,
        :] = 0  # Only keep the first 11 time samples for each antenna
        uH23_eq_est = np.fft.fft(tilde_y23, n=Nsc, axis=0)
    else:
        Y2_SIC = Y2_no_dir - (uH23_eq_est[comb_indexes] * r3[:, np.newaxis])
        tilde_y21 = np.fft.ifft(Y2_SIC * r1[:, np.newaxis].conj(), n=Nsc // 2,
                                axis=0)
        tilde_y21[11:,
        :] = 0  # Only keep the first 11 time samples for each antenna
        uH21_eq_est = np.fft.fft(tilde_y21, n=Nsc, axis=0)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx AN 3 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # UE 3 to AN 3
    tilde_y33 = np.fft.ifft(Y3 * r3[:, np.newaxis].conj(), n=Nsc // 2, axis=0)
    tilde_y33[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH33_eq_est = np.fft.fft(tilde_y33, n=Nsc, axis=0)
    Y3_no_dir = Y3 - (uH33_eq_est[comb_indexes] * r3[:, np.newaxis])

    # UE 1 to AN 3
    tilde_y31 = np.fft.ifft(Y3_no_dir * r1[:, np.newaxis].conj(), n=Nsc // 2,
                            axis=0)
    tilde_y31[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH31_eq_est = np.fft.fft(tilde_y31, n=Nsc, axis=0)

    # UE 2 to AN 3
    tilde_y32 = np.fft.ifft(Y3_no_dir * r2[:, np.newaxis].conj(), n=Nsc // 2,
                            axis=0)
    tilde_y32[11:,
    :] = 0  # Only keep the first 11 time samples for each antenna
    uH32_eq_est = np.fft.fft(tilde_y32, n=Nsc, axis=0)

    # Perform SIC for the weakest interfering link
    if np.linalg.norm(uH31_eq_est) > np.linalg.norm(uH32_eq_est):
        Y3_SIC = Y3_no_dir - (uH31_eq_est[comb_indexes] * r1[:, np.newaxis])
        tilde_y32 = np.fft.ifft(Y3_SIC * r2[:, np.newaxis].conj(), n=Nsc // 2,
                                axis=0)
        tilde_y32[11:,
        :] = 0  # Only keep the first 11 time samples for each antenna
        uH32_eq_est = np.fft.fft(tilde_y32, n=Nsc, axis=0)
    else:
        Y3_SIC = Y3_no_dir - (uH32_eq_est[comb_indexes] * r2[:, np.newaxis])
        tilde_y31 = np.fft.ifft(Y3_SIC * r1[:, np.newaxis].conj(), n=Nsc // 2,
                                axis=0)
        tilde_y31[11:,
        :] = 0  # Only keep the first 11 time samples for each antenna
        uH31_eq_est = np.fft.fft(tilde_y31, n=Nsc, axis=0)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return (uH11_eq_est, uH12_eq_est, uH13_eq_est, uH21_eq_est, uH22_eq_est,
            uH23_eq_est, uH31_eq_est, uH32_eq_est, uH33_eq_est)


def compute_channel_estimation_error_dB(H, Hest):
    return np.mean(linear2dB(np.abs(Hest - H) / np.abs(H)))


def main():
    # session = bp.Session(load_from_config=False)
    # bp.output_server(docname='simple_precoded_srs_AN1', session=session)
    bp.output_file('simple_precoded_srs.html', title="Simple Precoded SRS")

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Scenario Description xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # 3 Base Stations, each sending data to its own user while interfering
    # with the other users.

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Configuration xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    num_prbs = 25  # Number of PRBs to simulate
    Nsc = 12 * num_prbs  # Number of subcarriers
    Nzc = 149  # Size of the sequence
    u1 = 1  # Root sequence index of the first user
    u2 = 2  # Root sequence index of the first user
    u3 = 3  # Root sequence index of the first user
    numAnAnt = 4  # Number of Base station antennas
    numUeAnt = 2  # Number of UE antennas

    num_samples = 1  # Number of simulated channel samples (from
    # Jakes process)

    # Channel configuration
    speedTerminal = 0 / 3.6  # Speed in m/s
    fcDbl = 2.6e9  # Central carrier frequency (in Hz)
    timeTTIDbl = 1e-3  # Time of a single TTI
    subcarrierBandDbl = 15e3  # Subcarrier bandwidth (in Hz)
    numOfSubcarriersPRBInt = 12  # Number of subcarriers in each PRB
    L = 16  # The number of rays for the Jakes model.

    # Dependent parameters
    lambdaDbl = 3e8 / fcDbl  # Carrier wave length
    Fd = speedTerminal / lambdaDbl  # Doppler Frequency
    Ts = 1. / (Nsc * subcarrierBandDbl)  # Sampling time

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Generate the root sequence xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    a_u1 = get_extended_ZF(calcBaseZC(Nzc, u1), Nsc / 2)
    a_u2 = get_extended_ZF(calcBaseZC(Nzc, u2), Nsc / 2)
    a_u3 = get_extended_ZF(calcBaseZC(Nzc, u3), Nsc / 2)

    print("Nsc: {0}".format(Nsc))
    print("a_u.shape: {0}".format(a_u1.shape))

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Create shifted sequences for 3 users xxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # We arbitrarily choose some cyclic shift index and then we call
    # zadoffchu.get_srs_seq to get the shifted sequence.
    shift_index = 4
    r1 = get_srs_seq(a_u1, shift_index)
    r2 = get_srs_seq(a_u2, shift_index)
    r3 = get_srs_seq(a_u3, shift_index)

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Generate channels from users to the BS xxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    jakes_all_links = np.empty([3, 3], dtype=object)
    tdlchannels_all_links = np.empty([3, 3], dtype=object)
    impulse_responses = np.empty([3, 3], dtype=object)
    # Dimension: `UEs x ANs x num_subcarriers x numUeAnt x numAnAnt`
    freq_responses = np.empty([3, 3, Nsc, numUeAnt, numAnAnt], dtype=complex)

    for ueIdx in range(3):
        for anIdx in range(3):
            jakes_all_links[ueIdx, anIdx] = JakesSampleGenerator(
                Fd, Ts, L, shape=(numUeAnt, numAnAnt))

            tdlchannels_all_links[ueIdx, anIdx] = TdlChannel(
                jakes_all_links[ueIdx, anIdx],
                tap_powers_dB=COST259_TUx.tap_powers_dB,
                tap_delays=COST259_TUx.tap_delays)

            tdlchannels_all_links[ueIdx, anIdx].generate_impulse_response(
                num_samples)

            impulse_responses[ueIdx, anIdx] \
                = tdlchannels_all_links[
                ueIdx, anIdx].get_last_impulse_response()

            freq_responses[ueIdx, anIdx] = \
                impulse_responses[ueIdx, anIdx].get_freq_response(Nsc)[:, :, :,
                0]

    # xxxxxxxxxx Channels in downlink direction xxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Dimension: `Nsc x numUeAnt x numAnAnt`
    dH11 = freq_responses[0, 0]
    dH12 = freq_responses[0, 1]
    dH13 = freq_responses[0, 2]
    dH21 = freq_responses[1, 0]
    dH22 = freq_responses[1, 1]
    dH23 = freq_responses[1, 2]
    dH31 = freq_responses[2, 0]
    dH32 = freq_responses[2, 1]
    dH33 = freq_responses[2, 2]

    # xxxxxxxxxx Principal dimension in downlink direction xxxxxxxxxxxxxxxx
    sc_idx = 124  # Index of the subcarrier we are interested in
    [dU11, _, _] = np.linalg.svd(dH11[sc_idx])
    [dU22, _, _] = np.linalg.svd(dH22[sc_idx])
    [dU33, _, _] = np.linalg.svd(dH33[sc_idx])

    # xxxxxxxxxx Users precoders xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Users' precoders are the main column of the U matrix
    F11 = dU11[:, 0].conj()
    F22 = dU22[:, 0].conj()
    F33 = dU33[:, 0].conj()

    # xxxxxxxxxx Channels in uplink direction xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # int_path_loss = 0.05  # Path loss of the interfering links
    # int_g = math.sqrt(int_path_loss)  # Gain of interfering links
    # dir_d = 1.0                   #  Gain of direct links

    pl = np.array([[2.21e-08, 2.14e-09, 1.88e-08],
                   [3.45e-10, 2.17e-08, 4.53e-10],
                   [4.38e-10, 8.04e-10, 4.75e-08]])
    # pl = np.array([[  1,   0.1,   0.1],
    #                [  0.1,   1,   0.1],
    #                [  0.1,   0.1,   1]])


    # Dimension: `Nsc x numAnAnt x numUeAnt`
    uH11 = math.sqrt(pl[0, 0]) * np.transpose(dH11, axes=[0, 2, 1])
    uH12 = math.sqrt(pl[0, 1]) * np.transpose(dH12, axes=[0, 2, 1])
    uH13 = math.sqrt(pl[0, 2]) * np.transpose(dH13, axes=[0, 2, 1])
    uH21 = math.sqrt(pl[1, 0]) * np.transpose(dH21, axes=[0, 2, 1])
    uH22 = math.sqrt(pl[1, 1]) * np.transpose(dH22, axes=[0, 2, 1])
    uH23 = math.sqrt(pl[1, 2]) * np.transpose(dH23, axes=[0, 2, 1])
    uH31 = math.sqrt(pl[2, 0]) * np.transpose(dH31, axes=[0, 2, 1])
    uH32 = math.sqrt(pl[2, 1]) * np.transpose(dH32, axes=[0, 2, 1])
    uH33 = math.sqrt(pl[2, 2]) * np.transpose(dH33, axes=[0, 2, 1])

    # Compute the equivalent uplink channels
    uH11_eq = uH11.dot(F11)
    uH12_eq = uH12.dot(F22)
    uH13_eq = uH13.dot(F33)
    uH21_eq = uH21.dot(F11)
    uH22_eq = uH22.dot(F22)
    uH23_eq = uH23.dot(F33)
    uH31_eq = uH31.dot(F11)
    uH32_eq = uH32.dot(F22)
    uH33_eq = uH33.dot(F33)

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Compute Received Signals xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Calculate the received signals
    comb_indexes = np.r_[0:Nsc:2]
    Y1_term11 = uH11_eq[comb_indexes] * r1[:, np.newaxis]
    Y1_term12 = uH12_eq[comb_indexes] * r2[:, np.newaxis]
    Y1_term13 = uH13_eq[comb_indexes] * r3[:, np.newaxis]
    Y1 = Y1_term11 + Y1_term12 + Y1_term13

    Y2_term21 = uH21_eq[comb_indexes] * r1[:, np.newaxis]
    Y2_term22 = uH22_eq[comb_indexes] * r2[:, np.newaxis]
    Y2_term23 = uH23_eq[comb_indexes] * r3[:, np.newaxis]
    Y2 = Y2_term21 + Y2_term22 + Y2_term23

    Y3_term31 = uH31_eq[comb_indexes] * r1[:, np.newaxis]
    Y3_term32 = uH32_eq[comb_indexes] * r2[:, np.newaxis]
    Y3_term33 = uH33_eq[comb_indexes] * r3[:, np.newaxis]
    Y3 = Y3_term31 + Y3_term32 + Y3_term33

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Estimate the equivalent channel xxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    (uH11_eq_est, uH12_eq_est, uH13_eq_est, uH21_eq_est, uH22_eq_est,
     uH23_eq_est, uH31_eq_est, uH32_eq_est, uH33_eq_est
     ) = estimate_channels_remove_only_direct(Y1, Y2, Y3, r1, r2, r3, Nsc,
                                              comb_indexes)

    (uH11_eq_est_SIC, uH12_eq_est_SIC, uH13_eq_est_SIC, uH21_eq_est_SIC,
     uH22_eq_est_SIC,
     uH23_eq_est_SIC, uH31_eq_est_SIC, uH32_eq_est_SIC, uH33_eq_est_SIC
     ) = estimate_channels_remove_direct_and_perform_SIC(
        Y1, Y2, Y3, r1, r2, r3, Nsc, comb_indexes)

    # Compute the MSE reduction due to SIC
    improve11 = compute_channel_estimation_error_dB(uH11_eq,
                                                    uH11_eq_est) - compute_channel_estimation_error_dB(
        uH11_eq, uH11_eq_est_SIC)
    improve12 = compute_channel_estimation_error_dB(uH12_eq,
                                                    uH12_eq_est) - compute_channel_estimation_error_dB(
        uH12_eq, uH12_eq_est_SIC)
    improve13 = compute_channel_estimation_error_dB(uH13_eq,
                                                    uH13_eq_est) - compute_channel_estimation_error_dB(
        uH13_eq, uH13_eq_est_SIC)

    improve21 = compute_channel_estimation_error_dB(uH21_eq,
                                                    uH21_eq_est) - compute_channel_estimation_error_dB(
        uH21_eq, uH21_eq_est_SIC)
    improve22 = compute_channel_estimation_error_dB(uH22_eq,
                                                    uH22_eq_est) - compute_channel_estimation_error_dB(
        uH22_eq, uH22_eq_est_SIC)
    improve23 = compute_channel_estimation_error_dB(uH23_eq,
                                                    uH23_eq_est) - compute_channel_estimation_error_dB(
        uH23_eq, uH23_eq_est_SIC)

    improve31 = compute_channel_estimation_error_dB(uH31_eq,
                                                    uH31_eq_est) - compute_channel_estimation_error_dB(
        uH31_eq, uH31_eq_est_SIC)
    improve32 = compute_channel_estimation_error_dB(uH32_eq,
                                                    uH32_eq_est) - compute_channel_estimation_error_dB(
        uH32_eq, uH32_eq_est_SIC)
    improve33 = compute_channel_estimation_error_dB(uH33_eq,
                                                    uH33_eq_est) - compute_channel_estimation_error_dB(
        uH33_eq, uH33_eq_est_SIC)
    print(improve11)
    print(improve12)
    print(improve13)
    print(improve21)
    print(improve22)
    print(improve23)
    print(improve31)
    print(improve32)
    print(improve33)

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Plot the true and estimated channels xxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    p1 = plot_true_and_estimated_channel_with_bokeh_all_antennas(
        uH11_eq, uH11_eq_est,
        title='Direct Channel from UE1 to AN1')
    p2 = plot_true_and_estimated_channel_with_bokeh_all_antennas(
        uH12_eq, uH12_eq_est,
        title='Interfering Channel from UE2 to AN1')
    p3 = plot_true_and_estimated_channel_with_bokeh_all_antennas(
        uH13_eq, uH13_eq_est,
        title='Interfering Channel from UE3 to AN1')
    tab1 = bw.Panel(child=p1, title="UE1 to AN1")
    tab2 = bw.Panel(child=p2, title="UE2 to AN1")
    tab3 = bw.Panel(child=p3, title="UE3 to AN1")
    tabs_an1 = bw.Tabs(tabs=[tab1, tab2, tab3])

    p1 = plot_true_and_estimated_channel_with_bokeh_all_antennas(
        uH21_eq, uH21_eq_est,
        title='Interfering Channel from UE1 to AN2')
    p2 = plot_true_and_estimated_channel_with_bokeh_all_antennas(
        uH22_eq, uH22_eq_est,
        title='Direct Channel from UE2 to AN2')
    p3 = plot_true_and_estimated_channel_with_bokeh_all_antennas(
        uH23_eq, uH23_eq_est,
        title='Interfering Channel from UE3 to AN2')
    tab1 = bw.Panel(child=p1, title="UE1 to AN2")
    tab2 = bw.Panel(child=p2, title="UE2 to AN2")
    tab3 = bw.Panel(child=p3, title="UE3 to AN2")
    tabs_an2 = bw.Tabs(tabs=[tab1, tab2, tab3])

    p1 = plot_true_and_estimated_channel_with_bokeh_all_antennas(
        uH31_eq, uH31_eq_est,
        title='Interfering Channel from UE1 to AN3')
    p2 = plot_true_and_estimated_channel_with_bokeh_all_antennas(
        uH32_eq, uH32_eq_est,
        title='Interfering Channel from UE2 to AN3')
    p3 = plot_true_and_estimated_channel_with_bokeh_all_antennas(
        uH33_eq, uH33_eq_est,
        title='Direct Channel from UE3 to AN3')
    tab1 = bw.Panel(child=p1, title="UE1 to AN3")
    tab2 = bw.Panel(child=p2, title="UE2 to AN3")
    tab3 = bw.Panel(child=p3, title="UE3 to AN3")
    tabs_an3 = bw.Tabs(tabs=[tab1, tab2, tab3])

    # Put each AN tab as a panel of an "ANs tab" and show it
    tabs1 = bw.Panel(child=tabs_an1, title="AN1")
    tabs2 = bw.Panel(child=tabs_an2, title="AN2")
    tabs3 = bw.Panel(child=tabs_an3, title="AN3")
    tabs_all = bw.Tabs(tabs=[tabs1, tabs2, tabs3])
    bp.show(tabs_all)



    # f1 = plot_true_and_estimated_channel(
    #     uH11_eq, uH11_eq_est,
    #     title='Direct Channel from UE1 to AN1',
    #     antenna=0)
    # f2 = plot_true_and_estimated_channel(
    #     uH12_eq, uH12_eq_est,
    #     title='Interfering Channel from UE2 to AN1',
    #     antenna=0)
    # f3 = plot_true_and_estimated_channel(
    #     uH13_eq, uH13_eq_est,
    #     title='Interfering Channel from UE3 to AN1',
    #     antenna=0)
    # plt.show()


if __name__ == '__main__':
    main()
