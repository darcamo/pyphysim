#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('../')
# from matplotlib import pyplot as plt

import numpy as np
from matplotlib import pyplot as plt
from pyphysim.srs.zadoffchu import calcBaseZC, getShiftedZF, get_extended_ZF
from pyphysim.channels.fading import COST259_TUx, TdlChannel
from pyphysim.channels.fading_generators import JakesSampleGenerator

if __name__ == '__main__':
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
    speedTerminal = 3 / 3.6  # Speed in m/s
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
    # We arbitrarely choose some cyclic shift index and then we call
    # zadoffchu.getShiftedZF to get the shifted sequence.
    shift_index = 4
    r1 = getShiftedZF(a_u1, shift_index)
    r2 = getShiftedZF(a_u2, shift_index)
    r3 = getShiftedZF(a_u3, shift_index)

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Generate channels from users to the BS xxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    jakes_all_links = np.empty([3, 3], dtype=object)
    tdlchannels_all_links = np.empty([3, 3], dtype=object)
    fading_maps = np.empty([3, 3], dtype=object)
    # Dimension: `UEs x ANs x num_subcarriers x numUeAnt x numAnAnt`
    freq_responses = np.empty([3, 3, Nsc, numUeAnt, numAnAnt], dtype=complex)

    for ueIdx in range(3):
        for anIdx in range(3):
            jakes_all_links[ueIdx, anIdx] = JakesSampleGenerator(
                Fd, Ts, L, shape=(numUeAnt, numAnAnt))

            tdlchannels_all_links[ueIdx, anIdx] = TdlChannel(
                jakes_all_links[ueIdx, anIdx],
                COST259_TUx.tap_powers_dB,
                COST259_TUx.tap_delays)

            fading_maps[ueIdx, anIdx] \
                = tdlchannels_all_links[ueIdx, anIdx].get_fading_map(
                num_samples)

            freq_responses[ueIdx, anIdx] = tdlchannels_all_links[
                                               ueIdx, anIdx].get_channel_freq_response(
                fading_maps[ueIdx, anIdx], Nsc)[:, :, :, 0]

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
    [dU11, dS11, dV11_H] = np.linalg.svd(dH11[sc_idx])
    [dU22, dS22, dV22_H] = np.linalg.svd(dH22[sc_idx])
    [dU33, dS33, dV33_H] = np.linalg.svd(dH33[sc_idx])

    # xxxxxxxxxx Users precoders xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Users' precoders are the main column of the U matrix
    F11 = dU11[:, 0].conj()
    F22 = dU22[:, 0].conj()
    F33 = dU33[:, 0].conj()

    # xxxxxxxxxx Channels in uplink direction xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Dimension: `Nsc x numAnAnt x numUeAnt`
    uH11 = np.transpose(dH11, axes=[0, 2, 1])
    uH12 = np.transpose(dH12, axes=[0, 2, 1])
    uH13 = np.transpose(dH13, axes=[0, 2, 1])
    uH21 = np.transpose(dH21, axes=[0, 2, 1])
    uH22 = np.transpose(dH22, axes=[0, 2, 1])
    uH23 = np.transpose(dH23, axes=[0, 2, 1])
    uH31 = np.transpose(dH31, axes=[0, 2, 1])
    uH32 = np.transpose(dH32, axes=[0, 2, 1])
    uH33 = np.transpose(dH33, axes=[0, 2, 1])

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
    # xxxxxxxxxxxxxxx Plot the equivalent channel xxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Calculate the received signals
    comb_indexes = np.r_[0:Nsc:2]
    Y1 = uH11_eq[comb_indexes] * r1[:, np.newaxis] + uH12_eq[comb_indexes] * r2[
                                                                             :,
                                                                             np.newaxis] + \
         uH13_eq[comb_indexes] * r3[:, np.newaxis]
    Y2 = uH21_eq[comb_indexes] * r1[:, np.newaxis] + uH22_eq[comb_indexes] * r2[
                                                                             :,
                                                                             np.newaxis] + \
         uH23_eq[comb_indexes] * r3[:, np.newaxis]
    Y3 = uH31_eq[comb_indexes] * r1[:, np.newaxis] + uH32_eq[comb_indexes] * r2[
                                                                             :,
                                                                             np.newaxis] + \
         uH33_eq[comb_indexes] * r3[:, np.newaxis]


    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Estimate the equivalent channel xxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    tilde_y1 = np.fft.ifft(Y1 * r1[:, np.newaxis].conj(), n=Nsc / 2, axis=0)
    tilde_y1[15:, 0] = 0  # Only keep the first 15 time samples for each antenna

    plt.figure(figsize=(12, 14))
    plt.subplot(2, 1, 1)
    plt.stem(np.abs(tilde_y1[:, 0]))

    plt.subplot(2, 1, 2)
    plt.plot(np.abs(np.fft.fft(tilde_y1[:, 0], n=Nsc, axis=0)))
    plt.show()
