#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""module docstring"""

from __future__ import division

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import numpy as np

from ia import ia
from comm import modulators
from util.conversion import dB2Linear
from util import misc


if __name__ == '__main__':
    SNR = 50.0
    M = 16
    NSymbs = 200
    modulator = modulators.PSK(M)
    K = 3
    Nr = np.ones(K, dtype=int) * 2
    Nt = np.ones(K, dtype=int) * 2
    Ns = np.ones(K, dtype=int) * 1
    #ia_solver = ia.AlternatingMinIASolver()
    ia_solver = ia.MaxSinrIASolver()
    #ia_solver = ia.MinLeakageIASolver()
    ia_solver.max_iterations = 300

    # xxxxx Input Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # inputData has the data of all users (vertically stacked)
    inputData = np.random.randint(0, M, [np.sum(Ns), NSymbs])
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Modulate input data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # modulatedData has the data of all users (vertically stacked)
    modulatedData = modulator.modulate(inputData)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Perform the Interference Alignment xxxxxxxxxxxxxxxxxxxxxxxx
    cumNs = np.cumsum(Ns)
    # Split the data. transmit_signal will be a list and each element
    # is a numpy array with the data of a user
    transmit_signal = np.split(modulatedData, cumNs[:-1])

    ia_solver.randomizeH(Nr, Nt, K)
    ia_solver.randomizeF(Nt, Ns, K)

    ia_solver.solve()

    transmit_signal_precoded = map(np.dot, ia_solver._F, transmit_signal)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    noise_var = 1 / dB2Linear(SNR)
    multi_user_channel = ia_solver._multiUserChannel
    # received_data is an array of matrices, one matrix for each receiver.
    received_data = multi_user_channel.corrupt_data(
        transmit_signal_precoded, noise_var)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Perform the Interference Cancelation xxxxxxxxxxxxxxxxxxxxxx
    dot2 = lambda w, r: np.dot(w.transpose().conjugate(), r)
    # This will cancel the interference
    received_data_no_interference = map(dot2,
                                        ia_solver._W, received_data)

    # We still need to compensate the combined effect of the precoding and
    # IA receive filter
    compensate_filters = [np.linalg.inv(ia_solver.calc_equivalent_channel(k)) for k in range(K)]
    received_data_no_interference2 = map(np.dot,
                                         compensate_filters, received_data_no_interference)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Demodulate Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    received_data_no_interference = np.vstack(received_data_no_interference2)
    demodulated_data = modulator.demodulate(received_data_no_interference)
    # demodulated_data = map(modulator.demodulate, received_data_no_interference)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Debug xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # print "IA Cost: {0:f}".format(ia_solver.getCost())
    # print inputData - demodulated_data
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Calculates the symbol and bit error rates xxxxxxxxxxxxxxxxx
    symbolErrors = np.sum(inputData != demodulated_data)
    bitErrors = misc.count_bit_errors(inputData, demodulated_data)
    numSymbols = inputData.size
    numBits = inputData.size * modulators.level2bits(M)
    #ia_cost = ia_solver.getCost()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    print bitErrors
    print numBits
    BER = bitErrors / numBits
    print "BER: {0}".format(BER)

    # SINR_0 = ia_solver.calc_SINR_k(0)
    # SINR_1 = ia_solver.calc_SINR_k(1)
    # SINR_2 = ia_solver.calc_SINR_k(2)

    # print "SINR_0: {0}".format(SINR_0)
    # print "SINR_1: {0}".format(SINR_1)
    # print "SINR_2: {0}".format(SINR_2)

    # xxxxxxxxxx Debug info xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    H00 = ia_solver.get_channel(0, 0)
    H01 = ia_solver.get_channel(0, 1)
    H02 = ia_solver.get_channel(0, 2)
    H10 = ia_solver.get_channel(1, 0)
    H11 = ia_solver.get_channel(1, 1)
    H12 = ia_solver.get_channel(1, 2)
    H20 = ia_solver.get_channel(2, 0)
    H21 = ia_solver.get_channel(2, 1)
    H22 = ia_solver.get_channel(2, 2)

    F0 = ia_solver.F[0]
    F1 = ia_solver.F[1]
    F2 = ia_solver.F[2]

    W0 = ia_solver.W[0]
    W1 = ia_solver.W[1]
    W2 = ia_solver.W[2]
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
