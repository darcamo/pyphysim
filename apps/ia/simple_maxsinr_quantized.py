#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""module docstring"""

from __future__ import division

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
try:
    parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    grandparent_dir = os.path.split(parent_dir)[0]
    sys.path.append(grandparent_dir)
except NameError:
    sys.path.append('../../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import numpy as np

from pyphysim.ia import algorithms
from pyphysim.modulators import fundamental
from pyphysim.util.conversion import dB2Linear
from pyphysim.util import misc
from pyphysim.simulations.progressbar import ProgressbarText2
import pyphysim.channels.multiuser


# noinspection PyShadowingNames
def gen_codebook(codebook_size, dimension):
    """
    Generate a new codebook.

    Parameters
    ----------
    codebook_size : int
        The number of code words in the codebook.
    dimension : int
        The dimension of each precoder.

    Returns
    -------
    codebook : np.ndarray
        The generated codebook (2D numpy array).
    """
    codebook = np.empty([codebook_size, dimension], dtype=complex)
    for i in range(codebook_size):
        H = misc.randn_c(dimension, 1)
        [U, _, _] = np.linalg.svd(H)
        codebook[i] = U[:, 0]

    return codebook


def calc_dist(vec, codeword):
    """
    Calculate the distance between `vec` and `codeword`.

    First `vec` is normalized, since `codeword` has a unitary norm. Then
    the distance will be the square of the euclidean distance between
    normalized `vec` and `codeword`.

    Parameters
    ----------
    vec : np.ndarray
        The matrix to be quantized. This is a 1D numpy array.
    codeword : np.ndarray
        One codeword from the codebook. The norm of the codeword should be
        equal to 1. This is a 1D numpy array.

    Returns
    -------
    dist : float
        The distance between `vec` and `codeword`. This is basically the
        square of the sine between the two input vectors.
    """
    H = vec / np.linalg.norm(vec)
    return np.linalg.norm(H - codeword)**2


def calc_angle_dist(vec, codeword):
    """
    Calculate the distance (angle based) between `vec` and `codeword`.

    Parameters
    ----------
    vec : np.ndarray
        The matrix to be quantized (1D numpy array).
    codeword : np.ndarray
        One codeword from the codebook (1D numpy array).

    Returns
    -------
    dist : float
        The distance between `vec` and `codeword`. This is basically the
        square of the sine between the two input vectors.
    """
    cos_sq = (np.abs(vec.conj().dot(codeword) /
                     (np.linalg.norm(vec) * np.linalg.norm(codeword))))**2
    sin_sq = 1 - cos_sq

    return sin_sq


def quant_small_matrix(small_matrix, codebook):
    """
    Quantize the channel `small_matrix` of a single link.

    Parameters
    ----------
    small_matrix : np.ndarray
        The matrix to be quantized (2D numpy array).
    codebook : np.ndarray
        The codebook (2D numpy array), where each row corresponds to a codeword.

    Returns
    -------
    quantized_small_matrix
    """
    small_matrix_vec = small_matrix.reshape(-1)
    dists = [calc_dist(small_matrix_vec, c) for c in codebook]
    index = np.argmin(dists)

    quantized_small_matrix = codebook[index].reshape(small_matrix.shape)
    return quantized_small_matrix


def my_quant_func(true_matrix, Nr, Nt, K, codebook):
    """
    Quantize the channels of the multiple users in `true_matrix`.

    Parameters
    ----------
    true_matrix : np.ndarray
        The channel matrix correspondint to the channel from all
        transmitters to all receivers. This is a 2D numpy array.
    Nr : int | np.ndarray
        The number of receive antennas of each user.
    Nt : int | np.ndarray
        The number of transmit antennas of each user.
    K : int
        The number of users
    codebook : np.ndarray
        The codebook (generated with the `gen_codebook` function). This is a
        2D numpy array.

    Returns
    -------
    quantize_channel : np.ndarray
        The quantized channel (2D numpy array).
    """
    quantize_channel = np.empty(true_matrix.shape, dtype=complex)

    Nrvec = np.ones(K, dtype=int) * Nr
    Ntvec = np.ones(K, dtype=int) * Nt

    Nridx = np.hstack([0, np.cumsum(Nrvec)])
    Ntidx = np.hstack([0, np.cumsum(Ntvec)])

    for k1 in range(K):
        for k2 in range(K):
            rx_start, rx_end = Nridx[k1], Nridx[k1 + 1]
            tx_start, tx_end = Ntidx[k2], Ntidx[k2 + 1]

            quantize_channel[rx_start:rx_end, tx_start:tx_end] \
                = quant_small_matrix(
                    true_matrix[rx_start:rx_end, tx_start:tx_end], codebook)

    return quantize_channel


if __name__ == '__main__1':
    codebook_size = 5
    dimension = 4
    C = gen_codebook(codebook_size, dimension)

    Nt = 2
    Nr = 2
    K = 3

    H = misc.randn_c(Nr * K, Nt * K)

    Hquant = my_quant_func(H, Nr, Nt, K, C)

if __name__ == '__main__':
    SNR = 15.0
    noise_var = 1 / dB2Linear(SNR)
    M = 2
    NSymbs = 50
    rep_max = 300
    modulator = fundamental.BPSK()
    K = 3
    Nr = np.ones(K, dtype=int) * 2
    Nt = np.ones(K, dtype=int) * 2
    Ns = np.ones(K, dtype=int) * 1
    multi_user_channel = pyphysim.channels.multiuser.MultiUserChannelMatrix()
    multi_user_channel_quant = pyphysim.channels.multiuser.MultiUserChannelMatrix(
    )
    multi_user_channel.noise_var = noise_var
    multi_user_channel_quant.noise_var = noise_var

    ia_solver = algorithms.MaxSinrIASolver(multi_user_channel_quant)
    ia_solver.max_iterations = 120

    pb = ProgressbarText2(rep_max,
                          '*',
                          message="SNR {0} - {{elapsed_time}}".format(SNR))

    symbolErrors = 0
    bitErrors = 0
    numSymbols = 0
    numBits = 0

    # xxxxxxxxxx Quantization parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    codebook_size = 512
    dimension = 4
    codebook = gen_codebook(codebook_size, dimension)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    for rep in range(rep_max):
        pb.progress(rep)
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

        #multi_user_channel.randomize(Nr, Nt, K)
        big_matrix = misc.randn_c(np.sum(Nr), np.sum(Nt))
        big_matrix_quant = my_quant_func(big_matrix, Nr, Nt, K, codebook)

        multi_user_channel.init_from_channel_matrix(big_matrix, Nr, Nt, K)
        multi_user_channel_quant.init_from_channel_matrix(
            big_matrix_quant, Nr, Nt, K)

        #ia_solver.randomizeF(Ns)
        ia_solver.solve(Ns)

        transmit_signal_precoded = map(np.dot, ia_solver.F, transmit_signal)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Pass through the channel xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # received_data is an array of matrices, one matrix for each receiver.
        received_data = multi_user_channel.corrupt_data(
            transmit_signal_precoded)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Perform the Interference Cancellation xxxxxxxxxxxxxxxxxxxxx
        # dot2 = lambda w, r: np.dot(w.transpose().conjugate(), r)
        # This will cancel the interference
        received_data_no_interference = map(np.dot, ia_solver.W_H,
                                            received_data)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Demodulate Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        received_data_no_interference = np.vstack(received_data_no_interference)
        # received_data_no_interference = np.vstack(received_data_no_interference2)
        demodulated_data = modulator.demodulate(received_data_no_interference)
        # demodulated_data = map(modulator.demodulate, received_data_no_interference)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Calculates the symbol and bit error rates xxxxxxxxxxxxxxxxx
        symbolErrors = symbolErrors + np.sum(inputData != demodulated_data)
        bitErrors += misc.count_bit_errors(inputData, demodulated_data)
        numSymbols = numSymbols + inputData.size
        numBits += inputData.size * fundamental.level2bits(M)
        #ia_cost = ia_solver.get_cost()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    print()
    print(bitErrors)
    print(numBits)
    BER = bitErrors / numBits
    print("BER: {0}".format(BER))

    SINRs = multi_user_channel.calc_SINR(ia_solver.F, ia_solver.W)
    sum_capacity = np.sum(np.log2(1 + np.hstack(SINRs)))

    print("Sum Capacity: {0}".format(sum_capacity))

    # # SINR_0 = ia_solver._calc_SINR_k(0)
    # # SINR_1 = ia_solver._calc_SINR_k(1)
    # # SINR_2 = ia_solver._calc_SINR_k(2)

    # # print("SINR_0: {0}".format(SINR_0))
    # # print("SINR_1: {0}".format(SINR_1))
    # # print("SINR_2: {0}".format(SINR_2))

    # # xxxxxxxxxx Debug info xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # H00 = multi_user_channel.get_Hkl(0, 0)
    # H01 = multi_user_channel.get_Hkl(0, 1)
    # H02 = multi_user_channel.get_Hkl(0, 2)
    # H10 = multi_user_channel.get_Hkl(1, 0)
    # H11 = multi_user_channel.get_Hkl(1, 1)
    # H12 = multi_user_channel.get_Hkl(1, 2)
    # H20 = multi_user_channel.get_Hkl(2, 0)
    # H21 = multi_user_channel.get_Hkl(2, 1)
    # H22 = multi_user_channel.get_Hkl(2, 2)

    # F0 = ia_solver.F[0]
    # F1 = ia_solver.F[1]
    # F2 = ia_solver.F[2]

    # W0 = ia_solver.W[0]
    # W1 = ia_solver.W[1]
    # W2 = ia_solver.W[2]
    # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
