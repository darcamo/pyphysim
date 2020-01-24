#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""module docstring"""

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
from pyphysim.util.conversion import dB2Linear
from pyphysim.channels.multiuser import MultiUserChannelMatrix
from pyphysim.modulators.fundamental import PSK

if __name__ == '__main__':
    K = 3
    Nr = np.ones(K) * 4
    Nt = np.ones(K) * 4
    Ns = np.array([2, 2, 2])  # np.ones(K) * 2

    multiuserchannel = MultiUserChannelMatrix()
    modulator = PSK(4)

    SNR = 40
    noise_var = 1 / dB2Linear(SNR)
    print("SNR: {0}".format(SNR))
    print("noise_var: {0}".format(noise_var))

    multiuserchannel.randomize(Nr, Nt, K)
    multiuserchannel.noise_var = noise_var

    ia_solver = algorithms.AlternatingMinIASolver(multiuserchannel)
    ia_solver2 = algorithms.MMSEIASolver(multiuserchannel)
    ia_solver3 = algorithms.MaxSinrIASolver(multiuserchannel)

    ia_solver4 = algorithms.AlternatingMinIASolver(multiuserchannel)

    # ia_solver.initialize_with_closed_form = True
    # ia_solver2.initialize_with_closed_form = True
    # ia_solver3.initialize_with_closed_form = True

    ia_solver.randomizeF(Ns)
    ia_solver.max_iterations = 400
    ia_solver.solve(Ns)

    ia_solver2.randomizeF(Ns)
    ia_solver2.max_iterations = 100
    ia_solver2.solve(Ns)

    ia_solver3.randomizeF(Ns)
    ia_solver3.max_iterations = 100
    ia_solver3.solve(Ns)

    ia_solver4.randomizeF(Ns)
    ia_solver4.max_iterations = 100
    ia_solver4.solve(Ns)

    print("Final_Cost: {0}\n".format(ia_solver.get_cost()))

    # all_possibilities = itertools.product(range(K), range(K))
    # for ij in all_possibilities:
    #     i, j = ij
    #     print "Hij: H{0}{1}".format(i, j)
    #     Hij = multiuserchannel.get_Hkl(i, j)
    #     Hij_eff = np.dot(
    #         ia_solver.full_W_H[i], np.dot(Hij, ia_solver.full_F[j]))
    #     print "Eigenvalues: {0}".format(np.linalg.svd(Hij_eff)[1].round(6)[0])
    #     print "Eigenvector: {0}".format(
    #         np.linalg.svd(Hij_eff)[0].round(6)[0][0])
    #     print

    print("Sum Capacity (Alt Min): {0}".format(
        np.sum(np.log2(np.hstack(1.0 + ia_solver.calc_SINR())))))
    print("Sum Capacity (Alt Min): {0}".format(
        np.sum(np.log2(np.hstack(1.0 + ia_solver4.calc_SINR())))))
    print("Sum Capacity (MMSE): {0}".format(
        np.sum(np.log2(np.hstack(1.0 + ia_solver2.calc_SINR())))))
    print("Sum Capacity (Max SINR): {0}".format(
        np.sum(np.log2(np.hstack(1.0 + ia_solver3.calc_SINR())))))

    # print linear2dB(np.hstack(ia_solver.calc_SINR()))
    # print linear2dB(np.hstack(ia_solver2.calc_SINR()))
    # print linear2dB(np.hstack(ia_solver3.calc_SINR()))
