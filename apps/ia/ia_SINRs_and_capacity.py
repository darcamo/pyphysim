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

try:
    import cPickle as pickle
except ImportError as e:  # pragma: no cover
    import pickle

from matplotlib import pyplot as plt

import numpy as np

from pandas import DataFrame

from pyphysim.ia.algorithms import AlternatingMinIASolver, MaxSinrIASolver, \
    MMSEIASolver
from pyphysim.util.conversion import dB2Linear
from pyphysim.simulations.progressbar import ProgressbarText
import pyphysim.channels.multiuser


def calc_SINRs_and_capacity(solver):
    """
    Calculates the SINRs.

    Parameters
    ----------
    solver : T < IASolverBaseClass
        The IA solver.
    """
    SINRs = solver.calc_SINR_in_dB()
    sinrs = solver.calc_SINR()

    calc_capacity = lambda sirn: np.sum(np.log2(1 + sirn))

    capacity = np.array(list(map(calc_capacity, sinrs)))
    sum_capacity = np.sum(capacity)

    return SINRs, capacity, sum_capacity


if __name__ == '__main__':
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    K = 3
    Nr = 4
    Nt = 4
    Ns = 2
    SNR = 5
    max_iterations = 2000
    P = 1.0
    initialize_with = 'alt_min'
    # ---------------------------------------------------------------------
    noise_var = 1. / dB2Linear(SNR)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    multiuserchannel = pyphysim.channels.multiuser.MultiUserChannelMatrix()
    multiuserchannel.randomize(Nr, Nt, K)
    multiuserchannel.noise_var = noise_var

    alt_min_solver = AlternatingMinIASolver(multiuserchannel)
    alt_min_solver.max_iterations = max_iterations
    # alt_min_solver.noise_var = noise_var

    max_sinr_solver = MaxSinrIASolver(multiuserchannel)
    max_sinr_solver.max_iterations = max_iterations
    # max_sinr_solver.noise_var = noise_var
    max_sinr_solver.initialize_with = 'alt_min'

    mmse_solver = MMSEIASolver(multiuserchannel)
    mmse_solver.max_iterations = max_iterations
    # mmse_solver.noise_var = noise_var
    mmse_solver.initialize_with = 'alt_min'
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    calc_capacity = lambda sirn: np.sum(np.log2(1 + sirn))

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    rep_max = 100

    alt_min_SINRs = np.empty(rep_max, dtype=np.ndarray)
    max_sinr_SINRs = np.empty(rep_max, dtype=np.ndarray)
    mmse_SINRs = np.empty(rep_max, dtype=np.ndarray)
    alt_min_capacity = np.empty(rep_max, dtype=np.ndarray)
    max_sinr_capacity = np.empty(rep_max, dtype=np.ndarray)
    mmse_capacity = np.empty(rep_max, dtype=np.ndarray)
    alt_min_sum_capacity = np.empty(rep_max, dtype=float)
    max_sinr_sum_capacity = np.empty(rep_max, dtype=float)
    mmse_sum_capacity = np.empty(rep_max, dtype=float)

    alt_min_runned_iterations = np.empty(rep_max, dtype=int)
    max_sinr_runned_iterations = np.empty(rep_max, dtype=int)
    mmse_runned_iterations = np.empty(rep_max, dtype=int)

    pbar = ProgressbarText(rep_max,
                           message="Simulating for SNR: {0}".format(SNR))
    for rep in range(rep_max):
        multiuserchannel.randomize(Nr, Nt, K)

        alt_min_solver.clear()
        max_sinr_solver.clear()
        mmse_solver.clear()

        alt_min_runned_iterations[rep] = alt_min_solver.solve(Ns, P)
        max_sinr_runned_iterations[rep] = max_sinr_solver.solve(Ns, P)
        mmse_runned_iterations[rep] = mmse_solver.solve(Ns, P)

        mmse_solver.calc_sum_capacity()

        # print "Alt Min"
        (alt_min_SINRs[rep], alt_min_capacity[rep],
         alt_min_sum_capacity[rep]) = calc_SINRs_and_capacity(alt_min_solver)
        # print "SINRs:\n{0}".format(alt_min_SINRs[rep])
        # print "Capacity:\n{0}".format(alt_min_capacity[rep])
        # print "Sum_Capacity: {0}".format(alt_min_sum_capacity[rep])

        # print "\nMax SINR"
        (max_sinr_SINRs[rep], max_sinr_capacity[rep],
         max_sinr_sum_capacity[rep]) = calc_SINRs_and_capacity(max_sinr_solver)
        # print "SINRs:\n{0}".format(max_sinr_SINRs[rep])
        # print "Capacity:\n{0}".format(max_sinr_capacity[rep])
        # print "Sum_Capacity: {0}".format(max_sinr_sum_capacity[rep])

        # print "\nMMSE"
        (mmse_SINRs[rep], mmse_capacity[rep],
         mmse_sum_capacity[rep]) = calc_SINRs_and_capacity(mmse_solver)
        # print "SINRs:\n{0}".format(mmse_SINRs[rep])
        # print "Capacity:\n{0}".format(mmse_capacity[rep])
        # print "Sum_Capacity: {0}".format(mmse_sum_capacity[rep])

        pbar.progress(rep)

    df = DataFrame({
        'Min. Leakage': alt_min_sum_capacity,
        'Max SINR': max_sinr_sum_capacity,
        'MMSE': mmse_sum_capacity
    })
    df.to_csv(
        'sum_capacity_{Nr}x{Nt}_{Ns}_SNR_{SNR}_{initialize_with}_init.txt'.
        format(Nr=Nr, Ns=Ns, Nt=Nt, SNR=SNR, initialize_with=initialize_with),
        index_label="Index")

    plt.plot([sum(alt_min_capacity[a]) for a in range(50)])
    plt.plot([sum(max_sinr_capacity[a]) for a in range(50)])
    plt.plot([sum(mmse_capacity[a]) for a in range(50)])
    plt.legend(["Min Leakage", "Max SINR", "MMSE"])
