#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""module docstring"""

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
from time import time
import numpy as np
from pprint import pprint

from pyphysim.simulations.runner import SimulationRunner
from pyphysim.simulations.parameters import SimulationParameters
from pyphysim.simulations.results import SimulationResults, Result
from pyphysim.simulations.simulationhelpers import simulate_do_what_i_mean, get_common_parser
from pyphysim.comm import modulators, channels
from pyphysim.util.conversion import dB2Linear, linear2dB
from pyphysim.util import misc
from pyphysim.ia import algorithms
from pyphysim.simulations.progressbar import ProgressbarText
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

#calc_capacity = lambda sirn: np.sum(np.log2(1 + sirn))


def calc_capacity(sinr):
    """Calculate the Sum capacity.

    Parameters
    ----------
    sinr : 1D numpy array of 1D arrays.
        The sinr (in linear scale) of all streams of all users.

    Returns
    -------
    capacity : float
        The capacity of each user.
    """

    capacity = np.array(
        [np.sum(np.log2(1 + user_sinrs)) for user_sinrs in sinr])
    return capacity


def main():
    """Main function.
    """
    K = 3
    Nr = 4
    Nt = 4
    Ns = 2
    SNR = -30.0
    P = 1.0

    # Dependent parameters
    noise_var = 1 / dB2Linear(SNR)

    RepMax = 100
    mmse_sinrs = np.empty([RepMax, K, Ns], dtype=float)
    max_sinr_sinrs = np.empty([RepMax, K, Ns], dtype=float)
    mmse_capacity = np.empty(RepMax, dtype=float)
    max_sinr_capacity = np.empty(RepMax, dtype=float)

    pbar = ProgressbarText(RepMax, message="Simulating for SNR: {0}".format(SNR))

    for rep in range(RepMax):
        # Creat the channel
        multiUserChannel = channels.MultiUserChannelMatrix()
        multiUserChannel.randomize(Nr, Nt, K)
        multiUserChannel.noise_var = noise_var

        # Creat the IA solver object
        mmse_ia_solver = algorithms.MMSEIASolver(multiUserChannel)
        max_sinr_ia_solver = algorithms.MaxSinrIASolver(multiUserChannel)

        mmse_ia_solver.randomizeF(Ns, P)

        mmse_ia_solver.initialize_with = 'fix'
        max_sinr_ia_solver.initialize_with = 'fix'
        max_sinr_ia_solver._F = mmse_ia_solver._F

        #mmse_ia_solver.initialize_with = 'fix'

        # We wouldn't need to explicitly set ia_solver.noise_var
        # variable if the multiUserChannel object had the correct value at
        # this point.
        mmse_ia_solver.noise_var = noise_var
        mmse_ia_solver.max_iterations = 200
        mmse_ia_solver.solve(Ns)

        max_sinr_ia_solver.noise_var = noise_var
        max_sinr_ia_solver.max_iterations = 200

        max_sinr_ia_solver.solve(Ns)

        import pudb; pudb.set_trace()  ## DEBUG ##
        mmse_sinrs[rep] = map(linear2dB, mmse_ia_solver.calc_SINR())
        max_sinr_sinrs[rep] = map(linear2dB, max_sinr_ia_solver.calc_SINR())

        mmse_capacity[rep] = np.sum(calc_capacity(mmse_ia_solver.calc_SINR()))
        max_sinr_capacity[rep] = np.sum(calc_capacity(max_sinr_ia_solver.calc_SINR()))

        # print "MMSE Alt. SINRs:\n{0}".format(np.vstack(mmse_sinrs[rep]))
        # print "Max SINR Alg. SINRs:\n{0}".format(np.vstack(max_sinr_sinrs[rep]))

        # print "MMSE Alt. Capacity: {0}".format(np.sum(calc_capacity(mmse_sinrs[rep])))
        # print "Max SINR Alg. Capacity: {0}".format(np.sum(calc_capacity(max_sinr_sinrs[rep])))
        # print

        pbar.progress(rep)

    print "MMSE Average SINRs:\n{0}".format(mmse_sinrs.mean(0))
    print "Max SINR Average SINRs:\n{0}".format(max_sinr_sinrs.mean(0))
    print "MMSE Average Capacity: {0}".format(mmse_capacity.mean())
    print "Max SINR Average Capacity: {0}".format(max_sinr_capacity.mean())

    print "\nEnd!"

    import pudb; pudb.set_trace()  ## DEBUG ##


if __name__ == '__main__':
    main()
