#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=C0325
"""
Script to read the result files created by the simulate_greedy_ia.py
simulator and create a table with the stream statistics.
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

import itertools
import numpy as np

from pyphysim.simulations.parameters import SimulationParameters
from pyphysim.simulations.results import SimulationResults
from pyphysim.util import misc


def get_result_from_file():
    """
    Load the SimulationResults object from the file and return it.
    """
    config_file = 'greedy_config_file.txt'

    # xxxxxxxxxx Config spec for the config file xxxxxxxxxxxxxxxxxxxxxxxxxx
    spec = """[Grid]
        cell_radius=float(min=0.01, default=1.0)
        num_cells=integer(min=3,default=3)
        num_clusters=integer(min=1,default=1)
        [Scenario]
        NSymbs=integer(min=10, max=1000000, default=200)
        SNR=real_numpy_array(min=-50, max=100, default=0:5:31)
        M=integer(min=4, max=512, default=4)
        modulator=option('QPSK', 'PSK', 'QAM', 'BPSK', default="PSK")
        Nr=integer_scalar_or_integer_numpy_array_check(min=2,default=3)
        Nt=integer_scalar_or_integer_numpy_array_check(min=2,default=3)
        Ns=integer_scalar_or_integer_numpy_array_check(min=1,default=3)
        N0=float(default=-116.4)
        scenario=string_list(default=list('Random', 'NoPathLoss'))
        [IA Algorithm]
        max_iterations=integer(min=1, default=120)
        initialize_with=string_list(default=list('random'))
        stream_sel_method=string_list(default=list('greedy', 'brute'))
        [General]
        rep_max=integer(min=1, default=2000)
        max_bit_errors=integer(min=1, default=3000)
        unpacked_parameters=string_list(default=list('SNR','stream_sel_method','scenario','initialize_with'))
        """.split("\n")

    params = SimulationParameters.load_from_config_file(config_file, spec)

    # xxxxx Results base name xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    base_name = ("IA_stream_sel_results_{SNR}_{M}-{modulator}_{Nr}x{Nt}_({Ns})"
                 "_MaxIter_{max_iterations}_({initialize_with})")
    base_name = misc.replace_dict_values(base_name, params.parameters, True)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Get the SimulationResults objects xxxxxxxxxxxxxxxxxxxxxxxx
    results = SimulationResults.load_from_file(
        'greedy_{0}.pickle'.format(base_name))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    return results


def get_pretty_statistic_table(statistics, Ns=None, multiply=100):
    """
    Get a pretty table with the statistics for each stream.

    Parameters
    ----------
    statistics : 1D numpy array
        The array with the statistics of each combination.
    Ns : 1D numpy array
        The maximum number of streams for each user. If it is None it
        defaults to [3, 3, 3].
    multiply : int (default = 100)
        Value to multiply by each statistic value. If the values in
        `statistic_matrix` are average over N, set `multiply` to N to use
        the absolute values instead of the percentage.
    """
    if Ns is None:
        Ns = [3, 3, 3]

    my_range = lambda x: range(1, x + 1)

    all_ranges = map(my_range, Ns)
    all_combinations = itertools.product(*all_ranges)

    statistic_table = zip(all_combinations, np.round(statistics * multiply, 2))

    return statistic_table


def print_statistics_table(statistic_table):
    """
    Print the statistics table in a pretty way.

    Parameters
    ----------
    statistic_table : TYPE
    """
    # Remove rows in statistic_table which have a zero percentage
    filtered_table = [(a, b) for (a, b) in statistic_table if b != 0]
    for line in filtered_table:
        print('{0}: {1}%'.format(line[0], line[1]))


if __name__ == '__main__':
    results = get_result_from_file()

    greedy_nopl_stream_statistics = results.get_result_values_list(
        'stream_statistics',
        fixed_params={
            'initialize_with': 'random',
            'stream_sel_method': 'greedy',
            'scenario': 'NoPathLoss'
        })
    greedy_withpl_stream_statistics = results.get_result_values_list(
        'stream_statistics',
        fixed_params={
            'initialize_with': 'random',
            'stream_sel_method': 'greedy',
            'scenario': 'Random'
        })
    brute_force_nopl_stream_statistics = results.get_result_values_list(
        'stream_statistics',
        fixed_params={
            'initialize_with': 'random',
            'stream_sel_method': 'brute',
            'scenario': 'NoPathLoss'
        })
    brute_force_withpl_stream_statistics = results.get_result_values_list(
        'stream_statistics',
        fixed_params={
            'initialize_with': 'random',
            'stream_sel_method': 'brute',
            'scenario': 'Random'
        })

    SNR = results.params['SNR']

    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("xxxxxxxxxxxxxxx Greedy Results xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    for idx, snr in enumerate(SNR):
        print("SNR: {0}".format(snr))
        statistic_table = get_pretty_statistic_table(
            greedy_withpl_stream_statistics[idx])
        print_statistics_table(statistic_table)
        print()
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("xxxxxxxxxxxxxxx Brute Force Results xxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    for idx, snr in enumerate(SNR):
        print("SNR: {0}".format(snr))
        statistic_table = get_pretty_statistic_table(
            brute_force_withpl_stream_statistics[idx])
        print_statistics_table(statistic_table)
