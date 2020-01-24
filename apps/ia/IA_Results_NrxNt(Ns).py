#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to generate the plots."""

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

# xxxxxxxxxx Import Statements xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
from pyphysim.simulations.results import SimulationResults
import numpy as np
from pyphysim.extra.pgfplotshelper import *
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# In Ipython run this script with
#     run -i IA_Results_2x2(1).py
try:
    # noinspection PyUnboundLocalVariable,PyUnresolvedReferences
    initialized
except NameError as e:
    print("This line should not be executed")
    initialized = False
    max_iterations = "{0}".format(5)


# xxxxxxxxxx Function Definitions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def get_ber_for_given_num_iter(result_obj, max_iterations):
    """Docstring

    Parameters
    ----------
    result_obj : SimulationResults
        TRhe SimulationResults object.
    max_iterations : int
        The maximum number of iterations.

    Returns
    -------
    list
        The bit error rate (BER).
    """
    ber = result_obj.get_result_values_list(
        'ber', fixed_params={'max_iterations': max_iterations})
    return ber


def get_sum_capacity_for_given_num_iter(result_obj, max_iterations):
    """Docstring

    Parameters
    ----------
    result_obj : SimulationResults
        TRhe SimulationResults object.
    max_iterations : int
        The maximum number of iterations.

    Returns
    -------
    float
        The Sum Capacity.
    """
    sum_capacity = result_obj.get_result_values_list(
        'sum_capacity', fixed_params={'max_iterations': max_iterations})
    return sum_capacity


def get_mean_iterations(result_obj, max_iterations):
    """Docstring

    Parameters
    ----------
    result_obj : SimulationResults
        TRhe SimulationResults object.
    max_iterations : int
        The maximum number of iterations.

    Returns
    -------

    """
    mean_ia_terations = get_num_mean_ia_iterations(
        result_obj, {'max_iterations': max_iterations})
    return mean_ia_terations


def get_num_runned_reps(sim_results_object, fixed_params=None):
    """Docstring

    Parameters
    ----------
    sim_results_object : SimulationResults
        The SimulationResults object.
    fixed_params : dict
        Dictionary specifying the fixed values.

    Returns
    -------
    np.ndarray
        The number of runned repetitions."""
    if fixed_params is None:
        fixed_params = {}

    all_runned_reps = np.array(sim_results_object.runned_reps)
    indexes = sim_results_object.params.get_pack_indexes(fixed_params)
    return all_runned_reps[indexes]


def get_num_mean_ia_iterations(sim_results_object, fixed_params=None):
    """Docstring

    Parameters
    ----------
    sim_results_object : SimulationResults
        The SimulationResults object.
    fixed_params : dict
        Dictionary specifying the fixed values.

    Returns
    -------
    List
        A list with the ie runned iterations.
    """
    if fixed_params is None:
        fixed_params = {}
    return sim_results_object.get_result_values_list('ia_runned_iterations',
                                                     fixed_params)


if __name__ == '__main__':
    # xxxxxxxxxx Initializations xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    if initialized is False:
        print("Reading result files")
        # xxxxx Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # params = SimulationParameters.load_from_config_file(
        #     'ia_config_file.txt')
        K = 3
        Nr = 5
        Nt = 3
        Ns = 2
        M = 4
        max_iterations = '60'
        modulator = "PSK"
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Results base name xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        base_name = ('results_{M}-{modulator}_'
                     '{Nr}x{Nt}_({Ns})_MaxIter_[5_(5)_60]').format(
                         M=M, modulator=modulator, Nr=Nr, Nt=Nt, Ns=Ns)
        # Used only for the closed form algorithm, which is not iterative
        base_name_no_iter = ('results_{M}-{modulator}_{Nr}x{Nt}_({Ns})'
                             '_MaxIter_[5_(5)_60]').format(M=M,
                                                           modulator=modulator,
                                                           Nr=Nr,
                                                           Nt=Nt,
                                                           Ns=Ns)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        alt_min_results = SimulationResults.load_from_file(
            'ia_alt_min_{0}.pickle'.format(base_name))
        closed_form_results = SimulationResults.load_from_file(
            'ia_closed_form_{0}.pickle'.format(base_name_no_iter))
        # closed_form_first_results = SimulationResults.load_from_file(
        #     'ia_closed_form_first_init_{0}.pickle'.format(base_name))
        max_sinrn_results = SimulationResults.load_from_file(
            'ia_max_sinr_{0}.pickle'.format(base_name))
        # min_leakage_results = SimulationResults.load_from_file(
        #     'ia_min_leakage_{0}.pickle'.format(base_name))
        mmse_results = SimulationResults.load_from_file(
            'ia_mmse_{0}.pickle'.format(base_name))
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        initialized = True

    # xxxxxxxxxx SNR variables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # noinspection PyUnboundLocalVariable
    SNR_alt_min = np.array(alt_min_results.params['SNR'])
    # noinspection PyUnboundLocalVariable
    SNR_closed_form = np.array(closed_form_results.params['SNR'])
    # noinspection PyUnboundLocalVariable
    SNR_max_SINR = np.array(max_sinrn_results.params['SNR'])
    # SNR_min_leakage = np.array(min_leakage_results.params['SNR'])
    # noinspection PyUnboundLocalVariable
    SNR_mmse = np.array(mmse_results.params['SNR'])

    # xxxxxxxxxx BER Variables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ber_closed_form = get_ber_for_given_num_iter(closed_form_results, 5)

    ber_alt_min = {
        "5": get_ber_for_given_num_iter(alt_min_results, 5),
        "10": get_ber_for_given_num_iter(alt_min_results, 10),
        "15": get_ber_for_given_num_iter(alt_min_results, 15),
        "20": get_ber_for_given_num_iter(alt_min_results, 20),
        "25": get_ber_for_given_num_iter(alt_min_results, 25),
        "30": get_ber_for_given_num_iter(alt_min_results, 30),
        "35": get_ber_for_given_num_iter(alt_min_results, 35),
        "40": get_ber_for_given_num_iter(alt_min_results, 40),
        "45": get_ber_for_given_num_iter(alt_min_results, 45),
        "50": get_ber_for_given_num_iter(alt_min_results, 50),
        "55": get_ber_for_given_num_iter(alt_min_results, 55),
        "60": get_ber_for_given_num_iter(alt_min_results, 60)
    }

    ber_max_sinr = {
        "5": get_ber_for_given_num_iter(max_sinrn_results, 5),
        "10": get_ber_for_given_num_iter(max_sinrn_results, 10),
        "15": get_ber_for_given_num_iter(max_sinrn_results, 15),
        "20": get_ber_for_given_num_iter(max_sinrn_results, 20),
        "25": get_ber_for_given_num_iter(max_sinrn_results, 25),
        "30": get_ber_for_given_num_iter(max_sinrn_results, 30),
        "35": get_ber_for_given_num_iter(max_sinrn_results, 35),
        "40": get_ber_for_given_num_iter(max_sinrn_results, 40),
        "45": get_ber_for_given_num_iter(max_sinrn_results, 45),
        "50": get_ber_for_given_num_iter(max_sinrn_results, 50),
        "55": get_ber_for_given_num_iter(max_sinrn_results, 55),
        "60": get_ber_for_given_num_iter(max_sinrn_results, 60)
    }

    ber_mmse = {
        "5": get_ber_for_given_num_iter(mmse_results, 5),
        "10": get_ber_for_given_num_iter(mmse_results, 10),
        "15": get_ber_for_given_num_iter(mmse_results, 15),
        "20": get_ber_for_given_num_iter(mmse_results, 20),
        "25": get_ber_for_given_num_iter(mmse_results, 25),
        "30": get_ber_for_given_num_iter(mmse_results, 30),
        "35": get_ber_for_given_num_iter(mmse_results, 35),
        "40": get_ber_for_given_num_iter(mmse_results, 40),
        "45": get_ber_for_given_num_iter(mmse_results, 45),
        "50": get_ber_for_given_num_iter(mmse_results, 50),
        "55": get_ber_for_given_num_iter(mmse_results, 55),
        "60": get_ber_for_given_num_iter(mmse_results, 60)
    }

    # xxxxxxxxxx Sum Capacity Variables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    sum_capacity_closed_form = get_sum_capacity_for_given_num_iter(
        closed_form_results, 5)

    sum_capacity_alt_min = {
        "5": get_sum_capacity_for_given_num_iter(alt_min_results, 5),
        "10": get_sum_capacity_for_given_num_iter(alt_min_results, 10),
        "15": get_sum_capacity_for_given_num_iter(alt_min_results, 15),
        "20": get_sum_capacity_for_given_num_iter(alt_min_results, 20),
        "25": get_sum_capacity_for_given_num_iter(alt_min_results, 25),
        "30": get_sum_capacity_for_given_num_iter(alt_min_results, 30),
        "35": get_sum_capacity_for_given_num_iter(alt_min_results, 35),
        "40": get_sum_capacity_for_given_num_iter(alt_min_results, 40),
        "45": get_sum_capacity_for_given_num_iter(alt_min_results, 45),
        "50": get_sum_capacity_for_given_num_iter(alt_min_results, 50),
        "55": get_sum_capacity_for_given_num_iter(alt_min_results, 55),
        "60": get_sum_capacity_for_given_num_iter(alt_min_results, 60)
    }

    sum_capacity_max_sinr = {
        "5": get_sum_capacity_for_given_num_iter(max_sinrn_results, 5),
        "10": get_sum_capacity_for_given_num_iter(max_sinrn_results, 10),
        "15": get_sum_capacity_for_given_num_iter(max_sinrn_results, 15),
        "20": get_sum_capacity_for_given_num_iter(max_sinrn_results, 20),
        "25": get_sum_capacity_for_given_num_iter(max_sinrn_results, 25),
        "30": get_sum_capacity_for_given_num_iter(max_sinrn_results, 30),
        "35": get_sum_capacity_for_given_num_iter(max_sinrn_results, 35),
        "40": get_sum_capacity_for_given_num_iter(max_sinrn_results, 40),
        "45": get_sum_capacity_for_given_num_iter(max_sinrn_results, 45),
        "50": get_sum_capacity_for_given_num_iter(max_sinrn_results, 50),
        "55": get_sum_capacity_for_given_num_iter(max_sinrn_results, 55),
        "60": get_sum_capacity_for_given_num_iter(max_sinrn_results, 60)
    }

    sum_capacity_mmse = {
        "5": get_sum_capacity_for_given_num_iter(mmse_results, 5),
        "10": get_sum_capacity_for_given_num_iter(mmse_results, 10),
        "15": get_sum_capacity_for_given_num_iter(mmse_results, 15),
        "20": get_sum_capacity_for_given_num_iter(mmse_results, 20),
        "25": get_sum_capacity_for_given_num_iter(mmse_results, 25),
        "30": get_sum_capacity_for_given_num_iter(mmse_results, 30),
        "35": get_sum_capacity_for_given_num_iter(mmse_results, 35),
        "40": get_sum_capacity_for_given_num_iter(mmse_results, 40),
        "45": get_sum_capacity_for_given_num_iter(mmse_results, 45),
        "50": get_sum_capacity_for_given_num_iter(mmse_results, 50),
        "55": get_sum_capacity_for_given_num_iter(mmse_results, 55),
        "60": get_sum_capacity_for_given_num_iter(mmse_results, 60)
    }

    # xxxxxxxxxx Mean IA Iteration Variables xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    mean_ia_iterations_alt_min = {
        "5": get_mean_iterations(alt_min_results, 5),
        "10": get_mean_iterations(alt_min_results, 10),
        "15": get_mean_iterations(alt_min_results, 15),
        "20": get_mean_iterations(alt_min_results, 20),
        "25": get_mean_iterations(alt_min_results, 25),
        "30": get_mean_iterations(alt_min_results, 30),
        "35": get_mean_iterations(alt_min_results, 35),
        "40": get_mean_iterations(alt_min_results, 40),
        "45": get_mean_iterations(alt_min_results, 45),
        "50": get_mean_iterations(alt_min_results, 50),
        "55": get_mean_iterations(alt_min_results, 55),
        "60": get_mean_iterations(alt_min_results, 60)
    }

    mean_ia_iterations_max_sinr = {
        "5": get_mean_iterations(max_sinrn_results, 5),
        "10": get_mean_iterations(max_sinrn_results, 10),
        "15": get_mean_iterations(max_sinrn_results, 15),
        "20": get_mean_iterations(max_sinrn_results, 20),
        "25": get_mean_iterations(max_sinrn_results, 25),
        "30": get_mean_iterations(max_sinrn_results, 30),
        "35": get_mean_iterations(max_sinrn_results, 35),
        "40": get_mean_iterations(max_sinrn_results, 40),
        "45": get_mean_iterations(max_sinrn_results, 45),
        "50": get_mean_iterations(max_sinrn_results, 50),
        "55": get_mean_iterations(max_sinrn_results, 55),
        "60": get_mean_iterations(max_sinrn_results, 60)
    }

    mean_ia_iterations_mmse = {
        "5": get_mean_iterations(mmse_results, 5),
        "10": get_mean_iterations(mmse_results, 10),
        "15": get_mean_iterations(mmse_results, 15),
        "20": get_mean_iterations(mmse_results, 20),
        "25": get_mean_iterations(mmse_results, 25),
        "30": get_mean_iterations(mmse_results, 30),
        "35": get_mean_iterations(mmse_results, 35),
        "40": get_mean_iterations(mmse_results, 40),
        "45": get_mean_iterations(mmse_results, 45),
        "50": get_mean_iterations(mmse_results, 50),
        "55": get_mean_iterations(mmse_results, 55),
        "60": get_mean_iterations(mmse_results, 60)
    }

    # max_iterations = '10'

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Mean IA Iterations xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # noinspection PyUnboundLocalVariable
    ITER_ALL_ALGS = "{0}\n\n{1}\n\n{2}".format(
        generate_pgfplots_plotline(SNR_alt_min,
                                   mean_ia_iterations_alt_min[max_iterations],
                                   options="alt min iter style"),
        generate_pgfplots_plotline(SNR_max_SINR,
                                   mean_ia_iterations_max_sinr[max_iterations],
                                   options="max sinr iter style"),
        generate_pgfplots_plotline(SNR_mmse,
                                   mean_ia_iterations_mmse[max_iterations],
                                   options="mmse iter style"))

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx BER Plot xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    BER_ALL_ALGS = "{0}\n\n{1}\n\n{2}\n\n{3}".format(
        (generate_pgfplots_plotline(
            SNR_closed_form, ber_closed_form, options="closed form style") +
         "\n\\addlegendentry{Closed-Form};"),
        (generate_pgfplots_plotline(
            SNR_alt_min, ber_alt_min[max_iterations], options="alt min style") +
         "\n\\addlegendentry{Alt. Min.};"),
        (generate_pgfplots_plotline(SNR_max_SINR,
                                    ber_max_sinr[max_iterations],
                                    options="max sinr style") +
         "\n\\addlegendentry{Max SINR};"), (generate_pgfplots_plotline(
             SNR_mmse, ber_mmse[max_iterations], options="mmse style") +
                                            "\n\\addlegendentry{MMSE};"))

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Sum Capacity Plot xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    SUM_CAPACITY_ALL_ALGS = "{0}\n\n{1}\n\n{2}\n\n{3}".format(
        (generate_pgfplots_plotline(SNR_closed_form,
                                    sum_capacity_closed_form,
                                    options="closed form style") +
         "\n\\addlegendentry{Closed-Form};"),
        (generate_pgfplots_plotline(SNR_alt_min,
                                    sum_capacity_alt_min[max_iterations],
                                    options="alt min style") +
         "\n\\addlegendentry{Alt. Min.};"),
        (generate_pgfplots_plotline(SNR_max_SINR,
                                    sum_capacity_max_sinr[max_iterations],
                                    options="max sinr style") +
         "\n\\addlegendentry{Max SINR};"),
        (generate_pgfplots_plotline(
            SNR_mmse, sum_capacity_mmse[max_iterations], options="mmse style") +
         "\n\\addlegendentry{MMSE};"))

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxx Plot Filenames xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    second_tick = "{0}".format((int(max_iterations) // 10) + 1)

    with open('ber_plot_template.tikz', 'r') as fid:
        ber_plot_template = fid.read()

    with open('sum_capacity_template.tikz', 'r') as fid:
        sum_capacity_plot_template = fid.read()

    # noinspection PyUnboundLocalVariable
    ber_plot_filename = ("ber_all_ia_algs_{Nr}x{Nt}({Ns})_max_iter_"
                         "{max_iter}.tikz").format(Nr=Nr,
                                                   Nt=Nt,
                                                   Ns=Ns,
                                                   max_iter=max_iterations)
    sum_capacity_plot_filename = ("sum_capacity_all_ia_algs_{Nr}x{Nt}({Ns})"
                                  "_max_iter_{max_iter}.tikz").format(
                                      Nr=Nr,
                                      Nt=Nt,
                                      Ns=Ns,
                                      max_iter=max_iterations)

    ber_plot_file = open(ber_plot_filename, 'w')
    ber_plot_file.write(
        ber_plot_template.replace("MAXITER", max_iterations).replace(
            "BER_ALL_ALGS",
            BER_ALL_ALGS).replace("ITER_ALL_ALGS", ITER_ALL_ALGS).replace(
                "SECONDTICK", second_tick))
    ber_plot_file.close()

    sum_capacity_plot_file = open(sum_capacity_plot_filename, 'w')
    sum_capacity_plot_file.write(
        sum_capacity_plot_template.replace("MAXITER", max_iterations).replace(
            "SUM_CAPACITY_ALL_ALGS", SUM_CAPACITY_ALL_ALGS).replace(
                "ITER_ALL_ALGS",
                ITER_ALL_ALGS).replace("SECONDTICK",
                                       second_tick).replace("YMAX", "60"))
    sum_capacity_plot_file.close()
