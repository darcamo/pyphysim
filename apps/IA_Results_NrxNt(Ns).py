#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""module docstring"""


__revision__ = "$Revision$"

"Script para gerar os plots"


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
from pyphysim.util.simulations import SimulationRunner, SimulationParameters, SimulationResults, Result
from pyphysim.comm import modulators, channels
from pyphysim.util.conversion import dB2Linear
from pyphysim.util import misc
from pyphysim.ia import algorithms
import numpy as np
from pprint import pprint
from plot.pgfplotshelper import *
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# In Ipython run this script with
#     run -i IA_Results_2x2(1).py
try:
    initialized
except Exception as e:
    print "nao deveria estar aqui"
    initialized = False
    max_iterations = "{0}".format(5)

## xxxxxxxxxx Function Definitions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

def get_ber_for_given_num_iter(result_obj, max_iterations):
    """Docstring"""
    ber = result_obj.get_result_values_list(
            'ber',
            fixed_params={'max_iterations': max_iterations})
    return ber


def get_sum_capacity_for_given_num_iter(result_obj, max_iterations):
    """Docstring"""
    sum_capacity = result_obj.get_result_values_list(
            'sum_capacity',
            fixed_params={'max_iterations': max_iterations})
    return sum_capacity


def get_mean_iterations(result_obj, max_iterations):
    """Docstring"""
    mean_ia_terations = get_num_mean_ia_iterations(result_obj, {'max_iterations': max_iterations})
    return mean_ia_terations


def get_num_runned_reps(sim_results_object, fixed_params=None):
    """Docstring"""
    if fixed_params is None:
        fixed_params = {}

    all_runned_reps = np.array(sim_results_object.runned_reps)
    indexes = sim_results_object.params.get_pack_indexes(fixed_params)
    return all_runned_reps[indexes]


def get_num_mean_ia_iterations(sim_results_object, fixed_params=None):
    """Docstring"""
    if fixed_params is None:
        fixed_params = {}
    return sim_results_object.get_result_values_list('ia_runned_iterations', fixed_params)


## xxxxxxxxxx Initializations xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

if initialized is False:
    print "Lendo arquivos de resultados"
    # xxxxx Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #params = SimulationParameters.load_from_config_file('ia_config_file.txt')
    K = 3
    Nr = 5
    Nt = 3
    Ns = 2
    M = 4
    max_iterations = '60'
    modulator = "PSK"
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Results base name xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    base_name = 'results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_MaxIter_[5_(5)_60]'.format(M=M, modulator=modulator, Nr=Nr, Nt=Nt, Ns=Ns)
    base_name_no_iter = 'results_{M}-{modulator}_{Nr}x{Nt}_({Ns})_MaxIter_[5_(5)_60]'.format(M=M, modulator=modulator, Nr=Nr, Nt=Nt, Ns=Ns)  # Used only for the closed form algorithm, which is not iterative
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    initialized = True


## xxxxxxxxxx SNR variables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
SNR_alt_min = np.array(alt_min_results.params['SNR'])
SNR_closed_form = np.array(closed_form_results.params['SNR'])
SNR_max_SINR = np.array(max_sinrn_results.params['SNR'])
# SNR_min_leakage = np.array(min_leakage_results.params['SNR'])
SNR_mmse = np.array(mmse_results.params['SNR'])


## xxxxxxxxxx BER Variables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ber_closed_form = get_ber_for_given_num_iter(closed_form_results, 5)

ber_alt_min = {}
ber_alt_min["5"] = get_ber_for_given_num_iter(alt_min_results, 5)
ber_alt_min["10"] = get_ber_for_given_num_iter(alt_min_results, 10)
ber_alt_min["15"] = get_ber_for_given_num_iter(alt_min_results, 15)
ber_alt_min["20"] = get_ber_for_given_num_iter(alt_min_results, 20)
ber_alt_min["25"] = get_ber_for_given_num_iter(alt_min_results, 25)
ber_alt_min["30"] = get_ber_for_given_num_iter(alt_min_results, 30)
ber_alt_min["35"] = get_ber_for_given_num_iter(alt_min_results, 35)
ber_alt_min["40"] = get_ber_for_given_num_iter(alt_min_results, 40)
ber_alt_min["45"] = get_ber_for_given_num_iter(alt_min_results, 45)
ber_alt_min["50"] = get_ber_for_given_num_iter(alt_min_results, 50)
ber_alt_min["55"] = get_ber_for_given_num_iter(alt_min_results, 55)
ber_alt_min["60"] = get_ber_for_given_num_iter(alt_min_results, 60)

ber_max_sinr = {}
ber_max_sinr["5"] = get_ber_for_given_num_iter(max_sinrn_results, 5)
ber_max_sinr["10"] = get_ber_for_given_num_iter(max_sinrn_results, 10)
ber_max_sinr["15"] = get_ber_for_given_num_iter(max_sinrn_results, 15)
ber_max_sinr["20"] = get_ber_for_given_num_iter(max_sinrn_results, 20)
ber_max_sinr["25"] = get_ber_for_given_num_iter(max_sinrn_results, 25)
ber_max_sinr["30"] = get_ber_for_given_num_iter(max_sinrn_results, 30)
ber_max_sinr["35"] = get_ber_for_given_num_iter(max_sinrn_results, 35)
ber_max_sinr["40"] = get_ber_for_given_num_iter(max_sinrn_results, 40)
ber_max_sinr["45"] = get_ber_for_given_num_iter(max_sinrn_results, 45)
ber_max_sinr["50"] = get_ber_for_given_num_iter(max_sinrn_results, 50)
ber_max_sinr["55"] = get_ber_for_given_num_iter(max_sinrn_results, 55)
ber_max_sinr["60"] = get_ber_for_given_num_iter(max_sinrn_results, 60)

ber_mmse = {}
ber_mmse["5"] = get_ber_for_given_num_iter(mmse_results, 5)
ber_mmse["10"] = get_ber_for_given_num_iter(mmse_results, 10)
ber_mmse["15"] = get_ber_for_given_num_iter(mmse_results, 15)
ber_mmse["20"] = get_ber_for_given_num_iter(mmse_results, 20)
ber_mmse["25"] = get_ber_for_given_num_iter(mmse_results, 25)
ber_mmse["30"] = get_ber_for_given_num_iter(mmse_results, 30)
ber_mmse["35"] = get_ber_for_given_num_iter(mmse_results, 35)
ber_mmse["40"] = get_ber_for_given_num_iter(mmse_results, 40)
ber_mmse["45"] = get_ber_for_given_num_iter(mmse_results, 45)
ber_mmse["50"] = get_ber_for_given_num_iter(mmse_results, 50)
ber_mmse["55"] = get_ber_for_given_num_iter(mmse_results, 55)
ber_mmse["60"] = get_ber_for_given_num_iter(mmse_results, 60)


## xxxxxxxxxx Sum Capacity Variables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
sum_capacity_closed_form = get_sum_capacity_for_given_num_iter(closed_form_results, 5)

sum_capacity_alt_min = {}
sum_capacity_alt_min["5"] = get_sum_capacity_for_given_num_iter(alt_min_results, 5)
sum_capacity_alt_min["10"] = get_sum_capacity_for_given_num_iter(alt_min_results, 10)
sum_capacity_alt_min["15"] = get_sum_capacity_for_given_num_iter(alt_min_results, 15)
sum_capacity_alt_min["20"] = get_sum_capacity_for_given_num_iter(alt_min_results, 20)
sum_capacity_alt_min["25"] = get_sum_capacity_for_given_num_iter(alt_min_results, 25)
sum_capacity_alt_min["30"] = get_sum_capacity_for_given_num_iter(alt_min_results, 30)
sum_capacity_alt_min["35"] = get_sum_capacity_for_given_num_iter(alt_min_results, 35)
sum_capacity_alt_min["40"] = get_sum_capacity_for_given_num_iter(alt_min_results, 40)
sum_capacity_alt_min["45"] = get_sum_capacity_for_given_num_iter(alt_min_results, 45)
sum_capacity_alt_min["50"] = get_sum_capacity_for_given_num_iter(alt_min_results, 50)
sum_capacity_alt_min["55"] = get_sum_capacity_for_given_num_iter(alt_min_results, 55)
sum_capacity_alt_min["60"] = get_sum_capacity_for_given_num_iter(alt_min_results, 60)

sum_capacity_max_sinr = {}
sum_capacity_max_sinr["5"] = get_sum_capacity_for_given_num_iter(max_sinrn_results, 5)
sum_capacity_max_sinr["10"] = get_sum_capacity_for_given_num_iter(max_sinrn_results, 10)
sum_capacity_max_sinr["15"] = get_sum_capacity_for_given_num_iter(max_sinrn_results, 15)
sum_capacity_max_sinr["20"] = get_sum_capacity_for_given_num_iter(max_sinrn_results, 20)
sum_capacity_max_sinr["25"] = get_sum_capacity_for_given_num_iter(max_sinrn_results, 25)
sum_capacity_max_sinr["30"] = get_sum_capacity_for_given_num_iter(max_sinrn_results, 30)
sum_capacity_max_sinr["35"] = get_sum_capacity_for_given_num_iter(max_sinrn_results, 35)
sum_capacity_max_sinr["40"] = get_sum_capacity_for_given_num_iter(max_sinrn_results, 40)
sum_capacity_max_sinr["45"] = get_sum_capacity_for_given_num_iter(max_sinrn_results, 45)
sum_capacity_max_sinr["50"] = get_sum_capacity_for_given_num_iter(max_sinrn_results, 50)
sum_capacity_max_sinr["55"] = get_sum_capacity_for_given_num_iter(max_sinrn_results, 55)
sum_capacity_max_sinr["60"] = get_sum_capacity_for_given_num_iter(max_sinrn_results, 60)

sum_capacity_mmse = {}
sum_capacity_mmse["5"] = get_sum_capacity_for_given_num_iter(mmse_results, 5)
sum_capacity_mmse["10"] = get_sum_capacity_for_given_num_iter(mmse_results, 10)
sum_capacity_mmse["15"] = get_sum_capacity_for_given_num_iter(mmse_results, 15)
sum_capacity_mmse["20"] = get_sum_capacity_for_given_num_iter(mmse_results, 20)
sum_capacity_mmse["25"] = get_sum_capacity_for_given_num_iter(mmse_results, 25)
sum_capacity_mmse["30"] = get_sum_capacity_for_given_num_iter(mmse_results, 30)
sum_capacity_mmse["35"] = get_sum_capacity_for_given_num_iter(mmse_results, 35)
sum_capacity_mmse["40"] = get_sum_capacity_for_given_num_iter(mmse_results, 40)
sum_capacity_mmse["45"] = get_sum_capacity_for_given_num_iter(mmse_results, 45)
sum_capacity_mmse["50"] = get_sum_capacity_for_given_num_iter(mmse_results, 50)
sum_capacity_mmse["55"] = get_sum_capacity_for_given_num_iter(mmse_results, 55)
sum_capacity_mmse["60"] = get_sum_capacity_for_given_num_iter(mmse_results, 60)


## xxxxxxxxxx Mean IA Iteration Vairables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
mean_ia_iterations_alt_min = {}
mean_ia_iterations_alt_min["5"] = get_mean_iterations(alt_min_results, 5)
mean_ia_iterations_alt_min["10"] = get_mean_iterations(alt_min_results, 10)
mean_ia_iterations_alt_min["15"] = get_mean_iterations(alt_min_results, 15)
mean_ia_iterations_alt_min["20"] = get_mean_iterations(alt_min_results, 20)
mean_ia_iterations_alt_min["25"] = get_mean_iterations(alt_min_results, 25)
mean_ia_iterations_alt_min["30"] = get_mean_iterations(alt_min_results, 30)
mean_ia_iterations_alt_min["35"] = get_mean_iterations(alt_min_results, 35)
mean_ia_iterations_alt_min["40"] = get_mean_iterations(alt_min_results, 40)
mean_ia_iterations_alt_min["45"] = get_mean_iterations(alt_min_results, 45)
mean_ia_iterations_alt_min["50"] = get_mean_iterations(alt_min_results, 50)
mean_ia_iterations_alt_min["55"] = get_mean_iterations(alt_min_results, 55)
mean_ia_iterations_alt_min["60"] = get_mean_iterations(alt_min_results, 60)

mean_ia_iterations_max_sinr = {}
mean_ia_iterations_max_sinr["5"] = get_mean_iterations(max_sinrn_results, 5)
mean_ia_iterations_max_sinr["10"] = get_mean_iterations(max_sinrn_results, 10)
mean_ia_iterations_max_sinr["15"] = get_mean_iterations(max_sinrn_results, 15)
mean_ia_iterations_max_sinr["20"] = get_mean_iterations(max_sinrn_results, 20)
mean_ia_iterations_max_sinr["25"] = get_mean_iterations(max_sinrn_results, 25)
mean_ia_iterations_max_sinr["30"] = get_mean_iterations(max_sinrn_results, 30)
mean_ia_iterations_max_sinr["35"] = get_mean_iterations(max_sinrn_results, 35)
mean_ia_iterations_max_sinr["40"] = get_mean_iterations(max_sinrn_results, 40)
mean_ia_iterations_max_sinr["45"] = get_mean_iterations(max_sinrn_results, 45)
mean_ia_iterations_max_sinr["50"] = get_mean_iterations(max_sinrn_results, 50)
mean_ia_iterations_max_sinr["55"] = get_mean_iterations(max_sinrn_results, 55)
mean_ia_iterations_max_sinr["60"] = get_mean_iterations(max_sinrn_results, 60)

mean_ia_iterations_mmse = {}
mean_ia_iterations_mmse["5"] = get_mean_iterations(mmse_results, 5)
mean_ia_iterations_mmse["10"] = get_mean_iterations(mmse_results, 10)
mean_ia_iterations_mmse["15"] = get_mean_iterations(mmse_results, 15)
mean_ia_iterations_mmse["20"] = get_mean_iterations(mmse_results, 20)
mean_ia_iterations_mmse["25"] = get_mean_iterations(mmse_results, 25)
mean_ia_iterations_mmse["30"] = get_mean_iterations(mmse_results, 30)
mean_ia_iterations_mmse["35"] = get_mean_iterations(mmse_results, 35)
mean_ia_iterations_mmse["40"] = get_mean_iterations(mmse_results, 40)
mean_ia_iterations_mmse["45"] = get_mean_iterations(mmse_results, 45)
mean_ia_iterations_mmse["50"] = get_mean_iterations(mmse_results, 50)
mean_ia_iterations_mmse["55"] = get_mean_iterations(mmse_results, 55)
mean_ia_iterations_mmse["60"] = get_mean_iterations(mmse_results, 60)


#max_iterations = '10'


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Mean IA Iterations xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx BER Plot xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
BER_ALL_ALGS = "{0}\n\n{1}\n\n{2}\n\n{3}".format(
    (generate_pgfplots_plotline(SNR_closed_form,
                               ber_closed_form,
                               options="closed form style") +
     "\n\\addlegendentry{Closed-Form};"),
    (generate_pgfplots_plotline(SNR_alt_min,
                                ber_alt_min[max_iterations],
                                options="alt min style") +
     "\n\\addlegendentry{Alt. Min.};"),
    (generate_pgfplots_plotline(SNR_max_SINR,
                                ber_max_sinr[max_iterations],
                                options="max sinr style") +
     "\n\\addlegendentry{Max SINR};"),
    (generate_pgfplots_plotline(SNR_mmse,
                                ber_mmse[max_iterations],
                                options="mmse style") +
     "\n\\addlegendentry{MMSE};"))


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Sum Capacity Plot xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
    (generate_pgfplots_plotline(SNR_mmse,
                                sum_capacity_mmse[max_iterations],
                                options="mmse style") +
     "\n\\addlegendentry{MMSE};"))


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Plot Filenames xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
second_tick = "{0}".format((int(max_iterations) // 10) + 1)

with open('ber_plot_template.tikz', 'r') as fid:
    ber_plot_template = fid.read()

with open('sum_capacity_template.tikz', 'r') as fid:
    sum_capacity_plot_template = fid.read()

ber_plot_filename = "ber_all_ia_algs_{Nr}x{Nt}({Ns})_max_iter_{max_iter}.tikz".format(
    Nr=Nr,
    Nt=Nt,
    Ns=Ns,
    max_iter=max_iterations)
sum_capacity_plot_filename = "sum_capacity_all_ia_algs_{Nr}x{Nt}({Ns})_max_iter_{max_iter}.tikz".format(
    Nr=Nr,
    Nt=Nt,
    Ns=Ns,
    max_iter=max_iterations)

ber_plot_file = open(ber_plot_filename, 'w')
ber_plot_file.write(
    ber_plot_template.replace(
        "MAXITER",
        max_iterations).replace(
            "BER_ALL_ALGS",
            BER_ALL_ALGS).replace(
                "ITER_ALL_ALGS",
                ITER_ALL_ALGS).replace("SECONDTICK", second_tick)
)
ber_plot_file.close()

sum_capacity_plot_file = open(sum_capacity_plot_filename, 'w')
sum_capacity_plot_file.write(
    sum_capacity_plot_template.replace(
        "MAXITER",
        max_iterations).replace(
            "SUM_CAPACITY_ALL_ALGS",
            SUM_CAPACITY_ALL_ALGS).replace(
                "ITER_ALL_ALGS",
                ITER_ALL_ALGS).replace(
                    "SECONDTICK",
                    second_tick).replace("YMAX", "60")
)
sum_capacity_plot_file.close()
