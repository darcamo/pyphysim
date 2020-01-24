#!/usr/bin/env python
# -*- coding: utf-8 -*-

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
sys.path.append('../../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# xxxxxxxxxx Import Statements xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
from pyphysim.simulations.results import SimulationResults
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

if __name__ == '__main__':
    full_results_name = ("greedy_IA_stream_sel_results_[0.0_(5.0)_30.0]_"
                         "4-PSK_3x3_(3)_MaxIter_120_(['random']).pickle")
    full_result = SimulationResults.load_from_file(full_results_name)
    full_params = full_result.params

    name = ("partial_results/greedy_IA_stream_sel_results_[0.0_(5.0)_30.0]_"
            "4-PSK_3x3_(3)_MaxIter_120_(['random'])_unpack_{:0>2d}.pickle")

    for i in range(28):
        result = SimulationResults.load_from_file(name.format(i))
        params = result.params
        stream_sel_method = params['stream_sel_method']
        scenario = params['scenario']
        initialize_with = params['initialize_with']
        SNR = params['SNR']
        # print('Unpacked parameters')
        print(('scenario: {0:>10} | stream_sel_method: {1:>6s} | '
               'SNR: {2:>4}').format(scenario, stream_sel_method, SNR))

        # if scenario == 'NoPathLoss' and stream_sel_method == 'brute':
        #     print "SNR {0}: Ber {1}".format(SNR, result['ber'])

        # SNR 0.0: Ber [Result -> ber: 192516/6502400 -> 0.0296069143701]
        # SNR 5.0: Ber [Result -> ber: 32498/7090800 -> 0.00458312179162]
        # SNR 10.0: Ber [Result -> ber: 2108/10665200 -> 0.000197652177174]
        # SNR 15.0: Ber [Result -> ber: 11/11884000 -> 9.25614271289e-07]
        # SNR 20.0: Ber [Result -> ber: 0/12558000 -> 0.0]
        # SNR 25.0: Ber [Result -> ber: 0/13464400 -> 0.0]
        # SNR 30.0: Ber [Result -> ber: 0/13934400 -> 0.0]

    # print("\n\n")
    # params_list = full_params.get_unpacked_params_list()
    # for p in params_list:
    #     stream_sel_method = p['stream_sel_method']
    #     scenario = p['scenario']
    #     initialize_with = p['initialize_with']
    #     SNR = p['SNR']
    #     print('scenario: {0:>10} | stream_sel_method: {1:>6s} | '
    #           'SNR: {2:>4}').format(scenario, stream_sel_method, SNR)
