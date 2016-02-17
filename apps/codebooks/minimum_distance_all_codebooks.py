#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""module docstring"""

import numpy as np

Nt = 2
Ns = 1
# A script and a configuration file will be created for each value of K
K = range(8, 65, 4)

saved_data_file = "codebook_{0}_precoders_in_G({1},{2}).npz"

np.set_printoptions(precision=4)
min_dists = ""
for k in K:
    # print(saved_data_file.format(k, Nt, Ns))
    results = np.load(saved_data_file.format(k, Nt, Ns))
    min_dists += " | {:0.4f}".format(results['best_dist'].item())
    # Half part
    if k == 36:
        min_dists += " |\n"

    # print("{0} & {1}".format(k, results['best_dist']))

min_dists += " |"

print(min_dists)
