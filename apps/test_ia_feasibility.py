#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""module docstring"""
import numpy as np
import itertools

import sys
import os
parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
sys.path.append(parent_dir)

from ia import ia
from comm.channels import MultiUserChannelMatrix


if __name__ == '__main__':
    K = 3
    Nr = np.ones(K) * 2
    Nt = np.ones(K) * 2
    Ns = np.ones(K) * 1

    multiuserchannel = MultiUserChannelMatrix()

    alt = ia.AlternatingMinIASolver(multiuserchannel)
    multiuserchannel.randomize(Nr, Nt, K)
    alt.randomizeF(Ns)
    alt.max_iterations = 100

    alt.solve()

    print "Final_Cost: {0}\n".format(alt.getCost())

    all_possibilities = itertools.product(range(K), range(K))
    for ij in all_possibilities:
        i, j = ij
        print "Hij: H{0}{1}".format(i, j)
        Hij = multiuserchannel.get_channel(i, j)
        Hij_eff = np.dot(alt.W[i], np.dot(Hij, alt.F[j]))
        print "Eigenvalus: {0}".format(np.linalg.svd(Hij_eff)[1].round(6)[0])
        print "Eigenvector: {0}".format(np.linalg.svd(Hij_eff)[0].round(6)[0][0])
        print
