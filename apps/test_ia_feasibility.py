#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""module docstring"""
import numpy as np
import itertools

import sys
sys.path.append("../")

from ia import ia


if __name__ == '__main__':
    alternating_iterations = 400
    # K = 3
    # Nr = np.ones(K) * 4
    # Nt = np.ones(K) * 4
    # Ns = np.ones(K) * 2

    K = 5
    Nr = np.ones(K) * 4
    Nt = np.ones(K) * 8
    Ns = np.ones(K) * 2

    alt = ia.AlternatingMinIASolver()
    alt.randomizeH(Nr, Nt, K)
    alt.randomizeF(Nt, Ns, K)

    for i in xrange(alternating_iterations):
        alt.step()

    print alt.getCost()

    all_possibilities = itertools.product(range(K), range(K))
    for ij in all_possibilities:
        i, j = ij
        print "H{0}{1}".format(i, j)
        Hij = alt.get_channel(i, j)
        Hij_eff = np.dot(alt.W[i], np.dot(Hij, alt.F[j]))
        print np.linalg.svd(Hij_eff)[1].round(5)
        print np.linalg.svd(Hij_eff)[0].round(5)
        print
