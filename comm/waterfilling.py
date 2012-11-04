#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements a waterfilling method.

The doWF performs the waterfilling algorithm.

The genLatexCode returns latex code that can draw the provided solution by
the waterfilling algorithm, while the drawWF creates a file with this
code.
"""

import numpy as np


# TODO: Change comments from portuguese to english
def doWF(vtChannels, dPt, noiseVar=1.0, Es=1.0):
    """Performs the Waterfilling algorithm and returns the optimum power and water level.

    Arguments:
    - `vtChannels`: Numpy array with the channel POWER gains (power of the
                    parallel AWGN channels).
    - `dPt`: Total available power.
    - `noiseVar`: Noise variance (power in linear scale)
    - `Es`: Symbol energy (in linear scale)
    """
    ## Sort Channels (descending order)
    vtChannelsSortIndexes = np.argsort(vtChannels)[::-1]
    vtChannelsSorted = vtChannels[vtChannelsSortIndexes]

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Calcula o waterlevel que toca o pior canal (o mais alto) e
    # portanto que transmite potencia 0 no pior canal.  Depois disso
    # calcula a potencia em cada canal (o vetor Ps) para esse
    # waterlevel.  Se a soma dessas potencias for menor do que a
    # portencia total entao e so dividir a potencia restante igualmente
    # entre todos os canais (aumentar o waterlevel). Caso contrario,
    # removo o pior canal e repito o processo.
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Calculates minimum waterlevel $\mu$ required to use all channels
    dNChannels = vtChannels.size
    dRemoveChannels = 0

    minMu = float(noiseVar) / (
        Es * vtChannelsSorted[dNChannels - dRemoveChannels - 1])
    Ps = (minMu - float(noiseVar) / (
        Es * vtChannelsSorted[np.arange(0, dNChannels - dRemoveChannels)]))

    while (sum(Ps) > dPt) and (dRemoveChannels < dNChannels):
        dRemoveChannels = dRemoveChannels + 1
        minMu = float(noiseVar) / (
            Es * vtChannelsSorted[dNChannels - dRemoveChannels - 1])
        Ps = (minMu - float(noiseVar) / (
            Es * vtChannelsSorted[np.arange(0, dNChannels - dRemoveChannels)]))

    # Distribui a potencia restante entre todos os canais remanescentes
    dPdiff = dPt - Ps.sum()
    vtOptPaux = dPdiff / (dNChannels - dRemoveChannels) + Ps

    # Put optimum power in the original channel order
    vtOptP = np.zeros([vtChannels.size, ])
    vtOptP[vtChannelsSortIndexes[np.arange(0, dNChannels - dRemoveChannels)]] = vtOptPaux
    mu = vtOptPaux[0] + float(noiseVar) / vtChannelsSorted[0]

    return (vtOptP, mu)
