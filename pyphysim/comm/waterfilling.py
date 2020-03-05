#!/usr/bin/env python
"""Implements a waterfilling method.

The doWF method performs the waterfilling algorithm.
"""

from typing import Tuple

import numpy as np

__all__ = ['doWF']


# noinspection PyUnresolvedReferences
def doWF(vtChannels: np.ndarray,
         dPt: float,
         noiseVar: float = 1.0,
         Es: float = 1.0) -> Tuple[np.ndarray, float]:
    """
    Performs the Waterfilling algorithm and returns the optimum power and
    water level.

    Parameters
    ----------
    vtChannels : np.ndarray
        Numpy array with the channel POWER gains (power of the parallel
        AWGN channels).
    dPt : float
        Total available power.
    noiseVar : float
        Noise variance (power in linear scale).
    Es : float
        Symbol energy (in linear scale).

    Returns
    -------
    (vtOptP, mu) : (np.ndarray, float)
        A tuple with vtOptP and mu, where vtOptP are the optimum powers,
        while mu is the water level.
    """
    # Sort Channels (descending order)
    vtChannelsSortIndexes = np.argsort(vtChannels)[::-1]
    vtChannelsSorted = vtChannels[vtChannelsSortIndexes]
    assert isinstance(vtChannelsSorted, np.ndarray)

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Calculates the water level that touches the worst channel (the higher
    # one) and therefore transmits zero power in this worst channel. After
    # that, calculates the power in each channel (the vector 'Ps') for this
    # water level. If the sum of all of these powers in 'Ps' is less then
    # the total available power, then all we need to do is divide the
    # remaining power equally among all the channels (increase the water
    # level). On the other hand, if the sum of all of these powers in 'Ps'
    # is greater then the total available power then we remove the worst
    # channel and repeat the process.
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Calculates minimum water-level $\mu$ required to use all channels
    dNChannels = vtChannels.size
    dRemoveChannels = 0

    minMu = float(noiseVar) / (
        Es * vtChannelsSorted[dNChannels - dRemoveChannels - 1])
    Ps = (minMu - float(noiseVar) /
          (Es * vtChannelsSorted[np.arange(0, dNChannels - dRemoveChannels)]))

    # Ps should be a numpy array
    assert isinstance(Ps, np.ndarray)

    while (sum(Ps) > dPt) and (dRemoveChannels < dNChannels):
        dRemoveChannels += 1
        minMu = float(noiseVar) / (
            Es * vtChannelsSorted[dNChannels - dRemoveChannels - 1])
        Ps = (
            minMu - float(noiseVar) /
            (Es * vtChannelsSorted[np.arange(0, dNChannels - dRemoveChannels)])
        )

    # Distributes the remaining power among the all the remaining channels
    dPdiff = dPt - np.sum(Ps)
    vtOptPaux = dPdiff / (dNChannels - dRemoveChannels) + Ps

    # Put optimum power in the original channel order
    vtOptP = np.zeros([
        vtChannels.size,
    ])
    vtOptP[vtChannelsSortIndexes[np.arange(0, dNChannels -
                                           dRemoveChannels)]] = vtOptPaux
    mu = vtOptPaux[0] + float(noiseVar) / vtChannelsSorted[0]

    return vtOptP, mu
