#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1103
import scipy.constants
from ..util.conversion import linear2dBm


def calc_thermal_noise_power_dBm(T, delta_f):
    """
    Calculate the thermal noise power for the given room temperature
    `T` (in Cº) and bandwidth `delta_f` (in Hz).

    Parameters
    ----------
    T : float
        Room temperature in Cesium degrees.
    delta_f : float
        Bandwidth in Hz.

    Returns
    -------
    noise_var : float
        The noise power.
    """
    # Boltzmann constant
    B = scipy.constants.Boltzmann
    K = T + 273.0
    noise_power = B * K * delta_f
    noise_power_dBm = linear2dBm(noise_power)
    return noise_power_dBm
