#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing function related to several conversions, such as
linear to dB, binary to gray code, as well as the inverse of them.
"""

import numpy as np
from misc import xor


def dB2Linear(valueIndB):
    """Convert input from dB to linear scale.

    Arguments:
    - `valueIndB`: Value in dB

    Ex:
    >>> dB2Linear(30)
    1000.0
    """
    return pow(10, valueIndB / 10.0)


def linear2dB(valueInLinear):
    """Convert input from linear to dB scale.

    Arguments:
    - `valueInLinear`: Value in Linear scale

    Ex:
    >>> linear2dB(1000)
    30.0
    """
    return 10.0 * np.log10(valueInLinear)


def dBm2Linear(valueIndBm):
    """Convert input from dBm to linear scale.

    Arguments:
    - `valueIndBm`: Value in dBm

    Ex:
    >>> dBm2Linear(60)
    1000.0
    """
    return dB2Linear(valueIndBm) / 1000.


def linear2dBm(valueInLinear):
    """Convert input from linear to dBm scale.

    Arguments:
    - `valueInLinear`: Value in Linear scale

    Ex:
    >>> linear2dBm(1000)
    60.0
    """
    return linear2dB(valueInLinear * 1000.)


# Code from wikipedia
# http://en.wikipedia.org/wiki/Gray_code#Constructing_an_n-bit_Gray_code
def binary2gray(num):
    """Convert a number (in decimal format) to the corresponding Gray code
    (still in decimal format).

    Arguments:
    - `num`: The number in decimal encoding

    >>> binary2gray(np.arange(0,8))
    array([0, 1, 3, 2, 6, 7, 5, 4])
    """
    return xor((num >> 1), num)


def gray2binary(num):
    """Convert a number in Gray code (in decimal format) to its original
    value (in decimal format).

    Arguments:
    - `num`: The number in gray coding

    >>> gray2binary(binary2gray(np.arange(0,10)))
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    """
    temp = xor(num, (num >> 8))
    temp = xor(temp, (temp >> 4))
    temp = xor(temp, (temp >> 2))
    temp = xor(temp, (temp >> 1))

    return temp
