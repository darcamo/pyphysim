#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing function related to conversion.


"""

import numpy as np
from misc import xor


def dB2Linear(valueIndB):
    """Convert input from dB to linear.

    >>> dB2Linear(30)
    1000.0
    """
    return pow(10, valueIndB / 10.0)


def linear2dB(valueInLinear):
    """Convert input from linear to dB.

    >>> linear2dB(1000)
    30.0
    """
    return 10.0 * np.log10(valueInLinear)


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


# xxxxx Perform the doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    # When this module is run as a script the doctests are executed
    import doctest
    doctest.testmod()
