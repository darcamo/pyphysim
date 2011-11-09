#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing useful functions that I'd like to have in any python
section.

The folder with this module should be added to the python path, so that I
can use
    from darlan import *
and have access to all of these functions.

"""
__version__ = "$Revision: $"
# $Source$

import math
from numpy import unravel_index, prod

import numpy as np


def mmat(x, format='%+.12e'):
    """Display the ndarray 'x' in a format suitable for pasting to MATLAB

    mmat - a function to format arrays of arbitrary dimension for easy copy
    and paste to an interactive matlab session

    >>> a=np.arange(1,10)
    >>> a.shape=(3,3)
    >>> mmat(a)
    [ +1.000000000000e+00 +2.000000000000e+00 +3.000000000000e+00 ;   +4.000000000000e+00 +5.000000000000e+00 +6.000000000000e+00 ;   +7.000000000000e+00 +8.000000000000e+00 +9.000000000000e+00 ]
    """

    def print_row(row, format):
        if row.dtype == 'complex':
            for i in row:
                format_string = "%s%sj" % (format, format)
                print format_string % (i.real, i.imag)
        else:
            for i in row:
                print format % i,

    # if x.dtype=='complex':
    #     raise Exception ("conversion invalid for complex type")

    if x.ndim == 1:
        # 1d input
        print "[",
        print_row(x, format)
        print "]"
        print ""

    if x.ndim == 2:
        print "[",
        print_row(x[0], format)
        if x.shape[0] > 1:
            print ';',
        for row in x[1:-1]:
            print " ",
            print_row(row, format)
            print ";",
        if x.shape[0] > 1:
            print " ",
            print_row(x[-1], format)
        print "]",

    if x.ndim > 2:
        d_to_loop = x.shape[2:]
        sls = [slice(None, None)] * 2
        print "reshape([ ",
        # loop over flat index
        for i in range(prod(d_to_loop)):
            # reverse order for matlab
            # tricky double reversal to get first index to vary fastest
            ind_tuple = unravel_index(i, d_to_loop[::-1])[::-1]
            ind = sls + list(ind_tuple)
            mmat(x[ind], format)

        print '],[',
        for i in x.shape:
            print '%d' % i,
        print '])'


def randn_c(rows, cols):
    """Generates a random circularly complex gaussian matrix.

    Arguments:
    - `rows`: Number of rows for the random matrix
    - `cols`: Number of columns for the random matrix

    >>> a = randn_c(4,3)
    >>> a.shape
    (4, 3)
    >>> a.dtype
    dtype('complex128')
    """
    return (1.0 / math.sqrt(2.0)) * (
        np.random.randn(rows, cols) + (1j * np.random.randn(rows, cols)))


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


if __name__ == '__main__':
    # When this module is run as a script the doctests are executed
    import doctest
    doctest.testmod()
