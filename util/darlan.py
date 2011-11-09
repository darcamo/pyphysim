#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing useful functions that I'd like to have in any python
section.

The folder with this module should be added to the python path, so that I
can use
    from darlan import *
and have access to all of these functions.

"""

import math
from numpy import unravel_index, prod

import numpy as np


# xxxxx mmat xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def mmat(x, format='%+.12e'):
    """Display the ndarray 'x' in a format suitable for pasting to MATLAB

    mmat - a function to format arrays of arbitrary dimension for easy copy
    and paste to an interactive matlab session
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
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxx randn_c - Start - xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def randn_c(rows, cols):
    """

    Arguments:
    - `size`:
    """
    return (1.0 / math.sqrt(2.0)) * np.random.randn(rows, cols) + (1j * np.random.randn(rows,cols))
# xxxxx randn_c - End - xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxx Conversion from dB to Linear and vice-versa xxxxxxxxxxxxxxxxxxxxxxx
def dB2Linear(valueIndB):
    """Convert input from dB to linear."""
    return pow(10, valueIndB / 10.0)


def linear2dB(valueInLinear):
    """Convert input from linear to dB."""
    return 10.0 * np.log10(valueInLinear)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
