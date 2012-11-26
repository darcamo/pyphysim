#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with functions to easily moving data from python to MATLAB."""

import numpy as np


def mmat(x, format='%+.12e'):
    """Display the ndarray 'x' in a format suitable for pasting to MATLAB

    The mmat function formats numpy arrays of arbitrary dimension in a way
    which can easily copied and pasted into an interactive MATLAB session

    Parameters
    ----------
    format : str, optional
        The format string for the conversion.

    Returns
    -------
    converted_string : str
        A string that represents the converted numpy array. You can copy
        this string and past it into a MATLAB session.

    Examples
    --------
    >>> a=np.arange(1,10)
    >>> a.shape=(3,3)
    >>> # Print as a numpy matrix
    >>> print a
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    >>> # Call mmat(a) to print the string representation of the converted
    >>> # matrix
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
        for i in range(np.prod(d_to_loop)):
            # reverse order for matlab
            # tricky double reversal to get first index to vary fastest
            ind_tuple = np.unravel_index(i, d_to_loop[::-1])[::-1]
            ind = sls + list(ind_tuple)
            mmat(x[ind], format)

        print '],[',
        for i in x.shape:
            print '%d' % i,
        print '])'
