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
import numpy as np
from scipy.special import erfc
#import math.erf
# erf tb pode ser encontrada la biblioteca scipy.special
# erf tb pode ser encontrada la biblioteca math  -> python 2.7 ou superior
# erf tb pode ser encontrada la biblioteca mpmath


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


def xor(a, b):
    """Calculates the xor operation between a and b.

    In python this is performed with a^b. However, sage changed the "^"
    operator. This xor function was created so that it can be used in
    either sage or in regular python.

    Arguments:
    - `a`: first number
    - `b`: second number

    >>> xor(3,7)
    4
    >>> xor(15,6)
    9
    """
    return (a).__xor__(b)


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


def level2bits(n):
    """Calculates the number of needed to represent n different values.

    Arguments:
    - `n`: Number of different levels.

    >>> map(level2bits,range(1,20))
    [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5]
    """
    if n < 1:
        raise ValueError("level2bits: n must be greater then one")
    return int2bits(n - 1)


def int2bits(n):
    """Calculates the number of bits needed to represent an interger n.

    Arguments:
    - `n`: An Ingerger number

    >>> map(int2bits, range(0,19))
    [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5]
    """
    if n < 0:
        raise ValueError("int2bits: n must be greater then zero")

    if n == 0:
        return 1

    bits = 0
    while n:
        n >>= 1
        bits += 1
    return bits


def bitCount(n):
    """Count the number of bits that are set in an interger number.

    Arguments:
    - `n`: The number

    >>> 10
    """
    count = 0
    while n > 0:
        if n & 1 == 1:
            count += 1
        n >>= 1
    return count

# TODO: Because I convert bitCount to a ufunc, any doctest in bitCount is
# lost. Figure it out how to include a doctest in a ufunc.
#
# Make bitCount an ufunc
bitCount = np.frompyfunc(bitCount, 1, 1, doc=bitCount.__doc__)


def qfunc(x):
    """Calculates the qfunction of x.

    Arguments:
    - `x`:

    >>> qfunc(0.0)
    0.5
    >>> round(qfunc(1.0), 9)
    0.158655254
    >>> round(qfunc(3.0), 9)
    0.001349898
    """
    return 0.5 * erfc(x / math.sqrt(2))


# xxxxx Perform the doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    # When this module is run as a script the doctests are executed
    import doctest
    doctest.testmod()
