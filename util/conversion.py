#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing function related to several conversions, such as
linear to dB, binary to gray code, as well as the inverse of them.
"""

import numpy as np
from misc import xor


def single_matrix_to_matrix_of_matrices(single_matrix, nrows, ncols=None):
    """Converts a single numpy array to a numpy array of numpy arrays.

    For instance, a 6x6 numpy array may be converted to a 3x3 numpy array
    of 2x2 arrays.

    Parameters
    ----------
    single_matrix : 1D numpy array or 2D numpy array
        The single numpy array.
    nrows : 1D numpy array of ints
        The number of rows of each submatrix, or the number of elements in
        each subarray.
    ncols : 1D numpy array of ints
        The number of rows of each submatrix. If `single_matrix` is a 1D
        array then ncols should be None (default)

    Returns
    -------
    array_of_arrays : 1D or 2D numpy array
        The converted array (1D or 2D) of arrays (1D or 2D).

    Examples
    --------
    >>> single_array = np.array([2, 2, 4, 5, 6, 8, 8, 8, 8])
    >>> sizes = np.array([2, 3, 4])
    >>> print single_matrix_to_matrix_of_matrices(single_array, sizes)
    [[2 2] [4 5 6] [8 8 8 8]]
    """
    cumNrows = np.hstack([0, np.cumsum(nrows)])
    K = nrows.size
    if ncols is not None:
        cumNcols = np.hstack([0, np.cumsum(ncols)])
        output = np.zeros([K, K], dtype=np.ndarray)

        for rx in np.arange(K):
            for tx in np.arange(K):
                output[rx, tx] = single_matrix[
                    cumNrows[rx]:cumNrows[rx + 1],
                    cumNcols[tx]:cumNcols[tx + 1]]
        return output
    else:
        output = np.zeros(K, dtype=np.ndarray)

        for rx in np.arange(K):
            output[rx] = single_matrix[cumNrows[rx]:cumNrows[rx + 1]]
        return output


def dB2Linear(valueIndB):
    """Convert input from dB to linear scale.

    Parameters
    ----------
    valueIndB : float
        Value in dB

    Returns
    -------
    valueInLinear : float
        Value in Linear scale.

    Examples
    --------
    >>> dB2Linear(30)
    1000.0
    """
    return pow(10, valueIndB / 10.0)


def linear2dB(valueInLinear):
    """Convert input from linear to dB scale.

    Parameters
    ----------
    valueInLinear : floar
        Value in Linear scale.

    Returns
    -------
    valueIndB : float
        Value in dB scale.

    Examples
    --------
    >>> linear2dB(1000)
    30.0
    """
    return 10.0 * np.log10(valueInLinear)


def dBm2Linear(valueIndBm):
    """Convert input from dBm to linear scale.

    Parameters
    ----------
    valueIndBm : float
        Value in dBm.

    Returns
    -------
    valueInLinear : float
        Value in linear scale.

    Examples
    --------
    >>> dBm2Linear(60)
    1000.0
    """
    return dB2Linear(valueIndBm) / 1000.


def linear2dBm(valueInLinear):
    """Convert input from linear to dBm scale.

    Parameters
    ----------
    valueInLinear : float
        Value in Linear scale

    Returns
    -------
    valueIndBm : float
        Value in dBm.

    Examples
    --------
    >>> linear2dBm(1000)
    60.0
    """
    return linear2dB(valueInLinear * 1000.)


# Code from wikipedia
# http://en.wikipedia.org/wiki/Gray_code#Constructing_an_n-bit_Gray_code
def binary2gray(num):
    """Convert a number (in decimal format) to the corresponding Gray code
    (still in decimal format).

    Parameters
    ----------
    num : int
        The number in decimal encoding

    Returns
    -------
    num_gray : int
        Corresponding gray code (in decimal format) of `num`.

    Examples
    --------
    >>> binary2gray(np.arange(0,8))
    array([0, 1, 3, 2, 6, 7, 5, 4])
    """
    return xor((num >> 1), num)


def gray2binary(num):
    """Convert a number in Gray code (in decimal format) to its original
    value (in decimal format).

    Parameters
    ----------
    num : int
        The number in gray coding

    Returns
    -------
    num_orig : int
        The original number (in decimal format) whose Gray code
        correspondent is `num`.

    Examples
    --------
    >>> gray2binary(binary2gray(np.arange(0,10)))
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    temp = xor(num, (num >> 8))
    temp = xor(temp, (temp >> 4))
    temp = xor(temp, (temp >> 2))
    temp = xor(temp, (temp >> 1))

    return temp


def SNR_dB_to_EbN0_dB(SNR, bits_per_symb):
    """Convert an SNR value (in dB) to the equivalent Eb/N0 value (also in dB).

    Parameters
    ----------
    SNR : float
        SNR value (in dB)
    bits_per_symb : int
        Number of bits in a symbol.

    Returns
    -------
    EbN0 : float
        Eb/N0 value (in dB)

    """
    # ebn0 = dB2Linear(SNR) / float(bits_per_symb)
    # EbN0 = linear2dB(ebn0)

    EbN0 = SNR - 10 * np.log10(bits_per_symb)

    return EbN0


def EbN0_dB_to_SNR_dB(EbN0, bits_per_symb):
    """Convert an Eb/N0 value (in dB) to the equivalent SNR value (also in dB).

    Parameters
    ----------
    EbN0 : float
        Eb/N0 value (in dB)
    bits_per_symb : int
        Number of bits in a symbol.

    Returns
    -------
    SNR : float
        SNR value (in dB)

    """
    SNR = EbN0 + 10 * np.log10(bits_per_symb)
    return SNR
