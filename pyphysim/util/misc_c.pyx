#!/usr/bin/env python
# -*- coding: utf-8 -*-

#cython: language_level=3

cimport numpy as np
import numpy as np
cimport cython

"""This module re-implements functions in the util.misc module,
but using Cython for speeding up calculations. """

# Because we are using cdef, this function won't be available in python
cdef int _count_bits_single_element(int n):
    """
    Count the number of bits that are set in an integer number.

    Parameters
    ----------
    n : int
        The integer number.

    Returns
    -------
    int
        Number of bits that are equal to 1 in the bit representation of
        the number `n`.
    """
    cdef int count = 0
    while n > 0:
        if n & 1 == 1:
            count += 1
        n >>= 1
    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def count_bits_1D_array(np.ndarray[np.int_t, ndim=1] n):
    """
    Count the number of bits that are set.

    Parameters
    ----------
    n : np.ndarray
        An integer number or a numpy array of integer numbers.

    Returns
    -------
    num_bits : np.ndarray
        1D numpy array with the number of bits that are set for each
        element in `n`

    """
    assert n.dtype == np.int

    cdef int num_el = len(n)
    cdef Py_ssize_t index  # Since we will use index for indexing 'n', then
                           # using Py_ssize_t as the type for index give
                           # faster results then using a simple int.
    cdef np.ndarray[np.int_t, ndim=1] num_bits = np.empty(num_el, dtype=np.int)
    for index in range(num_el):
        num_bits[index] = _count_bits_single_element(n[index])

    return num_bits


@cython.boundscheck(False)
@cython.wraparound(False)
def count_bits_ND_array(n):
    cdef np.ndarray[np.int_t, ndim=1] flattened_input = n.flatten()

    cdef int num_el = len(flattened_input)
    cdef Py_ssize_t index  # Since we will use index for indexing 'n', then
                           # using Py_ssize_t as the type for index give
                           # faster results then using a simple int.
    cdef np.ndarray[np.int_t, ndim=1] num_bits_flat = np.empty(num_el, dtype=np.int)
    for index in range(num_el):
        num_bits_flat[index] = _count_bits_single_element(
            flattened_input[index])

    return np.reshape(num_bits_flat, n.shape)


def count_bits(n):
    """
    Count the number of bits that are set in `n`.

    Parameters
    ----------
    n : int | np.ndarray
        An integer number or a numpy array of integer numbers.

    Returns
    -------
    num_bits : int | np.ndarray
        Number of bits that are set in `n`. If `n` is a numpy array then
        `num_bits` will also be a numpy array with the number of bits that
        are set for each element in `n`
    """
    if not isinstance(n, np.ndarray):
        # If the input is not a numpy array we assume it to be an integer
        # and we call _count_bits_single_element directly
        return _count_bits_single_element(n)

    assert n.dtype == np.int

    if n.ndim == 1:
        return count_bits_1D_array(n)

    # General case
    return count_bits_ND_array(n)
