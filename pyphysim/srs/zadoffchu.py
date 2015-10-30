#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module containing Zadoff-chu related functions.
"""

import numpy as np

__all__ = ['calcBaseZC', 'getShiftedZF', 'get_extended_ZF']


def calcBaseZC(Nzc, u, q=0):
    """
    Calculate the root sequence of Zadoff-Chu sequences.

    Parameters
    ----------
    Nzc : int
        The size of the root Zadoff-Chu sequence.
    u : int
        The root sequence index.
    q : complex
        Any complex number. Usually this is just zero.

    Returns
    -------
    a_u : numpy array
        The root Zadoff-Chu sequence.
    """
    # In fact, 'u' must be lower than the largest prime number below or
    # equal to Nzc
    assert (u < Nzc)

    n = np.arange(Nzc)
    a_u = np.exp((-1j * np.pi * u * n * (n + 1 + 2 * q)) / Nzc)
    return a_u


def getShiftedZF(root_seq, n_cs):
    """
    Get the shifted Zadoff-Chu sequence from the root sequence.

    Parameters
    ----------
    root_seq : complex numpy array
        The Zadoff-Chu root sequence.
    n_cs : int
        The desired cyclic shift number. This should be an integer from 0
        to 7, where 0 will just return the base sequence, 1 gives the first
        shift, and so on.
    """
    assert (abs(n_cs) >= 0)
    assert (abs(n_cs) < 8)

    Nzc = root_seq.size
    alpha_m = 2 * np.pi * n_cs / 8
    shifted_seq = np.exp(1j * alpha_m * np.arange(Nzc)) * root_seq
    return shifted_seq


def get_extended_ZF(root_seq, size):
    """
    Cyclic Extend the Zadoff-Chu root sequence to have size equal to `size`.

    Parameters
    ----------
    root_seq : complex numpy array
        The root Zadoff-Chu sequence.
    size : int
        The size that the sequence should be extended to.

    Returns
    -------
    output : numpy complex array
        The extended root sequence.

    Example
    -------
    >>> root_seq = np.array([1, 2, 3, 4, 5])
    >>> get_extended_ZF(root_seq, 8)
    [1,2,3,4,5,1,2,3]
    """
    root_seq_size = root_seq.size
    if size - root_seq_size > root_seq_size:
        stack_list = [root_seq]
        num_full_repeats = (size // root_seq_size)
        # Repeat the full sequence by this amount
        stack_list *= num_full_repeats

        current_size = root_seq_size * num_full_repeats

        # Append remaining element to achieve the required size
        stack_list.append(root_seq[0:size - current_size])

        output = np.hstack(stack_list)
    else:
        output = np.hstack([root_seq, root_seq[0:size - root_seq_size]])

    return output

# if __name__ == '__main__':
#     np.set_printoptions(precision=4)
#     a_u = calcBaseZC(23, 4)
#     r1 = getShiftedZF(a_u, 2)
#     print(r1)
