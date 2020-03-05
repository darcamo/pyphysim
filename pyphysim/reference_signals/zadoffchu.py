#!/usr/bin/env python
"""
Module containing Zadoff-chu related functions.
"""

import numpy as np

__all__ = ['get_shifted_root_seq', 'calcBaseZC', 'get_extended_ZF']


def calcBaseZC(Nzc: int, u: int, q: complex = 0) -> np.ndarray:
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
    a_u : np.ndarray
        The root Zadoff-Chu sequence.
    """
    # $X = e^\frac{-j \pi u n (n+1)}{\text{Nzc}}$
    # In fact, 'u' must be lower than the largest prime number below or
    # equal to Nzc
    assert (u < Nzc)

    n = np.arange(Nzc)
    a_u = np.exp((-1j * np.pi * u * n * (n + 1 + 2 * q)) / Nzc)
    return a_u


def get_shifted_root_seq(root_seq: np.ndarray, n_cs: int,
                         denominator: int) -> np.ndarray:
    """
    Get the shifted root sequence suitable as the SRS sequence or the
    DMRS sequence of a user (depend on the `denominator` parameter).

    Parameters
    ----------
    root_seq : np.ndarray
        The root sequence to be shifted. This is a complex numpy array.
    n_cs : int
        The desired cyclic shift number. This should be an integer from
        0 to `denominator`-1, where 0 will just return the base
        sequence, 1 gives the first shift, and so on.
    denominator : int
        The denominator in the cyclic shift formula. This should be 8 for
        SRS and 12 for DMRS.

    Returns
    -------
    np.ndarray
        The shifted root sequence (a complex numpy array).

    See Also
    --------
    get_srs_seq, get_dmrs_seq
    """
    assert (abs(n_cs) >= 0)
    assert (abs(n_cs) < denominator)

    all_index_values = np.arange(root_seq.size)
    alpha_m = 2 * np.pi * n_cs / denominator
    shifted_seq = np.exp(1j * alpha_m * all_index_values) * root_seq
    return shifted_seq


def get_extended_ZF(root_seq: np.ndarray, size: int) -> np.ndarray:
    """
    Cyclic Extend the Zadoff-Chu root sequence to have size equal to
    `size`.

    Parameters
    ----------
    root_seq : np.ndarray
        The root Zadoff-Chu sequence. This is a complex numpy array.
    size : int
        The size that the sequence should be extended to.

    Returns
    -------
    output : np.ndarray
        The extended root sequence.

    Examples
    --------
    >>> root_seq = np.array([1, 2, 3, 4, 5])
    >>> get_extended_ZF(root_seq, 8)
    array([1, 2, 3, 4, 5, 1, 2, 3])
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
#     r1 = get_srs_seq(a_u, 2)
#     print(r1)
