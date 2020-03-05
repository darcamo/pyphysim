#!/usr/bin/env python
"""Module with Sounding Reference Signal (SRS) related functions"""

from typing import List, Optional, Union, cast

import numpy as np

from .zadoffchu import calcBaseZC, get_extended_ZF

# Type representing something that can be used to index a numpy array
Indexes = Union[np.ndarray, List[int], slice]

__all__ = ['RootSequence']

# List of prime numbers lower than 282.
_SMALL_PRIME_LIST = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
    331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
    421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503,
    509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607,
    613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701,
    709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811,
    821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911,
    919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009
])

# Table with root sequence for sequence size equal to 12 (number of
# subcarriers in one PRB)
ROOT_TABLE1 = {
    '0': np.array([-1, 1, 3, -3, 3, 3, 1, 1, 3, 1, -3, 3]),
    '1': np.array([1, 1, 3, 3, 3, -1, 1, -3, -3, 1, -3, 3]),
    '2': np.array([1, 1, -3, -3, -3, -1, -3, -3, 1, -3, 1, -1]),
    '3': np.array([-1, 1, 1, 1, 1, -1, -3, -3, 1, -3, 3, -1]),
    '4': np.array([-1, 3, 1, -1, 1, -1, -3, -1, 1, -1, 1, 3]),
    '5': np.array([1, -3, 3, -1, -1, 1, 1, -1, -1, 3, -3, 1]),
    '6': np.array([-1, 3, -3, -3, -3, 3, 1, -1, 3, 3, -3, 1]),
    '7': np.array([-3, -1, -1, -1, 1, -3, 3, -1, 1, -3, 3, 1]),
    '8': np.array([1, -3, 3, 1, -1, -1, -1, 1, 1, 3, -1, 1]),
    '9': np.array([1, -3, -1, 3, 3, -1, -3, 1, 1, 1, 1, 1]),
    '10': np.array([-1, 3, -1, 1, 1, -3, -3, -1, -3, -3, 3, -1]),
    '11': np.array([3, 1, -1, -1, 3, 3, -3, 1, 3, 1, 3, 3]),
    '12': np.array([1, -3, 1, 1, -3, 1, 1, 1, -3, -3, -3, 1]),
    '13': np.array([3, 3, -3, 3, -3, 1, 1, 3, -1, -3, 3, 3]),
    '14': np.array([-3, 1, -1, -3, -1, 3, 1, 3, 3, 3, -1, 1]),
    '15': np.array([3, -1, 1, -3, -1, -1, 1, 1, 3, 1, -1, -3]),
    '16': np.array([1, 3, 1, -1, 1, 3, 3, 3, -1, -1, 3, -1]),
    '17': np.array([-3, 1, 1, 3, -3, 3, -3, -3, 3, 1, 3, -1]),
    '18': np.array([-3, 3, 1, 1, -3, 1, -3, -3, -1, -1, 1, -3]),
    '19': np.array([-1, 3, 1, 3, 1, -1, -1, 3, -3, -1, -3, -1]),
    '20': np.array([-1, -3, 1, 1, 1, 1, 3, 1, -1, 1, -3, -1]),
    '21': np.array([-1, 3, -1, 1, -3, -3, -3, -3, -3, 1, -1, -3]),
    '22': np.array([1, 1, -3, -3, -3, -3, -1, 3, -3, 1, -3, 3]),
    '23': np.array([1, 1, -1, -3, -1, -3, 1, -1, 1, 3, -1, 1]),
    '24': np.array([1, 1, 3, 1, 3, 3, -1, 1, -1, -3, -3, 1]),
    '25': np.array([1, -3, 3, 3, 1, 3, 3, 1, -3, -1, -1, 3]),
    '26': np.array([1, 3, -3, -3, 3, -3, 1, -1, -1, 3, -1, -3]),
    '27': np.array([-3, -1, -3, -1, -3, 3, 1, -1, 1, 3, -3, -3]),
    '28': np.array([-1, 3, -3, 3, -1, 3, 3, -3, 3, 3, -1, -1]),
    '29': np.array([3, -3, -3, -1, -1, -3, -1, 3, -3, 3, 1, -1])
}

# Table with root sequence for sequence size equal to 24 (number of
# subcarriers in two PRBs)
ROOT_TABLE2 = {
    '0':
    np.array([
        -1, 3, 1, -3, 3, -1, 1, 3, -3, 3, 1, 3, -3, 3, 1, 1, -1, 1, 3, -3, 3,
        -3, -1, -3
    ]),
    '1':
    np.array([
        -3, 3, -3, -3, -3, 1, -3, -3, 3, -1, 1, 1, 1, 3, 1, -1, 3, -3, -3, 1,
        3, 1, 1, -3
    ]),
    '2':
    np.array([
        3, -1, 3, 3, 1, 1, -3, 3, 3, 3, 3, 1, -1, 3, -1, 1, 1, -1, -3, -1, -1,
        1, 3, 3
    ]),
    '3':
    np.array([
        -1, -3, 1, 1, 3, -3, 1, 1, -3, -1, -1, 1, 3, 1, 3, 1, -1, 3, 1, 1, -3,
        -1, -3, -1
    ]),
    '4':
    np.array([
        -1, -1, -1, -3, -3, -1, 1, 1, 3, 3, -1, 3, -1, 1, -1, -3, 1, -1, -3,
        -3, 1, -3, -1, -1
    ]),
    '5':
    np.array([
        -3, 1, 1, 3, -1, 1, 3, 1, -3, 1, -3, 1, 1, -1, -1, 3, -1, -3, 3, -3,
        -3, -3, 1, 1
    ]),
    '6':
    np.array([
        1, 1, -1, -1, 3, -3, -3, 3, -3, 1, -1, -1, 1, -1, 1, 1, -1, -3, -1, 1,
        -1, 3, -1, -3
    ]),
    '7':
    np.array([
        -3, 3, 3, -1, -1, -3, -1, 3, 1, 3, 1, 3, 1, 1, -1, 3, 1, -1, 1, 3, -3,
        -1, -1, 1
    ]),
    '8':
    np.array([
        -3, 1, 3, -3, 1, -1, -3, 3, -3, 3, -1, -1, -1, -1, 1, -3, -3, -3, 1,
        -3, -3, -3, 1, -3
    ]),
    '9':
    np.array([
        1, 1, -3, 3, 3, -1, -3, -1, 3, -3, 3, 3, 3, -1, 1, 1, -3, 1, -1, 1, 1,
        -3, 1, 1
    ]),
    '10':
    np.array([
        -1, 1, -3, -3, 3, -1, 3, -1, -1, -3, -3, -3, -1, -3, -3, 1, -1, 1, 3,
        3, -1, 1, -1, 3
    ]),
    '11':
    np.array([
        1, 3, 3, -3, -3, 1, 3, 1, -1, -3, -3, -3, 3, 3, -3, 3, 3, -1, -3, 3,
        -1, 1, -3, 1
    ]),
    '12':
    np.array([
        1, 3, 3, 1, 1, 1, -1, -1, 1, -3, 3, -1, 1, 1, -3, 3, 3, -1, -3, 3, -3,
        -1, -3, -1
    ]),
    '13':
    np.array([
        3, -1, -1, -1, -1, -3, -1, 3, 3, 1, -1, 1, 3, 3, 3, -1, 1, 1, -3, 1, 3,
        -1, -3, 3
    ]),
    '14':
    np.array([
        -3, -3, 3, 1, 3, 1, -3, 3, 1, 3, 1, 1, 3, 3, -1, -1, -3, 1, -3, -1, 3,
        1, 1, 3
    ]),
    '15':
    np.array([
        -1, -1, 1, -3, 1, 3, -3, 1, -1, -3, -1, 3, 1, 3, 1, -1, -3, -3, -1, -1,
        -3, -3, -3, -1
    ]),
    '16':
    np.array([
        -1, -3, 3, -1, -1, -1, -1, 1, 1, -3, 3, 1, 3, 3, 1, -1, 1, -3, 1, -3,
        1, 1, -3, -1
    ]),
    '17':
    np.array([
        1, 3, -1, 3, 3, -1, -3, 1, -1, -3, 3, 3, 3, -1, 1, 1, 3, -1, -3, -1, 3,
        -1, -1, -1
    ]),
    '18':
    np.array([
        1, 1, 1, 1, 1, -1, 3, -1, -3, 1, 1, 3, -3, 1, -3, -1, 1, 1, -3, -3, 3,
        1, 1, -3
    ]),
    '19':
    np.array([
        1, 3, 3, 1, -1, -3, 3, -1, 3, 3, 3, -3, 1, -1, 1, -1, -3, -1, 1, 3, -1,
        3, -3, -3
    ]),
    '20':
    np.array([
        -1, -3, 3, -3, -3, -3, -1, -1, -3, -1, -3, 3, 1, 3, -3, -1, 3, -1, 1,
        -1, 3, -3, 1, -1
    ]),
    '21':
    np.array([
        -3, -3, 1, 1, -1, 1, -1, 1, -1, 3, 1, -3, -1, 1, -1, 1, -1, -1, 3, 3,
        -3, -1, 1, -3
    ]),
    '22':
    np.array([
        -3, -1, -3, 3, 1, -1, -3, -1, -3, -3, 3, -3, 3, -3, -1, 1, 3, 1, -3, 1,
        3, 3, -1, -3
    ]),
    '23':
    np.array([
        -1, -1, -1, -1, 3, 3, 3, 1, 3, 3, -3, 1, 3, -1, 3, -1, 3, 3, -3, 3, 1,
        -1, 3, 3
    ]),
    '24':
    np.array([
        1, -1, 3, 3, -1, -3, 3, -3, -1, -1, 3, -1, 3, -1, -1, 1, 1, 1, 1, -1,
        -1, -3, -1, 3
    ]),
    '25':
    np.array([
        1, -1, 1, -1, 3, -1, 3, 1, 1, -1, -1, -3, 1, 1, -3, 1, 3, -3, 1, 1, -3,
        -3, -1, -1
    ]),
    '26':
    np.array([
        -3, -1, 1, 3, 1, 1, -3, -1, -1, -3, 3, -3, 3, 1, -3, 3, -3, 1, -1, 1,
        -3, 1, 1, 1
    ]),
    '27':
    np.array([
        -1, -3, 3, 3, 1, 1, 3, -1, -3, -1, -1, -1, 3, 1, -3, -3, -1, 3, -3, -1,
        -3, -1, -3, -1
    ]),
    '28':
    np.array([
        -1, -3, -1, -1, 1, -3, -1, -1, 1, -1, -3, 1, 1, -3, 1, -3, -3, 3, 1, 1,
        -1, 3, -1, -1
    ]),
    '29':
    np.array([
        1, 1, -1, -1, -3, -1, 3, -1, 3, -1, 1, 3, 1, -1, 3, 1, 3, -3, -3, 1,
        -1, -1, 1, 3
    ])
}


class RootSequence:
    """
    Class representing the root sequence of the reference signals.

    The root sequence is generated using two possible formulas, one used
    for a sequence size smaller than :math:`3M_{sc}^{RS}` and another for
    sequence size equal to or greater than :math:`3M_{sc}^{RS}`, where
    :math:`3M_{sc}^{RS}` is the number of subcarriers in a PRB (12
    subcarriers).

    Parameters
    ----------
    root_index : int
        The SRS root sequence index.
    size : int
        The size of the extended Zadoff-Chu sequence. If None then the
        sequence will not be extended and will thus have a size equal
        to Nzc.
    Nzc : int
        The size of the Zadoff-Chu sequence (without any extension). If not
        provided then the largest prime number lower than or equal to
        `size` will be used.
    """
    n_sc_PRB = 12  # Number of subcarriers in a PRB in LTE

    def __init__(self,
                 root_index: int,
                 size: Optional[int] = None,
                 Nzc: Optional[int] = None) -> None:
        if size is None and Nzc is None:
            raise AttributeError("Either 'size' or 'Nzc' (or both) must "
                                 "be provided.")
        if size is None:
            size = Nzc
        assert (isinstance(size, int))

        if Nzc is None:
            Nzc = self._get_largest_prime_lower_than_number(size)

        if size < Nzc:
            raise AttributeError("If 'size' and Nzc are provided, "
                                 "then size must be greater than Nzc")

        self._root_index = root_index
        self._seq_array: np.ndarray = None
        self._extended_seq_array: np.ndarray = None  # Extended Zadoff-Chu sequence

        if size > 2 * self.n_sc_PRB:
            # If size is greater then 2 * n_sc_PRB, the root
            # sequence is an extended Zadoff-Chu sequence. First let's
            # compute the Zadoff-Chu sequence.
            self._seq_array = calcBaseZC(Nzc, root_index)  # Zadoff-Chu seq

            # Now, if size is greater than Nzc, let's compute the cyclic
            # extension of the root sequence.
            if size > Nzc:
                self._extended_seq_array = get_extended_ZF(
                    self._seq_array, size)
        else:  # size must be either n_sc_PRB or 2*n_sc_PRB
            if size == self.n_sc_PRB:
                self._seq_array = np.exp(1j * (np.pi / 4.0) *
                                         ROOT_TABLE1['{0}'.format(root_index)])
            elif size == 2 * self.n_sc_PRB:
                self._seq_array = np.exp(1j * (np.pi / 4.0) *
                                         ROOT_TABLE2['{0}'.format(root_index)])
            else:
                raise AttributeError("Invalid root sequence size")

    @staticmethod
    def _get_largest_prime_lower_than_number(seq_size: int) -> int:
        """
        Get the largest prime number lower than `seq_size`.

        Parameters
        ----------
        seq_size : int
            The sequence size.

        Returns
        -------
        int
            The largest prime number lower than `seq_size`.
        """
        p = _SMALL_PRIME_LIST[_SMALL_PRIME_LIST <= seq_size][-1]
        return int(p)

    @property
    def Nzc(self) -> int:
        """
        Get the size of the Zadoff-Chu sequence (without any extension).

        Returns
        -------
        int
            The value of the Nzc property.
        """
        return cast(int, self._seq_array.size)

    @property
    def size(self) -> int:
        """
        Return the size (with extension) of the sequence.

        If the sequence is not extended than `size()` will return the same
        as `Nzc`.

        Returns
        -------
        size : int
            The size of the extended Zadoff-Chu sequence.

        Examples
        --------
        >>> seq1 = RootSequence(root_index=25, Nzc=139)
        >>> seq1.size
        139
        >>> seq1 = RootSequence(root_index=25, Nzc=139, size=150)
        >>> seq1.size
        150
        """
        if self._extended_seq_array is None:
            return self.Nzc

        return cast(int, self._extended_seq_array.size)

    @property
    def index(self) -> int:
        """
        Return the SRS root sequence index.

        Returns
        -------
        int
            The root sequence index.
        """
        return self._root_index

    def seq_array(self) -> np.ndarray:
        """
        Get the extended Zadoff-Chu root sequence as a numpy array.

        Returns
        -------
        seq : np.ndarray
            The extended Zadoff-Chu sequence
        """
        if self._extended_seq_array is None:
            return self._seq_array

        return self._extended_seq_array

    # xxxxxxxxxx Define some basic methods xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # We can always just get the equivalent numpy array and perform the
    # operations on it, but having these operations defined here is
    # convenient

    # TODO: Make these operation methods (add, mul, etc) also work with
    # RootSequence objects returning a new RootSequence object. Change the
    # docstring type information when you do that.
    def __add__(self, other: np.ndarray) -> np.ndarray:  # pragma: no cover
        """
        Perform addition with `other`.

        Parameters
        ----------
        other : np.ndarray

        Returns
        -------
        np.ndrray
        """
        return self.seq_array() + other

    def __radd__(self, other: np.ndarray) -> np.ndarray:  # pragma: no cover
        """
        Perform addition with `other`.

        Parameters
        ----------
        other : np.ndarray

        Returns
        -------
        np.ndrray
        """
        return self.seq_array() + other

    def __mul__(self, other: np.ndarray) -> np.ndarray:  # pragma: no cover
        """
        Perform multiplication with `other`.

        Parameters
        ----------
        other : np.ndarray

        Returns
        -------
        np.ndrray
        """
        return self.seq_array() * other

    def __rmul__(self, other: np.ndarray) -> np.ndarray:  # pragma: no cover
        """
        Perform multiplication with `other`.

        Parameters
        ----------
        other : np.ndarray

        Returns
        -------
        np.ndrray
        """
        return self.seq_array() * other

    def __getitem__(self, val: Indexes) -> np.ndarray:
        """
        Index the sequence.

        This will simply return the same indexing of the underlying numpy
        array.

        Parameters
        ----------
        val : Indexes
            Anything accepted as indexing by numpy arrays.

        Returns
        -------
        np.ndarray
        """
        return self.seq_array()[val]

    def conjugate(self) -> np.ndarray:  # pragma: no cover
        """
        Return the conjugate of the root sequence as a numpy array.

        Returns
        -------
        np.ndarray
            The conjugate of the root sequence.
        """
        return self.seq_array().conj()

    def conj(self) -> np.ndarray:  # pragma: no cover
        """
        Return the conjugate of the root sequence as a numpy array.

        Returns
        -------
        np.ndarray
            The conjugate of the root sequence.
        """

        return self.seq_array().conj()

    def __repr__(self) -> str:  # pragma: no cover
        """
        Get the representation of the object.

        Returns
        -------
        str
            The representation of the object.
        """
        if self._extended_seq_array is None:
            return ("<SrsRootSequence("
                    "root_index={0},Nzc={1})>").format(self._root_index,
                                                       self._seq_array.size)

        return ("<SrsRootSequence("
                "root_index={0},size={2},Nzc={1})>").format(
                    self._root_index, self._seq_array.size,
                    self._extended_seq_array.size)

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
