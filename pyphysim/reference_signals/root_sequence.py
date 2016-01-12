#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module with Sounding Reference Signal (SRS) related functions"""

import numpy as np

from .zadoffchu import calcBaseZC, get_extended_ZF

__all__ = ['RootSequence']


class RootSequence(object):
    """
    Class representing the root sequence of the reference signals.

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
    # List of prime numbers lower than 282.
    _SMALL_PRIME_LIST = np.array(
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
         61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
         137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
         199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271,
         277, 281])

    def __init__(self, root_index, size=None, Nzc=None):
        if size is None and Nzc is None:
            raise AttributeError("Either 'size' or 'Nzc' (or both) must "
                                 "be provided.")
        if Nzc is None:
            Nzc = self._get_largest_prime_lower_than_number(size)

        self._root_index = root_index
        self._zf_seq_array = calcBaseZC(Nzc, root_index)  # Zadoff-Chu seq
        self._extended_zf_seq_array = None  # Extended Zadoff-Chu sequence

        if size is not None and Nzc is not None:
            if size <= Nzc:
                raise AttributeError("If 'size' and Nzc are provided, "
                                     "then size must be greater than Nzc")
            else:
                self._extended_zf_seq_array = get_extended_ZF(
                    self._zf_seq_array, size)

    @classmethod
    def _get_largest_prime_lower_than_number(cls, seq_size):
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
        p = cls._SMALL_PRIME_LIST[cls._SMALL_PRIME_LIST <= seq_size][-1]
        return p

    @property
    def Nzc(self):
        """
        Get the size of the Zadoff-Chu sequence (without any extension).

        Returns
        -------
        int
            The value of the Nzc property.
        """
        return self._zf_seq_array.size

    @property
    def size(self):
        """
        Return the size (with extension) of the sequence.

        If the sequence is not extended than `size()` will return the same
        as `Nzc`.

        Returns
        -------
        size : int
            The size of the extended Zadoff-Chu sequence.

        Example
        -------
        >>> seq1 = RootSequence(root_index=25, Nzc=139)
        >>> seq1.size
        139
        >>> seq1 = RootSequence(root_index=25, Nzc=139, size=150)
        >>> seq1.size
        150
        """
        if self._extended_zf_seq_array is None:
            return self.Nzc
        else:
            return self._extended_zf_seq_array.size

    @property
    def index(self):
        """
        Return the SRS root sequence index.

        Returns
        -------
        int
            The root sequence index.
        """
        return self._root_index

    def seq_array(self):
        """
        Get the extended Zadoff-Chu root sequence as a numpy array.

        Returns
        -------
        seq : np.ndarray
            The extended Zadoff-Chu sequence
        """
        if self._extended_zf_seq_array is None:
            return self._zf_seq_array
        else:
            return self._extended_zf_seq_array

    # xxxxxxxxxx Define some basic methods xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # We can always just get the equivalent numpy array and perform the
    # operations on it, but having these operations defined here is
    # convenient

    # TODO: Make these operation methods (add, mul, etc) also work with
    # RootSequence objects returning a new RootSequence object. Change the
    # docstring type information when you do that.
    def __add__(self, other):  # pragma: no cover
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

    def __radd__(self, other):  # pragma: no cover
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

    def __mul__(self, other):  # pragma: no cover
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

    def __rmul__(self, other):  # pragma: no cover
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

    def conjugate(self):  # pragma: no cover
        """
        Return the conjugate of the root sequence as a numpy array.

        Returns
        -------
        np.ndarray
            The conjugate of the root sequence.
        """
        return self.seq_array().conj()

    def conj(self):  # pragma: no cover
        """
        Return the conjugate of the root sequence as a numpy array.

        Returns
        -------
        np.ndarray
            The conjugate of the root sequence.
        """

        return self.seq_array().conj()

    def __repr__(self):
        """
        Get the representation of the object.

        Returns
        -------
        str
            The representation of the object.
        """
        if self._extended_zf_seq_array is None:
            return ("<SrsRootSequence("
                    "root_index={0},Nzc={1})>").format(
                        self._root_index,
                        self._zf_seq_array.size)
        else:
            return ("<SrsRootSequence("
                    "root_index={0},size={2},Nzc={1})>").format(
                        self._root_index, self._zf_seq_array.size,
                        self._extended_zf_seq_array.size)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
