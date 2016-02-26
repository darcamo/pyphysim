#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module with Sounding Reference Signal (SRS) related functions"""

import numpy as np

from .root_sequence import RootSequence
from .zadoffchu import get_shifted_root_seq

__all__ = ['get_srs_seq', 'SrsUeSequence']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Module Functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def get_srs_seq(root_seq, n_cs):
    """
    Get the shifted root sequence suitable as the SRS sequence of a user.

    Parameters
    ----------
    root_seq : np.ndarray
        The root sequence to shift. This is a complex numpy array.
    n_cs : int
        The desired cyclic shift number. This should be an integer from 0
        to 7, where 0 will just return the base sequence, 1 gives the first
        shift, and so on.

    Returns
    -------
    np.ndarray
        The shifted root sequence.

    See Also
    --------
    get_shifted_root_seq, get_dmrs_seq
    """
    return get_shifted_root_seq(root_seq, n_cs, 8)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Classes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class UeSequence(object):
    """
    Reference signal sequence of a single user.

    You should not use this class directly and instead use a class that
    inherits from it and provides the desired reference sequence.

    Parameters
    ----------
    root_seq : RootSequence
        The root sequence of the base station the user is
        associated to. This should be an object of the RootSequence
        class.
    n_cs : int
        The shift index of the user. This can be an integer from 1 to 8.
    user_seq_array : np.ndarray
        The user sequence.
    """
    def __init__(self, root_seq, n_cs, user_seq_array):
        self._user_seq_array = user_seq_array
        self._n_cs = n_cs
        self._root_index = root_seq.index

    @property
    def size(self):
        """
        Return the size of the reference signal sequence.

        Returns
        -------
        size : int
            The size of the user's reference signal sequence.

        Example
        -------
        >>> root_seq1 = RootSequence(root_index=25, Nzc=139)
        >>> user_seq1 = SrsUeSequence(root_seq1, 3)
        >>> user_seq1.size
        139
        >>> root_seq2 = RootSequence(root_index=25, Nzc=139, size=150)
        >>> user_seq2 = SrsUeSequence(root_seq2, 3)
        >>> user_seq2.size
        150
        """
        return self._user_seq_array.size

    def seq_array(self):
        """
        Get reference signal sequence as a numpy array.

        Returns
        -------
        seq : np.ndarray
            The user's reference signal sequence.
        """
        return self._user_seq_array

    def __repr__(self):  # pragma: no cover
        """
        Get the representation of the object.

        Returns
        -------
        str
            The representation of the object.
        """
        return "<{0}(root_index={1}, n_cs={2})>".format(
            self.__class__.__name__, self._root_index, self._n_cs)

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
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class SrsUeSequence(UeSequence):
    """
    SRS sequence of a single user.

    Parameters
    ----------
    root_seq : RootSequence
        The SRS root sequence of the base station the user is
        associated to. This should be an object of the RootSequence
        class.
    n_cs : int
        The shift index of the user. This can be an integer from 0 to 7.
    """
    def __init__(self, root_seq, n_cs):
        root_seq_array = root_seq.seq_array()
        user_seq_array = get_srs_seq(root_seq_array, n_cs)
        super(SrsUeSequence, self).__init__(root_seq, n_cs, user_seq_array)
