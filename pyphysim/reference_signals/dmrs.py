#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module with Sounding Reference Signal (SRS) related functions"""

import numpy as np

from pyphysim.reference_signals.srs import UeSequence
from .zadoffchu import get_shifted_root_seq
from .root_sequence import RootSequence

__all__ = ['get_dmrs_seq', 'DmrsUeSequence']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Module Functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def get_dmrs_seq(root_seq, n_cs):
    """
    Get the shifted root sequence suitable as the DMRS sequence of a user.

    Parameters
    ----------
    root_seq : np.ndarray
        The root sequence to shift.
    n_cs : int
        The desired cyclic shift number. This should be an integer from
        0 to 11, where 0 will just return the base sequence, 1 gives the
        first shift, and so on.

    Returns
    -------
    np.ndarray
        The shifted root sequence.

    See Also
    --------
    .zadoffchu.get_shifted_root_seq, .srs.get_srs_seq
    """
    return get_shifted_root_seq(root_seq, n_cs, 12)


class DmrsUeSequence(UeSequence):
    """
    DMRS sequence of a single user.

    Parameters
    ----------
    root_seq : RootSequence
        The DMRS root sequence of the base station the user is
        associated to. This should be an object of the RootSequence
        class.
    n_cs : int
        The shift index of the user. This can be an integer from 0 to 11.
    cover_code : np.ndarray, optional
        Cover Code used by the UE. As an example, consider the cover code
        `np.array([1, -1])`. In that case, if the regular DMRS sequence
        (without the cover code) is `seq`, than the actual DMRS sequence
        with cover code will be a 2D numpy array equivalent with
        `seq_occ[0]==seq` and `seq_occ[1]==-seq`.
    normalize : bool
        True if the reference signal should be normalized. False otherwise.
    """

    def __init__(self, root_seq, n_cs, cover_code=None, normalize=False):
        root_seq_array = root_seq.seq_array()
        user_seq_array = get_dmrs_seq(root_seq_array, n_cs)

        # Orthogonal Cover Code. This is stored in an attribute only for
        # visualization purposes, since the stored user_seq_array will
        # already include its effect.
        self._occ = cover_code = cover_code

        if cover_code is not None:
            self._occ.flags.writeable = False
            user_seq_array = user_seq_array * cover_code[:, np.newaxis]

        super(DmrsUeSequence, self).__init__(root_seq,
                                             n_cs,
                                             user_seq_array,
                                             normalize=normalize)

    @property
    def cover_code(self):
        """Return the cover code."""
        return self._occ

    @property
    def size(self):
        """
        Return the size of the reference signal sequence.

        Returns
        -------
        size : int
            The size of the user's reference signal sequence.
        """
        if self._occ is None:
            return self._user_seq_array.shape[0]
        else:
            return self._user_seq_array.shape[1]

    def __repr__(self):
        """
        Get the representation of the object.

        Returns
        -------
        str
            The representation of the object.
        """
        return "<{0}(root_index={1}, n_cs={2}, cover_code={3})>".format(
            self.__class__.__name__, self._root_index, self._n_cs, self._occ)
