#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module with Sounding Reference Signal (SRS) related functions"""

import numpy as np

from pyphysim.reference_signals.srs import UeSequence
from .zadoffchu import get_shifted_root_seq
from .root_sequence import RootSequence

__all__ = ['get_dmrs_seq']


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
    get_shifted_root_seq, get_srs_seq
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
    """
    def __init__(self, root_seq, n_cs):
        root_seq_array = root_seq.seq_array()
        user_seq_array = get_dmrs_seq(root_seq_array, n_cs)
        super(DmrsUeSequence, self).__init__(
                root_seq, n_cs, user_seq_array)
