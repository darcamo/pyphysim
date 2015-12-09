#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module with Sounding Reference Signal (SRS) related functions"""

import numpy as np

from pyphysim.reference_signals.zadoffchu import get_shifted_root_seq

__all__ = ['get_shifted_dmrs_seq']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Module Functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def get_shifted_dmrs_seq(root_seq, n_cs):
    """
    Get the shifted root sequence suitable as the DMRS sequence of a user.

    Parameters
    ----------
    root_seq : complex numpy array
        The root sequence to shift.
    n_cs : int
        The desired cyclic shift number. This should be an integer from 0
        to 11, where 0 will just return the base sequence, 1 gives the first
        shift, and so on.

    Returns
    -------
    numpy array
        The shifted root sequence.

    See Also
    --------
    get_shifted_root_seq, get_shifted_srs_seq
    """
    return get_shifted_root_seq(root_seq, n_cs, 12)


class DmrsUeSequence(object):
    pass
