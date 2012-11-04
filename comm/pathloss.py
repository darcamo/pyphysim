#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implement classes for several Path loss models.

The base class PathLossBase implements the code common to every path loss
model and only two methods need to be implemented in subclasses:
which_distance_dB and _calc_deterministic_path_loss_dB.

The most common usage of a path loss class is to instantiate an object of
the desired path loss model and then call the calc_path_loss_dB or the
calc_path_loss methods to actually calculate the path loss.
"""

import sys
sys.path.append("../")

from util import conversion
import numpy as np
from collections import Iterable


class PathLossBase(object):
    """Base class for the different Path Loss models.

    The common interface for the path loss classes is provided by the
    calc_path_loss_dB or the calc_path_loss methods to actually calculate
    the path loss for a given distance, as well as the which_distance_dB or
    which_distance methods to determine the distance that yields the given
    path loss.

    Each subclass of PathLossBase NEED TO IMPLEMENT only the
    "which_distance_dB" and the "_calc_deterministic_path_loss_dB"
    functions.

    If the use_shadow_bool is set to True then calling calc_path_loss_dB or
    calc_path_loss will take the shadowing specified in the sigma_shadow
    variable into account. However, shadowing is not taken into account in
    the which_distance_dB and which_distance functions regardless of the
    value of the use_shadow_bool variable.
    """

    def __init__(self, ):
        self.sigma_shadow = 8  # Shadow standard deviation
        self.use_shadow_bool = False

    #
    # xxxxx Start - Implemented these functions in subclasses xxxxxxxxxxxxx
    def which_distance_dB(self, PL):
        """
        Calculates the distance that yields the given path loss (in dB).

        Arguments:
        - `PL`: Path Loss (in dB)
        """
        # Raises an exception if which_distance_dB is not implemented in a
        # subclass
        raise NotImplementedError('which_distance_dB must be reimplemented in the {0} class'.format(self.__class__.__name__))

    def _calc_deterministic_path_loss_dB(self, d):
        """Calculates the Path Loss (in dB) for a given distance (in Km)
        without including the shadowing.

        Arguments:
        - `d`: Distance (in Km)
        """
        raise NotImplementedError('_calc_deterministic_path_loss_dB must be reimplemented in the {0} class'.format(self.__class__.__name__))
    # xxxxx End - Implemented these functions in subclasses xxxxxxxxxxxxxxx
    #

    def calc_path_loss_dB(self, d):
        """Calculates the Path Loss (in dB) for a given distance (in
        kilometers).

        Note that the returned value is positive, but should be understood
        as "a loss".

        Arguments:
        - `d`: Distance (in Km)
        """
        PL = self._calc_deterministic_path_loss_dB(d)
        if self.use_shadow_bool is True:
            if isinstance(d, Iterable):
                # If 'd' is a numpy array (or something similar such as a
                # list), shadow must be a numpy array with the same shape
                shadow = conversion.dB2Linear(
                    np.random.randn(np.size(d)) * self.sigma_shadow)
                shadow.shape = np.shape(d)
            else:
                # If 'd' is not an array but add a scalar shadowing
                shadow = conversion.dB2Linear(
                    np.random.randn() * self.sigma_shadow)
            PL = PL + shadow

        return PL

    def calc_path_loss(self, d):
        """Calculates the path loss (in linear scale) for a given distance
        (in kilometers).

        Arguments:
        - `d`: Distance (in Km)
        """
        pl = conversion.dB2Linear(-self.calc_path_loss_dB(d))
        return pl

    def which_distance(self, pl):
        """Calculates the required distance (in kilometers) to achieve the
        given path loss. It is the inverse of the calcPathLoss function.

        Arguments:
        - `pl`: Path loss (in linear scale)
        """
        d = self.which_distance_dB(-conversion.linear2dB(pl))
        return d
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class PathLossFreeSpace(PathLossBase):
    """Class to calculate the Path Loss in the free space.

    The common interface for the path loss classes is provided by the
    calc_path_loss_dB or the calc_path_loss methods to actually calculate
    the path loss for a given distance, as well as the which_distance_dB or
    which_distance methods to determine the distance that yields the given
    path loss.

    For the path loss in free space you also need to set the `n` variable,
    corresponding to the path loss coefficient, and the `fc` variable,
    corresponding to the frequency. The `n` variable defaults to 2 and `fc`
    defaults to 900e6 (that is, 900MHz).

    Ex: Determining the path loss in the free space for a distance of 1Km
    (without considering shadowing).

    >>> pl = PathLossFreeSpace()
    >>> pl.calc_path_loss(1)        # linear scale
    7.036193308495632e-10
    >>> pl.calc_path_loss_dB(1)     # log scale
    91.526622374835199

    Ex: Determining the distance (in Km) that yields a path loss of 90dB.
    >>> pl.which_distance_dB(90)
    0.83882020174144778
    """

    def __init__(self):
        PathLossBase.__init__(self)
        self.n = 2       # Path Loss Coefficient
        self.fc = 900e6  # Frequency of the central carrier

    def which_distance_dB(self, PL):
        """Calculates the required distance (in kilometers) to achieve the
        given path loss (in dB). It is the inverse of the calcPathLoss
        function.

        $10^(PL/(10*n) - log10(fc) + 4.377911390697565)$
        d = obj.whichDistance(dB2Linear(-PL));

        Arguments:
        - `PL`: Path Loss (in dB)
        Returns:
        - Distance (in Km)
        """
        d = 10. ** (PL / (10. * self.n) - np.log10(self.fc) + 4.377911390697565)
        return d

    def _calc_deterministic_path_loss_dB(self, d):
        """Calculates the Path Loss (in dB) for a given distance (in
        kilometers).

        Note that the returned value is positive, but should be understood
        as "a loss".

        For d in Km and self.fc in Hz, the free space Path Loss is given by
        PL = 10 n ( log10(d) +log10(f) - 4.3779113907 )

        Arguments:
        - `d`: Distance (in Km)
        """
        PL = 10 * self.n * (np.log10(d) + np.log10(self.fc) - 4.377911390697565)
        return PL


class PathLoss3GPP1(PathLossBase):
    """Class to calculate the Path Loss according to the model from 3GPP
    (scenario 1). That is, the Path Loss (in dB) is equal to
    $128.1 + 37.6*\log10(d)$.

    This model is valid for LTE assumptions and at 2GHz frequency, where
    the distance is in Km.

    Ex: Determining the path loss in the free space for a distance of 1Km
    (without considering shadowing).

    >>> pl = PathLoss3GPP1()
    >>> pl.calc_path_loss(1)        # linear scale
    1.5488166189124858e-13
    >>> pl.calc_path_loss_dB(1)     # log scale
    128.09999999999999

    Ex: Determining the distance (in Km) that yields a path loss of 90dB.
    >>> pl.which_distance_dB(90)
    0.09698445455772262
    """

    def __init__(self):
        # super(PathLossFreeSpace, self).__init__()
        PathLossBase.__init__(self)

    def which_distance_dB(self, PL):
        """Calculates the required distance (in kilometers) to achieve the
        given path loss (in dB). It is the inverse of the calcPathLoss
        function.

        Arguments:
        - `PL`: Path loss (in dB)
        Returns:
        - Distance (in Km)
        """
        d = 10 ** ((PL - 128.1) / 37.6)
        return d

    def _calc_deterministic_path_loss_dB(self, d):
        """Calculates the Path Loss (in dB) for a given distance (in
        kilometers).

        Note that the returned value is positive, but should be understood
        as "a loss".

        For d in Km and self.fc in Hz, the free space Path Loss is given by
        PL = 10n ( log10(d) +log10(f) - 4.3779113907 )

        Arguments:
        - `d`: Distance (in Km)
        """
        PL = 128.1 + 37.6 * np.log10(d)
        return PL
