#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implement classes for several Path loss models.

The base class PathLossBase implements the code common to every path loss
model and only two methods need to be implemented in subclasses:
which_distance_dB and _calc_deterministic_path_loss_dB.

The most common usage of a path loss class is to instantiate an object of
the desired path loss model and then call the calc_path_loss_dB or the
calc_path_loss methods to actually calculate the path loss.

.. TODO:: Verify the equations in the docstrings

"""

__revision__ = "$Revision$"

try:
    from matplotlib import pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

import warnings
import numpy as np
from collections import Iterable

from ..util import conversion

__all__ = ['PathLossBase', 'PathLossFreeSpace', 'PathLoss3GPP1',
           'PathLossOkomuraHata']


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

    # xxxxx Start - Implemented these functions in subclasses xxxxxxxxxxxxx
    def which_distance_dB(self, PL):
        """Calculates the distance that yields the given path loss (in dB).

        Parameters
        ----------
        PL : float or numpy array
            Path Loss (in dB)

        Raises
        ------
        NotImplementedError
            If the which_distance_dB method of the PathLossBase class is
            called.

        """
        # Raises an exception if which_distance_dB is not implemented in a
        # subclass
        msg = 'which_distance_dB must be reimplemented in the {0} class'
        raise NotImplementedError(msg.format(self.__class__.__name__))

    def _calc_deterministic_path_loss_dB(self, d):
        """Calculates the Path Loss (in dB) for a given distance (in Km)
        without including the shadowing.

        Parameters
        ----------
        d : float or numpy array
            Distance (in Km)

        Raises
        ------
        NotImplementedError
            If the _calc_deterministic_path_loss_dB method of the
            PathLossBase class is called.

        """
        msg = ('_calc_deterministic_path_loss_dB must be reimplemented in '
               'the {0} class')
        raise NotImplementedError(msg.format(self.__class__.__name__))
    # xxxxx End - Implemented these functions in subclasses xxxxxxxxxxxxxxx

    def plot_deterministic_path_loss_in_dB(
            self, d, ax=None, extra_args=None):  # pragma: no cover
        """
        Parameters
        ----------
        d : float or numpy array
            Distance (in Km)
        ax : A matplotlib ax, optional
            The ax where the path loss will be plotted. If not provided, a
            new figure (and ax) will be created.
        extra_args : dict
            Extra arguments that will be passed to the ax.plot command as
            "**extra_args" (see Matplotlib documentation).
            Ex: {'label': 'curve name', 'linewidth': 2}
        """
        if extra_args is None:
            extra_args = {}

        stand_alone_plot = False

        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            ax = plt.axes()
            stand_alone_plot = True

        PL = self._calc_deterministic_path_loss_dB(d)
        ax.plot(d, PL, **extra_args)

        if stand_alone_plot is True:
            ax.set_ylabel('Path Loss (in dB)')
            ax.set_xlabel('Distance (in Km)')
            ax.grid(True)
            plt.show()

    def calc_path_loss_dB(self, d):
        """Calculates the Path Loss (in dB) for a given distance (in
        kilometers).

        Note that the returned value is positive, but should be understood
        as "a loss".

        Parameters
        ----------
        d : float or numpy array
            Distance (in Km)

        Returns
        -------
        PL : float or numpy array
            Path loss (in dB) for the given distance(s).
        """
        PL = self._calc_deterministic_path_loss_dB(d)
        if self.use_shadow_bool is True:  # pragma: no cover
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

        # The calculated path loss (in dB) must be positive. If it is not
        # positive that means that the distance 'd' is too small.
        if np.any(np.array(PL) < 0):
            msg = "The distance is too small to calculate a valid path loss."
            raise RuntimeError(msg.format(d))
        return PL

    def calc_path_loss(self, d):
        """Calculates the path loss (in linear scale) for a given distance
        (in kilometers).

        Parameters
        ----------
        d : float or numpy array
            Distance (in Km)

        Returns
        -------
        pl : float or numpy array
            Path loss (in linear scale) for the given distance(s).
        """
        pl = conversion.dB2Linear(-self.calc_path_loss_dB(d))
        return pl

    def which_distance(self, pl):
        """Calculates the required distance (in kilometers) to achieve the
        given path loss. It is the inverse of the calcPathLoss function.

        Parameters
        ----------
        pl : float or numpy array
            Path loss (in linear scale).

        Returns
        -------
        d : float or numpy array
            Distance(s) that will yield the path loss `pl`.
        """
        d = self.which_distance_dB(-conversion.linear2dB(pl))
        return d


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
    defaults to 900 (that is, 900MHz).

    Examples
    --------
    Determining the path loss in the free space for a distance of 1Km
    (without considering shadowing).

    >>> pl = PathLossFreeSpace()
    >>> pl.calc_path_loss(1)        # linear scale
    7.036193308495632e-10
    >>> pl.calc_path_loss_dB(1)     # log scale
    91.526622374835199

    Determining the distance (in Km) that yields a path loss of 90dB.

    >>> pl.which_distance_dB(90)
    0.83882020174144778

    """

    def __init__(self):
        PathLossBase.__init__(self)
        self.n = 2     # Path Loss Coefficient
        self.fc = 900  # Frequency of the central carrier (in MHz)

    def which_distance_dB(self, PL):
        """Calculates the required distance (in kilometers) to achieve the
        given path loss (in dB).

        It is the inverse of the calcPathLoss function.

        :math:`10^{(PL/(10n) - \\log_{10}(fc) + 4.377911390697565)}`

        d = obj.whichDistance(dB2Linear(-PL));

        Parameters
        ----------
        PL : float of numpy array
            Path Loss (in dB).

        Returns
        -------
        d : float of numpy array
            Distance (in Km).

        """
        # $10^{(PL/(10n) - \log_{10}(fc) + 4.377911390697565)}$
        # Note: the value "6.0" was subtracted to account the fact that
        # self.fc is in MHz.
        d = (10. ** (PL / (10. * self.n)
                     - np.log10(self.fc) - 6.0 + 4.377911390697565))
        return d

    def _calc_deterministic_path_loss_dB(self, d):
        """Calculates the Path Loss (in dB) for a given distance (in
        kilometers).

        Note that the returned value is positive, but should be understood
        as "a loss".

        For d in Km and self.fc in MHz, the free space Path Loss is given by
        PL = 10 n ( log10(d) +log10(f * 1e6) - 4.3779113907 )

        Parameters
        ----------
        d : float or numpy array
            Distance (in Km).

        Returns
        -------
        pl_dB : float or numpy array
            Path loss in dB.
        """
        # The value "6.0" was added to convert the frequency self.fc to MHz
        PL = (10 * self.n
              * (np.log10(d) + np.log10(self.fc) + 6.0 - 4.377911390697565))
        return PL


class PathLoss3GPP1(PathLossBase):
    """
    Class to calculate the Path Loss according to the model from 3GPP
    (scenario 1). That is, the Path Loss (in dB) is equal to

    $128.1 + 37.6*\log10(d)$.

    This model is valid for LTE assumptions and at 2GHz frequency, where
    the distance is in Km.

    Examples
    --------
    Determining the path loss in the free space for a distance of 1Km
    (without considering shadowing).

    >>> pl = PathLoss3GPP1()
    >>> pl.calc_path_loss(1)        # linear scale
    1.5488166189124858e-13
    >>> pl.calc_path_loss_dB(1)     # log scale
    128.09999999999999

    Determining the distance (in Km) that yields a path loss of 90dB.

    >>> pl.which_distance_dB(90)
    0.09698445455772262
    """

    def __init__(self):
        # super(PathLossFreeSpace, self).__init__()
        PathLossBase.__init__(self)

    def which_distance_dB(self, PL):
        """Calculates the required distance (in kilometers) to achieve the
        given path loss (in dB).

        It is the inverse of the calcPathLoss function.

        Parameters
        ----------
        PL : float or numpy array
            Path loss (in dB).

        Returns
        -------
        d : float of numpy array
            Distance (in Km).
        """
        d = 10 ** ((PL - 128.1) / 37.6)
        return d

    def _calc_deterministic_path_loss_dB(self, d):
        """
        Calculates the Path Loss (in dB) for a given distance (in kilometers).

        Note that the returned value is positive, but should be understood
        as "a loss".

        For d in Km and self.fc in Hz, the free space Path Loss is given by
        PL = 10n ( log10(d) +log10(f) - 4.3779113907 )

        Parameters
        ----------
        d : float or numpy array
            Distance (in Km).

        Returns
        -------
        PL : float
            Path loss in dB.
        """
        PL = 128.1 + 37.6 * np.log10(d)
        return PL


# TODO: Test this class
# See http://w3.antd.nist.gov/wctg/manet/calcmodels_r1.pdf
class PathLossOkomuraHata(PathLossBase):
    """
    Class to calculate the Path Loss according to the Okomura Hata model.

    The exact formula depend on the area type, but in general the path loss
    is given by (in dB):

    $$PL = 69.55 + 26.16 * \log(fc) - 13.82*\log(h_{bs}) - a(h_{ms}) + (44.9 - 6.55\log(h_{bs})) \log(d) - K$$

    The term '$a(h_{ms})$' is the mobile station correction factor (see
    `_calc_mobile_antenna_height_correction_factor`).

    The term 'K' is a correction factor that depends on the area type (see
    `_calc_K`).

    The possible area types are (in ascending order):
    'open', 'suburban', 'medium city' and 'large city'.
    """

    # A=69.55+26.16*log10(fc)-13.82*log10(hb)-a_hm;
    # B=44.9-6.55*log10(hb);

    # C=5.4+2*(log10(fc/28))^2;
    # D=40.94+4.78*(log10(fc))^2-18.33*log10(fc);

    # The equation below is valid for all area types, except the 'large city'
    # a_hm=(1.1*log10(fc)-0.7)*hm-(1.56*log10(fc)-0.8);

    # Lp_urban = A + B*log10(r);
    # Lp_suburban = A + B*log10(r)-C;
    # Lp_open = A + B*log10(r)-D;

    # f in MHz
    # d in Km

    # $L (\text{in dB}) = 69.55 + 26.16 \log(f) -13.82 \log(h_{bs}) - a(h_{ms}) + (44.9 - 6.55\log(h_{bs})) \log(d) - K$

    # d should be between 1Km and 20Km
    def __init__(self):
        # super(PathLossFreeSpace, self).__init__()
        PathLossBase.__init__(self)

        # Height of the Base Station (in meters) -> 30m to 200m
        self._hbs = 30.0

        # Height of the Mobile Station (in meters) - 1m to 10m
        self._hms = 1.0

        # Frequency of the central carrier (in MHz) - 150MHz to 1500MHz
        self._fc = 900

        # area_type can be 'open', 'suburban', 'medium city', 'large city'.
        # Note: The category of “large city” used by Hata implies building
        # heights greater than 15m.
        self._area_type = 'suburban'

    # xxxxxxxxxx fc property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def fc(self):
        """Get method for the fc property."""
        return self._fc

    @fc.setter
    def fc(self, value):
        """Set method for the fc property."""
        if value < 150.0 or value > 1500:
            msg = ("The carrier frequency for the Okomura Hata model must be"
                   " between 150 and 1500 (values in MHz).")
            raise RuntimeError(msg)
        self._fc = value
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx hbs property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def hbs(self):
        """Get method for the hbs property."""
        return self._hbs

    @hbs.setter
    def hbs(self, value):
        """Set method for the hbs property."""
        if value < 30.0 or value > 200.0:
            msg = ("The Base Station antenna height for the Okomura Hata "
                   "model must be between 30 and 200 (values in meters).")
            raise RuntimeError(msg)
        self._hbs = value
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx hms property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def hms(self):
        """Get method for the hms property."""
        return self._hms

    @hms.setter
    def hms(self, value):
        """Set method for the hms property."""
        if value < 1.0 or value > 10.0:
            msg = ("The Mobile Station antenna height for the Okomura Hata "
                   "model must be between 1 and 10 (values in meters).")
            raise RuntimeError(msg)
        self._hms = value
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx area_type property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def area_type(self):
        """Get method for the area_type property."""
        return self._area_type

    @area_type.setter
    def area_type(self, value):
        """Set method for the area_type property."""
        if value not in ['open', 'suburban', 'medium city', 'large city']:
            raise RuntimeError('Invalid area type: {0}'.format(value))
        self._area_type = value
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _calc_mobile_antenna_height_correction_factor(self):
        """
        Calculates the mobile antenna height correction factor.

        This factor is strongly impacted by surrounding buildings and is
        refined according to city sizes. For all area types, except 'large
        city', the mobile antenna correction factor is given by

        $$a(h_{ms}) = (1.1 \log(f) - 0.7) h_{ms} - 1.56 \log(f) + 0.8$$

        For the 'large city' area type, the mobile antenna height
        correction is given by

        $$a(h_{ms}) = 3.2 (\log(11.75*h_{ms})^2) - 4.97$$

        if the frequency is greater then 300MHz, or

        $$a(h_{ms}) = 8.29 (\log(1.54 h_{ms}))^2 - 1.10$$

        if the frequency is lower than 300MHz (and greater than 150MHz
        where the Okomura Hata model is valid).

        Returns
        -------
        a : float
            The mobile antenna height correction.
        """
        if self.area_type in ['open', 'suburban', 'medium city']:

            # Suburban and ruran areas (f in MHz
            # $a(h_{ms}) = (1.1 \log(f) - 0.7) h_{ms} - 1.56 \log(f) + 0.8$
            a = ((1.1 * np.log10(self.fc) - 0.7)
                 * self.hms - 1.56 * np.log10(self.fc) + 0.8)
        elif self.area_type == 'large city':
            # Note: The category of “large city” used by Hata implies
            # building heights greater than 15m.
            if self.fc > 300:
                # If frequency is greater then 300MHz then the factor is
                # given by
                # $3.2 (\log(11.75*h_{ms})^2) - 4.97$
                a = 3.2 * (np.log10(11.75 * self.hms)**2) - 4.97
            else:
                # If frequency is lower then 300MHz then the factor is
                # given by
                # $8.29 (\log(1.54 h_{ms}))^2 - 1.10$
                a = 8.29 * (np.log10(1.54 * self.hms)**2) - 1.10
        else:  # pragma: no cover
            raise RuntimeError('Invalid area type: {0}'.format(self.area_type))

        return a

    def _calc_K(self):
        """
        Calculates the "'medium city'/'suburban'/'open area'" correction
        factor.

        Returns
        -------
        K : float
            The correction factor "K".
        """
        if self.area_type == 'large city':
            K = 0
        elif self.area_type == 'open':
            # Value for 'open' areas
            # $K = 4.78 (\log(f))^2 - 18.33 \log(f) + 40.94$
            K = (4.78 * (np.log10(self.fc)**2) - 18.33 * np.log10(self.fc)
                 + 40.94)
        elif self.area_type == 'suburban':
            # Value for 'suburban' areas
            # $K = 2 [\log(f/28)^2] + 5.4$
            K = 2 * (np.log10(self.fc / 28.0)**2) + 5.4
        else:
            K = 0
        return K

    def _calc_deterministic_path_loss_dB(self, d):
        """
        Calculates the Path Loss (in dB) for a given distance (in kilometers).

        Note that the returned value is positive, but should be understood
        as "a loss".

        For d in Km and self.fc in Hz, the free space Path Loss is given by
        PL = 10n ( log10(d) +log10(f) - 4.3779113907 )

        Parameters
        ----------
        d : float or numpy array
            Distance (in Km). Can be between 1km and 20Km.

        Returns
        -------
        PL : float
            Path loss in dB.
        """
        if np.any(d < 1.0) or np.any(d > 20.0):
            msg = ('Distance for the Okomura Hata model should be between'
                   ' 1Km and 20Km')
            warnings.warn(Warning(msg))

        # $L (\text{in dB}) = 69.55 + 26.16 \log(f) -13.82 \log(h_{bs}) - a(h_{ms}) + (44.9 - 6.55\log(h_{bs})) \log(d) - K$

        # Calculate the mobile antenna height correction factor
        a = self._calc_mobile_antenna_height_correction_factor()

        # Calculates the "'suburban'/'open area'" correction factor.
        K = self._calc_K()

        L = (69.55 + 26.16 * np.log10(self.fc) - 13.82 * np.log10(self.hbs)
             - a + (44.9 - 6.55 * np.log10(self.hbs)) * np.log10(d) - K)
        return L

    def which_distance_dB(self, PL):
        """Calculates the required distance (in kilometers) to achieve the
        given path loss (in dB).

        It is the inverse of the calcPathLoss function.

        Parameters
        ----------
        PL : float or numpy array
            Path loss (in dB).

        Returns
        -------
        d : float of numpy array
            Distance (in Km).
        """
        # TODO: implement-me
        raise NotImplementedError("Implement-me")  # pragma: no cover
