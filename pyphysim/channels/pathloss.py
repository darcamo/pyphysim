#!/usr/bin/env python
"""
Implement classes for several Path loss models.

The :class:`PathLossBase` class implements the common code to every path loss model and
only two methods need to be implemented in subclasses: the
:meth:`PathLossBase.which_distance_dB` and the
:meth:`PathLossBase._calc_deterministic_path_loss_dB` methods. However, instead
of inheriting directly from :class:`PathLossBase`, inherit from either
:class:`PathLossIndoorBase` or :class:`PathLossOutdoorBase`.

The most common usage of a path loss class is to instantiate an object of
the desired path loss model and then call the **calc_path_loss_dB** or the
**calc_path_loss** methods to actually calculate the path loss.
"""

try:
    # noinspection PyUnresolvedReferences
    from matplotlib import pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

import math
import warnings
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable
from typing import Any, Optional, TypeVar

import numpy as np

from pyphysim.util import conversion

__all__ = [
    'PathLossBase', 'PathLossIndoorBase', 'PathLossOutdoorBase',
    'PathLossGeneral', 'PathLossFreeSpace', 'PathLoss3GPP1',
    'PathLossMetisPS7', 'PathLossOkomuraHata'
]

NumberOrArray = TypeVar("NumberOrArray", np.ndarray, float)


class PathLossBase:
    """
    Base class for the different Path Loss models.

    The common interface for the path loss classes is provided by the
    :meth:`calc_path_loss_dB` or the :meth:`calc_path_loss` methods to
    actually calculate the path loss for a given distance, as well as the
    :meth:`which_distance_dB` or :meth:`which_distance` methods to
    determine the distance that yields the given path loss.

    Each subclass of PathLossBase NEED TO IMPLEMENT only the
    :meth:`which_distance_dB` and the
    :meth:`_calc_deterministic_path_loss_dB` functions.

    If the `use_shadow_bool` attribute is set to True then calling
    :meth:`calc_path_loss_dB` or :meth:`calc_path_loss` will take the
    shadowing specified in the `sigma_shadow` attribute into
    account. However, shadowing is not taken into account in the
    :meth:`which_distance_dB` and :meth:`which_distance` functions,
    regardless of the value of the `use_shadow_bool` variable.

    Attributes
    ----------
    sigma_shadow : int
        The shadowing (in dB)
    use_shadow_bool : bool
        If True then shadowing will be used.
    handle_small_distances_bool : bool
        If this is True then any negative path loss (in dB) that appears
        because a distance is too small will considered as 0dB. If this s
        False then an exception will be raised instead.
    """
    # The PathLossBase class is an abstract class and all methods marked as
    # 'abstract' must be implemented in a subclass.
    __metaclass__ = ABCMeta

    # Path loss type, such as 'indoor', 'outdoor', 'outdoor2indoor',
    # etc. This should be set in subclasses appropriately. This is useful
    # mainly for introspection.
    _TYPE = 'base'

    def __init__(self) -> None:
        self.sigma_shadow: float = 8.0  # Shadowing standard deviation
        self.use_shadow_bool: bool = False  # True if shadowing should be used

        # If this is True, then any negative path loss (in dB) that appears
        # because a distance is too small will considered as 0dB. If this
        # is False then an exception will be raised instead.
        self.handle_small_distances_bool: bool = False

    @property
    def type(self) -> str:
        """Get method for the type property."""
        return self._TYPE

    # xxxxx Start - Implemented these functions in subclasses xxxxxxxxxxxxx
    @abstractmethod
    def which_distance_dB(
            self, PL: NumberOrArray) -> NumberOrArray:  # pragma: no cover
        """
        Calculates the distance that yields the given path loss (in dB).

        Parameters
        ----------
        PL : float | np.ndarray
            Path Loss (in dB)

        Returns
        -------
        d : float | np.ndarray
            Distance to get the desired path loss `PL`.

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

    @abstractmethod
    def _calc_deterministic_path_loss_dB(
            self, d: NumberOrArray,
            **kargs: Any) -> NumberOrArray:  # pragma: no cover
        """
        Calculates the Path Loss (in dB) for a given distance (in Km)
        without including the shadowing.

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in Km)
        kargs : dict, optional
            Optional parameters for use in subclasses if required.

        Other Parameters
        ----------------
        kwargs : dict
            Additional keywords that might be necessary in a subclass.

        Returns
        -------
        PL : float | np.ndarray
            Path loss (in dB).

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
            self,
            d: np.ndarray,
            ax: Optional[Any] = None,
            extra_args: Optional[Any] = None) -> None:  # pragma: no cover
        """
        Plot the path loss (in dB) for the distance values in `d` (in Km).

        Parameters
        ----------
        d : np.ndarray
            Distance (in Km)
        ax : A matplotlib ax, optional
            The ax where the path loss will be plotted. If not provided, a
            new figure (and ax) will be created.
        extra_args : dict
            Extra arguments that will be passed to the ax.plot command as
            ``**extra_args`` (see Matplotlib documentation).
            Ex: {'label': 'curve name', 'linewidth': 2}
        """
        self._plot_deterministic_path_loss_in_dB_impl(d, ax, extra_args, 'Km')

    def _plot_deterministic_path_loss_in_dB_impl(
            self,
            d: np.ndarray,
            ax: Optional[Any] = None,
            extra_args: Optional[Any] = None,
            distance_unit: str = 'Km') -> None:  # pragma: no cover
        """
        Plot the path loss (in dB) for the distance values in `d` (in Km).

        Parameters
        ----------
        d : np.ndarray
            Distance (in correct unit)
        ax : A matplotlib ax, optional
            The ax where the path loss will be plotted. If not provided, a
            new figure (and ax) will be created.
        extra_args : dict
            Extra arguments that will be passed to the ax.plot command as
            ``**extra_args`` (see Matplotlib documentation).
            Ex: {'label': 'curve name', 'linewidth': 2}
        """
        # First we disable the shadowing if it is set
        old_use_shadow_bool = self.use_shadow_bool
        self.use_shadow_bool = False

        if extra_args is None:
            extra_args = {}

        stand_alone_plot = False

        if ax is None:
            # This is a stand alone plot. Lets create a new axes.
            ax = plt.axes()
            stand_alone_plot = True

        # Calculate the deterministic path loss. Note that we use
        # calc_path_loss_dB instead of _calc_deterministic_path_loss_dB
        # because the latter does not respect handle_small_distances_bool.
        PL = self.calc_path_loss_dB(d)

        # Finally plot the path loss
        ax.plot(d, PL, **extra_args)

        # Restore shadowing
        self.use_shadow_bool = old_use_shadow_bool

        if stand_alone_plot is True:
            ax.set_ylabel('Path Loss (in dB)')
            ax.set_xlabel('Distance (in {0})'.format(distance_unit))
            ax.grid(True)
            plt.show()

    # noinspection PyUnresolvedReferences
    def calc_path_loss_dB(self, d: NumberOrArray,
                          **kargs: Any) -> NumberOrArray:
        """
        Calculates the Path Loss (in dB) for a given distance (in Km).

        Note that the returned value is positive, but should be understood
        as "a loss".

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in Km)
        kargs : dict
            Optional parameters for use in subclasses if required.

        Other Parameters
        ----------------
        kwargs : dict
            Additional keywords that might be necessary in a subclass.

        Returns
        -------
        PL : float | np.ndarray
            Path loss (in dB) for the given distance(s).
        """
        PL = self._calc_deterministic_path_loss_dB(d, **kargs)
        if self.use_shadow_bool is True:  # pragma: no cover
            # Shadowing modeled by a Gaussian Distribution (in dB)
            if isinstance(d, np.ndarray):
                # If 'd' is a numpy array (or something similar such as a
                # list), shadow must be a numpy array with the same shape
                shadow = np.random.standard_normal(
                    np.size(d)) * self.sigma_shadow
                shadow.shape = np.shape(d)
            else:
                # If 'd' is not an array but add a scalar shadowing
                shadow = np.random.standard_normal() * self.sigma_shadow
            # Sum the deterministic pathloss value (in dB) with the
            # shadowing (in dB)
            PL += shadow

        # The calculated path loss (in dB) must be positive. If it is not
        # positive that means that the distance 'd' is too small.
        if np.any(np.array(PL) < 0):
            if self.handle_small_distances_bool is True:
                if isinstance(PL, np.ndarray):
                    # If PL is the path loss for multiple distance values
                    # and one (or more) of the path loss values is (are)
                    # negative, set them to to zero (no path loss).
                    PL[PL < 0] = 0.0
                else:
                    # If PL is the path loss for a single distance value
                    # and it is negative (the distance is too small), let's
                    # assume the path loss is equal to 0dB
                    PL = 0.0
            else:
                msg = ("The distance is too small to calculate a valid"
                       " path loss.")
                raise RuntimeError(msg.format(d))
        return PL

    def calc_path_loss(self, d: NumberOrArray,
                       **kargs: Any) -> NumberOrArray:  # pragma: no cover
        """
        Calculates the path loss (linear scale) for a given distance (in
        Km).

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in Km)
        kargs : dict
            Extra named parameters. This is used in subclasses for extra
            parameters for the path loss calculation.

        Other Parameters
        ----------------
        kwargs : dict
            Additional keywords that might be necessary in a subclass.

        Returns
        -------
        pl : float | np.ndarray
            Path loss (in linear scale) for the given distance(s).
        """
        pl = conversion.dB2Linear(-self.calc_path_loss_dB(d, **kargs))
        return pl

    def which_distance(self, pl: NumberOrArray) -> NumberOrArray:
        """
        Calculates the required distance (in Km) to achieve the given path
        loss. It is the inverse of the calc_path_loss function.

        Parameters
        ----------
        pl : float | np.ndarray
            Path loss (in linear scale).

        Returns
        -------
        d : float | np.ndarray
            Distance(s) that will yield the path loss `pl`.
        """
        d = self.which_distance_dB(-conversion.linear2dB(pl))
        return d


class PathLossIndoorBase(PathLossBase):
    """
    Base class for the different Indoor Path Loss models.

    The common interface for the path loss classes is provided by the
    :meth:`calc_path_loss_dB` or the :meth:`calc_path_loss` methods to
    actually calculate the path loss for a given distance, as well as the
    :meth:`which_distance_dB` or which_distance methods to determine the
    distance that yields the given path loss.

    Each subclass of PathLossBase NEED TO IMPLEMENT only the
    :meth:`which_distance_dB` and the
    :meth:`_calc_deterministic_path_loss_dB` functions.

    If the `use_shadow_bool` is set to True then calling
    :meth:`calc_path_loss_dB` or :meth:`calc_path_loss` will take the
    shadowing specified in the `sigma_shadow` variable into
    account. However, shadowing is not taken into account in the
    :meth:`which_distance_dB` and :meth:`which_distance` functions,
    regardless of the value of the `use_shadow_bool` variable.
    """
    _TYPE = 'indoor'

    # xxxxx Start - Implemented these functions in subclasses xxxxxxxxxxxxx
    @abstractmethod
    def which_distance_dB(
            self, PL: NumberOrArray) -> NumberOrArray:  # pragma: no cover
        """
        Calculates the distance that yields the given path loss (in dB).

        Parameters
        ----------
        PL : float | np.ndarray
            Path Loss (in dB)

        Returns
        -------
        d : float | np.ndarray
            The distance to yield the given path loss.

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

    @abstractmethod
    def _calc_deterministic_path_loss_dB(
            self, d: NumberOrArray,
            **kargs: Any) -> NumberOrArray:  # pragma: no cover
        """
        Calculates the Path Loss (in dB) for a given distance (in meters)
        without including the shadowing.

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in meters)

        Other Parameters
        ----------------
        kwargs : dict
            Additional keywords that might be necessary in a subclass.

        Returns
        -------
        PL : float | np.ndarray
            The calculated path loss.

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
            self,
            d: NumberOrArray,
            ax: Optional[Any] = None,
            extra_args: Optional[Any] = None) -> None:  # pragma: no cover
        """
        Plot the path loss (in dB) for the distance values in `d` (in
        meters).

        Parameters
        ----------
        d : np.ndarray
            Distance (in meters)
        ax : A matplotlib ax, optional
            The ax where the path loss will be plotted. If not provided, a
            new figure (and ax) will be created.
        extra_args : dict
            Extra arguments that will be passed to the ax.plot command as
            ``**extra_args`` (see Matplotlib documentation).
            Ex: {'label': 'curve name', 'linewidth': 2}
        """
        self._plot_deterministic_path_loss_in_dB_impl(d, ax, extra_args,
                                                      'meters')

    def calc_path_loss_dB(self, d: NumberOrArray,
                          **kargs: Any) -> NumberOrArray:  # pragma: no cover
        """
        Calculates the Path Loss (in dB) for a given distance (in meters).

        Note that the returned value is positive, but should be understood
        as "a loss".

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in meters)
        kargs : dict
            Additional keywords that might be necessary in a subclass.

        Returns
        -------
        PL : float | np.ndarray
            Path loss (in dB) for the given distance(s).
        """
        return super().calc_path_loss_dB(d, **kargs)

    def calc_path_loss(self, d: NumberOrArray,
                       **kargs: Any) -> NumberOrArray:  # pragma: no cover
        """
        Calculates the path loss (linear scale) for a given distance (in
        meters).

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in meters)
        kargs : dict
            Additional keywords that might be necessary in a subclass.

        Returns
        -------
        pl : float | np.ndarray
            Path loss (in linear scale) for the given distance(s).
        """
        pl = conversion.dB2Linear(-self.calc_path_loss_dB(d, **kargs))
        return pl

    def which_distance(self,
                       pl: NumberOrArray) -> NumberOrArray:  # pragma: no cover
        """
        Calculates the required distance (in meters) to achieve the
        given path loss. It is the inverse of the calc_path_loss function.

        Parameters
        ----------
        pl : float | np.ndarray
            Path loss (in linear scale).

        Returns
        -------
        d : float | np.ndarray
            Distance(s) that will yield the path loss `pl`.
        """
        d = self.which_distance_dB(-conversion.linear2dB(pl))
        return d


class PathLossOutdoorBase(PathLossBase):
    """
    Base class for the different Outdoor Path Loss models.

    The common interface for the path loss classes is provided by the
    :meth:`calc_path_loss_dB` or the :meth:`calc_path_loss` methods to
    actually calculate the path loss for a given distance, as well as the
    :meth:`.which_distance_dB` or :meth:`PathLossBase.which_distance` methods to
    determine the distance that yields the given path loss.

    Each subclass of PathLossBase NEED TO IMPLEMENT only the
    :meth:`which_distance_dB` and the
    :meth:`_calc_deterministic_path_loss_dB` functions.

    If the `use_shadow_bool` is set to True then calling
    :meth:`calc_path_loss_dB` or :meth:`calc_path_loss` will take the
    shadowing specified in the `sigma_shadow` variable into
    account. However, shadowing is not taken into account in the
    :meth:`which_distance_dB` and :meth:`PathLossBase.which_distance` functions,
    regardless of the value of the `use_shadow_bool` variable.
    """
    _TYPE = 'outdoor'

    # xxxxx Start - Implemented these functions in subclasses xxxxxxxxxxxxx
    @abstractmethod
    def which_distance_dB(
            self, PL: NumberOrArray) -> NumberOrArray:  # pragma: no cover
        """
        Calculates the distance that yields the given path loss (in dB).

        Parameters
        ----------
        PL : float | np.ndarray
            Path Loss (in dB)

        Returns
        -------
        d : float | np.ndarray
            The distance that yields the given path loss.

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

    @abstractmethod
    def _calc_deterministic_path_loss_dB(
            self, d: NumberOrArray,
            **kargs: Any) -> NumberOrArray:  # pragma: no cover
        """
        Calculates the Path Loss (in dB) for a given distance (in Km)
        without including the shadowing.

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in Km)

        Other Parameters
        ----------------
        kwargs : dict
            Additional keywords that might be necessary in a subclass.

        Returns
        -------
        PL : float | np.ndarray
            The calculated path loss (in dB).

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
            self,
            d: NumberOrArray,
            ax: Optional[Any] = None,
            extra_args: Optional[Any] = None) -> None:  # pragma: no cover
        """
        Plot the path loss (in dB) for the distance values in ``d`` (in Km).

        Parameters
        ----------
        d : np.ndarray
            Distance (in Km)
        ax : A matplotlib ax, optional
            The ax where the path loss will be plotted. If not provided, a
            new figure (and ax) will be created.
        extra_args : dict
            Extra arguments that will be passed to the ax.plot command as
            ``**extra_args`` (see Matplotlib documentation).
            Ex: {'label': 'curve name', 'linewidth': 2}
        """
        self._plot_deterministic_path_loss_in_dB_impl(d, ax, extra_args, 'Km')

    def calc_path_loss_dB(self, d: NumberOrArray,
                          **kargs: Any) -> NumberOrArray:
        """
        Calculates the Path Loss (in dB) for a given distance (in Km).

        Note that the returned value is positive, but should be understood
        as "a loss".

        Parameters
        ----------
        d : float | np.ndarray | list[float]
            Distance (in Km)
        kargs : dict
            Additional keywords that might be necessary in a subclass.

        Returns
        -------
        PL : float | np.ndarray
            Path loss (in dB) for the given distance(s).
        """
        return super().calc_path_loss_dB(d)

    def calc_path_loss(self, d: NumberOrArray, **kargs: Any) -> NumberOrArray:
        """
        Calculates the path loss (linear scale) for a given distance (in
        Km).

        Parameters
        ----------
        d : float | np.ndarray | list[float]
            Distance (in Km)
        kargs : dict
            Additional keywords that might be necessary in a subclass.

        Returns
        -------
        pl : float | np.ndarray
            Path loss (in linear scale) for the given distance(s).
        """
        pl = conversion.dB2Linear(-self.calc_path_loss_dB(d))
        return pl


class PathLossGeneral(PathLossOutdoorBase):
    """
    Class to calculate the path loss given the path loss for a reference
    distance.

    In its simplest form, the path loss can be calculated using the formula

    :math:`PL = 10 n \\log_{10} (d) + C`

    where `PL` is in dB, `n` is the path loss exponent (usually in the
    range of 2 to 4) and `d` is the distance between the transmitter and
    the receiver.

    Parameters
    ----------
    n : float
        The path loss exponent.
    C : float
        The constant `C` in the path loss formula.
    """

    # $PL = 10n\log_{10}(d) + C$

    def __init__(self, n: float, C: float):
        """
        Initializes the path loss object.

        Parameters
        ----------
        n : float
            The path loss exponent.
        C : float
            The constant `C` in the path loss formula.
        """
        super().__init__()

        self._n: float = n
        self._C: float = C

    def _get_latex_repr(self) -> str:  # pragma: no cover
        """
        Get the Latex representation (equation) for the PathLossGeneral
        class.

        The general equation is given by

        .. math::
           PL = 10 n \\log_{10} (d) + C

        Returns
        -------
        str
            The Latex representation of the path loss object.
        """
        return '$PL = {0} \\log_{{10}} (d) + {1}$'.format(
            10 * self._n, self._C)

    def _repr_latex_(self) -> str:  # pragma: no cover
        """
        Get a Latex representation of the PathLossGeneral class.

        This is useful for representing the path loss object in an IPython
        notebook.

        Returns
        -------
        str
            The Latex representation of the path loss object.
        """
        return "PathLossGeneral (n={0}, C={1}): {2}".format(
            self._n, self._C, self._get_latex_repr())

    # @property
    # def n(self):
    #     """Get method for the n property."""
    #     return self._n

    # @n.setter
    # def n(self, value):
    #     """Set method for the n property."""
    #     self._n = value

    # @property
    # def C(self):
    #     """Get method for the C property."""
    #     return self._C

    # @C.setter
    # def C(self, value):
    #     """Set method for the C property."""
    #     self._C = value

    def which_distance_dB(self, PL: NumberOrArray) -> NumberOrArray:
        """
        Calculates the required distance (in Km) to achieve the given path loss
        (in dB).

        It is the inverse of the calc_path_loss function.

        .. math::
           10^{(PL/(10n) - C)}

        d = obj.whichDistance(dB2Linear(-PL));

        Parameters
        ----------
        PL : float | np.ndarray
            Path Loss (in dB).

        Returns
        -------
        d : float | np.ndarray
            Distance (in Km).
        """
        d = 10.**((PL - self._C) / (10. * self._n))
        return d

    def _calc_deterministic_path_loss_dB(self, d: NumberOrArray,
                                         **kargs: Any) -> NumberOrArray:
        """
        Calculates the Path Loss (in dB) for a given distance (in Km).

        Note that the returned value is positive, but should be understood
        as "a loss".

        For d in Km and self.fc in MHz, the free space Path Loss is given by

        .. math::
           PL = 10 n \\log_{10}(d) + C

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in Km).

        Returns
        -------
        pl_dB : float | np.ndarray
            Path loss in dB.
        """
        if isinstance(d, Iterable):
            log10 = np.log10
        else:
            log10 = math.log10

        PL = (10 * self._n * log10(d)) + self._C
        return PL


class PathLossFreeSpace(PathLossGeneral):
    """
    Class to calculate the Path Loss in the free space.

    The common interface for the path loss classes is provided by the
    :meth:`PathLossOutdoorBase.calc_path_loss_dB` or the
    :meth:`PathLossOutdoorBase.calc_path_loss` methods to actually calculate
    the path loss for a given distance, as well as the
    :meth:`PathLossGeneral.which_distance_dB` or
    :meth:`PathLossBase.which_distance` methods to determine the distance that
    yields the given path loss.

    For the path loss in free space you also need to set the `n` variable,
    corresponding to the path loss coefficient, and the `fc` variable,
    corresponding to the frequency. The `n` variable defaults to 2 and `fc`
    defaults to 900 (that is, 900MHz).

    The path loss (in dB) in free space is calculated as:

    .. math::
       PL = 10 n ( \\log_{10}(d)+\\log_{10}(fc * 1e6) - 4.3779113907)

    Likewise, the :meth:`PathLossGeneral.which_distance_dB` function calculates
    the value of

    .. math::
       10^{(PL/(10n) - \\log_{10}(fc) + 4.377911390697565)}

    Parameters
    ----------
    n : float
        Path loss exponent.
    fc : float
        Central carrier frequency (in MHz).

    Examples
    --------
    Determining the path loss in the free space for a distance of 1Km
    (without considering shadowing).

    >>> pl = PathLossFreeSpace()
    >>> pl.calc_path_loss(1)        # linear scale
    7.036193308495632e-10
    >>> pl.calc_path_loss_dB(1)     # log scale
    91.5266223748352

    Determining the distance (in Km) that yields a path loss of 90dB.

    >>> pl.which_distance_dB(90)
    0.8388202017414481
    """
    def __init__(self, n: float = 2.0, fc: float = 900.0):
        """
        Initializes the PathLossFreeSpace object

        Parameters
        ----------
        n : float
            Path loss exponent.
        fc : float
            Central carrier frequency (in MHz)
        """
        super().__init__(n=n, C=0)

        # Note that the set property of self.fc will update self._C
        self._fc: float = fc  # Frequency of the central carrier (in MHz)
        self._C: float = self._calculate_C_from_fc_and_n(self._fc, self.n)

    def _repr_latex_(self) -> str:  # pragma: no cover
        """
        Get a Latex representation of the PathLossFreeSpace class.

        This is useful for representing the path loss object in an IPython
        notebook.

        Returns
        -------
        str
            The Latex representation of the path loss object.
        """
        return "PathLossFreeSpace (n={0}, fc={1}): {2}".format(
            self.n, self.fc, self._get_latex_repr())

    @property
    def n(self) -> float:
        """
        Get method for the n property.

        Returns
        -------
        float
            The path loss exponent.
        """
        return self._n

    @n.setter
    def n(self, value: float) -> None:
        """
        Set method for the n property.

        Parameters
        ----------
        value : float
            The new path loss exponent.
        """
        self._n = value
        # If we change 'n', we need to update the C variable
        self._C = self._calculate_C_from_fc_and_n(self._fc, self.n)

    @property
    def fc(self) -> float:  # pragma: no cover
        """
        Get the central carrier frequency.

        Returns
        -------
        float
            The central carrier frequency.
         """
        return self._fc

    @fc.setter
    def fc(self, value: float) -> None:
        """
        Set the central carrier frequency (in MHz).

        Parameters
        ----------
        value : float
            Central carrier frequency (in MHz).
        """
        self._fc = value
        # If we change 'fc', we need to update the C variable
        self._C = self._calculate_C_from_fc_and_n(self._fc, self.n)

    @staticmethod
    def _calculate_C_from_fc_and_n(fc: float, n: float) -> float:
        """
        Calculate the value of the constant `C` for the frequency value `fc`
        and path loss exponent `n`.

        Parameters
        ----------
        fc : float
            Central carrier frequency (in MHz)
        n : float
            Path loss exponent

        Returns
        -------
        C : float
            Constant `C` for the Free Space path loss model for the given
            frequency and path loss exponent.
        """
        # $PL = 10 n (\log_{10}(d)+\log_{10}(f * 1e6) - 4.3779113907)$
        C = 10 * n * (math.log10(fc * 1e6) - 4.377911390697565)
        return C


class PathLoss3GPP1(PathLossGeneral):
    """
    Class to calculate the Path Loss according to the model from 3GPP
    (scenario 1). That is, the Path Loss (in dB) is equal to

    .. math::
       PL = 128.1 + 37.6*\\log10(d)

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
    128.1

    Determining the distance (in Km) that yields a path loss of 130dB.

    >>> pl.which_distance_dB(130)
    1.1233935211892188
    """
    def __init__(self) -> None:
        super().__init__(n=3.76, C=128.1)

    def _repr_latex_(self) -> str:  # pragma: no cover
        """
        Get a Latex representation of the PathLossFreeSpace class.

        This is useful for representing the path loss object in an IPython
        notebook.

        Returns
        -------
        str
            The Latex representation of the path loss object.
        """
        return "PathLoss3GPP1: {0}".format(self._get_latex_repr())


class PathLossMetisPS7(PathLossIndoorBase):
    """
    Class to calculate the Path Loss (indoor) according to the model
    described for the Propagation Scenario (PS) 7 of the METIS project.

    This model is an indoor-2-indoor model.
    """
    def __init__(self, fc: float = 900.0):
        """
        Initializes the PathLossFreeSpace object

        Parameters
        ----------
        fc : float
            Central carrier frequency (in MHz)
        """
        super().__init__()
        self._fc: float = fc  # Frequency (in MHz)

    @property
    def fc(self) -> float:
        """
        Get the central carrier frequency.

        Returns
        -------
        float
            The central carrier frequency.
        """
        return self._fc

    @fc.setter
    def fc(self, value: float) -> None:
        """
        Set the central carrier frequency (in MHz).

        Parameters
        ----------
        value : float
            Central carrier frequency (in MHz).
        """
        self._fc = value

    def _repr_latex_(self) -> str:  # pragma: no cover
        """
        Get a Latex representation of the PathLossMetisPS7 class.

        This is useful for representing the path loss object in an IPython
        notebook.

        Returns
        -------
        str
            The Latex representation of the path loss object.
        """
        return "PathLossMetisPS7 (fc={0}):\n{1}".format(
            self.fc, self.get_latex_repr())

    @staticmethod
    def get_latex_repr(
            num_walls: Optional[int] = None) -> str:  # pragma: no cover
        """
        Get the Latex representation (equation) for the PathLossGeneral class.

        The general equation is given by

        .. math::
           PL = A \\log_{10}(d) + B + C \\log_{10}(f_c/5) + X

        where the parameters A, B, C and X depend on the number of walls.

        Parameters
        ----------
        num_walls : int, None
            Number of walls. LOS is used if it is 0 and NLOS is used if it
            is greater than zero. If it is None, then letters are used
            instead of numeric values.

        Returns
        -------
        str
            The Latex representation (equation) for the PathLossGeneral class.
        """
        if num_walls is None:
            values = {'A': 'A', 'B': 'B', 'C': 'C', 'X': 'X'}
        else:
            if num_walls == 0:
                values = {'A': '18.7', 'B': '46.8', 'C': '20', 'X': '0'}
            elif num_walls > 0:
                values = {
                    'A': '36.8',
                    'B': '43.8',
                    'C': '20',
                    'X': str(5 * num_walls - 1)
                }
            else:
                raise ValueError("num_walls cannot be negative")

        return ("${A} \\log_{{10}}(d) + {B} + {C} \\log_{{10}}(f_c/5)"
                " + {X}$").format(**values)

    def _calc_PS7_path_loss_dB_same_floor(self,
                                          d: NumberOrArray,
                                          num_walls: int = 0) -> NumberOrArray:
        """
        Calculate the deterministic path loss according to the Propagation
        Scenario (PS) 7 of the METIS project.

        The path loss (in dB) is calculated as

        .. math::
           PL = A \\log_{10}(d) + B + C \\log_{10}(f_c/5) + X

        The distance :math:`d` is in meters, while the frequency
        :math:`f_c` is in GHz. Na others variables a different for the LOS
        and NLOS cases.

        For the LOS case we have:

        .. math::
           A = 18.7 B = 46.8, C = 20, X = 0

        For the NLOS case we have:

        .. math::
           A = 36.8 B = 43.8, C = 20, X = 5 ( n_w - 1 )

        where :math:`n_w` is the number of walls between the transmitter
        and receiver.

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in meters).
        num_walls : int | np.ndarray
            Indicates how many walls the signal has to pass. If num_walls
            is zero, then Line-of-Sight parameters are used. If it is
            greater than zero then Non-Sign-of-Sight parameters are
            used. If num_walls is a numpy array then it must have the same
            dimension as `d` and it then specifies the number of walls for
            each individual value of `d`

        Returns
        -------
        pl_dB : float | np.ndarray
            Path loss in dB.
        """
        if isinstance(num_walls, Iterable):
            # Code for num_walls array. Since num_walls is an array then
            # some values might be equal to zero while others might be
            # equal to zero. We will calculate them separately.
            #
            # The dimension of num_walls must be equal to the dimension of
            # d. Maybe some of the dimensions in num_walls are equal to 1
            # and we must broadcast them first to be equal to the
            # dimensions of d.
            [_, num_walls] = np.broadcast_arrays(d, num_walls)

            LOS_index = (num_walls == 0)
            NLOS_index = ~LOS_index

            pl_dB = np.empty(d.shape, dtype=float)
            pl_dB[LOS_index] \
                = self._calc_PS7_path_loss_dB_LOS_same_floor(d[LOS_index])

            pl_dB[NLOS_index] \
                = self._calc_PS7_path_loss_dB_NLOS_same_floor(
                    d[NLOS_index],
                    num_walls[NLOS_index])
        else:
            # Code for int num_walls
            if num_walls == 0:
                # Calculate the path loss for PS7 with LOS
                pl_dB = self._calc_PS7_path_loss_dB_LOS_same_floor(d)
            elif num_walls > 0:
                # Calculate the path loss for PS7 without LOS
                pl_dB = self._calc_PS7_path_loss_dB_NLOS_same_floor(
                    d, num_walls)
            else:
                raise ValueError("num_walls cannot be negative")

        return pl_dB

    def _calc_PS7_path_loss_dB_LOS_same_floor(
            self, d: NumberOrArray) -> NumberOrArray:
        """
        Calculate the deterministic path loss according to the Propagation
        Scenario (PS) 7 of the METIS project for the LOS case.

        The path loss (in dB) is calculated as

        .. math::
           PL = A \\log_{10}(d) + B + C \\log_{10}(f_c/5)

        The distance :math:`d` is in meters, while the frequency
        :math:`f_c` is in GHz.

        The other variables (NLOS case) are:

        .. math::
           A = 18.7 B = 46.8, C = 20

        where :math:`n_w` is the number of walls between the transmitter
        and receiver.

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in meters).

        Returns
        -------
        pl_dB : float | np.ndarray
            Path loss in dB.
        """
        if isinstance(d, Iterable):
            log10 = np.log10
        else:
            log10 = math.log10

        # LOS parameters
        A = 18.7
        B = 46.8
        C = 20

        # self.fc is in MHz
        fc_GHz = self.fc / 1e3

        pl_dB = A * log10(d) + B + C * log10(fc_GHz / 5.)
        return pl_dB

    def _calc_PS7_path_loss_dB_NLOS_same_floor(self,
                                               d: NumberOrArray,
                                               num_walls: int = 1
                                               ) -> NumberOrArray:
        """
        Calculate the deterministic path loss according to the Propagation
        Scenario (PS) 7 of the METIS project for the NLOS case.

        The path loss (in dB) is calculated as

        .. math::
           PL = A \\log_{10}(d) + B + C \\log_{10}(f_c/5) + X

        The distance :math:`d` is in meters, while the frequency
        :math:`f_c` is in Hz.

        The other variables (NLOS case) are:

        .. math::
           A = 36.8 B = 43.8, C = 20, X = 5 ( n_w - 1 )

        where :math:`n_w` is the number of walls between the transmitter
        and receiver.

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in meters).
        num_walls : int | np.ndarray
            Number of walls between the transmitter and the receiver. If it
            is an int array it must have the same dimension as `d`.

        Returns
        -------
        pl_dB : float | np.ndarray
            Path loss in dB.
        """
        if isinstance(d, Iterable):
            log10 = np.log10
        else:
            log10 = math.log10

        # NLOS parameters
        A = 36.8
        B = 43.8
        C = 20
        X = 5 * (num_walls - 1)

        # self.fc is in MHz
        fc_GHz = self.fc / 1e3

        # LOS values: A = 18.7 B = 46.8, C = 20, X = 0
        # NLOS values: A = 36.8 B = 43.8, C = 20, X = 5 ( n_w - 1 )

        # For the propagation between floors, we need to add the floor
        # losses if the transmitter and receiver are in different floors as
        # FL = 17 + 4 (n_f - 1)

        pl_dB = A * log10(d) + B + C * log10(fc_GHz / 5.) + X
        return pl_dB

    def which_distance_dB(
            self, PL: NumberOrArray) -> NumberOrArray:  # pragma: nocover
        pass

    def _calc_deterministic_path_loss_dB(  # type: ignore
            self,
            d: NumberOrArray,
            num_walls: int = 0) -> NumberOrArray:  # pragma: no cover
        """
        Calculates the Path Loss (in dB) for a given distance (in meters)
        without including the shadowing.

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in meters)
        num_walls : int | np.ndarray
            Number of walls between the transmitter and the receiver. If it
            is an int array it must have the same dimension as `d`.

        Returns
        -------
        PL : float | np.ndarray
            The calculated path loss.
        """
        return self._calc_PS7_path_loss_dB_same_floor(d, num_walls)


# TODO: Test this class
# See http://w3.antd.nist.gov/wctg/manet/calcmodels_r1.pdf
# noinspection PyPep8
class PathLossOkomuraHata(PathLossOutdoorBase):
    """
    Class to calculate the Path Loss according to the Okomura Hata model.

    The exact formula depend on the area type, but in general the path loss
    is given by (in dB):

    .. math::
       PL = 69.55 + 26.16 * \\log(fc) - 13.82*\\log(h_{bs}) - a(h_{ms}) + (44.9 - 6.55\\log(h_{bs})) \\log(d) - K

    The term :math:`a(h_{ms})` is the mobile station correction factor (see
    :meth:`_calc_mobile_antenna_height_correction_factor`).

    The term :math:`K` is a correction factor that depends on the area type
    (see :meth:`_calc_K`).

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
    def __init__(self) -> None:
        super().__init__()

        # Height of the Base Station (in meters) -> 30m to 200m
        self._hbs: float = 30.0

        # Height of the Mobile Station (in meters) - 1m to 10m
        self._hms: float = 1.0

        # Frequency of the central carrier (in MHz) - 150MHz to 1500MHz
        self._fc: float = 900.0

        # area_type can be 'open', 'suburban', 'medium city', 'large city'.
        # Note: The category of "large city" used by Hata implies building
        # heights greater than 15m.
        self._area_type: str = 'suburban'  # TODO: Change the area types to enumeration

    # xxxxxxxxxx fc property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def fc(self) -> float:
        """
        Get the central carrier frequency.

        Returns
        -------
        float
            The central carrier frequency.
        """
        return self._fc

    @fc.setter
    def fc(self, value: float) -> None:
        """
        Set the central carrier frequency (in MHz).

        Parameters
        ----------
        value : float
            Central carrier frequency (in MHz).
        """

        if value < 150.0 or value > 1500:
            msg = ("The carrier frequency for the Okomura Hata model must be"
                   " between 150 and 1500 (values in MHz).")
            raise RuntimeError(msg)
        self._fc = value

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx hbs property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def hbs(self) -> float:
        """
        Get the height of the Base Station property.

        Returns
        -------
        float
            Height of the Base Station (in meters).
        """
        return self._hbs

    @hbs.setter
    def hbs(self, value: float) -> None:
        """
        Set the height of the Base Station property (in meters).

        Parameters
        ----------
        value : float
            The Height of the Base Station (in meters). This should be
            between 30m to 200m.
        """
        if value < 30.0 or value > 200.0:
            msg = ("The Base Station antenna height for the Okomura Hata "
                   "model must be between 30 and 200 (values in meters).")
            raise RuntimeError(msg)
        self._hbs = value

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx hms property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def hms(self) -> float:
        """
        Get the height of the Mobile Station property.

        Returns
        -------
        float
            Height of the Mobile Station (in meters).
        """
        return self._hms

    @hms.setter
    def hms(self, value: float) -> None:
        """
        Set the height of the Mobile Station property (in meters).

        Parameters
        ----------
        value : float
            The Height of the Mobile Station (in meters). This should be
            between 1m to 10m.
        """
        if value < 1.0 or value > 10.0:
            msg = ("The Mobile Station antenna height for the Okomura Hata "
                   "model must be between 1 and 10 (values in meters).")
            raise RuntimeError(msg)
        self._hms = value

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx area_type property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def area_type(self) -> str:
        """Get method for the area_type property."""
        return self._area_type

    @area_type.setter
    def area_type(self, value: str) -> None:
        """
        Set method for the area_type property.

        Parameters
        ----------
        value : str
            The area type to set. This should be one of the values in
            ['open', 'suburban', 'medium city', 'large city'].
        """
        if value not in ['open', 'suburban', 'medium city', 'large city']:
            raise RuntimeError('Invalid area type: {0}'.format(value))
        self._area_type = value

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _calc_mobile_antenna_height_correction_factor(self) -> float:
        """
        Calculates the mobile antenna height correction factor.

        This factor is strongly impacted by surrounding buildings and is
        refined according to city sizes. For all area types, except 'large
        city', the mobile antenna correction factor is given by

        .. math::
          a(h_{ms}) = (1.1 \\log(f) - 0.7) h_{ms} - 1.56 \\log(f) + 0.8

        For the 'large city' area type, the mobile antenna height
        correction is given by

        .. math::
           a(h_{ms}) = 3.2 (\\log(11.75*h_{ms})^2) - 4.97

        if the frequency is greater then 300MHz, or

        .. math::
           a(h_{ms}) = 8.29 (\\log(1.54 h_{ms}))^2 - 1.10

        if the frequency is lower than 300MHz (and greater than 150MHz
        where the Okomura Hata model is valid).

        Returns
        -------
        float
            The mobile antenna height correction.
        """
        if self.area_type in ['open', 'suburban', 'medium city']:

            # Suburban and rural areas (f in MHz
            # $a(h_{ms}) = (1.1 \log(f) - 0.7) h_{ms} - 1.56 \log(f) + 0.8$
            a = ((1.1 * math.log10(self.fc) - 0.7) * self.hms -
                 1.56 * math.log10(self.fc) + 0.8)
        elif self.area_type == 'large city':
            # Note: The category of "large city" used by Hata implies
            # building heights greater than 15m.
            if self.fc > 300:
                # If frequency is greater then 300MHz then the factor is
                # given by
                # $3.2 (\log(11.75*h_{ms})^2) - 4.97$
                a = 3.2 * (math.log10(11.75 * self.hms)**2) - 4.97
            else:
                # If frequency is lower then 300MHz then the factor is
                # given by
                # $8.29 (\log(1.54 h_{ms}))^2 - 1.10$
                a = 8.29 * (math.log10(1.54 * self.hms)**2) - 1.10
        else:  # pragma: no cover
            raise RuntimeError('Invalid area type: {0}'.format(self.area_type))

        return a

    def _calc_K(self) -> float:
        """
        Calculates the "'medium city'/'suburban'/'open area'" correction
        factor.

        Returns
        -------
        K : float
            The correction factor "K".
        """
        if self.area_type == 'large city':
            K = 0.0
        elif self.area_type == 'open':
            # Value for 'open' areas
            # $K = 4.78 (\log(f))^2 - 18.33 \log(f) + 40.94$
            K = (4.78 * (math.log10(self.fc)**2) -
                 18.33 * math.log10(self.fc) + 40.94)
        elif self.area_type == 'suburban':
            # Value for 'suburban' areas
            # $K = 2 [\log(f/28)^2] + 5.4$
            K = 2 * (math.log10(self.fc / 28.0)**2) + 5.4
        else:
            K = 0
        return K

    def _calc_deterministic_path_loss_dB(self, d: NumberOrArray,
                                         **kargs: Any) -> NumberOrArray:
        """
        Calculates the Path Loss (in dB) for a given distance (in Km).

        Note that the returned value is positive, but should be understood
        as "a loss".

        For d in Km and self.fc in Hz, the free space Path Loss is given by

        .. math::
           PL = 10n ( \\log_{10}(d) + \\log_{10}(f) - 4.3779113907 )

        Parameters
        ----------
        d : float | np.ndarray
            Distance (in Km). Can be between 1km and 20Km.

        Returns
        -------
        PL : float | np.ndarray
            Path loss in dB.
        """
        if isinstance(d, Iterable):
            log10 = np.log10
        else:
            log10 = math.log10

        # noinspection PyTypeChecker
        if np.any(d < 1.0) or np.any(d > 20.0):
            msg = ('Distance for the Okomura Hata model should be between'
                   ' 1Km and 20Km')
            warnings.warn(Warning(msg))

        # $L (\text{in dB}) = 69.55 + 26.16 \log(f) -13.82 \log(h_{bs}) - a(h_{ms}) + (44.9 - 6.55\log(h_{bs})) \log(d) - K$

        # Calculate the mobile antenna height correction factor
        a = self._calc_mobile_antenna_height_correction_factor()

        # Calculates the "'suburban'/'open area'" correction factor.
        K = self._calc_K()

        L = (69.55 + 26.16 * log10(self.fc) - 13.82 * log10(self.hbs) - a +
             (44.9 - 6.55 * log10(self.hbs)) * log10(d) - K)
        return L

    def which_distance_dB(self, PL: NumberOrArray) -> NumberOrArray:
        """
        Calculates the required distance (in Km) to achieve the given path loss
        (in dB).

        It is the inverse of the calc_path_loss function.

        Parameters
        ----------
        PL : float | np.ndarray
            Path loss (in dB).

        Returns
        -------
        d : float | np.ndarray
            Distance (in Km).
        """
        raise NotImplementedError("which_distance_dB is not available for "
                                  "this path loss model")  # pragma: no cover
