#!/usr/bin/env python

from typing import Optional, TypeVar

import numpy as np

from ..util.conversion import dB2Linear

# See http://www.qtc.jp/3GPP/Specs/25996-a00.pdf

NumberOrArray = TypeVar("NumberOrArray", np.ndarray, float)


class AntGainBase:  # pragma: no cover
    """Base class for antenna models.
    """
    def get_antenna_gain(self, angle: NumberOrArray) -> NumberOrArray:
        """
        Get the antenna gain for the given angle.

        Parameters
        ----------
        angle : float | np.ndarray
            Angle between the direction of interest and the boresight of
            the antenna. This can also be a numpy array with angles.

        Returns
        -------
        float | np.ndarray
            The gain (in linear scale) for the provided angle.
        """
        raise NotImplementedError("Implement in a subclass")


class AntGainOmni(AntGainBase):
    """
    Class for Omnidirectional antenna gain model.

    Parameters
    ----------
    ant_gain : float, optional
        The antenna gain (in dBi). If not provided then 0dBi will be assumed.
    """
    def __init__(self, ant_gain: Optional[float] = None):
        super().__init__()
        self.ant_gain: float
        if ant_gain is None:
            self.ant_gain = 1.0
        else:
            self.ant_gain = dB2Linear(ant_gain)

    def get_antenna_gain(self, angle: NumberOrArray) -> NumberOrArray:
        """
        Get the antenna gain for the given angle.

        Parameters
        ----------
        angle : float | np.ndarray
            Angle between the direction of interest and the boresight of
            the antenna. This can also be a numpy array with angles.

        Returns
        -------
        float | np.ndarray
            The gain (in linear scale) for the provided angle.
        """
        if isinstance(angle, np.ndarray):
            return self.ant_gain * np.ones(angle.shape)  # type: ignore

        return self.ant_gain


class AntGainBS3GPP25996(AntGainBase):
    """
    Class for antenna model defined by 3GPP in the 25996 norm for sectorized
    Base Stations.

    The antenna gain (in dBi) will depend on the number of sectors of the
    Base Station.

    NOTE: The antenna pattern here is targeted for diversity-oriented
    implementations (i.e. large inter-element spacings). For beamforming
    applications that require small spacings, alternative antenna designs
    may have to be considered leading to a different antenna pattern.

    Parameters
    ----------
    number_of_sectors : int
        The number of sectors of the base station. It can be either 3 or 6.
    """
    def __init__(self, number_of_sectors: int = 3):
        super().__init__()

        self.ant_gain: float

        if number_of_sectors == 3:
            self.theta_3db = 70.  # Defined in the norm
            self.Am = 20.  # Maximum attenuation in dB
            self.ant_gain = dB2Linear(14.)  # Antenna gain (in dBi)
        elif number_of_sectors == 6:
            self.theta_3db = 35.  # Defined in the norm
            self.Am = 23.  # Maximum attenuation in dB
            self.ant_gain = dB2Linear(17.)  # Antenna gain (in dBi)
        else:
            raise ValueError(
                "Invalid number of sectors: {0}".format(number_of_sectors))

    # noinspection PyPep8
    def get_antenna_gain(self, angle: NumberOrArray) -> NumberOrArray:
        """
        Get the antenna gain for the given angle.

        Parameters
        ----------
        angle : float | np.ndarray
            Angle between the direction of interest and the boresight of
            the antenna. This can also be a numpy array with angles.

        Returns
        -------
        float | np.ndarray
            The gain (in linear scale) for the provided angle.
        """
        # For the antenna gain model defined in 3GPP 25996 the gain (in dB)
        # is given by
        #\(-\min\left[ 12\left( \frac{\theta}{\theta_{3dB}} \right)^2, A_m \right]\), where \(-180 \geq \theta \geq 180\)
        ant_pattern_gain = dB2Linear(
            -np.minimum(12 * (angle / self.theta_3db)**2, self.Am))
        return self.ant_gain * ant_pattern_gain  # type: ignore
