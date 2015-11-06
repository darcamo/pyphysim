#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing single user channels. """

import math
import numpy as np

from pyphysim.channels import fading


class SuSisoChannel(fading.TdlChannel):
    """
    Single User channel corresponding to a Tapped Delay Line channel model,
    which corresponds to a multipath channel. You can use a single tap in
    order to get a flat fading channel.

    You can create a new SuSisoChannel object either specifying the channel
    profile or specifying both the channel tap powers and delays. If only the
    fading_generator is specified then a single tap with unitary power and
    delay zero will be assumed, which corresponds to a flat fading channel
    model.

    Parameters
    ----------
    fading_generator : Subclass of FadingSampleGenerator
        The instance of a fading generator in the `fading_generators` module.
        It should be a subclass of FadingSampleGenerator. The fading
        generator will be used to generate the channel samples.
    channel_profile : TdlChannelProfile
        The channel profile, which specifies the tap powers and delays.
    tap_powers_dB : numpy real array
        The powers of each tap (in dB). Dimension: `L x 1`
        Note: The power of each tap will be a negative number (in dB).
    tap_delays : numpy real array
        The delay of each tap (in seconds). Dimension: `L x 1`
    """
    def __init__(self, fading_generator, channel_profile=None,
                 tap_powers_dB=None, tap_delays=None, Ts=None):
        if (channel_profile is None and
                tap_powers_dB is None and
                tap_delays is None and
                Ts is None):
            # Only the fading generator was provided. Let's assume a flat
            # fading channel
            super(SuSisoChannel, self).__init__(fading_generator,
                                                tap_powers_dB=np.zeros(1),
                                                tap_delays=np.zeros(1))

        else:
            # More parameters were provided. We will have then a TDL channel
            # model. Let's iust pass these parameters to the base class.
            super(SuSisoChannel, self).__init__(
                fading_generator,
                channel_profile, tap_powers_dB, tap_delays,
                Ts)

        # Path loss which will be multiplied by the impulse response when
        # corrupt_data is called
        self._pathloss_value = None

    def set_pathloss(self, pathloss_value=None):
        """
        Set the path loss (IN LINEAR SCALE) from each transmitter to each
        receiver.

        The path loss will be accounted when calling the corrupt_data
        method.

        If you want to disable the path loss, set `pathloss_value` to
        None.

        Parameters
        ----------
        pathloss_value : float
            The path loss (IN LINEAR SCALE) from the transmitter to the
            receiver. If you want to disable the path loss then set it to
            None.

        Notes
        -----
        Note that path loss is a power relation, which means that the
        channel coefficients will be multiplied by the square root of
        elements in `pathloss_value`.
        """
        if pathloss_value is not None:
            if pathloss_value < 0 or pathloss_value > 1:
                raise ValueError("Pathloss must be between 0 and 1")

        self._pathloss_value = pathloss_value

    def corrupt_data(self, signal):
        """
        Transmit the signal trhough the TDL channel.

        Parameters
        ----------
        signal : numpy array
            The signal to be transmitted.
        """
        output = super(SuSisoChannel, self).corrupt_data(signal)

        if self._pathloss_value is not None:
            output *= math.sqrt(self._pathloss_value)

        return output
