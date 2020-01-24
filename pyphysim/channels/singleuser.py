#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module containing single user channels. """

import math
import numpy as np

from . import fading, fading_generators


class SuChannel:
    """
    Single User channel corresponding to a Tapped Delay Line channel model,
    which corresponds to a multipath channel. You can use a single tap in
    order to get a flat fading channel.

    You can create a new SuChannel object either specifying the channel
    profile or specifying both the channel tap powers and delays. If
    only the fading_generator is specified then a single tap with
    unitary power and delay zero will be assumed, which corresponds to a
    flat fading channel model.

    Parameters
    ----------
    fading_generator : T <= fading_generators.FadingSampleGenerator
        The instance of a fading generator in the `fading_generators`
        module.  It should be a subclass of FadingSampleGenerator. The
        fading generator will be used to generate the channel samples. If
        not provided then RayleighSampleGenerator will be used
    channel_profile : fading.TdlChannelProfile
        The channel profile, which specifies the tap powers and delays.
    tap_powers_dB : np.ndarray
        The powers of each tap (in dB). Dimension: `L x 1`
        Note: The power of each tap will be a negative number (in dB).
    tap_delays : np.ndarray
        The delay of each tap (in seconds). Dimension: `L x 1`
    """

    def __init__(self,
                 fading_generator=None,
                 channel_profile=None,
                 tap_powers_dB=None,
                 tap_delays=None,
                 Ts=None):
        if fading_generator is None:
            fading_generator = fading_generators.RayleighSampleGenerator()
            if channel_profile is None and Ts is None:
                Ts = 1

        if (channel_profile is None and tap_powers_dB is None and
                tap_delays is None):
            # Only the fading generator was provided. Let's assume a flat
            # fading channel
            self._tdlchannel = fading.TdlChannel(fading_generator,
                                                 tap_powers_dB=np.zeros(1),
                                                 tap_delays=np.zeros(1),
                                                 Ts=Ts)
        else:
            # More parameters were provided. We will have then a TDL
            # channel model. Let's just pass these parameters to the
            # base class.
            self._tdlchannel = fading.TdlChannel(fading_generator,
                                                 channel_profile, tap_powers_dB,
                                                 tap_delays, Ts)

        # Path loss which will be multiplied by the impulse response when
        # corrupt_data is called
        self._pathloss_value = None

    def set_pathloss(self, pathloss_value=None):
        """
        Set the path loss (IN LINEAR SCALE).

        The path loss will be accounted when calling the corrupt_data
        method.

        If you want to disable the path loss, set `pathloss_value` to
        None.

        Parameters
        ----------
        pathloss_value : float | None
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

    def set_num_antennas(self, num_rx_antennas, num_tx_antennas):
        """
        Set the number of transmit and receive antennas for MIMO
        transmission.

        Set both `num_rx_antennas` and `num_tx_antennas` to None for SISO
        transmission

        Parameters
        ----------
        num_rx_antennas : int
            The number of receive antennas.
        num_tx_antennas : int
            The number of transmit antennas.
        """
        self._tdlchannel.set_num_antennas(num_rx_antennas, num_tx_antennas)

    def corrupt_data(self, signal):
        """
        Transmit the signal through the TDL channel.

        Parameters
        ----------
        signal : np.ndarray
            The signal to be transmitted.

        Returns
        -------
        np.ndarray
            The received signal after transmission through the TDL channel.
        """
        # output = super(SuChannel, self).corrupt_data(signal)
        output = self._tdlchannel.corrupt_data(signal)

        if self._pathloss_value is not None:
            # noinspection PyTypeChecker
            output *= math.sqrt(self._pathloss_value)

        return output

    def corrupt_data_in_freq_domain(self,
                                    signal,
                                    fft_size,
                                    carrier_indexes=None):
        """
        Transmit the signal through the TDL channel, but in the frequency
        domain.

        This is ROUGHLY equivalent to modulating `signal` with OFDM using
        `fft_size` subcarriers, transmitting through a regular TdlChannel,
        and then demodulating with OFDM to recover the received signal.

        One important difference is that here the channel is considered
        constant during the transmission of `fft_size` elements in
        `signal`, and then it is varied by the equivalent of the variation
        for that number of elements. That is, the channel is block static.

        Parameters
        ----------
        signal : np.ndarray
            The signal to be transmitted.
        fft_size : int
            The size of the Fourier transform to get the frequency
            response.
        carrier_indexes : slice | np.ndarray
            The indexes of the subcarriers where signal is to be
            transmitted. If it is None assume all subcarriers will be
            used. This can be a slice object or a numpy array of integers.

        Returns
        -------
        np.ndarray
            The received signal after transmission through the TDL channel
        """
        output = self._tdlchannel.corrupt_data_in_freq_domain(
            signal, fft_size, carrier_indexes)

        if self._pathloss_value is not None:
            # noinspection PyTypeChecker
            output *= math.sqrt(self._pathloss_value)
        return output

    def get_last_impulse_response(self):
        """
        Get the last generated impulse response.

        A new impulse response is generated when the method `corrupt_data`
        is called. You can use the `get_last_impulse_response` method to
        get the impulse response used to corrupt the last data.

        Returns
        -------
        fading.TdlImpulseResponse
            The impulse response of the channel that was used to corrupt
            the last data.
        """
        if self._pathloss_value is None:
            return self._tdlchannel.get_last_impulse_response()
        else:
            # noinspection PyTypeChecker
            return math.sqrt(self._pathloss_value) * \
                self._tdlchannel.get_last_impulse_response()

    @property
    def switched_direction(self):
        """
        Get the value of `switched_direction`.

        Returns
        -------
        bool
            True if direction is switched and False otherwise.
        """
        return self._tdlchannel.switched_direction

    @switched_direction.setter
    def switched_direction(self, value):
        """
        Set the value of `switched_direction`.

        Parameters
        ----------
        value : bool
            True to switch directions of false to use original direction.
        """
        self._tdlchannel.switched_direction = value

    @property
    def num_taps(self):
        """
        Get the number of taps in the profile.

        Returns
        -------
        int
            The number of taps in the channel (not including any zero
            padding).
        """
        return self._tdlchannel.num_taps

    @property
    def num_taps_with_padding(self):
        """
        Get the number of taps in the profile including zero-padding
        when the profile is discretized.

        If the profile is not discretized an exception is raised.

        Returns
        -------
        int
            The number of taps in the channel (including any zero padding).
        """
        return self._tdlchannel.num_taps_with_padding

    @property
    def channel_profile(self):
        """
        Return the channel profile.

        Returns
        -------
        fading.TdlChannelProfile
            The channel profile.
        """
        return self._tdlchannel.channel_profile

    @property
    def num_tx_antennas(self):
        """
        Get the number of transmit antennas.

        Returns
        -------
        int
            The number of transmit antennas.
        """
        return self._tdlchannel.num_tx_antennas

    @property
    def num_rx_antennas(self):
        """
        Get the number of receive antennas.

        Returns
        -------
        int
            The number of receive antennas.
        """
        return self._tdlchannel.num_rx_antennas


class SuMimoChannel(SuChannel):
    """
    Single User channel corresponding to a Tapped Delay Line channel model,
    which corresponds to a multipath channel. You can use a single tap in
    order to get a flat fading channel.

    You can create a new SuMimoChannel object either specifying the
    channel profile or specifying both the channel tap powers and
    delays. If only the fading_generator is specified then a single tap
    with unitary power and delay zero will be assumed, which corresponds
    to a flat fading channel model.

    Parameters
    ----------
    num_antennas : int
        Number of transmit and receive antennas.
    fading_generator : T <= fading_generators.FadingSampleGenerator
        The instance of a fading generator in the `fading_generators`
        module.  It should be a subclass of FadingSampleGenerator. The
        fading generator will be used to generate the channel samples. If
        not provided then RayleighSampleGenerator will be used
    channel_profile : fading.TdlChannelProfile
        The channel profile, which specifies the tap powers and delays.
    tap_powers_dB : np.ndarray
        The powers of each tap (in dB). Dimension: `L x 1`
        Note: The power of each tap will be a negative number (in dB).
    tap_delays : np.ndarray
        The delay of each tap (in seconds). Dimension: `L x 1`
    """

    def __init__(self,
                 num_antennas,
                 fading_generator=None,
                 channel_profile=None,
                 tap_powers_dB=None,
                 tap_delays=None,
                 Ts=None):
        # Before calling supper to initialize the base class we will set
        # the shape of the fading generator
        if fading_generator is None:
            fading_generator = fading_generators.RayleighSampleGenerator()
            if channel_profile is None and Ts is None:
                Ts = 1

        # Set the shape of the fading generator.
        fading_generator.shape = (num_antennas, num_antennas)

        # Initialize attributes from base class
        super(SuMimoChannel, self).__init__(fading_generator, channel_profile,
                                            tap_powers_dB, tap_delays, Ts)
