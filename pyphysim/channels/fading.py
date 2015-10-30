#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
from pyphysim.util.conversion import dB2Linear


class TdlChannelProfile(object):
    """
    Channel Profile class.

    This class is just a nice way to present known profiles from the norm
    or the literature, which are represented as instances of this class.

    A TDL channel profile store information about the TDL taps. That is, it
    stores the power and delay of each tap. The power and delay of each tap
    can be accessed through the `tap_powers` and `tap_delays` properties.

    Some profiles are defined as objects of this class, such as
    COST259_TUx, COST259_RAx and COST259_HTx. These can be used when
    instantiating a `TdlChannel` obejct.

    Ex:
        tdlchannel = TdlChannel(jakes_obj,
                                COST259_TUx.tap_powers,
                                COST259_TUx.tap_delays)
    """

    def __init__(self, name, tap_powers, tap_delays):
        """
        """
        self._name = name
        self._tap_powers = tap_powers.copy()
        self._tap_powers.flags['WRITEABLE'] = False
        self._tap_powers_linear = dB2Linear(tap_powers)
        self._tap_powers_linear.flags['WRITEABLE'] = False
        self._tap_delays = tap_delays.copy()
        self._tap_delays.flags['WRITEABLE'] = False
        self._num_taps = tap_delays.size

        self._mean_excess_delay \
            = (np.sum(self._tap_powers_linear * self._tap_delays) /
               np.sum(self._tap_powers_linear))

        aux = (np.sum(self._tap_powers_linear * self._tap_delays ** 2) /
               np.sum(self._tap_powers_linear))
        self._rms_delay_spread = math.sqrt(aux - self._mean_excess_delay ** 2)

    # noinspection PyPep8
    @property
    def mean_excess_delay(self):
        """
        The mean excess delay is the first moment of the power delay profile
        and is defined to be
            math:`\\overline{\\tau} = \\frac{\\sum_k P(\\tau_k)\\tau_k}{\\sum_k P(\\tau_k)}`
        """
        # $$\overline{\tau} = \frac{\sum_k P(\tau_k)\tau_k}{\sum_k P(\tau_k)}$$
        return self._mean_excess_delay

    # noinspection PyPep8
    @property
    def rms_delay_spread(self):
        """
        The RMS delay spread is the square root of the second central moment of
        the power delay profile. It is defined to be
            math:`\\sigma_t = \\sqrt{\\overline{t^2} - \\overline{\\tau}^2}`
        where
            math:`\overline{\tau^2}=\frac{\sum_k P(\tau_k)\tau_k^2}{\sum_k P(\tau_k)}`

         Typically, when the symbol time period is greater than 10 times
        the RMS delay spread, no ISI equalizer is needed in the receiver.
        """
        # $\sigma_t = \sqrt{\overline{t^2} - \overline{\tau}^2}$
        # where
        # $\overline{\tau^2}=\frac{\sum_k P(\tau_k)\tau_k^2}{\sum_k P(\tau_k)}$
        return self._rms_delay_spread

    @property
    def name(self):
        """Get the profile name"""
        return self._name

    @property
    def tap_powers(self):
        """Get the tap powers (in dB)"""
        return self._tap_powers

    @property
    def tap_powers_linear(self):
        """Get the tap powers (in linear scale)"""
        return self._tap_powers_linear

    @property
    def tap_delays(self):
        """Get the tap delays"""
        return self._tap_delays

    @property
    def num_taps(self):
        """Get the number of taps in the profile."""
        return self._num_taps

    def __repr__(self):  # pragma: no cover
        return "<TdlChannelProfile: '{0}' ({1} taps)>".format(
            self.name, self.num_taps)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Channel Profiel Classes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Channel profiles define the power and delays (according to the norm) for
# use with when creating TdlChannel objects
# xxxxxxxxxx Define some known profiles xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Reference: 3GPP TR 25.943 version 9.0.0 Release 9

# COST 259 Typical Urban
COST259_TUx = TdlChannelProfile(
    'COST259_TU',
    np.array([
        -5.7, -7.6, -10.1, -10.2, -10.2, -11.5, -13.4, -16.3, -16.9, -17.1,
        -17.4, -19, -19, -19.8, -21.5, -21.6, -22.1, -22.6, -23.5, -24.3]),
    np.array([0, 217, 512, 514, 517, 674, 882, 1230, 1287, 1311, 1349, 1533,
              1535, 1622, 1818, 1836, 1884, 1943, 2048, 2140]) * 1e-9)


# COST 259 Rural Area
COST259_RAx = TdlChannelProfile(
    'COST259_RA',
    np.array(
        [-5.2, -6.4, -8.4, -9.3, -10.0, -13.1, -15.3, -18.5, -20.4, -22.4]),
    np.array(
        [0., 42., 101., 129., 149., 245., 312., 410., 469., 528]) * 1e-9)


# COST 259 Hilly Terrain
COST259_HTx = TdlChannelProfile(
    'COST259_HT',
    np.array(
        [-3.6, -8.9, -10.2, -11.5, -11.8, -12.7, -13.0, -16.2, -17.3, -17.7,
         -17.6, -22.7, -24.1, -25.8, -25.8, -26.2, -29.0, -29.9, -30.0, -30.7]),
    np.array([0., 356., 441., 528., 546., 609., 625., 842., 916., 941., 15000.,
              16172., 16492., 16876., 16882., 16978., 17615., 17827., 17849.,
              18016.]) * 1e-9)


class TdlChannel(object):
    """
    Tapped Delay Line channel model, which corresponds to a multipath
    channel.

    Parameters
    ----------
    jakes_obj : JakesSampleGenerator object
        The instance of JakesSampleGenerator that will be used to generate
    tap_powers : numpy real array
        The powers of each tap (in dB). Dimension: `L x 1`
        Note: The power of each tap will be a negative number (in dB).
    tap_delays : numpy real array
        The delay of each tap (in seconds). Dimension: `L x 1`
    """

    def __init__(self, jakes_obj, tap_powers, tap_delays):
        """
        Init method.
        """
        self._tap_powers = tap_powers  # Tap powers before the discretization
        self._tap_delays = tap_delays  # Tap delays before the discretization
        self._num_taps = tap_delays.size  # Number of taps

        # These will be set in _calc_discretized_tap_powers_and_delays
        self._tap_linear_powers_discretized = None  # Discretized tap powers
        self._tap_delays_discretized = None  # Discretized tap delays
        self._num_discretized_taps = None  # Number of discretized taps
        # Size of the impulse response after discretization.
        self._num_discretized_taps_with_padding = None

        Ts = jakes_obj.Ts
        self._calc_discretized_tap_powers_and_delays(Ts)

        if jakes_obj.shape is None:
            jakes_obj.shape = (self._num_discretized_taps,)
        else:
            # Note that jakes_obj.shape must be a tuple
            jakes_obj.shape = (self._num_discretized_taps,) + jakes_obj.shape
        self._jakes_obj = jakes_obj

        # paramSt.speedTerminal = 3/3.6;
        # paramSt.fcDbl = 2.6e9;
        # paramSt.bandSystemInt = 5e6;
        # paramSt.timeTTIDbl = 1e-3;
        # paramSt.subcarrierBandDbl = 15e3;
        # paramSt.numOfSubcarriersPRBInt = 12;
        # paramSt.fadingModel = enums.MimoChannelModel.TU;

    @staticmethod
    def create_from_channel_profile(jakes_obj, channel_profile):
        """
        Create a new TdlChannel object from the channel_profile (an object of
        the TdlChannelProfile class).

        Parameters
        ----------
        jakes_obj : JakesSampleGenerator object
            The instance of JakesSampleGenerator that will be used to generate
        channel_profile : An object of TdlChannelProfile class.
            The channel profile knows the number of taps, the tap powers
            and the tap delays.
        """
        tap_powers = channel_profile.tap_powers
        tap_delays = channel_profile.tap_delays

        return TdlChannel(jakes_obj, tap_powers, tap_delays)

    def _calc_discretized_tap_powers_and_delays(self, Ts):
        """
        Discretize the taps according to the sampling time.

        The discretized taps will be equally spaced and the delta time from
        two taps corresponds to the sampling time.

        This method will set the `_tap_linear_powers_discretized` and
        `_tap_delays_discretized` attributes from the values in the
        `_tap_delays` and `_tap_powers` atributes.

        Parameters
        ----------
        Ts : float
            The sampling time (used in the Jakes object)
        """

        # Compute delay indices
        delay_indexes, idx_inverse = np.unique(
            np.round(self._tap_delays / Ts).astype(int).flatten(),
            return_inverse=True)

        # tap powers in linear scale
        tap_powers_lin = dB2Linear(self._tap_powers)
        # Force mean to 1
        tap_powers_lin = tap_powers_lin / np.sum(tap_powers_lin)

        self._tap_linear_powers_discretized = np.zeros(delay_indexes.size)
        for i, v in enumerate(tap_powers_lin):
            discretized_idx = idx_inverse[i]
            self._tap_linear_powers_discretized[discretized_idx] += v

        self._tap_delays_discretized = delay_indexes

        self._num_discretized_taps = delay_indexes.size
        self._num_discretized_taps_with_padding = delay_indexes[-1] + 1

    def get_fading_map(self, NSamples):
        """
        Generate `NSamples` of all (discretized) taps and return the generated
        map.

        The number of discretized taps will depend on the channel delay
        profile (the tap_delays passed during creation of the TdlChannel
        object) as well as on the sampling interval (configured in the
        jakes_obj passed to the TdlChannel object).

        As an example, the COST259 TU channel profile has 20 different taps
        where the last one has a delay equal to 2.14 microseconds. If the
        jakes_obj is configured with a sampling time equal to 3.25e-08 then
        the discretized channel will have more than 60 taps (most of them
        will be zeros, though). Alternatively, with a sampling time of
        1e-6 you will end up with only 3 discretized taps.

        Parameters
        ----------
        NSamples : int
            The number of samples to generate (for each tap).

        Returns
        -------
        samples : numpy complex array
            The generated samples. Dimens.: `Shape of the Jakes obj x NSamples`
        """
        jakes_samples = self._jakes_obj.generate_channel_samples(NSamples)

        # xxxxxxxxxx Apply the power to each tap xxxxxxxxxxxxxxxxxxxxxxxxxx
        # Note that here we only apply the power to the taps. The delays
        # will be applyed when the fading is actually used.

        # Note that self._tap_linear_powers_discretized has a single
        # dimension. We need to add singleton dimensions as necessary
        # before we multiply it by jakes_samples so that broadcasting
        # works.
        new_shape = [self._tap_linear_powers_discretized.shape[0]]
        new_shape.extend([1] * (jakes_samples.ndim - 1))

        samples = jakes_samples * np.sqrt(
            np.reshape(self._tap_linear_powers_discretized[:, np.newaxis],
                       new_shape)
        )

        return samples

    def include_the_zeros_in_fading_map(self, fading_map):
        """
        Return the fading_map including the zeros.

        Parameters
        ----------
        fading_map : numpy complex array
            The fading map including any required delays with zeros.

        Returns
        -------
        full_fading_map : numpy complex array
            The fading map including the extra delays containing zeros.
        """
        num_samples = fading_map.shape[1]
        full_fading_map = np.zeros(
            [self._num_discretized_taps_with_padding, num_samples],
            dtype=complex)

        full_fading_map[self._tap_delays_discretized] = fading_map
        return full_fading_map

    @staticmethod
    def get_channel_freq_response(full_fading_map, fft_size):
        """
        Get the frequency response for the given fadding map, computed with
        `np.fft.fft`.

        Parameters
        ----------
        full_fading_map : Numpy complex array
            The fading map (including any extra zeros) calculated by
            `get_fading_map`. The first dimension corresponds to the delay
            dimension (taps).
        fft_size : int
            The size of the FFT to be applied.

        Returns
        -------
        freq_response : numpy array
            The frequency response. Dimension: `fft_size x num_samples`
        """
        # Compute the FFT in the "delay" dimension, which captures the
        # multipath characteristics of the channel. The FFT is calculated
        # independently for each column (second dimension), which
        # corresponds to the second dimension is the time dimension (as the
        # channel response changes in time)
        freq_response = np.fft.fft(full_fading_map, fft_size, axis=0)
        return freq_response

    def transmit_signal_with_known_fading_map(self, signal, fading_map):
        """Transmit the signal trhough the TDL channel.

        Parameters
        ----------
        signal : numpy array
            The signal to be transmitted.
        fading_map : numpy array
            The fading map to use.
        """
        # Number of symbols to be transmitted
        num_symbols = signal.size
        # Maximum (discretized) delay of the channel. The output size will
        # be equal to the number of symbols to transit plus the max_delay.
        max_delay = self._num_discretized_taps_with_padding - 1
        output = np.zeros(num_symbols + max_delay, dtype=complex)

        for i, d in enumerate(self._tap_delays_discretized):
            output[d:d + num_symbols] += fading_map[i] * signal

        return output
