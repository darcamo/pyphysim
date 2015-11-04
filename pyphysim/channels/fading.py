#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

from pyphysim.channels import fading_generators
from ..util.conversion import dB2Linear, linear2dB

# TODO: change this module name to "singleuser.py"


class TdlChannelProfile(object):
    """
    Channel Profile class.

    This class is just a nice way to present known profiles from the norm
    or the literature, which are represented as instances of this class.

    A TDL channel profile store information about the TDL taps. That is, it
    stores the power and delay of each tap. The power and delay of each tap
    can be accessed through the `tap_powers_*` and `tap_delays` properties.

    Some profiles are defined as objects of this class, such as
    COST259_TUx, COST259_RAx and COST259_HTx. These can be used when
    instantiating a `TdlChannel` obejct.

    Note that the tap powers and delays are not necessarily `discretized`
    to some sampling interval.

    Ex:
        tdlchannel = TdlChannel(jakes_obj,
                                COST259_TUx.tap_powers,
                                COST259_TUx.tap_delays)
    """

    def __init__(self, tap_powers_dB, tap_delays, name='custom'):
        self._name = name
        self._tap_powers_dB = tap_powers_dB.copy()
        self._tap_powers_dB.flags['WRITEABLE'] = False
        self._tap_powers_linear = dB2Linear(tap_powers_dB)
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

        # Sampling interval when the channel profile is discretized. You
        # can call the
        self._Ts = None

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
    def tap_powers_dB(self):
        """Get the tap powers (in dB)"""
        return self._tap_powers_dB

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

    @property
    def num_taps_with_padding(self):
        """
        Get the number of taps in the profile including zero-padding when the
        profile is discretized.

        If the profile is not discretized an exception is raised.
        """
        if self.Ts is None:
            raise RuntimeError('TdlChannelProfile is not discretized')
        else:
            return self._tap_delays[-1] + 1

    @property
    def Ts(self, ):
        """
        Get the sampling interval used for discretizing this channel profile
        object.

        If it is not discretized then this returns None.
        """
        return self._Ts

    def get_discretize_profile(self, Ts):
        """
        Compute the discretized taps (power and delay) and return a new
        discretized TdlChannelProfile object.

        The tap powers and delays of the returned TdlChannelProfile object
        correspond to the taps and delays of the TdlChannelProfile object
        used to call `get_discretize_profile` after discretizing with the
        sampling interval `Ts`.

        Parameters
        ----------
        Ts : float
            The sampling time for the discretization of the tap powers and
            delays.

        Returns
        -------
        TdlChannelProfile
            The discretized channel profile
        """
        if self._Ts is not None:
            raise RuntimeError("Trying to discretize a TdlChannelProfile "
                               "object that is already discretized.")

        name = "{0} (discretized)".format(self.name)
        powers, delays = self._calc_discretized_tap_powers_and_delays(Ts)

        discretized_channel_profile = TdlChannelProfile(powers, delays, name)
        discretized_channel_profile._Ts = Ts

        return discretized_channel_profile

    def _calc_discretized_tap_powers_and_delays(self, Ts):
        """
        Discretize the taps according to the sampling time.

        The discretized taps will be equally spaced and the delta time from
        two taps corresponds to the sampling time.

        Parameters
        ----------
        Ts : float
            The sampling time (used in the Jakes object)

        Returns
        -------
        (discretized_powers_dB, discretized_delays)
            A tuple with the discretized powers and delays.
        """
        # Compute delay indices
        delay_indexes, idx_inverse = np.unique(
            np.round(self._tap_delays / Ts).astype(int).flatten(),
            return_inverse=True)

        discretized_powers_linear = np.zeros(delay_indexes.size)
        for i, v in enumerate(self.tap_powers_linear):
            discretized_idx = idx_inverse[i]
            discretized_powers_linear[discretized_idx] += v
        discretized_powers_linear /= np.sum(discretized_powers_linear)

        # Compute the discretized powers in dB
        discretized_powers_dB = linear2dB(discretized_powers_linear)

        return discretized_powers_dB, delay_indexes

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
COST259_TUx = TdlChannelProfile(np.array([
    -5.7, -7.6, -10.1, -10.2, -10.2, -11.5, -13.4, -16.3, -16.9, -17.1,
    -17.4, -19, -19, -19.8, -21.5, -21.6, -22.1, -22.6, -23.5, -24.3]),
    np.array([0, 217, 512, 514, 517, 674, 882, 1230, 1287, 1311, 1349, 1533,
              1535, 1622, 1818, 1836, 1884, 1943, 2048, 2140]) * 1e-9,
    'COST259_TU')


# COST 259 Rural Area
COST259_RAx = TdlChannelProfile(np.array(
    [-5.2, -6.4, -8.4, -9.3, -10.0, -13.1, -15.3, -18.5, -20.4, -22.4]),
    np.array(
        [0., 42., 101., 129., 149., 245., 312., 410., 469., 528]) * 1e-9,
    'COST259_RA')


# COST 259 Hilly Terrain
COST259_HTx = TdlChannelProfile(np.array(
    [-3.6, -8.9, -10.2, -11.5, -11.8, -12.7, -13.0, -16.2, -17.3, -17.7,
     -17.6, -22.7, -24.1, -25.8, -25.8, -26.2, -29.0, -29.9, -30.0, -30.7]),
    np.array([0., 356., 441., 528., 546., 609., 625., 842., 916., 941., 15000.,
              16172., 16492., 16876., 16882., 16978., 17615., 17827., 17849.,
              18016.]) * 1e-9, 'COST259_HT')


class TdlChannel(object):
    """
    Tapped Delay Line channel model, which corresponds to a multipath
    channel.

    You can create a new TdlChannel object either specifying the channel
    profile or specifying both the channel tap powers and delays.

    Parameters
    ----------
    jakes_obj : JakesSampleGenerator object
        The instance of JakesSampleGenerator that will be used to generate
        the samples.
    channel_profile : TdlChannelProfile
        The channel profile, which specifies the tap powers and delays.
    tap_powers_dB : numpy real array
        The powers of each tap (in dB). Dimension: `L x 1`
        Note: The power of each tap will be a negative number (in dB).
    tap_delays : numpy real array
        The delay of each tap (in seconds). Dimension: `L x 1`
    """

    # TODO: change jakes_obj to a generic fading generator
    def __init__(self, jakes_obj, channel_profile=None,
                 tap_powers_dB=None, tap_delays=None, Ts=None):
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        if isinstance(jakes_obj, fading_generators.JakesSampleGenerator):
            if Ts is None:
                # Ts was not provided, but the fading generator has
                # it. Let's use it then.
                Ts = jakes_obj.Ts
            elif Ts != jakes_obj.Ts:
                # Ts was provided and the fading generator also has it, but
                # they are not the same value. Let's raise an exception
                raise RuntimeError(
                    "The provided sampling interval Ts is "
                    "different from the one in the Jakes sample generator.")
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # If the user didn't provide the channel profile, but it provided
        # the tap powers and delays, then we use them to create a custom
        # channel profile
        if channel_profile is None:
            # If channel_profile is not provided, then tap_powers_dB and
            # tap_powers_dB must be provided and we will use them to create
            # the channel profile object
            channel_profile = TdlChannelProfile(tap_powers_dB, tap_delays)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # The channel profile is not discretized yet. We need to discretize it.
        if channel_profile.Ts is None:
            if Ts is None:
                raise RuntimeError(
                    "You must either provide the Ts argument or provide an "
                    "already discretized TdlChannelProfile object")
            else:
                channel_profile = channel_profile.get_discretize_profile(Ts)
        elif channel_profile.Ts != Ts:
            # Channel profile is already discretized but it does not agree
            # with the Ts value provided or the one in the fading generator
            raise RuntimeError(
                "Channel profile is already discretized, but it does not agree"
                " with the discretized parameter Ts")
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Finally save the channel profile to a member attribute
        self._channel_profile = channel_profile

        if jakes_obj.shape is None:
            jakes_obj.shape = (self.num_taps,)
        else:
            # Note that jakes_obj.shape must be a tuple
            jakes_obj.shape = (self.num_taps,) + jakes_obj.shape
        self._jakes_obj = jakes_obj

    @property
    def num_taps(self):
        """
        Number of taps not including zero taps after discretization.
        """
        return self._channel_profile.num_taps

    @property
    def num_taps_with_padding(self):
        """
        Number of taps including zero taps after discretization.
        """
        # This is only valid if _channel_profile is discretized and the
        # tap_delays correspond to integers
        return self._channel_profile.num_taps_with_padding

    def generate_and_get_samples(self, num_samples):
        """
        Generate `num_samples` of all discretized taps (not including possible
        zero padding) and return the generated samples.

        The number of discretized taps will depend on the channel delay
        profile (the tap_delays passed during creation of the TdlChannel
        object) as well as on the sampling interval (configured in the
        jakes_obj passed to the TdlChannel object).

        As an example, the COST259 TU channel profile has 20 different taps
        where the last one has a delay equal to 2.14 microseconds. If the
        jakes_obj is configured with a sampling time equal to 3.25e-08 then
        the discretized channel will have more than 60 taps (including the
        zeros padding), where only 15 taps are different from zero. These
        15 taps are what is returned by this method.

        Alternatively, with a sampling time of 1e-6 you will end up with
        only 3 discretized taps.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate (for each tap).

        Returns
        -------
        numpy complex array
            The generated samples. Dimension: `Shape of the Jakes obj x
            num_samples`.
        """
        self._jakes_obj.generate_more_samples(num_samples)
        jakes_samples = self._jakes_obj.get_samples()

        # xxxxxxxxxx Apply the power to each tap xxxxxxxxxxxxxxxxxxxxxxxxxx
        # Note that here we only apply the power to the taps. The delays
        # will be applyed when the fading is actually used.

        # Note that self._tap_linear_powers_discretized has a single
        # dimension. We need to add singleton dimensions as necessary
        # before we multiply it by jakes_samples so that broadcasting
        # works.
        new_shape = [self.num_taps]
        new_shape.extend([1] * (jakes_samples.ndim - 1))

        samples = (jakes_samples *
                   np.sqrt(np.reshape(
                       self._channel_profile.tap_powers_linear[:, np.newaxis],
                       new_shape)))

        return samples

    def include_the_zeros_in_fading_map(self, fading_map):
        """
        Return the `fading_map` including the zeros.

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
            [self.num_taps_with_padding, num_samples],
            dtype=complex)

        full_fading_map[self._channel_profile.tap_delays] = fading_map
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
            `generate_and_get_samples`. The first dimension corresponds to
            the delay dimension (taps).
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
        """
        Transmit the signal trhough the TDL channel.

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
        max_delay = self.num_taps_with_padding - 1
        output = np.zeros(num_symbols + max_delay, dtype=complex)

        for i, d in enumerate(self._channel_profile.tap_delays):
            output[d:d + num_symbols] += fading_map[i] * signal

        return output
