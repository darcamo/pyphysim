#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

from . import fading_generators
from ..util.conversion import dB2Linear, linear2dB


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

    Parameters
    ----------
    tap_powers_dB : real numpy array
        The tap powers (in dB). If both `tap_powers_dB` and `tap_delays`
        are None then a single tap with 0dB power will be assummed at delay
        0.
    tap_delays : real numpy array
        The tap delays.
    name : str
        A name for the channel profile

    Examples
    --------
    >>> jakes_generator = fading_generators.JakesSampleGenerator(Ts=3.25e-8)
    >>> tdlchannel = TdlChannel(jakes_generator, channel_profile=COST259_TUx)
    """

    def __init__(self, tap_powers_dB=None, tap_delays=None, name='custom'):
        self._name = name
        if tap_powers_dB is None and tap_delays is None:
            tap_powers_dB = np.zeros(1)
            tap_delays = np.zeros(1)

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

    @property
    def is_discretized(self):
        if self._Ts is None:
            return False
        else:
            return True

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
        if self.is_discretized:
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
            The sampling time.

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


class TdlImpulseResponse(object):
    """
    Class that represents impulse response for a TdlChannel object.

    This impulse response corresponds to the generated samples for one or
    more channel realization of the TdlChannel with the configured fading
    generator.

    Parameters
    ----------
    tap_values : numpy array
        The tap_values (not including zero padded taps) of a TDL channel
        generated for the non-zero taps. Dimension: `Num sparse taps x
        SHAPE x num_samples`. The value SHAPE here is the shape of the
        fading generator and corresponds to independent impulse
        responses. Often the shape of the used fading generator is None and
        thus the dimension of `tap_values` is just `Num sparse taps x
        num_samples`

    channel_profile : TdlChannelProfile
        The channel profile that was considering to generate this impulse
        response.
    """
    def __init__(self, tap_values, channel_profile):
        assert(isinstance(channel_profile, TdlChannelProfile))
        if channel_profile.Ts is None:
            raise RuntimeError('Channel profile must be discretized')

        self._channel_profile = channel_profile

        self._tap_values_sparse = tap_values
        self._tap_values_dense = None  # This will be set when needed

    @property
    def tap_values_sparse(self):
        """
        Return the tap values (not including zero padding) as a numpy array.
        """
        return self._tap_values_sparse

    @property
    def tap_indexes_sparse(self):
        """
        Return the (sparse) tap indexes.
        """
        return self._channel_profile.tap_delays

    @property
    def Ts(self):
        """
        Return the sampling interval of this impulse response.
        """
        return self._channel_profile.Ts

    @property
    def tap_delays_sparse(self):
        """
        Return the tap delays (which are multiples of the sampling interval).
        """
        return self.tap_indexes_sparse * self.Ts

    @property
    def tap_values(self):
        """
        Return the tap values (including zero padding) as a numpy array.
        """
        if self._tap_values_dense is None:
            self._tap_values_dense = \
                self._get_samples_including_the_extra_zeros()
        return self._tap_values_dense

    @property
    def num_samples(self):
        """
        Get the number of samples (different, "neighbor" impulse responses)
        stored here.
        """
        return self._tap_values_sparse.shape[-1]

    @property
    def channel_profile(self):
        """
        Return the channel profile.
        """
        return self._channel_profile

    def _get_samples_including_the_extra_zeros(self):
        """
        Return the `samples` including the zeros for the zero taps.

        Returns
        -------
        samples_with_zeros : numpy complex array
            The samples including the extra delays containing zeros.
        """
        num_taps_with_padding = self.tap_indexes_sparse[-1] + 1

        # Shape of sparse tap values
        orig_shape = self._tap_values_sparse.shape
        # The first dimension has the sparse taps. This dimension will
        # change to num_taps_with_padding. Note that the last dimension in
        # this shape corresponds to the number of samples.
        new_shape = (num_taps_with_padding, ) + orig_shape[1:]

        samples_with_zeros = np.zeros(new_shape, dtype=complex)

        # Fill the non-zero taps in samples_with_zeros in the correct
        # indexes
        samples_with_zeros[self.tap_indexes_sparse] = self._tap_values_sparse

        # Disable write on this array so that it has no risk of deviating
        # from self._tap_values_sparse
        samples_with_zeros.flags['WRITEABLE'] = False

        return samples_with_zeros

    def get_freq_response(self, fft_size):
        """
        Get the frequency response for this impulse response.

        Parameters
        ----------
        fft_size : int
            The size of the FFT to be applied.

        Returns
        -------
        numpy array
            The frequency response. Dimension: `fft_size x num_samples` for
            SISO impulse response or `fft_size x num_rx x num_tx x num_samples`
            for MIMO impulse response.
        """
        # Compute the FFT in the "delay" dimension, which captures the
        # multipath characteristics of the channel. The FFT is calculated
        # independently for each column (second dimension), which
        # corresponds to the second dimension is the time dimension (as the
        # channel response changes in time)
        freq_response = np.fft.fft(
            self._get_samples_including_the_extra_zeros(), fft_size, axis=0)
        return freq_response

    def __mul__(self, value):
        """
        Multiply the impulse response by a float returning a new (scaled)
        impulse response.

        Only the tap values are modified.

        Parameters
        ----------
        value : float
            The number to multiply the impulse response object.

        Returns
        -------
        TdlImpulseResponse
            A new (scaled by `value`) TdlImpulseResponse object.
        """
        return TdlImpulseResponse(value * self._tap_values_sparse,
                                  self._channel_profile)

    def __rmul__(self, value):
        """
        Multiply the impulse response by a float returning a new (scaled)
        impulse response.

        Only the tap values are modified.

        Parameters
        ----------
        value : float
            The number to multiply the impulse response object.

        Returns
        -------
        TdlImpulseResponse
            A new (scaled by `value`) TdlImpulseResponse object.
        """
        return self * value

    # noinspection PyUnresolvedReferences
    def plot_impulse_response(self):  # pragma: no cover
        """
        Plot the impulse response.
        """
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d.art3d as art3d

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        num_taps_with_padding = self._channel_profile.num_taps_with_padding
        x = np.arange(num_taps_with_padding)  # 0 -> 66

        for i in range(self.num_samples):
            z = np.abs(self.tap_values[:, i])
            ax.plot(x, [i] * num_taps_with_padding, z)

        # for y_ in y:
        #     z = np.abs(self.tap_values[:, y_])
        #     for x_, z_ in zip(x, z):
        #         line = Line3D(*zip((x_, y_, 0), (x_, y_, z_)),
        #                       marker='o', markevery=(1, 1))
        #         ax.add_line(line)

        ax.set_xlabel('Taps (delay domain)')
        ax.set_ylabel('Time Domain')
        ax.set_zlabel('Channel Amplitude')

        ax.set_xlim3d(0, num_taps_with_padding)
        ax.set_ylim3d(0, self.num_samples)
        ax.set_zlim3d(np.abs(self.tap_values).min(),
                      np.abs(self.tap_values).max())

        plt.show()

    # noinspection PyUnresolvedReferences
    def plot_frequency_response(self, fft_size):  # pragma: no cover
        """
        Plot the frequency response.

        Parameters
        ----------
        fft_size : int
            The size of the FFT to be applied.
        """
        import matplotlib.pyplot as plt
        import mpl_toolkits.mplot3d.art3d as art3d

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = np.arange(fft_size)
        freq_response = self.get_freq_response(fft_size)

        for i in range(self.num_samples):
            z = np.abs(freq_response[:, i])
            ax.plot(x, [i] * fft_size, z)

        ax.set_xlabel('Taps (delay domain)')
        ax.set_ylabel('Time Domain')
        ax.set_zlabel('Channel Amplitude')

        ax.set_xlim3d(0, fft_size)
        ax.set_ylim3d(0, self.num_samples)
        ax.set_zlim3d(np.abs(freq_response).min(),
                      np.abs(freq_response).max())

        plt.show()

    @staticmethod
    def concatenate_samples(list_of_impulse_responses):
        """
        Concatenate multiple TdlImpulseResponse objects and return the new
        concatenated TdlImpulseResponse.

        This contatenation is performed in the "samples" dimension.

        Parameters
        ----------
        list_of_impulse_responses : list[TdlImpulseResponse]
            A list of TdlImpulseResponse objects to be concatenated.

        Returns
        -------
        TdlImpulseResponse
            The new concatenated TdlImpulseResponse.
        """
        num_objs = len(list_of_impulse_responses)
        if num_objs < 2:
            if num_objs == 1:
                return list_of_impulse_responses[0]
            else:  # pragma: no cover
                raise ValueError("list_of_impulse_responses must contain at "
                                 "least two TdlImpulseResponse objects.")

        # We should test if all elements in list_of_impulse_responses have
        # the same profile, but in order to avoid too much overhead we only
        # test the first two.
        channel_profile1 = list_of_impulse_responses[0].channel_profile
        channel_profile2 = list_of_impulse_responses[1].channel_profile
        if channel_profile1 is not channel_profile2:
            raise ValueError("TdlImpulseResponse objects must have the same "
                             "channel profile object")

        tap_values_sparse = np.concatenate(
            [a.tap_values_sparse for a in list_of_impulse_responses], axis=-1)

        concatenated_impulse_response = TdlImpulseResponse(
            tap_values_sparse, channel_profile1)

        return concatenated_impulse_response


class TdlChannel(object):
    """
    Tapped Delay Line channel model, which corresponds to a multipath
    channel.

    You can create a new TdlChannel object either specifying the channel
    profile or specifying both the channel tap powers and delays.

    Parameters
    ----------
    fading_generator : Subclass of FadingSampleGenerator
        The instance of a fading generator in the `fading_generators` module.
        It should be a subclass of FadingSampleGenerator. The fading
        generator will be used to generate the channel samples.
        If the shape of the fading_generator is not None, then it must
        contain two positive integers, and a MIMO transmission will be
        employed, where the first integer in shape corresponds to the
        number of receive antennas while the second integer corresponds to
        the number of transmit antennas
    channel_profile : TdlChannelProfile
        The channel profile, which specifies the tap powers and delays.
    tap_powers_dB : numpy real array
        The powers of each tap (in dB). Dimension: `L x 1`
        Note: The power of each tap will be a negative number (in dB).
    tap_delays : numpy real array
        The delay of each tap (in seconds). Dimension: `L x 1`
    """
    # Note: It would be better to have only the first argument as
    # positional argument and all the others as keyword only arguments. We
    # can do this in Python3 by adding ",*," after the first positional
    # argument thus making all the other arguments keyword only. However,
    # this is not valid in Python2.
    def __init__(self, fading_generator, channel_profile=None,
                 tap_powers_dB=None, tap_delays=None, Ts=None):
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        if isinstance(fading_generator,
                      fading_generators.JakesSampleGenerator):
            if Ts is None:
                # Ts was not provided, but the fading generator has
                # it. Let's use it then.
                Ts = fading_generator.Ts
            elif Ts != fading_generator.Ts:
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
        else:
            assert(isinstance(channel_profile, TdlChannelProfile)),\
                'channel_profile must be an obj of the TdlChannelProfile class'
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # The channel profile is not discretized yet. We need to discretize it.
        if not channel_profile.is_discretized:
            if (isinstance(fading_generator,
                           fading_generators.RayleighSampleGenerator) and
                    Ts is None):
                Ts = 1
            # if Ts is None:
            #     raise RuntimeError(
            #         "You must either provide the Ts argument or provide an "
            #         "already discretized TdlChannelProfile object")
            # else:
            #     channel_profile = channel_profile.get_discretize_profile(Ts)
            channel_profile = channel_profile.get_discretize_profile(Ts)
        elif channel_profile.Ts != Ts and Ts is not None:
            # Channel profile is already discretized but it does not agree
            # with the Ts value provided or the one in the fading generator
            raise RuntimeError(
                "Channel profile is already discretized, but it does not "
                "agree with the discretized parameter Ts")
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Finally save the channel profile to a member attribute
        self._channel_profile = channel_profile

        shape = fading_generator.shape
        self._fading_generator = fading_generator
        self._set_fading_generator_shape(shape)

        # Last generated impulse response. This will be set when the
        # _generate_impulse_response method is called
        self._last_impulse_response = None

    def set_num_antennas(self, num_rx_antennas, num_tx_antennas):
        """
        Set the number of transmit and receive antennas for MIMO transmission.

        Set both `num_rx_antennas` and `num_tx_antennas` to None for SISO
        transmission

        Parameters
        ----------
        num_rx_antennas : int
            The number of receive antennas.
        num_tx_antennas : int
            The number of transmit antennas.
        """
        self._set_fading_generator_shape((num_rx_antennas, num_tx_antennas))

    def _set_fading_generator_shape(self, new_shape):
        """
        Set the shape of the fading generator.

        Parameters
        ----------
        new_shape : tuple of integers
            The new shape of the fading generator. Note that the actual
            shape will be set to (self.num_taps, new_shape)
        """
        if new_shape is None:
            self._fading_generator.shape = (self.num_taps,)
        else:
            # Note that fading_generator.shape must be a tuple
            self._fading_generator.shape = (self.num_taps,) + new_shape

    @property
    def channel_profile(self):
        """
        Return the channel profile.
        """
        return self._channel_profile

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

    def _generate_impulse_response(self, num_samples=1):
        """
        Generate a new impulse response of all discretized taps (not
        including possible zero padding) for `num_samples` channel
        realizations.

        The generated impulse response is saved in the
        `_last_impulse_response` attribute.

        The number of discretized taps of the generated impulse response will
        depend on the channel delay profile (the tap_delays passed during
        creation of the TdlChannel object) as well as on the sampling
        interval.

        As an example, the COST259 TU channel profile has 20 different taps
        where the last one has a delay equal to 2.14 microseconds. If the
        sampling interval is configured as 3.25e-08 then the discretized
        channel will have more than 60 taps ( including the zeros padding),
        where only 15 taps are different from zero. These 15 taps are what is
        returned by this method.

        Alternatively, with a sampling time of 1e-6 you will end up with
        only 3 discretized taps.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate (for each tap).
        """
        self._fading_generator.generate_more_samples(num_samples)
        channel_samples = self._fading_generator.get_samples()

        # xxxxxxxxxx Apply the power to each tap xxxxxxxxxxxxxxxxxxxxxxxxxx
        # Note that here we only apply the power to the taps. The delays
        # will be applyed when the fading is actually used.

        # Note that self._tap_linear_powers_discretized has a single
        # dimension. We need to add singleton dimensions as necessary
        # before we multiply it by channel_samples so that broadcasting
        # works.
        new_shape = [self.num_taps]
        new_shape.extend([1] * (channel_samples.ndim - 1))

        samples = (channel_samples *
                   np.sqrt(np.reshape(
                       self._channel_profile.tap_powers_linear[:, np.newaxis],
                       new_shape)))

        impulse_response = TdlImpulseResponse(samples, self._channel_profile)
        self._last_impulse_response = impulse_response

    @property
    def num_tx_antennas(self):
        """Get the number of transmit antennas.
        """
        if len(self._fading_generator.shape) == 1:
            return -1
        return self._fading_generator.shape[2]

    @property
    def num_rx_antennas(self):
        """Get the number of receive antennas.
        """
        if len(self._fading_generator.shape) == 1:
            return -1
        return self._fading_generator.shape[1]

    def get_last_impulse_response(self):
        """
        Get the last generated impulse response.

        A new impulse response is generated when the method `corrupt_data`
        is called. You can use the `get_last_impulse_response` method to
        get the impulse response used to corrupt the last data.

        Returns
        -------
        TdlImpulseResponse
            The impulse response of the channel that was used to corrupt
            the last data.
        """
        return self._last_impulse_response

    def corrupt_data(self, signal):
        """
        Transmit the signal trhough the TDL channel.

        Parameters
        ----------
        signal : numpy array
            The signal to be transmitted.

        Returns
        -------
        numpy array
            The received signal after transmission through the TDL channel
        """
        # Number of symbols to be transmitted
        num_symbols = signal.shape[-1]

        # Generate an impulse response with `num_symbols` samples that we
        # will use to corrupt the data.
        self._generate_impulse_response(num_symbols)

        # Get the channel memory (number of extra received symbols).
        channel_memory = self.num_taps_with_padding - 1

        # The indexes of the non-zero taps from our impulse response
        tap_indexes_sparse = self._last_impulse_response.tap_indexes_sparse
        # The values of the (sparse) tap
        tap_values_sparse = self._last_impulse_response.tap_values_sparse

        if len(self._fading_generator.shape) == 1:
            # xxxxxxxxxx SISO Case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            # The output size will be equal to the number of symbols to transit
            # plus the channel_memory.
            output = np.zeros(num_symbols + channel_memory, dtype=complex)

            for i, d in enumerate(tap_indexes_sparse):
                output[d:d + num_symbols] += tap_values_sparse[i] * signal
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        elif len(self._fading_generator.shape) == 3:
            # xxxxxxxxxx MIMO Case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # The output size will be equal to the number of symbols to transit
            # plus the channel_memory.
            _, num_rx_ant, num_tx_ant = self._fading_generator.shape

            output = np.zeros((num_rx_ant, num_symbols + channel_memory),
                              dtype=complex)

            for i, d in enumerate(tap_indexes_sparse):
                for tx_idx in range(num_tx_ant):
                    output[:, d:d + num_symbols] += (
                        tap_values_sparse[i, :, tx_idx, :] * signal[tx_idx])
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        else:
            raise RuntimeError(
                "Shape of the fading generator of the TdlChannel class must "
                "have either 1 (SISO) or 3 (MIMO) dimensions")

        return output

    def corrupt_data_in_freq_domain(self, signal, fft_size,
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
        signal : numpy array
            The signal to be transmitted.
        fft_size : int
            The size of the Fourier transform to get the frequency response.
        carrier_indexes : slice or numpy array of integers
            The indexes of the subcarriers where signal is to be
            transmitted. If it is None assume all subcarriers will be used.

        Returns
        -------
        numpy array
            The received signal after transmission through the TDL channel
        """
        # Number of symbols to be transmitted
        num_symbols = signal.shape[-1]

        # xxxxxxxxxx Get the block size xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        if carrier_indexes is None:
            block_size = fft_size
        else:
            # carrier_indexes may be either a slice object of a numpy array
            # of integers with indexes
            if isinstance(carrier_indexes, slice):
                # Get the indexes from the slice object. This is a tuple
                # with (start, stop, step)
                indexes = carrier_indexes.indices(fft_size)
                block_size = (indexes[1] - indexes[0]) // indexes[2]
            else:
                block_size = len(carrier_indexes)

        if num_symbols % block_size != 0:
            raise ValueError("The num of elements in `signal` must be a "
                             "multiple of number of sent elements per "
                             "`fft_size`.")
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Variable to stopre the impulse responses for each block. We will
        # concatenate these impulse responses at the end so that we can set
        # self._last_impulse_response to the impulse response of all blocks
        impulse_responses = []
        num_tx_ant = -1  # This will be set latter

        if len(self._fading_generator.shape) == 1:
            # Output variable representing the received signal
            output = np.empty(num_symbols, dtype=complex)
        elif len(self._fading_generator.shape) == 3:
            _, num_rx_ant, num_tx_ant = self._fading_generator.shape
            output = np.zeros((num_symbols, num_rx_ant), dtype=complex)
        else:
            raise RuntimeError(
                "Shape of the fading generator of the TdlChannel class must "
                "have either 1 (SISO) or 3 (MIMO) dimensions")

        # Number of full blocks in `signal`
        num_full_blocks = num_symbols // block_size

        for i in range(num_full_blocks):
            start_idx = block_size * i
            end_idx = block_size * (i + 1)

            # Generate next impulse response: the one we will use to
            # transmit the current block (the channel is static during
            # transmission of a single block)
            self._generate_impulse_response(1)
            impulse_responses.append(self.get_last_impulse_response())

            if len(self._fading_generator.shape) == 1:
                # xxxxxxxxxx SISO case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
                # Get the equivalent frequency response of the last generated
                # impulse response. That is what we will use to corrupt the
                # current block of signal
                if carrier_indexes is None:
                    freq_response = \
                        self._last_impulse_response.get_freq_response(
                            fft_size)[:, 0]
                else:
                    freq_response = \
                        self._last_impulse_response.get_freq_response(
                            fft_size)[carrier_indexes, 0]

                output[start_idx:end_idx] = (freq_response *
                                             signal[start_idx:end_idx])
                # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            else:  # len(self._fading_generator.shape) == 3
                # xxxxxxxxxx MIMO Case xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
                # Get the equivalent frequency response of the last generated
                # impulse response. That is what we will use to corrupt the
                # current block of signal
                if carrier_indexes is None:
                    freq_response = \
                        self._last_impulse_response.get_freq_response(
                            fft_size)[:, :, :, 0]
                else:
                    freq_response = \
                        self._last_impulse_response.get_freq_response(
                            fft_size)[carrier_indexes, :, :, 0]

                for tx_idx in range(num_tx_ant):
                    output[start_idx:end_idx, :] += (
                        freq_response[:, :, tx_idx] *
                        signal[tx_idx, start_idx:end_idx, np.newaxis])
                # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # Advance the fading generator by "fft_size - 1" to account how
            # much the channel has "changed" during the duration of the
            # current block
            self._fading_generator.skip_samples_for_next_generation(
                fft_size - 1)

        self._last_impulse_response = TdlImpulseResponse.concatenate_samples(
            impulse_responses)

        # Transposition has no effect for the SISO case. For the MIMO case
        # it will make output have dimension `num_rx_ant x num_samples`
        return output.T


class TdlMimoChannel(TdlChannel):
    """
    Tapped Delay Line channel model, which corresponds to a multipath
    channel.

    You can create a new TdlMimoChannel object either specifying the
    channel profile or specifying both the channel tap powers and delays.

    Note that the TdlChannel class can already work with multiple antennas
    if provided `fading_generator` has a shape with two elements (number of
    receive antennas and number of transmit antennas). The TdlMimoChannel
    only adds a slight better interface over TdlChannel class for working
    with MIMO. This class is also useful to test MIMO transmission, with
    the added `num_tx_antennas` and `num_rx_antennas` properties.

    Parameters
    ----------
    fading_generator : Subclass of FadingSampleGenerator
        The instance of a fading generator in the `fading_generators`
        module.  It should be a subclass of FadingSampleGenerator. The
        fading generator will be used to generate the channel samples.  The
        shape of the fading_generator will be ignored and replaced by
        provided number of antennas.
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
        if fading_generator.shape is None or len(fading_generator.shape) != 2:
            raise RuntimeError(
                "The provided fading_generator for the TdlMimoChannel class"
                " must have a shape with two values")

        super(TdlMimoChannel, self).__init__(fading_generator, channel_profile,
                                             tap_powers_dB, tap_delays, Ts)
