#!/usr/bin/env python
"""Module containing multiuser channels.

The :class:`MultiUserChannelMatrix` and
:class:`MultiUserChannelMatrixExtInt` classes implement the MIMO
Interference Channel (MIMO-IC) model, where the first one does not include
an external interference source while the last one includes it. The MIMO-IC
model is shown in the Figure below.

.. figure:: /_images/mimo_ic.svg
   :align: center

   MIMO Interference Channel
"""

import math
from numbers import Number
from typing import Iterable, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
from scipy.linalg import block_diag

from ..util.conversion import single_matrix_to_matrix_of_matrices
from ..util.misc import randn_c_RS
from . import singleuser
from .fading import TdlChannelProfile, TdlImpulseResponse
from .fading_generators import JakesSampleGenerator, RayleighSampleGenerator

# Either a jakes or a Rayleigh model can be used
FadingGenerator = Union[JakesSampleGenerator, RayleighSampleGenerator]

IntOrIntArrayUnion = Union[np.ndarray, int]
NumberOrArray = TypeVar("NumberOrArray", np.ndarray, float)

# Type representing something that can be used to index a numpy array
Indexes = Union[np.ndarray, List[int], slice]

# Seed is either int or an array_like type
Seed = Union[int, List[int], np.ndarray]


class MuChannel:
    """
    SISO multiuser channel.

    Each transmitter sends data to its own receiver while interfering to
    other receivers.

    Note that noise is NOT added.

    Parameters
    ----------
    N : int | tuple[int,int]
        The number of transmit/receive pairs.
    fading_generator : T <= fading_generators.FadingSampleGenerator
        The instance of a fading generator in the `fading_generators`
        module. It should be a subclass of FadingSampleGenerator. The
        fading generator will be used to generate the channel
        samples. However, since we have multiple links, the provided fading
        generator will actually be used to create similar (but independent)
        fading generators. If not provided then RayleighSampleGenerator
        will be used
    channel_profile : TdlChannelProfile
        The channel profile, which specifies the tap powers and delays.
    tap_powers_dB : np.ndarray
        The powers of each tap (in dB). Dimension: `L x 1`
        Note: The power of each tap will be a negative number (in dB).
    tap_delays : np.ndarray
        The delay of each tap (in seconds). Dimension: `L x 1`

    Returns
    -------
    MuChannel
        The created object.
    """
    def __init__(self,
                 N: Union[int, Tuple[int, int]],
                 fading_generator: Optional[FadingGenerator] = None,
                 channel_profile: Optional[TdlChannelProfile] = None,
                 tap_powers_dB: Optional[np.ndarray] = None,
                 tap_delays: Optional[np.ndarray] = None,
                 Ts: Optional[float] = None) -> None:

        if fading_generator is None:
            fading_generator = RayleighSampleGenerator()

        if isinstance(N, tuple):
            num_rx, num_tx = N
        else:
            num_rx = N
            num_tx = N

        # Variable to store the single user channels corresponding to each
        # link.
        self._su_siso_channels: np.ndarray = np.empty((num_rx, num_tx),
                                                      dtype=object)

        # Create each link's channel
        for rx in range(num_rx):
            for tx in range(num_tx):
                # Create a new fading generator for this link similar to
                # the one provided
                new_fading_generator = \
                    fading_generator.get_similar_fading_generator()

                # Create a new SuChannel object for this link
                self._su_siso_channels[rx, tx] = singleuser.SuChannel(
                    new_fading_generator,
                    channel_profile=channel_profile,
                    tap_powers_dB=tap_powers_dB,
                    tap_delays=tap_delays,
                    Ts=Ts)

                # Let's save the channel profile so that we use the same
                # object for the other link channels
                channel_profile = self._su_siso_channels[rx, tx].\
                    channel_profile

        self._pathloss_matrix: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        """
        String representation the object.

        Returns
        -------
        str
            The string representation of the object.
        """
        return "{0}(shape={1}, switched={2})".format(
            self.__class__.__name__,
            "{0}x{1}".format(*self._su_siso_channels.shape),
            self.switched_direction)

    @property
    def switched_direction(self) -> bool:
        """
        Get the value of `switched_direction`.

        Returns
        -------
        bool
            True if direction is switched and False otherwise.
        """
        return cast(bool, self._su_siso_channels[0, 0].switched_direction)

    @switched_direction.setter
    def switched_direction(self, value: bool) -> None:
        """
        Set the value of `switched_direction`.

        Parameters
        ----------
        value : bool
            True to switch directions of false to use original direction.
        """
        num_rx, num_tx = self._su_siso_channels.shape
        for rx_idx in range(num_rx):
            for tx_idx in range(num_tx):
                self._su_siso_channels[rx_idx, tx_idx].\
                    switched_direction = value

    @property
    def num_tx_antennas(self) -> int:
        """
        Get the number of transmit antennas.

        Returns
        -------
        np.ndarray
            The number of transmit antennas.
        """
        _, num_tx = self._su_siso_channels.shape
        num_tx_antennas = np.empty(num_tx, dtype=int)
        for tx_idx in range(num_tx):
            num_tx_antennas[tx_idx] = \
                self._su_siso_channels[0, tx_idx].num_tx_antennas
        return cast(int, num_tx_antennas)

    @property
    def num_rx_antennas(self) -> int:
        """
        Get the number of receive antennas.

        Returns
        -------
        int
            The number of receive antennas.
        """
        num_rx, _ = self._su_siso_channels.shape
        num_rx_antennas = np.empty(num_rx, dtype=int)
        for rx_idx in range(num_rx):
            num_rx_antennas[rx_idx] = \
                self._su_siso_channels[rx_idx, 0].num_rx_antennas
        return cast(int, num_rx_antennas)

    @property
    def channel_profile(self) -> TdlChannelProfile:
        """
        Return the channel profile.

        Returns
        -------
        TdlChannelProfile
            The channel profile.
        """
        return cast(TdlChannelProfile,
                    self._su_siso_channels[0, 0].channel_profile)

    @property
    def num_taps(self) -> int:
        """
        Get the number of taps in the profile.

        Note that all links have the same channel profile.

        Returns
        -------
        int
            The number of taps in the channel (not including any zero
            padding).
        """
        return cast(int, self._su_siso_channels[0, 0].num_taps)

    @property
    def num_taps_with_padding(self) -> int:
        """
        Get the number of taps in the profile including zero-padding
        when the profile is discretized.

        If the profile is not discretized an exception is raised.

        Note that all links have the same channel profile.

        Returns
        -------
        int
            The number of taps in the channel (including any zero padding).
        """
        return cast(int, self._su_siso_channels[0, 0].num_taps_with_padding)

    @property
    def pathloss_matrix(self) -> np.ndarray:
        """
        Get the matrix with the pathloss from each transmitter to each
        receiver.

        Returns
        -------
        np.ndarray
            The pathloss matrix, if it was set, or None if there is no
            pathloss.
        """
        return self._pathloss_matrix

    def set_pathloss(self, pathloss_matrix: np.ndarray) -> None:
        """
        Set the path loss (IN LINEAR SCALE) from each transmitter to each
        receiver.

        The path loss will be accounted when calling the corrupt_data
        method.

        If you want to disable the path loss, set `pathloss_matrix` to
        None.

        Parameters
        ----------
        pathloss_matrix : np.ndarray
            A matrix with dimension "K x K", where K is the number of
            users, with the path loss (IN LINEAR SCALE) from each
            transmitter (columns) to each receiver (rows). If you want to
            disable the path loss then set it to None.

        Notes
        -----
        Note that path loss is a power relation, which means that the
        channel coefficients will be multiplied by the square root of
        elements in `pathloss_matrix`.
        """
        num_rx, num_tx = self._su_siso_channels.shape

        # Set in an attribute for easy retriaval later
        self._pathloss_matrix = np.copy(pathloss_matrix)

        for rx in range(num_rx):
            for tx in range(num_tx):
                self._su_siso_channels[rx,
                                       tx].set_pathloss(pathloss_matrix[rx,
                                                                        tx])

    def corrupt_data(self, signal: np.ndarray) -> np.ndarray:
        """
        Corrupt data passed through the TDL channels of each link.

        Note that noise is NOT added in `corrupt_data`.

        Parameters
        ----------
        signal : np.ndarray
            Signal to be transmitted through the channel. This should be
            a 2D numpy array (1D array if there is only one
            transmitter), where each row corresponds to the transmit
            data of one transmitter.

        Returns
        -------
        np.ndarray
            Received signal at each receiver. Each row corresponds to one
            receiver.
        """
        if self.switched_direction:
            su_siso_channels = self._su_siso_channels.T
        else:
            su_siso_channels = self._su_siso_channels

        num_rx, num_tx = su_siso_channels.shape
        outputs = np.empty(num_rx, dtype=object)

        if num_tx == 1 and signal.ndim == 1:
            signal = np.reshape(signal, (1, -1))

        for rx in range(num_rx):
            suchannel = su_siso_channels[rx, 0]
            outputs[rx] = suchannel.corrupt_data(signal[0])
            for tx in range(1, num_tx):
                suchannel = su_siso_channels[rx, tx]
                outputs[rx] += suchannel.corrupt_data(signal[tx])

        return outputs

    def corrupt_data_in_freq_domain(
            self,
            signal: np.ndarray,
            fft_size: int,
            carrier_indexes: Indexes = None) -> np.ndarray:
        """
        Corrupt data passed through the TDL channels of each link,
        but in the frequency domain..

        For each link, this is ROUGHLY equivalent to modulating `signal`
        with OFDM using `fft_size` subcarriers, transmitting through a
        regular TdlChannel, and then demodulating with OFDM to recover the
        received signal.

        One important difference is that here the channel is considered
        constant during the transmission of `fft_size` elements in
        `signal`, and then it is varied by the equivalent of the variation
        for that number of elements. That is, the channel is block static.

        Note that noise is NOT added in `corrupt_data`.

        Parameters
        ----------
        signal : np.ndarray | list[np.ndarray]
            Signal to be transmitted through the channel. This should be a 2D
            numpy array where each row corresponds to the transmit data of
            one transmitter. It can also be a list of numpy arrays or,
            if there is only one transmitter, a single 1D numpy array.
        fft_size : int
            The size of the Fourier transform to get the frequency
            response.
        carrier_indexes : slice | np.ndarray | list[int]
            The indexes of the subcarriers where signal is to be
            transmitted (all users will use the same indexes). If it is
            None assume all subcarriers will be used.

        Returns
        -------
        np.ndarray
            Received signal at each receiver. Each row corresponds to one
            receiver.
        """
        if self.switched_direction:
            su_siso_channels = self._su_siso_channels.T
        else:
            su_siso_channels = self._su_siso_channels

        num_rx, num_tx = su_siso_channels.shape
        outputs = np.empty(num_rx, dtype=object)

        if num_tx == 1 and signal.ndim == 1:
            signal = np.reshape(signal, (1, -1))

        for rx in range(num_rx):
            suchannel = su_siso_channels[rx, 0]
            outputs[rx] = suchannel.corrupt_data_in_freq_domain(
                signal[0], fft_size, carrier_indexes)
            for tx in range(1, num_tx):
                suchannel = su_siso_channels[rx, tx]
                outputs[rx] += suchannel.corrupt_data_in_freq_domain(
                    signal[tx], fft_size, carrier_indexes)

        return outputs

    def get_last_impulse_response(self, rx_idx: int,
                                  tx_idx: int) -> TdlImpulseResponse:
        """
        Get the last generated impulse response.

        A new impulse response is generated when the method `corrupt_data`
        is called. You can use the `get_last_impulse_response` method to
        get the impulse response used to corrupt the last data.

        Parameters
        ----------
        rx_idx : int
            The index of the receiver.
        tx_idx : int
            The index of the transmitter

        Returns
        -------
        TdlImpulseResponse
            The impulse response of the channel that was used to corrupt
            the last data for the link from transmitter `tx_idx` to
            receiver `rx_idx`.
        """
        return cast(TdlImpulseResponse, self._su_siso_channels[rx_idx, tx_idx].\
            get_last_impulse_response())


# class MuSisoFlatFadingChannel:
#     """
#     SISO multiuser flat-fading channel.

#     This corresponds to an interference channel model, where each
#     transmitter sends data to its own receiver while interfering to other
#     receivers.

#     Note that noise is NOT added.

#     Parameters
#     ----------
#     N : int
#         The number of transmit/receive pairs.
#     fading_generator : Object of some class derived from
#         FadingSampleGenerator
#         The fading generator. If not provided, then an object of the
#         RayleighSampleGenerator class will be created.
#     """
#     def __init__(self, N, fading_generator=None):
#         self._H = None
#         if fading_generator is None:
#             fading_generator = RayleighSampleGenerator(shape=(N, N))

#         self._fading_generator = fading_generator
#         self._pathloss_matrix = None

#     def _get_channel_samples(self):
#         """
#         Get the fading generated channel samples while also applying the
#         path loss, if it was set with the `set_pathloss` method

#         Returns
#         -------
#         numpy array
#             The channel samples also including any path loss effect.
#         """
#         if self._pathloss_matrix is None:
#             samples = self._fading_generator.get_samples()
#         else:
#             samples = (self._fading_generator.get_samples() *
#                        self._pathloss_matrix)
#         return samples

#   # TODO: update fading samples after this method is called. After you do
#     # this, add a test for it in MuSisoChannelTestCase.test_corrupt_data
#     def corrupt_data(self, data):
#         """
#         Corrupt data passed through the channel.

#         Note that noise is NOT added in `corrupt_data`.

#         Parameters
#         ----------
#         data : 1D numpy array or 2D numpy array
#             If `data` is a 1D numpy array, the k-th element corresponds
#             to the symbol transmitted to the k-th user. If `data` is a 2D
#             numpy array then the k-th row corresponds to the symbols
#             transmitted to the k-th user.

#         Returns
#         -------
#         1D numpy array of 2D numpy arrays
#             A numpy array where each element (or row) contains the
#             received data of a user.
#         """
#         return np.dot(self._get_channel_samples(), data)

#     def set_pathloss(self, pathloss_matrix=None):
#         """
#         Set the path loss (IN LINEAR SCALE) from each transmitter to each
#         receiver.

#         The path loss will be accounted when calling the corrupt_data
#         method.

#         If you want to disable the path loss, set `pathloss_matrix` to
#         None.

#         Parameters
#         ----------
#         pathloss_matrix : 2D numpy array
#             A matrix with dimension "K x K", where K is the number of
#             users, with the path loss (IN LINEAR SCALE) from each
#             transmitter (columns) to each receiver (rows). If you want to
#             disable the path loss then set it to None.

#         Notes
#         -----
#         Note that path loss is a power relation, which means that the
#         channel coefficients will be multiplied by the square root of
#         elements in `pathloss_matrix`.
#         """
#         # A matrix with the path loss from each transmitter to each
#         # receiver.
#         self._pathloss_matrix = pathloss_matrix


class MuMimoChannel(MuChannel):
    """
    MIMO multiuser channel.

    Each transmitter sends data to its own receiver while interfering to
    other receivers.

    Note that noise is NOT added.

    Parameters
    ----------
    N : int | tuple[int, int]
        The number of transmit/receive pairs.
    num_rx_antennas : int
        Number of receive antennas of each user.
    num_tx_antennas : int
        Number of transmit antennas of each user.
    fading_generator : T <= fading_generators.FadingSampleGenerator
        The instance of a fading generator in the `fading_generators`
        module. It should be a subclass of FadingSampleGenerator. The
        fading generator will be used to generate the channel
        samples. However, since we have multiple links, the provided fading
        generator will actually be used to create similar (but independent)
        fading generators. If not provided then RayleighSampleGenerator
        will be used
    channel_profile : TdlChannelProfile
        The channel profile, which specifies the tap powers and delays.
    tap_powers_dB : np.ndarray
        The powers of each tap (in dB). Dimension: `L x 1`
        Note: The power of each tap will be a negative number (in dB).
    tap_delays : np.ndarray
        The delay of each tap (in seconds). Dimension: `L x 1`
    """
    def __init__(self,
                 N: Union[int, Tuple[int, int]],
                 num_rx_antennas: int,
                 num_tx_antennas: int,
                 fading_generator: Optional[FadingGenerator] = None,
                 channel_profile: Optional[TdlChannelProfile] = None,
                 tap_powers_dB: Optional[np.ndarray] = None,
                 tap_delays: Optional[np.ndarray] = None,
                 Ts: Optional[float] = None) -> None:
        super(MuMimoChannel,
              self).__init__(N, fading_generator, channel_profile,
                             tap_powers_dB, tap_delays, Ts)

        # Number of receivers and transmitters
        if isinstance(N, (tuple, list)):
            num_rx, num_tx = N
        else:
            num_rx = N
            num_tx = N

        # Create each link's channel
        for rx in range(num_rx):
            for tx in range(num_tx):
                self._su_siso_channels[rx, tx].set_num_antennas(
                    num_rx_antennas, num_tx_antennas)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Old Classes for backward compatibility xxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TODO: Maybe remove the N0_or_Rek argument of the "*calc_Bkl*" methods and
# use the value of self.noise_var whenever possible.
class MultiUserChannelMatrix:  # pylint: disable=R0902
    """
    Stores the (fast fading) channel matrix of a multi-user scenario.
    The path-loss from each transmitter to each receiver is also be
    accounted if the set_pathloss is called to set the path-loss matrix.

    This channel matrix can be seem as an concatenation of blocks (of
    non-uniform size) where each block is a channel from one transmitter to
    one receiver and the block size is equal to the number of receive
    antennas of the receiver times the number of transmit antennas of the
    transmitter.

    For instance, in a 3-users scenario the block (1,0) corresponds to the
    channel between the transmit antennas of user 0 and the receive
    antennas of user 1 (indexing staring at zero). If the number of receive
    antennas and transmit antennas of the three users are [2, 4, 6] and
    [2, 3, 5], respectively, then the block (1,0) would have a dimension of
    4x2. Likewise, the channel matrix would look similar to the block
    structure below.

      +-----+---------+---------------+
      |2 x 2|  2 x 3  |     2 x 5     |
      |     |         |               |
      +-----+---------+---------------+
      |4 x 2|  4 x 3  |     4 x 5     |
      |     |         |               |
      |     |         |               |
      |     |         |               |
      +-----+---------+---------------+
      |6 x 2|  6 x 3  |     6 x 5     |
      |     |         |               |
      |     |         |               |
      |     |         |               |
      |     |         |               |
      |     |         |               |
      +-----+---------+---------------+

    It is possible to initialize the channel matrix randomly by calling the
    `randomize` method, or from a given matrix by calling the
    `init_from_channel_matrix` method.

    In order to get the channel matrix of a specific user `k` to another
    user `l`, call the `get_Hkl` method.
    """
    def __init__(self) -> None:
        # The _big_H_no_pathloss variable is an internal variable with all
        # the channels from each transmitter to each receiver represented
        # as a single big matrix.
        self._big_H_no_pathloss = np.array([], dtype=np.ndarray)

        # The _H_no_pathloss variable is an internal variable with all the
        # channels from each transmitter to each receiver. It points to the
        # same data as the _big_H_no_pathloss variable, however, _H is a
        # "matrix of matrices" instead of a single big matrix.
        self._H_no_pathloss = np.array([], dtype=np.ndarray)

        # The _big_H_with_pathloss and _H_with_pathloss variables are
        # similar to their no_pathloss counterpart, but include the effect
        # of pathloss if it was set.
        self._big_H_with_pathloss: Optional[np.ndarray] = None
        self._H_with_pathloss: Optional[np.ndarray] = None

        self._Nr = np.array([])
        self._Nt = np.array([])
        self._K: int = 0
        self._pathloss_matrix: Optional[np.ndarray] = None
        # _pathloss_big_matrix should not be set directly. It is set when
        # _pathloss_matrix is set in the set_pathloss method.
        self._pathloss_big_matrix: Optional[np.ndarray] = None
        self._RS_channel = np.random.RandomState()
        self._RS_noise = np.random.RandomState()

        # Store the AWGN noise array from the last time any of the
        # corrupt*_data methods were called.
        self._last_noise: Optional[float] = None
        # Store the noise variance. If it is None, then no noise is added
        # in the "corrupt_*data" methods.
        self._noise_var: Optional[float] = None

        # Post processing filters (a list of 2D numpy arrays) for each user
        self._W: Optional[List[np.ndarray]] = None
        # Same as _W, but as a single block diagonal matrix.
        self._big_W: np.ndarray = None

    def set_channel_seed(self,
                         seed: Optional[Seed] = None
                         ) -> None:  # pragma: no cover
        """
        Set the seed of the RandomState object used to generate the random
        elements of the channel (when self.randomize is called).

        Parameters
        ----------
        seed : None | int | array_like
            Random seed initializing the pseudo-random number
            generator. See np.random.RandomState help for more info.
        """
        self._RS_channel.seed(seed=seed)

    def set_noise_seed(self,
                       seed: Optional[Seed] = None
                       ) -> None:  # pragma: no cover
        """
        Set the seed of the RandomState object used to generate the random
        noise elements (when the corrupt data function is called).

        Parameters
        ----------
        seed : None | int | array_like
            Random seed initializing the pseudo-random number
            generator. See np.random.RandomState help for more info.
        """
        self._RS_noise.seed(seed)

    def re_seed(self) -> None:  # pragma: no cover
        """
        Re-seed the channel and noise RandomState objects randomly.

        If you want to specify the seed for each of them call the
        `set_channel_seed` and `set_noise_seed` methods and pass the
        desired seed for each of them.
        """
        self.set_channel_seed(None)
        self.set_noise_seed(None)

    # Property to get the number of receive antennas
    @property
    def Nr(self) -> np.ndarray:
        """
        Get method for the Nr property.

        Returns
        -------
        np.ndarray
            The number of receive antennas of all users.
        """
        return self._Nr

    # Property to get the number of transmit antennas
    @property
    def Nt(self) -> np.ndarray:
        """
        Get method for the Nt property.

        Returns
        -------
        np.ndarray
            The number of transmit antennas of all users.
        """
        return self._Nt

    # Property to get the number of users
    @property
    def K(self) -> int:
        """
        Get method for the K property.

        Returns
        -------
        int
            The number of users (transmit-receive pairs).
        """
        return self._K

    # Property to get the matrix of channel matrices (with pass loss
    # applied if any)
    @property
    def H(self) -> np.ndarray:
        """
        Get method for the H property.

        Returns
        -------
        np.ndarray
            The channel from all transmitters to all receivers. This is a
            numpy array of numpy arrays.
        """
        if self._pathloss_matrix is None:
            # No path loss
            return self._H_no_pathloss

        if self._H_with_pathloss is None:
            # Apply path loss. Note that the _pathloss_big_matrix
            # matrix has the same dimension as the
            # self._big_H_no_pathloss matrix and we are performing
            # element-wise multiplication here.
            # noinspection PyTypeChecker
            self._H_with_pathloss = self._H_no_pathloss * np.sqrt(
                self._pathloss_matrix)
        return self._H_with_pathloss

    # Property to get the big channel matrix (with pass loss applied if
    # any)
    @property
    def big_H(self) -> np.ndarray:
        """
        Get method for the big_H property.

        Returns
        -------
        np.ndarray
            The channel from all transmitters to all receivers as a single
            big matrix (numpy complex array)
        """
        if self._pathloss_matrix is None:
            # No path loss
            return self._big_H_no_pathloss

        if self._big_H_with_pathloss is None:
            # Apply path loss. Note that the _pathloss_big_matrix
            # matrix has the same dimension as the
            # self._big_H_no_pathloss matrix and we are performing
            # element-wise multiplication here.
            # noinspection PyTypeChecker
            self._big_H_with_pathloss = (self._big_H_no_pathloss *
                                         np.sqrt(self._pathloss_big_matrix))
        return self._big_H_with_pathloss

    # Property to get the pathloss. Use the "set_pathloss" method to set
    # the pathloss.
    @property
    def pathloss(self) -> Optional[np.ndarray]:
        """
        Get method for the pathloss property.

        Returns
        -------
        None | np.ndarray
            The pathloss matrix (if one was set).
        """
        return self._pathloss_matrix

    @property
    def last_noise(self) -> Optional[np.ndarray]:
        """
        Get method for the last_noise property.

        Returns
        -------
        None | np.ndarray
            The last AWGN noise array added to corrupt the data.
        """
        return self._last_noise

    @property
    def noise_var(self) -> Optional[float]:
        """
        Get method for the noise_var property.

        Returns
        -------
        None | float
            The noise variance, if noise is being added in "corrupt_*data"
            methods.
        """
        return self._noise_var

    @noise_var.setter
    def noise_var(self, value: Optional[float]) -> None:
        """
        Set method for the noise_var property.

        Parameters
        ----------
        value: float | None
            The noise variance used when generating a new noise vector to add
            in the "corrupt_*data" methods. If `value` is None then noise
            addition is disabled.
        """
        if value is not None:
            assert value >= 0.0, "Noise variance must be >= 0."
        self._noise_var = value

    @staticmethod
    def _from_small_matrix_to_big_matrix(
            small_matrix: np.ndarray,
            Nr: np.ndarray,
            Nt: np.ndarray,
            Kr: int,
            Kt: Optional[int] = None) -> np.ndarray:
        """
        Convert from a small matrix to a big matrix by repeating elements
        according to the number of receive and transmit antennas.

        Parameters
        ----------
        small_matrix : np.ndarray
            Any 2D numpy array
        Nr : np.ndarray
            Number of antennas at each receiver. This should be a 1D numpy
            array.
        Nt : np.ndarray
            Number of antennas at each transmitter. This should be a 1D numpy
            array.
        Kr : int
            Number of receivers to consider.
        Kt : int, optional
            Number of transmitters to consider. It not provided the value of
            Kr will be used.

        Returns
        -------
        big_matrix : np.ndarray
            The converted matrix. This is a 2D numpy array

        Notes
        -----
        Since a 'user' is a transmit/receive pair then the small_matrix
        will be a square matrix and `Kr` must be equal to `Kt`.  However, in
        the :class:`MultiUserChannelMatrixExtInt` class we will only have the
        'transmitter part' for the external interference sources. That
        means that small_matrix will have more columns then rows and `Kt`
        will be greater then `Kr`.

        Examples
        --------
        >>> K = 3
        >>> Nr = np.array([2, 4, 6])
        >>> Nt = np.array([2, 3, 5])
        >>> small_matrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> MultiUserChannelMatrix._from_small_matrix_to_big_matrix(\
                small_matrix, Nr, Nt, K)
        array([[1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
               [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
               [4, 4, 5, 5, 5, 6, 6, 6, 6, 6],
               [4, 4, 5, 5, 5, 6, 6, 6, 6, 6],
               [4, 4, 5, 5, 5, 6, 6, 6, 6, 6],
               [4, 4, 5, 5, 5, 6, 6, 6, 6, 6],
               [7, 7, 8, 8, 8, 9, 9, 9, 9, 9],
               [7, 7, 8, 8, 8, 9, 9, 9, 9, 9],
               [7, 7, 8, 8, 8, 9, 9, 9, 9, 9],
               [7, 7, 8, 8, 8, 9, 9, 9, 9, 9],
               [7, 7, 8, 8, 8, 9, 9, 9, 9, 9],
               [7, 7, 8, 8, 8, 9, 9, 9, 9, 9]])
        """
        if Kt is None:
            Kt = Kr

        cumNr = np.hstack([0, np.cumsum(Nr)])
        cumNt = np.hstack([0, np.cumsum(Nt)])
        big_matrix = np.ones(
            [int(np.sum(Nr)), int(np.sum(Nt))], dtype=small_matrix.dtype)

        for rx in range(Kr):
            for tx in range(Kt):
                big_matrix[cumNr[rx]:cumNr[rx + 1], cumNt[tx]:cumNt[tx + 1]] \
                    *= small_matrix[rx, tx]
        return big_matrix

    def init_from_channel_matrix(self, channel_matrix: np.ndarray,
                                 Nr: IntOrIntArrayUnion,
                                 Nt: IntOrIntArrayUnion, K: int) -> None:
        """
        Initializes the multiuser channel matrix from the given
        `channel_matrix`.

        Parameters
        ----------
        channel_matrix : np.ndarray
            A matrix concatenating the channel of all users (from each
            transmitter to each receiver). This is a 2D numpy array.
        Nr : int | np.ndarray
            Number of antennas at each receiver.
        Nt : int | np.ndarray
            Number of antennas at each transmitter.
        K : int
            Number of transmit/receive pairs.

        Raises
        ------
        ValueError
            If the arguments are invalid.

        """
        # If Nt or Nr (or both) is (are) int assume the same value should
        # be used for all users.
        Nr_array = np.ones(K, dtype=int) * Nr if isinstance(Nr, int) else Nr
        Nt_array = np.ones(K, dtype=int) * Nt if isinstance(Nt, int) else Nt
        del Nt, Nr
        # if isinstance(Nr, int):  # pragma: no cover
        #     Nr = np.ones(K, dtype=int) * Nr
        # if isinstance(Nt, int):  # pragma: no cover
        #     Nt = np.ones(K, dtype=int) * Nt

        if channel_matrix.shape != (np.sum(Nr_array), np.sum(Nt_array)):
            msg = ("Shape of the channel_matrix must be equal to the sum or"
                   " receive antennas of all users times the sum of the "
                   "receive antennas of all users.")
            raise ValueError(msg)

        if (Nt_array.size != K) or (Nr_array.size != K):
            raise ValueError(
                "K must be equal to the number of elements in Nr and Nt")

        # Reset the _big_H_with_pathloss and _H_with_pathloss. They will be
        # correctly set the first time the _get_H or _get_big_H methods are
        # called.
        self._big_H_with_pathloss = None
        self._H_with_pathloss = None

        self._K = K
        self._Nr = Nr_array
        self._Nt = Nt_array

        self._big_H_no_pathloss = channel_matrix

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Lets convert the full channel_matrix matrix to our internal
        # representation of H as a matrix of matrices.
        self._H_no_pathloss = single_matrix_to_matrix_of_matrices(
            channel_matrix, Nr_array, Nt_array)

        # Assures that _big_H and _H will stay in sync by disallowing
        # modification of individual elements in both of them.
        self._big_H_no_pathloss.setflags(write=False)
        self._H_no_pathloss.setflags(write=False)

    def randomize(self, Nr: IntOrIntArrayUnion, Nt: IntOrIntArrayUnion,
                  K: int) -> None:
        """
        Generates a random channel matrix for all users.

        Parameters
        ----------
        Nr : int | np.ndarray
            Number of receive antennas of each user. If an integer is
            specified, all users will have that number of receive antennas.
        Nt : int | np.ndarray
            Number of transmit antennas of each user. If an integer is
            specified, all users will have that number of receive antennas.
        K : int
            Number of users.
        """
        # Reset the _big_H_with_pathloss and _H_with_pathloss. They will be
        # correctly set the first time the _get_H or _get_big_H methods are
        # called.
        self._big_H_with_pathloss = None
        self._H_with_pathloss = None

        if isinstance(Nr, int):
            Nr = np.ones(K, dtype=int) * Nr
        if isinstance(Nt, int):
            Nt = np.ones(K, dtype=int) * Nt

        self._Nr = Nr.astype(int)
        self._Nt = Nt.astype(int)
        self._K = int(K)

        self._big_H_no_pathloss = randn_c_RS(self._RS_channel,
                                             np.sum(self._Nr),
                                             np.sum(self._Nt))

        self._H_no_pathloss = single_matrix_to_matrix_of_matrices(
            self._big_H_no_pathloss, Nr, Nt)

        # Assures that _big_H and _H will stay in sync by disallowing
        # modification of individual elements in both of them.
        self._big_H_no_pathloss.setflags(write=False)
        self._H_no_pathloss.setflags(write=False)

    def get_Hkl(self, k: int, l: int) -> np.ndarray:
        """
        Get the channel matrix from user `l` to user `k`.

        Parameters
        ----------
        l : int
            Transmitting user.
        k : int
            Receiving user.

        Returns
        -------
        channel : np.ndarray
            Channel from transmitter `l` to receiver `k`. This is a 2D numpy
            array.

        See also
        --------
        get_Hk

        Examples
        --------
        >>> multiH = MultiUserChannelMatrix()
        >>> H = np.reshape(np.r_[0:16], [4,4])
        >>> Nt = np.array([2, 2])
        >>> Nr = np.array([2, 2])
        >>> multiH.init_from_channel_matrix(H, Nr, Nt, 2)
        >>> print(multiH.big_H)
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]
         [12 13 14 15]]
        >>> print(multiH.get_Hkl(0, 0))
        [[0 1]
         [4 5]]
        >>> print(multiH.get_Hkl(1, 0))
        [[ 8  9]
         [12 13]]
        """
        # This will call the _get_H method, which already applies the path
        # loss (if there is any)
        channel = self.H
        return channel[k, l]

    def get_Hk(self, k: int) -> np.ndarray:
        """
        Get the channel from all transmitters to receiver `k`.

        Parameters
        ----------
        k : int
            Receiving user.

        Returns
        -------
        channel_k : np.ndarray
            Channel from all transmitters to receiver `k`. This is a 2D numpy
            array.

        See also
        --------
        get_Hkl

        Examples
        --------
        >>> multiH = MultiUserChannelMatrix()
        >>> H = np.reshape(np.r_[0:16], [4,4])
        >>> Nt = np.array([2, 2])
        >>> Nr = np.array([2, 2])
        >>> multiH.init_from_channel_matrix(H, Nr, Nt, 2)
        >>> print(multiH.big_H)
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]
         [12 13 14 15]]
        >>> print(multiH.get_Hk(0))
        [[0 1 2 3]
         [4 5 6 7]]
        >>> print(multiH.get_Hk(1))
        [[ 8  9 10 11]
         [12 13 14 15]]
        """
        receive_channels = single_matrix_to_matrix_of_matrices(
            self.big_H, self.Nr)
        return receive_channels[k]

    def set_post_filter(self, filters: np.ndarray) -> None:
        """
        Set the post-processing filters.

        The post-processing filters will be applied to the data after if
        has been corrupted by the channel in either the `corrupt_data` or
        the `corrupt_concatenated_data` methods.

        Parameters
        ----------
        filters : list[np.ndarray] | np.ndarray
            The post processing filters of each user. This should be a list
            of 2D np arrays or a 1D np array of 2D np arrays.
        """
        self._W = filters
        # This will be set in the get property only when required.
        self._big_W = None

    @property
    def W(self) -> Optional[List[np.ndarray]]:
        """
        Post processing filters (a list of 2D numpy arrays) for each user.

        Returns
        -------
        list[np.ndarray]
            The Post processing filters for each user.
        """
        return self._W

    @property
    def big_W(self) -> np.ndarray:
        """
        Post processing filters (a block diagonal matrix) for each user.

        Returns
        -------
        np.ndarray
            The big block diagonal matrix with the post processing filters
            for each user.
        """
        if self._big_W is None and self.W is not None:
            # noinspection PyArgumentList
            self._big_W = block_diag(*self.W)
        return self._big_W

    def corrupt_concatenated_data(self, data: np.ndarray) -> np.ndarray:
        """
        Corrupt data passed through the channel.

        If self.noise_var is set to some scalar number then white noise
        will also be added.

        Parameters
        ----------
        data : np.ndarray
            A bi-dimensional numpy array with the concatenated data of all
            transmitters. The dimension of data is sum(self.Nt) x
            NSymb. That is, the number of rows corresponds to the sum of
            the number of transmit antennas of all users and the number of
            columns correspond to the number of transmitted symbols.

        Returns
        -------
        np.ndarray
            A bi-dimension numpy array where the number of rows corresponds
            to the sum of the number of receive antennas of all users and
            the number of columns correspond to the number of transmitted
            symbols.
        """
        # Note that self.big_H already accounts the path loss (while
        # self._big_H_no_pathloss does not)

        output = np.dot(self.big_H, data)

        # Add the noise, if self.noise_var is not None
        if self.noise_var is not None:
            awgn_noise = (randn_c_RS(self._RS_noise, *output.shape) *
                          math.sqrt(self.noise_var))
            output += awgn_noise
            self._last_noise = awgn_noise
        else:
            self._last_noise = None

        # Apply the post processing filter (if there is one set)
        if self.big_W is not None:
            output = np.dot(self.big_W.conjugate().T, output)

        return output

    def corrupt_data(self, data: np.ndarray) -> np.ndarray:
        """
        Corrupt data passed through the channel.

        If the noise_var is supplied then an white noise will also be
        added.

        Parameters
        ----------
        data : np.ndarray
            An array of numpy matrices with the data of the multiple
            users. The k-th element in `data` is a numpy array with
            dimension Nt_k x NSymbs, where Nt_k is the number of transmit
            antennas of the k-th user and NSymbs is the number of
            transmitted symbols.

        Returns
        -------
        np.ndarray
            A numpy array where each element contains the received data (a
            2D numpy array) of a user.

        """
        # Note that we intentionally use self.K instead of self._K in this
        # method. In the MultiUserChannelMatrix class they have the same
        # value. However, in the MultiUserChannelMatrixExtInt subclass the
        # self._K attribute will correspond to the number of users plus the
        # number of external interference sources, while the self.K
        # property will return only the number of users, which is what we
        # want here.
        concatenated_data = np.vstack(data)
        concatenated_output = self.corrupt_concatenated_data(concatenated_data)

        output = np.zeros(self.K, dtype=np.ndarray)
        cumNr = np.hstack([0, np.cumsum(self._Nr)])

        for k in np.arange(self.K):
            output[k] = concatenated_output[cumNr[k]:cumNr[k + 1], :]

        return output

    def set_pathloss(self,
                     pathloss_matrix: Optional[np.ndarray] = None) -> None:
        """
        Set the path loss (IN LINEAR SCALE) from each transmitter to each
        receiver.

        The path loss will be accounted when calling the get_Hkl, get_Hk, the
        corrupt_concatenated_data and the corrupt_data methods.

        If you want to disable the path loss, set `pathloss_matrix` to None.

        Parameters
        ----------
        pathloss_matrix : np.ndarray
            A matrix with dimension "K x K", where K is the number of
            users, with the path loss (IN LINEAR SCALE) from each
            transmitter (columns) to each receiver (rows). If you want to
            disable the path loss then set it to None.

        Notes
        -----
        Note that path loss is a power relation, which means that the
        channel coefficients will be multiplied by the square root of
        elements in `pathloss_matrix`.
        """
        # A matrix with the path loss from each transmitter to each
        # receiver.
        self._pathloss_matrix = pathloss_matrix

        # Reset the _big_H_with_pathloss and _H_with_pathloss. They will be
        # correctly set the first time the _get_H or _get_big_H methods are
        # called.
        self._big_H_with_pathloss = None
        self._H_with_pathloss = None

        if pathloss_matrix is None:
            self._pathloss_big_matrix = None
        else:
            assert (self._pathloss_matrix is not None)
            self._pathloss_big_matrix \
                = MultiUserChannelMatrix._from_small_matrix_to_big_matrix(
                    pathloss_matrix, self._Nr, self._Nt, self._K)

            # Assures that _pathloss_matrix and _pathloss_big_matrix will stay
            # in sync by disallowing modification of individual elements in
            # both of them.
            self._pathloss_matrix.setflags(write=False)
            self._pathloss_big_matrix.setflags(write=False)

    # noinspection PyPep8
    def _calc_Q_impl(self, k: int, F_all_users: np.ndarray) -> np.ndarray:
        """
        Calculates the interference covariance matrix (without any noise) at
        the :math:`k`-th receiver.

        See the documentation of the calc_Q method.

        Parameters
        ----------
        k : int
            Index of the desired receiver.
        F_all_users : np.ndarray
            The precoder of all users (already taking into account the
            transmit power). This should be a 1D numpy array of 2D numpy
            arrays.

        Returns
        -------
        np.ndarray
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{kj} \mtF_j \mtF_j^H \mtH_{kj}^H$$
        interfering_users = set(range(self.K)) - {k}
        Qk = np.zeros([self.Nr[k], self.Nr[k]], dtype=complex)

        for l in interfering_users:
            Hkl_F = np.dot(self.get_Hkl(k, l), F_all_users[l])
            Qk = Qk + np.dot(Hkl_F, Hkl_F.transpose().conjugate())

        return Qk

    # noinspection PyPep8
    def calc_Q(self, k: int, F_all_users: np.ndarray) -> np.ndarray:
        """
        Calculates the interference plus noise covariance matrix at the
        :math:`k`-th receiver.

        The interference covariance matrix at the :math:`k`-th receiver,
        :math:`\\mtQ k`, is given by

            :math:`\\mtQ k = \\sum_{j=1, j \\neq k}^{K} \\frac{P_j}{Ns_j} \\mtH_{kj} \\mtF_j \\mtF_j^H \\mtH_{kj}^H`

        where :math:`P_j` is the transmit power of transmitter :math:`j`,
        and :math:`Ns_j` is the number of streams for user :math:`j`.

        Parameters
        ----------
        k : int
            Index of the desired receiver.
        F_all_users : np.ndarray
            The precoder of all users (already taking into account the
            transmit power). This should be a 1D numpy array of 2D numpy
            arrays.

        Returns
        -------
        Qk : np.ndarray
            The interference covariance matrix at receiver :math:`k` (a 2D
            numpy complex array).
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{kj} \mtF_j \mtF_j^H \mtH_{kj}^H + \sigma_n^2 \mtI_{N_k}$$
        Qk = self._calc_Q_impl(k, F_all_users)

        if self.noise_var is not None:
            # If self.noise_var is not None we add the covariance matrix of
            # the noise.
            Rnk = np.eye(self.Nr[k]) * self.noise_var
            Qk += Rnk

        return Qk

    # noinspection PyPep8
    def _calc_JP_Q_impl(self, k: int, F_all_users: np.ndarray) -> np.ndarray:
        """
        Calculates the interference covariance matrix (without any noise) at
        the :math:`k`-th receiver with a joint processing scheme.

        See the documentation of the calc_JP_Q method.

        Parameters
        ----------
        k : int
            The user index.
        F_all_users : list[np.ndarray] | np.ndarray
            The precoders of all users. It can be a list of numpy arrays or a
            numpy array of numpy arrays.

        Returns
        -------
        np.ndarray
            The interference covariance matrix (without any noise).
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{k} \mtF_j \mtF_j^H \mtH_{k}^H$$
        interfering_users = set(range(self.K)) - {k}
        Qk = np.zeros([self.Nr[k], self.Nr[k]], dtype=complex)

        for l in interfering_users:
            Hk_F = np.dot(self.get_Hk(k), F_all_users[l])
            Qk = Qk + np.dot(Hk_F, Hk_F.transpose().conjugate())

        return Qk

    # noinspection PyPep8,PyPep8
    def calc_JP_Q(self, k: int, F_all_users: np.ndarray) -> np.ndarray:
        """
        Calculates the interference plus noise covariance matrix at the
        :math:`k`-th receiver with a joint processing scheme.

        The interference covariance matrix at the :math:`k`-th receiver,
        :math:`\\mtQ k`, is given by

            :math:`\\mtQ k = \\sum_{j=1, j \\neq k}^{K} \\frac{P_j}{Ns_j} \\mtH_{k} \\mtF_j \\mtF_j^H \\mtH_{k}^H`

        where :math:`P_j` is the transmit power of transmitter :math:`j`,
        and :math:`Ns_j` is the number of streams for user :math:`j`.

        Parameters
        ----------
        k : int
            Index of the desired receiver.
        F_all_users : np.ndarray | list[np.ndarray]
            The precoder of all users (already taking into account the
            transmit power).

        Returns
        -------
        Qk : np.ndarray
            The interference covariance matrix at receiver :math:`k`.
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{k} \mtF_j \mtF_j^H \mtH_{k}^H + \sigma_n^2 \mtI_{N_k}$$
        Qk = self._calc_JP_Q_impl(k, F_all_users)

        if self.noise_var is not None:
            Rnk = np.eye(self.Nr[k]) * self.noise_var
            return Qk + Rnk

        return Qk

    # noinspection PyPep8,PyPep8
    def _calc_Bkl_cov_matrix_first_part(
            self,
            F_all_users: np.ndarray,
            k: int,
            N0_or_Rek: NumberOrArray = 0.0) -> np.ndarray:
        """
        Calculates the first part in the equation of the Blk covariance matrix
        in equation (28) of [Cadambe2008]_.

        The first part is given by

            :math:`\\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star d}^{[j]} \\mtV_{\\star d}^{[j]\\dagger} \\mtH^{[kj]\\dagger} + \\mtI_{Nk}`

        Note that it only depends on the value of :math:`k`.

        Parameters
        ----------
        F_all_users : list[np.ndarray] | np.ndarray
            The precoder of all users (already taking into account the
            transmit power). It can be a list of numpy arrays or a
            numpy array of numpy arrays.
        k : int
            Index of the desired user.
        N0_or_Rek : float | np.ndarray
            If this is a 2D numpy array, it is interpreted as the
            covariance matrix of any external interference plus noise. If
            this is a number, it is interpreted as the noise power, in
            which case the covariance matrix will be an identity matrix
            times this noise power.

        Returns
        -------
        first_part : np.ndarray
        """
        # The first part in Bkl is given by
        # $$\sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger} + \mtR e_k$$
        # where $\mtR e_k$ is the covariance matrix of the (external
        # interference plus) noise.
        # Note that here the power is already included in `Fk`.
        if N0_or_Rek is None:
            N0_or_Rek = 0.0

        if isinstance(N0_or_Rek, Number):
            noise_power = N0_or_Rek
            # noinspection PyUnresolvedReferences
            Rek = (noise_power * np.eye(self.Nr[k]))
        else:
            Rek = N0_or_Rek

        first_part = 0.0
        for j in range(self.K):
            Hkj = self.get_Hkl(k, j)
            Hkj_H = Hkj.conjugate().transpose()
            Vj = F_all_users[j]
            Vj_H = Vj.conjugate().transpose()

            first_part = first_part + np.dot(Hkj,
                                             np.dot(np.dot(Vj, Vj_H), Hkj_H))
        first_part = first_part + Rek

        return first_part

    # noinspection PyPep8
    def _calc_Bkl_cov_matrix_second_part(self, Fk: np.ndarray, k: int,
                                         l: int) -> np.ndarray:
        """
        Calculates the second part in the equation of the Blk covariance
        matrix in equation (28) of [Cadambe2008]_ (note that it does not
        include the identity matrix).

        The second part is given by

            :math:`\\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger}`

        Parameters
        ----------
        Fk : np.ndarray
            The precoder of the desired user.
        k : int
            Index of the desired user.
        l : int
            Index of the desired stream.

        Returns
        -------
        second_part : np.ndarray
            Second part in equation (28) of [Cadambe2008]_.

        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger}$$
        Hkk = self.get_Hkl(k, k)
        Hkk_H = Hkk.transpose().conjugate()

        Vkl = Fk[:, l:l + 1]
        Vkl_H = Vkl.transpose().conjugate()
        second_part = np.dot(Hkk, np.dot(np.dot(Vkl, Vkl_H), Hkk_H))

        return second_part

    # noinspection PyPep8
    def _calc_Bkl_cov_matrix_all_l(
            self,
            F_all_users: np.ndarray,
            k: int,
            N0_or_Rek: NumberOrArray = 0.0) -> np.ndarray:
        """
        Calculates the interference-plus-noise covariance matrix for all
        streams at receiver :math:`k` according to equation (28) in [
        Cadambe2008]_.

        The interference-plus-noise covariance matrix for stream :math:`l`
        of user :math:`k` is given by Equation (28) in [Cadambe2008]_,
        which is reproduced below

            :math:`\\mtB^{[kl]} = \\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star l}^{[j]} \\mtV_{\\star l}^{[j]\\dagger} \\mtH^{[kj]\\dagger} - \\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger} + \\mtI_{N^{[k]}}`

        where :math:`P^{[k]}` is the transmit power of transmitter
        :math:`k`, :math:`d^{[k]}` is the number of degrees of freedom of
        user :math:`k`, :math:`\\mtH^{[kj]}` is the channel between
        transmitter :math:`j` and receiver :math:`k`, :math:`\\mtV_{\\star
        l}` is the :math:`l`-th column of the precoder of user :math:`k`
        and :math:`\\mtI_{N^{k}}` is an identity matrix with size equal to
        the number of receive antennas of receiver :math:`k`.

        Parameters
        ----------
        F_all_users : list[np.ndarray] | np.ndarray
            The precoder of all users (already taking into account the
            transmit power). This can be a list of numpy arrays or a 1D numpy
            array of numpy arrays.
        k : int
            Index of the desired user.
        N0_or_Rek : float | np.ndarray
            If this is a 2D numpy array, it is interpreted as the
            covariance matrix of any external interference plus noise. If
            this is a number, it is interpreted as the noise power, in
            which case the covariance matrix will be an identity matrix
            times this noise power.

        Returns
        -------
        Bkl : np.ndarray
            Covariance matrix of all streams of user k. Each element of the
            returned 1D numpy array is a 2D numpy complex array corresponding
            to the covariance matrix of one stream of user k.

        Notes
        -----
        To be simple, a function that returns the covariance matrix of only
        a single stream "l" of the desired user "k" could be implemented,
        but in the order to calculate the max SINR algorithm we need the
        covariance matrix of all streams and returning them in single
        function as is done here allows us to calculate the first part in
        equation (28) of [Cadambe2008]_ only once, since it is the same for
        all streams.

        """
        # $$\mtB^{[kl]} = \sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star l}^{[j]} \mtV_{\star l}^{[j]\dagger} \mtH^{[kj]\dagger} - \frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger} + \mtI_{N^{[k]}}$$

        Ns_k = F_all_users[k].shape[1]
        Bkl_all_l = np.empty(Ns_k, dtype=np.ndarray)
        first_part = self._calc_Bkl_cov_matrix_first_part(
            F_all_users, k, N0_or_Rek)
        for l in range(Ns_k):
            second_part = self._calc_Bkl_cov_matrix_second_part(
                F_all_users[k], k, l)
            Bkl_all_l[l] = first_part - second_part

        return Bkl_all_l

    # noinspection PyPep8
    def _calc_JP_Bkl_cov_matrix_first_part_impl(
            self, Hk: np.ndarray, F_all_users: np.ndarray,
            Rek: NumberOrArray) -> np.ndarray:
        """
        Common implementation of the _calc_JP_Bkl_cov_matrix_first_part.

        Parameters
        ----------
        Hk : np.ndarray
            The channel from all transmitters (not including external
            interference source, if any) to receiver k.
        F_all_users : list[np.ndarray]
            The precoder of all users (already taking into account the
            transmit power).
        Rek : np.ndarray | float
            Covariance matrix of the external interference (if there is
            any) plus noise.

        Returns
        -------
        np.ndarray
            The `first_part` for the Bkl matrix computation.
        """
        # $$\sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[k]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[k]\dagger} + \mtR e_k$$
        first_part = 0.0

        HK_H = Hk.conjugate().transpose()
        for j in range(self.K):
            Vj = F_all_users[j]
            Vj_H = Vj.conjugate().transpose()

            first_part = first_part + np.dot(Hk, np.dot(
                np.dot(Vj, Vj_H), HK_H))
        first_part += Rek

        return first_part

    # noinspection PyPep8,PyPep8
    def _calc_JP_Bkl_cov_matrix_first_part(
            self,
            F_all_users: np.ndarray,
            k: int,
            noise_power: float = 0.0) -> np.ndarray:
        """
        Calculates the first part in the equation of the Blk covariance matrix
        in equation (28) of [Cadambe2008]_ when joint process is employed.

        The first part is given by

            :math:`\\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star d}^{[j]} \\mtV_{\\star d}^{[j]\\dagger} \\mtH^{[kj]\\dagger} + \\mtI_{Nk}`

        Note that it only depends on the value of :math:`k`.

        Parameters
        ----------
        F_all_users : list[np.ndarray] | np.ndarray
            The precoder of all users (already taking into account the
            transmit power). It can be a list of numpy arrays or a numpy
            array of numpy arrays.
        k : int
            Index of the desired user.
        noise_power : float | None, optional
            The noise power.

        Returns
        -------
        np.ndarray
        """
        # The first part in Bkl is given by
        # $$\sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger} + \mtI_{N^{[k]}}$$
        # Note that here the power is already included in `Fk`.
        Rek = (noise_power * np.eye(self.Nr[k]))
        Hk = self.get_Hk(k)
        # noinspection PyTypeChecker
        return self._calc_JP_Bkl_cov_matrix_first_part_impl(
            Hk, F_all_users, Rek)

    # noinspection PyPep8
    @staticmethod
    def _calc_JP_Bkl_cov_matrix_second_part_impl(Hk: np.ndarray,
                                                 Fk: np.ndarray,
                                                 l: int) -> np.ndarray:
        """
        Common implementation of the _calc_JP_Bkl_cov_matrix_second_part
        method.

        Parameters
        ----------
        Hk : np.ndarray
        Fk : np.ndarray
        l : int

        Returns
        -------
        np.ndarray
        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[k]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[k]\dagger}$$
        Hk_H = Hk.transpose().conjugate()

        Vkl = Fk[:, l:l + 1]
        Vkl_H = Vkl.transpose().conjugate()
        second_part = np.dot(Hk, np.dot(np.dot(Vkl, Vkl_H), Hk_H))

        return second_part

    # noinspection PyPep8
    def _calc_JP_Bkl_cov_matrix_second_part(self, Fk: np.ndarray, k: int,
                                            l: int) -> np.ndarray:
        """Calculates the second part in the equation of the Blk covariance
        matrix in equation (28) of [Cadambe2008]_ (note that it does not
        include the identity matrix).

        The second part is given by

            :math:`\\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger}`

        Parameters
        ----------
        Fk : np.ndarray
            The precoder of the desired user.
        k : int
            Index of the desired user.
        l : int
            Index of the desired stream.

        Returns
        -------
        second_part : np.ndarray.
            Second part in equation (28) of [Cadambe2008]_.
        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[k]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[k]\dagger}$$
        Hk = self.get_Hk(k)
        return self._calc_JP_Bkl_cov_matrix_second_part_impl(Hk, Fk, l)

    # noinspection PyPep8
    def _calc_JP_Bkl_cov_matrix_all_l(
            self,
            F_all_users: np.ndarray,
            k: int,
            N0_or_Rek: NumberOrArray = 0.0) -> np.ndarray:
        """
        Calculates the interference-plus-noise covariance matrix for all
        streams at receiver :math:`k` according to equation (28) in [
        Cadambe2008]_.

        The interference-plus-noise covariance matrix for stream :math:`l`
        of user :math:`k` is given by Equation (28) in [Cadambe2008]_,
        which is reproduced below

            :math:`\\mtB^{[kl]} = \\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star l}^{[j]} \\mtV_{\\star l}^{[j]\\dagger} \\mtH^{[kj]\\dagger} - \\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger} + \\mtI_{N^{[k]}}`

        where :math:`P^{[k]}` is the transmit power of transmitter
        :math:`k`, :math:`d^{[k]}` is the number of degrees of freedom of
        user :math:`k`, :math:`\\mtH^{[kj]}` is the channel between
        transmitter :math:`j` and receiver :math:`k`, :math:`\\mtV_{\\star
        l}` is the :math:`l`-th column of the precoder of user :math:`k`
        and :math:`\\mtI_{N^{k}}` is an identity matrix with size equal to
        the number of receive antennas of receiver :math:`k`.

        Parameters
        ----------
        F_all_users : list[np.ndarray] | np.ndarray
            The precoder of all users (already taking into account the
            transmit power). This can be either a 1D numpy array of numpy
            arrays or a list of numpy arrays.
        k : int
            Index of the desired user.
        N0_or_Rek : float | np.ndarray
            If this is a 2D numpy array, it is interpreted as the
            covariance matrix of any external interference plus noise. If
            this is a number, it is interpreted as the noise power, in
            which case the covariance matrix will be an identity matrix
            times this noise power.

        Returns
        -------
        Bkl : np.ndarray
            Covariance matrix of all streams of user k. Each element of the
            returned 1D numpy array is a 2D numpy complex array
            corresponding to the covariance matrix of one stream of user k.

        Notes
        -----

        To be simple, a function that returns the covariance matrix of only
        a single stream "l" of the desired user "k" could be implemented,
        but in the order to calculate the max SINR algorithm we need the
        covariance matrix of all streams and returning them in single
        function as is done here allows us to calculate the first part in
        equation (28) of [Cadambe2008]_ only once, since it is the same for
        all streams.

        """
        # $$\mtB^{[kl]} = \sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star l}^{[j]} \mtV_{\star l}^{[j]\dagger} \mtH^{[kj]\dagger} - \frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger} + \mtI_{N^{[k]}}$$
        Ns_k = F_all_users[k].shape[1]
        Bkl_all_l = np.empty(Ns_k, dtype=np.ndarray)
        first_part = self._calc_JP_Bkl_cov_matrix_first_part(
            F_all_users, k, N0_or_Rek)
        for l in range(Ns_k):
            second_part = self._calc_JP_Bkl_cov_matrix_second_part(
                F_all_users[k], k, l)
            Bkl_all_l[l] = first_part - second_part

        return Bkl_all_l

    def _calc_SINR_k(self, k: int, Fk: np.ndarray, Uk: np.ndarray,
                     Bkl_all_l: np.ndarray) -> np.ndarray:
        """
        Calculates the SINR of all streams of user 'k'.

        Parameters
        ----------
        k : int
            Index of the desired user.
        Fk : np.ndarray
            The precoder of user k.
        Uk : np.ndarray
            The receive filter of user k (before applying the conjugate
            transpose).
        Bkl_all_l : list[np.ndarray] | np.ndarray
            A sequence (1D numpy array, a list, etc) of 2D numpy arrays
            corresponding to the Bkl matrices for all 'l's.

        Returns
        -------
        SINR_k : np.ndarray
            The SINR for the different streams of user k.
        """
        Ns_k = Fk.shape[1]

        SINR_k = np.empty(Ns_k, dtype=float)

        for l in range(Ns_k):
            Fkl = Fk[:, l:l + 1]
            Ukl = Uk[:, l:l + 1]
            Ukl_H = Ukl.conj().T

            aux = np.dot(Ukl_H, np.dot(self.get_Hkl(k, k), Fkl))
            numerator = np.dot(aux, aux.transpose().conjugate())
            denominator = np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))
            SINR_kl = numerator.item() / denominator.item()
            # The imaginary part should be negligible
            SINR_k[l] = np.abs(SINR_kl)

        return SINR_k

    def calc_SINR(self, F: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Calculates the SINR values (in linear scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property.

        Parameters
        ----------
        F : np.ndarray
            The precoders of all users. This should be a 1D numpy array
            of 2D numpy arrays.
        U : np.ndarray
            The receive filters of all users. This should be a 1D numpy
            array of 2D numpy arrays.

        Returns
        -------
        SINRs : np.ndarray
            The SINR (in linear scale) of all streams of all users. This is a
            1D numpy array of 1D numpy arrays (of floats)
        """
        K = self.K
        SINRs = np.empty(K, dtype=np.ndarray)

        for k in range(self.K):
            Bkl_all_l = self._calc_Bkl_cov_matrix_all_l(F, k, self.noise_var)
            SINRs[k] = self._calc_SINR_k(k, F[k], U[k], Bkl_all_l)
        return SINRs

    @staticmethod
    def _calc_JP_SINR_k_impl(Hk: np.ndarray, Fk: np.ndarray, Uk: np.ndarray,
                             Bkl_all_l: np.ndarray) -> np.ndarray:
        """
        Implementation of the :meth:`_calc_JP_SINR_k` method.

        Parameters
        ----------
        Hk : np.ndarray
            Channel from all transmitters to receiver k.
        Fk : np.ndarray
            The precoder of user k.
        Uk : np.ndarray
            The receive filter of user k (before applying the conjugate
            transpose).
        Bkl_all_l : list[np.ndarray]
            A sequence (1D numpy array, a list, etc) of 2D numpy arrays
            corresponding to the Bkl matrices for all 'l's.

        Returns
        -------
        SINR_k : np.ndarray
            The SINR for the different streams of user k.

        Notes
        -----

        The implementation of the _calc_JP_SINR_k method is almost the same
        for the MultiuserChannelMatrix and MultiuserChannelMatrixExtint
        class, except for the `Hk` argument. Therefore, the common code was
        put here and in each class the :meth:`_calc_JP_SINR_k` is
        implemented as simply getting the correct `Hk` argument and then
        calling :meth:`_calc_JP_SINR_k_impl`.
        """
        Ns_k = Fk.shape[1]

        SINR_k = np.empty(Ns_k, dtype=float)

        for l in range(Ns_k):
            Fkl = Fk[:, l:l + 1]
            Ukl = Uk[:, l:l + 1]
            Ukl_H = Ukl.conj().T

            aux = np.dot(Ukl_H, np.dot(Hk, Fkl))
            numerator = np.dot(aux, aux.transpose().conjugate())
            denominator = np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))
            SINR_kl = numerator.item() / denominator.item()
            # The imaginary part should be negligible
            SINR_k[l] = np.abs(SINR_kl)

        return SINR_k

    def _calc_JP_SINR_k(self, k: int, Fk: np.ndarray, Uk: np.ndarray,
                        Bkl_all_l: np.ndarray) -> np.ndarray:
        """
        Calculates the SINR of all streams of user 'k'.

        Parameters
        ----------
        k : int
            Index of the desired user.
        Fk : np.ndarray
            The precoder of user k.
        Uk : np.ndarray
            The receive filter of user k (before applying the conjugate
            transpose).
        Bkl_all_l : list[np.ndarray] | np.ndarray
            A sequence (1D numpy array, a list, etc) of 2D numpy arrays
            corresponding to the Bkl matrices for all 'l's.

        Returns
        -------
        SINR_k : np.ndarray
            The SINR for the different streams of user k.
        """
        Hk = self.get_Hk(k)
        return self._calc_JP_SINR_k_impl(Hk, Fk, Uk, Bkl_all_l)

    def calc_JP_SINR(self, F: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Calculates the SINR values (in linear scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property.

        Parameters
        ----------
        F : np.ndarray
            The precoders of all users. This should be a 1D numpy array
            of 2D numpy arrays.
        U : np.ndarray
            The receive filters of all users. This should be a 1D numpy
            array of 2D numpy arrays.

        Returns
        -------
        SINRs : np.ndarray
            The SINR (in linear scale) of all streams of all users. This is a
            1D numpy array of 1D numpy arrays (of floats).
        """
        K = self.K
        SINRs = np.empty(K, dtype=np.ndarray)

        noise_var = self.noise_var if self.noise_var is not None else 0.0
        for k in range(self.K):
            Bkl_all_l = self._calc_JP_Bkl_cov_matrix_all_l(F, k, noise_var)
            SINRs[k] = self._calc_JP_SINR_k(k, F[k], U[k], Bkl_all_l)
        return SINRs


class MultiUserChannelMatrixExtInt(  # pylint: disable=R0904
        MultiUserChannelMatrix):
    """
    Very similar to the MultiUserChannelMatrix class, but the
    MultiUserChannelMatrixExtInt also includes the effect of an external
    interference.

    This channel matrix can be seem as an concatenation of blocks (of
    non-uniform size) where each block is a channel from one transmitter to
    one receiver and the block size is equal to the number of receive
    antennas of the receiver times the number of transmit antennas of the
    transmitter. The difference compared with MultiUserChannelMatrix is
    that in the MultiUserChannelMatrixExtInt class the interference user
    counts as one more user, but with zero receive antennas.

    For instance, in a 3-users scenario the block (1,0) corresponds to the
    channel between the transmit antennas of user 0 and the receive
    antennas of user 1 (indexing staring at zero). If the number of receive
    antennas and transmit antennas of the three users are [2, 4, 6] and [2,
    3, 5], respectively, then the block (1,0) would have a dimension of
    4x2. The external interference will count as one more block where the
    number of columns of this block corresponds to the rank of the external
    interference. If the external interference has a rank 2 then the
    complete channel matrix would look similar to the block structure
    below.

      +-----+---------+---------------+-----+
      |2 x 2|  2 x 3  |     2 x 5     |2 x 2|
      |     |         |               |     |
      +-----+---------+---------------+-----+
      |4 x 2|  4 x 3  |     4 x 5     |4 x 2|
      |     |         |               |     |
      |     |         |               |     |
      |     |         |               |     |
      +-----+---------+---------------+-----+
      |6 x 2|  6 x 3  |     6 x 5     |6 x 2|
      |     |         |               |     |
      |     |         |               |     |
      |     |         |               |     |
      |     |         |               |     |
      |     |         |               |     |
      +-----+---------+---------------+-----+

    The methods from the MultiUserChannelMatrix class that makes sense were
    reimplemented here to include information regarding the external
    interference.
    """
    def __init__(self) -> None:
        super().__init__()
        self._extIntK: int = 0  # Number of external interference sources
        # Number of transmit antennas of the external interference sources.
        self._extIntNt: int = 0

    # Property to get the number of external interference sources
    @property
    def extIntK(self) -> int:
        """Get method for the extIntK property."""
        return self._extIntK

    # Property to get the number of transmit antennas (or the rank) of the
    # external interference sources
    @property
    def extIntNt(self) -> int:
        """Get method for the extIntNt property."""
        return self._extIntNt

    # Property to get the number of receive antennas of all users. We
    # overwrite the property from the MultiUserChannelMatrix to avoid
    # account the number of receive antennas of the external interference
    # sources.
    @property
    def Nr(self) -> np.ndarray:
        """Get method for the Nr property."""
        return self._Nr[:-self._extIntK]

    @property
    def Nt(self) -> np.ndarray:
        """Get method for the Nt property."""
        return self._Nt[:-self._extIntK]

    @property
    def K(self) -> int:
        """Get method for the K property."""
        return self._K - self._extIntK

    @property
    def big_H_no_ext_int(self) -> np.ndarray:
        """
        Get method for the big_H_no_est_int property.

        big_H_no_est_int is similar to big_H, but does not include the last
        column(s) corresponding to the external interference channel.
        """
        return self.big_H[:, :np.sum(self.Nt)]

    @property
    def H(self) -> np.ndarray:
        """Get method for the H property."""
        # We only care about the first self.K "rows". The remaining rows
        # are the channels from all transmitters to the "external
        # interference user".
        H = self._H_no_pathloss[0:self.K]

        if self._pathloss_matrix is None:
            # No path loss
            return H

        # Apply path loss. Note that the _pathloss_big_matrix matrix
        # has the same dimension as the self._big_H_no_pathloss matrix
        # and we are performing element-wise multiplication here.
        return H * np.sqrt(self._pathloss_matrix)

    @property
    def H_no_ext_int(self) -> np.ndarray:
        """Get method for the H_no_ext_int property."""
        # Call H property get method of the base class
        H = MultiUserChannelMatrix.H.fget(self)  # type: ignore
        return H[:self.K, :self.K]

    def corrupt_data(  # type: ignore
            self, data: np.ndarray, ext_int_data: np.ndarray) -> np.ndarray:
        """
        Corrupt data passed through the channel.

        If the noise_var member variable is not None then an white noise
        will also be added.

        Parameters
        ----------
        data : np.ndarray
            An array of numpy matrices with the data of the multiple
            users. The k-th element in `data` is a numpy array with
            dimension Nt_k x NSymbs, where Nt_k is the number of transmit
            antennas of the k-th user and NSymbs is the number of
            transmitted symbols.
        ext_int_data : list[np.ndarray] | np.ndarray
            An array of numpy matrices with the data of the external
            interference sources. The l-th element is the data transmitted
            by the l-th external interference source, which must have a
            dimension of NtEl x NSymbs, where NtEl is the number of
            transmit antennas of the l-th external interference source.

        Returns
        -------
        output : np.ndarray
            A numpy array where each element contains the received data (a
            2D numpy array) of a user.
        """
        input_data = np.hstack([data, ext_int_data])
        return MultiUserChannelMatrix.corrupt_data(self, input_data)

    def corrupt_concatenated_data(self, data: np.ndarray) -> np.ndarray:
        """
        Corrupt data passed through the channel.

        If the noise_var member variable is not None then an white noise
        will also be added.

        Parameters
        ----------
        data : np.ndarray
            A bi-dimensional numpy array with the concatenated data of all
            transmitters as well as the data from all external interference
            sources. The dimension of data is (sum(self._Nt) +
            sum(self.extIntNt)) x NSymb. That is, the number of rows
            corresponds to the sum of the number of transmit antennas of
            all users and external interference sources and the number of
            columns correspond to the number of transmitted symbols.

        Returns
        -------
        output : np.ndarray
            A bi-dimension numpy array where the number of rows corresponds
            to the sum of the number of receive antennas of all users and
            the number of columns correspond to the number of transmitted
            symbols.

        """
        return MultiUserChannelMatrix.corrupt_concatenated_data(self, data)

    def get_Hk_without_ext_int(self, k: int) -> np.ndarray:
        """
        Get the channel from all transmitters (without including the external
        interference sources) to receiver `k`.

        Parameters
        ----------
        k : int
            Receiving user.

        Returns
        -------
        channel_k : np.ndarray
            Channel from all transmitters to receiver `k` (2D numpy array).

        See also
        --------
        .get_Hkl,
        .get_Hk,
        get_Hk_with_ext_int

        Examples
        --------
        >>> multiH = MultiUserChannelMatrixExtInt()
        >>> H = np.reshape(np.r_[0:20], [4,5])
        >>> Nt = np.array([2, 2])
        >>> Nr = np.array([2, 2])
        >>> NextInt = np.array([1])
        >>> multiH.init_from_channel_matrix(H, Nr, Nt, 2, 1)
        >>> # Note that the last column of multiH.big_H corresponds to the
        >>> # external interference source
        >>> print(multiH.big_H)
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]]
        >>> print(multiH.get_Hk_without_ext_int(0))
        [[0 1 2 3]
         [5 6 7 8]]
        >>> print(multiH.get_Hk_without_ext_int(1))
        [[10 11 12 13]
         [15 16 17 18]]
        """
        receive_channels = single_matrix_to_matrix_of_matrices(
            self.big_H[:, :np.sum(self.Nt)], self.Nr)
        return receive_channels[k]

    # This is exactly the same as the
    # get_Hk_without_ext_int method from the
    # MultiUserChannelMatrix class. therefore, we don't need to test it.
    def get_Hk_with_ext_int(self, k: int) -> np.ndarray:
        """
        Get the channel from all transmitters (including the external
        interference sources) to receiver `k`.

        This method is essentially the same as the get_Hk method.

        Parameters
        ----------
        k : int
            Receiving user.

        Returns
        -------
        channel_k : np.ndarray
            Channel from all transmitters to receiver `k`.

        See also
        --------
        .get_Hkl,
        .get_Hk,
        get_Hk_without_ext_int

        Examples
        --------
        >>> multiH = MultiUserChannelMatrixExtInt()
        >>> H = np.reshape(np.r_[0:20], [4,5])
        >>> Nt = np.array([2, 2])
        >>> Nr = np.array([2, 2])
        >>> NextInt = np.array([1])
        >>> multiH.init_from_channel_matrix(H, Nr, Nt, 2, 1)
        >>> # Note that the last column of multiH.big_H corresponds to the
        >>> # external interference source
        >>> print(multiH.big_H)
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]]
        >>> print(multiH.get_Hk_with_ext_int(0))
        [[0 1 2 3 4]
         [5 6 7 8 9]]
        >>> print(multiH.get_Hk_with_ext_int(1))
        [[10 11 12 13 14]
         [15 16 17 18 19]]
        """
        return MultiUserChannelMatrix.get_Hk(self, k)  # pragma: no cover

    @staticmethod
    def _prepare_input_parans(
        Nr: np.ndarray, Nt: np.ndarray, K: int, NtE: Iterable[int]
    ) -> Tuple[np.ndarray, np.ndarray, int, int, np.ndarray]:
        """
        Helper method used in the init_from_channel_matrix and randomize
        method definitions.

        Parameters
        ----------
        Nr : np.ndarray
            Number of antennas at each receiver.
        Nt : np.ndarray
            Number of antennas at each transmitter.
        K : int
            Number of transmit/receive pairs.
        NtE : int | list[int] | np.ndarray
            Number of transmit antennas of the external interference
            source(s). If `NtE` is an iterable, the number of external
            interference sources will be the len(NtE).

        Returns
        -------
        output : tuple
            The tuple (full_Nr, full_Nt, full_K, extIntK, extIntNt).
        """
        if isinstance(NtE, (int, np.int_)):
            # NtE is a scalar number, which means we have a single external
            # interference source.
            extIntK = 1
            extIntNt = np.array([NtE])
        else:
            # We have multiple external interference sources
            extIntNt = np.array(NtE)
            extIntK = extIntNt.size

        # Number of receive antennas also including the number of receive
        # antennas of the interference users (which are zeros)
        full_Nr = np.hstack([Nr, np.zeros(extIntK, dtype=int)])
        # Number of transmit antennas also including the number of transmit
        # antennas of the interference users
        full_Nt = np.hstack([Nt, NtE])
        # Total number of users including the interference users.
        full_K = K + extIntK

        return full_Nr, full_Nt, full_K, extIntK, extIntNt

    def init_from_channel_matrix(  # type: ignore
            self, channel_matrix: np.ndarray, Nr: np.ndarray, Nt: np.ndarray,
            K: int, NtE: Iterable[int]) -> None:
        """
        Initializes the multiuser channel matrix from the given
        `channel_matrix`.

        Note that `channel_matrix` must also include the channel terms for
        the external interference, which must be the last `NtE` columns of
        `channel_matrix`. The number of rows in `channel_matrix` must be
        equal to np.sum(Nr), while the number of columns must be
        np.sum(Nt) + NtE.

        Parameters
        ----------
        channel_matrix : np.ndarray
            A matrix concatenating the channel of all transmitters
            (including the external interference sources) to all receivers.
        Nr : np.ndarray
            Number of antennas at each receiver.
        Nt : np.ndarray
            Number of antennas at each transmitter.
        K : int
            Number of transmit/receive pairs.
        NtE : int | list[int] | np.ndarray
            Number of transmit antennas of the external interference
            source(s). If NtE is an iterable, the number of external
            interference sources will be the len(NtE).

        Raises
        ------
        ValueError
            If the arguments are invalid.
        """
        (full_Nr, full_Nt, full_K, extIntK, extIntNt) \
            = MultiUserChannelMatrixExtInt._prepare_input_parans(
                Nr, Nt, K, NtE)

        self._extIntK = extIntK
        self._extIntNt = extIntNt

        MultiUserChannelMatrix.init_from_channel_matrix(
            self, channel_matrix, full_Nr, full_Nt, full_K)

    def randomize(  # type: ignore
            self, Nr: IntOrIntArrayUnion, Nt: IntOrIntArrayUnion, K: int,
            NtE: Iterable[int]) -> None:
        """
        Generates a random channel matrix for all users as well as for the
        external interference source(s).

        Parameters
        ----------
        Nr : int | np.ndarray
            Number of receive antennas of each user. If an integer is
            specified, all users will have that number of receive antennas.
        Nt : int | np.ndarray
            Number of transmit antennas of each user. If an integer is
            specified, all users will have that number of receive antennas.
        K : int
            Number of users.
        NtE : int | list[int] | np.ndarray
            Number of transmit antennas of the external interference
            source(s). If NtE is an iterable, the number of external
            interference sources will be the len(NtE).
        """
        if isinstance(Nr, int):
            Nr = np.ones(K, dtype=int) * Nr
        if isinstance(Nt, int):
            Nt = np.ones(K, dtype=int) * Nt

        (full_Nr, full_Nt, full_K, extIntK, extIntNt) \
            = MultiUserChannelMatrixExtInt._prepare_input_parans(
                Nr, Nt, K, NtE)

        self._extIntK = extIntK
        self._extIntNt = extIntNt

        MultiUserChannelMatrix.randomize(self, full_Nr, full_Nt, full_K)

    def set_pathloss(self,
                     pathloss_matrix: Optional[np.ndarray] = None,
                     ext_int_pathloss: Optional[np.ndarray] = None) -> None:
        """
        Set the path loss (IN LINEAR SCALE) from each transmitter to each
        receiver, as well as the path loss from the external interference
        source(s) to each receiver.

        The path loss will be accounted when calling the get_Hkl, get_Hk,
        the corrupt_concatenated_data and the corrupt_data methods.

        Note that path loss is a power relation, which means that the
        channel coefficients will be multiplied by the square root of
        elements in `pathloss_matrix`.

        If you want to disable the path loss, set pathloss_matrix to None.

        Parameters
        ----------
        pathloss_matrix : np.ndarray
            A matrix with dimension "K x K", where K is the number of
            users, with the path loss (IN LINEAR SCALE) from each
            transmitter (columns) to each receiver (rows). If you want to
            disable the path loss then set it to None.  ext_int_pathloss :
            2D numpy array The path loss from each interference source to
            each receiver. The number of rows of ext_int_pathloss must be
            equal to the number of receives, while the number of columns
            must be equal to the number of external interference sources.
        ext_int_pathloss : np.ndarray
            The external interference path loss.
        """
        # A matrix with the path loss from each transmitter to each
        # receiver.
        self._pathloss_matrix = pathloss_matrix

        if pathloss_matrix is None:
            self._pathloss_matrix = None
            self._pathloss_big_matrix = None
        else:
            pathloss_matrix_with_ext_int = np.hstack(
                [pathloss_matrix, ext_int_pathloss])
            self._pathloss_matrix = pathloss_matrix_with_ext_int

            self._pathloss_big_matrix \
                = MultiUserChannelMatrix._from_small_matrix_to_big_matrix(
                    pathloss_matrix_with_ext_int, self._Nr, self._Nt,
                    self.K, self._K)

            # Assures that _pathloss_matrix and _pathloss_big_matrix
            # will stay in sync by disallowing modification of
            # individual elements in both of them.
            self._pathloss_matrix.setflags(write=False)
            self._pathloss_big_matrix.setflags(write=False)

    def calc_cov_matrix_extint_without_noise(self,
                                             pe: float = 1.0) -> np.ndarray:
        """
        Calculates the covariance matrix of the external interference
        without include the noise.

        Parameters
        ----------
        pe : float, optional
            External interference power (in linear scale)

        Returns
        -------
        R_all_k : np.ndarray
            Return a numpy array, where each element is the covariance
            matrix of the external interference at one receiver.
        """
        # $$\mtR_e = \sum_{j=1}^{Ke} P_{e_j} \mtH_{k{e_j}} \mtH_{k{e_j}}^H$$
        R_all_k = np.empty(self.Nr.size, dtype=np.ndarray)
        cum_Nr = np.hstack([0, np.cumsum(self.Nr)])

        for ii in range(self.Nr.size):
            extH = self.big_H[cum_Nr[ii]:cum_Nr[ii + 1], np.sum(self.Nt):]
            R_all_k[ii] = pe * np.dot(extH, extH.transpose().conjugate())
        return R_all_k

    # noinspection PyPep8
    def calc_cov_matrix_extint_plus_noise(self, pe: float = 1.0) -> np.ndarray:
        """
        Calculates the covariance matrix of the external interference plus
        noise.

        Parameters
        ----------
        pe : float, optional [default=1]
            External interference power (in linear scale)

        Returns
        -------
        R_all_k : np.ndarray
            Return a numpy array, where each element is the covariance
            matrix of the external interference plus noise at one receiver.
        """
        # $$\mtR_e = \sum_{j=1}^{Ke} P_{e_j} \mtH_{k{e_j}} \mtH_{k{e_j}}^H + \sigma_n^2 \mtI$$
        # where $Ke$ is the number of external interference sources and
        # ${e_j}$ is the j-th external interference source.

        # Calculate the covariance matrix of the external interference
        # without noise.
        R_all_k = self.calc_cov_matrix_extint_without_noise(pe)

        if self.noise_var is not None:
            # If self.noise_var is not None then let's add the noise
            # covariance matrix
            noise_var = self.noise_var
            for i, R_all_k_i in enumerate(R_all_k):
                R_all_k_i += np.eye(self.Nr[i]) * noise_var

        return R_all_k

    # noinspection PyPep8,PyPep8
    def calc_Q(self,
               k: int,
               F_all_users: np.ndarray,
               pe: float = 1.0) -> np.ndarray:
        """
        Calculates the interference covariance matrix at the
        :math:`k`-th receiver.

        The interference covariance matrix at the :math:`k`-th receiver,
        :math:`\\mtQ k`, is given by

            :math:`\\mtQ k = \\sum_{j=1, j \\neq k}^{K} \\frac{P_j}{Ns_j} \\mtH_{kj} \\mtF_j \\mtF_j^H \\mtH_{kj}^H`

        where :math:`P_j` is the transmit power of transmitter :math:`j`,
        and :math:`Ns_j` is the number of streams for user :math:`j`.

        Parameters
        ----------
        k : int
            Index of the desired receiver.
        F_all_users : list[np.ndarray] | np.ndarray
            The precoder of all users (already taking into account the
            transmit power). This should be either a list of numpy
            2D arrays or a 1D numpy array of 2D numpy arrays.
        pe : float
            The power of the external interference source(s).

        Returns
        -------
        Qk : np.ndarray
            The interference covariance matrix at receiver :math:`k`.
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{kj} \mtF_j \mtF_j^H \mtH_{kj}^H + \mtR_e$$
        Rek_all_k = self.calc_cov_matrix_extint_plus_noise(pe)
        Qk = self._calc_Q_impl(k, F_all_users) + Rek_all_k[k]

        return Qk

    # noinspection PyPep8
    def _calc_JP_Q(self, k: int, F_all_users: np.ndarray) -> np.ndarray:
        """
        Calculates the interference covariance matrix at the :math:`k`-th
        receiver with a joint processing scheme (not including the
        covariance matrix of the external interference plus noise)

        Parameters
        ----------
        k : int
            Index of the desired receiver.
        F_all_users : np.ndarray
            The precoder of all users (already taking into account the
            transmit power). This is a 1D numpy array of 2D numpy array.

        See also
        --------
        calc_JP_Q
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{k} \mtF_j \mtF_j^H \mtH_{k}^H$$
        interfering_users = set(range(self.K)) - {k}
        Qk = np.zeros([self.Nr[k], self.Nr[k]], dtype=complex)

        for l in interfering_users:
            Hk_F = np.dot(self.get_Hk_without_ext_int(k), F_all_users[l])
            Qk = Qk + np.dot(Hk_F, Hk_F.transpose().conjugate())

        return Qk

    # noinspection PyPep8,PyPep8
    def calc_JP_Q(self,
                  k: int,
                  F_all_users: np.ndarray,
                  pe: float = 1.0) -> np.ndarray:
        """
        Calculates the interference covariance matrix at the
        :math:`k`-th receiver with a joint processing scheme.

        The interference covariance matrix at the :math:`k`-th receiver,
        :math:`\\mtQ k`, is given by

            :math:`\\mtQ k = \\sum_{j=1, j \\neq k}^{K} \\frac{P_j}{Ns_j} \\mtH_{k} \\mtF_j \\mtF_j^H \\mtH_{k}^H`

        where :math:`P_j` is the transmit power of transmitter :math:`j`,
        and :math:`Ns_j` is the number of streams for user :math:`j`.

        Parameters
        ----------
        k : int
            Index of the desired receiver.
        F_all_users : list[np.ndarray] | np.ndarray
            The precoder of all users (already taking into account the
            transmit power). This is a 1D numpy array of 2D numpy array.
        pe : float
            The power of the external interference source(s).

        Returns
        -------
        Qk : np.ndarray
            The interference covariance matrix at receiver :math:`k`.
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{k} \mtF_j \mtF_j^H \mtH_{k}^H + \mtR_e$$
        Rek_all_k = self.calc_cov_matrix_extint_plus_noise(pe)
        Qk = self._calc_JP_Q(k, F_all_users) + Rek_all_k[k]

        return Qk

    # noinspection PyUnresolvedReferences
    def calc_SINR(self,
                  F: np.ndarray,
                  U: np.ndarray,
                  pe: float = 1.0) -> np.ndarray:
        """
        Calculates the SINR values (in linear scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property.

        Parameters
        ----------
        F : list[np.ndarray] | np.ndarray
            The precoders of all users. This should be either a list of numpy
            2D arrays or a 1D numpy array of 2D numpy arrays.
        U : list[np.ndarray] | np.ndarray
            The receive filters of all users. This should be either a list of
            numpy 2D arrays or a 1D numpy array of 2D numpy arrays.
        pe : float
            Power of the external interference source.

        Returns
        -------
        SINRs : np.ndarray
            The SINR (in linear scale) of all streams of all users. This is a
            1D numpy array of 1D numpy arrays (of floats)
        """
        K = self.K
        SINRs = np.empty(K, dtype=np.ndarray)

        Re_all_k = self.calc_cov_matrix_extint_plus_noise(pe)

        for k in range(self.K):
            Bkl_all_l = self._calc_Bkl_cov_matrix_all_l(F, k, Re_all_k[k])
            SINRs[k] = self._calc_SINR_k(k, F[k], U[k], Bkl_all_l)
        return SINRs

    # pylint: disable=W0222
    # noinspection PyPep8
    def _calc_JP_Bkl_cov_matrix_first_part(  # type: ignore
            self, F_all_users: np.ndarray, k: int,
            Rek: NumberOrArray) -> np.ndarray:
        """
        Calculates the first part in the equation of the Blk covariance matrix
        in equation (28) of [Cadambe2008]_ when joint process is employed.

        The first part is given by

            :math:`\\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star d}^{[j]} \\mtV_{\\star d}^{[j]\\dagger} \\mtH^{[kj]\\dagger} + \\mtI_{Nk}`

        Note that it only depends on the value of :math:`k`.

        Parameters
        ----------
        F_all_users : list[np.ndarray] | np.ndarray
            The precoder of all users (already taking into account the
            transmit power). This can be either a 1D numpy array of numpy
            arrays or a list of numpy arrays.
        k : int
            Index of the desired user.
        Rek : np.ndarray | float
            Covariance matrix of the external interference plus noise.

        Returns
        -------
        np.ndarray
            The first part in the equation of the Blk covariance matrix.
        """
        # The first part in Bkl is given by
        # $$\sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[k]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[k]\dagger} + \mtR e_k$$
        # Note that here the power is already included in `Fk`.

        Hk = self.get_Hk_without_ext_int(k)
        return self._calc_JP_Bkl_cov_matrix_first_part_impl(
            Hk, F_all_users, Rek)

    # noinspection PyPep8,PyPep8
    def _calc_JP_Bkl_cov_matrix_second_part(self, Fk: np.ndarray, k: int,
                                            l: int) -> np.ndarray:
        """
        Calculates the second part in the equation of the Blk covariance
        matrix in equation (28) of [Cadambe2008]_ (note that it does not
        include the identity matrix).

        The second part is given by

            :math:`\\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger}`

        Parameters
        ----------
        Fk : np.ndarray
            The precoder of the desired user.
        k : int
            Index of the desired user.
        l : int
            Index of the desired stream.

        Returns
        -------
        second_part : np.ndarray
            Second part in equation (28) of [Cadambe2008]_.

        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[k]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[k]\dagger}$$
        Hk = self.get_Hk_without_ext_int(k)
        return self._calc_JP_Bkl_cov_matrix_second_part_impl(Hk, Fk, l)

    def _calc_JP_SINR_k(self, k: int, Fk: np.ndarray, Uk: np.ndarray,
                        Bkl_all_l: np.ndarray) -> np.ndarray:
        """
        Calculates the SINR of all streams of user 'k'.

        Parameters
        ----------
        Fk : np.ndarray
            The precoder of user k.
        Uk : np.ndarray
            The receive filter of user k (before applying the conjugate
            transpose).
        k : int
            Index of the desired user.
        Bkl_all_l : list[np.ndarray] | np.ndarray
            A sequence (1D numpy array, a list, etc) of 2D numpy arrays
            corresponding to the Bkl matrices for all 'l's.

        Returns
        -------
        np.ndarray
            The SINR for the different streams of user k.

        """
        Hk = self.get_Hk_without_ext_int(k)
        return self._calc_JP_SINR_k_impl(Hk, Fk, Uk, Bkl_all_l)

    def calc_JP_SINR(self,
                     F: np.ndarray,
                     U: np.ndarray,
                     pe: float = 1.0) -> np.ndarray:
        """
        Calculates the SINR values (in linear scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property.

        Parameters
        ----------
        F : np.ndarray
            The precoders of all users. This is a 1D numpy array of 2D
            numpy arrays.
        U : np.ndarray
            The receive filters of all users. This is a 1D numpy array
            of 2D numpy arrays.
        pe : float
            The external interference power.

        Returns
        -------
        np.ndarray
            The SINR (in linear scale) of all streams of all users. This
            is a 1D numpy array of 1D numpy arrays (of floats).
        """
        K = self.K
        SINRs = np.empty(K, dtype=np.ndarray)

        Re_all_k = self.calc_cov_matrix_extint_plus_noise(pe)

        for k in range(self.K):
            Bkl_all_l = self._calc_JP_Bkl_cov_matrix_all_l(F, k, Re_all_k[k])
            SINRs[k] = self._calc_JP_SINR_k(k, F[k], U[k], Bkl_all_l)
        return SINRs
