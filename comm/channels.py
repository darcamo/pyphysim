#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with implementation of channel related classes.

The :class:`MultiUserChannelMatrix` and
:class:`MultiUserChannelMatrixExtInt` classes implement the MIMO
Interference Channel (MIMO-IC) model, where the first one does not include
an external interference source while the last one includes it. The MIMO-IC
model is shown in the Figure below.

.. figure:: /_images/mimo_ic.svg
   :align: center

   MIMO Interference Channel

"""

__revision__ = "$Revision$"

from collections import Iterable
import numpy as np
from scipy.linalg import block_diag
from util.conversion import single_matrix_to_matrix_of_matrices
from util.misc import randn_c_RS

__all__ = ['MultiUserChannelMatrix', 'MultiUserChannelMatrixExtInt', 'JakesSampleGenerator', 'generate_jakes_samples']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def generate_jakes_samples(Fd, Ts=1e-3, NSamples=100, L=8, shape=None, RS=None, start_time=0.0):
    """
    Generates channel samples according to the Jakes model.

    This functions generates channel samples for a single tap according to
    the Jakes model given by

    .. math::
       :label: jakes_model

       h(t) = \\frac{1}{\\sqrt{L}}\\sum_{l=0}^{L-1}\\exp\\{j[2\\pi f_D \\cos(\\phi_l)t+\\psi_l]\\}


    Parameters
    ----------
    Fd : double
        The Doppler frequency (in Hetz).
    Ts : double
        The sample interval (in seconds).
    NSamples : int
        The number of samples to generate.
    L : int
        The number of rays for the Jakes model.
    shape : tuple (of integers)
        The shape of the generated channel. This is used to generate MIMO
        channels. For instance, in order to generate channels samples for a
        MIMO scenario with 3 receive antennas and 2 transmit antennas use a
        shape of (3, 2).
    RS : A numpy.random.RandomState object.
        The RandomState object used to generate the random values. If not
        provided, the global RandomState in numpy will be used.
    start_time : float
        The first value (start time) of `t` in :eq:`jakes_model`.

    Returns
    -------
    h : Numpy array
        The generated channel. If `shape` is None the the shape of the
        returned h is equal to (NSamples,). That is, h is a 1-dimensional
        numpy array. If `shape` was provided then the shape of h is the
        provided shape with an additional dimension for the time (the last
        dimension). For instance, if a `shape` of (3, 2) was provided then
        the shape of the returned h will be (3, 2, NSamples).
    """
    # $h(t) = \frac{1}{\sqrt{L}}\sum_{l=0}^{L-1}\exp\{j[2\pi f_D \cos(\phi_l)t+\psi_l]\}$

    # if RS is None:
    #     # If RS was not provided, we set it to the numpy.random module. That
    #     # way, when the rand "method" in RS is called it will actually call
    #     # the global rand function in numpy.random.  RandomState object in
    #     # numpy.
    #     RS = np.random

    # # xxxxx Time samples xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # t = np.arange(start_time, NSamples * Ts + start_time, Ts * 1.0000000001)
    # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # # xxxxx Generate phi_l and psi_l xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # if shape is None:
    #     # The dimension of phi_l and psi_l will be L x 1. We set the last
    #     # dimensions as 1, instead of setting the dimension of phi_l and
    #     # psi_l simply as (L,), because it will be broadcasted later by
    #     # numpy when we multiply with the time.
    #     phi_l = 2 * np.pi * RS.rand(L, 1)
    #     psi_l = 2 * np.pi * RS.rand(L, 1)
    # else:
    #     # The dimension of phi_l and psi_l will be L x Shape x 1. We set
    #     # the last dimensions as 1, instead of setting the dimension of
    #     # phi_l and psi_l simply as (L,), because it will be broadcasted
    #     # later by numpy when we multiply with the time.
    #     new_shape = [L]
    #     new_shape.extend(shape)
    #     new_shape.append(1)
    #     phi_l = 2 * np.pi * RS.rand(*new_shape)
    #     psi_l = 2 * np.pi * RS.rand(*new_shape)
    # # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # # xxxxx Calculates h xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # t_aux = np.tile(t, [L, 1])
    # # If shape is Not None we need to modify the shape of t_aux. We will
    # # add new dimensions (with value 1) for the dimensions specified in
    # # `shape` so that numpy broadcast will allow us to perform the
    # # multiplications required to calculate `h` later.
    # if shape is not None:
    #     new_shape = [L]
    #     new_shape.extend([1] * len(shape))
    #     new_shape.append(NSamples)
    #     t_aux.shape = new_shape

    # # $h = \sqrt{1.0 / L} \sum \exp(1j (2 \pi Fd \cos(\phi_l) t + \psi_l))$
    # h = np.sqrt(1.0 / L) * np.sum(np.exp(1j * (2 * np.pi * Fd * np.cos(phi_l) * t_aux + psi_l)), axis=0)
    # return h
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    obj = JakesSampleGenerator(Fd, Ts, L, shape, RS, start_time)
    return obj.generate_channel_samples(NSamples)


class JakesSampleGenerator(object):
    """
    The purpose of this class is to generate channel samples according to
    the Jakes model given by

    .. math:: h(t) = \\frac{1}{\\sqrt{L}}\\sum_{l=0}^{L-1}\\exp\\{j[2\\pi f_D \\cos(\\phi_l)t+\\psi_l]\\}

    This class is actually a wrapper to the :meth:`generate_jakes_samples`
    function in this module. Its main purpose is to allow easier usage of
    generate_jakes_samples as well as generating "more samples" continuing
    a previous call to generate_jakes_samples.

    Parameters
    ----------
    Fd : double
        The Doppler frequency (in Hetz).
    Ts : double
        The sample interval (in seconds).
    L : int
        The number of rays for the Jakes model.
    shape : tuple (of integers)
        The shape of the generated channel. This is used to generate MIMO
        channels. For instance, in order to generate channels samples for a
        MIMO scenario with 3 receive antennas and 2 transmit antennas use a
        shape of (3, 2).
    RS : A numpy.random.RandomState object.
        The RandomState object used to generate the random values. If not
        provided, the global RandomState in numpy will be used.
    start_time : float
        The first value (start time) of `t` in :eq:`jakes_model`.


    See also
    --------
    generate_jakes_samples
    """

    def __init__(self, Fd=100, Ts=1e-3, L=8,
                 shape=None, RS=None, start_time=0.0):
        self.Fd = Fd
        self.Ts = Ts
        self.L = L
        self.shape = shape

        if RS is None:
        # If RS was not provided, we set it to the numpy.random module. That
        # way, when the rand "method" in RS is called it will actually call
        # the global rand function in numpy.random.  RandomState object in
        # numpy.
            RS = np.random
        self.RS = RS

        # self._current_time will be update after each call to the
        # generate_channel_samples method.
        self._current_time = start_time

        # xxxxx Generate phi_l and psi_l xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        self._shape = shape
        if shape is None:
            # The dimension of phi_l and psi_l will be L x 1. We set the last
            # dimensions as 1, instead of setting the dimension of phi_l and
            # psi_l simply as (L,), because it will be broadcasted later by
            # numpy when we multiply with the time.
            self._phi_l = 2 * np.pi * RS.rand(L, 1)
            self._psi_l = 2 * np.pi * RS.rand(L, 1)
        else:
            # The dimension of phi_l and psi_l will be L x Shape x 1. We set
            # the last dimensions as 1, instead of setting the dimension of
            # phi_l and psi_l simply as (L,), because it will be broadcasted
            # later by numpy when we multiply with the time.
            new_shape = [L]
            new_shape.extend(shape)
            new_shape.append(1)
            self._phi_l = 2 * np.pi * RS.rand(*new_shape)
            self._psi_l = 2 * np.pi * RS.rand(*new_shape)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _generate_time_samples(self, NSamples):
        """
        Generate the time samples that will be used internally in
        generate_channel_samples method.

        Parameters
        ----------
        NSamples : int
            Number of samples to be generated.

        Returns
        -------
        t_aux : Numpy Array
            The numpy array with the time samples.

        Notes
        -----
        Each time _generate_time_samples is called it will update
        self._current_time to reflect the advance of the time after
        generating the new samples.
        """
        t = np.arange(
            self._current_time,  # Start time
            NSamples * self.Ts + self._current_time,
            self.Ts * 1.0000000001)
        t_aux = np.tile(t, [self.L, 1])
        if self._shape is not None:
            new_shape = [self.L]
            new_shape.extend([1] * len(self._shape))
            new_shape.append(NSamples)
            t_aux.shape = new_shape

        # Update the self._current_time variable with the value of the next
        # time sample that should be generated when _generate_time_samples
        # is called again.
        self._current_time = t[-1] + self.Ts
        return t_aux

    def generate_channel_samples(self, NSamples):
        """
        Generate more samples for the Jakes model.

        Parameters
        ----------
        NSamples : int
            Number of samples to be generated.

        Returns
        -------
        h : Numpy array
            The generated channel samples. The shape is in the form SHAPE x
            NSamples, where SHAPE is a tuple with the shape provided in the
            constructor of the JakesSampleGenerator class.

        Notes
        -----
        This method will update the self._current_time variable.
        """
        # This method will also update the _current_time member variable
        t = self._generate_time_samples(NSamples)

        # Finally calculate the channel samples
        h = np.sqrt(1.0 / self.L) * np.sum(np.exp(1j * (2 * np.pi * self.Fd * np.cos(self._phi_l) * t + self._psi_l)), axis=0)

        return h

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx MultiUserChannelMatrix Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MultiUserChannelMatrix(object):
    """Stores the (fast fading) channel matrix of a multi-user scenario. The
    path-loss from each transmitter to each receiver is also be accounted if
    the set_pathloss is called to set the path-loss matrix.

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

    .. aafig::
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
    user `l`, call the `get_channel` method.

    """

    def __init__(self):
        # The _big_H variable is an internal variable with all the channels
        # from each transmitter to each receiver represented as a single
        # big matrix.
        self._big_H = np.array([], dtype=np.ndarray)
        # The _H variable is an internal variable with all the channels
        # from each transmitter to each receiver. It points to the same
        # data as the _big_H variable, however, _H is a "matrix of
        # matrices" instead of a single big matrix.
        self._H = np.array([], dtype=np.ndarray)
        self._Nr = np.array([])
        self._Nt = np.array([])
        self._K = 0
        self._pathloss_matrix = None
        # _pathloss_big_matrix should not be set directly. It is set when
        # _pathloss_matrix is set in the set_pathloss method.
        self._pathloss_big_matrix = None
        self._RS_channel = np.random.RandomState()
        self._RS_noise = np.random.RandomState()

        self._last_noise = None  # Store the AWGN noise array from the last
                                 # time any of the corrupt*_data methods
                                 # were called.
        self._last_noise_var = 0.0  # Store the noise variance from the
                                    # last time any of the corrupt*_data
                                    # methods were called.

        self._W = None  # Post processing filters (a list of 2D numpy
                        # arrays) for each user
        self._big_W = None  # Same as _W, but as a single block diagonal
                            # matrix.

    def set_channel_seed(self, seed=None):
        """Set the seed of the RandomState object used to generate the random
        elements of the channel (when self.randomize is called).

        Parameters
        ----------
        seed : Int or array like
            Random seed initializing the pseudo-random number
            generator. See np.random.RandomState help for more info.

        """
        self._RS_channel.seed(seed=None)

    def set_noise_seed(self, seed):
        """Set the seed of the RandomState object used to generate the random
        noise elements (when the corrupt data function is called).

        Parameters
        ----------
        seed : Int or array like
            Random seed initializing the pseudo-random number
            generator. See np.random.RandomState help for more info.

        """
        self._RS_noise.seed(seed)

    def re_seed(self):
        """
        Re-seed the channel and noise RandomState objects randomly.

        If you want to specify the seed for each of them call the
        `set_channel_seed` and `set_noise_seed` methods and pass the
        desired seed for each of them.
        """
        self.set_channel_seed(None)
        self.set_noise_seed(None)

    # def randn_c(self, RS, *args):
    #     """Generates a random circularly complex gaussian matrix.

    #     This is essentially the same as the the randn_c function from the
    #     util.misc module. The only difference is that the randn_c function in
    #     util.misc uses the global RandomState object in numpy, while the
    #     method here used the randn method from a local RandomState
    #     object. This allow us greatter control.

    #     Parameters
    #     ----------
    #     RS : A numpy.random.RandomState object.
    #         The RandomState object used to generate the random values.
    #     *args : variable number of ints
    #         Variable number of arguments specifying the dimensions of the
    #         returned array. This is directly passed to the
    #         numpy.random.randn function.

    #     Returns
    #     -------
    #     result : N-dimensional numpy array
    #         A random N-dimensional numpy array (complex dtype) where the
    #         `N` is equal to the number of parameters passed to `randn_c`.
    #     """
    #     return (1.0 / math.sqrt(2.0)) * (
    #         RS.randn(*args) + (1j * RS.randn(*args)))

    # Property to get the number of receive antennas
    def _get_Nr(self):
        """Get method for the Nr property."""
        return self._Nr
    Nr = property(_get_Nr)

    # Property to get the number of transmit antennas
    def _get_Nt(self):
        """Get method for the Nt property."""
        return self._Nt
    Nt = property(_get_Nt)

    # Property to get the number of users
    def _get_K(self):
        """Get method for the K property."""
        return self._K
    K = property(_get_K)

    # Property to get the matrix of channel matrices (with pass loss
    # applied if any)
    def _get_H(self):
        """Get method for the H property."""
        if self._pathloss_matrix is None:
            # No path loss
            return self._H
        else:
            # Apply path loss. Note that the _pathloss_big_matrix matrix
            # has the same dimension as the self._big_H matrix and we are
            # performing element-wise multiplication here.
            return self._H * np.sqrt(self._pathloss_matrix)
    H = property(_get_H)

    # Property to get the big channel matrix (with pass loss applied if any)
    def _get_big_H(self):
        """Get method for the big_H property."""
        if self._pathloss_matrix is None:
            # No path loss
            return self._big_H
        else:
            # Apply path loss. Note that the _pathloss_big_matrix matrix
            # has the same dimension as the self._big_H matrix and we are
            # performing element-wise multiplication here.
            return self._big_H * np.sqrt(self._pathloss_big_matrix)
    big_H = property(_get_big_H)

    # Property to get the pathloss. Use the "set_pathloss" method to set
    # the pathloss.
    def _get_pathloss(self):
        """Get method for the pathloss property."""
        return self._pathloss_matrix
    pathloss = property(_get_pathloss)

    def _get_last_noise(self):
        """Get method for the last_noise property."""
        return self._last_noise

    last_noise = property(_get_last_noise)

    def _get_last_noise_var(self):
        """Get method for the last_noise_var property."""
        return self._last_noise_var

    last_noise_var = property(_get_last_noise_var)

    @staticmethod
    def _from_small_matrix_to_big_matrix(small_matrix, Nr, Nt, Kr, Kt=None):
        """Convert from a small matrix to a big matrix by repeating elements
        according to the number of receive and transmit antennas.

        Parameters
        ----------
        small_matrix : 2D numpy array
            Any 2D numpy array
        Nr : 1D numpy array
            Number of antennas at each receiver.
        Nt : 1D numpy array
            Number of antennas at each transmitter.
        Kr : int
            Number of receivers to consider.
        Kt : int, optional (default to the value of Kr)
            Number of transmitters to consider.

        Returns
        -------
        big_matrix : 2D numpy array
            The converted matrix.

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
        >>> MultiUserChannelMatrix._from_small_matrix_to_big_matrix(small_matrix, Nr, Nt, K)
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
        big_matrix = np.ones([np.sum(Nr), np.sum(Nt)], dtype=small_matrix.dtype)

        for rx in range(Kr):
            for tx in range(Kt):
                big_matrix[cumNr[rx]:cumNr[rx + 1], cumNt[tx]:cumNt[tx + 1]] *= small_matrix[rx, tx]
        return big_matrix

    def init_from_channel_matrix(self, channel_matrix, Nr, Nt, K):
        """Initializes the multiuser channel matrix from the given
        `channel_matrix`.

        Parameters
        ----------
        channel_matrix : 2D numpy array
            A matrix concatenating the channel of all users (from each
            transmitter to each receiver).
        Nr : 1D numpy array
            Number of antennas at each receiver.
        Nt : 1D numpy array
            Number of antennas at each transmitter.
        K : int
            Number of transmit/receive pairs.

        Raises
        ------
        ValueError
            If the arguments are invalid.

        """
        if channel_matrix.shape != (np.sum(Nr), np.sum(Nt)):
            raise ValueError("Shape of the channel_matrix must be equal to the sum or receive antennas of all users times the sum of the receive antennas of all users.")

        if (Nt.size != K) or (Nr.size != K):
            raise ValueError("K must be equal to the number of elements in Nr and Nt")

        self._K = K
        self._Nr = Nr
        self._Nt = Nt

        self._big_H = channel_matrix

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Lets convert the full channel_matrix matrix to our internal
        # representation of H as a matrix of matrices.
        self._H = single_matrix_to_matrix_of_matrices(channel_matrix, Nr, Nt)

        # Assures that _big_H and _H will stay in sync by disallowing
        # modification of individual elements in both of them.
        self._big_H.setflags(write=False)
        self._H.setflags(write=False)

    def randomize(self, Nr, Nt, K):
        """Generates a random channel matrix for all users.

        Parameters
        ----------
        K : int
            Number of users.
        Nr : 1D numpy array or integers or a single integer
            Number of receive antennas of each user. If an integer is
            specified, all users will have that number of receive antennas.
        Nt : 1D numpy array or integers or a single integer
            Number of transmit antennas of each user. If an integer is
            specified, all users will have that number of receive antennas.
        """
        if isinstance(Nr, int):
            Nr = np.ones(K, dtype=int) * Nr
        if isinstance(Nt, int):
            Nt = np.ones(K, dtype=int) * Nt

        self._Nr = Nr.astype(int)
        self._Nt = Nt.astype(int)
        self._K = int(K)

        self._big_H = randn_c_RS(self._RS_channel,
                                 np.sum(self._Nr), np.sum(self._Nt))

        self._H = single_matrix_to_matrix_of_matrices(self._big_H, Nr, Nt)

        # Assures that _big_H and _H will stay in sync by disallowing
        # modification of individual elements in both of them.
        self._big_H.setflags(write=False)
        self._H.setflags(write=False)

    def get_channel(self, k, l):
        """Get the channel from user `l` to user `k`.

        Parameters
        ----------
        l : int
            Transmitting user.
        k : int
            Receiving user.

        Returns
        -------
        channel : 2D numpy array
            Channel from transmitter `l` to receiver `k`.

        See also
        --------
        get_channel_all_tx_to_rx_k

        Examples
        --------
        >>> multiH = MultiUserChannelMatrix()
        >>> H = np.reshape(np.r_[0:16], [4,4])
        >>> Nt = np.array([2, 2])
        >>> Nr = np.array([2, 2])
        >>> multiH.init_from_channel_matrix(H, Nr, Nt, 2)
        >>> print multiH.big_H
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]
         [12 13 14 15]]
        >>> print multiH.get_channel(0, 0)
        [[0 1]
         [4 5]]
        >>> print multiH.get_channel(1, 0)
        [[ 8  9]
         [12 13]]
        """
        channel = self.H  # This will call the _get_H method, which already
                          # applies the path loss (of there is any)
        return channel[k, l]

    def get_channel_all_tx_to_rx_k(self, k):
        """Get the channel from all transmitters to receiver `k`.

        Parameters
        ----------
        k : int
            Receiving user.

        Returns
        -------
        channel_k : 2D numpy array
            Channel from all transmitters to receiver `k`.

        See also
        --------
        get_channel

        Examples
        --------
        >>> multiH = MultiUserChannelMatrix()
        >>> H = np.reshape(np.r_[0:16], [4,4])
        >>> Nt = np.array([2, 2])
        >>> Nr = np.array([2, 2])
        >>> multiH.init_from_channel_matrix(H, Nr, Nt, 2)
        >>> print multiH.big_H
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]
         [12 13 14 15]]
        >>> print multiH.get_channel_all_tx_to_rx_k(0)
        [[0 1 2 3]
         [4 5 6 7]]
        >>> print multiH.get_channel_all_tx_to_rx_k(1)
        [[ 8  9 10 11]
         [12 13 14 15]]
        """
        receive_channels = single_matrix_to_matrix_of_matrices(self.big_H, self.Nr)
        return receive_channels[k]

    def set_post_filter(self, filters):
        """
        Set the post-processing filters.

        The post-processing filters will be applyied to the data after if
        has been currupted by the channel in either the `corrupt_data` or
        the `corrupt_concatenated_data` methods.

        Parameters
        ----------
        filters : List of 2D numpy arrays or a 1D numpy array of 2D numpy
                  arrays.
        """
        self._W = filters
        self._big_W = None  # This will be set in the get property only
                            # when required.

    def _get_W(self):
        """Get method for the post processing filter W."""
        return self._W
    W = property(_get_W)

    def _get_big_W(self):
        """Get method for the big_W property."""
        if self._big_W is None:
            if self.W is not None:
                self._big_W = block_diag(*self.W)
        return self._big_W
    big_W = property(_get_big_W)

    def corrupt_concatenated_data(self, data, noise_var=None):
        """Corrupt data passed through the channel.

        If the noise_var is supplied then white noise will also be added.

        Parameters
        ----------
        data : 2D numpy array
            A bi-dimensional numpy array with the concatenated data of all
            transmitters. The dimension of data is sum(self.Nt) x
            NSymb. That is, the number of rows corresponds to the sum of
            the number of transmit antennas of all users and the number of
            columns correspond to the number of transmitted symbols.
        noise_var : float, optional (default to None)
            Variance of the AWGN noise. If not provided, no noise will be
            added.

        Returns
        -------
        output : 2D numpy array
            A bi-dimension numpy array where the number of rows corresponds
            to the sum of the number of receive antennas of all users and
            the number of columns correspond to the number of transmitted
            symbols.

        """
        # Note that self.big_H already accounts the path loss (while
        # self._big_H does not)

        output = np.dot(self.big_H, data)
        if noise_var is not None:
            awgn_noise = (
                randn_c_RS(self._RS_noise, *output.shape) * np.sqrt(noise_var))
            output = output + awgn_noise
            self._last_noise = awgn_noise
            self._last_noise_var = noise_var
        else:
            self._last_noise = None
            self._last_noise_var = 0.0

        # Apply the post processing filter (if there is one set)
        if self.big_W is not None:
            output = np.dot(self.big_W.conjugate().T, output)

        return output

    def corrupt_data(self, data, noise_var=None):
        """Corrupt data passed through the channel.

        If the noise_var is supplied then an white noise will also be
        added.

        Parameters
        ----------
        data : 2D numpy array
            An array of numpy matrices with the data of the multiple
            users. The k-th element in `data` is a numpy array with
            dimension Nt_k x NSymbs, where Nt_k is the number of transmit
            antennas of the k-th user and NSymbs is the number of
            transmitted symbols.
        noise_var : float, optional (default to None)
            Variance of the AWGN noise. If not provided, no noise will be
            added.

        Returns
        -------
        output : 1D numpy array of 2D numpy arrays
            A numpy array where each element contais the received data (a
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
        concatenated_output = self.corrupt_concatenated_data(
            concatenated_data, noise_var)

        output = np.zeros(self.K, dtype=np.ndarray)
        cumNr = np.hstack([0, np.cumsum(self._Nr)])

        for k in np.arange(self.K):
            output[k] = concatenated_output[cumNr[k]:cumNr[k + 1], :]

        return output

    def set_pathloss(self, pathloss_matrix=None):
        """Set the path loss from each transmitter to each receiver.

        The path loss will be accounted when calling the get_channel, the
        corrupt_concatenated_data and the corrupt_data methods.

        If you want to disable the path loss, set `pathloss_matrix` to None.

        Parameters
        ----------
        pathloss_matrix : 2D numpy array
            A matrix with dimension "K x K", where K is the number of
            users, with the path loss from each transmitter (columns) to
            each receiver (rows). If you want to disable the path loss then
            set it to None.

        Notes
        -----
        Note that path loss is a power relation, which means that the
        channel coefficients will be multiplied by the square root of
        elements in `pathloss_matrix`.

        """
        # A matrix with the path loss from each transmitter to each
        # receiver.
        self._pathloss_matrix = pathloss_matrix

        if pathloss_matrix is None:
            self._pathloss_big_matrix = None
        else:
            self._pathloss_big_matrix = MultiUserChannelMatrix._from_small_matrix_to_big_matrix(pathloss_matrix, self._Nr, self._Nt, self._K)

            # Assures that _pathloss_matrix and _pathloss_big_matrix will stay
            # in sync by disallowing modification of individual elements in
            # both of them.
            self._pathloss_matrix.setflags(write=False)
            self._pathloss_big_matrix.setflags(write=False)


class MultiUserChannelMatrixExtInt(MultiUserChannelMatrix):
    """Very similar to the MultiUserChannelMatrix class, but the
    MultiUserChannelMatrixExtInt also includes the effect of an external
    interference.

    This channel matrix can be seem as an concatenation of blocks (of
    non-uniform size) where each block is a channel from one transmitter to
    one receiver and the block size is equal to the number of receive
    antennas of the receiver times the number of transmit antennas of the
    transmitter. The difference compared with the MultiUserChannelMatrix
    class is that in the MultiUserChannelMatrixExtInt class the
    interference user counts as one more user, but with zero receive
    antennas.

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

    .. aafig::
                                                 Ext. Int.
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

    def __init__(self):
        MultiUserChannelMatrix.__init__(self)
        self._extIntK = 0  # Number of external interference sources
        self._extIntNt = 0  # Number of transmit antennas of the external
                            # interference sources.

    # Property to get the number of external interference sources
    def _get_extIntK(self):
        """Get method for the extIntK property."""
        return self._extIntK
    extIntK = property(_get_extIntK)

    # Property to get the number of transmit antennas (or the rank) of the
    # external interference sources
    def _get_extIntNt(self):
        """Get method for the extIntNt property."""
        return self._extIntNt
    extIntNt = property(_get_extIntNt)

    # Property to get the number of receive antennas of all users. We
    # overwrite the property from the MultiUserChannelMatrix to avoid
    # account the number of receive antennas of the external interference
    # sources.
    def _get_Nr(self):
        """Get method for the Nr property."""
        return self._Nr[:-self._extIntK]
    Nr = property(_get_Nr)

    def _get_Nt(self):
        """Get method for the Nt property."""
        return self._Nt[:-self._extIntK]
    Nt = property(_get_Nt)

    def _get_K(self):
        """Get method for the K property."""
        return self._K - self._extIntK
    K = property(_get_K)

    def _get_big_H_no_ext_int(self):
        """Get method for the big_H_no_est_int property.

        big_H_no_est_int is similar to big_H, but does not include the last
        column(s) corresponding to the external interference channel.

        """
        return self.big_H[:, :np.sum(self.Nt)]
    big_H_no_ext_int = property(_get_big_H_no_ext_int)

    def _get_H(self):
        """Get method for the H property."""
        # Call the _get_H method from the base class, which will apply the
        # path loss if any.
        H = MultiUserChannelMatrix._get_H(self)
        # Now all we need to do is ignore the last _extIntK "lines" since
        # they correspond to the receive antennas of the interference
        # sources 'users' (which are in fact empty)
        return H[:-self._extIntK]
    H = property(_get_H)

    def corrupt_data(self, data, ext_int_data, noise_var=None):
        """Corrupt data passed through the channel.

        If the noise_var is supplied then an white noise will also be
        added.

        Parameters
        ----------
        data : 2D numpy array
            An array of numpy matrices with the data of the multiple
            users. The k-th element in `data` is a numpy array with
            dimension Nt_k x NSymbs, where Nt_k is the number of transmit
            antennas of the k-th user and NSymbs is the number of
            transmitted symbols.
        ext_int_data : 1D numpy array of 2D numpy arrays
            An array of numpy matrices with the data of the external
            interference sources. The l-th element is the data transmitted
            by the l-th external interference source, which must have a
            dimension of NtEl x NSymbs, where NtEl is the number of
            transmit antennas of the l-th external interference source.
        noise_var : float, optional (default to None)
            Variance of the AWGN noise. If not provided, no noise will be
            added.

        Returns
        -------
        output : 1D numpy array of 2D numpy arrays
            A numpy array where each element contais the received data (a
            2D numpy array) of a user.
        """
        input_data = np.hstack([data, ext_int_data])
        return MultiUserChannelMatrix.corrupt_data(self, input_data, noise_var)

    def corrupt_concatenated_data(self, data, noise_var=None):
        """Corrupt data passed through the channel.

        If the noise_var is supplied then an white noise will also be
        added.

        Parameters
        ----------
        data : 2D numpy array
            A bi-dimensional numpy array with the concatenated data of all
            transmitters as well as the data from all external interference
            sources. The dimension of data is (sum(self._Nt) +
            sum(self.extIntNt)) x NSymb. That is, the number of rows
            corresponds to the sum of the number of transmit antennas of
            all users and external interference sources and the number of
            columns correspond to the number of transmitted symbols.
        noise_var : float, optional (default to None)
            Variance of the AWGN noise. If not provided, no noise will be
            added.

        Returns
        -------
        output : 2D numpy array
            A bi-dimension numpy array where the number of rows corresponds
            to the sum of the number of receive antennas of all users and
            the number of columns correspond to the number of transmitted
            symbols.

        """
        return MultiUserChannelMatrix.corrupt_concatenated_data(self,
                                                                data,
                                                                noise_var)

    def get_channel_all_tx_to_rx_k(self, k):
        """Get the channel from all transmitters (without including the
        external interference sources) to receiver `k`.

        Parameters
        ----------
        k : int
            Receiving user.

        Returns
        -------
        channel_k : 2D numpy array
            Channel from all transmitters to receiver `k`.

        See also
        --------
        get_channel,
        get_channel_all_tx_with_extint_to_rx_k

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
        >>> print multiH.big_H
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]]
        >>> print multiH.get_channel_all_tx_to_rx_k(0)
        [[0 1 2 3]
         [5 6 7 8]]
        >>> print multiH.get_channel_all_tx_to_rx_k(1)
        [[10 11 12 13]
         [15 16 17 18]]

        """
        receive_channels = single_matrix_to_matrix_of_matrices(
            self.big_H[:, :np.sum(self.Nt)], self.Nr)
        return receive_channels[k]

    # This is exactly the same as the
    # get_channel_all_tx_to_rx_k method from the
    # MultiUserChannelMatrix class. therefore, we don't need to test it.
    def get_channel_all_tx_with_extint_to_rx_k(self, k):
        """Get the channel from all transmitters (including the external
        interference sources) to receiver `k`.

        Parameters
        ----------
        k : int
            Receiving user.

        Returns
        -------
        channel_k : 2D numpy array
            Channel from all transmitters to receiver `k`.

        See also
        --------
        get_channel,
        get_channel_all_tx_to_rx_k

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
        >>> print multiH.big_H
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]]
        >>> print multiH.get_channel_all_tx_with_extint_to_rx_k(0)
        [[0 1 2 3 4]
         [5 6 7 8 9]]
        >>> print multiH.get_channel_all_tx_with_extint_to_rx_k(1)
        [[10 11 12 13 14]
         [15 16 17 18 19]]

        """
        return MultiUserChannelMatrix.get_channel_all_tx_to_rx_k(self, k)  # pragma: no cover

    @staticmethod
    def _prepare_input_parans(Nr, Nt, K, NtE):
        """Helper method used in the init_from_channel_matrix and randomize
        method definitions.

        Parameters
        ----------
        Nr : 1D numpy array
            Number of antennas at each receiver.
        Nt : 1D numpy array
            Number of antennas at each transmitter.
        K : int
            Number of transmit/receive pairs.
        NtE : int or iterable
            Number of transmit antennas of the external interference
            source(s). If `NtE` is an iterable, the number of external
            interference sources will be the len(NtE).

        Returns
        -------
        output : a tuple
            The tuple (full_Nr, full_Nt, full_K, extIntK, extIntNt)

        """
        if isinstance(NtE, Iterable):
            # We have multiple external interference sources
            extIntK = len(NtE)
            extIntNt = np.array(NtE)
        else:
            # NtE is a scalar number, which means we have a single external
            # interference source.
            extIntK = 1
            extIntNt = np.array([NtE])

        # Number of receive antennas also including the number of receive
        # antennas of the interference users (which are zeros)
        full_Nr = np.hstack([Nr, np.zeros(extIntK, dtype=int)])
        # Number of transmit antennas also including the number of transmit
        # antennas of the interference users
        full_Nt = np.hstack([Nt, NtE])
        # Total number of users including the interference users.
        full_K = K + extIntK

        return (full_Nr, full_Nt, full_K, extIntK, extIntNt)

    def init_from_channel_matrix(self, channel_matrix, Nr, Nt, K, NtE):
        """Initializes the multiuser channel matrix from the given
        `channel_matrix`.

        Note that `channel_matrix` must also include the channel terms for
        the external interference, which must be the last `NtE` columns of
        `channel_matrix`. The number of rows in `channel_matrix` must be
        equal to np.sum(Nr), while the number of columns must be
        np.sum(Nt) + NtE.

        Parameters
        ----------
        channel_matrix : 2D numpy array
            A matrix concatenating the channel of all transmitters
            (including the external interference sources) to all receivers.
        Nr : 1D numpy array
            Number of antennas at each receiver.
        Nt : 1D numpy array
            Number of antennas at each transmitter.
        K : int
            Number of transmit/receive pairs.
        NtE : int or iterable
            Number of transmit antennas of the external interference
            source(s). If NtE is an iterable, the number of external
            interference sources will be the len(NtE).

        Raises
        ------
        ValueError
            If the arguments are invalid.

        """
        (full_Nr, full_Nt, full_K, extIntK, extIntNt) = MultiUserChannelMatrixExtInt._prepare_input_parans(Nr, Nt, K, NtE)

        self._extIntK = extIntK
        self._extIntNt = extIntNt

        MultiUserChannelMatrix.init_from_channel_matrix(
            self, channel_matrix, full_Nr, full_Nt, full_K)

    def randomize(self, Nr, Nt, K, NtE):
        """Generates a random channel matrix for all users as well as for the
        external interference source(s).

        Parameters
        ----------
        K : int
            Number of users.
        Nr : 1D array or an int
            Number of receive antennas of each user. If an integer is
            specified, all users will have that number of receive antennas.
        Nt : 1D array or an int
            Number of transmit antennas of each user. If an integer is
            specified, all users will have that number of receive antennas.
        NtE : 1D array or an int
            Number of transmit antennas of the external interference
            source(s). If NtE is an iterable, the number of external
            interference sources will be the len(NtE).

        """
        (full_Nr, full_Nt, full_K, extIntK, extIntNt) = MultiUserChannelMatrixExtInt._prepare_input_parans(Nr, Nt, K, NtE)

        self._extIntK = extIntK
        self._extIntNt = extIntNt

        MultiUserChannelMatrix.randomize(self, full_Nr, full_Nt, full_K)

    def set_pathloss(self, pathloss_matrix=None, ext_int_pathloss=None):
        """Set the path loss from each transmitter to each receiver, as well
        as the path loss from the external interference source(s) to each
        receiver.

        The path loss will be accounted when calling the get_channel, the
        corrupt_concatenated_data and the corrupt_data methods.

        Note that path loss is a power relation, which means that the
        channel coefficients will be multiplied by the square root of
        elements in `pathloss_matrix`.

        If you want to disable the path loss, set pathloss_matrix to None.

        Parameters
        ----------
        pathloss_matrix : 2D numpy array
            A matrix with dimension "K x K", where K is the number of
            users, with the path loss from each transmitter (columns) to
            each receiver (rows). If you want to disable the path loss then
            set it to None.
        ext_int_pathloss : 2D numpy array
            The path loss from each interference source to each
            receiver. The number of rows of ext_int_pathloss must be equal
            to the number of receives, while the number of columns must be
            equal to the number of external interference sources.

        """
        # A matrix with the path loss from each transmitter to each
        # receiver.
        self._pathloss_matrix = pathloss_matrix

        if pathloss_matrix is None:
            self._pathloss_matrix = None
            self._pathloss_big_matrix = None
        else:
            pathloss_matrix_with_ext_int = np.hstack([
                pathloss_matrix,
                ext_int_pathloss])
            self._pathloss_matrix = pathloss_matrix_with_ext_int

            self._pathloss_big_matrix = MultiUserChannelMatrix._from_small_matrix_to_big_matrix(pathloss_matrix_with_ext_int, self._Nr, self._Nt, self.K, self._K)

            # Assures that _pathloss_matrix and _pathloss_big_matrix will stay
            # in sync by disallowing modification of individual elements in
            # both of them.
            self._pathloss_matrix.setflags(write=False)
            self._pathloss_big_matrix.setflags(write=False)

    def calc_cov_matrix_extint_plus_noise(self, noise_var=0, pe=1):
        """Calculates the covariance matrix of the external interference
        plus noise.

        Parameters
        ----------
        noise_var : float, optional [default=0]
            Noise variance. If not specified, then only the covariance
            matrix of the external interference will be returned.
        pe : float, optional [default=1]
            External interference power (in linear scale)

        Returns
        -------
        R : 1D array of numpy matrices
            Return a numpy array, where each element is the covariance
            matrix of the external interference plus noise at one receiver.

        """
        R = np.empty(self.Nr.size, dtype=np.ndarray)
        cum_Nr = np.hstack([0, np.cumsum(self.Nr)])

        # import pudb
        # pudb.set_trace()
        for ii in range(self.Nr.size):
            extH = self.big_H[cum_Nr[ii]:cum_Nr[ii + 1], np.sum(self.Nt):]
            R[ii] = pe * np.dot(extH, extH.transpose().conjugate()) + np.eye(self.Nr[ii]) * noise_var
        return R
