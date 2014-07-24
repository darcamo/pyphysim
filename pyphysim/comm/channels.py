#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=E1103

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
# The "Number" class is a base class that we can use to check with
# isinstance to see if something is a number
from numbers import Number
import numpy as np
from scipy.linalg import block_diag
from ..util.conversion import single_matrix_to_matrix_of_matrices
from ..util.misc import randn_c_RS

__all__ = ['MultiUserChannelMatrix', 'MultiUserChannelMatrixExtInt',
           'JakesSampleGenerator', 'generate_jakes_samples']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def generate_jakes_samples(Fd, Ts=1e-3, NSamples=100, L=8, shape=None,
                           RS=None):
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

    obj = JakesSampleGenerator(Fd, Ts, L, shape, RS)
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
    # $h(t) = \frac{1}{\sqrt{L}}\sum_{l=0}^{L-1}\exp\{j[2\pi f_D \cos(\phi_l)t+\psi_l]\}$

    def __init__(self, Fd=100, Ts=1e-3, L=8,
                 shape=None, RS=None):
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
        self._current_time = 0.0

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
        t : Numpy Array
            The numpy array with the time samples. The shape of the
            generated time variable is "(1, A, NSamples)", where 'A' is has
            as many '1's as the length of self._shape.
            Ex: If self._shape is None then the shape of the returned 't'
            variable is (1, NSamples). If self._shape is (2,3) then the
            shape of the returned 't' variable is (1, 1, 1, NSamples)

        Notes
        -----
        Each time _generate_time_samples is called it will update
        self._current_time to reflect the advance of the time after
        generating the new samples.
        """
        # Generate a 1D numpy with the time samples
        t = np.arange(
            self._current_time,  # Start time
            NSamples * self.Ts + self._current_time,
            self.Ts * 1.0000000001)

        # Update the self._current_time variable with the value of the next
        # time sample that should be generated when _generate_time_samples
        # is called again.
        self._current_time = t[-1] + self.Ts

        # Now we will change the shape of the 't' variable to an
        # appropriated shape for later use.
        if self._shape is not None:
            # Ex: If self._shape is (2,3) then the shape of the generated
            # 't' variable should be (1,1,1,NSNSamples). The first
            # dimension correspond to the number of taps (that is, self.L),
            # the following two dimensions correspond to the dimensions in
            # self._shape, and the last dimension corresponds to the number
            # of time samples.
            #
            # Note that we use '1' for all dimensions except the last one
            # and numpy will replicate to the correct value later thanks to
            # broadcast.
            t.shape = [1] * (len(self._shape) + 1) + [int(NSamples)]
        else:
            # Since self._shape is None, we only need one dimension for the
            # taps (that is, self.L) and another dimension for the actual
            # time samples.
            #
            # Note that we use '1' for all dimensions except the last one
            # and numpy will replicate to the correct value later thanks to
            # broadcast.
            t.shape = (1, NSamples)

        return t

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
        h = (np.sqrt(1.0 / self.L) *
             np.sum(np.exp(1j * (2 * np.pi * self.Fd
                                 * np.cos(self._phi_l) * t + self._psi_l)),
                    axis=0))

        return h

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx MultiUserChannelMatrix Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TODO: Maybe remove the N0_or_Rek argument of the "*calc_Bkl*" methods and
# use the value of self.noise_var whenever possible.
class MultiUserChannelMatrix(object):  # pylint: disable=R0902
    """
    Stores the (fast fading) channel matrix of a multi-user scenario. The
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
    user `l`, call the `get_Hkl` method.
    """

    def __init__(self):
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
        self._big_H_with_pathloss = None
        self._H_with_pathloss = None

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
        self._noise_var = None  # Store the noise variance. If it is None,
                                # then no noise is added in the
                                # "corrupt_*data" methods.

        self._W = None  # Post processing filters (a list of 2D numpy
                        # arrays) for each user
        self._big_W = None  # Same as _W, but as a single block diagonal
                            # matrix.

    def set_channel_seed(self, seed=None):
        """
        Set the seed of the RandomState object used to generate the random
        elements of the channel (when self.randomize is called).

        Parameters
        ----------
        seed : Int or array like
            Random seed initializing the pseudo-random number
            generator. See np.random.RandomState help for more info.
        """
        self._RS_channel.seed(seed=seed)

    def set_noise_seed(self, seed=None):
        """
        Set the seed of the RandomState object used to generate the random
        noise elements (when the corrupt data function is called).

        Parameters
        ----------
        seed : Int or array like
            Random seed initializing the pseudo-random number
            generator. See np.random.RandomState help for more info.
        """
        self._RS_noise.seed(seed)

    def re_seed(self):  # pragma: no cover
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
    def Nr(self):
        """Get method for the Nr property."""
        return self._Nr

    # Property to get the number of transmit antennas
    @property
    def Nt(self):
        """Get method for the Nt property."""
        return self._Nt

    # Property to get the number of users
    @property
    def K(self):
        """Get method for the K property."""
        return self._K

    # Property to get the matrix of channel matrices (with pass loss
    # applied if any)
    def _get_H(self):
        """Get method for the H property."""
        if self._pathloss_matrix is None:
            # No path loss
            return self._H_no_pathloss
        else:
            if self._H_with_pathloss is None:
                # Apply path loss. Note that the _pathloss_big_matrix
                # matrix has the same dimension as the
                # self._big_H_no_pathloss matrix and we are performing
                # element-wise multiplication here.
                self._H_with_pathloss = self._H_no_pathloss * np.sqrt(
                    self._pathloss_matrix)
            return self._H_with_pathloss
    H = property(_get_H)

    # Property to get the big channel matrix (with pass loss applied if any)
    @property
    def big_H(self):
        """Get method for the big_H property."""
        if self._pathloss_matrix is None:
            # No path lossr
            return self._big_H_no_pathloss
        else:
            if self._big_H_with_pathloss is None:
                # Apply path loss. Note that the _pathloss_big_matrix
                # matrix has the same dimension as the
                # self._big_H_no_pathloss matrix and we are performing
                # element-wise multiplication here.
                self._big_H_with_pathloss = self._big_H_no_pathloss * np.sqrt(
                    self._pathloss_big_matrix)
            return self._big_H_with_pathloss

    # Property to get the pathloss. Use the "set_pathloss" method to set
    # the pathloss.
    @property
    def pathloss(self):
        """Get method for the pathloss property."""
        return self._pathloss_matrix

    @property
    def last_noise(self):
        """Get method for the last_noise property."""
        return self._last_noise

    @property
    def noise_var(self):
        """Get method for the noise_var property."""
        return self._noise_var

    @noise_var.setter
    def noise_var(self, value):
        """Set method for the noise_var property."""
        if value is not None:
            assert value >= 0.0, "Noise variance must be >= 0."
        self._noise_var = value

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
        big_matrix = np.ones([np.sum(Nr), np.sum(Nt)],
                             dtype=small_matrix.dtype)

        for rx in range(Kr):
            for tx in range(Kt):
                big_matrix[cumNr[rx]:cumNr[rx + 1], cumNt[tx]:cumNt[tx + 1]] \
                    *= small_matrix[rx, tx]
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
            msg = ("Shape of the channel_matrix must be equal to the sum or"
                   " receive antennas of all users times the sum of the "
                   "receive antennas of all users.")
            raise ValueError(msg)

        if (Nt.size != K) or (Nr.size != K):
            raise ValueError(
                "K must be equal to the number of elements in Nr and Nt")

        # Reset the _big_H_with_pathloss and _H_with_pathloss. They will be
        # correctly set the first time the _get_H or _get_big_H methods are
        # called.
        self._big_H_with_pathloss = None
        self._H_with_pathloss = None

        self._K = K
        self._Nr = Nr
        self._Nt = Nt

        self._big_H_no_pathloss = channel_matrix

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Lets convert the full channel_matrix matrix to our internal
        # representation of H as a matrix of matrices.
        self._H_no_pathloss = single_matrix_to_matrix_of_matrices(
            channel_matrix, Nr, Nt)

        # Assures that _big_H and _H will stay in sync by disallowing
        # modification of individual elements in both of them.
        self._big_H_no_pathloss.setflags(write=False)
        self._H_no_pathloss.setflags(write=False)

    def randomize(self, Nr, Nt, K):
        """Generates a random channel matrix for all users.

        Parameters
        ----------
        Nr : 1D numpy array or integers or a single integer
            Number of receive antennas of each user. If an integer is
            specified, all users will have that number of receive antennas.
        Nt : 1D numpy array or integers or a single integer
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

        self._big_H_no_pathloss = randn_c_RS(
            self._RS_channel, np.sum(self._Nr), np.sum(self._Nt))

        self._H_no_pathloss = single_matrix_to_matrix_of_matrices(
            self._big_H_no_pathloss, Nr, Nt)

        # Assures that _big_H and _H will stay in sync by disallowing
        # modification of individual elements in both of them.
        self._big_H_no_pathloss.setflags(write=False)
        self._H_no_pathloss.setflags(write=False)

    def get_Hkl(self, k, l):
        """Get the channel matrix from user `l` to user `k`.

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
        channel = self.H  # This will call the _get_H method, which already
                          # applies the path loss (if there is any)
        return channel[k, l]

    def get_Hk(self, k):
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

    def set_post_filter(self, filters):
        """
        Set the post-processing filters.

        The post-processing filters will be applyied to the data after if
        has been currupted by the channel in either the `corrupt_data` or
        the `corrupt_concatenated_data` methods.

        Parameters
        ----------
        filters : List of 2D np arrays or a 1D np array of 2D np arrays.
            The post processing filters of each user.
        """
        self._W = filters
        self._big_W = None  # This will be set in the get property only
                            # when required.

    @property
    def W(self):
        """Get method for the post processing filter W."""
        return self._W

    @property
    def big_W(self):
        """Get method for the big_W property."""
        if self._big_W is None:
            if self.W is not None:
                self._big_W = block_diag(*self.W)
        return self._big_W

    def corrupt_concatenated_data(self, data):
        """
        Corrupt data passed through the channel.

        If self.noise_var is set to some scalar number then white noise
        will also be added.

        Parameters
        ----------
        data : 2D numpy array
            A bi-dimensional numpy array with the concatenated data of all
            transmitters. The dimension of data is sum(self.Nt) x
            NSymb. That is, the number of rows corresponds to the sum of
            the number of transmit antennas of all users and the number of
            columns correspond to the number of transmitted symbols.

        Returns
        -------
        output : 2D numpy array
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
            awgn_noise = (
                randn_c_RS(self._RS_noise, *output.shape)
                * np.sqrt(self.noise_var))
            output = output + awgn_noise
            self._last_noise = awgn_noise
        else:
            self._last_noise = None

        # Apply the post processing filter (if there is one set)
        if self.big_W is not None:
            output = np.dot(self.big_W.conjugate().T, output)

        return output

    def corrupt_data(self, data):
        """
        Corrupt data passed through the channel.

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
            concatenated_data)

        output = np.zeros(self.K, dtype=np.ndarray)
        cumNr = np.hstack([0, np.cumsum(self._Nr)])

        for k in np.arange(self.K):
            output[k] = concatenated_output[cumNr[k]:cumNr[k + 1], :]

        return output

    def set_pathloss(self, pathloss_matrix=None):
        """
        Set the path loss (IN LINEAR SCALE) from each transmitter to each
        receiver.

        The path loss will be accounted when calling the get_Hkl, get_Hk, the
        corrupt_concatenated_data and the corrupt_data methods.

        If you want to disable the path loss, set `pathloss_matrix` to None.

        Parameters
        ----------
        pathloss_matrix : 2D numpy array
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
            self._pathloss_big_matrix \
                = MultiUserChannelMatrix._from_small_matrix_to_big_matrix(
                    pathloss_matrix, self._Nr, self._Nt, self._K)

            # Assures that _pathloss_matrix and _pathloss_big_matrix will stay
            # in sync by disallowing modification of individual elements in
            # both of them.
            self._pathloss_matrix.setflags(write=False)
            self._pathloss_big_matrix.setflags(write=False)

    def _calc_Q_impl(self, k, F_all_users):
        """
        Calculates the interference covariance matrix (without any noise) at
        the :math:`k`-th receiver.

        See the documentation of the calc_Q method.
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{kj} \mtF_j \mtF_j^H \mtH_{kj}^H$$
        interfering_users = set(range(self.K)) - {k}
        Qk = np.zeros([self.Nr[k], self.Nr[k]], dtype=complex)

        for l in interfering_users:
            Hkl_F = np.dot(
                self.get_Hkl(k, l),
                F_all_users[l])
            Qk = Qk + np.dot(Hkl_F, Hkl_F.transpose().conjugate())

        return Qk

    def calc_Q(self, k, F_all_users):
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
        F_all_users : 1D numpy array of 2D numpy array
            The precoder of all users (already taking into account the
            transmit power).

        Returns
        -------
        Qk : 2D numpy complex array.
            The interference covariance matrix at receiver :math:`k`.
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{kj} \mtF_j \mtF_j^H \mtH_{kj}^H + \sigma_n^2 \mtI_{N_k}$$
        Qk = self._calc_Q_impl(k, F_all_users)

        if self.noise_var is not None:
            # If self.noise_var is not None we add the covariance matrix of
            # the noise.
            Rnk = np.eye(self.Nr[k]) * self.noise_var
            Qk += Rnk

        return Qk

    def _calc_JP_Q_impl(self, k, F_all_users):
        """
        Calculates the interference covariance matrix (without any noise) at
        the :math:`k`-th receiver with a joint processing scheme.

        See the documentation of the calc_JP_Q method.
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{k} \mtF_j \mtF_j^H \mtH_{k}^H$$
        interfering_users = set(range(self.K)) - {k}
        Qk = np.zeros([self.Nr[k], self.Nr[k]], dtype=complex)

        for l in interfering_users:
            Hk_F = np.dot(
                self.get_Hk(k),
                F_all_users[l])
            Qk = Qk + np.dot(Hk_F, Hk_F.transpose().conjugate())

        return Qk

    def calc_JP_Q(self, k, F_all_users):
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
        F_all_users : 1D numpy array of 2D numpy array
            The precoder of all users (already taking into account the
            transmit power).

        Returns
        -------
        Qk : 2D numpy complex array.
            The interference covariance matrix at receiver :math:`k`.
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{k} \mtF_j \mtF_j^H \mtH_{k}^H + \sigma_n^2 \mtI_{N_k}$$
        Qk = self._calc_JP_Q_impl(k, F_all_users)

        if self.noise_var is not None:
            Rnk = np.eye(self.Nr[k]) * self.noise_var
            return Qk + Rnk
        else:
            return Qk

    def _calc_Bkl_cov_matrix_first_part(self, F_all_users, k, N0_or_Rek=0.0):
        """
        Calculates the first part in the equation of the Blk covariance matrix
        in equation (28) of [Cadambe2008]_.

        The first part is given by

            :math:`\\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star d}^{[j]} \\mtV_{\\star d}^{[j]\\dagger} \\mtH^{[kj]\\dagger} + \\mtI_{Nk}`

        Note that it only depends on the value of :math:`k`.

        Parameters
        ----------
        F_all_users : 1D numpy array of 2D numpy array
            The precoder of all users (already taking into account the
            transmit power).
        k : int
            Index of the desired user.
        N0_or_Rek : float or a 2D numpy array
            If this is a 2D numpy array, it is interpreted as the
            covariance matrix of any external interference plus noise. If
            this is a number, it is interpreted as the noise power, in
            which case the covariance matrix will be an identity matrix
            times this noise power.
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
            Rek = (noise_power * np.eye(self.Nr[k]))
        else:
            Rek = N0_or_Rek

        first_part = 0.0
        for j in range(self.K):
            Hkj = self.get_Hkl(k, j)
            Hkj_H = Hkj.conjugate().transpose()
            Vj = F_all_users[j]
            Vj_H = Vj.conjugate().transpose()

            first_part = first_part + np.dot(
                Hkj,
                np.dot(
                    np.dot(Vj,
                           Vj_H),
                    Hkj_H))
        first_part = first_part + Rek

        return first_part

    def _calc_Bkl_cov_matrix_second_part(self, Fk, k, l):
        """Calculates the second part in the equation of the Blk covariance
        matrix in equation (28) of [Cadambe2008]_ (note that it does not
        include the identity matrix).

        The second part is given by

            :math:`\\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger}`

        Parameters
        ----------
        Fk : 2D numpy array
            The precoder of the desired user.
        k : int
            Index of the desired user.
        l : int
            Index of the desired stream.

        Returns
        -------
        second_part : 2D numpy complex array.
            Second part in equation (28) of [Cadambe2008]_.

        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger}$$
        Hkk = self.get_Hkl(k, k)
        Hkk_H = Hkk.transpose().conjugate()

        Vkl = Fk[:, l:l + 1]
        Vkl_H = Vkl.transpose().conjugate()
        second_part = np.dot(Hkk,
                             np.dot(np.dot(Vkl, Vkl_H),
                                    Hkk_H))

        return second_part

    def _calc_Bkl_cov_matrix_all_l(self, F_all_users, k, N0_or_Rek=0.0):
        """Calculates the interference-plus-noise covariance matrix for all
        streams at receiver :math:`k` according to equation (28) in
        [Cadambe2008]_.

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
        F_all_users : 1D numpy array of 2D numpy array
            The precoder of all users (already taking into account the
            transmit power).
        k : int
            Index of the desired user.
        N0_or_Rek : float or a 2D numpy array
            If this is a 2D numpy array, it is interpreted as the
            covariance matrix of any external interference plus noise. If
            this is a number, it is interpreted as the noise power, in
            which case the covariance matrix will be an identity matrix
            times this noise power.

        Returns
        -------
        Bkl : 1D numpy array of 2D numpy arrays
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
        first_part = self._calc_Bkl_cov_matrix_first_part(
            F_all_users, k, N0_or_Rek)
        for l in range(Ns_k):
            second_part = self._calc_Bkl_cov_matrix_second_part(
                F_all_users[k], k, l)
            Bkl_all_l[l] = first_part - second_part

        return Bkl_all_l

    def _calc_JP_Bkl_cov_matrix_first_part_impl(self, HK, F_all_users, Rek):
        """
        Common implementation of the _calc_JP_Bkl_cov_matrix_first_part.

        Parameters
        ----------
        Hk : 2D numpy array
            The channel from all transmitters (not including external
            interference source, if any) to receiver k.
        F_all_users : 1D numpy array of 2D numpy array
            The precoder of all users (already taking into account the
            transmit power).
        k : int
            Index of the desired user.
        Rek : 2D numpy array
            Covariance matrix of the external interference (if there is
            any) plus noise.
        """
        # $$\sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[k]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[k]\dagger} + \mtR e_k$$
        first_part = 0.0

        HK_H = HK.conjugate().transpose()
        for j in range(self.K):
            Vj = F_all_users[j]
            Vj_H = Vj.conjugate().transpose()

            first_part = first_part + np.dot(
                HK,
                np.dot(
                    np.dot(Vj,
                           Vj_H),
                    HK_H))
        first_part = first_part + Rek

        return first_part

    def _calc_JP_Bkl_cov_matrix_first_part(self, F_all_users, k,
                                           noise_power=0.0):
        """
        Calculates the first part in the equation of the Blk covariance matrix
        in equation (28) of [Cadambe2008]_ when joint process is employed.

        The first part is given by

            :math:`\\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star d}^{[j]} \\mtV_{\\star d}^{[j]\\dagger} \\mtH^{[kj]\\dagger} + \\mtI_{Nk}`

        Note that it only depends on the value of :math:`k`.

        Parameters
        ----------
        F_all_users : 1D numpy array of 2D numpy array
            The precoder of all users (already taking into account the
            transmit power).
        k : int
            Index of the desired user.
        noise_power : float
            The noise power.
        """
        # The first part in Bkl is given by
        # $$\sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger} + \mtI_{N^{[k]}}$$
        # Note that here the power is already included in `Fk`.
        if noise_power is None:
            noise_power = 0.0

        Rek = (noise_power * np.eye(self.Nr[k]))
        Hk = self.get_Hk(k)
        return self._calc_JP_Bkl_cov_matrix_first_part_impl(
            Hk, F_all_users, Rek)

    @staticmethod
    def _calc_JP_Bkl_cov_matrix_second_part_impl(Hk, Fk, l):
        """
        Common implementation of the _calc_JP_Bkl_cov_matrix_second_part
        method.
        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[k]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[k]\dagger}$$
        Hk_H = Hk.transpose().conjugate()

        Vkl = Fk[:, l:l + 1]
        Vkl_H = Vkl.transpose().conjugate()
        second_part = np.dot(Hk,
                             np.dot(np.dot(Vkl, Vkl_H),
                                    Hk_H))

        return second_part

    def _calc_JP_Bkl_cov_matrix_second_part(self, Fk, k, l):
        """Calculates the second part in the equation of the Blk covariance
        matrix in equation (28) of [Cadambe2008]_ (note that it does not
        include the identity matrix).

        The second part is given by

            :math:`\\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger}`

        Parameters
        ----------
        Fk : 2D numpy array
            The precoder of the desired user.
        k : int
            Index of the desired user.
        l : int
            Index of the desired stream.

        Returns
        -------
        second_part : 2D numpy complex array.
            Second part in equation (28) of [Cadambe2008]_.

        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[k]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[k]\dagger}$$
        Hk = self.get_Hk(k)
        return self._calc_JP_Bkl_cov_matrix_second_part_impl(Hk, Fk, l)

    def _calc_JP_Bkl_cov_matrix_all_l(self, F_all_users, k, N0_or_Rek=0.0):
        """Calculates the interference-plus-noise covariance matrix for all
        streams at receiver :math:`k` according to equation (28) in
        [Cadambe2008]_.

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
        F_all_users : 1D numpy array of 2D numpy array
            The precoder of all users (already taking into account the
            transmit power).
        k : int
            Index of the desired user.
        N0_or_Rek : float or a 2D numpy array
            If this is a 2D numpy array, it is interpreted as the
            covariance matrix of any external interference plus noise. If
            this is a number, it is interpreted as the noise power, in
            which case the covariance matrix will be an identity matrix
            times this noise power.

        Returns
        -------
        Bkl : 1D numpy array of 2D numpy arrays
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

    def _calc_SINR_k(self, k, Fk, Uk, Bkl_all_l):
        """
        Calculates the SINR of all streams of user 'k'.

        Parameters
        ----------
        Fk : 2D numpy array
            The precoder of user k.
        Uk : 2D numpy array
            The receive filter of user k (before applying the conjugate
            transpose).
        k : int
            Index of the desired user.
        Bkl_all_l : A sequence of 2D numpy arrays.
            A sequence (1D numpy array, a list, etc) of 2D numpy arrays
            corresponding to the Bkl matrices for all 'l's.

        Returns
        -------
        SINR_k : 1D numpy array
            The SINR for the different streams of user k.
        """
        Ns_k = Fk.shape[1]

        SINR_k = np.empty(Ns_k, dtype=float)

        for l in range(Ns_k):
            Fkl = Fk[:, l:l + 1]
            Ukl = Uk[:, l:l + 1]
            Ukl_H = Ukl.conj().T

            aux = np.dot(Ukl_H,
                         np.dot(self.get_Hkl(k, k),
                                Fkl))
            numerator = np.dot(aux,
                               aux.transpose().conjugate())
            denominator = np.dot(Ukl_H,
                                 np.dot(Bkl_all_l[l], Ukl))
            SINR_kl = np.asscalar(numerator) / np.asscalar(denominator)
            SINR_k[l] = np.abs(SINR_kl)  # The imaginary part should be
                                         # negligible

        return SINR_k

    def calc_SINR(self, F, U):
        """
        Calculates the SINR values (in linear scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property.

        Parameters
        ----------
        F : 1D numpy array of 2D numpy arrays
            The precoders of all users.
        U : 1D numpy array of 2D numpy arrays
            The receive filters of all users.

        Returns
        -------
        SINRs : 1D numpy array of 1D numpy arrays (of floats)
            The SINR (in linear scale) of all streams of all users.
        """
        K = self.K
        SINRs = np.empty(K, dtype=np.ndarray)

        for k in range(self.K):
            Bkl_all_l = self._calc_Bkl_cov_matrix_all_l(F, k,
                                                        self.noise_var)
            SINRs[k] = self._calc_SINR_k(k, F[k], U[k], Bkl_all_l)
        return SINRs

    @staticmethod
    def _calc_JP_SINR_k_impl(Hk, Fk, Uk, Bkl_all_l):
        """
        Implementation of the :meth:`_calc_JP_SINR_k method`.

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

            aux = np.dot(Ukl_H,
                         np.dot(Hk, Fkl))
            numerator = np.dot(aux,
                               aux.transpose().conjugate())
            denominator = np.dot(Ukl_H,
                                 np.dot(Bkl_all_l[l], Ukl))
            SINR_kl = np.asscalar(numerator) / np.asscalar(denominator)
            SINR_k[l] = np.abs(SINR_kl)  # The imaginary part should be
                                         # negligible

        return SINR_k

    def _calc_JP_SINR_k(self, k, Fk, Uk, Bkl_all_l):
        """Calculates the SINR of all streams of user 'k'.

        Parameters
        ----------
        Fk : 2D numpy array
            The precoder of user k.
        Uk : 2D numpy array
            The receive filter of user k (before applying the conjugate
            transpose).
        k : int
            Index of the desired user.
        Bkl_all_l : A sequence of 2D numpy arrays.
            A sequence (1D numpy array, a list, etc) of 2D numpy arrays
            corresponding to the Bkl matrices for all 'l's.

        Returns
        -------
        SINR_k : 1D numpy array
            The SINR for the different streams of user k.

        """
        Hk = self.get_Hk(k)
        return self._calc_JP_SINR_k_impl(Hk, Fk, Uk, Bkl_all_l)

    def calc_JP_SINR(self, F, U):
        """
        Calculates the SINR values (in linear scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property.

        Parameters
        ----------
        F : 1D numpy array of 2D numpy arrays
            The precoders of all users.
        U : 1D numpy array of 2D numpy arrays
            The receive filters of all users.

        Returns
        -------
        SINRs : 1D numpy array of 1D numpy arrays (of floats)
            The SINR (in linear scale) of all streams of all users.
        """
        K = self.K
        SINRs = np.empty(K, dtype=np.ndarray)

        for k in range(self.K):
            Bkl_all_l = self._calc_JP_Bkl_cov_matrix_all_l(F, k,
                                                           self.noise_var)
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
    @property
    def extIntK(self):
        """Get method for the extIntK property."""
        return self._extIntK

    # Property to get the number of transmit antennas (or the rank) of the
    # external interference sources
    @property
    def extIntNt(self):
        """Get method for the extIntNt property."""
        return self._extIntNt

    # Property to get the number of receive antennas of all users. We
    # overwrite the property from the MultiUserChannelMatrix to avoid
    # account the number of receive antennas of the external interference
    # sources.
    @property
    def Nr(self):
        """Get method for the Nr property."""
        return self._Nr[:-self._extIntK]

    @property
    def Nt(self):
        """Get method for the Nt property."""
        return self._Nt[:-self._extIntK]

    @property
    def K(self):
        """Get method for the K property."""
        return self._K - self._extIntK

    @property
    def big_H_no_ext_int(self):
        """
        Get method for the big_H_no_est_int property.

        big_H_no_est_int is similar to big_H, but does not include the last
        column(s) corresponding to the external interference channel.
        """
        return self.big_H[:, :np.sum(self.Nt)]

    @property
    def H(self):
        """Get method for the H property."""
        # We only care about the first self.K "rows". The remaining rows
        # are the channels from all transmitters to the "external
        # interference user".
        H = self._H_no_pathloss[0:self.K]
        if self._pathloss_matrix is None:
            # No path loss
            return H
        else:
            # Apply path loss. Note that the _pathloss_big_matrix matrix
            # has the same dimension as the self._big_H_no_pathloss matrix
            # and we are performing element-wise multiplication here.
            return H * np.sqrt(self._pathloss_matrix)

    @property
    def H_no_ext_int(self):
        """Get method for the H_no_ext_int property."""
        H = MultiUserChannelMatrix._get_H(self)
        return H[:self.K, :self.K]

    def corrupt_data(self, data, ext_int_data):
        """
        Corrupt data passed through the channel.

        If the noise_var member variable is not None then an white noise
        will also be added.

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

        Returns
        -------
        output : 1D numpy array of 2D numpy arrays
            A numpy array where each element contais the received data (a
            2D numpy array) of a user.
        """
        input_data = np.hstack([data, ext_int_data])
        return MultiUserChannelMatrix.corrupt_data(self, input_data)

    def corrupt_concatenated_data(self, data):
        """
        Corrupt data passed through the channel.

        If the noise_var member variable is not None then an white noise
        will also be added.

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

        Returns
        -------
        output : 2D numpy array
            A bi-dimension numpy array where the number of rows corresponds
            to the sum of the number of receive antennas of all users and
            the number of columns correspond to the number of transmitted
            symbols.

        """
        return MultiUserChannelMatrix.corrupt_concatenated_data(self, data)

    def get_Hk_without_ext_int(self, k):
        """
        Get the channel from all transmitters (without including the external
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
        get_Hkl,
        get_Hk,
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
    def get_Hk_with_ext_int(self, k):
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
        channel_k : 2D numpy array
            Channel from all transmitters to receiver `k`.

        See also
        --------
        get_Hkl,
        get_Hk,
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
    def _prepare_input_parans(Nr, Nt, K, NtE):
        """
        Helper method used in the init_from_channel_matrix and randomize
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
        (full_Nr, full_Nt, full_K, extIntK, extIntNt) \
            = MultiUserChannelMatrixExtInt._prepare_input_parans(
                Nr, Nt, K, NtE)

        self._extIntK = extIntK
        self._extIntNt = extIntNt

        MultiUserChannelMatrix.init_from_channel_matrix(
            self, channel_matrix, full_Nr, full_Nt, full_K)

    def randomize(self, Nr, Nt, K, NtE):
        """
        Generates a random channel matrix for all users as well as for the
        external interference source(s).

        Parameters
        ----------
        Nr : 1D array or an int
            Number of receive antennas of each user. If an integer is
            specified, all users will have that number of receive antennas.
        Nt : 1D array or an int
            Number of transmit antennas of each user. If an integer is
            specified, all users will have that number of receive antennas.
        K : int
            Number of users.
        NtE : 1D array or an int
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

    def set_pathloss(self, pathloss_matrix=None, ext_int_pathloss=None):
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
        pathloss_matrix : 2D numpy array
            A matrix with dimension "K x K", where K is the number of
            users, with the path loss (IN LINEAR SCALE) from each
            transmitter (columns) to each receiver (rows). If you want to
            disable the path loss then set it to None.  ext_int_pathloss :
            2D numpy array The path loss from each interference source to
            each receiver. The number of rows of ext_int_pathloss must be
            equal to the number of receives, while the number of columns
            must be equal to the number of external interference sources.
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

            self._pathloss_big_matrix \
                = MultiUserChannelMatrix._from_small_matrix_to_big_matrix(
                    pathloss_matrix_with_ext_int, self._Nr, self._Nt,
                    self.K, self._K)

            # Assures that _pathloss_matrix and _pathloss_big_matrix will stay
            # in sync by disallowing modification of individual elements in
            # both of them.
            self._pathloss_matrix.setflags(write=False)
            self._pathloss_big_matrix.setflags(write=False)

    def calc_cov_matrix_extint_without_noise(self, pe=1):
        """
        Calculates the covariance matrix of the external interference without
        include the noise.

        Parameters
        ----------
        pe : float, optional [default=1]
            External interference power (in linear scale)

        Returns
        -------
        R_all_k : 1D array of numpy matrices
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

    def calc_cov_matrix_extint_plus_noise(self, pe=1):
        """
        Calculates the covariance matrix of the external interference plus
        noise.

        Parameters
        ----------
        pe : float, optional [default=1]
            External interference power (in linear scale)

        Returns
        -------
        R_all_k : 1D array of numpy matrices
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
            for i in range(len(R_all_k)):
                R_all_k[i] += np.eye(self.Nr[i]) * noise_var

        return R_all_k

    def calc_Q(self, k, F_all_users, pe=1.0):
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
        F_all_users : 1D numpy array of 2D numpy array
            The precoder of all users (already taking into account the
            transmit power).
        pe : float
            The power of the external interference source(s).

        Returns
        -------
        Qk : 2D numpy complex array.
            The interference covariance matrix at receiver :math:`k`.
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{kj} \mtF_j \mtF_j^H \mtH_{kj}^H + \mtR_e$$
        Rek_all_k = self.calc_cov_matrix_extint_plus_noise(pe)
        Qk = self._calc_Q_impl(k, F_all_users) + Rek_all_k[k]

        return Qk

    def _calc_JP_Q(self, k, F_all_users):
        """
        Calculates the interference covariance matrix at the :math:`k`-th
        receiver with a joint processing scheme (not including the
        covariance matrix of the external interference plus noise)

        Parameters
        ----------
        k : int
            Index of the desired receiver.
        F_all_users : 1D numpy array of 2D numpy array
            The precoder of all users (already taking into account the
            transmit power).

        See also
        --------
        calc_JP_Q
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{k} \mtF_j \mtF_j^H \mtH_{k}^H$$
        interfering_users = set(range(self.K)) - {k}
        Qk = np.zeros([self.Nr[k], self.Nr[k]], dtype=complex)

        for l in interfering_users:
            Hk_F = np.dot(
                self.get_Hk_without_ext_int(k),
                F_all_users[l])
            Qk = Qk + np.dot(Hk_F, Hk_F.transpose().conjugate())

        return Qk

    def calc_JP_Q(self, k, F_all_users, pe=1.0):
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
        F_all_users : 1D numpy array of 2D numpy array
            The precoder of all users (already taking into account the
            transmit power).
        pe : float
            The power of the external interference source(s).

        Returns
        -------
        Qk : 2D numpy complex array.
            The interference covariance matrix at receiver :math:`k`.
        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{k} \mtF_j \mtF_j^H \mtH_{k}^H + \mtR_e$$
        Rek_all_k = self.calc_cov_matrix_extint_plus_noise(pe)
        Qk = self._calc_JP_Q(k, F_all_users) + Rek_all_k[k]

        return Qk

    def calc_SINR(self, F, U, pe=1.0):
        """
        Calculates the SINR values (in linear scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property.

        Parameters
        ----------
        F : 1D numpy array of 2D numpy arrays
            The precoders of all users.
        U : 1D numpy array of 2D numpy arrays
            The receive filters of all users.
        pe : float
            Power of the external interference source.

        Returns
        -------
        SINRs : 1D numpy array of 1D numpy arrays (of floats)
            The SINR (in linear scale) of all streams of all users.
        """
        K = self.K
        SINRs = np.empty(K, dtype=np.ndarray)

        Re_all_k = self.calc_cov_matrix_extint_plus_noise(pe)

        for k in range(self.K):
            Bkl_all_l = self._calc_Bkl_cov_matrix_all_l(F, k, Re_all_k[k])
            SINRs[k] = self._calc_SINR_k(k, F[k], U[k], Bkl_all_l)
        return SINRs

    # pylint: disable=W0222
    def _calc_JP_Bkl_cov_matrix_first_part(self, F_all_users, k, Rek):
        """
        Calculates the first part in the equation of the Blk covariance matrix
        in equation (28) of [Cadambe2008]_ when joint process is employed.

        The first part is given by

            :math:`\\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star d}^{[j]} \\mtV_{\\star d}^{[j]\\dagger} \\mtH^{[kj]\\dagger} + \\mtI_{Nk}`

        Note that it only depends on the value of :math:`k`.

        Parameters
        ----------
        F_all_users : 1D numpy array of 2D numpy array
            The precoder of all users (already taking into account the
            transmit power).
        k : int
            Index of the desired user.
        Rek : 2D numpy array
            Covariance matrix of the external interference plus noise.
        """
        # The first part in Bkl is given by
        # $$\sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[k]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[k]\dagger} + \mtR e_k$$
        # Note that here the power is already included in `Fk`.

        Hk = self.get_Hk_without_ext_int(k)
        return self._calc_JP_Bkl_cov_matrix_first_part_impl(
            Hk, F_all_users, Rek)

    def _calc_JP_Bkl_cov_matrix_second_part(self, Fk, k, l):
        """
        Calculates the second part in the equation of the Blk covariance
        matrix in equation (28) of [Cadambe2008]_ (note that it does not
        include the identity matrix).

        The second part is given by

            :math:`\\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger}`

        Parameters
        ----------
        Fk : 2D numpy array
            The precoder of the desired user.
        k : int
            Index of the desired user.
        l : int
            Index of the desired stream.

        Returns
        -------
        second_part : 2D numpy complex array.
            Second part in equation (28) of [Cadambe2008]_.

        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[k]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[k]\dagger}$$
        Hk = self.get_Hk_without_ext_int(k)
        return self._calc_JP_Bkl_cov_matrix_second_part_impl(Hk, Fk, l)

    def _calc_JP_SINR_k(self, k, Fk, Uk, Bkl_all_l):
        """
        Calculates the SINR of all streams of user 'k'.

        Parameters
        ----------
        Fk : 2D numpy array
            The precoder of user k.
        Uk : 2D numpy array
            The receive filter of user k (before applying the conjugate
            transpose).
        k : int
            Index of the desired user.
        Bkl_all_l : A sequence of 2D numpy arrays.
            A sequence (1D numpy array, a list, etc) of 2D numpy arrays
            corresponding to the Bkl matrices for all 'l's.

        Returns
        -------
        SINR_k : 1D numpy array
            The SINR for the different streams of user k.

        """
        Hk = self.get_Hk_without_ext_int(k)
        return self._calc_JP_SINR_k_impl(Hk, Fk, Uk, Bkl_all_l)

    def calc_JP_SINR(self, F, U, pe=1.0):
        """
        Calculates the SINR values (in linear scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property.

        Parameters
        ----------
        F : 1D numpy array of 2D numpy arrays
            The precoders of all users.
        U : 1D numpy array of 2D numpy arrays
            The receive filters of all users.

        Returns
        -------
        SINRs : 1D numpy array of 1D numpy arrays (of floats)
            The SINR (in linear scale) of all streams of all users.
        """
        K = self.K
        SINRs = np.empty(K, dtype=np.ndarray)

        Re_all_k = self.calc_cov_matrix_extint_plus_noise(pe)

        for k in range(self.K):
            Bkl_all_l = self._calc_JP_Bkl_cov_matrix_all_l(F, k, Re_all_k[k])
            SINRs[k] = self._calc_JP_SINR_k(k, F[k], U[k], Bkl_all_l)
        return SINRs
