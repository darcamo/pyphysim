#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with implementation of channel related classes"""

from collections import Iterable
import numpy as np
from util.conversion import single_matrix_to_matrix_of_matrices
from util.misc import randn_c


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
            Nr = np.ones(K) * Nr
        if isinstance(Nt, int):
            Nt = np.ones(K) * Nt

        self._Nr = Nr
        self._Nt = Nt
        self._K = K

        self._big_H = randn_c(np.sum(self._Nr), np.sum(self._Nt))

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
            awgn_noise = (randn_c(*output.shape) * np.sqrt(noise_var))
            output = output + awgn_noise
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

    def calc_cov_matrix_extint_plus_noise(self, noise_var=0):
        """Calculates the covariance matrix of the external interference
        plus noise.

        Parameters
        ----------
        noise_var : float, optional [default=0]
            Noise variance. If not specified, then only the covariance
            matrix of the external interference will be returned.

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
            R[ii] = np.dot(extH, extH.transpose().conjugate()) + np.eye(self.Nr[ii]) * noise_var
        return R
