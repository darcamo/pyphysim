#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with implementation of channel related classes"""


from collections import Iterable
import numpy as np
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
                +------+---------+---------------+
                |2 x 2 |  2 x 3  |     2 x 5     |
                |      |         |               |
                +------+---------+---------------+
                |4 x 2 |  4 x 3  |     4 x 5     |
                |      |         |               |
                |      |         |               |
                |      |         |               |
                +------+---------+---------------+
                |6 x 2 |  6 x 3  |     6 x 5     |
                |      |         |               |
                |      |         |               |
                |      |         |               |
                |      |         |               |
                |      |         |               |
                +------+---------+---------------+

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
        return self._Nr
    Nr = property(_get_Nr)

    # Property to get the number of transmit antennas
    def _get_Nt(self):
        return self._Nt
    Nt = property(_get_Nt)

    # Property to get the number of users
    def _get_K(self):
        return self._K
    K = property(_get_K)

    # Property to get the matrix of channel matrices (with pass loss
    # applied if any)
    def _get_H(self):
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
        return self._pathloss_matrix
    pathloss = property(_get_pathloss)

    @staticmethod
    def _from_big_matrix_to_matrix_of_matrices(big_matrix, Nr, Nt, K):
        """Convert from a big matrix that concatenates the channel of all
        users (from each transmitter to each receiver) to the matrix of
        matrices representation.

        Arguments:
        - `big_matrix`:
        - `Nr`: A numpy array with the number of antennas at each receiver.
        - `Nt`: A numpy array with the number of antennas at each transmitter.
        - `K`: Number of transmit/receive pairs.
        """
        cumNr = np.hstack([0, np.cumsum(Nr)])
        cumNt = np.hstack([0, np.cumsum(Nt)])

        output = np.zeros([K, K], dtype=np.ndarray)

        for rx in np.arange(K):
            for tx in np.arange(K):
                output[rx, tx] = big_matrix[
                    cumNr[rx]:cumNr[rx + 1], cumNt[tx]:cumNt[tx + 1]]

        return output

    @staticmethod
    def _from_small_matrix_to_big_matrix(small_matrix, Nr, Nt, Kr, Kt=None):
        """Convert from a small matrix to a big matrix by repeating elements
        according to the number of receive and transmit antennas.

        Note: Since a 'user' is a transmit/receive pair then the
        small_matrix will be a square matrix and Kr must be equal to Kt.
        However, in the MultiUserChannelMatrixExtInt class we will only
        have the 'transmitter part' for the external interference
        sources. That means that small_matrix will have more columns then
        rows and Kt will be greater then Kr.

        Arguments:
        - `small_matrix`:
        - `Nr`: A numpy array with the number of antennas at each receiver.
        - `Nt`: A numpy array with the number of antennas at each transmitter.
        - `Kr`: Number of receivers to consider.
        - `Kt`: Number of transmitters to consider (if not provided Kr will
                be used).

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

        Arguments:
        - `channel_matrix`: A matrix concatenating the channel of all users
                            (from each transmitter to each receiver).
        - `Nr`: A numpy array with the number of antennas at each receiver.
        - `Nt`: A numpy array with the number of antennas at each transmitter.
        - `K`: Number of transmit/receive pairs.

        Raises: ValueError if the arguments are invalid.
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
        self._H = MultiUserChannelMatrix._from_big_matrix_to_matrix_of_matrices(channel_matrix, Nr, Nt, K)

        # Assures that _big_H and _H will stay in sync by disallowing
        # modification of individual elements in both of them.
        self._big_H.setflags(write=False)
        self._H.setflags(write=False)

    def randomize(self, Nr, Nt, K):
        """Generates a random channel matrix for all users.

        Arguments:
        - `K`: (int) Number of users.
        - `Nr`: (array or int) Number of receive antennas of each user. If
                an integer is specified, all users will have that number of
                receive antennas.
        - `Nt`: (array or int) Number of transmit antennas of each user. If
                an integer is specified, all users will have that number of
                receive antennas.
        """
        if isinstance(Nr, int):
            Nr = np.ones(K) * Nr
        if isinstance(Nt, int):
            Nt = np.ones(K) * Nt

        self._Nr = Nr
        self._Nt = Nt
        self._K = K

        self._big_H = randn_c(np.sum(self._Nr), np.sum(self._Nt))
        self._H = MultiUserChannelMatrix._from_big_matrix_to_matrix_of_matrices(self._big_H, Nr, Nt, K)

        # Assures that _big_H and _H will stay in sync by disallowing
        # modification of individual elements in both of them.
        self._big_H.setflags(write=False)
        self._H.setflags(write=False)

    def get_channel(self, k, l):
        """Get the channel from user l to user k.

        Arguments:
        - `l`: Transmitting user.
        - `k`: Receiving user
        """
        channel = self.H  # This will call the _get_H method, which already
                          # applies the path loss (of there is any)
        return channel[k, l]

    def corrupt_concatenated_data(self, data, noise_var=None):
        """Corrupt data passed through the channel.

        If the noise_var is supplied then an white noise will also be
        added.

        Arguments:
        - `data`: A bi-dimensional numpy array with the concatenated data of
                  all transmitters. The dimension of data is
                  sum(self._Nt) x NSymb. That is, the number of rows
                  corresponds to the sum of the number of transmit antennas
                  of all users and the number of columns correspond to the
                  number of transmitted symbols.
        - `noise_var`: Variance of the AWGN noise.
        Output:
        - A bi-dimension numpy array where the number of rows corresponds
          to the sum of the number of receive antennas of all users and the
          number of columns correspond to the number of transmitted
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

        Arguments:
        - `data`: An array of numpy matrices with the data of the multiple
                  users. The k-th element in `data` is a numpy array with
                  dimension Nt_k x NSymbs, where Nt_k is the number of
                  transmit antennas of the k-th user and NSymbs is the
                  number of transmitted symbols.
        - `noise_var`: Variance of the AWGN noise.
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

        Note that path loss is a power relation, which means that the
        channel coefficients will be multiplied by the square root of
        elements in `pathloss_matrix`.

        If you want to disable the path loss, set pathloss_matrix to None.

        Arguments:
        - `pathloss_matrix`: A matrix with dimension "K x K", where K is
                             the number of users, with the path loss from
                             each transmitter (columns) to each receiver
                             (rows). If you want to disable the path loss
                             then set it to None.

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
                                                 Ext. Int.
                +------+---------+---------------+------+
                |2 x 2 |  2 x 3  |     2 x 5     |2 x 2 |
                |      |         |               |      |
                +------+---------+---------------+------+
                |4 x 2 |  4 x 3  |     4 x 5     |4 x 2 |
                |      |         |               |      |
                |      |         |               |      |
                |      |         |               |      |
                +------+---------+---------------+------+
                |6 x 2 |  6 x 3  |     6 x 5     |6 x 2 |
                |      |         |               |      |
                |      |         |               |      |
                |      |         |               |      |
                |      |         |               |      |
                |      |         |               |      |
                +------+---------+---------------+------+

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
        return self._extIntK
    extIntK = property(_get_extIntK)

    # Property to get the number of transmit antennas (or the rank) of the
    # external interference sources
    def _get_extIntNt(self):
        return self._extIntNt
    extIntNt = property(_get_extIntNt)

    # Property to get the number of receive antennas of all users. We
    # overwrite the property from the MultiUserChannelMatrix to avoid
    # account the number of receive antennas of the external interference
    # sources.
    def _get_Nr(self):
        return self._Nr[:-self._extIntK]
    Nr = property(_get_Nr)

    def _get_Nt(self):
        return self._Nt[:-self._extIntK]
    Nt = property(_get_Nt)

    def _get_K(self):
        return self._K - self._extIntK
    K = property(_get_K)

    def _get_H(self):
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

        Arguments:
        - `data`: An array of numpy matrices with the data of the multiple
                  users. The k-th element in `data` is a numpy array with
                  dimension Nt_k x NSymbs, where Nt_k is the number of
                  transmit antennas of the k-th user and NSymbs is the
                  number of transmitted symbols.
        - `ext_int_data`: An array of numpy matrices with the data of the
                          external interference sources. The l-th element
                          is the data transmitted by the l-th external
                          interference source, which must have a dimension
                          of NtEl x NSymbs, where NtEl is the number of
                          transmit antennas of the l-th external
                          interference source.
        - `noise_var`: Variance of the AWGN noise.

        """
        input_data = np.hstack([data, ext_int_data])
        return MultiUserChannelMatrix.corrupt_data(self, input_data, noise_var)

    def corrupt_concatenated_data(self, data, noise_var=None):
        """Corrupt data passed through the channel.

        If the noise_var is supplied then an white noise will also be
        added.

        Arguments:
        - `data`: A bi-dimensional numpy array with the concatenated data
                  of all transmitters as well as the data from all external
                  interference sources. The dimension of data is
                  (sum(self._Nt) + sum(self.extIntNt)) x NSymb. That is,
                  the number of rows corresponds to the sum of the number
                  of transmit antennas of all users and external
                  interference sources and the number of columns correspond
                  to the number of transmitted symbols.
        - `noise_var`: Variance of the AWGN noise.
        Output:
        - A bi-dimension numpy array where the number of rows corresponds
          to the sum of the number of receive antennas of all users and the
          number of columns correspond to the number of transmitted
          symbols.

        """

        return MultiUserChannelMatrix.corrupt_concatenated_data(self,
                                                                data,
                                                                noise_var)

    @staticmethod
    def _prepare_input_parans(Nr, Nt, K, NtE):
        """Helper method used in the init_from_channel_matrix and randomize
        method definitions.

        Arguments:
        - `Nr`: A numpy array with the number of antennas at each receiver.
        - `Nt`: A numpy array with the number of antennas at each transmitter.
        - `K`: Number of transmit/receive pairs.
        - `NtE`: (int, or iterable) Number of transmit antennas of the
                 external interference source(s). If NtE is an iterable,
                 the number of external interference sources will be the
                 len(NtE).
        Returns:
        - The tuple (full_Nr, full_Nt, full_K, extIntK, extIntNt)
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

        Arguments:
        - `channel_matrix`: A matrix concatenating the channel of all users
                            (from each transmitter to each receiver).
        - `Nr`: A numpy array with the number of antennas at each receiver.
        - `Nt`: A numpy array with the number of antennas at each transmitter.
        - `K`: Number of transmit/receive pairs.
        - `NtE`: (int, or iterable) Number of transmit antennas of the
                 external interference source(s). If NtE is an iterable,
                 the number of external interference sources will be the
                 len(NtE).

        Raises: ValueError if the arguments are invalid.

        """
        (full_Nr, full_Nt, full_K, extIntK, extIntNt) = MultiUserChannelMatrixExtInt._prepare_input_parans(Nr, Nt, K, NtE)

        self._extIntK = extIntK
        self._extIntNt = extIntNt

        MultiUserChannelMatrix.init_from_channel_matrix(
            self, channel_matrix, full_Nr, full_Nt, full_K)

    def randomize(self, Nr, Nt, K, NtE):
        """Generates a random channel matrix for all users as well as for the
        external interference source(s).

        Arguments:
        - `K`: (int) Number of users.
        - `Nr`: (array or int) Number of receive antennas of each user. If
                an integer is specified, all users will have that number of
                receive antennas.
        - `Nt`: (array or int) Number of transmit antennas of each user. If
                an integer is specified, all users will have that number of
                receive antennas.
        - `NtE`: (int, or iterable) Number of transmit antennas of the
                 external interference source(s). If NtE is an iterable,
                 the number of external interference sources will be the
                 len(NtE).
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

        Arguments:
        - `pathloss_matrix`: A matrix with dimension "K x K", where K is
                             the number of users, with the path loss from
                             each transmitter (columns) to each receiver
                             (rows). If you want to disable the path loss
                             then set it to None.
        - `ext_int_pathloss`:
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
