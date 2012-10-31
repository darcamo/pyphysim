#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with implementation of channel related classes"""

import sys
sys.path.append("../")

import numpy as np
from util.misc import randn_c


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx MultiUserChannelMatrix Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MultiUserChannelMatrix(object):
    """Stores the (fast fading) channel matrix of a multi-user scenario.

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

    def __init__(self, ):
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
            return self._H * self._pathloss_matrix
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
            return self._big_H * self._pathloss_big_matrix
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
    def _from_small_matrix_to_big_matrix(small_matrix, Nr, Nt, K):
        """Convert from a small matrix to a big matrix by repeating elements
        according to the number of receive and transmit antennas.

        Arguments:
        - `small_matrix`:
        - `Nr`: A numpy array with the number of antennas at each receiver.
        - `Nt`: A numpy array with the number of antennas at each transmitter.
        - `K`: Number of transmit/receive pairs.

        Ex:
        >>> small_matrix = np.array([[1, 2], [3, 4]])
        >>> K=2
        >>> Nr = np.array([2, 3])
        >>> Nt = np.array([2, 2])
        >>> print small_matrix
        [[1 2]
         [3 4]]
        >>> print MultiUserChannelMatrix._from_small_matrix_to_big_matrix(small_matrix, Nr, Nt, K)
        [[1 1 2 2]
         [1 1 2 2]
         [3 3 4 4]
         [3 3 4 4]
         [3 3 4 4]]
        """
        cumNr = np.hstack([0, np.cumsum(Nr)])
        cumNt = np.hstack([0, np.cumsum(Nt)])
        big_matrix = np.ones([np.sum(Nr), np.sum(Nt)], dtype=small_matrix.dtype)
        for rx in range(K):
            for tx in range(K):
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

        if Nr.size != Nt.size:
            raise ValueError("K must be equal to the number of elements in Nr and Nt")
        if Nt.size != K:
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
        channel = self._H[k, l]
        # Apply path loss if it was set.
        if self._pathloss_matrix is not None:
            channel = channel * self._pathloss_matrix[k, l]
        return channel

    def corrupt_concatenated_data(self, concatenated_data, noise_var=None):
        """Corrupt data passed through the channel.

        If the noise_var is supplied then an white noise will also be
        added.

        Arguments:
        - `data`: A bi-dimensional numpy array with the concatenated data of
                  all transmitters. The dimension of concatenated_data is
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
        if self._pathloss_matrix is None:
            # No path loss
            channel_matrix = self._big_H
        else:
            # Apply path loss. Note that the _pathloss_big_matrix matrix
            # has the same dimension as the self._big_H matrix and we are
            # performing element-wise multiplication here.
            channel_matrix = self._big_H * self._pathloss_big_matrix

        output = np.dot(channel_matrix, concatenated_data)
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
        concatenated_data = np.vstack(data)
        concatenated_output = self.corrupt_concatenated_data(
            concatenated_data, noise_var)

        output = np.zeros(self._K, dtype=np.ndarray)
        cumNr = np.hstack([0, np.cumsum(self._Nr)])

        for k in np.arange(self._K):
            output[k] = concatenated_output[cumNr[k]:cumNr[k + 1], :]

        return output

    def set_pathloss(self, pathloss_matrix=None):
        """Set the path loss from each transmitter to each receiver.

        The path loss will be accounted when calling the getChannel, the
        corrupt_concatenated_data and the corrupt_data methods.

        If you want to disable the path loss, set pathloss_matrix to None.

        Arguments:
        - `pathloss_matrix`: A matrix with dimension "K x K", where K is
                             the number of users, with the path loss from
                             each transmitter (columns) to each receiver
                             (rows).

        """
        # A matrix with the path loss from each transmitter to each
        # receiver
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

# xxxxxxxxxx Main method xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    multiH = MultiUserChannelMatrix()
    H = np.array(
        [
            [0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
            [0, 0, 1, 1, 1, 2, 2, 2, 2, 2],
            [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
            [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
            [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
            [3, 3, 4, 4, 4, 5, 5, 5, 5, 5],
            [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
            [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
            [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
            [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
            [6, 6, 7, 7, 7, 8, 8, 8, 8, 8],
            [6, 6, 7, 7, 7, 8, 8, 8, 8, 8]])

    K = 3
    Nr = np.array([2, 4, 6])
    Nt = np.array([2, 3, 5])
    multiH.init_from_channel_matrix(H, Nr, Nt, K)
    # print multiH.Nr
    # print multiH.Nt
    # print multiH._big_H.shape
    pathloss = np.array([[1, 1.1, 1.2],
                         [1.3, 1.4, 1.5],
                         [1.6, 1.7, 1.8]])
    multiH.set_pathloss(pathloss)
    print multiH._big_H
    print multiH._pathloss_big_matrix


# xxxxx Perform the doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__1':
    # When this module is run as a script the doctests are executed
    import doctest
    doctest.testmod()
    print "{0} executed".format(__file__)
