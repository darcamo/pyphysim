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
    user `l`, call the `getChannel` method.
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

    # Property to get the matrix of channel matrices
    def _get_H(self):
        return self._H
    H = property(_get_H)

    @staticmethod
    def _from_big_matrix_to_matrix_of_matrices(big_matrix, Nr, Nt, K):
        """Convert from a big matrix that concatenates the channel of all
        users (from each transmitter to each receiver) to the matrix of
        matrices representation.

        Arguments:
        - `big_matrix`:
        - `Nr`:
        - `Nt`:
        - `K`:
        """
        cumNr = np.hstack([0, np.cumsum(Nr)])
        cumNt = np.hstack([0, np.cumsum(Nt)])

        output = np.zeros([K, K], dtype=np.ndarray)

        for rx in np.arange(K):
            for tx in np.arange(K):
                output[rx, tx] = big_matrix[
                    cumNr[rx]:cumNr[rx + 1], cumNt[tx]:cumNt[tx + 1]]

        return output

    def init_from_channel_matrix(self, channel_matrix, Nr, Nt, K):
        """Initializes the multiuser channel matrix from the given
        `channel_matrix`.

        Arguments:
        - `channel_matrix`: A matrix concatenating the channel of all users
                            (from each transmitter to each receiver).
        - `Nr`: An array with the number of receive antennas of each user.
        - `Nt`: An array with the number of transmit antennas of each user.
        - `K`: (int) Number of users.

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
        self._H = MultiUserChannelMatrix._from_big_matrix_to_matrix_of_matrices(
            channel_matrix, Nr, Nt, K)

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
        self._H = MultiUserChannelMatrix._from_big_matrix_to_matrix_of_matrices(
            self._big_H, Nr, Nt, K)

    def getChannel(self, k, l):
        """Get the channel from user l to user k.

        Arguments:
        - `l`: Transmitting user.
        - `k`: Receiving user
        """
        return self._H[k, l]

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
        output = np.dot(self._big_H, concatenated_data)
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

    def set_pathloss(self, pathloss_matrix):
        """Set the path loss from each transmitter to each receiver.

        This path loss will be accounted when calling the getChannel, the
        corrupt_concatenated_data and the corrupt_data methods.

        Arguments:
        - `pathloss_matrix`: A matrix with dimension "K x K", where K is
                             the number of users, with the path loss from
                             each transmitter (columns) to each receiver
                             (rows).

        """
        self.pathloss_matrix = pathloss_matrix


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
    print multiH.Nr
    print multiH.Nt
    print multiH._big_H.shape
