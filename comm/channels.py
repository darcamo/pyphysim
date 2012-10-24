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
        self.H = np.array([], dtype=np.ndarray)
        self.Nr = np.array([])
        self.Nt = np.array([])
        self.K = 0

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
                output[rx, tx] = \
                big_matrix[cumNr[rx]:cumNr[rx + 1], cumNt[tx]:cumNt[tx + 1]]

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

        self.K = K
        self.Nr = Nr
        self.Nt = Nt

        self._H = channel_matrix
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Lets convert the full channel_matrix matrix to our internal
        # representation of H as a matrix of matrices.
        self.H = MultiUserChannelMatrix._from_big_matrix_to_matrix_of_matrices(
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

        self.Nr = Nr
        self.Nt = Nt
        self.K = K

        self._H = randn_c(np.sum(self.Nr), np.sum(self.Nt))
        self.H = MultiUserChannelMatrix._from_big_matrix_to_matrix_of_matrices(
            self._H, Nr, Nt, K)

        # self.H = np.zeros([self.K, self.K], dtype=np.ndarray)
        # for rx in np.arange(self.K):
        #     for tx in np.arange(self.K):
        #         self.H[rx, tx] = randn_c(Nr[rx], Nt[tx])

    def getChannel(self, k, l):
        """Get the channel from user l to user k.

        Arguments:
        - `l`: Transmitting user.
        - `k`: Receiving user
        """
        return self.H[k, l]

    def corrupt_concatenated_data(self, concatenated_data, noise_var=None):
        """Corrupt data passed through the channel.

        If the noise_var is supplied then an white noise will also be
        added.

        Arguments:
        - `data`: A bi-dimensional numpy array with the concatenated data of
                  all transmitters. The dimension of concatenated_data is
                  sum(self.Nt) x NSymb. That is, the number of rows
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
        output = np.dot(self._H, concatenated_data)
        if noise_var != None:
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

        output = np.zeros(self.K, dtype=np.ndarray)
        cumNr = np.hstack([0, np.cumsum(self.Nr)])

        for k in np.arange(self.K):
            output[k] = concatenated_output[cumNr[k]:cumNr[k + 1], :]

        return output
