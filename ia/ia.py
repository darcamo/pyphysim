#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Module with implementation of Interference Alignment algorithms"""

import numpy as np
import itertools

from misc import peig, leig, randn_c


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


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx AlternatingMinIASolver Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class AlternatingMinIASolver(object):
    """Implements the "Interference Alignment via % Alternating
    Minimization" algorithm from the paper with the same name.

    Read only variables:
    - K
    - Nt
    - Nr
    - Ns
    """
    def __init__(self):
        """
        """
        # The F and W variables will be numpy arrays OF numpy arrays.
        self.F = np.array([])  # Precoder: One precoder for each user
        self.W = np.array([])  # Receive filter: One for each user
        self._multiUserChannel = MultiUserChannelMatrix()  # Channel of all users
        self.C = []    # Basis of the interference subspace for each user

        # xxxxxxxxxx Private attributes xxxxxxxxxxxxxxx
        self._Ns = 0    # Number of streams per user

    # xxxxx Properties to read the channel related variables xxxxxxxxxxxxxx
    @property
    def K(self):
        """The number of users.
        """
        return self._multiUserChannel.K

    @property
    def Nr(self):
        """Number of receive antennas of all users.
        """
        return self._multiUserChannel.Nr

    @property
    def Nt(self):
        """Number of transmit antennas of all users.
        """
        return self._multiUserChannel.Nt

    @property
    def Ns(self):
        """Number of streams of all users.
        """
        return self._Ns

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def getCost(self):
        """Get the Cost of the algorithm for the current iteration of the
        precoder.
        """
        Cost = 0
        # This will get all combinations of (k,l) without repetition. This
        # is equivalent to two nested for loops with an if statement to
        # only execute the code only when `k` is different of `l`.
        all_kl_indexes = itertools.permutations(range(self.K), 2)

        for kl in all_kl_indexes:
            (k, l) = kl
            Hkl_Fl = np.dot(
                self.getChannel(k, l),
                self.F[l])
            Cost = Cost + np.linalg.norm(
                Hkl_Fl -
                np.dot(
                    np.dot(
                        self.C[k],
                        self.C[k].transpose().conjugate()),
                    Hkl_Fl
                ), 'fro') ** 2

        return Cost

    def step(self):
        """Step the algorithm
        """
        self.updateC()
        self.updateF()
        self.updateW()

    def updateC(self):
        """Update the value of Ck for all K users.

        Ck contains the orthogonal basis of the interference subspace of
        user k. It corresponds to the Nk-Sk dominant eigenvectors of
        $\sum_{l \neq k} H_{k,l} F_l F_l^H H_{k,l}^H$
        """
        Ni = self.Nr - self.Ns  # Ni: Dimension of the interference subspace

        self.C = np.zeros(self.K, dtype=np.ndarray)
        # This will get all combinations of (k,l) without repetition. This
        # is equivalent to two nested for loops with an if statement to
        # only execute the code only when `k` is different of `l`.
        all_kl_indexes = itertools.permutations(range(self.K), 2)

        # This code will store in self.C[k] the equivalent of
        # $\sum_{l \neq k} H_{k,l} F_l F_l^H H_{k,l}^H$
        for kl in all_kl_indexes:
            (k, l) = kl
            Hkl_F = np.dot(
                self.getChannel(k, l),
                self.F[l])
            self.C[k] = self.C[k] + np.dot(Hkl_F, Hkl_F.transpose().conjugate())

        # Every element in self.C[k] is a matrix. We want to replace each
        # element by the dominant eigenvectors of that element.
        for k in np.arange(self.K):
            # TODO: implement and test with external interference
            # # We are inside only of the first for loop
            # # Add the external interference contribution
            # self.C[k] = obj.C{k} + obj.Rk{k}

            # C[k] will receive the Ni most dominant eigenvectors of C[k]
            self.C[k] = peig(self.C[k], Ni[k])[0]

    def updateF(self):
        """Update the value of the precoder of all K users.

        Fl, the precoder of the l-th user, tries avoid as much as possible
        to send energy into the desired signal subspace of the other
        users. Fl contains the Sl least dominant eigenvectors of
        $\sum_{k \neq l} H_{k,l}^H (I - C_k C_k^H)H_{k,l}$
        """
        # xxxxx Calculates the temporary variable Y[k] for all k xxxxxxxxxx
        # Note that $Y[k] = (I - C_k C_k^H)$
        calc_Y = lambda Nr, C: np.eye(Nr, dtype=complex) - \
            np.dot(C, C.conjugate().transpose())
        Y = map(calc_Y, self.Nr, self.C)

        newF = np.zeros(self.K, dtype=np.ndarray)
        # This will get all combinations of (l,k) without repetition. This
        # is equivalent to two nested for loops with an if statement to
        # only execute the code only when `l` is different of `k`
        all_lk_indexes = itertools.permutations(range(self.K), 2)

        # This code will store in newF[l] the equivalent of
        # $\sum_{k \neq l} H_{k,l}^H (I - C_k C_k^H)H_{k,l}$
        for lk in all_lk_indexes:
            (l, k) = lk
            lH = self.getChannel(k, l)
            newF[l] = newF[l] + np.dot(
                np.dot(lH.conjugate().transpose(),
                       Y[k]),
                lH)

        # Every element in newF is a matrix. We want to replace each
        # element by the least dominant eigenvectors of that element.
        self.F = map(lambda x, y: leig(x, y)[0], newF, self.Ns)

    def updateW(self):
        """Update the zero-forcing filters.

        The zero-forcing filter is calculated in the paper "MIMO
        Interference Alignment Over Correlated Channels with Imperfect
        CSI".
        """
        newW = np.zeros(self.K, dtype=np.ndarray)
        for k in np.arange(self.K):
            tildeHi = np.hstack(
                [np.dot(self.getChannel(k, k), self.F[k]),
                 self.C[k]])
            newW[k] = np.linalg.inv(tildeHi)
            # We only want the first Ns[k] lines
            newW[k] = newW[k][0:self.Ns[k]]
        self.W = newW

    def randomizeF(self, Nt, Ns, K):
        """Generates a random precoder for each user.

        Arguments:
        - `K`: Number of users.
        - `Nt`: Number of transmit antennas of each user
        - `Ns`: Number of streams of each user.
        """
        if isinstance(Ns, int):
            Ns = np.ones(K) * Ns
        if isinstance(Nt, int):
            Nt = np.ones(K) * Nt

        # Lambda function that returns a normalized version of the input
        # numpy array
        normalized = lambda A: A / np.linalg.norm(A, 'fro')

        self.F = np.zeros(K, dtype=np.ndarray)
        for k in range(K):
            self.F[k] = normalized(randn_c(Nt[k], Ns[k]))
        #self.F = [normalized(randn_c(Nt[k], Ns[k])) for k in np.arange(0, K)]
        self._Ns = Ns

    def randomizeH(self, Nr, Nt, K):
        """Generates a random channel matrix for all users.

        Arguments:
        - `K`: (int) Number of users.
        - `Nr`: (array or int) Number of receive antennas of each user
        - `Nt`: (array or int) Number of transmit antennas of each user
        """
        self._multiUserChannel.randomize(Nr, Nt, K)

    # This method does not need testing, since the logic is implemented in
    # the MultiUserChannelMatrix class and it is already tested.
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
        self._multiUserChannel.init_from_channel_matrix(channel_matrix, Nr,
                                                       Nt, K)

    # This method does not need testing, since the logic is implemented in
    # the MultiUserChannelMatrix class and it is already tested.
    def getChannel(self, k, l):
        """Get the channel from user l to user k.

        Arguments:
        - `l`: Transmitting user.
        - `k`: Receiving user
        """
        return self._multiUserChannel.getChannel(k, l)


# xxxxx Perform the doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    # When this module is run as a script the doctests are executed
    import doctest
    doctest.testmod()
    print "Hello"
    print "{0} executed".format(__file__)
