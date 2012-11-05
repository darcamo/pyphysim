#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""module docstring"""

import numpy as np
import collections
#from scipy.linalg import block_diag

import sys
sys.path.append("../")

from util.misc import least_right_singular_vectors
import waterfilling


class BlockDiaginalizer(object):
    """Class to perform the block diagonalization algorithm in a joint
    transmission scenario. That is, multiple base stations act as a single
    transmitter to send data to the users.

    The waterfilling algorithm is also applied to optimally distribute the
    power. However, since we have multiple base stations, each one with a
    power restriction, then after the power is optimally allocated at each
    base station all powers will be normalized to respect the power
    restriction.

    Reference:
      "Zero-Forcing Methods for Downlink Spatial Multiplexing in Multiuser
      MIMO Channels"

    """

    def __init__(self, iNUsers, iPu, noiseVar):
        """
        - `iNUsers`: Number of users.
        - `iPu`: Power available for EACH user.
        - `noiseVar`: Noise variance (power in linear scale).
        """
        self.iNUsers = iNUsers
        self.iPu = iPu
        self.noiseVar = noiseVar

    def block_diagonalize(self, mtChannel):  # , iNStreams=None, Re=None):
        """Perform the block diagonalization.

        mtChannel is a matrix with the channel from the transmitter to all
        users, where each `iNUsers` rows correspond to one user.

        Arguments:
        - `mtChannel`: (numpy array) Channel from the transmitter to all users.
        - `iNStreams`:
        - `Re`:

        Return:
        - newH: Block diagonalized channel
        - Ms_good: Precoding matrix used to block diagonalize the channel
        """
        (iNr, iNt) = mtChannel.shape
        assert iNr % self.iNUsers == 0, "`block_diagonalize`: Number of rows of the channel must be a multiple of the number of users."

        # Number of antennas per user
        iNrU = iNr / self.iNUsers

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Ms_bad = []
        Sigma = []
        # Calculates the interfering channels $\tilde{\mat{H}}_j$ as well
        # as $\tilde{\mtV}_j^{(1)}$ and $\tilde{\mtV}_j^{(0)}$.
        # Note that $\tilde{\mat{H}}_j = \tilde{\mtU}_j \tilde{\Sigma}_j [\tilde{\mtV}_j^{(1)} \; \tilde{\mtV}_j^{(0)}]^H$ where $\tilde{\mtV}_j^{(1)}$ holds
        # the first $\tilde{L}_j$ right singular vectors and $\tilde{\mtV}_j^{(0)}$ holds the
        # last $(n_T - \tilde{L}_j)$ right singular values
        for user in range(0, self.iNUsers):
            # channel of all users except the current user
            tilde_H_cur_user = self._get_tilde_channel(mtChannel, user)

            # How many streams user `user` can receive is given by the
            # total number of receive antennas minus the rank of
            # tilde_H_cur_user
            nStreams = iNr - np.linalg.matrix_rank(tilde_H_cur_user)
            (tilde_V0, tilde_V1, tilde_S) = least_right_singular_vectors(
                tilde_H_cur_user,
                nStreams)

            # The equivalent channel of the current user corresponds to
            # $\mtH_j \tilde{\mtV}_j^{(0)}$

            # First we get $\mtH_j$
            H_cur_user = self._getSubChannel(mtChannel, user)

            # Now we get the right singular value of the equivalent channel
            (V0, V1, S) = least_right_singular_vectors(
                np.dot(H_cur_user, tilde_V0),
                # Number of receive antennas minus number of desired
                # streams
                iNrU - nStreams)

            # Get Ms and Sigma
            Ms_bad.append(np.dot(tilde_V0, V1))
            Sigma.extend(S)

        # Concatenates the precoder for each user to form the complete Ms
        # matrix. Ms_bad is the version without waterfilling.
        # This is equivalent to "concatenate(Ms_bad, axis=1)"
        Ms_bad = np.hstack(Ms_bad)
        Sigma = np.array(Sigma)

        # Perform water-filling for the parallel channel gains in Sigma
        # (but considering a global power constraint, each element (power)
        # in Sigma comes from all APs)
        total_power = self.iNUsers * self.iPu
        (vtOptP, mu) = waterfilling.doWF(Sigma ** 2, total_power, self.noiseVar)

        Ms_good = np.dot(
            Ms_bad,
            np.diag(np.sqrt(vtOptP)))

        # Since we used a global power constraint but we have in fact a
        # power constraint for each AP, we need to normalize the allocated
        # powers by the power of the AP with most energy (so that the
        # individual power constraints are satisfied). This will be
        # sub-optimum for the other bases, but it is what we can do.
        max_sqrt_P = 0
        for user in range(0, self.iNUsers):
            # Calculate the Frobenius norm of the matrix corresponding to
            # the transmitter `user`
            user_matrix = Ms_good[user * iNrU:user * iNrU + iNrU, :]
            # The power is actually the square of cur_sqrt_P
            cur_sqrt_P = np.linalg.norm(user_matrix, 'fro')
            if cur_sqrt_P > max_sqrt_P:
                max_sqrt_P = cur_sqrt_P

        # Normalize the power of the AP with highest transmitted power to
        # be equal to self.iPu
        Ms_good = Ms_good * np.sqrt(self.iPu) / max_sqrt_P

        # Finally calculates the Block diagonal channel
        newH = np.dot(mtChannel, Ms_good)

        # Return block diagonalized channel, the used precoding matrix, and
        # the power allocated to each parallel channel.
        #
        # Note: The power values in vtOptP are before any power
        # normalization. Maybe vtOptP is not useful to be returned.
        #return (newH, Ms_good, vtOptP)
        return (newH, Ms_good)

    def _get_tilde_channel(self, mtChannel, user):
        """Return the combined channel of all users except `user`."""
        vtAllUserIndexes = np.arange(0, self.iNUsers)
        desiredUsers = [i for i in vtAllUserIndexes if i != user]
        return self._getSubChannel(mtChannel, desiredUsers)

    def _getSubChannel(self, mt_channel, vtDesiredUsers):
        """Get a subchannel according to the vtDesiredUsers vector.

        Arguments:
        - `mt_channel`: Channel of all users
        - `vtDesiredUsers`: An iterable with the indexes of the desired users
                            or an integer.
        Return:
        - mtSubmatrix - Submatrix of the desired users

        Channel for 3 receivers, each with 2 receive antennas, where the
        transmitter has 6 transmit antennas.
        >>> BD = BlockDiaginalizer(3, 0, 0)
        >>> channel = np.vstack([np.ones([2, 6]), 2 * np.ones([2, 6]), 3 * np.ones([2, 6])])
        >>> BD._getSubChannel(channel, [0,2])
        array([[ 1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.],
               [ 3.,  3.,  3.,  3.,  3.,  3.],
               [ 3.,  3.,  3.,  3.,  3.,  3.]])
        >>> BD._getSubChannel(channel, 0)
        array([[ 1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.]])
        """
        (rows, cols) = mt_channel.shape
        iNrU = rows / self.iNUsers  # Number of receive antennas per user

        if isinstance(vtDesiredUsers, collections.Iterable):
            vtIndexes = []
            for index in vtDesiredUsers:
                vtIndexes.extend(range(iNrU * index, (index + 1) * iNrU))
        else:
            vtIndexes = range(iNrU * vtDesiredUsers, (vtDesiredUsers + 1) * iNrU)
        return mt_channel[vtIndexes, :]


def block_diagonalize(mtChannel, iNUsers, iPu, noiseVar, iNStreams=None, Re=None):
    """Performs the block diagonalization of `mtChannel`.

    Arguments:
    - `mtChannel`: (numpy bidimensional array) Global channel matrix
    - `iNUsers`: (int) Number of users
    - `iPu`: (float) Power available for each user
    - `noiseVar`: (float) Noise variance
    - `iNStreams`: Not used
    - `Re`: Not used

    Return:
    - newH: Block diagonalized channel
    - Ms_good: Precoding matrix used to block diagonalize the channel

    """
    BD = BlockDiaginalizer(iNUsers, iPu, noiseVar)
    results_tuble = BD.block_diagonalize(mtChannel)
    return results_tuble
