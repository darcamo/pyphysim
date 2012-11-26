#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Block Diagonalization module documentation.

1. Summary
2. Extended summary
3. Routine listings
4. See also
5. Notes
6. References
7. Examples

.. TODO:: Move everything in the blockdiagonalization module to the
   comp module.

"""

# TODO: Move the blockdiagonalization module to the comp package.

import numpy as np
import collections
#from scipy.linalg import block_diag

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

    Notes
    -----
    The block diagonalization algorithm is described in [1]_, where
    different power allocations are illustrated. The :class:`BlockDiaginalizer`
    class implement two power allocation methods, a global power
    allocation, and a 'per transmitter' power allocation.

    .. [1] Q. H. Spencer, A. L. Swindlehurst, and M. Haardt,
       "Zero-Forcing Methods for Downlink Spatial Multiplexing
       in Multiuser MIMO Channels," IEEE Transactions on Signal
       Processing, vol. 52, no. 2, pp. 461–471, Feb. 2004.

    """

    def __init__(self, iNUsers, iPu, noiseVar):
        """Initialize the BlockDiaginalizer object.

        Parameters
        ----------
        iNUsers : int
            Number of users.
        iPu : float
            Power available for EACH user.
        noiseVar : float
            Noise variance (power in linear scale).
        """
        self.iNUsers = iNUsers
        self.iPu = iPu
        self.noiseVar = noiseVar

    def _calc_BD_matrix_no_power_scaling(self, mtChannel):
        """Calculates the modulation matrix "M" without any king of power
        scaling.

        The "modulation matrix" is a matrix that changes the channel to a
        block diagonal structure and it is the first part in the Block
        Diagonalization algorithm. The returned modulation matrix is
        equivalent to Equation (12) of [1]_ but without the power scaling
        matrix $\Lambda$. Therefore, for the compelte BD algorithm it is
        still necessary to perform this power scalling in the output of
        _calc_BD_matrix.

        The reason why the Block Diagonalization algorithm was broken down
        into the code here in the _calc_BD_matrix method and the power
        scaling code is because the power scaling may changing depending
        on the scenario. For instance, if the transmitter corresponds to a
        single base station the the power may be distributed into all the
        dimensions of the Modulation matrix. On the other hand, if the
        transmitter corresponds to multiple base stations jointly
        transmitting to multiple users then the power of each base station
        must be distributed only into the dimensions corresponding to that
        base station.

        Parameters
        ----------
        mtChannel : 2D numpy array
            Channel from the transmitter to all users.

        Returns
        -------
        (Ms_bad, Sigma) : A tuple of numpy arrays
            The modulation matrix "Ms_bad" is a precoder that block
            diagonalizes the channel. The singular values of the equivalent
            channel when the modulation matrix is applied correspond to
            Sigma. Therefore, Sigma can be used latter in the power
            allocation process.
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
        return (Ms_bad, Sigma)

    def _perform_global_waterfilling_power_scaling(self, Ms_bad, Sigma):
        """Perform the power scaling based on the waterfilling algorithm for
        all the parallel channel gains in `Sigma`.

        This power scaling method corresponds to maximizing the sum rate
        when the transmitter is a single base station transmitting to all
        users. Note that this approach may result in one or two "strong
        users" taking a dominant share of the available power.

        Parameters
        ----------
        Ms_bad : 2D numpy array
            The previously calculated modulation matrix (without any powr
            scaling)
        Sigma : 1D numpy array of positive floats
            The singular values of the effective channel when Ms_bad is
            applied.

        Returns
        -------
        Ms_good : 2D numpy array
            The modulation matrix with the global power scaling applied.

        """
        # Perform water-filling for the parallel channel gains in Sigma
        # (but considering a global power constraint, each element (power)
        # in Sigma comes from all APs)
        total_power = self.iNUsers * self.iPu
        (vtOptP, mu) = waterfilling.doWF(Sigma ** 2,
                                         total_power,
                                         self.noiseVar)

        Ms_good = np.dot(Ms_bad,
                         np.diag(np.sqrt(vtOptP)))

        return Ms_good

    def _perform_normalized_waterfilling_power_scaling(self, Ms_bad, Sigma):
        """Perform the power scaling based on the waterfilling algorithm for
        all the parallel channel gains in `Sigma`, but normalize the result
        by the power of the base station transmitting with the highest
        power.

        When we have a joint transmission where multiple base stations act
        as a single base station, then performing the waterfilling on all
        the channels for the total available power may result in some base
        station transmitting with a higher power then it actually
        can. Therefore, we normalize the power of the strongest BS so that
        the power restriction at each BS is satisfied. This is sub-optimal
        since the other BSs will use less power then available, but it is
        simple and it works.

        Parameters
        ----------
        Ms_bad : 2D numpy array
            The previously calculated modulation matrix (without any powr
            scaling)
        Sigma : 1D numpy array of positive floats
            The singular values of the effective channel when Ms_bad is
            applied.

        Returns
        -------
        Ms_good : 2D numpy array
            The modulation matrix with the normalized power scaling applied.

        """
        # Number of receive antennas per user
        #
        # Note: I think this only works if the number of receive transmit
        # is equal to the number of receive antennas
        iNtU = Sigma.size / float(self.iNUsers)

        # First we perform the global waterfilling
        Ms_good = self._perform_global_waterfilling_power_scaling(
            Ms_bad, Sigma)

        # Since we used a global power constraint but we have in fact a
        # power constraint for each AP, we need to normalize the allocated
        # powers by the power of the AP with most energy (so that the
        # individual power constraints are satisfied). This will be
        # sub-optimal for the other bases, but it is what we can do.
        max_sqrt_P = 0
        for user in range(0, self.iNUsers):
            # Calculate the Frobenius norm of the matrix corresponding to
            # the transmitter `user`
            user_matrix = Ms_good[:, user * iNtU:user * iNtU + iNtU]
            # The power is actually the square of cur_sqrt_P
            cur_sqrt_P = np.linalg.norm(user_matrix, 'fro')
            if cur_sqrt_P > max_sqrt_P:
                max_sqrt_P = cur_sqrt_P

        # Normalize the power of the AP with highest transmitted power to
        # be equal to self.iPu
        Ms_good = Ms_good * np.sqrt(self.iPu) / max_sqrt_P

        return Ms_good

    def block_diagonalize(self, mtChannel):
        """Perform the block diagonalization.

        mtChannel is a matrix with the channel from the transmitter to all
        users, where each `iNUsers` rows correspond to one user.

        Parameters
        ----------
        mtChannel : 2D numpy array
            Channel from the transmitter to all users.

        Returns
        -------
        (newH, Ms_good) : A tuple of numpy arrays
            newH is a 2D numpy array corresponding to the Block
            diagonalized channel, while Ms_good is a 2D numpy array
            corresponding to the precoder matrix used to block diagonalize
            the channel.

        """
        # Calculates the modulation matrix and the singular values of the
        # effective channel when this modulation matrix is applied.
        (Ms_bad, Sigma) = self._calc_BD_matrix_no_power_scaling(mtChannel)

        # Scale the power of this modulation assuring the power restriction
        # is not violated in any of the base stations.
        Ms_good = self._perform_normalized_waterfilling_power_scaling(Ms_bad,
                                                                      Sigma)

        # Finally calculates the Block diagonal channel
        newH = np.dot(mtChannel, Ms_good)

        # Return block diagonalized channel and the used precoding matrix
        return (newH, Ms_good)

    @staticmethod
    def calc_receive_filter(newH):
        """Calculates the Zero-Forcing receive filter.

        Parameters
        ----------
        newH : 2D numpy array
            The block diagonalized channel.

        Returns
        -------
        W_bd : 2D numpy array
            The zero-forcing matrix to separate each stream of each user.
        """
        W_bd = np.linalg.pinv(newH)
        return W_bd

    def _get_tilde_channel(self, mtChannel, user):
        """Return the combined channel of all users except `user` .

        """
        vtAllUserIndexes = np.arange(0, self.iNUsers)
        desiredUsers = [i for i in vtAllUserIndexes if i != user]
        return self._getSubChannel(mtChannel, desiredUsers)

    def _getSubChannel(self, mt_channel, vtDesiredUsers):
        """Get a subchannel according to the vtDesiredUsers vector.

        Parameters
        ----------
        mt_channel : 2D numpy array
            Channel of all users
        vtDesiredUsers : iterable of integers
            An iterable with the indexes of the desired users or an
            integer.

        Returns
        -------
        mtSubmatrix : 2D numpy array
           Submatrix of the desired users

        Notes
        -------
        As an example, let's consider the case with a channel for 3
        receivers, each with 2 receive antennas, where the transmitter has
        6 transmit antennas.

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


def block_diagonalize(mtChannel, iNUsers, iPu, noiseVar):
    """Performs the block diagonalization of :attr:`mtChannel`.

    Parameters
    ----------
    mtChannel : 2D numpy array
        Global channel matrix
    iNUsers : int
        Number of users
    iPu : float
        Power available for each user
    noiseVar : float
        Noise variance

    Returns
    -------
    (newH, Ms_good) : A tuple of numpy arrays
        newH is a 2D numpy array corresponding to the Block
        diagonalized channel, while Ms_good is a 2D numpy array
        corresponding to the precoder matrix used to block diagonalize
        the channel.

    """
    BD = BlockDiaginalizer(iNUsers, iPu, noiseVar)
    results_tuple = BD.block_diagonalize(mtChannel)
    return results_tuple


def calc_receive_filter(newH):
    """Calculates the Zero-Forcing receive filter.

    Parameters
    ----------
    newH : 2D numpy array
        The block diagonalized channel.

    Returns
    -------
    W_bd : 2D numpy array
        The zero-forcing matrix to separate each stream of each user.
    """
    return BlockDiaginalizer.calc_receive_filter(newH)
