#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with implementation of Coordinated Multipoint (COmP) algorithms

"""

import numpy as np
import scipy.linalg as spl

# xxxxxxxxxx Remove latter xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import sys
sys.path.append("../")
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

from subspace.projections import calcProjectionMatrix
from comm.blockdiagonalization import BlockDiaginalizer

# Used for debug
from comm import blockdiagonalization
from comm import channels
from util.misc import least_right_singular_vectors


# TODO: the power of the external interference must be accounted. Is it
# already accounted in the concatenated_extint_channel???
def calc_cov_matrix_extint_plus_noise(concatenated_extint_channel, Nr, noise_var=0):
    """Calculates the covariance matrix of the external interference plus
    noise.

    Parameters
    ----------
    concatenated_extint_channel : 2D Array of complex numbers
        The channel from all external interference sources to all the
        receivers, where the rows correspond to the receivers antennas and
        the columns to the external interference sources.
    Nr : 1D Array of integers
        An array with the number of receive antennas of each receiver. If,
        for instance, the array is [2, 3], then
        :py:attr:`concatenated_extint_channel` must have 5 rows, where the
        first two correspond to the first receiver while the last three
        correspond to the second receiver.
    noise_var : float, optional [default=0]
        Noise variance. If not specified, then only the covariance matrix
        of the external interference will be returned.

    Returns
    -------
    R : 1D array of numpy matrices
        Return a numpy array, where each element is the covariance matrix
        of the external interference plus noise at one receiver.
    """
    R = np.empty(Nr.size, dtype=np.ndarray)
    cum_Nr = np.hstack([0, np.cumsum(Nr)])

    for ii in range(Nr.size):
        H = concatenated_extint_channel[cum_Nr[ii]:cum_Nr[ii + 1], :]
        R[ii] = np.dot(H, H.transpose().conjugate()) + np.eye(Nr[ii]) * noise_var
    return R


# TODO: Test-me
def _calc_stream_reduction_matrix(Re_k, num_red):
    """Calculates the `P` matrix that performs the stream reducion.

    Parameters
    ----------
    Re_k : 2D numpy array
        The external interference plus noise covariance matrix at a SINGLE
        receiver.
    num_red : int
        Number of streams to be sacrificed in order to mitigate the
        external interference

    Returns
    -------
    P : 2D numpy array
        A matrix whose columns corresponding to the `num_red` least
        significant singular vectors of Re_k

    """
    Nr, Nr = Re_k.shape
    (min_Vs, remaining_Vs, S) = least_right_singular_vectors(Re_k, num_red)
    return min_Vs


class Comp(BlockDiaginalizer):
    """Performs the Coordinated Multipoint transmission.

    This class basically performs the Block Diagonalization in a joint
    transmission assuring that the Power restriction of each Base Station
    is not violated.

    """

    def __init__(self, iNUsers, iPu, noiseVar):
        """Initializes the Comp object.

        Parameters
        ----------
        iNUsers : int
            Number of users.
        iPu : float
            Power available for EACH user.
        noiseVar : float
            Noise variance (power in linear scale).
        """
        BlockDiaginalizer.__init__(self, iNUsers, iPu, noiseVar)

    def perform_comp(self, mtChannel):
        """Perform the block diagonalization of `mtChannel`.

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
        return BlockDiaginalizer.block_diagonalize(self, mtChannel)


class CompExtInt(Comp):
    """Performs the Coordinated Multipoint transmission also taking into
    account the external interference.

    The CompExtInt class performs the block diagonalization characteristic
    to the joint transmission scenario where multiple base stations act as
    a single transmitter to send data to the users. However, in addition to
    what the BlockDiaginalizer class does the CompExtInt class can also
    take external interference into account.

    One way to reduce of eliminate the external interference is to
    sacrifice streams in directions strongly occupied by the external
    interference.

    Notes
    -----
    See the :class:`BlockDiaginalizer` class for details about the block
    diagonalization process.

    """

    def __init__(self, iNUsers, iPu, noiseVar):
        """Initializes the CompExtInt object.

        Parameters
        ----------
        iNUsers : int
            Number of users.
        iPu : float
            Power available for EACH user.
        noiseVar : float
            Noise variance (power in linear scale).
        """
        Comp.__init__(self, iNUsers, iPu, noiseVar)

    @staticmethod
    def calc_receive_filter(newH, P=None):
        """Calculates the Zero-Forcing receive filter.

        Parameters
        ----------
        newH : 2D numpy array
            The block diagonalized channel AFTER also applying any stream
            sacrificing.
        P : 1D numpy array of 2D numpy arrays
            P has the most significant singular vectors of the external
            interference plus noise covariance matrix for each
            receiver. Note that if one element of P is an empty array that
            means that there won't be any stream reduction for that user
            and therefore the receive filter fill be the same as the
            regular Block Diagonalization.

        Returns
        -------
        W : 1D array of 2D numpy arrays
            The zero-forcing matrix to separate each stream of each user.

        """
        if P is None:
            W = np.linalg.pinv(newH)

        # K = P.size
        # W_bd = np.empty(K)

        # for idx, p in enumerate(P):
        #     calcProjectionMatrix(p)
        #     H_ieq =
        #     W_bd[idx]

        return W

    @staticmethod
    def _calc_SNRs(H_ieq, W, Re):
        """Calculates the effective SNRs of each channel with the applied
        precoder matrix `precoder`.

        Parameters
        ----------
        H_ieq : 2D numpy array
            Equivalent channel matrix of all users after all precoding is
            applied.
        W : 2D numpy array
            Global receive filter for all users. This should actually a
            block diagonal matrix where each block corresponds to the
            receive filter at one receiver.
        Re : 1D numpy array of 2D numpy arrays.
            A numpy array where each element is the covariance matrix of
            the external interference plus noise seen by a user.

        Returns
        -------
        SNRs : 1D numpy array
            SNR (in linear scale) of all the parallel channels of all users.

        """
        #K = Re.size  # Number of users

        mtP = np.dot(W, H_ieq)
        desired_power = np.abs(np.diagonal(mtP)) ** 2
        internalInterference = np.sum(np.abs((mtP - np.diagflat(np.diagonal(mtP)))) ** 2, 1)

        block_diag_Re = spl.block_diag(*Re)

        W_H = W.transpose().conjugate()
        # Note that the noise is already accounted in the covariance matrix Re
        external_interference_plus_noise = np.diagonal(
            np.dot(W, np.dot(block_diag_Re, W_H))).real

        sinr = desired_power / (internalInterference + external_interference_plus_noise)
        return sinr

    def perform_comp(self, mtChannel, Re):
        """Perform the block diagonalization of `mtChannel` taking the external
        interference into account.

        Parameters
        ----------
        mtChannel : 2D numpy array
            Channel from all the transmitters (not including the external
            interference sources) to all the receivers.
        Re : 1D numpy array of 2D numpy arrays
            Covariance matrix of the external interference plus noise of
            each user. `Re` must be a numpy array of dimension (K x 1),
            where K is the number of users, and each element in `Re` is a
            numpy 2D array of size (Nrk x Nrk), where 'Nrk' is the number
            of receive antennas of the k-th user.

        Returns
        -------
        TODO: write me

        """
        Ms_bad, Sigma = self._calc_BD_matrix_no_power_scaling(mtChannel)

        # xxxxx SINRs with no stream reduction xxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Since there is no projection
        Ms_good = self._perform_normalized_waterfilling_power_scaling(Ms_bad,
                                                                      Sigma)
        newH = np.dot(mtChannel, Ms_good)
        W = self.calc_receive_filter(newH)

        SNRs = self._calc_SNRs(newH, W, Re)
        print SNRs

        #U, S, V_h = np.linalg



        # Q = np.empty(self.iNUsers, dtype=np.ndarray)
        # for k in range(self.iNUsers):
        #     # Calculates a projection matrix to the subspace orthogonal to
        #     # the external interference
        #     Q[k] = calcOrthogonalProjectionMatrix(Re)

        (newH, Ms_good) = BlockDiaginalizer.block_diagonalize(self, mtChannel)

        # xxxxxxxxxx Calculates the SINR with only BD xxxxxxxxxxxxxxxxxxxxx

        return (newH, Ms_good)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def perform_comp(mtChannel, iNUsers, iPu, noiseVar):
    """Performs the block diagonalization of `mtChannel`.

    Parameters
    ----------
    mtChannel : 2D numpy array
            Channel from the transmitter to all users.
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
    COMP = Comp(iNUsers, iPu, noiseVar)
    results_tuble = COMP.perform_comp(mtChannel)
    return results_tuble


def perform_comp_with_ext_int(mtChannel, iNUsers, iPu, noiseVar, Re):
    """Perform the block diagonalization of `mtChannel` taking the external
    interference into account.

    Parameters
    mtChannel : 2D numpy array
        Channel from all the transmitters (not including the external
        interference sources) to all the receivers.
    iNUsers : int
        Number of users
    iPu : float
        Power available for each user
    noiseVar : float
        Noise variance
    Re : 1D numpy array of 2D numpy arrays
        Covariance matrix of the external interference plus noise of each
        user. `Re` must be a numpy array of dimension (K x 1), where K is
        the number of users, and each element in `Re` is a numpy 2D array
        of size (Nrk x Nrk), where 'Nrk' is the number of receive antennas
        of the k-th user.

    Returns
    -------
    TODO: write me

    """
    COMP = CompExtInt(iNUsers, iPu, noiseVar)
    results_tuble = COMP.perform_comp(mtChannel, Re)
    return results_tuble


if __name__ == '__main__':
    Pu = 5.
    noise_var = 0.1
    num_users = 3
    num_antenas = 2
    # Total number of transmit and receive antennas
    #iNr = iNt = num_antenas * num_users

    #channel = randn_c(iNr, iNt)

    channel = channels.MultiUserChannelMatrix()
    channel.randomize(num_antenas, num_antenas, num_users)

    (newH, Ms) = blockdiagonalization.block_diagonalize(
        channel.big_H, num_users, Pu, noise_var)

    print newH.shape

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    for userindex in range(num_users):
        # Code here will be moved to the CompExtInt.perform_comp method later
        mtW_bd = np.linalg.pinv(newH)
        mtP = np.dot(mtW_bd, newH)

        # SNR without interference handling
        desired_power = np.abs(np.diag(mtP)) ** 2
        internalInterference = np.sum(np.abs(mtP - np.diag(np.diag(mtP))) ** 2, axis=1)

        #externalInterference =

        print internalInterference

        a = np.array([[1, 2, 3], [4, 5, 6], [5, 4, 3]])
        print a
        print np.sum(a, axis=1)
