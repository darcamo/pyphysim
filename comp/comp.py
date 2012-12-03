#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with implementation of Coordinated Multipoint (COmP) algorithms

"""

import numpy as np

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
from util.conversion import single_matrix_to_matrix_of_matrices, linear2dB


def _calc_stream_reduction_matrix(Re_k, kept_streams):
    """Calculates the `P` matrix that performs the stream reduction such
    that the subspace of the remaining streams span the dimensions with the
    lowest interference.

    Parameters
    ----------
    Re_k : 2D numpy array
        The external interference plus noise covariance matrix at a SINGLE
        receiver.
    kept_streams : int
        Number of streams that will be kept. This will be equal to the
        number of columns of the returned matrix.

    Returns
    -------
    Pk : 2D numpy array
        A matrix whose columns corresponding to the `kept_streams` least
        significant right singular vectors of Re_k.

    """
    min_Vs = least_right_singular_vectors(Re_k, kept_streams)[0]
    return min_Vs


# TODO: remove this class, since CompExtInt does everything this class
# does.
class Comp(BlockDiaginalizer):  # pragma: no cover
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

    def __init__(self, iNUsers, iPu, noiseVar, pe):
        """Initializes the CompExtInt object.

        Parameters
        ----------
        iNUsers : int
            Number of users.
        iPu : float
            Power available for EACH user (in linear scale).
        noiseVar : float
            Noise variance (power in linear scale).
        pe : float
            Power of the external interference source (in linear scale)
        """
        Comp.__init__(self, iNUsers, iPu, noiseVar)
        self.pe = pe

        # Function used to decide how many streams will be sacrificed to
        # mitigate external interference. This is set in the
        # set_ext_int_handling_metric method (as well as the _modulator and
        # _package_length attributes)
        self._metric_func = None  # The default metric will be None
        self._modulator = None
        self._package_length = None

    def set_ext_int_handling_metric(self, metric,
                                    modulator=None, package_length=None):
        """Set the metric used to decide how many streams to sacrifice for
        external interference handling.

        - If `metric` is None, then no streams will be sacrificed and the
          external interference won't be mitigated.

        - If `metric` is 'capacity', then the the sum capacity will be used
          to decide how many streams to sacrifice. That is, the
          _calc_shannon_sum_capacity method will be used in perform_comp to
          map the calculated sinrs into a sum capacity and the number of
          sacrificed streams will be the one with the highest sum capacity.

        - If `metric` is 'effective_throughput' then the effective
          throughput will be used to decide how many streams to
          sacrifice. That is, the _calc_effective_throughput method will be
          used in perform_comp to map the calculated sinrs into an
          effective throughput and the number of sacrificed streams will be
          the one with the highest sum capacity.

        Notes
        -----
        If `metric` is 'effective_throughput' then the modulator and
        package_length arguments must be provided. For the other metric
        options they will be ignored if provided.

        Parameters
        ----------
        metric : str, {None, 'capacity', 'effective_throughput'}
            The metric name. Must be one of the available metrics.
        modulator : comm.modulators.Modulator object
            The modulator object used in the simulation. It will be used to
            calculate the theoretical BER and (with the package length,
            the theoretical PER)
        package_length : int
            The package length used in the simulation.

        Raises
        ------
        AttributeError
            If the metric is not one of {None, 'capacity',
            'effective_throughput'}

        """
        if metric is None:
            self._metric_func = None
            self._modulator = None
            self._package_length = None

        elif metric == 'capacity':
            self._metric_func = self._calc_shannon_sum_capacity
            self._modulator = None
            self._package_length = None

        elif metric == 'effective_throughput':
            self._metric_func = self._calc_effective_throughput
            if (modulator is None) or (package_length is None):
                raise AttributeError("The modulator and package_length attributes must be provided for the 'effective_throughput' metric.")
            self._modulator = modulator
            self._package_length = package_length
        else:
            raise AttributeError("The `metric` attribute can only be one of {None, 'capacity', 'effective_throughput'}")

    @staticmethod
    def calc_receive_filter_user_k(Heq_k, P=None):
        """Calculates the Zero-Forcing receive filter of a single user `k`
        with or without the stream reduction.

        Parameters
        ----------
        Heq_k : 2D numpy array
            The equivalent channel of user `k` after the block
            diagonalization process, but without including the stream
            reduction.
        P : 2D numpy array
            P has the most significant singular vectors of the external
            interference plus noise covariance matrix for each
            receiver. Note that if one element of P is an empty array that
            means that there won't be any stream reduction for that user
            and therefore the receive filter fill be the same as the
            regular Block Diagonalization.

        Returns
        -------
        W : 2D numpy array
            The receive filter of user `k`.

        Notes
        -----
        If `P` is not None then the number of transmit streams will be
        equal to the number of columns in `P`. Also, the receive filter `W`
        includes a projection into the subspace spanned by the columns of
        `P`. Since `P` was calculated to be in the directions with weaker
        (or no) external interference then the receive filter `W` will
        mitigate external interference.

        """
        if P is None:
            W = np.linalg.pinv(Heq_k)
        else:
            overbar_P = calcProjectionMatrix(P)
            # Calculate the equivalent channel including the stream
            # reduction
            Heq_k_red = np.dot(Heq_k, P)
            W = np.dot(
                np.linalg.pinv(np.dot(overbar_P, Heq_k_red)),
                overbar_P)

        return W

    @staticmethod
    def _calc_linear_SINRs(Heq_k_red, Wk, Re_k):
        """Calculates the effective SINRs of each channel with the applied
        precoder matrix `precoder`.

        Parameters
        ----------
        Heq_k_red : 2D numpy array
            Equivalent channel matrix of user `k` including the block
            diagonalization and any stream reduction applied.
        Wk : 2D numpy array
            Receive filter for user `k`.
        Re_k : 1D numpy array of 2D numpy arrays.
            A numpy array where each element is the covariance matrix of
            the external interference PLUS noise seen by a user.

        Returns
        -------
        sinrs : 1D numpy array
            SINR (in linear scale) of all the parallel channels of all users.

        """
        #K = Re_k.size  # Number of users
        mtP = np.dot(Wk, Heq_k_red)
        desired_power = np.abs(np.diagonal(mtP)) ** 2
        internalInterference = np.sum(np.abs((mtP - np.diagflat(np.diagonal(mtP)))) ** 2, 1)

        Wk_H = Wk.transpose().conjugate()

        # Note that the noise is already accounted in the covariance matrix
        # Re_k
        external_interference_plus_noise = np.diagonal(
            np.dot(Wk, np.dot(Re_k, Wk_H))
        ).real

        sinr = desired_power / (internalInterference + np.abs(external_interference_plus_noise))
        return sinr

    # Since only `sinrs` is required and it is passed as an argument, then
    # this method is a static method.
    @staticmethod
    def _calc_shannon_sum_capacity(sinrs):
        """Calculate the sum of the Shannon capacity of the values in `sinrs`

        Parameters
        ----------
        sinrs : 1D numpy array or float
            SINR values (in linear scale).

        Returns
        -------
        sum_capacity : float
            Sum capacity.

        Examples
        --------
        >>> sinrs_linear = np.array([11.4, 20.3])
        >>> print CompExtInt._calc_shannon_sum_capacity(sinrs_linear)
        8.04504974084
        """
        sum_capacity = np.sum(np.log2(1 + sinrs))

        return sum_capacity

    def _calc_effective_throughput(self, sinrs):
        """Calculates the effective throughput of the values in `sinrs`.

        The effective throughput is equivalent to the package error for a
        specific package error rate and package length, times the nominal
        throughput.

        Parameters
        ----------
        sinrs : 1D numpy array or float
            SINR values (in linear scale).

        Returns
        -------
        effective_throughput : float
            Effective throughput that can be obtained.

        """
        SINRs = linear2dB(sinrs)
        se = self._modulator.calcTheoreticalSpectralEfficiency(SINRs, self._package_length)
        total_se = np.sum(se)

        return total_se

    def perform_comp(self, mu_channel):
        """Perform the block diagonalization of `mu_channel` taking the
        external interference into account.

        Two important parameters used here are the `noise_var` (noise
        variance) and the `pe` (external interference power) attributes.

        Parameters
        ----------
        mu_channel : MultiUserChannelMatrixExtInt object.
            A MultiUserChannelMatrixExtInt object, which has the channel
            from all the transmitters to all the receivers, as well as th
            external interference.

        Returns
        -------
        MsPk_all_users : 1D numpy array of 2D numpy arrays
            A 1D numpy array where each element corresponds to the precoder
            for a user.

        """
        K = mu_channel.K
        Nr = mu_channel.Nr
        Nt = mu_channel.Nt
        H_matrix = mu_channel.big_H_no_ext_int
        Re = mu_channel.calc_cov_matrix_extint_plus_noise(
            self.noise_var, self.pe)

        if self._metric_func is None:
            # If _metric_func is None then we are not handling external
            # interference. Therefore, we simple call the perform_comp
            # method of the Comp class.
            (newH, Ms_good) = BlockDiaginalizer.block_diagonalize(self, mu_channel.big_H_no_ext_int)
            MsPk_all_users = single_matrix_to_matrix_of_matrices(Ms_good, None, Nt)
            return MsPk_all_users

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # If we are here, then _metric_func is not None, which means that
        # we are handling the external interference.
        Ms_bad, Sigma = self._calc_BD_matrix_no_power_scaling(H_matrix)

        # The k-th 'element' in Ms_bad_ks is a matrix containing the
        # columns of Ms_bad of the k-th user.
        Ms_bad_ks = single_matrix_to_matrix_of_matrices(Ms_bad, None, Nt)
        H_all_ks = single_matrix_to_matrix_of_matrices(H_matrix, Nr)

        # Loop for the users
        MsPk_all_users = np.empty(K, dtype=np.ndarray)
        for userindex in range(K):
            #print 'User: {0}'.format(userindex)
            Ntk = Nt[userindex]
            Rek = Re[userindex]
            Hk = H_all_ks[userindex]
            Msk = Ms_bad_ks[userindex]

            # Equivalent channel of user k after the block diagonalization
            # process, but without any stream reduction
            Heq_k = np.dot(Hk, Msk)

            # DARLAN CONTINUE AQUI

            # We can have from a single stream to all streams (the number
            # of transmit antennas). This loop varies the number of
            # transmit streams of user k.
            sum_capacity_k = np.empty(Ntk)
            Pk_all = np.empty(Ntk, dtype=np.ndarray)
            norm_term_all = np.empty(Ntk)

            for index in range(Ntk):
                Ns_k = index + 1
                # Find Pk
                Pk = _calc_stream_reduction_matrix(Rek, Ns_k)
                Pk_all[index] = Pk  # Save for later

                # Normalization term for the combined BD matrix Msk and stream
                # reduction matrix Pk
                norm_term = np.linalg.norm(np.dot(Msk, Pk), 'fro') / np.sqrt(self.iPu)
                norm_term_all[index] = norm_term  # Save for later

                # Equivalent channel with stream reduction
                Heq_k_red = np.dot(Heq_k, Pk / norm_term)

                # Calculates the receive filter W_k (note
                # calc_receive_filter_user_k receives the channel without
                # stream reduction as an argument and the stream reduction
                # matrix)
                W_k = self.calc_receive_filter_user_k(Heq_k, Pk)

                # SINR (in linear scale) of all streams of user k.
                sinrs_k = self._calc_linear_SINRs(Heq_k_red, W_k, Rek)

                #sum_capacity_k[index] = self._calc_shannon_sum_capacity(sinrs_k)
                sum_capacity_k[index] = self._metric_func(sinrs_k)

                #print 'SumCapacity: {0}'.format(sum_capacity_k[index])

            # The index with the highest metric value. This is equivalent
            # to the number of transmit streams which yields the highest
            # value of the metric.
            best_index = np.argmax(sum_capacity_k)
            MsPk_all_users[userindex] = np.dot(Msk, Pk_all[best_index]) / norm_term_all[best_index]

        return MsPk_all_users


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
    ----------
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
    output : lalala
        Write me

    """
    COMP = CompExtInt(iNUsers, iPu, noiseVar)
    results_tuble = COMP.perform_comp(mtChannel, Re)
    return results_tuble


if __name__ == '__main__1':
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
