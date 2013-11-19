#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module implementing the block diagonalization algorithm.

There are two ways to use this module. You can either use the
:class:`BlockDiaginalizer` class, or you can use the
:meth:`block_diagonalize` and the :meth:`calc_receive_filter` functions
(which use the BlockDiaginalizer class in their implementation).

"""

__all__ = ['BlockDiaginalizer', 'block_diagonalize', 'calc_receive_filter']

__revision__ = "$Revision$"

import numpy as np
import collections
#from scipy.linalg import block_diag

from util.misc import least_right_singular_vectors
from comm import waterfilling
from util.conversion import single_matrix_to_matrix_of_matrices, linear2dB
from subspace.projections import calcProjectionMatrix

__all__ = ['BlockDiaginalizer', 'block_diagonalize', 'calc_receive_filter']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
## xxxxxxxxxx Helper Functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def _calc_stream_reduction_matrix(Re_k, kept_streams):
    """Calculates the `P` matrix that performs the stream reduction such that
    the subspace of the remaining streams span the dimensions with the
    lowest interference (according to the external interference plus noise
    covariance matrix Re_k).

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


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
## xxxxxxxxxx Classes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class BlockDiaginalizer(object):
    """Class to perform the block diagonalization algorithm in a joint
    transmission scenario.

    In the block diagonalization algorithm either a single base station
    with more antennas transmits for multiple users at the same time or a
    group of base stations acts as a single transmitter to send data to the
    multiple users at the same time. In both cases the block
    diagonalization algorithm assures that each receiver does not see
    interference from the other receivers.

    The waterfilling algorithm is also applied to optimally distribute the
    power. However, in the case with multiple base stations, the power
    restriction in each base station must be respected. Therefore, after
    the power is optimally allocated at each base station all powers will
    be normalized to respect the power restriction of the base transmitting
    the highest energy. This is what is done in the
    :meth:`block_diagonalize` method.

    If the power should not be optimally allocated with the waterfilling
    algorithm, use the :meth:`block_diagonalize_no_waterfilling` method
    instead. The power restriction in each base station will still be
    respected, but the base station will equally divide its power among the
    available dimensions. Note that the result will be similar to
    :meth:`block_diagonalize` in the high SNR regime.

    Parameters
    ----------
    num_users : int
        Number of users.
    iPu : float
        Power available for EACH user.
    noise_var : float
        Noise variance (power in linear scale).

    Examples
    --------

    Consider the case where we have 3 base station (BSs) jointly
    transmitting to 3 users, where each base station has a power of 1.5, the
    number of antennas (at each BS and at each receiver) is 2, and the
    noise variance is 1e-4. The channel can be block diagonalized for this
    scenario with

    >>> bs_power = 1.5
    >>> noise_var = 1e-4
    >>> num_users = 2
    >>> Ntx = 2  # Number of transmit antennas (per BS)
    >>> Nrx = 2  # Number of receive antennas (per user)
    >>> # Create the BlockDiaginalizer object
    >>> bd = BlockDiaginalizer(num_users, bs_power, noise_var)
    >>> channel = np.array([[-0.9834-0.0123j,  0.6503-0.3189j,  0.5484+1.7049j, -1.0891-0.1025j], [-0.5911-0.3055j, -0.6205+0.3375j, -0.7995+0.3723j,  0.7412-1.2537j], [-0.2732+0.475j , -0.4191+0.4019j,  0.1047-0.5592j,  0.7548-1.0214j], [ 0.5377-0.208j , -0.1480-1.0527j, -0.6373+0.4081j, -0.5854-0.8135j]])
    >>> (newH, Ms) = bd.block_diagonalize(channel)

    We can see that the equivalent channel (after applying the Ms
    modulation matrix) is really block diagonalized.

    >>> print (newH + 1e-10 + 1e-10j).round(4)
    [[ 0.0916+0.0135j -1.7449-0.4328j  0.0000+0.j      0.0000+0.j    ]
     [-0.0114-0.146j   0.0213-1.1366j  0.0000+0.j      0.0000+0.j    ]
     [ 0.0000+0.j      0.0000+0.j      0.0868+0.1565j -0.3673+0.2289j]
     [ 0.0000+0.j      0.0000+0.j     -0.0396+0.0407j  1.0240+0.8997j]]

    Notice how the power restriction of each BS is respected (although only
    one BS will transmit with its maximum power).

    >>> print (np.linalg.norm(Ms[:,0:Ntx])**2).round(4)
    1.4997
    >>> print (np.linalg.norm(Ms[:,Ntx:])**2).round(4)
    1.5


    Notes
    -----
    The block diagonalization algorithm is described in [1]_, where
    different power allocations are illustrated. The
    :class:`BlockDiaginalizer` class implement two power allocation
    methods, a global power allocation, and a 'per transmitter' power
    allocation.

    .. [1] Q. H. Spencer, A. L. Swindlehurst, and M. Haardt,
       "Zero-Forcing Methods for Downlink Spatial Multiplexing
       in Multiuser MIMO Channels," IEEE Transactions on Signal
       Processing, vol. 52, no. 2, pp. 461–471, Feb. 2004.

    """

    def __init__(self, num_users, iPu, noise_var):
        """Initialize the BlockDiaginalizer object.

        Parameters
        ----------
        num_users : int
            Number of users.
        iPu : float
            Power available for EACH user.
        noise_var : float
            Noise variance (power in linear scale).
        """
        self.num_users = num_users
        self.iPu = iPu
        self.noise_var = noise_var  # Noise power is used in the
                                    # waterfilling calculations

    def _calc_BD_matrix_no_power_scaling(self, mtChannel):
        """Calculates the modulation matrix "M" that block diagonalizes the
        channel `mtChannel`, but without any king of power scaling.

        The "modulation matrix" is a matrix that changes the channel to a
        block diagonal structure and it is the first part in the Block
        Diagonalization algorithm. The returned modulation matrix is
        equivalent to Equation (12) of [1]_ but without the power scaling
        matrix $\Lambda$. Therefore, for the complete BD algorithm it is
        still necessary to perform this power scalling in the output of
        _calc_BD_matrix_no_power_scaling.

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

        Notes
        -----
        The reason why the Block Diagonalization algorithm was broken down
        into the code here and the power scaling code is because the power
        scaling may changing depending on the scenario. For instance, if
        the transmitter corresponds to a single base station the the power
        may be distributed into all the dimensions of the Modulation
        matrix. On the other hand, if the transmitter corresponds to
        multiple base stations jointly transmitting to multiple users then
        the power of each base station must be distributed only into the
        dimensions corresponding to that base station.
        """
        iNr = mtChannel.shape[0]
        assert iNr % self.num_users == 0, "`block_diagonalize`: Number of rows of the channel must be a multiple of the number of users."

        # Number of antennas per user
        iNrU = iNr // self.num_users

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Ms_bad = []
        Sigma = []

        # Calculates the interfering channels $\tilde{\mat{H}}_j$ as well
        # as $\tilde{\mtV}_j^{(1)}$ and $\tilde{\mtV}_j^{(0)}$.
        # Note that $\tilde{\mat{H}}_j = \tilde{\mtU}_j \tilde{\Sigma}_j [\tilde{\mtV}_j^{(1)} \; \tilde{\mtV}_j^{(0)}]^H$ where $\tilde{\mtV}_j^{(1)}$ holds
        # the first $\tilde{L}_j$ right singular vectors and $\tilde{\mtV}_j^{(0)}$ holds the
        # last $(n_T - \tilde{L}_j)$ right singular values
        for user in range(0, self.num_users):
            # channel of all users except the current user
            tilde_H_cur_user = self._get_tilde_channel(mtChannel, user)

            # How many streams user `user` can receive is given by the
            # total number of receive antennas minus the rank of
            # tilde_H_cur_user
            nStreams = iNr - np.linalg.matrix_rank(tilde_H_cur_user)
            tilde_V0 = least_right_singular_vectors(
                tilde_H_cur_user,
                nStreams)[0]

            # The equivalent channel of the current user corresponds to
            # $\mtH_j \tilde{\mtV}_j^{(0)}$

            # First we get $\mtH_j$
            H_cur_user = self._get_sub_channel(mtChannel, user)

            # Now we get the right singular value of the equivalent channel
            (_, V1, S) = least_right_singular_vectors(
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
        total_power = self.num_users * self.iPu
        vtOptP = waterfilling.doWF(Sigma ** 2,
                                   total_power,
                                   self.noise_var)[0]
        # print "Darlan"
        # print vtOptP
        # print "Cav"

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
        iNtU = Sigma.size / float(self.num_users)

        # First we perform the global waterfilling
        Ms_good = self._perform_global_waterfilling_power_scaling(
            Ms_bad, Sigma)

        # Since we used a global power constraint but we have in fact a
        # power constraint for each AP, we need to normalize the allocated
        # powers by the power of the AP with most energy (so that the
        # individual power constraints are satisfied). This will be
        # sub-optimal for the other bases, but it is what we can do.
        max_sqrt_P = 0
        for user in range(0, self.num_users):
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

        For an example, see the documentation of the
        :class:`BlockDiaginalizer` class.

        Parameters
        ----------
        mtChannel : 2D numpy array
            Channel from (all) the transmitter(s) to all users.

        Returns
        -------
        (newH, Ms_good) : A tuple of numpy arrays
            newH is a 2D numpy array corresponding to the Block
            diagonalized channel, while Ms_good is a 2D numpy array
            corresponding to the precoder matrix used to block diagonalize
            the channel.

        See also
        --------
        block_diagonalize_no_waterfilling

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

    def block_diagonalize_no_waterfilling(self, mtChannel):
        """Performs the block diagonalization, but without applying the
        waterfilling algorithm.

        The power of each base station is equally divided such that the
        square of the Frobenius norm or the columns of Ms_good
        corresponding to that base station is equal to its power.

        Parameters
        ----------
        mtChannel : 2D numpy array
            Channel from (all) the transmitter(s) to all users.

        Returns
        -------
        (newH, Ms_good) : A tuple of numpy arrays
            newH is a 2D numpy array corresponding to the Block
            diagonalized channel, while Ms_good is a 2D numpy array
            corresponding to the precoder matrix used to block diagonalize
            the channel.

        See also
        --------
        block_diagonalize

        """
        # This only works of the number of transmit antennas is the same
        # for all transmitters.
        iNtU = mtChannel.shape[1] / self.num_users

        # Calculates the modulation matrix and the singular values of the
        # effective channel when this modulation matrix is applied.
        (Ms_bad, Sigma) = self._calc_BD_matrix_no_power_scaling(mtChannel)

        # Scale the power of this modulation assuring the power restriction
        # is not violated in any of the base stations.
        Ms_good = np.empty(Ms_bad.shape, dtype=complex)
        for user in range(0, self.num_users):
            # Calculate the Frobenius norm of the matrix corresponding to
            # the transmitter `user`
            user_matrix = Ms_bad[:, user * iNtU:user * iNtU + iNtU]
            # The power is actually the square of cur_sqrt_P
            cur_sqrt_P = np.linalg.norm(user_matrix, 'fro')
            Ms_good[:, user * iNtU:user * iNtU + iNtU] = user_matrix * np.sqrt(self.iPu) / cur_sqrt_P

        # Ms_good = self._perform_normalized_power_scaling(Ms_bad,
        #                                                  Sigma)

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
        """
        Return the combined channel of all users except `user` .

        Let $k$ be the index for `user`. If the channel from all
        transmitters to receiver $k$ is $\mtH_k$, then this method returns
        $\tilde{\mtH_k} = [\mtH_1^T, \ldots, \mtH_{k-1}^T, \mtH_{k+1}^T, \ldots, \mtH_K]^T$.

        Parameters
        ----------
        mt_channel : 2D numpy array
            Channel of all users

        user : int
            Index of the user.
        """
        vtAllUserIndexes = np.arange(0, self.num_users)
        desiredUsers = [i for i in vtAllUserIndexes if i != user]
        return self._get_sub_channel(mtChannel, desiredUsers)

    def _get_sub_channel(self, mt_channel, desired_users):
        """Get a subchannel according to the desired_users vector.

        Parameters
        ----------
        mt_channel : 2D numpy array
            Channel of all users
        desired_users : iterable of integers
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
        >>> BD._get_sub_channel(channel, [0,2])
        array([[ 1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.],
               [ 3.,  3.,  3.,  3.,  3.,  3.],
               [ 3.,  3.,  3.,  3.,  3.,  3.]])
        >>> BD._get_sub_channel(channel, 0)
        array([[ 1.,  1.,  1.,  1.,  1.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.]])

        """
        nrows = mt_channel.shape[0]
        iNrU = nrows // self.num_users  # Number of receive antennas per user

        if isinstance(desired_users, collections.Iterable):
            vtIndexes = []
            for index in desired_users:
                vtIndexes.extend(range(iNrU * index, (index + 1) * iNrU))
        else:
            vtIndexes = range(iNrU * desired_users, (desired_users + 1) * iNrU)
        return mt_channel[vtIndexes, :]


class EnhancedBD(BlockDiaginalizer):
    """Performs the Coordinated Multipoint transmission also taking into
    account the external interference.

    The EnhancedBD class performs the block diagonalization characteristic
    to the joint transmission scenario where multiple base stations act as
    a single transmitter to send data to the users. However, in addition to
    what the BlockDiaginalizer class does the EnhancedBD class can also
    take external interference into account.

    One way to reduce of eliminate the external interference is to
    sacrifice streams in directions strongly occupied by the external
    interference.

    Notes
    -----
    See the :class:`BlockDiaginalizer` class for details about the block
    diagonalization process.

    """

    def __init__(self, num_users, iPu, noise_var, pe):
        """Initializes the EnhancedBD object.

        Parameters
        ----------
        num_users : int
            Number of users.
        iPu : float
            Power available for EACH user (in linear scale).
        noise_var : float
            Noise variance (power in linear scale).
        pe : float
            Power of the external interference source (in linear scale)
        """
        BlockDiaginalizer.__init__(self, num_users, iPu, noise_var)
        self.pe = pe

        # Function used to decide how many streams will be sacrificed to
        # mitigate external interference. This is set in the
        # set_ext_int_handling_metric method (as well as the _modulator and
        # _packet_length attributes)
        self._metric_func = None  # The default metric will be None

        self._metric_func_name = 'None'

        # Extra arguments that will be passed to the self._metric_func when
        # it is called.
        self._metric_func_extra_args = {}

    def set_ext_int_handling_metric(self, metric,
                                    metric_func_extra_args_dict={}):
        """Set the metric used to decide how many streams to sacrifice for
        external interference handling.

        The modification to the standard Block Diagonalization algorithm
        performed in this class consists in avoid transmit data in the
        subspace strongly occupied by the external interference source.

        This sacrificing (not transmitting) of streams may or may not be
        worth it and different metrics can be used to decide this.

        The valid values for the `metric` argument are 'None' (python None
        object), "fixed", "naive", "capacity" and "effective_throughput".
        Each of these values will impact on how the number of transmit
        streams is chosen and which subspace is actually used for the
        desired signal.

        For the "fixed" and "naive" metrics, the number of transmit streams
        is determined by the value of the 'num_streams' key in the
        metric_func_extra_args_dict. The difference between them is how the
        subspace where the useful data is determined (for the given number
        of sacrificed streams).

        For the "naive" metric, the stream reduction is performed by
        multiplying the usual block diagonalizing matrix M by a subset of
        the identity matrix. For the "fixed" metric the subspace containing
        the lowest remaining external interference energy is chosen by
        multiplying the block diagonalizing matrix M by the singular vectors
        of the external interference covariance matrix corresponding to the
        lowest singular values. The same procedure is used for the other
        metrics.

        Differently from the "naive" and "fixed" metrics, the "capacity"
        and "effective_throughput" metrics try to determine this best
        number of sacrificed streams.

        - If `metric` is None, then all streams will be used. That is, no
          streams will be sacrificed and the external interference won't be
          mitigated.

        - If metric is "None" or "naive" then the specified number of
          streams will be used.

        - If `metric` is 'capacity', then the metric used to decide how
          many streams to sacrifice will be the sum capacity. The function
          :meth:`._calc_shannon_sum_capacity` will be used to calculate the
          sum capacity metric, and since it only uses the SINR values, no
          extra arguments are required in the metric_func_extra_args_dict
          dictionary.

        - If `metric` is 'effective_throughput' then the metric used to
          decide how many streams to sacrifice will be the effective
          throughput that can be obtained. The function
          :meth:`._calc_effective_throughput` will be used to calculate the
          effective throughput. Since it requires the a modulator and a
          packet length you should set the metric_func_extra_args_dict so
          that it has the keys 'modulator' and 'packet_length' with the
          correct values (a modulator object and an integer, respectively)

        Parameters
        ----------
        metric : str, {None, 'capacity', 'effective_throughput'}
            The metric name. Must be one of the available metrics.
        metric_func_extra_args_dict : dict
            A dictionary containing the extra arguments that must be passed
            to the metric function. For the "naive" and "fixed" metrics,
            this dictionary must contain the "num_streams" keyword with the
            desired number of transmit streams. For the
            "effective_throughput" metric this dictionary mst contain the
            "modulatro" and "packet_length" keywords with a modulator
            object and an integer, respectivelly. For the other metrics
            metric_func_extra_args_dict will be ignored.

        Raises
        ------
        AttributeError
            If the metric is not one of the available metrics or if the
            metric_func_extra_args_dict does not contain the required
            keywords.

        """
        if metric is None or metric == 'None':
            self._metric_func_name = 'None'
            self._metric_func = None
            self._metric_func_extra_args = {}

        elif metric == 'capacity':
            self._metric_func_name = 'capacity'
            self._metric_func = self._calc_shannon_sum_capacity
            self._metric_func_extra_args = {}

        elif metric == 'naive':
            self._metric_func_name = 'naive'
            self._metric_func = None
            if 'num_streams' not in metric_func_extra_args_dict.keys():
                raise AttributeError("The 'naive' metric requires that metric_func_extra_args_dict is provided and has the 'num_streams' key")

            # Set self._metric_func_extra_args as a dictionary containing
            # the 'num_stream' key (and value) in
            # metric_func_extra_args_dict
            self._metric_func_extra_args = {k: metric_func_extra_args_dict[k]
                                            for k in ('num_streams',)}
            self._metric_func_extra_args = metric_func_extra_args_dict

        elif metric == 'fixed':
            self._metric_func_name = 'fixed'
            self._metric_func = None
            if 'num_streams' not in metric_func_extra_args_dict.keys():
                raise AttributeError("The 'fixed' metric requires that metric_func_extra_args_dict is provided and has the 'num_streams' key")  # pragma: no cover

            # Set self._metric_func_extra_args as a dictionary containing
            # the 'num_stream' key (and value) in
            # metric_func_extra_args_dict
            self._metric_func_extra_args = {k: metric_func_extra_args_dict[k]
                                            for k in ('num_streams',)}

        elif metric == 'effective_throughput':
            self._metric_func_name = 'effective_throughput'
            self._metric_func = self._calc_effective_throughput
            keys = metric_func_extra_args_dict.keys()
            if ('modulator' not in keys) or ('packet_length' not in keys):
                raise AttributeError("The 'effective_throughput' metric requires that metric_func_extra_args_dict is provided and has the 'modulator' and package_length' keys")

            # Set self._metric_func_extra_args as a dictionary containing
            # the 'modulator' and 'packet_length' keys (and values) in
            # metric_func_extra_args_dict
            self._metric_func_extra_args = {k: metric_func_extra_args_dict[k]
                                            for k in ('modulator', 'packet_length')}
        else:
            raise AttributeError("The `metric` attribute can only be one of {None, 'capacity', 'effective_throughput'}")

    def _get_metric_name(self):
        """Get name of the method used to decide how many streams to
        sacrifice.
        """
        return self._metric_func_name
    metric_name = property(_get_metric_name)

    @staticmethod
    def calc_receive_filter_user_k(Heq_k_P, P=None):
        """Calculates the Zero-Forcing receive filter of a single user `k`
        with or without the stream reduction.

        Parameters
        ----------
        Heq_k_P : 2D numpy array
            The equivalent channel of user `k` after the block
            diagonalization process and any stream reduction.
        P : 2D numpy array
            P has the most significant singular vectors of the external
            interference plus noise covariance matrix for each
            receiver.

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
        mitigate the external interference.

        """
        if P is None:
            W = np.linalg.pinv(Heq_k_P)
        else:
            overbar_P = calcProjectionMatrix(P)

            # Calculate the equivalent channel including the stream
            # reduction
            #Heq_k_red = np.dot(Heq_k, P)
            W = np.dot(
                np.linalg.pinv(np.dot(overbar_P, Heq_k_P)),
                overbar_P)

        return W

    # NOTE: PROBABLY, THIS IS ONLY VALID FOR SPATIAL MULTIPLEXING.
    @staticmethod
    def _calc_linear_SINRs(Heq_k_red, Wk, Re_k):
        """Calculates the effective SINRs of each parallel channel.

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
        >>> print(EnhancedBD._calc_shannon_sum_capacity(sinrs_linear))
        8.04504974084
        """
        sum_capacity = np.sum(np.log2(1 + sinrs))

        return sum_capacity

    @staticmethod
    def _calc_effective_throughput(sinrs, modulator, packet_length):
        """Calculates the effective throughput of the values in `sinrs` considering
        the given modulator and packet_length.

        The effective throughput is equivalent to the packet error for a
        specific packet error rate and packet length, times the nominal
        throughput.

        Parameters
        ----------
        sinrs : 1D numpy array or float
            SINR values (in linear scale).
        modulator : A modulator object.
            A modulator object such as M-PSK, M-QAM, etc. See the
            :mod:`.modulators` module.
        packet_length: int
            The package length. That is, the number of bits in each
            package.

        Returns
        -------
        effective_throughput : float
            Effective throughput that can be obtained.

        """
        SINRs = linear2dB(sinrs)
        se = modulator.calcTheoreticalSpectralEfficiency(SINRs, packet_length)
        total_se = np.sum(se)
        return total_se

    def _perform_BD_no_waterfilling_no_stream_reduction(self, mu_channel):
        """Function called inside perform_BD_no_waterfilling when no stream
        reduction should be performed.

        """
        # xxxxxxxxxx Some initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        K = mu_channel.K
        Nr = mu_channel.Nr
        Nt = mu_channel.Nt
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Output variables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # No matter which metric is used, these variables should be set and
        # returned
        MsPk_all_users = np.empty(K, dtype=np.ndarray)
        Wk_all_users = np.empty(K, dtype=np.ndarray)
        Ns_all_users = np.empty(K, dtype=int)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Since we are not handling external interference, we simple call
        # the block_diagonalize_no_waterfilling method from the
        # BlockDiaginalizer class.
        (newH, Ms_good) = BlockDiaginalizer.block_diagonalize_no_waterfilling(self, mu_channel.big_H_no_ext_int)

        # Since there is no stream reduction, the number of streams of each
        # user will transmit is equal to the number of transmit antennas of
        # that user
        Ns_all_users = Nt
        MsPk_all_users = single_matrix_to_matrix_of_matrices(Ms_good, None, Nt)
        newH_all_k = single_matrix_to_matrix_of_matrices(newH, Nr, Nt)
        for userindex in range(K):
            Wk_all_users[userindex] = self.calc_receive_filter_user_k(
                newH_all_k[userindex, userindex], None)

        return (MsPk_all_users, Wk_all_users, Ns_all_users)

    def _perform_BD_no_waterfilling_fixed_or_naive_reduction(self, mu_channel):
        """Function called inside perform_BD_no_waterfilling when the naive or
        the fixed stream reduction should be performed.

        For the naive or the fixed stream reduction cases the number of
        transmitted streams is always equal to
        self._metric_func_extra_args['num_streams']. That is, the number of
        sacrificed streams is equal to the number of transmit antennas
        minus num_streams.

        The only difference between the naive and the fixed cases is that
        in the fixed case the reduction matrix P is chosen so that it gets
        as orthogonal to the external interference as possible, while the
        naive case simple chooses P as a submatrix of the diagonal matrix.

        """
        # xxxxxxxxxx Some initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        K = mu_channel.K
        Nr = mu_channel.Nr
        Nt = mu_channel.Nt
        H_matrix = mu_channel.big_H_no_ext_int
        Re = mu_channel.calc_cov_matrix_extint_plus_noise(
            self.noise_var, self.pe)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Output variables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # No matter which metric is used, these variables should be set and
        # returned
        MsPk_all_users = np.empty(K, dtype=np.ndarray)
        Wk_all_users = np.empty(K, dtype=np.ndarray)
        Ns_all_users = np.empty(K, dtype=int)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Ms_bad, Sigma = self._calc_BD_matrix_no_power_scaling(H_matrix)
        Ms_bad_ks = single_matrix_to_matrix_of_matrices(Ms_bad, None, Nt)
        H_all_ks = single_matrix_to_matrix_of_matrices(H_matrix, Nr)

        # Number of streams (per user) that will be transmitted. This
        # should be greater than 0 and lower than the number of transmit
        # antennas of each user
        num_streams = self._metric_func_extra_args['num_streams']
        # Loop for the users
        for userindex in range(K):
            Ntk = Nt[userindex]
            Hk = H_all_ks[userindex]
            Msk = Ms_bad_ks[userindex]
            Re_k = Re[userindex]

            # Equivalent channel of user k after the block diagonalization
            # process, but without any stream reduction
            Heq_k = np.dot(Hk, Msk)

            if self.metric_name == 'naive':
                Pk = np.eye(Ntk)[:, 0:num_streams]
            if self.metric_name == 'fixed':
                Pk = _calc_stream_reduction_matrix(Re_k, num_streams)

            norm_term = np.linalg.norm(np.dot(Msk, Pk), 'fro') / np.sqrt(self.iPu)
            # Equivalent channel with stream reduction
            Heq_k_red = np.dot(Heq_k, Pk / norm_term)

            W_k = self.calc_receive_filter_user_k(Heq_k_red, Pk)

            # Save results
            MsPk_all_users[userindex] = np.dot(Msk, Pk) / norm_term
            Wk_all_users[userindex] = W_k
            Ns_all_users[userindex] = num_streams

        return (MsPk_all_users, Wk_all_users, Ns_all_users)

    def _perform_BD_no_waterfilling_decide_number_streams(self, mu_channel):
        """Function called inside perform_BD_no_waterfilling when the stream
        reduction is performed and the number of sacrificed streams depend
        on the metric used (the function set as self._metric_func)

        """
        # xxxxxxxxxx Some initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        K = mu_channel.K
        Nr = mu_channel.Nr
        Nt = mu_channel.Nt
        H_matrix = mu_channel.big_H_no_ext_int
        Re = mu_channel.calc_cov_matrix_extint_plus_noise(
            self.noise_var, self.pe)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Output variables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # No matter which metric is used, these variables should be set and
        # returned
        MsPk_all_users = np.empty(K, dtype=np.ndarray)
        Wk_all_users = np.empty(K, dtype=np.ndarray)
        Ns_all_users = np.empty(K, dtype=int)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        Ms_bad, Sigma = self._calc_BD_matrix_no_power_scaling(H_matrix)

        # The k-th 'element' in Ms_bad_ks is a matrix containing the
        # columns of Ms_bad of the k-th user.
        Ms_bad_ks = single_matrix_to_matrix_of_matrices(Ms_bad, None, Nt)
        H_all_ks = single_matrix_to_matrix_of_matrices(H_matrix, Nr)

        # Loop for the users
        for userindex in range(K):
            Ntk = Nt[userindex]
            Rek = Re[userindex]
            Hk = H_all_ks[userindex]
            Msk = Ms_bad_ks[userindex]

            # Equivalent channel of user k after the block diagonalization
            # process, but without any stream reduction
            Heq_k = np.dot(Hk, Msk)

            # We can have from a single stream to all streams (the number
            # of transmit antennas). This loop varies the number of
            # transmit streams of user k.
            #
            # The metric that will be calculated and used to determine how
            # many streams to sacrifice.
            metric_value_for_user_k = np.zeros(Ntk)
            Pk_all = np.empty(Ntk, dtype=np.ndarray)
            norm_term_all = np.empty(Ntk)
            Wk_all = np.empty(Ntk, dtype=np.ndarray)
            for index in range(Ntk):
                Ns_k = index + 1
                # xxxxx Find Pk xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
                # Note that if index is equal to Ntk - 1 that means that no
                # stream reduction will be performed. In that case, we set
                # Pk as the identity matrix, since setting it as the
                # singular vectors of the external interference covariance
                # matrix with _calc_stream_reduction_matrix won't help (may
                # even get worse results)
                if index == Ntk - 1:
                    Pk = np.eye(Ntk)
                else:
                    Pk = _calc_stream_reduction_matrix(Rek, Ns_k)
                # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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
                W_k = self.calc_receive_filter_user_k(Heq_k_red, Pk)
                Wk_all[index] = W_k

                # SINR (in linear scale) of all streams of user k.
                sinrs_k = self._calc_linear_SINRs(Heq_k_red, W_k, Rek)

                metric_value_for_user_k[index] = self._metric_func(
                    sinrs_k,
                    # Use use the '**' magic to pass the values in the
                    # self._metric_func_extra_args dictionary as the
                    # arguments of the metric function.
                    ** self._metric_func_extra_args)

            # The index with the highest metric value. This is equivalent
            # to the number of transmit streams which yields the highest
            # value of the metric.
            best_index = np.argmax(metric_value_for_user_k)
            MsPk_all_users[userindex] = np.dot(Msk, Pk_all[best_index]) / norm_term_all[best_index]
            Wk_all_users[userindex] = Wk_all[best_index]
            Ns_all_users[userindex] = Pk_all[best_index].shape[1]

        return (MsPk_all_users, Wk_all_users, Ns_all_users)

    def block_diagonalize_no_waterfilling(self, mu_channel):
        """Perform the block diagonalization of `mu_channel` taking the
        external interference into account.

        This is the main method calculating the BD algorithm. Two
        important parameters used here are the noise variance (an attribute
        of the `mu_channel` object) and the external interference power
        (the `pe` attribute) attributes.

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
        Wk_all_users : 1D numpy array of 2D numpy arrays
            A 1D numpy array where each element corresponds to the receive
            filter for a user.
        Ns_all_users: 1D numpy array of ints
            Number of streams of each user.

        """
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Case where no stream reduction is performed xxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # The value of self._metric_func_extra_args is not used for this
        # case
        if self._metric_func_name == "None":
            return self._perform_BD_no_waterfilling_no_stream_reduction(
                mu_channel)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxx Case where the naive stream reduction is performed xxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        if (self._metric_func_name == "naive" or
           self._metric_func_name == "fixed"):

            return self._perform_BD_no_waterfilling_fixed_or_naive_reduction(
                mu_channel)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Case where self._metric_func is used xxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # In this case the number of sacrified streams is automatically
        # determined according to the self._metric_func function. This
        # function is set in the set_ext_int_handling_metric method, where
        # any extra arguments (besides sinr) that should be passed to this
        # function are set in the _metric_func_extra_args dictionary.
        return self._perform_BD_no_waterfilling_decide_number_streams(mu_channel)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
## xxxxxxxxxx Module functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def block_diagonalize(mtChannel, num_users, iPu, noise_var):
    """Performs the block diagonalization of :attr:`mtChannel`.

    Parameters
    ----------
    mtChannel : 2D numpy array
        Global channel matrix
    num_users : int
        Number of users
    iPu : float
        Power available for each user
    noise_var : float
        Noise variance

    Returns
    -------
    (newH, Ms_good) : A tuple of numpy arrays
        newH is a 2D numpy array corresponding to the Block
        diagonalized channel, while Ms_good is a 2D numpy array
        corresponding to the precoder matrix used to block diagonalize
        the channel.

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
    BD = BlockDiaginalizer(num_users, iPu, noise_var)
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
