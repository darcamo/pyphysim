#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: disable=R0902,R0904

"""
Module with implementation of Interference Alignment (IA) algorithms.

Note that all IA algorithms require the channel object and any change to
the channel object must be performed before calling the `solve` method of
the IA algorithm object. This includes generating the channel and setting
the noise variance.
"""

__revision__ = "$Revision$"

import numpy as np
from scipy import optimize
import itertools

from ..util.misc import peig, leig, randn_c_RS, update_inv_sum_diag, \
    get_principal_component_matrix
from ..comm import channels

__all__ = ['AlternatingMinIASolver', 'MaxSinrIASolver',
           'MinLeakageIASolver', 'ClosedFormIASolver']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Base Class for all IA Algorithms xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class IASolverBaseClass(object):
    """
    Base class for all Interference Alignment Algorithms.

    At least the `_updateW`, `_updateF` and `solve` methods must be
    implemented in the subclasses of IASolverBaseClass, where the `solve`
    method uses the `_updateW` and `_updateF` methods in its
    implementation.

    The implementation of the `_updateW` method should call the
    `_clear_receive_filter` method in the beginning and after that set
    either the _W or the _W_H variables with the correct value.

    The implementation of the `_updateF` method should call the
    clear_precoder_filter in the beginning and after that set _W variable
    with the correct precoder (normalized to have a Frobenius norm equal to
    one.

    The implementation of the `_updateF` method must set the _F variable
    with the correct value.

    Another method that can be implemented is the get_cost method. It should
    return the cost of the current IA solution. What is considered "the
    cost" varies from one IA algorithm to another, but should always be a
    real non-negative number. If get_cost is not implemented a value of -1
    is returned.

    Parameters
    ----------
    multiUserChannel : A MultiUserChannelMatrix object.
        The multiuser channel.
    """
    def __init__(self, multiUserChannel):
        """Initialize the variables that every IA solver will have.

        Parameters
        ----------
        multiUserChannel : A MultiUserChannelMatrix object.
            The multiuser channel.
        """
        # xxxxxxxxxx Private attributes xxxxxxxxxxxxxxx
        if not isinstance(multiUserChannel, channels.MultiUserChannelMatrix):
            raise ValueError("multiUserChannel must be an object of the"
                             " comm.channels.MultiUserChannelMatrix class"
                             " (or a subclass).")
        # Channel of all users
        self._multiUserChannel = multiUserChannel

        # Number of streams per user
        self._Ns = None

        self._P = None  # Power of each user (P is an 1D numpy array). If
                        # not set (_P is None), then a power of 1 will be
                        # used for each transmitter.
        self._noise_var = None  # If None, then the value of last_noise_var
                                # in the multiUserChannel object will be
                                # used.

        # Precoder and receive filters (numpy arrays of numpy arrays)
        self._F = None  # Precoder: One precoder for each user
        self._full_F = None  # Precoder: Same as _F, but scaled with the
                             # correct power value in self.P
        self._W = None  # Receive filter: One for each user
        self._W_H = None  # Receive filter: One for each user
        self._full_W_H = None
        self._full_W = None

        # Other member variables
        self._rs = np.random.RandomState()  # RandomState object used to
                                            # randomize the precoder

    def _clear_receive_filter(self):
        """
        Clear the receive filter.

        This should be called in the beginning of the implementation of the
        `_updateW` method in subclasses.
        """
        self._W = None
        self._W_H = None
        self._full_W_H = None
        self._full_W = None

    def _clear_precoder_filter(self):
        """
        Clear the precoder filter.

        This should be called in the beginning of the implemetnation of the
        `_updateF` method in subclasses.
        """
        self._F = None
        self._full_F = None

    def clear(self):
        """
        Clear the IA Solver object.

        All member attributes that are updated during the solve method,
        such as the precoder and receive filters, will be cleared. The
        other attributes that correspond to "configuration" such as the
        channel object won't be changed.

        Notes
        -----
        You should overwrite this method in subclasses that pass parameters
        to the __init__ method, since here we call __init__ without
        arguments which is probably not what you want.
        """
        # The F and W variables will be numpy arrays of numpy arrays.
        self._clear_precoder_filter()  # Set _F and _full_F to None
        self._clear_receive_filter()  # Set _W, _W_H and _full_W_H to None
        self._P = None
        self._Ns = None

        # Don't clear the self._noise_var attribute

    def get_cost(self):  # pylint: disable=R0201
        """
        Get the current cost of the IA Solution.

        This method should be implemented in subclasses and return a number
        greater than or equal to zero..

        Returns
        -------
        cost : float (real non-negative number)
            The Cost of the current IA solution.
        """
        return -1

    @property
    def noise_var(self):
        """Get method for the noise_var property."""
        if self._noise_var is None:
            return self._multiUserChannel.last_noise_var
        else:
            return self._noise_var

    @noise_var.setter
    def noise_var(self, value):
        """Set method for the noise_var property."""
        assert value >= 0.0, "Noise variance must be >= 0."
        self._noise_var = value

    @property
    def F(self):
        """Transmit precoder of all users."""
        return self._F

    @property
    def full_F(self):
        """Transmit precoder of all users."""
        if self._full_F is None:
            self._full_F = self._F * np.sqrt(self.P)
        return self._full_F

    # The W property should return a receive filter that can be directly
    # multiplied (no need to calculate the hermitian of W) by the received
    # signal to cancel interference and compensate the effect of the
    # channel.
    @property
    def W(self):
        """Receive filter of all users."""
        # If self._W is None but self._W_H is not None than we need to
        # update self_W from self._W_H.
        if self._W is None:
            if self._W_H is not None:
                self._W = np.empty(self.K, dtype=np.ndarray)
                for k in range(self.K):
                    self._W[k] = self._W_H[k].conj().T
        return self._W

    @property
    def W_H(self):
        """Get method for the W_H property."""
        # If self._W_H is None but self._W is not None than we need to
        # update self_W_H from self._W.
        if self._W_H is None:
            if self._W is not None:
                self._W_H = np.empty(self.K, dtype=np.ndarray)
                for k in range(self.K):
                    self._W_H[k] = self._W[k].conj().T
        return self._W_H

    @property
    def full_W_H(self, ):
        """
        Get method for the full_W_H property.

        The full_W_H property returns the equivalent filter of the IA
        filter plus the post processing filter.
        """
        if self._full_W_H is None:
            if self.W_H is not None:
                self._full_W_H = np.empty(self.K, dtype=np.ndarray)
                for k in range(self.K):
                    # Equivalent channel with the effect of the precoder,
                    # channel and receive filter
                    Hieq = self._calc_equivalent_channel(k)
                    self._full_W_H[k] = np.linalg.solve(Hieq, self.W_H[k])

        return self._full_W_H

    @property
    def full_W(self, ):
        """
        Get method for the full_W property.

        The full_W property returns the equivalent filter of the IA
        filter plus the post processing filter.
        """
        if self._full_W is None:
            self._full_W = np.empty(self.K, dtype=np.ndarray)
            for k in range(self.K):
                self._full_W[k] = self.full_W_H[k].conj().T
        return self._full_W

    def _calc_equivalent_channel(self, k):
        """
        Calculates the equivalent channel for user :math:`k` considering the
        effect of the precoder (including transmit power), the actual
        channel, and the receive filter (without power compensation).

        Parameters
        ----------
        k : int
            The index of the desired user.

        Notes
        -----
        This method is used only internaly in order to calculate the "W"
        get property so that the returned filter W compensates the effect
        of the direct channel.
        """
        # Note that here Wk_H is the self.Wk_H property and not the
        # self.full_W_H property. Since _calc_equivalent_channel is used in
        # the full_W_H get property if we had used self.full_W_H here we
        # would get an infinity recursion.
        Wk_H = self.W_H[k]
        full_Fk = self.full_F[k]
        Hkk = self._get_channel(k, k)
        Hk_eq = Wk_H.dot(Hkk.dot(full_Fk))
        return Hk_eq

    @property
    def P(self):
        """Transmit power of all users.
        """
        if self._P is None:
            P = np.ones(self.K, dtype=float)
        else:
            P = self._P
        return P

    @P.setter
    def P(self, value):
        """Transmit power of all users.
        """
        if value is None:
            # Note that if self._P is None then the getter property will
            # return a numpy array of ones with the appropriated size.
            self._P = None
        elif np.isscalar(value):
            if value > 0.0:
                self._P = np.ones(self.K, dtype=float) * value
            else:
                raise ValueError("P cannot be negative or equal to zero.")
        else:
            if len(value) != self.K:
                raise ValueError("P must be set to a sequency of length K")
            else:
                value = np.array(value)
                if np.all(value > 0.0):
                    self._P = np.array(value)
                else:
                    raise ValueError("P cannot be negative or equal to zero.")

    @property
    def Ns(self):
        """Number of streams of all users.

        Returns
        -------
        Ns : 1D numpy array
            Number of streams of all users.
        """
        return self._Ns

    # xxxxx Properties to read the channel related variables xxxxxxxxxxxxxx
    @property
    def K(self):
        """The number of users.

        Returns
        -------
        K : int
            The number of users.
        """
        return self._multiUserChannel.K

    @property
    def Nr(self):
        """Number of receive antennas of all users.

        Returns
        -------
        Nr : 1D numpy array
            Number of receive antennas of all users.
        """
        return self._multiUserChannel.Nr

    @property
    def Nt(self):
        """Number of transmit antennas of all users.

        Returns
        -------
        Nt : 1D numpy array
            Number of transmit antennas of all users.
        """
        return self._multiUserChannel.Nt

    def randomizeF(self, Ns, P=None):
        """Generates a random precoder for each user.

        Parameters
        ----------
        Ns : int or 1D numpy array
            Number of streams of each user.
        P : 1D numpy array
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        self._clear_precoder_filter()

        if isinstance(Ns, int):
            Ns = np.ones(self.K, dtype=int) * Ns

        self.P = P

        # Lambda function that returns a normalized version of the input
        # numpy array
        normalized = lambda A: A / np.linalg.norm(A, 'fro')

        self._F = np.zeros(self.K, dtype=np.ndarray)
        for k in range(self.K):
            self._F[k] = normalized(randn_c_RS(self._rs, self.Nt[k], Ns[k]))
        #self._F = [normalized(randn_c(Nt[k], Ns[k])) for k in np.arange(0, K)]

        self._Ns = np.array(Ns)  # This will create a new array so that we
                                 # can modify self._Ns internally without
                                 # changing the original Ns variable passed
                                 # to the randomizeF method.

    # This method is just an alias for the get_channel method of the
    # multiuserchannel object associated with the IA Solver.xs
    def _get_channel(self, k, l):
        """Get the channel from transmitter l to receiver k.

        Parameters
        ----------
        l : int
            Transmitting user.
        k : int
            Receiving user.

        Returns
        -------
        H : 2D numpy array
            The channel matrix between transmitter l and receiver k.
        """
        return self._multiUserChannel.get_Hkl(k, l)

    def _get_channel_rev(self, k, l):
        """Get the channel from transmitter l to receiver k in the reverse
        network.

        Let the matrix :math:`\\mtH_{kl}` be the channel matrix between the
        transmitter :math:`l` to receiver :math:`k` in the direct
        network. The channel matrix between the transmitter :math:`l` to
        receiver :math:`k` in the reverse network, denoted as
        :math:`\\overleftarrow{\\mtH}_{kl}`, is then given by
        :math:`\\overleftarrow{\\mtH}_{kl} = \\mtH_{lk}^\\dagger` where
        :math:`\\mtA^\\dagger` is the conjugate transpose of :math:`\\mtA`.

        Parameters
        ----------
        l : int
            Transmitting user of the reverse network.
        k : int
            Receiving user of the reverse network.

        Returns
        -------
        H : 2D numpy array
            The channel matrix between transmitter l and receiver k in the
            reverse network.

        Notes
        -----
        See Section III of [Cadambe2008]_ for details.

        """
        return self._get_channel(l, k).transpose().conjugate()

    def calc_Q(self, k):
        """Calculates the interference covariance matrix at the
        :math:`k`-th receiver.

        The interference covariance matrix at the :math:`k`-th receiver,
        :math:`\\mtQ k`, is given by

            :math:`\\mtQ k = \\sum_{j=1, j \\neq k}^{K} \\frac{P_j}{Ns_j} \\mtH_{kj} \\mtF_j \\mtF_j^H \\mtH_{kj}^H`

        where :math:`P_j` is the transmit power of transmitter :math:`j`,
        and :math:`Ns_j` is the number of streams for user :math:`j`.

        Parameters
        ----------
        k : int
            Index of the desired receiver.

        Returns
        -------
        Qk : 2D numpy complex array.
            The interference covariance matrix at receiver :math:`k`.

        Notes
        -----
        This is impacted by the self.P attribute.

        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{kj} \mtF_j \mtF_j^H \mtH_{kj}^H$$
        # interfering_users = set(range(self.K)) - set([k])
        # Qk = np.zeros([self.Nr[k], self.Nr[k]], dtype=complex)

        # for l in interfering_users:
        #     Hkl_F = np.dot(
        #         self._get_channel(k, l),
        #         self.full_F[l])
        #     Qk = Qk + np.dot(Hkl_F, Hkl_F.transpose().conjugate())

        Qk = self._multiUserChannel.calc_Q(k, self.full_F)
        return Qk

    # This method must be tested in a subclass of IASolverBaseClass, since
    # we need the receive filter and IASolverBaseClass does not know how to
    # calculate it
    def calc_Q_rev(self, k):
        """Calculates the interference covariance matrix at the
        :math:`k`-th receiver in the reverse network.

        Parameters
        ----------
        k : int
            Index of the desired receiver.

        Returns
        -------
        Qk_rev : 2D numpy complex array.
            The interference covariance matrix at receiver :math:`k` in the
            reverse network.

        See also
        --------
        calc_Q
        """
        P = self.P
        interfering_users = set(range(self.K)) - set([k])
        Qk = np.zeros([self.Nt[k], self.Nt[k]], dtype=complex)

        for l in interfering_users:
            # The lets make sure the receive filter norm is equal to one so
            # that we can correctly scale it to the desired power.
            assert np.linalg.norm(self._W[l], 'fro') - 1.0 < 1e-6
            Hkl_F_rev = np.dot(
                self._get_channel_rev(k, l),
                self._W[l])
            Qk = Qk + np.dot(P[l] * Hkl_F_rev, Hkl_F_rev.conjugate().T)

        return Qk

    def calc_remaining_interference_percentage(self, k, Qk=None):
        """Calculates the percentage of the interference in the desired
        signal space according to equation (30) in [Cadambe2008]_.

        The percentage :math:`p_k` of the interference in the desired
        signal space is given by

            :math:`p_k = \\frac{\\sum_{j=1}^{Ns[k]} \\lambda_j [\\mtQ k]}{Tr[\\mtQ k]}`

        where :math:`\\lambda_j[\\mtA]` denotes the :math:`j`-th smallest
        eigenvalue of :math:`\\mtA`.

        Parameters
        ----------
        k : int
            The index of the desired user.
        Qk : 2D numpy complex array
            The covariance matrix of the remaining interference at receiver
            k. If not provided, it will be automatically calculated. In
            that case, the `P` attribute will also be taken into account if
            it is set.

        Notes
        -----
        `Qk` must be a symmetric matrix so that its eigenvalues are real and
        positive (any covariance matrix is a symmetric matrix).

        """
        # $$p_k = \frac{\sum_{j=1}^{Ns[k]} \lambda_j [\mtQ k]}{Tr[\mtQ k]}$$

        if Qk is None:
            Qk = self.calc_Q(k)

        [_, D] = leig(Qk, self.Ns[k])
        pk = np.sum(np.abs(D)) / np.trace(np.abs(Qk))
        return pk

    def calc_SINR_old(self):
        """
        Calculates the SINR values (in linear scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property, which, if not explicitly set, will use the
        last_noise_var property of the multiuserchannel object.

        This method is deprecated since it's not the correct way to
        calculate the SINR. Use the calc_SINR method instead.

        Returns
        -------
        SINRs : 1D numpy array of 1D numpy arrays (of floats)
            The SINR (in linear scale) of all streams of all users.
        """
        K = self.K
        SINRs = np.empty(K, dtype=np.ndarray)

        for j in range(K):
            numerator = 0.0
            denominator = 0.0
            Wj_H = self.W_H[j]
            for i in range(K):
                Hji = self._get_channel(j, i)
                Fi = self.F[i]
                aux = np.dot(Wj_H, np.dot(Hji, Fi))
                if i == j:
                    aux = np.dot(aux, aux.transpose().conjugate())
                    # Numerator will be a 1D numpy array with length equal
                    # to the number of streams
                    numerator = numerator + np.diag(np.abs(aux))
                else:
                    denominator = denominator + aux

            # pylint: disable=E1103
            denominator = np.dot(denominator,
                                 denominator.transpose().conjugate())
            noise_power = self.noise_var * np.dot(
                Wj_H, Wj_H.transpose().conjugate())
            denominator = denominator + noise_power
            denominator = np.diag(np.abs(denominator))

            SINRs[j] = numerator / denominator

        return SINRs

    def calc_SINR(self):
        """
        Calculates the SINR values (in linear scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property, which, if not explicitly set, will use the
        last_noise_var property of the multiuserchannel object.

        Returns
        -------
        SINRs : 1D numpy array of 1D numpy arrays (of floats)
            The SINR (in linear scale) of all streams of all users.
        """
        K = self.K
        SINRs = np.empty(K, dtype=np.ndarray)

        for k in range(self.K):
            Bkl_all_l = self._calc_Bkl_cov_matrix_all_l(k, self.noise_var)
            SINRs[k] = self._calc_SINR_k(k, Bkl_all_l)
        return SINRs

    def _calc_Bkl_cov_matrix_first_part(self, k):
        """Calculates the first part in the equation of the Blk covariance
        matrix in equation (28) of [Cadambe2008]_.

        The first part is given by

            :math:`\\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star d}^{[j]} \\mtV_{\\star d}^{[j]\\dagger} \\mtH^{[kj]\\dagger}`

        Note that it only depends on the value of :math:`k`.

        Parameters
        ----------
        k : int
            Index of the desired user.

        Returns
        -------
        Bkl_first_part : 2D numpy complex array
            First part in equation (28) of [Cadambe2008]_.

        """
        # $$\sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$$
        first_part = 0.0
        for j in range(self.K):
            Hkj = self._get_channel(k, j)
            Hkj_H = Hkj.conjugate().transpose()
            Vj = self.full_F[j]
            Vj_H = Vj.conjugate().transpose()

            first_part = first_part + np.dot(
                Hkj,
                np.dot(
                    np.dot(Vj,
                           Vj_H),
                    Hkj_H))

        return first_part

    def _calc_Bkl_cov_matrix_second_part(self, k, l):
        """Calculates the second part in the equation of the Blk covariance
        matrix in equation (28) of [Cadambe2008]_ (note that it does not
        include the identity matrix).

        The second part is given by

            :math:`\\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger}`

        Parameters
        ----------
        k : int
            Index of the desired user.
        l : int
            Index of the desired stream.

        Returns
        -------
        second_part : 2D numpy complex array.
            Second part in equation (28) of [Cadambe2008]_.

        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger}$$
        Hkk = self._get_channel(k, k)
        Hkk_H = Hkk.transpose().conjugate()

        Vkl = self.full_F[k][:, l:l + 1]
        Vkl_H = Vkl.transpose().conjugate()
        second_part = np.dot(Hkk,
                             np.dot(np.dot(Vkl, Vkl_H),
                                    Hkk_H))

        return second_part

    def _calc_Bkl_cov_matrix_all_l(self, k, noise_power=None):
        """Calculates the interference-plus-noise covariance matrix for all
        streams at receiver :math:`k` according to equation (28) in
        [Cadambe2008]_.

        The interference-plus-noise covariance matrix for stream :math:`l`
        of user :math:`k` is given by Equation (28) in [Cadambe2008]_,
        which is reproduced below

            :math:`\\mtB^{[kl]} = \\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star l}^{[j]} \\mtV_{\\star l}^{[j]\\dagger} \\mtH^{[kj]\\dagger} - \\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger} + \\mtI_{N^{[k]}}`

        where :math:`P^{[k]}` is the transmit power of transmitter
        :math:`k`, :math:`d^{[k]}` is the number of degrees of freedom of
        user :math:`k`, :math:`\\mtH^{[kj]}` is the channel between
        transmitter :math:`j` and receiver :math:`k`, :math:`\\mtV_{\\star
        l}` is the :math:`l`-th column of the precoder of user :math:`k`
        and :math:`\\mtI_{N^{k}}` is an identity matrix with size equal to
        the number of receive antennas of receiver :math:`k`.

        Parameters
        ----------
        k : int
            Index of the desired user.
        noise_power : float
            Noise power (variance).

        Returns
        -------
        Bkl : 1D numpy array of 2D numpy arrays
            Covariance matrix of all streams of user k. Each element of the
            returned 1D numpy array is a 2D numpy complex array
            corresponding to the covariance matrix of one stream of user k.

        Notes
        -----

        To be simple, a function that returns the covariance matrix of only
        a single stream "l" of the desired user "k" could be implemented,
        but in the order to calculate the max SINR algorithm we need the
        covariance matrix of all streams and returning them in single
        function as is done here allows us to calculate the first part in
        equation (28) of [Cadambe2008]_ only once, since it is the same for
        all streams.

        """
        # $$\mtB^{[kl]} = \sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star l}^{[j]} \mtV_{\star l}^{[j]\dagger} \mtH^{[kj]\dagger} - \frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger} + \mtI_{N^{[k]}}$$
        if noise_power is None:
            noise_power = self.noise_var

        Bkl_all_l = np.empty(self._Ns[k], dtype=np.ndarray)
        first_part = self._calc_Bkl_cov_matrix_first_part(k)
        for l in range(self._Ns[k]):
            second_part = self._calc_Bkl_cov_matrix_second_part(k, l)
            Bkl_all_l[l] = first_part - second_part + (
                noise_power * np.eye(self.Nr[k]))

        return Bkl_all_l

    def _calc_SINR_k(self, k, Bkl_all_l):
        """Calculates the SINR of all streams of user 'k'.

        Parameters
        ----------
        k : int
            Index of the desired user.
        Bkl_all_l : A sequence of 2D numpy arrays.
            A sequence (1D numpy array, a list, etc) of 2D numpy arrays
            corresponding to the Bkl matrices for all 'l's.

        Returns
        -------
        SINR_k : 1D numpy array
            The SINR for the different streams of user k.

        """
        Hkk = self._get_channel(k, k)
        Vk = self.full_F[k]
        Uk_H = self.full_W_H[k]

        SINR_k = np.empty(self.Ns[k], dtype=float)

        for l in range(self.Ns[k]):
            Vkl = Vk[:, l:l + 1]
            Ukl_H = Uk_H[l:l + 1, :]
            Ukl = Ukl_H.conj().T
            aux = np.dot(Ukl_H,
                         np.dot(Hkk, Vkl))
            numerator = np.dot(aux,
                               aux.transpose().conjugate())
            denominator = np.dot(Ukl_H,
                                 np.dot(Bkl_all_l[l], Ukl))
            SINR_kl = np.asscalar(numerator) / np.asscalar(denominator)
            SINR_k[l] = np.abs(SINR_kl)  # The imaginary part should be
                                         # negligible

        return SINR_k

    def solve(self, Ns, P=None):
        """
        Find the IA solution.

        This method must be implemented in a subclass and should updates
        the 'F' and 'W' member variables.

        Parameters
        ----------
        Ns : int or 1D numpy array
            Number of streams of each user.
        P : 1D numpy array
            Power of each user. If not provided, a value of 1 will be used
            for each user.

        Notes
        -----
        This function should be implemented in the derived classes
        """
        raise NotImplementedError("solve: Not implemented")


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx ClosedFormIASolver class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class ClosedFormIASolver(IASolverBaseClass):
    """
    Implements the closed form Interference Alignment algorithm as
    described in the paper "Interference Alignment and Degrees of Freedom
    of the K User Interference Channel [CadambeDoF2008]_".

    Parameters
    ----------
    multiUserChannel : A MultiUserChannelMatrix object.
        The multiuser channel.

    Notes
    -----

    .. [CadambeDoF2008] V. R. Cadambe and S. A. Jafar, "Interference
       Alignment and Degrees of Freedom of the K User Interference
       Channel," IEEE Transactions on Information Theory 54, pp. 3425–3441,
       Aug. 2008.
    """
    def __init__(self, multiUserChannel, use_best_init=True):
        """
        Parameters
        ----------
        multiUserChannel : A MultiUserChannelMatrix object.
            The multiuser channel.
        use_best_init : bool
            If true, all possible initializations (subsets of the
            eigenvectors of the 'E' matrix)for the first precoder will be
            tested and the best solution will be used. If false, the first
            initialization will be used.
        """
        IASolverBaseClass.__init__(self, multiUserChannel)
        self._use_best_init = use_best_init

    def _calc_E(self):
        """
        Calculates the "E" matrix, given by

        :math:`\\mtE = \\mtH_{31}^{-1}\\mtH_{32}\\mtH_{12}^{-1}\\mtH_{13}\\mtH_{23}^{-1}\\mtH_{21}`.
        """
        # $\mtE = \mtH_{31}^{-1}\mtH_{32}\mtH_{12}^{-1}\mtH_{13}\mtH_{23}^{-1}\mtH_{21}$

        H31 = self._get_channel(2, 0)
        H32 = self._get_channel(2, 1)
        H12 = self._get_channel(0, 1)
        H13 = self._get_channel(0, 2)
        H23 = self._get_channel(1, 2)
        H21 = self._get_channel(1, 0)

        E = (np.linalg.solve(H31, H32)).dot(
             (np.linalg.solve(H12, H13)).dot(
                 np.linalg.solve(H23, H21)))

        return E

    def _calc_all_F_initializations(self, Ns):
        """
        Calculates all possible initializations for the first precoder
        (self._F[0]).

        The precoder self._F[0] is initialized with a subset of the
        eigenvectors of the matrix 'E'. Therefore, this method returns all
        possible subsets of the eigenvectors of the matrix 'E'.

        Parameters
        ----------
        Ns : int
            Number of streams of the first user.

        Returns
        -------
        all_initializations : A list of numpy arrays.
            All possible subsets (with size Ns) of the eigenvectors of the
            matrix E
        """
        E = self._calc_E()
        eigenvectors = np.linalg.eig(E)[1]
        num_eigenvectors = eigenvectors.shape[1]

        all_subsets = []

        for comb_index in itertools.combinations(range(num_eigenvectors), Ns):
            all_subsets.append(eigenvectors[:, comb_index])

        return all_subsets

    def _updateF(self, F0=None):
        """
        Find the precoders.

        Parameters
        ----------
        F0 : numpy array
            The first precoder. If not provided, the matrix 'E' will be
            calculated (with the _calc_E method) and the the first Ns
            eigenvectors will be used as F0.
        """
        self._clear_precoder_filter()

        # The number of users is always 3 for the ClosedFormIASolver class
        self._F = np.zeros(3, dtype=np.ndarray)

        if F0 is None:
            E = self._calc_E()
            Ns0 = self.Ns[0]

            # The first precoder is given by any subset of the eigenvectors of
            # the "E" matrix. We simple get the first Ns_0 eigenvectors of E.
            eigenvectors = np.linalg.eig(E)[1]
            F0 = eigenvectors[:, 0:Ns0]
            self._F[0] = F0
        else:
            self._F[0] = F0

        # The second precoder is given by $\mtH_{32}^{-1}\mtH_{31}\mtV_1$
        invH32 = np.linalg.pinv(self._get_channel(2, 1))
        H31 = self._get_channel(2, 0)
        self._F[1] = np.dot(invH32, np.dot(H31, F0))
        # The third precoder is given by $\mtH_{23}^{-1}\mtH_{21}\mtV_1$
        invH23 = np.linalg.pinv(self._get_channel(1, 2))
        H21 = self._get_channel(1, 0)
        self._F[2] = np.dot(invH23, np.dot(H21, F0))

        # Normalize the precoders
        self._F[0] = self._F[0] / np.linalg.norm(self._F[0], 'fro')
        self._F[1] = self._F[1] / np.linalg.norm(self._F[1], 'fro')
        self._F[2] = self._F[2] / np.linalg.norm(self._F[2], 'fro')

    def _updateW(self):
        """Find the receive filters
        """
        self._clear_receive_filter()

        # The number of users is always 3 for the ClosedFormIASolver class
        self._W = np.zeros(3, dtype=np.ndarray)

        A0 = np.dot(self._get_channel(0, 1), self.F[1])
        self._W[0] = leig(
            np.dot(A0, A0.transpose().conjugate()),
            self.Ns[0])[0]

        A1 = np.dot(self._get_channel(1, 0), self.F[0])
        self._W[1] = leig(
            np.dot(A1, A1.transpose().conjugate()),
            self.Ns[1])[0]

        A2 = np.dot(self._get_channel(2, 0), self.F[0])
        self._W[2] = leig(
            np.dot(A2, A2.transpose().conjugate()),
            self.Ns[2])[0]

    # @property
    # def W(self):
    #     """Receive filter of all users."""
    #     return self._get_W_property_alt()

    def solve(self, Ns, P=None):
        """
        Find the IA solution.

        This method updates the 'F' and 'W' member variables.

        Parameters
        ----------
        Ns : int or 1D numpy array
            Number of streams of each user.
        P : 1D numpy array
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        assert self.K == 3, ('The ClosedFormIASolver class only works'
                             ' in a MIMO-IC scenario with 3 users.')

        if isinstance(Ns, int):
            Ns = np.ones(3, dtype=int) * Ns
        else:
            assert len(Ns) == 3
        self._Ns = np.array(Ns)  # This will create a new array so that we
                                 # can modify self._Ns internally without
                                 # changing the original Ns variable passed
                                 # to the solve method.

        self.P = P

        if self._use_best_init is True:
            # xxxxx Case when the best solution should be used xxxxxxxxxxxx
            best_sum_capacity = 0
            all_initializations = self._calc_all_F_initializations(Ns[0])

            # Lambda function to calculate the sum capacity from the SINR
            # values (in linear scale)
            calc_capacity = lambda sinr: np.sum(np.log2(1 + sinr))

            for F0 in all_initializations:
                self._updateF(F0)
                self._updateW()

                # Calculates the sum capacity
                sinr_all_k = self.calc_SINR()

                # Array with the sum capacity of each user
                sum_capacity = map(calc_capacity, sinr_all_k)
                # Total sum capacity
                total_sum_capacity = np.sum(list(sum_capacity))

                if total_sum_capacity > best_sum_capacity:
                    best_sum_capacity = total_sum_capacity
                    best_F = self._F
                    best_W = self._W

            self._clear_precoder_filter()
            self._clear_receive_filter()
            self._F = best_F
            self._W = best_W
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        else:
            # xxxxx Case when the first solution should be used xxxxxxxxxxx
            self._updateF()
            self._updateW()
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx Base Class for all iterative IA algorithms xxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class IterativeIASolverBaseClass(IASolverBaseClass):
    """
    Base class for all Iterative IA algorithms.

    All subclasses of IterativeIASolverBaseClass must implement at least
    the _updateF and _updateW methods.

    Solving an iterative algorithm usually involves some initialization and
    then performing a "step" a given number of times until convergence. The
    initialization code is performed in the `_solve_init` method while the
    "step" corresponds to the `_step` method.

    The initialization code is defined here as simply initializing the
    precoder with a random matrix and then calling the `_updateW` method
    (which must be implemented in a subclass) to update the receive
    filter. This is usually what you want but any subclass can redefine
    `_solve_init` if a different initialization is required.

    The "step" part usually involves updating the precoder and then
    updating the receive filters. The definition of `_step` here calls two
    methods that MUST be defined in subclasses, the _updateF method and the
    _updateW method. If anything else is required in the "step" part then
    the _step method can be redefined in a subclass, but even in that case
    it should call the _updateF and _updateW methods instead of
    implementing everything in your redefined _step method.

    Parameters
    ----------
    multiUserChannel : A MultiUserChannelMatrix object.
        The multiuser channel.
    """

    def __init__(self, multiUserChannel):
        """
        Parameters
        ----------
        multiUserChannel : A MultiUserChannelMatrix object.
            The multiuser channel.
        """
        IASolverBaseClass.__init__(self, multiUserChannel)

        self._runned_iterations = 0  # Count how many times the step method
                                     # was called. This will be reseted to
                                     # zero whenever the channel or the
                                     # precoder is reinitialized.
        self.max_iterations = 50  # Number of times the step method is
                                  # called in the solve method.

        # Relative change of the precoder in one iteration to the next
        # one. If the relative change from one iteration to the next one is
        # lower than this factor then the algorithm will stop the
        # iterations before the max_iterations limit is reached.
        self.relative_factor = 1e-6

        # We can use the closed form IA solver to initialize the iterative
        # algorithm. This can reduce the number of iterations required for
        # convergence. Note that this will be done only if
        # initialize_with_closed_form is True when the solve method is
        # called.
        self._closed_form_ia_solver = ClosedFormIASolver(multiUserChannel,
                                                         use_best_init=True)
        self.initialize_with_closed_form = False

    def _get_runned_iterations(self):
        """Get method for the runned_iterations property."""
        return self._runned_iterations
    runned_iterations = property(_get_runned_iterations)

    def clear(self):
        """
        Clear the IA Solver object.

        All member attributes that are updated during the solve method,
        such as the precoder and receive filters, will be cleared. The
        other attributes that correspond to "configuration" such as the
        channel object won't be changed

        Notes
        -----
        You should overwrite this method in subclasses that pass parameters
        to the __init__ method, since here we call __init__ without
        arguments which is probably not what you want.
        """
        IASolverBaseClass.clear(self)
        self._runned_iterations = 0

    def _updateF(self):  # pragma: no cover
        """
        Update the precoders.

        Notes
        -----
        This method should be implemented in the derived classes

        See also
        --------
        _step
        """
        raise NotImplementedError("_updateF: Not implemented")

    def _updateW(self):  # pragma: no cover
        """
        Update the receive filters.

        Notes
        -----
        This method should be implemented in the derived classes

        See also
        --------
        _step
        """
        raise NotImplementedError("_updateW: Not implemented")

    def _step(self):
        """Performs one iteration of the algorithm.

        This method does not return anything, but instead updates the
        precoder and receive filter.
        """
        self._updateF()
        self._updateW()

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Overwrite method in the IASolverBaseClass that change the precoder so
    # that they also reset the _runned_iterations attribute to zero.
    def randomizeF(self, Ns, P=None):
        self._runned_iterations = 0
        super(IterativeIASolverBaseClass, self).randomizeF(Ns, P)
    randomizeF.__doc__ = IASolverBaseClass.randomizeF.__doc__
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _initialize_F_randomly_and_find_W(self, Ns, P):
        """
        Initialize the IA Solution from a random matrix.

        The implementation here simple initializes the precoder variable
        and then calculates the initial receive filter.

        Parameters
        ----------
        Ns : int or 1D numpy array
            Number of streams of each user.
        P : 1D numpy array
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        self.randomizeF(Ns, P)
        self._updateW()

    def _initialize_F_and_W_from_closed_form(self, Ns, P):
        """Initialize the IA Solution from the closed form IA solver.

        Parameters
        ----------
        Ns : int or 1D numpy array
            Number of streams of each user.
        P : 1D numpy array
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        # Clear all precoders and receive filters
        self._clear_precoder_filter()
        self._clear_receive_filter()

        self._closed_form_ia_solver.solve(Ns, P)
        self._F = self._closed_form_ia_solver.F
        self._W = self._closed_form_ia_solver.W

    def _solve_init(self, Ns, P):
        """
        Code run in the `solve` method before the loop that run the `step`
        method.

        The implementation here simple initializes the precoder variable
        and then calculates the initial receive filter.

        Parameters
        ----------
        Ns : int or 1D numpy array
            Number of streams of each user.
        P : 1D numpy array
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        if self.initialize_with_closed_form is True:
            self._initialize_F_and_W_from_closed_form(Ns, P)
        else:
            self._initialize_F_randomly_and_find_W(Ns, P)

    def _solve_finalize(self):  # pragma: no cover
        """Perform any post processing after the solution has been found.
        """
        # Some of the found precoders may be a singular matrix. In that
        # case, we need to remove the dimensions with zero energy from both
        # the found precoder and the receive filter.
        mod_users = []  # Store the index of the users from which we need
                        # to modify the precoders and receive filters
        num_significant_sing_values = []
        for k in range(self.K):
            # We only need to perform further actions if more then one
            # streams is transmitted. In that case we need to test if the
            # final precoder has a large condition number indicating that
            # some dimensions has very low energy and could be
            # discarded. If only one stream is transmitted then the
            # condition number is always equal to 1.
            if self.Ns[k] > 1:
                [_, S, _] = np.linalg.svd(self._F[k])
                # If the condition number os large, then there is some
                # dimension with zero energy
                cond = S.max() / S.min()
                if cond > 1e4:
                    mod_users.append(k)
                    # [U, S, V_H] = np.linalg.svd(self._F[k])
                    max_sing_value = S.max()
                    # Calculate the number of significative singular
                    # values. Basically, any singular value (and corresponding
                    # dimension) lower then max_sing_value/1e8 will be
                    # discarded.
                    n = np.sum(S > max_sing_value / 1.0e4)

                    # Store the number of significative singular values for
                    # that user
                    num_significant_sing_values.append(n)

                    new_F = get_principal_component_matrix(self._F[k], n)
                    new_F = new_F / np.linalg.norm(new_F, 'fro')
                    self._F[k] = new_F
                    self.Ns[k] = n

        # If we modified any of the precoders then the mod_users list has
        # the index of the users whose precoders were modified. We need to
        # also modify the receive filter for those users.
        if len(mod_users) > 0:
            # Note that we still need to remove the dead dimensions of the
            # receive filter. However, depending on the algorithm, either the
            # _W or the _W_H member variable was set while the other is None
            # (at this point).
            if self._W_H is None:
                # Since _W_H is None that means that we need to modify the _W
                # member variable
                for k, n in zip(mod_users, num_significant_sing_values):
                    new_W = get_principal_component_matrix(self._W[k], n)
                    self._W[k] = new_W

            elif self._W is None:
                # Since _W is None that means that we need to modify the _W_H
                # member variable
                for k, n in zip(mod_users, num_significant_sing_values):
                    W = self._W_H[k].conj().T
                    new_W = get_principal_component_matrix(W, n)
                    self._W_H[k] = new_W.conj().T
            else:
                # If both self._W and self._W_H are not None then something
                # wrong happened. Maybe you called the self.W or the self.W_H
                # properties by mistake before _solve_finalize is called
                # (in the solve method).
                raise Exception("I should not be here.")

    @classmethod
    def _is_diff_significant(cls, F_old, F_new, relative_factor):
        """
        Test if there was any significant change from `F_old` to `F_new`.

        This method is used internally in the solve method of the
        IterativeIASolverBaseClass to detect when the precoder of a given
        iteration didn't change significantly from one iteration to
        another. This is used to stop the iterations of the algorithm and
        avoid unnecessary computations.

        Parameters
        ----------
        F_old : 1D numpy array of numpy arrays
            The precoder of all users (in a previous iteration).
        F_new : 1D numpy array of numpy arrays
            The precoder of all users (in the current iteration).

        Returns
        -------
        out : bool
            True if the difference is significant, False otherwise.

        Notes
        -----
        A difference is considered significant if it is larger then 1/1000
        of the minimum value in the precoder.
        """
        K = F_old.size
        for k in range(K):
            Fk_old = F_old[k]
            Fk_new = F_new[k]
            min_value = np.abs(Fk_new).min()
            diff = np.abs(Fk_new - Fk_old)
            max_diff = diff.max()
            if max_diff > (min_value * relative_factor):
                return True

        return False

    def solve(self, Ns, P=None):
        """
        Find the IA solution by performing the `step` method several times.

        The number of times the `step` method is run is controlled by the
        max_iterations member variable.

        Before calling the `step` method for the first time the
        `_solve_init` method is called to perform any required
        initializations. Since iterative IA algorithms usually starts with
        a random precoder then the `_solve_init` implementation in
        IterativeIASolverBaseClass calls randomizeF.

        Parameters
        ----------
        Ns : int or 1D numpy array
            Number of streams of each user.
        P : 1D numpy array
            Power of each user. If not provided, a value of 1 will be used
            for each user.

        Returns
        -------
        Number of iterations the iterative interference alignment algorithm
        run.

        Notes
        -----
        You probably should not overwrite this method in sub-classes of the
        IterativeIASolverBaseClass. If you want to change the
        initialization of the algorithm overwrite the _solve_init method
        instead.

        Subclasses must implement the _updateF() and _updateW() methods. If
        something else besides calling these two methods is required in the
        "step" part of the algorithm, then reimplement also the _step()
        method.
        """
        self._solve_init(Ns, P)

        if isinstance(Ns, int):
            Ns = np.ones(self.K, dtype=int) * Ns
        else:
            assert len(Ns) == self.K

        self._Ns = np.array(Ns)  # This will create a new array so that we
                                 # can modify self._Ns internally without
                                 # changing the original Ns variable passed
                                 # to the randomizeF method.

        # This will be used to detect of the precoder did not
        # significativelly change
        old_F = self._F
        i = -1  # Initialize the i variable in case the for loop is not run
        for i in range(self.max_iterations):
            self._runned_iterations = self._runned_iterations + 1
            self._step()

            # Stop the iteration earlier if the precoder does not change
            # too much
            if self._is_diff_significant(old_F, self._F, self.relative_factor) is False:
                break
            else:
                old_F = self._F

        # Perform any post processing after the precoder and receive
        # filters where found. One possible usage for this method is to
        # remove dimensions of the precoder (and receive filter) that have
        # actually zero energy. That is, if the precoder ends up being a
        # singular matrix we can implement _solve_finalize to remove the
        # dimensions that do not contribute.
        self._solve_finalize()

        # Return the number of iterations the algorithm run
        return i + 1


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx AlternatingMinIASolver Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class AlternatingMinIASolver(IterativeIASolverBaseClass):
    """
    Implements the "Interference Alignment via Alternating Minimization"
    algorithm from the paper with the same name [PetersHeathAltMin2009]_.

    This algorithm is applicable to a "K-user" scenario and it is very
    flexible in the sense that you can change the number of transmit
    antennas, receive antennas and streams per user, as well as the number
    of users involved in the IA process. However, note that alignment is
    only feasible for some cases configurations.

    An example of a common exenario is a scenario with 3 pairs or
    transmitter/receiver with 2 antennas in each node and 1 stream
    transmitted per node.

    You can determine the scenario of an AlternatingMinIASolver object by
    infering the variables K, Nt, Nr and Ns.

    Parameters
    ----------
    multiUserChannel : A MultiUserChannelMatrix object.
        The multiuser channel.

    Notes
    -----

    .. [PetersHeathAltMin2009] Peters, S.W.; Heath, R.W., "Interference
       alignment via alternating minimization," Acoustics, Speech and
       Signal Processing, 2009. ICASSP 2009. IEEE International Conference
       on, pp.2445,2448, 19-24 April 2009
    """
    def __init__(self, multiUserChannel):
        """
        Parameters
        ----------
        multiUserChannel : A MultiUserChannelMatrix object.
            The multiuser channel.
        """
        IterativeIASolverBaseClass.__init__(self, multiUserChannel)

        self._C = []    # Basis of the interference subspace for each user
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def get_cost(self):
        """
        Get the Cost of the algorithm for the current iteration of the
        precoder.

        Returns
        -------
        cost : float (real non-negative number)
            The Cost of the algorithm for the current iteration of the
            precoder
        """
        Cost = 0
        # This will get all combinations of (k,l) without repetition. This
        # is equivalent to two nested for loops with an if statement to
        # only execute the code only when `k` is different of `l`.
        all_kl_indexes = itertools.permutations(range(self.K), 2)

        for kl in all_kl_indexes:
            (k, l) = kl
            Hkl_Fl = np.dot(
                self._get_channel(k, l),
                self._F[l])
            Cost = Cost + np.linalg.norm(
                Hkl_Fl -
                np.dot(
                    np.dot(
                        self._C[k],
                        self._C[k].transpose().conjugate()),
                    Hkl_Fl
                ), 'fro') ** 2

        return Cost

    def _solve_init(self, Ns, P):
        """
        Code run in the `solve` method before the loop that run the `step`
        method.

        The implementation here simple initializes the precoder variable
        and then calculates the initial receive filter.
        """
        self.randomizeF(Ns, P)
        self._updateC()
        self._updateW()

    def _step(self):
        """Performs one iteration of the algorithm.

        The step method is usually all you need to call to perform an
        iteration of the Alternating Minimization algorithm. It will update
        C, then update F and finally update W.

        See also
        --------
        updateC, updateF, updateW

        """

        self._updateC()
        self._updateF()
        self._updateW()

    def _updateC(self):
        """Update the value of Ck for all K users.

        Ck contains the orthogonal basis of the interference subspace of
        user k. It corresponds to the Nk-Sk dominant eigenvectors of

            :math:`\\sum_{l \\neq k} \\mtH_{k,l} \\mtF_l \\mtF_l^H \\mtH_{k,l}^H`.

        Notes
        -----
        This method is called in the :meth:`step` method.

        See also
        --------
        step
        """
        # $$\sum_{l \neq k} \mtH_{k,l} \mtF_l \mtF_l^H \mtH_{k,l}^H$$

        # xxxxxxxxxx New Implementation using calc_Q xxxxxxxxxxxxxxxxxxxxxx
        Ni = self.Nr - self.Ns  # Ni: Dimension of the interference subspace

        self._C = np.empty(self.K, dtype=np.ndarray)

        for k in np.arange(self.K):
            ### TODO: Implement and test with external interference
            # We are inside only of the first for loop
            # Add the external interference contribution
            #self._C[k] = self.calc_Q(k) + self.Rk[k]

            # C[k] will receive the Ni most dominant eigenvectors of C[k]
            self._C[k] = peig(self.calc_Q(k), Ni[k])[0]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _updateF(self):
        """
        Update the value of the precoder of all K users.

        Fl, the precoder of the l-th user, tries avoid as much as possible
        to send energy into the desired signal subspace of the other
        users. Fl contains the Sl least dominant eigenvectors of
        :math:`\\sum_{k \\neq l} H_{k,l}^H (I - C_k C_k^H)H_{k,l}`

        Notes
        -----
        This method is called in the :meth:`step` method.

        See also
        --------
        step
        """
        # $\sum_{k \neq l} \mtH_{k,l}^H (\mtI - \mtC_k \mtC_k^H)\mtH_{k,l}$
        # xxxxx Calculates the temporary variable Y[k] for all k xxxxxxxxxx
        # Note that $\mtY[k] = (\mtI - \mtC_k \mtC_k^H)$

        self._clear_precoder_filter()

        # The number of users is always 3 for the ClosedFormIASolver class
        self._F = np.zeros(self.K, dtype=np.ndarray)

        calc_Y = lambda Nr, C: np.eye(Nr, dtype=complex) - \
            np.dot(C, C.conjugate().transpose())
        Y = map(calc_Y, self.Nr, self._C)

        newF = np.zeros(self.K, dtype=np.ndarray)
        # This will get all combinations of (l,k) without repetition. This
        # is equivalent to two nested for loops with an if statement to
        # only execute the code only when `l` is different of `k`
        all_lk_indexes = itertools.permutations(range(self.K), 2)

        # This code will store in newF[l] the equivalent of
        # $\sum_{k \neq l} \mtH_{k,l}^H (\mtI - \mtC_k \mtC_k^H)H_{k,l}$
        for lk in all_lk_indexes:
            (l, k) = lk
            lH = self._get_channel(k, l)
            newF[l] = newF[l] + np.dot(
                np.dot(lH.conjugate().transpose(),
                       Y[k]),
                lH)

        # Every element in newF is a matrix. We want to replace each
        # element by the least dominant eigenvectors of that element.

        #self._F = map(lambda x, y: leig(x, y)[0], newF, self.Ns)
        for k in range(self.K):
            self._F[k] = leig(newF[k], self.Ns[k])[0]

    def _updateW(self):
        """
        Update the zero-forcing filters.

        The zero-forcing filter is calculated in the paper "MIMO
        Interference Alignment Over Correlated Channels with Imperfect
        CSI".

        Notes
        -----
        This method is called in the :meth:`step` method.

        See also
        --------
        step
        """
        self._clear_receive_filter()

        # Note that the formula for the receive filter in the "Interference
        # Alignment via Alternating Minimization" paper actually calculates
        # W_H instead of W.
        newW_H = np.zeros(self.K, dtype=np.ndarray)
        for k in np.arange(self.K):
            tildeHi = np.hstack(
                [np.dot(self._get_channel(k, k), self._F[k]),
                 self._C[k]])
            newW_H[k] = np.linalg.inv(tildeHi)
            # We only want the first Ns[k] lines
            newW_H[k] = newW_H[k][0:self.Ns[k]]
        self._W_H = newW_H


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MinLeakageIASolver class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MinLeakageIASolver(IterativeIASolverBaseClass):
    """
    Implements the Minimum Leakage Interference Alignment algorithm as
    described in the paper "Approaching the Capacity of Wireless Networks
    through Distributed Interference Alignment [Cadambe2008]_".

    Parameters
    ----------
    multiUserChannel : A MultiUserChannelMatrix object.
        The multiuser channel.
    """

    def __init__(self, multiUserChannel):
        """
        Parameters
        ----------
        multiUserChannel : A MultiUserChannelMatrix object.
            The multiuser channel.
        """
        IterativeIASolverBaseClass.__init__(self, multiUserChannel)

    def get_cost(self):
        """
        Get the Cost of the algorithm for the current iteration of the
        precoder.

        For the Minimum Leakage Interference Alignment algorithm the cost
        is equivalent to the sum of the interference that all users see
        after applying the receive filter. That is,

            :math:`C = Tr[\\mtU_k^H \\mtQ_k \\mtU_k]`

        Returns
        -------
        cost : float (real non-negative number)
            The Cost of the algorithm for the current iteration of the
            precoder
        """
        # $$C = Tr[\mtU_k^H \mtQ_k \mtU_k]$$
        cost = 0
        for k in range(self.K):
            Qk = self.calc_Q(k)
            Wk = self._W[k]
            aux = np.dot(
                np.dot(Wk.transpose().conjugate(), Qk),
                Wk)
            cost = cost + np.trace(np.abs(aux))
        return cost

    def _calc_Uk_all_k(self):
        """Calculates the receive filter of all users.
        """
        Uk = np.empty(self.K, dtype=np.ndarray)

        for k in range(self.K):
            Qk = self.calc_Q(k)
            [V, _] = leig(Qk, self.Ns[k])
            Uk[k] = V
        return Uk

    def _calc_Uk_all_k_rev(self):
        """Calculates the receive filter of all users in the reverse
        network.

        """
        Uk_rev = np.empty(self.K, dtype=np.ndarray)
        for k in range(self.K):
            Qk_rev = self.calc_Q_rev(k)
            [V, _] = leig(Qk_rev, self.Ns[k])
            Uk_rev[k] = V
        return Uk_rev

    def _updateF(self):
        """
        Update the precoders.

        Notes
        -----
        This method is called in the :meth:`_step` method.

        See also
        --------
        _step

        """
        self._clear_precoder_filter()
        self._F = self._calc_Uk_all_k_rev()

    def _updateW(self):
        """
        Update the receive filters.

        Notes
        -----
        This method is called in the :meth:`_step` method.

        See also
        --------
        _step
        """
        self._clear_receive_filter()
        self._W = self._calc_Uk_all_k()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MaxSinrIASolver class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MaxSinrIASolver(IterativeIASolverBaseClass):
    """Implements the "Interference Alignment via Max SINR" algorithm.

    This algorithm is applicable to a "K-user" scenario and it is
    described in [Cadambe2008]_.

    An example of a common exenario is a scenario with 3 pairs or
    transmitter/receiver with 2 antennas in each node and 1 stream
    transmitted per node.

    You can determine the scenario of an MaxSinrIASolver object by
    infering the variables K, Nt, Nr and Ns.

    Parameters
    ----------
    multiUserChannel : A MultiUserChannelMatrix object.
        The multiuser channel.

    Notes
    -----

    .. [Cadambe2008] K. Gomadam, V. R. Cadambe, and S. A. Jafar,
       "Approaching the Capacity of Wireless Networks through Distributed
       Interference Alignment," in IEEE GLOBECOM 2008 - 2008 IEEE Global
       Telecommunications Conference, 2008, pp. 1-6.

    """

    def __init__(self, multiUserChannel):
        """
        Parameters
        ----------
        multiUserChannel : A MultiUserChannelMatrix object.
            The multiuser channel.
        """
        IterativeIASolverBaseClass.__init__(self, multiUserChannel)

    def _calc_Bkl_cov_matrix_first_part_rev(self, k):
        """Calculates the first part in the equation of the Blk covariance
        matrix of the reverse channel.

        Parameters
        ----------
        k : int
            Index of the desired user.

        Returns
        -------
        Bkl_first_part_rev : 2D numpy complex array
            First part in equation (28) of [Cadambe2008]_, but for the
            reverse channel.

        See also
        --------
        _calc_Bkl_cov_matrix_first_part_rev

        """
        # $$\sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$$
        P = self.P
        first_part = 0.0

        for j in range(self.K):
            Hkj = self._get_channel_rev(k, j)
            Hkj_H = Hkj.conjugate().transpose()
            Vj = self._W[j]

            # The lets make sure the receive filter norm is equal to one so
            # that we can correctly scale it to the desired power.
            assert np.linalg.norm(Vj, 'fro') - 1.0 < 1e-6

            Vj_H = Vj.conjugate().transpose()
            first_part = first_part + (float(P[j]) / self._Ns[j]) * np.dot(
                Hkj,
                np.dot(
                    np.dot(Vj,
                           Vj_H),
                    Hkj_H))

        return first_part

    def _calc_Bkl_cov_matrix_second_part_rev(self, k, l):
        """Calculates the second part in the equation of the Blk covariance
        matrix of the reverse channel..

        The second part is given by

            :math:`\\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger}`

        Parameters
        ----------
        k : int
            Index of the desired user.
        l : int
            Index of the desired stream.

        Returns
        -------
        second_part : 2D numpy complex array.
            Second part in equation (28) of [Cadambe2008]_.

        See also
        --------
        _calc_Bkl_cov_matrix_second_part
        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger}$$
        P = self.P

        Hkk = self._get_channel_rev(k, k)
        Hkk_H = Hkk.transpose().conjugate()

        Vkl = self._W[k][:, l:l + 1]
        Vkl_H = Vkl.transpose().conjugate()
        second_part = np.dot(Hkk,
                             np.dot(np.dot(Vkl, Vkl_H),
                                    Hkk_H))

        return second_part * (float(P[k]) / self._Ns[k])

    def _calc_Bkl_cov_matrix_all_l_rev(self, k):
        """Calculates the interference-plus-noise covariance matrix for all
        streams at "receiver" :math:`k` for the reverse channel.

        Parameters
        ----------
        k : int
            Index of the desired user.

        Returns
        -------
        Bkl_rev : 1D numpy array of 2D numpy arrays
            Covariance matrix of all streams of user k. Each element of the
            returned 1D numpy array is a 2D numpy complex array
            corresponding to the covariance matrix of one stream of user k.

        See also
        --------
        calc_Bkl_cov_matrix_all_l

        """
        # $$\mtB^{[kl]} = \sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star l}^{[j]} \mtV_{\star l}^{[j]\dagger} \mtH^{[kj]\dagger} - \frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger} + \mtI_{N^{[k]}}$$
        Bkl_all_l_rev = np.empty(self._Ns[k], dtype=np.ndarray)
        first_part = self._calc_Bkl_cov_matrix_first_part_rev(k)

        for l in range(self._Ns[k]):
            second_part = self._calc_Bkl_cov_matrix_second_part_rev(k, l)
            Bkl_all_l_rev[l] = first_part - second_part + (
                self.noise_var * np.eye(self.Nt[k]))

        return Bkl_all_l_rev

    @classmethod
    def _calc_Ukl(cls, Hkk, Vk, Bkl, k, l):
        """Calculates the Ukl matrix in equation (29) of [Cadambe2008]_.

        Parameters
        ----------
        Hkk : 2D numpy complex array
            Channel from transmitter K to receiver K.
        Vk : 2D numpy array
            Precoder of user k.
        Bkl : 2D numpy complex array
            The previously calculates Bkl matrix in equation (28) of
            [Cadambe2008]_
        k : int
            Index of the desired user
        l : int
            Index of the desired stream

        Returns
        -------
        Ukl : 2D numpy array (with self.Nr[k] rows and a single column)
            The calculated Ukl matrix.

        """
        Vkl = Vk[:, l:l + 1]

        Ukl = np.linalg.solve(Bkl, np.dot(Hkk, Vkl))

        Ukl = Ukl / np.linalg.norm(Ukl, 'fro')
        return Ukl

    @classmethod
    def _calc_Uk(cls, Hkk, Vk, Bkl_all_l, k):
        """Similar to the :meth:`calc_Ukl` method, but while :meth:`calc_Ukl`
        calculates the receive filter (a vector) only for the :math:`l`-th
        stream :meth:`calc_Uk` calculates a receive filter (a matrix) for
        all streams.

        Parameters
        ----------
        Hkk : 2D numpy complex array
            Channel from transmitter K to receiver K.
        Vk : 2D numpy array
            Precoder of user k.
        Bkl_all_l : 1D numpy array of 2D numpy arrays.
            Covariance matrix of all streams of user k. Each element of the
            returned 1D numpy array is a 2D numpy complex array
            corresponding to the covariance matrix of one stream of user k.
        k : int
            Index of the desired user.

        Returns
        -------
        Uk : 2D numpy array.
            The receive filver for all streams of user k.
        """
        num_streams = Bkl_all_l.size
        num_Rx = Bkl_all_l[0].shape[0]
        Uk = np.zeros([num_Rx, num_streams], dtype=complex)
        for l in range(num_streams):
            Uk[:, l] = MaxSinrIASolver._calc_Ukl(
                Hkk, Vk, Bkl_all_l[l], k, l)[:, 0]

        return Uk / np.linalg.norm(Uk, 'fro')

    def _calc_Uk_all_k(self):
        """Calculates the receive filter of all users.
        """
        Uk = np.empty(self.K, dtype=np.ndarray)

        for k in range(self.K):
            Hkk = self._get_channel(k, k)
            Bkl_all_l = self._calc_Bkl_cov_matrix_all_l(k, self.noise_var)
            Uk[k] = self._calc_Uk(Hkk, self._F[k], Bkl_all_l, k)
        return Uk

    def _calc_Uk_all_k_rev(self):
        """Calculates the receive filter of all users for the reverse channel.
        """
        Uk = np.empty(self.K, dtype=np.ndarray)
        F = self._W  # The precoder is the receive filter of the direct
                    # channel
        for k in range(self.K):
            Hkk = self._get_channel_rev(k, k)
            Bkl_all_l = self._calc_Bkl_cov_matrix_all_l_rev(k)
            Uk[k] = self._calc_Uk(Hkk, F[k], Bkl_all_l, k)
        return Uk

    def _updateF(self):
        """
        Update the precoders.

        Notes
        -----
        This method is called in the :meth:`_step` method.

        See also
        --------
        _step
        """
        self._clear_precoder_filter()
        self._F = self._calc_Uk_all_k_rev()

    def _updateW(self):
        """
        Update the receive filters.

        Notes
        -----
        This method is called in the :meth:`_step` method.

        See also
        --------
        _step
        """
        self._clear_receive_filter()
        self._W = self._calc_Uk_all_k()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MMSEIASolver Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MMSEIASolver(IterativeIASolverBaseClass):
    """
    Implements the MMSE based Interference Alignment algorithm.

    This algorithm is applicable to a "K-user" scenario and it is
    described in [Peters2011]_.

    An example of a common exenario is a scenario with 3 pairs or
    transmitter/receiver with 2 antennas in each node and 1 stream
    transmitted per node.

    You can determine the scenario of an MMSEIASolver object by
    infering the variables K, Nt, Nr and Ns.

    Parameters
    ----------
    multiUserChannel : A MultiUserChannelMatrix object.
        The multiuser channel.

    Notes
    -----

    .. [Peters2011] S. W. Peters and R. W. Heath, "Cooperative Algorithms
       for MIMO Interference Channels," vol. 60, no. 1, pp. 206–218, 2011.
    """

    def __init__(self, multiUserChannel):
        """
        Parameters
        ----------
        multiUserChannel : A MultiUserChannelMatrix object.
            The multiuser channel.
        """
        IterativeIASolverBaseClass.__init__(self, multiUserChannel)

        self._mu = None
        self._bisection_tol = 1e-3  # Tolerance used to stop the bisection method

    def _solve_init(self, Ns, P):
        """
        Code run in the `solve` method before the loop that run the `step`
        method.

        The implementation here simple initializes the precoder variable
        and then calculates the initial receive filter.

        Parameters
        ----------
        Ns : int or 1D numpy array
            Number of streams of each user.
        P : 1D numpy array
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        self._mu = np.zeros(self.K, dtype=float)
        IterativeIASolverBaseClass._solve_init(self, Ns, P)

    def _calc_Uk(self, k):
        """Calculates the receive filter of the k-th user.

        Parameters
        ----------
        k : int
            User index
        """
        # $$\mtU_k = \left( \sum_{i=1}^K \mtH_{ki} \mtV_i \mtV_i^H \mtH_{ki}^H + \sigma_n^2 \mtI \right)^{-1} \mtH_{kk} \mtV_k$$
        Hkk = self._get_channel(k, k)
        Vk = self.full_F[k]

        sum_term = 0
        for i in range(self.K):
            Hki = self._get_channel(k, i)
            Vi = self.full_F[i]
            aux = np.dot(Hki, Vi)
            sum_term = sum_term + np.dot(aux,
                                         aux.conj().T)

        sum_term2 = sum_term + self.noise_var * np.eye(self.Nr[k])

        Uk = np.linalg.solve(sum_term2, np.dot(Hkk, Vk))
        return Uk

    def _updateW(self):
        """
        Updates the receive filter of all users.
        """
        new_W = np.zeros(self.K, dtype=np.ndarray)
        for k in range(self.K):
            new_W[k] = self._calc_Uk(k)

        self._clear_receive_filter()
        self._W = new_W

    @staticmethod
    def _calc_Vi_for_a_given_mu(sum_term, mu_i, H_herm_U):
        """
        Calculates the value of Vi for the given parameters.

        This method is called inside _calc_Vi.

        Parameters
        ----------
        sum_term : numpy array
            The sumation term in the formula to calculate the precoder.
        mu_i : float
            The value of the lagrange multiplier
        H_herm_U : numpy array
            The value of :math:`H_ii^H U_i`
        """
        N = sum_term.shape[0]
        Vi = np.linalg.solve(sum_term + mu_i * np.eye(N), H_herm_U)
        # Vi = np.dot(np.linalg.inv(sum_term + mu_i * np.eye(N)),
        #             H_herm_U)

        return Vi

    @staticmethod
    def _calc_Vi_for_a_given_mu2(inv_sum_term, mu_i, H_herm_U):
        """
        Calculates the value of Vi for the given parameters.

        This method is called inside _calc_Vi.

        Parameters
        ----------
        inv_sum_term : numpy array
            The inverse of the sumation term in the formula to calculate
            the precoder when mu_i is equal to zero.
        mu_i : float
            The value of the lagrange multiplier
        H_herm_U : numpy array
            The value of :math:`H_ii^H U_i`
        """
        N = inv_sum_term.shape[0]
        diagonal = mu_i * np.ones(N)  # Vector of N elements
        new_inv = update_inv_sum_diag(inv_sum_term, diagonal)
        Vi = np.dot(new_inv, H_herm_U)
        return Vi

    def _calc_Vi(self, i, mu_i=None):
        """
        Calculates the precoder of the i-th user.

        Parameters
        ----------
        i : int
            User index
        mu_i : float or None
            The value of the Lagrange multiplier. If it is None (default),
            then the best value will be found and used to calculate the
            precoder.

        Returns
        -------
        Vi : numpy array
            The calculate precoder of the i-th user.
        """
        # $$\mtV_i = \left( \sum_{k=1}^K \mtH_{ki}^H \mtU_k \mtU_k^H \mtH_{ki} + \mu_i \mtI \right)^{-1} \mtH_{ii}^H \mtU_i$$
        Hii = self._get_channel(i, i)
        Ui = self.W[i]

        Hii_herm_U = np.dot(Hii.conj().T, Ui)

        # xxxxx Calculates the Summation term xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        sum_term = 0
        for k in range(self.K):
            Hki = self._get_channel(k, i)
            Uk = self.W[k]
            aux = np.dot(Hki.conj().T, Uk)
            sum_term = sum_term + np.dot(aux, aux.conj().T)
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Perform diagonal loading if it is necessary xxxxxxxxxx
        # Occasionally (specially for the low SNR values) sum_term may be a
        # singular matrix. In that case, we add a diagonal loading factor
        # to make the sum_term non-singular. This diagonal loading
        # corresponding to summing an identity matrix times a load_factor
        # to the sum_term, where this load_factor is arbitrarily calculated
        # as 1/100 of the mean of the eigen values of the singular
        # sum_term.

        # Calculates the SVD of sum_term so that we can calculate the
        # condition number.
        [_, S, _] = np.linalg.svd(sum_term)
        cond = cond = S.max() / S.min()
        load_factor = 0.0
        # If the condition number is larger than 1e8 we consider sum_term
        # as a singular matrix, which means that we will perform the
        # diagonal loading
        if cond > 5e4:
            # Calculates the load_factor (arbitrarily choosen as 1/100 the
            # mean of the current singular values of sum_term).
            load_factor = S.mean() / 100.0
            # pylint: disable= E1103
            sum_term = sum_term + np.eye(sum_term.shape[0]) * load_factor
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Case when the best mu value must be found xxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        if mu_i is None:
            # xxxxx Define the function that will be optimized xxxxxxxxxxxx
            def func(mu, sum_term, Hii_herm_U, P):
                """
                Function that will be optimized to find the best value of mu.
                """
                Vi = self._calc_Vi_for_a_given_mu(sum_term, mu, Hii_herm_U)
                norm = np.linalg.norm(Vi, 'fro')
                cost = (norm ** 2) - P
                return cost
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            min_mu_i = 0

            cost = func(min_mu_i, sum_term, Hii_herm_U, self.P[i])
            # If cost is lower than or equal to zero then the power
            # constraint is already satisfied and we are done. The value of
            # mu will be min_mu_i.
            if cost <= 0:
                mu_i = min_mu_i
                Vi = self._calc_Vi_for_a_given_mu(
                    sum_term, mu_i, Hii_herm_U)
                self._mu[i] = mu_i

            else:
                # If we are not done yet then we need to perform the
                # bisection method to find the best mu value between
                # min_mu_i and max_mu_i
                mu_i = optimize.fsolve(
                    func, min_mu_i,
                    args=(sum_term, Hii_herm_U, self.P[i]))[0]

                # Now that we have the best value for mu_i, lets calculate Vi
                Vi = self._calc_Vi_for_a_given_mu(
                    sum_term, mu_i, Hii_herm_U)
                # Vi = self._calc_Vi_for_a_given_mu2(
                #     inv_sum_term, mu_i, Hii_herm_U)

                # If any load_factor was added (in case the original
                # sum_term is a singular matrix) we will add it to the
                # optimum mu_i, since this is the effective value of mu_i.
                self._mu[i] = mu_i + load_factor

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Case when the mu value is provided xxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        else:
            self._mu[i] = mu_i
            Vi = self._calc_Vi_for_a_given_mu(sum_term, mu_i, Hii_herm_U)
            # Vi = self._calc_Vi_for_a_given_mu2(inv_sum_term, mu_i, Hii_herm_U)

        return Vi


    # def _calc_Vi_orig(self, i, mu_i=None):
    #     """
    #     Calculates the precoder of the i-th user.

    #     Parameters
    #     ----------
    #     i : int
    #         User index
    #     mu_i : float or None
    #         The value of the Lagrange multiplier. If it is None (default),
    #         then the best value will be found and used to calculate the
    #         precoder.

    #     Returns
    #     -------
    #     Vi : numpy array
    #         The calculate precoder of the i-th user.
    #     """
    #     # $$\mtV_i = \left( \sum_{k=1}^K \mtH_{ki}^H \mtU_k \mtU_k^H \mtH_{ki} + \mu_i \mtI \right)^{-1} \mtH_{ii}^H \mtU_i$$
    #     Hii = self._get_channel(i, i)
    #     Ui = self.W[i]

    #     Hii_herm_U = np.dot(Hii.conj().T, Ui)

    #     # xxxxx Calculates the Summation term xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #     sum_term = 0
    #     for k in range(self.K):
    #         Hki = self._get_channel(k, i)
    #         Uk = self.W[k]
    #         aux = np.dot(Hki.conj().T, Uk)
    #         sum_term = sum_term + np.dot(aux, aux.conj().T)
    #     # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    #     # xxxxxxxxxx Perform diagonal loading if it is necessary xxxxxxxxxx
    #     # Occasionally (specially for the low SNR values) sum_term may be a
    #     # singular matrix. In that case, we add a diagonal loading factor
    #     # to make the sum_term non-singular. This diagonal loading
    #     # corresponding to summing an identity matrix times a load_factor
    #     # to the sum_term, where this load_factor is arbitrarily calculated
    #     # as 1/100 of the mean of the eigen values of the singular
    #     # sum_term.

    #     # Calculates the SVD of sum_term so that we can calculate the
    #     # condition number.
    #     [_, S, _] = np.linalg.svd(sum_term)
    #     cond = cond = S.max() / S.min()
    #     load_factor = 0.0
    #     # If the condition number is larger than 1e8 we consider sum_term
    #     # as a singular matrix, which means that we will perform the
    #     # diagonal loading
    #     if cond > 5e4:
    #         # Calculates the load_factor (arbitrarily choosen as 1/100 the
    #         # mean of the current singular values of sum_term).
    #         load_factor = S.mean() / 100.0
    #         # pylint: disable= E1103
    #         sum_term = sum_term + np.eye(sum_term.shape[0]) * load_factor
    #     # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    #     # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #     # xxxxxxxxxx Case when the best mu value must be found xxxxxxxxxxxx
    #     # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #     if mu_i is None:
    #         # xxxxx Define the function that will be optimized xxxxxxxxxxxx
    #         def func(mu, sum_term, Hii_herm_U, P):
    #             Vi = self._calc_Vi_for_a_given_mu(sum_term, mu, Hii_herm_U)
    #             norm = np.linalg.norm(Vi, 'fro')
    #             cost = norm**2  - P
    #             return cost
    #         # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    #         min_mu_i = 0
    #         max_mu_i = 10  # (10 was arbitrarily chosen, but seems good enough)

    #         cost = func(min_mu_i, sum_term, Hii_herm_U, self.P[i])
    #         # If cost is lower than or equal to zero then the power
    #         # constraint is already satisfied and we are done. The value of
    #         # mu will be min_mu_i.
    #         if cost <= 0:
    #             mu_i = min_mu_i
    #             Vi = self._calc_Vi_for_a_given_mu(
    #                 sum_term, mu_i, Hii_herm_U)
    #             self._mu[i] = mu_i

    #         else:
    #             # If we are not done yet then we need to perform the
    #             # bisection method to find the best mu value between
    #             # min_mu_i and max_mu_i
    #             tol = self._bisection_tol  # Tolerance used to stop the bisection method

    #             # Maximum number of iterations of the bisection
    #             max_iter = int(1 + np.round(
    #                 (np.log(max_mu_i - min_mu_i) - np.log(tol)) / np.log(2)))

    #             # Perform the bisection
    #             for _ in range(max_iter):
    #                 mu_i = (max_mu_i + min_mu_i) / 2.0
    #                 cost = func(mu_i, sum_term, Hii_herm_U, self.P[i])
    #                 if cost > 0:
    #                     # The current value of mu_i yields a precoder with
    #                     # a power higher then the allowed value. Lets
    #                     # increase the value of min_mu_i
    #                     min_mu_i = mu_i
    #                 else:
    #                     # The current value of mu_i yields a precoder with
    #                     # a power lower then the allowed value. Lets
    #                     # decrease the value of max_mu_i
    #                     max_mu_i = mu_i
    #                 if np.abs(cost) < tol:
    #                     break

    #             # Now that we have the best value for mu_i, lets calculate Vi
    #             Vi = self._calc_Vi_for_a_given_mu(
    #                 sum_term, mu_i, Hii_herm_U)
    #             # Vi = self._calc_Vi_for_a_given_mu2(
    #             #     inv_sum_term, mu_i, Hii_herm_U)

    #             # If any load_factor was added (in case the original
    #             # sum_term is a singular matrix) we will add it to the
    #             # optimum mu_i, since this is the effective value of mu_i.
    #             self._mu[i] = mu_i + load_factor

    #     # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #     # xxxxxxxxxx Case when the mu value is provided xxxxxxxxxxxxxxxxxxx
    #     # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #     else:
    #         self._mu[i] = mu_i
    #         Vi = self._calc_Vi_for_a_given_mu(sum_term, mu_i, Hii_herm_U)
    #         # Vi = self._calc_Vi_for_a_given_mu2(inv_sum_term, mu_i, Hii_herm_U)

    #     return Vi

    def _updateF(self):
        """
        Updates the precoder of all users.
        """
        # Note that _mu should never be negative. By setting the values to
        # -1 here if after the solve method some element in _mu is still
        # negative then something was not write.
        self._mu = -1.0 * np.ones(self.K, dtype=float)

        Vi = np.zeros(self.K, dtype=np.ndarray)
        norm_Vi = np.zeros(self.K, dtype=np.ndarray)
        for k in range(self.K):
            # Note: The square of the Frobenius norm of Vi is NOT equal to
            # one. Since the full_F property will apply the correct power
            # scaling, the norm of self._F must be equal to one.
            Vi[k] = self._calc_Vi(k)
            norm_Vi[k] = Vi[k] / np.linalg.norm(Vi[k], 'fro')

        self._clear_precoder_filter()
        self._full_F = Vi
        self._F = norm_Vi


# xxxxxxxxxx End of the File xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
