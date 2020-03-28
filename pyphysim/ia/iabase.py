#!/usr/bin/env python
"""
Module containing the base class for Interference Alignment (IA)
Algorithms.

This module should probably only be imported in the other modules inside
the 'ia' package that implement the IA algorithms.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Sequence, TypeVar, Union, cast

import numpy as np

import pyphysim.channels.multiuser as muchannels

from ..util.conversion import linear2dB
from ..util.misc import leig, randn_c_RS

NumberOrArray = TypeVar("NumberOrArray", np.ndarray, float)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Base Class for all IA Algorithms xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class IASolverBaseClass:  # pylint: disable=R0902
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

    Another method that can be implemented is the get_cost method. It
    should return the cost of the current IA solution. What is
    considered "the cost" varies from one IA algorithm to another,
    but should always be a real non-negative number. If get_cost is not
    implemented a value of -1 is returned.

    Parameters
    ----------
    multiUserChannel : muchannels.MultiUserChannelMatrix
        The multiuser channel.
    """

    # The IASolverBaseClass is an abstract class and the 'solve' method
    # (marked as abstract) must be implemented in a subclass.
    __metaclass__ = ABCMeta

    def __init__(self, multiUserChannel: muchannels.MultiUserChannelMatrix):
        # xxxxxxxxxx Private attributes xxxxxxxxxxxxxxx
        if not isinstance(multiUserChannel, muchannels.MultiUserChannelMatrix):
            raise ValueError("multiUserChannel must be an object of the "
                             "MultiUserChannelMatrix class (or a subclass).")
        # Channel of all users
        self._multiUserChannel = multiUserChannel

        # Number of streams per user
        self._Ns: Optional[np.ndarray] = None
        # Power of each user (P is an 1D numpy array). If not set (_P is
        # None), then a power of 1 will be used for each transmitter.
        self._P: Optional[np.ndarray] = None

        # xxxxxxxxxx Precoder and receive filters xxxxxxxxxxxxxxxxxxxxxxxxx
        # These are numpy arrays of numpy arrays

        # Precoder: One precoder for each user
        self._F: Optional[np.ndarray] = None
        # Precoder: Same as _F, but scaled with the correct power value in
        # self.P
        self._full_F: Optional[np.ndarray] = None
        # Receive filter: One for each user
        self._W: Optional[np.ndarray] = None
        # Receive filter (hermitian): One for each user
        self._W_H: Optional[np.ndarray] = None
        self._full_W_H: Optional[np.ndarray] = None
        self._full_W: Optional[np.ndarray] = None
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxx Other member variables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # RandomState object used to randomize the precoder
        self._rs = np.random.RandomState()
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _clear_receive_filter(self) -> None:
        """
        Clear the receive filter.

        This should be called in the beginning of the implementation of the
        `_updateW` method in subclasses.
        """
        self._W = None
        self._W_H = None
        self._full_W_H = None
        self._full_W = None

    def _clear_precoder_filter(self) -> None:
        """
        Clear the precoder filter.

        This should be called in the beginning of the implementation of the
        `_updateF` method in subclasses.
        """
        self._F = None
        self._full_F = None

    def clear(self) -> None:
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

    def get_cost(self) -> float:  # pylint: disable=R0201
        """
        Get the current cost of the IA Solution.

        This method should be implemented in subclasses and return a number
        greater than or equal to zero..

        Returns
        -------
        cost : float
            The Cost of the current IA solution (a real non-negative
            number).
        """
        return -1

    @property
    def noise_var(self) -> float:
        """
        Get method for the noise_var property.

        Returns
        -------
        float
            The noise variance (a real non-negative number).
        """
        noise_var = self._multiUserChannel.noise_var
        if noise_var is None:
            return 0.0

        return noise_var

    @property
    def F(self) -> np.ndarray:
        """
        Transmit precoder of all users.

        Returns
        -------
        np.ndarray
            The precoders of all users (a 1D numpy array of 2D numpy
            arrays).
        """
        return self._F

    @property
    def full_F(self) -> np.ndarray:
        """
        Transmit precoder of all users.

        Returns
        -------
        np.ndarray
            The precoders of all users (a 1D numpy array of 2D numpy
            arrays).
        """
        if self._full_F is None:
            self._full_F = self._F * np.sqrt(self.P)
        return self._full_F

    # noinspection PyUnresolvedReferences
    def set_precoders(self,
                      F: Optional[Sequence[np.ndarray]] = None,
                      full_F: Optional[Sequence[np.ndarray]] = None,
                      P: Optional[np.ndarray] = None) -> None:
        """
        Set the precoders of each user.

        Either `F` or `full_F` (or both of them) must be provided.

        If only `full_F` is provided then the value of `F` will be
        calculated from `full_F`.

        In any case, the value of `self.Ns` will be updated according to
        the dimensions of the `F` or `full_F`.

        Parameters
        ----------
        F : np.ndarray | list[np.ndarray], optional
            A numpy array where each element is the (normalized) precoder
            (a 2D numpy array) of one user.
        full_F : np.ndarray | list[np.ndarray], optional
            A numpy array where each element is the precoder (a 2D numpy
            array) of one user.
        P : np.ndarray, optional
            The maximum transmit power. If not provided the current value
            of self.P will be kept as it is.
        """
        if F is None and full_F is None:
            raise RuntimeError("Either 'F' or 'full_F' must be provided.")

        self._clear_precoder_filter()

        if P is not None:
            self._P = P

        self._full_F = full_F

        if F is None:
            assert (full_F is not None)
            K = len(full_F)
            self._F = np.empty(K, dtype=np.ndarray)
            for k in range(K):
                self._F[k] = full_F[k] / np.linalg.norm(full_F[k], 'fro')
        else:
            self._F = F

        # Update the number of streams
        self._Ns = np.empty(self.K, dtype=int)
        for k in range(self.K):
            self._Ns[k] = self._F[k].shape[1]

    # The W property should return a receive filter that can be directly
    # multiplied (no need to calculate the hermitian of W) by the received
    # signal to cancel interference and compensate the effect of the
    # channel.
    @property
    def W(self) -> np.ndarray:
        """
        Receive filter of all users.

        Returns
        -------
        np.ndarray
            The receive filter of all users. (a 1D numpy array of 2D numpy
            arrays).
        """
        # If self._W is None but self._W_H is not None than we need to
        # update self_W from self._W_H.
        if self._W is None:
            if self._W_H is not None:
                self._W = np.empty(self.K, dtype=np.ndarray)
                for k in range(self.K):
                    self._W[k] = self._W_H[k].conj().T
        return self._W

    @property
    def W_H(self) -> np.ndarray:
        """
        Get method for the W_H property.

        Returns
        -------
        np.ndarray
            The conjugate of the receive filter of all users. (a 1D numpy
            array of 2D numpy arrays).
        """
        # If self._W_H is None but self._W is not None than we need to
        # update self_W_H from self._W.
        if self._W_H is None:
            if self._W is not None:
                self._W_H = np.empty(self.K, dtype=np.ndarray)
                for k in range(self.K):
                    self._W_H[k] = self._W[k].conj().T
        return self._W_H

    @property
    def full_W_H(self) -> np.ndarray:
        """
        Get method for the full_W_H property.

        The full_W_H property returns the equivalent filter of the IA
        filter plus the post processing filter.

        Returns
        -------
        np.ndarray
            The equivalent filter of the IA filter plus the post processing
            filter.
        """
        if self._full_W_H is None:
            if self.W_H is not None:
                self._full_W_H = np.empty(self.K, dtype=np.ndarray)
                for k in range(self.K):
                    # Equivalent channel with the effect of the precoder,
                    # channel and receive filter
                    Hieq = self._calc_equivalent_channel(k)
                    # TODO: Put this in a try-except block for the case
                    # that Hieq is singular (a linalg.LinAlgError exception
                    # is thrown). In order to handle the exception, you
                    # could set full_W_H to just W_H or you could try the
                    # stream reduction (but stream reduction should be
                    # performed at the precoders too)
                    self._full_W_H[k] = np.linalg.solve(Hieq, self.W_H[k])

        return self._full_W_H

    @property
    def full_W(self) -> np.ndarray:
        """
        Get method for the full_W property.

        The full_W property returns the equivalent filter of the IA
        filter plus the post processing filter.

        Returns
        -------
        np.ndarray
            the equivalent filter of the IA filter plus the post processing
            filter.
        """
        if self._full_W is None:
            self._full_W = np.empty(self.K, dtype=np.ndarray)
            for k in range(self.K):
                self._full_W[k] = self.full_W_H[k].conj().T
        return self._full_W

    def set_receive_filters(self,
                            W_H: Optional[Sequence[np.ndarray]] = None,
                            W: Optional[Sequence[np.ndarray]] = None) -> None:
        """
        Set the receive filters.

        You only need to pass either `W_H` or `W`, but not both of them,
        since one is calculated from the other.

        Parameters
        ----------
        W_H : np.ndarray | list[np.ndarray]
            A numpy array where each element is the receive filter (a 2D
            numpy array) of one user. This is a 1D numpy array of 2D numpy
            arrays.
        W : np.ndarray | list[np.ndarray]
            A numpy array where each element is the receive filter (a 2D
            numpy array) of one user. This is a 1D numpy array of 2D numpy
            arrays.
        """
        self._clear_receive_filter()

        if W is None and W_H is None:
            raise RuntimeError("Either 'W' or 'W_H' must be provided.")

        if W is not None and W_H is not None:
            raise RuntimeError("Either 'W' or 'W_H' must be provided ("
                               "but not both of them.")

        self._W = W
        self._W_H = W_H

    def _calc_equivalent_channel(self, k: int) -> np.ndarray:
        """
        Calculates the equivalent channel for user :math:`k` considering
        the effect of the precoder (including transmit power),
        the actual channel, and the receive filter (without power
        compensation).

        Parameters
        ----------
        k : int
            The index of the desired user.

        Returns
        -------
        Hk_eq : np.ndarray
            The equivalent channel.

        Notes
        -----
        This method is used only internally in order to calculate the "W"
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
    def P(self) -> np.ndarray:
        """
        Transmit power of all users.

        Returns
        -------
        np.ndarray
            The power of all users.
        """
        if self._P is None:
            return np.ones(self.K, dtype=float)

        return self._P

    @P.setter
    def P(self, value: Optional[NumberOrArray]) -> None:
        """Transmit power of all users.

        Parameters
        ----------
        value : float | np.ndarray
            The new power of all users.
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
            assert (not isinstance(value, float))
            if len(value) != self.K:
                raise ValueError("P must be set to a sequence of length K")

            value = np.array(value)
            assert (isinstance(value, np.ndarray))
            if np.all(value > 0.0):
                self._P = np.array(value)
            else:
                raise ValueError("P cannot be negative or equal to zero.")

    @property
    def Ns(self) -> np.ndarray:
        """
        Number of streams of all users.

        Returns
        -------
        Ns : np.ndarray
            Number of streams of all users.
        """
        return self._Ns

    # xxxxx Properties to read the channel related variables xxxxxxxxxxxxxx
    @property
    def K(self) -> int:
        """
        The number of users.

        Returns
        -------
        K : int
            The number of users.
        """
        K = self._multiUserChannel.K
        assert K > 0, ("You must initialize the channel object before"
                       " using the IA solver object.")
        return K

    @property
    def Nr(self) -> np.ndarray:
        """
        Number of receive antennas of all users.

        Returns
        -------
        Nr : np.ndarray
            Number of receive antennas of all users.
        """
        return self._multiUserChannel.Nr

    @property
    def Nt(self) -> np.ndarray:
        """
        Number of transmit antennas of all users.

        Returns
        -------
        Nt : np.ndarray
            Number of transmit antennas of all users.
        """
        return self._multiUserChannel.Nt

    def randomizeF(self,
                   Ns: Union[int, List[int], Sequence[int]],
                   P: Optional[np.ndarray] = None) -> None:
        """
        Generates a random precoder for each user.

        Parameters
        ----------
        Ns : int | list[int] | np.ndarray
            Number of streams of each user.
        P : np.ndarray, optional
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        self._clear_precoder_filter()

        if isinstance(Ns, int):
            Ns = np.ones(self.K, dtype=int) * Ns
        assert (not isinstance(Ns, int))

        self.P = P

        # Local function that returns a normalized version of the input
        # numpy array
        def normalized(A: np.ndarray) -> np.ndarray:
            return A / np.linalg.norm(A, 'fro')

        self._F = np.zeros(self.K, dtype=np.ndarray)
        for k in range(self.K):
            self._F[k] = normalized(randn_c_RS(self._rs, self.Nt[k], Ns[k]))

        # This will create a new array so that we can modify self._Ns
        # internally without changing the original Ns variable passed to
        # the randomizeF method.
        self._Ns = np.array(Ns)

    # This method is just an alias for the get_channel method of the
    # multiuserchannel object associated with the IA Solver.xs
    def _get_channel(self, k: int, l: int) -> np.ndarray:
        """
        Get the channel from transmitter l to receiver k.

        Parameters
        ----------
        l : int
            Transmitting user.
        k : int
            Receiving user.

        Returns
        -------
        H : np.ndarray
            The channel matrix between transmitter l and receiver k.
        """
        return self._multiUserChannel.get_Hkl(k, l)

    def _get_channel_rev(self, k: int, l: int) -> np.ndarray:
        """
        Get the channel from transmitter l to receiver k in the reverse
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
        H : np.ndarray
            The channel matrix between transmitter l and receiver k in the
            reverse network.

        Notes
        -----
        See Section III of [Cadambe2008]_ for details.
        """
        return self._get_channel(l, k).transpose().conjugate()

    # noinspection PyPep8
    def calc_Q(self, k: int) -> np.ndarray:
        """
        Calculates the interference covariance matrix at the :math:`k`-th
        receiver.

        The interference covariance matrix at the :math:`k`-th receiver,
        :math:`\\mtQ k`, is given by

            .. math::

               \\mtQ k = \\sum_{j=1, j \\neq k}^{K} \\frac{P_j}{Ns_j} \\mtH_{kj} \\mtF_j \\mtF_j^H \\mtH_{kj}^H

        where :math:`P_j` is the transmit power of transmitter :math:`j`,
        and :math:`Ns_j` is the number of streams for user :math:`j`.

        Parameters
        ----------
        k : int
            Index of the desired receiver.

        Returns
        -------
        Qk : np.ndarray
            The interference covariance matrix at receiver :math:`k`.

        Notes
        -----
        This is impacted by the self.P attribute.
        """
        Qk = self._multiUserChannel.calc_Q(k, self.full_F)
        return Qk

    # This method must be tested in a subclass of IASolverBaseClass, since
    # we need the receive filter and IASolverBaseClass does not know how to
    # calculate it
    def calc_Q_rev(self, k: int) -> np.ndarray:
        """
        Calculates the interference covariance matrix at the :math:`k`-th
        receiver in the reverse network.

        Parameters
        ----------
        k : int
            Index of the desired receiver.

        Returns
        -------
        Qk_rev : np.ndarray
            The interference covariance matrix at receiver :math:`k` in the
            reverse network.

        See also
        --------
        calc_Q
        """
        P = self.P
        interfering_users = set(range(self.K)) - {k}
        Qk = np.zeros([self.Nt[k], self.Nt[k]], dtype=complex)

        for l in interfering_users:
            # The lets make sure the receive filter norm is equal to one so
            # that we can correctly scale it to the desired power.
            assert (isinstance(self._W, np.ndarray))
            assert np.linalg.norm(self._W[l], 'fro') - 1.0 < 1e-6
            Hkl_F_rev = np.dot(self._get_channel_rev(k, l), self._W[l])
            Qk = Qk + np.dot(P[l] * Hkl_F_rev, Hkl_F_rev.conjugate().T)

        return Qk

    # noinspection PyPep8
    def calc_remaining_interference_percentage(self,
                                               k: int,
                                               Qk: Optional[np.ndarray] = None
                                               ) -> float:
        """
        Calculates the percentage of the interference in the desired signal
        space according to equation (30) in [Cadambe2008]_.

        The percentage :math:`p_k` of the interference in the desired
        signal space is given by

            .. math::

               p_k = \\frac{\\sum_{j=1}^{Ns[k]} \\lambda_j [\\mtQ k]}{Tr[\\mtQ k]}

        where :math:`\\lambda_j[\\mtA]` denotes the :math:`j`-th smallest
        eigenvalue of :math:`\\mtA`.

        Parameters
        ----------
        k : int
            The index of the desired user.
        Qk : np.ndarray
            The covariance matrix of the remaining interference at receiver
            k. If not provided, it will be automatically calculated. In
            that case, the `P` attribute will also be taken into account if
            it is set.

        Returns
        -------
        float
            The percentage of the interference in the desired signal space.

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
        return cast(float, pk)

    def calc_SINR_old(self) -> np.ndarray:
        """
        Calculates the SINR values (in linear scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property, which, if not explicitly set, will use the
        noise_var property of the multiuserchannel object.

        This method is deprecated since it's not the correct way to
        calculate the SINR. Use the calc_SINR method instead.

        Returns
        -------
        SINRs : np.ndarray
            The SINR (in linear scale) of all streams of all users. This
            is a 1D numpy array of 1D numpy arrays (of floats).
        """
        K = self.K
        SINRs: np.array = np.empty(K, dtype=np.ndarray)

        for j in range(K):
            numerator = 0.0
            denominator = 0.0
            Wj_H = self.W_H[j]
            for i in range(K):
                Hji = self._get_channel(j, i)
                Fi = self.F[i]
                aux: np.ndarray = np.dot(Wj_H, np.dot(Hji, Fi))
                if i == j:
                    aux = np.dot(aux, aux.transpose().conjugate())
                    # Numerator will be a 1D numpy array with length equal
                    # to the number of streams
                    numerator = numerator + np.diag(np.abs(aux))
                else:
                    denominator = denominator + aux

            # pylint: disable=E1103
            # noinspection PyUnresolvedReferences
            denominator = np.dot(
                denominator,
                denominator.transpose().conjugate())  # type: ignore
            noise_power = self.noise_var * np.dot(Wj_H,
                                                  Wj_H.transpose().conjugate())
            denominator += noise_power
            denominator = np.diag(np.abs(denominator))

            SINRs[j] = numerator / denominator

        return SINRs

    def calc_SINR(self) -> np.ndarray:
        """
        Calculates the SINR values (in linear scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property, which, if not explicitly set, will use the
        noise_var property of the multiuserchannel object.

        Returns
        -------
        SINRs : np.ndarray
            The SINR (in linear scale) of all streams of all users. This is a
            1D numpy array of 1D numpy arrays (of floats).
        """
        K = self.K
        SINRs = np.empty(K, dtype=np.ndarray)

        for k in range(self.K):
            Bkl_all_l = self._calc_Bkl_cov_matrix_all_l(k, self.noise_var)
            SINRs[k] = self._calc_SINR_k(k, Bkl_all_l)
        return SINRs

    def calc_SINR_in_dB(self) -> np.ndarray:
        """
        Calculates the SINR values (in dB scale) of all streams of all
        users with the current IA solution.

        The noise variance used will be the value of the noise_var
        property, which, if not explicitly set, will use the
        noise_var property of the multiuserchannel object.

        Returns
        -------
        SINRs : np.ndarray
            The SINR (in dB scale) of all streams of all users. This is a 1D
            numpy array of 1D numpy arrays (of floats).
        """
        K = self.K
        SINRs = np.empty(K, dtype=np.ndarray)

        for k in range(self.K):
            Bkl_all_l = self._calc_Bkl_cov_matrix_all_l(k, self.noise_var)
            SINRs[k] = linear2dB(self._calc_SINR_k(k, Bkl_all_l))
        return SINRs

    def calc_sum_capacity(self) -> float:
        """
        Calculates the sum capacity of the current solution.

        The SINRs are estimated and applied to the Shannon capacity formula

        Returns
        -------
        float
            The sum capacity of the current solution.
        """
        return cast(float, np.sum(np.log2(1 + np.hstack(self.calc_SINR()))))

    # noinspection PyPep8
    def _calc_Bkl_cov_matrix_first_part(self, k: int) -> np.ndarray:
        """
        Calculates the first part in the equation of the Blk covariance matrix
        in equation (28) of [Cadambe2008]_.

        The first part is given by

            .. math::

               \\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star d}^{[j]} \\mtV_{\\star d}^{[j]\\dagger} \\mtH^{[kj]\\dagger}

        Note that it only depends on the value of :math:`k`.

        Parameters
        ----------
        k : int
            Index of the desired user.

        Returns
        -------
        Bkl_first_part : np.ndarray
            First part in equation (28) of [Cadambe2008]_.
        """
        # $$\sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$$
        first_part = 0.0
        for j in range(self.K):
            Hkj = self._get_channel(k, j)
            Vj = self.full_F[j]

            aux = np.dot(Hkj, Vj)

            first_part = first_part + np.dot(aux, aux.conjugate().T)

        return first_part

    # noinspection PyPep8
    def _calc_Bkl_cov_matrix_second_part(self, k: int, l: int) -> np.ndarray:
        """
        Calculates the second part in the equation of the Blk covariance matrix
        in equation (28) of [Cadambe2008]_ (note that it does not include
        the identity matrix).

        The second part is given by

            .. math::

               \\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger}

        Parameters
        ----------
        k : int
            Index of the desired user.
        l : int
            Index of the desired stream.

        Returns
        -------
        second_part : np.ndarray
            Second part in equation (28) of [Cadambe2008]_.
        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger}$$
        Hkk = self._get_channel(k, k)
        Vkl = self.full_F[k][:, l:l + 1]
        aux = np.dot(Hkk, Vkl)
        second_part = np.dot(aux, aux.conjugate().T)

        return second_part

    # noinspection PyPep8
    def _calc_Bkl_cov_matrix_all_l(self,
                                   k: int,
                                   noise_power: Optional[float] = None
                                   ) -> np.ndarray:
        """
        Calculates the interference-plus-noise covariance matrix for all
        streams at receiver :math:`k` according to equation (28) in
        [Cadambe2008]_.

        The interference-plus-noise covariance matrix for stream :math:`l`
        of user :math:`k` is given by Equation (28) in [Cadambe2008]_,
        which is reproduced below

            .. math::

               \\mtB^{[kl]} = \\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star l}^{[j]} \\mtV_{\\star l}^{[j]\\dagger} \\mtH^{[kj]\\dagger} - \\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger} + \\mtI_{N^{[k]}}

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
        Bkl : np.ndarray
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
        # $$\mtB^{[kl]} = \sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star l}^{[j]} \mtV_{\star l}^{[j]\dagger} \mtH^{[kj]\dagger} - \frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger} + \sigma_n^2 \mtI_{N^{[k]}}$$
        if noise_power is None:
            noise_power = self.noise_var

        assert (self._Ns is not None)
        Bkl_all_l = np.empty(self._Ns[k], dtype=np.ndarray)
        first_part = self._calc_Bkl_cov_matrix_first_part(k)
        for l in range(self._Ns[k]):
            second_part = self._calc_Bkl_cov_matrix_second_part(k, l)
            Bkl_all_l[l] = first_part - second_part + (noise_power *
                                                       np.eye(self.Nr[k]))

        return Bkl_all_l

    def _calc_SINR_k(self, k: int,
                     Bkl_all_l: Sequence[np.ndarray]) -> np.ndarray:
        """
        Calculates the SINR of all streams of user 'k'.

        Parameters
        ----------
        k : int
            Index of the desired user.
        Bkl_all_l : list[np.ndarray] | nd.ndarray
            A sequence (1D numpy array, a list, etc) of 2D numpy arrays
            corresponding to the Bkl matrices for all 'l's.

        Returns
        -------
        SINR_k : np.ndarray
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
            aux = np.dot(Ukl_H, np.dot(Hkk, Vkl))
            numerator = np.dot(aux, aux.transpose().conjugate())
            denominator = np.dot(Ukl_H, np.dot(Bkl_all_l[l], Ukl))
            SINR_kl = numerator.item() / denominator.item()
            # The imaginary part should be negligible
            SINR_k[l] = np.abs(SINR_kl)

        return SINR_k

    @abstractmethod
    def solve(self,
              Ns: Union[int, np.ndarray],
              P: Optional[np.ndarray] = None) -> Any:  # pragma: no cover
        """
        Find the IA solution.

        This method must be implemented in a subclass and should updates
        the 'F' and 'W' member variables.

        Parameters
        ----------
        Ns : int | np.ndarray
            Number of streams of each user.
        P : np.ndarray
            Power of each user. If not provided, a value of 1 will be used
            for each user.

        Notes
        -----
        This function should be implemented in the derived classes
        """
        raise NotImplementedError("solve: Not implemented")
