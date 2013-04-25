#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Module with implementation of Interference Alignment algorithms"""

__revision__ = "$Revision$"

import numpy as np
import itertools

from util.misc import peig, leig, randn_c
from comm.channels import MultiUserChannelMatrix

__all__ = ['IASolverBaseClass', 'AlternatingMinIASolver',
           'MaxSinrIASolverIASolver']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Base Class for IA Algorithms xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class IASolverBaseClass(object):
    """Base class for Interference Alignment Algorithms.
    """

    def __init__(self):
        """Initialize the variables that every IA solver will have.
        """
        # The F and W variables will be numpy arrays OF numpy arrays.
        self.F = np.array([])  # Precoder: One precoder for each user
        self.W = np.array([])  # Receive filter: One for each user

        # xxxxxxxxxx Private attributes xxxxxxxxxxxxxxx
        # Number of streams per user
        self._Ns = np.array([])
        # Channel of all users
        self._multiUserChannel = MultiUserChannelMatrix()

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
        return self._multiUserChannel._Nt

    def randomizeF(self, Nt, Ns, K):
        """Generates a random precoder for each user.

        Parameters
        ----------
        K : int
            Number of users.
        Nt : int or 1D numpy array
            Number of transmit antennas of each user.
        Ns : int or 1D numpy array
            Number of streams of each user.
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

    # This method does not need testing, since the logic is implemented in
    # the MultiUserChannelMatrix class and it is already tested.
    def randomizeH(self, Nr, Nt, K):
        """Generates a random channel matrix for all users.

        Parameters
        ----------
        K : int
            Number of users.
        Nr : int or 1D numpy array
            Number of receive antennas of each user.
        Nt : int or 1D numpy array
            Number of transmit antennas of each user.
        """
        self._multiUserChannel.randomize(Nr, Nt, K)

    # This method does not need testing, since the logic is implemented in
    # the MultiUserChannelMatrix class and it is already tested.
    def init_from_channel_matrix(self, channel_matrix, Nr, Nt, K):
        """Initializes the multiuser channel matrix from the given
        `channel_matrix`.

        Parameters
        ----------
        channel_matrix : 2D numpy array
            A matrix concatenating the channel of all users (from each
            transmitter to each receiver).
        Nr : 1D numpy array
            The number of receive antennas of each user.
        Nt : 1D numpy array
            The number of transmit antennas of each user.
        K : int
            Number of users.

        Raises
        ------
        ValueError
            If the arguments are invalid.

        """
        self._multiUserChannel.init_from_channel_matrix(channel_matrix, Nr,
                                                        Nt, K)

    # This method does not need testing, since the logic is implemented in
    # the MultiUserChannelMatrix class and it is already tested.
    def get_channel(self, k, l):
        """Get the channel from user l to user k.

        Parameters
        ----------
        l : int
            Transmitting user.
        k : int
            Receiving user

        Returns
        -------
        H : 2D numpy array
            The channel matrix between transmitter l and receiver k.
        """
        return self._multiUserChannel.get_channel(k, l)

    def calc_Q(self, k, P):
        """Calculates the interference covariance matrix at the :math:`k`-th
        receiver.

        The interference covariance matrix at the :math:`k`-th receiver,
        :math:`\mtQ k`, is given by

            :math:`\\mtQ k = \\sum_{j=1, j \\neq k}^{K} \\frac{P_j}{Ns_j} \\mtH_{kj} \\mtF_j \\mtF_j^H \\mtH_{kj}^H`

        where :math:`P_j` is the transmit power of transmitter :math:`j`,
        and :math:`Ns_j` is the number of streams for user :math:`j`.

        Parameters
        ----------
        k : int
            Index of the desired receiver.
        P : 1D numpy array
            Transmit power of all users. If not provided, a transmit power
            equal to 1.0 will be used for each user. Note that `P` has the
            power of all users to be consistent with other methods in this
            module that require the power of the users, but the power of
            the :math:`k`-th user in `P` will be ignored.

        Returns
        -------
        Qk : 2D numpy complex array.
            The interference covariance matrix at receiver :math:`k`.

        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{kj} \mtF_j \mtF_j^H \mtH_{kj}^H$$

        interfering_users = set(range(self.K)) - set([k])
        Qk = np.zeros([self.Nr[k], self.Nr[k]], dtype=complex)

        for l in interfering_users:
            Hkl_F = np.dot(
                self.get_channel(k, l),
                self.F[l])
            Qk = Qk + np.dot(P[l] * Hkl_F, Hkl_F.transpose().conjugate())

        return Qk

    def solve(self):
        """Find the IA solution.

        This method does not return anythin, but instead updates the 'F'
        and 'W' member variables.

        Notes
        -----
        This function should be implemented in the derived classes
        """
        raise NotImplementedError("solve: Not implemented")


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx AlternatingMinIASolver Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class AlternatingMinIASolver(IASolverBaseClass):
    """Implements the "Interference Alignment via Alternating Minimization"
    algorithm from the paper with the same name.

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

    """
    def __init__(self):
        IASolverBaseClass.__init__(self)

        self.C = []    # Basis of the interference subspace for each user
        self.max_iterations = 50

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def getCost(self):
        """Get the Cost of the algorithm for the current iteration of the
        precoder.

        Returns
        -------
        cost : float
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
                self.get_channel(k, l),
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

        The step method is usually all you need to call to perform an
        iteration of the Alternating Minimization algorithm. It will update
        C, then update F and finally update W.

        See also
        --------
        updateC, updateF, updateW

        """
        self.updateC()
        self.updateF()
        self.updateW()

    def updateC(self):
        """Update the value of Ck for all K users.

        Ck contains the orthogonal basis of the interference subspace of
        user k. It corresponds to the Nk-Sk dominant eigenvectors of

            :math:`\\sum_{l \\neq k} \mtH_{k,l} \mtF_l \mtF_l^H \mtH_{k,l}^H`.

        Notes
        -----
        This method is called in the :meth:`step` method.

        See also
        --------
        step
        """
        # $$\sum_{l \neq k} \mtH_{k,l} \mtF_l \mtF_l^H \mtH_{k,l}^H$$
        Ni = self.Nr - self.Ns  # Ni: Dimension of the interference subspace

        self.C = np.zeros(self.K, dtype=np.ndarray)
        # This will get all combinations of (k,l) without repetition. This
        # is equivalent to two nested for loops with an if statement to
        # only execute the code only when `k` is different of `l`.
        all_kl_indexes = itertools.permutations(range(self.K), 2)

        # This code will store in self.C[k] the equivalent of
        # $\sum_{l \neq k} \mtH_{k,l} \mtF_l \mtF_l^H \mtH_{k,l}^H$
        for k, l in all_kl_indexes:
            Hkl_F = np.dot(
                self.get_channel(k, l),
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
        calc_Y = lambda Nr, C: np.eye(Nr, dtype=complex) - \
            np.dot(C, C.conjugate().transpose())
        Y = map(calc_Y, self.Nr, self.C)

        newF = np.zeros(self.K, dtype=np.ndarray)
        # This will get all combinations of (l,k) without repetition. This
        # is equivalent to two nested for loops with an if statement to
        # only execute the code only when `l` is different of `k`
        all_lk_indexes = itertools.permutations(range(self.K), 2)

        # This code will store in newF[l] the equivalent of
        # $\sum_{k \neq l} \mtH_{k,l}^H (\mtI - \mtC_k \mtC_k^H)H_{k,l}$
        for lk in all_lk_indexes:
            (l, k) = lk
            lH = self.get_channel(k, l)
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

        Notes
        -----
        This method is called in the :meth:`step` method.

        See also
        --------
        step
        """
        newW = np.zeros(self.K, dtype=np.ndarray)
        for k in np.arange(self.K):
            tildeHi = np.hstack(
                [np.dot(self.get_channel(k, k), self.F[k]),
                 self.C[k]])
            newW[k] = np.linalg.inv(tildeHi)
            # We only want the first Ns[k] lines
            newW[k] = newW[k][0:self.Ns[k]]
        self.W = newW

    def solve(self):
        """Find the IA solution with the Alternating Minimizations algorithm.

        The number of iterations of the algorithm must be specified in the
        max_iterations member variable.

        Notes
        -----

        You need to call :meth:`randomizeF` at least once before calling
        :meth:`solve` as well as initialize the channel either calling the
        :meth:`init_from_channel_matrix` or the :meth:`randomizeH` methods.

        """
        for i in range(self.max_iterations):
            self.step()


# TODO: Finish the implementation
class MaxSinrIASolverIASolver(IASolverBaseClass):
    """Implements the "Interference Alignment via Max SINR" algorithm.

    This algorithm is applicable to a "K-user" scenario and it is
    described in [Cadambe2008]_.

    An example of a common exenario is a scenario with 3 pairs or
    transmitter/receiver with 2 antennas in each node and 1 stream
    transmitted per node.

    You can determine the scenario of an AlternatingMinIASolver object by
    infering the variables K, Nt, Nr and Ns.

    Notes
    -----

    .. [Cadambe2008] K. Gomadam, V. R. Cadambe, and S. A. Jafar,
       "Approaching the Capacity of Wireless Networks through Distributed
       Interference Alignment," in IEEE GLOBECOM 2008 - 2008 IEEE Global
       Telecommunications Conference, 2008, pp. 1-6.

    """

    def __init__(self, ):
        """
        """
        IASolverBaseClass.__init__(self)

    def _calc_Bkl_cov_matrix_first_part(self, k, P=None):
        """Calculates the first part in the equation of the Blk covariance
        matrix in equation (28) of [Cadambe2008]_.

        The first part is given by

            :math:`\\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star l}^{[j]} \\mtV_{\\star l}^{[j]\\dagger} \\mtH^{[kj]\\dagger}`

        Note that it only depends on the value of :math:`k`.

        Parameters
        ----------
        k : int
            Index of the desired user.
        P : 1D numpy array
            Transmit power of all users. If not provided, a transmit power
            equal to 1.0 will be used for each user.

        Returns
        -------
        Bkl_first_part : 2D numpy complex array
            First part in equation (28) of [Cadambe2008]_.
        """
        # $$\sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star l}^{[j]} \mtV_{\star l}^{[j]\dagger} \mtH^{[kj]\dagger}$$
        if P is None:
            P = np.ones(self.K)

        first_part = 0.0
        for j in range(self.K):
            Hkj = self.get_channel(k, j)
            Hkj_H = Hkj.conjugate().transpose()
            Vj = self.F[j]
            Vj_H = Vj.conjugate().transpose()

            first_part = first_part + (float(P[j]) / self._Ns[j]) * np.dot(
                Hkj,
                np.dot(
                    np.dot(Vj,
                           Vj_H),
                    Hkj_H))

        return first_part

    def _calc_Bkl_cov_matrix_second_part(self, k, l, P=None):
        """Calculates the second part in the equation of the Blk covariance
        matrix in equation (28) of [Cadambe2008]_.

        The second part is given by

            :math:`\\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger}`

        Parameters
        ----------
        k : int
            Index of the desired user.
        l : int
            Index of the desired stream.
        P : 1D numpy array
            Transmit power of all users. If not provided, a transmit power
            equal to 1.0 will be used for each user.

        Returns
        -------
        second_part : 2D numpy complex array.
            Second part in equation (28) of [Cadambe2008]_.

        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger}$$
        if P is None:
            P = np.ones(self.K)

        Hkk = self.get_channel(k, k)
        Hkk_H = Hkk.transpose().conjugate()

        Vkl = self.F[k][:, l:l + 1]
        Vkl_H = Vkl.transpose().conjugate()
        second_part = np.dot(Hkk,
                             np.dot(np.dot(Vkl, Vkl_H),
                                    Hkk_H))

        return second_part * (float(P[k]) / self._Ns[k])

    def calc_Bkl_cov_matrix_all_l(self, k, P=None):
        """Calculates the interference-plus-noise covariance matrix for all
        streams at receiver :math:`k` according to equation (28) in
        [Cadambe2008]_.

        The interference-plus-noise covariance matrix for stream :math:`l`
        of user :math:`k` is given by Equation (28) in [Cadambe2008]_,
        which is reproduced below

            :math:`\\mtB^{[kl]} = \\sum_{j=1}^{K} \\frac{P^{[j]}}{d^{[j]}} \\sum_{d=1}^{d^{[j]}} \\mtH^{[kj]}\\mtV_{\\star l}^{[j]} \\mtV_{\\star l}^{[j]\\dagger} \\mtH^{[kj]\\dagger} - \\frac{P^{[k]}}{d^{[k]}} \\mtH^{[kk]} \\mtV_{\\star l}^{[k]} \\mtV_{\\star l}^{[k]\\dagger} \\mtH^{[kk]\\dagger} + \\mtI_{N^{[k]}}`

        where :math:`P^{[k]}` is the transmit power of transmitter
        :math:`k`, :math:`d^{[k]}` is the number of degrees of freedom of
        user :math:`k`, :math:`\mtH^{[kj]}` is the channel between
        transmitter :math:`j` and receiver :math:`k`, :math:`\mtV_{\star
        l}` is the :math:`l`-th column of the precoder of user :math:`k`
        and :math:`\mtI_{N^{k}}` is an identity matrix with size equal to
        the number of receive antennas of receiver :math:`k`.

        Parameters
        ----------
        k : int
            Index of the desired user.
        P : 1D numpy array
            Transmit power of all users. If not provided, a transmit power
            equal to 1.0 will be used for each user.

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
        Bkl_all_l = np.empty(self._Ns[k], dtype=np.ndarray)
        first_part = self._calc_Bkl_cov_matrix_first_part(k, P)
        for l in range(self._Ns[k]):
            second_part = self._calc_Bkl_cov_matrix_second_part(k, l, P)
            Bkl_all_l[l] = first_part - second_part + np.eye(self.Nr[k])

        return Bkl_all_l

    def calc_Ukl(self, Bkl, k, l):
        """Calculates the Ukl matrix in equation (29) of [Cadambe2008]_.

        Parameters
        ----------
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
        Hkk = self.get_channel(k, k)
        Vkl = self.F[k][:, l:l + 1]
        invBkl = np.linalg.inv(Bkl)
        Ukl = np.dot(invBkl,
                     np.dot(Hkk, Vkl))
        Ukl = Ukl / np.linalg.norm(Ukl, 'fro')
        return Ukl

    def calc_Uk(self, Bkl_all_l, k):
        """Similar to the :meth:`calc_Ukl` method, but while :meth:`calc_Ukl`
        calculates the receive filter (a vector) only for the :math:`l`-th
        stream :meth:`calc_Uk` calculates a receive filter (a matrix) for
        all streams.

        Parameters
        ----------
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
            Uk[:, l] = self.calc_Ukl(Bkl_all_l[l], k, l)[:, 0]

        return Uk

    def calc_SINR_k(self, Bkl_all_l, Uk, k, P=None):
        """Calculates the SINR of all streams of user 'k'.

        Parameters
        ----------
        Bkl_all_l : A sequence of 2D numpy arrays.
            A sequence (1D numpy array, a list, etc) of 2D numpy arrays
            corresponding to the Bkl matrices for all 'l's.
        Uk: 2D numpy arrays.
            The receive filter for all streams of user k.
        k : int
            Index of the desired user.
        P : 1D numpy array.
            Transmit power of all users. If not provided, a transmit power
            equal to 1.0 will be used for each user.

        Returns
        -------
        SINR_k : 1D numpy array
            The SINR for the different streams of user k.

        """
        if P is None:
            P = np.ones(self.K)

        Hkk = self.get_channel(k, k)
        Vk = self.F[k]
        Pk = P[k]

        SINR_k = np.empty(self.Ns[k], dtype=float)

        for l in range(self.Ns[k]):
            Vkl = Vk[:, l:l + 1]
            Ukl = Uk[:, l:l + 1]
            Ukl_H = Ukl.transpose().conjugate()
            aux = np.dot(Ukl_H,
                         np.dot(Hkk, Vkl))
            numerator = np.dot(aux,
                               aux.transpose().conjugate()) * Pk / self.Ns[k]
            denominator = np.dot(Ukl_H,
                                 np.dot(Bkl_all_l[l], Ukl))
            SINR_kl = np.asscalar(numerator) / np.asscalar(denominator)
            SINR_k[l] = np.abs(SINR_kl)  # The imaginary part should be
                                         # negligible

        return SINR_k
