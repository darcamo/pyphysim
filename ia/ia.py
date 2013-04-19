#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Module with implementation of Interference Alignment algorithms"""

__revision__ = "$Revision$"

import numpy as np
import itertools

from util.misc import peig, leig, randn_c
from comm.channels import MultiUserChannelMatrix

__all__ = ['IASolverBaseClass', 'AlternatingMinIASolver']


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
        :math:`\\sum_{l \\neq k} H_{k,l} F_l F_l^H H_{k,l}^H`.

        Notes
        -----
        This method is called in the :meth:`step` method.

        See also
        --------
        step
        """
        # $\sum_{l \neq k} H_{k,l} F_l F_l^H H_{k,l}^H$
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
        # $\sum_{k \neq l} H_{k,l}^H (I - C_k C_k^H)H_{k,l}$
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
    described in [1].

    An example of a common exenario is a scenario with 3 pairs or
    transmitter/receiver with 2 antennas in each node and 1 stream
    transmitted per node.

    You can determine the scenario of an AlternatingMinIASolver object by
    infering the variables K, Nt, Nr and Ns.

    References
    ----------

    [1] K. Gomadam, V. R. Cadambe, and S. A. Jafar, "Approaching the
    Capacity of Wireless Networks through Distributed Interference
    Alignment," in IEEE GLOBECOM 2008 - 2008 IEEE Global Telecommunications
    Conference, 2008, pp. 1-6.

    """

    def __init__(self, ):
        """
        """
        IASolverBaseClass.__init__(self)

    def calc_Bkl_cov_matrix(self, k, l):
        """Calculates the interference-plus-noise covariance matrix for stream
        $l$ at receiver $k$ according to eqaution (28) in [1].

        Equation (28) in [1] is reproduced below

          $$B^{[kl]} = \sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}}
        \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star l}^{[j]} \mtV_{\star l}^{[j]\dagger} \mtH^{[kj]\dagger} -
        \frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]}
        \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger} +
        \mtI_{N^{[k]}}$$

        where $P^{[k]}$ is the transmit power of transmitter $k$, $d^{[k]}$
        is the number of degrees of freedom of user $k$, $\mtH^{[kj]}$ is
        the channel between transmitter $j$ and receiver $k$,
        $\mtV_{\star l}$ is the $l$-th column of the precoder of user $k$
        and $\mtI_{N^{k}}$ is an identity matrix with size equal to the
        number of receive antennas of receiver $k$.


        Returns
        -------
        Bkl : 1D numpy array of numpy arrays
            Covariance matrix of all users.

        References
        ----------

        [1] K. Gomadam, V. R. Cadambe, and S. A. Jafar, "Approaching the
        Capacity of Wireless Networks through Distributed Interference
        Alignment," in IEEE GLOBECOM 2008 - 2008 IEEE Global
        Telecommunications Conference, 2008, pp. 1-6.

        """

        pass
