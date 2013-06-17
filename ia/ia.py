#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Module with implementation of Interference Alignment algorithms"""

__revision__ = "$Revision$"

import numpy as np
import itertools

from util.misc import peig, leig, randn_c
from comm.channels import MultiUserChannelMatrix

__all__ = ['AlternatingMinIASolver', 'MaxSinrIASolver',
           'MinLeakageIASolver', 'ClosedFormIASolver']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Base Class for all IA Algorithms xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class IASolverBaseClass(object):
    """Base class for all Interference Alignment Algorithms.

    Parameters
    ----------
    multiUserChannel : A MultiUserChannelMatrix object.
        The multiuser channel.
    """
    def __init__(self, multiUserChannel):
        """Initialize the variables that every IA solver will have.
        """
        # The F and W variables will be numpy arrays OF numpy arrays.
        self._F = np.array([])  # Precoder: One precoder for each user
        self._W = np.array([])  # Receive filter: One for each user

        # xxxxxxxxxx Private attributes xxxxxxxxxxxxxxx
        # Number of streams per user
        self._Ns = np.array([])
        # Channel of all users
        self._multiUserChannel = multiUserChannel

        self._P = None  # Power of each user (P is an 1D numpy array). If
                        # not set (_P is None), then a power of 1 will be
                        # used for each transmitter.

    # def clear(self, ):
    #     """
    #     Clear the IA Solver object.

    #     The object becames as if it was just created.

    #     Notes
    #     -----
    #     You should overwrite this method in subclasses that pass parameters
    #     to the __init__ method, since here we call __init__ without
    #     arguments which is probably not what you want.
    #     """
    #     # The F and W variables will be numpy arrays OF numpy arrays.
    #     self._F = np.array([])  # Precoder: One precoder for each user
    #     self._W = np.array([])  # Receive filter: One for each user

    @property
    def F(self):
        """Transmit precoder of all users."""
        return self._F

    # The W property should return a receive filter that can be directly
    # multiplied (no need to calculate the hermitian of W) by the received
    # signal to cancel interference and compensate the effect of the
    # channel.
    @property
    def W(self):
        """Receive filter of all users."""
        return self._W

    # In some of the derived classes the the W property must be the
    # hermitian of _W combined with the inverse of the effective channel
    # (so that the effect of the precoder, channel and receive filter is a
    # identity matrix). The code to do that is here so that the subclasses
    # can define the W property and use _get_W_property_alt in that
    # definition.
    def _get_W_property_alt(self):
        """
        Receive filter of all users.

        The returned filter is equivalent to first calculating the
        hermitian of self._W. Then calculate the inverse of the effective
        channel and then multiply them.
        """
        W = np.empty(self.K, dtype=np.ndarray)
        for k in range(self.K):
            # Equivalent channel with the effect of the precoder, channel
            # and receive filter
            Hieq = self._calc_equivalent_channel(k)

            try:
                Hieq_inv = np.linalg.inv(Hieq)
            except Exception:
                # Should not be here
                Hieq_inv = np.eye(Hieq.shape[0])

            W[k] = Hieq_inv.dot(self._W[k].transpose().conjugate())
            # W is the only receive filter required to cancel the
            # interference and compensate the effect of the channel and
            # transmit precoder.
        return W

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
            self._P = np.ones(self.K, dtype=float) * value
        else:
            if len(value) != self.K:
                raise ValueError("P must be set to a sequency of length K")
            else:
                self._P = np.array(value)

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

    def randomizeF(self, Ns, P=None):
        """Generates a random precoder for each user.

        Parameters
        ----------
        K : int
            Number of users.
        Nt : int or 1D numpy array
            Number of transmit antennas of each user.
        Ns : int or 1D numpy array
            Number of streams of each user.
        P : 1D numpy array
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        if isinstance(Ns, int):
            Ns = np.ones(self.K, dtype=int) * Ns

        self.P = P

        # Lambda function that returns a normalized version of the input
        # numpy array
        normalized = lambda A: A / np.linalg.norm(A, 'fro')

        self._F = np.zeros(self.K, dtype=np.ndarray)
        for k in range(self.K):
            self._F[k] = normalized(randn_c(self.Nt[k], Ns[k]))
        #self._F = [normalized(randn_c(Nt[k], Ns[k])) for k in np.arange(0, K)]
        self._Ns = Ns

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
        return self._multiUserChannel.get_channel(k, l)

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
        :math:`\mtQ k`, is given by

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

        This is impacted by the self._P attribute.

        """
        # $$\mtQ k = \sum_{j=1, j \neq k}^{K} \frac{P_j}{Ns_j} \mtH_{kj} \mtF_j \mtF_j^H \mtH_{kj}^H$$
        P = self._P
        if P is None:
            P = np.ones(self.K)

        interfering_users = set(range(self.K)) - set([k])
        Qk = np.zeros([self.Nr[k], self.Nr[k]], dtype=complex)

        for l in interfering_users:
            Hkl_F = np.dot(
                self._get_channel(k, l),
                self._F[l])
            Qk = Qk + np.dot(P[l] * Hkl_F, Hkl_F.transpose().conjugate())

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
        # TODO: The power in the reverse network is probably different and
        # therefore maybe we should not use self._P directly.
        P = self.P
        interfering_users = set(range(self.K)) - set([k])
        Qk = np.zeros([self.Nt[k], self.Nt[k]], dtype=complex)

        for l in interfering_users:
            Hkl_F_rev = np.dot(
                self._get_channel_rev(k, l),
                self._W[l])
            Qk = Qk + np.dot(P[l] * Hkl_F_rev, Hkl_F_rev.transpose().conjugate())

        return Qk

    def _calc_equivalent_channel(self, k):
        """
        Calculates the equivalent channel for user :math:`k` considering
        the effect of the precoder, the actual channel, and the receive
        filter.

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
        # Note that here Wk is the self._Wk attribute and not the W
        # property. Since _calc_equivalent_channel is used in the W get
        # property if we had used self.W here we would get an infinity
        # recursion.
        Wk = self._W[k].transpose().conjugate()
        Fk = self.F[k]
        Hkk = self._get_channel(k, k)
        Hk_eq = Wk.dot(Hkk.dot(Fk))
        return Hk_eq

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

        [V, D] = leig(Qk, self.Ns[k])
        pk = np.sum(np.abs(D)) / np.trace(np.abs(Qk))
        return pk

    def solve(self):
        """
        Find the IA solution.

        This method instead updates the 'F' and 'W' member variables.

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
    Ns : int or 1D numpy array
        Number of streams of all users.

    Notes
    -----

    .. [CadambeDoF2008] V. R. Cadambe and S. A. Jafar, "Interference
       Alignment and Degrees of Freedom of the K User Interference
       Channel," IEEE Transactions on Information Theory 54, pp. 3425–3441,
       Aug. 2008.
    """

    def __init__(self, multiUserChannel, Ns):
        """

        Paramters
        ---------
        multiUserChannel : A MultiUserChannelMatrix object.
            The multiuser channel.
        Ns : int or 1D numpy array
            Number of streams of all users.
        """
        IASolverBaseClass.__init__(self, multiUserChannel)

        if isinstance(Ns, int):
            Ns = np.ones(3, dtype=int) * Ns
        else:
            assert len(Ns) == 3
        self._Ns = Ns

    def _calc_E(self):
        """
        Calculates the "E" matrix, given by

        :math:`\\mtE = \\mtH_{31}^{-1}\\mtH_{32}\\mtH_{12}^{-1}\\mtH_{13}\\mtH_{23}^{-1}\\mtH_{21}`.
        """
        # $\mtE = \mtH_{31}^{-1}\mtH_{32}\mtH_{12}^{-1}\mtH_{13}\mtH_{23}^{-1}\mtH_{21}$
        inv = np.linalg.inv
        H31 = self._get_channel(2, 0)
        H32 = self._get_channel(2, 1)
        H12 = self._get_channel(0, 1)
        H13 = self._get_channel(0, 2)
        H23 = self._get_channel(1, 2)
        H21 = self._get_channel(1, 0)
        E = np.dot(inv(H31),
                   np.dot(H32,
                          np.dot(inv(H12),
                                 np.dot(H13, np.dot(inv(H23), H21)))))
        return E

    def updateF(self):
        """Find the precoders.
        """
        E = self._calc_E()
        Ns0 = self.Ns[0]

        # The number of users is always 3 for the ClosedFormIASolver class
        self._F = np.zeros(3, dtype=np.ndarray)

        # The first precoder is given by any subset of the eigenvectors of
        # the "E" matrix. We simple get the first Ns_0 eigenvectors of E.
        eigenvectors = np.linalg.eig(E)[1]
        V1 = eigenvectors[:, 0:Ns0]
        self._F[0] = V1

        # The second precoder is given by $\mtH_{32}^{-1}\mtH_{31}\mtV_1$
        invH32 = np.linalg.inv(self._get_channel(2, 1))
        H31 = self._get_channel(2, 0)
        self._F[1] = np.dot(invH32, np.dot(H31, V1))
        # The third precoder is given by $\mtH_{23}^{-1}\mtH_{21}\mtV_1$
        invH23 = np.linalg.inv(self._get_channel(1, 2))
        H21 = self._get_channel(1, 0)
        self._F[2] = np.dot(invH23, np.dot(H21, V1))

        # Normalize the precoders
        self._F[0] = self._F[0] / np.linalg.norm(self._F[0], 'fro')
        self._F[1] = self._F[1] / np.linalg.norm(self._F[1], 'fro')
        self._F[2] = self._F[2] / np.linalg.norm(self._F[2], 'fro')

    def updateW(self):
        """Find the receive filters
        """
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
        pass

    @property
    def W(self):
        """Receive filter of all users."""
        return self._get_W_property_alt()

    def solve(self):
        """
        Find the IA solution.
        """
        assert self.K == 3, 'The ClosedFormIASolver class only works in a MIMO-IC scenario with 3 users.'
        self.updateF()
        self.updateW()


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx Base Class for all iterative IA algorithms xxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class IterativeIASolverBaseClass(IASolverBaseClass):
    """Base class for all Iterative IA algorithms.

    All iterative IA algorithms must implement a `_step` method. The solve
    method will be then simply running the _step method a given number of
    times, which is controlled by the max_iterations attribute.

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

    def _step(self):
        """Performs one iteration of the algorithm.

        This method does not return anything, but instead updates the
        precoder and receive filter.

        Notes
        -----
        This function should be implemented in the derived classes

        """
        raise NotImplementedError("step: Not implemented")  # pragma: no cover

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Overwrite method in the IASolverBaseClass that change the precoder so
    # that they also reset the _runned_iterations attribute to zero.
    def randomizeF(self, Ns, P=None):
        self._runned_iterations = 0
        super(IterativeIASolverBaseClass, self).randomizeF(Ns, P)
    randomizeF.__doc__ = IASolverBaseClass.randomizeF.__doc__
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def solve(self):
        """
        Find the IA solution by performing the `step` method several times.

        The number of iterations of the algorithm must be specified in the
        max_iterations member variable.

        Returns
        -------
        Number of iterations the iterative interference alignment algorithm
        run.

        Notes
        -----

        You need to call :meth:`randomizeF` at least once before calling
        :meth:`solve` as well as initialize the channel either calling the
        :meth:`init_from_channel_matrix` or the :meth:`randomize` methods.
        """
        for i in range(self.max_iterations):
            self._runned_iterations = self._runned_iterations + 1
            self._step()

        # Return the number of iterations the algorithm run
        return i + 1


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx AlternatingMinIASolver Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class AlternatingMinIASolver(IterativeIASolverBaseClass):
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

        self.C = []    # Basis of the interference subspace for each user
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def getCost(self):
        """
        Get the Cost of the algorithm for the current iteration of the
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
                self._get_channel(k, l),
                self._F[l])
            Cost = Cost + np.linalg.norm(
                Hkl_Fl -
                np.dot(
                    np.dot(
                        self.C[k],
                        self.C[k].transpose().conjugate()),
                    Hkl_Fl
                ), 'fro') ** 2

        return Cost

    def _step(self):
        """Performs one iteration of the algorithm.

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

        # xxxxxxxxxx New Implementation using calc_Q xxxxxxxxxxxxxxxxxxxxxx
        Ni = self.Nr - self.Ns  # Ni: Dimension of the interference subspace

        self.C = np.empty(self.K, dtype=np.ndarray)

        for k in np.arange(self.K):
            ### TODO: Implement and test with external interference
            # We are inside only of the first for loop
            # Add the external interference contribution
            #self.C[k] = self.calc_Q(k) + self.Rk[k]

            # C[k] will receive the Ni most dominant eigenvectors of C[k]
            self.C[k] = peig(self.calc_Q(k), Ni[k])[0]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # # xxxxxxxxxx Old Implementation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Ni = self.Nr - self.Ns  # Ni: Dimension of the interference subspace

        # self.C = np.zeros(self.K, dtype=np.ndarray)
        # # This will get all combinations of (k,l) without repetition. This
        # # is equivalent to two nested for loops with an if statement to
        # # only execute the code only when `k` is different of `l`.
        # all_kl_indexes = itertools.permutations(range(self.K), 2)

        # # This code will store in self.C[k] the equivalent of
        # # $\sum_{l \neq k} \mtH_{k,l} \mtF_l \mtF_l^H \mtH_{k,l}^H$
        # for k, l in all_kl_indexes:
        #     Hkl_F = np.dot(
        #         self.get_channel(k, l),
        #         self.F[l])
        #     self.C[k] = self.C[k] + np.dot(Hkl_F, Hkl_F.transpose().conjugate())

        # # Every element in self.C[k] is a matrix. We want to replace each
        # # element by the dominant eigenvectors of that element.
        # for k in np.arange(self.K):
        #     # TODO: implement and test with external interference
        #     # # We are inside only of the first for loop
        #     # # Add the external interference contribution
        #     # self.C[k] = obj.C{k} + obj.Rk{k}

        #     # C[k] will receive the Ni most dominant eigenvectors of C[k]
        #     self.C[k] = peig(self.C[k], Ni[k])[0]

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
            lH = self._get_channel(k, l)
            newF[l] = newF[l] + np.dot(
                np.dot(lH.conjugate().transpose(),
                       Y[k]),
                lH)

        # Every element in newF is a matrix. We want to replace each
        # element by the least dominant eigenvectors of that element.
        self._F = map(lambda x, y: leig(x, y)[0], newF, self.Ns)

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
                [np.dot(self._get_channel(k, k), self._F[k]),
                 self.C[k]])
            newW[k] = np.linalg.inv(tildeHi)
            # We only want the first Ns[k] lines
            newW[k] = newW[k][0:self.Ns[k]]
        self._W = newW


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

    def getCost(self):
        """
        Get the Cost of the algorithm for the current iteration of the
        precoder.

        For the Minimum Leakage Interference Alignment algorithm the cost
        is equivalent to the sum of the interference that all users see
        after applying the receive filter. That is,

            :math:`C = Tr[\\mtU_k^H \\mtQ_k \\mtU_k]`

        Returns
        -------
        cost : float
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
            [V, D] = leig(Qk, self.Ns[k])
            Uk[k] = V
        return Uk

    def _calc_Uk_all_k_rev(self):
        """Calculates the receive filter of all users in the reverse
        network.

        """
        Uk_rev = np.empty(self.K, dtype=np.ndarray)
        for k in range(self.K):
            Qk_rev = self.calc_Q_rev(k)
            [V, D] = leig(Qk_rev, self.Ns[k])
            Uk_rev[k] = V
        return Uk_rev

    @property
    def W(self):
        """Receive filter of all users."""
        return self._get_W_property_alt()

    def _step(self):
        """Performs one iteration of the algorithm.

        The step method is usually all you need to call to perform an
        iteration of the algorithm. Note that it is necesary to initialize
        the precoders and the channel before calling the step method for
        the first time.

        See also
        --------
        randomizeF

        """
        self._F = self._calc_Uk_all_k_rev()
        self._W = self._calc_Uk_all_k()

    def solve(self):
        """Find the IA solution with the Min Leakage algorithm.

        The number of iterations of the algorithm must be specified in the
        max_iterations member variable.

        Returns
        -------
        Number of iterations the iterative interference alignment algorithm
        run.


        Notes
        -----

        You need to call :meth:`randomizeF` at least once before calling
        :meth:`solve` as well as initialize the channel either calling the
        :meth:`init_from_channel_matrix` or the :meth:`randomize` methods.

        """
        self._W = self._calc_Uk_all_k()
        return IterativeIASolverBaseClass.solve(self)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MaxSinrIASolver class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# TODO: Test and validate this implementation
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
    noise_var : float
        Noise power in dBm.

    Notes
    -----

    .. [Cadambe2008] K. Gomadam, V. R. Cadambe, and S. A. Jafar,
       "Approaching the Capacity of Wireless Networks through Distributed
       Interference Alignment," in IEEE GLOBECOM 2008 - 2008 IEEE Global
       Telecommunications Conference, 2008, pp. 1-6.

    """

    def __init__(self, multiUserChannel, noise_power=1):
        """
        Parameters
        ----------
        multiUserChannel : A MultiUserChannelMatrix object.
            The multiuser channel.
        noise_var : float
            Noise power in dBm.
        """
        IterativeIASolverBaseClass.__init__(self, multiUserChannel)
        self.noise_power = noise_power

    @property
    def W(self):
        """Receive filter of all users."""
        return self._get_W_property_alt()

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
        P = self._P
        if P is None:
            P = np.ones(self.K)  # pragma: no cover

        first_part = 0.0
        for j in range(self.K):
            Hkj = self._get_channel(k, j)
            Hkj_H = Hkj.conjugate().transpose()
            Vj = self._F[j]
            Vj_H = Vj.conjugate().transpose()

            first_part = first_part + (float(P[j]) / self._Ns[j]) * np.dot(
                Hkj,
                np.dot(
                    np.dot(Vj,
                           Vj_H),
                    Hkj_H))

        return first_part

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
        P = self._P
        if P is None:
            P = np.ones(self.K)  # pragma: no cover

        first_part = 0.0
        for j in range(self.K):
            Hkj = self._get_channel_rev(k, j)
            Hkj_H = Hkj.conjugate().transpose()
            Vj = self._W[j]
            Vj_H = Vj.conjugate().transpose()

            first_part = first_part + (float(P[j]) / self._Ns[j]) * np.dot(
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
        P = self._P
        if P is None:
            P = np.ones(self.K)  # pragma: no cover

        Hkk = self._get_channel(k, k)
        Hkk_H = Hkk.transpose().conjugate()

        Vkl = self._F[k][:, l:l + 1]
        Vkl_H = Vkl.transpose().conjugate()
        second_part = np.dot(Hkk,
                             np.dot(np.dot(Vkl, Vkl_H),
                                    Hkk_H))

        return second_part * (float(P[k]) / self._Ns[k])

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
        P = self._P
        if P is None:
            P = np.ones(self.K)  # pragma: no cover

        Hkk = self._get_channel_rev(k, k)
        Hkk_H = Hkk.transpose().conjugate()

        Vkl = self._W[k][:, l:l + 1]
        Vkl_H = Vkl.transpose().conjugate()
        second_part = np.dot(Hkk,
                             np.dot(np.dot(Vkl, Vkl_H),
                                    Hkk_H))

        return second_part * (float(P[k]) / self._Ns[k])

    def _calc_Bkl_cov_matrix_all_l(self, k):
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
        first_part = self._calc_Bkl_cov_matrix_first_part(k)
        for l in range(self._Ns[k]):
            second_part = self._calc_Bkl_cov_matrix_second_part(k, l)
            Bkl_all_l[l] = first_part - second_part + (self.noise_power * np.eye(self.Nr[k]))

        return Bkl_all_l

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
            Bkl_all_l_rev[l] = first_part - second_part + (self.noise_power * np.eye(self.Nt[k]))

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
        invBkl = np.linalg.inv(Bkl)
        Ukl = np.dot(invBkl,
                     np.dot(Hkk, Vkl))
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
            Uk[:, l] = MaxSinrIASolver._calc_Ukl(Hkk, Vk, Bkl_all_l[l], k, l)[:, 0]

        return Uk

    def _calc_Uk_all_k(self):
        """Calculates the receive filter of all users.
        """
        Uk = np.empty(self.K, dtype=np.ndarray)
        for k in range(self.K):
            Hkk = self._get_channel(k, k)
            Bkl_all_l = self._calc_Bkl_cov_matrix_all_l(k)
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

    # NOTE: This method is specific to the MaxSinrIASolver algorithm and that is
    # why it is an internal method (starting with a "_")
    def _calc_SINR_k(self, Bkl_all_l, Uk, k):
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

        Returns
        -------
        SINR_k : 1D numpy array
            The SINR for the different streams of user k.

        """
        Pk = self.P[k]
        Hkk = self._get_channel(k, k)
        Vk = self._F[k]

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

    def _step(self):  # pragma: no cover
        """Performs one iteration of the algorithm.

        The step method is usually all you need to call to perform an
        iteration of the algorithm. Note that it is necesary to initialize
        the precoders and the channel before calling the step method for
        the first time.

        See also
        --------
        randomizeF

        """
        self._F = self._calc_Uk_all_k_rev()
        self._W = self._calc_Uk_all_k()

    def solve(self):  # pragma: no cover
        """Find the IA solution with the Max SINR algorithm.

        The number of iterations of the algorithm must be specified in the
        max_iterations member variable.

        Returns
        -------
        Number of iterations the iterative interference alignment algorithm
        run.


        Notes
        -----

        You need to call :meth:`randomizeF` at least once before calling
        :meth:`solve` as well as initialize the channel either calling the
        :meth:`init_from_channel_matrix` or the :meth:`randomize` methods.

        """
        self._W = self._calc_Uk_all_k()
        return IterativeIASolverBaseClass.solve(self)
