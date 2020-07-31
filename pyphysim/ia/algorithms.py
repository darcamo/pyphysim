#!/usr/bin/env python

# pylint: disable=R0904
"""
Module with implementation of Interference Alignment (IA) algorithms.

Note that all IA algorithms require the channel object and any change to
the channel object must be performed before calling the `solve` method of
the IA algorithm object. This includes generating the channel and setting
the noise variance.
"""

import itertools
from abc import ABCMeta, abstractmethod
from copy import copy
from itertools import product
from typing import (Any, List, Optional, Sequence, Tuple, Type, TypeVar, Union,
                    cast)

import numpy as np
from scipy import optimize

from ..channels import multiuser as muchannels
from ..util.misc import (get_principal_component_matrix,
                         least_right_singular_vectors, leig, peig,
                         update_inv_sum_diag)
from .iabase import IASolverBaseClass

__all__ = [
    'AlternatingMinIASolver', 'MaxSinrIASolver', 'MinLeakageIASolver',
    'ClosedFormIASolver', 'MMSEIASolver', 'GreedStreamIASolver',
    'BruteForceStreamIASolver', 'IterativeIASolverBaseClass'
]

FloatOrFloatSequence = TypeVar("FloatOrFloatSequence", float, Sequence[float])
IntOrIntSequence = TypeVar("IntOrIntSequence", int, Sequence[int])


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
    multiUserChannel : muchannels.MultiUserChannelMatrix
        The multiuser channel.
    use_best_init : bool
        If true, all possible initializations (subsets of the
        eigenvectors of the 'E' matrix)for the first precoder will be
        tested and the best solution will be used. If false, the first
        initialization will be used.

    Notes
    -----

    .. [CadambeDoF2008] V. R. Cadambe and S. A. Jafar, "Interference
       Alignment and Degrees of Freedom of the K User Interference
       Channel," IEEE Transactions on Information Theory 54, pp. 3425-3441,
       Aug. 2008.
    """
    def __init__(self,
                 multiUserChannel: muchannels.MultiUserChannelMatrix,
                 use_best_init: bool = True):
        super().__init__(multiUserChannel)
        self._use_best_init = use_best_init

    # noinspection PyPep8
    def _calc_E(self) -> np.ndarray:
        """
        Calculates the "E" matrix, given by

        :math:`\\mtE = \\mtH_{31}^{-1}\\mtH_{32}\\mtH_{12}^{-1}\\mtH_{13}\\mtH_{23}^{-1}\\mtH_{21}`.

        Returns
        -------
        np.ndarray
            The "E" matrix.
        """
        # $\mtE = \mtH_{31}^{-1}\mtH_{32}\mtH_{12}^{-1}\mtH_{13}\mtH_{23}^{-1}\mtH_{21}$

        H31 = self._get_channel(2, 0)
        H32 = self._get_channel(2, 1)
        H12 = self._get_channel(0, 1)
        H13 = self._get_channel(0, 2)
        H23 = self._get_channel(1, 2)
        H21 = self._get_channel(1, 0)

        E = (np.linalg.solve(H31, H32)).dot(
            (np.linalg.solve(H12, H13)).dot(np.linalg.solve(H23, H21)))

        return E

    def _calc_all_F_initializations(self, Ns: int) -> List[np.ndarray]:
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
        all_initializations : list[np.ndarray]
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

    def _updateF(self, F0: Optional[np.ndarray] = None) -> None:
        """
        Find the precoders.

        Parameters
        ----------
        F0 : np.ndarray
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

            # The first precoder is given by any subset of the
            # eigenvectors of the "E" matrix. We simple get the first
            # Ns_0 eigenvectors of E.
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
        self._F[0] /= np.linalg.norm(self._F[0], 'fro')
        self._F[1] /= np.linalg.norm(self._F[1], 'fro')
        self._F[2] /= np.linalg.norm(self._F[2], 'fro')

    # noinspection PyUnresolvedReferences
    def _updateW(self) -> None:
        """Find the receive filters
        """
        self._clear_receive_filter()

        # The number of users is always 3 for the ClosedFormIASolver class
        self._W = np.zeros(3, dtype=np.ndarray)

        A0 = np.dot(self._get_channel(0, 1), self.F[1])
        self._W[0] = leig(np.dot(A0,
                                 A0.transpose().conjugate()), self.Ns[0])[0]

        A1 = np.dot(self._get_channel(1, 0), self.F[0])
        self._W[1] = leig(np.dot(A1,
                                 A1.transpose().conjugate()), self.Ns[1])[0]

        A2 = np.dot(self._get_channel(2, 0), self.F[0])
        self._W[2] = leig(np.dot(A2,
                                 A2.transpose().conjugate()), self.Ns[2])[0]

    # noinspection PyUnboundLocalVariable
    def solve(self,
              Ns: IntOrIntSequence,
              P: Optional[FloatOrFloatSequence] = None) -> None:
        """
        Find the IA solution.

        This method updates the 'F' and 'W' member variables.

        Parameters
        ----------
        Ns : int | np.ndarray
            Number of streams of each user.
        P : np.ndarray | List[float] | float, optional
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        assert self.K == 3, ('The ClosedFormIASolver class only works'
                             ' in a MIMO-IC scenario with 3 users.')

        if isinstance(Ns, int):
            Ns = np.ones(3, dtype=int) * Ns
        else:
            # noinspection PyTypeChecker
            assert len(Ns) == 3

        # Help the type system knowing that at this point Ns is a Sequence[int]
        assert (not isinstance(Ns, int))

        # This will create a new array so that we can modify self._Ns
        # internally without changing the original Ns variable passed to
        # the solve method.
        self._Ns = np.array(Ns)

        self.P = P

        if self._use_best_init is True:
            # xxxxx Case when the best solution should be used xxxxxxxxxxxx
            best_sum_capacity = 0
            all_initializations = self._calc_all_F_initializations(Ns[0])

            # Lambda function to calculate the sum capacity from the SINR
            # values (in linear scale)
            def calc_capacity(sinr: np.ndarray) -> float:
                return cast(float, np.sum(np.log2(1 + sinr)))

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
    multiUserChannel : muchannels.MultiUserChannelMatrix
        The multiuser channel.
    """

    # The IterativeIASolverBaseClass is an abstract class and the
    # '_updateF' and '_updateW' methods (marked as abstract) must be
    # implemented in a subclass.
    __metaclass__ = ABCMeta

    def __init__(self, multiUserChannel: muchannels.MultiUserChannelMatrix):
        super().__init__(multiUserChannel)

        # Count how many times the step method was called. This will be
        # reseted to zero whenever the channel or the precoder is
        # reinitialized.
        self._runned_iterations = 0
        # Number of times the step method is called in the solve method.
        self.max_iterations = 50

        # Relative change of the precoder in one iteration to the next
        # one. If the relative change from one iteration to the next one is
        # lower than this factor then the algorithm will stop the
        # iterations before the max_iterations limit is reached.
        self.relative_factor = 1e-6

        # We can use the closed form IA solver or the Alternating
        # Minimization IA solver to initialize the iterative
        # algorithm. This can reduce the number of iterations required for
        # convergence or even get better performance.
        #
        # Note that this is controlled by the 'initialize_with'
        # variable. If it is 'random' the initialization is performed with
        # random precoders. If it is 'closed_form' the initialization is
        # performed with the Closed Form algorithm, while if it is
        # 'alt_min' the initialization is performed with the Alternating
        # Minimizations algorithm.
        self._closed_form_ia_solver = ClosedFormIASolver(multiUserChannel,
                                                         use_best_init=True)

        # If self is of the class AlternatingMinIASolver, then the
        # initialization with the Alternating Minimizations algorithm makes
        # no sense.
        self._alt_min_ia_solver: Optional[AlternatingMinIASolver]
        if isinstance(self, AlternatingMinIASolver):
            self._alt_min_ia_solver = None
        else:
            self._alt_min_ia_solver = AlternatingMinIASolver(multiUserChannel)

        # Can be: 'random', 'closed_form', 'alt_min', or 'fix'
        #
        # 'random' -> Precoder will be initialized randomly and then the
        #             receive filter will be updated (using the _updateW
        #             method which is implemented in the subclass..
        #
        # 'closed_form' : -> The precoder and receive filters will be
        #                    initialized with the solution of the closed
        #                    form algorithm.
        #
        # 'alt_min' : -> The precoder and receive filters will be
        #                initialized with the solution of the alternating
        #                minimizations algorithm.
        #
        # 'fix' : The precoder is not initialized and the current value of
        #         self.F is used. This initialization type should only be
        #         used to call the 'solve' method a second time after it
        #         has been called before with another of the initialization
        #         type. This allows performing more iterations continuing
        #         from the previous solution . This is specially useful for
        #         debugging.
        self._initialize_with = 'random'

    # xxxxxxxxxx initialize_with property xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    @property
    def initialize_with(self) -> str:
        """Get method for the initialize_with property."""
        return self._initialize_with

    @initialize_with.setter
    def initialize_with(self, value: str) -> None:
        """
        Set method for the initialize_with property.

        Parameters
        ----------
        value : str
            A string with the initialization method.
        """
        options = ['random', 'alt_min', 'closed_form', 'fix', 'svd']
        if value in options:
            self._initialize_with = value
        else:
            msg = "unknown initialization option: '{0}'".format(value)
            raise RuntimeError(msg)

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    @property
    def runned_iterations(self) -> int:
        """Get method for the runned_iterations property."""
        return self._runned_iterations

    def clear(self) -> None:
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

    @abstractmethod
    def _updateF(self) -> None:  # pragma: no cover
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

    @abstractmethod
    def _updateW(self) -> None:  # pragma: no cover
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

    def _step(self) -> None:
        """Performs one iteration of the algorithm.

        This method does not return anything, but instead updates the
        precoder and receive filter.
        """
        self._updateF()
        self._updateW()

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Overwrite method in the IASolverBaseClass that change the precoder so
    # that they also reset the _runned_iterations attribute to zero.
    def randomizeF(self,
                   Ns: Union[int, List[int], Sequence[int]],
                   P: Optional[np.ndarray] = None) -> None:
        self._runned_iterations = 0
        # randomizeF in the base class will set `self._P` to `P`
        super().randomizeF(Ns, P)

    randomizeF.__doc__ = IASolverBaseClass.randomizeF.__doc__

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _before_initialize_W_func(self) -> None:
        """
        Method run in any of the initialize methods after the precoder is
        initialized but before the receive filter is initialized.
        """
        pass

    def _initialize_F_randomly_and_find_W(self, Ns: Sequence[int],
                                          P: np.ndarray) -> None:
        """
        Initialize the IA Solution from a random matrix.

        The implementation here simple initializes the precoder variable
        and then calculates the initial receive filter.

        Parameters
        ----------
        Ns : int np.ndarray
            Number of streams of each user.
        P : np.ndarray
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        self.randomizeF(Ns, P)

        # Method called before the _updateW method
        self._before_initialize_W_func()

        # Calculate the receive filters
        self._updateW()

    # noinspection PyUnusedLocal
    def _initialize_F_with_svd_and_find_W(self, Ns: IntOrIntSequence,
                                          P: np.ndarray) -> None:
        """
        Initialize the IA Solution from the most significant singular
        vectors of each user's channel.

        The implementation here simple initializes the precoder variable of
        each user as the most significant singular vector(s) of that
        user. After that, calculate the initial receive filters.

        Parameters
        ----------
        Ns : int | np.ndarray
            Number of streams of each user.
        P : np.ndarray
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        if isinstance(Ns, int):
            Ns = np.ones(self.K, dtype=int) * Ns

        # Help the type system knowing that at this point Ns is a Sequence[int]
        assert (not isinstance(Ns, int))

        # Create the precoder variable
        self._F = np.empty(self.K, dtype=np.ndarray)

        # Set the precoder of each user with the most significant singular
        # vector(s) of that user's channel
        for k in range(self.K):
            Hkk = self._get_channel(k, k)
            # The second variable returned by least_right_singular_vectors
            # has the corresponds to the most significant singular
            # vectors.
            _, V1, _ = least_right_singular_vectors(Hkk, self.Nr[k] - Ns[k])
            self._F[k] = V1 / np.linalg.norm(V1, 'fro')

        # Method called before the _updateW method
        self._before_initialize_W_func()

        # Calculate the receive filters
        self._updateW()

    def _dont_initialize_F_and_only_and_find_W(self, *_: Any) -> None:
        """
        Initialize the IA Solution from a random matrix.

        The implementation here simple initializes the precoder variable
        and then calculates the initial receive filter.

        Note: The `dummy1` and `dummy2` arguments have no effect. They only
        exist to keep the signature of this method equal to the signature
        of other *initialize* methods.
        """
        # The current value of self._F and self._full_F will be used.
        if self._F is None:
            msg = ("The precoder must be manually set, since you specified"
                   " the 'fix' initialize_with option.")
            raise RuntimeError(msg)

        # For the 'fix' initialization type we get the number of streams
        # from the initialized precoders.
        self._Ns = np.array([F.shape[1] for F in self._F])

        # Method called before the _updateW method
        self._before_initialize_W_func()

        self._updateW()

    def _initialize_F_and_W_from_closed_form(self, Ns: IntOrIntSequence,
                                             P: np.ndarray) -> None:
        """
        Initialize the IA Solution from the closed form IA solver.

        Parameters
        ----------
        Ns : int | np.ndarray
            Number of streams of each user.
        P : np.ndarray
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        # Clear all precoders and receive filters
        self._clear_precoder_filter()
        self._clear_receive_filter()

        self.P = P

        self._closed_form_ia_solver.solve(Ns, P)
        self._F = self._closed_form_ia_solver.F

        # Method called before the _updateW method
        self._before_initialize_W_func()

        self._W = self._closed_form_ia_solver.W

    def _initialize_F_and_W_from_alt_min(self, Ns: IntOrIntSequence,
                                         P: np.ndarray) -> None:
        """
        Initialize the IA Solution from the Alternating Minimizations IA
        solver.

        Parameters
        ----------
        Ns : int | np.ndarray
            Number of streams of each user.
        P : np.ndarray
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        # Clear all precoders and receive filters
        self._clear_precoder_filter()
        self._clear_receive_filter()

        self.P = P

        assert (self._alt_min_ia_solver is not None)
        self._alt_min_ia_solver.max_iterations = self.max_iterations
        self._alt_min_ia_solver.solve(Ns, P)

        self._F = self._alt_min_ia_solver.F

        # Method called before the _updateW method
        self._before_initialize_W_func()

        self._W = np.empty(self.K, dtype=np.ndarray)
        for k in range(self.K):
            Wk = self._alt_min_ia_solver.W[k]
            self._W[k] = Wk / np.linalg.norm(Wk, 'fro')

    def _solve_init(self, Ns: IntOrIntSequence,
                    P: FloatOrFloatSequence) -> None:
        """
        Code run in the `solve` method before the loop that run the :meth:`_step`
        method.

        The implementation here simple initializes the precoder variable
        and then calculates the initial receive filter.

        Parameters
        ----------
        Ns : int | np.ndarray
            Number of streams of each user.
        P : float | np.ndarray
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        self.P = P

        # initialize_with can be: 'random', 'fix', 'closed_form',
        # 'alt_min', or 'svd'
        options = {
            'random': self._initialize_F_randomly_and_find_W,
            'alt_min': self._initialize_F_and_W_from_alt_min,
            'closed_form': self._initialize_F_and_W_from_closed_form,
            'fix': self._dont_initialize_F_and_only_and_find_W,
            'svd': self._initialize_F_with_svd_and_find_W
        }

        initialzie_func = options[self.initialize_with]
        initialzie_func(Ns, P)  # type: ignore

    def _solve_finalize(self) -> None:  # pragma: no cover
        """
        Perform any post processing after the solution has been found.

        Some of the found precoders may be a singular matrix. In that case,
        we need to remove the dimensions with zero energy from both the
        found precoder and the receive filter.
        """
        # Store the index of the users from which we need to modify the
        # precoders and receive filters
        mod_users = []
        num_significant_sing_values = []
        assert (self._F is not None)
        assert (self._full_F is not None)
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
                    max_sing_value = S.max().item()
                    # Calculate the number of significative singular
                    # values. Basically, any singular value (and
                    # corresponding dimension) lower then
                    # max_sing_value/1e4 will be discarded.
                    n = np.count_nonzero(np.greater(S, max_sing_value / 1.0e4))

                    # Store the number of significative singular values for
                    # that user
                    num_significant_sing_values.append(n)

                    new_F = get_principal_component_matrix(self._F[k], n)

                    # Normalize new_F
                    new_F /= np.linalg.norm(new_F, 'fro')

                    self._F[k] = new_F

                    if self._full_F[k] is not None:
                        # Original norm of the _full_F[k] precoder
                        original_norm = np.linalg.norm(self._full_F[k], 'fro')
                        new_full_F = get_principal_component_matrix(
                            self._full_F[k], n)
                        # Restore the original norm
                        new_full_F = new_full_F / np.linalg.norm(
                            new_full_F, 'fro') * original_norm
                        self._full_F[k] = new_full_F

                    self.Ns[k] = n

        # If we modified any of the precoders then the mod_users list has
        # the index of the users whose precoders were modified. We need to
        # also modify the receive filter for those users.
        if len(mod_users) > 0:
            # Note that we still need to remove the dead dimensions of
            # the receive filter. However, depending on the algorithm,
            # either the _W or the _W_H member variable was set while
            # the other is None (at this point).
            if self._W_H is None:
                assert (self._W is not None)
                # Since _W_H is None that means that we need to modify
                # the _W member variable
                for k, n in zip(mod_users, num_significant_sing_values):
                    new_W = get_principal_component_matrix(self._W[k], n)
                    self._W[k] = new_W

            elif self._W is None:
                # Since _W is None that means that we need to modify the
                #  _W_H member variable
                for k, n in zip(mod_users, num_significant_sing_values):
                    W = self._W_H[k].conj().T
                    new_W = get_principal_component_matrix(W, n)
                    self._W_H[k] = new_W.conj().T
            else:
                # If both self._W and self._W_H are not None then
                # something wrong happened. Maybe you called the self.W
                # or the self.W_H properties by mistake before
                # _solve_finalize is called (in the solve method).
                raise Exception("I should not be here.")

    @classmethod
    def _is_diff_significant(cls, F_old: np.ndarray, F_new: np.ndarray,
                             relative_factor: float) -> bool:
        """
        Test if there was any significant change from `F_old` to `F_new`.

        This method is used internally in the solve method of the
        IterativeIASolverBaseClass to detect when the precoder of a given
        iteration didn't change significantly from one iteration to
        another. This is used to stop the iterations of the algorithm and
        avoid unnecessary computations.

        Parameters
        ----------
        F_old : np.ndarray
            The precoder of all users (in a previous iteration). This is a 1D
            numpy array of numpy arrays.
        F_new : np.ndarray
            The precoder of all users (in the current iteration). This is a 1D
            numpy array of numpy arrays.
        relative_factor : float
            Relative change of the precoder in one iteration to the next one.
            If the relative change from one iteration to the next one is lower
            than this factor then the algorithm will stop the iterations before
            the max_iterations limit is reached.

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

    def solve(self,
              Ns: IntOrIntSequence,
              P: Optional[FloatOrFloatSequence] = None) -> int:
        """
        Find the IA solution by performing the :meth:`_step` method several times.

        The number of times the :meth:`_step` method is run is controlled by the
        max_iterations member variable.

        Before calling the :meth:`_step` method for the first time the
        `_solve_init` method is called to perform any required
        initializations. Since iterative IA algorithms usually starts with
        a random precoder then the `_solve_init` implementation in
        IterativeIASolverBaseClass calls randomizeF.

        Parameters
        ----------
        Ns : int | np.ndarray
            Number of streams of each user.
        P : np.ndarray | List[float] | float, optional
            Power of each user. If not provided, a value of 1 will be used
            for each user.

        Returns
        -------
        Number of iterations that the iterative interference alignment
        algorithm run.

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
        if isinstance(Ns, int):
            Ns = np.ones(self.K, dtype=int) * Ns
        else:
            # noinspection PyTypeChecker
            assert len(Ns) == self.K

        # This will create a new array so that we can modify self._Ns
        # internally without changing the original Ns variable passed to
        # the randomizeF method.
        self._Ns = np.array(Ns, dtype=int)
        self._solve_init(Ns, P)  # type: ignore

        # This will be used to detect of the precoder did not
        # significative change
        # TODO: Should I use full_F instead of F???
        old_F = self._F
        for _ in range(self.max_iterations):
            self._runned_iterations += 1
            self._step()

            # Stop the iteration earlier if the precoder does not change
            # too much
            if self._is_diff_significant(old_F, self._F,
                                         self.relative_factor) is False:
                break  # pragma: no cover

            old_F = self._F

        # Perform any post processing after the precoder and receive
        # filters where found. One possible usage for this method is to
        # remove dimensions of the precoder (and receive filter) that have
        # actually zero energy. That is, if the precoder ends up being a
        # singular matrix we can implement _solve_finalize to remove the
        # dimensions that do not contribute.
        self._solve_finalize()

        # Return the number of iterations the algorithm run
        return self._runned_iterations


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

    An example of a common scenario is a scenario with 3 pairs or
    transmitter/receiver with 2 antennas in each node and 1 stream
    transmitted per node.

    You can determine the scenario of an AlternatingMinIASolver object by
    inferring the variables K, Nt, Nr and Ns.

    Parameters
    ----------
    multiUserChannel : muchannels.MultiUserChannelMatrix
        The multiuser channel.

    Notes
    -----

    .. [PetersHeathAltMin2009] Peters, S.W.; Heath, R.W., "Interference
       alignment via alternating minimization," Acoustics, Speech and
       Signal Processing, 2009. ICASSP 2009. IEEE International Conference
       on, pp.2445,2448, 19-24 April 2009
    """
    def __init__(self, multiUserChannel: muchannels.MultiUserChannelMatrix):
        super().__init__(multiUserChannel)

        self._C: List[np.ndarray] = [
        ]  # Basis of the interference subspace for each user

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # Overwrite the set property for 'initialize_with' to remove the
    # 'alt_min' initialization type, since it does not make sense.
    # noinspection PyMethodOverriding,PyIncorrectDocstring
    @IterativeIASolverBaseClass.initialize_with.setter  # type: ignore
    def initialize_with(self, value: str) -> None:
        """Set method for the initialize_with property."""
        if value == 'alt_min':  # pragma: nocover
            msg = "Can't use '{0}' initialization with '{1}' class '{0}'"
            raise RuntimeError(msg.format(value, self.__class__.__name__))

        IterativeIASolverBaseClass.initialize_with.fset(  # type: ignore
            self, value)

    def get_cost(self) -> float:
        """
        Get the Cost of the algorithm for the current iteration of the
        precoder.

        Returns
        -------
        cost : float
            The Cost of the algorithm for the current iteration of the
            precoder. This is a real non-negative number.
        """
        Cost = 0
        # This will get all combinations of (k,l) without repetition. This
        # is equivalent to two nested for loops with an if statement to
        # only execute the code only when `k` is different of `l`.
        all_kl_indexes = itertools.permutations(range(self.K), 2)

        for kl in all_kl_indexes:
            (k, l) = kl
            Hkl_Fl = np.dot(self._get_channel(k, l), self.full_F[l])
            Cost += np.linalg.norm(
                Hkl_Fl -
                np.dot(np.dot(self._C[k], self._C[k].transpose().conjugate()),
                       Hkl_Fl), 'fro')**2

        return Cost

    def _before_initialize_W_func(self) -> None:
        """
        Method run in any of the initialize methods after the precoder is
        initialized but before the receive filter is initialized.
        """
        self._updateC()

    def _step(self) -> None:
        """
        Performs one iteration of the algorithm.

        The step method is usually all you need to call to perform an
        iteration of the Alternating Minimization algorithm. It will update
        C, then update F and finally update W.

        See also
        --------
        _updateC, _updateF, _updateW
        """
        # Note that before the `_step` method is called the first time, the
        # _solve_init method must be called. It will initialize the
        # precoders, the C matrices and the receive filters.

        self._updateF()  # Depend on the value of C
        self._updateC()  # The value of C depend on the precoders F

        # Note that in the Alternating Minimizations algorithm we do not
        # need to update the receive filters at each iteration, since the
        # precoders only depend on the values of the 'C' matrices. Because
        # of that, we will only update the receive filters in the
        # _solve_finalize method, which is called after the algorithm
        # converged.

    def _solve_finalize(self) -> None:  # pragma: no cover
        """Perform any post processing after the solution has been found.
        """
        # Note that in the Alternating Minimizations algorithm we do not
        # need to update the receive filters at each iteration, since the
        # precoders only depend on the values of the 'C' matrices. Because
        # of that, we will only update the receive filters in the
        # _solve_finalize method, which is called only once after the
        # algorithm converged.
        self._updateW()  # Depend on the value of C
        IterativeIASolverBaseClass._solve_finalize(self)

    # noinspection PyPep8
    def _updateC(self) -> None:
        """Update the value of Ck for all K users.

        Ck contains the orthogonal basis of the interference subspace of
        user k. It corresponds to the Nk-Sk dominant eigenvectors of

            :math:`\\sum_{l \\neq k} \\mtH_{k,l} \\mtF_l \\mtF_l^H \\mtH_{k,l}^H`.

        Notes
        -----
        This method is called in the :meth:`_step` method.

        See also
        --------
        _step
        """
        # $$\sum_{l \neq k} \mtH_{k,l} \mtF_l \mtF_l^H \mtH_{k,l}^H$$

        # xxxxxxxxxx New Implementation using calc_Q xxxxxxxxxxxxxxxxxxxxxx
        Ni = self.Nr - self.Ns  # Ni: Dimension of the interference subspace

        self._C = np.empty(self.K, dtype=np.ndarray)

        for k in np.arange(self.K):
            # TODO: Implement and test with external interference
            # # We are inside only of the first for loop
            # # Add the external interference contribution
            # self._C[k] = self.calc_Q(k) + self.Rk[k]

            # C[k] will receive the Ni most dominant eigenvectors of C[k]
            self._C[k] = peig(self.calc_Q(k), Ni[k])[0]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    def _updateF(self) -> None:
        """
        Update the value of the precoder of all K users.

        Fl, the precoder of the l-th user, tries avoid as much as possible
        to send energy into the desired signal subspace of the other
        users. Fl contains the Sl least dominant eigenvectors of
        :math:`\\sum_{k \\neq l} H_{k,l}^H (I - C_k C_k^H)H_{k,l}`

        Notes
        -----
        This method is called in the :meth:`_step` method.

        See also
        --------
        _step
        """
        # $\sum_{k \neq l} \mtH_{k,l}^H (\mtI - \mtC_k \mtC_k^H)\mtH_{k,l}$

        # xxxxx Calculates the temporary variable Y[k] for all k xxxxxxxxxx
        # Note that $\mtY[k] = (\mtI - \mtC_k \mtC_k^H)$

        self._clear_precoder_filter()

        # The number of users is always 3 for the ClosedFormIASolver class
        self._F = np.zeros(self.K, dtype=np.ndarray)

        def calc_Y(Nr: int, C: np.ndarray) -> np.ndarray:
            return (np.eye(Nr, dtype=complex) -
                    np.dot(C,
                           C.conjugate().transpose()))

        Y = list(map(calc_Y, self.Nr, self._C))

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
                np.dot(lH.conjugate().transpose(), Y[k]), lH)

        # Every element in newF is a matrix. We want to replace each
        # element by the least dominant eigenvectors of that element.
        for k in range(self.K):
            self._F[k] = leig(newF[k], self.Ns[k])[0]
            self._F[k] /= np.linalg.norm(self._F[k], 'fro')

    def _updateW(self) -> None:
        """
        Update the zero-forcing filters.

        The zero-forcing filter is calculated in the paper "MIMO
        Interference Alignment Over Correlated Channels with Imperfect
        CSI".

        Notes
        -----
        This method is called in the :meth:`_step` method.

        See also
        --------
        _step
        """
        self._clear_receive_filter()

        # Note that the formula for the receive filter in the "Interference
        # Alignment via Alternating Minimization" paper actually calculates
        # W_H instead of W.
        newW_H = np.zeros(self.K, dtype=np.ndarray)
        assert (self._F is not None)
        for k in np.arange(self.K):
            tildeHi = np.hstack(
                [np.dot(self._get_channel(k, k), self._F[k]), self._C[k]])
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
    multiUserChannel : muchannels.MultiUserChannelMatrix
        The multiuser channel.
    """
    def __init__(self, multiUserChannel: muchannels.MultiUserChannelMatrix):
        super().__init__(multiUserChannel)

    def get_cost(self) -> float:
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
            precoder. This is a (real non-negative number).
        """
        # $$C = Tr[\mtU_k^H \mtQ_k \mtU_k]$$
        assert (self._W is not None)
        cost = 0
        for k in range(self.K):
            Qk = self.calc_Q(k)
            Wk = self._W[k]
            aux = np.dot(np.dot(Wk.transpose().conjugate(), Qk), Wk)
            cost = cost + np.trace(np.abs(aux))
        return cost

    def _calc_Uk_all_k(self) -> np.ndarray:
        """
        Calculates the receive filter of all users.

        Returns
        -------
        np.ndarray
            The Uk array of each user. This is a 1D numpy array of numpy
            arrays.
        """
        Uk = np.empty(self.K, dtype=np.ndarray)

        for k in range(self.K):
            Qk = self.calc_Q(k)
            [V, _] = leig(Qk, self.Ns[k])
            Uk[k] = V
        return Uk

    def _calc_Uk_all_k_rev(self) -> np.ndarray:
        """
        Calculates the receive filter of all users in the reverse network.

        Returns
        -------
        np.ndarray
            The Uk array of each user. This is a 1D numpy array of numpy
            arrays.
        """
        Uk_rev = np.empty(self.K, dtype=np.ndarray)
        for k in range(self.K):
            Qk_rev = self.calc_Q_rev(k)
            [V, _] = leig(Qk_rev, self.Ns[k])
            Uk_rev[k] = V
        return Uk_rev

    def _updateF(self) -> None:
        """
        Update the precoders.

        Notes
        -----
        This method is called in the :meth:`IterativeIASolverBaseClass._step` method.

        See also
        --------
        IterativeIASolverBaseClass._step

        """
        self._clear_precoder_filter()
        self._F = self._calc_Uk_all_k_rev()

    def _updateW(self) -> None:
        """
        Update the receive filters.

        Notes
        -----
        This method is called in the :meth:`IterativeIASolverBaseClass._step` method.

        See also
        --------
        IterativeIASolverBaseClass._step
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

    An example of a common scenario is a scenario with 3 pairs or
    transmitter/receiver with 2 antennas in each node and 1 stream
    transmitted per node.

    You can determine the scenario of an MaxSinrIASolver object by
    inferring the variables K, Nt, Nr and Ns.

    Parameters
    ----------
    multiUserChannel : muchannels.MultiUserChannelMatrix
        The multiuser channel.
    """
    def __init__(self, multiUserChannel: muchannels.MultiUserChannelMatrix):
        super().__init__(multiUserChannel)

    # noinspection PyPep8
    def _calc_Bkl_cov_matrix_first_part_rev(self, k: int) -> np.ndarray:
        """
        Calculates the first part in the equation of the Blk covariance
        matrix of the reverse channel.

        Parameters
        ----------
        k : int
            Index of the desired user.

        Returns
        -------
        Bkl_first_part_rev : np.ndarray
            First part in equation (28) of [Cadambe2008]_, but for the
            reverse channel.

        See also
        --------
        _calc_Bkl_cov_matrix_first_part_rev

        """
        # $$\sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star d}^{[j]} \mtV_{\star d}^{[j]\dagger} \mtH^{[kj]\dagger}$$
        P = self.P
        first_part = 0.0

        assert (self._W is not None)
        for j in range(self.K):
            Hkj = self._get_channel_rev(k, j)
            Hkj_H = Hkj.conjugate().transpose()
            Vj = self._W[j]

            # The lets make sure the receive filter norm is equal to one so
            # that we can correctly scale it to the desired power.
            assert np.linalg.norm(Vj, 'fro') - 1.0 < 1e-6
            assert (isinstance(P, np.ndarray))
            assert (isinstance(self._Ns, np.ndarray))
            Vj_H = Vj.conjugate().transpose()
            first_part += (float(P[j]) / self._Ns[j]) * np.dot(
                Hkj, np.dot(np.dot(Vj, Vj_H), Hkj_H))

        return first_part

    # noinspection PyPep8,PyPep8
    def _calc_Bkl_cov_matrix_second_part_rev(self, k: int,
                                             l: int) -> np.ndarray:
        """
        Calculates the second part in the equation of the Blk covariance
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
        second_part : np.ndarray
            Second part in equation (28) of [Cadambe2008]_.
        """
        # $$\frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger}$$
        P = self.P

        Hkk = self._get_channel_rev(k, k)
        Hkk_H = Hkk.transpose().conjugate()

        assert (self._W is not None)
        Vkl = self._W[k][:, l:l + 1]
        Vkl_H = Vkl.transpose().conjugate()
        second_part = np.dot(Hkk, np.dot(np.dot(Vkl, Vkl_H), Hkk_H))
        assert (isinstance(P, np.ndarray))
        assert (isinstance(self._Ns, np.ndarray))
        return second_part * (float(P[k]) / self._Ns[k])

    # noinspection PyPep8
    def _calc_Bkl_cov_matrix_all_l_rev(self, k: int) -> np.ndarray:
        """
        Calculates the interference-plus-noise covariance matrix for all
        streams at "receiver" :math:`k` for the reverse channel.

        Parameters
        ----------
        k : int
            Index of the desired user.

        Returns
        -------
        Bkl_rev : np.ndarray
            Covariance matrix of all streams of user k. Each element of the
            returned 1D numpy array is a 2D numpy complex array
            corresponding to the covariance matrix of one stream of user k.
        """
        # $$\mtB^{[kl]} = \sum_{j=1}^{K} \frac{P^{[j]}}{d^{[j]}} \sum_{d=1}^{d^{[j]}} \mtH^{[kj]}\mtV_{\star l}^{[j]} \mtV_{\star l}^{[j]\dagger} \mtH^{[kj]\dagger} - \frac{P^{[k]}}{d^{[k]}} \mtH^{[kk]} \mtV_{\star l}^{[k]} \mtV_{\star l}^{[k]\dagger} \mtH^{[kk]\dagger} + \mtI_{N^{[k]}}$$
        assert (isinstance(self._Ns, np.ndarray))
        Bkl_all_l_rev = np.empty(self._Ns[k], dtype=np.ndarray)
        first_part = self._calc_Bkl_cov_matrix_first_part_rev(k)

        for l in range(self._Ns[k]):
            second_part = self._calc_Bkl_cov_matrix_second_part_rev(k, l)
            Bkl_all_l_rev[l] = first_part - second_part + (self.noise_var *
                                                           np.eye(self.Nt[k]))

        return Bkl_all_l_rev

    @classmethod
    def _calc_Ukl(cls, Hkk: np.ndarray, Vk: np.ndarray, Bkl: np.ndarray,
                  l: int) -> np.ndarray:
        """
        Calculates the Ukl matrix in equation (29) of [Cadambe2008]_.

        Parameters
        ----------
        Hkk : np.ndarray
            Channel from transmitter K to receiver K.
        Vk : np.ndarray
            Precoder of user k.
        Bkl : np.ndarray
            The previously calculates Bkl matrix in equation (28) of
            [Cadambe2008]_
        l : int
            Index of the desired stream

        Returns
        -------
        Ukl : np.ndarray
            The calculated Ukl matrix. This is a 2D numpy array (with
            self.Nr[k] rows and a single column).
        """
        Vkl = Vk[:, l:l + 1]

        Ukl = np.linalg.solve(Bkl, np.dot(Hkk, Vkl))

        Ukl /= np.linalg.norm(Ukl, 'fro')
        return Ukl

    @classmethod
    def _calc_Uk(cls, Hkk: np.ndarray, Vk: np.ndarray,
                 Bkl_all_l: np.ndarray) -> np.ndarray:
        """
        Similar to the :meth:`_calc_Ukl` method, but while :meth:`_calc_Ukl`
        calculates the receive filter (a vector) only for the :math:`l`-th
        stream :meth:`_calc_Uk` calculates a receive filter (a matrix) for all
        streams.

        Parameters
        ----------
        Hkk : np.ndarray
            Channel from transmitter K to receiver K.
        Vk : np.ndarray
            Precoder of user k.
        Bkl_all_l : np.ndarray
            Covariance matrix of all streams of user k. Each element of the
            returned 1D numpy array is a 2D numpy complex array
            corresponding to the covariance matrix of one stream of user k.

        Returns
        -------
        Uk : np.ndarray
            The receive filter for all streams of user k.
        """
        num_streams = Bkl_all_l.size
        num_Rx = Bkl_all_l[0].shape[0]
        Uk = np.zeros([num_Rx, num_streams], dtype=complex)
        for l in range(num_streams):
            Uk[:, l] = MaxSinrIASolver._calc_Ukl(Hkk, Vk, Bkl_all_l[l], l)[:,
                                                                           0]

        return Uk / np.linalg.norm(Uk, 'fro')

    def _calc_Uk_all_k(self) -> np.ndarray:
        """
        Calculates the receive filter of all users.

        Returns
        -------
        np.ndarray
            The receive filter of all users. This is a numpy array of numpy
            arrays.
        """
        Uk = np.empty(self.K, dtype=np.ndarray)
        assert (self._F is not None)
        for k in range(self.K):
            Hkk = self._get_channel(k, k)
            Bkl_all_l = self._calc_Bkl_cov_matrix_all_l(k, self.noise_var)
            Uk[k] = self._calc_Uk(Hkk, self._F[k], Bkl_all_l)
        return Uk

    def _calc_Uk_all_k_rev(self) -> np.ndarray:
        """
        Calculates the receive filter of all users for the reverse channel.

         Returns
        -------
        np.ndarray
            The receive filter of all users for the reverse channel. This is
            a numpy array of numpy arrays.
        """
        Uk = np.empty(self.K, dtype=np.ndarray)
        assert (self._W is not None)
        F = self._W  # The precoder is the receive filter of the direct channel
        for k in range(self.K):
            Hkk = self._get_channel_rev(k, k)
            Bkl_all_l = self._calc_Bkl_cov_matrix_all_l_rev(k)
            Uk[k] = self._calc_Uk(Hkk, F[k], Bkl_all_l)
        return Uk

    def _updateF(self) -> None:
        """
        Update the precoders.

        Notes
        -----
        This method is called in the :meth:`IterativeIASolverBaseClass._step` method.

        See also
        --------
        IterativeIASolverBaseClass._step
        """
        self._clear_precoder_filter()
        self._F = self._calc_Uk_all_k_rev()

    def _updateW(self) -> None:
        """
        Update the receive filters.

        Notes
        -----
        This method is called in the :meth:`IterativeIASolverBaseClass._step` method.

        See also
        --------
        IterativeIASolverBaseClass._step
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

    An example of a common scenario is a scenario with 3 pairs or
    transmitter/receiver with 2 antennas in each node and 1 stream
    transmitted per node.

    You can determine the scenario of an MMSEIASolver object by
    inferring the variables K, Nt, Nr and Ns.

    Parameters
    ----------
    multiUserChannel : muchannels.MultiUserChannelMatrix
        The multiuser channel.

    Notes
    -----

    .. [Peters2011] S. W. Peters and R. W. Heath, "Cooperative Algorithms
       for MIMO Interference Channels," vol. 60, no. 1, pp. 206-218, 2011.
    """
    def __init__(self, multiUserChannel: muchannels.MultiUserChannelMatrix):
        super().__init__(multiUserChannel)

        self._mu: Optional[np.ndarray] = None

    def _solve_init(self, Ns: IntOrIntSequence, P: np.ndarray) -> None:
        """
        Code run in the `solve` method before the loop that run the
        :meth:`IterativeIASolverBaseClass._step` method.

        The implementation here simple initializes the precoder variable
        and then calculates the initial receive filter.

        Parameters
        ----------
        Ns : int | np.ndarray
            Number of streams of each user.
        P : np.ndarray
            Power of each user. If not provided, a value of 1 will be used
            for each user.
        """
        IterativeIASolverBaseClass._solve_init(self, Ns, P)
        self._mu = np.zeros(self.K, dtype=float)

    # noinspection PyPep8
    def _calc_Uk(self, k: int) -> np.ndarray:
        """
        Calculates the receive filter of the k-th user.

        Parameters
        ----------
        k : int
            User index

        Returns
        -------
        Uk : np.ndarray
            The receive filter of the user 'k'.
        """
        # $$\mtU_k = \left( \sum_{i=1}^K \mtH_{ki} \mtV_i \mtV_i^H \mtH_{ki}^H + \sigma_n^2 \mtI \right)^{-1} \mtH_{kk} \mtV_k$$
        Hkk = self._get_channel(k, k)
        Vk = self.full_F[k]

        sum_term = 0
        for i in range(self.K):
            Hki = self._get_channel(k, i)
            Vi = self.full_F[i]
            aux = np.dot(Hki, Vi)
            sum_term = sum_term + np.dot(aux, aux.conj().T)

        sum_term2: np.ndarray = sum_term + self.noise_var * np.eye(self.Nr[k])

        Uk = np.linalg.solve(sum_term2, np.dot(Hkk, Vk))
        return Uk

    def _updateW(self) -> None:
        """
        Updates the receive filter of all users.
        """
        new_W = np.zeros(self.K, dtype=np.ndarray)
        for k in range(self.K):
            new_W[k] = self._calc_Uk(k)

        self._clear_receive_filter()
        self._W = new_W

    @staticmethod
    def _calc_Vi_for_a_given_mu(sum_term: np.ndarray, mu_i: float,
                                H_herm_U: np.ndarray) -> np.ndarray:
        """
        Calculates the value of Vi for the given parameters.

        This method is called inside _calc_Vi.

        Parameters
        ----------
        sum_term : np.ndarray
            The summation term in the formula to calculate the precoder.
        mu_i : float
            The value of the lagrange multiplier
        H_herm_U : np.ndarray
            The value of :math:`H_ii^H U_i`

        Returns
        -------
        np.ndarray
            The Vi matrix for the given parameters.
        """
        N = sum_term.shape[0]
        Vi = np.linalg.solve(sum_term + mu_i * np.eye(N), H_herm_U)
        # Vi = np.dot(np.linalg.inv(sum_term + mu_i * np.eye(N)),
        #             H_herm_U)

        return Vi

    @staticmethod
    def _calc_Vi_for_a_given_mu2(inv_sum_term: np.ndarray, mu_i: float,
                                 H_herm_U: np.ndarray) -> np.ndarray:
        """
        Calculates the value of Vi for the given parameters.

        This method is called inside _calc_Vi.

        Parameters
        ----------
        inv_sum_term : np.ndarray
            The inverse of the summation term in the formula to calculate
            the precoder when mu_i is equal to zero.
        mu_i : float
            The value of the lagrange multiplier
        H_herm_U : np.ndarray
            The value of :math:`H_ii^H U_i`

        Returns
        -------
        np.ndarray
            The Vi matrix for the given parameters.
        """
        N = inv_sum_term.shape[0]
        diagonal = mu_i * np.ones(N)  # Vector of N elements
        new_inv = update_inv_sum_diag(inv_sum_term, diagonal)
        Vi = np.dot(new_inv, H_herm_U)
        return Vi

    # noinspection PyPep8
    def _calc_Vi(self, i: int, mu_i: Optional[float] = None) -> np.ndarray:
        """
        Calculates the precoder of the i-th user.

        Parameters
        ----------
        i : int
            User index
        mu_i : float, optional
            The value of the Lagrange multiplier. If it is None (default),
            then the best value will be found and used to calculate the
            precoder.

        Returns
        -------
        Vi : np.ndarray
            The calculate precoder of the i-th user.
        """
        # $$\mtV_i = \left( \sum_{k=1}^K \mtH_{ki}^H \mtU_k \mtU_k^H \mtH_{ki} + \mu_i \mtI \right)^{-1} \mtH_{ii}^H \mtU_i$$

        # H_{ii}^H * Ui
        Hii_herm_U = np.dot(self._get_channel(i, i).conj().T, self.W[i])

        # xxxxx Calculates the Summation term xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        sum_term = np.array([0.0])
        for k in range(self.K):
            # H_{ki}^H * Uk
            aux = np.dot(self._get_channel(k, i).conj().T, self.W[k])
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
        cond = S.max() / S.min()
        load_factor = 0.0
        # If the condition number is larger than 1e8 we consider sum_term
        # as a singular matrix, which means that we will perform the
        # diagonal loading
        if cond > 5e4:  # pragma: no cover
            # Calculates the load_factor (arbitrarily chosen as 1/100 the
            # mean of the current singular values of sum_term).
            load_factor = S.mean() / 100.0
            # pylint: disable= E1103
            sum_term += np.eye(sum_term.shape[0]) * load_factor
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxx Case when the best mu value must be found xxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        assert (isinstance(self._mu, np.ndarray))
        if mu_i is None:
            # xxxxx Define the function that will be optimized xxxxxxxxxxxx
            # noinspection PyShadowingNames,PyIncorrectDocstring
            def func(local_mu: float, local_sum_term: np.ndarray,
                     local_Hii_herm_U: np.ndarray, local_P: float) -> float:
                """
                Function that will be optimized to find the best value of
                local_mu.
                """
                Vi = self._calc_Vi_for_a_given_mu(local_sum_term, local_mu,
                                                  local_Hii_herm_U)
                norm = np.linalg.norm(Vi, 'fro')
                cost = (norm**2) - local_P
                return cast(float, cost)

            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxxxxxxx Scale Hii_herm_U and sum_term xxxxxxxxxxxxxxxxxxxx
            # Depending of the transmit power the value of 'Hii^H Ui' and
            # `sum_term` can be very big (or maybe very small). This may
            # cause precision problems in the optimization of the Lagrange
            # multiplier mu. To avoid this problem we will scale both
            # 'Hii^H Ui' and `sum_term` by this `scale_factor` to avoid
            # that.
            #
            # Note that THIS WILL NOT CHANGE the final value of Vi,
            # although the Lagrange multiplier will be different.
            scale_factor = np.linalg.norm(Hii_herm_U)
            Hii_herm_U /= scale_factor
            sum_term /= scale_factor
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            min_mu_i = 0.0

            cost = func(min_mu_i, sum_term, Hii_herm_U, self.P[i])
            # If cost is lower than or equal to zero then the power
            # constraint is already satisfied and we are done. The value of
            # mu will be min_mu_i.
            if cost <= 0:
                assert (self._mu is not None)
                mu_i = min_mu_i
                Vi = self._calc_Vi_for_a_given_mu(sum_term, mu_i, Hii_herm_U)
                self._mu[i] = mu_i
            else:
                try:
                    # If we are not done yet then we need to perform the
                    # bisection method to find the best mu value between
                    # min_mu_i and max_mu_i
                    mu_i = optimize.newton(  # pylint: disable= E1101
                        func,
                        min_mu_i,
                        args=(sum_term, Hii_herm_U, self.P[i]),
                        maxiter=200)
                except RuntimeError:  # pragma: nocover
                    # We get a RuntimeError if the maximum number of
                    # iterations has been reached.
                    raise RuntimeError(
                        "Could not find optimum Lagrange multiplier in 200"
                        " iterations.")

                # xxxxxxxxxx Handle case where a bad mu_i was found xxxxxxx
                # Sometimes the optimization algorithm finds a solution,
                # but it is clearly wrong with a very high value of
                # mu_i. In that case we will do some scaling and try again.
                assert (isinstance(mu_i, float))
                if abs(mu_i) > 1e20:
                    mu_i = optimize.newton(func,
                                           min_mu_i,
                                           args=(sum_term * 10,
                                                 Hii_herm_U * 10, self.P[i]),
                                           maxiter=200)
                    mu_i /= 10.
                    cost = func(mu_i, sum_term, Hii_herm_U, self.P[i])
                    # If our new solution is still bad then we raise a
                    # RuntimeError exception to indicate that a good
                    # solution was not found
                    if cost > self.P[i] / 1e6:
                        # Cost is still positive. The current value for mu
                        # can't be used, since the power restriction will
                        # not be valid. Note that we allow a positive cost
                        # lower then self.P[i]/1e6 since this will be
                        # relatively close to zero.
                        msg = "Could not find a good Lagrange multiplier"
                        raise RuntimeError(msg)
                # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

                # Now that we have the best value for mu_i, lets calculate Vi
                Vi = self._calc_Vi_for_a_given_mu(sum_term, mu_i, Hii_herm_U)
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
            # Vi = self._calc_Vi_for_a_given_mu2(inv_sum_term,
            #                                    mu_i, Hii_herm_U)

        return Vi

    def _updateF(self) -> None:
        """
        Updates the precoder of all users.
        """
        # Note that _mu should never be negative. By setting the values to
        # -1 here if after the solve method some element in _mu is still
        # negative then something was not right.
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


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx GreedStreamIASolver Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class GreedStreamIASolver:
    """
    Implements the Greed Stream Interference Alignment algorithm variation.

    This is not a new IA algorithm, but rather a variation of existing IA
    algorithms. The idea is to use another IA algorithm to find the IA
    solution for the desired maximum number of streams. After the solution
    is found, we remove the worst stream and use the same algorithm again
    to find a new solution. If the solution after the stream reduction
    provided a larger sum capacity we remove the worst stream again and
    keep going until each user has only one stream or the sum capacity
    after stream reduction is lower. The final solution will be for the
    number of streams that yielded the largest sum capacity.

    Parameters
    ----------
    iasolver_obj : T <= IASolverBaseClass
        Must be an object of a derived class of IterativeIASolverBaseClass.
    """
    def __init__(self, iasolver_obj: Type[IASolverBaseClass]):
        self._iasolver = iasolver_obj
        self._runned_iterations = 0

        # #super().__init__(multiUserChannel)

        # # Maximum number of iterations that the underlying iterative IA
        # # algorithm can run for a given stream configuration.
        # self.max_iterations_per_run = 50

        # # Relative change of the precoder of the underlying algorithm to
        # # consider if the algorithm converged.
        # self.relative_factor = 1e-6

        # Store the full_F, W_H and Ns for the previous stream
        # configuration
        self._old_F: Optional[Sequence[np.ndarray]] = None
        self._old_full_F: Optional[Sequence[np.ndarray]] = None
        self._old_W_H: Optional[Sequence[np.ndarray]] = None
        self._old_Ns: Optional[Sequence[np.ndarray]] = None

    @property
    def runned_iterations(self) -> int:
        """
        Get method for the runned_iterations property.

        Returns
        -------
        int
            The number of runned iterations.
        """
        return self._runned_iterations

    def solve(self,
              Ns: IntOrIntSequence,
              P: Optional[FloatOrFloatSequence] = None) -> int:
        """
        Find the IA solution.

        This method updates the 'F' and 'W' member variables.

        Parameters
        ----------
        Ns : int | np.ndarray
            Number of streams of each user.
        P : np.ndarray | List[float] | float, optional
            Power of each user. If not provided, a value of 1 will be used
            for each user.

        Returns
        -------
        int
            Number of iterations the iterative interference alignment
            algorithm run.
        """
        assert (isinstance(self._iasolver, IterativeIASolverBaseClass))
        self._iasolver.clear()
        self._runned_iterations = 0

        # Find the solution for the number of asked streams. Note that
        # depending of the underlying IA algorithm the number of streams in
        # the solution for some user(s) can be lower then the values in Ns
        self._runned_iterations += self._iasolver.solve(Ns, P)

        # We check if any user has more then one stream, since otherwise we
        # can't remove any stream.
        #
        # The keep_going variable will then indicate if stream reduction should
        # be tried. If there is no user with more then one stream then we have
        # no way to reduce a stream.
        keep_going = bool(np.any(np.greater(self._iasolver.Ns, 1)))

        while keep_going is True:
            # xxxxxxxxxx Store the current solution xxxxxxxxxxxxxxxxxxxxxxx
            self._old_F = [F.copy() for F in self._iasolver.F]
            self._old_full_F = [
                full_F.copy() for full_F in self._iasolver.full_F
            ]
            self._old_W_H = [W_H.copy() for W_H in self._iasolver.W_H]
            self._old_Ns = copy(self._iasolver.Ns)

            old_sum_capacity = self._iasolver.calc_sum_capacity()
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxxxxxxx Find the index of the stream to be removed xxxxxxx
            # Note that you need to have called the solve method of the
            # self._iasolver object before you call this method here.
            user_idx, stream_idx = \
                self._find_index_stream_with_worst_sinr()
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Remove the stream and find a new IA solution xxxxxxxxxx
            self._iasolver.F[user_idx] = np.delete(self._iasolver.F[user_idx],
                                                   stream_idx, 1)
            self._iasolver.full_F[user_idx] = np.delete(
                self._iasolver.full_F[user_idx], stream_idx, 1)
            self._iasolver.Ns[user_idx] -= 1

            # Note that the F member variable is the normalized
            # precoder. Since we removed one column, let's normalize it
            # again.
            self._iasolver.F[user_idx] /= np.linalg.norm(
                self._iasolver.F[user_idx], 'fro')

            #
            self._iasolver.initialize_with = 'fix'
            self._runned_iterations += self._iasolver.solve(
                self._iasolver.Ns, self._iasolver.P)
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxxxxxxx Check if the new solution is better xxxxxxxxxxxxxx
            new_sum_capacity = self._iasolver.calc_sum_capacity()

            # If the new solution is not better, we restore the previous
            # solution and set keep_going to False to stop the stream
            # reduction.
            if old_sum_capacity > new_sum_capacity:
                # Lets restore the previous solution. First we clear the
                # current solution.
                self._iasolver.clear()

                # Now we set the precoders
                self._iasolver.set_precoders(F=self._old_F,
                                             full_F=self._old_full_F,
                                             P=P)
                self._iasolver.set_receive_filters(W_H=self._old_W_H)

                keep_going = False
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            # xxxxx Check if at least one user has more then 1 stream xxxxx
            # noinspection PyTypeChecker
            if not np.any(np.greater(self._iasolver.Ns, 1)):
                # If there is no user with more then 1 stream, then we
                # can't reduce streams anymore. Let's stop the while loop
                # then.
                keep_going = False  # pragma: no cover
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return self.runned_iterations

    def _find_index_stream_with_worst_sinr(self) -> Tuple[int, int]:
        """
        Considering the current solution (precoders and receive filters) in
        self._iasolver, find the index of the user and stream corresponding
        to the worst SINR.

        Returns
        -------
        user_idx : int
            The index of the user that has the stream with the worst SINR.
        stream_idx : int
            The index of the stream (for user `user_idx`) with the worst
            SINR.
        """
        sinrs = self._iasolver.calc_SINR()  # type: ignore
        # First we find the index of the minimum SINR for each user
        min_sinr_indexes = [np.argmin(s) for s in sinrs]

        # For each user we get his minimum sinr.
        min_sinrs = [
            sinrs[i][min_sinr_indexes[i]]
            for i in range(self._iasolver.K)  # type: ignore
        ]

        # Index of the users in ascending order of the sinrs. The fist
        # element is the index of the user with the minimum SINR.
        min_sinr_user_idx = np.argsort(min_sinrs)

        # Let's discard any user that has only one stream, since we can't
        # reduce the number of streams of that user.
        valid_users_idx = \
            np.arange(self._iasolver.K)[self._iasolver.Ns > 1]  # type: ignore
        min_sinr_user_idx = [
            i for i in min_sinr_user_idx if i in valid_users_idx
        ]

        user_idx = min_sinr_user_idx[0]
        stream_idx = min_sinr_indexes[user_idx]
        return user_idx, stream_idx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx BruteForceStreamIASolver xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class BruteForceStreamIASolver:
    """
    Implements the Brute Force Stream Interference Alignment algorithm
    variation.

    This is not a new IA algorithm, but rather a variation of existing IA
    algorithms. The idea is to use another IA algorithm to find the IA
    solution for each possible stream configuration (number of streams for
    each user) and keep the best one.

    Note: IA algorithms can provide different performance for the same
    number of streams but with different initializations. To reduce this
    variability we set the initialization method to 'svd' (see
    'initialize_with' property in the IterativeIASolverBaseClass class) so
    that the initialization is always the same.

    Parameters
    ----------
    iasolver_obj : T <= IASolverBaseClass
        Must be an object of a derived class of IterativeIASolverBaseClass.
    """
    def __init__(self, iasolver_obj: IterativeIASolverBaseClass):
        self._iasolver = iasolver_obj
        self._runned_iterations = 0

        # store every possible stream combination
        self._stream_combinations: List[int] = []
        # store sum capacity for each stream combination
        self._every_sum_capacity: List[float] = []

        # Store the full_F, W_H and Ns for the previous stream
        # configuration
        self._best_F: Optional[Sequence[np.ndarray]] = None
        self._best_full_F: Optional[Sequence[np.ndarray]] = None
        self._best_W_H: Optional[Sequence[np.ndarray]] = None
        self._best_Ns: Optional[Sequence[np.ndarray]] = None

    def clear(self) -> None:
        """
        Clear the BruteForceStreamIASolver object.
        """
        # store every possible stream combination
        self._stream_combinations = []
        # store sum capacity for each stream combination
        self._every_sum_capacity = []

        # Store the full_F, W_H and Ns for the previous stream
        # configuration
        self._best_F = None
        self._best_full_F = None
        self._best_W_H = None
        self._best_Ns = None

    @property
    def runned_iterations(self) -> int:
        """
        Get method for the runned_iterations property.

        Returns
        -------
        int
            The number of runned iterations.
        """
        return self._runned_iterations

    @property
    def stream_combinations(self) -> Sequence[int]:
        """
        Get method for the stream_combinations property.

        Returns
        -------
        tuple
            Tuple containing every possible stream combination.
        """
        return self._stream_combinations

    @property
    def every_sum_capacity(self) -> List[float]:
        """
        Get method for the every_sum_capacity property.

        Returns
        -------
        list
            Tuple containing the sum capacity for each stream
            combination in `self.stream_combinations`.
        """
        return self._every_sum_capacity

    def solve(self,
              Ns: IntOrIntSequence,
              P: Optional[FloatOrFloatSequence] = None) -> int:
        """
        Find the IA solution.

        This method updates the 'F' and 'W' member variables.

        Parameters
        ----------
        Ns : int | np.ndarray
            MAXIMUM number of streams of each user. All possible values
            from 1 to Ns (or Ns[k], if Ns is an array) will tried.
        P : np.ndarray | List[float] | float, optional
            Power of each user. If not provided, a value of 1 will be used
            for each user.

        Returns
        -------
        int
            Number of iterations the iterative interference alignment
            algorithm run.
        """
        self._iasolver.clear()
        self._runned_iterations = 0

        self._iasolver.initialize_with = 'svd'
        K = self._iasolver.K

        if isinstance(Ns, int):
            Ns = np.ones(K, dtype=int) * Ns
        assert (not isinstance(Ns, int))

        # xxxxxxxxxx Find all possible stream configurations xxxxxxxxxxxxxx
        # First we create a list of K lists, where each inner list has the
        # possible number of streams of one user.
        # Ex: If Ns is [2, 2, 3] then the list of lists will be
        # [[1, 2], [1, 2], [1, 2, 3]]
        each_user_variation = [range(1, Ns[i] + 1) for i in range(K)]

        # Calculate all possible combinations of the inner lists.
        self._stream_combinations = list(
            product(*each_user_variation))  # type: ignore

        # xxxxx Find the solution for each stream configuration xxxxxxxxxxx
        self._every_sum_capacity = []

        # xxxxx Compute the solution for the first combination xxxxxxxxxxxx
        stream_combinations = iter(self._stream_combinations)
        comb = next(stream_combinations)
        self._iasolver.clear()
        self._runned_iterations += self._iasolver.solve(np.array(comb), P)
        self._every_sum_capacity.append(self._iasolver.calc_sum_capacity())

        self._best_F = self._iasolver._F
        self._best_full_F = self._iasolver._full_F
        self._best_W_H = self._iasolver._W_H
        self._best_Ns = self._iasolver.Ns
        best_sum_capacity = self._every_sum_capacity[-1]
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # xxxxx Compute the solution for the remaining combinations xxxxxxx
        for comb in stream_combinations:
            self._iasolver.clear()
            self._runned_iterations += \
                self._iasolver.solve(np.array(comb), P)
            self._every_sum_capacity.append(self._iasolver.calc_sum_capacity())

            # If the current solution is better then the best one, store it
            # as the new best solution.
            if self._every_sum_capacity[-1] > best_sum_capacity:
                best_sum_capacity = self._every_sum_capacity[-1]
                self._best_F = self._iasolver._F
                self._best_full_F = self._iasolver._full_F
                self._best_W_H = self._iasolver._W_H
                self._best_Ns = self._iasolver.Ns
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        # Now that we tested every possible solution, lets keep the best
        # one we found
        self._iasolver.clear()
        self._iasolver._F = self._best_F
        self._iasolver._full_F = self._best_full_F
        self._iasolver._W_H = self._best_W_H
        self._iasolver._Ns = self._best_Ns
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        return self._runned_iterations


# xxxxxxxxxx End of the File xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
