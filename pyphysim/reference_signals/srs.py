#!/usr/bin/env python
"""Module with Sounding Reference Signal (SRS) related functions"""

from typing import List, Tuple, Union, cast

import numpy as np

from .root_sequence import RootSequence
from .zadoffchu import get_shifted_root_seq

# Type representing a shape
Shape = Tuple[int, ...]

# Type representing something that can be used to index a numpy array
Indexes = Union[np.ndarray, List[int], slice]

__all__ = ['get_srs_seq', 'SrsUeSequence']


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Module Functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def get_srs_seq(root_seq: np.ndarray, n_cs: int) -> np.ndarray:
    """
    Get the shifted root sequence suitable as the SRS sequence of a user.

    Parameters
    ----------
    root_seq : np.ndarray
        The root sequence to shift. This is a complex numpy array.
    n_cs : int
        The desired cyclic shift number. This should be an integer from 0
        to 7, where 0 will just return the base sequence, 1 gives the first
        shift, and so on.

    Returns
    -------
    np.ndarray
        The shifted root sequence.

    See Also
    --------
    .get_shifted_root_seq, .dmrs.get_dmrs_seq
    """
    return get_shifted_root_seq(root_seq, n_cs, 8)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Classes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class UeSequence:
    """
    Reference signal sequence of a single user.

    You should not use this class directly and instead use a class that
    inherits from it and provides the desired reference sequence.

    Parameters
    ----------
    root_seq : RootSequence
        The root sequence of the base station the user is
        associated to. This should be an object of the RootSequence
        class.
    n_cs : int
        The shift index of the user. This can be an integer from 1 to 8.
    user_seq_array : np.ndarray
        The user sequence.
    normalize : bool
        True if the reference signal should be normalized. False otherwise.
    """
    def __init__(self,
                 root_seq: RootSequence,
                 n_cs: int,
                 user_seq_array: np.ndarray,
                 normalize: bool = False):
        self._n_cs = n_cs
        self._root_index = root_seq.index
        self._normalized = normalize

        if normalize is True:
            ndim = user_seq_array.ndim
            if ndim == 1:
                # No cover code -> user_seq_array is a 1D array
                norm_factor = np.linalg.norm(user_seq_array)
            else:
                # Case with Cover code -> user_seq_array first dimension is
                # the cover code dimension, while the second dimension is
                # the sequence elements dimension
                norm_factor = np.linalg.norm(user_seq_array[0])
            self._user_seq_array = \
                user_seq_array / norm_factor
        else:
            self._user_seq_array = user_seq_array

    @property
    def normalized(self) -> bool:
        """True if the reference signal is normalized.
        """
        return self._normalized

    @property
    def size(self) -> int:
        """
        Return the size of the reference signal sequence.

        Returns
        -------
        size : int
            The size of the user's reference signal sequence.

        Examples
        --------
        >>> root_seq1 = RootSequence(root_index=25, Nzc=139)
        >>> user_seq1 = SrsUeSequence(root_seq1, 3)
        >>> user_seq1.size
        139
        >>> root_seq2 = RootSequence(root_index=25, Nzc=139, size=150)
        >>> user_seq2 = SrsUeSequence(root_seq2, 3)
        >>> user_seq2.size
        150
        """
        return cast(int, self.seq_array().size)

    @property
    def shape(self) -> Shape:
        """
        Return the shape of the reference signal sequence.

        Returns
        -------
        Tuple[int]
            The shape of the reference signal sequence.
        """
        return cast(Shape, self.seq_array().shape)

    def seq_array(self) -> np.ndarray:
        """
        Get reference signal sequence as a numpy array.

        Returns
        -------
        seq : np.ndarray
            The user's reference signal sequence.
        """
        return self._user_seq_array

    def __getitem__(self, val: Indexes) -> np.ndarray:
        """
        Index the sequence.

        This will simply return the same indexing of the underlying numpy
        array.

        Parameters
        ----------
        val : any
            Anything accepted as indexing by numpy arrays.
        """
        return self.seq_array()[val]

    def __repr__(self) -> str:  # pragma: no cover
        """
        Get the representation of the object.

        Returns
        -------
        str
            The representation of the object.
        """
        return "<{0}(root_index={1}, n_cs={2})>".format(
            self.__class__.__name__, self._root_index, self._n_cs)

    # xxxxxxxxxx Define some basic methods xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # We can always just get the equivalent numpy array and perform the
    # operations on it, but having these operations defined here is
    # convenient

    # TODO: Make these operation methods (add, mul, etc) also work with
    # RootSequence objects returning a new RootSequence object. Change the
    # docstring type information when you do that.
    def __add__(self, other: np.ndarray) -> np.ndarray:  # pragma: no cover
        """
        Perform addition with `other`.

        Parameters
        ----------
        other : np.ndarray

        Returns
        -------
        np.ndrray
        """

        return self.seq_array() + other

    def __radd__(self, other: np.ndarray) -> np.ndarray:  # pragma: no cover
        """
        Perform addition with `other`.

        Parameters
        ----------
        other : np.ndarray

        Returns
        -------
        np.ndrray
        """
        return self.seq_array() + other

    def __mul__(self, other: np.ndarray) -> np.ndarray:  # pragma: no cover
        """
        Perform multiplication with `other`.

        Parameters
        ----------
        other : np.ndarray

        Returns
        -------
        np.ndrray
        """
        return self.seq_array() * other

    def __rmul__(self, other: np.ndarray) -> np.ndarray:  # pragma: no cover
        """
        Perform multiplication with `other`.

        Parameters
        ----------
        other : np.ndarray

        Returns
        -------
        np.ndrray
        """
        return self.seq_array() * other

    def conjugate(self) -> np.ndarray:  # pragma: no cover
        """
        Return the conjugate of the root sequence as a numpy array.

        Returns
        -------
        np.ndarray
            The conjugate of the root sequence.
        """

        return self.seq_array().conj()

    def conj(self) -> np.ndarray:  # pragma: no cover
        """
        Return the conjugate of the root sequence as a numpy array.

        Returns
        -------
        np.ndarray
            The conjugate of the root sequence.
        """

        return self.seq_array().conj()

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


class SrsUeSequence(UeSequence):
    """
    SRS sequence of a single user.

    Parameters
    ----------
    root_seq : RootSequence
        The SRS root sequence of the base station the user is
        associated to. This should be an object of the RootSequence
        class.
    n_cs : int
        The shift index of the user. This can be an integer from 0 to 7.
    normalize : bool
        True if the reference signal should be normalized. False otherwise.
    """
    def __init__(self,
                 root_seq: RootSequence,
                 n_cs: int,
                 normalize: bool = False):
        root_seq_array = root_seq.seq_array()
        user_seq_array = get_srs_seq(root_seq_array, n_cs)
        super().__init__(root_seq, n_cs, user_seq_array, normalize=normalize)
