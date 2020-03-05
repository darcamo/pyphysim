#!/usr/bin/env python
"""Module related to subspace projection."""

import numpy as np


class Projection:
    """
    Class to calculate the projection, orthogonal projection and
    reflection of a given matrix in a Subspace `S` spanned by the
    columns of a matrix `A`.

    The matrix A is provided in the constructor and after that the
    functions `project`, `oProject` and `reflect` can be called with M
    as an argument.

    Parameters
    ----------
    A : np.ndarray
        The matrix whose columns form a basis for the projected
        subspace.

    Examples
    --------
    >>> A = np.array([[1 + 1j, 2 - 2j], [3 - 2j, 0], [-1 - 1j, 2 - 3j]])
    >>> v = np.array([1, 2, 3])
    >>> P = Projection(A)
    >>> P.project(v)
    array([1.69577465+0.87887324j, 1.33802817+0.41408451j,
           2.32957746-0.56901408j])
    >>> P.oProject(v)
    array([-0.69577465-0.87887324j,  0.66197183-0.41408451j,
            0.67042254+0.56901408j])
    >>> P.reflect(v)
    array([-2.3915493 -1.75774648j, -0.67605634-0.82816901j,
           -1.65915493+1.13802817j])
    """
    def __init__(self, A: np.ndarray):
        self._A = A
        self.Q = Projection.calcProjectionMatrix(A)

        # Matrix to project in the orthogonal subspace. Note that self.Q is
        # always a square matrix
        self.oQ = Projection.calcOrthogonalProjectionMatrix(A)

    def project(self, M: np.ndarray) -> np.ndarray:
        """
        Project the matrix (or vector) M in the desired subspace.

        Parameters
        ----------
        M : np.ndarray
            The matrix to be projected.

        Returns
        -------
        np.ndarray
            The projection of `M` into the desired subspace.
        """
        return self.Q.dot(M)

    def oProject(self, M: np.ndarray) -> np.ndarray:
        """
        Project the matrix (or vector) M the subspace ORTHOGONAL to the
        subspace projected with `project`.

        Parameters
        ----------
        M : np.ndarray
            The matrix to be projected.

        Returns
        -------
        np.ndarray
            The projection of `M` into the orthogonal subspace.
        """
        return self.oQ.dot(M)

    def reflect(self, M: np.ndarray) -> np.ndarray:
        """Find the reflection of the matrix in the subspace spanned by
        the columns of `A`

        Parameters
        ----------
        M : np.ndarray
            The matrix to be projected.

        Returns
        -------
        np.ndarray
            The reflection of `M` in the subspace.
        """
        return (np.eye(self.Q.shape[0]) - 2 * self.Q).dot(M)

    @staticmethod
    def calcProjectionMatrix(A: np.ndarray) -> np.ndarray:
        """
        Calculates the projection matrix that projects a vector (or a
        matrix) into the signal space spanned by the columns of `A`.

        Parameters
        ----------
        A : np.ndarray
            A matrix whose columns form a basis for the desired subspace.

        Returns
        -------
        np.ndarray
            The projection matrix that can be used to project a vector or a
            matrix into the subspace spanned by the columns of `A`

        See also
        --------
        calcOrthogonalProjectionMatrix

        Examples
        --------
        >>> A = np.array([[1 + 1j, 2 - 2j], [3 - 2j, 0], \
                          [-1 - 1j, 2 - 3j]])
        >>> # Matrix that projects into the subspace spanned by the columns
        >>> # of A
        >>> Q = calcProjectionMatrix(A)
        >>> np.allclose(Q.round(4), np.array( \
          [[ 0.5239+0.j, 0.0366+0.3296j, 0.3662+0.0732j], \
           [ 0.0366-0.3296j, 0.7690+0.j, -0.0789+0.2479j], \
           [ 0.3662-0.0732j, -0.0789-0.2479j, 0.7070-0.j]]))
        True
        """
        # MATLAB version: A/(A'*A)*A';
        A_H = A.conjugate().transpose()
        return (A.dot(np.linalg.inv(A_H.dot(A)))).dot(A_H)

    @staticmethod
    def calcOrthogonalProjectionMatrix(A: np.ndarray) -> np.ndarray:
        """
        Calculates the projection matrix that projects a vector (or a
        matrix) into the signal space orthogonal to the signal space
        spanned by the columns of M.

        Parameters
        ----------
        A : np.ndarray
            A matrix whose columns form a basis for the "desired subspace".

        Returns
        -------
        np.ndarray
            The projection matrix that can be used to project a vector or a
            matrix into the subspace orthogonal to the subspace spanned by
            the columns of `A`

        See also
        --------
        calcProjectionMatrix

        Examples
        --------
        >>> A = np.array([[1, 2], [2, 2], [4, 3]])
        >>> # Matrix that projects into the subspace orthogonal to the
        >>> # subspace spanned by the columns of A
        >>> oQ = calcOrthogonalProjectionMatrix(A)
        >>> print(oQ)
        [[ 0.12121212 -0.3030303   0.12121212]
         [-0.3030303   0.75757576 -0.3030303 ]
         [ 0.12121212 -0.3030303   0.12121212]]
        """
        Q = Projection.calcProjectionMatrix(A)
        return np.eye(Q.shape[0]) - Q


# xxxxx Alias for the static methods of Projection class xxxxxxxxxxxxxxxxxx
calcProjectionMatrix = Projection.calcProjectionMatrix
calcOrthogonalProjectionMatrix = Projection.calcOrthogonalProjectionMatrix
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
