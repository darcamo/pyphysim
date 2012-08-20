#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module related to subspace projection."""

import numpy as np


class Projection():
    """Class to calculate the projection, orthogonal projection and
    reflection of a given matrix in a Subspace S spanned by the columns of
    a matrix A.

    The matrix A is provied in the constructor and after that the functions
    `project`, `oProject` and `reflect` can be called with M as an
    argument.

    Example:
    >>> A = np.array([[1+1j, 2-2j], [3-2j, 0], [-1-1j, 2-3j]])
    >>> v = np.array([1,2,3])
    >>> P = Projection(A)
    >>> P.project(v)
    array([ 1.69577465+0.87887324j,  1.33802817+0.41408451j,
            2.32957746-0.56901408j])
    >>> P.oProject(v)
    array([-0.69577465-0.87887324j,  0.66197183-0.41408451j,
            0.67042254+0.56901408j])
    >>> P.reflect(v)
    array([-2.39154930-1.75774648j, -0.67605634-0.82816901j,
           -1.65915493+1.13802817j])
    """

    def __init__(self, A):
        """
        Arguments:
        - `A`:
        """
        self._A = A
        self.Q = Projection.calcProjectionMatrix(A)

        # Matrix to project in the orthogonal subspace. Note that self.Q is
        # always a square matrix
        self.oQ = np.eye(self.Q.shape[0]) - self.Q

    def project(self, M):
        """Project the matrix (or vector) M in the desired subspace.

        Arguments:
        - `M`: Matrix to be projected

        Return:
        - Projection of M in the desired subspace
        """
        return self.Q.dot(M)

    def oProject(self, M):
        """Project the matrix (or vector) M the subspace ORTHOGONAL to the
        subspace projected with `project`.

        Arguments:
        - `M`: Matrix to be projected

        Return:
        - Projection of M in the orthogonal subspace
        """
        return self.oQ.dot(M)

    def reflect(self, M):
        """Find the reflection of the matrix in the subspace spanned by the
        columns of A

        Arguments:
        - `M`: Matrix to be reflected.

        Return:
        - The reflection of M with the subspace
        """
        return (np.eye(self.Q.shape[0]) - 2 * self.Q).dot(M)

    @staticmethod
    def calcProjectionMatrix(A):
        """Calculates the projection matrix that projects a vector (or a
        matrix) into the sinal space spanned by the columns of A.

        Arguments:
        - `A`:

        Example:
        >>> A = np.array([[1+1j, 2-2j], [3-2j, 0], [-1-1j, 2-3j]])
        >>> Q = Projection.calcProjectionMatrix(A)
        >>> Q.round(4)
        array([[ 0.5239-0.j    ,  0.0366+0.3296j,  0.3662+0.0732j],
               [ 0.0366-0.3296j,  0.7690-0.j    , -0.0789+0.2479j],
               [ 0.3662-0.0732j, -0.0789-0.2479j,  0.7070+0.j    ]])

        """
        # MATLAB version: A/(A'*A)*A';
        A_H = A.conjugate().transpose()
        return (A.dot(np.linalg.inv(A_H.dot(A)))).dot(A_H)

    @staticmethod
    def calcOrthogonalProjectionMatrix(A):
        """Calculates the projection matrix that projects a vector (or a
        matrix) into the signal space orthogonal to the signal space
        spanned by the columns of M.

        Arguments:
        - `A`:

        Example:
        >>> M = np.array([[1+1j, 2-2j], [3-2j, 0], [-1-1j, 2-3j]])
        >>> oQ = calcOrthogonalProjectionMatrix(M)
        >>> oQ
        array([[ 0.47605634 +2.77555756e-17j, -0.03661972 -3.29577465e-01j,
                -0.36619718 -7.32394366e-02j],
               [-0.03661972 +3.29577465e-01j,  0.23098592 +1.38777878e-17j,
                 0.07887324 -2.47887324e-01j],
               [-0.36619718 +7.32394366e-02j,  0.07887324 +2.47887324e-01j,
                 0.29295775 -2.77555756e-17j]])
        """
        Q = Projection.calcProjectionMatrix(A)
        return np.eye(Q.shape[0]) - Q


# xxxxx Alias for the static methods of Projection class xxxxxxxxxxxxxxxxxx
calcProjectionMatrix = Projection.calcProjectionMatrix
calcOrthogonalProjectionMatrix = Projection.calcOrthogonalProjectionMatrix
calcProjectionMatrix.__doc__ += "\nThis is just an alias to Projection.calcProjectionMatrix"
calcOrthogonalProjectionMatrix.__doc__ += "\nThis is just an alias to Projection.calcOrthogonalProjectionMatrix"
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxx Perform the doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    print "{0} executed".format(__file__)
