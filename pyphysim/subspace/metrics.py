#!/usr/bin/env python
"""Implement several metrics for subspaces."""

import math
from typing import cast

import numpy as np

from .projections import calcProjectionMatrix

__all__ = [
    "calc_principal_angles", "calc_chordal_distance_from_principal_angles",
    "calc_chordal_distance", "calc_chordal_distance_2"
]


# TODO: I think calc_principal_angles is not correct when matrix1 e matrix2
# have different sizes. At least obtaining the chordal distance from the
# principal angles does not work when matrix1 and matrix2 have different
# shapes.
def calc_principal_angles(matrix1: np.ndarray,
                          matrix2: np.ndarray) -> np.ndarray:
    """
    Calculates the principal angles between `matrix1` and `matrix2`.

    Parameters
    ----------
    matrix1 : np.ndarray
        A 2D numpy array.
    matrix2 : np.ndarray
        A 2D numpy array.

    Returns
    -------
    np.ndarray
        The principal angles between `matrix1` and `matrix2`. This is a
        1D numpy array.

    See also
    --------
    calc_chordal_distance_from_principal_angles

    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4], [5, 6]])
    >>> B = np.array([[1, 5], [3, 7], [5, -1]])
    >>> print(calc_principal_angles(A, B))
    [0.         0.54312217]
    """
    # First we need to find the orthogonal basis for matrix1 and
    # matrix2. This can be done with the QR decomposition. Note that if
    # matrix1 has 'n' columns then its orthogonal basis is given by the
    # first 'n' columns of the 'Q' matrix from its QR decomposition.
    Q1 = np.linalg.qr(matrix1)[0]
    Q2 = np.linalg.qr(matrix2)[0]

    # TODO: Test who has more columns. Q1 must have dimension grater than
    # or equal to Q2 so that the SVD can be calculated in the order below.
    #
    # See the algorithm in
    # http://sensblogs.wordpress.com/2011/09/07/matlab-codes-for-principal-angles-also-termed-as-canonical-correlation-between-any-arbitrary-subspaces-redirected-from-jen-mei-changs-dissertation/
    S = np.linalg.svd(Q1.conjugate().transpose().dot(Q2),
                      full_matrices=False)[1]

    # The singular values of S vary between 0 and 1, but due to
    # computational impressions there can be some value above 1 (by a very
    # small value). Below we change values greater then 1 to be equal to 1
    # to avoid problems with the arc-cos call later.
    S[S > 1] = 1  # Change values greater then 1 to 1

    # The singular values in the matrix S are equal to the cosine of the
    # principal angles. We can calculate the arc-cosine of each element
    # then.
    return np.arccos(S)


# noinspection PyPep8
def calc_chordal_distance_from_principal_angles(
        principalAngles: np.ndarray) -> float:
    """
    Calculates the chordal distance from the principal angles.

    It is given by the square root of the sum of the squares of the sin of
    the principal angles.

    Parameters
    ----------
    principalAngles : np.ndarray
        Numpy array with the principal angles. This is a 1D numpy array.

    Returns
    -------
    chord_dist : float
        The chordal distance.

    See also
    --------
    calc_principal_angles,
    calc_chordal_distance,
    calc_chordal_distance_2

    Examples
    --------
    >>> A = np.arange(1, 9.)
    >>> A.shape = (4, 2)
    >>> B = np.array([[1.2, 2.1], [2.9, 4.3], [5.2, 6.1], [6.8, 8.1]])
    >>> princ_angles = calc_principal_angles(A, B)
    >>> print(round(calc_chordal_distance_from_principal_angles(princ_angles), 8))
    0.47386786
    """
    # noinspection PyTypeChecker
    summation = (np.sum(np.sin(principalAngles)**2)).item()
    return math.sqrt(summation)


def calc_chordal_distance(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
    Calculates the chordal distance between the two matrices

    Parameters
    ----------
    matrix1 : np.ndarray
        A 2D numpy array.
    matrix2 : np.ndarray
        A 2D numpy array.

    Returns
    -------
    chord_dist : float
        The chordal distance.

    Notes
    -----
    Same as :func:`calc_chordal_distance_2`, but implemented differently.

    See also
    --------
    calc_chordal_distance_2,
    calc_chordal_distance_from_principal_angles

    Examples
    --------
    >>> A = np.arange(1, 9.)
    >>> A.shape = (4, 2)
    >>> B = np.array([[1.2, 2.1], [2.9, 4.3], [5.2, 6.1], [6.8, 8.1]])
    >>> print(round(calc_chordal_distance(A, B), 8))
    0.47386786
    """
    Q1 = np.linalg.qr(matrix1)[0]
    Q2 = np.linalg.qr(matrix2)[0]

    # ncols = matrix1.shape[1]  # Must be equal to matrix2.shape[1].

    # The first ncols columns of Q1 and Q2 make orthogonal basis of
    # ran(matrix1) and ran(matrix2), respectively
    Q1_sqr = Q1.dot(Q1.conjugate().transpose())
    Q2_sqr = Q2.dot(Q2.conjugate().transpose())
    return cast(float, np.linalg.norm(Q1_sqr - Q2_sqr, 'fro') / math.sqrt(2.))


def calc_chordal_distance_2(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
    Calculates the chordal distance between the two matrices

    Parameters
    ----------
    matrix1 : np.ndarray
        A 2D numpy array.
    matrix2 : np.ndarray
        A 2D numpy array.

    Returns
    -------
    chord_dist : float
        The chordal distance.

    Notes
    -----
    Same as :func:`calc_chordal_distance`, but implemented differently.

    See also
    --------
    calc_chordal_distance,
    calc_chordal_distance_from_principal_angles

    Examples
    --------
    >>> A = np.arange(1, 9.)
    >>> A.shape = (4, 2)
    >>> B = np.array([[1.2, 2.1], [2.9, 4.3], [5.2, 6.1], [6.8, 8.1]])
    >>> print(round(calc_chordal_distance_2(A, B), 8))
    0.47386786
    """
    return cast(
        float,  #
        (np.linalg.norm(
            calcProjectionMatrix(matrix1) - calcProjectionMatrix(matrix2),
            'fro') / math.sqrt(2))  #
    )
