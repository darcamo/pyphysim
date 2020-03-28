#!/usr/bin/env python
"""
Module containing useful general functions that don't belong to another
module.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, cast

import numba
import numpy as np
from scipy.special import erfc

IntOrIntArray = TypeVar("IntOrIntArray", np.ndarray, int)
NumberOrArrayUnion = Union[np.ndarray, float]


def gmd(U: np.ndarray,
        S: np.ndarray,
        V_H: np.ndarray,
        tol: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform the Geometric Mean Decomposition of a matrix A, whose SVD is
    given by `[U, S, V_H] = np.linalg.svd(A)`.

    The Geometric Mean Decomposition (GMD) is described in paper "Joint
    Transceiver Design for MIMO Communications Using Geometric Mean
    Decomposition."

    Parameters
    ----------
    U : np.ndarray
        First matrix obtained from the SVD decomposition of the original
        matrix you want to decompose.
    S : np.ndarray
        Second matrix obtained from the SVD decomposition of the original
        matrix you want to decompose.
    V_H : np.ndarray
       Third matrix obtained from the SVD decomposition of the original
       matrix you want to decompose.
    tol : float
        The tolerance.

    Returns
    -------
    (np.ndarray,np.ndarray,np.ndarray)
        The three matrices `Q`, `R` and `P` such that `A = QRP^H`, `R` is
        an upper triangular matrix and `Q` and `P` are unitary matrices.
    """
    # Note: The code here was adapted from the MATLAB code provided by the
    # original GMD authors in
    # http://www.sal.ufl.edu/yjiang/papers/gmd.m

    # \(\mtA = \mtU \mtS \mtV^H\)
    # \(\mtR = \mtU_r \mtS \mtV_r^H\)
    # \(A = \mtU \mtU_r^H S \mtV_R \mtV\)
    m = U.shape[0]
    n = V_H.shape[0]

    # Initialize R, P and Q
    R = np.zeros([m, n])
    P = V_H.conj().T.copy()
    Q = U.copy()

    # 'd' is a vector with the singular values
    d = np.copy(S)  # We copy here to avoid changing 'S'

    # l = min(m, n)
    # noinspection PyTypeChecker
    p = np.sum(S >= tol).item()  # Number of singular values >= tol

    # If there is no singular value greater than the tolerance, then we
    # throw an exception
    if p < 1:
        raise RuntimeError(
            "This is no singular value greater than the tolerance")

    # If we only have one singular value, that will be our diagonal
    # element
    if p < 2:
        R[0, 0] = d[0]

    z = np.zeros([p - 1])  # Vector
    large = 1  # index of the largest diagonal element
    small = p - 1  # index of the smallest diagonal element
    perm = np.r_[0:p]  # perm (i) = location in d of i-th largest entry
    invperm = np.r_[0:p]  # maps diagonal entries to perm

    # Geometric Mean of the 'p' largest singular values
    sigma_bar = np.prod(S[0:p])**(1. / p)

    for k in range(p - 1):
        flag = 0

        # xxxxx If flag is changed to 1 here we will not rotate xxxxxxx
        if d[k] >= sigma_bar:
            i = perm[small]
            small -= 1
            if d[i] >= sigma_bar:
                flag = 1
        else:
            i = perm[large]
            large += 1
            if d[i] <= sigma_bar:
                flag = 1
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        k1 = k + 1
        if i != k1:  # Apply permutation Pi of paper
            t = d[k1]  # Interchange d[i] and d[k1]
            d[k1] = d[i]
            d[i] = t

            j = invperm[k1]  # Update perm arrays
            perm[j] = i
            invperm[i] = j

            # Interchange columns i and k+1 of the Q and P matrices
            I = np.array([k1, i])
            J = np.array([i, k1])
            Q[:, I] = Q[:, J]
            P[:, I] = P[:, J]

        # Deltas
        delta1 = d[k]
        delta2 = d[k1]
        sq_delta1 = delta1**2
        sq_delta2 = delta2**2
        if flag:
            c = 1.0
            s = 0.0
        else:
            c = math.sqrt((sigma_bar**2 - sq_delta2) / (sq_delta1 - sq_delta2))
            s = math.sqrt(1 - c**2)

        d[k1] = delta1 * delta2 / sigma_bar  # = y in paper
        z[k] = s * c * (sq_delta2 - sq_delta1) / sigma_bar  # = x in paper
        R[k, k] = sigma_bar

        if k > 0:
            R[0:k, k] = z[0:k] * c  # new column of R
            z[0:k] = -z[0:k] * s  # new column of Z

        # First Givens Rotation matrix
        G1 = np.array([[c, -s], [s, c]])

        J = np.array([k, k1])
        P[:, J] = P[:, J].dot(G1)  # apply G1 to P

        # Second Givens Rotation Matrix
        G2 = (1. / sigma_bar) * np.array([[c * delta1, -s * delta2],
                                          [s * delta2, c * delta1]])

        Q[:, J] = Q[:, J].dot(G2)  # apply G2 to Q

    R[p - 1, p - 1] = sigma_bar
    R[0:p - 1, p - 1] = z

    return Q, R, P


def peig(A: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a matrix whose columns are the `n` dominant eigenvectors of
    `A` (eigenvectors corresponding to the `n` dominant eigenvalues).

    Parameters
    ----------
    A : np.ndarray
        A symmetric matrix (bi-dimensional numpy array).
    n : int
        Number of desired dominant eigenvectors.

    Returns
    -------
    np.ndarray, np.ndarray
        A list with two elements where the first element is a 2D numpy
        array with the desired eigenvectors, while the second element is a
        1D numpy array with the corresponding eigenvalues.

    Notes
    -----
    `A` must be a symmetric matrix so that its eigenvalues are real and
    positive.

    Raises
    ------
    ValueError
        If `n` is greater than the number of columns of `A`.

    Examples
    --------
    >>> A = np.random.randn(3,3) + 1j*np.random.randn(3,3)
    >>> V, D = peig(A, 1)

    """
    (_, ncols) = A.shape
    if n > ncols:  # A is symmetric -> we could get either nrows or ncols
        raise ValueError("`n` must be lower then the number of columns "
                         "in `A`")

    [D, V] = np.linalg.eig(A)
    indexes = np.argsort(D.real)
    indexes = indexes[::-1]
    V = V[:, indexes[0:n]]
    D = D[indexes[0:n]]
    return V, D


def leig(A: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a matrix whose columns are the `n` least significant
    eigenvectors of `A` (eigenvectors corresponding to the `n` dominant
    eigenvalues).

    Parameters
    ----------
    A : np.ndarray
        A symmetric matrix (bi-dimensional numpy array)
    n : int
        Number of desired least significant eigenvectors.

    Returns
    -------
    np.ndarray,np.ndarray
        A list with two elements where the first element is a 2D numpy
        array with the desired eigenvectors, while the second element is a
        1D numpy array with the corresponding eigenvalues.

    Notes
    -----
    `A` must be a symmetric matrix so that its eigenvalues are real and
    positive.

    Raises
    ------
    ValueError
        If `n` is greater than the number of columns of `A`.

    Examples
    --------
    >>> A = np.random.randn(3,3) + 1j*np.random.randn(3,3)
    >>> V, D = peig(A, 1)

    """
    (_, ncols) = A.shape
    if n > ncols:  # A is symmetric -> we could get either nrows or ncols
        raise ValueError("`n` must be lower then the number of columns "
                         "in `A`")

    [D, V] = np.linalg.eig(A)
    indexes = np.argsort(D.real)
    V = V[:, indexes[0:n]]
    D = D[indexes[0:n]]
    return V, D


def pretty_time(time_in_seconds: float) -> str:
    """
    Return the time in a more friendly way.

    Parameters
    ----------
    time_in_seconds : float
        Time in seconds.

    Returns
    -------
    time_string : str
        Pretty time representation as a string.

    Examples
    --------
    >>> pretty_time(30)
    '30.00s'
    >>> pretty_time(76)
    '1m:16s'
    >>> pretty_time(4343)
    '1h:12m:23s'
    """
    seconds = time_in_seconds
    minutes = int(seconds) // 60
    seconds = int(round(seconds % 60))

    hours = minutes // 60
    minutes %= 60

    if hours > 0:
        return "%sh:%02dm:%02ds" % (hours, minutes, seconds)

    if minutes > 0:
        return "%sm:%02ds" % (minutes, seconds)

    return "%.2fs" % time_in_seconds


def xor(a: int, b: int) -> int:
    """
    Calculates the xor operation between a and b.

    In python this is performed with a^b. However, sage changed the "^"
    operator. This xor function was created so that it can be used in
    either sage or in regular python.

    Parameters
    ----------
    a : int
        First number.
    b : int
        Second number.

    Returns
    -------
    int
        The result of the `xor` operation between `a` and `b`.

    Examples
    --------
    >>> xor(3,7)
    4
    >>> xor(15,6)
    9
    """
    return a.__xor__(b)


def randn_c(*args: int) -> np.ndarray:
    """
    Generates a random circularly complex gaussian matrix.

    Parameters
    ----------
    *args : any
        Variable number of arguments (int values) specifying the
        dimensions of the returned array. This is directly passed to the
        numpy.random.randn function.

    Returns
    -------
    result : np.ndarray
        A random N-dimensional numpy array (complex dtype) where the `N` is
        equal to the number of parameters passed to `randn_c`.

    Examples
    --------
    >>> a = randn_c(4,3)
    >>> a.shape
    (4, 3)
    >>> a.dtype
    dtype('complex128')

    """
    # noinspection PyArgumentList
    return (1.0 / math.sqrt(2.0)) * (np.random.randn(*args) +
                                     (1j * np.random.randn(*args)))


def randn_c_RS(RS: np.random.RandomState,
               *args: int) -> np.ndarray:  # pragma: no cover
    """
    Generates a random circularly complex gaussian matrix.

    This is essentially the same as the the randn_c function. The only
    difference is that the randn_c function uses the global RandomState
    object in numpy, while randn_c_RS use the provided RandomState
    object. This allow us greater control.

    Parameters
    ----------
    RS : np.random.RandomState
        The RandomState object used to generate the random values.
    *args : any
        Variable number of arguments specifying the dimensions of the
        returned array. This is directly passed to the
        numpy.random.randn function.

    Returns
    -------
    result : np.ndarray
        A random N-dimensional numpy array (complex dtype) where the
        `N` is equal to the number of parameters passed to `randn_c`.

    """
    if RS is None:
        # noinspection PyArgumentList
        return randn_c(*args)

    # noinspection PyArgumentList
    return (1.0 / math.sqrt(2.0)) * (RS.randn(*args) + (1j * RS.randn(*args)))


def level2bits(n: int) -> int:
    """
    Calculates the number of bits needed to represent n different
    values.

    Parameters
    ----------
    n : int
        Number of different levels.

    Returns
    -------
    num_bits : int
        Number of bits required to represent `n` levels.

    Examples
    --------
    >>> list(map(level2bits,range(1,20)))
    [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5]
    """
    if n < 1:
        raise ValueError("level2bits: n must be greater then one")
    return int2bits(n - 1)


def int2bits(n: int) -> int:
    """
    Calculates the number of bits needed to represent an integer n.

    Parameters
    ----------
    n : int
        The integer number.

    Returns
    -------
    num_bits : int
        The number of bits required to represent the number `n`.

    Examples
    --------
    >>> list(map(int2bits, range(0,19)))
    [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5]
    """
    if n < 0:
        raise ValueError("int2bits: n must be greater then zero")

    if n == 0:
        return 1

    bits = 0
    while n:
        n >>= 1
        bits += 1
    return bits


@numba.vectorize
def count_bits(n: IntOrIntArray) -> IntOrIntArray:  # pragma: no cover
    """
    Count the number of bits that are set in `n`.

    Parameters
    ----------
    n : int | np.ndarray
        An integer number or a numpy array of integer numbers.

    Returns
    -------
    Number of bits that are equal to 1 in the bit representation of the
    number `n`.

    Examples
    --------
    >>> a = np.array([3, 0, 2])
    >>> print(count_bits(a))
    [2 0 1]

    """
    count = 0
    while n > 0:
        if n & 1 == 1:
            count += 1
        n >>= 1
    return count


# # Note: This method works only for an integer `n` and returns an
# # integer. However, we are writing the documentation as if it were a numpy
# # ufunc because we will create the count_bits ufunc with it using
# # numpy.vectorize and count_bits will inherit the documentation.
# def _count_bits_single_element(n: IntOrIntArray
#                                ) -> IntOrIntArray:  # pragma: no cover
#     """
#     Count the number of bits that are set in `n`.

#     Parameters
#     ----------
#     n : int | np.ndarray
#         An integer number or a numpy array of integer numbers.

#     Returns
#     -------
#     Number of bits that are equal to 1 in the bit representation of the
#     number `n`.

#     Examples
#     --------
#     >>> a = np.array([3, 0, 2])
#     >>> print(count_bits(a))
#     [2 0 1]

#     """
#     count = 0
#     while n > 0:
#         if n & 1 == 1:
#             count += 1
#         n >>= 1
#     return count

# # Make count_bits an ufunc
# count_bits = np.vectorize(_count_bits_single_element)

# # count_bits = np.frompyfunc(_count_bits_single_element, 1, 1,
# #                            doc=_count_bits_single_element.__doc__)


def count_bit_errors(first: IntOrIntArray,
                     second: IntOrIntArray,
                     axis: Optional[Any] = None) -> IntOrIntArray:
    """
    Compare `first` and `second` and count the number of equivalent bit
    errors.

    The two arguments are assumed to have the index of transmitted and
    decoded symbols. The count_bit_errors function will compare each
    element in `first` with the corresponding element in `second`,
    determine how many bits changed and then return the total number of
    changes bits. For instance, if we compare the numbers 3 and 0, we see
    that 2 bits have changed, since 3 corresponds to '11', while 0
    corresponds to '00'.

    Parameters
    ----------
    first : int | np.ndarray
        The decoded symbols.
    second : int | np.ndarray
        The transmitted symbols.
    axis : int, optional
        Since first and second can be numpy arrays, when axis is not
        provided (that is, it is None) then the total number of bit errors
        of all the elements of the 'difference array' is returned. If axis
        is provided, then an array of bit errors is returned where the
        number of bit errors summed along the provided axis is returned.

    Returns
    -------
    bit_errors : int | np.ndarray
        The total number of bit errors.

    Examples
    --------
    >>> first = np.array([[2, 3, 3, 0], [1, 3, 1, 2]])
    >>> second = np.array([[0, 3, 2, 0], [2, 0, 1, 2]])
    >>> # The number of changed bits in each element is equal to
    >>> # array([1, 0, 1, 0, 2, 2, 0])
    >>> count_bit_errors(first, second)
    6
    >>> count_bit_errors(first, second, 0)
    array([3, 2, 1, 0])
    >>> count_bit_errors(first, second, 1)
    array([2, 4])
    """
    different_bits = xor(first, second)
    return np.sum(count_bits(different_bits), axis)  # type: ignore


def qfunc(x: float) -> float:
    """
    Calculates the 'q' function of x.

    Parameters
    ----------
    x : float
        The value to apply the Q function.

    Returns
    -------
    result : float
        Qfunc(x)

    Examples
    --------
    >>> qfunc(0.0)
    0.5
    >>> round(qfunc(1.0), 9)
    0.158655254
    >>> round(qfunc(3.0), 9)
    0.001349898
    """
    return cast(float, 0.5 * erfc(x / math.sqrt(2)))


def least_right_singular_vectors(
        A: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the three matrices. The first one is formed by the `n` least
    significative right singular vectors of `A`, the second one is
    formed by the remaining right singular vectors of `A` and the third
    one has the singular values of the singular vectors of the second
    matrix (the most significative ones).

    Parameters
    ----------
    A : np.ndarray
        A 2D numpy array.
    n : int
        An integer between 0 and the number of columns of `A`.

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        The tree matrices V0, V1 and S.

        The matrix V0 has the right singular vectors corresponding to
        the `n` least significant singular values.

        The matrix V1 has the remaining right singular vectors.

        The matrix S has the singular values corresponding to the
        remaining singular vectors `V1`.

    Notes
    -----
    Because of the sort operation, if you call
    least_right_singular_vectors(A, ncols_of_A) you will get all the right
    singular vectors of A with the column order reversed.

    Examples
    --------
    >>> A = np.array([1,2,3,6,5,4,2,2,1])
    >>> A.shape = (3,3)
    >>> (min_Vs, remaining_Vs, S) = least_right_singular_vectors(A,1)
    >>> min_Vs
    array([[-0.4474985 ],
           [ 0.81116484],
           [-0.3765059 ]])
    >>> remaining_Vs
    array([[-0.62341491, -0.64116998],
           [ 0.01889071, -0.5845124 ],
           [ 0.78166296, -0.49723869]])
    >>> S
    array([1.88354706, 9.81370681])
    """
    # Note that numpy.linalg.svd returns the hermitian of V
    [_, S, V_H] = np.linalg.svd(A, full_matrices=True)

    V = V_H.conjugate().transpose()

    # Index in crescent order of the singular values

    # Since the SVD gives the values in descending order, we just need to
    # reverse the order instead of performing a full sort
    sort_indexes = list(reversed(range(0, V.shape[0])))
    # sort_indexes = S.argsort()

    # The `n` columns corresponding to the least significative singular
    # values
    V0 = V[:, sort_indexes[0:n]]
    V1 = V[:, sort_indexes[n:]]

    return V0, V1, S[sort_indexes[n:]]


# New versions of numpy already have this method
# https://gist.github.com/1511969/222e3316048bce5763b1004331af898088ffcd9e
# def ravel_multi_index(indexes, shape):
#     """
#     Get the linear index corresponding to `indexes`.

#     The linear index is calculated in 'C' order. That is, it "travels"
#     the array faster in the fist dimension than in the last (row order
#     in bi-dimensional arrays).

#     Arguments
#     - `indexes`: A list with the indexes of each dimension in the array.
#     - `shape`: Shape of the array

#     Ex:
#     For shape=[3,3] we get the matrix
#     array([[0, 1, 2],
#            [3, 4, 5],
#            [6, 7, 8]])
#     Therefore (the indexes start at zero),
#     >>> ravel_multi_index([0,2],[3,3])
#     2

#     Similarly
#     >>> ravel_multi_index([3,1],[4,3])
#     10
#     """
#     #c order only
#     base_c = np.arange(np.prod(shape)).reshape(*shape)
#     return base_c[tuple(indexes)]


def calc_unorm_autocorr(x: np.ndarray) -> np.ndarray:
    """
    Calculates the unormalized auto-correlation of an array x starting
    from lag 0.

    Parameters
    ----------
    x : np.ndarray
        A 1D numpy array.

    Returns
    -------
    result : np.ndarray
        The unormalized auto-correlation of `x`.

    Examples
    --------
    >>> x = np.array([4, 2, 1, 3, 7, 3, 8])
    >>> calc_unorm_autocorr(x)
    array([152,  79,  82,  53,  42,  28,  32])

    """
    # R = np.convolve(x, x[::-1], 'full')
    R = np.correlate(x, x, 'full')

    # Return the auto-correlation for indexes greater then or equal to 0
    return R[R.size // 2:]


def calc_autocorr(x: np.ndarray) -> np.ndarray:
    """
    Calculates the (normalized) auto-correlation of an array x starting
    from lag 0.

    Parameters
    ----------
    x : np.ndarray
        A 1D numpy array.

    Returns
    -------
    result : np.ndarray
        The normalized auto-correlation of `x`.

    Examples
    --------
    >>> x = np.array([4, 2, 1, 3, 7, 3, 8])
    >>> calc_autocorr(x)
    array([ 1.   , -0.025,  0.15 , -0.175, -0.25 , -0.2  ,  0.   ])
    """
    x2 = x - np.mean(x)
    variance = float(np.var(x2))  # Biased variance of x2

    # We divide by x2.size because the calculated variance is the biased
    # version. If it was the unbiased we would have to divide by x2.size-1
    # instead.
    return calc_unorm_autocorr(x2) / (x2.size * variance)


# noinspection PyPep8
def update_inv_sum_diag(invA: np.ndarray, diagonal: np.ndarray) -> np.ndarray:
    """
    Calculates the inverse of a matrix :math:`(A + D)`, where :math:`D` is a diagonal
    matrix, given the inverse of :math`A` and the diagonal of :math:`D`.

    This calculation is performed using the Sherman-Morrison formula, given
    my

    .. math::
       (A+uv^T)^{-1} = A^{-1} - {A^{-1}uv^T A^{-1} \\over 1 + v^T A^{-1}u},

    where :math:`u` and :math:`v` are vectors.

    Parameters
    ----------
    invA : np.ndarray
        A 2D numpy array.
    diagonal : np.ndarray
        A 1D numpy array with the elements in the diagonal of :math:`D`.

    Returns
    -------
    new_inv : np.ndarray
        The inverse of :math:`A+D`.
    """

    #$$(A+uv^T)^{-1} = A^{-1} - {A^{-1}uv^T A^{-1} \over 1 + v^T A^{-1}u}$$

    # This function updates the inverse as the equation above when the
    # vectors "u" and "v" are equal and correspond to a column of the
    # identity matrix multiplied by a constant (only one element is
    # different of zero).
    # pylint: disable=C0111
    def calc_update_term(inv_matrix: np.ndarray, p_index: int,
                         p_indexed_element: float,
                         p_diagonal_element: float) -> np.ndarray:
        term1 = (p_diagonal_element *
                 np.outer(inv_matrix[:, p_index], inv_matrix[p_index, :]))
        return term1 / (1 + p_diagonal_element * p_indexed_element)

    new_inv = invA.copy()
    for index, diagonal_element in zip(range(diagonal.size), diagonal):
        indexed_element = new_inv[index, index]
        new_inv -= calc_update_term(new_inv, index, indexed_element,
                                    diagonal_element)

    return new_inv


def calc_confidence_interval(mean: float,
                             std: float,
                             n: int,
                             P: float = 95.0) -> Tuple[float, float]:
    """
    Calculate the confidence interval that contains the true mean (of a
    normal random variable) with a certain probability `P`, given the
    measured `mean`, standard deviation `std` for number of samples `n`.

    Only a few values are allowed for the probability `P`, which are: 50%,
    60%, 70%, 80%, 90%, 95%, 98%, 99%, 99.5%, 99.8% and 99.9%.

    Parameters
    ----------
    mean : float
        The measured mean value.
    std : float
        The measured standard deviation.
    n : int
        The number of samples used to measure the mean and standard
        deviation.
    P : float
        The desired confidence (probability in %) that true value is inside
        the calculated interval.

    Returns
    -------
    float, float
        A list with two float elements, the interval minimum and maximum
        values.

    Notes
    -----
    This function assumes that the estimated random variable is a normal
    variable.
    """
    # Dictionary that maps a desired "confidence" to the corresponding
    # critical value. See https://en.wikipedia.org/wiki/Student%27s_
    # t-distribution
    table_of_values = {
        50: 0.674,
        60: 0.842,
        70: 1.036,
        80: 1.282,
        90: 1.645,
        95: 1.960,
        98: 2.326,
        99: 2.576,
        99.5: 2.807,
        99.8: 3.090,
        99.9: 3.291
    }

    # Critical value used in the calculation of the confidence interval
    C = table_of_values[P]

    norm_std = std / np.sqrt(n)
    min_value = mean - (C * norm_std)
    max_value = mean + (C * norm_std)

    return min_value, max_value


def get_principal_component_matrix(A: np.ndarray,
                                   num_components: int) -> np.ndarray:
    """
    Returns a matrix without the "principal components" of `A`.

    This function returns a new matrix formed by the most significative
    components of `A`.

    Parameters
    ----------
    A : np.ndarray
        The original matrix, which is a 2D numpy matrix.
    num_components : int
        Number of components to be kept.

    Returns
    -------
    out : np.ndarray
        The new matrix with the dead dimensions removed.

    Notes
    -----
    You might want to normalize the returned matrix `out` after calling
    get_principal_component_matrix to have the same norm as the original
    matrix `A`.
    """
    # Note 'S' as returned by np.linalg.svd contains the singular values in
    # descending order. Therefore, we only need to keep the first
    # 'num_components' singular values (and vectors).
    [U, S, V_H] = np.linalg.svd(A)
    num_rows = U.shape[0]
    num_cols = V_H.shape[1]
    newS = np.zeros(num_rows, dtype=A.dtype)
    newS[:num_components] = S[:num_components]
    newS = np.diag(newS)[:, :num_cols]

    out = np.dot(U, np.dot(newS, V_H[:, :num_components]))

    return out


def get_range_representation(array: np.ndarray,
                             filename_mode: bool = False) -> Optional[str]:
    """
    Get the "range representation" of a numpy array consisting of a
    arithmetic progression. If no valid range representation exists,
    return None.

    Suppose you have the array
    n = [5, 10, 15, 20, 25, 30, 35, 40]
    This array is an arithmetic progression with step equal to 5 and can be
    represented as "5:5:40", which is exactly what get_range_representation
    will return for such array.

    Parameters
    ----------
    array : np.ndarray
        The array to be represented as a range expression.
    filename_mode : bool, optional
        If True, the returned representation will be more suitable to be
        used as part of a file-name. That is instead of "5:5:40" the string
        "5_(5)_40" would be returned.

    Returns
    -------
    expr : str
        A string expression representing `array`.
    """
    # Special case-> If len(array) < 4 we simply return the array
    if array.size < 4:
        return ','.join(array.astype(str))

    step = array[1] - array[0]

    # Change step from numpy.int64 or numpy.float64 to a regular int or
    # float. In the case of float, we round to 12 decimal digits. All of
    # this is only important in Python3.
    if step.dtype == int:
        step = int(step)
    elif step.dtype == float:
        step = round(float(step), 12)

    # noinspection PyTypeChecker
    if np.allclose(array[1:] - step, array[0:-1]):
        # array is an arithmetic progression
        if filename_mode is True:
            return "{0}_({1})_{2}".format(array[0], step, array[-1])
        return "{0}:{1}:{2}".format(array[0], step, array[-1])

    # array is not an arithmetic progression
    return None


def get_mixed_range_representation(array: np.ndarray,
                                   filename_mode: bool = False) -> str:
    """
    Get the "range representation" of a numpy array. This is similar to
    get_range_representation, but it no pure range representation is
    possible it will try to represent as least part of the array as range
    representations.

    Suppose you have the array
    n = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 100]

    Except for the 3 initial and the final elements, this array is an
    arithmetic progression with step equal to 5. Lets keep the 3 initial
    and the final values and represent the other values as a range
    representation.

    Parameters
    ----------
    array : np.ndarray
        The array to be represented as a range expression.
    filename_mode : bool, optional
        If True, the returned representation will be more suitable to be
        used as part of a file-name. That is instead of "5:5:40" the string
        "5_(5)_40" would be returned.

    Returns
    -------
    expr : str
        A string expression representing `array`.
    """
    if len(array) < 2:
        return '{0}'.format(array[0])

    diff = array[1:] - array[0:-1]
    diff = np.hstack([diff[0], diff])
    start = 0

    current_value = diff[0]

    output_expressions = []

    while start < len(diff):
        i = -1  # Just a start value in case the range below is empty
        for i in range(start, len(diff)):
            # if diff[i] != current_value:
            if not np.allclose(diff[i], current_value):
                end = i
                # Interval including the start, but not including the end
                output_expressions.append([start, end])
                start = end
                current_value = diff[end]
                # Break the for loop. The else statements will not run.
                break
        else:
            # If the for loop terminated normally, this code will run
            end = i
            output_expressions.append([start, end + 1])
            start = end + 1

    # xxxxx Process the results from the previous while loop. xxxxxxxxxxxxx
    # The first pair will never be changed
    for i, pair in enumerate(output_expressions[1:]):
        if pair[1] - pair[0] > 3:
            # This is a range expression

            # Get the step of this range expression
            step = array[pair[0] + 1] - array[pair[0]]

            # Get the first element in the range
            first_element = array[pair[0]]

            # Get the previous element (the element in array before this
            # range)
            previous_element = array[output_expressions[i][1] - 1]

            # If the difference of the first element in the range to the
            # previous element in the range is equal to the step of the
            # range, that means that this previous element should be in
            # this pair, and not in the previous one.
            if np.allclose(first_element - previous_element, step):
                output_expressions[i][1] -= 1
                output_expressions[i + 1][0] -= 1

    out = []
    for pair in output_expressions:
        value = get_range_representation(array[pair[0]:pair[1]], filename_mode)
        assert (value is not None)
        if value != '':
            out.append(value)

    return ','.join(out)


# noinspection PyPep8
def replace_dict_values(name: str,
                        dictionary: Dict[str, str],
                        filename_mode: bool = False) -> str:
    """
    Perform the replacements in `name` with the value of dictionary[name].

    See the usage example below:

    >>> name = "results_snr_{snr}_param_a_{param_a}"
    >>> replacements = {'snr': np.array([0,5,10,15,20]), 'param_a': 'something'}
    >>> replace_dict_values(name, replacements)
    'results_snr_[0:5:20]_param_a_something'

    Note that some small changes are performed in the dictionary prior to the
    replacement. More specifically, modifications such as changing a numpy
    array to a more compact representation (when possible). This is done by
    converting the numpy arrays with the get_mixed_range_representation
    function.

    If the string is going to be used as a filename, pass True to
    `filename_mode` as in the example below

    >>> replace_dict_values(name, replacements, True)
    'results_snr_[0_(5)_20]_param_a_something'

    Parameters
    ----------
    name : str
        The name fo be formatted.
    dictionary : dict
        The dictionary with the values to be replaced in `name`.
    filename_mode : bool, optional
        Extra parameter passed to the get_mixed_range_representation
        function. If True, the returned representation will be more
        suitable to be used as part of a file-name. That is instead of
        "5:5:40" the string "5_(5)_40" would be used.

    Returns
    -------
    new_name : str
        The value of `name` after the replacements in `dictionary`.

    Examples
    --------
    >>> name = "something {value1} - {value2} something else {value3}"
    >>> dictionary = {'value1':'bla bla', \
                      'value2':np.array([5, 10, 15, 20, 25, 30]), \
                      'value3': 76}
    >>> replace_dict_values(name, dictionary, True)
    'something bla bla - [5_(5)_30] something else 76'
    """
    new_dict = {}
    for n, v in dictionary.items():
        if isinstance(v, np.ndarray):
            v = "[{0}]".format(get_mixed_range_representation(
                v, filename_mode))
        new_dict[n] = v

    return name.format(**new_dict)


# Function taken from
# http://stackoverflow.com/questions/10480806/compare-dictionaries-ignoring-specific-keys
def equal_dicts(a: Dict[Any, Any], b: Dict[Any, Any],
                ignore_keys: List[Any]) -> bool:
    """
    Test if two dictionaries are equal ignoring certain keys.

    Parameters
    ----------
    a : dict
        The first dictionary
    b : dict
        The second dictionary
    ignore_keys : list
        A list or tuple with the keys to be ignored.
    """
    ka = set(a).difference(ignore_keys)
    kb = set(b).difference(ignore_keys)
    return ka == kb and all(a[k] == b[k] for k in ka)


def calc_decorrelation_matrix(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the decorrelation matrix that can be applied to a data vector
    whose covariance matrix is ``cov_matrix`` so that the new vector covariance
    matrix is a diagonal matrix.

    Parameters
    ----------
    cov_matrix : np.ndarray
        The covariance matrix of the original data that will be
        decorrelated. This must be a symmetric and positive semi-definite
        matrix

    Returns
    -------
    np.ndarray
        The decorrelation matrix :math:`\\mtW_D`. If the original data is a vector
        :math:`\\vtX` it can be decorrelated with :math:`\\mtW_D^T \\vtX`.

    See also
    --------
    calc_whitening_matrix
    """
    _, V = np.linalg.eig(cov_matrix)
    return V


# noinspection PyPep8
def calc_whitening_matrix(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the whitening matrix that can be applied to a data vector
    whose covariance matrix is ``cov_matrix`` so that the new vector
    covariance matrix is an identity matrix

    Parameters
    ----------
    cov_matrix : np.ndarray
        The covariance matrix of the original data that will be
        decorrelated. This must be a symmetric and positive semi-definite
        matrix

    Returns
    -------
    whitening_matrix : np.ndarray
        The whitening matrix :math:`\\mtW_W`. If the original data is a
        vector :math:`$\\vtX$` it can be whitened with
        :math:`\\mtW_W^H \\vtX`.

    Notes
    -----
    The returned `whitening_matrix` matrix will make the covariance of the
    filtered data an identity matrix. If you only need the the covariance
    matrix of the filtered data to be a diagonal matrix (not necessarily an
    identity matrix) what you want to calculate is the "decorrelation
    matrix". See the :func:`calc_decorrelation_matrix` function for that.

    See also
    --------
    calc_decorrelation_matrix
    """
    L, V = np.linalg.eig(cov_matrix)
    W = np.dot(V, np.diag(1. / (L**0.5)))
    return W


def calc_shannon_sum_capacity(sinrs: NumberOrArrayUnion) -> float:
    """
    Calculate the sum of the Shannon capacity of the values in `sinrs`

    Parameters
    ----------
    sinrs : float | np.ndarray
        SINR values (in linear scale).

    Returns
    -------
    sum_capacity : float
        Sum capacity.

    Examples
    --------
    >>> calc_shannon_sum_capacity(11.4)
    3.6322682154995127
    >>> calc_shannon_sum_capacity(20.3)
    4.412781525338476
    >>> sinrs_linear = np.array([11.4, 20.3])
    >>> print(calc_shannon_sum_capacity(sinrs_linear))
    8.045049740837989
    """
    sum_capacity = np.sum(np.log2(1 + sinrs))

    return cast(float, sum_capacity)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# # xxxxx Load Cython reimplementation of functions here xxxxxxxxxxxxxxxxxxxx
# # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# try:
#     # If the misc_c.so extension was compiled then any method defined there
#     # will replace the corresponding method defined here.
#     # pylint: disable=E0611,F0401
#     from ..c_extensions.misc_c import *  # type: ignore
#     USING_CYTHON = True
# except ImportError:  # pragma: no cover
#     import warnings
#     USING_CYTHON = False
#     warnings.warn(
#         "util.misc.count_bits will be slow, since cythonized version was not used"
#     )
# # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

if __name__ == '__main__':  # pragma: nocover
    import doctest

    doctest.testmod()
