#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implement several metrics for subspaces."""

import numpy as np
import math

from projections import calcProjectionMatrix

# TODO: Create doctests for this module


# TODO: I think calcPrincipalAngles is not correct when matrix1 e matrix2
# have different sizes. At least obtaining the chordal distance from the
# principal angles does not work when matrix1 and matrix2 have different
# shapes.
def calcPrincipalAngles(matrix1, matrix2):
    """Calculates the principal angles between matrix1 and matrix2.

    Arguments:
    - `matrix1`: bi-dimensional numpy array
    - `matrix2`: bi-dimensional numpy array
    """
    # Primeiro preciso achar as matrizes de base ortonormal para matrix1 e
    # matrix2, o que consigo com a decomposicao QR. Note que se matrix1
    # possui `n` colunas então sua base orthogonal é formada pelas `n`
    # primeiras colunas de Q1
    (Q1, R1) = np.linalg.qr(matrix1)
    (Q2, R2) = np.linalg.qr(matrix2)

    # TODO: Teste quem tem mais colunas que quem. Q1 deve ter dimensao
    # maior ou igual que Q2 para que a SVD seja calculada nessa ordem
    # abaixo.
    #
    # Veja um algoritmo em
    # http://sensblogs.wordpress.com/2011/09/07/matlab-codes-for-principal-angles-also-termed-as-canonical-correlation-between-any-arbitrary-subspaces-redirected-from-jen-mei-changs-dissertation/
    (U, S, V_H) = np.linalg.svd(Q1.conjugate().transpose().dot(Q2), full_matrices=False)

    # Os valores singulares em S variam entre 0 e 1, mas devido a
    # imprecisões computacionais pode ter algum valor acima de um (por um
    # erro bem pequeno). Abaixo mudo valores maiores que 1 para 1 para
    # evitar problemas com o arccos mais tarde.
    S[S > 1] = 1  # Muda valores maiores que 1 para 1

    # Os valores singulares na matriz S sao iguais ao cosseno dos angulos
    # principais. Basta calcular o arcocosseno de cada elemento entao.
    return np.arccos(S)


def calcChordalDistanceFromPrincipalAngles(principalAngles):
    """Calculates the chordal distance from the principal angles.

    It is given by the square root of the sum of the squares of the sin of
    the principal angles.

    Arguments:
    - `principalAngles`: Numpy array with the principal angles.
    """
    return np.sqrt(np.sum(np.sin(principalAngles) ** 2))


def calcChordalDistance(matrix1, matrix2):
    """Calculates the chordal distance between the two matrices

    Arguments:
    - `matrix1`: bi-dimensional numpy array
    - `matrix2`: bi-dimensional numpy array

    """
    (Q1, R1) = np.linalg.qr(matrix1)
    (Q2, R2) = np.linalg.qr(matrix2)

    #ncols = matrix1.shape[1]  # Deve ser igual a matrix2.shape[1].

    # As primeiras ncols colunas de Q1 e Q2 formam a base ortonormal de
    # ran(matrix1) e ran(matrix2), respectivamente
    Q1_sqr = Q1.dot(Q1.conjugate().transpose())
    Q2_sqr = Q2.dot(Q2.conjugate().transpose())
    return np.linalg.norm(Q1_sqr - Q2_sqr, 'fro') / math.sqrt(2.)
    # return (Q1*Q1.conjugate_transpose() - Q2*Q2.conjugate_transpose()).norm('frob')/math.sqrt(2).n()


def calcChordalDistance2(matrix1, matrix2):
    """Calculates the chordal distance between the two matrices

    Arguments:
    - `matrix1`: bi-dimensional numpy array
    - `matrix2`: bi-dimensional numpy array

    """
    return np.linalg.norm(calcProjectionMatrix(matrix1) - calcProjectionMatrix(matrix2), 'fro') / math.sqrt(2)


# Tests
if __name__ == '__main__1':
    A = np.arange(1, 9.)
    A.shape = (4, 2)
    B = np.array([[1.2, 2.1], [2.9, 4.3], [5.2, 6.1], [6.8, 8.1]])
    print calcPrincipalAngles(A, B)
    print calcChordalDistance(A, B)
    print calcChordalDistance2(A, B)


if __name__ == '__main__':
    # Pegue do livro Matrix computations
    A = np.array([1, 2, 3, 4, 5, 6])
    A.shape = (3, 2)
    B = np.array([1, 5, 3, 7, 5, -1])
    B.shape = (3, 2)
    print (calcPrincipalAngles(A, B))


# xxxxx Perform the doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    # When this module is run as a script the doctests are executed
    import doctest
    doctest.testmod()
