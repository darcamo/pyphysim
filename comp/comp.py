#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with implementation of Coordinated Multipoint (COmP) algorithms"""

import numpy as np

# xxxxxxxxxx Remove latter xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import sys
sys.path.append("../")
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

from subspace.projections import calcOrthogonalProjectionMatrix
from comm.blockdiagonalization import BlockDiaginalizer

# Used for debug
from util.misc import randn_c
from comm import blockdiagonalization
from comm import channels


class Comp(BlockDiaginalizer):
    """Performs the Coordinated Multipoint transmission.

    This class basically performs the Block Diagonalization in a joint
    transmission assuring that the Power restriction of each Base Station
    is not violated..

    """

    def __init__(self, iNUsers, iPu, noiseVar):
        """
        - `iNUsers`: Number of users.
        - `iPu`: Power available for EACH user.
        - `noiseVar`: Noise variance (power in linear scale).
        """
        BlockDiaginalizer.__init__(self, iNUsers, iPu, noiseVar)

    def perform_comp(self, mtChannel):
        """Perform the block diagonalization and possible interference handling.

        Arguments:
        - `mtChannel`: (numpy array) Channel from the transmitter to all
                       users.

        """
        return BlockDiaginalizer.block_diagonalize(self, mtChannel)


class CompExtInt(Comp):
    """Performs the Coordinated Multipoint transmission also taking into
    account the external interference.

    The CompExtInt class performs the block diagonalization characteristic
    to the joint transmission scenario where multiple base stations act as
    a single transmitter to send data to the users. However, in addition to
    what the BlockDiaginalizer class does the CompExtInt class can also
    take external interference into account.

    One way to reduce of eliminate the external interference is to
    sacrifice streams in directions strongly occupied by the external
    interference.

    Reference:
      See the `BlockDiaginalizer` class for details about the block
      diagonalization process.

    """

    def __init__(self, iNUsers, iPu, noiseVar):
        """
        - `iNUsers`: Number of users.
        - `iPu`: Power available for EACH user.
        - `noiseVar`: Noise variance (power in linear scale).
        """
        BlockDiaginalizer.__init__(self, iNUsers, iPu, noiseVar)

    def perform_comp(self, mtChannel, Re):
        """Perform the block diagonalization and possible interference handling.

        Arguments:
        - `mtChannel`: (numpy array) Channel from the transmitter to all
                       users.
        - `Re`: (numpy array of matrices) Covariance matrix of the external
                interference plus noise of each user. `Re` must be a numpy
                array of dimension (K x 1), where K is the number of users,
                and each element in `Re` is a numpy bidimensional array of
                size (Nrk x Nrk), where 'Nrk' is the number of receive
                antennas of the k-th user.

        """
        # Q = np.empty(self.iNUsers, dtype=np.ndarray)
        # for k in range(self.iNUsers):
        #     # Calculates a projection matrix to the subspace orthogonal to
        #     # the external interference
        #     Q[k] = calcOrthogonalProjectionMatrix(Re)



        (newH, Ms_good) = BlockDiaginalizer.block_diagonalize(self, mtChannel)

        # xxxxxxxxxx Calculates the SINR with only BD xxxxxxxxxxxxxxxxxxxxx
        print "lala"




        return (newH, Ms_good)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def perform_comp(mtChannel, iNUsers, iPu, noiseVar):
    """Performs the block diagonalization of `mtChannel`.

    Arguments:
    - `mtChannel`: (numpy bidimensional array) Global channel matrix
    - `iNUsers`: (int) Number of users
    - `iPu`: (float) Power available for each user
    - `noiseVar`: (float) Noise variance

    Return:
    - newH: Block diagonalized channel
    - Ms_good: Precoding matrix used to block diagonalize the channel

    """
    COMP = Comp(iNUsers, iPu, noiseVar)
    results_tuble = COMP.perform_comp(mtChannel)
    return results_tuble


def perform_comp_with_ext_int(mtChannel, iNUsers, iPu, noiseVar, Re):
    """Performs the block diagonalization of `mtChannel` and possible
    sacrificing of streams to reduce external interference.

    Arguments:
    - `mtChannel`: (numpy bidimensional array) Global channel matrix
    - `iNUsers`: (int) Number of users
    - `iPu`: (float) Power available for each user
    - `noiseVar`: (float) Noise variance
    - `Re`: (numpy array of matrices) Covariance matrix of the external
            interference plus noise of each user. `Re` must be a numpy
            array of dimension (K x 1), where K is the number of users, and
            each element in `Re` is a numpy bidimensional array of size
            (Nrk x Nrk), where 'Nrk' is the number of receive antennas of
            the k-th user.

    Return:
    - newH: Block diagonalized channel
    - Ms_good: Precoding matrix used to block diagonalize the channel

    """
    COMP = CompExtInt(iNUsers, iPu, noiseVar)
    results_tuble = COMP.perform_comp(mtChannel)
    return results_tuble


if __name__ == '__main__':
    Pu = 5.
    noise_var = 0.1
    num_users = 3
    num_antenas = 2
    # Total number of transmit and receive antennas
    #iNr = iNt = num_antenas * num_users

    #channel = randn_c(iNr, iNt)

    channel = channels.MultiUserChannelMatrix()
    channel.randomize(num_antenas, num_antenas, num_users)

    (newH, Ms) = blockdiagonalization.block_diagonalize(
        channel.big_H, num_users, Pu, noise_var)

    print newH.shape

    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    for userindex in range(num_users):
        # Code here will be moved to the CompExtInt.perform_comp method later
        mtW_bd = np.linalg.pinv(newH)
        mtP = np.dot(mtW_bd, newH)

        # SNR without interference handling
        desiredPower = np.abs(np.diag(mtP)) ** 2
        internalInterference = np.sum(np.abs(mtP - np.diag(np.diag(mtP))) ** 2, axis=1)

        #externalInterference =

        print internalInterference

        a = np.array([[1, 2, 3], [4, 5, 6], [5, 4, 3]])
        print a
        print np.sum(a, axis=1)
