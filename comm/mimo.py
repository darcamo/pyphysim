#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""module docstring"""

__version__ = "$Revision$"
# $Source$

import numpy as np


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Mimo Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Mimo():
    """Base Class for MIMO schemes.
    """

    def __init__(self):
        """
        """

    def getNumberOfLayers(self):
        """Get the number of layers of the MIMO scheme."""
        raise NotImplementedError('getNumberOfLayers still needs to be implemented in the {0} class'.format(self.__class__.__name__))

    @staticmethod
    def _calcZeroForceFilter(channel):
        """Calculates the Zero-Force filter to cancel the inter-stream
        interference.

        The Zero-Force filter basically corresponds to the pseudo-inverse
        of the channel matrix.

        Arguments:
        - `channel`: MIMO Channel Matrix
        """
        print "Zero-Force used"
        return np.linalg.pinv(channel)

    @staticmethod
    def _calcMMSEFilter(channel, noise_var):
        """Calculates the MMSE filter to cancel the inter-stream
        interference.

        Arguments:
        - `channel`: MIMO Channel Matrix
        - `noise_var`: Noise variance
        """
        print "MMSE used"
        H = channel
        H_H = H.transpose().conjugate()
        Nr, Nt = H.shape
        W = np.dot(np.linalg.inv(np.dot(H_H, H) + noise_var * np.eye(Nt)), H_H)
        return W

    def encode(self, transmit_data):
        """
        """
        raise NotImplementedError('encode still needs to be implemented in the {0} class'.format(self.__class__.__name__))

    def decode(self, received_data, channel):
        """
        """
        raise NotImplementedError('decode still needs to be implemented in the {0} class'.format(self.__class__.__name__))


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Blast Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Blast(Mimo):
    """
    """

    def __init__(self, Nt):
        """
        """
        Mimo.__init__(self)
        # Function to calculate the receive filter
        self.calc_filter = Mimo._calcZeroForceFilter
        self.noise_var = 0
        self.Nt = Nt

    def getNumberOfLayers(self, ):
        """Get the number of layers of the MIMO scheme.
        """
        return self.Nt

    def set_noise_var(self, noise_var):
        """Set the noise variance for the MMSE receive filter.

        If noise_var is non-positive then the Zero-Force filter will be
        used instead.

        Arguments:
        - `noise_var`: Noise variance.
        """
        self.noise_var = noise_var
        if noise_var > 0:
            self.calc_filter = lambda H: Mimo._calcMMSEFilter(H, self.noise_var)
        else:
            self.calc_filter = Mimo._calcZeroForceFilter

    def encode(self, transmit_data):
        """Encode the transmit data array to be transmitted using the BLAST
        scheme.

        Arguments:
        - `transmit_data`: A numpy array with a number of elements which is a multiple of the number of transmit antennas.
        """
        num_elements = transmit_data.size
        assert num_elements % Nt == 0, "Input array number of elements must be a multiple of the number of transmit antennas"
        return transmit_data.reshape(self.Nt, num_elements / Nt, order='F') / np.sqrt(self.Nt)

    def decode(self, received_data, channel):
        """Decode the received data array.

        Arguments:
        - `received_data`:
        - `channel`: Channel matrix
        """
        (Nr, Ns) = received_data.shape
        W = self.calc_filter(channel)
        decoded_data = W.dot(received_data) * np.sqrt(Nt)
        return decoded_data.reshape(self.Nt * Ns, order='F')


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Alamouti Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Alamouti(Mimo):
    """
    """

    def __init__(self, ):
        """
        """
        Mimo.__init__(self)

    def getNumberOfLayers(self, ):
        """Get the number of layers of the MIMO scheme.
        """
        return 1

    # TODO: Falta dividir a potência entre as antenas transmissoras
    def encode(self, transmit_data):
        """

        Arguments:
        - `transmit_data`:
        """
        Ns = transmit_data.size
        encoded_data = np.empty((2, Ns), dtype=complex)
        for n in range(0, Ns, 2):
            encoded_data[0, n] = transmit_data[n]
            encoded_data[0, n + 1] = -(transmit_data[n + 1]).conjugate()
            encoded_data[1, n] = transmit_data[n + 1]
            encoded_data[1, n + 1] = (transmit_data[n]).conjugate()
        return encoded_data

    # TODO: Falta dividir a potência entre as antenas transmissoras
    # TODO: Apagar depois
    # Mais lento que encode
    def encode2(self, transmit_data):
        """

        Arguments:
        - `transmit_data`:
        """
        # transmit_data will have symbols $s_1, s_2, s_3, s_4, \ldots$

        # Number of symbols
        Ns = transmit_data.size

        # aux is the conjugate of transmit_data and the signal of the first
        # of each two symbols will also be changed
        aux = transmit_data.conjugate()
        aux[np.arange(0, Ns) % 2 == 1] = -aux[np.arange(0, Ns) % 2 == 1]

        # Then we use some clever reshaping
        transmit_data = transmit_data.reshape(2, Ns / 2, order='F')
        aux = aux.reshape(2, Ns / 2, order='F')

        # Swap the two rows of aux
        aux[0], aux[1] = aux[1].copy(), aux[0].copy()

        # Column i of transmit_data corresponds to the first column of the
        # i-th codeword, while column i of aux corresponds to the second
        # column of the i-th codeword.
        encoded_data = np.empty((2, Ns), dtype=complex)
        encoded_data[:, np.arange(0, Ns) % 2 == 0] = transmit_data
        encoded_data[:, np.arange(0, Ns) % 2 == 1] = aux

        # encoded_data will then have the form
        # $\begin{bmatrix} s_1 & -s_2^* & s_3 & -s_4^* \\ s_2 & s_1^* & s_4 & s_3^*\end{bmatrix} \ldots$
        return encoded_data

    def decode(self, received_data, channel):
        """

        Arguments:
        - `received_data`:
        - `channel`:
        """
        Nr, Ns = received_data.shape
        # Number of Alamouti codewords
        number_of_blocks = Ns / 2
        decoded_data = np.empty(Ns, dtype=complex)

        # Conjugate of the first column of the channel (first transmit
        # antenna to all receive antennas)
        h0_conj = channel[:, 0].conjugate()
        minus_h0 = -channel[:, 0]
        # Second column of the channel (second transmit antenna to all
        # receive antennas)
        h1 = channel[:, 1]
        h1_conj = channel[:, 1].conjugate()

        for i in range(0, number_of_blocks):
            # decoded_data[2 * i] = np.dot(channel[:, 0].conjugate(), received_data[:, 2 * i]) + np.dot(channel[:, 1], received_data[:, 2 * i + 1].conjugate())
            decoded_data[2 * i] = np.dot(h0_conj, received_data[:, 2 * i]) + np.dot(h1, received_data[:, 2 * i + 1].conjugate())
            decoded_data[2 * i + 1] = np.dot(h1_conj, received_data[:, 2 * i]) + np.dot(minus_h0, received_data[:, 2 * i + 1].conjugate())

        # The Alamouti code gives a gain of the square of the frobenius
        # norm of the channel. We need to compensate that gain.
        decoded_data = decoded_data / np.linalg.norm(channel, 'fro') ** 2
        return decoded_data


if __name__ == '__main__':
    Nt = 2
    Nr = 3
    Ns = 16

    transmit_data = np.arange(0, Ns)
    transmit_data = transmit_data + 1j * transmit_data
    alamouti = Alamouti()
    encoded_data = alamouti.encode(transmit_data)

    channel = np.random.randn(Nr, Nt)
    received_data = channel.dot(encoded_data)

    decoded_data = alamouti.decode(received_data, channel)
    print decoded_data


if __name__ == '__main__1':
    data = np.arange(0, 9)
    print data
    Nt = 3
    Nr = 4
    blast = Blast(Nt)
    print blast.getNumberOfLayers()
    transmitted_data = blast.encode(data)
    print transmitted_data

    channel = np.random.randn(Nr, Nt)
    received_data = channel.dot(transmitted_data)

    blast.set_noise_var(0.001)
    decoded_data = blast.decode(received_data, channel)
    print decoded_data
