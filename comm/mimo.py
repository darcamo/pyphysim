#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with several MIMO related classes, such as classes for the BLAST
and Alamouti MIMO schemes.
"""

__version__ = "$Revision$"
# $Source$

import numpy as np


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MimoBase Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MimoBase(object):
    """Base Class for MIMO schemes.

    All subclasses must implement the following methods:

    - :meth:`getNumberOfLayers`:
      Should return the number of layers of that specific MIMO scheme
    - :meth:`encode`:
      The encode method must perform everything executed at the transmitter
      for that specific MIMO scheme. This also include the power division
      among the transmit antennas.
    - :meth:`decode`:
      Analogous to the encode method, the decode method must perform
      everything performed at the receiver.

    """

    def __init__(self):
        pass

    def getNumberOfLayers(self):  # pragma: no cover
        """Get the number of layers of the MIMO scheme.

        Notes
        -----
        This method must be implemented in each subclass of `MimoBase`.
        """
        raise NotImplementedError('getNumberOfLayers still needs to be implemented in the {0} class'.format(self.__class__.__name__))

    @staticmethod
    def _calcZeroForceFilter(channel):
        """Calculates the Zero-Force filter to cancel the inter-stream
        interference.

        Parameters
        ----------
        channel : 2D numpy array
            MIMO channel matrix.

        Returns
        -------
        W : 2D numpy array
            The Zero-Forcing receive filter.

        Notes
        -----
        The Zero-Force filter basically corresponds to the pseudo-inverse
        of the channel matrix.

        """
        return np.linalg.pinv(channel)

    @staticmethod
    def _calcMMSEFilter(channel, noise_var):
        """Calculates the MMSE filter to cancel the inter-stream interference.

        Parameters
        ----------
        channel : 2D numpy array
            MIMO channel matrix.
        noise_var : float
            Noise variance.

        Returns
        -------
        W : 2D numpy array
            The MMSE receive filter.
        """
        H = channel
        H_H = H.transpose().conjugate()
        Nr, Nt = H.shape
        W = np.dot(np.linalg.inv(np.dot(H_H, H) + noise_var * np.eye(Nt)), H_H)
        return W

    def encode(self, transmit_data):  # pragma: no cover
        raise NotImplementedError('encode still needs to be implemented in the {0} class'.format(self.__class__.__name__))

    def decode(self, received_data, channel):  # pragma: no cover
        raise NotImplementedError('decode still needs to be implemented in the {0} class'.format(self.__class__.__name__))


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Blast Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Blast(MimoBase):
    """MIMO class for the BLAST scheme.

    The number of streams need to be specified during object creation.

    The receive filter used will depend on the noise variance (see the
    :meth:`set_noise_var` method). Of the noise variance is positive the
    MMSE filter will be used, otherwise noise variance will be ignored and
    the Zero-Forcing filter will be used.

    """

    def __init__(self, nStreams):
        """Initialized the Blast object.

        Parameters
        ----------
        nStreams : int
            The number of transmit streams.
        """
        MimoBase.__init__(self)
        # Function to calculate the receive filter
        self.calc_filter = MimoBase._calcZeroForceFilter
        self.noise_var = 0
        self.nStreams = nStreams

    def getNumberOfLayers(self):
        """Get the number of layers of the Blast scheme.

        Returns
        -------
        Nl : int
            Number of layers of the MIMO scheme.
        """
        return self.nStreams

    def set_noise_var(self, noise_var):
        """Set the noise variance for the MMSE receive filter.

        If noise_var is non-positive then the Zero-Force filter will be
        used instead.

        Parameters
        ----------
        noise_var : float
            Noise variance for the MMSE filter (if `noise_var` is
            positive). If `noise_var` is negative then the Zero-Forcing
            filter will be used and `noise_var` will be ignored.

        """
        self.noise_var = noise_var
        if noise_var > 0:
            self.calc_filter = lambda H: MimoBase._calcMMSEFilter(H, self.noise_var)
        else:
            self.calc_filter = MimoBase._calcZeroForceFilter

    def _encode(self, transmit_data):
        """Encode the transmit data array to be transmitted using the BLAST
        scheme, but **WITHOUT** dividing the power among the transmit antennas.

        The idea is that the encode method will call _encode and perform
        the power division. This separation allows better code reuse.

        Parameters
        ----------
        transmit_data : 1D numpy array
            A numpy array with a number of elements which is a multiple of
            the number of transmit antennas.

        Returns
        -------
        encoded_data : 2D numpy array
            The encoded `transmit_data` (without dividing the power among
            transmit antennas).

        Raises
        ------
        ValueError
            If the number of elements in `transmit_data` is not multiple of
            the number of transmit antennas.

        See also
        --------
        encode

        """
        num_elements = transmit_data.size
        if num_elements % self.nStreams != 0:
            raise ValueError("Input array number of elements must be a multiple of the number of transmit antennas")

        return transmit_data.reshape(self.nStreams, num_elements / self.nStreams, order='F')

    def encode(self, transmit_data):
        """Encode the transmit data array to be transmitted using the BLAST
        scheme.

        Parameters
        ----------
        transmit_data : 1D numpy array
            A numpy array with a number of elements which is a multiple of
            the number of transmit antennas.

        Returns
        -------
        encoded_data : 2D numpy array
            The encoded `transmit_data`.

        Raises
        ------
        ValueError
            If the number of elements in `transmit_data` is not multiple of
            the number of transmit antennas.
        """
        return self._encode(transmit_data) / np.sqrt(self.nStreams)

    def _decode(self, received_data, channel):
        """Decode the received data array, but does not compensate for the power
        division among transmit antennas.

        The idea is that the decode method will call _decode and perform
        the power compensation. This separation allows better code reuse.

        Parameters
        ----------
        received_data : 2D received data
            Received data, which was encoded with the Blast scheme and
            corrupted by the channel `channel`.
        channel : 2D numpy array
            MIMO channel matrix.

        Returns
        -------
        decoded_data : 1D numpy array
            The decoded data (without power compensating the power division
            performed during transmission).

        See also
        --------
        decode

        """
        (Nr, Ns) = received_data.shape
        W = self.calc_filter(channel)
        decoded_data = W.dot(received_data).reshape(self.nStreams * Ns, order='F')
        return decoded_data

    def decode(self, received_data, channel):
        """Decode the received data array.

        Parameters
        ----------
        received_data : 2D received data
            Received data, which was encoded with the Blast scheme and
            corrupted by the channel `channel`.
        channel : 2D numpy array
            MIMO channel matrix.

        Returns
        -------
        decoded_data : 1D numpy array
            The decoded data.
        """
        return self._decode(received_data, channel) * np.sqrt(self.nStreams)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Alamouti Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Alamouti(MimoBase):
    """MIMO class for the Alamouti scheme.
    """

    def __init__(self, ):
        """Initialized the Alamouti object.
        """
        MimoBase.__init__(self)

    def getNumberOfLayers(self):
        """Get the number of layers of the Alamouti scheme.

        The number of layers in the Alamouti scheme is always equal to
        one.

        Returns
        -------
        Nl : int
            Number of layers of the Alamouti scheme, which is always one.

        """
        return 1

    def _encode(self, transmit_data):
        """Perform the Alamouti encoding, but without dividing the power
        among the transmit antennas.

        The idea is that the encode method will call _encode and perform
        the power division. This separation allows better code reuse.

        Parameters
        ----------
        transmit_data : 1D numpy array
            Data to be encoded by the Alamouit scheme.

        Returns
        -------
        encoded_data : 2D numpy array
            The encoded `transmit_data` (without dividing the power among
            transmit antennas).

        See also
        --------
        encode
        """
        Ns = transmit_data.size
        encoded_data = np.empty((2, Ns), dtype=complex)
        for n in range(0, Ns, 2):
            encoded_data[0, n] = transmit_data[n]
            encoded_data[0, n + 1] = -(transmit_data[n + 1]).conjugate()
            encoded_data[1, n] = transmit_data[n + 1]
            encoded_data[1, n + 1] = (transmit_data[n]).conjugate()
        return encoded_data

    def encode(self, transmit_data):
        """Perform the Alamouiti encoding.

        Parameters
        ----------
        transmit_data : 1D numpy array
            Data to be encoded by the Alamouit scheme.

        Returns
        -------
        encoded_data : 2D numpy array
            The encoded `transmit_data`.
        """
        return self._encode(transmit_data) / np.sqrt(2)

    def _decode(self, received_data, channel):
        """Perform the decoding of the received_data for the Alamouit
        scheme with the channel `channel`, but does not compensate for the
        power division among transmit antennas.

        The idea is that the decode method will call _decode and perform
        the power compensation. This separation allows better code reuse.

        Parameters
        ----------
        received_data`: 2D numpy array
            Received data, which was encoded with the Alamouit scheme and
            corrupted by the channel `channel`.
        channel : 2D numpy array
            MIMO channel matrix.

        Returns
        -------
        decoded_data : 1D numpy array
            The decoded data (without power compensating the power division
            performed during transmission).

        See also
        --------
        decode

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
        decoded_data = decoded_data / (np.linalg.norm(channel, 'fro') ** 2)
        return decoded_data

    def decode(self, received_data, channel):
        """Perform the decoding of the received_data for the Alamouit
        scheme with the channel `channel`.

        Parameters
        ----------
        received_data`: 2D numpy array
            Received data, which was encoded with the Alamouit scheme and
            corrupted by the channel `channel`.
        channel : 2D numpy array
            MIMO channel matrix.

        Returns
        -------
        decoded_data : 1D numpy array
            The decoded data.
        """
        return self._decode(received_data, channel) * np.sqrt(2)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
