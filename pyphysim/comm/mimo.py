#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module implementing different MIMO schemes.

Each MIMO scheme is implemented as a class inheriting from
:class:`MimoBase` and implements at least the methods `encode`, `decode`
and `getNumberOfLayers`.

"""

import numpy as np
import math

__all__ = ['MimoBase', 'Blast', 'Alamouti']

# TODO: maybe you can use the weave module (inline or blitz methods) from
# scipy to speed up things here.
# See http://docs.scipy.org/doc/scipy/reference/tutorial/weave.html


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MimoBase Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MimoBase(object):
    """
    Base Class for MIMO schemes.

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
        """
        Get the number of layers of the MIMO scheme.

        Notes
        -----
        This method must be implemented in each subclass of `MimoBase`.
        """
        m = 'getNumberOfLayers still needs to be implemented in the {0} class'
        raise NotImplementedError(m.format(self.__class__.__name__))

    @staticmethod
    def _calcZeroForceFilter(channel):
        """
        Calculates the Zero-Force filter to cancel the inter-stream
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
        """
        Calculates the MMSE filter to cancel the inter-stream interference.

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
        H_H = H.conj().T
        Nt = H.shape[1]
        W = np.linalg.solve(np.dot(H_H, H) + noise_var * np.eye(Nt), H_H)

        return W

    def encode(self, transmit_data):  # pragma: no cover, pylint: disable=W0613
        """
        Method to encode the transmit data array to be transmitted using some
        MIMO shceme. This method must be implemented in a subclass.
        """
        msg = 'encode still needs to be implemented in the {0} class'
        raise NotImplementedError(msg.format(self.__class__.__name__))

    def decode(self,
               received_data,   # pylint: disable=W0613
               channel):        # pragma: no cover, pylint: disable=W0613
        """
        Method to decode the transmit data array to be transmitted using some
        MIMO shceme. This method must be implemented in a subclass.
        """
        msg = 'decode still needs to be implemented in the {0} class'
        raise NotImplementedError(msg.format(self.__class__.__name__))


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Blast Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Blast(MimoBase):
    """
    MIMO class for the BLAST scheme.

    The number of streams need to be specified during object creation.

    The receive filter used will depend on the noise variance (see the
    :meth:`set_noise_var` method). Of the noise variance is positive the
    MMSE filter will be used, otherwise noise variance will be ignored and
    the Zero-Forcing filter will be used.
    """

    def __init__(self, nStreams):
        """
        Initialized the Blast object.

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
        """
        Get the number of layers of the Blast scheme.

        Returns
        -------
        Nl : int
            Number of layers of the MIMO scheme.
        """
        return self.nStreams

    def set_noise_var(self, noise_var):
        """
        Set the noise variance for the MMSE receive filter.

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
            self.calc_filter = lambda H: MimoBase._calcMMSEFilter(
                H,
                self.noise_var)
        else:
            self.calc_filter = MimoBase._calcZeroForceFilter

    def _encode(self, transmit_data):
        """
        Encode the transmit data array to be transmitted using the BLAST
        scheme, but **WITHOUT** dividing the power among the transmit
        antennas.

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
            # Note this is a single string
            msg = ("Input array number of elements must be a multiple of the"
                   " number of transmit antennas")
            raise ValueError(msg)

        return transmit_data.reshape(
            self.nStreams, num_elements / self.nStreams, order='F')

    def encode(self, transmit_data):
        """
        Encode the transmit data array to be transmitted using the BLAST
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
        return self._encode(transmit_data) / math.sqrt(self.nStreams)

    def _decode(self, received_data, channel):
        """
        Decode the received data array, but does not compensate for the power
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
        Ns = received_data.shape[1]
        W = self.calc_filter(channel)
        decoded_data = W.dot(received_data).reshape(self.nStreams * Ns,
                                                    order='F')
        return decoded_data

    def decode(self, received_data, channel):
        """
        Decode the received data array.

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
        return self._decode(received_data, channel) * math.sqrt(self.nStreams)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MRT xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MRT(MimoBase):
    """
    MIMO class for the MRT scheme.

    The number of streams for the MRT scheme is always equal to one, but it
    still employs multiple transmit antennas.
    """

    def __init__(self, ):
        """
        Initialized the MRT object.
        """
        MimoBase.__init__(self)

    def getNumberOfLayers(self):  # pragma: no cover
        """
        Get the number of layers of the MRT scheme.

        The returned value is always equal to 1.
        """
        return 1

    def encode(self, transmit_data, channel):
        """
        Encode the transmit data array to be transmitted using the MRT scheme.

        The MRT scheme corresponds to multiplying the symbol from each
        transmit antenna with a complex number corresponding to the inverse
        of the phase of the channel so as to ensure that the signals add
        constructively at the receiver. This also means that the MRT echeme
        only be applied to senarios with a single receive antenna.

        Parameters
        ----------
        transmit_data : 1D numpy array
            A numpy array with the data to be transmitted.
        channel : 1D numpy array
            MISO channel vector. It must be a 1D numpy array, where the
            number of receive antennas is assumed to be equal to 1.

        Returns
        -------
        encoded_data : 2D numpy array
            The encoded `transmit_data`.
        """
        Nt = channel.shape[-1]

        # Add an extra first dimension so that broadcast does the right
        # thing later
        x = transmit_data[np.newaxis, :]

        # Calculate the transmit filter 'W'
        if len(channel.shape) == 1:
            # Channel must be a 1D numpy array with dimention Nt
            # W will have dimension (Nt x 1)
            W = np.exp(-1j * np.angle(channel[:, np.newaxis]))
        else:
            Nr = channel.shape[0]
            if Nr != 1:
                raise ValueError("The MRT scheme is only defined for the "
                                 "scenario with a single receive antenna")
            W = np.exp(-1j * np.angle(channel)).T

        # Elementwise multiplication
        encoded_data = (W * x) / math.sqrt(Nt)
        return encoded_data

    def decode(self, received_data, channel):
        """
        Decode the received data array.

        Parameters
        ----------
        received_data : 2D or 1D numpy array
            Received data, which was encoded with the MRT scheme and
            corrupted by the channel `channel`.
        channel : 1D or 2D numpy array
            MIMO channel matrix. If it is a 1D numpy array assume the
            number of receive antennas is equal to 1. If it is 1D then
            `received_data` also needs to be 1D.

        Returns
        -------
        decoded_data : 1D numpy array
            The decoded data.
        """
        Nt = channel.shape[-1]
        if len(channel.shape) == 1:
            # Channel is 1D. Since this is MRT, there is no need for a
            # fancy receiver. All we need to do is to divide by the channel
            # sum of the channel absolute values and compensate for the
            # power division applied at the transmission side.
            decoded_data \
                = math.sqrt(Nt) * received_data / np.sum(np.abs(channel))
        else:
            # Channel is 2D. Note that the first dimension corresponding
            # to the number of receive antennas MUST be equal to 1.
            Nr = channel.shape[0]
            if Nr != 1:
                raise ValueError("The MRT scheme is only defined for the "
                                 "scenario with a single receive antenna")

            decoded_data \
                = math.sqrt(Nt) * received_data / np.sum(np.abs(channel))
            decoded_data.shape = (decoded_data.size)

        return decoded_data


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MRC Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MRC(Blast):
    """
    MIMO class for the MRC scheme.

    The number of streams need to be specified during object creation.

    The receive filter used will depend on the noise variance (see the
    :meth:`set_noise_var` method). Of the noise variance is positive the
    MMSE filter will be used, otherwise noise variance will be ignored and
    the Zero-Forcing filter will be used.

    The receive filter in the `Blast` class already does the maximum ratio
    combining. Therefore, this MRC class simply inherits from the Blast
    class and only exists for completion.
    """

    def __init__(self, nStreams):
        """
        Initialized the MRC object.

        Parameters
        ----------
        nStreams : int
            The number of transmit streams.
        """
        Blast.__init__(self, nStreams)

    # def decode(self, received_data, channel):
    #     """
    #     Decode the received data array.

    #     Parameters
    #     ----------
    #     received_data : 2D numpy array
    #         Received data, which was encoded with the Blast scheme and
    #         corrupted by the channel `channel`.
    #     channel : 2D numpy array
    #         MIMO channel matrix.

    #     Returns
    #     -------
    #     decoded_data : 1D numpy array
    #         The decoded data.
    #     """
    #     equiv_channel = np.dot(channel.conj().T, channel)
    #     equiv_received_data = np.dot(channel.conj().T, received_data)
    #     return self._decode(equiv_received_data, equiv_channel) * math.sqrt(
    #         self.nStreams)



# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Alamouti Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Alamouti(MimoBase):
    """
    MIMO class for the Alamouti scheme.
    """

    def __init__(self, ):
        """
        Initialized the Alamouti object.
        """
        MimoBase.__init__(self)

    def getNumberOfLayers(self):
        """
        Get the number of layers of the Alamouti scheme.

        The number of layers in the Alamouti scheme is always equal to
        one.

        Returns
        -------
        Nl : int
            Number of layers of the Alamouti scheme, which is always one.
        """
        return 1

    @staticmethod
    def _encode(transmit_data):
        """
        Perform the Alamouti encoding, but without dividing the power among
        the transmit antennas.

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
        """
        Perform the Alamouiti encoding.

        Parameters
        ----------
        transmit_data : 1D numpy array
            Data to be encoded by the Alamouit scheme.

        Returns
        -------
        encoded_data : 2D numpy array
            The encoded `transmit_data`.
        """
        return self._encode(transmit_data) / math.sqrt(2)

    @staticmethod
    def _decode(received_data, channel):
        """
        Perform the decoding of the received_data for the Alamouit scheme with
        the channel `channel`, but does not compensate for the power
        division among transmit antennas.

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
        Ns = received_data.shape[1]
        # Number of Alamouti codewords
        number_of_blocks = Ns // 2
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
            decoded_data[2 * i] = (
                np.dot(h0_conj, received_data[:, 2 * i]) +
                np.dot(h1, received_data[:, 2 * i + 1].conjugate())
            )
            decoded_data[2 * i + 1] = (
                np.dot(h1_conj, received_data[:, 2 * i]) +
                np.dot(minus_h0, received_data[:, 2 * i + 1].conjugate())
            )

        # The Alamouti code gives a gain of the square of the frobenius
        # norm of the channel. We need to compensate that gain.
        decoded_data = decoded_data / (np.linalg.norm(channel, 'fro') ** 2)
        return decoded_data

    def decode(self, received_data, channel):
        """
        Perform the decoding of the received_data for the Alamouit scheme with
        the channel `channel`.

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
        return self._decode(received_data, channel) * math.sqrt(2)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
