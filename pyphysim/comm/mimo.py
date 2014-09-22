#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module implementing different MIMO schemes.

Each MIMO scheme is implemented as a class inheriting from
:class:`MimoBase` and implements at least the methods `encode`, `decode`
and `getNumberOfLayers`.

"""

from abc import ABCMeta, abstractmethod
import numpy as np
import math
import warnings
from pyphysim.util.misc import gmd

__all__ = ['MimoBase', 'Blast', 'Alamouti', 'MRT', 'MRC', 'SVDMimo']

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
    # The MimoBase class is an abstract class and all methods marked as
    # 'abstract' must be implemented in a subclass.
    __metaclass__ = ABCMeta

    def __init__(self, channel=None):
        """
        Initialized the MimoBase object.

        Parameters
        ----------
        channel : 1D or 2D numpy array
            MIMO channel matrix. The allowed dimensions will depend on the
            particular MIMO scheme implemented in a subclass.
        """
        self._channel = None
        if channel is not None:
            self.set_channel_matrix(channel)

    def set_channel_matrix(self, channel):
        """
        Set the channel matrix.

        Parameters
        ----------
        channel : 1D or 2D numpy array
            MIMO channel matrix. The allowed dimensions will depend on the
            particular MIMO scheme implemented in a subclass.
        """
        self._channel = channel

    @property
    def Nt(self):
        """
        Get the number of transmit antennas
        """
        return self._channel.shape[1]

    @property
    def Nr(self):
        """
        Get the number of receive antennas
        """
        return self._channel.shape[0]

    @abstractmethod
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

    @abstractmethod
    def encode(self, transmit_data):  # pragma: no cover, pylint: disable=W0613
        """
        Method to encode the transmit data array to be transmitted using some
        MIMO shceme. This method must be implemented in a subclass.
        """
        msg = 'encode still needs to be implemented in the {0} class'
        raise NotImplementedError(msg.format(self.__class__.__name__))

    @abstractmethod
    def decode(self, received_data):  # pragma: no cover, pylint: disable=W0613
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

    The receive filter used will depend on the noise variance (see the
    :meth:`set_noise_var` method). If the noise variance is positive the
    MMSE filter will be used, otherwise noise variance will be ignored and
    the Zero-Forcing filter will be used.
    """

    def __init__(self, channel=None):
        """
        Initialized the Blast object.

        If `channel` is not provided you need to call the
        `set_channel_matrix` method before calling the `decode` or the
        `getNumberOfLayers` methods.

        Parameters
        ----------
        channel : 2D numpy array
            MIMO channel matrix.
        """
        MimoBase.__init__(self, channel)
        # Function to calculate the receive filter
        self.calc_filter = MimoBase._calcZeroForceFilter
        self.noise_var = 0

    def set_channel_matrix(self, channel):
        """
        Set the channel matrix.

        Parameters
        ----------
        channel : 2D numpy array
            MIMO channel matrix.
        """
        Nr, Nt = channel.shape
        if Nt > Nr:
            # Since the number of streams will be equal to the number of
            # transmit antennas, the number of receive antennas should be
            # greater than or equal to the number of transmit antennas. If
            # this is not the case, then you won't be able to recover the
            # streams in the decode method.
            msg = ("The number of transmit antennas for {0} should not be "
                   "greater than the number of receive antennas.").format(
                       self.__class__.__name__)
            warnings.warn(msg)

        self._channel = channel

    def getNumberOfLayers(self):
        """
        Get the number of layers of the Blast scheme.

        Returns
        -------
        Nl : int
            Number of layers of the MIMO scheme.
        """
        return self.Nt

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

    def _encode(self, transmit_data, reshape_order='F'):
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
        reshape_order : str
            Memory layout used when reshaping the numpy array. This can be
            either 'F' (default) or 'C'.

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
        nStreams = self.getNumberOfLayers()
        if num_elements % nStreams != 0:
            # Note this is a single string
            msg = ("Input array number of elements must be a multiple of the"
                   " number of transmit antennas")
            raise ValueError(msg)

        return transmit_data.reshape(nStreams, -1, order=reshape_order)

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
        return self._encode(transmit_data) / math.sqrt(self.Nt)

    def _decode(self, received_data, channel, reshape_order='F'):
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
        reshape_order : str
            Memory layout used when reshaping the numpy array. This can be
            either 'F' (default) or 'C'.

        Returns
        -------
        decoded_data : 1D numpy array
            The decoded data (without power compensating the power division
            performed during transmission).

        See also
        --------
        decode
        """
        nStreams = self.getNumberOfLayers()
        Ns = received_data.shape[1]
        W = self.calc_filter(channel)
        decoded_data = W.dot(received_data).reshape(-1, order=reshape_order)
        return decoded_data

    def decode(self, received_data):
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
        return self._decode(received_data, self._channel) * math.sqrt(self.Nt)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MRT xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MRT(MimoBase):
    """
    MIMO class for the MRT scheme.

    The number of streams for the MRT scheme is always equal to one, but it
    still employs multiple transmit antennas.
    """

    def __init__(self, channel=None):
        """
        Initialized the MRT object.

        If `channel` is not provided you need to call the
        `set_channel_matrix` method before calling the other methods.

        Parameters
        ----------
        channel : 1D numpy array
            MISO channel vector. It must be a 1D numpy array, where the
            number of receive antennas is assumed to be equal to 1.
        """
        MimoBase.__init__(self, channel=None)
        if channel is not None:
            self.set_channel_matrix(channel)

    def set_channel_matrix(self, channel):
        """
        Set the channel matrix.

        Parameters
        ----------
        channel : 1D or 2D numpy array
            MISO channel vector. The MRT MIMO scheme is defined for the
            scenario with multiple transmit antennas and a single receive
            antenna. If channel is 2D then the first dimension size must be
            equal to 1.
        """
        # We will store the channel as a 2D numpy to be consistent with the
        # other MIMO classes
        if len(channel.shape) == 1:
            self._channel = channel[np.newaxis, :]
        else:
            Nr = channel.shape[0]
            if Nr != 1:
                raise ValueError("The MRT scheme is only defined for the "
                                 "scenario with a single receive antenna")
            self._channel = channel

    def getNumberOfLayers(self):  # pragma: no cover
        """
        Get the number of layers of the MRT scheme.

        The returned value is always equal to 1.
        """
        return 1

    def encode(self, transmit_data):
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

        Returns
        -------
        encoded_data : 2D numpy array
            The encoded `transmit_data`.
        """
        # Add an extra first dimension so that broadcast does the right
        # thing later
        x = transmit_data[np.newaxis, :]
        W = np.exp(-1j * np.angle(self._channel)).T / math.sqrt(self.Nt)

        # Elementwise multiplication (use broadcast)
        encoded_data = (W * x)
        return encoded_data

    def decode(self, received_data):
        """
        Decode the received data array.

        Parameters
        ----------
        received_data : 2D or 1D numpy array
            Received data, which was encoded with the MRT scheme and
            corrupted by the channel `channel`.

        Returns
        -------
        decoded_data : 1D numpy array
            The decoded data.
        """
        decoded_data \
            = math.sqrt(self.Nt) * received_data / np.sum(np.abs(self._channel))
        decoded_data.shape = (decoded_data.size)

        return decoded_data


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MRC Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MRC(Blast):
    """
    MIMO class for the MRC scheme.

    The receive filter used will depend on the noise variance (see the
    :meth:`set_noise_var` method). If the noise variance is positive the
    MMSE filter will be used, otherwise noise variance will be ignored and
    the Zero-Forcing filter will be used.

    The receive filter in the `Blast` class already does the maximum ratio
    combining. Therefore, this MRC class simply inherits from the Blast
    class and only exists for completion.
    """

    def __init__(self, channel=None):
        """
        Initialized the MRC object.

        Parameters
        ----------
        channel : 2D numpy array
            MIMO channel matrix.
        """
        Blast.__init__(self, channel)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SVD MIMO xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SVDMimo(Blast):
    """
    MIMO class for the SVD MIMO scheme.
    """

    def __init__(self, channel=None):
        """
        Initialized the SVD MIMO object.

        Parameters
        ----------
        channel : 2D numpy array
            MIMO channel matrix.
        """
        Blast.__init__(self, channel)

    def encode(self, transmit_data):
        """
        Encode the transmit data array to be transmitted using the SVD MIMO
        scheme.

        The SVD MIMO scheme corresponds to using the 'U' and 'V' matrices
        from the SVD decomposition of the channel as the precoder and
        receive filter.

        Parameters
        ----------
        transmit_data : 1D numpy array
            A numpy array with the data to be transmitted.

        Returns
        -------
        encoded_data : 2D numpy array
            The encoded `transmit_data`.
        """
        num_elements = transmit_data.size
        if num_elements % self.Nt != 0:
            msg = ("Input array number of elements must be a multiple of the"
                   " number of transmit antennas")
            raise ValueError(msg)

        X = transmit_data.reshape(self.Nt, -1)
        U, S, V_H = np.linalg.svd(self._channel)

        # The transmit filter is the 'V' matrix from the SVD decomposition
        # of the channel. Notice that we also need to divide by sqrt(Nt) to
        # make sure 'W' has a unitary norm.
        W = V_H.conj().T / math.sqrt(self.Nt)
        encoded_data = W.dot(X)

        return encoded_data

    def decode(self, received_data):
        """
        Perform the decoding of the received_data for the SVD MIMO scheme with
        the channel `channel`.

        Parameters
        ----------
        received_data`: 2D numpy array
            Received data, which was encoded with the Alamouit scheme and
            corrupted by the channel `channel`.

        Returns
        -------
        decoded_data : 1D numpy array
            The decoded data.
        """
        U, S, V_H = np.linalg.svd(self._channel)
        G = np.diag(1./S).dot(U.conj().T) * math.sqrt(self.Nt)

        decoded_data = G.dot(received_data)

        # Return the decoded data as a 1D numpy array
        return decoded_data.reshape(decoded_data.size,)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx GMD MIMO xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class GMDMimo(Blast):
    """
    MIMO class for the GMD based MIMO scheme.
    """

    def __init__(self, channel=None):
        """
        Initialized the SVD MIMO object.

        Parameters
        ----------
        channel : 2D numpy array
            MIMO channel matrix.
        """
        Blast.__init__(self, channel)

    def encode(self, transmit_data):
        """
        Encode the transmit data array to be transmitted using the GMD MIMO
        scheme.

        The GMD MIMO scheme is based on the Geometric Mean Decomposition
        (GMD) of the channel. The channel is decomposed into `H = Q R P^H`,
        where `R` is an upper triangular matrix with all diagonal elements
        being equal to the geometric mean of the singular values of the
        channel matrix `H`.

        corresponds to using the 'U' and 'V' matrices
        from the SVD decomposition of the channel as the precoder and
        receive filter.

        Parameters
        ----------
        transmit_data : 1D numpy array
            A numpy array with the data to be transmitted.

        Returns
        -------
        encoded_data : 2D numpy array
            The encoded `transmit_data`.
        """
        num_elements = transmit_data.size
        if num_elements % self.Nt != 0:
            msg = ("Input array number of elements must be a multiple of the"
                   " number of transmit antennas")
            raise ValueError(msg)

        # The encode method will precode the transmit_data using the
        # matrix 'P' obtained from the gmd.
        U, S, V_H = np.linalg.svd(self._channel)
        Q, R, P = gmd(U, S, V_H)
        W = P / math.sqrt(self.Nt)

        X = transmit_data.reshape(self.Nt, -1)
        encoded_data = W.dot(X)

        return encoded_data

    def decode(self, received_data):
        """
        Perform the decoding of the received_data for the GMD MIMO.

        Parameters
        ----------
        received_data`: 2D numpy array
            Received data, which was encoded with the Alamouit scheme and
            corrupted by the channel `channel`.

        Returns
        -------
        decoded_data : 1D numpy array
            The decoded data.
        """
        U, S, V_H = np.linalg.svd(self._channel)
        Q, R, P = gmd(U, S, V_H)
        channel_eq = Q.dot(R)

        decoded_data = self._decode(
            received_data, channel_eq, reshape_order='C') * math.sqrt(self.Nt)
        return decoded_data


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Alamouti Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Alamouti(MimoBase):
    """
    MIMO class for the Alamouti scheme.
    """

    def __init__(self, channel=None):
        """
        Initialized the Alamouti object.

        Parameters
        ----------
        channel : 2D numpy array
            MIMO channel matrix.
        """
        MimoBase.__init__(self, channel)
        # if self.Nt != 2:
        #     msg = ("Invalid channel dimensions. Alamouti MIMO scheme requires"
        #            " exact two transmit antennas.")
        #     raise ValueError(msg)

    def set_channel_matrix(self, channel):
        """
        Set the channel matrix.

        Parameters
        ----------
        channel : 1D or 2D numpy array
            MIMO channel matrix.
        """
        if len(channel.shape) == 1:
            self._channel = channel[np.newaxis, :]
        else:
            _, Nt = channel.shape
            if Nt != 2:
                msg = ("The number of transmit antennas must be equal to 2 for"
                       " the {0} scheme").format(self.__class__.__name__)
                raise ValueError(msg)
            self._channel = channel

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

    def decode(self, received_data):
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
        return self._decode(received_data, self._channel) * math.sqrt(2)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
