#!/usr/bin/env python
"""Module implementing different MIMO schemes.

Each MIMO scheme is implemented as a class inheriting from
:class:`MimoBase` and implements at least the methods `encode`, `decode`
and `getNumberOfLayers`.

"""

import math
import warnings
from abc import ABCMeta, abstractmethod
from typing import Optional, cast

import numpy as np

from pyphysim.util.conversion import linear2dB
from pyphysim.util.misc import gmd

__all__ = [
    'MimoBase', 'MisoBase', 'Blast', 'Alamouti', 'MRT', 'MRC', 'SVDMimo',
    'GMDMimo'
]

# TODO: maybe you can use the weave module (inline or blitz methods) from
# scipy to speed up things here.
# See http://docs.scipy.org/doc/scipy/reference/tutorial/weave.html


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Functions to calculate the SINRs xxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def calc_post_processing_SINRs(
        channel: np.ndarray,
        W: np.ndarray,
        G_H: np.ndarray,
        noise_var: Optional[float] = None) -> np.ndarray:
    """
    Calculate the post processing SINRs (in dB) of all streams for a given
    MIMO scheme.

    Parameters
    ----------
    channel : np.ndarray
        The MIMO channel. This should be a 2D numpy array.
    W : np.ndarray
        The precoder for the MIMO scheme. This should be a 2D numpy array.
    G_H : np.ndarray
        The receive filter for the MIMO scheme. This should be a 2D numpy
        array.
    noise_var : float
        The noise variance

    Returns
    -------
    np.ndarray
        The SINR of all streams (in linear scale).
    """
    return linear2dB(
        calc_post_processing_linear_SINRs(channel, W, G_H, noise_var))


def calc_post_processing_linear_SINRs(
        channel: np.ndarray,
        W: np.ndarray,
        G_H: np.ndarray,
        noise_var: Optional[float] = None) -> np.ndarray:
    """
    Calculate the post processing SINRs (in linear scale) of all streams
    for a given MIMO scheme.

    Parameters
    ----------
    channel : np.ndarray
        The MIMO channel. This should be a 2D numpy array.
    W : np.ndarray
        The precoder for the MIMO scheme. This should be a 2D numpy array.
    G_H : np.ndarray
        The receive filter for the MIMO scheme. This should be a 2D
        numpy array.
    noise_var : float
        The noise variance

    Returns
    -------
    np.ndarray
        The SINR of all streams (in linear scale).
    """
    if noise_var is None:  # pragma: nocover
        noise_var = 0.0

    # This matrix will always be square
    channel_eq = np.dot(G_H, channel.dot(W))
    sum_all_antennas = np.sum(channel_eq, axis=1)
    s = np.diag(channel_eq)
    i = sum_all_antennas - s

    S = np.abs(s)**2
    I = np.abs(i)**2

    if isinstance(G_H, np.ndarray):
        # G_H is a numpy array. Lets calculate the norm considering the
        # second axis (axis 1). That is, calculate the norm of each row in
        # G_H. The square of this norm will gives us the amount of noise
        # amplification in each stream.
        N = noise_var * np.linalg.norm(G_H, axis=1)**2
    else:
        # G_H is a single number. The square of its absolute value will
        # gives us the noise amplification of the single stream.
        N = noise_var * abs(G_H)**2

    sinrs = S / (I + N)

    return sinrs


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MimoBase Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MimoBase:
    """
    Base Class for MIMO schemes.

    All subclasses must implement at least the following methods:

    - :meth:`getNumberOfLayers`:
      Should return the number of layers of that specific MIMO scheme
    - :meth:`encode`:
      The encode method must perform everything executed at the transmitter
      for that specific MIMO scheme. This also include the power division
      among the transmit antennas.
    - :meth:`decode`:
      Analogous to the encode method, the decode method must perform
      everything performed at the receiver.

    If possible, subclasses should implement the `_calc_precoder` and
    `_calc_receive_filter` static methods and use them in the
    implementation of `encode` and `decode`. This will allow using the
    `calc_linear_SINRs` and `calc_SINRs` methods to calculate the post
    processing SINRs. Note that calling the `_calcZeroForceFilter` and
    `_calcMMSEFilter` methods in the implementation of the receive filter
    calculation can be useful.

    If you can't implement the `_calc_precoder` and `_calc_receive_filter`
    static methods` (because there is no linear precoder or receive filter
    for the MIMO scheme in the subclass, for instance), then you should
    implement the `calc_linear_SINRs` method in the subclass instead.

    Parameters
    ----------
    channel : np.ndarray | None
        MIMO channel matrix. This should be a 1D or 2D numpy array. The
        allowed dimensions will depend on the particular MIMO scheme
        implemented in a subclass.
    """
    # The MimoBase class is an abstract class and all methods marked as
    # 'abstract' must be implemented in a subclass.
    __metaclass__ = ABCMeta

    def __init__(self, channel: Optional[np.ndarray] = None):
        self._channel = channel

        if channel is not None:
            self.set_channel_matrix(channel)

    def set_channel_matrix(self, channel: np.ndarray) -> None:
        """
        Set the channel matrix.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix. This should be a 1D or 2D numpy array. The
            allowed dimensions will depend on the particular MIMO scheme
            implemented in a subclass.
        """
        self._channel = channel

    @property
    def Nt(self) -> int:
        """
        Get the number of transmit antennas

        Returns
        -------
        int
            The number of transmit antennas.
        """
        assert (self._channel is not None)
        return cast(int, self._channel.shape[1])

    @property
    def Nr(self) -> int:
        """
        Get the number of receive antennas

        Returns
        -------
        int
            The number of receive antennas.
        """
        assert (self._channel is not None)
        return cast(int, self._channel.shape[0])

    @staticmethod
    def _calc_precoder(channel: np.ndarray) -> np.ndarray:  # pragma: nocover
        """
        Calculate the linear precoder for the MIMO scheme, if there is any.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix.

        Returns
        -------
        W : np.ndarray
            The precoder that can be applied to the input data.
        """
        raise NotImplementedError(
            '_calc_precoder still needs to be implemented')

    @staticmethod
    def _calc_receive_filter(
            channel: np.ndarray,
            noise_var: Optional[float] = None
    ) -> np.ndarray:  # pragma: nocover
        """
        Calculate the receive filter for the MIMO scheme, if there is any.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix.
        noise_var : float
            The noise variance.

        Returns
        -------
        G_H : np.ndarray
            The receive_filter that can be applied to the input data.
        """
        raise NotImplementedError(
            '_calc_receive_filter still needs to be implemented')

    @abstractmethod
    def getNumberOfLayers(self) -> int:  # pragma: no cover
        """
        Get the number of layers of the MIMO scheme.

        Notes
        -----
        This method must be implemented in each subclass of `MimoBase`.

        Returns
        -------
        int
            The number of layers.
        """
        m = ('getNumberOfLayers still needs to '
             'be implemented in the {0} class')
        raise NotImplementedError(m.format(self.__class__.__name__))

    @staticmethod
    def _calcZeroForceFilter(channel: np.ndarray) -> np.ndarray:
        """
        Calculates the Zero-Force filter to cancel the inter-stream
        interference.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix.

        Returns
        -------
        W : np.ndarray
            The Zero-Forcing receive filter.

        Notes
        -----
        The Zero-Force filter basically corresponds to the pseudo-inverse
        of the channel matrix.
        """
        return np.linalg.pinv(channel)

    @staticmethod
    def _calcMMSEFilter(channel: np.ndarray, noise_var: float) -> np.ndarray:
        """
        Calculates the MMSE filter to cancel the inter-stream interference.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix.
        noise_var : float
            Noise variance.

        Returns
        -------
        W : np.ndarray
            The MMSE receive filter.
        """
        H = channel
        H_H = H.conj().T
        Nt = H.shape[1]
        W = np.linalg.solve(np.dot(H_H, H) + noise_var * np.eye(Nt), H_H)

        return W

    def calc_linear_SINRs(self, noise_var: float) -> np.ndarray:
        """
        Calculate the SINRs (in linear scale) of the multiple streams.

        Parameters
        ----------
        noise_var : float
            The noise variance.

        Returns
        -------
        sinrs : np.ndarray
            The sinrs (in linear scale) of the multiple streams.
        """
        W = self._calc_precoder(self._channel)
        G_H = self._calc_receive_filter(self._channel, noise_var)
        sinrs = calc_post_processing_SINRs(self._channel, W, G_H, noise_var)
        return sinrs

    def calc_SINRs(self, noise_var: float) -> np.ndarray:
        """
        Calculate the SINRs (in dB) of the multiple streams.

        Parameters
        ----------
        noise_var : float
            The noise variance.

        Returns
        -------
        SINRs : np.ndarray
            The SINRs (in dB) of the multiple streams.
        """
        return linear2dB(self.calc_linear_SINRs(noise_var))

    # noinspection PyPep8
    @abstractmethod
    def encode(self, transmit_data: np.ndarray) -> np.ndarray:  # pragma: no cover, pylint: disable=W0613
        """
        Method to encode the transmit data array to be transmitted using
        some MIMO scheme. This method must be implemented in a subclass.

        Parameters
        ----------
        transmit_data : np.ndarray
            The data to be transmitted.

        Returns
        -------
        encoded_data : np.ndarray
            The encoded `transmit_data`.
        """
        msg = 'encode still needs to be implemented in the {0} class'
        raise NotImplementedError(msg.format(self.__class__.__name__))

    # noinspection PyPep8
    @abstractmethod
    def decode(self, received_data: np.ndarray) -> np.ndarray:  # pragma: no cover, pylint: disable=W0613
        """
        Method to decode the transmit data array to be transmitted using
        some MIMO scheme. This method must be implemented in a subclass.

        Parameters
        ----------
        received_data : np.ndarray
            The received data.

        Returns
        -------
        decoded_data : np.ndarray
            The decoded data.
        """
        msg = 'decode still needs to be implemented in the {0} class'
        raise NotImplementedError(msg.format(self.__class__.__name__))


# noinspection PyAbstractClass
class MisoBase(MimoBase):  # pylint: disable=W0223
    """
    Base Class for MISO schemes.

    All subclasses must implement at least the following methods:

    - :meth:`MimoBase.encode`:
      The encode method must perform everything executed at the transmitter
      for that specific MIMO scheme. This also include the power division
      among the transmit antennas.
    - :meth:`MimoBase.decode`:
      Analogous to the encode method, the decode method must perform
      everything performed at the receiver.

    Other optional methods that might be useful implementing in subclasses
    are the `_calc_precoder` and `_calc_receive_filter` methods.

    Parameters
    ----------
    channel : np.ndarray
        MISO channel matrix/vector. MISO schemes are defined for
        scenarios with multiple transmit antennas and a single receive
        antenna. If `channel` is 2D, then the first dimension size must
        be equal to 1.
    """
    def __init__(self, channel: Optional[np.ndarray] = None):
        super().__init__(channel=None)
        if channel is not None:
            self.set_channel_matrix(channel)

    def set_channel_matrix(self, channel: np.ndarray) -> None:
        """
        Set the channel matrix.

        Parameters
        ----------
        channel : np.ndarray
            MISO channel vector. A MISO scheme is defined for the scenario
            with multiple transmit antennas and a single receive
            antenna. If channel is 2D then the first dimension size must be
            equal to 1.

        Returns
        -------
        None
        """
        # We will store the channel as a 2D numpy to be consistent with the
        # other MIMO classes
        if len(channel.shape) == 1:
            super().set_channel_matrix(channel[np.newaxis, :])
        else:
            Nr = channel.shape[0]
            if Nr != 1:
                raise ValueError("The MRT scheme is only defined for the "
                                 "scenario with a single receive antenna")
            # By calling the parent set_channel_matrix method the
            # self._W and self._G_H will be set to None
            super().set_channel_matrix(channel)

    def getNumberOfLayers(self) -> int:  # pragma: no cover
        """
        Get the number of layers of the MISO scheme.

        Because a MISO scheme only has one receive antenna then then number
        of layers is always equal to 1.

        Returns
        -------
        int
            The number of layers.
        """
        return 1


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

    Parameters
    ----------
    channel : np.ndarray
        MIMO channel matrix.
    """
    def __init__(self, channel: Optional[np.ndarray] = None):
        """
        Initialized the Blast object.

        If `channel` is not provided you need to call the
        `set_channel_matrix` method before calling the `decode` or the
        `getNumberOfLayers` methods.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix.
        """
        super().__init__(channel)
        self._noise_var: float = 0.0

    def set_channel_matrix(self, channel: np.ndarray) -> None:
        """
        Set the channel matrix.

        Parameters
        ----------
        channel : np.ndarray
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

        super().set_channel_matrix(channel)

    def getNumberOfLayers(self) -> int:
        """
        Get the number of layers of the Blast scheme.

        Returns
        -------
        Nl : int
            Number of layers of the MIMO scheme.
        """
        return self.Nt

    def set_noise_var(self, noise_var: Optional[float]) -> None:
        """
        Set the noise variance for the MMSE receive filter.

        If noise_var is non-positive then the Zero-Force filter will be
        used instead.

        Parameters
        ----------
        noise_var : float | None
            Noise variance for the MMSE filter (if `noise_var` is
            positive). If `noise_var` is 0.0 or None then the Zero-Forcing
            filter will be used.

        Returns
        -------
        None
        """
        if noise_var is None:
            self._noise_var = 0.0

        elif noise_var >= 0.0:
            self._noise_var = noise_var
        else:
            raise ValueError('Noise variance must be a non-negative value.')

    @staticmethod
    def _calc_precoder(channel: np.ndarray) -> np.ndarray:
        """
        Calculate the linear precoder for the BLAST scheme.

        The BLAST scheme simple send the data through the multiple streams
        without any particular precoding. Therefore, its linear precoder is
        equivalent to an identity matrix.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix.

        Returns
        -------
        W : np.ndarray
            The precoder that can be applied to the input data.
        """
        Nt = channel.shape[1]
        return np.eye(Nt) / math.sqrt(Nt)

    @staticmethod
    def _calc_receive_filter(channel: np.ndarray,
                             noise_var: Optional[float] = None) -> np.ndarray:
        """
        Calculate the receive filter for the MIMO scheme, if there is any.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix.
        noise_var : float
            The noise variance. If a value is provided then MMSE filter
            will be used. If it is not provided (or None is passes) then
            Zero Force filter will be used.

        Returns
        -------
        G_H : np.ndarray
            The receive_filter that can be applied to the input data.
        """
        Nt = channel.shape[1]

        if noise_var is None:  # pragma: nocover
            noise_var = 0.0

        if noise_var > 0:
            G_H = Blast._calcMMSEFilter(channel, noise_var)
        else:
            G_H = Blast._calcZeroForceFilter(channel)

        return G_H * math.sqrt(Nt)

    def encode(self, transmit_data: np.ndarray) -> np.ndarray:
        """
        Encode the transmit data array to be transmitted using the BLAST
        scheme.

        Parameters
        ----------
        transmit_data : np.ndarray
            A numpy array with a number of elements which is a multiple of
            the number of transmit antennas.

        Returns
        -------
        encoded_data : np.ndarray
            The encoded `transmit_data`.

        Raises
        ------
        ValueError
            If the number of elements in `transmit_data` is not multiple of
            the number of transmit antennas.
        """
        num_elements = transmit_data.size
        nStreams = self.getNumberOfLayers()
        if num_elements % nStreams != 0:
            # Note this is a single string
            msg = ("Input array number of elements must be a multiple of the"
                   " number of transmit antennas")
            raise ValueError(msg)

        encoded_data = (transmit_data.reshape(
            (nStreams, -1), order='F') / math.sqrt(self.Nt))
        return encoded_data

    def decode(self, received_data: np.ndarray) -> np.ndarray:
        """
        Decode the received data array.

        Parameters
        ----------
        received_data : np.ndarray
            Received data, which was encoded with the Blast scheme and
            corrupted by the channel `channel`.

        Returns
        -------
        decoded_data : np.ndarray
            The decoded data.
        """
        G_H = self._calc_receive_filter(self._channel, self._noise_var)
        decoded_data = G_H.dot(received_data).reshape(-1, order='F')
        return decoded_data


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MRT xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MRT(MisoBase):
    """
    MIMO class for the MRT scheme.

    The number of streams for the MRT scheme is always equal to one, but it
    still employs multiple transmit antennas.

    If `channel` is not provided you need to call the `set_channel_matrix`
    method before calling the other methods.

    Parameters
    ----------
    channel : np.ndarray
        MISO channel vector. It must be a 1D numpy array, where the
        number of receive antennas is assumed to be equal to 1.
    """
    def __init__(self, channel: Optional[np.ndarray] = None):
        super().__init__(channel)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _calc_precoder(channel: np.ndarray) -> np.ndarray:
        """
        Calculate the linear precoder for the MRT scheme.

        The MRT scheme corresponds to multiplying the symbol from each
        transmit antenna with a complex number corresponding to the inverse
        of the phase of the channel so as to ensure that the signals add
        constructively at the receiver. This also means that the MRT scheme
        only be applied to scenarios with a single receive antenna.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix with dimension (1, Nt).

        Returns
        -------
        W : np.ndarray
            The precoder that can be applied to the input data.
        """
        Nt = channel.shape[1]
        W = np.exp(-1j * np.angle(channel)).T / math.sqrt(Nt)
        return W

    @staticmethod
    def _calc_receive_filter(channel: np.ndarray,
                             noise_var: Optional[float] = None) -> np.ndarray:
        """
        Calculate the receive filter for the MRT scheme.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix.
        noise_var : float
            The noise variance.

        Returns
        -------
        G_H : np.ndarray
            The receive_filter that can be applied to the input data.
        """
        Nt = channel.shape[1]
        G_H = math.sqrt(Nt) / np.sum(np.abs(channel))
        return G_H

    def encode(self, transmit_data: np.ndarray) -> np.ndarray:
        """
        Encode the transmit data array to be transmitted using the MRT
        scheme.

        The MRT scheme corresponds to multiplying the symbol from each
        transmit antenna with a complex number corresponding to the
        inverse of the phase of the channel so as to ensure that the
        signals add constructively at the receiver. This also means that
        the MRT scheme only be applied to scenarios with a single receive
        antenna.

        Parameters
        ----------
        transmit_data : np.ndarray
            A numpy array with the data to be transmitted.

        Returns
        -------
        encoded_data : np.ndarray
            The encoded `transmit_data`.
        """
        # Add an extra first dimension so that broadcast does the right
        # thing later
        x = transmit_data[np.newaxis, :]
        W = self._calc_precoder(self._channel)

        # Element-wise multiplication (use broadcast)
        encoded_data = (W * x)
        return encoded_data

    def decode(self, received_data: np.ndarray) -> np.ndarray:
        """
        Decode the received data array.

        Parameters
        ----------
        received_data : np.ndarray
            Received data, which was encoded with the MRT scheme and
            corrupted by the channel `channel`.

        Returns
        -------
        decoded_data : np.ndarray
            The decoded data.
        """
        G_H = self._calc_receive_filter(self._channel)
        decoded_data = G_H * received_data
        decoded_data.shape = decoded_data.size

        return decoded_data


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx MRC Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class MRC(Blast):
    """
    MIMO class for the MRC scheme.

    The receive filter used will depend on the noise variance (see the
    :meth:`.set_noise_var` method). If the noise variance is positive the
    MMSE filter will be used, otherwise noise variance will be ignored and
    the Zero-Forcing filter will be used.

    The receive filter in the `Blast` class already does the maximum ratio
    combining. Therefore, this MRC class simply inherits from the Blast
    class and only exists for completion.

    Parameters
    ----------
    channel : np.ndarray
        MIMO channel matrix.
    """
    def __init__(self, channel: Optional[np.ndarray] = None):
        super().__init__(channel)

    def set_channel_matrix(self, channel: np.ndarray) -> None:
        """
        Set the channel matrix.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix. The MRC MIMO scheme is defined for the
            scenario with multiple receive antennas and a single receive
            antenna. If channel is 1D assume that the number of transmit
            antennas is equal to 1.
        """
        # We will store the channel as a 2D numpy to be consistent with the
        # other MIMO classes
        if len(channel.shape) == 1:
            super().set_channel_matrix(channel[:, np.newaxis])
        else:
            super().set_channel_matrix(channel)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx SVD MIMO xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class SVDMimo(Blast):
    """
    MIMO class for the SVD MIMO scheme.

    Parameters
    ----------
    channel : np.ndarray
        MIMO channel matrix.
    """
    def __init__(self, channel: Optional[np.ndarray] = None):
        super().__init__(channel)

    @staticmethod
    def _calc_precoder(channel: np.ndarray) -> np.ndarray:
        """
        Calculate the linear precoder for the SVD MIMO scheme.

        The SVD MIMO scheme employs as precoder the right singular matrix
        from SVD (Singular Value Decomposition) of the channel.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix with dimension (1, Nt).

        Returns
        -------
        W : np.ndarray
            The precoder that can be applied to the input data.
        """
        Nt = channel.shape[1]
        _, _, V_H = np.linalg.svd(channel)
        # The precoder is the 'V' matrix from the SVD decomposition of the
        # channel. Notice that we also need to divide by sqrt(Nt) to make
        # sure 'W' has a unitary norm.
        W = V_H.conj().T / math.sqrt(Nt)
        return W

    @staticmethod
    def _calc_receive_filter(channel: np.ndarray,
                             noise_var: Optional[float] = None) -> np.ndarray:
        """
        Calculate the receive filter for the SVD MIMO scheme.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix.
        noise_var : float
            The noise variance.

        Returns
        -------
        G_H : np.ndarray
            The receive_filter that can be applied to the input data.
        """
        Nt = channel.shape[1]
        U, S, _ = np.linalg.svd(channel)
        G_H = np.diag(1. / S).dot(U.conj().T) * math.sqrt(Nt)
        return G_H

    def encode(self, transmit_data: np.ndarray) -> np.ndarray:
        """
        Encode the transmit data array to be transmitted using the SVD MIMO
        scheme.

        The SVD MIMO scheme corresponds to using the 'U' and 'V' matrices
        from the SVD decomposition of the channel as the precoder and
        receive filter.

        Parameters
        ----------
        transmit_data : np.ndarray
            A numpy array with the data to be transmitted.

        Returns
        -------
        encoded_data : np.ndarray
            The encoded `transmit_data`.
        """
        num_elements = transmit_data.size
        if num_elements % self.Nt != 0:
            msg = ("Input array number of elements must be a multiple of the"
                   " number of transmit antennas")
            raise ValueError(msg)

        X = transmit_data.reshape(self.Nt, -1)

        W = self._calc_precoder(self._channel)
        encoded_data = W.dot(X)

        return encoded_data

    def decode(self, received_data: np.ndarray) -> np.ndarray:
        """
        Perform the decoding of the received_data for the SVD MIMO
        scheme with the channel `channel`.

        Parameters
        ----------
        received_data : np.ndarray
            Received data, which was encoded with the Alamouti scheme and
            corrupted by the channel `channel`.

        Returns
        -------
        decoded_data : np.ndarray
            The decoded data.
        """
        G_H = self._calc_receive_filter(self._channel)
        decoded_data = G_H.dot(received_data)

        # Return the decoded data as a 1D numpy array
        return decoded_data.reshape(decoded_data.size, )


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx GMD MIMO xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class GMDMimo(Blast):
    """
    MIMO class for the GMD based MIMO scheme.

    Parameters
    ----------
    channel : np.ndarray
        MIMO channel matrix.
    """
    def __init__(self, channel: Optional[np.ndarray] = None):
        super().__init__(channel)

    @staticmethod
    def _calc_precoder(channel: np.ndarray) -> np.ndarray:
        """
        Calculate the linear precoder for the GMD scheme.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix with dimension (1, Nt).

        Returns
        -------
        W : np.ndarray
            The precoder that can be applied to the input data.
        """
        Nt = channel.shape[1]
        # The encode method will precode the transmit_data using the
        # matrix 'P' obtained from the gmd.
        U, S, V_H = np.linalg.svd(channel)
        _, _, P = gmd(U, S, V_H)
        W = P / math.sqrt(Nt)
        return W

    @staticmethod
    def _calc_receive_filter(channel: np.ndarray,
                             noise_var: Optional[float] = None) -> np.ndarray:
        """
        Calculate the receive filter for the MRT scheme.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix.
        noise_var : float
            The noise variance.

        Returns
        -------
        G_H : np.ndarray
            The receive_filter that can be applied to the input data.
        """
        U, S, V_H = np.linalg.svd(channel)
        Q, R, _ = gmd(U, S, V_H)
        channel_eq = Q.dot(R)

        # Use the _calc_receive_filter method from the base class (Blast)
        G_H = Blast._calc_receive_filter(channel_eq, noise_var)
        return G_H

    def encode(self, transmit_data: np.ndarray) -> np.ndarray:
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
        transmit_data : np.ndarray
            A numpy array with the data to be transmitted.

        Returns
        -------
        encoded_data : np.ndarray
            The encoded `transmit_data`.
        """
        num_elements = transmit_data.size
        if num_elements % self.Nt != 0:
            msg = ("Input array number of elements must be a multiple of the"
                   " number of transmit antennas")
            raise ValueError(msg)

        W = self._calc_precoder(self._channel)
        X = transmit_data.reshape(self.Nt, -1)
        encoded_data = W.dot(X)

        return encoded_data

    def decode(self, received_data: np.ndarray) -> np.ndarray:
        """
        Perform the decoding of the received_data for the GMD MIMO.

        Parameters
        ----------
        received_data : np.ndarray
            Received data, which was encoded with the Alamouti scheme and
            corrupted by the channel `channel`.

        Returns
        -------
        decoded_data : np.ndarray
            The decoded data.
        """
        G_H = self._calc_receive_filter(self._channel, self._noise_var)
        decoded_data = G_H.dot(received_data).reshape(-1)
        return decoded_data


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxx Alamouti Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Alamouti(MimoBase):
    """
    MIMO class for the Alamouti scheme.

    Parameters
    ----------
    channel : np.ndarray
        MIMO channel matrix.
    """
    def __init__(self, channel: Optional[np.ndarray] = None):
        super().__init__(channel)

    @staticmethod
    def _calc_precoder(channel: np.ndarray) -> np.ndarray:  # pragma: nocover
        """
        Not defined.

        There is no linear precoder for the Almost scheme and thus an
        exception is called if this method is ever called.
        """
        raise RuntimeError("Alamouti scheme has no linear precoder")

    @staticmethod
    def _calc_receive_filter(
            channel: np.ndarray,
            noise_var: Optional[float] = None
    ) -> np.ndarray:  # pragma: nocover
        """
        Not defined.

        There is no linear receive filter for the Alamouti scheme that can
        be directly applied to the received data. Thus, an exception is
        called if this method is ever called.
        """
        raise RuntimeError("Alamouti scheme has no linear receive filter that"
                           " can be directly applied to the received data")

    def set_channel_matrix(self, channel: np.ndarray) -> None:
        """
        Set the channel matrix.

        Parameters
        ----------
        channel : np.ndarray
            MIMO channel matrix.

        Returns
        -------
        None
        """
        if len(channel.shape) == 1:
            super().set_channel_matrix(channel[np.newaxis, :])
        else:
            _, Nt = channel.shape
            if Nt != 2:
                msg = ("The number of transmit antennas must be equal to "
                       "2 for the {0} scheme").format(self.__class__.__name__)
                raise ValueError(msg)
            super().set_channel_matrix(channel)

    def getNumberOfLayers(self) -> int:
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

    def calc_linear_SINRs(self, noise_var: float) -> np.ndarray:
        """
        Calculate the SINRs (in linear scale) of the multiple streams.

        Parameters
        ----------
        noise_var : float
            The noise variance.

        Returns
        -------
        sinrs : np.ndarray
            The sinrs (in linear scale) of the multiple streams.
        """
        # The linear post-processing SINR for the Alamouti scheme is
        # given by
        # \[\frac{\Vert \mtH \Vert_F^2}{2 \sigma_N} \]
        sinr = np.linalg.norm(self._channel, 'fro')**2 / noise_var
        return sinr

    @staticmethod
    def _encode(transmit_data: np.ndarray) -> np.ndarray:
        """
        Perform the Alamouti encoding, but without dividing the power among
        the transmit antennas.

        The idea is that the encode method will call _encode and perform
        the power division. This separation allows better code reuse.

        Parameters
        ----------
        transmit_data : np.ndarray
            Data to be encoded by the Alamouti scheme.

        Returns
        -------
        encoded_data : np.ndarray
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

    def encode(self, transmit_data: np.ndarray) -> np.ndarray:
        """
        Perform the Alamouti encoding.

        Parameters
        ----------
        transmit_data : np.ndarray
            Data to be encoded by the Alamouti scheme.

        Returns
        -------
        encoded_data : np.ndarray
            The encoded `transmit_data`.
        """
        return self._encode(transmit_data) / math.sqrt(2)

    @staticmethod
    def _decode(received_data: np.ndarray, channel: np.ndarray) -> np.ndarray:
        """
        Perform the decoding of the received_data for the Alamouti
        scheme with the channel `channel`, but does not compensate for
        the power division among transmit antennas.

        The idea is that the decode method will call _decode and perform
        the power compensation. This separation allows better code reuse.

        Parameters
        ----------
        received_data : np.ndarray
            Received data, which was encoded with the Alamouti scheme and
            corrupted by the channel `channel`.
        channel : np.ndarray
            MIMO channel matrix.

        Returns
        -------
        decoded_data : np.ndarray
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
                np.dot(h1, received_data[:, 2 * i + 1].conjugate()))
            decoded_data[2 * i + 1] = (
                np.dot(h1_conj, received_data[:, 2 * i]) +
                np.dot(minus_h0, received_data[:, 2 * i + 1].conjugate()))

        # The Alamouti code gives a gain of the square of the Frobenius
        # norm of the channel. We need to compensate that gain.
        decoded_data /= np.linalg.norm(channel, 'fro')**2
        return decoded_data

    def decode(self, received_data: np.ndarray) -> np.ndarray:
        """
        Perform the decoding of the received_data for the Alamouti
        scheme with the channel `channel`.

        Parameters
        ----------
        received_data : np.ndarray
            Received data, which was encoded with the Alamouti scheme and
            corrupted by the channel `channel`.

        Returns
        -------
        decoded_data : np.ndarray
            The decoded data.
        """
        return self._decode(received_data, self._channel) * math.sqrt(2)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
