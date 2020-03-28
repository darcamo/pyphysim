#!/usr/bin/env python
"""
Module implementing OFDM modulation and demodulation.
"""

import math
from typing import Optional, Tuple

import numpy as np

from ..channels import fading

__all__ = ['OFDM', 'OfdmOneTapEqualizer']


class OFDM:
    """
    OFDM class.
    """
    def __init__(self,
                 fft_size: int,
                 cp_size: int,
                 num_used_subcarriers: Optional[int] = None) -> None:
        """
        Initialize the OFDM object.

        Parameters
        ----------
        fft_size : int
            Size of the FFT and IFFT used by the OFDM class.
        cp_size : int
            Size of the cyclic prefix (in samples).
        num_used_subcarriers : int, optional
            Number of used subcarriers. Must be greater than or equal to 2
            and lower than or equal to fft_size. If not provided, fft_size
            will be used

        Returns
        -------
        OFDM

        Raises
        ------
        ValueError
            If the any of the parameters are invalid."""
        self.fft_size: int = 0
        self.cp_size: int = 0
        self.num_used_subcarriers: int = 0

        self.set_parameters(fft_size, cp_size, num_used_subcarriers)

    def set_parameters(self,
                       fft_size: int,
                       cp_size: int,
                       num_used_subcarriers: Optional[int] = None) -> None:
        """
        Set the OFDM parameters.

        Parameters
        ----------
        fft_size : int
            Size of the FFT and IFFT used by the OFDM class.
        cp_size : int
            Size of the cyclic prefix (in samples).
        num_used_subcarriers : int, optional
            Number of used subcarriers. Must be greater than or equal to 2
            and lower than or equal to fft_size. If not provided, fft_size
            will be used

        Raises
        ------
        ValueError
            If the any of the parameters are invalid.
        """
        if (cp_size < 0) or cp_size > fft_size:
            msg = ("cp_size must be nonnegative and cannot be greater "
                   "than fft_size")
            raise ValueError(msg)

        if num_used_subcarriers is None:
            num_used_subcarriers = fft_size

        if num_used_subcarriers > fft_size:
            msg = ("Number of used subcarriers cannot be greater than the "
                   "fft_size")
            raise ValueError(msg)

        if (num_used_subcarriers % 2 != 0) or (num_used_subcarriers < 2):
            msg = "Number of used subcarriers must be a multiple of 2"
            raise ValueError(msg)

        self.fft_size = fft_size
        self.cp_size = cp_size
        self.num_used_subcarriers = num_used_subcarriers

    def _calc_zeropad(self, input_data_size: int) -> Tuple[int, int]:
        """
        Calculates the number of zeros that must be added to the input data
        to make it a multiple of the OFDM size.

        The number of zeros that must be added to the input data is
        returned along with the number of OFDM symbols that will be
        generated.

        Parameters
        ----------
        input_data_size : int
            Size the the data that will be modulated by the OFDM object.

        Returns
        -------
        (zeropad, num_ofdm_symbols) : tuple[int,int]
            A tuple with zeropad and num_ofdm_symbols. Zeropad is the
            number of zeros added to the input data to make the total
            number of elements a multiple of the number of used
            subcarriers. Num_ofdm_symbols is the number of OFDM symbols
            required to transmit `input_data_size` symbols.
        """
        num_ofdm_symbols = (int(
            np.ceil(float(input_data_size) / self.num_used_subcarriers)))
        zeropad = (self.num_used_subcarriers * num_ofdm_symbols -
                   input_data_size)
        return zeropad, num_ofdm_symbols

    def _get_subcarrier_numbers(self) -> np.ndarray:
        """
        Get the indexes of all subcarriers, including the negative, the DC
        and the positive subcarriers.

        Note that these indexes are not suitable for indexing in
        python. They are the actual indexes of the subcarriers in an OFDM
        symbol. For instance, an OFDM symbol with 16 subcarriers will have
        indexes from -8 to 7. However, due to the way the fft is
        implemented in numpy the indexes here are actually from 0 to 7
        followed by -8 to -1.

        Returns
        -------
        np.ndarray
            Numbers of all subcarriers, including the negative, the DC and
            the positive subcarriers

        Examples
        --------
        >> ofdm_obj = OFDM(16, 4, 16)
        >> ofdm_obj._get_subcarrier_numbers()
        array([ 0,  1,  2,  3,  4,  5,  6,  7, \
                -8, -7, -6, -5, -4, -3, -2, -1])
        """
        indexes_regular_order = (np.arange(self.fft_size) - self.fft_size // 2)
        return np.fft.fftshift(indexes_regular_order)

    def _get_used_subcarrier_numbers(self) -> np.ndarray:
        """
        Get the subcarrier indexes of the actually used subcarriers.

        Note that these indexes are not suitable for indexing in
        python. They are the actual indexes of the subcarriers in an OFDM
        symbol. See the documentation of the _get_subcarrier_numbers
        function.

        Returns
        -------
        np.ndarray
            Number of the actually used subcarriers.

        Examples
        --------
        >> ofdm_obj = OFDM(16, 4, 10)
        >> ofdm_obj._get_used_subcarrier_numbers()
        array([ 1,  2,  3,  4,  5, -5, -4, -3, -2, -1])
        >> ofdm_obj = OFDM(16, 4, 14)
        >> ofdm_obj._get_used_subcarrier_numbers()
        array([ 1,  2,  3,  4,  5,  6,  7, -7, -6, -5, -4, -3, -2, -1])
        """
        if self.num_used_subcarriers == self.fft_size:
            return self._get_subcarrier_numbers()

        # Calculates half the number of subcarriers. This is only valid if
        # num_used_subcarriers is a multiple of 2.
        half_used_sc = self.num_used_subcarriers // 2

        first_half = np.r_[1:half_used_sc + 1]
        second_half = np.r_[-half_used_sc:0]
        indexes = np.hstack([first_half, second_half])
        return indexes

    def get_used_subcarrier_indexes(self) -> np.ndarray:
        """
        Get the subcarrier indexes of the subcarriers actually used in a
        way suitable for python indexing (going from 0 to fft_size-1).

        Returns
        -------
        indexes : np.ndarray
            Subcarrier indexes of the subcarriers actually used in a way
            suitable for python indexing.

        Notes
        -----
        This is the function actually used in the modulate function.

        Examples
        --------
        Consider the example below where we have 16 subcarriers and only
        10 subcarriers are used. The lower and higher subcarrier as well
        as the DC subcarrier will not be used. The index of the used
        subcarriers should go then from 11 to 15 (5 subcarriers),
        skip subcarrier 0, and then go from 1 to 5 (the other 5
        subcarriers).

        >>> ofdm_obj = OFDM(16, 4, 10)
        >>> ofdm_obj.get_used_subcarrier_indexes()
        array([11, 12, 13, 14, 15,  1,  2,  3,  4,  5])
        >>> ofdm_obj = OFDM(16,4,14)
        >>> ofdm_obj.get_used_subcarrier_indexes()
        array([ 9, 10, 11, 12, 13, 14, 15,  1,  2,  3,  4,  5,  6,  7])
        """
        numbers = self._get_used_subcarrier_numbers()
        half_used = self.num_used_subcarriers // 2

        indexes_proper = np.hstack(
            [self.fft_size + numbers[half_used:], numbers[0:half_used]])
        return indexes_proper

    def _prepare_input_signal(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Prepare the input signal to be passed to the IFFT in the modulate
        function.

        The input signal must be prepared before it is passed to the IFFT
        in the OFDM modulate function.

        - First, zeros must be added so that the input signal size is
          multiple of the number of used subcarriers.
        - After that the input signal must be allocated to subcarriers in
          the center of the spectrum (except for the DC component). That
          is, zeros will be allocated to the lower and higher subcarriers
          such that only num_used_subcarriers are used from fft_size
          subcarriers.

        This preparation is performed by the _prepare_input_signal
        function.

        Parameters
        ----------
        input_signal : np.ndarray
            Input signal that must be modulated by the OFDM modulate
            function.

        Returns
        -------
        input_ifft : np.ndarray
            Signal suitable to be passed to the IFFT function to actually
            perform the OFDM modulation.

        See also
        --------
        _prepare_decoded_signal
        """
        # Number of data symbols passed to the modulate function
        num_symbs = input_signal.size
        # Calculates how many zeros need to be added as well as the number
        # of equivalent OFDM symbols.
        num_ofdm_symbols = self._calc_zeropad(num_symbs)[1]

        # Finally add the zeros to the input data
        input_signal = np.hstack([
            input_signal,
            np.zeros(self.num_used_subcarriers * num_ofdm_symbols - num_symbs)
        ])

        # Change the shape of the input data. Each row will be modulated as
        # one OFDM symbol.
        input_signal.shape = (num_ofdm_symbols, self.num_used_subcarriers)

        input_ifft = np.zeros([num_ofdm_symbols, self.fft_size], dtype=complex)
        input_ifft[:, self.get_used_subcarrier_indexes()] \
            = input_signal[:, :]

        return input_ifft

    def _prepare_decoded_signal(self,
                                decoded_signal: np.ndarray) -> np.ndarray:
        """
        Prepare the decoded signal that was processed by the FFT in the
        demodulate function.

        This is equivalent of reversing the indexing that was done by the
        _prepare_input_signal method.

        Parameters
        ----------
        decoded_signal : np.ndarray
            Signal that was decoded by the FFT in the OFDM demodulate
            method.

        Returns
        -------
        demodulated_samples : np.ndarray
            Demodulated samples of the symbols that were modulated by the
            OFDM object (for instance the PSK or M-QAM symbols passed to
            OFDM).

        Notes
        -----
        This method should be called AFTER the Cyclic Prefix was removed
        and the FFT was performed.

        Also, because the number of zeropad was not saved, then
        _prepare_decoded_signal has no way to remove them.

        See also
        --------
        _prepare_input_signal

        """
        return decoded_signal[:, self.get_used_subcarrier_indexes()].flatten()

    def _add_CP(self, input_data: np.ndarray) -> np.ndarray:
        """
        Add the Cyclic prefix to the input data.

        Parameters
        ----------
        input_data : np.ndarray
            OFDM modulated data (after the IFFT). This must be a 2D numpy
            array with shape (Number of OFDM symbols, IFFT size).

        Returns
        -------
        output : np.ndarray
            The `input_data` with the cyclic prefix added. The shape of the
            output is (Number of OFDM symbols, IFFT size + CP Size).

        """
        if self.cp_size != 0:
            output = np.hstack([input_data[:, -self.cp_size:], input_data])
        else:
            output = input_data
        return output

    def _remove_CP(self, received_data: np.ndarray) -> np.ndarray:
        """
        Remove the Cyclic prefix of the received data.

        Parameters
        ----------
        received_data : np.ndarray
            Data that must be demodulated by the OFDM object.

        Returns
        -------
        output : np.ndarray
            Received data without the Cyclic prefix.

        Notes
        -----
        The _remove_CP method will also change the shape so that it is
        suitable to be passed to the FFT function.

        """
        num_ofdm_symbols = (received_data.size //
                            (self.fft_size + self.cp_size))
        received_data.shape = (num_ofdm_symbols, self.fft_size + self.cp_size)
        received_data_no_CP = received_data[:, self.cp_size:]

        return received_data_no_CP

    def _calculate_power_scale(self) -> float:
        """
        Calculate the power scale that needs to be applied in the
        modulator and removed in the demodulate methods.

        The power is applied in the modulator method so that the total
        power of the OFDM samples is similar to the total power of the
        symbols modulated by OFDM.

        Note that this total power is shared among useful samples and the
        cyclic prefix in one OFDM symbol. Therefore, the larger the cyclic
        prefix size the lower is this power scale to account energy loss
        due to sending the cyclic prefix.

        Returns
        -------
        power_scale : float
            The calculated power scale. You should take the square root of
            this before multiplying by the samples.
        """
        power_scale = (float(self.fft_size) ** 2) / \
                      (float(self.num_used_subcarriers) + self.cp_size)
        return power_scale

    def modulate(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Perform the OFDM modulation of the input_signal.

        Parameters
        ----------
        input_signal : np.ndarray
            Input signal that must be modulated by the OFDM modulate
            function.

        Returns
        -------
        output : np.ndarray
            An array with the samples of the modulated OFDM symbols.

        """
        # _prepare_input_signal will perform any zero padding needed as
        # well as deactivating the DC subcarrier and the guard subcarriers
        # when the number of used subcarriers is lower then the IFFT size.
        # Notice that the output of _prepare_input_signal will be a
        # bi-dimensional array, where each row has the input data for a
        # single OFDM symbol.
        input_ifft = self._prepare_input_signal(input_signal)

        # Now we calculate the ifft for the second axis. That is equivalent
        # to calculate the ifft separately for each row in the input_ifft
        # variable.
        output_ifft = (math.sqrt(self._calculate_power_scale()) *
                       np.fft.ifft(input_ifft, self.fft_size, 1))
        assert isinstance(output_ifft, np.ndarray)

        # Add the Cyclic prefix
        modulated_ofdm = self._add_CP(output_ifft)

        # Change the shape to one dimensional array and return that array
        return modulated_ofdm.flatten()

    def demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        Perform the OFDM demodulation of the received_signal.

        Parameters
        ----------
        received_signal : np.ndarray
            An array with the samples of the received OFDM symbols.

        Returns
        -------
        demodulated_data : np.ndarray
            Demodulated symbols.
        """
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # - Remove the Cyclic Prefix -> the output will have a shape of
        # num_ofdm_symbols x fft_size
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        received_signal_no_CP = self._remove_CP(received_signal)

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # - Call The FFT
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now we calculate the FFT for the second axis. That is equivalent
        # to calculate the fft separately for each row in the
        # received_signal variable.
        output_fft = (np.fft.fft(received_signal_no_CP, self.fft_size, 1) /
                      math.sqrt(self._calculate_power_scale()))
        assert isinstance(output_fft, np.ndarray)

        # - Call the `_prepare_decoded_signal` method to get the data only
        # from the useful subcarriers
        decoded_symbols = self._prepare_decoded_signal(output_fft)

        # Return the decoded data
        return decoded_symbols


class OfdmOneTapEqualizer:
    """
    The OfdmOneTapEqualizer class performs the one-tap equalization often
    required in OFDM transmissions to compensate the effect of the channel
    at each subcarrier.

    Parameters
    ----------
    ofdm_obj : OFDM
        The OFDM object used to modulate/demodulate the data.
    """
    def __init__(self, ofdm_obj: OFDM):
        self._ofdm_obj = ofdm_obj

    def _equalize_data(self, data_reshaped: np.ndarray,
                       mean_freq_response: np.ndarray) -> np.ndarray:
        """
        Perform the one-tap equalization and return `data` after the
        channel compensation.

        Parameters
        ----------
        data_reshaped : np.ndarray
            The data to be equalized. If must be a 2D numpy array, where
            different rows correspond to different OFDM symbols and the
            different columns correspond to the USED
            subcarriers.
            Dimension: `num OFDM symbols x num Used subcarriers`
        mean_freq_response : np.ndarray
            The frequency response for each OFDM symbol.
            Dimension: `num OFDM symbols x FFT size`

        Returns
        -------
        np.ndarray
            The received `data` after the one-tap equalization to
            compensate the channel effect.
            Dimension: `num OFDM symbols x num Used subcarriers`
        """
        used_subcarriers_idx = self._ofdm_obj.get_used_subcarrier_indexes()

        equalized_ofdm_demodulated_data = \
            data_reshaped / mean_freq_response[:, used_subcarriers_idx]

        return equalized_ofdm_demodulated_data

    def equalize_data(
            self, data: np.ndarray,
            impulse_response: fading.TdlImpulseResponse) -> np.ndarray:
        """
        Perform the one-tap equalization and return `data` after the
        channel compensation.

        Parameters
        ----------
        data : np.ndarray
            The data to be equalized.
        impulse_response : fading.TdlImpulseResponse
            The impulse response of the channel.

        Returns
        -------
        np.ndarray
            The received `data` after the one-tap equalization to
            compensate the channel effect.
        """
        fft_size = self._ofdm_obj.fft_size
        num_used_subcarriers = self._ofdm_obj.num_used_subcarriers
        num_ofdm_symbols = data.size // num_used_subcarriers

        data_reshaped = np.reshape(data, (-1, num_used_subcarriers))

        freq_response = impulse_response.get_freq_response(fft_size)

        # Reshape and get the average frequency response for all samples in
        # each OFDM symbol
        freq_response = np.reshape(freq_response,
                                   (fft_size, num_ofdm_symbols, -1))
        mean_freq_response = np.mean(freq_response, axis=2)
        mean_freq_response = mean_freq_response.T

        equalized_ofdm_demodulated_data = self._equalize_data(
            data_reshaped, mean_freq_response)
        return equalized_ofdm_demodulated_data.flatten()
