#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module implementing OFDM modulation and demodulation."""

import numpy as np
import math

__all__ = ['OFDM']


class OFDM(object):
    """OFDM class.
    """
    def __init__(self, fft_size, cp_size, num_used_subcarriers=None):
        """Initializates the OFDM object.

        Parameters
        ----------
        fft_size : int
            Size of the FFT and IFFT used by the OFDM class.
        cp_size : int
            Size of the cyclic prefix (in samples).
        num_used_subcarriers : int, optional (default to fft_size)
            Number of used subcarriers. Must be greater than or equal to 2
            and lower than or equal to fft_size. If not provided, fft_size
            will be used

        Raises
        ------
        ValueError
            If the any of the parameters are invalid.
        """
        self.fft_size = 0
        self.cp_size = 0
        self.num_used_subcarriers = 0

        self.set_parameters(fft_size, cp_size, num_used_subcarriers)

    def set_parameters(self, fft_size, cp_size, num_used_subcarriers=None):
        """Set the OFDM parameters.

        Parameters
        ----------
        fft_size : int
            Size of the FFT and IFFT used by the OFDM class.
        cp_size : int
            Size of the cyclic prefix (in samples).
        num_used_subcarriers : int, optional (default to fft_size)
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

    def _calc_zeropad(self, input_data_size):
        """Calculates the number of zeros that must be added to the input
        data to make it a multiple of the OFDM size.

        The number of zeros that must be added to the input data is
        returned along with the number of OFDM symbols that will be
        generated.

        Parameters
        ----------
        input_data_size : int
            Size the the data that will be modulated by the OFDM object.

        Returns
        -------
        (zeropad, num_ofdm_symbols) : tuple
            A tuple with zeropad and num_ofdm_symbols. Zeropad is the
            number of zeros added to the input data to make the total
            number of elements a multiple of the number of used
            subcarriers. Num_ofdm_symbols is the number of OFDM symbols
            required to transmit `input_data_size` symbols.

        """
        num_ofdm_symbols = (int(np.ceil(float(input_data_size)
                                        / self.num_used_subcarriers)))
        zeropad = (self.num_used_subcarriers * num_ofdm_symbols
                   - input_data_size)
        return (zeropad, num_ofdm_symbols)

    def get_subcarrier_indexes(self):
        """Get the indexes of all subcarriers, including the negative, the
        DC and the positive subcarriers.

        Note that these indexes are not suitable for indexing in
        python. They are the actual indexes of the subcarriers in an OFDM
        symbol. For instance, an OFDM symbol with 16 subcarriers will have
        indexes from -8 to 7. However, due to the way the fft is
        implemented in numpy the indexes here are actually from 0 to 7
        followed by -8 to -1.

        Returns
        -------
        indexes : 1D numpy array
            Indexes of all subcarriers, including the negative, the DC and
            the positive subcarriers

        Examples
        --------
        >>> ofdm = OFDM(16,4,16)
        >>> ofdm.get_subcarrier_indexes()
        array([ 0,  1,  2,  3,  4,  5,  6,  7, -8, -7, -6, -5, -4, -3, -2, -1])
        """
        # first_half = np.r_[0:self.fft_size // 2]
        # second_half = np.r_[-self.fft_size // 2:0]
        # indexes = np.hstack([first_half, second_half])

        indexes_regular_order = np.r_[0:self.fft_size] - self.fft_size // 2
        return np.fft.fftshift(indexes_regular_order)

    def _get_used_subcarrier_indexes(self):
        """Get the subcarrier indexes of the actually used subcarriers.

        Note that these indexes are not suitable for indexing in
        python. They are the actual indexes of the subcarriers in an OFDM
        symbol. See the documentation of the get_subcarrier_indexes
        function.

        Returns
        -------
        indexes : 1D numpy array
            Indexes of the actually used subcarriers.

        Examples
        --------
        >>> ofdm = OFDM(16,4,10)
        >>> ofdm._get_used_subcarrier_indexes()
        array([ 1,  2,  3,  4,  5, -5, -4, -3, -2, -1])
        >>> ofdm = OFDM(16,4,14)
        >>> ofdm._get_used_subcarrier_indexes()
        array([ 1,  2,  3,  4,  5,  6,  7, -7, -6, -5, -4, -3, -2, -1])
        """
        if self.num_used_subcarriers == self.fft_size:
            return self.get_subcarrier_indexes()

        # Calculates half the number of subcarriers. This is only valid if
        # num_used_subcarriers is a multiple of 2.
        half_used_sc = self.num_used_subcarriers // 2

        first_half = np.r_[1:half_used_sc + 1]
        second_half = np.r_[-half_used_sc:0]
        indexes = np.hstack([first_half, second_half])
        return indexes

    def _get_used_subcarrier_indexes_proper(self, ):
        """Get the subcarrier indexes of the subcarriers actually used, but
        in a way suitable for python indexing (going from 0 to fft_size-1).

        Returns
        -------
        indexes : 1D numpy array
            Subcarrier indexes of the subcarriers actually used in a way
            suitable for python indexing.

        Notes
        -----
        This is the function actually used in the modulate function.

        Examples
        --------
        >>> ofdm = OFDM(16,4,10)
        >>> ofdm._get_used_subcarrier_indexes_proper()
        array([11, 12, 13, 14, 15,  1,  2,  3,  4,  5])
        >>> ofdm = OFDM(16,4,14)
        >>> ofdm._get_used_subcarrier_indexes_proper()
        array([ 9, 10, 11, 12, 13, 14, 15,  1,  2,  3,  4,  5,  6,  7])

        """
        indexes = self._get_used_subcarrier_indexes()
        half_used = self.num_used_subcarriers // 2

        indexes_proper = np.hstack([
            self.fft_size + indexes[half_used:],
            indexes[0:half_used]])
        return indexes_proper

    def _prepare_input_signal(self, input_signal):
        """Prepare the input signal to be passed to the IFFT in the
        modulate function.

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
        input_data : 1D numpy array
            Input signal that must be modulated by the OFDM modulate
            function.

        Returns
        -------
        input_ifft : 1D numpy array
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
        input_signal = np.hstack(
            [input_signal, np.zeros(
                self.num_used_subcarriers * num_ofdm_symbols - num_symbs)])

        # Change the shape of the imput data. Each row will be modulated as
        # one OFDM symbol.
        input_signal.shape = (num_ofdm_symbols, self.num_used_subcarriers)

        input_ifft = np.zeros([num_ofdm_symbols, self.fft_size], dtype=complex)
        input_ifft[:, self._get_used_subcarrier_indexes_proper()] \
            = input_signal[:, :]

        return input_ifft

    # def _prepare_received_signal(self, received_signal):
    #     """Prepare the received signal that will still be passed to the FFT
    #     function in the demodulate method.

    #     NOTE: The received_signal will be modified. That is, no copy will
    #     be performed, but the input of the _prepare_received_signal will be
    #     modified.

    #     Arguments:
    #     - `received_signal`: Received signal that will still be passed to
    #                          the FFT in the demodulate function.
    #     Output:
    #     - `input_fft`: Signal suitable to be passed to the FFT function
    #                    to actually perform the OFDM demodulation.

    #     """
    #     num_ofdm_symbols = received_signal.size // self.fft_size
    #     received_signal.shape = (num_ofdm_symbols, self.fft_size)
    #     return received_signal

    def _prepare_decoded_signal(self, decoded_signal):
        """Prepare the decoded signal that was processed by the FFT in the
        demodulate function.

        This is equivalent of reversing the indexing that was done by the
        _prepare_input_signal method.

        Parameters
        ----------
        decoded_signal : 1D numpy array
            Signal that was decoded by the FFT in the OFDM demodulate
            method.

        Returns
        -------
        demodulated_samples : 1D numpy array
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
        return decoded_signal[
            :, self._get_used_subcarrier_indexes_proper()].flatten()

    def _add_CP(self, input_data):
        """Add the Cyclic prefix to the input data.

        Note that `input_data` must have a shape of (Number of OFDM
        symbols, IFFT size).

        Parameters
        ----------
        input_data : 2D numpy array
            OFDM modulated data (after the IFFT).

        Returns
        -------
        output : 1D numpy array
            The `input_data` with the cyclic prefix added. The shape of the
            output is (Number of OFDM symbols, IFFT size + CP Size).

        """
        if self.cp_size != 0:
            output = np.hstack([input_data[:, -self.cp_size:], input_data])
        else:
            output = input_data
        return output

    def _remove_CP(self, received_data):
        """Remove the Cyclic prefix of the received data.

        Parameters
        ----------
        received_data : 2D numpy array
            Data that must be demodulated by the OFDM object. This is a one
            dimensional array.

        Returns
        -------
        output : 1D numpy array
            Received data without the Cyclic prefix. This is a
            bi-dimensional array.

        Notes
        -----
        The _remove_CP method will also change the shape so that it is
        suitable to be passed to the FFT function.

        """
        num_ofdm_symbols = received_data.size // (self.fft_size + self.cp_size)
        received_data.shape = (num_ofdm_symbols, self.fft_size + self.cp_size)
        received_data_no_CP = received_data[:, self.cp_size:]

        return received_data_no_CP

    def _calculate_power_scale(self):
        """
        Calculate the power scale that needs to be applied in the modulator and
        removed in the demodulate methods.

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
        power_scale = (float(self.fft_size) ** 2) / (float(self.num_used_subcarriers) + self.cp_size)
        return power_scale

    def modulate(self, input_signal):
        """Perform the OFDM modulation of the input_signal.

        .. TODO:: Write here about the performed zeropadding as well as CP
           addition.

        Parameters
        ----------
        input_signal : 1D numpy array
            Input signal that must be modulated by the OFDM modulate
            function.

        Returns
        -------
        output : 1D numpy array
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
        # to calculate the ifft separatelly for each row in the input_ifft
        # variable.
        output_ifft = math.sqrt(self._calculate_power_scale()) \
                      * np.fft.ifft(input_ifft, self.fft_size, 1)

        # Add the Cyclic prefix
        modulated_ofdm = self._add_CP(output_ifft)

        # Change the shape to one dimensional array and return that array
        return modulated_ofdm.flatten()

    def demodulate(self, received_signal):
        """Perform the OFDM demodulation of the received_signal.

        .. TODO:: Write here about the zeropadding (which is not removed) as
           well as CP removal.

        Parameters
        ----------
        received_signal : 1D numpy array
            An array with the samples of the received OFDM symbols.

        Returns
        -------
        demodulated_data : 1D numpy array
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
        # to calculate the fft separatelly for each row in the
        # received_signal variable.
        output_fft = np.fft.fft(received_signal_no_CP, self.fft_size, 1) \
                     / math.sqrt(self._calculate_power_scale())

        # - CALL THE _prepare_decoded_signal METHOD TO GET THE DATA ONLY
        # FROM THE USEFUL SUBCARRIERS
        decoded_symbols = self._prepare_decoded_signal(output_fft)

        # - RETURN THE DECODED DATA
        return decoded_symbols
        #
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # received_signal_no_CP = self._remove_CP(received_signal)

        # num_ofdm_symbols = received_signal_no_CP.size / self.fft_size
        # received_signal_no_CP.shape = (num_ofdm_symbols, self.fft_size)

        # demodulated_data = np.fft.fft(received_signal_no_CP,
        #                               self.fft_size, 1)
        # return demodulated_data
