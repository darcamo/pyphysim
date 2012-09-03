#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""module docstring"""

import numpy as np
from matplotlib import pylab
from matplotlib import pyplot as plt


class OFDM():
    """OFDM class.
    """
    def __init__(self, fft_size, cp_size, num_used_subcarriers=0):
        """Initializates the OFDM object.

        See the documentation of the `set_parameters` method.
        """
        self.set_parameters(fft_size, cp_size, num_used_subcarriers)

    def set_parameters(self, fft_size, cp_size, num_used_subcarriers=0):
        """

        Arguments:
        - `fft_size`: Size of the FFT and IFFT used by the OFDM class.
        - `cp_size`: Size of the cyclic prefix (in samples)
        - `num_used_subcarriers`: Number of used subcarriers. Must be
                                  greater than or equal to 2 and lower than
                                  or equal to fft_size. Otherwise fft_size
                                  will be used instead.
        """
        if (cp_size < 0) or cp_size > fft_size:
            raise ValueError("cp_size must be nonnegative and cannot be greater than fft_size")

        if num_used_subcarriers < 0:
            num_used_subcarriers = fft_size

        if num_used_subcarriers > fft_size:
            raise ValueError("Number of used subcarriers cannot be greater than the fft_size")

        if (num_used_subcarriers % 2 != 0) or (num_used_subcarriers < 2):
            raise ValueError("Number of used subcarriers must be a multiple of 2")

        self.fft_size = fft_size
        self.cp_size = cp_size
        self.num_used_subcarriers = num_used_subcarriers

    def _calc_zeropad(self, input_data_size):
        """Calculates the number of zeros that must be added to the input
        data to make it a multiple of the OFDM size.

        The number of zeros that must be added to the input data is
        returned along with the number of OFDM symbols that will be
        generated.

        Arguments:
        - `input_data_size`: Size the the data that will be modulated by
                             the OFDM object.
        """
        num_ofdm_symbols = int(np.ceil(float(input_data_size) / self.num_used_subcarriers))
        zeropad = self.num_used_subcarriers * num_ofdm_symbols - input_data_size
        return (zeropad, num_ofdm_symbols)

    def get_subcarrier_indexes(self):
        """Get the subcarrier indexes of all subcarriers, including the
        negative, the DC and the positive subcarriers.

        Note that these indexes are not suitable for indexing in
        python. They are the actual indexes of the subcarriers in an OFDM
        symbol. For instance, an OFDM symbol with 16 subcarriers will have
        indexes from -8 to 7. However, due to the way the fft is
        implemented in numpy the indexes here are actually from 0 to 7
        followed by -8 to -1. That is
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
        """Get the subcarrier indexes of the subcarriers actually used.

        Note that these indexes are not suitable for indexing in
        python. They are the actual indexes of the subcarriers in an OFDM
        symbol. See the documentation of the get_subcarrier_indexes
        function.

        Ex:
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

        This is the function actually used in the modulate function.
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
                indexes[0:half_used]
                ])
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

        Arguments:
        - `input_data`: Input signal that must be modulated by the OFDM
                        modulate function.
        Output:
        - `input_ifft`: Signal suitable to be passed to the IFFT function
                        to actually perform the OFDM modulation.
        """
        # Number of data symbols passed to the modulate function
        num_symbs = input_signal.size
        # Calculates how many zeros need to be added as well as the number
        # of equivalent OFDM symbols.
        zeropad, num_ofdm_symbols = self._calc_zeropad(num_symbs)

        # Finally add the zeros to the input data
        input_signal = np.hstack([input_signal, np.zeros(self.num_used_subcarriers * num_ofdm_symbols - num_symbs)])

        # Change the shape of the imput data. Each row will be modulated as
        # one OFDM symbol.
        input_signal.shape = (num_ofdm_symbols, self.num_used_subcarriers)

        input_ifft = np.zeros([num_ofdm_symbols, self.fft_size])
        input_ifft[:, self._get_used_subcarrier_indexes_proper()] = input_signal[:, :]

        return input_ifft

    def _add_CP(self, input_data):
        """Add the Cyclic prefix to the input data.

        `input_data` must have a shape of (Number of OFDM symbols, IFFT
        size).

        Arguments:
        - `input_data`: OFDM modulated data (after the IFFT)
        Output:
        - `output`: input_data with the cyclic prefix added. The shape of
                    the output is (Number of OFDM symbols, IFFT size + CP
                    Size).
        """
        if self.cp_size != 0:
            output = np.hstack([input_data[:, -self.cp_size:], input_data])
        else:
            output = input_data
        return output

    def _remove_CP(self, received_data):
        """Remove the Cyclic prefix of the received data.

        Arguments:
        - `received_data`: Data that must be demodulated by the OFDM
                           object.
        Output:
        - `output`: received data without the Cyclic prefix.

        """
        # TODO: Implement-me
        return received_data

    def modulate(self, input_signal):
        """

        Arguments:

        - `input_signal`: Input signal that must be modulated by the OFDM
                          modulate function.
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
        output_ifft = np.fft.ifft(input_ifft, self.fft_size, 1)

        # Add the Cyclic prefix
        modulated_ofdm = self._add_CP(output_ifft)

        # Change the shape to one dimensional array and return that array
        return modulated_ofdm.flatten()

    def demodulate(self, received_signal):
        """

        Arguments:
        - `received_signal`:
        Output:
        - `demodulated_data`: Demodulated symbols
        """
        received_signal_no_CP = self._remove_CP(received_signal)

        num_ofdm_symbols = received_signal_no_CP.size / self.fft_size
        received_signal_no_CP.shape = (num_ofdm_symbols, self.fft_size)

        demodulated_data = np.fft.fft(received_signal_no_CP, self.fft_size, 1)
        return demodulated_data

if __name__ == '__main__':
    # xxxxxxxxxx Input generation (not part of OFDM) xxxxxxxxxxxxxxxxxxxxxx
    num_bits = 2500
    # generating 1's and 0's
    ip_bits = np.random.random_integers(0, 1, num_bits)
    # Number of modulated symbols
    num_mod_symbols = num_bits * 1
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxxxxxxx BPSK modulation xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # bit0 --> -1
    # bit1 --> +1
    ip_mod = 2 * ip_bits - 1

    ofdm = OFDM(64, 16, 52)
    ofdm_symbols = ofdm.modulate(ip_mod)

    # MATLAB code to plot the power spectral density
    # close all
    fsMHz = 20e6
    Pxx, W = pylab.psd(ofdm_symbols, NFFT=ofdm.fft_size, Fs=fsMHz)
    # [Pxx,W] = pwelch(st,[],[],4096,20);
    plt.plot(
        W,
        #10 * np.log10(np.fft.fftshift(Pxx))
        10 * np.log10(Pxx)
        )
    plt.xlabel('frequency, MHz')
    plt.ylabel('power spectral density')
    plt.title('Transmit spectrum OFDM (based on 802.11a)')
    plt.show()
