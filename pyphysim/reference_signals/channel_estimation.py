#!/usr/bin/env python
"""
Module with channel estimation implementations based on the reference signals in
this package.
"""

from typing import Union

import numpy as np

from .dmrs import DmrsUeSequence
from .srs import SrsUeSequence, UeSequence


class CazacBasedChannelEstimator:
    """
    Estimated the (uplink) channel based on CAZAC (Constant Amplitude Zero
    AutoCorrelation) reference sequences sent by one user (either SRS or
    DMRS).

    The estimation is performed according to the paper [Bertrand2011]_,
    where the received signal in the FREQUENCY DOMAIN is used by the
    estimator.

    Note that for SRS sequences usually a comb pattern is employed such
    that only half of the subcarriers is used to send pilot
    symbols. Therefore, an FFT in the during the estimation will
    effectively interpolate for the other subcarriers. This is controlled
    by the `size_multiplier` argument (default is 2 to accommodate comb
    pattern). If all subcarriers are used to send pilot symbols then set
    `size_multiplier` to 1.

    Parameters
    ----------
    ue_ref_seq : SrsUeSequence | DmrsUeSequence | np.ndarray
        The reference signal sequence.
    size_multiplier : int, optional
        Multiplication factor for the FFT to get the actual channel size.
        When using the comb pattern for SRS this should be 2 (default value),
        but for DMRS, which does not employ the comb pattern, this should be
        set to 1.

    Notes
    -----

    .. [Bertrand2011] Bertrand, Pierre, "Channel Gain Estimation from
       Sounding Reference Signal in LTE," Conference: Proceedings of the
       73rd IEEE Vehicular Technology Conference.
    """
    def __init__(self,
                 ue_ref_seq: Union[SrsUeSequence, DmrsUeSequence, np.ndarray],
                 size_multiplier: int = 2) -> None:
        # If ue_ref_seq is not an instance of UeSequence (or a subclass)
        # assume it is a numpy array.
        if isinstance(ue_ref_seq, UeSequence):
            self._normalized_ref_seq = ue_ref_seq.normalized
            ue_ref_seq = ue_ref_seq.seq_array()
        else:
            self._normalized_ref_seq = False

        self._ue_ref_sequence = ue_ref_seq
        self._size_multiplier = size_multiplier

    @property
    def ue_ref_seq(self) -> np.ndarray:
        """Get the sequence of the UE."""
        return self._ue_ref_sequence

    def estimate_channel_freq_domain(self, received_signal: np.ndarray,
                                     num_taps_to_keep: int) -> np.ndarray:
        """
        Estimate the channel based on the received signal.

        Parameters
        ----------
        received_signal : np.ndarray
            The received reference signal after being transmitted through the
            channel (in the frequency domain). If this is a 2D numpy array
            the first dimensions is assumed to be "receive antennas" while
            the second dimension are the received sequence elements. The
            number of elements in the received signal (per antenna) is
            equal to the channel size (number of subcarriers) divided by
            `size_multiplier`.
        num_taps_to_keep : int
            Number of taps (in delay domain) to keep. All taps from 0 to
            `num_taps_to_keep`-1 will be kept and all other taps will be
            zeroed before applying the FFT to get the channel response in
            the frequency domain.

        Returns
        -------
        freq_response : np.ndarray
            The channel frequency response. Note that for SRS sequences
            this will have twice as many elements as the sent SRS signal,
            since the SRS signal is sent every other subcarrier.
        """
        # Reference signal sequence
        r = self.ue_ref_seq

        if received_signal.ndim == 1:
            # First we multiply (element-wise) the received signal by the
            # conjugate of the reference signal sequence
            y = np.fft.ifft(np.conj(r) * received_signal, r.size)

            # The channel impulse response consists of the first
            # `num_taps_to_keep` elements in `y`.
            tilde_h = y[0:num_taps_to_keep + 1]
        elif received_signal.ndim == 2:
            # Case with multiple receive antennas
            y = np.fft.ifft(
                np.conj(r)[np.newaxis, :] * received_signal, r.size)

            # The channel impulse response consists of the first
            # `num_taps_to_keep` elements in `y`.
            tilde_h = y[:, 0:num_taps_to_keep + 1]
        else:
            raise ValueError(  # pragma: no cover
                "received_signal must have either one dimension ("
                "one receive antenna) or two dimensions (first "
                "dimension being the receive antenna dimension).")

        # Now we can apply the FFT to get the frequency response.
        # The number of subcarriers is twice the number of elements in the
        # SRS sequence due to the comb pattern
        Nsc = r.size
        tilde_H = np.fft.fft(tilde_h, self._size_multiplier * Nsc)

        if self._normalized_ref_seq is True:
            tilde_H *= Nsc

        return tilde_H


class CazacBasedWithOCCChannelEstimator(CazacBasedChannelEstimator):
    """
    Estimated the (uplink) channel based on CAZAC (Constant Amplitude Zero
    AutoCorrelation) reference sequences sent by one user including the
    Orthogonal Cover Code (OCC).

    With OCC the user will send reference signal in multiple time slots, in
    each slot multiplied with the respective OCC sequence element.

    Parameters
    ----------
    ue_ref_seq : DmrsUeSequence
        The reference signal sequence.
    """
    def __init__(self, ue_ref_seq: DmrsUeSequence) -> None:
        cover_code = ue_ref_seq.cover_code
        ue_ref_seq_array = ue_ref_seq.seq_array()
        reference_seq = ue_ref_seq_array[0] * cover_code[0]

        super().__init__(reference_seq, size_multiplier=1)

        self._cover_code = cover_code
        self._normalized_ref_seq = ue_ref_seq.normalized

    @property
    def cover_code(self) -> np.ndarray:
        """Get the cover code of the UE."""
        return self._cover_code

    def estimate_channel_freq_domain(
            self,
            received_signal: np.ndarray,
            num_taps_to_keep: int,
            extra_dimension: bool = True) -> np.ndarray:
        """
        Estimate the channel based on the received signal with cover codes.

        Parameters
        ----------
        received_signal : np.ndarray
            The received reference signal after being transmitted through
            the channel (in the frequency domain).

            Dimension: Depend if there are multiple receive antennas and if
            `extra_dimension` is True or False. Let :math:`Nr` be the
            number of receive antennas, :math:`Ne` be the number of reference
            signal elements (reference signal size without cover code) and
            :math:`Nc` be the cover code size. The dimension of
            `received_signal` must match the table below.

            =================  =======================  ======================
                   /            extra_dimension: True   extra_dimension: False
            =================  =======================  ======================
            Single Antenna      Nc x Ne           (2D)   Ne * Nc          (1D)
            Multiple Antennas   Nr x Nc x Ne      (3D)   Nr x (Ne * Nc)   (2D)
            =================  =======================  ======================

        num_taps_to_keep : int
            Number of taps (in delay domain) to keep. All taps from 0 to
            `num_taps_to_keep`-1 will be kept and all other taps will be
            zeroed before applying the FFT to get the channel response in
            the frequency domain.
        extra_dimension : bool
            If True then the should be an extra dimension in
            `received_signal` corresponding to the cover code dimension. If
            False then the cover code is included in the dimension of the
            reference signal elements.

        Returns
        -------
        freq_response : np.ndarray
            The channel frequency response.
        """
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxx Add the extra dimension if it does not exist xxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Create a view for the received signals. If extra_dimension is
        # false we will reshape this view to add a dimension for the cover
        # code
        r = received_signal.view()
        if extra_dimension is False:
            # Let's reorganize the received signal so that we have the
            # extra dimension
            if received_signal.ndim == 1:
                # Case with a single antenna. Cover code dimension will be
                # the first dimension.
                r.shape = (self.cover_code.size, -1)

            elif received_signal.ndim == 2:
                # Case with multiple antennas. Cover code dimension will be
                # the second dimension.
                num_antennas = r.shape[0]
                r.shape = (num_antennas, self.cover_code.size, -1)
            else:
                raise RuntimeError(
                    'Invalid dimension for received_signal: {0}'.format(
                        r.ndim))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxx Average over the cover code dimension xxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Now we can consider the case with the extra cover_code dimension
        if r.ndim == 2:
            # Apply the cover code
            r_mean = np.mean(r * self.cover_code[:, np.newaxis], axis=0)
        elif r.ndim == 3:
            r_mean = np.mean(r * self.cover_code[np.newaxis, :, np.newaxis],
                             axis=1)

        else:
            raise RuntimeError(
                'Invalid dimension for received_signal: {0}'.format(r.ndim))

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxx Perform the estimation xxxxxxxxxxxxxxxxxxxxxxxxxx
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # Call the estimate_channel_freq_domain from the base class
        return super().estimate_channel_freq_domain(r_mean, num_taps_to_keep)
