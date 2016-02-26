#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module with channel estimation implementations based on the reference
signals in this package. """
import numpy as np
from .srs import SrsUeSequence, UeSequence
from .dmrs import DmrsUeSequence


class CazacBasedChannelEstimator(object):
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

    def __init__(self, ue_ref_seq, size_multiplier=2):
        # If ue_ref_seq is not an instance of UeSequence (or a subclass)
        # assume it is a numpy array.
        if isinstance(ue_ref_seq, UeSequence):
            ue_ref_seq = ue_ref_seq.seq_array()

        self._ue_ref_sequence = ue_ref_seq
        self._size_multiplier = size_multiplier

    @property
    def ue_ref_seq(self):
        """Get the sequence of the UE."""
        return self._ue_ref_sequence

    def estimate_channel_freq_domain(self, received_signal,
                                     num_taps_to_keep):
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
            The channel frequency response. Note that this will have twice
            as many elements as the sent SRS signal, since the SRS signal
            is sent every other subcarrier.
        """
        # Reference signal sequence
        r = self.ue_ref_seq

        if received_signal.ndim == 1:
            # First we multiply (element-wise) the received signal by the
            # conjugate of the reference signal sequence
            y = np.fft.ifft(np.conj(r) * received_signal, r.size)

            # The channel impulse response consists of the first
            # `num_taps_to_keep` elements in `y`.
            tilde_h = y[0:num_taps_to_keep+1]
        elif received_signal.ndim == 2:
            # Case with multiple receive antennas
            y = np.fft.ifft(np.conj(r)[np.newaxis, :] * received_signal,
                            r.size)

            # The channel impulse response consists of the first
            # `num_taps_to_keep` elements in `y`.
            tilde_h = y[:, 0:num_taps_to_keep+1]
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

        return tilde_H
