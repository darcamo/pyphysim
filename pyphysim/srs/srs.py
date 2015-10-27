#!/usr/bin/env python
# -*- coding: utf-8 -*-
"Module with Sounding Reference Signal (SRS) related functions"

import numpy as np
from pyphysim.util.zadoffchu import calcBaseZC, getShiftedZF, get_extended_ZF


class SrsRootSequence(object):
    """
    SRS root sequence class.

    Parameters
    ----------
    root_index : int
        The SRS root sequence index.
    Nzc : int
        The size of the Zadoff-Chu sequence (without any extension).
    extend_to : int
        The size of the extended Zadoff-Chu sequence. It None then the
        sequence will not be extended and will thus have a size equal
        to Nzc.
    """

    def __init__(self, root_index, Nzc, extend_to=None):
        self._root_index = root_index
        self._zf_seq_array = calcBaseZC(Nzc, root_index)  # Zadoff-Chu sequence
        self._extended_zf_seq_array = None  # Extended Zadoff-Chu sequence

        if extend_to is not None:
            if extend_to <= Nzc:
                raise AttributeError(
                    "If 'extend_to' is provided it must be greater than Nzc")
            else:
                self._extended_zf_seq_array = get_extended_ZF(
                    self._zf_seq_array, extend_to)

    @property
    def Nzc(self):
        """Get the size of the Zadoff-Chu sequence (without any extension)"""
        return self._zf_seq_array.size

    @property
    def size(self):
        """
        Return the size (with extension) of the sequece.

        If the sequence is not extended than `size()` will return the same
        as `Nzc`.

        Returns
        -------
        size : int
            The size of the extended Zadoff-Chu sequence.

        Example
        -------
        >>> seq1 = SrsRootSequence(root_index=25, Nzc=139)
        >>> seq1.size
        139
        >>> seq1 = SrsRootSequence(root_index=25, Nzc=139, extend_to=150)
        >>> seq1.size
        150
        """
        if self._extended_zf_seq_array is None:
            return self.Nzc
        else:
            return self._extended_zf_seq_array.size

    def seq_array(self):
        """
        Get the extended Zadoff-Chu root sequence as a numpy array.

        Returns
        -------
        seq : numpy array
            The extended Zadoff-Chu sequence
        """
        if self._extended_zf_seq_array is None:
            return self._zf_seq_array
        else:
            return self._extended_zf_seq_array


class SrsUeSequence(object):
    """
    SRS sequence of a single user.

    Parameters
    ----------
    n_cs : int
        The shift index of the user. This can be an integer from 1 to 8.
    root_seq : SrsRootSequence object
        The SRS root sequence of the base station the user is
        associated to. This should be an object of the SrsRootSequence
        class.
    """

    def __init__(self, n_cs, root_seq):
        root_seq_array = root_seq.seq_array()
        self._user_seq_array = getShiftedZF(root_seq_array, n_cs)

    @property
    def size(self):
        """
        Return the size of the user's SRS sequence.

        Returns
        -------
        size : int
            The size of the user's SRS sequence.

        Example
        -------
        >>> root_seq1 = SrsRootSequence(root_index=25, Nzc=139)
        >>> user_seq1 = SrsUeSequence(3, root_seq1)
        >>> user_seq1.size
        139
        >>> root_seq2 = SrsRootSequence(root_index=25, Nzc=139, extend_to=150)
        >>> user_seq2 = SrsUeSequence(3, root_seq2)
        >>> user_seq2.size
        150
        """
        return self._user_seq_array.size

    def seq_array(self):
        """
        Get the user's SRS sequence as a numpy array.

        Returns
        -------
        seq : numpy array
            The user's SRS sequence.
        """
        return self._user_seq_array


class SrsChannelEstimator(object):
    """
    Estimated the (uplink) channel based on the SRS sequence sent by one
    user.

    The estimation is performed according to the paper REFERENCE, where the
    received signal in the FREQUENCY DOMAIN is used by the estimator.

    Parameters
    ----------
    srs_ue : SrsUeSequence object
        The user's SRS sequence.
    """

    def __init__(self, srs_ue):
        self._srs_ue = srs_ue

    def estimate_channel(self, received_signal, num_taps_to_keep=16):
        """
        Estimate the channel based on the received signal.

        Parameters
        ----------
        received_signal : numpy array
            The received SRS signal after being transmitted through the
            channel (in the frequency domain).
        num_taps_to_keep : int
            Number of taps (in delay domain) to keep. All taps from 0 to
            `num_taps_to_keep`-1 will be kept and all other taps will be
            zeroed before applying the FFT to get the channel response in
            the frequency domain.

        Returns
        -------
        freq_response : numpy array
            The channel frequency response. Note that this will have twice
            as many elements as the sent SRS signal, since the SRS signal
            is sent every other subcarrier.
        """
        # User's SRS sequence
        r = self._srs_ue.seq_array()

        # First we multiply (elementwise) the received signal by the
        # conjugate of the user's SRS sequence
        y = np.fft.ifft(np.conj(r) * received_signal, r.size)

        # The channel impulse response consists of the first
        # `num_taps_to_keep` elements in `y`.
        tilde_h = y[0:num_taps_to_keep]

        # Now we can apply the FFT to get the frequency response
        Nsc = r.size  # Number of subcarriers is twice the number of
                      # elements in the SRS sequence due to the comb
                      # pattern
        tilde_H = np.fft.fft(tilde_h, 2 * Nsc)

        return tilde_H
