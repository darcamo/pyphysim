#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot the Power Spectral Density of OFDM modulated data"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys

import os

try:
    parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    grandparent_dir = os.path.split(parent_dir)[0]
    sys.path.append(grandparent_dir)
except NameError:
    sys.path.append('../../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

import numpy as np
from matplotlib import pylab
from matplotlib import pyplot as plt
from pyphysim.modulators.ofdm import OFDM

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

    ofdm_obj = OFDM(64, 16, 52)
    ofdm_symbols = ofdm_obj.modulate(ip_mod)

    # Code to plot the power spectral density
    fsMHz = 20e6
    Pxx, W = pylab.psd(ofdm_symbols, NFFT=ofdm_obj.fft_size, Fs=fsMHz)
    # [Pxx,W] = pwelch(st,[],[],4096,20);
    plt.plot(
        W,
        # 10 * np.log10(np.fft.fftshift(Pxx))
        10 * np.log10(Pxx))
    plt.xlabel('frequency, MHz')
    plt.ylabel('power spectral density')
    plt.title('Transmit spectrum OFDM (based on 802.11a)')
    plt.show()
