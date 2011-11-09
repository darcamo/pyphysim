#!/usr/bin/env python
# -*- coding: utf-8 -*-

# http://www.doughellmann.com/PyMOTW/struct/
# import struct
# import binascii

"""Module docstring"""

__version__ = "$Revision: $"
# $Source$

import sys
sys.path.append("/home/darlan/cvs_files/Python_Funcs/")
from darlan import *

import numpy as np
import math
import matplotlib.pyplot as plt
#import matplotlib.patches as patches

from scipy.special import erfc
#import math.erf
# erf tb pode ser encontrada la biblioteca scipy.special
# erf tb pode ser encontrada la biblioteca math  -> python 2.7 ou superior
# erf tb pode ser encontrada la biblioteca mpmath

PI = np.pi


# xxxxx Misc Functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def level2bits(n):
    """Calculates the number of needed to represent n different values.

    Arguments:
    - `n`: Number of different levels.
    """
    return int2bits(n - 1)


def int2bits(n):
    """Calculates the number of bits needed to represent an interger n.

    Arguments:
    - `n`: An Ingerger number
    """
    assert n >= 0, "int2bits: Improper argument value"
    if n < 0:
        raise Exception("int2bits: n must be greater then zero")

    if n == 0:
        return 1

    bits = 0
    while n:
        n >>= 1
        bits += 1
    return bits


def xor(a, b):
    """Calculates the xor operation between a and b.

    In python this is performed with a^b. However, sage changed the "^"
    operator. This xor function was created so that it can be used in
    either sage or in regular python.

    Arguments:
    - `a`: first number
    - `b`: second number

    """
    return (a).__xor__(b)


# Code from wikipedia
# http://en.wikipedia.org/wiki/Gray_code#Constructing_an_n-bit_Gray_code
def binary2gray(num):
    """Convert a number (in decimal format) to the corresponding Gray code
    (still in decimal format).

    Arguments:
    - `num`:
    """
    return xor((num >> 1), num)


def gray2binary(num):
    """Convert a number in Gray code (in decimal format) to its original
    value (in decimal format).

    Arguments:
    - `num`:

    """
    temp = xor(num, (num >> 8))
    temp = xor(temp, (temp >> 4))
    temp = xor(temp, (temp >> 2))
    temp = xor(temp, (temp >> 1))

    return temp


def bitCount(n):
    """Count the number of bits that are set in an interger number.

    Arguments:
    - `n`:
    """
    count = 0
    while n > 0:
        if n & 1 == 1:
            count += 1
        n >>= 1
    return count

# Make bitCount an ufunc
bitCount = np.frompyfunc(bitCount, 1, 1)


def qfunc(x):
    """Calculates the qfunction of x.

    Arguments:
    - `x`:
    """
    return 0.5 * erfc(x / math.sqrt(2))

# xxxxx End of misc functions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxx Modulator Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Modulator:
    """
    """

    def __init__(self):
        """
        """
        # This should be set in a subclass of the Modulator Class.
        self.M = 0
        self.symbols = np.array([])

    def __repr__(self):
        return "{0:d}-{1:s} object".format(self.M, self.__class__.__name__)

    def setConstellation(self, symbols):
        """Set the constelation of the modulator.

        This function should be called in the constructor of the derived
        classes.

        Arguments:
        - `symbols`: A an numpy array with the symbol table

        """
        self.M = symbols.size
        self.symbols = symbols

    def plotConstellation(self):
        """Plot the constellation (in a scatter plot).
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)  # one row, one column, first plot
        # circlePatch = patches.Circle((0,0),1)
        # circlePatch.set_fill(False)
        # ax.add_patch(circlePatch)
        ax.scatter(self.symbols.real, self.symbols.imag)

        ax.axis('equal')
        ax.grid()

        formatString = "{0:0=" + str(level2bits(self.M)) + "b} ({0})"

        index = 0
        for symbol in self.symbols:
            ax.text(
                symbol.real,  # Coordinate X
                symbol.imag + 0.03,  # Coordinate Y
                formatString.format(index, format_spec="0"),  # Text
                verticalalignment='bottom',  # From now on, text properties
                horizontalalignment='center')
            index += 1

        plt.show()

    def modulate(self, inputData):
        """Modulate the input data (decimal data).

        Arguments:
        - `inputData`: Data to be modulated
        """
        # TODO: test is inputData is valid (elements between 0 and M-1)
        return self.symbols[inputData]

    def demodulate(self, receivedData):
        """Demodulate the data.

        Arguments:
        - `receivedData`: Data to be demodulated
        """
        def getClosestSymbol(symb):
            """
            """
            closestSymbolIndex = abs(self.symbols - symb).argmin()
            return closestSymbolIndex
        getClosestSymbol = np.frompyfunc(getClosestSymbol, 1, 1)
        return getClosestSymbol(receivedData)

    def calcTheoreticalSER(self, snr):
        raise NotImplementedError("calcTheoreticalSER: Not implemented")

    def calcTheoreticalBER(self, snr):
        raise NotImplementedError("calcTheoreticalBER: Not implemented")
# xxxxx End of Modulator Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxx PSK Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class PSK(Modulator):
    """
    """
    def __init__(self, M, phaseOffset=0):
        """
        """
        Modulator.__init__(self)
        # Check if M is a power of 2
        assert 2**math.log(M, 2) == M

        # Generates the constellation
        symbols = self.___createConstellation(M, phaseOffset)

        # Change to Gray mapping
        symbols = symbols[gray2binary(np.arange(0, M))]

        self.setConstellation(symbols)

    # def __repr__(self):
    #     return "{0:d}-PSK object".format(self.M)

    @staticmethod
    def ___createConstellation(M, phaseOffset):
        """Generates the Constellation for the PSK modulation scheme

        Arguments:
        - `M`: Modulation cardinality
        - `phaseOffset`: phase offset (in radians)
        """
        phases = 2 * PI / M * np.arange(0, M) + phaseOffset
        realPart = np.cos(phases)
        imagPart = np.sin(phases)

        # Any number inferior to 1e-15 will be considered as 0
        realPart[abs(realPart) < 1e-15] = 0
        imagPart[abs(imagPart) < 1e-15] = 0
        return realPart + 1j * imagPart

    def setPhaseOffset(phaseOffset):
        """Set a new phase offset for the constellation

        Arguments:
        - `phaseOffset`: phase offset (in radians)
        """
        self.setConstellation(self.__createConstellation(self.M, phaseOffset))

    def calcTheoreticalSER(self, snr):
        """Calculates the theoretical (approximation) symbol error rate for
        the M-PSK squeme.

        Arguments:
        - `snr`: Signal to noise ration (in linear)
        """
        # $P_s \approx 2Q\left(\sqrt{2\gamma_s}\sin\frac{\pi}{M}\right)$
        ser = 2 * qfunc(np.sqrt(2 * snr) * math.sin(PI / self.M))
        return ser

    def calcTheoreticalBER(self, snr):
        """Calculates the theoretical (approximation) bit error rate for
        the M-PSK squeme using Gray coding.

        Arguments:
        - `snr`: Signal to noise ration (in linear)
        """
        # $P_b = \frac{1}{k}P_s$
        # Number of bits per symbol
        k = level2bits(self.M)
        return 1.0 / k * self.calcTheoreticalSER(snr)


# xxxxx End of PSK Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxx QPSK Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class QPSK(PSK):
    """QPSK Class.
    """

    def __init__(self, ):
        """
        """
        PSK.__init__(self, 4, PI / 4)

    def __repr__(self):
        return "QPSK object"

    # def calcTheoreticalBER(self, EbOverN0):
    #     """

    #     Arguments:
    #     - `EbOverN0`: Bit energy over noise power
    #     """
    #     # $P_b = Q\left(\sqrt{\frac{2E_b}{N_0}}\right)$
    #     Pb = qfunc(math.sqrt(2*EbOverN0))
    #     return Pb
# xxxxx End of QPSK Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxx BPSK Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class BPSK(Modulator):
    """
    """
    def __init__(self, ):
        """
        """
        Modulator.__init__(self)
        self.setConstellation(np.array([-1, 1]))

    def __repr__(self):
        return "BPSK object"

    def calcTheoreticalSER(self, snr):
        """Calculates the theoretical (approximation) symbol error rate for
        the BPSK squeme.

        Arguments:
        - `snr`: Signal to noise ration (in linear)
        """
        # $P_b = Q\left(\sqrt{\frac{2E_b}{N_0}}\right)$
        ser = qfunc(math.sqrt(2 * snr))
        return ser

    def calcTheoreticalBER(self, snr):
        """Calculates the theoretical (approximation) bit error rate for
        the BPSK squeme.

        Arguments:
        - `snr`: Signal to noise ration (in linear)
        """
        return self.calcTheoreticalSER(snr)

# xxxxx End of BPSK Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxx QAM Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class QAM(Modulator):
    """
    """

    def __init__(self, M):
        """
        """
        Modulator.__init__(self)

        # Check if M is an even power of 2
        power = math.log(M, 2)
        assert power % 2 == 0, "M must be a square power of 2"
        assert 2**power == M, "M must be a square power of 2"

        symbols = self.__createConstellation(M)

        # TODO: Change to Gray mapping
        L = int(round(math.sqrt(M)))
        grayMappingIndexes = self.__calculateGrayMappingIndexQAM(L)
        symbols = symbols[grayMappingIndexes]

        # Set the constelation
        self.setConstellation(symbols)

        # __createConstellation deve retornar os simbolos
    def __createConstellation(self, M):
        """Generates the Constellation for the (SQUARE) M-QAM modulation
        scheme.

        Arguments:
        - `M`: Modulation cardinality
        """
        # Size of the square. The square root is exact
        symbols = np.empty(M, dtype=complex)
        L = int(round(math.sqrt(M)))
        for jj in range(0, L):
            for ii in range(0, L):
                symbol = complex(-(L - 1) + jj * 2, (L - 1) - ii * 2)
                #print symbol
                symbols[ii * L + jj] = symbol

        average_energy = (M - 1) * 2.0 / 3.0
        # Normalize the constellation, so that the mean symbol energy is
        # equal to one.
        return symbols / math.sqrt(average_energy)

    def __calculateGrayMappingIndexQAM(self, L):
        """Calculates the indexes that should be applied to the
        constellation created by __createConstellation in order to
        correspond to Gray mapping.

        Notice that the square M-QAM constellation is a matrix of dimension
        L x L, where L is the square root of M. Since the constellation was
        generated without taking into account the Gray mapping, then we
        need to reorder the generated constellation and this function
        calculates the indexes that can be applied to the original
        constellation in order to do this.

        As an example, for the 16-QAM modulation the indexes can be
        organized (row order) in the matrix below
                  00     01     11     10
               |------+------+------+------|
            00 | 0000 | 0001 | 0011 | 0010 |
            01 | 0100 | 0101 | 0111 | 0110 |
            11 | 1100 | 1101 | 1111 | 1110 |
            10 | 1000 | 1001 | 1011 | 1010 |
               |------+------+------+------|
        This is equivalent to concatenate a Gray mapping for the row with a
        Gray mapping for the column, and the corresponding indexes are
        [0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10]

        Arguments:
        - `L`: Square root of the modulation cardinality (must be an interger)

        """
        # Row vector with the column variation (second half of the index in
        # binary form)
        column = binary2gray(np.arange(0, L, dtype=int))
        # Column vector with the row variation
        row = column.reshape(L, 1)  # Column vector with the row variation
                                    # (first half of the index in binary
                                    # form)
        columns = np.tile(column, (L, 1))
        rows = np.tile(row, (1, L))
        # Shift the first part by half the number of bits and sum with the
        # second part to form each element in the index matrix
        index_matrix = (rows << (level2bits(L**2) / 2)) + columns

        # Return the indexes as a vector (row order, which is the default
        # in numpy)
        return np.reshape(index_matrix, L**2)
    #
    # TODO: Implement calcTheoreticalSER and calcTheoreticalBER for square
    # QAM systems
# xxxxx End of QAM Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxx Tests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def testGrayCodeConversion(maxNum=8):
    for i in range(0, maxNum):
        grayNumber = binary2gray(i)
        print ("Normal: ({0:2}) {0:0=4b} | Gray: ({1:2}) {1:0=4b} -> Conver"
               "tido {2:2}").format(i, grayNumber, gray2binary(grayNumber))
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


if __name__ == '__main__':
    M = 4
    psk = PSK(M)

    inputData = np.random.randint(0, M, 20)
    modulatedData = psk.modulate(inputData)

    receivedData = (modulatedData + 1e-4 * np.random.randn(20) +
                    1e-4 * 1j * np.random.randn())
    demodulatedData = psk.demodulate(receivedData)

    # Calculates the symbol error rate
    SER = 1.0 * sum(inputData != demodulatedData) / inputData.size

    # testGrayCodeConversion(16)

    qam = QAM(16)
