#!/usr/bin/env python
# -*- coding: utf-8 -*-

# http://www.doughellmann.com/PyMOTW/struct/
# import struct
# import binascii

"""Module docstring"""

__version__ = "$Revision: $"
# $Source$

import numpy as np
import math
import matplotlib.pyplot as plt

from util.misc import level2bits, qfunc
from util.conversion import gray2binary, binary2gray, dB2Linear

PI = np.pi


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxx Modulator Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class Modulator:
    """Base class for digital modulators.

    The derived classes need to at least call setConstellation to set the
    constellation in their constructors as well as implement
    calcTheoreticalSER and calcTheoreticalBER.

    >>> constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j])
    >>> m=Modulator()
    >>> m.setConstellation(constellation)
    >>> m.symbols
    array([ 1.+1.j, -1.+1.j, -1.-1.j,  1.-1.j])
    >>> m.M
    4
    >>> m
    4-Modulator object
    >>> m.modulate(np.array([0, 0, 3, 3, 1, 3, 3, 3, 2, 2]))
    array([ 1.+1.j,  1.+1.j,  1.-1.j,  1.-1.j, -1.+1.j,  1.-1.j,  1.-1.j,
            1.-1.j, -1.-1.j, -1.-1.j])

    >>> m.demodulate(np.array([ 1.+1.j, 1.+1.j, 1.-1.j, 1.-1.j, -1.+1.j, \
                                1.-1.j, 1.-1.j, 1.-1.j, -1.-1.j, -1.-1.j]))
    array([0, 0, 3, 3, 1, 3, 3, 3, 2, 2])
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
        try:
            return self.symbols[inputData]
        except IndexError:
            raise IndexError("Input data must be between 0 and 2^M")

    def demodulate(self, receivedData):
        """Demodulate the data.

        Arguments:
        - `receivedData`: Data to be demodulated
        """
        def getClosestSymbol(symb):
            closestSymbolIndex = abs(self.symbols - symb).argmin()
            return closestSymbolIndex
        getClosestSymbol = np.frompyfunc(getClosestSymbol, 1, 1)
        return getClosestSymbol(receivedData).astype(int)

    def calcTheoreticalSER(self, SNR):
        """Calculates the theoretical symbol error rate.

        This function should be implemented in the derived classes

        Arguments:
        - `SNR`: Signal-to-noise-value (in dB)
        """
        raise NotImplementedError("calcTheoreticalSER: Not implemented")

    def calcTheoreticalBER(self, SNR):
        """Calculates the theoretical bit error rate.

        This function should be implemented in the derived classes

        Arguments:
        - `SNR`: Signal-to-noise-value (in dB)
        """
        raise NotImplementedError("calcTheoreticalBER: Not implemented")
# xxxxx End of Modulator Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxx PSK Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class PSK(Modulator):
    """PSK Class
    """
    def __init__(self, M, phaseOffset=0):
        """
        """
        Modulator.__init__(self)
        # Check if M is a power of 2
        assert 2 ** math.log(M, 2) == M

        # Generates the constellation
        symbols = self.___createConstellation(M, phaseOffset)

        # Change to Gray mapping
        symbols = symbols[gray2binary(np.arange(0, M))]

        self.setConstellation(symbols)

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

    def setPhaseOffset(self, phaseOffset):
        """Set a new phase offset for the constellation

        Arguments:
        - `phaseOffset`: phase offset (in radians)
        """
        self.setConstellation(self.__createConstellation(self.M, phaseOffset))

    def calcTheoreticalSER(self, SNR):
        """Calculates the theoretical (approximation for high M and high
        SNR) symbol error rate for the M-PSK squeme.

        Arguments:
        - `SNR`: Signal-to-noise-value (in dB)
        """
        snr = dB2Linear(SNR)

        # $P_s \approx 2Q\left(\sqrt{2\gamma_s}\sin\frac{\pi}{M}\right)$
        ser = 2 * qfunc(np.sqrt(2 * snr) * math.sin(PI / self.M))
        return ser

    def calcTheoreticalBER(self, SNR):
        """Calculates the theoretical (approximation) bit error rate for
        the M-PSK squeme using Gray coding.

        Arguments:
        - `SNR`: Signal to noise ration (in dB)
        """
        # $P_b = \frac{1}{k}P_s$
        # Number of bits per symbol
        k = level2bits(self.M)
        return 1.0 / k * self.calcTheoreticalSER(SNR)


# xxxxx End of PSK Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxx QPSK Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class QPSK(PSK):
    """QPSK Class
    """

    def __init__(self, ):
        """
        """
        PSK.__init__(self, 4, PI / 4)

    def __repr__(self):
        return "QPSK object"
# xxxxx End of QPSK Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxx BPSK Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class BPSK(Modulator):
    """BPSK Class
    """
    def __init__(self, ):
        """
        """
        Modulator.__init__(self)
        self.setConstellation(np.array([-1, 1]))

    def __repr__(self):
        return "BPSK object"

    def calcTheoreticalSER(self, SNR):
        """Calculates the theoretical (approximation) symbol error rate for
        the BPSK squeme.

        Arguments:
        - `snr`: Signal to noise ration (in dB)
        """
        snr = dB2Linear(SNR)
        # $P_b = Q\left(\sqrt{\frac{2E_b}{N_0}}\right)$
        ser = qfunc(math.sqrt(2 * snr))
        return ser

    def calcTheoreticalBER(self, SNR):
        """Calculates the theoretical (approximation) bit error rate for
        the BPSK squeme.

        Arguments:
        - `snr`: Signal to noise ration (in dB)
        """
        return self.calcTheoreticalSER(SNR)

# xxxxx End of BPSK Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxx QAM Class xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class QAM(Modulator):
    """QAM Class
    """

    def __init__(self, M):
        """
        """
        Modulator.__init__(self)

        # Check if M is an even power of 2
        power = math.log(M, 2)
        assert power % 2 == 0, "M must be a square power of 2"
        assert 2 ** power == M, "M must be a square power of 2"

        symbols = self.__createConstellation(M)

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
        index_matrix = (rows << (level2bits(L ** 2) / 2)) + columns

        # Return the indexes as a vector (row order, which is the default
        # in numpy)
        return np.reshape(index_matrix, L ** 2)

    def _calcTheoreticalSingleCarrierErrorRate(self, SNR):
        """Calculates the theoretical (approximation) error rate of a
        single carrier in the QAM system (QAM has two carriers).

        Arguments:
        - `SNR`: Signal to noise ration (in dB)
        """
        snr = dB2Linear(SNR)
        # Probability of error of each carrier in a square QAM
        # $P_{sc} = 2\left(1 - \frac{1}{\sqrt M}\right)Q\left(\sqrt{\frac{3}{M-1}\frac{E_s}{N_0}}\right)$
        sqrtM = np.sqrt(self.M)
        Psc = 2. * (1. - (1. / sqrtM)) * qfunc(np.sqrt(snr * 3. / (self.M - 1.)))
        return Psc

    def calcTheoreticalSER(self, SNR):
        """Calculates the theoretical (approximation) symbol error rate for
        the QAM squeme.

        Arguments:
        - `SNR`: Signal to noise ration (in dB)
        """
        Psc = self._calcTheoreticalSingleCarrierErrorRate(SNR)
        # The SER is then given by
        # $ser = 1 - (1 - Psc)^2$
        ser = 1 - (1 - Psc) ** 2
        return ser

    def calcTheoreticalBER(self, SNR):
        """Calculates the theoretical (approximation) bit error rate for
        the QAM squeme.

        Arguments:
        - `SNR`: Signal to noise ration (in dB)
        """
        # For higher SNR values and gray mapping, each symbol error
        # corresponds to aproximatelly a bit error. The BER is then given
        # by the probability of error of a single carrier in the QAM system
        # divided by the number of bits transported in that carrier.
        k = level2bits(self.M)
        Psc = self._calcTheoreticalSingleCarrierErrorRate(SNR)
        ber = 2 * Psc / k
        return ber
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


if __name__ == '__main__1':
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


# xxxxx Perform the doctests xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':
    # When this module is run as a script the doctests are executed
    import doctest
    doctest.testmod()
    print "{0} executed".format(__file__)
