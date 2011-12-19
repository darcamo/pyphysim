#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np
from itertools import combinations
import multiprocessing

from subspace.metrics import calcChordalDistance, calcPrincipalAngles
from util.progressbar import ProgressbarText, DummyProgressbar

"""Module to find good codebooks"""


class CodebookFinder():
    """Class to find good codebooks using random search.
    """
    (COMPLEX, REAL, COMPLEX_QEGT) = range(3)

    def __init__(self, Nt, Ns, K, codebook_type=COMPLEX, prng_seed=None):
        """
        Arguments:
        - `Nt`: Number of rows in each precoder in the codebook
        - `Ns`: Number of columns of each precoder in the codebook
        - `K`: Number of precoders in the codebook
        - `codebook_type`: Type of the desired codebook. The allowed values are: COMPLEX, REAL, and COMPLEX_QEGT
        - `prng_seed`: Seed for the pseudo-random number generator. This is
                       passed to numpy and, if not provided, numpy will
                       provide a random seed. You only need to provide this
                       you you need the results to be reproductible or if
                       you are creating CodebookFinder multiples objects to
                       work in multiple process and you want to guarantee
                       that they will have a different seed.
        """
        self._rs = np.random.RandomState(prng_seed)

        # Codebook parameters
        assert Ns < Nt, "Ns must be lower then Nt"
        self._Nt = Nt
        self._Ns = Ns
        self._K = K

        # We want the codebook with maximum minimum distance. Let's
        # initialize max_min_dist with the worst possible value
        self._min_dist = 0
        self._principal_angle = 0
        self._best_C = None

        # The type affects how the codebook is generated
        self._codebook_type = codebook_type

        # General Configurations
        self.use_progressbar = True

    def __repr__(self):
        return "CodebookFinder: {0} {1} precoders in G({2},{3}) with minimum distance {4:.4f}".format(self._K, self._get_type_as_string(), self._Nt, self._Ns, self._min_dist)

    def _get_type_as_string(self):
        """Get the codebook type as a string.
        """
        types = {
            CodebookFinder.COMPLEX: "Complex",
            CodebookFinder.COMPLEX_QEGT: "Complex QEG",
            CodebookFinder.REAL: "Real",
            }
        return types[self._codebook_type]

    def _generate_complex_random_codebook(self, K, Nt, Ns):
        """Generates a complex random codebook.

        Arguments:
        - `K`: Number of precoders in the codebook
        - `Nt`: Number of rows (transmit antennas) in each precoder
        - `Ns`: Number of columns (number of streams) in each precoder
        """
        C = (1. / math.sqrt(2.0)) * (self._rs.randn(K, Nt, Ns) + (1j * self._rs.randn(K, Nt, Ns)))
        for k in xrange(0, K):
            C[k, :, :] /= np.linalg.norm(C[k, :, :], 'fro')

        return C

    def _generate_real_random_codebook(self, K, Nt, Ns):
        """Generates a real random codebook.

        Arguments:
        - `K`: Number of precoders in the codebook
        - `Nt`: Number of rows (transmit antennas) in each precoder
        - `Ns`: Number of columns (number of streams) in each precoder
        """
        C = self._rs.randn(K, Nt, Ns)
        for k in xrange(0, K):
            C[k, :, :] /= np.linalg.norm(C[k, :, :], 'fro')

        return C

    def _generate_complex_qegt_random_codebook(self, K, Nt, Ns):
        """Generates a complex Quantazed Equal Gain Transmission random
        codebook.

        Arguments:
        - `K`: Number of precoders in the codebook
        - `Nt`: Number of rows (transmit antennas) in each precoder
        - `Ns`: Number of columns (number of streams) in each precoder

        """
        C = self._rs.rand(K, Nt, Ns) * np.pi
        C = np.exp(1j * C)
        return C

    @staticmethod
    def calc_min_chordal_dist(codebook):
        """Claculates the minimum chordal distance in the Codebook.

        Note that the codebook is a 3-dimensional complex numpy array with
        dimension `K x Nt x Ns` (K is the number of precoders in the codebook,
        Nt and Ns are the number of rows and columns, respectively, of each
        precoder.

        Arguments:
        - `codebook`: A 3-dimensional (K x Nt x Ns) complex numpy array

        """
        K = codebook.shape[0]
        #s = codebook.shape[2]

        #Se pegar todas as combinações possíveis (sem repetoção e sem ligar para
        # ordem) vc tera (ncols**2-ncols)/2 possibilidades. Isso Equivale a pegar
        # uma matriz matrix.ncols() x matrix.ncols() e contar todos os elementos
        # abaixo (ou acima) da diagonal.
        num_possibilidades = (K ** 2 - K) / 2
        dists = np.empty(num_possibilidades)
        principal_angles = []
        index = 0

        # for comb in calc_all_comb_indexes(K):
        for comb in combinations(range(0, K), 2):
            #comb is a tuple with two elements
            dists[index] = calcChordalDistance(codebook[comb[0]], codebook[comb[1]])
            principal_angles.append(calcPrincipalAngles(codebook[comb[0]], codebook[comb[1]]))
            index += 1

        min_index = dists.argmin()  # Index of the minimum distance (in the
                                    # flattened array)
        min_dist = dists.flatten()[min_index]  # Same as dists.min()
        principal_angles = np.array(principal_angles[min_index])

        return (min_dist, principal_angles)

    def find_codebook(self, rep_max=100):
        """
        Arguments:
        - `rep_max`: Number of simulations, that is, number of generated
                     random codebooks
        """
        # The function used to create a random codebook depends on the
        # self._codebook_type variable
        gen_functions = {
            CodebookFinder.REAL: CodebookFinder._generate_real_random_codebook,
            CodebookFinder.COMPLEX: CodebookFinder._generate_complex_random_codebook,
            CodebookFinder.COMPLEX_QEGT: CodebookFinder._generate_complex_qegt_random_codebook
        }

        if self.use_progressbar:
            pb = ProgressbarText(
                rep_max,
                message="Find {0} {1} precoders in G({2},{3})".format(self._K, self.type, self._Nt, self._Ns)
            )
        else:
            pb = DummyProgressbar()

        # Simulation
        for rep in xrange(0, rep_max + 1):
            pb.progress(rep)
            # Call the apropriated codebook generating function and passes
            # the K, Nt, and Ns arguments to it.
            C = gen_functions[self._codebook_type](self, self._K, self._Nt, self._Ns)
            # C = generate_complex_random_codebook(self._K, self._Nt, self._Ns)
            # C = generate_real_random_codebook(self._K, self._Nt, self._Ns)
            (min_dist, principal_angles) = CodebookFinder.calc_min_chordal_dist(C)
            if min_dist > self._min_dist:
                # Yes! We found a better codebook. Let's save the data
                self._min_dist = min_dist
                self._principal_angle = principal_angles
                self._best_C = C

    @property
    def min_dist(self):
        """Minimum distance between the precoders in the found codebook."""
        return self._min_dist

    @property
    def principal_angles(self):
        """Pricipal angles between the precoders in the found codebook."""
        return self._principal_angle

    @property
    def codebook(self):
        """Best Codebook which was found."""
        return self._best_C

    @property
    def type(self):
        """Return a string with the type representation of the codebook."""
        return self._get_type_as_string()
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def find_codebook(rep_max=100, seed=None):
    """Find a codebook using the CodebookFinder class.

    Arguments:
    - `seed`: seed passed to the CodebookFinder object
    """

    # Codebook parameters
    Nt = 3                      # Number of tx antennas
    Ns = 1                      # Number of streams
    K = 16                      # Number of precoders in the codebook

    cf = CodebookFinder(Nt, Ns, K, CodebookFinder.COMPLEX, seed)
    cf.find_codebook(rep_max)
    print "Maximum minimum distance is: {0:.2f}".format(cf.min_dist)
    print "Principal angles are (radians): {0}".format(cf.principal_angles)
    print "Principal angles are (degrees): {0}".format(180 / np.pi * cf.principal_angles)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Acha melhores codebooks e salva em um arquivo
def simula_para_o_relatorio():
    """
    """
    # get number of cpus -> multiprocessing.cpu_count()

    # Darlan, implemente essa funcao
    Nt = 3                      # Number of tx antennas
    Ns = 1                      # Number of streams
    K = 16                      # Number of precoders in the codebook
    finename = "codebook_%s_precoders_in_G(%s,%s)" % (K, Nt, Ns)
    seed_1 = 1234
    seed_2 = 1234

    cbf1 = CodebookFinder(Nt, Ns, K, CodebookFinder.COMPLEX, seed_1)
    cbf2 = CodebookFinder(Nt, Ns, K, CodebookFinder.COMPLEX, seed_2)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxs


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxx Abaixo tentativas para rodar em paralelo xxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
def run(args):
    # name = multiprocessing.current_process().name
    # print name

    (Nt, Ns, K, seed) = args
    cf = CodebookFinder(Nt, Ns, K, CodebookFinder.COMPLEX, seed)
    cf.use_progressbar = True
    cf.find_codebook(100)
    return cf


def run2(args, queue):
    cf = run(args)
    queue.put(cf)
    #return cf


def find_codebook_parallel():
    """Call find_codebook in multiple processes.
    """
# Veja http://www.slideshare.net/pvergain/multiprocessing-with-python-presentation

    # Codebook parameters
    Nt = 3                      # Number of tx antennas
    Ns = 1                      # Number of streams
    K = 16                      # Number of precoders in the codebook
    seed_1 = 1234
    seed_2 = 5678
    args1 = (Nt, Ns, K, seed_1)
    args2 = (Nt, Ns, K, seed_2)

    # xxxxxxxxxx Abordagem 1 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Note que o parâmetro "args" é um iterable com todos os
    # argumentos. Como quero que args1 e args2 sejam reconhecidos como um
    # único argumento, então coloquei dentro de uma lista.
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=run2, args=[args1, queue])
    p2 = multiprocessing.Process(target=run2, args=[args2, queue])
    p.start()
    p2.start()

    p.join()
    p2.join()
    # print cf1.min_dist
    # print cf2.min_dist
    result1 = queue.get()
    result2 = queue.get()
    print result1.min_dist
    print result2.min_dist
    print result1
    return (result1, result2)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxxxxxxx Abordagem 2 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # pool = multiprocessing.Pool(processes=2)
    # # Map retorna uma lista com o retorno de cada execução da função
    # results = pool.map(run, [args1, args2])
    # (cf1, cf2) = results
    # print cf1.min_dist
    # print cf2.min_dist
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx



# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

if __name__ == '__main__':
    # xxxxx Add parent folder to the path xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # import os
    # import sys

    # cmd_folder = os.path.dirname(os.path.abspath(__file__))
    # if cmd_folder not in sys.path:
    #     # Add the parent folder to the beggining of the path
    #     sys.path.insert(0, cmd_folder)

    import sys
    sys.path.append("../")
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    find_codebook()
    #find_codebook_parallel()
