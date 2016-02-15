#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module to find good codebooks"""

# xxxxxxxxxx Add the parent folder to the python path. xxxxxxxxxxxxxxxxxxxx
import sys
import os
try:
    parent_dir = os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
    sys.path.append(parent_dir)
except NameError:
    sys.path.append('../')
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

from configobj import ConfigObj
#from exceptions import IOError
from itertools import combinations
from optparse import OptionParser
from time import time
import math
import multiprocessing
import numpy as np
import scipy.io

from pyphysim.subspace.metrics import calc_chordal_distance_from_principal_angles, calc_principal_angles
from pyphysim.simulations import progressbar
from pyphysim.util.misc import pretty_time


# noinspection PyShadowingNames
class CodebookFinder(object):
    """Class to find good codebooks using random search.
    """
    (COMPLEX, REAL, COMPLEX_QEGT) = range(3)

    def __init__(self, Nt, Ns, K, codebook_type=COMPLEX, prng_seed=None):
        """
        Arguments:
        - `Nt`: Number of rows in each precoder in the codebook
        - `Ns`: Number of columns of each precoder in the codebook
        - `K`: Number of precoders in the codebook
        - `codebook_type`: Type of the desired codebook. The allowed values are: COMPLEX, REAL, and COMPLEX_QEGT.
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

        # For now we set self.progressbar to a dummy progressbar.
        # Set this to a useful progressbar to track progress
        self.progressbar = progressbar.DummyProgressbar()

    def __repr__(self):
        return "CodebookFinder: {0} {1} precoders in G({2},{3}) with minimum distance {4:.4f}".format(self._K, self.type, self._Nt, self._Ns, self._min_dist)

    def _generate_complex_random_codebook(self, K, Nt, Ns):
        """Generates a complex random codebook.

        Parameters
        ----------
        K : int
            Number of precoders in the codebook
        Nt : int
            Number of rows (transmit antennas) in each precoder
        Ns : int
            Number of columns (number of streams) in each precoder

        Returns
        -------
        np.ndarray
        """
        C = (1. / math.sqrt(2.0)) * (
            self._rs.randn(K, Nt, Ns) + (1j * self._rs.randn(K, Nt, Ns)))
        ":type: np.ndarray"
        for k in range(0, K):
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
        for k in range(0, K):
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
    def type_to_string(codebook_type):
        """Get the codebook type as a string.
        """
        types = {
            CodebookFinder.COMPLEX: "Complex",
            CodebookFinder.COMPLEX_QEGT: "Complex QEG",
            CodebookFinder.REAL: "Real",
            }
        return types[codebook_type]

    @staticmethod
    def calc_min_chordal_dist(codebook):
        """Claculates the minimum chordal distance in the Codebook.

        Note that the codebook is a 3-dimensional complex numpy array with
        dimension `K x Nt x Ns` (K is the number of precoders in the codebook,
        Nt and Ns are the number of rows and columns, respectively, of each
        precoder.

        Parameters
        ----------
        codebook : A 3-dimensional (K x Nt x Ns) complex numpy array
            The codebook for which the monimum chordal distance should be
            calculated.

        """
        K = codebook.shape[0]

        #Se pegar todas as combinacoes possiveis (sem repeticao e sem ligar para
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
            pa = calc_principal_angles(codebook[comb[0]], codebook[comb[1]])
            principal_angles.append(pa)
            dists[index] = calc_chordal_distance_from_principal_angles(pa)
            index += 1

        min_index = dists.argmin()  # Index of the minimum distance (in the
                                    # flattened array)
        min_dist = dists.flatten()[min_index]  # Same as dists.min()
        principal_angles = np.array(principal_angles[min_index])

        return min_dist, principal_angles

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

        # Simulation
        for rep in range(0, rep_max + 1):
            self.progressbar.progress(rep)
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
        return CodebookFinder.type_to_string(self._codebook_type)
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


def find_codebook(Nt, Ns, K, rep_max, prng_seed=None, codebook_type=CodebookFinder.COMPLEX, progressbar=None):
    """Create a CodebookFinder object, use it to find a codebook and return
 the codebook found.

    Arguments:
    - `Kt`: Number of rows in each precoder in the codebook
    - `Ns`: Number of columns of each precoder in the codebook
    - `K`: Number of precoders in the codebook
    - `rep_max`: Number of simulations, that is, number of generated random
                 codebooks
    - `prng_seed`: Seed for the pseudo-random number generator
    - `codebook_type`: Type of the codebook. The allowed values are:
                       COMPLEX, REAL, and COMPLEX_QEGT
    """
    cb = CodebookFinder(Nt, Ns, K, CodebookFinder.COMPLEX, prng_seed)
    # An object is always true
    if progressbar:
        cb.progressbar = progressbar

    cb.find_codebook(rep_max)
    return cb.codebook
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# # xxxxxxxxxx Functions that perform a complete simulation xxxxxxxxxxxxxxxxx
# # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# def find_codebook_single_process(rep_max=100, seed=None):
#     """Find a codebook using find_codebook function.

#     Arguments:
#     - `rep_max`:
#     - `seed`: seed passed to the CodebookFinder object
#     """

#     # Codebook parameters
#     Nt = 3                      # Number of tx antennas
#     Ns = 1                      # Number of streams
#     K = 16                      # Number of precoders in the codebook
#     codebook_type = CodebookFinder.COMPLEX

#     bar = ProgressbarText(
#         rep_max,
#         message="Find {0} {1} precoders in G({2},{3})".format(
#             K,
#             CodebookFinder.type_to_string(codebook_type),
#             Nt,
#             Ns)
#         )
#     codebook = find_codebook(Nt, Ns, K, rep_max, codebook_type=codebook_type, progressbar=bar)
#     (min_dist, principal_angles) = CodebookFinder.calc_min_chordal_dist(codebook)
#     print "Maximum minimum distance is: {0:.2f}".format(min_dist)
#     print "Principal angles are (radians): {0}".format(principal_angles)
#     print "Principal angles are (degrees): {0}".format(180 / np.pi * principal_angles)
# # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Acha melhores codebooks e salva em um arquivo
def find_codebook_multiple_processes(Nt, Ns, K, rep_max=100):
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    def find_codebook_wrapper(queue, args):
        """Wrapper that calls find_codebook and put the result value in a
        queue.
        """
        queue.put(find_codebook(*args))

    def save_results(best_dist, best_codebook, best_principal_angles, filename):
        # Save matlab version
        scipy.io.savemat(
            filename,
            {'codebook': best_codebook, 'shape': best_codebook.shape},
            oned_as='row')
        # Save Python Version.
        np.savez(
            filename + ".npz",
            best_codebook=best_codebook,
            best_dist=best_dist.item(),
            best_principal_angles=best_principal_angles)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # get number of cpus -> multiprocessing.cpu_count()
    num_process = multiprocessing.cpu_count()
    print("Processes: {0}".format(num_process))
    print("Repmax: {0}".format(rep_max))

    # xxxxx Simulation Parameters xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Nt = 3                      # Number of tx antennas
    # Ns = 2                      # Number of streams
    # K = 64                      # Number of precoders in the codebook
    # rep_max = 100
    codebook_type = CodebookFinder.COMPLEX

    # The .mat extension will be added by the scipy.io.savemat function
    filename = "codebook_%s_precoders_in_G(%s,%s)" % (K, Nt, Ns)

    # Queue to store the codebooks found in each process
    queue = multiprocessing.Queue()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Multiprocess progressbar xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    pb = progressbar.ProgressbarMultiProcessServer(message="Find {0} {1} precoders in G({2},{3})".format(K, CodebookFinder.type_to_string(codebook_type), Nt, Ns))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Create the processes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    procs = []
    for i in range(0, num_process):
        proc_args = [
            Nt,
            Ns,
            K,
            rep_max,
            # TODO: Pensar em um modo de garantir sementes diferentes
            np.random.randint(0, 10000, 1).item(),
            codebook_type,
            # Register a progressbar proxy for the process to be tracked by
            # the ProgressbarMultiProcessText processbar
            pb.register_function_and_get_proxy_progressbar(rep_max)]

        procs.append(multiprocessing.Process(target=find_codebook_wrapper, args=[queue, proc_args]))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Start all processes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    for proc in procs:
        proc.start()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Start the processbar xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    pb.start_updater()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Join all processes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    for proc in procs:
        proc.join()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Stop the processbar xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    pb.stop_updater()
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Process the results xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Get all codebooks in the queue
    codebooks = [queue.get() for i in range(0, num_process)]
    min_dists = map(CodebookFinder.calc_min_chordal_dist, codebooks)
    min_dists = np.array([i[0] for i in min_dists])

    # Index of the maximum distance (index of the best codebook)
    best_index = min_dists.argmax()

    best_codebook = codebooks[best_index]
    (best_dist, best_principal_angles) = CodebookFinder.calc_min_chordal_dist(best_codebook)
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    print("Maximum minimum distance found: {0}".format(best_dist))
    print("Principal angles found: {0}".format(best_principal_angles))

    # xxxxx Open previously stored results (if there is any) xxxxxxxxxxxxxx
    try:
        previous_results = np.load(filename + ".npz")
        previous_best_dist = previous_results['best_dist']
        print("Previous minimum distance: {0}".format(previous_best_dist))
    except IOError:
        print("Could not open file `{0}`".format(filename + ".npz"))
        previous_best_dist = 0
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Save results to a file in the disk xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # Only save if it is better then the previous results
    if previous_best_dist < best_dist:
        print("Saving new results")
        save_results(best_dist, best_codebook, best_principal_angles, filename)
    else:
        print("Keeping previous results")
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


if __name__ == '__main__':
    # xxxxx Add parent folder to the path xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # import os
    # import sys
    # from exceptions import NameError

    # # Add the parent folder to the path.
    # try:
    #     # If this file is executed the __file__ will be defined and we add
    #     # the parent folder to the path, considering the file location
    #     cmd_folder = os.path.dirname(os.path.abspath(__file__))
    # except NameError, e:
    #     # If the content of this file is executed as a script then __file__
    #     # will not be defined and we add the parent folder of the current
    #     # working directory to the path
    #         cmd_folder = os.getcwd()
    # finally:
    #     if cmd_folder not in sys.path:
    #         # Add the parent folder to the beggining of the path
    #         sys.path.insert(0, cmd_folder)

    # sys.path.append("../")
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    tic = time()

    # xxxxx Get configuration filename from command line xxxxxxxxxxxxxxxxxx
    comm_line_parser = OptionParser()
    comm_line_parser.add_option("-c", "--config_file", help="Specify the configuration file", default="find_codebook_config.txt")
    (command_line_options, args) = comm_line_parser.parse_args()

    config_file_name = command_line_options.config_file
    # if config_file_name is None:
    #     config_file_name = "config.txt"
    print('Using Config file: "{0}"'.format(config_file_name))
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # xxxxx Read configuration from config file xxxxxxxxxxxxxxxxxxxxxxxxxxx
    conf_file_parser = ConfigObj(config_file_name)
    Nt = int(conf_file_parser["Precoder"]["Nt"])
    Ns = int(conf_file_parser["Precoder"]["Ns"])
    K = int(conf_file_parser["Precoder"]["K"])
    rep_max = int(conf_file_parser["Simulation"]["rep_max"])
    #results_folder = conf_file_parser["Simulation"]["results_folder"]
    # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

    #find_codebook_single_process(100)
    find_codebook_multiple_processes(Nt, Ns, K, rep_max)

    toc = time()
    print("Elapsed Time: {0}".format(pretty_time(toc - tic)))
    print("---------- End -------------------------------------------\n\n")
