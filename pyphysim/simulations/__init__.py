#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module containing useful classes to implement Monte Carlo simulations.

The main class for Monte Carlo simulations is the :class:`.runner.SimulationRunner`
class, but a few other classes are also implemented to handle simulation
parameters and simulation results.

More specifically, the :mod:`.simulations` module implements the classes:
 - :class:`.runner.SimulationRunner`
 - :class:`.parameters.SimulationParameters`
 - :class:`.results.SimulationResults`
 - :class:`.results.Result`

For a description of how to implement Monte Carlo simulations using the
classes defined in the :mod:`.simulations` module see the section
:ref:`implementing_monte_carlo_simulations`.
"""

# pylint: disable=W0614,W0401
from .parameters import *
from .runner import *
from .results import *

# xxxxxxxxxx HDF5 xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# http://scipy-user.10969.n7.nabble.com/Should-I-use-pickle-for-numpy-array-td144.html
# >>> import h5py
# >>> f = h5py.File('example.hdf5', 'w')
# >>> import numpy
# >>> f['my_array'] = numpy.arange(10)
# >>> f.close()

# xxxxxxxxxx PyTables xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# http://stackoverflow.com/questions/8447926/loading-matlab-sparse-matrix-using-python-pytables

# xxxxxxxxxx Conclusions xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Its better to use h5py instead of PyTables. The h5py library provides an
# interface similar to numpy arrays and I don't need the extra abstraction
# layer from PyTables and its complex database like operations.

# xxxxxxxxxx Good answer xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# http://stackoverflow.com/questions/10075661/how-to-save-dictionaries-and-arrays-in-the-same-archive-with-numpy-savez

# xxxxxxxxxx Quick Start Guide do h5py xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# http://www.h5py.org/docs/intro/quick.html

# You can use the vitables program to visualize the files created with
# pytables or hdf5
